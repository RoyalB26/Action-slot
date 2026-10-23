"""Decoupled Execution Engine managing Training, Evaluation, Tracking, and Distributed Lifecycle."""

import os
import sys
import gc
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List

from utils.metrics import AverageMeter
from utils.tracker import PerformanceTracker
from utils.excel_logger import ExcelTrainingLogger
from utils.visualizer import save_attention_grid
from utils.attention_dumper import (
    save_attention_matrix,
    visualize_best_and_worst_slots,
    render_all_slots_from_file,
)
from utils.taco_labels import TACO_ACTOR_LABELS
from utils.tracking_logger import append_validation_tracking_log


def calculate_mAP(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean Average Precision (mAP) across active classes."""
    ap_list = []
    num_classes = y_true.shape[1]
    for i in range(num_classes):
        if np.sum(y_true[:, i]) > 0:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            ap_list.append(ap)
    return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0


def extract_gradient_stats(model: torch.nn.Module) -> Tuple[float, Dict[str, float]]:
    """Extract total grad norm and module-wise gradient norms from model parameters.

    Returns:
        tuple[float, dict[str, float]]: (total_grad_norm, grad_dict)
    """
    total_grad_sq = 0.0
    grad_dict = {}

    # Strip wrapper (e.g. DDP, DataParallel) if present
    base_model = getattr(model, 'module', model)

    for name, param in base_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g_norm = param.grad.data.norm(2).item()
            total_grad_sq += g_norm ** 2
            prefix = name.split('.')[0]
            grad_dict[f"grad_norm_{prefix}"] = grad_dict.get(f"grad_norm_{prefix}", 0.0) + (g_norm ** 2)

            # Detailed breakdown for Deformable Head subcomponents
            if "head.value_proj" in name:
                grad_dict["grad_norm_head_value_proj"] = grad_dict.get("grad_norm_head_value_proj", 0.0) + (g_norm ** 2)
            elif "head.out_proj" in name:
                grad_dict["grad_norm_head_out_proj"] = grad_dict.get("grad_norm_head_out_proj", 0.0) + (g_norm ** 2)
            elif "head.offset_proj" in name or "head.weight_proj" in name:
                grad_dict["grad_norm_head_sampling_projs"] = grad_dict.get("grad_norm_head_sampling_projs", 0.0) + (g_norm ** 2)
            elif "head.fc_actor" in name:
                grad_dict["grad_norm_head_fc_actor"] = grad_dict.get("grad_norm_head_fc_actor", 0.0) + (g_norm ** 2)

    total_grad_norm = float(np.sqrt(total_grad_sq))
    for k in list(grad_dict.keys()):
        grad_dict[k] = float(np.sqrt(grad_dict[k]))
    return total_grad_norm, grad_dict


class Trainer:
    """Standardized training engine handling multi-GPU DDP, validation, logging, and memory lifecycle."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        args = None,
        accelerator = None,
        scheduler = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.accelerator = accelerator
        self.scheduler = scheduler

        self.device = accelerator.device if accelerator is not None else torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.is_main = accelerator is None or accelerator.is_main_process

        self.num_actor_class = getattr(args, 'num_actor_class', 64)
        self.num_ego_class = getattr(args, 'num_ego_class', 7)

        # Logging & output tracking setup
        self.use_wandb = getattr(args, 'wandb', False)
        self.save_dir = getattr(args, 'save_dir', 'checkpoints')
        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)

        self.excel_logger = None
        if self.is_main:
            excel_path = os.path.join(self.save_dir, "training_full_metrics.xlsx")
            self.excel_logger = ExcelTrainingLogger(excel_path)

        self.attn_vis_dir = os.path.join(self.save_dir, "attention_vis") if (getattr(args, 'plot_attn', False) and self.is_main) else None
        if self.attn_vis_dir and self.is_main:
            os.makedirs(self.attn_vis_dir, exist_ok=True)

        # Performance error tracker
        self.train_tracker = PerformanceTracker(self.num_actor_class, self.num_ego_class) if getattr(args, 'diagnose', False) else None
        self.val_tracker = PerformanceTracker(self.num_actor_class, self.num_ego_class) if getattr(args, 'diagnose', False) else None

        self.best_map = 0.0
        self.best_val_loss = float('inf')
        self.patience = getattr(args, 'patience', 0)
        self.early_stopping_metric = getattr(args, 'early_stopping_metric', 'val_map')
        self.patience_counter = 0
        self.early_stop_triggered = False

    def train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Optional[Tuple[float, Dict[str, float]]]]:
        """Execute one complete training epoch with detailed loss breakdown."""
        self.model.train()
        loss_meter = AverageMeter()
        loss_component_meters: Dict[str, AverageMeter] = {}
        self.optimizer.zero_grad()

        if self.train_tracker is not None:
            self.train_tracker.reset()

        last_grad_info = None

        grad_accum_steps = getattr(self.args, 'grad_accum_steps', 1)
        clip_grad_norm = getattr(self.args, 'clip_grad_norm', 0.0)
        diagnose_every_steps = getattr(self.args, 'diagnose_every_steps', 0)

        show_tqdm = (not self.use_wandb) and self.is_main
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", disable=not show_tqdm)
        iterator = pbar if show_tqdm else self.train_loader
        total_steps = len(self.train_loader)

        for step, batch in enumerate(iterator):
            # Input tensors
            if self.accelerator is not None:
                videos = [f.to(dtype=torch.float32) for f in batch['videos']]
                targets = {"target_actor": batch['gt_actor'], "target_ego": batch['gt_ego']}
            else:
                videos = [f.to(self.device, dtype=torch.float32) for f in batch['videos']]
                targets = {"target_actor": batch['gt_actor'].to(self.device), "target_ego": batch['gt_ego'].to(self.device)}

            # Standardized forward pass returning Dictionary
            preds = self.model(videos)
            total_loss, loss_dict = self.criterion(preds, targets)
            loss = total_loss / grad_accum_steps

            # Backward pass
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # Optimizer step with gradient accumulation
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                if self.accelerator is not None:
                    if clip_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=clip_grad_norm
                        )
                    # Capture gradients right before zero_grad
                    unwrapped_for_grad = self.accelerator.unwrap_model(self.model)
                    last_grad_info = extract_gradient_stats(unwrapped_for_grad)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=clip_grad_norm
                        )
                    # Capture gradients right before zero_grad
                    last_grad_info = extract_gradient_stats(self.model)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            if self.train_tracker is not None:
                self.train_tracker.update(
                    pred_actor=preds.get("pred_actor"),
                    target_actor=targets.get("target_actor"),
                    pred_ego=preds.get("pred_ego"),
                    target_ego=targets.get("target_ego")
                )

            loss_meter.update(total_loss.item(), n=targets["target_actor"].size(0))
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if k != "total_loss" and v is not None:
                        if k not in loss_component_meters:
                            loss_component_meters[k] = AverageMeter()
                        loss_component_meters[k].update(v.item() if isinstance(v, torch.Tensor) else float(v), n=targets["target_actor"].size(0))

            # Logging & terminal status
            if self.is_main:
                quick = self.train_tracker.get_quick_status() if (self.train_tracker is not None and len(self.train_tracker.actor_preds) >= 10) else {}
                if show_tqdm:
                    postfix = {'loss': f"{loss_meter.avg:.4f}", 'accum': f"{(step % grad_accum_steps) + 1}/{grad_accum_steps}"}
                    if "loss_actor" in loss_component_meters:
                        postfix['act_l'] = f"{loss_component_meters['loss_actor'].avg:.3f}"
                    if "loss_ego" in loss_component_meters:
                        postfix['ego_l'] = f"{loss_component_meters['loss_ego'].avg:.3f}"
                    if "attn_loss" in loss_component_meters:
                        postfix['att_l'] = f"{loss_component_meters['attn_loss'].avg:.3f}"
                    if quick:
                        postfix['mAP'] = f"{quick.get('mAP', 0.0):.3f}"
                    pbar.set_postfix(postfix)
                else:
                    if self.use_wandb:
                        import wandb
                        if wandb.run is not None:
                            wandb_log_dict = {
                                "train/loss_step": total_loss.item(),
                                "train/loss_avg": loss_meter.avg,
                                "train/step": (epoch - 1) * total_steps + step,
                            }
                            for comp_k, comp_m in loss_component_meters.items():
                                wandb_log_dict[f"train/{comp_k}_avg"] = comp_m.avg
                            wandb.log(wandb_log_dict)
                    if (step + 1) % 50 == 0 or (step + 1) == total_steps:
                        mAP_str = f" | mAP: {quick.get('mAP', 0.0):.4f}" if quick else ""
                        comp_str = ""
                        if "loss_actor" in loss_component_meters:
                            comp_str += f" | Actor Loss: {loss_component_meters['loss_actor'].avg:.4f}"
                        if "loss_ego" in loss_component_meters:
                            comp_str += f" | Ego Loss: {loss_component_meters['loss_ego'].avg:.4f}"
                        if "attn_loss" in loss_component_meters:
                            comp_str += f" | Attn Loss: {loss_component_meters['attn_loss'].avg:.4f}"
                        if "bg_attn_loss" in loss_component_meters:
                            comp_str += f" | Bg Attn Loss: {loss_component_meters['bg_attn_loss'].avg:.4f}"
                        print(f"Epoch [{epoch}] Step [{step+1}/{total_steps}] Loss: {loss_meter.avg:.4f}{comp_str}{mAP_str}")

                if self.train_tracker is not None and diagnose_every_steps > 0 and (step + 1) % diagnose_every_steps == 0:
                    diag_file = os.path.join(self.save_dir, "train_diagnostics_live.json")
                    self.train_tracker.save_json(diag_file)

        avg_loss_components = {k: v.avg for k, v in loss_component_meters.items()}
        return loss_meter.avg, avg_loss_components, last_grad_info

    @torch.no_grad()
    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, dict]:
        """Execute evaluation loop and compute mAP and diagnostic metrics."""
        self.model.eval()
        loss_meter = AverageMeter()
        loss_component_meters: Dict[str, AverageMeter] = {}
        all_preds = []
        all_gts = []

        if self.val_tracker is not None:
            self.val_tracker.reset()

        show_tqdm = (not self.use_wandb) and self.is_main
        iterator = tqdm(self.val_loader, desc="[Validation]") if show_tqdm else self.val_loader
        attn_saved = False
        saved_sample_videos = None
        saved_sample_attn = None

        # Buffers to find specific samples where best slot is correct and worst slot is incorrect
        best_vis_sample = None    # {"videos": ..., "attn": ..., "batch_idx": ...}
        worst_vis_sample = None   # {"videos": ..., "attn": ..., "batch_idx": ...}
        buffered_batches = []     # Keep up to 20 batches of (videos, attn, pred_probs, label_actor) for slot targeting

        for batch_idx, batch in enumerate(iterator):
            if self.accelerator is not None:
                videos = [f.to(dtype=torch.float32) for f in batch['videos']]
                targets = {"target_actor": batch['gt_actor'], "target_ego": batch['gt_ego']}
            else:
                videos = [f.to(self.device, dtype=torch.float32) for f in batch['videos']]
                targets = {"target_actor": batch['gt_actor'].to(self.device), "target_ego": batch['gt_ego'].to(self.device)}

            preds = self.model(videos)
            total_loss, loss_dict = self.criterion(preds, targets)
            loss_meter.update(total_loss.item(), n=targets["target_actor"].size(0))
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    if k != "total_loss" and v is not None:
                        if k not in loss_component_meters:
                            loss_component_meters[k] = AverageMeter()
                        loss_component_meters[k].update(v.item() if isinstance(v, torch.Tensor) else float(v), n=targets["target_actor"].size(0))

            pred_actor = preds.get("pred_actor")
            if pred_actor.ndim == 3:
                pred_actor = pred_actor.max(dim=1)[0]
            if pred_actor.shape[-1] > self.num_actor_class:
                pred_actor = pred_actor[:, :self.num_actor_class]

            pred_probs = torch.sigmoid(pred_actor.float())
            label_actor = targets["target_actor"]

            if self.accelerator is not None:
                gathered_preds, gathered_gts = self.accelerator.gather_for_metrics((pred_probs, label_actor))
                all_preds.append(gathered_preds.cpu().numpy())
                all_gts.append(gathered_gts.cpu().numpy())
            else:
                all_preds.append(pred_probs.cpu().numpy())
                all_gts.append(label_actor.cpu().numpy())

            if self.val_tracker is not None:
                self.val_tracker.update(
                    pred_actor=preds.get("pred_actor"),
                    target_actor=targets.get("target_actor"),
                    pred_ego=preds.get("pred_ego"),
                    target_ego=targets.get("target_ego")
                )

            # Save sample attention matrix and cache candidates for best/worst visualization
            attn = preds.get("attn")
            if self.is_main and self.attn_vis_dir and attn is not None:
                if not attn_saved:
                    ep_tag = f"epoch_{epoch}_" if epoch is not None else ""
                    mat_file = os.path.join(self.attn_vis_dir, f"{ep_tag}attention_matrix.npz")
                    try:
                        save_attention_matrix(
                            attn_tensor=attn,
                            frames=videos,
                            save_path=mat_file,
                            extra_metadata={"epoch": epoch, "batch_idx": batch_idx},
                        )
                        saved_sample_videos = videos
                        saved_sample_attn = attn
                        attn_saved = True
                    except Exception as e:
                        print(f"Warning: Failed to save attention matrix: {e}")
                        attn_saved = True

                # Buffer up to 30 batches for finding true positive (correct) and error (incorrect) samples
                if len(buffered_batches) < 30:
                    cpu_videos = [f.detach().cpu() for f in videos]
                    cpu_attn = attn.detach().cpu()
                    cpu_preds = pred_probs.detach().cpu().numpy()
                    cpu_gts = label_actor.detach().cpu().numpy()
                    buffered_batches.append({
                        "videos": cpu_videos,
                        "attn": cpu_attn,
                        "preds": cpu_preds,
                        "gts": cpu_gts,
                    })

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
        mAP = calculate_mAP(all_gts, all_preds)

        del all_preds, all_gts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        diag_summary = {}
        if self.val_tracker is not None:
            diag_summary = self.val_tracker.compute_summary(clear_buffers=True)

        # Generate 1 Best Slot and 1 Worst Slot visualizations across all T frames
        if self.is_main and self.attn_vis_dir and saved_sample_videos is not None and saved_sample_attn is not None:
            try:
                per_class = diag_summary.get("all_actor_classes", [])
                
                # Determine best and worst slot indices
                active_classes = [c for c in per_class if c.get("support", 0) > 0 and c.get("class_id", 0) < self.num_actor_class]
                if active_classes:
                    best_cand = max(active_classes, key=lambda x: x.get("ap", 0.0))
                    worst_cand = min(active_classes, key=lambda x: x.get("ap", 0.0))
                    best_idx = best_cand["class_id"]
                    worst_idx = worst_cand["class_id"]
                else:
                    best_idx = 0
                    worst_idx = min(1, self.num_actor_class - 1)

                # Search buffered batches for a sample where best_slot is CORRECT (TP preferred, or TN)
                # and a sample where worst_slot is INCORRECT (FN or FP)
                best_frames = None
                best_attn = None
                worst_frames = None
                worst_attn = None

                # 1. Search for best slot correct sample (prioritize True Positive where gt == 1 and pred >= 0.5)
                for cand in buffered_batches:
                    preds_b = cand["preds"]
                    gts_b = cand["gts"]
                    # TP check
                    tp_mask = np.where((preds_b[:, best_idx] >= 0.5) & (gts_b[:, best_idx] == 1))[0]
                    if len(tp_mask) > 0:
                        i = tp_mask[0]
                        best_frames = [f[i] for f in cand["videos"]]
                        best_attn = cand["attn"][i]
                        break

                # Fallback for best slot if no TP found in buffer: check TN (gt == 0 and pred < 0.5)
                if best_frames is None:
                    for cand in buffered_batches:
                        preds_b = cand["preds"]
                        gts_b = cand["gts"]
                        correct_mask = np.where((preds_b[:, best_idx] >= 0.5) == (gts_b[:, best_idx] == 1))[0]
                        if len(correct_mask) > 0:
                            i = correct_mask[0]
                            best_frames = [f[i] for f in cand["videos"]]
                            best_attn = cand["attn"][i]
                            break

                # 2. Search for worst slot incorrect sample (prioritize False Negative where gt == 1 and pred < 0.5, then False Positive)
                for cand in buffered_batches:
                    preds_b = cand["preds"]
                    gts_b = cand["gts"]
                    # FN check
                    fn_mask = np.where((preds_b[:, worst_idx] < 0.5) & (gts_b[:, worst_idx] == 1))[0]
                    if len(fn_mask) > 0:
                        i = fn_mask[0]
                        worst_frames = [f[i] for f in cand["videos"]]
                        worst_attn = cand["attn"][i]
                        break

                # If no FN, look for FP (gt == 0 and pred >= 0.5)
                if worst_frames is None:
                    for cand in buffered_batches:
                        preds_b = cand["preds"]
                        gts_b = cand["gts"]
                        fp_mask = np.where((preds_b[:, worst_idx] >= 0.5) & (gts_b[:, worst_idx] == 0))[0]
                        if len(fp_mask) > 0:
                            i = fp_mask[0]
                            worst_frames = [f[i] for f in cand["videos"]]
                            worst_attn = cand["attn"][i]
                            break

                # Fallback to default saved batch sample if not found in buffer
                if best_frames is None:
                    best_frames = saved_sample_videos
                    best_attn = saved_sample_attn
                if worst_frames is None:
                    worst_frames = saved_sample_videos
                    worst_attn = saved_sample_attn

                visualize_best_and_worst_slots(
                    frames=best_frames,
                    attn_tensor=best_attn,
                    output_dir=self.attn_vis_dir,
                    epoch=epoch,
                    slot_names=TACO_ACTOR_LABELS,
                    per_class_summary=per_class,
                    worst_frames=worst_frames,
                    worst_attn_tensor=worst_attn,
                    best_slot_idx=best_idx,
                    worst_slot_idx=worst_idx,
                )
            except Exception as e:
                print(f"Warning: Failed to render best/worst slot visualizations: {e}")

        avg_val_loss_components = {k: v.avg for k, v in loss_component_meters.items()}
        diag_summary["loss_components"] = avg_val_loss_components

        return loss_meter.avg, mAP, diag_summary

    def fit(self, epochs: int, val_every: int = 5):
        """Run complete multi-epoch training and validation workflow."""
        if self.is_main:
            print("Starting training via Modular Engine...")

        for epoch in range(26, epochs + 1):
            train_loss, train_loss_dict, train_grad_info = self.train_one_epoch(epoch)

            train_diag = None
            if self.is_main:
                loss_breakdown_str = ""
                if train_loss_dict:
                    if "loss_actor" in train_loss_dict:
                        loss_breakdown_str += f" | Actor Loss: {train_loss_dict['loss_actor']:.4f}"
                    if "loss_ego" in train_loss_dict:
                        loss_breakdown_str += f" | Ego Loss: {train_loss_dict['loss_ego']:.4f}"
                    if "attn_loss" in train_loss_dict:
                        loss_breakdown_str += f" | Attn Loss: {train_loss_dict['attn_loss']:.4f}"
                    if "bg_attn_loss" in train_loss_dict:
                        loss_breakdown_str += f" | Bg Attn Loss: {train_loss_dict['bg_attn_loss']:.4f}"
                print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}{loss_breakdown_str}")
                if self.train_tracker is not None:
                    train_diag = self.train_tracker.compute_summary(clear_buffers=True)
                    if train_loss_dict:
                        train_diag["loss_components"] = train_loss_dict
                    train_diag_path = os.path.join(self.save_dir, f"train_diagnostics_epoch_{epoch}.json")
                    self.train_tracker.save_json(train_diag_path, train_diag)

                if self.use_wandb:
                    import wandb
                    wandb_train_dict = {"epoch": epoch, "train/epoch_loss": train_loss}
                    if train_loss_dict:
                        for l_name, l_val in train_loss_dict.items():
                            wandb_train_dict[f"train/epoch_{l_name}"] = l_val
                    wandb.log(wandb_train_dict)

            val_loss, val_map, val_diag = None, None, None
            if self.val_loader is not None and (epoch % val_every == 0 or epoch == epochs):
                val_loss, val_map, val_diag = self.evaluate(epoch=epoch)
                if self.is_main:
                    val_comp_str = ""
                    val_loss_comps = val_diag.get("loss_components", {}) if val_diag else {}
                    if val_loss_comps:
                        if "loss_actor" in val_loss_comps:
                            val_comp_str += f" | Val Actor Loss: {val_loss_comps['loss_actor']:.4f}"
                        if "loss_ego" in val_loss_comps:
                            val_comp_str += f" | Val Ego Loss: {val_loss_comps['loss_ego']:.4f}"
                        if "attn_loss" in val_loss_comps:
                            val_comp_str += f" | Val Attn Loss: {val_loss_comps['attn_loss']:.4f}"
                    print(f"Epoch [{epoch}/{epochs}] Val Loss: {val_loss:.4f}{val_comp_str} | Val mAP: {val_map:.4f}")
                    if self.use_wandb:
                        import wandb
                        wandb_val_dict = {"val/loss": val_loss, "val/mAP": val_map, "epoch": epoch}
                        if val_loss_comps:
                            for vl_name, vl_val in val_loss_comps.items():
                                wandb_val_dict[f"val/{vl_name}"] = vl_val
                        wandb.log(wandb_val_dict)

                    if val_diag and self.val_tracker is not None:
                        diag_path = os.path.join(self.save_dir, f"diagnostics_epoch_{epoch}.json")
                        self.val_tracker.save_json(diag_path, val_diag)

                    # Checkpoints
                    latest_path = os.path.join(self.save_dir, "latest_model.pth")
                    unwrapped = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
                    torch.save(unwrapped.state_dict(), latest_path)

                    improved = False
                    if self.early_stopping_metric == 'val_loss':
                        if val_loss is not None and val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            improved = True
                    else:
                        if val_map is not None and val_map > self.best_map:
                            self.best_map = val_map
                            improved = True

                    if val_map is not None and val_map > self.best_map:
                        self.best_map = val_map

                    if improved:
                        self.patience_counter = 0
                        best_path = os.path.join(self.save_dir, "best_model.pth")
                        torch.save(unwrapped.state_dict(), best_path)
                        metric_val_str = f"val_loss: {self.best_val_loss:.4f}" if self.early_stopping_metric == 'val_loss' else f"mAP: {self.best_map:.4f}"
                        print(f">> Saved new best model ({metric_val_str}) to {best_path}")
                    elif self.patience > 0:
                        self.patience_counter += 1
                        print(f">> Early stopping patience: {self.patience_counter}/{self.patience}")
                        if self.patience_counter >= self.patience:
                            print(f">> Early stopping triggered! Validation {self.early_stopping_metric} did not improve for {self.patience} evaluations.")
                            self.early_stop_triggered = True

                    # Append to validation tracking report file and print to console
                    tracking_file = os.path.join(self.save_dir, "val_tracking.txt")
                    per_class_list = val_diag.get("all_actor_classes", []) if val_diag else []
                    append_validation_tracking_log(
                        file_path=tracking_file,
                        epoch=epoch,
                        best_map=self.best_map,
                        current_map=val_map,
                        per_class_reports=per_class_list,
                        print_to_terminal=True,
                    )

            # Log to Excel & WandB on main process
            if self.is_main and self.excel_logger is not None:
                unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
                train_map_quick = train_diag.get('mAP', None) if train_diag else None
                self.excel_logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mAP=val_map,
                    train_mAP=train_map_quick,
                    model=unwrapped_model,
                    optimizer=self.optimizer,
                    diag_summary=val_diag or train_diag,
                    grad_info=train_grad_info,
                    train_loss_components=train_loss_dict,
                    val_loss_components=val_diag.get("loss_components", {}) if val_diag else None,
                )

                if self.use_wandb:
                    import wandb
                    wandb_metrics = {"epoch": epoch}
                    if self.excel_logger.epoch_records:
                        for k, v in self.excel_logger.epoch_records[-1].items():
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                wandb_metrics[f"overview/{k.lower()}"] = v
                    if self.excel_logger.grad_records:
                        for k, v in self.excel_logger.grad_records[-1].items():
                            if k != "Epoch":
                                wandb_metrics[f"gradients/{k}"] = v
                    if self.excel_logger.qkv_records:
                        for k, v in self.excel_logger.qkv_records[-1].items():
                            if k != "Epoch":
                                wandb_metrics[f"qkv_attention/{k}"] = v

                    if self.attn_vis_dir and os.path.exists(self.attn_vis_dir):
                        plot_files = sorted([os.path.join(self.attn_vis_dir, f) for f in os.listdir(self.attn_vis_dir) if f.endswith(".png")])
                        if plot_files:
                            wandb_metrics["attention/sample_heatmap"] = wandb.Image(plot_files[-1], caption=f"Epoch {epoch} Heatmap")

                    wandb.log(wandb_metrics)

            if self.early_stop_triggered:
                print(">> Terminating training loop early due to early stopping criteria.")
                break

            if self.scheduler is not None:
                self.scheduler.step()

            # Garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self.use_wandb and self.is_main:
            import wandb
            wandb.finish()
