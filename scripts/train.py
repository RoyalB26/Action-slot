"""Training script for Action-Slot with X3D backbone on the TACO benchmark."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import time
import gc
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco import TACO
from model import generate_model
from scripts.loss import ActionSlotLoss
from scripts.args import get_train_parser
from utils.metrics import AverageMeter
from utils.tracker import PerformanceTracker
from utils.excel_logger import ExcelTrainingLogger
from utils.visualizer import save_attention_grid


def calculate_mAP(y_true, y_pred):
    """Compute mean Average Precision (mAP) across active classes.

    Args:
        y_true (np.ndarray): Binary ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: Mean average precision score across all active classes.
    """
    ap_list = []
    num_classes = y_true.shape[1]
    for i in range(num_classes):
        if np.sum(y_true[:, i]) > 0:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            ap_list.append(ap)
    return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler=None,
    amp_enabled=False,
    grad_accum_steps=1,
    clip_grad_norm=0.0,
    train_tracker=None,
    diagnose_every_steps=0,
    save_dir=None,
    use_wandb=False,
    accelerator=None,
):
    """Run one epoch of model training with AMP, gradient accumulation, and live performance tracking.

    Args:
        model (torch.nn.Module): Action-Slot or X3D model.
        dataloader (DataLoader): PyTorch training DataLoader.
        criterion (ActionSlotLoss): Loss criterion.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Compute device (CUDA or CPU).
        epoch (int): Current epoch number.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for AMP.
        amp_enabled (bool): Enable automatic mixed precision.
        grad_accum_steps (int): Gradient accumulation iterations.
        clip_grad_norm (float): Max gradient norm for clipping.
        train_tracker (PerformanceTracker, optional): Real-time training metrics tracker.
        diagnose_every_steps (int): Frequency of in-training diagnostic breakdowns written to file.
        save_dir (str, optional): Directory to save in-training diagnostic logs.
        use_wandb (bool): If True, disables terminal tqdm bar and logs to Weights & Biases.
        accelerator (Accelerator, optional): HuggingFace Accelerate instance.

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    if train_tracker is not None:
        train_tracker.reset()

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    is_main = accelerator is None or accelerator.is_main_process

    # When wandb or multi-process is active, disable tqdm on non-main processes or when wandb clean mode is on
    show_tqdm = (not use_wandb) and is_main
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not show_tqdm)
    iterator = pbar if show_tqdm else dataloader

    total_steps = len(dataloader)
    for step, batch in enumerate(iterator):
        # Input video frames: list of 16 tensors with shape (B, 3, H, W)
        if accelerator is not None:
            videos = [frame.to(dtype=torch.float32) for frame in batch['videos']]
            label_actor = batch['gt_actor']
            label_ego = batch['gt_ego']
        else:
            videos = [frame.to(device, dtype=torch.float32) for frame in batch['videos']]
            label_actor = batch['gt_actor'].to(device)
            label_ego = batch['gt_ego'].to(device)

        # Forward pass (autocast handled by accelerator or native torch.amp)
        if accelerator is not None:
            outputs = model(videos)
            if len(outputs) == 3:
                pred_ego, pred_actor, attn = outputs
            else:
                pred_ego, pred_actor = outputs
                attn = None
            total_loss, loss_ego, loss_actor = criterion(pred_ego, pred_actor, label_ego, label_actor)
            loss = total_loss / grad_accum_steps
            accelerator.backward(loss)
        else:
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                outputs = model(videos)
                if len(outputs) == 3:
                    pred_ego, pred_actor, attn = outputs
                else:
                    pred_ego, pred_actor = outputs
                    attn = None

                total_loss, loss_ego, loss_actor = criterion(pred_ego, pred_actor, label_ego, label_actor)
                loss = total_loss / grad_accum_steps

            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Step optimizer when accumulation threshold is reached or at epoch end
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
            if accelerator is not None:
                if clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max_norm=clip_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
            else:
                if scaler is not None and amp_enabled:
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            max_norm=clip_grad_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            max_norm=clip_grad_norm,
                        )
                    optimizer.step()
                optimizer.zero_grad()

        # Update running training metrics in tracker
        if train_tracker is not None:
            train_tracker.update(
                pred_actor=pred_actor,
                target_actor=label_actor,
                pred_ego=pred_ego,
                target_ego=label_ego,
            )

        loss_meter.update(total_loss.item(), n=label_actor.size(0))

        # Real-time progress updates (rank 0 only)
        if is_main:
            quick = train_tracker.get_quick_status() if (train_tracker is not None and len(train_tracker.actor_preds) >= 10) else {}
            if show_tqdm:
                postfix = {
                    'loss': f"{loss_meter.avg:.4f}",
                    'accum': f"{(step % grad_accum_steps) + 1}/{grad_accum_steps}",
                }
                if quick:
                    postfix['mAP'] = f"{quick.get('mAP', 0.0):.3f}"
                    postfix['ego'] = f"{quick.get('ego_acc', 0.0):.2f}"
                pbar.set_postfix(postfix)
            else:
                if use_wandb:
                    import wandb
                    if wandb.run is not None:
                        log_data = {
                            "train/loss_step": total_loss.item(),
                            "train/loss_avg": loss_meter.avg,
                            "train/step": (epoch - 1) * total_steps + step,
                        }
                        if quick:
                            log_data["train/mAP_live"] = quick.get("mAP", 0.0)
                            log_data["train/ego_acc_live"] = quick.get("ego_acc", 0.0)
                        wandb.log(log_data)

                if (step + 1) % 50 == 0 or (step + 1) == total_steps:
                    mAP_str = f" | mAP: {quick.get('mAP', 0.0):.4f}" if quick else ""
                    ego_str = f" | EgoAcc: {quick.get('ego_acc', 0.0):.4f}" if quick else ""
                    print(f"Epoch [{epoch}] Step [{step+1}/{total_steps}] Loss: {loss_meter.avg:.4f}{mAP_str}{ego_str}")

            # Periodic in-training detailed error breakdown saved to file (silent on terminal)
            if (
                train_tracker is not None
                and diagnose_every_steps > 0
                and save_dir is not None
                and (step + 1) % diagnose_every_steps == 0
            ):
                diag_file = os.path.join(save_dir, "train_diagnostics_live.json")
                train_tracker.save_json(diag_file)

    return loss_meter.avg


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_actor_class=64,
    amp_enabled=False,
    tracker=None,
    save_attn_dir=None,
    epoch=None,
    use_wandb=False,
    accelerator=None,
):
    """Run validation loop, measure loss, mAP, diagnose failures, and visualize attention.

    Args:
        model (torch.nn.Module): Model under evaluation.
        dataloader (DataLoader): Validation DataLoader.
        criterion (ActionSlotLoss): Loss criterion.
        device (torch.device): Target device.
        num_actor_class (int): Number of target actor action categories.
        amp_enabled (bool): Enable mixed precision in evaluation.
        tracker (PerformanceTracker, optional): Error breakdown & diagnostic tracker.
        save_attn_dir (str, optional): Directory to save attention heatmap plots.
        epoch (int, optional): Current epoch for filename tagging.
        use_wandb (bool): If True, disables tqdm progress bar and logs eval metrics to wandb.
        accelerator (Accelerator, optional): HuggingFace Accelerate instance.

    Returns:
        tuple[float, float, dict]:
            - avg_loss: Average validation loss.
            - mAP: Mean Average Precision over validation set.
            - diag_summary: Performance diagnostic dictionary (or empty dict).
    """
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_gts = []

    if tracker is not None:
        tracker.reset()

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    attn_saved = False

    is_main = accelerator is None or accelerator.is_main_process
    show_tqdm = (not use_wandb) and is_main
    iterator = tqdm(dataloader, desc="[Validation]") if show_tqdm else dataloader

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if accelerator is not None:
                videos = [frame.to(dtype=torch.float32) for frame in batch['videos']]
                label_actor = batch['gt_actor']
                label_ego = batch['gt_ego']
            else:
                videos = [frame.to(device, dtype=torch.float32) for frame in batch['videos']]
                label_actor = batch['gt_actor'].to(device)
                label_ego = batch['gt_ego'].to(device)

            if accelerator is not None:
                outputs = model(videos)
                if len(outputs) == 3:
                    pred_ego, pred_actor, attn = outputs
                else:
                    pred_ego, pred_actor = outputs
                    attn = None
                total_loss, _, _ = criterion(pred_ego, pred_actor, label_ego, label_actor)
            else:
                with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                    outputs = model(videos)
                    if len(outputs) == 3:
                        pred_ego, pred_actor, attn = outputs
                    else:
                        pred_ego, pred_actor = outputs
                        attn = None

                    total_loss, _, _ = criterion(pred_ego, pred_actor, label_ego, label_actor)

            loss_meter.update(total_loss.item(), n=label_actor.size(0))

            # Process multi-label probabilities: reduce slot dimension if shape is (B, num_slots, C)
            actor_logits = pred_actor
            if actor_logits.ndim == 3:
                actor_logits = actor_logits.max(dim=1)[0]
            if actor_logits.shape[-1] > num_actor_class:
                actor_logits = actor_logits[:, :num_actor_class]

            pred_probs = torch.sigmoid(actor_logits.float())
            if accelerator is not None:
                gathered_preds, gathered_gts = accelerator.gather_for_metrics((pred_probs, label_actor))
                all_preds.append(gathered_preds.cpu().numpy())
                all_gts.append(gathered_gts.cpu().numpy())
            else:
                all_preds.append(pred_probs.cpu().numpy())
                all_gts.append(label_actor.cpu().numpy())

            if tracker is not None:
                tracker.update(pred_actor=pred_actor, target_actor=label_actor, pred_ego=pred_ego, target_ego=label_ego)

            # Save sample attention heatmap for the first validation batch (main process only)
            if is_main and save_attn_dir and attn is not None and not attn_saved:
                ep_tag = f"epoch_{epoch}_" if epoch is not None else ""
                plot_file = os.path.join(save_attn_dir, f"{ep_tag}val_batch_{batch_idx}_attention.png")
                try:
                    save_attention_grid(
                        frames=videos,
                        attn_tensor=attn,
                        save_path=plot_file,
                        title=f"Action-Slot Attention Map [Epoch {epoch if epoch is not None else ''}] (Red=Peak, Orange=Med, Blue=Low)",
                    )
                    attn_saved = True
                except Exception as e:
                    print(f"Warning: Failed to save attention heatmap: {e}")
                    attn_saved = True

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N, 64)
    all_gts = np.concatenate(all_gts, axis=0)      # Shape: (N, 64)
    mAP = calculate_mAP(all_gts, all_preds)

    del all_preds, all_gts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    diag_summary = {}
    if tracker is not None:
        diag_summary = tracker.compute_summary(clear_buffers=True)

    return loss_meter.avg, mAP, diag_summary


def main():
    """Main training workflow."""
    args = get_train_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup Accelerate if requested
    accelerator = None
    if getattr(args, 'accelerate', False):
        from accelerate import Accelerator, DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision='fp16' if args.amp else 'no',
            gradient_accumulation_steps=args.grad_accum_steps,
            kwargs_handlers=[ddp_kwargs],
        )
        device = accelerator.device

    is_main = accelerator is None or accelerator.is_main_process

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        if accelerator is not None:
            print(f"Accelerate initialized on device: {device} (Total processes: {accelerator.num_processes})")

    # Setup WandB if requested (rank 0 only)
    use_wandb = getattr(args, 'wandb', False)
    if use_wandb and is_main:
        import wandb
        wandb.init(
            project=getattr(args, 'wandb_project', 'Action-Slot-TACO'),
            name=getattr(args, 'wandb_run_name', None),
            config=vars(args),
        )
        print(f"Weights & Biases initialized: Project '{args.wandb_project}' (tqdm progress bar disabled for clean logging)")

    if is_main:
        print(f"Loading TACO datasets...")
    train_dataset = TACO(
        args,
        lmdb_train_path=args.lmdb_train_path,
        lmdb_val_path=args.lmdb_val_path,
        split='train',
        root=args.root,
        path_to_idx_file=args.path_to_idx_file,
    )
    val_dataset = TACO(
        args,
        lmdb_train_path=args.lmdb_train_path,
        lmdb_val_path=args.lmdb_val_path,
        split='val',
        root=args.root,
        path_to_idx_file=args.path_to_idx_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_ego_class = 7    # 7 ego actions in TACO
    num_actor_class = 64  # 64 actor actions in TACO

    print(f"Initializing model '{args.model_name}' with X3D backbone...")
    model = generate_model(args, num_ego_class, num_actor_class)
    if accelerator is None:
        model = model.to(device)

    criterion = ActionSlotLoss(args, num_actor_class)
    if accelerator is None:
        criterion = criterion.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

    if accelerator is not None:
        model, optimizer, train_loader, val_loader, criterion = accelerator.prepare(
            model, optimizer, train_loader, val_loader, criterion
        )
        scaler = None
    else:
        # Initialize GradScaler for Mixed Precision (AMP FP16)
        use_cuda = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and use_cuda))

    num_procs = accelerator.num_processes if accelerator is not None else 1
    effective_batch_size = args.batch_size * args.grad_accum_steps * num_procs
    if is_main:
        print(f"Training configuration:")
        print(f"  - Micro batch size (per GPU): {args.batch_size}")
        print(f"  - Accumulation steps: {args.grad_accum_steps}")
        print(f"  - Total GPUs / processes: {num_procs}")
        print(f"  - Effective batch size: {effective_batch_size}")
        print(f"  - AMP Mixed Precision: {'Enabled' if (args.amp and (use_cuda if accelerator is None else True)) else 'Disabled'}")
        print(f"  - Accelerate Distributed: {'Enabled' if accelerator is not None else 'Disabled'}")
        print(f"  - WandB Logging: {'Enabled (Clean mode without tqdm)' if use_wandb else 'Disabled'}")
        if args.clip_grad_norm > 0:
            print(f"  - Gradient Clipping: max_norm={args.clip_grad_norm}")

    # Optional diagnostic tracker and attention visualizer output dirs
    val_tracker = PerformanceTracker(num_actor_class, num_ego_class) if getattr(args, 'diagnose', False) else None
    train_tracker = PerformanceTracker(num_actor_class, num_ego_class) if getattr(args, 'diagnose', False) else None
    diagnose_every_steps = getattr(args, 'diagnose_every_steps', 100) if getattr(args, 'diagnose', False) else 0

    # Initialize Excel full-metrics logger (rank 0 only)
    excel_logger = None
    if is_main:
        excel_path = os.path.join(args.save_dir, "training_full_metrics.xlsx")
        excel_logger = ExcelTrainingLogger(excel_path)
        print(f"Excel full metrics tracking enabled: '{excel_path}'")

    attn_vis_dir = os.path.join(args.save_dir, "attention_vis") if (getattr(args, 'plot_attn', False) and is_main) else None
    if attn_vis_dir and is_main:
        os.makedirs(attn_vis_dir, exist_ok=True)

    best_map = 0.0

    if is_main:
        print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_enabled=(args.amp and (device.type == 'cuda' if accelerator is None else False)),
            grad_accum_steps=args.grad_accum_steps,
            clip_grad_norm=args.clip_grad_norm,
            train_tracker=train_tracker,
            diagnose_every_steps=diagnose_every_steps,
            save_dir=args.save_dir,
            use_wandb=use_wandb,
            accelerator=accelerator,
        )

        train_diag = None
        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")

            # Save training diagnostics to JSON file for this epoch (silent on terminal)
            if train_tracker is not None:
                train_diag = train_tracker.compute_summary()
                train_diag_path = os.path.join(args.save_dir, f"train_diagnostics_epoch_{epoch}.json")
                train_tracker.save_json(train_diag_path, train_diag)

            if use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "train/epoch_loss": train_loss})

        val_loss, val_map, val_diag = None, None, None
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, val_map, val_diag = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_actor_class=num_actor_class,
                amp_enabled=(args.amp and (device.type == 'cuda' if accelerator is None else False)),
                tracker=val_tracker,
                save_attn_dir=attn_vis_dir,
                epoch=epoch,
                use_wandb=use_wandb,
                accelerator=accelerator,
            )
            if is_main:
                print(f"Epoch [{epoch}/{args.epochs}] Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")

                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/mAP": val_map, "epoch": epoch})

                # Save diagnostics JSON report
                if val_diag and val_tracker is not None:
                    diag_path = os.path.join(args.save_dir, f"diagnostics_epoch_{epoch}.json")
                    val_tracker.save_json(diag_path, val_diag)
                    print(f">> Saved detailed validation diagnostics report to {diag_path}")

                # Save latest checkpoint
                latest_path = os.path.join(args.save_dir, "latest_model.pth")
                unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
                torch.save(unwrapped.state_dict(), latest_path)

                # Save best checkpoint
                if val_map > best_map:
                    best_map = val_map
                    best_path = os.path.join(args.save_dir, "best_model.pth")
                    torch.save(unwrapped.state_dict(), best_path)
                    print(f">> Saved new best model (mAP: {best_map:.4f}) to {best_path}")

        # Update and save Excel workbook with all training, gradient, and Q/K/V parameters (rank 0 only)
        if is_main and excel_logger is not None:
            unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
            train_map_quick = train_diag.get('mAP', None) if train_diag else None
            excel_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mAP=val_map,
                train_mAP=train_map_quick,
                model=unwrapped_model,
                optimizer=optimizer,
                diag_summary=val_diag or train_diag,
            )

            # Log all detailed tracker, gradient, and Q/K/V metrics to WandB
            if use_wandb:
                import wandb
                # Get latest overview, gradients, and QKV from excel logger records
                wandb_metrics = {"epoch": epoch}
                if excel_logger.epoch_records:
                    latest_overview = excel_logger.epoch_records[-1]
                    for k, v in latest_overview.items():
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            wandb_metrics[f"overview/{k.lower()}"] = v

                if excel_logger.grad_records:
                    latest_grad = excel_logger.grad_records[-1]
                    for k, v in latest_grad.items():
                        if k != "Epoch":
                            wandb_metrics[f"gradients/{k}"] = v

                if excel_logger.qkv_records:
                    latest_qkv = excel_logger.qkv_records[-1]
                    for k, v in latest_qkv.items():
                        if k != "Epoch":
                            wandb_metrics[f"qkv_attention/{k}"] = v

                # Log Ego vehicle per-class accuracy if available
                diag_for_wandb = val_diag or train_diag
                if diag_for_wandb and "ego_metrics" in diag_for_wandb:
                    ego_metrics = diag_for_wandb["ego_metrics"]
                    wandb_metrics["ego/overall_accuracy"] = ego_metrics.get("overall_accuracy", 0.0)
                    for ego_cls, stats in ego_metrics.get("per_class", {}).items():
                        wandb_metrics[f"ego_classes/{ego_cls}"] = stats["accuracy"]

                # Log latest attention visualization artifact to WandB
                if attn_vis_dir and os.path.exists(attn_vis_dir):
                    plot_files = sorted([os.path.join(attn_vis_dir, f) for f in os.listdir(attn_vis_dir) if f.endswith(".png")])
                    if plot_files:
                        wandb_metrics["attention/sample_heatmap"] = wandb.Image(plot_files[-1], caption=f"Epoch {epoch} Attention Heatmap")

                wandb.log(wandb_metrics)

        # Garbage collection & CUDA cache cleaning after each epoch to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if accelerator is not None:
            accelerator.wait_for_everyone()

    if accelerator is not None:
        accelerator.wait_for_everyone()

    if use_wandb and is_main:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
