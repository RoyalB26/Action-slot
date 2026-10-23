"""Excel logger and deep diagnostics tracker recording all training parameters, gradients, and Q/K/V slot dynamics."""

import os
import pandas as pd
import numpy as np
import torch


class ExcelTrainingLogger:
    """Comprehensive training stats and internal parameter tracker exporting to an Excel workbook.

    Sheets exported:
        1. 'Epoch_Overview': Loss, mAP, ego acc, learning rate, gradient norms, q/k/v activation magnitudes.
        2. 'Gradients_Detail': Per-module gradient norms (Backbone, Conv3d, SlotAttention, Head, etc.).
        3. 'QKV_Attention_Dynamics': Norms, means, stds of Query, Key, Value vectors and slot attention entropy.
        4. 'Per_Class_Performance': Per-class AP, precision, recall, support, and failure diagnostic breakdown.
    """

    def __init__(self, excel_path="checkpoints/training_full_metrics.xlsx"):
        """
        Args:
            excel_path (str): Destination path for the .xlsx file.
        """
        self.excel_path = excel_path
        parent_dir = os.path.dirname(excel_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.epoch_records = []
        self.grad_records = []
        self.qkv_records = []
        self.per_class_dfs = []

    def extract_model_stats(self, model, optimizer=None, grad_info=None):
        """Extract all gradient statistics, parameter norms, and SlotAttention Q, K, V metrics.

        Args:
            model (torch.nn.Module): Action-Slot or X3D model.
            optimizer (torch.optim.Optimizer, optional): Optimizer to get learning rate.
            grad_info (tuple[float, dict], optional): Pre-extracted (total_grad_norm, grad_dict)
                captured before optimizer.zero_grad().

        Returns:
            tuple[dict, dict, dict]: (overview_dict, grad_dict, qkv_dict)
        """
        # 1. Learning rate
        current_lr = 0.0
        if optimizer is not None and len(optimizer.param_groups) > 0:
            current_lr = optimizer.param_groups[0].get('lr', 0.0)

        # 2. Extract Gradients per module and total
        total_param_sq = 0.0
        param_dict = {}

        if grad_info is not None:
            total_grad_norm, grad_dict = grad_info
            for name, param in model.named_parameters():
                if param.requires_grad:
                    p_norm = param.data.norm(2).item()
                    total_param_sq += p_norm ** 2
                    prefix = name.split('.')[0]
                    param_dict[f"param_norm_{prefix}"] = param_dict.get(f"param_norm_{prefix}", 0.0) + (p_norm ** 2)
        else:
            total_grad_sq = 0.0
            grad_dict = {}

            # Inspect major submodules
            for name, param in model.named_parameters():
                if param.requires_grad:
                    p_norm = param.data.norm(2).item()
                    total_param_sq += p_norm ** 2
                    
                    if param.grad is not None:
                        g_norm = param.grad.data.norm(2).item()
                        total_grad_sq += g_norm ** 2
                    else:
                        g_norm = 0.0

                    # Group by top-level module prefix
                    prefix = name.split('.')[0]
                    grad_dict[f"grad_norm_{prefix}"] = grad_dict.get(f"grad_norm_{prefix}", 0.0) + (g_norm ** 2)
                    param_dict[f"param_norm_{prefix}"] = param_dict.get(f"param_norm_{prefix}", 0.0) + (p_norm ** 2)

            total_grad_norm = float(np.sqrt(total_grad_sq))
            for k in list(grad_dict.keys()):
                grad_dict[k] = float(np.sqrt(grad_dict[k]))

        # Finalize L2 norms for params
        total_param_norm = float(np.sqrt(total_param_sq))
        for k in list(param_dict.keys()):
            param_dict[k] = float(np.sqrt(param_dict[k]))

        # 3. Extract Q, K, V and Attention Dynamics from SlotAttention
        qkv_dict = {}
        slot_attn_module = None
        if hasattr(model, 'slot_attention'):
            slot_attn_module = model.slot_attention
        elif hasattr(model, 'module') and hasattr(model.module, 'slot_attention'):
            slot_attn_module = model.module.slot_attention

        if slot_attn_module is not None:
            # Query stats
            if hasattr(slot_attn_module, 'last_q') and slot_attn_module.last_q is not None:
                q = slot_attn_module.last_q.float()
                qkv_dict['q_norm_mean'] = float(q.norm(dim=-1).mean().item())
                qkv_dict['q_std'] = float(q.std().item())
                qkv_dict['q_mean'] = float(q.mean().item())

            # Key stats
            if hasattr(slot_attn_module, 'last_k') and slot_attn_module.last_k is not None:
                k = slot_attn_module.last_k.float()
                qkv_dict['k_norm_mean'] = float(k.norm(dim=-1).mean().item())
                qkv_dict['k_std'] = float(k.std().item())
                qkv_dict['k_mean'] = float(k.mean().item())

            # Value stats
            if hasattr(slot_attn_module, 'last_v') and slot_attn_module.last_v is not None:
                v = slot_attn_module.last_v.float()
                qkv_dict['v_norm_mean'] = float(v.norm(dim=-1).mean().item())
                qkv_dict['v_std'] = float(v.std().item())
                qkv_dict['v_mean'] = float(v.mean().item())

            # Attention weights distribution and Shannon entropy (degree of slot specialization)
            if hasattr(slot_attn_module, 'last_attn') and slot_attn_module.last_attn is not None:
                attn = slot_attn_module.last_attn.float()
                # attn shape (B, num_slots, T*H*W)
                eps = 1e-9
                attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + eps)
                entropy = -(attn_norm * torch.log(attn_norm + eps)).sum(dim=-1).mean().item()
                qkv_dict['attn_entropy'] = float(entropy)
                qkv_dict['attn_peak_max'] = float(attn.max().item())
                qkv_dict['attn_mean'] = float(attn.mean().item())

            # Slot parameters stats (mu, sigma)
            if hasattr(slot_attn_module, 'slots_mu'):
                qkv_dict['slots_mu_norm'] = float(slot_attn_module.slots_mu.norm(2).item())
            if hasattr(slot_attn_module, 'slots_sigma'):
                qkv_dict['slots_sigma_norm'] = float(slot_attn_module.slots_sigma.norm(2).item())

        # 4. Deformable Head Aggregation Diagnostics (Idea 6: Measure dominance)
        head_module = getattr(model, 'head', None)
        if head_module is not None:
            if hasattr(head_module, 'last_slots_norm') and head_module.last_slots_norm is not None:
                qkv_dict['deform_slots_norm'] = head_module.last_slots_norm
                qkv_dict['deform_agg_norm'] = getattr(head_module, 'last_agg_norm', 0.0)
                qkv_dict['deform_agg_to_slots_ratio'] = getattr(head_module, 'last_agg_ratio', 0.0)

        overview_dict = {
            "lr": current_lr,
            "total_grad_norm": total_grad_norm,
            "total_param_norm": total_param_norm,
            "q_norm": qkv_dict.get('q_norm_mean', 0.0),
            "k_norm": qkv_dict.get('k_norm_mean', 0.0),
            "v_norm": qkv_dict.get('v_norm_mean', 0.0),
            "attn_entropy": qkv_dict.get('attn_entropy', 0.0),
            "deform_agg_ratio": qkv_dict.get('deform_agg_to_slots_ratio', None),
        }

        return overview_dict, grad_dict, qkv_dict

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss=None,
        val_mAP=None,
        train_mAP=None,
        model=None,
        optimizer=None,
        diag_summary=None,
        grad_info=None,
        train_loss_components=None,
        val_loss_components=None,
    ):
        """Record epoch full metrics and dump updated spreadsheet to Excel.

        Args:
            epoch (int): Epoch index.
            train_loss (float): Average training loss.
            val_loss (float, optional): Average validation loss.
            val_mAP (float, optional): Validation actor mAP.
            train_mAP (float, optional): Training actor mAP.
            model (nn.Module, optional): Model instance for gradient/internal extraction.
            optimizer (Optimizer, optional): Optimizer.
            diag_summary (dict, optional): Diagnostic report from PerformanceTracker.
            grad_info (tuple, optional): Pre-extracted (total_grad_norm, grad_dict) before zero_grad.
            train_loss_components (dict, optional): Breakdown of training losses (actor, ego, attn, etc.).
            val_loss_components (dict, optional): Breakdown of validation losses.
        """
        overview_dict, grad_dict, qkv_dict = {}, {}, {}
        if model is not None:
            overview_dict, grad_dict, qkv_dict = self.extract_model_stats(model, optimizer, grad_info=grad_info)

        # 1. Append Overview Record with specific loss breakdown
        rec = {
            "Epoch": epoch,
            "Train_Loss": round(train_loss, 5),
            "Val_Loss": round(val_loss, 5) if val_loss is not None else None,
            "Train_Actor_Loss": round(train_loss_components.get("loss_actor"), 5) if (train_loss_components and train_loss_components.get("loss_actor") is not None) else None,
            "Train_Ego_Loss": round(train_loss_components.get("loss_ego"), 5) if (train_loss_components and train_loss_components.get("loss_ego") is not None) else None,
            "Train_Attn_Loss": round(train_loss_components.get("attn_loss"), 5) if (train_loss_components and train_loss_components.get("attn_loss") is not None) else None,
            "Train_Bg_Attn_Loss": round(train_loss_components.get("bg_attn_loss"), 5) if (train_loss_components and train_loss_components.get("bg_attn_loss") is not None) else None,
            "Val_Actor_Loss": round(val_loss_components.get("loss_actor"), 5) if (val_loss_components and val_loss_components.get("loss_actor") is not None) else None,
            "Val_Ego_Loss": round(val_loss_components.get("loss_ego"), 5) if (val_loss_components and val_loss_components.get("loss_ego") is not None) else None,
            "Val_Attn_Loss": round(val_loss_components.get("attn_loss"), 5) if (val_loss_components and val_loss_components.get("attn_loss") is not None) else None,
            "Train_mAP": round(train_mAP, 4) if train_mAP is not None else None,
            "Val_mAP": round(val_mAP, 4) if val_mAP is not None else None,
            "LR": overview_dict.get("lr", 0.0),
            "Total_Grad_Norm": round(overview_dict.get("total_grad_norm", 0.0), 4),
            "Total_Param_Norm": round(overview_dict.get("total_param_norm", 0.0), 4),
            "Q_Norm": round(overview_dict.get("q_norm", 0.0), 4),
            "K_Norm": round(overview_dict.get("k_norm", 0.0), 4),
            "V_Norm": round(overview_dict.get("v_norm", 0.0), 4),
            "Slot_Attn_Entropy": round(overview_dict.get("attn_entropy", 0.0), 4),
        }
        self.epoch_records.append(rec)

        # 2. Append Grad Record
        grad_rec = {"Epoch": epoch, "Total_Grad_Norm": round(overview_dict.get("total_grad_norm", 0.0), 4)}
        for k, v in grad_dict.items():
            grad_rec[k] = round(v, 4)
        self.grad_records.append(grad_rec)

        # 3. Append QKV Record
        if qkv_dict:
            qkv_rec = {"Epoch": epoch}
            for k, v in qkv_dict.items():
                qkv_rec[k] = round(v, 5)
            self.qkv_records.append(qkv_rec)

        # 4. Save per-class breakdown if available
        if diag_summary and "all_actor_classes" in diag_summary:
            cls_df = pd.DataFrame(diag_summary["all_actor_classes"])
            cls_df.insert(0, "Epoch", epoch)
            self.per_class_dfs.append(cls_df)

        # Save to multi-sheet Excel file
        self.save_excel()

    def save_excel(self):
        """Write all accumulated records to multi-tab Excel file."""
        try:
            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                # Sheet 1: Epoch Overview
                if self.epoch_records:
                    pd.DataFrame(self.epoch_records).to_excel(writer, sheet_name="Epoch_Overview", index=False)

                # Sheet 2: Gradients
                if self.grad_records:
                    pd.DataFrame(self.grad_records).to_excel(writer, sheet_name="Gradients_Detail", index=False)

                # Sheet 3: QKV Dynamics
                if self.qkv_records:
                    pd.DataFrame(self.qkv_records).to_excel(writer, sheet_name="QKV_Attention_Dynamics", index=False)

                # Sheet 4: Latest Per-Class Metrics
                if self.per_class_dfs:
                    latest_cls_df = self.per_class_dfs[-1]
                    latest_cls_df.to_excel(writer, sheet_name="Per_Class_Latest", index=False)
        except Exception as e:
            print(f"Warning: Failed to write Excel tracking log: {e}")
