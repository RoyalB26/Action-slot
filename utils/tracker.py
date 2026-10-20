"""Tracking and diagnostic utilities for per-class error analysis and performance breakdown."""

import os
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from utils.taco_labels import TACO_ACTOR_LABELS, TACO_EGO_LABELS


class PerformanceTracker:
    """Detailed performance tracker for per-class metrics, confusion, and failure mode diagnosis."""

    def __init__(self, num_actor_class=64, num_ego_class=7, actor_names=None, ego_names=None):
        """
        Args:
            num_actor_class (int): Number of actor action classes.
            num_ego_class (int): Number of ego action classes.
            actor_names (list[str], optional): Descriptive names for actor action classes.
            ego_names (list[str], optional): Descriptive names for ego action classes.
        """
        self.num_actor_class = num_actor_class
        self.num_ego_class = num_ego_class

        # Default to canonical TACO labels
        if actor_names is not None:
            self.actor_names = actor_names
        elif num_actor_class == 64:
            self.actor_names = TACO_ACTOR_LABELS[:64]
        elif num_actor_class == 65:
            self.actor_names = TACO_ACTOR_LABELS[:65]
        else:
            self.actor_names = [f"Actor_Class_{i}" for i in range(num_actor_class)]

        if ego_names is not None:
            self.ego_names = ego_names
        elif num_ego_class == 7:
            self.ego_names = TACO_EGO_LABELS
        else:
            self.ego_names = [f"Ego_Class_{i}" for i in range(num_ego_class)]

        self.reset()

    def reset(self):
        """Reset internal accumulator buffers."""
        self.actor_preds = []
        self.actor_targets = []
        self.ego_preds = []
        self.ego_targets = []

    def update(self, pred_actor, target_actor, pred_ego=None, target_ego=None):
        """Append batch outputs.

        Args:
            pred_actor (torch.Tensor or np.ndarray): Raw logits or sigmoid probs (B, num_actor_class).
            target_actor (torch.Tensor or np.ndarray): Ground truth binary targets (B, num_actor_class).
            pred_ego (torch.Tensor or np.ndarray, optional): Ego logits (B, num_ego_class).
            target_ego (torch.Tensor or np.ndarray, optional): Ego target indices (B,).
        """
        if isinstance(pred_actor, torch.Tensor):
            if pred_actor.ndim == 3:
                pred_actor = pred_actor.max(dim=1)[0]
            if pred_actor.shape[-1] > self.num_actor_class:
                pred_actor = pred_actor[:, :self.num_actor_class]
            probs_actor = torch.sigmoid(pred_actor.float()).detach().cpu().numpy()
        else:
            probs_actor = np.asarray(pred_actor)

        if isinstance(target_actor, torch.Tensor):
            target_actor = target_actor.detach().cpu().numpy()

        self.actor_preds.append(probs_actor)
        self.actor_targets.append(np.asarray(target_actor))

        if pred_ego is not None and target_ego is not None:
            if isinstance(pred_ego, torch.Tensor):
                pred_ego = torch.softmax(pred_ego.float(), dim=-1).detach().cpu().numpy()
            if isinstance(target_ego, torch.Tensor):
                target_ego = target_ego.detach().cpu().numpy()
            self.ego_preds.append(np.asarray(pred_ego))
            self.ego_targets.append(np.asarray(target_ego))

    def compute_summary(self, threshold=0.5, clear_buffers=True):
        """Compute comprehensive performance breakdown and diagnose poor performers.

        Args:
            threshold (float): Decision boundary threshold for binary classification.
            clear_buffers (bool): If True, immediately clear internal arrays to free RAM.

        Returns:
            dict: Detailed diagnostic report.
        """
        if len(self.actor_preds) == 0:
            return {}

        all_actor_probs = np.concatenate(self.actor_preds, axis=0)
        all_actor_gts = np.concatenate(self.actor_targets, axis=0)

        # Per-class AP, Precision, Recall, F1, False Positives, False Negatives
        actor_class_reports = []
        ap_list = []

        binary_preds = (all_actor_probs >= threshold).astype(int)

        for i in range(self.num_actor_class):
            gt_i = all_actor_gts[:, i]
            prob_i = all_actor_probs[:, i]
            pred_i = binary_preds[:, i]

            pos_count = int(np.sum(gt_i))
            pred_pos_count = int(np.sum(pred_i))

            if pos_count > 0:
                ap = float(average_precision_score(gt_i, prob_i))
                ap_list.append(ap)
            else:
                ap = 0.0

            tp = int(np.sum((pred_i == 1) & (gt_i == 1)))
            fp = int(np.sum((pred_i == 1) & (gt_i == 0)))
            fn = int(np.sum((pred_i == 0) & (gt_i == 1)))
            tn = int(np.sum((pred_i == 0) & (gt_i == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if pos_count == 0:
                failure_cause = "Zero test support (Class never appeared in ground truth)"
            elif ap >= 0.6:
                failure_cause = "Good performance"
            elif pos_count < 10:
                failure_cause = "Severe class imbalance / extreme few-shot support"
            elif recall < 0.2 and precision >= 0.5:
                failure_cause = "High Miss Rate (Model is overly conservative, high false negatives)"
            elif precision < 0.2 and recall >= 0.5:
                failure_cause = "High False Alarm (Model over-predicts, high false positives)"
            elif precision < 0.3 and recall < 0.3:
                failure_cause = "Severe confusion / feature overlap with background or other classes"
            else:
                failure_cause = "Moderate confusion or insufficient capacity"

            actor_class_reports.append({
                "class_id": i,
                "name": self.actor_names[i] if i < len(self.actor_names) else f"Class_{i}",
                "support": pos_count,
                "predicted_pos": pred_pos_count,
                "ap": round(ap, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "primary_issue": failure_cause,
            })

        mean_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

        # Sort classes by AP ascending to find worst performers
        active_classes = [c for c in actor_class_reports if c["support"] > 0]
        worst_classes = sorted(active_classes, key=lambda x: x["ap"])[:10]
        best_classes = sorted(active_classes, key=lambda x: x["ap"], reverse=True)[:10]

        # Ego vehicle breakdown
        ego_summary = {}
        if len(self.ego_preds) > 0:
            all_ego_probs = np.concatenate(self.ego_preds, axis=0)
            all_ego_gts = np.concatenate(self.ego_targets, axis=0)
            ego_pred_cls = np.argmax(all_ego_probs, axis=-1)
            acc = float(np.mean(ego_pred_cls == all_ego_gts))

            per_ego_acc = {}
            for e_idx in range(self.num_ego_class):
                mask = (all_ego_gts == e_idx)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    per_ego_acc[self.ego_names[e_idx]] = {
                        "accuracy": round(float(np.mean(ego_pred_cls[mask] == e_idx)), 4),
                        "count": cnt,
                    }
            ego_summary = {
                "overall_accuracy": round(acc, 4),
                "per_class": per_ego_acc,
            }

        report = {
            "mAP": round(mean_ap, 4),
            "num_evaluated_samples": len(all_actor_gts),
            "best_performing_actor_classes": best_classes,
            "worst_performing_actor_classes": worst_classes,
            "all_actor_classes": actor_class_reports,
            "ego_metrics": ego_summary,
        }

        if clear_buffers:
            self.reset()

        return report

    def get_quick_status(self) -> dict:
        """Compute quick lightweight status for training progress bars."""
        if len(self.actor_preds) == 0:
            return {}
        try:
            preds = np.concatenate(self.actor_preds, axis=0)
            gts = np.concatenate(self.actor_targets, axis=0)
            aps = []
            for i in range(min(self.num_actor_class, preds.shape[1])):
                if np.sum(gts[:, i]) > 0:
                    aps.append(average_precision_score(gts[:, i], preds[:, i]))
            map_score = float(np.mean(aps)) if len(aps) > 0 else 0.0

            ego_acc = 0.0
            if len(self.ego_preds) > 0:
                ego_probs = np.concatenate(self.ego_preds, axis=0)
                ego_gts = np.concatenate(self.ego_targets, axis=0)
                ego_acc = float(np.mean(np.argmax(ego_probs, axis=-1) == ego_gts))

            return {"mAP": round(map_score, 4), "ego_acc": round(ego_acc, 4)}
        except Exception:
            return {}

    def save_json(self, save_path: str, summary: dict = None):
        """Save performance report to a JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if summary is None:
            summary = self.compute_summary(clear_buffers=False)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
