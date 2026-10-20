"""Tracking and diagnostic utilities for per-class error analysis and performance breakdown."""

import os
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


EGO_ACTION_NAMES = [
    "Stop",
    "Go / Accelerate",
    "Turn Left",
    "Turn Right",
    "Lane Change Left",
    "Lane Change Right",
    "Slow Down / Brake",
]


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
        self.actor_names = actor_names or [f"Actor_Class_{i}" for i in range(num_actor_class)]
        self.ego_names = ego_names or (EGO_ACTION_NAMES if num_ego_class == 7 else [f"Ego_Class_{i}" for i in range(num_ego_class)])
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
            # If actor logits are 3D (B, num_slots, C), reduce max over slots
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

    def compute_summary(self, threshold=0.5):
        """Compute comprehensive performance breakdown and diagnose poor performers.

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

            # Confusion counts
            tp = int(np.sum((pred_i == 1) & (gt_i == 1)))
            fp = int(np.sum((pred_i == 1) & (gt_i == 0)))
            fn = int(np.sum((pred_i == 0) & (gt_i == 1)))
            tn = int(np.sum((pred_i == 0) & (gt_i == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Diagnose failure mode
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

        return {
            "mAP": round(mean_ap, 4),
            "num_evaluated_samples": len(all_actor_gts),
            "best_performing_actor_classes": best_classes,
            "worst_performing_actor_classes": worst_classes,
            "all_actor_classes": actor_class_reports,
            "ego_metrics": ego_summary,
        }

    def print_diagnostic_table(self, summary=None, title="MODEL DIAGNOSTIC REPORT"):
        """Format and print an informative diagnostic console report."""
        if summary is None:
            summary = self.compute_summary()

        print("\n" + "=" * 80)
        print(f" {title} (Overall Actor mAP: {summary.get('mAP', 0.0):.4f})")
        print("=" * 80)

        # 1. Top 5 Best Classes
        print("\n[+] TOP 5 BEST PERFORMING ACTOR ACTIONS:")
        print(f"{'Class':<25} | {'Support':<8} | {'AP':<8} | {'Prec':<8} | {'Rec':<8} | Status")
        print("-" * 80)
        for c in summary.get("best_performing_actor_classes", [])[:5]:
            print(f"{c['name'][:24]:<25} | {c['support']:<8} | {c['ap']:<8.4f} | {c['precision']:<8.4f} | {c['recall']:<8.4f} | {c['primary_issue']}")

        # 2. Top 5 Worst Classes with Root Cause
        print("\n[-] TOP 5 WORST PERFORMING ACTOR ACTIONS & ROOT CAUSE ANALYSIS:")
        print(f"{'Class':<25} | {'Support':<8} | {'AP':<8} | {'Prec':<8} | {'Rec':<8} | Root Cause Diagnosis")
        print("-" * 80)
        for c in summary.get("worst_performing_actor_classes", [])[:5]:
            print(f"{c['name'][:24]:<25} | {c['support']:<8} | {c['ap']:<8.4f} | {c['precision']:<8.4f} | {c['recall']:<8.4f} | {c['primary_issue']}")

        # 3. Ego performance
        ego_met = summary.get("ego_metrics", {})
        if ego_met:
            print("\n[*] EGO-VEHICLE ACTION PREDICTION:")
            print(f"Overall Accuracy: {ego_met.get('overall_accuracy', 0.0):.4f}")
            for name, d in ego_met.get("per_class", {}).items():
                print(f"  - {name:<22}: Acc = {d['accuracy']:.4f} (count={d['count']})")
        print("=" * 80 + "\n")

    def get_quick_status(self):
        """Compute quick metrics (mAP, ego acc, worst 3 classes) for real-time train progress.

        Returns:
            dict: Concise tracking status dictionary.
        """
        if len(self.actor_preds) == 0:
            return {}

        all_actor_probs = np.concatenate(self.actor_preds, axis=0)
        all_actor_gts = np.concatenate(self.actor_targets, axis=0)

        ap_list = []
        class_aps = []
        for i in range(self.num_actor_class):
            gt_i = all_actor_gts[:, i]
            if np.sum(gt_i) > 0:
                ap = float(average_precision_score(gt_i, all_actor_probs[:, i]))
                ap_list.append(ap)
                class_aps.append((self.actor_names[i], ap, int(np.sum(gt_i))))

        mean_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
        class_aps.sort(key=lambda x: x[1])
        worst_3 = [f"{name} ({ap:.2f})" for name, ap, _ in class_aps[:3]]

        ego_acc = 0.0
        if len(self.ego_preds) > 0:
            all_ego_probs = np.concatenate(self.ego_preds, axis=0)
            all_ego_gts = np.concatenate(self.ego_targets, axis=0)
            ego_acc = float(np.mean(np.argmax(all_ego_probs, axis=-1) == all_ego_gts))

        return {
            "mAP": mean_ap,
            "ego_acc": ego_acc,
            "worst_classes": worst_3,
        }

    def save_json(self, filepath, summary=None):
        """Save detailed diagnostics as JSON file."""
        if summary is None:
            summary = self.compute_summary()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
