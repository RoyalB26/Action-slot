"""Modular loss functions with registration and standardized dict inputs."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import LOSSES


class BaseLoss(nn.Module):
    """Base class for modular loss functions receiving prediction and target dictionaries."""

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss.

        Args:
            preds: Model outputs containing 'pred_actor', 'pred_ego', 'attn'.
            targets: Targets containing 'target_actor', 'target_ego'.

        Returns:
            Tuple of:
                - total_loss (torch.Tensor): Scalar loss for backpropagation.
                - loss_dict (Dict[str, torch.Tensor]): Detailed loss components for tracking.
        """
        raise NotImplementedError


@LOSSES.register("action_slot_loss")
@LOSSES.register("default")
class ActionSlotLoss(BaseLoss):
    """Standard multi-task loss for Action-Slot (BCE / CE for actor + CrossEntropy for ego)."""

    def __init__(
        self,
        num_actor_class: int = 64,
        bce_pos_weight: float = 10.0,
        ce_pos_weight: float = 1.0,
        ce_neg_weight: float = 0.05,
        ego_loss_weight: float = 0.5,
        allocated_slot: bool = False,
    ):
        super().__init__()
        self.num_actor_class = num_actor_class
        self.ego_loss_weight = ego_loss_weight
        self.allocated_slot = allocated_slot
        self.bce_pos_weight = bce_pos_weight

        self.ego_ce = nn.CrossEntropyLoss(reduction="mean")

        if allocated_slot:
            pos_weight = torch.ones([num_actor_class]) * bce_pos_weight
            self.actor_bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        else:
            self.actor_bce = None

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_actor = preds.get("pred_actor")
        pred_ego = preds.get("pred_ego")

        target_actor = targets.get("target_actor")
        target_ego = targets.get("target_ego")

        loss_dict = {}

        # 1. Compute Actor Multi-label Loss
        if pred_actor.ndim == 3:
            # (B, num_slots, num_classes) -> pool over slot dimension
            pred_actor = pred_actor.max(dim=1)[0]
        if pred_actor.shape[-1] > self.num_actor_class:
            pred_actor = pred_actor[:, :self.num_actor_class]

        pos_weight = torch.ones([self.num_actor_class], device=pred_actor.device) * self.bce_pos_weight
        loss_actor = F.binary_cross_entropy_with_logits(
            pred_actor,
            target_actor.float(),
            pos_weight=pos_weight
        )
        loss_dict["loss_actor"] = loss_actor

        # 2. Compute Ego-Action Loss
        if pred_ego is not None and target_ego is not None:
            loss_ego = self.ego_ce(pred_ego, target_ego)
            loss_dict["loss_ego"] = loss_ego
            total_loss = loss_actor + self.ego_loss_weight * loss_ego
        else:
            total_loss = loss_actor

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict


@LOSSES.register("focal_loss")
class FocalActionLoss(BaseLoss):
    """Focal Loss variant for addressing severe class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ego_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ego_weight = ego_weight
        self.ego_ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets, **kwargs):
        pred_actor = preds["pred_actor"]
        if pred_actor.ndim == 3:
            pred_actor = pred_actor.max(dim=1)[0]
        target_actor = targets["target_actor"].float()

        p = torch.sigmoid(pred_actor)
        ce_loss = F.binary_cross_entropy_with_logits(pred_actor, target_actor, reduction="none")
        p_t = p * target_actor + (1 - p) * (1 - target_actor)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()

        loss_dict = {"loss_actor": focal_loss}
        if preds.get("pred_ego") is not None and targets.get("target_ego") is not None:
            loss_ego = self.ego_ce(preds["pred_ego"], targets["target_ego"])
            loss_dict["loss_ego"] = loss_ego
            total_loss = focal_loss + self.ego_weight * loss_ego
        else:
            total_loss = focal_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict


def build_loss(args, num_actor_class: int = 64):
    """Factory builder for loss module based on args."""
    loss_name = getattr(args, 'loss_name', 'action_slot_loss')
    loss_kwargs = {
        "num_actor_class": num_actor_class,
        "bce_pos_weight": getattr(args, 'bce_pos_weight', 10.0),
        "ce_pos_weight": getattr(args, 'ce_pos_weight', 1.0),
        "ce_neg_weight": getattr(args, 'ce_neg_weight', 0.05),
        "ego_loss_weight": getattr(args, 'ego_loss_weight', 0.5),
        "allocated_slot": getattr(args, 'allocated_slot', False),
    }
    return LOSSES.build(loss_name, **loss_kwargs)
