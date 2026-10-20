"""Loss function definitions for Action-Slot learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionSlotLoss(nn.Module):
    """Multi-task loss module for Action-Slot covering actor and ego actions."""

    def __init__(self, args, num_actor_class, attention_res=None):
        """
        Args:
            args (argparse.Namespace): Hyperparameters containing weights:
                - ce_pos_weight (float): Positive class weight for CrossEntropy.
                - ce_neg_weight (float): Negative/background class weight.
                - bce_pos_weight (float): Positive class weight for BCEWithLogits.
                - ego_loss_weight (float): Loss scaling factor for ego classification.
            num_actor_class (int): Number of actor action classes (64 for TACO).
            attention_res (tuple[int, int], optional): Spatial resolution for attention loss.
        """
        super(ActionSlotLoss, self).__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.attention_res = attention_res
        self.ego_ce = nn.CrossEntropyLoss(reduction="mean")
        self.actor_loss_type = self._parse_actor_loss(args)
        self.attn_loss_type = self._parse_attn_loss(args)

    def _parse_actor_loss(self, args):
        """Parse actor loss type based on whether slots are class-allocated."""
        if ("slot" in args.model_name and not getattr(args, 'allocated_slot', False)) or getattr(args, 'box', False):
            ce_weights = torch.ones(self.num_actor_class + 1) * getattr(args, 'ce_pos_weight', 1.0)
            ce_weights[-1] = getattr(args, 'ce_neg_weight', 0.05)
            self.bce = nn.CrossEntropyLoss(reduction="none", weight=ce_weights)
            return 1
        else:
            pos_weight = torch.ones([self.num_actor_class]) * getattr(args, 'bce_pos_weight', 10.0)
            self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
            return 0

    def _parse_attn_loss(self, args):
        """Determine attention supervision configuration."""
        flag = 0
        if ("slot" in args.model_name and not getattr(args, 'allocated_slot', False)) or (getattr(args, 'box', False) and getattr(args, 'obj_mask', False)):
            flag = 1
        elif "slot" in args.model_name and getattr(args, 'allocated_slot', False):
            if not getattr(args, 'bg_mask', False) and getattr(args, 'action_attn_weight', 0.0) > 0:
                flag = 2
            elif getattr(args, 'obj_mask', False):
                flag = 1
            elif (
                getattr(args, 'bg_slot', False)
                and getattr(args, 'bg_mask', False)
                and getattr(args, 'action_attn_weight', 0.0) > 0.0
                and getattr(args, 'bg_attn_weight', 0.0) > 0.0
            ):
                flag = 3
            elif (
                getattr(args, 'bg_slot', False)
                and getattr(args, 'bg_mask', False)
                and getattr(args, 'bg_attn_weight', 0.0) > 0.0
                and not (getattr(args, 'action_attn_weight', 0.0) > 0.0)
            ):
                flag = 4
        if flag > 0:
            self.obj_bce = nn.BCELoss()
        return flag

    def ego_loss(self, pred, label):
        """Calculate cross-entropy loss for ego-vehicle action.

        Args:
            pred (torch.Tensor or None): Ego action logits of shape (B, num_ego_class).
            label (torch.Tensor): Ground truth ego action indices of shape (B,).

        Returns:
            torch.Tensor or None: Mean cross-entropy loss scalar.
        """
        if pred is None:
            return None
        return self.ego_ce(pred, label)

    def actor_loss(self, pred, label):
        """Calculate loss for multi-label actor actions.

        Args:
            pred (torch.Tensor): Actor action logits of shape (B, num_actor_class)
                or (B, num_slots, num_actor_class + 1).
            label (torch.Tensor): Binary ground truth of shape (B, num_actor_class).

        Returns:
            torch.Tensor: Mean actor loss scalar.
        """
        if self.actor_loss_type == 1:
            # If pred is (B, num_slots, num_actor_class + 1)
            if pred.ndim == 3:
                # Aggregate across slots: take max over slot dimension for the foreground classes
                pred = pred.max(dim=1)[0]
            if pred.shape[-1] > self.num_actor_class:
                pred = pred[:, :self.num_actor_class]
            pos_weight = torch.ones([self.num_actor_class], device=pred.device) * getattr(self.args, 'bce_pos_weight', 10.0)
            loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=pos_weight)
            return loss
        else:
            loss = self.bce(pred, label)
            return torch.mean(loss)

    def forward(self, pred_ego, pred_actor, label_ego, label_actor):
        """Compute total weighted loss.

        Args:
            pred_ego (torch.Tensor or None): Ego logits of shape (B, num_ego_class).
            pred_actor (torch.Tensor): Actor logits of shape (B, num_actor_class).
            label_ego (torch.Tensor): Ego target indices of shape (B,).
            label_actor (torch.Tensor): Actor multi-label targets of shape (B, num_actor_class).

        Returns:
            tuple[torch.Tensor, torch.Tensor or None, torch.Tensor]:
                - total_loss: Sum of actor and weighted ego loss.
                - loss_ego: Ego cross-entropy loss scalar.
                - loss_actor: Actor classification loss scalar.
        """
        loss_actor = self.actor_loss(pred_actor, label_actor)
        loss_ego = self.ego_loss(pred_ego, label_ego)
        if loss_ego is not None:
            total_loss = loss_actor + getattr(self.args, 'ego_loss_weight', 0.5) * loss_ego
        else:
            total_loss = loss_actor
        return total_loss, loss_ego, loss_actor
