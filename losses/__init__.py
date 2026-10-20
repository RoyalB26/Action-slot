"""Modular loss functions with registration and standardized dict inputs."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from core.registry import LOSSES
from utils.calculation_support import inter_and_union


class BaseLoss(nn.Module):
    """Base class for modular loss functions receiving prediction and target dictionaries."""

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        validate: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss.

        Args:
            preds: Model outputs containing 'pred_actor', 'pred_ego', 'attn'.
            targets: Targets containing 'target_actor', 'target_ego', and optional masks.
            validate: Whether running during validation evaluation.

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
        head_name: str = "linear_head",
        **kwargs
    ):
        super().__init__()
        self.num_actor_class = num_actor_class
        self.ego_loss_weight = ego_loss_weight
        self.allocated_slot = allocated_slot or (head_name == "allocated_head")
        self.bce_pos_weight = bce_pos_weight

        self.ego_ce = nn.CrossEntropyLoss(reduction="mean")

        if self.allocated_slot:
            pos_weight = torch.ones([num_actor_class]) * bce_pos_weight
            self.actor_bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        else:
            self.actor_bce = None

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        validate: bool = False,
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

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ego_weight: float = 0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ego_weight = ego_weight
        self.ego_ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets, validate: bool = False, **kwargs):
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


@LOSSES.register("better_action_slot_loss")
class BetterActionSlotLoss(BaseLoss):
    """Refactored BetterActionSlotLoss with Hungarian matching and attention mask supervision.

    Fully compatible with the modular registry and standard BaseLoss (preds, targets) interface.
    """

    def __init__(
        self,
        args=None,
        num_actor_class: int = 64,
        attention_res: Optional[Tuple[int, int]] = None,
        bce_pos_weight: float = 10.0,
        ce_pos_weight: float = 1.0,
        ce_neg_weight: float = 0.05,
        ego_loss_weight: float = 0.5,
        allocated_slot: bool = False,
        head_name: str = "linear_head",
        **kwargs
    ):
        super().__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.attention_res = attention_res or (8, 24)
        self.ego_loss_weight = ego_loss_weight

        # Hyperparameters with robust defaults
        self.bce_pos_weight = getattr(args, 'bce_pos_weight', bce_pos_weight)
        self.ce_pos_weight = getattr(args, 'ce_pos_weight', ce_pos_weight)
        self.ce_neg_weight = getattr(args, 'ce_neg_weight', ce_neg_weight)
        self.action_attn_weight = getattr(args, 'action_attn_weight', 0.0)
        self.bg_attn_weight = getattr(args, 'bg_attn_weight', 0.0)
        self.seq_len = getattr(args, 'seq_len', 16)
        self.mask_every_frame = getattr(args, 'mask_every_frame', 2)
        self.bg_upsample = getattr(args, 'bg_upsample', 4)

        self.allocated_slot = allocated_slot or getattr(args, 'allocated_slot', False) or (head_name == "allocated_head")
        self.model_name = getattr(args, 'model_name', 'action_slot')
        self.box = getattr(args, 'box', False)
        self.obj_mask = getattr(args, 'obj_mask', False)
        self.bg_mask = getattr(args, 'bg_mask', False)
        self.bg_slot = getattr(args, 'bg_slot', False)

        self.ego_ce = nn.CrossEntropyLoss(reduction='mean')
        self.actor_loss_type = self._parse_actor_loss()
        self.attn_loss_type = self._parse_attn_loss()

    def _parse_actor_loss(self):
        """Parse actor loss type based on whether slots are class-allocated or Hungarian matched."""
        if (('slot' in self.model_name and not self.allocated_slot) or self.box):
            ce_weights = torch.ones(self.num_actor_class + 1) * self.ce_pos_weight
            ce_weights[-1] = self.ce_neg_weight
            self.actor_ce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights)
            return 1
        else:
            pos_weight = torch.ones([self.num_actor_class]) * self.bce_pos_weight
            self.actor_bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
            return 0

    def _parse_attn_loss(self):
        """Determine attention supervision configuration."""
        flag = 0
        if (('slot' in self.model_name and not self.allocated_slot) or self.box) and self.obj_mask:
            flag = 1
        elif 'slot' in self.model_name and self.allocated_slot:
            if not self.bg_mask and self.action_attn_weight > 0:
                flag = 2
            elif self.obj_mask:
                flag = 1
            elif self.bg_slot and self.bg_mask and self.action_attn_weight > 0.0 and self.bg_attn_weight > 0.0:
                flag = 3
            elif self.bg_slot and self.bg_mask and self.bg_attn_weight > 0.0 and not (self.action_attn_weight > 0.0):
                flag = 4

        if flag > 0:
            self.obj_bce = nn.BCELoss()
        return flag

    def ego_loss(self, pred_ego: Optional[torch.Tensor], target_ego: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pred_ego is None or target_ego is None:
            return None
        return self.ego_ce(pred_ego, target_ego)

    def actor_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate loss for multi-label actor actions (with Hungarian matching if type == 1)."""
        if self.actor_loss_type == 1 and pred.ndim == 3 and not self.allocated_slot:
            bs, num_queries = pred.shape[:2]
            out_prob = pred.clone().detach().flatten(0, 1).softmax(-1)
            actor_gt_np = label.clone().detach()

            # Format targets for bipartite matching
            sizes = []
            tgt_ids_list = []
            for b_idx in range(bs):
                pos_indices = torch.nonzero(actor_gt_np[b_idx]).squeeze(-1)
                sizes.append(len(pos_indices))
                if len(pos_indices) > 0:
                    tgt_ids_list.append(pos_indices)

            if len(tgt_ids_list) > 0:
                tgt_ids = torch.cat(tgt_ids_list)
                C = -out_prob[:, tgt_ids].clone().detach()
                C = C.view(bs, num_queries, -1).cpu()

                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                indx = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

                batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)])
                src_idx = torch.cat([src for (src, _) in indx])
                idx = (batch_idx, src_idx)

                target_classes_o = torch.cat([tgt_ids_list[i][J] for i, (_, J) in enumerate(indx)]).to(pred.device)
                target_classes = torch.full(
                    pred.shape[:2],
                    self.num_actor_class,
                    dtype=torch.int64,
                    device=pred.device
                )
                target_classes[idx] = target_classes_o
                return self.actor_ce(pred.transpose(1, 2), target_classes)
            else:
                # No positive actor actions present in batch: supervise all queries with background class
                target_classes = torch.full(
                    pred.shape[:2],
                    self.num_actor_class,
                    dtype=torch.int64,
                    device=pred.device
                )
                return self.actor_ce(pred.transpose(1, 2), target_classes)
        else:
            # Multi-label BCE
            if pred.ndim == 3:
                pred = pred.max(dim=1)[0]
            if pred.shape[-1] > self.num_actor_class:
                pred = pred[:, :self.num_actor_class]

            pos_weight = torch.ones([self.num_actor_class], device=pred.device) * self.bce_pos_weight
            return F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=pos_weight)

    def attn_loss(self, attn: Optional[torch.Tensor], targets: Dict[str, torch.Tensor], validate: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        res = {'attn_loss': None, 'bg_attn_loss': None}
        if attn is None or self.attn_loss_type == 0:
            return res

        device = attn.device
        b, l, n, h, w = attn.shape

        bg_seg = None
        if self.bg_attn_weight > 0 and 'bg_seg' in targets:
            bg_seg_in = targets['bg_seg']
            bg_seg_list = []
            stride = max(1, self.mask_every_frame)
            for i in range(self.seq_len // stride):
                if i < len(bg_seg_in):
                    bg_seg_list.append(bg_seg_in[i].to(device, dtype=torch.float32))
            if bg_seg_list:
                bg_seg = torch.stack(bg_seg_list, 0).permute((1, 0, 2, 3))  # [b, len, h, w]

        action_attn = None
        bg_attn = None

        if self.attn_loss_type == 2:
            action_attn = attn[:, :, :self.num_actor_class, :, :]
            if 'target_actor' in targets:
                class_idx = (targets['target_actor'] == 0.0)
                class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w).permute((0, 2, 1, 3, 4))
                attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32, device=device)
                res['attn_loss'] = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

        elif self.attn_loss_type == 3:
            action_attn = attn[:, :, :self.num_actor_class, :, :]
            stride = max(1, self.mask_every_frame)
            bg_attn = attn[:, ::stride, -1, :, :].reshape(b, -1, h, w)

            if 'target_actor' in targets:
                class_idx = (targets['target_actor'] == 0.0)
                class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w).permute((0, 2, 1, 3, 4))
                attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32, device=device)
                res['attn_loss'] = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

            if bg_seg is not None and bg_attn is not None:
                res['bg_attn_loss'] = self.obj_bce(bg_attn, bg_seg)

        elif self.attn_loss_type == 4:
            stride = max(1, self.mask_every_frame)
            bg_attn = attn[:, ::stride, -1, :, :].reshape(b, -1, h, w)
            if bg_seg is not None:
                res['bg_attn_loss'] = self.obj_bce(bg_attn, bg_seg)

        return res

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        validate: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute unified loss returning total_loss and loss breakdown dictionary."""
        loss_dict = {}

        # 1. Ego loss
        pred_ego = preds.get('pred_ego')
        target_ego = targets.get('target_ego')
        loss_ego = self.ego_loss(pred_ego, target_ego)
        if loss_ego is not None:
            loss_dict['loss_ego'] = loss_ego

        # 2. Actor loss
        pred_actor = preds.get('pred_actor')
        target_actor = targets.get('target_actor')
        loss_actor = self.actor_loss(pred_actor, target_actor)
        loss_dict['loss_actor'] = loss_actor

        # 3. Attention supervision loss
        attn_map = preds.get('attn')
        attn_losses = self.attn_loss(attn_map, targets, validate=validate)

        # 4. Total weighted loss
        total_loss = loss_actor
        if loss_ego is not None:
            total_loss = total_loss + self.ego_loss_weight * loss_ego

        if attn_losses.get('attn_loss') is not None:
            loss_dict['attn_loss'] = attn_losses['attn_loss']
            total_loss = total_loss + self.action_attn_weight * attn_losses['attn_loss']

        if attn_losses.get('bg_attn_loss') is not None:
            loss_dict['bg_attn_loss'] = attn_losses['bg_attn_loss']
            total_loss = total_loss + self.bg_attn_weight * attn_losses['bg_attn_loss']

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


def build_loss(args, num_actor_class: int = 64):
    """Factory builder for loss module based on args."""
    loss_name = getattr(args, 'loss_name', 'action_slot_loss')
    loss_kwargs = {
        "args": args,
        "num_actor_class": num_actor_class,
        "bce_pos_weight": getattr(args, 'bce_pos_weight', 10.0),
        "ce_pos_weight": getattr(args, 'ce_pos_weight', 1.0),
        "ce_neg_weight": getattr(args, 'ce_neg_weight', 0.05),
        "ego_loss_weight": getattr(args, 'ego_loss_weight', 0.5),
        "head_name": getattr(args, 'head_name', 'linear_head'),
        "allocated_slot": getattr(args, 'allocated_slot', False),
    }
    return LOSSES.build(loss_name, **loss_kwargs)
