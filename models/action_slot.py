"""Action-Slot model architecture composed of modular Backbone, SlotAttention, and Head."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base_model import BaseActionModel
from core.registry import MODELS, BACKBONES, HEADS
from models.slot_attention import SlotAttention


@MODELS.register("action_slot")
class ActionSlotModel(BaseActionModel):
    """Spatiotemporal Action-Slot model.

    Modularly connects:
        1. Backbone (X3D, VideoSwin, etc.)
        2. Spatiotemporal Projection & Ego Branch
        3. Slot Attention (Iterative Object/Action feature clustering)
        4. Multi-task Classification Head (Linear, Allocated, Deformable)
    """

    def __init__(
        self,
        num_ego_class: int = 7,
        num_actor_class: int = 64,
        backbone_name: str = "x3d",
        head_name: str = "linear_head",
        num_slots: int = 64,
        slot_dim: int = 256,
        ego_c: int = 64,
        bg_slot: bool = False,
        resolution3d: tuple = (16, 8, 24),
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.ego_c = ego_c
        self.head_name = head_name

        # 1. Modular Backbone
        bb_kwargs = {
            "freeze_stages": kwargs.get("freeze_stages", 3),
            "freeze_all": kwargs.get("freeze_all", False),
            "unfreeze_all": kwargs.get("unfreeze_all", False),
        }
        self.backbone = BACKBONES.build(backbone_name, **bb_kwargs)
        in_c = self.backbone.out_channels

        # 2. Ego Feature Branch
        if self.num_ego_class > 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(in_c),
                nn.Conv3d(in_c, self.ego_c, (1, 1, 1), stride=1),
            )
            self.pool_ego = nn.AdaptiveAvgPool3d(output_size=1)
        else:
            self.conv3d_ego = None
            self.pool_ego = None

        # 3. Spatiotemporal Feature Projection
        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, self.slot_dim, (1, 1, 1), stride=1),
            nn.ReLU(),
        )

        # 4. Slot Attention
        actual_slots = self.num_slots + 1 if bg_slot else self.num_slots
        self.slot_attention = SlotAttention(
            num_slots=actual_slots,
            dim=self.slot_dim,
            eps=1e-8,
            input_dim=self.slot_dim,
            resolution=list(resolution3d),
            num_actor_class=num_actor_class,
        )

        # 5. Modular Classification Head
        actor_classes_in_head = num_actor_class if head_name == "allocated_head" else (num_actor_class + 1)
        self.head = HEADS.build(
            head_name,
            in_channel=self.slot_dim,
            num_ego_classes=self.num_ego_class,
            num_actor_classes=actor_classes_in_head,
            ego_channel=self.ego_c,
        )

        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Unified forward pass returning standard dictionary."""
        seq_len = len(videos) if isinstance(videos, list) else videos.shape[2]
        batch_size = videos[0].shape[0] if isinstance(videos, list) else videos.shape[0]

        # 1. Backbone feature extraction: (B, C, T_feat, H_feat, W_feat)
        feats = self.backbone(videos)

        # 2. Ego feature branch
        ego_x = None
        if self.conv3d_ego is not None:
            ego_feat = self.conv3d_ego(feats)
            ego_feat = self.pool_ego(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))

        new_seq_len = feats.shape[2]
        new_h, new_w = feats.shape[3], feats.shape[4]

        # 3. Project to slot dimension: (B, T_feat, H_feat, W_feat, slot_dim)
        x = self.conv3d(feats)
        x = x.permute((0, 2, 3, 4, 1))
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))
        feat_map = x

        # 4. Slot Attention
        slots, attn_masks = self.slot_attention(x)

        # 5. Attention map upsampling back to input frame count
        b, n, _ = attn_masks.shape
        attn_masks = attn_masks.view(b * n, 1, new_seq_len, new_h, new_w)
        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode='trilinear')
        attn_masks = torch.reshape(attn_masks, (b, n, seq_len, new_h, new_w))
        attn_masks = attn_masks.permute((0, 2, 1, 3, 4))  # (B, T, num_slots, H, W)

        # 6. Apply Dropout & Head
        slots = self.drop(slots)
        if ego_x is not None:
            ego_x = self.drop(ego_x)

        pred_ego, pred_actor = self.head(slots, feat_map=feat_map, ego_x=ego_x)

        return {
            "pred_actor": pred_actor,
            "pred_ego": pred_ego,
            "attn": attn_masks,
        }
