"""Baseline 3D Video Classification Model (e.g. standard X3D without slot attention)."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from core.base_model import BaseActionModel
from core.registry import MODELS


@MODELS.register("x3d")
@MODELS.register("x3d_baseline")
class X3D_Baseline(BaseActionModel):
    """Standard X3D video model baseline."""

    def __init__(
        self,
        num_ego_class: int = 7,
        num_actor_class: int = 64,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class

        x3d_hub = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.blocks = x3d_hub.blocks
        in_feat = 2048

        self.actor_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, num_actor_class),
        )

        if num_ego_class > 0:
            self.ego_head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_feat, num_ego_class),
            )
        else:
            self.ego_head = None

    def forward(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> Dict[str, Optional[torch.Tensor]]:
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0).permute((1, 2, 0, 3, 4))

        # Pass through backbone
        for block in self.blocks[:-1]:
            videos = block(videos)

        # Global average pool: (B, C, 1, 1, 1) -> (B, C)
        pool = self.blocks[-1].pool(videos)
        feat = torch.flatten(pool, 1)

        pred_actor = self.actor_head(feat)
        pred_ego = self.ego_head(feat) if self.ego_head is not None else None

        return {
            "pred_actor": pred_actor,
            "pred_ego": pred_ego,
            "attn": None,
        }
