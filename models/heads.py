"""Action and Ego Classification Heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import HEADS


@HEADS.register("linear_head")
@HEADS.register("head")
class Head(nn.Module):
    """Standard multi-task classification head for ego and actor predictions."""

    def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        if self.num_ego_classes != 0:
            feat_c = ego_channel if ego_channel != 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(feat_c, num_ego_classes)
            )
        else:
            self.fc_ego = None

        self.fc_actor = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel, num_actor_classes)
        )

    def forward(self, x: torch.Tensor, ego_x: torch.Tensor = None, **kwargs):
        """
        Args:
            x: Actor feature tensor (B, dim) or (B, num_slots, dim).
            ego_x: Optional ego feature tensor (B, ego_dim).

        Returns:
            tuple[Optional[Tensor], Tensor]: (pred_ego, pred_actor)
        """
        y_actor = self.fc_actor(x)
        y_ego = None
        if self.fc_ego is not None:
            target_ego = ego_x if ego_x is not None else x
            y_ego = self.fc_ego(target_ego)
        return y_ego, y_actor


@HEADS.register("allocated_head")
class Allocated_Head(nn.Module):
    """Slot-allocated classification head."""

    def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        if self.num_ego_classes != 0:
            feat_c = ego_channel if ego_channel != 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(feat_c, num_ego_classes)
            )
        else:
            self.fc_ego = None

        self.fc_actor = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel, 1)
        )

    def forward(self, x: torch.Tensor, feat_map: torch.Tensor = None, ego_x: torch.Tensor = None, **kwargs):
        # x is (B, num_slots, dim)
        y_actor = self.fc_actor(x).squeeze(-1)  # (B, num_slots)
        y_ego = None
        if self.fc_ego is not None:
            target_ego = ego_x if ego_x is not None else x
            y_ego = self.fc_ego(target_ego)
        return y_ego, y_actor


@HEADS.register("deformable_head")
class DeformableSlotTemporalDetectionHead(nn.Module):
    """Deformable cross-attention and temporal pooling head for action slots."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        H: int = 8,
        W: int = 24,
        num_actor_classes: int = 64,
        num_ego_classes: int = 7,
        num_heads: int = 4,
        num_points: int = 4,
        ego_channel: int = 64,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.H = H
        self.W = W
        self.num_actor_classes = num_actor_classes
        self.num_ego_classes = num_ego_classes
        self.num_heads = num_heads
        self.num_points = num_points

        self.offset_proj = nn.Linear(dim, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc_actor = nn.Linear(dim, 1)

        if num_ego_classes > 0:
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(ego_channel, num_ego_classes)
            )
        else:
            self.fc_ego = None

    def forward(self, slots: torch.Tensor, feat_map: torch.Tensor, ego_x: torch.Tensor = None, **kwargs):
        """Cross-attention between slots and spatiotemporal feature map."""
        B, S, D = slots.shape
        _, T, H, W, _ = feat_map.shape

        # Compute sampling offsets & weights from slot queries
        offsets = self.offset_proj(slots).view(B, S, self.num_heads, self.num_points, 2).tanh()
        weights = F.softmax(self.weight_proj(slots).view(B, S, self.num_heads, self.num_points), dim=-1)

        # Spatially sample from dense feat_map across frames
        V = self.value_proj(feat_map)
        V_pooled = V.mean(dim=1)  # (B, H, W, D)

        base_coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, int(self.num_slots**0.5) if int(self.num_slots**0.5)**2 == self.num_slots else 8),
            torch.linspace(-1, 1, 8),
            indexing='ij'
        ), dim=-1).view(-1, 2)[:S].to(slots.device)

        grid = base_coords.view(1, S, 1, 1, 2) + offsets * 0.2
        grid = grid.view(B, S * self.num_heads * self.num_points, 1, 2)

        V_in = V_pooled.permute(0, 3, 1, 2)
        sampled = F.grid_sample(V_in, grid, align_corners=False, mode='bilinear')
        sampled = sampled.view(B, D, S, self.num_heads, self.num_points).permute(0, 2, 3, 4, 1)

        aggregated = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3))
        slot_features = slots + self.out_proj(aggregated)

        # Classification logits
        y_actor = self.fc_actor(slot_features).squeeze(-1)
        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None
        return y_ego, y_actor
