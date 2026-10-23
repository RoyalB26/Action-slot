"""Action and Ego Classification Heads."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import HEADS


class GradScaleFunction(torch.autograd.Function):
    """Scales gradients flowing backward through this tensor."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale == 0.0:
            return None, None
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply gradient scaling: if scale=0.0, detach; if scale=1.0, identity."""
    if scale == 0.0:
        return x.detach()
    if scale == 1.0:
        return x
    return GradScaleFunction.apply(x, scale)


@HEADS.register("linear_head")
@HEADS.register("head")
class Head(nn.Module):
    """Standard multi-task classification head for ego and actor predictions."""

    def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0, **kwargs):
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

    def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0, **kwargs):
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
    """Deformable cross-attention and temporal pooling head for action slots.

    Integrates:
        1. Gated LayerScale Residual: Keeps aggregated feature magnitude constrained relative to slots.
        2. Backward Gradient Scaling: Restricts gradient flowing from deformable head back into backbone.
        3. Attention-guided Anchoring: Anchors sample points to slot attention spatial centers of mass.
    """

    def __init__(
        self,
        in_channel: int = 256,
        dim: int = None,
        num_slots: int = 64,
        H: int = 8,
        W: int = 24,
        num_actor_classes: int = 64,
        num_ego_classes: int = 7,
        num_heads: int = 4,
        num_points: int = 4,
        ego_channel: int = 64,
        agg_grad_scale: float = 0.1,
        agg_gate_init: float = 0.1,
        disable_agg_gate: bool = False,
        anchor_from_attn: bool = True,
        **kwargs
    ):
        super().__init__()
        dim = dim or in_channel
        self.in_channel = in_channel
        self.num_slots = num_slots
        self.dim = dim
        self.H = H
        self.W = W
        self.num_actor_classes = num_actor_classes
        self.num_ego_classes = num_ego_classes
        self.num_heads = num_heads
        self.num_points = num_points
        self.agg_grad_scale = agg_grad_scale
        self.disable_agg_gate = disable_agg_gate
        self.anchor_from_attn = anchor_from_attn

        self.offset_proj = nn.Linear(dim, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Idea 1: LayerNorm + Gated Residual (LayerScale)
        self.agg_norm = nn.LayerNorm(dim)
        if not self.disable_agg_gate:
            self.gamma = nn.Parameter(torch.full((dim,), float(agg_gate_init)))
            self.slot_gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        else:
            self.gamma = None
            self.slot_gate = None

        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        # Support allocated head with background slot (e.g. num_slots = 65, num_actor_classes = 64)
        if self.num_slots == self.num_actor_classes or self.num_slots == self.num_actor_classes + 1:
            self.fc_actor = nn.Linear(dim, 1)
        else:
            self.fc_actor = nn.Linear(dim, self.num_actor_classes)

        if num_ego_classes > 0:
            # Enhanced ego prediction fusing conv3d_ego feature with holistic deformable slot context
            ego_in_dim = (ego_channel if ego_channel > 0 else 0) + dim
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.1),
                nn.Linear(ego_in_dim, num_ego_classes)
            )
        else:
            self.fc_ego = None

    def compute_anchor_coords(self, attn_masks: Optional[torch.Tensor], B: int, S: int, T_target: int, device: torch.device) -> torch.Tensor:
        """Idea 3 (Spatiotemporal): Anchor sampling grid to slot attention center-of-mass across space per frame.
        
        Returns:
            base_coords: (B, T_target, S, 2) normalized coordinates in [-1, 1].
        """
        if self.anchor_from_attn and attn_masks is not None:
            # attn_masks can be (B, T, S, H, W) or (B, S, T, H, W)
            if attn_masks.ndim == 5:
                if attn_masks.shape[1] == S:
                    # (B, S, T, H, W) -> permute to (B, T, S, H, W)
                    m = attn_masks.permute(0, 2, 1, 3, 4)
                elif attn_masks.shape[2] == S:
                    # (B, T, S, H, W)
                    m = attn_masks
                else:
                    m = None
            else:
                m = None

            if m is not None:
                # Interpolate or pool along time dimension if attention map frame count != T_target
                T_curr = m.shape[1]
                if T_curr != T_target:
                    # Reshape to (B*S, 1, T_curr, H, W)
                    H_m, W_m = m.shape[3], m.shape[4]
                    m_reshaped = m.permute(0, 2, 1, 3, 4).reshape(B * S, 1, T_curr, H_m, W_m)
                    m_reshaped = F.interpolate(m_reshaped, size=(T_target, H_m, W_m), mode='trilinear', align_corners=False)
                    m = m_reshaped.view(B, S, T_target, H_m, W_m).permute(0, 2, 1, 3, 4)

                H_att, W_att = m.shape[3], m.shape[4]
                # Normalize over spatial dimensions per frame: (B, T, S, H, W)
                denom = m.sum(dim=(-2, -1), keepdim=True) + 1e-8
                prob = m / denom

                # Normalized coordinates in [-1, 1]
                xs = torch.linspace(-1.0, 1.0, W_att, device=device)
                ys = torch.linspace(-1.0, 1.0, H_att, device=device)

                # Expectation of X and Y (Center of Mass per frame): (B, T, S)
                cx = (prob.sum(dim=-2) * xs).sum(dim=-1)
                cy = (prob.sum(dim=-1) * ys).sum(dim=-1)

                # Detach coordinates so anchor does not backprop into attention map directly
                base_coords = torch.stack([cx, cy], dim=-1).detach()  # (B, T_target, S, 2)
                return base_coords

        # Fallback to static uniform grid across all frames
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, int(self.num_slots**0.5) if int(self.num_slots**0.5)**2 == self.num_slots else 8),
            torch.linspace(-1, 1, 8),
            indexing='ij'
        ), dim=-1).view(-1, 2)[:S].to(device)
        return base_grid.unsqueeze(0).unsqueeze(0).expand(B, T_target, -1, -1)  # (B, T_target, S, 2)

    def forward(
        self,
        slots: torch.Tensor,
        feat_map: torch.Tensor,
        ego_x: torch.Tensor = None,
        attn: torch.Tensor = None,
        **kwargs
    ):
        """Spatiotemporal cross-attention between slots and feature map across all individual frames."""
        B, S, D = slots.shape
        _, T, H, W, _ = feat_map.shape

        # Idea 2: Scale gradient of feat_map flowing into Deformable Head
        feat_map_scaled = grad_scale(feat_map, self.agg_grad_scale)

        # Compute sampling offsets & weights from slot queries: (B, S, num_heads, num_points, 2)
        offsets = self.offset_proj(slots).view(B, S, self.num_heads, self.num_points, 2).tanh()
        weights = F.softmax(self.weight_proj(slots).view(B, S, self.num_heads, self.num_points), dim=-1)

        # Value projection preserving temporal sequence: (B, T, H, W, D)
        V = self.value_proj(feat_map_scaled)

        # Spatiotemporal anchor coordinates per frame: (B, T, S, 2)
        attn_tensor = attn if attn is not None else kwargs.get("attn_masks", None)
        base_coords = self.compute_anchor_coords(attn_tensor, B, S, T, slots.device)

        # Apply offsets relative to anchor base per frame:
        # base_coords: (B, T, S, 1, 1, 2), offsets: (B, 1, S, num_heads, num_points, 2)
        grid = base_coords.unsqueeze(3).unsqueeze(3) + offsets.unsqueeze(1) * 0.2  # (B, T, S, num_heads, num_points, 2)

        # Flatten B and T to perform efficient parallel 2D bilinear sampling across all frames
        # grid: (B*T, S * num_heads * num_points, 1, 2)
        grid_flat = grid.view(B * T, S * self.num_heads * self.num_points, 1, 2)
        # V_in: (B*T, D, H, W)
        V_in = V.reshape(B * T, H, W, D).permute(0, 3, 1, 2)

        # Sample features at dynamic trajectory locations
        sampled = F.grid_sample(V_in, grid_flat, align_corners=False, mode='bilinear')
        # sampled: (B, T, D, S, num_heads, num_points) -> permute to (B, T, S, num_heads, num_points, D)
        sampled = sampled.view(B, T, D, S, self.num_heads, self.num_points).permute(0, 1, 3, 4, 5, 2)

        # Aggregate across heads and sampling points with attention weights:
        # weights: (B, 1, S, num_heads, num_points, 1)
        sampled_weighted = sampled * weights.unsqueeze(1).unsqueeze(-1)
        aggregated_t = sampled_weighted.sum(dim=(3, 4))  # (B, T, S, D)

        # Temporal pooling: pool feature representation over the 16 time frames
        aggregated = aggregated_t.mean(dim=1)  # (B, S, D)

        # Idea 1: Gated Residual (LayerNorm + Slot Gate + Learnable Gamma scale, or Direct Residual if disabled)
        normed_agg = self.agg_norm(aggregated)
        agg_proj = self.out_proj(normed_agg)
        if not self.disable_agg_gate:
            gate_weights = self.slot_gate(slots)  # Slot decides what context to accept
            scaled_agg = self.gamma * gate_weights * agg_proj
        else:
            scaled_agg = agg_proj

        slot_features = slots + scaled_agg

        # Diagnostics for measuring dominance of aggregated branch vs slots
        with torch.no_grad():
            self.last_slots_norm = float(slots.detach().norm(dim=-1).mean().item())
            self.last_agg_norm = float(scaled_agg.detach().norm(dim=-1).mean().item())
            self.last_agg_ratio = float((self.last_agg_norm / (self.last_slots_norm + 1e-8)))

        # Classification logits
        if self.num_slots == self.num_actor_classes:
            y_actor = self.fc_actor(slot_features).squeeze(-1)  # (B, num_actor_classes)
        elif self.num_slots == self.num_actor_classes + 1:
            # Exclude extra background slot from action classification logits
            action_slots = slot_features[:, :self.num_actor_classes, :]
            y_actor = self.fc_actor(action_slots).squeeze(-1)  # (B, num_actor_classes)
        else:
            y_actor = self.fc_actor(slot_features)  # (B, S, num_actor_classes)
        # Ego prediction: fuse deformable context with ego_x if available, or predict purely from deformable slots
        y_ego = None
        if self.fc_ego is not None:
            # Global pooled context from all enhanced slot features: (B, D)
            slot_ego_context = slot_features.mean(dim=1)
            if ego_x is not None:
                ego_combined = torch.cat([ego_x, slot_ego_context], dim=-1)
            elif self.fc_ego[2].in_features == (self.dim + self.ego_channel) and self.ego_channel > 0:
                # If ego_x is omitted during testing but model expected ego_x, pad zeros
                dummy_ego = torch.zeros(slot_features.shape[0], self.ego_channel, device=slot_features.device, dtype=slot_features.dtype)
                ego_combined = torch.cat([dummy_ego, slot_ego_context], dim=-1)
            else:
                ego_combined = slot_ego_context
            y_ego = self.fc_ego(ego_combined)
        return y_ego, y_actor
