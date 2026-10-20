"""Classification and detection heads for ego-vehicle and other actor action prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Standard linear classification head for ego and actor predictions."""

    def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
        """
        Args:
            in_channel (int): Dimension of incoming actor feature vector.
            num_ego_classes (int): Number of ego action classes (0 if unused).
            num_actor_classes (int): Number of other actor action classes.
            ego_channel (int): Input feature dimension for ego classification (if different).
        """
        super(Head, self).__init__()
        self.num_ego_classes = num_ego_classes
        if self.num_ego_classes != 0:
            feat_c = ego_channel if ego_channel != 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(feat_c, num_ego_classes)
            )
        self.fc_actor = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel, num_actor_classes)
        )

    def forward(self, x, ego_x=None):
        """
        Args:
            x (torch.Tensor): Actor feature tensor of shape (B, dim) or (B, num_slots, dim).
            ego_x (torch.Tensor, optional): Ego feature tensor of shape (B, ego_channel).

        Returns:
            tuple[torch.Tensor or None, torch.Tensor]:
                - y_ego: Ego logits of shape (B, num_ego_classes) or None.
                - y_actor: Actor logits of shape (B, num_actor_classes) or (B, num_slots, num_actor_classes).
        """
        y_ego = None
        y_actor = self.fc_actor(x)
        if self.num_ego_classes != 0:
            target_ego = ego_x if ego_x is not None else x
            y_ego = self.fc_ego(target_ego)
        return y_ego, y_actor


class DeformableSlotTemporalDetectionHead(nn.Module):
    """Deformable cross-attention and temporal pooling head for action slots."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        H: int = 8,
        W: int = 24,
        num_points: int = 8,
        num_heads: int = 8,
    ):
        """
        Args:
            num_slots (int): Number of action slots (S).
            dim (int): Slot and feature embedding dimension.
            H (int): Feature map height.
            W (int): Feature map width.
            num_points (int): Number of sampling points per attention head.
            num_heads (int): Number of multi-head deformable attention heads.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_slots = num_slots
        self.dim = dim
        self.H = H
        self.W = W
        self.num_points = num_points
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_embed = nn.Embedding(num_slots, dim)
        self.slot_embed = nn.Embedding(num_slots, dim)

        self.slot_modulator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.ref_point_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Sigmoid(),
        )

        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

        self.spatial_pool_attn = nn.Linear(dim, 1)
        self.spatial_pool_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.temporal_pool_attn = nn.Linear(dim, 1)
        self.shared_fc_obj = nn.Linear(dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize projection and sampling offset parameters."""
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        nn.init.constant_(self.slot_modulator[-1].weight, 0.0)
        nn.init.constant_(self.slot_modulator[-1].bias, 0.0)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.spatial_pool_attn.weight)
        nn.init.constant_(self.spatial_pool_attn.bias, 0.0)
        nn.init.xavier_uniform_(self.spatial_pool_proj.weight)
        nn.init.constant_(self.spatial_pool_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.temporal_pool_attn.weight)
        nn.init.constant_(self.temporal_pool_attn.bias, 0.0)

    def forward(
        self,
        slot_feat: torch.Tensor,
        feat_map: torch.Tensor,
    ) -> torch.Tensor:
        """Deformable cross-attention and spatiotemporal feature aggregation.

        Args:
            slot_feat (torch.Tensor): Extracted slot features of shape (B, S, dim).
            feat_map (torch.Tensor): Dense spatiotemporal feature volume of shape (B, T, H, W, dim).

        Returns:
            torch.Tensor: Actor classification logits of shape (B, S).
        """
        B, S, _ = slot_feat.shape
        _, T, H, W, _ = feat_map.shape
        device = slot_feat.device

        # Generate slot context and modulated query: shape (B, S, dim)
        slot_idx = torch.arange(S, device=device)
        slot_emb = self.slot_embed(slot_idx).unsqueeze(0).expand(B, -1, -1)
        slot_ctx = self.slot_modulator(slot_feat + slot_emb)

        query = self.query_embed(slot_idx).unsqueeze(0).expand(B, -1, -1) + slot_ctx
        reference_points = self.ref_point_head(query)  # (B, S, 2)

        # Expand across temporal frames: (B, T, S, dim) and (B, T, S, 2)
        query_t = query.unsqueeze(1).expand(-1, T, -1, -1)
        ref_t = reference_points.unsqueeze(1).expand(-1, T, -1, -1)

        BTS = B * T
        Q = S
        query_flat = query_t.contiguous().view(BTS, Q, self.dim)
        ref_flat = ref_t.contiguous().view(BTS, Q, 2)

        # Value projection: (B*T, num_heads, head_dim, H, W)
        feat_flat = feat_map.contiguous().view(BTS, H, W, self.dim)
        value = self.value_proj(feat_flat)
        value = value.view(BTS, H, W, self.num_heads, self.head_dim)
        value = value.permute(0, 3, 4, 1, 2).contiguous().view(BTS * self.num_heads, self.head_dim, H, W)

        # Predict deformable offsets: (B*T, S, num_heads, num_points, 2)
        offsets = self.sampling_offsets(query_flat).view(BTS, Q, self.num_heads, self.num_points, 2)
        spatial_scale = torch.tensor([1.0 / W, 1.0 / H], device=device).view(1, 1, 1, 1, 2)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_scale

        # Normalize grid to [-1, 1] for grid_sample: (B*T * num_heads, Q, num_points, 2)
        sampling_grid = 2.0 * sampling_locations - 1.0
        grid_permuted = sampling_grid.permute(0, 2, 1, 3, 4).contiguous().view(
            BTS * self.num_heads, Q, self.num_points, 2
        )

        attn_weights = self.attention_weights(query_flat).view(BTS, Q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Bilinear sampling: (BTS * num_heads, head_dim, Q, num_points)
        sampled_feat = F.grid_sample(
            value,
            grid_permuted,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        sampled_feat = sampled_feat.view(BTS, self.num_heads, self.head_dim, Q, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # (BTS, Q, num_heads, num_points, head_dim)

        # Aggregate across points: (BTS, Q, dim)
        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = self.output_proj(out.view(BTS, Q, self.dim))

        # Spatial attention pooling: (BTS, dim)
        spatial_weights = F.softmax(self.spatial_pool_attn(out), dim=1)
        pooled_spatial = (out * spatial_weights).sum(dim=1)
        delta_feat = self.spatial_pool_proj(pooled_spatial).view(B, T, S, self.dim)

        # Temporal attention pooling: (B, S, dim)
        delta_feat_t = delta_feat.permute(0, 2, 1, 3)
        temporal_weights = F.softmax(self.temporal_pool_attn(delta_feat_t), dim=2)
        enhanced_delta = (delta_feat_t * temporal_weights).sum(dim=2)

        # Residual update: (B, S, dim)
        enhanced_feat = slot_feat + enhanced_delta
        # Output action prediction logits: (B, S)
        obj_logits = self.shared_fc_obj(enhanced_feat).squeeze(-1)
        return obj_logits


class Allocated_Head(nn.Module):
    """Allocated head coupling ego classification with Deformable Action-Slot detection."""

    def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
        """
        Args:
            in_channel (int): Dimension of incoming slot feature vectors.
            num_ego_classes (int): Number of ego action classes.
            num_actor_classes (int): Number of actor action classes (corresponds to num_slots).
            ego_channel (int): Input feature dimension for ego classification branch.
        """
        super(Allocated_Head, self).__init__()
        self.num_ego_classes = num_ego_classes
        self.num_actor_classes = num_actor_classes
        if self.num_ego_classes != 0:
            feat_c = ego_channel if ego_channel != 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(feat_c, num_ego_classes),
            )
        self.fc_actor = DeformableSlotTemporalDetectionHead(num_actor_classes, in_channel, 8, 24, 8, 8)

    def forward(self, x, feat_map, ego_x=None):
        """
        Args:
            x (torch.Tensor): Slot features of shape (B, num_actor_classes, in_channel).
            feat_map (torch.Tensor): Feature volume of shape (B, T, H, W, in_channel).
            ego_x (torch.Tensor, optional): Pooled ego feature tensor of shape (B, ego_channel).

        Returns:
            tuple[torch.Tensor or None, torch.Tensor]:
                - y_ego: Ego action logits of shape (B, num_ego_classes) or None.
                - y_actor: Actor action logits of shape (B, num_actor_classes).
        """
        y_actor = self.fc_actor(x, feat_map)
        y_ego = None
        if self.num_ego_classes != 0 and ego_x is not None:
            y_ego = self.fc_ego(ego_x)
        return y_ego, y_actor
