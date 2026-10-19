import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
		super(Head, self).__init__()
		self.num_ego_classes = num_ego_classes
		if self.num_ego_classes !=0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
					nn.ReLU(inplace=False),
					nn.Linear(in_channel, num_ego_classes)
					)
			else:
				self.fc_ego = nn.Sequential(
					nn.ReLU(inplace=False),
					nn.Linear(ego_channel, num_ego_classes)
					)
		self.fc_actor = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Linear(in_channel, num_actor_classes)
			)
		
	def forward(self, x, ego_x=None):
		y_ego = None
		y_actor = self.fc_actor(x)
		if self.num_ego_classes != 0:
			if ego_x != None:
				y_ego = self.fc_ego(ego_x)
			else:
				y_ego = self.fc_ego(x)
		return y_ego, y_actor

class DeformableSlotTemporalDetectionHead(nn.Module):
    def __init__(
        self,
        num_slots: int,
        dim: int,
        H: int = 10,
        W: int = 10,
        num_points: int = 4,   # Số điểm lấy mẫu xung quanh mỗi query
        num_heads: int = 4,    # Số attention heads
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.num_slots = num_slots
        self.dim = dim
        self.H = H
        self.W = W
        self.num_queries = H * W
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # ========================================================
        # 1. REFERENCE GRID CHO QUERIES (CHUẨN HÓA [0, 1])
        # x tương ứng với W, y tương ứng với H
        # ========================================================
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0.5 / H, 1.0 - 0.5 / H, H),
            torch.linspace(0.5 / W, 1.0 - 0.5 / W, W),
            indexing="ij",
        )
        # grid shape: [H * W, 2] với định dạng (x, y)
        grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
        self.register_buffer("grid", grid)

        # Query Embeddings mã hóa vị trí không gian
        self.query_embed = nn.Embedding(self.num_queries, dim)

        # ========================================================
        # 2. SLOT-SPECIFIC EMBEDDING & MODULATOR
        # ========================================================
        self.slot_embed = nn.Embedding(num_slots, dim)
        self.slot_modulator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2),  # gamma, beta
        )

        # ========================================================
        # 3. DEFORMABLE ATTENTION ENGINE
        # ========================================================
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

        # ========================================================
        # 4. GOM TỤ KHÔNG GIAN & THỜI GIAN (TIẾT KIỆM THAM SỐ)
        # ========================================================
        # Spatial Attention Pooling: nén H*W queries về 1 vector
        self.spatial_pool_attn = nn.Linear(dim, 1)
        self.spatial_pool_proj = nn.Linear(dim, dim)

        # Temporal Attention Pooling: nén T frames về 1 vector
        self.temporal_pool_attn = nn.Linear(dim, 1)

        # ========================================================
        # 5. OBJECTNESS HEAD (CHIA SẺ CHO TẤT CẢ SLOTS)
        # ========================================================
        self.shared_fc_obj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # 1. Embeddings
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        # 2. Modulator
        nn.init.constant_(self.slot_modulator[-1].weight, 0.0)
        nn.init.constant_(self.slot_modulator[-1].bias, 0.0)

        # 3. Sampling Offsets: Tỏa đều theo các góc hướng
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

        # 4. Attention & Projections
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        # 5. Pooling Projections
        nn.init.xavier_uniform_(self.spatial_pool_attn.weight)
        nn.init.constant_(self.spatial_pool_attn.bias, 0.0)
        nn.init.xavier_uniform_(self.spatial_pool_proj.weight)
        nn.init.constant_(self.spatial_pool_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.temporal_pool_attn.weight)
        nn.init.constant_(self.temporal_pool_attn.bias, 0.0)

    def forward(
        self,
        slot_feat: torch.Tensor,  # [B, S, dim]
        feat_map: torch.Tensor,   # [B, T, W, H, dim]
    ) -> torch.Tensor:
        B, S, _ = slot_feat.shape
        _, T, W, H, _ = feat_map.shape
        device = slot_feat.device
        Q = self.num_queries  # H * W

        # 1. Điều chế Slot Feature
        slot_ids = torch.arange(S, device=device)
        slot_id_embed = self.slot_embed(slot_ids)
        combined_slot = slot_feat + slot_id_embed.unsqueeze(0)  # [B, S, dim]

        gamma, beta = self.slot_modulator(combined_slot).chunk(2, dim=-1)
        modulated_slot = (1.0 + gamma) * combined_slot + beta   # [B, S, dim]

        # 2. Mở rộng Query theo thời gian T: [B, T, S, Q, dim]
        # Query kết hợp giữa thông tin slot và spatial position embedding
        query_spatial = modulated_slot.unsqueeze(2) + self.query_embed.weight.view(1, 1, Q, self.dim) # [B, S, Q, dim]
        query_temporal = query_spatial.unsqueeze(1).expand(B, T, S, Q, self.dim)                       # [B, T, S, Q, dim]
        
        BTS = B * T * S
        query_flat = query_temporal.reshape(BTS, Q, self.dim)  # [B*T*S, Q, dim]

        # 3. Chuẩn bị Reference Points dạng (x, y) trong [0, 1]
        ref_flat = self.grid.view(1, 1, 1, Q, 2).expand(B, T, S, -1, -1).reshape(BTS, Q, 2)

        # 4. Chiếu Value từ feat_map [B, T, W, H, dim] -> [BTS * num_heads, head_dim, H, W]
        # Chuyển đổi (W, H) thành (H, W) chuẩn định dạng ảnh (y, x) cho F.grid_sample
        feat_map_hw = feat_map.permute(0, 1, 3, 2, 4)  # [B, T, H, W, dim]
        value = self.value_proj(feat_map_hw)            # [B, T, H, W, dim]
        value = value.view(B, T, H, W, self.num_heads, self.head_dim)

        # Mở rộng qua các Slot: [B, T, S, H, W, num_heads, head_dim]
        value = value.unsqueeze(2).expand(B, T, S, H, W, self.num_heads, self.head_dim)
        value = value.permute(0, 1, 2, 5, 6, 3, 4).reshape(BTS * self.num_heads, self.head_dim, H, W)

        # 5. Tính Sampling Offsets & Attention Weights
        offsets = self.sampling_offsets(query_flat).view(BTS, Q, self.num_heads, self.num_points, 2)
        spatial_scale = torch.tensor([1.0 / W, 1.0 / H], device=device).view(1, 1, 1, 1, 2)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_scale  # [BTS, Q, heads, points, 2]

        # Đưa về [-1, 1] cho F.grid_sample
        sampling_grid = 2.0 * sampling_locations - 1.0
        grid_permuted = sampling_grid.permute(0, 2, 1, 3, 4).contiguous().view(
            BTS * self.num_heads, Q, self.num_points, 2
        )

        attn_weights = self.attention_weights(query_flat).view(BTS, Q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Bilinear interpolation
        sampled_feat = F.grid_sample(
            value,
            grid_permuted,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [BTS * num_heads, head_dim, Q, num_points]

        sampled_feat = sampled_feat.view(BTS, self.num_heads, self.head_dim, Q, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [BTS, Q, num_heads, num_points, head_dim]

        # Tổng hợp qua các điểm lấy mẫu
        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)  # [BTS, Q, num_heads, head_dim]
        out = self.output_proj(out.view(BTS, Q, self.dim))            # [BTS, Q, dim]

        # 6. Gom tụ không gian (Spatial Attention Pooling)
        spatial_weights = F.softmax(self.spatial_pool_attn(out), dim=1)  # [BTS, Q, 1]
        pooled_spatial = (out * spatial_weights).sum(dim=1)               # [BTS, dim]
        delta_feat = self.spatial_pool_proj(pooled_spatial)               # [BTS, dim]
        delta_feat = delta_feat.view(B, T, S, self.dim)                   # [B, T, S, dim]

        # 7. Gom tụ thời gian (Temporal Attention Pooling qua T frames)
        delta_feat_t = delta_feat.permute(0, 2, 1, 3)                     # [B, S, T, dim]
        temporal_weights = F.softmax(self.temporal_pool_attn(delta_feat_t), dim=2)  # [B, S, T, 1]
        enhanced_delta = (delta_feat_t * temporal_weights).sum(dim=2)               # [B, S, dim]

        # Cộng dư vào slot feature ban đầu
        enhanced_feat = slot_feat + enhanced_delta                        # [B, S, dim]

        # 8. Dự đoán Objectness song song
        obj_logits = self.shared_fc_obj(enhanced_feat).squeeze(-1)         # [B, S]

        return obj_logits



class Allocated_Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
		super(Allocated_Head, self).__init__()
		self.num_ego_classes = num_ego_classes
		self.num_actor_classes = num_actor_classes
		if self.num_ego_classes != 0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(in_channel, num_ego_classes),
		                )
			else:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(ego_channel, num_ego_classes),
		                )
				
		self.fc_actor = DeformableSlotTemporalDetectionHead(num_actor_classes, in_channel, 8, 24, 8, 8)

	def forward(self, x, feat_map, ego_x=None):

		b, n, _ = x.shape
		y_actor = self.fc_actor(x, feat_map)
		y_ego = None
		# x = torch.reshape(x, (b, n))
		if self.num_ego_classes != 0:
			y_ego = self.fc_ego(ego_x)
			
		return y_ego, y_actor
