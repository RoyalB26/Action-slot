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

class DeformableSlotDetectionHead3D(nn.Module):
    def __init__(
        self,
        num_slots: int,
        dim: int,
        T: int = 4,
        W: int = 10,
        H: int = 10,
        num_points: int = 16,   # Số điểm lấy mẫu 3D quanh mỗi query
        num_heads: int = 8,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.num_slots = num_slots
        self.dim = dim
        self.T = T
        self.W = W
        self.H = H
        self.num_queries = T * W * H
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # ========================================================
        # 1. 3D REFERENCE GRID (CHUẨN HÓA [0, 1] CHO X, Y, T)
        # ========================================================
        t_coords = torch.linspace(0.5 / T, 1.0 - 0.5 / T, T)
        x_coords = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        y_coords = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)

        # Meshgrid thứ tự t, x, y
        t_grid, x_grid, y_grid = torch.meshgrid(t_coords, x_coords, y_coords, indexing="ij")
        
        # Grid format: [x, y, t] tương ứng với trục [W, H, T] để khớp grid_sample
        grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1), t_grid.reshape(-1)], dim=-1)  # [Q, 3]
        self.register_buffer("grid", grid)

        eps = 1e-5
        grid_clamped = grid.clamp(min=eps, max=1.0 - eps)
        grid_logits = torch.log(grid_clamped / (1.0 - grid_clamped))
        self.register_buffer("grid_logits", grid_logits)

        # Query Embeddings cho T * W * H vị trí không-thời gian
        self.query_embed = nn.Embedding(self.num_queries, dim)

        # ========================================================
        # 2. SLOT-SPECIFIC EMBEDDING & MODULATOR
        # ========================================================
        self.slot_embed = nn.Embedding(num_slots, dim)
        self.slot_modulator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2)  # gamma, beta
        )
        # Scale kích thước hộp 3D (w, h, duration)
        self.slot_box_scale = nn.Parameter(torch.zeros(num_slots, 3))

        # ========================================================
        # 3. 3D DEFORMABLE ATTENTION ENGINE
        # ========================================================
        self.value_proj = nn.Linear(dim, dim)

        # Offsets 3D gồm: delta_x, delta_y, delta_t -> nhân 3
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 3)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

        # ========================================================
        # 4. PREDICTION HEADS
        # ========================================================
        self.out_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.num_queries
        )

        self.fc_obj_slots = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, 1)
            )
            for _ in range(num_slots)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        nn.init.constant_(self.slot_modulator[-1].weight, 0.0)
        nn.init.constant_(self.slot_modulator[-1].bias, 0.0)

        # Khởi tạo 3D Sampling Offsets: phân bổ các hướng theo hình cầu Fibonacci
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        indices = torch.arange(0, self.num_heads, dtype=torch.float32) + 0.5
        phi = torch.arccos(1.0 - 2.0 * indices / self.num_heads)
        theta = math.pi * (1.0 + 5.0 ** 0.5) * indices

        # Tọa độ mặt cầu đơn vị [num_heads, 3] -> (x, y, t)
        unit_offsets = torch.stack([
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi)
        ], dim=-1)

        grid_init = (
            unit_offsets.view(self.num_heads, 1, 1, 3)
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

    def forward(
        self,
        slot_feat: torch.Tensor, # [B, S, dim]
        feat_map: torch.Tensor   # [B, T, W, H, dim]
    ):
        B, S, _ = slot_feat.shape
        _, T, W, H, C = feat_map.shape
        assert C == self.dim, f"feat_map dim ({C}) không khớp module dim ({self.dim})"
        device = slot_feat.device
        BS = B * S

        # 1. Slot Modulation
        slot_ids = torch.arange(S, device=device)
        slot_id_embed = self.slot_embed(slot_ids)
        combined_slot = slot_feat + slot_id_embed.unsqueeze(0)

        gamma, beta = self.slot_modulator(combined_slot).chunk(2, dim=-1)
        modulated_slot = (1.0 + gamma) * combined_slot + beta

        # 2. Spatiotemporal Query Features: [BS, Q, dim]
        query_feat = modulated_slot.unsqueeze(2) + self.query_embed.weight.view(1, 1, self.num_queries, -1)
        query_flat = query_feat.view(BS, self.num_queries, self.dim)

        # 3. Reference Grid 3D: [BS, Q, 3]
        ref_flat = self.grid.view(1, 1, self.num_queries, 3).expand(B, S, -1, -1).reshape(BS, self.num_queries, 3)

        # 4. Chuẩn bị 5D Value Tensor cho Trilinear Sampling
        # feat_map gốc: [B, T, W, H, dim] -> Permute thành [B, dim, T, H, W]
        feat_map_5d = feat_map.permute(0, 4, 1, 3, 2).contiguous() # [B, dim, T, H, W]
        value = self.value_proj(feat_map_5d.view(B, self.dim, -1).transpose(1, 2)).transpose(1, 2)
        value = value.view(B, self.num_heads, self.head_dim, T, H, W)
        
        # Mở rộng cho số lượng slots S -> [BS * num_heads, head_dim, T, H, W]
        value = (
            value.unsqueeze(1)
            .expand(-1, S, -1, -1, -1, -1, -1)
            .contiguous()
            .view(BS * self.num_heads, self.head_dim, T, H, W)
        )

        # 5. 3D Sampling Offsets & Attention Weights
        offsets = self.sampling_offsets(query_flat).view(BS, self.num_queries, self.num_heads, self.num_points, 3)
        
        # Scale không gian - thời gian chuẩn hóa: (scale_x, scale_y, scale_t)
        spatial_temp_scale = torch.tensor([1.0 / W, 1.0 / H, 1.0 / T], device=device).view(1, 1, 1, 1, 3)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_temp_scale

        # Chuẩn hóa về [-1, 1] phục vụ F.grid_sample
        sampling_grid = 2.0 * sampling_locations - 1.0

        # grid_sample 5D nhận input grid shape: [N, D_out, H_out, W_out, 3]
        # Tại đây gán: D_out = Q, H_out = num_points, W_out = 1
        grid_permuted = (
            sampling_grid.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(BS * self.num_heads, self.num_queries, self.num_points, 1, 3)
        )

        attn_weights = self.attention_weights(query_flat).view(BS, self.num_queries, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1) # [BS, Q, num_heads, P]

        # 6. Trilinear Grid Sample 3D
        # sampled_feat shape: [BS * num_heads, head_dim, Q, num_points, 1]
        sampled_feat = F.grid_sample(
            value,
            grid_permuted,
            mode="bilinear",        # Trong volumetric sampling 5D, mode="bilinear" hoạt động như trilinear
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1)

        sampled_feat = sampled_feat.view(BS, self.num_heads, self.head_dim, self.num_queries, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [BS, Q, num_heads, num_points, head_dim]

        # 7. Aggregation & Projection
        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)  # [BS, Q, num_heads, head_dim]
        out = self.output_proj(out.view(BS, self.num_queries, self.dim)).view(B, S, self.num_queries, self.dim)

        # Gộp đặc trưng qua Conv1D
        out = out.permute(0, 1, 3, 2).reshape(BS, self.dim, self.num_queries)
        out = self.out_conv(out).squeeze(-1).view(B, S, self.dim)
        enhanced_feat = slot_feat + out  # [B, S, dim]

        # 8. Dự đoán Objectness
        obj_logits_list = [self.fc_obj_slots[s](enhanced_feat[:, s]) for s in range(self.num_slots)]
        obj_logits = torch.cat(obj_logits_list, dim=-1)  # [B, S]

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
				
		self.fc_actor = DeformableSlotDetectionHead3D(num_actor_classes, in_channel, 16, 24, 8)

	def forward(self, x, feat_map, ego_x=None):

		b, n, _ = x.shape
		y_actor = self.fc_actor(x, feat_map)
		y_ego = None
		# x = torch.reshape(x, (b, n))
		if self.num_ego_classes != 0:
			y_ego = self.fc_ego(ego_x)
			
		return y_ego, y_actor
