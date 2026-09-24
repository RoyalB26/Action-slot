import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalDeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, dropout=0.1):
        """
        Deformable Attention chuyên biệt cho Video Trajectory trong Action-slot.
        Mỗi token tại frame t sẽ lấy mẫu các điểm lân cận động trên các frame lân cận.
        Args:
            d_model: Số chiều kênh đặc trưng (mặc định 256 của slot_dim).
            n_heads: Số lượng attention heads.
            n_points: Số điểm lấy mẫu trên mỗi frame lân cận.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Ta xét lấy mẫu trên 3 frame cục bộ: [t-1 (quá khứ), t (hiện tại), t+1 (tương lai)]
        self.n_temporal_offsets = 3 

        # Dự đoán 2D spatial offsets (dx, dy) cho mỗi frame lân cận và mỗi điểm mẫu
        self.sampling_offsets = nn.Linear(
            d_model, n_heads * self.n_temporal_offsets * n_points * 2
        )
        # Dự đoán trọng số attention cho các điểm mẫu
        self.attention_weights = nn.Linear(
            d_model, n_heads * self.n_temporal_offsets * n_points
        )
        
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN Block
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Khởi tạo góc offset ban đầu tỏa đều ra các hướng theo vòng tròn
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_temporal_offsets, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, x):
        """
        Args:
            x: Tensor đầu vào dạng [B, T, H, W, D] từ backbone conv3d.
        Returns:
            Tensor cùng kích thước [B, T, H, W, D] đã được làm giàu thông tin quỹ đạo và che khuất.
        """
        B, T, H, W, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        # 1. Sinh Reference Points chuẩn hóa trong đoạn [0, 1] trên lưới (H, W)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=x.device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        ref_pts = torch.stack((ref_x / W, ref_y / H), -1)  # [H, W, 2]
        ref_pts = ref_pts.view(1, 1, H * W, 2).expand(B, T, -1, -1)  # [B, T, H*W, 2]

        # 2. Chiếu Value
        value = self.value_proj(x_norm)  # [B, T, H, W, D]
        # Gom B và T lại để dùng grid_sample 2D cho từng frame
        value = value.view(B * T, H, W, self.n_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        # value: [B * T, n_heads, head_dim, H, W]

        # 3. Dự đoán Offsets và Attention Weights
        x_flat = x_norm.view(B, T, H * W, D)
        offsets = self.sampling_offsets(x_flat).view(
            B, T, H * W, self.n_heads, self.n_temporal_offsets, self.n_points, 2
        )
        weights = self.attention_weights(x_flat).view(
            B, T, H * W, self.n_heads, self.n_temporal_offsets * self.n_points
        )
        weights = F.softmax(weights, dim=-1).view(
            B, T, H * W, self.n_heads, self.n_temporal_offsets, self.n_points
        )

        # Chuẩn hóa offset theo (W, H)
        offset_normalizer = torch.tensor([W, H], dtype=torch.float32, device=x.device)
        sampling_locations = ref_pts[:, :, :, None, None, None, :] + offsets / offset_normalizer

        # 4. Lấy mẫu thông tin qua 3 bước thời gian: t-1, t, t+1
        # Chuyển tọa độ sang [-1, 1] cho F.grid_sample
        sampling_grid = 2.0 * sampling_locations - 1.0  # [B, T, H*W, n_heads, 3, n_points, 2]

        sampled_values = []
        # Định nghĩa các bước nhảy thời gian delta_t tương ứng với 3 mức
        temporal_deltas = [-1, 0, 1]

        # value định hình: [B, T, n_heads, head_dim, H, W]
        value_by_time = value.view(B, T, self.n_heads, self.head_dim, H, W)

        for step_idx, dt in enumerate(temporal_deltas):
            # Clamp để không bị vượt biên thời gian (tại frame đầu t=0 hoặc frame cuối t=T-1)
            target_t = torch.clamp(torch.arange(T, device=x.device) + dt, 0, T - 1)
            # Lấy feature map tại target_t
            v_target = value_by_time[:, target_t] # [B, T, n_heads, head_dim, H, W]
            v_target = v_target.reshape(B * T * self.n_heads, self.head_dim, H, W)

            # Lấy lưới sampling tương ứng với temporal step này
            # [B, T, H*W, n_heads, n_points, 2] -> [B * T * n_heads, H*W, n_points, 2]
            grid_step = sampling_grid[:, :, :, :, step_idx, :, :]
            grid_step = grid_step.permute(0, 1, 3, 2, 4, 5).reshape(B * T * self.n_heads, H * W, self.n_points, 2)

            # Bilinear interpolation
            sampled_val = F.grid_sample(
                v_target, grid_step, mode='bilinear', padding_mode='zeros', align_corners=False
            ) # [B * T * n_heads, head_dim, H*W, n_points]

            sampled_val = sampled_val.view(B, T, self.n_heads, self.head_dim, H * W, self.n_points)
            sampled_val = sampled_val.permute(0, 1, 4, 2, 3, 5) # [B, T, H*W, n_heads, head_dim, n_points]
            sampled_values.append(sampled_val)

        # [B, T, H*W, n_heads, head_dim, 3, n_points]
        sampled_values = torch.stack(sampled_values, dim=-2)

        # 5. Tổng hợp đặc trưng theo trọng số chú ý
        # weights: [B, T, H*W, n_heads, 1, 3, n_points]
        w = weights.unsqueeze(4)
        out = (sampled_values * w).sum(dim=(-2, -1)) # [B, T, H*W, n_heads, head_dim]
        out = out.reshape(B, T, H, W, D)

        out = self.output_proj(out)
        x = residual + self.dropout(out)
        x = x + self.ffn(self.norm2(x))

        return x