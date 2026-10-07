import math
import numpy as np
from ptflops import get_model_complexity_info
from pytorchvideo.models.hub import csn_r101, i3d_r50, mvit_base_16x4
import r50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =========================================================================
# 1. TEMPORAL CSWIN ATTENTION (Dải sọc thời gian - không gian trực giao)
# =========================================================================


class SpatioTemporalCSWinBlock(nn.Module):

  def __init__(self, dim, num_heads=4):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.half_heads = num_heads // 2
    self.head_dim = dim // num_heads
    self.scale = self.head_dim**-0.5

    self.qkv = nn.Linear(dim, dim * 3)
    self.proj = nn.Linear(dim, dim)
    self.norm = nn.LayerNorm(dim)

  def forward(self, x):
    """x: [B, T, H, W, C]"""
    B, T, H, W, C = x.shape
    residual = x
    x = self.norm(x)

    qkv = (
        self.qkv(x)
        .reshape(B, T, H, W, 3, self.num_heads, self.head_dim)
        .permute(4, 0, 5, 1, 2, 3, 6)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, H, W, head_dim]

    # --- Nhánh 1: Sọc ngang theo thời gian (T x W) trên H heads đầu ---
    q_w = (
        q[:, : self.half_heads]
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B * self.half_heads * H, T * W, self.head_dim)
    )
    k_w = (
        k[:, : self.half_heads]
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B * self.half_heads * H, T * W, self.head_dim)
    )
    v_w = (
        v[:, : self.half_heads]
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B * self.half_heads * H, T * W, self.head_dim)
    )

    attn_w = (
        torch.bmm(q_w, k_w.transpose(-1, -2)) * self.scale
    ).softmax(  # [B * half_heads * H, TW, TW]
        dim=-1
    )
    out_w = torch.bmm(attn_w, v_w).reshape(
        B, self.half_heads, H, T, W, self.head_dim
    )
    out_w = out_w.permute(0, 3, 2, 4, 1, 5)  # [B, T, H, W, half_heads, head_dim]

    # --- Nhánh 2: Sọc dọc theo thời gian (T x H) trên H heads sau ---
    q_h = (
        q[:, self.half_heads :]
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(B * self.half_heads * W, T * H, self.head_dim)
    )
    k_h = (
        k[:, self.half_heads :]
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(B * self.half_heads * W, T * H, self.head_dim)
    )
    v_h = (
        v[:, self.half_heads :]
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(B * self.half_heads * W, T * H, self.head_dim)
    )

    attn_h = (
        torch.bmm(q_h, k_h.transpose(-1, -2)) * self.scale
    ).softmax(  # [B * half_heads * W, TH, TH]
        dim=-1
    )
    out_h = torch.bmm(attn_h, v_h).reshape(
        B, self.half_heads, W, T, H, self.head_dim
    )
    out_h = out_h.permute(0, 3, 4, 2, 1, 5)  # [B, T, H, W, half_heads, head_dim]

    # Ghép lại
    out = torch.cat([out_w, out_h], dim=-2).reshape(B, T, H, W, C)
    return residual + self.proj(out)


# =========================================================================
# 2. TEMPORAL ATTENTION POOLING (Chống pha loãng đặc trưng khi xuất hiện trễ)
# =========================================================================


class TemporalAttentionPooling(nn.Module):

  def __init__(self, in_channel):
    super().__init__()
    self.score_net = nn.Sequential(
        nn.Linear(in_channel, in_channel // 2),
        nn.Tanh(),
        nn.Linear(in_channel // 2, 1),
    )

  def forward(self, slot_trajectories):
    """slot_trajectories: [B, num_slots, T, C] Trả về: pooled_feat: [B,

    num_slots, C], temporal_weights: [B, num_slots, T]
    """
    scores = self.score_net(slot_trajectories)  # [B, num_slots, T, 1]
    temporal_weights = F.softmax(scores, dim=2)  # [B, num_slots, T, 1]
    pooled_feat = (slot_trajectories * temporal_weights).sum(
        dim=2
    )  # [B, num_slots, C]
    return pooled_feat, temporal_weights.squeeze(-1)


# =========================================================================
# 3. POSITION EMBEDDING VÀ ALLOCATED HEAD NÂNG CẤP
# =========================================================================


def build_3d_grid(resolution):
  ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
  grid = torch.meshgrid(*ranges, indexing="ij")
  grid = torch.stack(grid, dim=-1)
  grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
  grid = grid.unsqueeze(0)
  return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed3D(nn.Module):

  def __init__(self, hidden_size, resolution):
    super().__init__()
    self.embedding = nn.Linear(6, hidden_size, bias=True)
    self.register_buffer("grid", build_3d_grid(resolution))

  def forward(self, inputs):
    grid = self.embedding(self.grid)
    return inputs + grid


class Upgraded_Allocated_Head(nn.Module):

  def __init__(
      self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0
  ):
    super().__init__()
    self.num_ego_classes = num_ego_classes
    self.num_actor_classes = num_actor_classes

    if self.num_ego_classes != 0:
      c_in = ego_channel if ego_channel != 0 else in_channel
      self.fc_ego = nn.Sequential(
          nn.ReLU(inplace=False), nn.Linear(c_in, num_ego_classes)
      )

    self.fc_actor = nn.ModuleList([
        nn.Sequential(nn.ReLU(inplace=False), nn.Linear(in_channel, 1))
        for _ in range(num_actor_classes)
    ])
    self.temp_pool = TemporalAttentionPooling(in_channel)

  def forward(self, x, ego_x=None):
    """x có thể là [B, num_actor, C] hoặc [B, num_actor, T, C]"""
    temp_weights = None
    if x.dim() == 4:
      x, temp_weights = self.temp_pool(x)  # [B, num_actor, C]

    b, n, _ = x.shape
    y_actor = []
    y_ego = None
    for i in range(self.num_actor_classes):
      y_actor.append(self.fc_actor[i](x[:, i, :]))

    y_actor = torch.stack(y_actor, dim=0).permute((1, 0, 2)).reshape(b, n)
    if self.num_ego_classes != 0:
      y_ego = self.fc_ego(ego_x)

    return y_ego, y_actor, temp_weights


# =========================================================================
# 4. SLOT ATTENTION TÍCH HỢP QUY TRÌNH TRACKING
# =========================================================================


class SlotAttention(nn.Module):

  def __init__(
      self,
      num_slots,
      dim,
      num_actor_class=64,
      eps=1e-8,
      input_dim=64,
      resolution=[16, 8, 24],
      allocated_slot=True,
  ):
    super().__init__()
    self.dim = dim
    self.num_slots = num_slots
    self.num_actor_class = num_actor_class
    self.allocated_slot = allocated_slot
    self.eps = eps
    self.scale = dim**-0.5
    self.resolution = resolution

    # Learnable Slot Init
    self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).cuda()
    self.slots_sigma = torch.randn(1, 1, dim).cuda()
    self.slots_sigma = nn.Parameter(self.slots_sigma.absolute())

    self.FC1 = nn.Linear(dim, dim)
    self.FC2 = nn.Linear(dim, dim)
    self.LN = nn.LayerNorm(dim)

    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, dim)

    self.norm_input = nn.LayerNorm(dim)
    self.norm_slots = nn.LayerNorm(dim)
    self.norm_pre_ff = nn.LayerNorm(dim)

    self.pe = SoftPositionEmbed3D(
        dim, [resolution[0], resolution[1], resolution[2]]
    )

    mu = self.slots_mu.expand(1, self.num_slots, -1)
    sigma = self.slots_sigma.expand(1, self.num_slots, -1)
    self.register_buffer("slots", torch.normal(mu, sigma).contiguous())

  def forward(self, inputs):
    """inputs: [B, T, H, W, C]"""
    b, t, h, w, d = inputs.shape
    inputs = self.pe(inputs)

    # CSWin Feature qua LN và FC
    inputs = self.LN(inputs.reshape(b, -1, d))
    inputs = self.FC2(F.relu(self.FC1(inputs)))
    inputs = self.norm_input(inputs)

    k, v = self.to_k(inputs), self.to_v(inputs)
    slots = self.slots.expand(b, -1, -1)
    slots_norm = self.norm_slots(slots)
    q = self.to_q(slots_norm)

    dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
    attn_ori = dots.softmax(dim=1) + self.eps  # Cạnh tranh Softmax giữa các slot
    attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

    # Cập nhật đặc trưng slot
    slots_update = torch.einsum("bjd,bij->bid", v, attn)
    if self.allocated_slot:
      slots_update = slots_update[:, : self.num_actor_class, :]
    else:
      slots_update = slots_update[:, : self.num_slots, :]

    slots_out = slots_update + self.fc2(
        F.relu(self.fc1(self.norm_pre_ff(slots_update)))
    )

    # Tái tạo slot trajectory theo trục T: [B, num_slots, T, C]
    # Phân rã attn_ori: [B, S, T, H*W]
    attn_5d = attn_ori.view(b, -1, t, h * w)
    v_5d = v.view(b, t, h * w, d)
    # Gom đặc trưng riêng từng frame cho mỗi slot
    slot_trajectories = torch.einsum(
        "bstp,btpd->bstd", attn_5d, v_5d
    )  # [B, S, T, C]
    if self.allocated_slot:
      slot_trajectories = slot_trajectories[:, : self.num_actor_class, :, :]

    return slots_out, slot_trajectories, attn_ori


# =========================================================================
# 5. ACTION_SLOT TỔNG HỢP VỚI MODULE BÙ TRỪ VẬN TỐC XE CHỦ & CHỈ SỐ STATS
# =========================================================================


class ACTION_SLOT(nn.Module):

  def __init__(
      self,
      args,
      num_ego_class,
      num_actor_class,
      num_slots=64,
      box=False,
      videomae=None,
  ):
    super(ACTION_SLOT, self).__init__()
    self.args = args
    self.hidden_dim = args.channel
    self.hidden_dim2 = args.channel
    self.slot_dim = args.channel
    self.num_ego_class = num_ego_class
    self.num_actor_class = num_actor_class
    self.ego_c = 128
    self.num_slots = num_slots

    self.resnet = i3d_r50(True)
    if args.backbone == 'r50':
        self.resnet = r50.R50()
        self.in_c = 2048
        if args.dataset == 'taco':
            self.resolution = (8, 24)
            self.resolution3d = (args.seq_len, 5, 5)
        elif args.dataset == 'oats':
            self.resolution = (7, 7)
            self.resolution3d = (args.seq_len, 7, 7)

    elif args.backbone == 'i3d':
        self.resnet = self.resnet.blocks[:-1]
        self.in_c = 2048
        if args.dataset == 'taco':
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)
        elif args.dataset == 'oats':
            self.resolution = (7, 7)
            self.resolution3d = (4, 7, 7)

    elif args.backbone == 'x3d':
        self.resnet = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.resnet = self.resnet.blocks[:-1]
        self.in_c = 192
        
        if (args.dataset == 'oats' or args.pretrain == 'oats') and args.pretrain != 'taco':
            self.resolution = (7, 7)
            self.resolution3d = (16, 7, 7)
        else:
            self.resolution = (8, 24)
            self.resolution3d = (16, 8, 24)
        
    elif args.backbone == 'slowfast':
        self.resnet = torch.hub.load('facebookresearch/pytorchvideo:main', 'slowfast_r50', pretrained=True)
        self.resnet = self.resnet.blocks[:-2]
        self.path_pool = nn.AdaptiveAvgPool3d((4, 8, 24))
        self.in_c = 2304
        if args.dataset == 'oats':
            self.resolution = (7, 7)
            self.resolution3d = (4, 7, 7)
        else:
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)

    # 1. Nhánh Ego-motion
    if self.num_ego_class != 0:
      self.conv3d_ego = nn.Sequential(
          nn.ReLU(),
          nn.BatchNorm3d(self.in_c),
          nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
      )
      # Ego-Motion Compensation Projection (Chiếu vận tốc xe chủ để bù trừ optical flow)
      self.ego_compensator = nn.Sequential(
          nn.Linear(self.ego_c, self.hidden_dim2),
          nn.Tanh(),
      )

    # 2. Nhánh Feature Spatio-temporal Conv
    self.conv3d = nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm3d(self.in_c),
        nn.Conv3d(self.in_c, self.hidden_dim2, (1, 1, 1), stride=1),
        nn.ReLU(),
    )

    # 3. Spatio-Temporal CSWin Attention
    self.cswin = SpatioTemporalCSWinBlock(self.hidden_dim2, num_heads=4)

    # 4. Slot Attention
    self.slot_attention = SlotAttention(
        num_slots=self.num_slots + (1 if args.bg_slot else 0),
        dim=self.slot_dim,
        input_dim=self.hidden_dim2,
        resolution=self.resolution3d,
        num_actor_class=num_actor_class,
        allocated_slot=args.allocated_slot,
    )

    # 5. Upgraded Classification Head
    self.head = Upgraded_Allocated_Head(
        self.slot_dim, num_ego_class, num_actor_class, self.ego_c
    )
    self.drop = nn.Dropout(p=0.5)
    self.pool = nn.AdaptiveAvgPool3d(output_size=1)

  def forward(self, x, box=False):
    seq_len = len(x)
    batch_size = x[0].shape[0]

    # --- Backbone Forward ---
    if isinstance(x, list):
      x = torch.stack(x, dim=0)
      x = x.permute((1, 2, 0, 3, 4))  # [B, C, T, H, W]

    for i in range(len(self.resnet)):
      x = self.resnet[i](x)

    x = self.drop(x)
    ego_x = None
    if self.num_ego_class != 0:
      ego_feat = self.conv3d_ego(x)
      ego_x = self.pool(ego_feat).reshape(batch_size, self.ego_c)

    # Giảm chiều kênh
    x = self.conv3d(x)  # [B, C, T, H, W]
    B, C, T, H, W = x.shape

    # --- Ego-Motion Compensation ---
    if ego_x is not None:
      ego_bias = self.ego_compensator(ego_x).view(
          B, C, 1, 1, 1
      )  # Vector bù trừ vận tốc
      x = x - ego_bias  # Khử rung lắc và trôi dạt do xe mình

    # Chuẩn bị vào CSWin: [B, T, H, W, C]
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    x = self.cswin(x)

    # --- Slot Attention ---
    slots_out, slot_trajectories, raw_attn = self.slot_attention(x)

    # --- Head Forward (Tích hợp Temporal Attention Pooling) ---
    ego_x_drop = self.drop(ego_x) if ego_x is not None else None
    y_ego, y_actor, temp_weights = self.head(slot_trajectories, ego_x_drop)

    # =========================================================================
    # TÍNH TOÁN CÁC THÔNG SỐ TRACKING DIAGNOSTICS (METRICS ĐÁNH GIÁ THỰC NGHIỆM)
    # =========================================================================
    with torch.no_grad():
      # raw_attn: [B, S, T*H*W] -> [B, S, T, H, W]
      attn_spatial = raw_attn[:, : self.num_actor_class].view(
          B, self.num_actor_class, T, H, W
      )

      # 1. Spatial Center Entropy (Đo mức độ dồn cục vào tâm ngã tư)
      # Tâm ngã tư: H in [2, 6], W in [6, 18]
      center_mass = attn_spatial[:, :, :, 2:6, 6:18].sum(dim=(-2, -1))
      total_mass = attn_spatial.sum(dim=(-2, -1)) + 1e-6
      center_ratio = (
          (center_mass / total_mass).mean().item()
      )  # Càng thấp (<0.40) chứng tỏ đã thoát bẫy ngã tư

      # 2. Early Frame Responsiveness (Tỷ lệ năng lượng chú ý ở 4 frame đầu)
        # Chuẩn hóa ma trận của từng slot về tổng = 1 trên toàn bộ không-thời gian
      slot_attn_norm = attn_spatial / (
        attn_spatial.sum(dim=(2, 3, 4), keepdim=True) + 1e-6
      )
        # Lúc này năng lượng 4 frame đầu của từng slot mới phản ánh độ nhạy thực sự:
      early_energy = (
        slot_attn_norm[:, :, :4].sum(dim=(2, 3, 4)).mean().item()
      )  

      # 3. Temporal Attention Peak Index (Khám phá frame nào được chú ý nhất cho mỗi slot)
      # temp_weights: [B, num_actor_class, T]
      peak_frames = temp_weights.argmax(dim=-1).float().mean().item()

      tracking_stats = {
          'center_bias_ratio': center_ratio,  # Nếu > 0.70 là bị bẫy ngã tư; mong muốn ~ 0.35 - 0.45
          'early_frame_energy': early_energy,  # Nếu < 0.10 là bị mù frame đầu; mong muốn ~ 0.20 - 0.30
          'avg_peak_frame': peak_frames,  # Cho biết model tập trung nhất ở frame thứ mấy
        #   'temp_weights': (
        #       temp_weights.detach()
        #   ),  # Dùng để visualize biểu đồ frame 1->16
      }

    # Nội suy attn_masks về kích thước ban đầu để trả về
    attn_masks = raw_attn.view(B, -1, T, H, W).permute(0, 2, 1, 3, 4)
    if seq_len > T:
      attn_masks = attn_masks.permute(0, 2, 1, 3, 4).reshape(B * -1, 1, T, H, W)
      attn_masks = F.interpolate(
          attn_masks,
          size=(seq_len, self.resolution[0], self.resolution[1]),
          mode='trilinear',
          align_corners=False,
      )
      attn_masks = attn_masks.view(
          B, -1, seq_len, self.resolution[0], self.resolution[1]
      ).permute(0, 2, 1, 3, 4)

    if self.num_ego_class != 0:
      return ego_x, y_actor, attn_masks, tracking_stats
    else:
      return y_actor, attn_masks, tracking_stats