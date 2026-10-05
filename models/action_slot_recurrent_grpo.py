from math import ceil
import numpy as np
from ptflops import get_model_complexity_info
from pytorchvideo.models.hub import csn_r101, i3d_r50, mvit_base_16x4
import r50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from classifier import Allocated_Head, Head


# -------------------------------------------------------------
# 1. 3D Spatial Position Embedding cho từng chunk 4 frames
# -------------------------------------------------------------
def build_3d_grid(resolution):
  ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
  grid = torch.meshgrid(*ranges, indexing="ij")
  grid = torch.stack(grid, dim=-1)
  grid = torch.reshape(
      grid, [resolution[0], resolution[1], resolution[2], -1]
  )
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


# -------------------------------------------------------------
# 2. Recurrent Slot Attention xử lý theo từng Micro-Clip (chunk 4 frames)
# -------------------------------------------------------------
class RecurrentChunkSlotAttention(nn.Module):

  def __init__(
      self,
      num_slots,
      dim,
      num_actor_class=64,
      eps=1e-8,
      chunk_resolution=[4, 8, 24],
      allocated_slot=True,
  ):
    super().__init__()
    self.dim = dim
    self.num_slots = num_slots
    self.num_actor_class = num_actor_class
    self.allocated_slot = allocated_slot
    self.eps = eps
    self.scale = dim**-0.5
    self.chunk_resolution = chunk_resolution

    # Prior distribution để sample initial slots S_0
    self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
    self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

    # Projection cho inputs (f_k)
    self.FC1 = nn.Linear(dim, dim)
    self.FC2 = nn.Linear(dim, dim)
    self.LN = nn.LayerNorm(dim)

    # Slot Attention Projections
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)

    self.norm_input = nn.LayerNorm(dim)
    self.norm_slots = nn.LayerNorm(dim)
    self.norm_pre_ff = nn.LayerNorm(dim)

    # Cập nhật recurrent qua GRU
    self.gru = nn.GRUCell(dim, dim)
    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, dim)

    self.pe = SoftPositionEmbed3D(dim, chunk_resolution)

  def sample_initial_slots(self, batch_size, num_groups=1):
        mu = self.slots_mu.expand(batch_size * num_groups, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size * num_groups, self.num_slots, -1)

        if self.training and num_groups > 1:
            eps = torch.randn_like(mu)
            slots = mu + eps * sigma
            dist = torch.distributions.Normal(mu, sigma)
            
            log_prob = dist.log_prob(slots).mean(dim=[-1, -2])  # [B * G]
        else:
            slots = mu
            log_prob = torch.zeros(batch_size * num_groups, device=mu.device)

        return slots, log_prob

  def forward_chunk(self, slots_prev, chunk_inputs):
    b, f, h, w, d = chunk_inputs.shape
    chunk_inputs = self.pe(chunk_inputs)
    chunk_inputs = torch.reshape(chunk_inputs, (b, -1, d))

    inputs = self.LN(chunk_inputs)
    inputs = self.FC1(inputs)
    inputs = F.relu(inputs)
    inputs = self.FC2(inputs)

    inputs = self.norm_input(inputs)
    k, v = self.to_k(inputs), self.to_v(inputs)

    slots_norm = self.norm_slots(slots_prev)
    q = self.to_q(slots_norm)

    dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
    attn_ori = dots.softmax(dim=1) + self.eps
    attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
    updates = torch.einsum("bjd,bij->bid", v, attn)

    # Cập nhật GRU tuần tự
    slots_flat = slots_prev.reshape(-1, d)
    updates_flat = updates.reshape(-1, d)
    slots_next = self.gru(updates_flat, slots_flat)
    slots_next = slots_next.reshape(b, self.num_slots, d)

    slots_next = slots_next + self.fc2(
        F.relu(self.fc1(self.norm_pre_ff(slots_next)))
    )

    # GIỮ NGUYÊN num_slots xuyên suốt chu kỳ, KHÔNG slice ở đây
    return slots_next, attn_ori


# -------------------------------------------------------------
# 3. ACTION_SLOT_RECURRENT ĐẦY ĐỦ VÀ CHUẨN HOÁ
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from classifier import Head, Allocated_Head
from pytorchvideo.models.hub import i3d_r50, csn_r101, mvit_base_16x4
import r50


class ACTION_SLOT_RECURRENT(nn.Module):
    def __init__(
        self,
        args,
        num_ego_class,
        num_actor_class,
        num_slots=21,
        chunk_size=4,
        num_groups=4,
    ):
        super().__init__()
        self.args = args
        self.chunk_size = chunk_size
        self.num_groups = num_groups
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        self.ego_c = 128
        self.hidden_dim = args.channel
        self.hidden_dim2 = args.channel
        self.slot_dim = args.channel
        self.num_slots = num_slots

        if (
            args.dataset == "nuscenes"
            and args.pretrain == "oats"
            and "nuscenes" not in args.cp
        ):
            self.num_actor_class = 35
        if args.dataset == "nuscenes" and args.pretrain == "oats":
            self.num_slots = 35
        if args.dataset == "oats" and args.pretrain == "taco":
            self.num_slots = 64

        # Backbone Setup
        self.resnet = i3d_r50(True)
        if args.backbone == "r50":
            self.resnet = r50.R50()
            self.in_c = 2048
            if args.dataset == "taco":
                self.resolution = (8, 24)
            elif args.dataset == "oats":
                self.resolution = (7, 7)
        elif args.backbone == "i3d":
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            if args.dataset == "taco":
                self.resolution = (8, 24)
            elif args.dataset == "oats":
                self.resolution = (7, 7)
        elif args.backbone == "x3d":
            self.resnet = torch.hub.load(
                "facebookresearch/pytorchvideo:main",
                "x3d_m",
                pretrained=True,
            )
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 192
            self.resolution = (
                (7, 7)
                if (args.dataset == "oats" or args.pretrain == "oats")
                and args.pretrain != "taco"
                else (8, 24)
            )
        elif args.backbone == "slowfast":
            self.resnet = torch.hub.load(
                "facebookresearch/pytorchvideo:main",
                "slowfast_r50",
                pretrained=True,
            )
            self.resnet = self.resnet.blocks[:-2]
            self.path_pool = nn.AdaptiveAvgPool3d((4, 8, 24))
            self.in_c = 2304
            self.resolution = (7, 7) if args.dataset == "oats" else (8, 24)

        # Resolution cho chunk 4 frames
        self.chunk_resolution = [chunk_size, self.resolution[0], self.resolution[1]]

        # Head
        if args.allocated_slot:
            self.head = Allocated_Head(
                self.slot_dim, num_ego_class, self.num_actor_class, self.ego_c
            )
        else:
            self.head = Head(
                self.slot_dim, num_ego_class, self.num_actor_class + 1, self.ego_c
            )

        # Conv 3D
        if self.num_ego_class != 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
            )

        if args.backbone == "r50":
            self.conv3d = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.in_c // 2, (1, 1, 1), stride=1),
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c // 2),
                nn.Conv3d(
                    self.in_c // 2,
                    self.in_c // 2,
                    (3, 3, 3),
                    stride=1,
                    padding="same",
                ),
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c // 2),
                nn.Conv3d(
                    self.in_c // 2,
                    self.in_c // 2,
                    (3, 3, 3),
                    stride=1,
                    padding="same",
                ),
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c // 2),
                nn.Conv3d(self.in_c // 2, self.hidden_dim2, (1, 1, 1), stride=1),
                nn.ReLU(),
            )
        else:
            self.conv3d = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.hidden_dim2, (1, 1, 1), stride=1),
                nn.ReLU(),
            )

        self.slot_attention = RecurrentChunkSlotAttention(
            num_slots=self.num_slots,
            dim=self.slot_dim,
            num_actor_class=self.num_actor_class,
            chunk_resolution=self.chunk_resolution,
            allocated_slot=args.allocated_slot,
        )

        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def setup_stage(self, stage=1):
        """
        stage 1: Supervised Warmup (Học toàn bộ bằng Actor Supervised Loss, không bật GRPO)
        stage 2: GRPO Fine-tuning (Đóng băng Backbone và Classifier Head, chỉ tối ưu Slot Policy qua GRPO)
        """
        if stage == 1:
            # Mở khóa toàn bộ mô hình để học đặc trưng cơ bản
            for p in self.parameters():
                p.requires_grad = True

        elif stage == 2:
            # 1. Đóng băng Backbone và 3D Conv (giữ nguyên feature trích xuất)
            for p in self.resnet.parameters():
                p.requires_grad = False
            for p in self.conv3d.parameters():
                p.requires_grad = False
            if hasattr(self, "conv3d_ego"):
                for p in self.conv3d_ego.parameters():
                    p.requires_grad = False

            # 2. Đóng băng Head phân loại (Classifier) để Head đóng vai trò Reward Evaluator cố định
            for p in self.head.parameters():
                p.requires_grad = False

            # 3. Mở khóa Slot Attention & GRU để tối ưu hóa quỹ đạo phân bổ slot
            for p in self.slot_attention.parameters():
                p.requires_grad = True

    def forward(self, x, num_groups=None):
        """x: Danh sách T tensors từ dataloader [T, B, C, H, W]

        num_groups: Số nhóm G (GRPO Rollouts). Khi Val/Test sẽ mặc định = 1.
        """
        if num_groups is None:
            num_groups = self.num_groups if self.training else 1

        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]

        # --- 1. TRÍCH XUẤT FEATURE QUA BACKBONE ---
        if self.args.backbone == "r50":
            x = torch.stack(x, dim=0)  # [T, B, C, H, W]
            x = torch.reshape(x, (seq_len * batch_size, 3, height, width))
            x = self.resnet(x)
            _, c, h, w = x.shape
            x = torch.reshape(x, (self.args.seq_len, batch_size, c, h, w))
            x = x.permute(1, 2, 0, 3, 4)  # [B, C, T, H, W]

        elif self.args.backbone == "slowfast":
            slow_x = [x[i] for i in range(0, seq_len, 4)]
            x = torch.stack(x, dim=0).permute((1, 2, 0, 3, 4))
            slow_x = torch.stack(slow_x, dim=0).permute((1, 2, 0, 3, 4))
            x = [slow_x, x]
            for i in range(len(self.resnet)):
                x = self.resnet[i](x)
            x[1] = self.path_pool(x[1])
            x = torch.cat((x[0], x[1]), dim=1)

        else:
            x = torch.stack(x, dim=0).permute((1, 2, 0, 3, 4))
            for i in range(len(self.resnet)):
                x = self.resnet[i](x)

        x = self.drop(x)

        # Ego Feature
        ego_x = None
        if self.num_ego_class != 0:
            ego_feat = self.conv3d_ego(x)
            ego_feat = self.pool(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))

        # Conv 3D Feature Map
        x = self.conv3d(x)  # [B, hidden_dim2, T_feat, H_feat, W_feat]
        x = x.permute((0, 2, 3, 4, 1))  # [B, T_feat, H_feat, W_feat, C]
        B, T_feat, H_feat, W_feat, C = x.shape

        assert (
            T_feat % self.chunk_size == 0
        ), f"T_feat ({T_feat}) phải chia hết cho chunk_size ({self.chunk_size})"

        # --- 2. CHIA CHUNKS 4 FRAMES & MỞ RỘNG G NHÓM ---
        chunks = torch.split(x, self.chunk_size, dim=1)
        expanded_chunks = [
            chunk.unsqueeze(1)
            .repeat(1, num_groups, 1, 1, 1, 1)
            .view(B * num_groups, self.chunk_size, H_feat, W_feat, C)
            for chunk in chunks
        ]

        # Khởi tạo S_0 cho G nhóm
        slots_t, initial_log_probs = self.slot_attention.sample_initial_slots(
            batch_size=B, num_groups=num_groups
        )

        attns_list = []
        # --- 3. RECURRENT ATTENTION TUẦN TỰ ---
        for f_k in expanded_chunks:
            slots_t, attn = self.slot_attention.forward_chunk(slots_t, f_k)
            attns_list.append(attn)

        # Cắt slot theo actor class sau khi hoàn thành toàn bộ chu kỳ
        if self.args.allocated_slot:
            final_slots = slots_t[:, : self.num_actor_class, :]
        else:
            final_slots = slots_t[:, : self.num_slots, :]

        final_slots = self.drop(final_slots)

        # --- 4. HEAD CLASSIFICATION TƯƠNG THÍCH VỚI PIPELINE ---
        if self.num_ego_class != 0:
            ego_x = self.drop(ego_x)
            # Mở rộng ego_x nếu có G nhóm để tính toán qua Head
            if num_groups > 1:
                ego_x_exp = (
                    ego_x.unsqueeze(1)
                    .repeat(1, num_groups, 1)
                    .view(B * num_groups, self.ego_c)
                )
                pred_ego, pred_actor = self.head(final_slots, ego_x_exp)
            else:
                pred_ego, pred_actor = self.head(final_slots, ego_x)
        else:
            pred_ego = None
            pred_actor = self.head(final_slots)

        # Reshape về lại cấu trúc batch chuẩn [B, G, num_classes] nếu num_groups > 1
        if num_groups > 1:
            pred_actor = pred_actor.view(B, num_groups, -1)
            initial_log_probs = initial_log_probs.view(B, num_groups)
            if pred_ego is not None:
                pred_ego = pred_ego.view(B, num_groups, -1).mean(dim=1)  # Ego dùng chung đại diện video
        else:
            pred_actor = pred_actor.view(B, -1)
            initial_log_probs = initial_log_probs.view(B)

        # Nối attention masks của các chunks theo trục thời gian
        attn_masks = torch.cat(attns_list, dim=-1)

        return pred_ego, pred_actor, attn_masks, initial_log_probs