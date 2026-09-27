import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head, Allocated_Head
from pytorchvideo.models.hub import i3d_r50
from pytorchvideo.models.hub import csn_r101
from pytorchvideo.models.hub import mvit_base_16x4
import r50
import numpy as np
from math import ceil 
from ptflops import get_model_complexity_info
from deformable_DETR import SpatioTemporalDeformableAttention


def build_3d_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=int(res)) for res in resolution]
    try:
        grid = torch.meshgrid(*ranges, indexing='ij')
    except TypeError:
        grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1).float()


class SoftPositionEmbed3D(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution))

    def forward(self, inputs):
        # Đảm bảo grid luôn cùng device và dtype với inputs
        grid = self.grid.to(device=inputs.device, dtype=inputs.dtype)
        return inputs + self.embedding(grid)


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_actor_class=64, eps=1e-6, input_dim=64, resolution=[16, 8, 24], allocated_slot=True):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.allocated_slot = allocated_slot
        self.eps = eps
        self.scale = dim ** -0.5
        self.resolution = resolution

        # Không gọi .cuda() cứng ở __init__
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim).abs() + 0.05)

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(dim, [resolution[0], resolution[1], resolution[2]])

    def get_3d_slot(self, slots, inputs):
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))

        inputs = self.LN(inputs)
        inputs = self.FC1(inputs)
        inputs = F.relu(inputs)
        inputs = self.FC2(inputs)

        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        # Đảm bảo contiguous trước khi nhân ma trận
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # 2. Tính tích vô hướng an toàn qua bmm: [B, N_slots, d] x [B, d, N_tokens] -> [B, N_slots, N_tokens]
        scale = float(d) ** -0.5
        dots = torch.bmm(q, k.transpose(1, 2)) * scale

        # 3. Khử triệt để NaN/Inf tiềm ẩn trước khi vào hàm toán học
        dots = torch.nan_to_num(dots, nan=0.0, posinf=15.0, neginf=-15.0)

        # 4. Kẹp miền giá trị tránh bùng nổ số mũ
        dots = torch.clamp(dots, min=-15.0, max=15.0)

        # 5. Kỹ thuật trừ Max (Log-Sum-Exp Trick) giúp exp() <= 1.0, chống 100% overflow CUDA
        dots_max = dots.max(dim=1, keepdim=True)[0].detach()
        dots_stable = dots - dots_max

        # 6. Softmax trên trục Slot (dim=1) và thêm epsilon an toàn
        attn_ori = F.softmax(dots_stable, dim=1) + 1e-6

        # 7. Chuẩn hóa qua toàn bộ tokens, dùng add thay vì chia trần để triệt tiêu chia cho 0
        denom = attn_ori.sum(dim=-1, keepdim=True)
        attn = attn_ori / (denom + 1e-5)

        # 8. Nhân với Value tensor
        slots = torch.bmm(attn, v)
        print("5. OK")
        slots = slots.reshape(b, -1, d)
        if self.allocated_slot:
            slots = slots[:, :self.num_actor_class, :]
        else:
            slots = slots[:, :self.num_slots, :]
        print("6. OK")
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        print("7. OK")
        return slots, attn_ori

    def forward(self, inputs, num_slots=None):
        b, nf, h, w, d = inputs.shape
        
        # Khởi tạo slots động trực tiếp trên GPU của inputs
        mu = self.slots_mu.to(device=inputs.device, dtype=inputs.dtype).expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.to(device=inputs.device, dtype=inputs.dtype).expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=inputs.device, dtype=inputs.dtype)

        slots_out, attns = self.get_3d_slot(slots, inputs)
        return slots_out, attns


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, "num_heads has to divides dim"

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 2, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, T, H, W, D = x.shape
        residual = x
        x_norm = self.norm(x)

        x_temp = x_norm.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        qkv = self.qkv(x_temp).reshape(B * H * W, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-25.0, max=25.0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B * H * W, T, D)
        out = self.proj_drop(self.proj(out))
        out = out.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)

        x = residual + out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ACTION_SLOT(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class, num_slots=21, box=False, videomae=None):
        super(ACTION_SLOT, self).__init__()
        self.hidden_dim = args.channel
        self.hidden_dim2 = args.channel
        self.slot_dim, self.temp_dim = args.channel, args.channel
        self.num_ego_class = num_ego_class
        self.ego_c = 128
        self.num_slots = num_slots
        if args.dataset == 'nuscenes' and args.pretrain == 'oats' and not 'nuscenes' in args.cp:
            num_actor_class = 35
        if args.dataset == 'nuscenes' and args.pretrain == 'oats':
            self.num_slots = 35
        if args.dataset == 'oats' and args.pretrain == 'taco':
            self.num_slots = 6

        self.resnet = i3d_r50(True)
        self.args = args
        self.temporal_attn = SpatioTemporalDeformableAttention(self.slot_dim)
        self.num_obj_slot = 6

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
            
        if args.allocated_slot:
            self.head = Allocated_Head(self.slot_dim, num_ego_class, num_actor_class, self.ego_c)
        else:
            self.head = Head(self.slot_dim, num_ego_class, num_actor_class+1, self.ego_c)

        if self.num_ego_class != 0:
            self.conv3d_ego = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c),
                    nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
                    )

        if args.backbone == 'r50':
            self.conv3d = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c),
                    nn.Conv3d(self.in_c, self.in_c//2, (1, 1, 1), stride=1),
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c//2),
                    nn.Conv3d(self.in_c//2, self.in_c//2, (3, 3, 3), stride=1, padding='same'),
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c//2),
                    nn.Conv3d(self.in_c//2, self.in_c//2, (3, 3, 3), stride=1, padding='same'),
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c//2),
                    nn.Conv3d(self.in_c//2, self.hidden_dim2, (1, 1, 1), stride=1),
                    nn.ReLU(),)
        else:
            self.conv3d = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c),
                    nn.Conv3d(self.in_c, self.hidden_dim2, (1, 1, 1), stride=1),
                    nn.ReLU(),)

        if args.bg_slot:
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots+1,
                dim=self.slot_dim,
                eps=1e-6,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class=num_actor_class
                ) 
            self.object_attention = SlotAttention(
                num_slots=self.num_obj_slot + 1,
                dim=self.slot_dim,
                eps=1e-6,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class=self.num_obj_slot,
                allocated_slot=False
            )
        else:
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                dim=self.slot_dim,
                eps=1e-6,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class=num_actor_class
                ) 
            self.object_attention = SlotAttention(
                num_slots=self.num_obj_slot,
                dim=self.slot_dim,
                eps=1e-6,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class=self.num_obj_slot,
                allocated_slot=False
            )

        self.feedback_proj = nn.Sequential(
            nn.Linear(self.slot_dim, self.hidden_dim2),
            nn.LayerNorm(self.hidden_dim2)
        )

        self.temporal_pool_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.slot_dim,
                out_channels=self.hidden_dim2,
                kernel_size=(16, 1, 1),
                stride=(1, 1, 1),
                padding=0
            ),
            nn.BatchNorm3d(self.hidden_dim2),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(p=0.5)         
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x, box=False):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]

        if self.args.backbone == 'r50':
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
                x = torch.reshape(x, (seq_len*batch_size, 3, height, width))
                x = self.resnet(x)
                _, c, h, w = x.shape
                x = torch.reshape(x, (self.args.seq_len, batch_size, c, h, w))
                x = x.permute(1, 2, 0, 3, 4)

        elif self.args.backbone == 'slowfast':
            slow_x = []
            for i in range(0, seq_len, 4):
                slow_x.append(x[i])
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
                slow_x = torch.stack(slow_x, dim=0)
                x = x.permute((1,2,0,3,4))
                slow_x = slow_x.permute((1,2,0,3,4))
                x = [slow_x, x]

                for i in range(len(self.resnet)):
                    x = self.resnet[i](x)
                x[1] = self.path_pool(x[1])
                x = torch.cat((x[0], x[1]), dim=1)

        else:
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
                x = x.permute((1,2,0,3,4)).contiguous()
            for i in range(len(self.resnet)):
                x = self.resnet[i](x)

        x = self.drop(x)
        if self.num_ego_class != 0:
            ego_x = self.conv3d_ego(x)
            ego_x = self.pool(ego_x)
            ego_x = torch.reshape(ego_x, (batch_size, self.ego_c))

        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        x = self.conv3d(x)
        x = x.permute((0, 2, 3, 4, 1))
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1)).contiguous()

        x = self.temporal_attn(x)

        # 1. Trích xuất object slots & masks
        object_slots, obj_attns = self.object_attention(x)
        if obj_attns.shape[1] == object_slots.shape[1] + 1:
            attn_masks_obj = obj_attns[:, 1:, :]
        else:
            attn_masks_obj = obj_attns

        # 2. Chiếu ngược về từng patch
        slot_per_patch = torch.bmm(attn_masks_obj.transpose(1, 2), object_slots)

        B = x.shape[0]
        C = slot_per_patch.shape[-1]
        H, W = self.resolution[0], self.resolution[1]
        T_orig = slot_per_patch.shape[1] // (H * W)

        # 3. Reshape 5D và đưa qua Temporal Conv3D
        slot_per_patch_5d = slot_per_patch.view(B, T_orig, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        slot_per_patch_1frame = self.temporal_pool_conv(slot_per_patch_5d).permute(0, 2, 3, 4, 1)

        # 4. Hồi tiếp có kiểm soát qua cổng gamma
        x = x + self.gamma * self.feedback_proj(slot_per_patch_1frame)
        
        # 5. Slot Attention cho các tác tử hành vi
        x, attn_masks = self.slot_attention(x)

        b, n, thw = attn_masks.shape
        attn_masks = attn_masks.reshape(b, n, -1)
        attn_masks = attn_masks.view(b, n, new_seq_len, self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)
        attn_masks = attn_masks.reshape(b, n, -1)
        attn_masks = attn_masks.view(b, n, new_seq_len, self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)
        attn_masks = attn_masks.view(b*n, 1, new_seq_len, attn_masks.shape[3], attn_masks.shape[4])
        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode='trilinear')
        attn_masks = torch.reshape(attn_masks, (b, n, seq_len, new_h, new_w))
        attn_masks = attn_masks.permute((0, 2, 1, 3, 4))

        x = self.drop(x)
        attn_dict = {'action_attn': attn_masks, 'obj_attn': obj_attns}

        if self.num_ego_class != 0:
            ego_x = self.drop(ego_x)
            ego_x, x = self.head(x, ego_x)
            return ego_x, x, attn_dict
        else:
            x = self.head(x)
            return x, attn_dict