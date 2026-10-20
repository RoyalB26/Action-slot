"""Action-Slot model architecture tailored for TACO dataset using X3D backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .slot_attention import SlotAttention
from .heads import Head, Allocated_Head


class ACTION_SLOT(nn.Module):
    """Spatiotemporal Action-Slot model with X3D backbone for TACO benchmark."""

    def __init__(self, args, num_ego_class, num_actor_class, num_slots=64):
        """
        Args:
            args (argparse.Namespace): Configuration namespace containing hyperparameters:
                - channel (int): Feature channel dimension.
                - allocated_slot (bool): Whether each slot is assigned to a specific action class.
                - bg_slot (bool): Whether to include an extra slot for background.
            num_ego_class (int): Number of ego-vehicle action classes (e.g. 7 for TACO).
            num_actor_class (int): Number of other actor action classes (e.g. 64 for TACO).
            num_slots (int): Number of slots in slot attention.
        """
        super(ACTION_SLOT, self).__init__()
        self.args = args
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots

        self.slot_dim = getattr(args, 'channel', 256)
        self.hidden_dim2 = self.slot_dim
        self.ego_c = 64

        # Backbone: X3D-M pretrained on Kinetics, removing the final classification projection
        x3d_hub = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.resnet = x3d_hub.blocks[:-1]
        self.in_c = 192  # Channel output dimension of X3D-M before head

        # Feature resolution for TACO (T=16, H=8, W=24)
        self.resolution = (8, 24)
        self.resolution3d = (16, 8, 24)

        if getattr(args, 'allocated_slot', False):
            self.head = Allocated_Head(self.slot_dim, num_ego_class, num_actor_class, self.ego_c)
        else:
            self.head = Head(self.slot_dim, num_ego_class, num_actor_class + 1, self.ego_c)

        if self.num_ego_class != 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
            )

        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.in_c),
            nn.Conv3d(self.in_c, self.hidden_dim2, (1, 1, 1), stride=1),
            nn.ReLU(),
        )

        num_slots_init = self.num_slots + 1 if getattr(args, 'bg_slot', False) else self.num_slots
        self.slot_attention = SlotAttention(
            num_slots=num_slots_init,
            dim=self.slot_dim,
            eps=1e-8,
            input_dim=self.hidden_dim2,
            resolution=self.resolution3d,
            num_actor_class=num_actor_class,
            allocated_slot=getattr(args, 'allocated_slot', False),
        )

        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x, box=False):
        """Forward pass for video clips.

        Args:
            x (list[torch.Tensor] or torch.Tensor): List of T frame tensors, each of shape (B, C, H, W),
                or single stacked tensor of shape (B, C, T, H, W).
            box (bool, optional): Ignored placeholder for compatibility.

        Returns:
            If num_ego_class != 0:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - ego_x: Ego-action logits of shape (B, num_ego_class).
                    - x: Actor-action logits of shape (B, num_actor_class) or (B, num_slots, num_actor_class).
                    - attn_masks: Spatiotemporal attention weights of shape (B, T, num_slots, H, W).
            Else:
                tuple[torch.Tensor, torch.Tensor]:
                    - x: Actor-action logits of shape (B, num_actor_class).
                    - attn_masks: Attention weights of shape (B, T, num_slots, H, W).
        """
        seq_len = len(x)
        batch_size = x[0].shape[0]

        # Prepare 5D tensor for 3D CNN: (B, C, T, H, W)
        if isinstance(x, list):
            x = torch.stack(x, dim=0)        # Shape: (T, B, C, H, W)
            x = x.permute((1, 2, 0, 3, 4))    # Shape: (B, C, T, H, W)

        # Forward through X3D backbone blocks
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)  # Shape: (B, 192, T_feat, H_feat, W_feat)

        x = self.drop(x)

        # Ego branch feature extraction
        if self.num_ego_class != 0:
            ego_x = self.conv3d_ego(x)                       # Shape: (B, ego_c, T_feat, H_feat, W_feat)
            ego_x = self.pool(ego_x)                         # Shape: (B, ego_c, 1, 1, 1)
            ego_x = torch.reshape(ego_x, (batch_size, self.ego_c))  # Shape: (B, ego_c)

        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        # Project channels into hidden_dim2 (slot_dim)
        x = self.conv3d(x)                              # Shape: (B, slot_dim, T_feat, H_feat, W_feat)
        x = x.permute((0, 2, 3, 4, 1))                  # Shape: (B, T_feat, H_feat, W_feat, slot_dim)
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))
        feat_map = x                                    # Keep dense spatiotemporal map for deformable head

        # Slot Attention extraction: slots (B, num_slots, slot_dim), attn (B, num_slots, T*H*W)
        x, attn_masks = self.slot_attention(x)

        # Reshape and interpolate attention maps back to video resolution: (B, T, num_slots, H, W)
        b, n, _ = attn_masks.shape
        attn_masks = attn_masks.view(b * n, 1, new_seq_len, new_h, new_w)

        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode='trilinear')

        attn_masks = torch.reshape(attn_masks, (b, n, seq_len, new_h, new_w))
        attn_masks = attn_masks.permute((0, 2, 1, 3, 4))  # Shape: (B, T, num_slots, H, W)

        x = self.drop(x)
        if self.num_ego_class != 0:
            ego_x = self.drop(ego_x)
            if isinstance(self.head, Allocated_Head):
                ego_x, x = self.head(x, feat_map, ego_x)
            else:
                ego_x, x = self.head(x, ego_x)
            return ego_x, x, attn_masks
        else:
            if isinstance(self.head, Allocated_Head):
                x = self.head(x, feat_map)
            else:
                x = self.head(x)
            return x, attn_masks
