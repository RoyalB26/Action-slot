"""Standard 3D CNN Baseline model using X3D architecture."""

import torch
import torch.nn as nn
from .heads import Head


class X3D_Baseline(nn.Module):
    """X3D-M baseline model for video-level multi-label action recognition."""

    def __init__(self, num_ego_class, num_actor_class, args=None):
        """
        Args:
            num_ego_class (int): Number of ego classes (e.g. 7).
            num_actor_class (int): Number of actor action classes (e.g. 64).
            args (argparse.Namespace, optional): Configuration arguments.
        """
        super(X3D_Baseline, self).__init__()
        self.num_ego_class = num_ego_class

        # Load X3D-M pretrained on Kinetics
        self.model = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)

        # Replace classification projection with custom 2048-dim projection
        self.projection = nn.Sequential(
            nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(16, 3, 3), stride=1, padding=0),
            nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.model.blocks[-1] = self.projection
        self.head = Head(2048, num_ego_class, num_actor_class)
        self.model.blocks.append(self.head)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x):
        """Forward pass through X3D baseline.

        Args:
            x (list[torch.Tensor] or torch.Tensor): Input video frames of shape (B, C, T, H, W)
                or list of T tensors with shape (B, C, H, W).

        Returns:
            If num_ego_class != 0:
                tuple[torch.Tensor, torch.Tensor]:
                    - ego: Ego logits of shape (B, num_ego_class).
                    - out: Actor action logits of shape (B, num_actor_class).
            Else:
                torch.Tensor: Actor logits of shape (B, num_actor_class).
        """
        batch_size = x[0].shape[0]
        if isinstance(x, list):
            x = torch.stack(x, dim=0)        # Shape: (T, B, C, H, W)
            x = x.permute((1, 2, 0, 3, 4))    # Shape: (B, C, T, H, W)

        num_block = len(self.model.blocks)
        for i in range(num_block - 1):
            x = self.model.blocks[i](x)

        # Global average pooling: (B, 2048, 1, 1, 1) -> (B, 2048)
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, 2048))

        if self.num_ego_class != 0:
            ego, out = self.model.blocks[-1](x)
            return ego, out
        else:
            out = self.model.blocks[-1](x)
            return out
