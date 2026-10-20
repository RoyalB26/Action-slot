"""Modular Video Backbones with registration."""

from abc import ABC, abstractmethod
from typing import List, Union
import torch
import torch.nn as nn
from core.registry import BACKBONES


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all spatiotemporal feature extractors."""

    @abstractmethod
    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Extract spatiotemporal feature volume.

        Returns:
            torch.Tensor: Feature map of shape (B, C_out, T_feat, H_feat, W_feat)
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Output feature dimension of the backbone."""
        pass


@BACKBONES.register("x3d")
@BACKBONES.register("x3d_m")
class X3DBackbone(BaseBackbone):
    """Pretrained X3D-M backbone from PyTorchVideo hub."""

    def __init__(self, pretrained: bool = True, freeze_stages: int = 3, freeze_all: bool = False, unfreeze_all: bool = False):
        super().__init__()
        x3d_hub = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=pretrained)
        # All residual stages except the final 1x1x1 classification head
        self.blocks = x3d_hub.blocks[:-1]
        self._out_channels = 192

        # Configure freezing options for finetuning
        if freeze_all:
            for param in self.blocks.parameters():
                param.requires_grad = False
        elif unfreeze_all:
            for param in self.blocks.parameters():
                param.requires_grad = True
        elif freeze_stages > 0:
            for block in self.blocks[:freeze_stages]:
                for param in block.parameters():
                    param.requires_grad = False

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            # Shape transition: list of T * (B, C, H, W) -> (B, C, T, H, W)
            x = torch.stack(x, dim=0).permute((1, 2, 0, 3, 4))

        for block in self.blocks:
            x = block(x)
        return x
