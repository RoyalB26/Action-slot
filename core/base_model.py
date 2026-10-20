"""Base interface for all action video models ensuring standardized output dictionary."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


class BaseActionModel(nn.Module, ABC):
    """Abstract base class that enforces unified outputs across architectures.

    All models returning outputs must provide a dictionary containing at least:
        - 'pred_actor': Tensor of actor action predictions. Shape: (B, num_actor_class) or (B, num_slots, num_actor_class)
        - 'pred_ego': Optional Tensor of ego action predictions. Shape: (B, num_ego_class)
        - 'attn': Optional Tensor of spatiotemporal slot attention masks. Shape: (B, T, num_slots, H, W)
    """

    @abstractmethod
    def forward(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Process input video and return standardized dictionary.

        Args:
            videos: List of T frame tensors (B, 3, H, W) or a 5D tensor (B, 3, T, H, W).
            **kwargs: Extra parameters (e.g. bounding boxes, extra context).

        Returns:
            Dict containing keys:
                'pred_actor': torch.Tensor
                'pred_ego': Optional[torch.Tensor]
                'attn': Optional[torch.Tensor]
        """
        pass

    def get_trainable_parameters(self, stage: Optional[str] = None):
        """Helper to retrieve fine-tuning parameters with layer-specific rules."""
        return filter(lambda p: p.requires_grad, self.parameters())
