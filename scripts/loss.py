"""Loss function compatibility layer mapping to modular losses package."""

from losses import ActionSlotLoss, FocalActionLoss, build_loss
from core.registry import LOSSES

__all__ = [
    "ActionSlotLoss",
    "FocalActionLoss",
    "build_loss",
    "LOSSES",
]
