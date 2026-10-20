"""Loss function compatibility layer mapping to modular losses package."""

from losses import ActionSlotLoss, FocalActionLoss, BetterActionSlotLoss, build_loss
from core.registry import LOSSES

__all__ = [
    "ActionSlotLoss",
    "FocalActionLoss",
    "BetterActionSlotLoss",
    "build_loss",
    "LOSSES",
]
