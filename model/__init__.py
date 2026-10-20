"""Backward compatibility layer for legacy imports: from model import generate_model."""

from models import (
    build_model,
    ActionSlotModel,
    X3D_Baseline,
    Head,
    Allocated_Head,
    DeformableSlotTemporalDetectionHead,
)
from model.slot_attention import SlotAttention
from core.registry import MODELS, BACKBONES, HEADS


def generate_model(args, num_ego_class: int = 7, num_actor_class: int = 64):
    """Instantiate model using the modular builder."""
    return build_model(args, num_ego_class, num_actor_class)


# Aliases for compatibility
ACTION_SLOT = ActionSlotModel

__all__ = [
    "generate_model",
    "build_model",
    "ACTION_SLOT",
    "ActionSlotModel",
    "X3D_Baseline",
    "Head",
    "Allocated_Head",
    "DeformableSlotTemporalDetectionHead",
    "SlotAttention",
    "MODELS",
    "BACKBONES",
    "HEADS",
]
