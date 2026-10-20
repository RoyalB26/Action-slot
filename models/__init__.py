"""Models package exposing modular architectures and registry."""

from core.registry import MODELS, BACKBONES, HEADS
from models.backbones import BaseBackbone, X3DBackbone
from models.heads import Head, Allocated_Head, DeformableSlotTemporalDetectionHead
from models.action_slot import ActionSlotModel
from models.x3d_baseline import X3D_Baseline


def build_model(args, num_ego_class: int = 7, num_actor_class: int = 64):
    """Factory helper to build model from parsed CLI arguments or dictionary."""
    model_name = getattr(args, 'model_name', 'action_slot')

    kwargs = {
        "num_ego_class": num_ego_class,
        "num_actor_class": num_actor_class,
        "backbone_name": getattr(args, 'backbone', 'x3d'),
        "head_name": getattr(args, 'head_name', 'linear_head'),
        "num_slots": getattr(args, 'num_slots', 64),
        "slot_dim": getattr(args, 'channel', 256),
        "bg_slot": getattr(args, 'bg_slot', False),
        "dropout": 0.5,
        "freeze_stages": getattr(args, 'freeze_stages', 3),
        "freeze_all": getattr(args, 'freeze_backbone', False),
        "unfreeze_all": getattr(args, 'unfreeze_backbone', False),
    }

    model = MODELS.build(model_name, **kwargs)
    return model


__all__ = [
    "MODELS",
    "BACKBONES",
    "HEADS",
    "BaseBackbone",
    "X3DBackbone",
    "Head",
    "Allocated_Head",
    "DeformableSlotTemporalDetectionHead",
    "ActionSlotModel",
    "X3D_Baseline",
    "build_model",
]
