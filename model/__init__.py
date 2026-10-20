"""Model instantiation factory for Action-Slot and X3D baseline."""

import numpy as np
import torch
from .action_slot_x3d import ACTION_SLOT
from .x3d_baseline import X3D_Baseline
from .heads import Head, Allocated_Head, DeformableSlotTemporalDetectionHead
from .slot_attention import SlotAttention


def generate_model(args, num_ego_class, num_actor_class):
    """Instantiate and configure model backbone and trainable parameters.

    Args:
        args (argparse.Namespace): Command line options containing:
            - model_name (str): Either 'action_slot' or 'x3d'.
            - num_slots (int): Number of slots for slot attention.
        num_ego_class (int): Number of ego action classes.
        num_actor_class (int): Number of actor action classes.

    Returns:
        torch.nn.Module: Configured model instance ready for training or evaluation.
    """
    model_name = getattr(args, 'model_name', 'action_slot')

    if model_name == 'action_slot':
        model = ACTION_SLOT(args, num_ego_class, num_actor_class, num_slots=args.num_slots)
        for t in model.parameters():
            t.requires_grad = True
        # Freeze initial X3D blocks, fine-tune the last two residual stages
        for t in model.resnet.parameters():
            t.requires_grad = False
        for t in model.resnet[-1].parameters():
            t.requires_grad = True
        for t in model.resnet[-2].parameters():
            t.requires_grad = True

    elif model_name == 'x3d':
        model = X3D_Baseline(num_ego_class, num_actor_class, args)
        for t in model.model.parameters():
            t.requires_grad = False
        # Fine-tune top blocks
        for t in model.model.blocks[-1].parameters():
            t.requires_grad = True
        for t in model.model.blocks[-2].parameters():
            t.requires_grad = True
        for t in model.model.blocks[-3].parameters():
            t.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'action_slot' and 'x3d' are supported.")

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    trainable_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")

    return model


__all__ = [
    "ACTION_SLOT",
    "X3D_Baseline",
    "Head",
    "Allocated_Head",
    "DeformableSlotTemporalDetectionHead",
    "SlotAttention",
    "generate_model",
]
