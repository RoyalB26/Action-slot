"""Attention matrix dumper and full-slot multi-frame visualizer.

Utilities:
1. save_attention_matrix: Save raw attention tensor and frame metadata to a .npz / .pt file.
2. render_all_slots_from_file: Load saved attention matrix and render 64 separate images (one for each slot)
   covering all T frames with progressive red heatmap overlay on original images.
3. visualize_best_worst_slots: Identify the highest and lowest performing slots and render full T-frame images.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import re
import numpy as np
import torch
from typing import List, Optional, Union

from utils.visualizer import denormalize_image, render_attention_heatmap, save_single_slot_all_frames
from utils.taco_labels import TACO_ACTOR_LABELS


def sanitize_filename(name: str) -> str:
    """Sanitize class label string into safe file name component."""
    return re.sub(r'[^a-zA-Z0-9_\-\+]', '_', name)


def save_attention_matrix(
    attn_tensor: Union[torch.Tensor, np.ndarray],
    frames: Union[List[torch.Tensor], torch.Tensor],
    save_path: str,
    extra_metadata: Optional[dict] = None,
):
    """Save spatiotemporal attention tensor and video frames to a compressed NumPy .npz file.

    Args:
        attn_tensor: Attention tensor of shape (B, T, S, H, W) or (T, S, H, W).
        frames: Video frames list of T tensors (B, 3, H, W) or 4D tensor (T, 3, H, W).
        save_path: Output file path (.npz).
        extra_metadata: Optional dictionary with epoch, scenario, loss info.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(attn_tensor, torch.Tensor):
        attn_np = attn_tensor.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn_tensor)

    # Denormalize frames into uint8 numpy array: (T, H, W, 3)
    if isinstance(frames, list):
        frames_list = [denormalize_image(f) for f in frames]
        frames_np = np.stack(frames_list, axis=0)  # (T, H, W, 3)
    elif isinstance(frames, torch.Tensor):
        f = frames.detach().cpu()
        if f.ndim == 5:
            f = f[0]
        frames_list = [denormalize_image(f[t]) for t in range(f.shape[0])]
        frames_np = np.stack(frames_list, axis=0)
    else:
        frames_np = np.asarray(frames)

    metadata = extra_metadata or {}
    np.savez_compressed(
        save_path,
        attention=attn_np,
        frames=frames_np,
        metadata=metadata,
    )
    print(f"Saved attention matrix to '{save_path}'")


def render_all_slots_from_file(
    matrix_file: str,
    output_dir: str,
    slot_names: Optional[List[str]] = None,
    epoch: Optional[int] = None,
    prefix: str = "slot",
) -> List[str]:
    """Load saved attention matrix file and render an image for each slot across all T frames.

    Generates 64 (or S) images in output_dir:
        format: {prefix}_epoch_{epoch}_slot_{slot_idx:02d}_{sanitized_name}.png

    Args:
        matrix_file: Path to .npz file saved by save_attention_matrix.
        output_dir: Directory where the slot images will be stored.
        slot_names: List of class names for each slot (defaults to TACO_ACTOR_LABELS).
        epoch: Optional epoch integer for filename tagging.
        prefix: Filename prefix.

    Returns:
        List of generated image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(matrix_file, allow_pickle=True)
    attn = data['attention']  # (B, T, S, H, W) or (T, S, H, W)
    frames_rgb = data['frames']  # (T, H, W, 3)

    if attn.ndim == 5:
        attn = attn[0]

    T, S, H_att, W_att = attn.shape
    names = slot_names or TACO_ACTOR_LABELS
    num_to_render = len(slot_names) if slot_names is not None else S
    num_to_render = min(num_to_render, S)

    ep_tag = f"epoch_{epoch}_" if epoch is not None else ""
    generated_files = []

    for s_idx in range(num_to_render):
        name = names[s_idx] if s_idx < len(names) else f"slot_{s_idx}"
        safe_name = sanitize_filename(name)
        filename = f"{prefix}_{ep_tag}slot_{s_idx:02d}_{safe_name}.png"
        out_path = os.path.join(output_dir, filename)

        title = f"Slot #{s_idx:02d} [{name}] - All {T} Frames (Progressive Red Heatmap)"
        save_single_slot_all_frames(
            frames=frames_rgb,
            attn_tensor=attn,
            slot_idx=s_idx,
            slot_name=name,
            save_path=out_path,
            title=title,
        )
        generated_files.append(out_path)

    print(f"Rendered {len(generated_files)} slot attention images to '{output_dir}'")
    return generated_files


def visualize_best_and_worst_slots(
    frames,
    attn_tensor,
    output_dir: str,
    epoch: Optional[int] = None,
    slot_names: Optional[List[str]] = None,
    per_class_summary: Optional[List[dict]] = None,
) -> dict:
    """Identify 1 best and 1 worst slot and save their full T-frame visualizations.

    Slot ranking strategy:
        - If per_class_summary is provided (from tracker), picks class with highest AP as best
          and class with lowest AP (with positive support) as worst.
        - Otherwise, ranks slots by mean spatiotemporal attention activation energy.

    Args:
        frames: Video frames (T, 3, H, W).
        attn_tensor: (T, S, H, W) or (1, T, S, H, W).
        output_dir: Destination folder.
        epoch: Current epoch.
        slot_names: Labels for the slots.
        per_class_summary: Optional per-class performance dictionary from PerformanceTracker.

    Returns:
        dict: Paths to generated best and worst images.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(attn_tensor, torch.Tensor):
        attn = attn_tensor.detach().cpu().numpy()
    else:
        attn = np.array(attn_tensor)

    if attn.ndim == 5:
        attn = attn[0]

    T, S, H_att, W_att = attn.shape
    names = slot_names or TACO_ACTOR_LABELS

    best_idx = 0
    worst_idx = min(1, S - 1)

    # 1. Determine best and worst slots
    if per_class_summary and len(per_class_summary) > 0:
        # Filter classes with support > 0
        active = [c for c in per_class_summary if c.get("support", 0) > 0 and c.get("class_id", 0) < S]
        if active:
            best_cand = max(active, key=lambda x: x.get("ap", 0.0))
            worst_cand = min(active, key=lambda x: x.get("ap", 0.0))
            best_idx = best_cand["class_id"]
            worst_idx = worst_cand["class_id"]
        else:
            best_cand = max(per_class_summary, key=lambda x: x.get("ap", 0.0))
            worst_cand = min(per_class_summary, key=lambda x: x.get("ap", 0.0))
            best_idx = min(best_cand.get("class_id", 0), S - 1)
            worst_idx = min(worst_cand.get("class_id", 1), S - 1)
    else:
        # Fallback to slot attention activation magnitude
        slot_energies = [float(np.mean(attn[:, s])) for s in range(S)]
        ranked = list(np.argsort(slot_energies))
        worst_idx = ranked[0]
        best_idx = ranked[-1]

    ep_tag = f"epoch_{epoch}_" if epoch is not None else ""

    best_name = names[best_idx] if best_idx < len(names) else f"slot_{best_idx}"
    worst_name = names[worst_idx] if worst_idx < len(names) else f"slot_{worst_idx}"

    best_file = os.path.join(
        output_dir,
        f"{ep_tag}best_slot_{best_idx:02d}_{sanitize_filename(best_name)}.png"
    )
    worst_file = os.path.join(
        output_dir,
        f"{ep_tag}worst_slot_{worst_idx:02d}_{sanitize_filename(worst_name)}.png"
    )

    save_single_slot_all_frames(
        frames=frames,
        attn_tensor=attn,
        slot_idx=best_idx,
        slot_name=best_name,
        save_path=best_file,
        title=f"[BEST SLOT] Epoch {epoch} - Slot #{best_idx:02d} [{best_name}] (All {T} Frames)",
    )

    save_single_slot_all_frames(
        frames=frames,
        attn_tensor=attn,
        slot_idx=worst_idx,
        slot_name=worst_name,
        save_path=worst_file,
        title=f"[WORST SLOT] Epoch {epoch} - Slot #{worst_idx:02d} [{worst_name}] (All {T} Frames)",
    )

    print(f">> Saved best slot image: {best_file}")
    print(f">> Saved worst slot image: {worst_file}")

    return {
        "best_slot": {"index": best_idx, "name": best_name, "path": best_file},
        "worst_slot": {"index": worst_idx, "name": worst_name, "path": worst_file},
    }
