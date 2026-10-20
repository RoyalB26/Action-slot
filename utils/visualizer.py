"""Attention heatmap overlay and visualization utilities with thermal-gradient color mapping.

Color Palette Specification:
- High attention focus (Peak)   : Red
- Medium attention focus        : Orange
- Low / Background attention     : Deep Blue
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils.normalization import get_mean, get_std


def create_attention_colormap():
    """Create custom colormap transitioning smoothly: Blue (low) -> Cyan -> Green -> Orange -> Red (high).

    This strictly implements:
        - Low attention: Deep Blue
        - Medium attention: Orange
        - High attention: Red
    """
    colors = [
        (0.0, "#00008b"),   # Deep Blue (low attention)
        (0.25, "#0080ff"),  # Bright Blue
        (0.5, "#ff8c00"),   # Orange (medium attention)
        (0.75, "#ff4500"),  # Orange-Red
        (1.0, "#ff0000"),   # Pure Red (high attention)
    ]
    return LinearSegmentedColormap.from_list("blue_orange_red", colors, N=256)


# Global custom colormap instance
ATTN_COLORMAP = create_attention_colormap()


def denormalize_image(img_tensor):
    """Denormalize a 3-channel image tensor into a [0, 255] RGB numpy uint8 array.

    Args:
        img_tensor (torch.Tensor or np.ndarray): Shape (3, H, W) or (B, 3, H, W) or (H, W, 3).

    Returns:
        np.ndarray: Image uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)

    # If batched tensor (B, 3, H, W) is passed, take the first sample (3, H, W)
    if img.ndim == 4:
        img = img[0]

    if img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        img = np.transpose(img, (1, 2, 0))

    mean = np.array(get_mean(norm_value=1))
    std = np.array(get_std(norm_value=1))

    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def render_attention_heatmap(frame_rgb, attn_map, alpha=0.55):
    """Overlay attention heatmap on a video frame using Blue -> Orange -> Red thermal colormap.

    Args:
        frame_rgb (np.ndarray): Base RGB image array (H, W, 3) in range [0, 255].
        attn_map (np.ndarray or torch.Tensor): 2D attention map (H_attn, W_attn) or (H, W).
        alpha (float): Transparency factor blending the heatmap over the frame.

    Returns:
        np.ndarray: Blended RGB image uint8 array of shape (H, W, 3).
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    h, w = frame_rgb.shape[:2]

    # Resize attention map to frame resolution if needed
    if attn_map.shape[:2] != (h, w):
        attn_map = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize heatmap strictly to [0, 1]
    attn_min, attn_max = float(attn_map.min()), float(attn_map.max())
    if attn_max > attn_min:
        norm_attn = (attn_map - attn_min) / (attn_max - attn_min)
    else:
        norm_attn = np.zeros_like(attn_map)

    # Apply Blue -> Orange -> Red colormap (produces RGBA float in [0, 1])
    heatmap_rgba = ATTN_COLORMAP(norm_attn)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)

    # Alpha blending: blended = alpha * heatmap + (1 - alpha) * frame
    blended = cv2.addWeighted(heatmap_rgb, alpha, frame_rgb, 1.0 - alpha, 0)
    return blended


def save_attention_grid(
    frames,
    attn_tensor,
    save_path,
    slot_idx=None,
    top_k_slots=4,
    frame_indices=(0, 4, 8, 12, 15),
    title="Action-Slot Spatiotemporal Attention",
):
    """Render and save a spatiotemporal grid showing attention evolution across video frames.

    Args:
        frames (list[torch.Tensor] or torch.Tensor): Video clip frames of shape (T, 3, H, W).
        attn_tensor (torch.Tensor or np.ndarray): Attention map of shape (T, num_slots, H, W)
            or (1, T, num_slots, H, W).
        save_path (str): Output destination image path (.jpg / .png).
        slot_idx (int, optional): Specific slot index to plot. If None, plots top active slots.
        top_k_slots (int): Number of top slots to plot when slot_idx is None.
        frame_indices (tuple[int]): Temporal frame steps to sample for visualization.
        title (str): Figure title.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(attn_tensor, torch.Tensor):
        attn = attn_tensor.detach().cpu().numpy()
    else:
        attn = np.array(attn_tensor)

    if attn.ndim == 5:  # (B, T, num_slots, H, W) -> take first sample
        attn = attn[0]

    T, S, H_att, W_att = attn.shape

    # Choose slot indices to visualize
    if slot_idx is not None:
        target_slots = [slot_idx]
    else:
        # Pick top active slots by highest peak activation energy
        slot_energies = [float(np.mean(attn[:, s])) for s in range(S)]
        target_slots = list(np.argsort(slot_energies)[::-1][:top_k_slots])

    valid_frames = [f for f in frame_indices if f < T]
    num_cols = len(valid_frames)
    num_rows = len(target_slots) + 1  # 1 row for original frames + rows for slots

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.2))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # 1. First row: Original video frames
    for col_idx, t_idx in enumerate(valid_frames):
        frame_rgb = denormalize_image(frames[t_idx])
        ax = axes[0, col_idx]
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame t={t_idx}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # 2. Subsequent rows: Attention overlays for selected slots
    for row_idx, s in enumerate(target_slots, start=1):
        for col_idx, t_idx in enumerate(valid_frames):
            frame_rgb = denormalize_image(frames[t_idx])
            slot_map = attn[t_idx, s]
            blended = render_attention_heatmap(frame_rgb, slot_map, alpha=0.58)

            ax = axes[row_idx, col_idx]
            ax.imshow(blended)
            if col_idx == 0:
                ax.set_ylabel(f"Slot #{s}", fontsize=11, fontweight="bold", rotation=0, labelpad=35)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
