"""Attention heatmap overlay and visualization utilities.

Color Palette Specification:
- Base video frames maintain natural RGB colors.
- Attention highlights appear as intense red progressively intensifying with attention strength.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from utils.normalization import get_mean, get_std


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

    if img.ndim == 4:
        img = img[0]

    if img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        img = np.transpose(img, (1, 2, 0))

    mean = np.array(get_mean(norm_value=1))
    std = np.array(get_std(norm_value=1))

    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def render_attention_heatmap(frame_rgb, attn_map, alpha_max=0.75):
    """Overlay progressive red highlight on original RGB frame based on attention intensity.

    - Low attention: Keeps original frame natural colors.
    - High attention: Progressively blends into pure glowing red ([255, 0, 0]).

    Args:
        frame_rgb (np.ndarray): Base RGB image array (H, W, 3) in range [0, 255].
        attn_map (np.ndarray or torch.Tensor): 2D attention map (H_attn, W_attn).
        alpha_max (float): Maximum opacity factor for peak attention points.

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

    # Progressive red mask: pure red in RGB is (255, 0, 0)
    # Blend: output = (1 - alpha) * frame + alpha * red
    alpha = (norm_attn ** 1.5) * alpha_max  # Smooth non-linear ramp
    alpha = np.expand_dims(alpha, axis=-1)  # (H, W, 1)

    red_color = np.zeros_like(frame_rgb, dtype=np.float32)
    red_color[:, :, 0] = 255.0  # Red channel

    frame_float = frame_rgb.astype(np.float32)
    blended = (1.0 - alpha) * frame_float + alpha * red_color
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_single_slot_all_frames(
    frames,
    attn_tensor,
    slot_idx: int,
    slot_name: str,
    save_path: str,
    title: str = None,
):
    """Render and save an image containing ALL T frames for a specific slot.

    Args:
        frames (list[torch.Tensor] or torch.Tensor): Video frames of shape (T, 3, H, W).
        attn_tensor (torch.Tensor or np.ndarray): Shape (T, S, H, W) or (1, T, S, H, W).
        slot_idx (int): Target slot index.
        slot_name (str): Label name of the slot.
        save_path (str): Output file path (.png).
        title (str, optional): Header title.
    """
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if isinstance(attn_tensor, torch.Tensor):
        attn = attn_tensor.detach().cpu().numpy()
    else:
        attn = np.array(attn_tensor)

    if attn.ndim == 5:
        attn = attn[0]

    T, S, H_att, W_att = attn.shape

    # Layout: arrange T frames into grid (e.g. 4x4 for T=16, or 2x8)
    num_cols = 4 if T >= 4 else T
    num_rows = int(np.ceil(T / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.8, num_rows * 2.5))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    disp_title = title or f"Slot #{slot_idx:02d} [{slot_name}] - All {T} Frames (Progressive Red Heatmap)"
    fig.suptitle(disp_title, fontsize=14, fontweight="bold", y=0.98)

    for t_idx in range(num_rows * num_cols):
        r = t_idx // num_cols
        c = t_idx % num_cols
        ax = axes[r, c]

        if t_idx < T:
            frame_rgb = denormalize_image(frames[t_idx])
            slot_map = attn[t_idx, slot_idx]
            blended = render_attention_heatmap(frame_rgb, slot_map, alpha_max=0.75)
            ax.imshow(blended)
            ax.set_title(f"Frame t={t_idx:02d}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_attention_grid(
    frames,
    attn_tensor,
    save_path,
    slot_idx=None,
    top_k_slots=4,
    frame_indices=(0, 4, 8, 12, 15),
    title="Action-Slot Spatiotemporal Attention",
):
    """Backward-compatible grid visualization for selected frames and top slots."""
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if isinstance(attn_tensor, torch.Tensor):
        attn = attn_tensor.detach().cpu().numpy()
    else:
        attn = np.array(attn_tensor)

    if attn.ndim == 5:
        attn = attn[0]

    T, S, H_att, W_att = attn.shape

    if slot_idx is not None:
        target_slots = [slot_idx]
    else:
        slot_energies = [float(np.mean(attn[:, s])) for s in range(S)]
        target_slots = list(np.argsort(slot_energies)[::-1][:top_k_slots])

    valid_frames = [f for f in frame_indices if f < T]
    num_cols = len(valid_frames)
    num_rows = len(target_slots) + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.2))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for col_idx, t_idx in enumerate(valid_frames):
        frame_rgb = denormalize_image(frames[t_idx])
        ax = axes[0, col_idx]
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame t={t_idx}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for row_idx, s in enumerate(target_slots, start=1):
        for col_idx, t_idx in enumerate(valid_frames):
            frame_rgb = denormalize_image(frames[t_idx])
            slot_map = attn[t_idx, s]
            blended = render_attention_heatmap(frame_rgb, slot_map, alpha_max=0.75)
            ax = axes[row_idx, col_idx]
            ax.imshow(blended)
            if col_idx == 0:
                ax.set_ylabel(f"Slot #{s}", fontsize=11, fontweight="bold", rotation=0, labelpad=35)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
