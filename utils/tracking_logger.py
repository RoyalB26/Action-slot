"""Formatting and logging utilities for TACO multi-action evaluation tracking."""

import os
from typing import Dict, List, Optional
import numpy as np


# Category groupings defined by TACO action label prefixes:
# c:  car actions (12 classes)
# b:  bus/truck actions (12 classes)
# p:  pedestrian actions (8 classes)
# c+: car+ actions (12 classes)
# b+: bus+ actions (12 classes)
# p+: pedestrian+ actions (8 classes)
TACO_ACTION_GROUPS = ["c", "b", "p", "c+", "b+", "p+"]


def format_validation_tracking_text(
    epoch: int,
    best_map: float,
    current_map: float,
    per_class_reports: List[Dict],
    category_order: Optional[List[str]] = None,
) -> str:
    """Format evaluation statistics into the exact requested tracking text layout.

    Layout structure:
    epoch: {epoch}
    best mAP: {best_map:.4f}
    mAP: {current_map:.4f}
    mAP of c: {map_c:.4f}
    mAP of b: {map_b:.4f}
    mAP of p: {map_p:.4f}
    mAP of c+: {map_c_plus:.4f}
    mAP of b+: {map_b_plus:.4f}
    mAP of p+: {map_p_plus:.4f}
    c per class:
    0.1098 0.0949 ...
    b per class:
    ...
    c+ per class:
    ...
    b+ per class:
    ...
    p per class:
    ...
    p+ per class:
    ...
    ***************

    Args:
        epoch: Epoch index.
        best_map: Current best mAP score.
        current_map: mAP score for the current epoch.
        per_class_reports: List of dicts with 'name', 'ap', and 'support' from tracker.
        category_order: Prefix grouping order.

    Returns:
        str: Formatted multiline string.
    """
    order = category_order or TACO_ACTION_GROUPS

    # Organize APs into category buckets preserving original class order
    group_aps: Dict[str, List[float]] = {cat: [] for cat in order}

    for item in per_class_reports:
        name = item.get("name", "")
        ap = float(item.get("ap", 0.0))
        # Determine prefix before ':'
        if ":" in name:
            prefix = name.split(":")[0]
            if prefix in group_aps:
                group_aps[prefix].append(ap)

    lines = []
    lines.append(f"epoch: {epoch}")
    lines.append(f"best mAP: {best_map:.4f}")
    lines.append(f"mAP: {current_map:.4f}")

    # 1. Output group mAPs: mAP of {cat}: {val:.4f}
    for cat in order:
        aps = group_aps[cat]
        cat_mean = float(np.mean(aps)) if len(aps) > 0 else 0.0
        lines.append(f"mAP of {cat}: {cat_mean:.4f}")

    # 2. Output per-class scores per group:
    # {cat} per class:
    # 0.xxxx 0.xxxx ...
    for cat in order:
        aps = group_aps[cat]
        lines.append(f"{cat} per class:")
        val_strs = [f"{ap:.4f}" for ap in aps]
        lines.append(" ".join(val_strs))

    lines.append("***************")
    return "\n".join(lines)


def append_validation_tracking_log(
    file_path: str,
    epoch: int,
    best_map: float,
    current_map: float,
    per_class_reports: List[Dict],
    category_order: Optional[List[str]] = None,
    print_to_terminal: bool = True,
) -> str:
    """Append formatted tracking log to file and optionally print to console.

    Args:
        file_path: Destination text file (e.g. checkpoints/val_tracking.txt).
        epoch: Current epoch number.
        best_map: Best recorded mAP.
        current_map: Current epoch mAP.
        per_class_reports: Per-class statistics from PerformanceTracker.
        category_order: Optional category prefix ordering.
        print_to_terminal: If True, prints directly to terminal.

    Returns:
        str: Appended tracking text block.
    """
    text_block = format_validation_tracking_text(
        epoch=epoch,
        best_map=best_map,
        current_map=current_map,
        per_class_reports=per_class_reports,
        category_order=category_order,
    )

    if print_to_terminal:
        print("\n" + text_block + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text_block + "\n")

    return text_block
