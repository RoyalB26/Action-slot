"""Normalization utilities for video datasets."""

import numpy as np


def get_mean(norm_value=255, dataset='activitynet'):
    """Return mean channel values for video normalization.

    Args:
        norm_value (float): Scaling factor (e.g. 255 or 1).
        dataset (str): Dataset name ('activitynet' or 'kinetics').

    Returns:
        list[float]: Per-channel mean values [R, G, B].
    """
    assert dataset in ['activitynet', 'kinetics'], f"Unsupported dataset: {dataset}"
    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    """Return standard deviation channel values for video normalization.

    Args:
        norm_value (float): Scaling factor (e.g. 255 or 1).

    Returns:
        list[float]: Per-channel standard deviation values [R, G, B].
    """
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]
