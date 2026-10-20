"""Evaluation metrics and tracking utilities."""

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        """Update meter with a new value.

        Args:
            val (float): Current measured value.
            n (int): Batch size or weight of the value.
        """
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class=1, start_class=0):
    """Compute intersection and union for binary/multiclass masks.

    Args:
        pred (torch.Tensor or np.ndarray): Predicted mask, shape (H, W) or (B, H, W).
        mask (torch.Tensor or np.ndarray): Ground truth mask, shape (H, W) or (B, H, W).
        num_class (int): Number of classes.
        start_class (int): Starting index for class histogram.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - area_inter: Intersection area per class.
            - area_union: Union area per class.
    """
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    mask = mask.data.cpu().numpy().astype(np.uint8)
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()
    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(start_class, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(start_class, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(start_class, num_class))
    area_union = area_pred + area_mask - area_inter
    return (area_inter, area_union)
