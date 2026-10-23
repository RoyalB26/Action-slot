"""Regularization and spatiotemporal video augmentation utilities."""

import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F_tv
from utils.normalization import get_mean, get_std


class VideoColorJitter:
    """Apply consistent color jittering across all video frames in a clip."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, imgs):
        if random.random() > self.p:
            return imgs

        # Sample transformation factors once per clip
        b_factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness) if self.brightness > 0 else 1.0
        c_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast) if self.contrast > 0 else 1.0
        s_factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation) if self.saturation > 0 else 1.0
        h_factor = random.uniform(-self.hue, self.hue) if self.hue > 0 else 0.0

        transformed = []
        for img in imgs:
            res = img
            if b_factor != 1.0:
                res = F_tv.adjust_brightness(res, b_factor)
            if c_factor != 1.0:
                res = F_tv.adjust_contrast(res, c_factor)
            if s_factor != 1.0:
                res = F_tv.adjust_saturation(res, s_factor)
            if h_factor != 0.0:
                res = F_tv.adjust_hue(res, h_factor)
            transformed.append(res)
        return transformed


class VideoRandomHorizontalFlip:
    """Random horizontal flip applied consistently across all frames in a clip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return [F_tv.hflip(img) for img in imgs]
        return imgs


class VideoRandomErasing:
    """Random erasing (Cutout) applied across video tensor frames."""

    def __init__(self, p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, tensor_list):
        """Args:
            tensor_list: List of T frame tensors each of shape (C, H, W).
        """
        if random.random() > self.p or len(tensor_list) == 0:
            return tensor_list

        c, h, w = tensor_list[0].shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            erase_h = int(round((target_area * aspect_ratio) ** 0.5))
            erase_w = int(round((target_area / aspect_ratio) ** 0.5))

            if erase_h < h and erase_w < w:
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)

                out = []
                for t in tensor_list:
                    t_mod = t.clone()
                    t_mod[:, i:i + erase_h, j:j + erase_w] = self.value
                    out.append(t_mod)
                return out

        return tensor_list


class VideoClipTransform:
    """Top-level pickleable callable for video clip transformations (safe for Windows multiprocessing)."""

    def __init__(self, img_size, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.norm = transforms.Normalize(mean=get_mean(norm_value=1), std=get_std(norm_value=1))

        if self.augment:
            self.color_jitter = VideoColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5)
            self.random_flip = VideoRandomHorizontalFlip(p=0.5)
            self.random_erase = VideoRandomErasing(p=0.25, scale=(0.02, 0.15))
        else:
            self.color_jitter = None
            self.random_flip = None
            self.random_erase = None

    def __call__(self, imgs):
        if self.augment:
            imgs = self.color_jitter(imgs)
            imgs = self.random_flip(imgs)

        tensors = []
        for img in imgs:
            t = transforms.functional.resize(img, self.img_size)
            t = transforms.functional.to_tensor(t)
            t = self.norm(t)
            tensors.append(t)

        if self.augment and self.random_erase is not None:
            tensors = self.random_erase(tensors)

        return tensors


def build_train_video_transforms(img_size, augment=False):
    """Build video clip transform pipeline supporting optional spatial and photometric augmentation."""
    return VideoClipTransform(img_size, augment=augment)
