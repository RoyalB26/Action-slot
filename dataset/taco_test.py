"""TACO Test dataset loader for evaluation and inference."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils.normalization import get_mean, get_std


class TACO_TEST(Dataset):
    """TACO Dataset loader formatted for testing and offline evaluation."""

    def __init__(
        self,
        args,
        split='val',
        root=None,
        Max_N=20,
        label_file=None,
    ):
        """
        Args:
            args (argparse.Namespace): Arguments object.
            split (str): 'val' or 'test'.
            root (str, optional): Root directory containing video scenarios.
            Max_N (int): Maximum object/slot bound.
            label_file (str, optional): Optional path to explicit label JSON.
        """
        self.args = args
        self.split = split
        self.root = root if root is not None else getattr(args, 'root', '')
        self.seq_len = getattr(args, 'seq_len', 16)
        self.Max_N = Max_N
        self.num_class = 64

        self.maps = []
        self.id = []
        self.variants = []
        self.scenario_name = []
        self.gt_ego = []
        self.gt_actor = []

        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f_label:
                label_list = json.load(f_label)
        else:
            default_label = os.path.join(os.path.dirname(__file__), f"taco_{split}_label.json")
            if os.path.exists(default_label):
                with open(default_label, 'r', encoding='utf-8') as f_label:
                    label_list = json.load(f_label)
            else:
                label_list = {}

        for k, v in label_list.items():
            parts = k.split('/')
            if len(parts) >= 4:
                self.maps.append(parts[0])
                self.id.append(parts[1])
                self.variants.append(parts[3])
                self.scenario_name.append(parts[0] + '/' + parts[1] + '/' + parts[3])
                self.gt_ego.append(v['ego'])
                self.gt_actor.append(v['actor'])

        img_h = getattr(args, 'img_height', 128)
        img_w = getattr(args, 'img_width', 384)
        self.img_size = (img_h, img_w)

        self.test_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=get_mean(norm_value=1), std=get_std(norm_value=1)),
        ])

    def __len__(self):
        """Total number of evaluation video sequences."""
        return len(self.gt_ego)

    def __getitem__(self, idx):
        """Retrieve video frames uniformly sampled across scenario clip.

        Args:
            idx (int): Video index.

        Returns:
            dict:
                - 'map' (str): Town or map name.
                - 'id' (str): Scenario ID.
                - 'variants' (str): Variant ID.
                - 'scenario' (str): Scenario identifier string.
                - 'videos' (list[torch.Tensor]): List of seq_len frame tensors (3, 128, 384).
                - 'gt_actor' (torch.Tensor): Binary multi-label actor vector of shape (64,).
                - 'gt_ego' (torch.Tensor): Ego action label scalar of shape ().
        """
        scenario = f"{self.maps[idx]}/{self.id[idx]}/{self.variants[idx]}"
        scenario_path = os.path.join(self.root, scenario) if self.root else ""
        imgs = []

        if scenario_path and os.path.isdir(scenario_path):
            frame_files = sorted(os.listdir(scenario_path))
            step = max(1, len(frame_files) // self.seq_len)
            selected_files = frame_files[::step][:self.seq_len]
            for f_name in selected_files:
                img_path = os.path.join(scenario_path, f_name)
                img = Image.open(img_path).convert('RGB')
                imgs.append(self.test_transforms(img))

        # Pad with zero frames if video is short
        if len(imgs) < self.seq_len:
            pad_count = self.seq_len - len(imgs)
            for _ in range(pad_count):
                imgs.append(torch.zeros(3, 128, 384))

        # Multi-label actor ground truth: (64,)
        gt_actor = torch.zeros(self.num_class, dtype=torch.float32)
        if len(self.gt_actor[idx]) > 0:
            gt_actor[self.gt_actor[idx]] = 1.0

        # Ego-action ground truth: ()
        gt_ego = torch.tensor(self.gt_ego[idx], dtype=torch.long)

        return {
            'map': self.maps[idx],
            'id': self.id[idx],
            'variants': self.variants[idx],
            'scenario': scenario,
            'videos': imgs,        # list of seq_len tensors with shape (3, 128, 384)
            'gt_actor': gt_actor,  # shape: (64,)
            'gt_ego': gt_ego,      # shape: ()
        }
