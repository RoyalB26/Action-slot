"""TACO Test dataset loader for evaluation and inference."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils.normalization import get_mean, get_std

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class TACO_TEST(Dataset):
    """TACO Dataset loader formatted for testing and offline evaluation.

    Supports:
    1. Direct path to a single image file.
    2. Direct path to a directory containing images (frames of a video, or multiple test images).
    3. Classic TACO test set scenario layout via root and label_file/json.
    """

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
            root (str, optional): Root directory containing video scenarios or test input path.
            Max_N (int): Maximum object/slot bound.
            label_file (str, optional): Optional path to explicit label JSON.
        """
        self.args = args
        self.split = split
        self.seq_len = getattr(args, 'seq_len', 16)
        self.Max_N = Max_N
        self.num_class = 64

        img_h = getattr(args, 'img_height', 128)
        img_w = getattr(args, 'img_width', 384)
        self.img_size = (img_h, img_w)

        self.test_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=get_mean(norm_value=1), std=get_std(norm_value=1)),
        ])

        # Priority: input_path > root
        input_target = getattr(args, 'input_path', None) or root or getattr(args, 'root', '')
        self.input_target = input_target

        self.samples = []
        self.mode = 'dataset'  # 'single_image', 'image_dir_clip', 'multi_scenario', or 'dataset'

        if input_target and os.path.exists(input_target):
            if os.path.isfile(input_target):
                # Mode 1: A single image file provided
                self.mode = 'single_image'
                self.samples = [{
                    'scenario': os.path.splitext(os.path.basename(input_target))[0],
                    'frame_paths': [input_target],
                    'map': 'custom',
                    'id': os.path.basename(input_target),
                    'variants': 'none',
                    'gt_actor': torch.zeros(self.num_class, dtype=torch.float32),
                    'gt_ego': torch.tensor(0, dtype=torch.long),
                }]
            elif os.path.isdir(input_target):
                # Check if directory directly contains image files
                direct_images = sorted([
                    os.path.join(input_target, f) for f in os.listdir(input_target)
                    if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(input_target, f))
                ])

                if len(direct_images) > 0:
                    # Directory directly contains images -> Treat as a video sequence or collection of frames
                    self.mode = 'image_dir_clip'
                    self.samples = [{
                        'scenario': os.path.basename(os.path.abspath(input_target)),
                        'frame_paths': direct_images,
                        'map': 'custom',
                        'id': os.path.basename(os.path.abspath(input_target)),
                        'variants': 'none',
                        'gt_actor': torch.zeros(self.num_class, dtype=torch.float32),
                        'gt_ego': torch.tensor(0, dtype=torch.long),
                    }]
                else:
                    # Check if subdirectories contain images (e.g. multiple scenarios or clips)
                    subdirs_with_images = []
                    for root_dir, _, files in os.walk(input_target):
                        img_files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
                        if len(img_files) > 0:
                            subdirs_with_images.append((root_dir, sorted([os.path.join(root_dir, f) for f in img_files])))

                    if len(subdirs_with_images) > 0:
                        self.mode = 'multi_scenario'
                        for sub_path, img_paths in sorted(subdirs_with_images, key=lambda x: x[0]):
                            rel_name = os.path.relpath(sub_path, input_target).replace('\\', '/')
                            self.samples.append({
                                'scenario': rel_name,
                                'frame_paths': img_paths,
                                'map': rel_name.split('/')[0] if '/' in rel_name else 'custom',
                                'id': rel_name,
                                'variants': 'none',
                                'gt_actor': torch.zeros(self.num_class, dtype=torch.float32),
                                'gt_ego': torch.tensor(0, dtype=torch.long),
                            })

        if len(self.samples) == 0:
            # Fallback to standard TACO dataset structure using labels JSON
            self.mode = 'dataset'
            self.root = input_target
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

    def __len__(self):
        """Total number of evaluation video sequences or image samples."""
        if self.mode != 'dataset':
            return len(self.samples)
        return len(self.gt_ego)

    def __getitem__(self, idx):
        """Retrieve video frames uniformly sampled across scenario clip or image(s).

        Args:
            idx (int): Video index.

        Returns:
            dict:
                - 'map' (str): Town or map name.
                - 'id' (str): Scenario ID.
                - 'variants' (str): Variant ID.
                - 'scenario' (str): Scenario identifier string.
                - 'videos' (list[torch.Tensor]): List of seq_len frame tensors (3, img_h, img_w).
                - 'gt_actor' (torch.Tensor): Binary multi-label actor vector of shape (64,).
                - 'gt_ego' (torch.Tensor): Ego action label scalar of shape ().
        """
        if self.mode != 'dataset':
            sample = self.samples[idx]
            frame_paths = sample['frame_paths']
            imgs = []

            if len(frame_paths) == 1:
                # Single image provided: replicate across seq_len frames
                img = Image.open(frame_paths[0]).convert('RGB')
                tensor_img = self.test_transforms(img)
                imgs = [tensor_img for _ in range(self.seq_len)]
            elif len(frame_paths) > 0:
                step = max(1, len(frame_paths) // self.seq_len)
                selected_files = frame_paths[::step][:self.seq_len]
                for f_path in selected_files:
                    img = Image.open(f_path).convert('RGB')
                    imgs.append(self.test_transforms(img))

                # Pad if needed
                if len(imgs) < self.seq_len:
                    pad_count = self.seq_len - len(imgs)
                    last_img = imgs[-1] if len(imgs) > 0 else torch.zeros(3, self.img_size[0], self.img_size[1])
                    for _ in range(pad_count):
                        imgs.append(last_img)
            else:
                imgs = [torch.zeros(3, self.img_size[0], self.img_size[1]) for _ in range(self.seq_len)]

            return {
                'map': sample['map'],
                'id': sample['id'],
                'variants': sample['variants'],
                'scenario': sample['scenario'],
                'videos': imgs,
                'gt_actor': sample['gt_actor'],
                'gt_ego': sample['gt_ego'],
            }

        # Legacy dataset mode
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

        # Pad with zero or last frames if video is short
        if len(imgs) < self.seq_len:
            pad_count = self.seq_len - len(imgs)
            last_img = imgs[-1] if len(imgs) > 0 else torch.zeros(3, self.img_size[0], self.img_size[1])
            for _ in range(pad_count):
                imgs.append(last_img)

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
            'videos': imgs,        # list of seq_len tensors with shape (3, img_h, img_w)
            'gt_actor': gt_actor,  # shape: (64,)
            'gt_ego': gt_ego,      # shape: ()
        }
