"""TACO Dataset loader supporting LMDB and disk image folder structures."""

import os
import json
import io
import pickle
from PIL import Image
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.normalization import get_mean, get_std
from dataset.transforms import build_train_video_transforms


class TACO(Dataset):
    """TACO (Traffic Action and Condition) video dataset loader."""

    def __init__(
        self,
        args,
        lmdb_train_path=None,
        lmdb_val_path=None,
        split='val',
        root='',
        Max_N=20,
        accelerator=None,
        path_to_idx_file=None,
        label_file=None,
    ):
        """
        Args:
            args (argparse.Namespace): Arguments object.
            lmdb_train_path (str, optional): Path to LMDB train database.
            lmdb_val_path (str, optional): Path to LMDB val database.
            split (str): 'train' or 'val'.
            root (str): Fallback root directory of raw video frames.
            Max_N (int): Maximum object/slot bound.
            accelerator (object, optional): Distributed accelerator instance.
            path_to_idx_file (str, optional): Path to JSON mapping paths to LMDB keys.
            label_file (str, optional): Path to TACO split label JSON file.
        """
        self.args = args
        self.split = split
        self.root = root
        self.Max_N = Max_N
        self.accelerator = accelerator

        self.videos_list = []
        self.id = []
        self.variants = []
        self.gt_ego = []
        self.gt_actor = []
        self.num_class = 64
        self.maps = []
        self._env = None

        if split == 'train':
            target_lmdb_dir = lmdb_train_path
        else:
            target_lmdb_dir = lmdb_val_path

        self.lmdb_path = None
        resolved_idx_file = path_to_idx_file

        if target_lmdb_dir and os.path.exists(target_lmdb_dir):
            if os.path.isdir(target_lmdb_dir):
                # Search for .lmdb file or folder inside target_lmdb_dir
                possible_lmdbs = [
                    os.path.join(target_lmdb_dir, f"taco_{split}.lmdb"),
                    os.path.join(target_lmdb_dir, f"{split}.lmdb"),
                    os.path.join(target_lmdb_dir, "taco.lmdb"),
                ]
                # Also check any file/folder ending with .lmdb or containing data.mdb
                for cand in possible_lmdbs:
                    if os.path.exists(cand):
                        self.lmdb_path = cand
                        break
                if self.lmdb_path is None:
                    for item in os.listdir(target_lmdb_dir):
                        item_path = os.path.join(target_lmdb_dir, item)
                        if item.endswith(".lmdb") or os.path.exists(os.path.join(item_path, "data.mdb")):
                            self.lmdb_path = item_path
                            break
                if self.lmdb_path is None and os.path.exists(os.path.join(target_lmdb_dir, "data.mdb")):
                    self.lmdb_path = target_lmdb_dir

                # Look for taco_path_to_idx.json in the folder if not explicitly provided
                if not resolved_idx_file or not os.path.exists(resolved_idx_file):
                    possible_idx = [
                        os.path.join(target_lmdb_dir, "taco_path_to_idx.json"),
                        os.path.join(target_lmdb_dir, f"taco_{split}_path_to_idx.json"),
                        os.path.join(target_lmdb_dir, "path_to_idx.json"),
                    ]
                    for cand_idx in possible_idx:
                        if os.path.exists(cand_idx):
                            resolved_idx_file = cand_idx
                            break
            else:
                self.lmdb_path = target_lmdb_dir

        path_to_idx = {}
        if resolved_idx_file and os.path.exists(resolved_idx_file):
            with open(resolved_idx_file, 'r', encoding='utf-8') as f:
                path_to_idx = json.load(f)

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

        # Build scenario-to-frame-indices mapping from path_to_idx
        # Path format: .../<category_or_map>/<scenario_id>/variant_scenario/<scenario_num>/rgb/downsampled/<frame_file>.jpg
        from collections import defaultdict
        scenario_to_frames = defaultdict(list)
        for full_path, lmdb_idx in path_to_idx.items():
            parts = full_path.split('/')
            if 'rgb' in parts:
                rgb_pos = parts.index('rgb')
                var_scenario = parts[rgb_pos - 1]
                scenario_id = parts[rgb_pos - 3]
                cat_or_map = parts[rgb_pos - 4]
                key = f"{cat_or_map}/{scenario_id}/{var_scenario}"
                frame_file = parts[-1]
                scenario_to_frames[key].append((frame_file, lmdb_idx))
            else:
                key = "/".join(parts[7:11]) if len(parts) >= 11 else full_path
                scenario_to_frames[key].append((parts[-1], lmdb_idx))

        # Process each scenario from label_list
        for scenario_key, label_meta in label_list.items():
            parts = scenario_key.split('/')
            if len(parts) >= 3:
                self.maps.append(parts[0])
                self.id.append(parts[1])
                self.variants.append(parts[2])
            elif len(parts) >= 2:
                self.maps.append(parts[0])
                self.id.append(parts[1])
                self.variants.append(parts[1])
            else:
                self.maps.append(scenario_key)
                self.id.append(scenario_key)
                self.variants.append(scenario_key)

            self.gt_ego.append(label_meta.get('ego', 0))
            # Actor multi-label vector: support both 'agents' (length 64) and legacy 'actor'
            if 'agents' in label_meta:
                self.gt_actor.append(label_meta['agents'])
            elif 'actor' in label_meta:
                self.gt_actor.append(label_meta['actor'])
            else:
                self.gt_actor.append([0] * self.num_class)

            # Match scenario frames in LMDB
            if scenario_key in scenario_to_frames:
                # Sort frames by filename and take first 16 frame indices
                sorted_frames = sorted(scenario_to_frames[scenario_key], key=lambda x: x[0])
                frame_indices = [item[1] for item in sorted_frames[:16]]
                self.videos_list.append(frame_indices)
            else:
                self.videos_list.append(scenario_key)

        img_h = getattr(args, 'img_height', 128)
        img_w = getattr(args, 'img_width', 384)
        self.img_size = (img_h, img_w)

        # Video transforms: video-level augmentation for train, deterministic for val
        use_aug = getattr(args, 'augment', False) and (split == 'train')
        self.clip_transform = build_train_video_transforms(self.img_size, augment=use_aug)

    @property
    def env(self):
        """Lazy initialization of LMDB environment."""
        if self._env is None and self.lmdb_path and os.path.exists(self.lmdb_path):
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                max_dbs=1,
            )
        return self._env

    def __len__(self):
        """Return total number of video samples."""
        return len(self.gt_ego)

    def __getitem__(self, idx):
        """Fetch video clip frames and labels for a single sample.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Sample dictionary with videos, gt_actor, and gt_ego.
        """
        video_key = self.videos_list[idx]
        raw_pil_imgs = []

        # Read from LMDB if available
        if self.env is not None and isinstance(video_key, list):
            with self.env.begin(write=False) as txn:
                for frame_idx in video_key:
                    byteflow = txn.get(f"{frame_idx:08}".encode('ascii'))
                    if byteflow is not None:
                        entry = pickle.loads(byteflow)
                        frame_bytes = entry['frame'] if isinstance(entry, dict) and 'frame' in entry else entry
                        if isinstance(frame_bytes, bytes):
                            img = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
                        else:
                            img = Image.fromarray(np.uint8(frame_bytes)).convert('RGB')
                        raw_pil_imgs.append(img)
        # Fallback for single integer index containing list of frames
        elif self.env is not None and isinstance(video_key, int):
            with self.env.begin(write=False) as txn:
                byteflow = txn.get(f"{video_key:08}".encode('ascii'))
            if byteflow is not None:
                entry = pickle.loads(byteflow)
                imgs_bytes = entry if isinstance(entry, list) else [entry]
                for b in imgs_bytes:
                    img_data = b['frame'] if isinstance(b, dict) and 'frame' in b else b
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    raw_pil_imgs.append(img)
        # Otherwise fallback to raw images on disk
        elif self.root and os.path.exists(self.root):
            scenario_path = os.path.join(self.root, str(video_key))
            if os.path.isdir(scenario_path):
                frame_files = sorted(os.listdir(scenario_path))
                for f_name in frame_files[:16]:
                    img_path = os.path.join(scenario_path, f_name)
                    img = Image.open(img_path).convert('RGB')
                    raw_pil_imgs.append(img)

        # Pad with last image if sequence is short
        if len(raw_pil_imgs) < 16:
            pad_needed = 16 - len(raw_pil_imgs)
            fallback_img = raw_pil_imgs[-1] if len(raw_pil_imgs) > 0 else Image.new('RGB', (self.img_size[1], self.img_size[0]))
            for _ in range(pad_needed):
                raw_pil_imgs.append(fallback_img)
        else:
            raw_pil_imgs = raw_pil_imgs[:16]

        # Apply video clip transform (consistent augmentation across 16 frames)
        imgs = self.clip_transform(raw_pil_imgs)

        # Multi-label binary target for actor actions: shape (64,)
        gt_actor = torch.zeros(self.num_class, dtype=torch.float32)
        raw_actor = self.gt_actor[idx]
        if isinstance(raw_actor, list):
            if len(raw_actor) == self.num_class:
                gt_actor = torch.tensor(raw_actor, dtype=torch.float32)
            else:
                gt_actor[raw_actor] = 1.0
        elif isinstance(raw_actor, (int, np.integer)):
            gt_actor[raw_actor] = 1.0

        # Categorical target for ego-vehicle action: shape ()
        gt_ego = torch.tensor(self.gt_ego[idx], dtype=torch.long)

        return {
            'map': self.maps[idx],
            'id': self.id[idx],
            'variants': self.variants[idx],
            'videos': imgs,           # list of 16 tensors with shape (3, 128, 384)
            'gt_actor': gt_actor,     # shape: (64,)
            'gt_ego': gt_ego,         # shape: ()
        }
