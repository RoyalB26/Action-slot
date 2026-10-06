import os
import json
from PIL import Image
import cv2
import torch.nn.functional as F

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import sys
import json 
import random
import torchvision.transforms as transforms
import lmdb
import io
import pickle
class TACO(Dataset):
    def __init__(self, 
                args,
                lmdb_train_path, 
                lmdb_val_path,
                split='val',
                root='',
                Max_N=20,
                accelerator= None):
        self.args= args
        self.split= split
        self.root= root
        self.Max_N= Max_N
        self.accelerator= accelerator
        
        self.videos_list= []
        self.id = []
        self.variants= []
        self.gt_ego = []
        self.gt_actor = []
        self.num_class = 64
        self.maps= []
        self.lmdb_path = lmdb_val_path
        path_to_idx= []
        self._env= None
        if split == 'train':
            self.lmdb_path = lmdb_train_path
            with open("/kaggle/input/datasets/royalb26/taco-dataset/taco_path_to_idx.json", 'r', encoding= 'utf-8') as f:
                path_to_idx= json.load(f)
        else:
            with open("/kaggle/input/datasets/royalb26/taco-val-dataset/taco_path_to_idx.json", 'r', encoding= 'utf-8') as f:
                path_to_idx= json.load(f)
        
        f_label = open('../datasets/taco_'+split+'_label.json')
        label_list = json.load(f_label)

        dir_to_idx= {}
        for k, v in path_to_idx.items():
            parts= k.split('/')
            path= parts[7:11]
            path= "/".join(path)
            if path not in dir_to_idx:
                dir_to_idx[path]= v
        
        for step_idx, (scenario, v) in enumerate(label_list.items()):
            parent_folder, basic, variant = scenario.split('/')
            sample_dir= os.path.join(parent_folder,basic,'variant_scenario',variant)
            idx= dir_to_idx.get(sample_dir, -1)
            if idx != -1:
                self.videos_list.append(idx)
                gt = v
                self.id.append(basic)
                self.variants.append(variant)
                gt_ego, gt_actor = get_labels(args, gt, num_slots=args.num_slots)
                self.gt_ego.append(gt_ego)
                self.gt_actor.append(gt_actor)
                self.maps.append(parent_folder)
            
        print('num_videos: ' + str(len(self.variants)))

    @property
    def env(self):
        # Nếu chưa mở thì mở, nếu mở rồi thì dùng lại
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        return self._env
    
    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.videos_list)

    def __getitem__(self, index):
        data = dict()
        data['videos'] = []
        data['bg_seg'] = []
        data['obj_masks'] = []
        data['raw'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]
        data['map'] = self.maps[index]

        for frame_idx, idx in enumerate(range(index, index + 16)):
            with self.env.begin() as txn:
                raw_data = txn.get(f"{idx:08d}".encode("ascii"))
                sample = pickle.loads(raw_data)

            frame = Image.open(io.BytesIO(sample["frame"])).convert("RGB")
            data["videos"].append(frame)

            if self.args.plot:
                data["raw"].append(frame)

            if self.split == "train" or self.split == "val":
                # Kiểm tra theo frame_idx (cục bộ từ 0 -> 16), KHÔNG dùng idx
                if (
                    self.args.bg_mask
                    and frame_idx % self.args.mask_every_frame == 0
                ):
                    if sample["bg"] is not None:
                        bg = Image.open(io.BytesIO(sample["bg"])).convert("L")
                        data["bg_seg"].append(bg)
                    else:
                        pass

                if self.args.obj_mask:
                    if frame_idx % self.args.mask_every_frame == 0 or (
                        self.args.plot and self.args.plot_mode == ""
                    ):
                        npy_array = np.load(io.BytesIO(sample["npy"]))
                        data["obj_masks"].append(get_obj_mask(npy_array))

        while len(data["bg_seg"]) < len(data["videos"]):
            data['bg_seg'].append(data['bg_seg'][-1])

        data["videos"] = to_np(
            data["videos"], self.args.model_name, self.args.backbone
        )
        

        data["bg_seg"] = to_np_no_norm(data["bg_seg"])

        # Đảm bảo obj_masks cũng được stack thành tensor nếu có sử dụng
        if self.args.obj_mask and len(data["obj_masks"]) > 0:
            data["obj_masks"] = torch.stack(data["obj_masks"], dim=0)

        return data
                            
def get_obj_mask(obj_masks):
    if obj_masks.shape[0] == 0:
        obj_masks = torch.zeros([64, 32, 96], dtype=torch.int32)
    else:
        obj_masks = torch.from_numpy(np.stack(obj_masks, 0))
    # img = torch.flip(torch.from_numpy(img).type(torch.int).permute(2,0,1),[0])
    obj_masks = obj_masks.type(torch.int)
    pad_num = 64 - obj_masks.shape[0]
    obj_masks = torch.cat((obj_masks, torch.zeros([pad_num, 32, 96], dtype=torch.int32)), dim=0)
    obj_masks = obj_masks.type(torch.float32)

    return obj_masks



def get_labels(args, gt, num_slots=64):   
    num_class = 64
    agent_label = gt['agents']
    ego_label = gt['ego']

    ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3}

    actor_table = { 'c:z1-z2': 0, 'c:z1-z3':1, 'c:z1-z4':2,
                    'c:z2-z1': 3, 'c:z2-z3': 4, 'c:z2-z4': 5,
                    'c:z3-z1': 6, 'c:z3-z2': 7, 'c:z3-z4': 8,
                    'c:z4-z1': 9, 'c:z4-z2': 10, 'c:z4-z3': 11,

                    'c+:z1-z2': 12, 'c+:z1-z3':13, 'c+:z1-z4':14,
                    'c+:z2-z1': 15, 'c+:z2-z3': 16, 'c+:z2-z4': 17,
                    'c+:z3-z1': 18, 'c+:z3-z2': 19, 'c+:z3-z4': 20,
                    'c+:z4-z1': 21, 'c+:z4-z2': 22, 'c+:z4-z3': 23,

                    'b:z1-z2': 24, 'b:z1-z3':25, 'b:z1-z4':26,
                    'b:z2-z1': 27, 'b:z2-z3': 28, 'b:z2-z4': 29,
                    'b:z3-z1': 30, 'b:z3-z2': 31, 'b:z3-z4': 32,
                    'b:z4-z1': 33, 'b:z4-z2': 34, 'b:z4-z3': 35,

                    'b+:z1-z2': 36, 'b+:z1-z3':37, 'b+:z1-z4':38,
                    'b+:z2-z1': 39, 'b+:z2-z3': 40, 'b+:z2-z4': 41,
                    'b+:z3-z1': 42, 'b+:z3-z2': 43, 'b+:z3-z4': 44,
                    'b+:z4-z1': 45, 'b+:z4-z2': 46, 'b+:z4-z3': 47,


                    'p:c1-c2': 48, 'p:c1-c4': 49, 
                    'p:c2-c1': 50, 'p:c2-c3': 51, 
                    'p:c3-c2': 52, 'p:c3-c4': 53, 
                    'p:c4-c1': 54, 'p:c4-c3': 55,

                    'p+:c1-c2': 56, 'p+:c1-c4': 57, 
                    'p+:c2-c1': 58, 'p+:c2-c3': 59, 
                    'p+:c3-c2': 60, 'p+:c3-c4': 61, 
                    'p+:c4-c1': 62, 'p+:c4-c3': 63 
                    }

    ego_label = torch.tensor(ego_label)
    agent_label = torch.FloatTensor(agent_label)
    return ego_label, agent_label


def scale(image, scale=2.0, model_name=None):

    if scale == -1.0:
        (width, height) = (224, 224)
    else:
        (width, height) = (int(image.width // scale), int(image.heighft // scale))
    # (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    return im_resized



def to_np(v, model_name, backbone):
    if backbone != 'inception':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
    else:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v

def to_np_no_norm(v):
    transform = transforms.Compose([
                transforms.ToTensor(),
                ])
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v


