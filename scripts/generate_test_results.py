import argparse
import os
import sys
from tqdm import tqdm
import pickle

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')   
import taco_test
from model import generate_model
from utils import *
from parser_test import get_test_parser
from eval_taco import plot_mvit
torch.backends.cudnn.benchmark = True
import cv2

def plot_slot_test(
    attn,
    raw,
    map_name,
    clip_id,
    v,
    logdir,
    pred_actor=None,
    threshold=0.5,
    alpha=0.4,
    use_pred_filter=True,
    draw_heatmap=False,
):
    """Trực quan hóa vùng Attention của các Slot trên tập Test (Không cần nhãn Ground Truth).
    
    Args:
        attn: Tensor chú ý từ mô hình, shape (B, T, N_slots, H_feat, W_feat)
        raw: List gồm các frame ảnh gốc (Tensor) hoặc Tensor (T, C, H, W)
        map_name (str): Tên map/town (ví dụ: 'Town10HD')
        clip_id (str): ID của clip test
        v (str): Góc nhìn hoặc camera ID
        logdir (str): Thư mục lưu kết quả
        pred_actor: Output logits của actor từ Allocated_Head, shape (B, N_classes)
        threshold (float): Ngưỡng lọc mask nhị phân sau min-max normalization
        alpha (float): Độ trong suốt khi đè mask màu lên ảnh gốc
        use_pred_filter (bool): Nếu True, chỉ hiển thị slot mà mô hình dự đoán dương tính (> 0.5)
        draw_heatmap (bool): Nếu True, vẽ heatmap tổng hợp (dạng Jet); nếu False, vẽ binary masks phân màu
    """
    # 1. Khởi tạo đường dẫn lưu ảnh
    out_dir = os.path.join(logdir, "test_vis", f"{map_name}_{clip_id}_{v}")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Xử lý tensor dự đoán (nếu có)
    active_slot_indices = []
    if pred_actor is not None and use_pred_filter:
        pred_probs = torch.sigmoid(pred_actor[0]).detach().cpu()
        active_slot_indices = (pred_probs > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(active_slot_indices, int):
            active_slot_indices = [active_slot_indices]
    
    # 3. Chuẩn hóa chuỗi ảnh gốc raw: (T, H, W, 3) với định dạng BGR [0, 255] cho OpenCV
    if isinstance(raw, list):
        raw = torch.stack(raw, dim=0)
    if raw.dim() == 5:  # (B, T, C, H, W) hoặc (T, B, C, H, W)
        if raw.shape[0] == 1:
            raw = raw.squeeze(0)
        else:
            raw = raw.permute(1, 0, 2, 3, 4)[0]

    # Resize raw frames về chuẩn (128, 384)
    # raw đầu vào: (T, C, H, W) -> F.interpolate nhận (N, C, H, W)
    if raw.shape[-2:] != (128, 384):
        raw_resized = F.interpolate(raw, size=(128, 384), mode='bilinear', align_corners=False)
    else:
        raw_resized = raw

    raw_frames = raw_resized.permute(0, 2, 3, 1).detach().cpu().numpy()
    if raw_frames.max() <= 1.0:
        raw_frames = (raw_frames * 255.0).astype(np.uint8)
    else:
        raw_frames = raw_frames.astype(np.uint8)

    # Chuyển kênh RGB sang BGR để ghi bằng OpenCV
    raw_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in raw_frames]

    # 4. Xử lý và nội suy Attention Map: (T, N_slots, 128, 384)
    attn = attn.detach()
    if attn.dim() == 5:
        attn = attn[0]  # Bỏ batch dim -> (T, N_slots, H, W)

    T_len, N_slots, H_a, W_a = attn.shape
    # Gộp T và N_slots để nội suy song tuyến 2D nhanh chóng
    attn_flat = attn.reshape(T_len * N_slots, 1, H_a, W_a)
    attn_interpolated = F.interpolate(attn_flat, size=(128, 384), mode='bilinear', align_corners=False)
    attn_maps = attn_interpolated.reshape(T_len, N_slots, 128, 384).cpu().numpy()

    # Nếu không lọc theo threshold pred_actor hoặc không có pred_actor, lấy toàn bộ slots
    if not active_slot_indices:
        active_slot_indices = list(range(N_slots))

    # Bảng màu cố định cho các slot (định dạng BGR cho OpenCV)
    COLOR_PALETTE = [
        np.array([0, 0, 255]),      # Red
        np.array([0, 255, 0]),      # Green
        np.array([255, 0, 0]),      # Blue
        np.array([0, 255, 255]),    # Yellow
        np.array([255, 0, 255]),    # Magenta
        np.array([255, 255, 0]),    # Cyan
        np.array([0, 165, 255]),    # Orange
        np.array([128, 0, 128]),    # Purple
        np.array([0, 128, 128]),    # Olive
        np.array([205, 133, 63]),   # Steel Blue
    ]

    # 5. Duyệt và kết xuất từng frame
    for t in range(T_len):
        base_img = raw_frames_bgr[t].copy().astype(np.float32)

        if draw_heatmap:
            # Chế độ Heatmap: gom tổng attention của các slot được chọn
            combined_attn = np.zeros((128, 384), dtype=np.float32)
            for idx in active_slot_indices:
                if idx < N_slots:
                    combined_attn += attn_maps[t, idx]
            
            # Chuẩn hóa về [0, 255]
            denom = (combined_attn.max() - combined_attn.min()) + 1e-8
            norm_heat = ((combined_attn - combined_attn.min()) / denom * 255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)

            # Trộn với ảnh gốc
            blended = cv2.addWeighted(heatmap.astype(np.float32), alpha, base_img, 1.0 - alpha, 0)
            out_img = np.clip(blended, 0, 255).astype(np.uint8)

        else:
            # Chế độ Binary Mask đè màu theo từng Slot
            overlay = base_img.copy()
            applied_any_mask = False

            for slot_order, idx in enumerate(active_slot_indices):
                if idx >= N_slots:
                    continue
                
                mask_i = attn_maps[t, idx]
                m_min, m_max = mask_i.min(), mask_i.max()
                norm_mask = (mask_i - m_min) / (m_max - m_min + 1e-8)
                binary_mask = norm_mask > threshold

                if not binary_mask.any():
                    continue

                color = COLOR_PALETTE[slot_order % len(COLOR_PALETTE)]
                overlay[binary_mask] = color
                applied_any_mask = True

            if applied_any_mask:
                out_img = cv2.addWeighted(overlay, alpha, base_img, 1.0 - alpha, 0)
                out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            else:
                out_img = raw_frames_bgr[t]

        # Ghi ảnh trực tiếp ra ổ đĩa
        img_out_path = os.path.join(out_dir, f"frame_{t:02d}.jpg")
        cv2.imwrite(img_out_path, out_img)


class Engine(object):
    """Engine that runs training and inference.
    self.args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, args, logdir, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.args = args
        self.num_actor_class= args.num_slots
        self.logdir= logdir
    def test(self, model, dataloader, epoch):
        save_results = {}
        model.eval()
        with torch.no_grad():   
            for batch_num, data in enumerate(tqdm(dataloader)):

                # -------get video name------
                map = data['map'][0]
                id = data['id'][0]
                v = data['variants'][0]
                video_in = data['videos']
                scenario = map + '/'+id + '/' + v
                raw = data['raw']
                # -------get input------                    
                inputs = []
                for i in range(self.args.seq_len):
                    inputs.append(video_in[i].to(self.args.device, dtype=torch.float32))

                batch_size = inputs[0].shape[0]
                # -------get prediction------  
                if ('slot' in self.args.model_name) or self.args.box or 'mvit' in self.args.model_name:
                        pred_ego, pred_actor, attn = model(inputs)
                        if self.args.plot:
                            
                            if ('mvit' in self.args.model_name):
                                channel_idx = [-1]
                                for j,(attn,thw) in enumerate(attn):
                                    for c_idx in channel_idx:
                                        plot_mvit(attn[0], c_idx, raw, self.logdir , id, v, j, grid_size=(thw[1],thw[2]))
                            else:
                                plot_slot_test(attn, raw, map, id, v, self.logdir, pred_actor, self.args.plot_threshold)
                        
                else:
                    pred_ego, pred_actor = model(inputs)

                # -------transform object-detector-based's output (instance-level) to multilabel ------
        #         if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
        #             pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
        #             _, pred_actor_idx = torch.max(pred_actor.data, -1)
        #             pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
        #             map_batch_new_pred_actor = []
        #             for i, b in enumerate(pred_actor_idx):
        #                 map_new_pred = np.zeros(self.num_actor_class, dtype=np.float32)+1e-5
        #                 for j, pred in enumerate(b):
        #                     if pred != self.num_actor_class:
        #                         if pred_actor[i, j, pred] > map_new_pred[pred]:
        #                             map_new_pred[pred] = pred_actor[i, j, pred]
        #                 map_batch_new_pred_actor.append(map_new_pred)
        #             map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
        #             save_results[scenario] = map_batch_new_pred_actor
        #         # -------output multilabel for video-level models and Action-slot------
        #         else:
        #             pred_actor = torch.sigmoid(pred_actor)
        #             pred_actor = pred_actor.detach().cpu().numpy()
        #             save_results[scenario] = pred_actor

        # with open('prediction_results.pkl', 'wb') as handle:
        #     pickle.dump(save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    torch.cuda.empty_cache()
    args, logdir = get_test_parser()
    print(args)
    torch.cuda.empty_cache() 
    # -------start testing------
    seq_len = args.seq_len
    num_ego_class = 4
    num_actor_class = 64

    # Data
    test_set = taco_test.TACO_TEST(args=args, split=args.split)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = generate_model(args, num_ego_class, num_actor_class).cuda()
    trainer = Engine(args, logdir)

    # model_path = os.path.join(args.cp)
    # print(f"===> Đang nạp weights từ: {model_path}")
    # checkpoint = torch.load(model_path, map_location='cpu')

    # # 1. Bóc tách dictionary nếu bị lồng key
    # if isinstance(checkpoint, dict):
    #     if 'model_state_dict' in checkpoint:
    #         state_dict = checkpoint['model_state_dict']
    #     elif 'state_dict' in checkpoint:
    #         state_dict = checkpoint['state_dict']
    #     elif 'model' in checkpoint:
    #         state_dict = checkpoint['model']
    #     else:
    #         state_dict = checkpoint
    # else:
    #     state_dict = checkpoint

    trainer.test(model, dataloader_test, None)

if __name__ == "__main__":

    main()
