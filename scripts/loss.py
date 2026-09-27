import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils import inter_and_union

class ActionSlotLoss(nn.Module):
    def __init__(self, args, num_actor_class, attention_res=None):
        super(ActionSlotLoss, self).__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.attention_res = attention_res
        self.ego_ce = nn.CrossEntropyLoss(reduction='mean')
        self.actor_loss_type =  self._parse_actor_loss(args)
        self.attn_loss_type = self._parse_attn_loss(args)
        self.obj_slot_bce = nn.BCELoss(reduction='mean')

        mapping = torch.zeros(64, dtype=torch.long)
        mapping[0:12] = 0    # c
        mapping[12:24] = 1   # c+
        mapping[24:36] = 2   # b
        mapping[36:48] = 3   # b+
        mapping[48:56] = 4   # p
        mapping[56:64] = 5   # p+
        self.register_buffer('action_to_obj', mapping)

    def _parse_actor_loss(self,args):
        if ('slot' in args.model_name and not args.allocated_slot) or args.box:
            ce_weights = torch.ones(self.num_actor_class+1)*args.ce_pos_weight
            ce_weights[-1] = args.ce_neg_weight
            self.bce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights)
            return 1 
        else:
            pos_weight = torch.ones([self.num_actor_class])*args.bce_pos_weight
            self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
            return 0

    def _parse_attn_loss(self,args):
        flag = 0
        if (('slot' in args.model_name and not args.allocated_slot) or args.box) and args.obj_mask:
            flag = 1
        elif 'slot' in args.model_name and args.allocated_slot:
            if not args.bg_mask and args.action_attn_weight>0:
                flag = 2
            elif args.obj_mask:
                flag = 1
            elif args.bg_slot and args.bg_mask and args.action_attn_weight>0. and args.bg_attn_weight>0.:
                flag = 3
            elif args.bg_slot and args.bg_mask and args.bg_attn_weight>0. and not args.action_attn_weight > 0.:
                flag = 4
        if flag >0:
            self.obj_bce = nn.BCELoss()

        
        return flag
    
    def ego_loss(self,pred, label):
        if pred is None:
            return None
        return self.ego_ce(pred, label)

    def actor_loss(self, pred, label):
        if self.actor_loss_type == 1:
            bs, num_queries = pred.shape[:2]
            out_prob = pred.clone().detach().flatten(0, 1).softmax(-1)
            actor_gt_np = label.clone().detach()
            tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
            C = -out_prob[:, tgt_ids].clone().detach()
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v) for v in actor_gt_np]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
            src_idx = torch.cat([src for (src, _) in indx]).detach()
            idx = (batch_idx, src_idx)
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(label, indx)]).cuda()
            target_classes = torch.full(pred.shape[:2], self.num_actor_class,
                                dtype=torch.int64, device=out_prob.device)

            target_classes[idx] = target_classes_o
            actor_loss = self.bce(pred.transpose(1, 2), target_classes)
        else:
            # print(pred.device,label.device)
            actor_loss = self.bce(pred, label)
        return actor_loss

    def attn_loss(self, attn, label, actor,validate):
        attn_loss = None
        bg_attn_loss = None
        action_attn = None
        bg_attn = None

        # prepare background mask
        if self.args.bg_attn_weight>0:
            bg_seg = []
            bg_seg_in = label['bg_seg']
            for i in range(self.args.seq_len//self.args.mask_every_frame):
                bg_seg.append(bg_seg_in[i].to(self.args.device, dtype=torch.float32))
            h, w = bg_seg[0].shape[-2], bg_seg[0].shape[-1]
            bg_seg = torch.stack(bg_seg, 0)
            l, b, _, h, w = bg_seg.shape
            bg_seg = torch.reshape(bg_seg, (l, b, h, w))
            bg_seg = torch.permute(bg_seg, (1, 0, 2, 3)) #[batch, len, h, w]
            
        # object mask supervision for action slot
        if self.attn_loss_type == 1:
            obj_mask_list = []
            obj_mask = label['obj_masks']
            for i in range(self.args.seq_len//self.args.mask_every_frame):
                obj_mask_list.append(obj_mask[i].to(self.args.device, dtype=torch.float32))
            obj_mask_list = torch.stack(obj_mask_list, 0)
            obj_mask_list = torch.permute(obj_mask_list, (1, 0, 2, 3, 4)) #[batch, len, n, h, w]
            b, l, n, h, w = obj_mask_list.shape

            attn_loss = 0.0
            # 8, 16, 64, 8, 24
            b, l, n, h, w = attn.shape
            attn = torch.reshape(attn, (-1, 1 ,8, 24))
            attn = F.interpolate(attn, size=(32, 96))
            attn = torch.reshape(attn, (b, l , n, 32, 96))

            # sup_idx = [1,3,5,7,9,11,13,15]
            # attn = attn[:,sup_idx,:,:,:].reshape((b,8,n,32,96))
            attn = attn[:, ::self.args.mask_every_frame, :, :, :].reshape(b, -1, n, 32, 96)

            b, seq, n_obj, h, w = obj_mask_list.shape
            mask_detach = attn.detach().flatten(3,4)
            mask_detach = mask_detach.cpu().numpy()
            mask_gt_np = obj_mask_list.flatten(3,4)
            mask_gt_np = mask_gt_np.detach().cpu().numpy()
            scores = np.zeros((b, 4, n, n_obj))
            for i in range(b):
                for j in range(4):
                    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
                    scores[i,j] += cross_entropy_cur
            for i in range(b):
                for j in range(4):
                    matches = linear_sum_assignment(-scores[i,j])
                    id_slot, id_gt = matches 
                    attn_loss += self.obj_bce(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])

        elif self.attn_loss_type == 2:
            b, l, n, h, w = attn.shape
            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            action_attn = attn[:, :, :self.num_actor_class, :, :]

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32).cuda()
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

        # Action-slot, background mask + negative mask
        elif self.attn_loss_type == 3:
            b, l, n, h, w = attn.shape

            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            action_attn = attn[:, :, :self.num_actor_class, :, :]
            bg_attn = attn[:, ::self.args.mask_every_frame, -1, :, :].reshape(b, -1, h, w)

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32).cuda()

            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])
            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)
            # attn_loss = self.args.action_attn_weight*action_attn_loss + self.args.bg_attn_weight*bg_attn_loss
            
        # background mask only
        elif self.attn_loss_type == 4:
            b, l, n, h, w = attn.shape

            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            bg_attn = attn[:, ::self.args.mask_every_frame, -1, :, :].reshape(b, l//self.args.mask_every_frame, h, w)
            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)

        loss = {'attn_loss':attn_loss,'bg_attn_loss':bg_attn_loss}
        if validate:
            loss['action_inter'] = None
            loss['action_union'] = None
            loss['bg_inter'] = None
            loss['bg_union'] = None

            if action_attn is not None:
                action_attn_pred = action_attn[class_idx] > 0.5
                inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                loss['action_inter'] = inter
                loss['action_union'] = union
            if bg_attn is not None:
                bg_attn_pred = bg_attn > 0.5
                inter, union = inter_and_union(bg_attn_pred, bg_seg, 1, 1)
                loss['bg_inter'] = inter
                loss['bg_union'] = union

        return loss

    def object_slot_loss(self, obj_attn, obj_gt):
        """
        obj_attn: [B, K, thw] (K = 7 nếu có bg_slot, K = 6 nếu không có)
        obj_gt:   [B, 6] (tensor float32 nhị phân trên cùng device)
        """
        if obj_attn is None:
            return torch.tensor(0.0, device=self.args.device)

        # Tách Background mask nếu có bg_slot (7 slots -> lấy 6 object slots)
        if self.args.bg_slot and obj_attn.shape[1] == 7:
            action_obj_attn = obj_attn[:, 1:, :]  # [B, 6, thw]
        else:
            action_obj_attn = obj_attn             # [B, 6, thw]

        loss = torch.tensor(0.0, device=obj_attn.device)

        # 1. Triệt tiêu các slot vắng mặt (Absent slots -> ép attention mask về 0)
        absent_mask = (obj_gt == 0.0)  # Tensor bool shape [B, 6]
        if absent_mask.any():
            empty_slots_attn = action_obj_attn[absent_mask]  # [num_absent, thw]
            loss += 0.5 * F.binary_cross_entropy(empty_slots_attn, torch.zeros_like(empty_slots_attn))

        # 2. Ép co cụm không gian (Spatial Compactness / Entropy) cho các slot thực sự xuất hiện
        present_mask = (obj_gt == 1.0)  # Tensor bool shape [B, 6]
        if present_mask.any():
            active_slots = action_obj_attn[present_mask]     # [num_present, thw]
            p = active_slots / (active_slots.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
            loss += 0.05 * entropy.mean()

        return loss

    def forward(self, pred, label, validate=False):
        ego_loss = self.ego_loss(pred['ego'], label['ego'])

        # Bóc tách action_attn và obj_attn từ dictionary
        if isinstance(pred['attn'], dict):
            action_attn = pred['attn']['action_attn']
            obj_attn = pred['attn']['obj_attn']
        else:
            action_attn = pred['attn']
            obj_attn = None

        actor_loss = self.actor_loss(pred['actor'], label['actor'])
        attention_loss = self.attn_loss(action_attn, label, pred['actor'], validate)

        # ----------------------------------------------------
        # Xây dựng ground-truth obj_gt [B, 6] từ label['actor'] [B, 64]
        # ----------------------------------------------------
        actor_label = label['actor']
        B = actor_label.shape[0]
        obj_gt = torch.zeros((B, 6), device=actor_label.device, dtype=torch.float32)
        obj_gt[:, 0] = (actor_label[:, 0:12].sum(dim=-1) > 0).float()   # c 
        obj_gt[:, 1] = (actor_label[:, 12:24].sum(dim=-1) > 0).float()  # c+
        obj_gt[:, 2] = (actor_label[:, 24:36].sum(dim=-1) > 0).float()  # b 
        obj_gt[:, 3] = (actor_label[:, 36:48].sum(dim=-1) > 0).float()  # b+
        obj_gt[:, 4] = (actor_label[:, 48:56].sum(dim=-1) > 0).float()  # p 
        obj_gt[:, 5] = (actor_label[:, 56:64].sum(dim=-1) > 0).float()  # p+

        # 1. Loss attention mask (TRUYỀN ĐÚNG obj_gt THAY VÌ label)
        obj_slot_loss = self.object_slot_loss(obj_attn, obj_gt)

        # 2. Loss phân loại trực tiếp cho object slots từ obj_head
        pred_obj = pred.get('pred_obj', None)
        if pred_obj is not None:
            obj_cls_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_gt)
        else:
            obj_cls_loss = torch.tensor(0.0, device=self.args.device)

        total_obj_loss = obj_slot_loss + obj_cls_loss

        return {
            "ego": ego_loss,
            "actor": actor_loss,
            "attn": attention_loss,
            "obj_slot_loss": total_obj_loss
        }