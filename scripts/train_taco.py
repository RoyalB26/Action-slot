import sys
sys.path.append('../')

import argparse
import json
import os
from PIL import Image
import numpy as np
from get_parser import parser
import math
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import average_precision_score
from scipy.optimize import linear_sum_assignment
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb

torch.backends.cudnn.benchmark = True

from datasets.taco import TACO
from model import generate_model
from loss import ActionSlotLoss
from utils import AverageMeter
import time
from put_lmdb import put_into_working
import warnings

# Suppress Python standard warnings (sklearn, PyTorch user warnings, etc.)
warnings.filterwarnings('ignore')

# Suppress native C++/CUDA and Hugging Face library-level warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

logger = get_logger(__name__)


def plot_result(result, args):
    """
    result : [mAP, loss]
    """
    text = ["mAP", "loss"]
    x = [i + 1 for i in range(0, args.epochs, args.val_every)]
    x = x if x[-1] == args.epochs else x + [args.epochs]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(2):
        ax[i].plot(x, result[:, i])
        ax[i].set_title(text[i])
    plt.show()


def lambda_lr(epoch):
    if epoch < 11:
        lr = math.pow(1.1, epoch)
    else:
        lr = math.pow(0.7, int(epoch / 3))
    return lr


def set_lr(model):
    target_model = getattr(model, "module", model)
    params = list(filter(lambda kv: kv[0].startswith("head"), target_model.named_parameters()))
    base_params = list(filter(lambda kv: not kv[0].startswith("head"), target_model.named_parameters()))
    return [
        {'params': [temp[1] for temp in base_params]},
        {'params': [temp[1] for temp in params], 'lr': 2e-2}
    ]


class Engine(object):
    """Engine that runs training and inference with Accelerate & WandB."""

    def __init__(self, args, model, optimizer, num_actor_class, accelerator, scheduler=None):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_actor_class = num_actor_class

        target_model = getattr(self.model, "module", self.model)
        if hasattr(target_model, 'resolution'):
            attention_res = (
                target_model.resolution[0] * args.bg_upsample,
                target_model.resolution[1] * args.bg_upsample
            )
        else:
            attention_res = None

        self.criterion = ActionSlotLoss(args, num_actor_class, attention_res).to(self.accelerator.device)

        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.best_mAP = 1e-5
        self.best_log = []
        self.reset_log()

    def reset_log(self):
        self.loss_epoch = 0.
        self.ego_loss_epoch = 0.
        self.seg_loss_epoch = 0.
        self.attn_loss_epoch = 0.
        self.action_attn_loss_epoch = 0.
        self.bg_attn_loss_epoch = 0.
        self.actor_loss_epoch = 0.
        self.correct_ego = 0
        self.total_ego = 0
        self.label_actor_list = []
        self.map_pred_actor_list = []
        self.action_inter = AverageMeter()
        self.action_union = AverageMeter()
        self.bg_inter = AverageMeter()
        self.bg_union = AverageMeter()

    def step(self, batch, mode):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.accelerator.device)

        video_in = batch['videos']
        if self.args.box:
            box_in = batch['box']
            if isinstance(box_in, np.ndarray):
                boxes = torch.from_numpy(box_in).to(self.accelerator.device, dtype=torch.float32)
            else:
                boxes = box_in.to(self.accelerator.device, dtype=torch.float32)

        inputs = [video_in[i].to(self.accelerator.device, dtype=torch.float32) for i in range(self.args.seq_len)]

        attn = None
        if self.args.box:
            pred_ego, pred_actor = self.model(inputs, boxes)
        else:
            if 'slot' in self.args.model_name or 'mvit' in self.args.model_name:
                pred_ego, pred_actor, attn = self.model(inputs)
            else:
                pred_ego, pred_actor = self.model(inputs)

        loss_dict = self.criterion(
            {'ego': pred_ego, 'actor': pred_actor, 'attn': attn},
            batch,
            False if mode == 'train' else True
        )

        ego_loss = loss_dict['ego']
        if ego_loss is None:
            ego_loss = torch.tensor([0.0], device=self.accelerator.device)
        actor_loss = loss_dict['actor']
        action_attn_loss = loss_dict['attn']['attn_loss']
        bg_attn_loss = loss_dict['attn']['bg_attn_loss']

        attn_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.criterion.attn_loss_type == 1:
            attn_loss = action_attn_loss
        elif self.criterion.attn_loss_type == 2:
            attn_loss = action_attn_loss * self.args.action_attn_weight
            self.attn_loss_epoch += float(attn_loss.item())
            self.action_attn_loss_epoch += float(action_attn_loss.item())
        elif self.criterion.attn_loss_type == 3:
            attn_loss = self.args.action_attn_weight * action_attn_loss + self.args.bg_attn_weight * bg_attn_loss
            self.attn_loss_epoch += float(attn_loss.item())
            self.action_attn_loss_epoch += float(action_attn_loss.item())
            self.bg_attn_loss_epoch += float(bg_attn_loss.item())
        elif self.criterion.attn_loss_type == 4:
            attn_loss = self.args.bg_attn_weight * bg_attn_loss
            self.attn_loss_epoch += float(attn_loss.item())
            self.bg_attn_loss_epoch += float(bg_attn_loss.item())

        if 'slot' in self.args.model_name and (self.args.action_attn_weight > 0. or self.args.bg_attn_weight > 0. or self.args.obj_mask):
            loss = actor_loss + self.args.ego_loss_weight * ego_loss + attn_loss
        else:
            loss = actor_loss + self.args.ego_loss_weight * ego_loss

        self.loss_epoch += float(loss.item())

        _, pred_ego_idx = torch.max(pred_ego.data, 1)
        ego, actor = batch['ego'], batch['actor']
        self.total_ego += ego.size(0)
        self.correct_ego += (pred_ego_idx == ego).sum().item()

        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            pred_actor_prob = torch.nn.functional.softmax(pred_actor, dim=-1)
            _, pred_actor_idx = torch.max(pred_actor_prob.data, -1)
            pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
            pred_actor_prob = pred_actor_prob.detach().cpu().numpy()

            map_batch_new_pred_actor = []
            for i, b in enumerate(pred_actor_idx):
                map_new_pred = np.zeros(self.num_actor_class, dtype=np.float32) + 1e-5
                for j, pred in enumerate(b):
                    if pred != self.num_actor_class:
                        if pred_actor_prob[i, j, pred] > map_new_pred[pred]:
                            map_new_pred[pred] = pred_actor_prob[i, j, pred]
                map_batch_new_pred_actor.append(map_new_pred)

            map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
            self.map_pred_actor_list.append(map_batch_new_pred_actor)
            self.label_actor_list.append(batch['slot_eval_gt'].cpu().numpy())
        else:
            pred_actor_prob = torch.sigmoid(pred_actor)
            self.map_pred_actor_list.append(pred_actor_prob.detach().cpu().numpy())
            self.label_actor_list.append(actor.detach().cpu().numpy())

        actor_loss_val, ego_loss_val = actor_loss.mean(), ego_loss.mean()
        self.actor_loss_epoch += float(actor_loss_val.item())
        self.ego_loss_epoch += float(ego_loss_val.item())

        if mode == 'train':
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
        else:
            if loss_dict['attn']['action_inter'] is not None:
                self.action_inter.update(loss_dict['attn']['action_inter'])
                self.action_union.update(loss_dict['attn']['action_union'])
            if loss_dict['attn']['bg_inter'] is not None:
                self.bg_inter.update(loss_dict['attn']['bg_inter'])
                self.bg_union.update(loss_dict['attn']['bg_union'])

    def train(self, dataloader_train):
        self.reset_log()
        self.model.train()

        current_lr = self.optimizer.param_groups[0]['lr']
        self.accelerator.print(f"Current epoch: {self.cur_epoch}, lr: {current_lr}")
        logging.info(f"Current epoch: {self.cur_epoch}, lr: {current_lr}")

        num_batches = len(dataloader_train)
        start_time = time.time()
        step_start_time = time.time()
        prev_loss = 0.0
        for step_idx, data in enumerate(dataloader_train):
            self.step(data, 'train')

            # Cứ mỗi args.print_every step hoặc batch cuối cùng thì cập nhật tiến độ
            if (step_idx + 1) % self.args.print_every == 0 or (step_idx + 1) == num_batches:
                curr_steps = step_idx + 1
                elapsed = time.time() - start_time
                step_elapsed = time.time() - step_start_time
                
                # Tốc độ và ETA
                it_per_sec = (self.args.print_every if curr_steps < num_batches else (curr_steps % self.args.print_every or self.args.print_every)) / max(step_elapsed, 1e-5)

                progress_pct = (curr_steps / num_batches) * 100
                # Cập nhật real-time lên WandB theo từng step interval
                self.accelerator.log({
                    "train_step/progress_pct": progress_pct,
                    "train_step/speed_it_s": it_per_sec,
                })

                step_start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        map_pred_actor_list = np.concatenate(self.map_pred_actor_list, axis=0)
        label_actor_list = np.concatenate(self.label_actor_list, axis=0)

        train_mAP = average_precision_score(label_actor_list, map_pred_actor_list.astype(np.float32))

        loss_epoch = self.loss_epoch / num_batches
        actor_loss_epoch = self.actor_loss_epoch / num_batches
        ego_loss_epoch = self.ego_loss_epoch / num_batches
        ego_acc = self.correct_ego / max(self.total_ego, 1)

        self.accelerator.print(f'acc of the ego: {ego_acc}')
        logging.info(f'acc of the ego: {ego_acc}')
        self.accelerator.print('total loss')
        logging.info('total loss')
        self.accelerator.print(loss_epoch)
        logging.info(loss_epoch)

        if self.args.action_attn_weight > 0 or self.args.bg_attn_weight > 0:
            attn_loss_epoch = self.attn_loss_epoch / num_batches
            self.accelerator.print('attn loss:')
            logging.info('attn loss:')
            self.accelerator.print(attn_loss_epoch)
            logging.info(attn_loss_epoch)

            if self.args.action_attn_weight > 0:
                action_attn_loss_epoch = self.action_attn_loss_epoch / num_batches
                self.accelerator.print('action_attn_loss')
                logging.info('action_attn_loss')
                self.accelerator.print(action_attn_loss_epoch)
                logging.info(action_attn_loss_epoch)
            if self.args.bg_attn_weight > 0:
                bg_attn_loss_epoch = self.bg_attn_loss_epoch / num_batches
                self.accelerator.print('bg_attn_loss_epoch')
                logging.info('bg_attn_loss_epoch')
                self.accelerator.print(bg_attn_loss_epoch)
                logging.info(bg_attn_loss_epoch)

        self.accelerator.print('-' * 20)
        logging.info('-' * 20)
        self.accelerator.print('actor loss:')
        logging.info('actor loss:')
        self.accelerator.print(actor_loss_epoch)
        logging.info(actor_loss_epoch)
        self.accelerator.print('ego loss:')
        logging.info('ego loss:')
        self.accelerator.print(ego_loss_epoch)
        logging.info(ego_loss_epoch)
        self.accelerator.print(f'(train) mAP of the actor: {train_mAP}')
        logging.info(f'(train) mAP of the actor: {train_mAP}')

        self.accelerator.log({
            "train/total_loss": loss_epoch,
            "train/actor_loss": actor_loss_epoch,
            "train/ego_loss": ego_loss_epoch,
            "train/ego_acc": ego_acc,
            "train/mAP": train_mAP,
            "train/lr": current_lr,
            "epoch": self.cur_epoch
        }, step=self.cur_epoch)

        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self, dataloader_val, abs_logdir):
        self.model.eval()
        save_cp = False
        self.reset_log()

        num_batches = len(dataloader_val)
        with torch.no_grad():
            for step_idx, data in enumerate(dataloader_val):
                self.step(data, 'val')

            if self.args.action_attn_weight > 0. or self.args.bg_attn_weight > 0.:
                attn_loss_epoch = self.attn_loss_epoch / num_batches
                self.accelerator.print('attn loss:')
                logging.info('attn loss:')
                self.accelerator.print(attn_loss_epoch)
                logging.info(attn_loss_epoch)

                if self.args.action_attn_weight > 0.:
                    action_attn_loss_epoch = self.action_attn_loss_epoch / num_batches
                    self.accelerator.print('action_attn_loss')
                    logging.info('action_attn_loss')
                    self.accelerator.print(action_attn_loss_epoch)
                    logging.info(action_attn_loss_epoch)
                if self.args.bg_attn_weight > 0.:
                    bg_attn_loss_epoch = self.bg_attn_loss_epoch / num_batches
                    self.accelerator.print('bg_attn_loss_epoch')
                    logging.info('bg_attn_loss_epoch')
                    self.accelerator.print(bg_attn_loss_epoch)
                    logging.info(bg_attn_loss_epoch)

            if self.args.action_attn_weight > 0 and self.args.bg_attn_weight > 0:
                if self.bg_inter.sum and self.bg_union.sum:
                    iou = self.bg_inter.sum / (self.bg_union.sum + 1e-10)
                    for i, val in enumerate(iou):
                        self.accelerator.print('BG IoU {0}: {1:.2f}'.format(i, val * 100))
                        logging.info('BG IoU {0}: {1:.2f}'.format(i, val * 100))

            map_pred_actor_list = np.concatenate(self.map_pred_actor_list, axis=0)
            label_actor_list = np.concatenate(self.label_actor_list, axis=0)

            mAP = average_precision_score(
                label_actor_list,
                map_pred_actor_list.astype(np.float32),
            )
            c_mAP = average_precision_score(
                label_actor_list[:, :12],
                map_pred_actor_list[:, :12].astype(np.float32)
            )
            b_mAP = average_precision_score(
                label_actor_list[:, 24:36],
                map_pred_actor_list[:, 24:36].astype(np.float32)
            )
            p_mAP = average_precision_score(
                label_actor_list[:, 48:56],
                map_pred_actor_list[:, 48:56].astype(np.float32),
            )
            group_c_mAP = average_precision_score(
                label_actor_list[:, 12:24],
                map_pred_actor_list[:, 12:24].astype(np.float32)
            )
            group_b_mAP = average_precision_score(
                label_actor_list[:, 36:48],
                map_pred_actor_list[:, 36:48].astype(np.float32)
            )
            group_p_mAP = average_precision_score(
                label_actor_list[:, 56:64],
                map_pred_actor_list[:, 56:64].astype(np.float32),
            )
            mAP_per_class = average_precision_score(
                label_actor_list,
                map_pred_actor_list.astype(np.float32), 
                average=None
            )

            self.accelerator.print(f'(val) mAP: {mAP}')
            self.accelerator.print(f'(val) mAP of the c: {c_mAP}')
            self.accelerator.print(f'(val) mAP of the b: {b_mAP}')
            self.accelerator.print(f'(val) mAP of the p: {p_mAP}')
            self.accelerator.print(f'(val) mAP of the c+: {group_c_mAP}')
            self.accelerator.print(f'(val) mAP of the b+: {group_b_mAP}')
            self.accelerator.print(f'(val) mAP of the p+: {group_p_mAP}')

            logging.info(f'(val) mAP: {mAP}')
            logging.info(f'(val) mAP of the c: {c_mAP}')
            logging.info(f'(val) mAP of the b: {b_mAP}')
            logging.info(f'(val) mAP of the p: {p_mAP}')
            logging.info(f'(val) mAP of the c+: {group_c_mAP}')
            logging.info(f'(val) mAP of the b+: {group_b_mAP}')
            logging.info(f'(val) mAP of the p+: {group_p_mAP}')

            ego_acc = self.correct_ego / max(self.total_ego, 1)
            self.accelerator.print(f'acc of the ego: {ego_acc}')
            logging.info(f'acc of the ego: {ego_acc}')

            if mAP > self.best_mAP:
                self.best_mAP = mAP
                self.best_log = [
                    f'(val) mAP: {mAP}',
                    f'(val) mAP of the c: {c_mAP}',
                    f'(val) mAP of the b: {b_mAP}',
                    f'(val) mAP of the p: {p_mAP}',
                    f'(val) mAP of the c+: {group_c_mAP}',
                    f'(val) mAP of the b+: {group_b_mAP}',
                    f'(val) mAP of the p+: {group_p_mAP}'
                ]
                save_cp = True

            self.accelerator.print(f'best mAP : {self.best_mAP}')
            logging.info(f'best mAP : {self.best_mAP}')

            if self.accelerator.is_main_process:
                with open(os.path.join(abs_logdir, 'mAP.txt'), 'a') as f:
                    f.write('epoch: ' + str(self.cur_epoch) + '\n')
                    f.write('best mAP: %.4f\n' % self.best_mAP)
                    f.write('mAP: %.4f\n' % mAP)
                    f.write('mAP of c: %.4f\n' % c_mAP)
                    f.write('mAP of b: %.4f\n' % b_mAP)
                    f.write('mAP of p: %.4f\n' % p_mAP)
                    f.write('mAP of c+: %.4f\n' % group_c_mAP)
                    f.write('mAP of b+: %.4f\n' % group_b_mAP)
                    f.write('mAP of p+: %.4f\n' % group_p_mAP)

                    # 1. Nhóm c: 0 đến 12
                    f.write("c per class: \n")
                    for ap in mAP_per_class[:12].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")

                    # 2. Nhóm c+: 12 đến 24
                    f.write("c+ per class: \n")
                    for ap in mAP_per_class[12:24].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")

                    # 3. Nhóm b: 24 đến 36
                    f.write("b per class: \n")
                    for ap in mAP_per_class[24:36].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")

                    # 4. Nhóm b+: 36 đến 48
                    f.write("b+ per class: \n")
                    for ap in mAP_per_class[36:48].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")

                    # 5. Nhóm p: 48 đến 56
                    f.write("p per class: \n")
                    for ap in mAP_per_class[48:56].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")

                    # 6. Nhóm p+: 56 đến 64
                    f.write("p+ per class: \n")
                    for ap in mAP_per_class[56:64].tolist():
                        f.write("%.4f " % ap)
                    f.write("\n")
                    f.write('*' * 15 + '\n')

            logging.info('epoch: ' + str(self.cur_epoch))
            logging.info('best mAP: %.4f' % self.best_mAP)
            logging.info('mAP: %.4f' % mAP)
            logging.info('mAP of c: %.4f' % c_mAP)
            logging.info('mAP of b: %.4f' % b_mAP)
            logging.info('mAP of p: %.4f' % p_mAP)
            logging.info('mAP of c+: %.4f' % group_c_mAP)
            logging.info('mAP of b+: %.4f' % group_b_mAP)
            logging.info('mAP of p+: %.4f' % group_p_mAP)

            logging.info('c per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[:12].tolist()
            ))
            logging.info('c+ per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[12:24].tolist()
            ))
            logging.info('b per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[24:36].tolist()
            ))
            logging.info('b+ per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[36:48].tolist()
            ))
            logging.info('p per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[48:56].tolist()
            ))
            logging.info('p+ per class: ' + ' '.join(
                "%.4f" % ap for ap in mAP_per_class[56:64].tolist()
            ))
            logging.info('*' * 15)

            total_loss = self.loss_epoch / float(num_batches)
            self.accelerator.print(f'Epoch {self.cur_epoch:03d} Loss: {total_loss:3.3f}')

            # WandB logging
            self.accelerator.log({
                "val/total_loss": total_loss,
                "val/ego_acc": ego_acc,
                "val/mAP": mAP,
                "val/c_mAP": c_mAP,
                "val/b_mAP": b_mAP,
                "val/p_mAP": p_mAP,
                "val/group_c_mAP": group_c_mAP,
                "val/group_b_mAP": group_b_mAP,
                "val/group_p_mAP": group_p_mAP,
                "val/best_mAP": self.best_mAP,
                "epoch": self.cur_epoch
            }, step=self.cur_epoch)

            self.val_loss.append(total_loss)
            return save_cp, [mAP, total_loss]

    def save(self, is_best, abs_logdir):
        if is_best and self.accelerator.is_main_process:
            self.bestval = self.val_loss[-1]
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = os.path.join(abs_logdir, 'best_model.pth')
            self.accelerator.save(unwrapped_model.state_dict(), save_path)
            self.accelerator.print(f"====== Overwrote best model at {save_path} ======>")


if __name__ == '__main__':
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info('\n' * 5)

    args, logdir = parser()
    print(args)
    logging.info(args)

    # Initialize Accelerator with wandb logging backend
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb",
    )

    logdir = logdir.replace(':', '_').replace('\n', '_').replace(" ", "")
    abs_logdir = os.path.abspath(logdir)
    if os.name == 'nt' and not abs_logdir.startswith('\\\\?\\'):
        abs_logdir = f'\\\\?\\{abs_logdir}'

    num_ego_class = 4
    num_actor_class = 64
    lmdb_train_path = "/kaggle/working/taco_train.lmdb"
    lmdb_val_path = "/kaggle/working/taco_val.lmdb"

    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        lmdb_train_path, lmdb_val_path = put_into_working()

        accelerator.init_trackers(
            project_name=getattr(args, 'wandb_project', 'action-slot-taco'),
            config=vars(args),
            init_kwargs={"wandb": {"name": getattr(args, 'run_name', None), "dir": "./"}}
        )

    accelerator.print('initialize train set')
    logging.info('initialize train set')
    train_set = TACO(args, lmdb_train_path, lmdb_val_path, split='train')

    accelerator.print('initialize val set')
    logging.info('initialize val set')
    val_set = TACO(args, lmdb_train_path, lmdb_val_path, split='val')

    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    model = generate_model(args, num_ego_class, num_actor_class)

    if args.model_name == 'mvit':
        params = set_lr(model)
    else:
        params = [{'params': model.parameters()}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr) if args.scheduler else None

    # Prepare model, optimizer, dataloaders, and scheduler with Accelerate
    if scheduler is not None:
        model, optimizer, dataloader_train, dataloader_val, scheduler = accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val, scheduler
        )
    else:
        model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val
        )

    trainer = Engine(args, model, optimizer, num_actor_class, accelerator, scheduler)

    accelerator.print(f'Checkpoint path: {logdir}')

    result_list = []
    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train(dataloader_train)
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            is_best, res = trainer.validate(dataloader_val, abs_logdir)
            trainer.save(is_best, abs_logdir)
            result_list.append(res)

    accelerator.print('********** Best model **********')
    for s in trainer.best_log:
        accelerator.print(s)

    accelerator.end_training()

    if accelerator.is_main_process and len(result_list) > 0:
        plot_result(np.array(result_list), args)