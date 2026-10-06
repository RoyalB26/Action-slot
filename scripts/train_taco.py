import sys
sys.path.append('../')

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import math
import logging
import argparse
import json
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
# --- Hugging Face Accelerate & WandB ---
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb

# --- Rich formatting (tùy chọn hiển thị console đẹp) ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None

from get_parser import parser
from datasets.taco import TACO
from model import generate_model
from loss import ActionSlotLoss
from utils import AverageMeter
from put_lmdb import *

def display_metrics_table(epoch, metrics_dict, title="Epoch Summary"):
    """In bảng kết quả chuyên nghiệp ra console."""
    if console is not None:
        table = Table(title=f"[bold cyan]{title} - Epoch {epoch}[/bold cyan]", show_header=True, header_style="bold magenta")
        table.add_column("Chỉ số (Metric)", style="dim", width=26)
        table.add_column("Giá trị", justify="right", style="green")

        for k, v in metrics_dict.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        console.print(table)
    else:
        print(f"\n===== {title} - Epoch {epoch} =====")
        for k, v in metrics_dict.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<24}: {val_str}")
        print("=" * 35)


def plot_result(result, args):
    """result: mAP, loss"""
    text = ["mAP", "loss"]
    x = [i + 1 for i in range(0, args.epochs, args.val_every)]
    x = x if (len(x) > 0 and x[-1] == args.epochs) else x + [args.epochs]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    for i in range(2):
        ax[i].plot(x, result[:, i])
        ax[i].title.set_text(text[i])
        ax[i].grid(True)
    plt.tight_layout()
    plt.show()


def lambda_lr(epoch):
    if epoch < 11:
        return math.pow(1.1, epoch)
    else:
        return math.pow(0.7, int(epoch / 3))


def set_lr(model):
    unwrapped = model.module if hasattr(model, 'module') else model
    params = list(filter(lambda kv: kv[0].startswith("head"), unwrapped.named_parameters()))
    base_params = list(filter(lambda kv: not kv[0].startswith("head"), unwrapped.named_parameters()))
    return [
        {'params': [temp[1] for temp in base_params]},
        {'params': [temp[1] for temp in params], 'lr': 2e-2}
    ]


class Engine(object):
    def __init__(self, args, accelerator, model, optimizer, num_actor_class, criterion, scheduler=None, writer=None):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.writer = writer
        self.num_actor_class = num_actor_class

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
        self.tracking_stats = {
            'center_bias_ratio': 0.0,
            'early_frame_energy': 0.0,
            'avg_peak_frame': 0.0,
        }
        self.label_actor_list = []
        self.map_pred_actor_list = []
        self.action_inter = AverageMeter()
        self.action_union = AverageMeter()
        self.bg_inter = AverageMeter()
        self.bg_union = AverageMeter()

    def step(self, batch, mode):
        video_in = batch['videos']
        inputs = [v.to(dtype=torch.float32) for v in video_in]

        attn = None
        if self.args.box:
            box_in = batch['box']
            boxes = torch.from_numpy(box_in).to(self.accelerator.device, dtype=torch.float32) if isinstance(box_in, np.ndarray) else box_in.to(dtype=torch.float32)
            pred_ego, pred_actor = self.model(inputs, boxes)
            tracking_stats = {k: 0.0 for k in self.tracking_stats}
        else:
            if 'slot' in self.args.model_name or 'mvit' in self.args.model_name:
                pred_ego, pred_actor, attn, tracking_stats = self.model(inputs)
            else:
                pred_ego, pred_actor, tracking_stats = self.model(inputs)

        loss_dict = self.criterion(
            {'ego': pred_ego, 'actor': pred_actor, 'attn': attn},
            batch,
            False if mode == 'train' else True
        )

        ego_loss = loss_dict['ego']
        if ego_loss is None:
            ego_loss = torch.tensor(0.0, device=self.accelerator.device)
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

        # Prediction evaluation
        _, pred_ego_cls = torch.max(pred_ego.data, 1)
        ego = batch['ego']
        actor = batch['actor']

        self.total_ego += ego.size(0)
        self.correct_ego += (pred_ego_cls == ego).sum().item()

        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            pred_actor_sm = torch.nn.functional.softmax(pred_actor, dim=-1)
            _, pred_actor_idx = torch.max(pred_actor_sm.data, -1)
            pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
            pred_actor_np = pred_actor_sm.detach().cpu().numpy()

            map_batch_new_pred_actor = []
            for i, b in enumerate(pred_actor_idx):
                map_new_pred = np.zeros(self.num_actor_class, dtype=np.float32) + 1e-5
                for j, pred in enumerate(b):
                    if pred != self.num_actor_class:
                        if pred_actor_np[i, j, pred] > map_new_pred[pred]:
                            map_new_pred[pred] = pred_actor_np[i, j, pred]
                map_batch_new_pred_actor.append(map_new_pred)

            self.map_pred_actor_list.append(np.array(map_batch_new_pred_actor))
            self.label_actor_list.append(batch['slot_eval_gt'].detach().cpu().numpy())
        else:
            pred_actor_sig = torch.sigmoid(pred_actor)
            self.map_pred_actor_list.append(pred_actor_sig.detach().cpu().numpy())
            self.label_actor_list.append(actor.detach().cpu().numpy())

        self.actor_loss_epoch += float(actor_loss.mean().item())
        self.ego_loss_epoch += float(ego_loss.mean().item())
        for k in self.tracking_stats:
            if k in tracking_stats:
                self.tracking_stats[k] += float(tracking_stats[k])

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

    def train_epoch(self, dataloader_train):
        self.reset_log()
        self.model.train()
        num_batches = len(dataloader_train)

        lr_current = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]['lr']

        progress_bar = tqdm(
            dataloader_train,
            desc=f"Epoch {self.cur_epoch:03d} [Train]",
            disable=not self.accelerator.is_local_main_process,
            leave=False
        )
        for data in progress_bar:
            self.step(data, 'train')
            progress_bar.set_postfix({"loss": f"{self.loss_epoch / max(1, progress_bar.n):.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        # Metrics aggregation
        map_pred = np.concatenate(self.map_pred_actor_list, axis=0)
        label_pred = np.concatenate(self.label_actor_list, axis=0)
        map_pred = map_pred.reshape(-1, self.num_actor_class)
        label_pred = label_pred.reshape(-1, self.num_actor_class)

        mAP = average_precision_score(label_pred, map_pred.astype(np.float32))
        loss_epoch = self.loss_epoch / num_batches
        actor_loss = self.actor_loss_epoch / num_batches
        ego_loss = self.ego_loss_epoch / num_batches
        ego_acc = self.correct_ego / max(1, self.total_ego)
        tracking_avg = {k: self.tracking_stats[k] / num_batches for k in self.tracking_stats}

        self.train_loss.append(loss_epoch)

        # Logging
        train_metrics = {
            "train/loss": loss_epoch,
            "train/actor_loss": actor_loss,
            "train/ego_loss": ego_loss,
            "train/ego_acc": ego_acc,
            "train/mAP": float(mAP),
            "train/lr": lr_current,
            **{f"tracking/{k}": v for k, v in tracking_avg.items()}
        }

        if self.accelerator.is_main_process:
            display_metrics_table(self.cur_epoch, {
                "Train Loss": loss_epoch,
                "Actor Loss": actor_loss,
                "Ego Loss": ego_loss,
                "Ego Accuracy": f"{ego_acc * 100:.2f}%",
                "Actor mAP": f"{mAP:.4f}",
                "Learning Rate": f"{lr_current:.2e}"
            }, title="Training Summary")

            if wandb.run is not None:
                wandb.log(train_metrics, step=self.cur_epoch)

        self.cur_epoch += 1

    def validate(self, dataloader_val, logdir):
        self.model.eval()
        self.reset_log()
        num_batches = len(dataloader_val)

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader_val,
                desc=f"Epoch {self.cur_epoch - 1:03d} [Val]",
                disable=not self.accelerator.is_local_main_process,
                leave=False
            )
            for data in progress_bar:
                self.step(data, 'val')

            map_pred = np.concatenate(self.map_pred_actor_list, axis=0).reshape(-1, self.num_actor_class)
            label_pred = np.concatenate(self.label_actor_list, axis=0).reshape(-1, self.num_actor_class)

            mAP = average_precision_score(label_pred, map_pred.astype(np.float32))
            c_mAP = average_precision_score(label_pred[:, :12], map_pred[:, :12].astype(np.float32))
            group_c_mAP = average_precision_score(label_pred[:, 12:24], map_pred[:, 12:24].astype(np.float32))
            b_mAP = average_precision_score(label_pred[:, 24:36], map_pred[:, 24:36].astype(np.float32))
            group_b_mAP = average_precision_score(label_pred[:, 36:48], map_pred[:, 36:48].astype(np.float32))
            p_mAP = average_precision_score(label_pred[:, 48:56], map_pred[:, 48:56].astype(np.float32))
            group_p_mAP = average_precision_score(label_pred[:, 56:64], map_pred[:, 56:64].astype(np.float32))

            total_loss = self.loss_epoch / float(max(1, num_batches))
            ego_acc = self.correct_ego / max(1, self.total_ego)
            self.val_loss.append(total_loss)

            save_cp = False
            if mAP > self.best_mAP:
                self.best_mAP = mAP
                save_cp = True

            val_metrics = {
                "val/loss": total_loss,
                "val/mAP": float(mAP),
                "val/c_mAP": float(c_mAP),
                "val/group_c_mAP": float(group_c_mAP),
                "val/b_mAP": float(b_mAP),
                "val/group_b_mAP": float(group_b_mAP),
                "val/p_mAP": float(p_mAP),
                "val/group_p_mAP": float(group_p_mAP),
                "val/ego_acc": float(ego_acc),
                "val/best_mAP": float(self.best_mAP)
            }

            if self.accelerator.is_main_process:
                display_metrics_table(self.cur_epoch - 1, {
                    "Val Loss": total_loss,
                    "Total mAP": float(mAP),
                    "c_mAP (Cars)": float(c_mAP),
                    "c+_mAP": float(group_c_mAP),
                    "b_mAP (Bikes)": float(b_mAP),
                    "b+_mAP": float(group_b_mAP),
                    "p_mAP (Peds)": float(p_mAP),
                    "p+_mAP": float(group_p_mAP),
                    "Best mAP": float(self.best_mAP)
                }, title="Validation Evaluation")

                if wandb.run is not None:
                    wandb.log(val_metrics, step=self.cur_epoch - 1)

                if self.writer:
                    self.writer.add_scalar('val/mAP', mAP, self.cur_epoch - 1)
                    self.writer.add_scalar('val/loss', total_loss, self.cur_epoch - 1)

                # Lưu log ra file text
                with open(os.path.join(logdir, 'mAP.txt'), 'a') as f:
                    f.write(f"epoch: {self.cur_epoch - 1}\nbest mAP: {self.best_mAP:.4f}\nmAP: {mAP:.4f}\n")
                    f.write(f"c_mAP: {c_mAP:.4f} | b_mAP: {b_mAP:.4f} | p_mAP: {p_mAP:.4f}\n{'*'*25}\n")

            return save_cp, [mAP, total_loss]

    def save_checkpoint(self, logdir, is_best):
        if is_best and self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = os.path.join(logdir, 'best_model.pth')
            torch.save(unwrapped_model.state_dict(), save_path)
            if console:
                console.print(f"[bold green]✔ Saved Best Model to {save_path}[/bold green]")
            else:
                print(f"[BEST] Overwrote best model at {save_path}")


if __name__ == '__main__':
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    args, logdir = parser()
    seq_len = args.seq_len
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")

    logdir = logdir.replace(':', '_').replace('\n', '_').replace(" ", "")
    abs_logdir = os.path.abspath(logdir)
    if os.name == 'nt' and not abs_logdir.startswith('\\\\?\\'):
        abs_logdir = f'\\\\?\\{abs_logdir}'
    logdir = abs_logdir
    lmdb_train_path = "/kaggle/working/taco_train.lmdb"
    lmdb_val_path = "/kaggle/working/taco_val.lmdb"
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        # Khởi tạo WandB
        accelerator.init_trackers(
            project_name=getattr(args, "wandb_project", "TACO-Action-Slot"),
            config=vars(args),
            init_kwargs={"wandb": {"name": os.path.basename(logdir)}}
        )
        writer = SummaryWriter(log_dir=logdir)
        lmdb_train_path, lmdb_val_path= put_into_working()
    else:
        writer = None
    accelerator.wait_for_everyone()
    num_ego_class = 4
    num_actor_class = 64

    # Dataloaders
    
    train_set = TACO(args,lmdb_train_path, lmdb_val_path, split='train')
    val_set = TACO(args,lmdb_train_path, lmdb_val_path, split='val')

    dataloader_train = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    dataloader_val = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # Model
    model = generate_model(args, num_ego_class, num_actor_class)

    # Optimizer & Scheduler
    params = set_lr(model) if 'mvit' == args.model_name else [{'params': model.parameters()}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr) if args.scheduler else None

    # Loss criterion
    attention_res = (model.resolution[0] * args.bg_upsample, model.resolution[1] * args.bg_upsample) if hasattr(model, 'resolution') else None
    criterion = ActionSlotLoss(args, num_actor_class, attention_res).to(accelerator.device)

    # Chuẩn bị model & components với Accelerate
    model, optimizer, dataloader_train, dataloader_val, scheduler = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val, scheduler
    )

    trainer = Engine(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        num_actor_class=num_actor_class,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer
    )

    result_list = []
    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train_epoch(dataloader_train)
        if (epoch % args.val_every == 0 or epoch == args.epochs - 1):
            is_best, res = trainer.validate(dataloader_val, logdir)
            trainer.save_checkpoint(logdir, is_best)
            result_list.append(res)

    if accelerator.is_main_process:
        accelerator.end_training()
        if len(result_list) > 0:
            plot_result(np.array(result_list), args)