import argparse
import json
import math
import os
import sys

sys.path.append("../")

from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

torch.backends.cudnn.benchmark = True

from datasets.taco import TACO
from get_parser import parser
from loss import ActionSlotLoss
from model import generate_model
from utils import AverageMeter

def plot_result(result, args):
    """
    result : mAP, loss
    """
    text = ["mAP", "loss"]
    x = [i + 1 for i in range(0, args.epochs, args.val_every)]
    x = x if x[-1] == args.epochs else x + [args.epochs]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    for i in range(2):
        ax[i].plot(x, result[:, i])
        ax[i].title.set_text(text[i])
    plt.show()


def lambda_lr(epoch):
    if epoch < 11:
        lr = math.pow(1.1, epoch)
    else:
        lr = math.pow(0.7, int(epoch / 3))
    return lr


def set_lr(model):
    # Trích xuất model gốc trong trường hợp bọc bởi DDP/Accelerate
    unwrapped_model = (
        model.module if hasattr(model, "module") else model
    )
    params = list(
        filter(lambda kv: kv[0].startswith("head"), unwrapped_model.named_parameters())
    )
    base_params = list(
        filter(lambda kv: not kv[0].startswith("head"), unwrapped_model.named_parameters())
    )
    return [
        {"params": [temp[1] for temp in base_params]},
        {"params": [temp[1] for temp in params], "lr": 2e-2},
    ]


class Engine(object):

    def __init__(
        self,
        args,
        model,
        optimizer,
        num_actor_class,
        accelerator: Accelerator,
        scheduler=None,
        logdir="runs",
    ):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_actor_class = num_actor_class
        self.num_groups = getattr(args, "num_groups", 4)
        self.logdir = logdir

        unwrapped = self.accelerator.unwrap_model(self.model)
        attention_res = (
            (
                unwrapped.resolution[0] * args.bg_upsample,
                unwrapped.resolution[1] * args.bg_upsample,
            )
            if hasattr(unwrapped, "resolution")
            else None
        )
        self.criterion = ActionSlotLoss(
            args, num_actor_class, attention_res
        ).to(self.accelerator.device)

        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.best_mAP = 1e-5
        self.best_log = []
        self.reset_log()

    def reset_log(self):
        self.loss_epoch = 0.0
        self.ego_loss_epoch = 0.0
        self.actor_loss_epoch = 0.0
        self.grpo_loss_epoch = 0.0
        self.attn_loss_epoch = 0.0
        self.correct_ego = 0
        self.total_ego = 0
        self.label_actor_list = []
        self.map_pred_actor_list = []
        self.action_inter = AverageMeter()
        self.action_union = AverageMeter()
        self.bg_inter = AverageMeter()
        self.bg_union = AverageMeter()

    def step(self, batch, mode):
        video_in = batch["videos"]
        seq_len = self.args.seq_len
        inputs = [video_in[i].to(dtype=torch.float32) for i in range(seq_len)]

        # =========================================================================
        # HUẤN LUYỆN (TRAIN MODE) VỚI GRPO (G NHÓM TRAJECTORY)
        # =========================================================================
        if mode == "train":
            # 1. Rollout Policy cũ (pi_old) để lấy mẫu G nhóm S
            with torch.no_grad():
                pred_ego_old, pred_actor_old, _, old_log_prob = self.model(
                    inputs, num_groups=self.num_groups
                )

            # 2. Forward Policy hiện tại (pi_theta)
            pred_ego, pred_actor, attn, log_prob = self.model(
                inputs, num_groups=self.num_groups
            )

            # 3. Tính toán Loss (Supervised + GRPO)
            pred_dict = {
                "ego": pred_ego,
                "actor": pred_actor,
                "attn": attn,
                "log_prob": log_prob,
                "old_log_prob": old_log_prob,
            }
            loss_dict = self.criterion(pred_dict, batch, validate=False)

            ego_loss = (
                loss_dict["ego"]
                if loss_dict["ego"] is not None
                else torch.tensor(0.0, device=self.accelerator.device)
            )
            actor_loss = loss_dict["actor"]
            grpo_loss = loss_dict["grpo"]
            attn_loss = loss_dict["attn"]["attn_loss"]

            total_loss = (
                actor_loss
                + self.args.ego_loss_weight * ego_loss
                + grpo_loss
                + attn_loss
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Gather metrics qua các device
            total_loss_gathered = self.accelerator.gather(total_loss).mean().item()
            actor_loss_gathered = self.accelerator.gather(actor_loss).mean().item()
            grpo_loss_gathered = self.accelerator.gather(grpo_loss).mean().item()
            ego_loss_gathered = self.accelerator.gather(ego_loss).mean().item()

            self.loss_epoch += float(total_loss_gathered)
            self.actor_loss_epoch += float(actor_loss_gathered)
            self.grpo_loss_epoch += float(grpo_loss_gathered)
            self.ego_loss_epoch += float(ego_loss_gathered)

            # Lấy trung bình dự đoán qua G nhóm để đo độ chính xác lúc train
            pred_actor_mean = torch.sigmoid(pred_actor.mean(dim=1))
            gathered_actor_preds = self.accelerator.gather_for_metrics(pred_actor_mean)
            gathered_actor_labels = self.accelerator.gather_for_metrics(batch["actor"])

            self.map_pred_actor_list.append(gathered_actor_preds.detach().cpu().numpy())
            self.label_actor_list.append(gathered_actor_labels.detach().cpu().numpy())

            if pred_ego is not None:
                _, pred_ego_idx = torch.max(pred_ego.data, 1)
                correct = (pred_ego_idx == batch["ego"]).sum()
                total = torch.tensor(batch["ego"].size(0), device=self.accelerator.device)
                self.correct_ego += self.accelerator.gather(correct).sum().item()
                self.total_ego += self.accelerator.gather(total).sum().item()

        # =========================================================================
        # ĐÁNH GIÁ (VAL / TEST MODE) - CHẠY DETERMINISTIC VỚI G = 1 (STREAMING EDGE)
        # =========================================================================
        else:
            with torch.no_grad():
                pred_ego, pred_actor, attn, _ = self.model(inputs, num_groups=1)
                if pred_actor.dim() == 3 and pred_actor.shape[1] == 1:
                    pred_actor = pred_actor.squeeze(1)

                pred_dict = {
                    "ego": pred_ego,
                    "actor": pred_actor,
                    "attn": attn,
                    "log_prob": None,
                    "old_log_prob": None,
                }
                loss_dict = self.criterion(pred_dict, batch, validate=True)

                actor_loss = loss_dict["actor"]
                ego_loss = (
                    loss_dict["ego"]
                    if loss_dict["ego"] is not None
                    else torch.tensor(0.0, device=self.accelerator.device)
                )
                total_loss = actor_loss + self.args.ego_loss_weight * ego_loss
                total_loss_gathered = self.accelerator.gather(total_loss).mean().item()
                self.loss_epoch += float(total_loss_gathered)

                pred_actor_sig = torch.sigmoid(pred_actor)
                gathered_preds = self.accelerator.gather_for_metrics(pred_actor_sig)
                gathered_labels = self.accelerator.gather_for_metrics(batch["actor"])

                self.map_pred_actor_list.append(gathered_preds.detach().cpu().numpy())
                self.label_actor_list.append(gathered_labels.detach().cpu().numpy())

                if pred_ego is not None:
                    _, pred_ego_idx = torch.max(pred_ego.data, 1)
                    correct = (pred_ego_idx == batch["ego"]).sum()
                    total = torch.tensor(batch["ego"].size(0), device=self.accelerator.device)
                    self.correct_ego += self.accelerator.gather(correct).sum().item()
                    self.total_ego += self.accelerator.gather(total).sum().item()

    def train(self, dataloader_train):
        self.reset_log()
        self.model.train()
        self.num_batches = len(dataloader_train)

        # Nếu có wandb thì disable tqdm
        disable_pbar = self.args.wandb or (not self.accelerator.is_local_main_process)

        pbar = tqdm(
            dataloader_train,
            desc=f"Train Epoch {self.cur_epoch}",
            disable=disable_pbar,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=False,
        )

        for step_idx, data in enumerate(pbar):
            self.step(data, "train")

        loss_epoch = self.loss_epoch / self.num_batches
        actor_loss_epoch = self.actor_loss_epoch / self.num_batches
        grpo_loss_epoch = self.grpo_loss_epoch / self.num_batches

        # Log metrics lên WandB / Tracker
        self.accelerator.log(
            {
                "train/total_loss": loss_epoch,
                "train/actor_loss": actor_loss_epoch,
                "train/grpo_loss": grpo_loss_epoch,
                "train/ego_loss": self.ego_loss_epoch / self.num_batches,
                "epoch": self.cur_epoch,
            },
            step=self.cur_epoch,
        )

        self.accelerator.print(f"\n[Epoch {self.cur_epoch}] Total Loss: {loss_epoch:.4f}")
        self.accelerator.print(
            f"Actor Loss: {actor_loss_epoch:.4f} | GRPO Loss: {grpo_loss_epoch:.4f}"
        )
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self, dataloader):
        self.model.eval()
        self.reset_log()
        self.num_batches = len(dataloader)
        save_cp = False

        disable_pbar = self.args.wandb or (not self.accelerator.is_local_main_process)

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc="Validating (G=1)",
                disable=disable_pbar,
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.5,
                leave=False,
            )
            for data in pbar:
                self.step(data, "val")

            total_loss = self.loss_epoch / float(self.num_batches)

            if self.accelerator.is_main_process:
                map_pred_actor_list = np.concatenate(self.map_pred_actor_list, axis=0)
                label_actor_list = np.concatenate(self.label_actor_list, axis=0)

                mAP = average_precision_score(
                    label_actor_list, map_pred_actor_list.astype(np.float32)
                )
                c_mAP = average_precision_score(
                    label_actor_list[:, :12],
                    map_pred_actor_list[:, :12].astype(np.float32),
                )
                b_mAP = average_precision_score(
                    label_actor_list[:, 24:36],
                    map_pred_actor_list[:, 24:36].astype(np.float32),
                )
                p_mAP = average_precision_score(
                    label_actor_list[:, 48:56],
                    map_pred_actor_list[:, 48:56].astype(np.float32),
                )
                group_c_mAP = average_precision_score(
                    label_actor_list[:, 12:24],
                    map_pred_actor_list[:, 12:24].astype(np.float32),
                )
                group_b_mAP = average_precision_score(
                    label_actor_list[:, 36:48],
                    map_pred_actor_list[:, 36:48].astype(np.float32),
                )
                group_p_mAP = average_precision_score(
                    label_actor_list[:, 56:64],
                    map_pred_actor_list[:, 56:64].astype(np.float32),
                )

                ego_acc = self.correct_ego / max(self.total_ego, 1)

                # Log toàn bộ metrics validation lên WandB
                self.accelerator.log(
                    {
                        "val/loss": total_loss,
                        "val/mAP": mAP,
                        "val/c_mAP": c_mAP,
                        "val/b_mAP": b_mAP,
                        "val/p_mAP": p_mAP,
                        "val/group_c_mAP": group_c_mAP,
                        "val/group_b_mAP": group_b_mAP,
                        "val/group_p_mAP": group_p_mAP,
                        "val/ego_acc": ego_acc,
                        "epoch": self.cur_epoch,
                    },
                    step=self.cur_epoch,
                )

                if mAP > self.best_mAP:
                    self.best_mAP = mAP
                    self.best_log = [...]
                    save_cp = True

                self.val_loss.append(total_loss)
                return save_cp, [mAP, total_loss]
            else:
                return False, [0.0, total_loss]

    def save(self, is_best):
        if is_best and self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = os.path.join(self.logdir, "best_model.pth")
            self.accelerator.save(unwrapped_model.state_dict(), save_path)
            tqdm.write("====== Overwrote best model ======>")


if __name__ == "__main__":
    args, logdir = parser()
    print(args)
    logdir = logdir.replace(":", "_").replace("\n", "_").replace(" ", "")

    abs_logdir = os.path.abspath(logdir)
    if os.name == "nt" and not abs_logdir.startswith("\\\\?\\"):
        abs_logdir = f"\\\\?\\{abs_logdir}"

    os.makedirs(abs_logdir, exist_ok=True)

    # Khởi tạo Accelerator với logging TensorBoard
    log_tracker = "wandb" if args.wandb else "tensorboard"

    accelerator = Accelerator(
        log_with=log_tracker,
        project_dir=abs_logdir
    )

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb:
            init_kwargs = {
                "wandb": {
                    "name": os.path.basename(abs_logdir),  # Tên run hiển thị trên W&B
                    "config": vars(args)                   # Lưu toàn bộ hyperparameter
                }
            }
        accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT", "taco_experiments"), 
            init_kwargs=init_kwargs
        )

    args.device = accelerator.device
    seq_len = args.seq_len

    num_ego_class = 4
    num_actor_class = 64

    accelerator.print("initialize train set")
    train_set = TACO(args=args, split="train", accelerator= accelerator)
    accelerator.print("initialize val set")
    val_set = TACO(args=args, split="val", accelerator= accelerator)

    dataloader_train = DataLoader(
        train_set,
        batch_size=6,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = generate_model(args, num_ego_class, num_actor_class)

    if "mvit" == args.model_name:
        params = set_lr(model)
    else:
        params = [{"params": model.parameters()}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    else:
        scheduler = None

    # Chuẩn bị model, optimizer, dataloaders qua accelerator
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val
    )
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    trainer = Engine(
        args,
        model,
        optimizer,
        num_actor_class,
        accelerator=accelerator,
        scheduler=scheduler,
        logdir=abs_logdir,
    )

    accelerator.print(f"Checkpoint path: {abs_logdir}")

    result_list = []
    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train(dataloader_train)
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            is_best, res = trainer.validate(dataloader_val)
            trainer.save(is_best)
            if accelerator.is_main_process:
                result_list.append(res)

    if accelerator.is_main_process:
        print("********** Best model **********")
        for s in trainer.best_log:
            print(s)
        plot_result(np.array(result_list), args)
        accelerator.end_training()