import argparse
import json
import math
import os
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

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
    params = list(
        filter(lambda kv: kv[0].startswith("head"), model.named_parameters())
    )
    base_params = list(
        filter(
            lambda kv: not kv[0].startswith("head"), model.named_parameters()
        )
    )
    return [
        {"params": [temp[1] for temp in base_params]},
        {"params": [temp[1] for temp in params], "lr": 2e-2},
    ]


class Engine(object):

    def __init__(
        self, args, model, optimizer, num_actor_class, scheduler=None
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_actor_class = num_actor_class
        self.num_groups = getattr(args, "num_groups", 4)  # G nhóm S cho GRPO

        attention_res = (
            (
                self.model.resolution[0] * args.bg_upsample,
                self.model.resolution[1] * args.bg_upsample,
            )
            if hasattr(self.model, "resolution")
            else None
        )
        self.criterion = ActionSlotLoss(
            args, num_actor_class, attention_res
        ).to(self.args.device)

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
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.args.device)

        video_in = batch["videos"]
        seq_len = self.args.seq_len
        inputs = [
            video_in[i].to(self.args.device, dtype=torch.float32)
            for i in range(seq_len)
        ]

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
                else torch.tensor(0.0).cuda()
            )
            actor_loss = loss_dict["actor"]
            grpo_loss = loss_dict["grpo"]
            attn_loss = loss_dict["attn"]["attn_loss"]

            # Tổng hợp Loss
            total_loss = (
                actor_loss
                + self.args.ego_loss_weight * ego_loss
                + grpo_loss
                + attn_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Metrics logging
            self.loss_epoch += float(total_loss.item())
            self.actor_loss_epoch += float(actor_loss.item())
            self.grpo_loss_epoch += float(grpo_loss.item())
            self.ego_loss_epoch += float(ego_loss.item())

            # Lấy trung bình dự đoán qua G nhóm để đo độ chính xác lúc train
            pred_actor_mean = torch.sigmoid(pred_actor.mean(dim=1))
            self.map_pred_actor_list.append(
                pred_actor_mean.detach().cpu().numpy()
            )
            self.label_actor_list.append(batch["actor"].detach().cpu().numpy())

            if pred_ego is not None:
                _, pred_ego_idx = torch.max(pred_ego.data, 1)
                self.correct_ego += (pred_ego_idx == batch["ego"]).sum().item()
                self.total_ego += batch["ego"].size(0)

        # =========================================================================
        # ĐÁNH GIÁ (VAL / TEST MODE) - CHẠY DETERMINISTIC VỚI G = 1 (STREAMING EDGE)
        # =========================================================================
        else:
            with torch.no_grad():
                # G=1: chỉ giữ 1 chuỗi slot chuyển tiếp giữa các 4-frame chunks
                pred_ego, pred_actor, attn, _ = self.model(inputs, num_groups=1)
                # Squeeze chiều G = 1
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
                    else torch.tensor(0.0).cuda()
                )
                total_loss = actor_loss + self.args.ego_loss_weight * ego_loss
                self.loss_epoch += float(total_loss.item())

                pred_actor_sig = torch.sigmoid(pred_actor)
                self.map_pred_actor_list.append(
                    pred_actor_sig.detach().cpu().numpy()
                )
                self.label_actor_list.append(
                    batch["actor"].detach().cpu().numpy()
                )

                if pred_ego is not None:
                    _, pred_ego_idx = torch.max(pred_ego.data, 1)
                    self.correct_ego += (
                        (pred_ego_idx == batch["ego"]).sum().item()
                    )
                    self.total_ego += batch["ego"].size(0)

    def train(self):
        self.reset_log()
        self.model.train()
        self.num_batches = len(dataloader_train)

        for data in tqdm(
            dataloader_train, desc=f"Train Epoch {self.cur_epoch}"
        ):
            self.step(data, "train")

        loss_epoch = self.loss_epoch / self.num_batches
        actor_loss_epoch = self.actor_loss_epoch / self.num_batches
        grpo_loss_epoch = self.grpo_loss_epoch / self.num_batches

        print(f"\n[Epoch {self.cur_epoch}] Total Loss: {loss_epoch:.4f}")
        print(
            f"Actor Loss: {actor_loss_epoch:.4f} | GRPO Loss: {grpo_loss_epoch:.4f}"
        )
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self, dataloader):
        self.model.eval()
        self.reset_log()
        self.num_batches = len(dataloader)
        save_cp = False

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Validating (G=1 Streaming)"):
                self.step(data, "val")

            map_pred_actor_list = np.concatenate(
                self.map_pred_actor_list, axis=0
            )
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
            mAP_per_class = average_precision_score(
                label_actor_list,
                map_pred_actor_list.astype(np.float32),
                average=None,
            )

            print(f"(val) mAP: {mAP}")
            print(f"(val) mAP of the c: {c_mAP}")
            print(f"(val) mAP of the b: {b_mAP}")
            print(f"(val) mAP of the p: {p_mAP}")
            print(f"(val) mAP of the c+: {group_c_mAP}")
            print(f"(val) mAP of the b+: {group_b_mAP}")
            print(f"(val) mAP of the p+: {group_p_mAP}")

            print(f"acc of the ego: {self.correct_ego/self.total_ego}")
            writer.add_scalar(
                "ego", self.correct_ego / self.total_ego, self.cur_epoch
            )

            if mAP > self.best_mAP:
                self.best_mAP = mAP
                self.best_log = [
                    f"(val) mAP: {mAP}",
                    f"(val) mAP of the c: {c_mAP}",
                    f"(val) mAP of the b: {b_mAP}",
                    f"(val) mAP of the p: {p_mAP}",
                    f"(val) mAP of the c+: {group_c_mAP}",
                    f"(val) mAP of the b+: {group_b_mAP}",
                    f"(val) mAP of the p+: {group_p_mAP}",
                ]
                save_cp = True
            print(f"best mAP : {self.best_mAP}")

            with open(os.path.join(logdir, "mAP.txt"), "a") as f:
                f.write("epoch: " + str(self.cur_epoch) + "\n")
                f.write("best mAP: %.4f\n" % self.best_mAP)
                f.write("mAP: %.4f\n" % mAP)
                f.write("mAP of c: %.4f\n" % c_mAP)
                f.write("mAP of b: %.4f\n" % b_mAP)
                f.write("mAP of p: %.4f\n" % p_mAP)
                f.write("mAP of c+: %.4f\n" % group_c_mAP)
                f.write("mAP of b+: %.4f\n" % group_b_mAP)
                f.write("mAP of p+: %.4f\n" % group_p_mAP)

                f.write("c per class: \n")
                for ap in mAP_per_class[:12].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")

                f.write("b per class: \n")
                for ap in mAP_per_class[12:24].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")

                f.write("c+ per class: \n")
                for ap in mAP_per_class[24:36].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")

                f.write("b+ per class: \n")
                for ap in mAP_per_class[36:48].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")

                f.write("p per class: \n")
                for ap in mAP_per_class[48:56].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")

                f.write("p+ per class: \n")
                for ap in mAP_per_class[56:64].tolist():
                    f.write("%.4f " % ap)
                f.write("\n")
                f.write("*" * 15 + "\n")

            total_loss = self.loss_epoch / float(self.num_batches)
            tqdm.write(f"Epoch {self.cur_epoch:03d} Loss: {total_loss:3.3f}")
            self.val_loss.append(total_loss)

        return save_cp, [mAP, total_loss]

    def save(self, is_best):
        save_best = False
        if is_best:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        if save_best:
            torch.save(
                self.model.state_dict(), os.path.join(logdir, "best_model.pth")
            )
            tqdm.write("====== Overwrote best model ======>")


if __name__ == "__main__":
    args, logdir = parser()
    print(args)
    logdir = logdir.replace(":", "_").replace("\n", "_").replace(" ", "")

    abs_logdir = os.path.abspath(logdir)
    if os.name == "nt" and not abs_logdir.startswith("\\\\?\\"):
        abs_logdir = f"\\\\?\\{abs_logdir}"

    os.makedirs(abs_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    seq_len = args.seq_len

    num_ego_class = 4
    num_actor_class = 64

    print("initialize train set")
    train_set = TACO(args=args, split="train")
    print("initialize val set")
    val_set = TACO(args=args, split="val")

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

    model = generate_model(args, num_ego_class, num_actor_class).cuda()

    if "mvit" == args.model_name:
        params = set_lr(model)
    else:
        params = [{"params": model.parameters()}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr
        )
    else:
        scheduler = None

    trainer = Engine(args, model, optimizer, num_actor_class, scheduler)

    print(f"Checkpoint path: {logdir}")

    result_list = []
    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train()
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            is_best, res = trainer.validate(dataloader_val)
            trainer.save(is_best)
            result_list.append(res)

    print("********** Best model **********")
    for s in trainer.best_log:
        print(s)
    plot_result(np.array(result_list), args)