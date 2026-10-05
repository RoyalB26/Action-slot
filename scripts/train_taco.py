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
from pcgrad import PCGrad
from datasets.taco import TACO
from get_parser import parser
from loss import ActionSlotLoss
from model import generate_model
from utils import AverageMeter
import warnings

warnings.filterwarnings("ignore")
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

def check_real_performance(y_true, y_pred, threshold=0.5):
    """
    y_true: [N, 64] (numpy array)
    y_pred: [N, 64] (numpy array - sau khi qua sigmoid)
    """
    # 1. Tìm các class có ít nhất 1 sample xuất hiện trong batch
    active_classes = np.where(y_true.sum(axis=0) > 0)[0]
    print(f"Tổng số class thực tế xuất hiện trong 6 batch: {len(active_classes)} / 64")
    
    # 2. Tính AP riêng cho các class đang có mẫu
    ap_list = []
    for cls in active_classes:
        ap = average_precision_score(y_true[:, cls], y_pred[:, cls])
        ap_list.append(ap)
        print(f"  Class {cls:02d}: AP = {ap:.4f} (Số mẫu dương: {int(y_true[:, cls].sum())})")
        
    real_mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0
    print(f"===> mAP thực tế trên các active classes: {real_mAP * 100:.2f}%")
    
    # 3. Tính thêm F1-score / Accuracy tại ngưỡng threshold
    pred_binary = (y_pred > threshold).astype(int)
    correct_pos = ((pred_binary == 1) & (y_true == 1)).sum()
    total_pos = y_true.sum()
    print(f"===> Tỉ lệ bắt trúng nhãn dương (Recall): {correct_pos} / {int(total_pos)} ({correct_pos/max(total_pos,1)*100:.1f}%)")

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

        self.raw_optimizer = optimizer
        self.optimizer = PCGrad(optimizer)

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

        self.cur_epoch = self.args.start_epoch
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
        self.rl_metrics= {
            "rl/mean_reward": 0.0,
            "rl/std_reward": 0.0,
            "rl/approx_kl": 0.0,
            "rl/clip_fraction": 0.0,
            "rl/advantages_mean": 0.0,
            "rl/policy_loss": 0.0,
            "rl/kl_penalty": 0.0
        }
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

        stage1_epochs = getattr(self.args, "stage1_epochs", 30)
        stage = 1 if self.cur_epoch < stage1_epochs else 2

        if mode == "train":
            # =================================================================
            # STAGE 1: SUPERVISED WARMUP (TẮT GRPO, CHẠY G=1)
            # =================================================================
            if stage == 1:
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
                loss_dict = self.criterion(pred_dict, batch, validate=False)

                ego_loss = (
                    loss_dict["ego"]
                    if loss_dict["ego"] is not None
                    else torch.tensor(0.0, device=self.accelerator.device)
                )
                actor_loss = loss_dict["actor"]
                attn_loss = loss_dict["attn"]["attn_loss"]

                total_loss = actor_loss + self.args.ego_loss_weight * ego_loss + attn_loss

                self.optimizer.zero_grad()
                self.accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logging
                self.loss_epoch += float(self.accelerator.gather(total_loss).mean().item())
                self.actor_loss_epoch += float(self.accelerator.gather(actor_loss).mean().item())
                self.ego_loss_epoch += float(self.accelerator.gather(ego_loss).mean().item())
                self.attn_loss_epoch += float(self.accelerator.gather(attn_loss).mean().item())
                self.grpo_loss_epoch += 0.0

                pred_actor_sig = torch.sigmoid(pred_actor)
                self.map_pred_actor_list.append(
                    self.accelerator.gather_for_metrics(pred_actor_sig).detach().cpu().numpy()
                )
                self.label_actor_list.append(
                    self.accelerator.gather_for_metrics(batch["actor"]).detach().cpu().numpy()
                )

                if pred_ego is not None:
                    target_ego = batch["ego"].view(-1)
                    _, pred_ego_idx = torch.max(pred_ego.data, dim=1)
                    correct = (pred_ego_idx == target_ego).sum()
                    total = torch.tensor(target_ego.size(0), device=self.accelerator.device)
                    self.correct_ego += self.accelerator.gather(correct).sum().item()
                    self.total_ego += self.accelerator.gather(total).sum().item()

            # =================================================================
            # STAGE 2: GRPO FINE-TUNING (CHUYỂN TIẾP MỎ NEO AN TOÀN)
            # =================================================================
            else:
                # 1. Rollout pi_old lấy G nhóm trajectories
                with torch.no_grad():
                    pred_ego_old, pred_actor_old, _, old_log_prob = self.model(
                        inputs, num_groups=self.num_groups
                    )

                # 2. Forward pi_theta
                pred_ego, pred_actor, attn, log_prob = self.model(
                    inputs, num_groups=self.num_groups
                )

                pred_dict = {
                    "ego": pred_ego,
                    "actor": pred_actor,
                    "attn": attn,
                    "log_prob": log_prob,
                    "old_log_prob": old_log_prob,
                }
                loss_dict = self.criterion(pred_dict, batch, validate=False)

                grpo_loss = loss_dict["grpo"]
                actor_loss = loss_dict["actor"]
                attn_loss = loss_dict["attn"]["attn_loss"]
                metrics = loss_dict['metrics']
                ego_loss = (
                    loss_dict["ego"]
                    if loss_dict["ego"] is not None
                    else torch.tensor(0.0, device=self.accelerator.device)
                )

                # 4 Epochs chuyển tiếp (Transition): Giữ Actor Loss làm mỏ neo tránh vỡ chính sách
                if self.cur_epoch < stage1_epochs + 4:
                    total_loss = (
                        actor_loss 
                        + self.args.ego_loss_weight * ego_loss 
                        + attn_loss 
                        + 0.05 * grpo_loss
                    )
                else:
                    # Giai đoạn sau: Tối ưu GRPO kèm nhánh Ego cố định
                    total_loss = 0.2 * actor_loss + grpo_loss + self.args.ego_loss_weight * ego_loss + 0.2 * attn_loss

                self.optimizer.zero_grad()
                self.accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Logging Stage 2
                self.loss_epoch += float(self.accelerator.gather(total_loss).mean().item())
                self.actor_loss_epoch += float(self.accelerator.gather(actor_loss).mean().item())
                self.grpo_loss_epoch += float(self.accelerator.gather(grpo_loss).mean().item())
                self.ego_loss_epoch += float(self.accelerator.gather(ego_loss).mean().item())
                self.attn_loss_epoch += float(self.accelerator.gather(attn_loss).mean().item())
                if metrics:
                    self.rl_metrics = {k: self.rl_metrics[k] + metrics[k] for k in self.rl_metrics}
                # Gom dự đoán trung bình của G nhóm để tính metric
                pred_actor_mean = torch.sigmoid(pred_actor.mean(dim=1))
                self.map_pred_actor_list.append(
                    self.accelerator.gather_for_metrics(pred_actor_mean).detach().cpu().numpy()
                )
                self.label_actor_list.append(
                    self.accelerator.gather_for_metrics(batch["actor"]).detach().cpu().numpy()
                )

                if pred_ego is not None:
                    target_ego = batch["ego"].view(-1)
                    _, pred_ego_idx = torch.max(pred_ego.data, dim=1)
                    correct = (pred_ego_idx == target_ego).sum()
                    total = torch.tensor(target_ego.size(0), device=self.accelerator.device)
                    self.correct_ego += self.accelerator.gather(correct).sum().item()
                    self.total_ego += self.accelerator.gather(total).sum().item()

        # =====================================================================
        # VALIDATE / TEST (DETERMINISTIC VỚI G=1 STREAMING)
        # =====================================================================
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

                # Logging loss
                self.loss_epoch += float(self.accelerator.gather(total_loss).mean().item())

                # Metric Actor
                pred_actor_sig = torch.sigmoid(pred_actor)
                self.map_pred_actor_list.append(
                    self.accelerator.gather_for_metrics(pred_actor_sig).detach().cpu().numpy()
                )
                self.label_actor_list.append(
                    self.accelerator.gather_for_metrics(batch["actor"]).detach().cpu().numpy()
                )

                # Metric Ego
                if pred_ego is not None:
                    target_ego = batch["ego"].view(-1)
                    _, pred_ego_idx = torch.max(pred_ego.data, dim=1)
                    correct = (pred_ego_idx == target_ego).sum()
                    total = torch.tensor(target_ego.size(0), device=self.accelerator.device)
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

        log_interval = max(1, self.num_batches // 10)

        for step_idx, data in enumerate(pbar if not disable_pbar else dataloader_train):
            self.step(data, "train")
            if self.args.wandb and step_idx % log_interval == 0:
                progress_pct = (step_idx + 1) / self.num_batches * 100
                self.accelerator.log(
                    {
                        "train/batch_progress_pct": progress_pct,
                        "train/current_batch": step_idx + 1,
                        "epoch": self.cur_epoch,
                    }
                )

        map_pred_actor_list = np.concatenate(self.map_pred_actor_list, axis=0)
        label_actor_list = np.concatenate(self.label_actor_list, axis=0)

        try:
            ap_per_class = average_precision_score(
                label_actor_list,
                map_pred_actor_list.astype(np.float32),
                average=None
            )
            valid_ap = ap_per_class[~np.isnan(ap_per_class)]
            mAP = float(np.mean(valid_ap)) if len(valid_ap) > 0 else 0.0
        except Exception:
            mAP = 0.0

        loss_epoch = self.loss_epoch / self.num_batches
        actor_loss_epoch = self.actor_loss_epoch / self.num_batches
        grpo_loss_epoch = self.grpo_loss_epoch / self.num_batches
        attn_loss_epoch = self.attn_loss_epoch / self.num_batches
        epoch_metrics = {k: self.rl_metrics[k] / self.num_batches for k in self.rl_metrics}
        # Log metrics lên WandB / Tracker
        self.accelerator.log(
            {
                "train/total_loss": loss_epoch,
                "train/actor_loss": actor_loss_epoch,
                "train/grpo_loss": grpo_loss_epoch,
                "train/ego_loss": self.ego_loss_epoch / self.num_batches,
                "train/mAP": mAP,
                "train/attn_loss": attn_loss_epoch,
                "epoch": self.cur_epoch,
            },
            step=self.cur_epoch,
        )

        self.accelerator.log(self.rl_metrics, step= self.cur_epoch)

        self.accelerator.print(f"\n[Epoch {self.cur_epoch}] Total Loss: {loss_epoch:.4f}")
        self.accelerator.print(
            f"Actor Loss: {actor_loss_epoch:.4f} | GRPO Loss: {grpo_loss_epoch:.4f}"
        )
        
        self.accelerator.print(f"--- RL Stats ---")
        self.accelerator.print(f"Mean Reward: {epoch_metrics['rl/mean_reward']:.4f} | Std Reward: {epoch_metrics['rl/std_reward']:.4f}")
        self.accelerator.print(f"Approx KL: {epoch_metrics['rl/approx_kl']:.5f} | Clip Fraction: {epoch_metrics['rl/clip_fraction']*100:.2f}%")

        # check_real_performance(label_actor_list, map_pred_actor_list)
        self.accelerator.print(f"(train) mAP of the actor: {mAP:.4f}")
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1



    def validate(self, dataloader):
        self.model.eval()
        self.reset_log()
        self.num_batches = len(dataloader)
        save_cp = False

        disable_pbar = self.args.wandb or (not self.accelerator.is_local_main_process)
        log_interval = max(1, self.num_batches // 10)

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

            for step_idx, data in enumerate(pbar if not disable_pbar else dataloader_val):
                self.step(data, "val")

                if self.args.wandb and step_idx % log_interval == 0:
                    if self.accelerator.is_local_main_process:
                        progress_pct = (step_idx + 1) / self.num_batches * 100
                        print(f"[Validating] {step_idx + 1}/{self.num_batches} ({progress_pct:.1f}%)", flush=True)

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
                # Tính chi tiết từng lớp cho actor
                mAP_per_class = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32),
                    average=None,
                )

                ego_acc = self.correct_ego / max(self.total_ego, 1)

                # 1. In ra màn hình console
                self.accelerator.print(f"\n--- Validation Epoch {self.cur_epoch} Results ---")
                self.accelerator.print(f"(val) Loss: {total_loss:.4f}")
                self.accelerator.print(f"(val) mAP: {mAP:.4f}")
                self.accelerator.print(f"(val) mAP of c: {c_mAP:.4f} | b: {b_mAP:.4f} | p: {p_mAP:.4f}")
                self.accelerator.print(f"(val) mAP of c+: {group_c_mAP:.4f} | b+: {group_b_mAP:.4f} | p+: {group_p_mAP:.4f}")
                self.accelerator.print(f"(val) Ego Acc: {ego_acc:.4f}")

                # 2. Log lên WandB
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

                # 3. Cập nhật best model flag
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
                    self.accelerator.print(f"--> Found new best mAP: {self.best_mAP:.4f}")

                # 4. Ghi chi tiết kết quả vào file mAP.txt
                with open(os.path.join(self.logdir, "mAP.txt"), "a") as f:
                    f.write(f"epoch: {self.cur_epoch}\n")
                    f.write(f"best mAP: {self.best_mAP:.4f}\n")
                    f.write(f"mAP: {mAP:.4f}\n")
                    f.write(f"mAP of c: {c_mAP:.4f}\n")
                    f.write(f"mAP of b: {b_mAP:.4f}\n")
                    f.write(f"mAP of p: {p_mAP:.4f}\n")
                    f.write(f"mAP of c+: {group_c_mAP:.4f}\n")
                    f.write(f"mAP of b+: {group_b_mAP:.4f}\n")
                    f.write(f"mAP of p+: {group_p_mAP:.4f}\n")

                    f.write("c per class: \n")
                    for ap in mAP_per_class[:12].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")

                    f.write("b per class: \n")
                    for ap in mAP_per_class[12:24].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")

                    f.write("c+ per class: \n")
                    for ap in mAP_per_class[24:36].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")

                    f.write("b+ per class: \n")
                    for ap in mAP_per_class[36:48].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")

                    f.write("p per class: \n")
                    for ap in mAP_per_class[48:56].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")

                    f.write("p+ per class: \n")
                    for ap in mAP_per_class[56:64].tolist():
                        f.write(f"{ap:.4f} ")
                    f.write("\n")
                    f.write("*" * 15 + "\n")

                self.val_loss.append(total_loss)
                return save_cp, [mAP, total_loss]
            else:
                return False, [0.0, total_loss]

    def save(self, is_best):

       if is_best and self.accelerator.is_main_process:
            
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

    # Khởi tạo Accelerator với logging TensorBoard/W&B
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
                    "name": os.path.basename(abs_logdir),
                    "config": vars(args),
                }
            }
        accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT", "taco_experiments"),
            init_kwargs=init_kwargs,
        )

    args.device = accelerator.device
    seq_len = args.seq_len

    num_ego_class = 4
    num_actor_class = 64

    accelerator.print("initialize train set")
    train_set = TACO(args=args, split="train", accelerator=accelerator)
    accelerator.print("initialize val set")
    val_set = TACO(args=args, split="val", accelerator=accelerator)

    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
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

    # =========================================================================
    # NẠP CHECKPOINT (RESUME HOẶC KHỞI ĐỘNG STAGE 2 TỪ CHECKPOINT STAGE 1)
    # =========================================================================
    checkpoint_path = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise Exception("Lỗi đường dẫn checkpoint")
        accelerator.print(f"\n>>> Đang nạp checkpoint từ: {checkpoint_path}")
        # Map về device hiện tại thông qua accelerator
        ckpt = torch.load(checkpoint_path, map_location=accelerator.device)

        # 1. Trích xuất state_dict của model
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Nạp weights vào model đã unwrap
        unwrapped_model = accelerator.unwrap_model(model)
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"    Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")

        # 2. Xử lý logic khôi phục Optimizer / Epoch:
        # Kiểm tra xem đây là nạp để chạy Stage 2 luôn hay resume bình thường
        start_at_stage2 = getattr(args, "start_stage2", False) or getattr(args, "stage1_epochs", 30) == 0

        if start_at_stage2:
            accelerator.print(">>> Nạp checkpoint Stage 1 để vào thẳng STAGE 2: Khởi tạo lại LR = 1e-5 & Reset Scheduler")
            trainer.cur_epoch = args.stage1_epochs  # Đưa thẳng con trỏ epoch về mốc Stage 2
            trainer.best_mAP = ckpt.get("best_mAP", ckpt.get("mAP", 1e-5))

            # Set LR = 1e-5 cho toàn bộ param groups
            raw_opt = trainer.raw_optimizer if hasattr(trainer, "raw_optimizer") else trainer.optimizer
            for param_group in raw_opt.param_groups:
                param_group['lr'] = 1e-5
            trainer.scheduler = None  # Không dùng scheduler cũ
        else:
            # Resume tiếp tục quá trình bình thường
            if "epoch" in ckpt:
                trainer.cur_epoch = ckpt["epoch"] + 1
                accelerator.print(f"    Khôi phục tiếp tục từ epoch: {trainer.cur_epoch}")
            if "best_mAP" in ckpt:
                trainer.best_mAP = ckpt["best_mAP"]
            if "optimizer_state_dict" in ckpt and optimizer is not None:
                try:
                    raw_opt = trainer.raw_optimizer if hasattr(trainer, "raw_optimizer") else trainer.optimizer
                    raw_opt.load_state_dict(ckpt["optimizer_state_dict"])
                    accelerator.print("    Đã nạp lại trạng thái Optimizer thành công.")
                except Exception as e:
                    accelerator.print(f"    Cảnh báo: Không thể nạp optimizer_state_dict ({e}), giữ optimizer mới.")
            if "scheduler_state_dict" in ckpt and trainer.scheduler is not None:
                try:
                    trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass

    accelerator.print(f"Checkpoint path lưu logs: {abs_logdir}")

    result_list = []
    unwrapped_model = accelerator.unwrap_model(model)

    for epoch in range(trainer.cur_epoch, args.epochs):
        # 1. Trạng thái Stage 1
        if epoch < args.stage1_epochs:
            unwrapped_model.setup_stage(stage=1)
            accelerator.print(f"\n>>> [Epoch {epoch}] Đang chạy STAGE 1: Supervised Warmup")

        # 2. Bước chuyển giao sang Stage 2 (Chỉ thực hiện thiết lập 1 lần ở epoch đầu tiên của Stage 2)
        elif epoch == args.stage1_epochs:
            unwrapped_model.setup_stage(stage=2)
            
            # Lấy đúng raw optimizer bất kể có dùng PCGrad hay không
            raw_opt = trainer.raw_optimizer if hasattr(trainer, "raw_optimizer") else trainer.optimizer
            for param_group in raw_opt.param_groups:
                param_group['lr'] = 2.5e-5

            # Tắt scheduler cũ để không bị ghi đè LR cũ
            trainer.scheduler = None  
            
            accelerator.print(f"\n>>> [Epoch {epoch}] BẮT ĐẦU STAGE 2: Set LR = 2.5e-5, Tắt Scheduler cũ & Đóng băng Backbone + Head")

        # 3. Các epoch còn lại của Stage 2
        else:
            accelerator.print(f"\n>>> [Epoch {epoch}] Đang chạy STAGE 2: GRPO Fine-tuning (Backbone & Head Frozen)")

        trainer.train(dataloader_train)
        
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            is_best, res = trainer.validate(dataloader_val)
            trainer.save(is_best)

    if accelerator.is_main_process:
        print("********** Best model **********")
        for s in trainer.best_log:
            print(s)
        plot_result(np.array(result_list), args)
        accelerator.end_training()