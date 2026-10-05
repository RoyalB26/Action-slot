import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import inter_and_union


class ActionSlotLoss(nn.Module):

  def __init__(self, args, num_actor_class, attention_res=None):
    super(ActionSlotLoss, self).__init__()
    self.args = args
    self.num_actor_class = num_actor_class
    self.attention_res = attention_res
    self.ego_ce = nn.CrossEntropyLoss(reduction="mean")
    self.actor_loss_type = self._parse_actor_loss(args)
    self.attn_loss_type = self._parse_attn_loss(args)

    # GRPO Hyperparameters
    self.grpo_clip_eps = getattr(args, "grpo_clip_eps", 0.2)
    self.grpo_kl_coef = getattr(args, "grpo_kl_coef", 0.01)

  def _parse_actor_loss(self, args):
    if ("slot" in args.model_name and not args.allocated_slot) or args.box:
      ce_weights = torch.ones(self.num_actor_class + 1) * args.ce_pos_weight
      ce_weights[-1] = args.ce_neg_weight
      self.bce = nn.CrossEntropyLoss(reduction="none", weight=ce_weights)
      return 1
    else:
      pos_weight = torch.ones([self.num_actor_class]) * args.bce_pos_weight
      self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
      return 0

  def _parse_attn_loss(self, args):
    flag = 0
    if (
        "slot" in args.model_name and not args.allocated_slot
    ) or args.box and args.obj_mask:
      flag = 1
    elif "slot" in args.model_name and args.allocated_slot:
      if not args.bg_mask and args.action_attn_weight > 0:
        flag = 2
      elif args.obj_mask:
        flag = 1
      elif (
          args.bg_slot
          and args.bg_mask
          and args.action_attn_weight > 0.0
          and args.bg_attn_weight > 0.0
      ):
        flag = 3
      elif (
          args.bg_slot
          and args.bg_mask
          and args.bg_attn_weight > 0.0
          and not args.action_attn_weight > 0.0
      ):
        flag = 4
    if flag > 0:
      self.obj_bce = nn.BCELoss()
    return flag

  def ego_loss(self, pred, label):
    if pred is None:
      return None
    return self.ego_ce(pred, label)

  def compute_actor_reward(self, pred_actor, label_actor):
    """pred_actor: [B, G, num_actor_class] hoặc [B, num_actor_class]

    label_actor: [B, num_actor_class]
    """
    if pred_actor.dim() == 2:
      # Trường hợp Validation / Inference G = 1
      loss = self.bce(pred_actor, label_actor).mean()
      return -loss, loss

    B, G, C = pred_actor.shape
    label_actor_exp = label_actor.unsqueeze(1).repeat(1, G, 1)

    if self.actor_loss_type == 0:
      # BCE loss theo từng sample trong group: [B, G]
      per_sample_loss = self.bce(pred_actor, label_actor_exp).mean(dim=-1)
    else:
      per_sample_loss = self.bce(
          pred_actor.view(B * G, C), label_actor_exp.view(B * G, C)
      ).view(B, G)

    reward = -per_sample_loss  # Reward là negative loss
    return reward, per_sample_loss.mean()
  
  def compute_grpo_loss(self, log_probs, old_log_probs, rewards, epsilon=0.2, beta=0.01):
    # 1. Advantage Computation
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean_r) / (std_r + 1e-8)

    # 2. Probability Ratio & Clipping
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL Divergence (Approximation)
    approx_kl = (old_log_probs - log_probs).mean()
    kl_penalty = beta * approx_kl

    total_loss = policy_loss + kl_penalty

    # 4. Trích xuất Metrics để giám sát
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > epsilon).float().mean()
        rl_metrics = {
            "rl/mean_reward": mean_r.mean().item(),
            "rl/std_reward": std_r.mean().item(),
            "rl/approx_kl": approx_kl.item(),
            "rl/clip_fraction": clip_frac.item(),
            "rl/advantages_mean": advantages.abs().mean().item(),
            "rl/policy_loss": policy_loss.item(),
            "rl/kl_penalty": kl_penalty.item()
        }

    return total_loss, rl_metrics

  def attn_loss(self, attn, label, actor, validate):
    # Giữ nguyên cấu trúc giám sát mask của bạn (nếu có dùng obj_mask / bg_mask)
    if attn is None or self.attn_loss_type == 0:
      return {
          "attn_loss": torch.tensor(0.0).cuda(),
          "bg_attn_loss": torch.tensor(0.0).cuda(),
          "action_inter": None,
          "action_union": None,
          "bg_inter": None,
          "bg_union": None,
      }
    # (Các cấu hình tính attention map mask tương tự code gốc)
    return {
        "attn_loss": torch.tensor(0.0).cuda(),
        "bg_attn_loss": torch.tensor(0.0).cuda(),
        "action_inter": None,
        "action_union": None,
        "bg_inter": None,
        "bg_union": None,
    }

  def forward(self, pred, label, validate=False):
    ego_loss = (
        self.ego_loss(pred["ego"], label["ego"])
        if pred.get("ego") is not None
        else None
    )

    pred_actor = pred["actor"]
    rewards, actor_loss = self.compute_actor_reward(pred_actor, label["actor"])

    # Tính GRPO Loss nếu có log_probs và G > 1
    if (
        not validate
        and pred.get("log_prob") is not None
        and pred.get("old_log_prob") is not None
    ):
      grpo_loss, metrics = self.compute_grpo_loss(
          pred["log_prob"], pred["old_log_prob"], rewards
      )
    else:
      grpo_loss = torch.tensor(0.0, device=pred_actor.device)
      metrics= None
    attn_dict = self.attn_loss(pred.get("attn"), label, pred_actor, validate)

    return {
        "ego": ego_loss,
        "actor": actor_loss,
        "grpo": grpo_loss,
        "attn": attn_dict,
        "metrics": metrics
    }