"""Refactored, decoupled training entry point for Action-Slot and arbitrary video architectures."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco import TACO
from models import build_model
from losses import build_loss
from core.engine import Trainer
from scripts.args import get_train_parser


def main():
    """Main training workflow using modular components."""
    args = get_train_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Setup Distributed / Mixed Precision Accelerator
    accelerator = None
    if getattr(args, 'accelerate', False):
        from accelerate import Accelerator, DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision='fp16' if args.amp else 'no',
            gradient_accumulation_steps=args.grad_accum_steps,
            kwargs_handlers=[ddp_kwargs],
        )
        device = accelerator.device

    is_main = accelerator is None or accelerator.is_main_process

    # 2. Setup WandB remote tracking (rank 0 only)
    use_wandb = getattr(args, 'wandb', False)
    if use_wandb and is_main:
        import wandb
        wandb.init(
            project=getattr(args, 'wandb_project', 'Action-Slot-TACO'),
            name=getattr(args, 'wandb_run_name', None),
            config=vars(args),
        )
        print(f"Weights & Biases initialized: Project '{args.wandb_project}'")

    if is_main:
        print("Loading TACO datasets...")

    # 3. DataLoaders
    train_dataset = TACO(
        args,
        lmdb_train_path=args.lmdb_train_path,
        lmdb_val_path=args.lmdb_val_path,
        split='train',
        root=args.root,
        path_to_idx_file=args.path_to_idx_file,
    )
    val_dataset = TACO(
        args,
        lmdb_train_path=args.lmdb_train_path,
        lmdb_val_path=args.lmdb_val_path,
        split='val',
        root=args.root,
        path_to_idx_file=args.path_to_idx_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    num_ego_class = 7
    num_actor_class = 64
    args.num_ego_class = num_ego_class
    args.num_actor_class = num_actor_class

    # 4. Instantiate Modular Model & Loss via Registry
    if is_main:
        print(f"Instantiating model '{args.model_name}' (backbone='{getattr(args, 'backbone', 'x3d')}, head= {getattr(args, 'head_name', 'linear_head')}') via Registry...")
    model = build_model(args, num_ego_class=num_ego_class, num_actor_class=num_actor_class)

    # 4.1 Load Pretrained Checkpoint for Finetuning
    finetune_path = getattr(args, 'finetune', None)
    if finetune_path and os.path.exists(finetune_path):
        state_dict = torch.load(finetune_path, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remove potential 'module.' prefix from DDP saving
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if is_main:
            print(f">> Successfully loaded pretrained weights for finetuning from: '{finetune_path}'")
            if missing_keys:
                print(f"   Missing keys (e.g. newly initialized heads): {len(missing_keys)} layers")
            if unexpected_keys:
                print(f"   Unexpected keys (skipped): {len(unexpected_keys)} layers")
    elif finetune_path:
        if is_main:
            print(f"Warning: Specified finetune path '{finetune_path}' does not exist! Training from scratch.")

    criterion = build_loss(args, num_actor_class=num_actor_class)

    if accelerator is None:
        model = model.to(device)
        criterion = criterion.to(device)

    # 5. Optimizer with Layer-wise Learning Rate support
    backbone_lr = getattr(args, 'backbone_lr', None)
    if backbone_lr is not None and hasattr(model, 'backbone'):
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': args.lr},
        ]
        if is_main:
            print(f"Differential LR enabled: Backbone LR={backbone_lr}, Head/Slots LR={args.lr}")
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd
        )

    # 6. Accelerate DDP wrap
    if accelerator is not None:
        model, optimizer, train_loader, val_loader, criterion = accelerator.prepare(
            model, optimizer, train_loader, val_loader, criterion
        )

    # 7. Print configuration
    if is_main:
        base_m = accelerator.unwrap_model(model) if accelerator is not None else model
        total_params = sum(p.numel() for p in base_m.parameters())
        trainable_params = sum(p.numel() for p in base_m.parameters() if p.requires_grad)
        num_train_data = len(train_dataset)
        num_val_data = len(val_dataset)
        num_procs = accelerator.num_processes if accelerator is not None else 1
        effective_batch_size = args.batch_size * args.grad_accum_steps * num_procs

        print(f"Training configuration:")
        print(f"  - Total Parameters : {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  - Trainable Params : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(f"  - Total Train Data : {num_train_data:,} samples")
        print(f"  - Total Val Data   : {num_val_data:,} samples")
        print(f"  - Total Epochs     : {args.epochs}")
        print(f"  - Learning Rate    : {args.lr}")
        print(f"  - Micro batch size : {args.batch_size}")
        print(f"  - Accumulation step: {args.grad_accum_steps}")
        print(f"  - Total processes  : {num_procs}")
        print(f"  - Effective batch  : {effective_batch_size}")
        print(f"  - AMP Mixed Prec   : {'Enabled' if args.amp else 'Disabled'}")
        print(f"  - WandB Remote Log : {'Enabled' if use_wandb else 'Disabled'}")

    # 8. Hand off execution to decoupled Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        accelerator=accelerator,
    )

    trainer.fit(epochs=args.epochs, val_every=args.val_every)


if __name__ == '__main__':
    main()
