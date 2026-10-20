"""Command-line interface arguments for training, evaluation, and inference."""

import argparse


def get_train_parser():
    """Build and parse CLI arguments for model training on TACO.

    Returns:
        argparse.Namespace: Parsed training options.
    """
    parser = argparse.ArgumentParser(description="Action-Slot Training Parser for TACO")

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='taco', help='Target dataset name')
    parser.add_argument('--root', type=str, default='', help='Path to raw dataset frames directory')
    parser.add_argument('--lmdb_train_path', type=str, default=None, help='Path to lmdb train folder or database')
    parser.add_argument('--lmdb_val_path', type=str, default=None, help='Path to lmdb val folder or database')
    parser.add_argument('--path_to_idx_file', type=str, default=None, help='Optional explicit mapping path to index json')
    parser.add_argument('--seq_len', type=int, default=16, help='Number of video frames per clip')
    parser.add_argument('--img_height', type=int, default=128, help='Input frame image height')
    parser.add_argument('--img_width', type=int, default=384, help='Input frame image width')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='action_slot', choices=['action_slot', 'x3d'], help='Model type')
    parser.add_argument('--backbone', type=str, default='x3d', help='Feature extractor backbone')
    parser.add_argument('--head_name', type=str, default='linear_head', choices=['linear_head', 'allocated_head', 'deformable_head', 'head'], help='Classification head type')
    parser.add_argument('--num_slots', type=int, default=64, help='Number of slots')
    parser.add_argument('--channel', type=int, default=256, help='Feature channel dimension')
    parser.add_argument('--bg_slot', action='store_true', help='Append extra background slot')

    # Attention parameters
    parser.add_argument('--action_attn_weight', type=float, default=0.0, help='Weight for action attention loss')
    parser.add_argument('--bg_attn_weight', type=float, default=0.0, help='Weight for background attention loss')
    parser.add_argument('--bg_mask', action='store_true', help='Supervise background mask')
    parser.add_argument('--obj_mask', action='store_true', help='Supervise object masks')
    parser.add_argument('--bg_upsample', type=int, default=4, help='Background mask upsampling factor')
    parser.add_argument('--mask_every_frame', type=int, default=2, help='Downsample frame stride for attention/mask supervision')
    parser.add_argument('--box', action='store_true', help='Enable bounding box / Hungarian matching in actor loss')

    # Loss parameters
    parser.add_argument('--loss_name', type=str, default='action_slot_loss', help='Loss function name registered in LOSSES (e.g. action_slot_loss, focal_loss)')
    parser.add_argument('--bce_pos_weight', type=float, default=10.0, help='Positive weight for BCE loss')
    parser.add_argument('--ce_pos_weight', type=float, default=1.0, help='Positive weight for CE loss')
    parser.add_argument('--ce_neg_weight', type=float, default=0.05, help='Negative weight for CE loss')
    parser.add_argument('--ego_loss_weight', type=float, default=0.5, help='Weight for ego-vehicle loss')

    # Optimization
    parser.add_argument('--device', type=str, default='cuda', help='Device target (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of total epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.07, help='Weight decay')
    parser.add_argument('--scheduler', action='store_true', help='Enable learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per iteration')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP FP16)')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers (default: 2 for multi-process memory safety)')
    parser.add_argument('--val_every', type=int, default=5, help='Validation frequency in epochs')
    parser.add_argument('--diagnose', action='store_true', help='Run per-class performance tracking and failure diagnosis during validation')
    parser.add_argument('--diagnose_every_steps', type=int, default=100, help='Print running diagnostic stats every N training steps (0 to disable)')
    parser.add_argument('--plot_attn', action='store_true', help='Save attention heatmap visualizations (Blue->Orange->Red)')
    parser.add_argument('--accelerate', action='store_true', help='Use HuggingFace Accelerate for distributed/mixed training')
    parser.add_argument('--wandb', action='store_true', help='Log metrics to Weights & Biases (disables tqdm bar for clean remote logging)')
    parser.add_argument('--wandb_project', type=str, default='Action-Slot-TACO', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run display name')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint output directory')

    return parser.parse_args()


def get_eval_parser():
    """Build and parse CLI arguments for model evaluation on TACO validation set.

    Returns:
        argparse.Namespace: Parsed evaluation options.
    """
    parser = argparse.ArgumentParser(description="Action-Slot Evaluation Parser for TACO")
    parser.add_argument('--dataset', type=str, default='taco', help='Target dataset')
    parser.add_argument('--root', type=str, default='', help='Path to raw dataset frames')
    parser.add_argument('--model_name', type=str, default='action_slot', choices=['action_slot', 'x3d'], help='Model type')
    parser.add_argument('--backbone', type=str, default='x3d', help='Backbone network')
    parser.add_argument('--head_name', type=str, default='linear_head', choices=['linear_head', 'allocated_head', 'deformable_head', 'head'], help='Classification head type')
    parser.add_argument('--num_slots', type=int, default=64, help='Number of slots')
    parser.add_argument('--seq_len', type=int, default=16, help='Input sequence length')
    parser.add_argument('--img_height', type=int, default=128, help='Input frame image height')
    parser.add_argument('--img_width', type=int, default=384, help='Input frame image width')
    parser.add_argument('--channel', type=int, default=256, help='Feature channel dimension')
    parser.add_argument('--bg_slot', action='store_true', help='Include background slot')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Worker threads')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='Evaluation results directory')
    parser.add_argument('--plot', action='store_true', help='Visualize attention maps')
    parser.add_argument('--plot_threshold', type=float, default=0.3, help='Threshold for visualization')
    return parser.parse_args()


def get_test_parser():
    """Build and parse CLI arguments for offline test-set inference.

    Returns:
        argparse.Namespace: Parsed inference options.
    """
    parser = argparse.ArgumentParser(description="Action-Slot Inference Parser for TACO")
    parser.add_argument('--dataset', type=str, default='taco', help='Dataset name')
    parser.add_argument('--root', type=str, required=True, help='Path to test video frames')
    parser.add_argument('--model_name', type=str, default='action_slot', choices=['action_slot', 'x3d'], help='Model type')
    parser.add_argument('--backbone', type=str, default='x3d', help='Backbone network')
    parser.add_argument('--head_name', type=str, default='linear_head', choices=['linear_head', 'allocated_head', 'deformable_head', 'head'], help='Classification head type')
    parser.add_argument('--num_slots', type=int, default=64, help='Number of slots')
    parser.add_argument('--seq_len', type=int, default=16, help='Frames per clip')
    parser.add_argument('--img_height', type=int, default=128, help='Input frame image height')
    parser.add_argument('--img_width', type=int, default=384, help='Input frame image width')
    parser.add_argument('--channel', type=int, default=256, help='Feature channel dimension')
    parser.add_argument('--bg_slot', action='store_true', help='Include background slot')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device target')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for test inference')
    parser.add_argument('--num_workers', type=int, default=2, help='Worker threads')
    parser.add_argument('--output_pkl', type=str, default='prediction_results.pkl', help='Output pickle file')
    return parser.parse_args()
