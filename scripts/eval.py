"""Evaluation script computing mAP, Precision, and Recall on the TACO dataset."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_score, recall_score
from tqdm import tqdm

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco import TACO
from model import generate_model
from scripts.args import get_eval_parser


TACO_ACTOR_LABELS = [
    'c:z1-z2', 'c:z1-z3', 'c:z1-z4',
    'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
    'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
    'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
    'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
    'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
    'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
    'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
    'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
    'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
    'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
    'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
    'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
    'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
    'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
    'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
    'p:c1-c2', 'p:c1-c4', 
    'p:c2-c1', 'p:c2-c3', 
    'p:c3-c2', 'p:c3-c4', 
    'p:c4-c1', 'p:c4-c3', 
    'p+:c1-c2', 'p+:c1-c4', 
    'p+:c2-c1', 'p+:c2-c3', 
    'p+:c3-c2', 'p+:c3-c4', 
    'p+:c4-c1', 'p+:c4-c3',
    'bg'
]


def evaluate_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute detailed evaluation metrics.

    Args:
        y_true (np.ndarray): Binary ground truth labels of shape (N, num_classes).
        y_pred_prob (np.ndarray): Predicted probability matrix of shape (N, num_classes).
        threshold (float): Binarization decision threshold.

    Returns:
        dict[str, float]: Dictionary containing mAP, Precision, and Recall scores.
    """
    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    num_classes = y_true.shape[1]
    aps = []
    for i in range(num_classes):
        if np.sum(y_true[:, i]) > 0:
            aps.append(average_precision_score(y_true[:, i], y_pred_prob[:, i]))

    mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0
    precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)

    return {
        'mAP': mean_ap,
        'Precision': precision,
        'Recall': recall,
    }


def main():
    """Main evaluation execution flow."""
    args = get_eval_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading TACO validation dataset...")
    val_dataset = TACO(args, split='val', root=args.root)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_ego_class = 7
    num_actor_class = 64

    print(f"Instantiating model {args.model_name}...")
    model = generate_model(args, num_ego_class, num_actor_class).to(device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_preds = []
    all_gts = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Input video frames: list of 16 tensors with shape (B, 3, 128, 384)
            videos = [f.to(device, dtype=torch.float32) for f in batch['videos']]
            label_actor = batch['gt_actor'].to(device)  # Shape: (B, 64)

            if hasattr(model, 'slot_attention'):
                _, pred_actor, _ = model(videos)       # pred_actor shape: (B, 64)
            else:
                _, pred_actor = model(videos)

            pred_probs = torch.sigmoid(pred_actor).cpu().numpy()  # Shape: (B, 64)
            all_preds.append(pred_probs)
            all_gts.append(label_actor.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N, 64)
    all_gts = np.concatenate(all_gts, axis=0)      # Shape: (N, 64)

    metrics = evaluate_metrics(all_gts, all_preds)
    print("\n" + "=" * 40)
    print(" TACO EVALUATION RESULTS ")
    print("=" * 40)
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()
