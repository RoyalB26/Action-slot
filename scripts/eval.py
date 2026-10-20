"""Refactored, decoupled evaluation entry point for any registered video model."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_score, recall_score
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco import TACO
from models import build_model
from scripts.args import get_eval_parser


def evaluate_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute detailed evaluation metrics."""
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
        'Precision': float(precision),
        'Recall': float(recall),
    }


def main():
    args = get_eval_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    val_dataset = TACO(
        args,
        lmdb_train_path=args.lmdb_train_path,
        lmdb_val_path=args.lmdb_val_path,
        split='val',
        root=args.root,
        path_to_idx_file=args.path_to_idx_file,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_ego_class = 7
    num_actor_class = 64

    # Build model via registry
    model = build_model(args, num_ego_class=num_ego_class, num_actor_class=num_actor_class)
    if args.model_path and os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {args.model_path}")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            videos = [f.to(device, dtype=torch.float32) for f in batch['videos']]
            label_actor = batch['gt_actor'].to(device)

            preds = model(videos)
            pred_actor = preds["pred_actor"]

            if pred_actor.ndim == 3:
                pred_actor = pred_actor.max(dim=1)[0]
            if pred_actor.shape[-1] > num_actor_class:
                pred_actor = pred_actor[:, :num_actor_class]

            pred_probs = torch.sigmoid(pred_actor).cpu().numpy()
            all_preds.append(pred_probs)
            all_gts.append(label_actor.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    metrics = evaluate_metrics(all_gts, all_preds)

    print("\n" + "=" * 40)
    print("      EVALUATION METRICS SUMMARY")
    print("=" * 40)
    print(f"Mean Average Precision (mAP) : {metrics['mAP']:.4f}")
    print(f"Macro Precision              : {metrics['Precision']:.4f}")
    print(f"Macro Recall                 : {metrics['Recall']:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()
