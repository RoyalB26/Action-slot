"""Refactored, decoupled test set inference script."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco_test import TACO_TEST
from models import build_model
from scripts.args import get_test_parser


def main():
    args = get_test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dataset = TACO_TEST(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
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
        print(f"Loaded checkpoint: {args.model_path}")
    model = model.to(device)
    model.eval()

    save_results = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            scenario = batch['scenario'][0]
            videos = [f.to(device, dtype=torch.float32) for f in batch['videos']]

            preds = model(videos)
            pred_actor = preds["pred_actor"]

            if pred_actor.ndim == 3:
                pred_actor = pred_actor.max(dim=1)[0]
            if pred_actor.shape[-1] > num_actor_class:
                pred_actor = pred_actor[:, :num_actor_class]

            pred_actor_prob = torch.sigmoid(pred_actor).cpu().numpy()
            save_results[scenario] = pred_actor_prob

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(save_results, f)
    print(f"Predictions saved to {args.save_path}")


if __name__ == '__main__':
    main()
