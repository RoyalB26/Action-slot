"""Inference script generating test set predictions saved to a pickle file."""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset.taco_test import TACO_TEST
from model import generate_model
from scripts.args import get_test_parser


def main():
    """Main inference loop for unseen test scenarios."""
    args = get_test_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading TACO test dataset from: {args.root}")
    test_dataset = TACO_TEST(args, split='test', root=args.root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_ego_class = 7
    num_actor_class = 64

    print(f"Instantiating model {args.model_name}...")
    model = generate_model(args, num_ego_class, num_actor_class).to(device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    save_results = {}

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            scenario = batch['scenario'][0]
            # Input video frames: list of seq_len tensors with shape (B, 3, 128, 384)
            videos = [f.to(device, dtype=torch.float32) for f in batch['videos']]

            outputs = model(videos)
            if len(outputs) == 3:
                pred_ego, pred_actor, _ = outputs
            else:
                pred_ego, pred_actor = outputs

            if getattr(args, 'allocated_slot', False):
                pred_actor_prob = torch.sigmoid(pred_actor).cpu().numpy()  # Shape: (B, 64)
            else:
                pred_actor_prob = torch.softmax(pred_actor, dim=-1).cpu().numpy()

            save_results[scenario] = pred_actor_prob

    print(f"Saving predictions to: {args.output_pkl}")
    with open(args.output_pkl, 'wb') as handle:
        pickle.dump(save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Inference completed successfully!")


if __name__ == '__main__':
    main()
