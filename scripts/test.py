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
from utils.attention_dumper import save_attention_matrix, render_all_slots_from_file
from utils.taco_labels import TACO_ACTOR_LABELS
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
    saved_attn_sample = None
    saved_video_sample = None
    plot_enabled = getattr(args, 'plot', False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            scenario = batch['scenario'][0]
            videos = [f.to(device, dtype=torch.float32) for f in batch['videos']]

            preds = model(videos)

            if plot_enabled and saved_attn_sample is None and preds.get("attn") is not None:
                saved_attn_sample = preds["attn"]
                saved_video_sample = videos
            pred_actor = preds["pred_actor"]

            if pred_actor.ndim == 3:
                pred_actor = pred_actor.max(dim=1)[0]
            if pred_actor.shape[-1] > num_actor_class:
                pred_actor = pred_actor[:, :num_actor_class]

            pred_actor_prob = torch.sigmoid(pred_actor).cpu().numpy()
            save_results[scenario] = pred_actor_prob

    save_file_path = getattr(args, 'save_path', None) or getattr(args, 'output_pkl', 'prediction_results.pkl')
    parent_dir = os.path.dirname(save_file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(save_file_path, 'wb') as f:
        pickle.dump(save_results, f)
    print(f"Predictions saved to {save_file_path}")

    # Render attention matrix and generate all 64 slot visualizations if --plot is enabled
    if plot_enabled and saved_attn_sample is not None and saved_video_sample is not None:
        plot_dir = os.path.join(getattr(args, 'save_dir', 'test_results'), "attention_plots")
        os.makedirs(plot_dir, exist_ok=True)
        mat_path = os.path.join(plot_dir, "test_attention_matrix.npz")
        save_attention_matrix(
            attn_tensor=saved_attn_sample,
            frames=saved_video_sample,
            save_path=mat_path,
            extra_metadata={"scenario": scenario},
        )
        print("Rendering 64 slot visualizations across all T frames for test sample...")
        render_all_slots_from_file(
            matrix_file=mat_path,
            output_dir=plot_dir,
            slot_names=TACO_ACTOR_LABELS,
            prefix="test",
        )


if __name__ == '__main__':
    main()
