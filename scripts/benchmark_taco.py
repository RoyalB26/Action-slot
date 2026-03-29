#!/usr/bin/env python3
"""
Benchmark script for TACO action-slot models
Compares multiple model checkpoints and configurations
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score

# Add paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import taco
from model import generate_model
from parser_eval import get_eval_parser

torch.backends.cudnn.benchmark = True

class ModelBenchmark:
    def __init__(self, args):
        self.args = args
        self.results = {}

    def load_model(self, model_path, num_ego_class=4, num_actor_class=20):
        """Load model from checkpoint"""
        model = generate_model(self.args, num_ego_class, num_actor_class).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def evaluate_model(self, model, dataloader):
        """Evaluate single model and return metrics"""
        map_pred_actor_list = []
        label_actor_list = []
        correct_ego = 0
        total_ego = 0

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Evaluating"):
                rgb = data['rgb'].cuda()
                ego_gt = data['ego_gt'].cuda()
                actor_gt = data['actor_gt'].cuda()

                ego_pred, actor_pred = model(rgb)

                # Ego accuracy
                ego_pred_class = torch.argmax(ego_pred, dim=1)
                correct_ego += (ego_pred_class == ego_gt).sum().item()
                total_ego += ego_gt.size(0)

                # Actor predictions
                map_pred_actor_list.extend(actor_pred.cpu().numpy())
                label_actor_list.extend(actor_gt.cpu().numpy())

        # Convert to numpy arrays
        map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
        label_actor_list = np.stack(label_actor_list, axis=0)
        map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], 20))
        label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], 20))

        # Calculate metrics
        mAP = average_precision_score(label_actor_list, map_pred_actor_list.astype(np.float32))
        z_mAP = average_precision_score(label_actor_list[:, :12], map_pred_actor_list[:, :12].astype(np.float32))
        c_mAP = average_precision_score(label_actor_list[:, 12:20], map_pred_actor_list[:, 12:20].astype(np.float32))
        mAP_per_class = average_precision_score(label_actor_list, map_pred_actor_list.astype(np.float32), average=None)

        # F1 scores
        pred_binary = (map_pred_actor_list > 0.5).astype(int)
        label_binary = label_actor_list.astype(int)
        macro_f1 = f1_score(label_binary, pred_binary, average='macro', zero_division=0)
        z_f1 = f1_score(label_binary[:, :12], pred_binary[:, :12], average='macro', zero_division=0)
        c_f1 = f1_score(label_binary[:, 12:20], pred_binary[:, 12:20], average='macro', zero_division=0)

        ego_acc = correct_ego / total_ego

        return {
            'mAP': float(mAP),
            'z_mAP': float(z_mAP),
            'c_mAP': float(c_mAP),
            'macro_f1': float(macro_f1),
            'z_f1': float(z_f1),
            'c_f1': float(c_f1),
            'ego_accuracy': float(ego_acc),
            'per_class_mAP': mAP_per_class.tolist()
        }

    def benchmark_models(self, model_configs, dataloader):
        """Benchmark multiple models"""
        print("=== STARTING BENCHMARK ===")
        print(f"Dataset: TACO (action-only, 20 classes)")
        print(f"Number of models to evaluate: {len(model_configs)}")
        print()

        for config_name, config in model_configs.items():
            print(f"Evaluating: {config_name}")
            try:
                if config.get('type') == 'baseline':
                    # Random baseline
                    metrics = self.evaluate_random_baseline(dataloader)
                else:
                    model = self.load_model(config['path'])
                    metrics = self.evaluate_model(model, dataloader)

                self.results[config_name] = {
                    'config': config,
                    'metrics': metrics
                }

                print(".4f")
                print(".4f")
                print(".4f")
                print(".4f")
                print(".4f")
                print()

            except Exception as e:
                print(f"Error evaluating {config_name}: {e}")
                self.results[config_name] = {'error': str(e)}

        return self.results

    def evaluate_random_baseline(self, dataloader):
        """Evaluate random baseline"""
        label_actor_list = []
        correct_ego = 0
        total_ego = 0

        # Collect all labels
        for data in tqdm(dataloader, desc="Collecting labels for baseline"):
            ego_gt = data['ego_gt']
            actor_gt = data['actor_gt']

            # Random ego prediction
            ego_pred_class = torch.randint(0, 4, ego_gt.shape).cuda()
            correct_ego += (ego_pred_class == ego_gt.cuda()).sum().item()
            total_ego += ego_gt.size(0)

            label_actor_list.extend(actor_gt.cpu().numpy())

        label_actor_list = np.stack(label_actor_list, axis=0)
        label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], 20))

        # Random predictions
        np.random.seed(42)
        random_pred = np.random.rand(*label_actor_list.shape)

        # Calculate metrics
        mAP = average_precision_score(label_actor_list, random_pred.astype(np.float32))
        z_mAP = average_precision_score(label_actor_list[:, :12], random_pred[:, :12].astype(np.float32))
        c_mAP = average_precision_score(label_actor_list[:, 12:20], random_pred[:, 12:20].astype(np.float32))
        mAP_per_class = average_precision_score(label_actor_list, random_pred.astype(np.float32), average=None)

        # F1 scores
        pred_binary = (random_pred > 0.5).astype(int)
        label_binary = label_actor_list.astype(int)
        macro_f1 = f1_score(label_binary, pred_binary, average='macro', zero_division=0)
        z_f1 = f1_score(label_binary[:, :12], pred_binary[:, :12], average='macro', zero_division=0)
        c_f1 = f1_score(label_binary[:, 12:20], pred_binary[:, 12:20], average='macro', zero_division=0)

        ego_acc = correct_ego / total_ego

        return {
            'mAP': float(mAP),
            'z_mAP': float(z_mAP),
            'c_mAP': float(c_mAP),
            'macro_f1': float(macro_f1),
            'z_f1': float(z_f1),
            'c_f1': float(c_f1),
            'ego_accuracy': float(ego_acc),
            'per_class_mAP': mAP_per_class.tolist()
        }

    def save_results(self, output_file):
        """Save benchmark results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Benchmark results saved to: {output_file}")

    def print_summary(self):
        """Print summary of results"""
        print("=== BENCHMARK SUMMARY ===")
        print("<20")
        print("-" * 60)

        for model_name, result in self.results.items():
            if 'error' in result:
                print("<20")
                continue

            metrics = result['metrics']
            print("<20")

        print("\n=== TOP PERFORMERS ===")
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            best_mAP = max(valid_results.items(), key=lambda x: x[1]['metrics']['mAP'])
            print(f"Best mAP: {best_mAP[0]} ({best_mAP[1]['metrics']['mAP']:.4f})")

            best_f1 = max(valid_results.items(), key=lambda x: x[1]['metrics']['macro_f1'])
            print(f"Best Macro F1: {best_f1[0]} ({best_f1[1]['metrics']['macro_f1']:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Benchmark TACO models')
    parser.add_argument('--root', type=str, required=True, help='Dataset root path')
    parser.add_argument('--model_configs', type=str, required=True,
                       help='JSON file with model configurations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length')

    args = parser.parse_args()

    # Load model configurations
    with open(args.model_configs, 'r') as f:
        model_configs = json.load(f)

    # Setup data
    eval_args = get_eval_parser()[0]  # Get default args
    eval_args.root = args.root
    eval_args.seq_len = args.seq_len

    val_set = taco.TACO(args=eval_args, split='val')
    dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False,
                               num_workers=4, pin_memory=True, drop_last=True)

    # Run benchmark
    benchmark = ModelBenchmark(eval_args)
    results = benchmark.benchmark_models(model_configs, dataloader_val)
    benchmark.save_results(args.output)
    benchmark.print_summary()

if __name__ == "__main__":
    main()