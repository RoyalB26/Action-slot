# TACO Action-Slot Benchmark

This script provides comprehensive benchmarking for TACO action-slot models, comparing multiple checkpoints and configurations.

## Features

- **Multiple Metrics**: mAP, F1-score, ego accuracy, per-class performance
- **Model Comparison**: Evaluate multiple trained models side-by-side
- **Baseline Comparison**: Includes random baseline for reference
- **Detailed Results**: Saves comprehensive results to JSON
- **Action-Only Focus**: Optimized for 20-class action recognition (z + c actions)

## Usage

### 1. Prepare Model Configurations

Create a JSON file with model configurations (see `benchmark_configs.json` for example):

```json
{
  "model_name_1": {
    "path": "path/to/model1.pth",
    "description": "Description of model 1",
    "backbone": "x3d",
    "num_slots": 21
  },
  "model_name_2": {
    "path": "path/to/model2.pth",
    "description": "Description of model 2"
  },
  "random_baseline": {
    "path": null,
    "description": "Random baseline",
    "type": "baseline"
  }
}
```

### 2. Run Benchmark

```bash
python scripts/benchmark_taco.py \
  --root "C:\TACO" \
  --model_configs benchmark_configs.json \
  --output benchmark_results.json \
  --seq_len 16
```

### 3. Results

The script will output:
- Real-time progress for each model
- Summary table comparing all models
- Top performers ranking
- Detailed JSON results file

## Metrics

- **mAP**: Mean Average Precision (overall and per action type)
- **F1-Score**: Macro and micro F1 scores
- **Ego Accuracy**: Accuracy for ego vehicle action prediction
- **Per-Class mAP**: Performance breakdown by individual action classes
- **Improvement over Random**: How much better than random baseline

## Enhanced Eval Script

The `eval_taco.py` script now includes additional benchmark metrics:
- F1-scores for z and c actions
- Random baseline comparison
- JSON export of results

## Example Output

```
=== BENCHMARK SUMMARY ===
Model                  mAP      F1       Ego Acc
action_slot_x3d        0.1234   0.5678   0.7890
random_baseline        0.0500   0.1234   0.2500

=== TOP PERFORMERS ===
Best mAP: action_slot_x3d (0.1234)
Best Macro F1: action_slot_x3d (0.5678)
```

## Notes

- All models are evaluated on the TACO validation set
- Action-only mapping (20 classes: 12 z-actions + 8 c-actions)
- Ego vehicle has 4 classes
- Results are saved to JSON for further analysis