# Action-Slot (TACO Dataset & Modular Video Architecture)

This repository provides an optimized, modular, and extensible deep learning codebase for the **TACO (Traffic Action and Condition)** video benchmark. It centers on the **Action-Slot** architecture combining 3D spatiotemporal backbones (such as **X3D**), **3D Slot Attention**, and multi-task classification heads.

It features full support for **Automatic Mixed Precision (AMP FP16)**, **Gradient Accumulation**, **Differential Learning Rates**, **Spatiotemporal Video Augmentations**, **Attention Regularization**, and **Early Stopping** to prevent overfitting.

---

## Directory Structure

```text
action_slot/
├── core/                        # Modular registry & training engine
│   ├── base_model.py            # BaseActionModel interface (standard output dict)
│   ├── engine.py                # Decoupled Trainer (training loop, eval, early stopping, Excel logging)
│   └── registry.py              # Dynamic module registries (MODELS, BACKBONES, HEADS, LOSSES)
├── dataset/                     # TACO Data Loaders & Augmentations
│   ├── taco.py                  # Train/Val Dataset loader supporting LMDB and disk folders
│   ├── taco_test.py             # Inference/Test loader (single image, image folders, or full dataset)
│   ├── transforms.py            # Spatiotemporal video augmentations (ColorJitter, Flip, Cutout/Erasing)
│   ├── taco_train_data.json     # Training split metadata
│   ├── taco_train_label.json    # Training split action labels
│   ├── taco_val_data.json       # Validation split metadata
│   ├── taco_val_label.json      # Validation split action labels
│   └── taco_test_data.json      # Test split metadata
├── models/                      # Plug-and-Play Model Modules
│   ├── action_slot.py           # Composite Action-Slot model with regularized slot attention
│   ├── backbones.py             # Feature extractor backbones (X3D, etc.)
│   ├── heads.py                 # Multi-task heads (Linear Head, Allocated Head, Deformable Head)
│   ├── slot_attention.py        # 3D Slot Attention with attention dropout & soft position embeds
│   ├── x3d_baseline.py          # Video-level X3D baseline model
│   └── __init__.py              # Model factory (build_model)
├── losses/                      # Loss Functions
│   └── __init__.py              # Multi-task losses (ActionSlotLoss, FocalActionLoss, Hungarian matching)
├── scripts/                     # Execution Entry Points
│   ├── args.py                  # CLI argument parser definitions for train, eval, and test
│   ├── train.py                 # Training script with Accelerate, AMP & Differential LR
│   ├── eval.py                  # Quantitative evaluation script (mAP, Precision, Recall)
│   └── test.py                  # Inference script supporting single images or video frame folders
└── utils/                       # Diagnostics, Logging & Visualization
    ├── attention_dumper.py      # Slot attention matrix dumping & heatmap rendering
    ├── excel_logger.py          # Comprehensive logging of weights, Q/K/V norms & metrics to Excel
    ├── metrics.py               # Metric trackers (AverageMeter)
    ├── normalization.py         # Dataset mean/std normalization configurations
    ├── taco_labels.py           # Canonical list of 64 actor action labels
    ├── tracker.py               # Per-class performance tracking & diagnostic summaries
    └── visualizer.py            # Attention heatmap visualization (Blue -> Orange -> Red)
```

---

## Installation & Environment

Create and activate a conda environment with PyTorch and torchvision:

```bash
conda create -n Action_Slot python=3.10 -y && conda activate Action_Slot && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && pip install lmdb pillow tqdm scikit-learn numpy openpyxl
```

---

## Usage Guide

### 1. Training

The training script supports **Automatic Mixed Precision (AMP FP16)**, **Gradient Accumulation**, **Differential Learning Rates**, and **Overfitting Prevention Regularizations**.

#### Recommended Training (Anti-Overfitting Configuration)
To maintain strong validation generalization and avoid early saturation on training data:
```bash
python scripts/train.py --model_name action_slot --backbone x3d --lmdb_train_path "/path/to/TACO/train" --lmdb_val_path "/path/to/TACO/val" --lr 5e-4 --backbone_lr 5e-5 --wd 0.1 --dropout 0.5 --slot_dropout 0.15 --augment --patience 3 --early_stopping_metric val_map --batch_size 8 --val_every 5 --diagnose
```

#### Memory-Efficient Training (For 4GB–6GB VRAM GPUs)
When training at higher resolutions (e.g., $T=16, H=256, W=768$), use **AMP FP16** and **Gradient Accumulation** to keep peak VRAM under **3.5 GB**:

```bash
python scripts/train.py --model_name action_slot --lmdb_train_path "D:\apps\Datasets\TACO\train" --lmdb_val_path "D:\apps\Datasets\TACO\val" --img_height 256 --img_width 768 --batch_size 2 --grad_accum_steps 4 --amp --clip_grad_norm 1.0 --backbone_lr 5e-5 --augment --diagnose --diagnose_every_steps 500 --plot_attn --epochs 100 --lr 5e-4
```

**Key Training Options:**
* `--augment`: Enables spatiotemporal video augmentation (ColorJitter, RandomHorizontalFlip, and RandomErasing/Cutout).
* `--slot_dropout`: Dropout probability applied to the Slot Attention affinity matrix (prevents slot over-specialization and key-norm explosion).
* `--dropout`: Dropout probability before classification heads (default: `0.5`).
* `--backbone_lr`: Sets a separate, lower learning rate for the feature extraction backbone (e.g. `5e-5` vs `5e-4` for slots/heads).
* `--patience`: Early stopping patience (number of validation evaluations without improvement before halting training; set `0` to disable).
* `--early_stopping_metric`: Metric monitored for early stopping (`val_map` or `val_loss`).
* `--amp`: Enables PyTorch `torch.amp.autocast` for FP16 mixed precision.
* `--grad_accum_steps`: Accumulates gradients across multiple micro-batches (`Effective Batch Size = batch_size * grad_accum_steps * num_gpus`).
* `--accelerate`: Enables HuggingFace Accelerate for distributed multi-GPU training.
* `--wandb`: Enables logging to Weights & Biases.
* `--diagnose`: Tracks per-class TP, FP, FN, AP, and outputs diagnostic reports and `training_full_metrics.xlsx`.

---

### 2. Quantitative Evaluation

Evaluate model checkpoints on the validation dataset:

```bash
python scripts/eval.py --model_name action_slot --checkpoint checkpoints/best_model.pth --lmdb_val_path "/path/to/TACO/val" --batch_size 8
```

You can also pass `--plot` to generate attention matrix heatmaps for all 64 actor categories:
```bash
python scripts/eval.py --model_name action_slot --checkpoint checkpoints/best_model.pth --lmdb_val_path "/path/to/TACO/val" --batch_size 8 --plot
```

---

### 3. Inference / Testing

The testing mechanism supports direct inference on **a single image**, **a folder of video frames**, or **the full benchmark dataset**:

#### A. Test a Single Image
Replicates the image across $T=16$ frames and outputs the Top-5 predicted actor actions directly to the console:
```bash
python scripts/test.py --checkpoint checkpoints/best_model.pth --input_path "path/to/sample_image.jpg"
```

#### B. Test a Directory of Frames (Video Clip)
Loads and uniformly samples 16 frames from the folder:
```bash
python scripts/test.py --checkpoint checkpoints/best_model.pth --input_path "path/to/frames_folder" --plot --save_dir test_results
```

#### C. Full Benchmark Test Set
Evaluates across all scenarios defined in the benchmark JSON:
```bash
python scripts/test.py --checkpoint checkpoints/best_model.pth --root "/path/to/test_frames_root" --output_pkl prediction_results.pkl
```

---

## Memory Footprint & VRAM Estimation

Reference peak VRAM consumption when training with video input shape **$T=16, H=256, W=768$**:

| Batch Size | Precision | Peak VRAM | Recommended Hardware |
| :---: | :---: | :---: | :--- |
| **$B = 1$** | FP32 | $\approx 3.24\text{ GB}$ | $\ge 4\text{ GB}$ GPU |
| **$B = 2$** | FP32 | $\approx 6.47\text{ GB}$ | $\ge 8\text{ GB}$ GPU |
| **$B = 2$** | **AMP FP16** | **$\approx 3.28\text{ GB}$** | **$4\text{ GB} - 6\text{ GB}$ GPU (e.g., RTX 3050 / 4050 Laptop)** |
| **$B = 4$** | **AMP FP16** | **$\approx 6.50\text{ GB}$** | $\ge 8\text{ GB}$ GPU (e.g., RTX 3070 / 4060 Ti) |
| **$B = 8$** | **AMP FP16** | **$\approx 13.0\text{ GB}$** | $\ge 16\text{ GB}$ GPU (e.g., V100 / RTX 4090 / A100) |
