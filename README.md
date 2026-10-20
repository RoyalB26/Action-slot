# Action-Slot (TACO Dataset & X3D Backbone)

This repository provides an optimized, standalone implementation tailored for the **TACO (Traffic Action and Condition)** video benchmark, focusing on the **Action-Slot** architecture with an **X3D** spatiotemporal backbone (as well as the pure X3D baseline).

It includes full support for **Automatic Mixed Precision (AMP FP16)**, **Gradient Accumulation**, and customizable spatiotemporal feature extraction to enable scalable training across various GPU setups (including constrained setups like 4GB–6GB VRAM GPUs).

---

## Directory Structure

```text
action_slot/
├── dataset/
│   ├── taco.py              # TACO Dataset loader supporting LMDB and disk image folders
│   ├── taco_test.py         # TACO Dataset loader for test-set inference
│   ├── taco_train_data.json # Training split metadata
│   ├── taco_train_label.json# Training split action labels
│   ├── taco_val_data.json   # Validation split metadata
│   ├── taco_val_label.json  # Validation split action labels
│   └── taco_test_data.json  # Test split metadata
├── model/
│   ├── action_slot_x3d.py   # Spatiotemporal Action-Slot model with X3D backbone
│   ├── x3d_baseline.py      # Standard video-level X3D baseline model
│   ├── slot_attention.py    # 3D Slot Attention mechanism & soft positional embeddings
│   ├── heads.py             # Prediction heads (Standard, Allocated, Deformable Temporal)
│   └── __init__.py          # Model factory (generate_model)
├── scripts/
│   ├── args.py              # CLI argument parser definitions for train, eval, and test
│   ├── loss.py              # Multi-task loss functions (BCE, CrossEntropy, Ego Loss)
│   ├── train.py             # Training & validation loop with AMP & Gradient Accumulation
│   ├── eval.py              # Quantitative evaluation script (mAP calculation)
│   └── test.py              # Offline inference script saving predictions to pickle
└── utils/
    ├── metrics.py           # Metric trackers (AverageMeter, IoU utilities)
    ├── normalization.py     # Mean/std dataset normalization configurations
    └── __init__.py
```

---

## Installation & Environment

Create and activate a conda environment with PyTorch and torchvision:

```bash
conda create -n Action_Slot python=3.10 -y
conda activate Action_Slot
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lmdb pillow tqdm scikit-learn numpy
```

---

## Usage Guide

### 1. Training

The training script supports **Automatic Mixed Precision (AMP FP16)**, **Gradient Accumulation**, and **Gradient Clipping**.

#### Standard Training (Default Batch Size 8)
```bash
python scripts/train.py --model_name action_slot --backbone x3d --lmdb_train_path /path/to/TACO/train --lmdb_val_path /path/to/TACO/val --batch_size 8 --epochs 100 --lr 5e-4
```
> **Note on LMDB path**: You only need to pass the directory containing the dataset (e.g. `D:\apps\Datasets\TACO\train`). The loader will automatically detect `taco_train.lmdb` and `taco_path_to_idx.json` inside that folder.

#### Memory-Efficient Training (Recommended for GPUs with 4GB–6GB VRAM)
When training with higher resolutions (e.g., $T=16, H=256, W=768$), use **AMP FP16** and **Gradient Accumulation** to maintain an effective batch size of 8 while keeping peak VRAM below **3.3 GB**:

```bash
python scripts/train.py --model_name action_slot --lmdb_train_path "D:\apps\Datasets\TACO\train" --lmdb_val_path "D:\apps\Datasets\TACO\val" --img_height 256 --img_width 768 --batch_size 2 --grad_accum_steps 4 --amp --clip_grad_norm 1.0 --diagnose --diagnose_every_steps 500 --plot_attn --epochs 100 --lr 5e-4
```

**Key Training Options:**
* `--img_height`, `--img_width`: Set input frame dimensions (defaults: `128` and `384`; use `256` and `768` for high resolution).
* `--amp`: Enables PyTorch `torch.amp.autocast` and `torch.amp.GradScaler` for FP16 mixed precision.
* `--grad_accum_steps`: Number of forward/backward steps before updating model weights (`Effective Batch Size = batch_size * grad_accum_steps`).
* `--accelerate`: Enables HuggingFace Accelerate for distributed multi-GPU training.
* `--wandb`: Enables logging to Weights & Biases (automatically disables the local `tqdm` progress bar for clean logging).
* `--wandb_project`: Project name for Weights & Biases (default: `Action-Slot-TACO`).
* `--wandb_run_name`: Optional custom display name for the run in wandb.
* `--clip_grad_norm`: Maximum gradient norm value (e.g. `1.0`) to avoid gradient explosions.
* `--allocated_slot`: Assigns slots directly to action categories.
* `--save_dir`: Directory to store the checkpoints (`best_model.pth` and `latest_model.pth`).

---

### 2. Evaluation

Evaluate model checkpoints on the validation dataset:

```bash
python scripts/eval.py --model_name action_slot --checkpoint checkpoints/best_model.pth --batch_size 8
```

---

### 3. Inference / Testing

Generate offline action predictions and save to a `.pkl` file:

```bash
python scripts/test.py --model_name action_slot --checkpoint checkpoints/best_model.pth --root /path/to/test_frames --output_pkl prediction_results.pkl
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
