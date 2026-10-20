# Hướng Dẫn Kỹ Thuật Dành Cho Developer (Action-Slot Core Architecture)

Tài liệu này cung cấp hướng dẫn chi tiết dành cho developers về cấu trúc mã nguồn đã được tái cấu trúc (refactored), thiết kế theo mẫu **Registry & Modular Pattern**, quy chuẩn dữ liệu đầu vào/đầu ra, cùng cách mở rộng thêm **Backbone**, **Head**, **Loss**, và **Model mới** mà không phá vỡ logic huấn luyện hiện tại.

---

## 1. Cấu Trúc Mã Nguồn (Architecture Overview)

```
action_slot/
├── core/                               <-- Nền tảng cốt lõi (Core Engine)
│   ├── registry.py                     # Quản lý đăng ký động: MODELS, BACKBONES, HEADS, LOSSES
│   ├── base_model.py                   # Interface BaseActionModel ép chuẩn Output Dict
│   └── engine.py                       # Trainer Engine (DDP, AMP, Eval, mAP, Memory Cleanup)
├── models/                             <-- Khối mô hình độc lập (Plug & Play)
│   ├── backbones.py                    # Trích xuất đặc trưng video (X3D, VideoSwin, v.v.)
│   ├── heads.py                        # Các loại phân loại (Linear, Allocated, Deformable)
│   ├── slot_attention.py               # Spatiotemporal 3D Slot Attention
│   ├── action_slot.py                  # Mô hình ghép nối ActionSlot
│   └── x3d_baseline.py                 # Baseline X3D
├── losses/                             <-- Các hàm mục tiêu / Loss
│   └── __init__.py                     # ActionSlotLoss, FocalActionLoss, build_loss
├── dataset/                            <-- Dataloaders cho TACO
│   ├── taco.py                         # Train/Val dataset loader
│   └── taco_test.py                    # Test dataset loader
├── utils/                              <-- Công cụ bổ trợ
│   ├── excel_logger.py                 # Ghi toàn bộ gradients, Q/K/V, per-class ra Excel
│   ├── tracker.py                      # Đánh giá chi tiết mAP, precision, recall, confusion
│   └── visualizer.py                   # Vẽ attention heatmap 3 màu (Xanh -> Cam -> Đỏ)
└── scripts/                            <-- Điểm vào thực thi (Entry Points)
    ├── args.py                         # CLI argument parser
    ├── train.py                        # Script huấn luyện chính (gọi Trainer.fit)
    ├── eval.py                         # Đánh giá mAP trên tập validation
    └── test.py                         # Sinh file kết quả nộp bài / kiểm thử
```

---

## 2. Tiêu Chuẩn Output Bắt Buộc (`BaseActionModel`)

Mọi mô hình được phát triển mới **bắt buộc** phải kế thừa từ `core.base_model.BaseActionModel` và trả về một `dict` (thay vì tuple):

```python
from core.base_model import BaseActionModel

class MyCustomModel(BaseActionModel):
    def forward(self, videos, **kwargs) -> dict:
        # Xử lý ...
        return {
            "pred_actor": actor_logits,   # Tensor (B, num_actor_class) hoặc (B, num_slots, num_actor_class)
            "pred_ego": ego_logits,       # Tensor (B, num_ego_class) hoặc None
            "attn": attention_masks,      # Tensor (B, T, num_slots, H, W) hoặc None
        }
```

> **Lưu ý:** Việc trả về dạng Dictionary giúp ngăn chặn hoàn toàn lỗi `ValueError: too many values to unpack`, cho phép thay đổi cấu trúc đầu ra mà không phải sửa `Trainer`, `eval.py` hay `loss.py`.

---

## 3. Hướng Dẫn Mở Rộng Hệ Thống (How-To Extend)

Hệ thống sử dụng cơ chế **Registry**. Khi thêm thành phần mới, bạn chỉ cần dùng decorator `@REGISTRY.register("tên_thành_phần")`.

### 3.1. Thêm một Backbone mới (ví dụ: VideoSwin hoặc SlowFast)

Mở [`models/backbones.py`](models/backbones.py) và khai báo:

```python
from core.registry import BACKBONES
from models.backbones import BaseBackbone

@BACKBONES.register("videoswin")
class VideoSwinBackbone(BaseBackbone):
    def __init__(self, pretrained=True):
        super().__init__()
        # Khởi tạo mô hình VideoSwin
        self.swin = ...
        self._out_channels = 768  # Kích thước channel đầu ra

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        # x có thể là list of T tensors (B, 3, H, W) hoặc 5D tensor (B, 3, T, H, W)
        feat_map = self.swin(x)   # (B, C_out, T_feat, H_feat, W_feat)
        return feat_map
```

**Sử dụng ngay trong CLI:**
```bash
python scripts/train.py --backbone videoswin
```

---

### 3.2. Thêm một Head phân loại mới (Classification Head)

Mở [`models/heads.py`](models/heads.py):

```python
from core.registry import HEADS
import torch.nn as nn

@HEADS.register("mlp_head")
class MLPHead(nn.Module):
    def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
        super().__init__()
        self.mlp_actor = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.ReLU(),
            nn.Linear(512, num_actor_classes)
        )
        self.mlp_ego = nn.Linear(ego_channel, num_ego_classes) if num_ego_classes > 0 else None

    def forward(self, x, ego_x=None, **kwargs):
        pred_actor = self.mlp_actor(x)
        pred_ego = self.mlp_ego(ego_x) if self.mlp_ego and ego_x is not None else None
        return pred_ego, pred_actor
```

---

### 3.3. Thêm một hàm Loss mới (ví dụ: Asymmetric Loss / Dice Loss)

Mở [`losses/__init__.py`](losses/__init__.py):

```python
from core.registry import LOSSES
from losses import BaseLoss

@LOSSES.register("asl_loss")
class AsymmetricLoss(BaseLoss):
    def __init__(self, num_actor_class=64, gamma_neg=4, gamma_pos=1, **kwargs):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.ego_ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets, **kwargs):
        pred_actor = preds["pred_actor"]
        target_actor = targets["target_actor"]
        
        # Tính ASL Loss cho actor ...
        loss_actor = ...

        # Tính Loss cho Ego
        loss_ego = self.ego_ce(preds["pred_ego"], targets["target_ego"]) if preds["pred_ego"] is not None else 0.0

        total_loss = loss_actor + 0.5 * loss_ego
        return total_loss, {"loss_actor": loss_actor, "loss_ego": loss_ego, "total_loss": total_loss}
```

**Sử dụng trong CLI:**
```bash
python scripts/train.py --loss_name asl_loss
```

---

### 3.4. Thêm một Kiến trúc hoàn toàn mới (New Model Architecture)

Tạo file mới trong thư mục `models/` (ví dụ `models/pure_transformer.py`):

```python
from core.registry import MODELS
from core.base_model import BaseActionModel

@MODELS.register("pure_transformer")
class PureTransformerModel(BaseActionModel):
    def __init__(self, num_ego_class=7, num_actor_class=64, **kwargs):
        super().__init__()
        self.encoder = ...
        self.head = ...

    def forward(self, videos, **kwargs):
        # Tính toán
        return {
            "pred_actor": actor_preds,
            "pred_ego": ego_preds,
            "attn": None
        }
```

Sau đó trong `models/__init__.py`, import file này để đăng ký. Khi chạy:
```bash
python scripts/train.py --model_name pure_transformer
```

---

## 4. Cơ Chế Bộ Nhớ & Distributed (DDP / Accelerate)

### Quản lý Bộ nhớ & Tránh OOM (SIGKILL -9)
Kaggle và các cụm GPU thường có giới hạn RAM hệ thống (16-30 GB). Để tránh tràn RAM:
1. `core/engine.py` tự động kích hoạt `gc.collect()` và `torch.cuda.empty_cache()` sau mỗi epoch.
2. `utils/tracker.py`: Hàm `compute_summary(clear_buffers=True)` tự động xoá sạch các mảng lưu trữ tạm thời sau khi tính mAP.
3. Tham số `--num_workers` mặc định là `2`. Nếu chạy multi-GPU, không nên tăng quá `4` vì $2 \text{ GPUs} \times 4 \text{ workers} = 8$ tiến trình đọc LMDB đồng thời.

### Chế độ Multi-GPU (DDP Safe)
- `DistributedDataParallelKwargs(find_unused_parameters=True)` đã được bật sẵn trong `scripts/train.py`, đảm bảo an toàn ngay cả khi một số layer không tham gia vào backward pass.
- Các tác vụ ghi đĩa (WandB, lưu Checkpoint `.pth`, ghi Excel `training_full_metrics.xlsx`) được bọc bởi `if self.is_main:`, đảm bảo chỉ duy nhất tiến trình chính (`rank 0`) thực thi, không gây xung đột lock file.
- Tính toán mAP validation được đồng bộ tự động trên toàn bộ các GPU qua `accelerator.gather_for_metrics`.

---

## 5. Danh Sách Lệnh Thường Dùng (Cheatsheet)

### Huấn luyện Multi-GPU với Accelerate (Kaggle 2x T4):
```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  scripts/train.py \
  --model_name action_slot \
  --lmdb_train_path "/kaggle/input/datasets/royalb26/taco-dataset" \
  --lmdb_val_path "/kaggle/input/datasets/royalb26/taco-dataset" \
  --img_height 256 \
  --img_width 768 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --num_workers 2 \
  --amp \
  --clip_grad_norm 1.0 \
  --diagnose \
  --diagnose_every_steps 500 \
  --plot_attn \
  --epochs 100 \
  --lr 5e-4 \
  --wandb \
  --accelerate
```

### Đánh giá Checkpoint (Validation):
```bash
python scripts/eval.py \
  --model_name action_slot \
  --model_path checkpoints/best_model.pth \
  --lmdb_val_path "/kaggle/input/datasets/royalb26/taco-dataset" \
  --img_height 256 \
  --img_width 768 \
  --batch_size 8
```

### Chạy suy luận tạo file kết quả (Inference / Test):
```bash
python scripts/test.py \
  --model_name action_slot \
  --model_path checkpoints/best_model.pth \
  --save_path results/test_preds.pkl
```
