# Super-Resolution Project Introduction

## 1. Project Overview

This project focuses on **image super-resolution**, where the goal is to reconstruct high-resolution (HR) images from low-resolution (LR) inputs. The implementation is designed for a Kaggle-style super-resolution task and includes three major parts:

1. **Training script**: fine-tunes or retrains a super-resolution model using LR-HR image pairs.
2. **Single-model inference script**: generates super-resolution predictions from one trained checkpoint.
3. **Voting / Ensemble inference script**: combines multiple checkpoints to improve prediction stability and final output quality.

The project mainly uses models from the `super_image` library, including **MSRN**, **EDSR**, and **DRLN**. In the current setting, **DRLN** is used as the main model backbone.

---

## 2. File Structure

| File | Purpose |
|---|---|
| `train3.py` | Trains or retrains a super-resolution model using paired LR-HR images. |
| `inf3.py` | Performs single-model inference and generates a Kaggle submission CSV. |
| `voting.py` | Performs ensemble / voting inference using multiple checkpoints. |

---

## 3. Training Pipeline (`train3.py`)

The training script uses paired low-resolution and high-resolution images for supervised learning. The LR images are loaded from `./data_sr/train/lr`, while the HR images are loaded from `./data_sr/train/hr`.

### Main Training Settings

| Setting | Value |
|---|---:|
| Model | DRLN |
| Scale factor | ×4 |
| Patch size | 128 |
| Epochs | 150 |
| Batch size | 8 |
| Learning rate | 9e-5 |
| Validation ratio | 0.02 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| AMP | Enabled |
| EMA | Enabled |
| EMA decay | 0.999 |
| Loss | 0.5 × L1 + 0.5 × Charbonnier |

### Training Strategy

The training process applies **patch-based supervised learning**. Instead of feeding full images directly, the script randomly crops LR patches and matches them with the corresponding HR patches. This reduces GPU memory usage and increases the diversity of training samples.

To further improve robustness, the script applies simple data augmentation, including horizontal flip, vertical flip, and 90-degree rotation. These operations help the model learn more general image structures and reduce overfitting.

The model is optimized using a combined loss function:

```text
Loss = 0.5 × L1 Loss + 0.5 × Charbonnier Loss
```

L1 Loss helps preserve pixel-level accuracy, while Charbonnier Loss provides a smoother and more stable optimization objective for image restoration tasks.

### Model Saving

During training, the script saves:

- `last.pth`: the latest checkpoint.
- `best.pth`: the checkpoint with the highest validation PSNR.
- `train_log.csv`: the training log, including loss, L1 loss, Charbonnier loss, validation PSNR, and best PSNR.

The best model is selected based on **validation PSNR**.

---

## 4. Single-Model Inference (`inf3.py`)

The inference script loads one trained checkpoint and applies it to the test LR images. The output is converted into the required Kaggle submission format.

### Main Inference Settings

| Setting | Value |
|---|---:|
| Test LR directory | `./data_sr/test/lr` |
| Sample submission | `./data_sr/sample_submission.csv` |
| Checkpoint | `./outputs_super_image_drln_retrain_vnew/28.32best.pth` |
| Output CSV | `./n28.32submission_tta_x8_64.csv` |
| Tile inference | Enabled |
| Tile size | 256 |
| Tile overlap | 64 |
| TTA | Enabled |
| TTA mode | x8 |

### Tile Inference

Tile inference divides a large image into smaller patches before feeding it into the model. This is useful when the image is too large to process at once due to GPU memory limitations.

The script also uses overlapping tiles. The overlapping regions are averaged to reduce boundary artifacts between neighboring tiles.

### Test-Time Augmentation

The script supports TTA modes:

| Mode | Description |
|---|---|
| `none` | Direct inference only. |
| `x4` | Original image + horizontal flip + vertical flip + both flips. |
| `x8` | x4 TTA plus transpose-based augmentations. |

The current setting uses **x8 TTA**, which is slower but usually produces more stable predictions.


### Fusion Modes

| Fusion mode | Description |
|---|---|
| `mean` | Averages all model predictions equally. |
| `weighted_mean` | Averages predictions according to the checkpoint weights. |
| `median` | Uses pixel-level median fusion, which is more robust to outlier predictions. |


