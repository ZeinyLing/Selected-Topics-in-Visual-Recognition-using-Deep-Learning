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

### Submission Encoding

After generating the super-resolution image, the result is converted from RGB to BGR and encoded using the official Kaggle-style RLE + zlib + base64 format. The final output is saved as a CSV file with the following columns:

```text
id, filename, rle
```

---

## 5. Voting / Ensemble Inference (`voting.py`)

The voting script improves inference by combining predictions from multiple trained checkpoints. This can reduce the effect of individual model errors and produce more stable results.

### Ensemble Checkpoints

The current script uses two checkpoints:

| Checkpoint | Weight |
|---|---:|
| `./outputs_super_image_drln_retrain_vnew/28.32best.pth` | 0.9 |
| `./outputs_super_image_drln_finetune_v3_from_f26055/best.pth` | 1.0 |

More checkpoints can be added to the `ENSEMBLE_CKPTS` list.

### Fusion Modes

| Fusion mode | Description |
|---|---|
| `mean` | Averages all model predictions equally. |
| `weighted_mean` | Averages predictions according to the checkpoint weights. |
| `median` | Uses pixel-level median fusion, which is more robust to outlier predictions. |

The current setting uses:

```text
FUSION_MODE = "mean"
```

This means each model prediction contributes equally to the final output.

### Why Ensemble Helps

Single-model inference may produce unstable details in some images. Ensemble inference combines multiple models or checkpoints, which can smooth unstable predictions and improve visual consistency. In super-resolution tasks, this is especially useful because different checkpoints may reconstruct textures and edges slightly differently.

---

## 6. Overall Workflow

```text
1. Prepare LR-HR training image pairs
        ↓
2. Train DRLN model with patch-based supervised learning
        ↓
3. Save best checkpoint according to validation PSNR
        ↓
4. Run single-model inference or ensemble inference
        ↓
5. Apply tile inference and x8 TTA
        ↓
6. Encode predictions into Kaggle submission CSV
```

---

## 7. Key Techniques

### Patch-Based Training

Patch-based training allows the model to learn local textures and edges while reducing memory consumption. It also increases the number of training samples because multiple random crops can be generated from the same image.

### Charbonnier Loss

Charbonnier Loss is a smooth version of L1 Loss. It is commonly used in low-level vision tasks because it is less sensitive to small pixel-level noise and provides stable training.

### EMA

Exponential Moving Average keeps a smoothed version of the model weights during training. The EMA model is often more stable than the raw model and can improve validation PSNR.

### Tile Inference

Tile inference makes it possible to process large images with limited GPU memory. Overlapping tiles help reduce visible seams in the final reconstructed image.

### TTA and Ensemble

TTA improves prediction stability by averaging outputs from transformed versions of the same input. Ensemble inference further improves robustness by averaging predictions from multiple checkpoints.

---

## 8. Conclusion

This project builds a complete super-resolution pipeline from training to Kaggle submission generation. The training script uses DRLN with patch-based learning, data augmentation, mixed precision, EMA, and a combined L1 + Charbonnier loss. The inference scripts further improve output quality through tile inference, x8 test-time augmentation, and optional ensemble voting.

Overall, the design focuses on improving **PSNR**, reducing inference artifacts, and producing stable high-resolution predictions suitable for competition submission.
