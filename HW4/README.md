
# NYCU  Computer Vision 2026 HW4

* **Student ID:** 314551087
* **Name:** 黃奕睿


## Introduction

This project focuses on instance segmentation for colored medical images. The goal is to detect, segment, and classify individual cells into four categories: class1–class4. Since cells may touch, overlap, or appear densely distributed, the model must separate each cell instance accurately.
The dataset contains 209 training/validation images and 101 test images in .tif format. Because the dataset is small, preprocessing, data augmentation, and model design are important for improving generalization.
The evaluation metric is AP50, which measures mask prediction quality at an IoU threshold of 0.5. Therefore, the model needs good classification, localization, and mask segmentation ability.
In this work, Mask R-CNN is used as the baseline. We compare it with CBAM Mask R-CNN, Cascade Mask R-CNN, PointRend Mask R-CNN, and PointRend Cascade Mask R-CNN. These models are evaluated to find the most suitable architecture under the constraints of no external data, pure vision-based models, and fewer than 200M trainable parameters.


## Environment Setup

### Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
.
├── cbam_inference.py                       # inference cbam maskrcnn
├── cascade_inference.py                    # inference cascade maskrcnn
├── maskrcnn_inference.py                   # inference maskrcnn
├── pointrend_cascade_inference.py          # inference pointrend cascade maskrcnn
├── cbam_train.py                           # Training and validation cbam maskrcnn
├── cascade_train.py                        # Training and validation cascade maskrcnn
├── maskrcnn_train.py                       # Training and validation maskrcnn
├── pointrend_cascade_train.py              # Training and validation pointrend cascade maskrcnn
├── requirements.txt                        # Project dependencies
└── data/                                   # Dataset directory
```

## Usage

### Training

```bash
python maskrcnn_train.py 
```

### Configuration
```bash
# DATA PATH
DATA_ROOT = "./hw3-data-release"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
OUTPUT_DIR = "./outputs_maskrcnn_train"
CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
```
Hyperparameter:
- `Image Size`: 256 × 256
- `Epochs`: 100
- `Batch Size`: 8
- `Learning Rate`: 1e-4
- `Weight Decay`: 1e-4
- `Validation Ratio`: 0.1
- `Optimizer`: AdamW
- `Scheduler`: CosineAnnealingLR
- `Loss Function`: Charbonnier Loss

### Inference

```bash
python maskrcnn_inference.py
```
## Strategy and Adjustments

The following modifications and strategies are applied in the model and training process:

1. Apply horizontal flip and vertical flip for data augmentation. 
2. Update masks and bounding boxes after image flipping. 
3. Use ImageNet pretrained weights to improve feature extraction. 
4. Train the model with AdamW optimizer.
5. Use CosineAnnealingLR to adjust the learning rate.

## Additional experiments

### cbam maskrcnn
```bash
python cbam_train.py                  # Train  

python cbam_inference.py              # inference
```

### cascade maskrcnn
```bash
python cascade_train.py               # Train

python cascade_inference.py           # inference
```

###  pointrend cascade maskrcnn
```bash
python pointrend_cascade_train.py      # Train

python pointrend_cascade_inference.py  # inference
```

## Performance

- Public test data PSNR : 30.90

| Model | Best Val PSNR | Scores |
|------|------|------|
| PromptIR | 27.963 | 29.75 |
| PromptIR V2 | 28.455| 30.27 |
| PromptIR V3 | 29.799 | 30.69 |
| PromptIR V4 | 29.860 | 30.90 |

## Performance snapshot
<img src="img/pubscore.png" width="1000">
