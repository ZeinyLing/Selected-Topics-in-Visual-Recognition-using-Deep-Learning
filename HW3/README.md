
# NYCU  Computer Vision 2026 HW3

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
TRAIN_IMG_DIR = "./dataset/train"
TRAIN_JSON = "./dataset/annotations/train.json"
VAL_IMG_DIR = "./dataset/val"
VAL_JSON = "./dataset/annotations/val.json"
```
Hyperparameter:
- `Optimizer`: AdamW
- `Weight Decay`: 1e-4
- `Learning rate`: 1e-4
- `Scheduler`: CosineAnnealingLR
- `Batch Size`: 2
- `Epochs`: 40
- `Validation Ratio`: 0.15
- `Mask Threshold`: 0.5
- `Evaluation Score Threshold`: 0.05
- `Minimum Instance Area`: 8

### Inference

```bash
python maskrcnn_inference.py
```
## Strategy and Adjustments

The following modifications and strategies are applied in the model and training process:

1. **Differential Learning Rates**:A smaller learning rate is applied to the pretrained backbone (1e-5), and a larger learning rate is used for the Transformer and prediction heads (1e-4). 
2. **Data Cleaning and Validation**: Invalid and extremely small bounding boxes are removed, and all annotations are clamped within image boundaries to prevent numerical instability. 
3. **Custom Collate Function**: Manual padding and pixel mask generation are implemented to handle variable image sizes and ensure compatibility across library versions. 
4. **Training Stability Techniques**: Gradient clipping (0.1) and disabling mixed precision (fp16) are used to avoid unstable training and NaN issues.

## Additional experiments

### Training Conditional DETR
```bash
python train_conditional.py
```
### Inference Conditional DETR

```bash
python inference_conditional.py
```

### Training Deformable DETR
```bash
python train_deformable.py
```

### Inference Deformable DETR

```bash
python inference_deformable.py
```

## Performance

- Public test data AP50 :  0.5605

| Model | Best Val AP50 | Scores |
|------|------|------|
| Mask R-CNN | 0.4522 | 0.3300 |
| Cbam Mask R-CNN | 0.4726| 0.3579 |
| Cascade Mask R-CNN | 0.4944 | 0.3893 |
| PointRend Cascade Mask R-CNN | 0.7736 | 0.5605 |

## Performance snapshot

![image](https://github.com/ZeinyLing/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/imgs/score.png)

