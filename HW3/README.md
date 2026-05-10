
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
├── inference_conditional.py   # inference conditional detr
├── inference_deformable.py    # inference deformable detr
├── inference_detr.py          # inference detr
├── inference_detr.py          # inference detr
├── train_conditional.py       # Training and validation conditional detr
├── train_deformable.py        # Training and validation deformable detr
├── train_detr.py              # Training and validation detr
├── train_detr.py              # Training and validation detr
├── requirements.txt           # Project dependencies
└── data/                      # Dataset directory
```

## Usage

### Training

```bash
python train_detr.py 
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
- `Batch size`: 2
- `Epochs`: 200
- `Optimizer`: AdamW
- `Learning rate`: 1e-5
- `Transformer + Heads LR`: 1e-4
- `Weight decay`: 1e-4
- `Learning rate scheduler`: Linear
- `Loss function`: Classification loss + Box regression loss + Box overlap loss
- `Gradient clipping`: 0.1

### Inference

```bash
python inference_detr.py
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

- Public test data COCO mAP@0.5:0.95 : 0.4

| Model | Best Val mAP50:90 | Scores |
|------|------|------|
| DETR | 0.3688 | 0.34 |
|DETR (7 layers) | 0.3596| 0.33 |
| Conditional DETR | 0.4206 | 0.38 |
| Deformable DETR | 0.4809 | 0.4 |

## Performance snapshot

![image](https://github.com/ZeinyLing/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/imgs/score.png)

