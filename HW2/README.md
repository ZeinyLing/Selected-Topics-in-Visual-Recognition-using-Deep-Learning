
# NYCU  Computer Vision 2026 HW2

* **Student ID:** 314551087
* **Name:** 黃奕睿


## Introduction

The dataset consists of RGB images containing multiple digits with variations in scale, orientation, background, and illumination, making detection challenging. It includes 30,062 training images, 3,340 validation images, and 13,068 test images, where each image may contain multiple digits from different classes, requiring strong multi-object detection capability. The annotations follow the COCO-style JSON format, with each object represented by a bounding box [xmin, ymin, w, h] and a class label, where coordinates are in pixel values and category_id starts from 1. The assignment requires using Detection Transformer (DETR) with a ResNet-50 backbone as the primary model, and further incorporates Deformable DETR and Conditional DETR to address limitations such as slow convergence and weaker performance on small object detection.


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
├── train_conditional.py       # Training and validation conditional detr
├── train_deformable.py        # Training and validation deformable detr
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

### Conditional DETR
```bash
python train_conditional.py
```
### Inference

```bash
python inference_conditional.py
```

### Deformable DETR
```bash
python train_deformable.py
```

### Inference

```bash
python inference_deformable.py
```

## Performance

- Validation COCO mAP@0.5:0.95 : 0.9133
- Public test data COCO mAP@0.5:0.95 : 0.4

## Performance snapshot

![image](https://github.com/ZeinyLing/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/imgs/score.png)

