
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
├── config.py       # Configuration parameters
├── config.py       # Configuration parameters
├── inference.py    # Test prediction code
├── train.py        # Training and validation routines
├── requirements.txt # Project dependencies
└── data/           # Dataset directory
    ├── train/      # Training images (100 classes)
    ├── val/        # Validation images
    └── test/       # Test images
```

## Usage

### Training

```bash
python train.py 
```

Additional training options:
- `--num_epochs 20`: Set number of training epochs (default: 20)
- `--batch_size 10`: Change batch size (default: 10)
- `--learning_rate 1e-5`: Adjust learning rate (default: 1e-5)
- `--criterion focal`: Select loss function, options: "focal" or "cross_entropy" (default: "focal")
- `--nodropout`: Disable dropout (default: dropout enabled with p=0.5)
- `--seed 42`: Set random seed for reproducibility (default: 42)
- `--device cuda`: Select device for training (default: "cuda")
- `--weighted_loss`: Enable class weighting in loss function

### Inference

```bash
python inference.py 
```

Options:
- `--test_data_dir data/test`: Directory containing test images (default: "./data/test")
- `--model_path`: Path to the trained model weights (required)
- `--save_dir ./results`: Directory to save prediction results (default: "./results")
- `--tta`: Enable Test-Time Augmentation for improved accuracy
- `--batch_size 10`: Adjust batch size for inference (default: 10)
- `--nodropout`: Disable dropout (should match training configuration)
- `--device cuda`: Select device for inference (default: "cuda")

## Strategy and Adjustments

The following modifications and strategies are applied in the model and training process:

1. **SE Block**: Squeeze-and-Excitation (SE) blocks are integrated into ResNet50 to enhance the model’s ability to focus on important channel features.
2. **Dropout**: A dropout layer (0.4) is added before the classification layer to reduce the risk of overfitting.
3. **Channel Attention**: A Squeeze-and-Excitation module with reduction ratio 16 recalibrates feature importance
4. **Classification Head**: A classifier with optional dropout (p=0.5) produces the final prediction across 100 classes

The implementation uses mixed precision training for efficiency and includes early stopping to prevent overfitting.

## Performance snapshot

- Validation COCO mAP@0.5:0.95 : 0.9133
- Public test data COCO mAP@0.5:0.95 : 0.4

## Performance snapshot

![image](https://github.com/ZeinyLing/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/imgs/score.png)

