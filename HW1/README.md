
# NYCU Visual Recognition using Deep Learning 2026 HW1

* **Student ID:** 314551087
* **Name:** 黃奕睿

## Introduction

Based on these requirements, this work adopts a modified ResNet50 architecture enhanced with Squeeze-and-Excitation (SE) blocks. Specifically, SE modules are inserted after each of the four main residual stages to recalibrate channel-wise feature responses. By applying global average pooling followed by channel-wise weighting, the SE mechanism enables the model to emphasize informative features while suppressing less relevant ones. Finally, the model applies global average pooling, followed by dropout and a fully connected layer to produce predictions over the 100 classes.


## Environment Setup

How to install dependence
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

- Validation accuracy: 0.9133
- Public test data accuracy: 0.96
- Parameters: 89.1M (within competition constraint of 100M)

## Performance snapshot

![image](https://github.com/user-attachments/assets/9b3865ff-0032-469e-8676-e21e3fb029fc)

