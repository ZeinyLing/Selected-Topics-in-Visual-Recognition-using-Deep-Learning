
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

Hyperparameter:
- `Batch size`: 32
- `Epochs`: 60
- `Optimizer`: SGD
- `Learning rate`: 0.01
- `Momentum`: 0.9
- `Weight decay`: 5e-4
- `Learning rate scheduler`: Cosine Annealing LR
- `Loss function`: CrossEntropyLoss
- `Label smoothing`: 0.05

### Inference

```bash
python inference.py 
```
### Soft Voting

```bash
python inference.py 
```

### Hard Voting

```bash
python inference.py 
```
Hyperparameter:
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
3. **Label Smoothing**: Label smoothing is applied in the loss function to improve decision boundary learning and reduce overconfidence.
4. **Learning Rate Scheduler**: A cosine learning rate scheduler is used to gradually decrease the learning rate, leading to more stable convergence.
5. **Test-Time Augmentation (TTA)**: During inference, horizontal flipping is applied and predictions are averaged to improve stability and accuracy.
6. **Model Ensemble**: Three models with the same architecture but different random seeds are trained and ensembled. Both hard voting and soft voting methods are explored to further improve overall classification performance.
The implementation uses mixed precision training for efficiency and includes early stopping to prevent overfitting.

## Performance

- Parameters: 24.4M (within competition constraint of 100M)
- Parameters: 73.2M (within competition constraint of 100M)
- Validation accuracy: 0.9133
- Public test data accuracy: 0.96


## Performance snapshot

![image](https://github.com/user-attachments/assets/9b3865ff-0032-469e-8676-e21e3fb029fc)

