# NYCU Computer Vision 2026 HW1

* **Student ID:** 314551087
* **Name:** 黃奕睿

## Introduction
This HW focuses on a multi-class image classification task, aiming to predict the correct category from RGB images. The dataset includes 100 classes (e.g., plants, insects, and birds), with 21,024 training/validation images and 2,344 test images. Due to visual similarity and complex backgrounds, the model must learn strong features and generalize well.

The model must use ResNet as the backbone, allow pretrained weights, and keep parameters under 100M. Therefore, the design must balance performance and efficiency.

This work adopts a modified ResNet50 with Squeeze-and-Excitation (SE) blocks. SE modules are added after each stage to enhance important features. The final predictions are produced using global pooling, dropout, and a fully connected layer, improving classification performance while maintaining the original backbone structure.


## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
