# NYCU Computer Vision 2026 HW1

* **Student ID:** 314551087
* **Name:** 黃奕睿

## Introduction
This Homework focuses on a multi-class image classification task, where the objective is to predict the corresponding object category from input RGB images. The dataset consists of 100 classes, including natural objects such as plants, insects, and birds, with 21,024 images for training and validation, and 2,344 images for testing. Due to the presence of visually similar categories, complex backgrounds, and significant variations in object scale, the model is required not only to extract discriminative features effectively but also to maintain strong generalization ability in order to achieve reliable performance on unseen test data.
In Homework, the model design must satisfy several constraints. First, ResNet must be used as the primary backbone, although modifications to its structure are allowed with proper justification. Second, pretrained weights are permitted to leverage transfer learning, which helps accelerate convergence and improve feature representation. In addition, the total number of model parameters must be limited to under 100 million, ensuring a balance between performance and computational efficiency. Therefore, the model design must carefully consider both effectiveness and compliance with these constraints.
Based on these requirements, this work adopts a modified ResNet50 architecture enhanced with Squeeze-and-Excitation (SE) blocks. Specifically, SE modules are inserted after each of the four main residual stages to recalibrate channel-wise feature responses. By applying global average pooling followed by channel-wise weighting, the SE mechanism enables the model to emphasize informative features while suppressing less relevant ones. Finally, the model applies global average pooling, followed by dropout and a fully connected layer to produce predictions over the 100 classes. This design preserves the original ResNet50 structure while enhancing feature representation, aiming to improve overall classification performance


## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
