# -*- coding: utf-8 -*-
"""
Torchvision-only Cascade-like PointRend Mask R-CNN

IMPORTANT:
- This is NOT the official Cascade Mask R-CNN / official PointRend.
- True Cascade Mask R-CNN + official PointRend head usually needs Detectron2 or MMDetection.
- This script avoids Detectron2 and keeps your original torchvision pipeline.

What is improved from your original code:
1. ResNet101-FPN backbone instead of ResNet50-FPN.
2. Small-object anchors are preserved.
3. Dice + PointRend-like uncertain-point mask loss is preserved.
4. AMP mixed precision training.
5. Cosine LR scheduler.
6. Gradient clipping.
7. Multi-scale / flip TTA inference.
8. Class-wise mask NMS after TTA to reduce duplicate masks.
9. Validation AP50 + per-class AP50.
10. Save best model, loss curve, AP50 curve, LR curve, history.json.
11. Output COCO-style RLE submission.json.

Dataset format:
hw3-data-release/
    train/
        sample_xxx/
            image.tif
            class1.tif
            class2.tif
            class3.tif
            class4.tif
    test_release/
        xxx.tif
    test_image_name_to_ids.json

Class rule:
- Background = 0
- class1 = 1
- class2 = 2
- class3 = 3
- class4 = 4
"""

import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import roi_heads as roi_heads_module
from torchvision.models.detection.roi_heads import project_masks_on_boxes
from torchvision.models.detection.anchor_utils import AnchorGenerator

from pycocotools import mask as mask_utils


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "./hw3-data-release"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

OUTPUT_DIR = "./outputs_torchvision_cascade_like_pointrend"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_CKPT_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")
SUBMISSION_JSON = os.path.join(OUTPUT_DIR, "submission.json")

LOSS_CURVE_PNG = os.path.join(OUTPUT_DIR, "loss_curve.png")
AP50_CURVE_PNG = os.path.join(OUTPUT_DIR, "ap50_curve.png")
LR_CURVE_PNG = os.path.join(OUTPUT_DIR, "lr_curve.png")
HISTORY_JSON = os.path.join(OUTPUT_DIR, "history.json")

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# background + 4 classes
NUM_CLASSES = 5

EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
MIN_INSTANCE_AREA = 8
GRAD_CLIP_NORM = 5.0
USE_AMP = True

# inference / validation thresholds
SCORE_THRESH = 0.30
MASK_THRESH = 0.50
MASK_NMS_IOU = 0.50

# PointRend-like refinement settings
POINT_REFINEMENT_ENABLED = True
POINT_NUM_POINTS = 2048
POINT_REFINE_WEIGHT = 0.15
DICE_WEIGHT = 1.0

# augmentation
ENABLE_AUG = True
CROP_PROB = 0.4
SCALE_PROB = 0.5
BLUR_PROB = 0.2
NOISE_PROB = 0.25
COLOR_PROB = 0.5
CUTOUT_PROB = 0.15

# TTA
ENABLE_TTA = True
# 1.25 may improve AP but uses more memory/time. If OOM, change to [1.0].
TTA_SCALES = [1.0, 1.25]
TTA_FLIPS = ["none", "h", "v"]

# Model
BOX_DETECTIONS_PER_IMG = 800
RPN_POST_NMS_TOP_N_TRAIN = 1500
RPN_POST_NMS_TOP_N_TEST = 1500
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 2000


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic makes training slower but reproducible.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================================================
# IO
# =========================================================
def read_rgb_tif(path):
    try:
        arr = tifffile.imread(path)
    except Exception:
        try:
            arr = np.array(Image.open(path))
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Failed to read image tif: {path}") from e

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in [1, 3] and arr.shape[-1] not in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr.astype(np.uint8)


def read_mask_tif(path):
    try:
        arr = tifffile.imread(path)
    except Exception:
        try:
            arr = np.array(Image.open(path))
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Failed to read mask tif: {path}") from e

    if arr.ndim > 2:
        arr = np.squeeze(arr)
    return arr


def collate_fn(batch):
    return tuple(zip(*batch))


def binary_mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": rle["counts"],
    }


# =========================================================
# AUGMENTATION HELPERS
# =========================================================
def masks_to_boxes_np(masks: np.ndarray):
    if len(masks) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 1, 1), dtype=np.uint8),
        )

    boxes = []
    keep_masks = []

    for m in masks:
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        if m.sum() < MIN_INSTANCE_AREA:
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        keep_masks.append(m.astype(np.uint8))

    if len(boxes) == 0:
        h, w = masks.shape[1], masks.shape[2]
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, h, w), dtype=np.uint8),
        )

    return np.array(boxes, dtype=np.float32), np.stack(keep_masks).astype(np.uint8)


def sanitize_target(target):
    boxes = target["boxes"]
    labels = target["labels"]
    masks = target["masks"]

    if boxes.numel() == 0:
        return target

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 1e-6) & (heights > 1e-6)

    if masks.numel() > 0:
        areas = masks.flatten(1).sum(dim=1)
        valid = valid & (areas >= MIN_INSTANCE_AREA)

    target["boxes"] = boxes[valid]
    target["labels"] = labels[valid]
    target["masks"] = masks[valid]
    return target


def apply_photometric_aug(image):
    img = image.astype(np.float32)

    if random.random() < COLOR_PROB:
        alpha = random.uniform(0.85, 1.15)
        img = (img - 127.5) * alpha + 127.5

    if random.random() < COLOR_PROB:
        beta = random.uniform(-15, 15)
        img = img + beta

    if random.random() < NOISE_PROB:
        noise = np.random.normal(0, 5, img.shape)
        img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)

    if random.random() < BLUR_PROB:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if random.random() < CUTOUT_PROB:
        h, w = img.shape[:2]
        cut_h = random.randint(max(4, h // 20), max(8, h // 10))
        cut_w = random.randint(max(4, w // 20), max(8, w // 10))
        y0 = random.randint(0, max(0, h - cut_h))
        x0 = random.randint(0, max(0, w - cut_w))
        img[y0:y0 + cut_h, x0:x0 + cut_w] = random.randint(0, 255)

    return img


def safe_random_crop(image, masks, boxes, labels, min_keep=1):
    h, w = image.shape[:2]
    crop_scale = random.uniform(0.75, 1.0)
    ch = int(h * crop_scale)
    cw = int(w * crop_scale)

    if ch >= h or cw >= w:
        return image, masks, boxes, labels

    y0 = random.randint(0, h - ch)
    x0 = random.randint(0, w - cw)
    y1 = y0 + ch
    x1 = x0 + cw

    cropped_img = image[y0:y1, x0:x1]

    if len(masks) == 0:
        return cropped_img, masks, boxes, labels

    cropped_masks = masks[:, y0:y1, x0:x1]

    new_boxes = []
    new_masks = []
    keep_labels = []

    for m, lab in zip(cropped_masks, labels):
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        if m.sum() < MIN_INSTANCE_AREA:
            continue

        x_min, y_min, x_max, y_max = xs.min(), ys.min(), xs.max(), ys.max()
        if x_max <= x_min or y_max <= y_min:
            continue

        new_boxes.append([x_min, y_min, x_max, y_max])
        new_masks.append(m.astype(np.uint8))
        keep_labels.append(lab)

    if len(new_boxes) < min_keep:
        return image, masks, boxes, labels

    return (
        cropped_img,
        np.stack(new_masks).astype(np.uint8),
        np.array(new_boxes, dtype=np.float32),
        np.array(keep_labels, dtype=np.int64),
    )


def safe_random_scale(image, masks, boxes, labels):
    scale = random.uniform(0.85, 1.15)
    h, w = image.shape[:2]
    nh, nw = max(32, int(h * scale)), max(32, int(w * scale))

    image_scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if len(masks) == 0:
        return image_scaled, masks, boxes, labels

    new_boxes = []
    new_masks = []
    keep_labels = []

    for m, lab in zip(masks, labels):
        ms = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(ms > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        if ms.sum() < MIN_INSTANCE_AREA:
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        if x2 <= x1 or y2 <= y1:
            continue

        new_boxes.append([x1, y1, x2, y2])
        new_masks.append(ms.astype(np.uint8))
        keep_labels.append(lab)

    if len(new_boxes) == 0:
        return (
            image_scaled,
            np.zeros((0, nh, nw), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        image_scaled,
        np.stack(new_masks).astype(np.uint8),
        np.array(new_boxes, dtype=np.float32),
        np.array(keep_labels, dtype=np.int64),
    )


def apply_geometric_aug(image, target):
    h, w = image.shape[:2]

    boxes = (
        target["boxes"].numpy()
        if target["boxes"].numel() > 0
        else np.zeros((0, 4), dtype=np.float32)
    )
    labels = (
        target["labels"].numpy()
        if target["labels"].numel() > 0
        else np.zeros((0,), dtype=np.int64)
    )
    masks = (
        target["masks"].numpy()
        if target["masks"].numel() > 0
        else np.zeros((0, h, w), dtype=np.uint8)
    )

    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        if len(masks) > 0:
            masks = np.ascontiguousarray(masks[:, :, ::-1])

    if random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        if len(masks) > 0:
            masks = np.ascontiguousarray(masks[:, ::-1, :])

    k = random.choice([0, 1, 2, 3])
    if k > 0:
        image = np.rot90(image, k).copy()
        if len(masks) > 0:
            masks = np.rot90(masks, k, axes=(1, 2)).copy()

    if len(masks) > 0:
        new_boxes = []
        new_masks = []
        new_labels = []
        for m, lab in zip(masks, labels):
            ys, xs = np.where(m > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            if m.sum() < MIN_INSTANCE_AREA:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 <= x1 or y2 <= y1:
                continue
            new_boxes.append([x1, y1, x2, y2])
            new_masks.append(m.astype(np.uint8))
            new_labels.append(lab)

        if len(new_boxes) == 0:
            h2, w2 = image.shape[:2]
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, h2, w2), dtype=np.uint8)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(new_boxes, dtype=np.float32)
            masks = np.stack(new_masks).astype(np.uint8)
            labels = np.array(new_labels, dtype=np.int64)

    if random.random() < SCALE_PROB:
        image, masks, boxes, labels = safe_random_scale(image, masks, boxes, labels)

    if random.random() < CROP_PROB:
        image, masks, boxes, labels = safe_random_crop(image, masks, boxes, labels)

    target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
    target["labels"] = torch.tensor(labels, dtype=torch.int64)

    h2, w2 = image.shape[:2]
    if len(masks) == 0:
        target["masks"] = torch.zeros((0, h2, w2), dtype=torch.uint8)
    else:
        target["masks"] = torch.tensor(masks, dtype=torch.uint8)

    return image, target


# =========================================================
# DATASET
# =========================================================
class NucleiDataset(Dataset):
    def __init__(self, dirs, training=False):
        self.dirs = dirs
        self.training = training

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        d = self.dirs[idx]

        image = read_rgb_tif(os.path.join(d, "image.tif"))
        h, w = image.shape[:2]

        boxes, labels, masks = [], [], []

        for cls in range(1, 5):
            p = os.path.join(d, f"class{cls}.tif")
            if not os.path.exists(p):
                continue

            inst = read_mask_tif(p)
            ids = np.unique(inst)
            ids = ids[ids > 0]

            for i in ids:
                m = (inst == i).astype(np.uint8)
                ys, xs = np.where(m > 0)

                if len(xs) == 0 or m.sum() < MIN_INSTANCE_AREA:
                    continue

                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
                masks.append(m)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
            }

        if self.training and ENABLE_AUG:
            image = apply_photometric_aug(image)
            image, target = apply_geometric_aug(image, target)

        image = TF.to_tensor(image)
        target = sanitize_target(target)

        return image, target


class NucleiTestDataset(Dataset):
    def __init__(self, test_root):
        self.paths = sorted(list(Path(test_root).glob("*.tif")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = str(self.paths[idx])
        image = read_rgb_tif(p)
        tensor = TF.to_tensor(image)
        return tensor, os.path.basename(p), image.shape[0], image.shape[1]


# =========================================================
# DICE + POINTREND-LIKE MASK LOSS PATCH
# =========================================================
def dice_loss(pred, gt):
    p = torch.sigmoid(pred).flatten(1)
    g = gt.flatten(1).float()

    inter = (p * g).sum(1)
    union = p.sum(1) + g.sum(1)

    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def pointrend_boundary_loss(pred, gt, num_points=1024):
    n, h, w = pred.shape
    if n == 0:
        return pred.sum() * 0

    probs = torch.sigmoid(pred)
    uncertainty = -torch.abs(probs - 0.5).reshape(n, -1)

    p = min(num_points, h * w)
    idx = torch.topk(uncertainty, k=p, dim=1).indices

    pred_flat = pred.reshape(n, -1)
    gt_flat = gt.float().reshape(n, -1)

    pred_pts = torch.gather(pred_flat, 1, idx)
    gt_pts = torch.gather(gt_flat, 1, idx)

    return F.binary_cross_entropy_with_logits(pred_pts, gt_pts)


def mask_loss(mask_logits, proposals, gt_masks, gt_labels, idxs):
    size = mask_logits.shape[-1]

    labels = [gt_label[i] for gt_label, i in zip(gt_labels, idxs)]
    targets = [
        project_masks_on_boxes(m, p, i, size)
        for m, p, i in zip(gt_masks, proposals, idxs)
    ]

    labels = torch.cat(labels)
    targets = torch.cat(targets)

    if targets.numel() == 0:
        return mask_logits.sum() * 0

    pred = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]

    bce = F.binary_cross_entropy_with_logits(pred, targets)
    dloss = dice_loss(pred, targets)

    if POINT_REFINEMENT_ENABLED:
        ploss = pointrend_boundary_loss(
            pred,
            targets,
            num_points=min(POINT_NUM_POINTS, pred.shape[-1] * pred.shape[-2])
        )
    else:
        ploss = pred.sum() * 0

    return bce + DICE_WEIGHT * dloss + POINT_REFINE_WEIGHT * ploss


roi_heads_module.maskrcnn_loss = mask_loss


# =========================================================
# MODEL
# =========================================================
class CascadeLikePointRendMaskRCNN(nn.Module):
    """
    Torchvision-only stronger Mask R-CNN.

    Not true Cascade Mask R-CNN.
    Real Cascade needs multiple ROI box heads with progressive IoU thresholds.
    Torchvision does not provide this directly.

    This version is a practical no-Detectron2 alternative:
    - ResNet101-FPN backbone
    - high proposal count
    - small anchors
    - Dice + uncertain-point mask loss
    - multi-scale TTA + mask NMS inference
    """

    def __init__(self):
        super().__init__()

        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=ResNet101_Weights.IMAGENET1K_V2,
            trainable_layers=5,
        )

        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        model = MaskRCNN(
            backbone=backbone,
            num_classes=NUM_CLASSES,
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=BOX_DETECTIONS_PER_IMG,
            rpn_pre_nms_top_n_train=RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
        )

        model.rpn.nms_thresh = 0.7

        # Replace predictors explicitly, keeps compatibility with custom backbone variants.
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)

        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)

        # In eval, use a lower internal score threshold.
        # Final threshold is handled by SCORE_THRESH.
        model.roi_heads.score_thresh = 0.05
        model.roi_heads.nms_thresh = 0.5
        model.roi_heads.detections_per_img = BOX_DETECTIONS_PER_IMG

        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)


# =========================================================
# AP50 METRIC
# =========================================================
def compute_mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


@torch.no_grad()
def collect_val_predictions(model, loader):
    model.eval()
    cache = []

    for imgs, targets in tqdm(loader, desc="CollectVal", leave=False):
        imgs = [i.to(DEVICE) for i in imgs]
        outputs = model(imgs)

        for out, tgt in zip(outputs, targets):
            cache.append({
                "gt_masks": tgt["masks"].cpu().numpy(),
                "gt_labels": tgt["labels"].cpu().numpy(),
                "pred_scores": out["scores"].detach().cpu().numpy(),
                "pred_labels": out["labels"].detach().cpu().numpy(),
                "pred_masks": out["masks"].detach().cpu().numpy(),
            })

    return cache


def evaluate_ap50_from_cache(
    cache,
    score_thresh=SCORE_THRESH,
    mask_thresh=MASK_THRESH,
    min_area=MIN_INSTANCE_AREA,
    iou_thresh=0.5,
):
    all_gt = {c: 0 for c in range(1, NUM_CLASSES)}
    pred_records = {c: [] for c in range(1, NUM_CLASSES)}

    for item in cache:
        gt_masks = item["gt_masks"] if len(item["gt_masks"]) > 0 else np.zeros((0, 1, 1), dtype=np.uint8)
        gt_labels = item["gt_labels"] if len(item["gt_labels"]) > 0 else np.zeros((0,), dtype=np.int64)
        pred_scores = item["pred_scores"]
        pred_labels = item["pred_labels"]
        pred_masks = item["pred_masks"]

        gt_by_class = {c: gt_masks[gt_labels == c] for c in range(1, NUM_CLASSES)}
        used_gt = {c: np.zeros((len(gt_by_class[c]),), dtype=bool) for c in range(1, NUM_CLASSES)}

        for c in range(1, NUM_CLASSES):
            all_gt[c] += int(np.sum(gt_labels == c))

        order = np.argsort(-pred_scores)
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]
        pred_masks = pred_masks[order]

        for score, label, mask_prob in zip(pred_scores, pred_labels, pred_masks):
            if float(score) < score_thresh:
                continue
            if int(label) < 1 or int(label) >= NUM_CLASSES:
                continue

            pred_mask = (mask_prob[0] >= mask_thresh).astype(np.uint8)
            if int(pred_mask.sum()) < min_area:
                continue

            gt_pool = gt_by_class[int(label)]
            best_iou = 0.0
            best_idx = -1

            for i, gt_mask in enumerate(gt_pool):
                if used_gt[int(label)][i]:
                    continue
                iou = compute_mask_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            is_tp = 0
            if best_iou >= iou_thresh and best_idx >= 0:
                used_gt[int(label)][best_idx] = True
                is_tp = 1

            pred_records[int(label)].append((float(score), is_tp))

    per_class_ap = {}
    ap_list = []

    for c in range(1, NUM_CLASSES):
        n_gt = all_gt[c]
        records = pred_records[c]

        if n_gt == 0:
            per_class_ap[c] = None
            continue

        if len(records) == 0:
            per_class_ap[c] = 0.0
            ap_list.append(0.0)
            continue

        records.sort(key=lambda x: x[0], reverse=True)
        tp = np.array([r[1] for r in records], dtype=np.float32)
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(n_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        idx = np.where(recalls[1:] != recalls[:-1])[0]
        ap = float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))

        per_class_ap[c] = ap
        ap_list.append(ap)

    mean_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    return mean_ap, per_class_ap


# =========================================================
# TTA INFERENCE HELPERS
# =========================================================
@torch.no_grad()
def model_predict_single(model, image_tensor):
    outputs = model([image_tensor.to(DEVICE)])
    return outputs[0]


def resize_tensor_image(image_tensor, scale):
    if abs(scale - 1.0) < 1e-6:
        return image_tensor

    img = image_tensor.unsqueeze(0)
    img = F.interpolate(
        img,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    return img.squeeze(0)


def resize_mask_to_hw(mask, out_h, out_w):
    # mask: numpy [H, W]
    return cv2.resize(
        mask.astype(np.float32),
        (out_w, out_h),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_flip_tensor(image_tensor, flip_type):
    if flip_type == "h":
        return torch.flip(image_tensor, dims=[2])
    if flip_type == "v":
        return torch.flip(image_tensor, dims=[1])
    return image_tensor


def unflip_mask(mask, flip_type):
    if flip_type == "h":
        return mask[:, ::-1].copy()
    if flip_type == "v":
        return mask[::-1, :].copy()
    return mask


def unflip_boxes(boxes, flip_type, h, w):
    if len(boxes) == 0:
        return boxes

    out = boxes.copy()

    if flip_type == "h":
        out[:, 0] = w - 1 - boxes[:, 2]
        out[:, 2] = w - 1 - boxes[:, 0]

    if flip_type == "v":
        out[:, 1] = h - 1 - boxes[:, 3]
        out[:, 3] = h - 1 - boxes[:, 1]

    return out


def apply_tta_predict(model, image_tensor):
    orig_h, orig_w = image_tensor.shape[-2:]
    pred_items = []

    if not ENABLE_TTA:
        scales = [1.0]
        flips = ["none"]
    else:
        scales = TTA_SCALES
        flips = TTA_FLIPS

    for scale in scales:
        scaled_img = resize_tensor_image(image_tensor, scale)
        sh, sw = scaled_img.shape[-2:]

        for flip_type in flips:
            aug_img = apply_flip_tensor(scaled_img, flip_type)
            out = model_predict_single(model, aug_img)

            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            masks = out["masks"].detach().cpu().numpy()

            # Undo boxes in scaled coordinate.
            boxes = unflip_boxes(boxes, flip_type, sh, sw)

            # Scale boxes back to original coordinate.
            if abs(scale - 1.0) > 1e-6 and len(boxes) > 0:
                boxes = boxes / scale

            for box, label, score, mask_prob in zip(boxes, labels, scores, masks):
                if float(score) < SCORE_THRESH:
                    continue
                if int(label) < 1 or int(label) >= NUM_CLASSES:
                    continue

                m = mask_prob[0]
                m = unflip_mask(m, flip_type)

                if abs(scale - 1.0) > 1e-6:
                    m = resize_mask_to_hw(m, orig_h, orig_w)

                binary = (m >= MASK_THRESH).astype(np.uint8)

                if int(binary.sum()) < MIN_INSTANCE_AREA:
                    continue

                pred_items.append({
                    "box": box.astype(np.float32),
                    "label": int(label),
                    "score": float(score),
                    "mask": binary,
                })

    return pred_items


def mask_iou_np(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def classwise_mask_nms(items, iou_thr=0.5):
    if len(items) == 0:
        return []

    final = []

    for c in range(1, NUM_CLASSES):
        cls_items = [x for x in items if int(x["label"]) == c]
        cls_items = sorted(cls_items, key=lambda x: x["score"], reverse=True)

        kept = []
        for item in cls_items:
            duplicate = False
            for kept_item in kept:
                if mask_iou_np(item["mask"], kept_item["mask"]) >= iou_thr:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)

        final.extend(kept)

    final = sorted(final, key=lambda x: x["score"], reverse=True)
    return final


# =========================================================
# TRAIN / VAL
# =========================================================
def train_epoch(model, loader, opt, scaler):
    model.train()
    total = 0.0
    steps = 0

    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        valid_imgs = []
        valid_targets = []
        for img, t in zip(imgs, targets):
            if t["boxes"].shape[0] == 0:
                continue
            valid_imgs.append(img)
            valid_targets.append(t)

        if len(valid_imgs) == 0:
            continue

        opt.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE.type == "cuda":
            with torch.cuda.amp.autocast():
                loss_dict = model(valid_imgs, valid_targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
        else:
            loss_dict = model(valid_imgs, valid_targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

        total += float(loss.item())
        steps += 1

    return total / max(steps, 1)


@torch.no_grad()
def val_epoch(model, loader):
    # Torchvision detection model returns loss only in train mode.
    model.train()

    total = 0.0
    steps = 0

    for imgs, targets in tqdm(loader, desc="ValLoss", leave=False):
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        valid_imgs = []
        valid_targets = []
        for img, t in zip(imgs, targets):
            if t["boxes"].shape[0] == 0:
                continue
            valid_imgs.append(img)
            valid_targets.append(t)

        if len(valid_imgs) == 0:
            continue

        loss = sum(model(valid_imgs, valid_targets).values())
        total += float(loss.item())
        steps += 1

    return total / max(steps, 1)


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def run_test_inference(model_path=CKPT_PATH):
    model = CascadeLikePointRendMaskRCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_dataset = NucleiTestDataset(TEST_ROOT)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    with open(TEST_NAME2ID_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        name_to_id = {k: int(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        name_to_id = {item["file_name"]: int(item["id"]) for item in raw}
    else:
        raise ValueError("Unsupported test_image_name_to_ids.json format")

    results = []

    for imgs, names, heights, widths in tqdm(test_loader, desc="Inference"):
        img = imgs[0]
        name = names[0]
        h = int(heights[0])
        w = int(widths[0])
        image_id = int(name_to_id[name])

        pred_items = apply_tta_predict(model, img)
        pred_items = classwise_mask_nms(pred_items, iou_thr=MASK_NMS_IOU)

        for item in pred_items:
            binary_mask = item["mask"].astype(np.uint8)

            if binary_mask.shape[0] != h or binary_mask.shape[1] != w:
                binary_mask = cv2.resize(
                    binary_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)

            if int(binary_mask.sum()) < MIN_INSTANCE_AREA:
                continue

            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(xs.max())
            y2 = float(ys.max())

            bw = x2 - x1 + 1.0
            bh = y2 - y1 + 1.0

            x1 = max(0.0, min(x1, float(w - 1)))
            y1 = max(0.0, min(y1, float(h - 1)))
            bw = max(0.0, min(bw, float(w) - x1))
            bh = max(0.0, min(bh, float(h) - y1))

            results.append({
                "image_id": image_id,
                "bbox": [x1, y1, bw, bh],
                "score": float(item["score"]),
                "category_id": int(item["label"]),
                "segmentation": binary_mask_to_rle(binary_mask),
            })

    with open(SUBMISSION_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"Saved submission to: {SUBMISSION_JSON}")
    print(f"Total predictions: {len(results)}")


# =========================================================
# PLOT
# =========================================================
def save_curves(train_losses, val_losses, val_ap50s, lrs):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=200)
    plt.close()
    print(f"Saved {LOSS_CURVE_PNG}")

    plt.figure()
    plt.plot(val_ap50s, label="val_ap50")
    plt.legend()
    plt.title("Val AP50 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("AP50")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(AP50_CURVE_PNG, dpi=200)
    plt.close()
    print(f"Saved {AP50_CURVE_PNG}")

    plt.figure()
    plt.plot(lrs, label="lr")
    plt.legend()
    plt.title("LR Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LR_CURVE_PNG, dpi=200)
    plt.close()
    print(f"Saved {LR_CURVE_PNG}")


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(42)

    dirs = [str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir()]
    random.shuffle(dirs)

    n = max(1, int(len(dirs) * VAL_RATIO))
    train_dirs = dirs[n:]
    val_dirs = dirs[:n]

    print(f"Total folders: {len(dirs)}")
    print(f"Train split: {len(train_dirs)}")
    print(f"Val split: {len(val_dirs)}")
    print(f"Device: {DEVICE}")

    train_loader = DataLoader(
        NucleiDataset(train_dirs, training=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        NucleiDataset(val_dirs, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = CascadeLikePointRendMaskRCNN().to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS,
        eta_min=LR * 0.05,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    train_losses = []
    val_losses = []
    val_ap50s = []
    lrs = []
    per_class_ap50s = []

    best_val_ap50 = -1.0
    best_epoch = -1

    for e in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_loader, opt, scaler)
        vl = val_epoch(model, val_loader)

        val_cache = collect_val_predictions(model, val_loader)
        va, pc_ap = evaluate_ap50_from_cache(val_cache)

        scheduler.step()

        cur_lr = opt.param_groups[0]["lr"]

        train_losses.append(tl)
        val_losses.append(vl)
        val_ap50s.append(va)
        lrs.append(cur_lr)
        per_class_ap50s.append(pc_ap)

        pc_str = " | ".join(
            [
                f"C{k}:{v:.4f}" if v is not None else f"C{k}:NA"
                for k, v in pc_ap.items()
            ]
        )

        print(
            f"[Epoch {e:03d}/{EPOCHS}] "
            f"lr={cur_lr:.8f} "
            f"train={tl:.4f} "
            f"val_loss={vl:.4f} "
            f"val_ap50={va:.4f} "
            f"| {pc_str}"
        )

        torch.save(model.state_dict(), LAST_CKPT_PATH)

        if va > best_val_ap50:
            best_val_ap50 = va
            best_epoch = e
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"Saved best model to: {CKPT_PATH} | Best Val AP50={best_val_ap50:.4f}")

        save_curves(train_losses, val_losses, val_ap50s, lrs)

        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_ap50s": val_ap50s,
                "per_class_ap50s": per_class_ap50s,
                "lrs": lrs,
                "best_val_ap50": best_val_ap50,
                "best_epoch": best_epoch,
                "config": {
                    "model": "Torchvision Cascade-like PointRend Mask R-CNN",
                    "backbone": "ResNet101-FPN",
                    "num_classes": NUM_CLASSES,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "score_thresh": SCORE_THRESH,
                    "mask_thresh": MASK_THRESH,
                    "mask_nms_iou": MASK_NMS_IOU,
                    "tta_scales": TTA_SCALES,
                    "tta_flips": TTA_FLIPS,
                    "use_amp": USE_AMP,
                }
            }, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Best model: {CKPT_PATH}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AP50: {best_val_ap50:.4f}")

    print("Running test inference with best model...")
    run_test_inference(CKPT_PATH)


if __name__ == "__main__":
    main()