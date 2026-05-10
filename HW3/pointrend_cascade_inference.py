# -*- coding: utf-8 -*-
"""
Pure inference script for Torchvision-only Cascade-like PointRend Mask R-CNN.

Modified from your training script:
- No training
- No validation
- No plotting
- Load best_model.pth
- Run test_release inference
- Use multi-scale + flip TTA
- Use class-wise mask NMS
- Save COCO-style submission.json
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm

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
from torchvision.models.detection.anchor_utils import AnchorGenerator

from pycocotools import mask as mask_utils


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "./hw3-data-release"
TEST_ROOT = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

OUTPUT_DIR = "./outputs_torchvision_cascade_like_pointrend"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_JSON = os.path.join(OUTPUT_DIR, "submission.json")

# Change to "cuda:0" / "cuda:1" if needed.
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# background + class1~class4
NUM_CLASSES = 5

MIN_INSTANCE_AREA = 8

SCORE_THRESH = 0.10
MASK_THRESH = 0.30
MASK_NMS_IOU = 0.30

ENABLE_TTA = True
TTA_SCALES = [1.0]      # If OOM/slow, use [1.0]
TTA_FLIPS = ["none"]  # If slow, use ["none"]

BOX_DETECTIONS_PER_IMG = 800
RPN_POST_NMS_TOP_N_TRAIN = 1500
RPN_POST_NMS_TOP_N_TEST = 1500
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 2000


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
# TEST DATASET
# =========================================================
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
# MODEL
# =========================================================
class CascadeLikePointRendMaskRCNN(nn.Module):
    """
    Must match the training model exactly.
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

        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)

        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)

        model.roi_heads.score_thresh = 0.05
        model.roi_heads.nms_thresh = 0.5
        model.roi_heads.detections_per_img = BOX_DETECTIONS_PER_IMG

        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)


# =========================================================
# TTA HELPERS
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


@torch.no_grad()
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

            boxes = unflip_boxes(boxes, flip_type, sh, sw)

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
# INFERENCE
# =========================================================
@torch.no_grad()
def run_test_inference(model_path=CKPT_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {model_path}")
    print(f"TTA scales: {TTA_SCALES if ENABLE_TTA else [1.0]}")
    print(f"TTA flips: {TTA_FLIPS if ENABLE_TTA else ['none']}")

    model = CascadeLikePointRendMaskRCNN().to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
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

        if name not in name_to_id:
            raise KeyError(f"{name} not found in {TEST_NAME2ID_JSON}")

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


def main():
    run_test_inference(CKPT_PATH)


if __name__ == "__main__":
    main()