import os
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from pycocotools import mask as mask_utils

# =========================================================
# 0. Config
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./hw3-data-release"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

OUTPUT_DIR = "./outputs_maskrcnn_cbam"
CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "submission.json")

NUM_CLASSES = 5  # background + 4 classes
BATCH_SIZE = 2
NUM_WORKERS = 0
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
MIN_INSTANCE_AREA = 8
DEFAULT_MASK_THRESH = 0.5
DEFAULT_SCORE_THRESH = 0.3

# threshold tuning candidates
TUNE_SCORE_THRESH = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
TUNE_MASK_THRESH = [0.3, 0.4, 0.5, 0.6]
TUNE_MIN_AREA = [4, 8, 12]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2. TIFF IO
# =========================================================
def read_rgb_tif(path: str) -> np.ndarray:
    try:
        arr = tifffile.imread(path)
    except Exception:
        try:
            arr = np.array(Image.open(path))
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Failed to read image tif: {path}") from e

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in [1, 3] and arr.shape[-1] not in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr.astype(np.uint8)


def read_mask_tif(path: str) -> np.ndarray:
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


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": rle["counts"],
    }


# =========================================================
# 3. CBAM modules
# =========================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(7)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =========================================================
# 4. Dataset + augmentation
# =========================================================
class NucleiTrainDataset(Dataset):
    def __init__(self, sample_dirs: List[str], train: bool = True):
        self.sample_dirs = sample_dirs
        self.train = train

    def __len__(self):
        return len(self.sample_dirs)

    def _build_target(self, sample_dir: str, h: int, w: int) -> Dict[str, torch.Tensor]:
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for class_idx in range(1, 5):
            mask_path = os.path.join(sample_dir, f"class{class_idx}.tif")
            if not os.path.exists(mask_path):
                continue

            try:
                inst_map = read_mask_tif(mask_path)
            except Exception as e:
                print(f"[Warning] Skip unreadable mask: {mask_path} | {e}")
                continue

            unique_ids = np.unique(inst_map)
            unique_ids = unique_ids[unique_ids > 0]

            for inst_id in unique_ids:
                binary_mask = (inst_map == inst_id).astype(np.uint8)
                ys, xs = np.where(binary_mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                area = float(binary_mask.sum())
                if area < MIN_INSTANCE_AREA:
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_idx)
                masks.append(binary_mask)
                areas.append(area)
                iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([0], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        image_path = os.path.join(sample_dir, "image.tif")
        image = read_rgb_tif(image_path)
        h, w = image.shape[:2]
        target = self._build_target(sample_dir, h, w)
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)

        if self.train:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                if target["boxes"].shape[0] > 0:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = w - 1 - boxes[:, [2, 0]]
                    target["boxes"] = boxes
                    target["masks"] = torch.flip(target["masks"], dims=[2])

            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1, :])
                if target["boxes"].shape[0] > 0:
                    boxes = target["boxes"].clone()
                    boxes[:, [1, 3]] = h - 1 - boxes[:, [3, 1]]
                    target["boxes"] = boxes
                    target["masks"] = torch.flip(target["masks"], dims=[1])

            k = random.choice([0, 1, 2, 3])
            if k > 0:
                image = np.rot90(image, k).copy()
                if target["boxes"].shape[0] > 0:
                    masks_np = target["masks"].numpy()
                    masks_np = np.rot90(masks_np, k, axes=(1, 2)).copy()
                    target["masks"] = torch.as_tensor(masks_np, dtype=torch.uint8)
                    boxes = masks_to_boxes_np(masks_np)
                    target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

            image = apply_color_jitter_np(image)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, target


class NucleiTestDataset(Dataset):
    def __init__(self, test_dir: str):
        self.image_paths = sorted(list(Path(test_dir).glob("*.tif")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = str(self.image_paths[idx])
        image = read_rgb_tif(image_path)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_tensor, os.path.basename(image_path), image.shape[0], image.shape[1]


def masks_to_boxes_np(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return np.array(boxes, dtype=np.float32)


def apply_color_jitter_np(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)

    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)  # contrast
        img = (img - 127.5) * alpha + 127.5

    if random.random() < 0.5:
        beta = random.uniform(-12, 12)  # brightness
        img = img + beta

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


# =========================================================
# 5. Backbone with CBAM
# =========================================================
class CBAMBackboneWithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V1", norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.body = create_feature_extractor(
            backbone,
            return_nodes={
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3",
            },
        )
        self.cbam0 = CBAM(256)
        self.cbam1 = CBAM(512)
        self.cbam2 = CBAM(1024)
        self.cbam3 = CBAM(2048)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def forward(self, x):
        feats = self.body(x)
        feats["0"] = self.cbam0(feats["0"])
        feats["1"] = self.cbam1(feats["1"])
        feats["2"] = self.cbam2(feats["2"])
        feats["3"] = self.cbam3(feats["3"])
        feats = self.fpn(feats)
        return feats


def get_model(num_classes: int = 5):
    backbone = CBAMBackboneWithFPN()

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=400,
        max_size=800,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# =========================================================
# 6. Metrics / evaluation
# =========================================================
def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


@torch.no_grad()
def collect_val_predictions(model, loader, device):
    model.eval()
    data = []
    for images, targets in tqdm(loader, desc="CollectVal", leave=False):
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)
        for output, target in zip(outputs, targets):
            data.append({
                "gt_masks": target["masks"].cpu().numpy(),
                "gt_labels": target["labels"].cpu().numpy(),
                "pred_scores": output["scores"].detach().cpu().numpy(),
                "pred_labels": output["labels"].detach().cpu().numpy(),
                "pred_masks": output["masks"].detach().cpu().numpy(),
            })
    return data


def evaluate_ap50_from_cache(cached_data, num_classes: int, score_thresh: float, mask_thresh: float, min_area: int, iou_thresh: float = 0.5):
    all_gt = {c: 0 for c in range(1, num_classes)}
    pred_records = {c: [] for c in range(1, num_classes)}

    for item in cached_data:
        gt_masks = item["gt_masks"] if len(item["gt_masks"]) > 0 else np.zeros((0, 1, 1), dtype=np.uint8)
        gt_labels = item["gt_labels"] if len(item["gt_labels"]) > 0 else np.zeros((0,), dtype=np.int64)
        pred_scores = item["pred_scores"]
        pred_labels = item["pred_labels"]
        pred_masks = item["pred_masks"]

        gt_by_class = {c: gt_masks[gt_labels == c] for c in range(1, num_classes)}
        used_gt = {c: np.zeros((len(gt_by_class[c]),), dtype=bool) for c in range(1, num_classes)}

        for c in range(1, num_classes):
            all_gt[c] += int(np.sum(gt_labels == c))

        order = np.argsort(-pred_scores)
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]
        pred_masks = pred_masks[order]

        for score, label, mask_prob in zip(pred_scores, pred_labels, pred_masks):
            if float(score) < score_thresh:
                continue
            if int(label) < 1 or int(label) >= num_classes:
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

    ap_list = []
    for c in range(1, num_classes):
        n_gt = all_gt[c]
        records = pred_records[c]
        if n_gt == 0:
            continue
        if len(records) == 0:
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
        ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
        ap_list.append(float(ap))

    return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0





# =========================================================
# 7. Train / val / plot
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="ValLoss", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / max(len(loader), 1)


def save_training_curves(train_losses, val_losses, val_ap50s, output_dir: str):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_ap50s, label="Val AP50")
    plt.xlabel("Epoch")
    plt.ylabel("AP50")
    plt.title("Validation AP50")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ap50_curve.png"), dpi=200)
    plt.close()


# =========================================================
# 8. Inference
# =========================================================
@torch.no_grad()
def build_submission(model, test_loader, name_to_id: Dict[str, int], device, score_thresh: float, mask_thresh: float, min_area: int):
    model.eval()
    results = []

    for images, file_names, heights, widths in tqdm(test_loader, desc="Inference"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, file_name, h, w in zip(outputs, file_names, heights, widths):
            image_id = int(name_to_id[file_name])
            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()

            for box, label, score, mask_prob in zip(boxes, labels, scores, masks):
                if float(score) < score_thresh:
                    continue
                if int(label) < 1 or int(label) >= NUM_CLASSES:
                    continue

                binary_mask = (mask_prob[0] >= mask_thresh).astype(np.uint8)
                if int(binary_mask.sum()) < min_area:
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
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": binary_mask_to_rle(binary_mask),
                })
    return results


def run_inference_only(score_thresh=DEFAULT_SCORE_THRESH, mask_thresh=DEFAULT_MASK_THRESH, min_area=MIN_INSTANCE_AREA):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    test_dataset = NucleiTestDataset(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    with open(TEST_NAME2ID_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        name_to_id = {k: int(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        name_to_id = {item["file_name"]: int(item["id"]) for item in raw}
    else:
        raise ValueError("Unsupported test_image_name_to_ids.json format")

    submission = build_submission(model, test_loader, name_to_id, DEVICE, score_thresh, mask_thresh, min_area)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)

    print(f"Submission saved to: {OUTPUT_JSON}")
    print(f"Total predicted instances: {len(submission)}")


# =========================================================
# 9. Main
# =========================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_dirs = sorted([str(p) for p in Path(TRAIN_DIR).iterdir() if p.is_dir() and (p / "image.tif").exists()])
    assert len(sample_dirs) > 0, f"No training folders found in {TRAIN_DIR}"

    random.shuffle(sample_dirs)
    n_val = max(1, int(len(sample_dirs) * VAL_RATIO))
    val_dirs = sample_dirs[:n_val]
    train_dirs = sample_dirs[n_val:]

    print(f"Total folders: {len(sample_dirs)}")
    print(f"Train split: {len(train_dirs)}")
    print(f"Val split: {len(val_dirs)}")
    print(f"Device: {DEVICE}")

    train_dataset = NucleiTrainDataset(train_dirs, train=True)
    val_dataset = NucleiTrainDataset(val_dirs, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_ap50 = -1.0
    best_cfg = (DEFAULT_SCORE_THRESH, DEFAULT_MASK_THRESH, MIN_INSTANCE_AREA)  # 固定 threshold
    train_losses, val_losses, val_ap50s = [], [], []

    print("Start training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, DEVICE)

        # 使用固定 threshold（加速訓練）
        val_ap50 = evaluate_ap50_from_cache(
            collect_val_predictions(model, val_loader, DEVICE),
            NUM_CLASSES,
            score_thresh=DEFAULT_SCORE_THRESH,
            mask_thresh=DEFAULT_MASK_THRESH,
            min_area=MIN_INSTANCE_AREA,
            iou_thresh=0.5,
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ap50s.append(val_ap50)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_AP50={val_ap50:.4f} | "
            f"score={best_cfg[0]:.2f} | mask={best_cfg[1]:.2f} | min_area={best_cfg[2]} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_ap50 > best_val_ap50:
            best_val_ap50 = val_ap50
            # 不再更新 threshold（固定）
            torch.save(copy.deepcopy(model.state_dict()), CKPT_PATH)
            with open(os.path.join(OUTPUT_DIR, "best_thresholds.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "score_thresh": best_cfg[0],
                    "mask_thresh": best_cfg[1],
                    "min_area": best_cfg[2],
                    "val_ap50": best_val_ap50,
                }, f, ensure_ascii=False, indent=2)
            print(f"Saved best model to: {CKPT_PATH}")

    save_training_curves(train_losses, val_losses, val_ap50s, OUTPUT_DIR)

    print("Training finished.")
    print(f"Best checkpoint: {CKPT_PATH}")
    print(f"Best val AP50: {best_val_ap50:.4f}")
    print(f"Using fixed thresholds: score={best_cfg[0]:.2f}, mask={best_cfg[1]:.2f}, min_area={best_cfg[2]}")

    print("Run inference with saved best model...")
    run_inference_only(score_thresh=best_cfg[0], mask_thresh=best_cfg[1], min_area=best_cfg[2])


if __name__ == "__main__":
    main()
