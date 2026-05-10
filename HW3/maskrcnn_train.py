import os
import random
import copy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# =========================================================
# 0. Basic config
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== modify these paths ========
DATA_ROOT = "./hw3-data-release"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
OUTPUT_DIR = "./outputs_maskrcnn_train"
CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# background + 4 classes
NUM_CLASSES = 5
BATCH_SIZE = 2
NUM_WORKERS = 0
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
MIN_INSTANCE_AREA = 8
MASK_THRESH = 0.5
EVAL_SCORE_THRESH = 0.05

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
# 2. Utilities
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


# =========================================================
# 3. Dataset
# train/
#   sample_xxx/
#       image.tif
#       class1.tif
#       class2.tif
#       class3.tif
#       class4.tif
# Note:
# some folders may not contain all class*.tif files.
# This code will skip missing class files automatically.
# =========================================================
class NucleiTrainDataset(Dataset):
    def __init__(self, sample_dirs: List[str], train: bool = True):
        self.sample_dirs = sample_dirs
        self.train = train

    def __len__(self):
        return len(self.sample_dirs)

    def _build_target(self, sample_dir: str, h: int, w: int) -> Dict[str, torch.Tensor]:
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

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
                if (x_max - x_min + 1) < 1 or (y_max - y_min + 1) < 1:
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

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([0], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return target

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

        image = F.to_tensor(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, target


# =========================================================
# 4. Model
# =========================================================
def get_model(num_classes: int = 5):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


# =========================================================
# 5. Metric: simplified mask AP50
# =========================================================
def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


@torch.no_grad()
def evaluate_ap50(model, loader, device, score_thresh: float = 0.05, iou_thresh: float = 0.5):
    model.eval()

    all_gt = {c: 0 for c in range(1, NUM_CLASSES)}
    pred_records = {c: [] for c in range(1, NUM_CLASSES)}

    for images, targets in tqdm(loader, desc="AP50", leave=False):
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)

        for output, target in zip(outputs, targets):
            gt_masks = target["masks"].cpu().numpy() if len(target["masks"]) > 0 else np.zeros((0, 1, 1), dtype=np.uint8)
            gt_labels = target["labels"].cpu().numpy() if len(target["labels"]) > 0 else np.zeros((0,), dtype=np.int64)

            pred_scores = output["scores"].detach().cpu().numpy()
            pred_labels = output["labels"].detach().cpu().numpy()
            pred_masks = output["masks"].detach().cpu().numpy()

            gt_by_class = {c: gt_masks[gt_labels == c] for c in range(1, NUM_CLASSES)}
            used_gt = {c: np.zeros((len(gt_by_class[c]),), dtype=bool) for c in range(1, NUM_CLASSES)}

            for c in range(1, NUM_CLASSES):
                all_gt[c] += int(np.sum(gt_labels == c))

            order = np.argsort(-pred_scores)
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]
            pred_masks = pred_masks[order]

            for score, label, mask_prob in zip(pred_scores, pred_labels, pred_masks):
                if score < score_thresh:
                    continue
                if label < 1 or label >= NUM_CLASSES:
                    continue

                pred_mask = (mask_prob[0] >= MASK_THRESH).astype(np.uint8)
                gt_pool = gt_by_class[label]

                best_iou = 0.0
                best_idx = -1
                for i, gt_mask in enumerate(gt_pool):
                    if used_gt[label][i]:
                        continue
                    iou = compute_mask_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                is_tp = 0
                if best_iou >= iou_thresh and best_idx >= 0:
                    used_gt[label][best_idx] = True
                    is_tp = 1

                pred_records[label].append((float(score), is_tp))

    ap_list = []
    for c in range(1, NUM_CLASSES):
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

    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))


# =========================================================
# 6. Plot helpers
# =========================================================
def save_training_curves(train_losses, val_losses, val_ap50s, output_dir: str):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
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
# 7. Train / Validate
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

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / max(len(loader), 1)


# =========================================================
# 8. Main
# =========================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_dirs = sorted([
        str(p) for p in Path(TRAIN_DIR).iterdir()
        if p.is_dir() and (p / "image.tif").exists()
    ])
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_ap50 = -1.0
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    val_ap50s = []

    print("Start training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, DEVICE)
        val_ap50 = evaluate_ap50(model, val_loader, DEVICE, score_thresh=EVAL_SCORE_THRESH, iou_thresh=0.5)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ap50s.append(val_ap50)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AP50={val_ap50:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_ap50 > best_val_ap50:
            best_val_ap50 = val_ap50
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, CKPT_PATH)
            print(f"Saved best model to: {CKPT_PATH}")

    save_training_curves(train_losses, val_losses, val_ap50s, OUTPUT_DIR)

    print("Training finished.")
    print(f"Best checkpoint: {CKPT_PATH}")
    print(f"Best val AP50: {best_val_ap50:.4f}")
    print(f"Best val loss at saved checkpoint: {best_val_loss:.4f}")
    print(f"Loss curve saved to: {os.path.join(OUTPUT_DIR, 'loss_curve.png')}")
    print(f"AP50 curve saved to: {os.path.join(OUTPUT_DIR, 'ap50_curve.png')}")


if __name__ == "__main__":
    main()
