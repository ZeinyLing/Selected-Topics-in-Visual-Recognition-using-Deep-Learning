import os
import json
import random
from pathlib import Path

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

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import roi_heads as roi_heads_module
from torchvision.models.detection.roi_heads import project_masks_on_boxes
from pycocotools import mask as mask_utils
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "./hw3-data-release"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

OUTPUT_DIR = "./outputs_cascade_maskrcnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_JSON = os.path.join(OUTPUT_DIR, "submission.json")
LOSS_CURVE_PNG = os.path.join(OUTPUT_DIR, "loss_curve.png")
AP50_CURVE_PNG = os.path.join(OUTPUT_DIR, "ap50_curve.png")
HISTORY_JSON = os.path.join(OUTPUT_DIR, "history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
EPOCHS = 40
BATCH_SIZE = 2
LR = 1e-4
VAL_RATIO = 0.10
MIN_INSTANCE_AREA = 8
DICE_WEIGHT = 1.0

# inference / validation thresholds
SCORE_THRESH = 0.30
MASK_THRESH = 0.50


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# DATASET
# =========================================================
class NucleiDataset(Dataset):
    def __init__(self, dirs):
        self.dirs = dirs

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
                ys, xs = np.where(m)

                if len(xs) == 0 or m.sum() < MIN_INSTANCE_AREA:
                    continue

                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                labels.append(cls)
                masks.append(m)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.stack(masks), dtype=torch.uint8)

        image = TF.to_tensor(image)

        return image, {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }


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
# DICE LOSS PATCH
# =========================================================
def dice_loss(pred, gt):
    p = torch.sigmoid(pred).flatten(1)
    g = gt.flatten(1).float()

    inter = (p * g).sum(1)
    union = p.sum(1) + g.sum(1)

    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def mask_loss(mask_logits, proposals, gt_masks, gt_labels, idxs):
    size = mask_logits.shape[-1]

    labels = [gt_label[i] for gt_label, i in zip(gt_labels, idxs)]
    targets = [project_masks_on_boxes(m, p, i, size) for m, p, i in zip(gt_masks, proposals, idxs)]

    labels = torch.cat(labels)
    targets = torch.cat(targets)

    if targets.numel() == 0:
        return mask_logits.sum() * 0

    pred = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]

    return F.binary_cross_entropy_with_logits(pred, targets) + DICE_WEIGHT * dice_loss(pred, targets)


roi_heads_module.maskrcnn_loss = mask_loss


# =========================================================
# MODEL
# =========================================================
class CascadeMaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet_fpn_backbone(
            'resnet101',
            weights='IMAGENET1K_V2',
            trainable_layers=5
        )

        base = MaskRCNN(
            backbone,
            num_classes=NUM_CLASSES
        )

        in_feat = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)

        in_mask = base.roi_heads.mask_predictor.conv5_mask.in_channels
        base.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)

        self.backbone = base.backbone
        self.rpn = base.rpn
        self.roi_heads = base.roi_heads
        self.transform = base.transform

        self.iou = [0.5, 0.6, 0.7]
        self.heads = nn.ModuleList([FastRCNNPredictor(in_feat, NUM_CLASSES) for _ in self.iou])

    def forward(self, images, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        feats = self.backbone(images.tensors)

        if not self.training:
            props, _ = self.rpn(images, feats)
            det = None
            for i, h in enumerate(self.heads):
                self.roi_heads.box_predictor = h
                if i == 0:
                    det, _ = self.roi_heads(feats, props, images.image_sizes)
                else:
                    if hasattr(self.roi_heads, "proposal_matcher") and hasattr(self.roi_heads.proposal_matcher, "high_threshold"):
                        self.roi_heads.proposal_matcher.high_threshold = self.iou[i]
                    det, _ = self.roi_heads(feats, [d["boxes"] for d in det], images.image_sizes)
            det = self.transform.postprocess(det, images.image_sizes, original_image_sizes)
            return det

        props, ploss = self.rpn(images, feats, targets)

        losses = []
        for i, h in enumerate(self.heads):
            self.roi_heads.box_predictor = h
            if hasattr(self.roi_heads, "proposal_matcher") and hasattr(self.roi_heads.proposal_matcher, "high_threshold"):
                self.roi_heads.proposal_matcher.high_threshold = self.iou[i]
            _, dl = self.roi_heads(feats, props, images.image_sizes, targets)
            losses.append({k: v / (i + 1) for k, v in dl.items()})

        out = {}
        for k in losses[0]:
            out[k] = sum(l[k] for l in losses)

        out.update(ploss)
        return out


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


def evaluate_ap50_from_cache(cache, score_thresh=SCORE_THRESH, mask_thresh=MASK_THRESH, min_area=MIN_INSTANCE_AREA, iou_thresh=0.5):
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

    return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0


# =========================================================
# TRAIN / VAL
# =========================================================
def train_epoch(model, loader, opt):
    model.train()
    total = 0.0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss = sum(model(imgs, targets).values())

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
    return total / max(len(loader), 1)


def val_epoch(model, loader):
    model.train()  # MaskRCNN val 要 train mode 才有 loss
    total = 0.0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="ValLoss", leave=False):
            imgs = [i.to(DEVICE) for i in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss = sum(model(imgs, targets).values())
            total += loss.item()
    return total / max(len(loader), 1)


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def run_test_inference(model_path=CKPT_PATH):
    model = CascadeMaskRCNN().to(DEVICE)
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
        imgs = [i.to(DEVICE) for i in imgs]
        outputs = model(imgs)

        for out, name, h, w in zip(outputs, names, heights, widths):
            image_id = int(name_to_id[name])

            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            masks = out["masks"].detach().cpu().numpy()

            for box, label, score, mask_prob in zip(boxes, labels, scores, masks):
                if float(score) < SCORE_THRESH:
                    continue
                if int(label) < 1 or int(label) >= NUM_CLASSES:
                    continue

                binary_mask = (mask_prob[0] >= MASK_THRESH).astype(np.uint8)
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
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": binary_mask_to_rle(binary_mask),
                })

    with open(SUBMISSION_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"Saved submission to: {SUBMISSION_JSON}")
    print(f"Total predictions: {len(results)}")


# =========================================================
# MAIN + PLOT
# =========================================================
def main():
    set_seed()

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
        NucleiDataset(train_dirs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        NucleiDataset(val_dirs),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CascadeMaskRCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    val_ap50s = []
    best_val_ap50 = -1.0

    for e in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_loader, opt)
        vl = val_epoch(model, val_loader)
        val_cache = collect_val_predictions(model, val_loader)
        va = evaluate_ap50_from_cache(val_cache)

        train_losses.append(tl)
        val_losses.append(vl)
        val_ap50s.append(va)

        print(f"[Epoch {e}] train={tl:.4f} val_loss={vl:.4f} val_ap50={va:.4f}")

        if va > best_val_ap50:
            best_val_ap50 = va
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"Saved best model to: {CKPT_PATH}")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(LOSS_CURVE_PNG)
    print(f"Saved {LOSS_CURVE_PNG}")

    plt.figure()
    plt.plot(val_ap50s, label="val_ap50")
    plt.legend()
    plt.title("Val AP50 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("AP50")
    plt.savefig(AP50_CURVE_PNG)
    print(f"Saved {AP50_CURVE_PNG}")

    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_ap50s": val_ap50s,
            "best_val_ap50": best_val_ap50,
        }, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Best model: {CKPT_PATH}")
    print(f"Best val AP50: {best_val_ap50:.4f}")
    print("Running test inference with best model...")
    run_test_inference(CKPT_PATH)


if __name__ == "__main__":
    main()
