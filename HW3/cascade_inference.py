import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm

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


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "./hw3-data-release"
TEST_ROOT = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

OUTPUT_DIR = "./outputs_cascade_maskrcnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_JSON = os.path.join(OUTPUT_DIR, "submission.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
MIN_INSTANCE_AREA = 8
DICE_WEIGHT = 1.0

# inference thresholds
SCORE_THRESH = 0.10
MASK_THRESH = 0.30


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
# DICE LOSS PATCH
# 與訓練版保持一致，確保載入權重時結構一致
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

        base = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
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
# INFERENCE
# =========================================================
@torch.no_grad()
def run_test_inference(model_path=CKPT_PATH):
    model = CascadeMaskRCNN().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
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


if __name__ == "__main__":
    run_test_inference()