import os
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile
from tqdm import tqdm

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./hw3-data-release"
TEST_DIR = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

CKPT_PATH = "./outputs_maskrcnn_cbam/best_model.pth"
OUTPUT_JSON = "./outputs_maskrcnn_cbam/submission.json"

NUM_CLASSES = 5  # background + 4 classes
NUM_WORKERS = 0

# 這三個可以手動調
SCORE_THRESH = 0.1
MASK_THRESH = 0.3
MIN_INSTANCE_AREA = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# 1. TIFF IO
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
# 2. CBAM
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
# 3. Test Dataset
# =========================================================
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


# =========================================================
# 4. Backbone with CBAM + FPN
# =========================================================
class CBAMBackboneWithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(
            weights="IMAGENET1K_V1",
            norm_layer=misc_nn_ops.FrozenBatchNorm2d
        )

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
# 5. Inference
# official format:
# [{
#   "image_id": int,
#   "bbox": [x, y, width, height],
#   "score": float,
#   "category_id": int,
#   "segmentation": {"size": [H, W], "counts": "..."}
# }]
# =========================================================
@torch.no_grad()
def build_submission(model, test_loader, name_to_id: Dict[str, int], device):
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

    return results


# =========================================================
# 6. Main
# =========================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    model = get_model(NUM_CLASSES).to(DEVICE)

    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    test_dataset = NucleiTestDataset(TEST_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    with open(TEST_NAME2ID_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        name_to_id = {k: int(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        name_to_id = {item["file_name"]: int(item["id"]) for item in raw}
    else:
        raise ValueError("Unsupported test_image_name_to_ids.json format")

    submission = build_submission(model, test_loader, name_to_id, DEVICE)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)

    print(f"Submission saved to: {OUTPUT_JSON}")
    print(f"Total predicted instances: {len(submission)}")


if __name__ == "__main__":
    main()