import os
import json
from pathlib import Path
from typing import Dict

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
from pycocotools import mask as mask_utils

# =========================================================
# 0. Config
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./hw3-data-release"
TEST_DIR = os.path.join(DATA_ROOT, "test_release")
TEST_NAME2ID_JSON = os.path.join(DATA_ROOT, "test_image_name_to_ids.json")

CKPT_PATH = "./outputs_maskrcnn_train/best_model.pth"
OUTPUT_JSON = "./outputs_maskrcnn_train/submission2.json"

NUM_CLASSES = 5  # background + 4 classes
NUM_WORKERS = 0
MASK_THRESH = 0.3
SCORE_THRESH = 0.1
MIN_INSTANCE_AREA = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# 1. Read tif
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
# 2. Dataset
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
# 3. Model
# =========================================================
def get_model(num_classes: int = 5):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# =========================================================
# 4. Inference
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
# 5. Main
# =========================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
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
