import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

import albumentations as A

from transformers import (
    DetrConfig,
    DetrImageProcessor,
    DetrForObjectDetection,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
from torch.optim import AdamW


# =========================================================
# 0. 基本設定
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"目前使用的硬體環境: {device}")

# 路徑設定
TRAIN_IMG_DIR = "./dataset/train"
TRAIN_JSON = "./dataset/annotations/train.json"
VAL_IMG_DIR = "./dataset/val"
VAL_JSON = "./dataset/annotations/val.json"

OUTPUT_DIR = "./detr_scratch_transformer_pretrained_backbone"
FINAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "final_model")


# =========================================================
# 1. Albumentations 資料增強
# =========================================================
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,
            p=0.3,
        ),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_area=1.0,
        min_visibility=0.1,
    )
)

val_transform = None


# =========================================================
# 2. Processor
#    只載 image processor，不載 pretrained DETR model
# =========================================================
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


# =========================================================
# 3. 類別映射：原始 COCO category_id -> 連續索引 0..N-1
# =========================================================
def build_category_mappings(coco):
    original_cat_ids = sorted(coco.cats.keys())
    cat_id_to_contiguous = {cat_id: idx for idx, cat_id in enumerate(original_cat_ids)}
    contiguous_to_cat_id = {idx: cat_id for cat_id, idx in cat_id_to_contiguous.items()}
    id2label = {idx: coco.cats[cat_id]["name"] for idx, cat_id in contiguous_to_cat_id.items()}
    label2id = {name: idx for idx, name in id2label.items()}
    return cat_id_to_contiguous, contiguous_to_cat_id, id2label, label2id


# =========================================================
# 4. Dataset
# =========================================================
class DetrCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor, cat_id_to_contiguous, transforms=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.cat_id_to_contiguous = cat_id_to_contiguous
        self.aug = transforms   # 改這裡，不要叫 self.transforms

    def _filter_and_convert_annotations(self, anns, image_w, image_h):
        valid = []
        for ann in anns:
            if "bbox" not in ann or "category_id" not in ann:
                continue

            x, y, w, h = ann["bbox"]

            # 過濾掉異常框
            if w <= 1e-3 or h <= 1e-3:
                continue

            # clamp 到影像範圍
            x = max(0, x)
            y = max(0, y)
            w = min(w, image_w - x)
            h = min(h, image_h - y)

            if w <= 1e-3 or h <= 1e-3:
                continue

            original_cat_id = ann["category_id"]
            if original_cat_id not in self.cat_id_to_contiguous:
                continue

            valid.append({
                "bbox": [x, y, w, h],
                "category_id": self.cat_id_to_contiguous[original_cat_id],
                "area": float(w * h),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
        return valid

    def __getitem__(self, idx):
        # 這裡現在不會再碰到 albumentations 衝突
        img, anns = super().__getitem__(idx)
        image_id = self.ids[idx]

        image_w, image_h = img.size
        anns = self._filter_and_convert_annotations(anns, image_w, image_h)

        image_np = np.array(img)

        # Albumentations augmentation
        if self.aug is not None:
            bboxes = [ann["bbox"] for ann in anns]
            category_ids = [ann["category_id"] for ann in anns]

            transformed = self.aug(
                image=image_np,
                bboxes=bboxes,
                category_ids=category_ids
            )

            image_np = transformed["image"]
            new_bboxes = transformed["bboxes"]
            new_category_ids = transformed["category_ids"]

            anns = []
            h_img, w_img = image_np.shape[:2]
            for bbox, cat_id in zip(new_bboxes, new_category_ids):
                x, y, w, h = bbox

                if w <= 1e-3 or h <= 1e-3:
                    continue

                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                if w <= 1e-3 or h <= 1e-3:
                    continue

                anns.append({
                    "bbox": [x, y, w, h],
                    "category_id": int(cat_id),
                    "area": float(w * h),
                    "iscrowd": 0,
                })

        target = {
            "image_id": image_id,
            "annotations": anns
        }

        encoding = self.processor(
            images=image_np,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# =========================================================
# 5. Collate function
#    用 processor.pad 自動處理 padding
# =========================================================
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(img.shape[1] for img in pixel_values)
    max_w = max(img.shape[2] for img in pixel_values)

    padded_images = []
    pixel_masks = []

    for img in pixel_values:
        c, h, w = img.shape

        padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        padded_img[:, :h, :w] = img
        padded_images.append(padded_img)

        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


# =========================================================
# 6. 建立資料集
# =========================================================
print("正在載入資料集與類別資訊...")
tmp_train_dataset = torchvision.datasets.CocoDetection(TRAIN_IMG_DIR, TRAIN_JSON)

cat_id_to_contiguous, contiguous_to_cat_id, id2label, label2id = build_category_mappings(tmp_train_dataset.coco)

print("偵測到的類別映射:")
print(id2label)

train_dataset = DetrCocoDataset(
    img_folder=TRAIN_IMG_DIR,
    ann_file=TRAIN_JSON,
    processor=processor,
    cat_id_to_contiguous=cat_id_to_contiguous,
    transforms=train_transform
)

val_dataset = DetrCocoDataset(
    img_folder=VAL_IMG_DIR,
    ann_file=VAL_JSON,
    processor=processor,
    cat_id_to_contiguous=cat_id_to_contiguous,
    transforms=val_transform
)

print(f"訓練集數量: {len(train_dataset)}")
print(f"驗證集數量: {len(val_dataset)}")


# =========================================================
# 7. 建立 DETR 模型
#    注意：不用 from_pretrained()
#    -> encoder-decoder / heads 會是 random init
# =========================================================
config = DetrConfig(
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    auxiliary_loss=False,
)

model = DetrForObjectDetection(config)


# =========================================================
# 8. 只載 backbone 的 pretrained weights
# =========================================================
def get_detr_backbone_module(model):
    candidates = [
        ["model", "backbone", "conv_encoder", "model"],
        ["model", "backbone", "model"],
        ["backbone", "conv_encoder", "model"],
        ["backbone", "model"],
    ]

    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj, ".".join(path)

    raise RuntimeError("找不到 DETR 內部 backbone 模組，請先 print(model) 檢查結構。")


backbone_module, backbone_path = get_detr_backbone_module(model)
print(f"找到 backbone 路徑: {backbone_path}")

tv_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

missing_keys, unexpected_keys = backbone_module.load_state_dict(tv_resnet.state_dict(), strict=False)
print("已成功載入 torchvision ResNet-50 pretrained weights 到 DETR backbone")
print(f"Backbone missing keys: {len(missing_keys)}")
print(f"Backbone unexpected keys: {len(unexpected_keys)}")

model.to(device)


# =========================================================
# 9. 參數分組：backbone 與其他層分開設定 learning rate
# =========================================================
def is_backbone_param_name(name):
    return name.startswith("model.backbone.") or ".backbone." in name

backbone_params = []
other_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if is_backbone_param_name(name):
        backbone_params.append(param)
    else:
        other_params.append(param)

print(f"backbone tensors: {len(backbone_params)}")
print(f"other tensors: {len(other_params)}")


# =========================================================
# 10. 自訂 Trainer
# =========================================================
class DetrTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": backbone_params,
                    "lr": 1e-5,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": other_params,
                    "lr": 1e-4,
                    "weight_decay": self.args.weight_decay,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name="linear",
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler


# =========================================================
# 11. TrainingArguments
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs= 200,

    fp16=False,
    max_grad_norm=0.1,
    weight_decay=1e-4,

    remove_unused_columns=False,

    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=50,
    save_total_limit=2,

    dataloader_num_workers=4,
    report_to="none",
)


# =========================================================
# 12. 建立 Trainer
# =========================================================
trainer = DetrTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=collate_fn,
)


# =========================================================
# 13. 開始訓練
# =========================================================
print("開始訓練...")
trainer.train()


# =========================================================
# 14. 儲存模型
# =========================================================
print("訓練完成！儲存模型...")
trainer.save_model(FINAL_SAVE_DIR)
processor.save_pretrained(FINAL_SAVE_DIR)

meta = {
    "id2label": {int(k): v for k, v in id2label.items()},
    "label2id": label2id,
    "cat_id_to_contiguous": {int(k): int(v) for k, v in cat_id_to_contiguous.items()},
    "contiguous_to_cat_id": {int(k): int(v) for k, v in contiguous_to_cat_id.items()},
}

with open(os.path.join(FINAL_SAVE_DIR, "label_mappings.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"模型已成功儲存至: {FINAL_SAVE_DIR}")
