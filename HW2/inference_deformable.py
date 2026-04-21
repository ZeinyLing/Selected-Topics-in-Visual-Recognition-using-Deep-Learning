import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection
)

# ==========================================
# 1. 基本設定
# ==========================================
MODEL_PATH = "./deformable-detr-my-coco-model/final_model"
TEST_IMAGE_DIR = "./dataset/test"
OUTPUT_JSON = "pred_deformable0.01.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用硬體: {device}")

SCORE_THRESHOLD = 0.01
CLAMP_BOX_TO_IMAGE = True

# ==========================================
# 2. 載入模型
# ==========================================
print("正在載入 Deformable DETR 模型...")
processor = DeformableDetrImageProcessor.from_pretrained(MODEL_PATH)
model = DeformableDetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()

print("模型載入完成")
print("decoder_layers =", model.config.decoder_layers)
print("num_queries =", model.config.num_queries)
print("id2label =", model.config.id2label)
print("label2id =", model.config.label2id)

predictions = []

# ==========================================
# 3. 讀取測試圖片
# ==========================================
def numeric_sort_key(filename):
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem)
    except ValueError:
        return stem

image_files = sorted(
    [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    key=numeric_sort_key
)

print(f"找到 {len(image_files)} 張測試圖片")

# ==========================================
# 4. 開始推論
# ==========================================
print("開始產生 pred.json ...")

for img_file in tqdm(image_files, desc="Processing Test Images"):
    img_path = os.path.join(TEST_IMAGE_DIR, img_file)

    try:
        image_id = int(os.path.splitext(img_file)[0])
    except ValueError:
        print(f"⚠️ 圖片名稱 {img_file} 無法轉為整數 image_id，已跳過")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 無法讀取圖片 {img_file}: {e}")
        continue

    img_w, img_h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img_h, img_w]], device=device)

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=SCORE_THRESHOLD,
        top_k=100
    )[0]

    scores = results["scores"].detach().cpu()
    labels = results["labels"].detach().cpu()
    boxes = results["boxes"].detach().cpu()

    for score, label_idx, box in zip(scores, labels, boxes):
        xmin, ymin, xmax, ymax = box.tolist()

        if CLAMP_BOX_TO_IMAGE:
            xmin = max(0.0, min(xmin, img_w - 1))
            ymin = max(0.0, min(ymin, img_h - 1))
            xmax = max(0.0, min(xmax, img_w - 1))
            ymax = max(0.0, min(ymax, img_h - 1))

        width = xmax - xmin
        height = ymax - ymin

        if width <= 1 or height <= 1:
            continue

        class_idx = int(label_idx.item())

        # 轉回原始 category_id
        if hasattr(model.config, "contig2cat_id"):
            contig2cat_id = model.config.contig2cat_id
            if isinstance(contig2cat_id, dict):
                if str(class_idx) in contig2cat_id:
                    category_id = int(contig2cat_id[str(class_idx)])
                elif class_idx in contig2cat_id:
                    category_id = int(contig2cat_id[class_idx])
                else:
                    category_id = class_idx
            else:
                category_id = class_idx
        else:
            category_id = class_idx

        predictions.append({
            "image_id": image_id,
            "bbox": [
                round(float(xmin), 15),
                round(float(ymin), 15),
                round(float(width), 15),
                round(float(height), 15)
            ],
            "score": round(float(score.item()), 15),
            "category_id": category_id
        })

# ==========================================
# 5. 輸出 JSON
# ==========================================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print(f"\n成功產生 {OUTPUT_JSON}")
print(f"總共輸出 {len(predictions)} 個預測框")