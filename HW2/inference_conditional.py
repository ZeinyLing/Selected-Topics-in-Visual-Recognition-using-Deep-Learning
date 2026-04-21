import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    ConditionalDetrImageProcessor,
    ConditionalDetrForObjectDetection
)

# ==========================================
# 1. 基本設定
# ==========================================
MODEL_PATH = "./conditional-detr-my-coco-model6/final_model"
TEST_IMAGE_DIR = "./dataset/test"
OUTPUT_JSON = "pred_cond6.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用硬體: {device}")

# 分數閾值
SCORE_THRESHOLD = 0.01

# 是否要把 bbox 限制在影像範圍內
CLAMP_BOX_TO_IMAGE = False

# ==========================================
# 2. 載入模型
# ==========================================
print("正在載入 Conditional DETR 模型...")
processor = ConditionalDetrImageProcessor.from_pretrained(MODEL_PATH)
model = ConditionalDetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()

print("模型載入完成。")
print("id2label =", model.config.id2label)
print("label2id =", model.config.label2id)

predictions = []

# ==========================================
# 3. 讀取測試圖片清單
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
# 4. 推論
# ==========================================
print("開始產生 pred.json ...")

for img_file in tqdm(image_files, desc="Processing Test Images"):
    img_path = os.path.join(TEST_IMAGE_DIR, img_file)

    # image_id 由檔名轉整數
    try:
        image_id = int(os.path.splitext(img_file)[0])
    except ValueError:
        print(f"\n⚠️ 圖片名稱 {img_file} 無法轉為整數 image_id，已跳過")
        continue

    # 讀圖
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"\n⚠️ 無法讀取圖片 {img_file}: {e}")
        continue

    img_w, img_h = image.size

    # 前處理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)

    # 還原到原圖大小
    target_sizes = torch.tensor([[img_h, img_w]], device=device)

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=SCORE_THRESHOLD
    )[0]

    scores = results["scores"].detach().cpu()
    labels = results["labels"].detach().cpu()
    boxes = results["boxes"].detach().cpu()

    for score, label_idx, box in zip(scores, labels, boxes):
        xmin, ymin, xmax, ymax = box.tolist()

        # 可選：把框限制在影像範圍內
        if CLAMP_BOX_TO_IMAGE:
            xmin = max(0.0, min(xmin, img_w - 1))
            ymin = max(0.0, min(ymin, img_h - 1))
            xmax = max(0.0, min(xmax, img_w - 1))
            ymax = max(0.0, min(ymax, img_h - 1))

        width = xmax - xmin
        height = ymax - ymin

        # 過濾非法框
        if width <= 1 or height <= 1:
            continue

        # label idx -> class name
        class_name = model.config.id2label[int(label_idx)]

        # class name -> category_id
        # 這裡對應你訓練時存進 config 的 label2id
        category_id = int(model.config.label2id[class_name])

        predictions.append({
            "image_id": image_id,
            "bbox": [
                round(float(xmin), 15),
                round(float(ymin), 15),
                round(float(width), 15),
                round(float(height), 15)
            ],
            "score": round(float(score.item()), 15),
            "category_id": category_id+1
        })

# ==========================================
# 5. 輸出 JSON
# ==========================================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print(f"\n成功產生 {OUTPUT_JSON}")
print(f"總共輸出 {len(predictions)} 個預測框")