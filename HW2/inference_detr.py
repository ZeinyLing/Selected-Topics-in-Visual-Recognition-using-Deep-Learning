import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


# =========================================================
# 0. 基本設定
# =========================================================
MODEL_DIR = "./detr_scratch_transformer_pretrained_backbone/final_model"
TEST_IMG_DIR = "./dataset/test"
OUTPUT_JSON = "pred.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"目前使用裝置: {DEVICE}")

# 分數閾值，可自行調整
SCORE_THRESHOLD = 0.001

# 是否將 bbox clamp 到影像範圍內
CLAMP_BOX_TO_IMAGE = False


# =========================================================
# 1. 載入 processor / model
# =========================================================
print("正在載入模型...")
processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
model = DetrForObjectDetection.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("模型載入完成")


# =========================================================
# 2. 載入 label mapping
#    你訓練時有存 label_mappings.json
# =========================================================
mapping_path = os.path.join(MODEL_DIR, "label_mappings.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

contiguous_to_cat_id = {int(k): int(v) for k, v in meta["contiguous_to_cat_id"].items()}
id2label = {int(k): v for k, v in meta["id2label"].items()}

print("類別映射載入完成")
print("contiguous_to_cat_id =", contiguous_to_cat_id)
print("id2label =", id2label)


# =========================================================
# 3. 檔名轉 image_id
#    預設假設 test 檔名像 1.jpg / 2.png
# =========================================================
def filename_to_image_id(filename):
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem)
    except ValueError:
        raise ValueError(
            f"檔名 {filename} 無法直接轉成 int image_id，"
            "若你的 image_id 不是檔名，請改成由 test annotation 對應。"
        )


# =========================================================
# 4. 單張圖推論
# =========================================================
@torch.no_grad()
def run_inference_on_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

    # target_sizes: (batch_size, 2) -> (height, width)
    target_sizes = torch.tensor([[h, w]], device=DEVICE)

    results = processor.post_process_object_detection(
        outputs,
        threshold=SCORE_THRESHOLD,
        target_sizes=target_sizes
    )[0]

    return results, w, h


# =========================================================
# 5. 全部 test 圖片推論
# =========================================================
image_files = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
])

print(f"找到 {len(image_files)} 張測試圖片")
print("開始進行 inference...")

predictions = []

for file_name in tqdm(image_files):
    image_path = os.path.join(TEST_IMG_DIR, file_name)
    image_id = filename_to_image_id(file_name)

    results, img_w, img_h = run_inference_on_image(image_path)

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    for box, score, category_id in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.tolist()

        if CLAMP_BOX_TO_IMAGE:
            xmin = max(0.0, min(xmin, img_w))
            ymin = max(0.0, min(ymin, img_h))
            xmax = max(0.0, min(xmax, img_w))
            ymax = max(0.0, min(ymax, img_h))

        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            continue

        # -------------------------------------------------
        # 安全寫法：用 mapping 轉回原始 category_id
        # -------------------------------------------------
        original_category_id = contiguous_to_cat_id[int(category_id)]

        predictions.append({
            "image_id": int(image_id),
            "bbox": [
                round(float(xmin), 15),
                round(float(ymin), 15),
                round(float(width), 15),
                round(float(height), 15)
            ],
            "score": round(float(score.item()), 15),
            "category_id": int(original_category_id)
        })

        # -------------------------------------------------
        # 如果你非常確定原始類別就是 1~N 連號
        # 才可以改成下面這種：
        #
        # predictions.append({
        #     "image_id": int(image_id),
        #     "bbox": [
        #         round(float(xmin), 15),
        #         round(float(ymin), 15),
        #         round(float(width), 15),
        #         round(float(height), 15)
        #     ],
        #     "score": round(float(score.item()), 15),
        #     "category_id": int(category_id) + 1
        # })
        # -------------------------------------------------


# =========================================================
# 6. 儲存 JSON
# =========================================================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"推論完成，共輸出 {len(predictions)} 筆框")
print(f"結果已儲存至: {OUTPUT_JSON}")