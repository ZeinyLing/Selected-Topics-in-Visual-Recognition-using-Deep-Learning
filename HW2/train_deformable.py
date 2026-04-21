import os
import copy
import json
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    DeformableDetrConfig,
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from pycocotools.cocoeval import COCOeval


# =========================================================
# 0. SETTINGS
# =========================================================
SEED = 42
TRAIN_IMG_DIR = "./dataset/train"
TRAIN_JSON = "./dataset/annotations/train.json"
VAL_IMG_DIR = "./dataset/val"
VAL_JSON = "./dataset/annotations/val.json"
OUTPUT_DIR = "./deformable-detr-my-coco-model"

PROCESSOR_NAME = "SenseTime/deformable-detr"

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 50

# 若驗證集太大，可先設成 200 / 500；None = 完整驗證集
MAX_EVAL_SAMPLES = None

# mAP 時計算檢測結果時的 threshold
MAP_THRESHOLD = 0.05

# 嘗試只讓 backbone 用 pretrained
TRY_PRETRAINED_BACKBONE = True

device =  torch.device("cuda:0")
print(f"目前使用裝置: {device}")


# =========================================================
# 1. DATASET
# =========================================================
class DetrCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.cat_id_map = None

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        w_img, h_img = img.size
        valid_targets = []

        for t in target:
            x, y, w, h = t["bbox"]

            # 基本檢查
            if w <= 1 or h <= 1:
                continue

            # clamp 到 image 範圍
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            if w <= 1 or h <= 1:
                continue

            new_t = copy.deepcopy(t)
            new_t["category_id"] = self.cat_id_map[t["category_id"]]
            new_t["bbox"] = [x, y, w, h]
            valid_targets.append(new_t)

        image_id = self.ids[idx]
        target_dict = {
            "image_id": image_id,
            "annotations": valid_targets
        }

        encoding = self.processor(
            images=img,
            annotations=target_dict,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0]
        }


# =========================================================
# 2. COLLATE FN
# =========================================================
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_h = max(img.shape[1] for img in pixel_values)
    max_w = max(img.shape[2] for img in pixel_values)

    batch_pixel_values = []
    batch_pixel_mask = []

    for img in pixel_values:
        _, h, w = img.shape

        padded_img = torch.zeros((3, max_h, max_w), dtype=img.dtype)
        padded_img[:, :h, :w] = img
        batch_pixel_values.append(padded_img)

        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        batch_pixel_mask.append(mask)

    return {
        "pixel_values": torch.stack(batch_pixel_values),
        "pixel_mask": torch.stack(batch_pixel_mask),
        "labels": labels
    }


# =========================================================
# 3. CURVE PLOTS
# =========================================================
def plot_loss_curve(log_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_df = pd.DataFrame(log_history)

    csv_path = os.path.join(save_dir, "training_log.csv")
    fig_path = os.path.join(save_dir, "loss_curve.png")

    log_df.to_csv(csv_path, index=False)
    print(f"訓練紀錄已儲存: {csv_path}")

    plt.figure(figsize=(10, 6))

    if "loss" in log_df.columns:
        train_df = log_df.dropna(subset=["loss"]).copy()
        if len(train_df) > 0 and "step" in train_df.columns:
            plt.plot(
                train_df["step"],
                train_df["loss"],
                label="Train Loss",
                marker="o",
                markersize=2
            )

    if "eval_loss" in log_df.columns:
        eval_df = log_df.dropna(subset=["eval_loss"]).copy()
        if len(eval_df) > 0 and "step" in eval_df.columns:
            plt.plot(
                eval_df["step"],
                eval_df["eval_loss"],
                label="Validation Loss",
                marker="s",
                markersize=3
            )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Loss 曲線已儲存: {fig_path}")
    plt.show()


def plot_map_curve(map_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if len(map_history) == 0:
        print("沒有 mAP 紀錄，略過 map curve。")
        return

    map_df = pd.DataFrame(map_history)
    csv_path = os.path.join(save_dir, "map_log.csv")
    fig_path = os.path.join(save_dir, "map_curve.png")

    map_df.to_csv(csv_path, index=False)
    print(f"mAP 紀錄已儲存: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        map_df["step"],
        map_df["mAP_50_95"],
        label="mAP@0.5:0.95",
        marker="o",
        markersize=4
    )

    plt.xlabel("Step")
    plt.ylabel("mAP")
    plt.title("Validation mAP@0.5:0.95 Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"mAP 曲線已儲存: {fig_path}")
    plt.show()


# =========================================================
# 4. COCO mAP EVALUATION
# =========================================================
def evaluate_map(model, processor, dataset, device, threshold=0.05, max_eval_samples=None):
    model.eval()

    coco_gt = dataset.coco
    coco_results = []

    total_samples = len(dataset) if max_eval_samples is None else min(len(dataset), max_eval_samples)
    print(f"開始計算 COCO mAP，樣本數: {total_samples}")

    for idx in range(total_samples):
        image_id = dataset.ids[idx]
        img_info = coco_gt.loadImgs(image_id)[0]
        image = dataset._load_image(image_id)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        h, w = img_info["height"], img_info["width"]
        target_sizes = torch.tensor([[h, w]], device=device)

        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            top_k=100
        )[0]

        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()
        boxes = results["boxes"].detach().cpu()

        for score, label, box in zip(scores, labels, boxes):
            xmin, ymin, xmax, ymax = box.tolist()

            width = xmax - xmin
            height = ymax - ymin

            if width <= 1 or height <= 1:
                continue

            class_idx = int(label.item())
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

            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(xmin), float(ymin), float(width), float(height)],
                "score": float(score.item())
            })

    if len(coco_results) == 0:
        print("沒有任何預測框，mAP 設為 0.0")
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return float(coco_eval.stats[0])  # mAP@[0.5:0.95]


class MAPCallback(TrainerCallback):
    def __init__(self, processor, val_dataset, device, max_eval_samples=None, threshold=0.05):
        self.processor = processor
        self.val_dataset = val_dataset
        self.device = device
        self.max_eval_samples = max_eval_samples
        self.threshold = threshold
        self.map_history = []

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        print("\n==============================")
        print("開始計算驗證集 mAP@0.5:0.95")
        print("==============================")

        mAP = evaluate_map(
            model=model,
            processor=self.processor,
            dataset=self.val_dataset,
            device=self.device,
            threshold=self.threshold,
            max_eval_samples=self.max_eval_samples
        )

        print(f"step {state.global_step} | mAP@0.5:0.95 = {mAP:.6f}")

        self.map_history.append({
            "step": int(state.global_step),
            "mAP_50_95": float(mAP)
        })


# =========================================================
# 5. MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Processor
    # -----------------------------
    processor = DeformableDetrImageProcessor.from_pretrained(PROCESSOR_NAME)

    # -----------------------------
    # Dataset
    # -----------------------------
    print("正在載入資料集...")
    train_dataset = DetrCocoDataset(
        img_folder=TRAIN_IMG_DIR,
        ann_file=TRAIN_JSON,
        processor=processor
    )
    val_dataset = DetrCocoDataset(
        img_folder=VAL_IMG_DIR,
        ann_file=VAL_JSON,
        processor=processor
    )

    print(f"訓練集數量: {len(train_dataset)}")
    print(f"驗證集數量: {len(val_dataset)}")

    # -----------------------------
    # 類別映射
    # -----------------------------
    cats = train_dataset.coco.cats
    sorted_cat_ids = sorted(cats.keys())

    id2label = {i: cats[cat_id]["name"] for i, cat_id in enumerate(sorted_cat_ids)}
    label2id = {cats[cat_id]["name"]: i for i, cat_id in enumerate(sorted_cat_ids)}

    cat_id_map = {cat_id: i for i, cat_id in enumerate(sorted_cat_ids)}
    contig2cat_id = {i: cat_id for i, cat_id in enumerate(sorted_cat_ids)}

    train_dataset.cat_id_map = cat_id_map
    val_dataset.cat_id_map = cat_id_map

    print(f"偵測到類別: {id2label}")

    # -----------------------------
    # Config
    # -----------------------------
    config = DeformableDetrConfig.from_pretrained(PROCESSOR_NAME)

    config.num_labels = len(sorted_cat_ids)
    config.id2label = id2label
    config.label2id = label2id
    config.contig2cat_id = contig2cat_id

    # Deformable DETR 常見預設
    config.num_queries = 300
    config.encoder_layers = 6
    config.decoder_layers = 6
    config.dropout = 0.1
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    config.num_feature_levels = 4
    config.encoder_n_points = 4
    config.decoder_n_points = 4

    # 可加速/加強收斂的常見選項
    config.with_box_refine = True
    config.two_stage = False
    config.auxiliary_loss = True

    # matching / loss
    config.class_cost = 2
    config.bbox_cost = 5
    config.giou_cost = 2
    config.bbox_loss_coefficient = 5
    config.giou_loss_coefficient = 2
    config.eos_coefficient = 0.1

    # 嘗試只開 backbone pretrained
    if TRY_PRETRAINED_BACKBONE:
        if hasattr(config, "use_pretrained_backbone"):
            config.use_pretrained_backbone = True
            print("偵測到 use_pretrained_backbone，已設為 True")
        else:
            print("目前這版 transformers 的 DeformableDetrConfig 沒明確暴露 use_pretrained_backbone；")
            print("本程式會避免載入完整 detector checkpoint，但 backbone 是否單獨 pretrained 取決於你的安裝版本。")

    # -----------------------------
    # Model
    # 關鍵：不要用 DeformableDetrForObjectDetection.from_pretrained(PROCESSOR_NAME)
    # 這樣至少不會把整個 detector checkpoint 載入
    # -----------------------------
    model = DeformableDetrForObjectDetection(config)
    model.to(device)

    print("模型建立完成")
    print("encoder_layers =", model.config.encoder_layers)
    print("decoder_layers =", model.config.decoder_layers)
    print("num_queries =", model.config.num_queries)
    print("with_box_refine =", model.config.with_box_refine)
    print("two_stage =", model.config.two_stage)

    # -----------------------------
    # TrainingArguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=0.1,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none"
    )

    # -----------------------------
    # mAP callback
    # -----------------------------
    map_callback = MAPCallback(
        processor=processor,
        val_dataset=val_dataset,
        device=device,
        max_eval_samples=MAX_EVAL_SAMPLES,
        threshold=MAP_THRESHOLD
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks=[map_callback]
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("開始訓練...")
    trainer.train()

    # -----------------------------
    # Save final model
    # -----------------------------
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print(f"模型已儲存至: {final_model_dir}")

    trainer.state.save_to_json(os.path.join(OUTPUT_DIR, "trainer_state.json"))
    print(f"Trainer state 已儲存至: {os.path.join(OUTPUT_DIR, 'trainer_state.json')}")

    # -----------------------------
    # Plot curves
    # -----------------------------
    plot_loss_curve(trainer.state.log_history, OUTPUT_DIR)
    plot_map_curve(map_callback.map_history, OUTPUT_DIR)

    print("全部完成。")