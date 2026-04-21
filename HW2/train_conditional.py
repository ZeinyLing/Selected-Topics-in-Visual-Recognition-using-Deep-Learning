import os
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrImageProcessor,
    ConditionalDetrForObjectDetection,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. Dataset
# ==========================================
class DetrCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        w_img, h_img = img.size

        valid_targets = []

        for t in target:
            x, y, w, h = t["bbox"]

            # 1. 基本檢查
            if w <= 1 or h <= 1:
                continue

            # 2. clamp 到 image 範圍
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))

            w = min(w, w_img - x)
            h = min(h, h_img - y)

            if w <= 1 or h <= 1:
                continue

            # 3. category 修正（超重要）
            t["category_id"] = self.cat_id_map[t["category_id"]]

            t["bbox"] = [x, y, w, h]
            valid_targets.append(t)

        if len(valid_targets) == 0:
            # 避免 empty label crash
            valid_targets = []

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
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": encoding["labels"][0]
        }

# ==========================================
# 2. collate_fn
# ==========================================
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

# ==========================================
# 3. 畫 loss
# ==========================================
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

# ==========================================
# 4. Main
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"目前使用裝置: {device}")

    # --------------------------------------
    # 路徑
    # --------------------------------------
    TRAIN_IMG_DIR = "./dataset/train"
    TRAIN_JSON = "./dataset/annotations/train.json"
    VAL_IMG_DIR = "./dataset/val"
    VAL_JSON = "./dataset/annotations/val.json"

    output_dir = "./conditional-detr-my-coco-model4"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------
    # Processor
    # --------------------------------------
    processor_name = "microsoft/conditional-detr-resnet-50"
    processor = ConditionalDetrImageProcessor.from_pretrained(processor_name)

    # --------------------------------------
    # Dataset
    # --------------------------------------
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

    # --------------------------------------
    # 類別映射
    # --------------------------------------
    cats = train_dataset.coco.cats

    id2label = {i: v["name"] for i, v in enumerate(cats.values())}
    label2id = {v: k for k, v in id2label.items()}

    # ⭐ 關鍵：建立 mapping
    cat_id_map = {k: i for i, k in enumerate(cats.keys())}

    train_dataset.cat_id_map = cat_id_map
    val_dataset.cat_id_map = cat_id_map
    print(f"偵測到類別: {id2label}")

    # --------------------------------------
    # Config：加強模型架構
    # --------------------------------------
    config = ConditionalDetrConfig.from_pretrained(processor_name)

    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    # 可以加強的幾個地方
    config.num_queries = 300          # DETR 常用 query 數，可先維持或加大
    config.encoder_layers = 6         # 可改 6~8
    config.decoder_layers = 6         # 可改 6~8
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    # bbox / giou / class 損失權重可依資料集調
    # config.bbox_loss_coefficient = 5
    # config.giou_loss_coefficient = 2
    # config.class_cost = 1
    # config.bbox_cost = 5
    # config.giou_cost = 2

    # --------------------------------------
    # Model
    # --------------------------------------
    model = ConditionalDetrForObjectDetection.from_pretrained(
        processor_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # --------------------------------------
    # TrainingArguments
    # 保留 steps 設定
    # --------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        num_train_epochs=25,

        learning_rate=1e-5,
        weight_decay=1e-4,
        max_grad_norm=0.1,

        fp16=False,

        logging_steps=50,

        eval_strategy="steps",
        save_steps=1000,
        eval_steps=1000,

        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none"
    )

    # --------------------------------------
    # Trainer
    # --------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # --------------------------------------
    # Train
    # --------------------------------------
    print("開始訓練...")
    trainer.train()

    # --------------------------------------
    # Save final model
    # --------------------------------------
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print(f"模型已儲存至: {final_model_dir}")

    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
    print(f"Trainer state 已儲存至: {os.path.join(output_dir, 'trainer_state.json')}")

    # --------------------------------------
    # Plot loss
    # --------------------------------------
    plot_loss_curve(trainer.state.log_history, output_dir)

    print("全部完成。")