import os
import csv
import base64
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

from super_image import MsrnModel, EdsrModel, DrlnModel


# =========================================================
# Config
# =========================================================
TEST_LR_DIR = "./data_sr/test/lr"
SAMPLE_SUBMISSION = "./data_sr/sample_submission.csv"

# 多個 checkpoint 放這裡
ENSEMBLE_CKPTS = [
    {
        "path": "./outputs_super_image_drln_retrain_vnew/28.32best.pth",
        "weight": 0.9,
    },
    {
        "path": "./outputs_super_image_drln_finetune_v3_from_f26055/best.pth",
        "weight": 1.0,
    },
    # 可繼續加
    # {
    #     "path": "./outputs_super_image_edsr/best.pth",
    #     "weight": 0.8,
    # },
]

OUT_CSV = "./submission_voting_ensemble_tta_x8_v1.csv"

# Fusion mode:
# "mean"          : 平均 voting，最常用
# "weighted_mean" : 依照上面 weight 做加權平均
# "median"        : pixel-level median，較像 voting，抗 outlier
FUSION_MODE = "mean"

# Tile inference
USE_TILE = True
TILE_SIZE = 256
TILE_OVERLAP = 64

# Test-time augmentation
USE_TTA = True
TTA_MODE = "x8"   # "none", "x4", "x8"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Build model
# =========================================================
def build_model(model_name, scale):
    if model_name == "msrn":
        model = MsrnModel.from_pretrained("eugenesiow/msrn", scale=scale)

    elif model_name == "edsr":
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)

    elif model_name == "drln":
        model = DrlnModel.from_pretrained("eugenesiow/drln", scale=scale)

    elif model_name == "drln-bam":
        model = DrlnModel.from_pretrained("eugenesiow/drln-bam", scale=scale)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# =========================================================
# Image tensor utils
# =========================================================
def pil_to_tensor_rgb(img):
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def tensor_to_uint8_rgb(x):
    x = torch.clamp(x, 0, 1)
    arr = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


# =========================================================
# Kaggle encode
# =========================================================
def encode(img: np.ndarray) -> bytes:
    img_to_encode = img.astype(np.uint8)
    img_to_encode = img_to_encode.flatten()
    img_to_encode = np.append(img_to_encode, -1)

    cnt, rle = 1, []

    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1

            if cnt > 255:
                rle += [int(img_to_encode[i - 1]), 255]
                cnt = 1
        else:
            rle += [int(img_to_encode[i - 1]), cnt]
            cnt = 1

    compressed = zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)
    base64_bytes = base64.b64encode(compressed)

    return base64_bytes


def encode_rgb_prediction_to_str(sr_rgb: np.ndarray) -> str:
    sr_bgr = sr_rgb[:, :, ::-1].copy()
    encoded = encode(sr_bgr)
    return str(encoded)


# =========================================================
# Tile inference
# =========================================================
@torch.no_grad()
def forward_tiled(model, x, scale=4, tile_size=256, overlap=32):
    b, c, h, w = x.shape
    device = x.device

    if tile_size <= 0:
        return model(x)

    output = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=x.dtype,
    )

    weight = torch.zeros_like(output)

    stride = tile_size - overlap
    if stride <= 0:
        stride = tile_size

    for y in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y + tile_size, h)
            x1 = min(x0 + tile_size, w)

            patch = x[:, :, y:y1, x0:x1]
            sr_patch = model(patch)

            oy0 = y * scale
            ox0 = x0 * scale
            oy1 = y1 * scale
            ox1 = x1 * scale

            output[:, :, oy0:oy1, ox0:ox1] += sr_patch
            weight[:, :, oy0:oy1, ox0:ox1] += 1.0

    output = output / torch.clamp(weight, min=1e-6)
    return output


@torch.no_grad()
def forward_once(model, x, scale=4, use_tile=True, tile_size=256, overlap=32):
    if use_tile:
        return forward_tiled(
            model=model,
            x=x,
            scale=scale,
            tile_size=tile_size,
            overlap=overlap,
        )
    return model(x)


# =========================================================
# TTA
# =========================================================
@torch.no_grad()
def forward_x4_tta(model, x, scale=4, use_tile=True, tile_size=256, overlap=32):
    outs = []

    y = forward_once(model, x, scale, use_tile, tile_size, overlap)
    outs.append(y)

    x_aug = torch.flip(x, dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3])
    outs.append(y)

    x_aug = torch.flip(x, dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2])
    outs.append(y)

    x_aug = torch.flip(x, dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3])
    outs.append(y)

    sr = torch.stack(outs, dim=0).mean(dim=0)
    return sr


@torch.no_grad()
def forward_x8_tta(model, x, scale=4, use_tile=True, tile_size=256, overlap=32):
    outs = []

    y = forward_once(model, x, scale, use_tile, tile_size, overlap)
    outs.append(y)

    x_aug = torch.flip(x, dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3])
    outs.append(y)

    x_aug = torch.flip(x, dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2])
    outs.append(y)

    x_aug = torch.flip(x, dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3])
    outs.append(y)

    x_aug = x.transpose(2, 3)
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = y.transpose(2, 3)
    outs.append(y)

    x_aug = torch.flip(x.transpose(2, 3), dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3]).transpose(2, 3)
    outs.append(y)

    x_aug = torch.flip(x.transpose(2, 3), dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2]).transpose(2, 3)
    outs.append(y)

    x_aug = torch.flip(x.transpose(2, 3), dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3]).transpose(2, 3)
    outs.append(y)

    sr = torch.stack(outs, dim=0).mean(dim=0)
    return sr


@torch.no_grad()
def infer_sr(model, x, scale=4):
    mode = TTA_MODE.lower()

    if not USE_TTA or mode == "none":
        return forward_once(
            model=model,
            x=x,
            scale=scale,
            use_tile=USE_TILE,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
        )

    if mode == "x4":
        return forward_x4_tta(
            model=model,
            x=x,
            scale=scale,
            use_tile=USE_TILE,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
        )

    if mode == "x8":
        return forward_x8_tta(
            model=model,
            x=x,
            scale=scale,
            use_tile=USE_TILE,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
        )

    raise ValueError(f"Unsupported TTA_MODE: {TTA_MODE}")


# =========================================================
# Ensemble / Voting
# =========================================================
def load_one_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = ckpt["model_name"]
    scale = int(ckpt["scale"])

    model = build_model(model_name, scale)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    info = {
        "model": model,
        "model_name": model_name,
        "scale": scale,
        "epoch": ckpt.get("epoch", "unknown"),
        "best_psnr": ckpt.get("best_psnr", "unknown"),
    }

    return info


@torch.no_grad()
def ensemble_infer(models_info, x):
    preds = []
    weights = []

    base_scale = models_info[0]["scale"]

    for item in models_info:
        model = item["model"]
        scale = item["scale"]
        weight = float(item["weight"])

        if scale != base_scale:
            raise ValueError(
                f"Scale mismatch: got {scale}, expected {base_scale}. "
                "All ensemble models must use the same scale."
            )

        sr = infer_sr(model, x, scale=scale)
        sr = torch.clamp(sr, 0, 1)

        preds.append(sr)
        weights.append(weight)

    stack = torch.stack(preds, dim=0)
    # shape: [N, 1, 3, H, W]

    mode = FUSION_MODE.lower()

    if mode == "mean":
        fused = stack.mean(dim=0)

    elif mode == "median":
        fused = stack.median(dim=0).values

    elif mode == "weighted_mean":
        w = torch.tensor(weights, device=stack.device, dtype=stack.dtype)
        w = w / torch.clamp(w.sum(), min=1e-8)
        w = w.view(-1, 1, 1, 1, 1)
        fused = (stack * w).sum(dim=0)

    else:
        raise ValueError(
            f"Unsupported FUSION_MODE: {FUSION_MODE}. "
            "Use 'mean', 'weighted_mean', or 'median'."
        )

    fused = torch.clamp(fused, 0, 1)
    return fused


# =========================================================
# Main
# =========================================================
def main():
    print("Device:", DEVICE)
    print("USE_TILE:", USE_TILE)
    print("TILE_SIZE:", TILE_SIZE)
    print("TILE_OVERLAP:", TILE_OVERLAP)
    print("USE_TTA:", USE_TTA)
    print("TTA_MODE:", TTA_MODE)
    print("FUSION_MODE:", FUSION_MODE)
    print("Num ensemble ckpts:", len(ENSEMBLE_CKPTS))

    sample = pd.read_csv(SAMPLE_SUBMISSION)

    print("Sample submission:")
    print(sample.head())
    print("Rows:", len(sample))

    # Load ensemble models
    models_info = []

    for cfg in ENSEMBLE_CKPTS:
        ckpt_path = cfg["path"]
        weight = float(cfg.get("weight", 1.0))

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        item = load_one_model(ckpt_path)
        item["path"] = ckpt_path
        item["weight"] = weight

        models_info.append(item)

        print("-" * 80)
        print("Loaded:", ckpt_path)
        print("Model:", item["model_name"])
        print("Scale:", item["scale"])
        print("Epoch:", item["epoch"])
        print("Best PSNR:", item["best_psnr"])
        print("Weight:", weight)

    # Check scale consistency
    scales = [m["scale"] for m in models_info]
    if len(set(scales)) != 1:
        raise ValueError(f"All models must have same scale, got: {scales}")

    rows = []

    with torch.no_grad():
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Voting Inference"):
            idx = int(row["id"])
            filename = str(row["filename"])

            img_path = Path(TEST_LR_DIR) / filename

            if not img_path.exists():
                raise FileNotFoundError(f"Missing test image: {img_path}")

            img = Image.open(img_path).convert("RGB")
            x = pil_to_tensor_rgb(img).to(DEVICE)

            sr = ensemble_infer(models_info, x)
            sr_rgb = tensor_to_uint8_rgb(sr)

            rle_str = encode_rgb_prediction_to_str(sr_rgb)

            rows.append({
                "id": idx,
                "filename": filename,
                "rle": rle_str,
            })

    submission = pd.DataFrame(rows)
    submission = submission[["id", "filename", "rle"]]

    submission.to_csv(
        OUT_CSV,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    print("=" * 80)
    print("Saved:", OUT_CSV)
    print(submission.head())
    print("=" * 80)

    unique_rle = submission["rle"].nunique()
    print("Unique rle count:", unique_rle)

    if unique_rle <= 1:
        print("Warning: all RLE strings are identical. Please check inference output.")


if __name__ == "__main__":
    main()