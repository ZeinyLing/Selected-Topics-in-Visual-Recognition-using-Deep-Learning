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
# Config: 直接改這裡
# =========================================================
TEST_LR_DIR = "./data_sr/test/lr"
SAMPLE_SUBMISSION = "./data_sr/sample_submission.csv"

CKPT_PATH = "./outputs_super_image_drln_retrain_vnew/28.32best.pth"
OUT_CSV = "./n28.32submission_tta_x8_64.csv"

# Tile inference: 大圖建議開
USE_TILE = True
TILE_SIZE = 256
TILE_OVERLAP = 64   # 原本 16，可改 32 減少接縫

# Test-time augmentation / self-ensemble
USE_TTA = True
TTA_MODE = "x8"     # "none", "x4", "x8"

# 如果你用 CUDA_VISIBLE_DEVICES=1 指定 GPU，這裡顯示 cuda:0 是正常的
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
    """
    PIL RGB image -> torch tensor [1, 3, H, W], range [0, 1]
    """
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def tensor_to_uint8_rgb(x):
    """
    torch tensor [1, 3, H, W] -> uint8 RGB image [H, W, 3]
    """
    x = torch.clamp(x, 0, 1)
    arr = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


# =========================================================
# Official Kaggle encode
# =========================================================
def encode(img: np.ndarray) -> bytes:
    """
    Lossless encoding of images for submission on Kaggle platform.

    Parameters
    ----------
    img : np.ndarray
        BGR image in (h, w, c) format, c = 3.
        This follows the official example using cv2.imread().

    Returns
    -------
    bytes
        Encoded image as base64 bytes.
    """
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
    """
    Model output is RGB.
    Official example uses cv2.imread(), which is BGR.
    Therefore RGB must be converted to BGR before encode().
    """
    sr_bgr = sr_rgb[:, :, ::-1].copy()
    encoded = encode(sr_bgr)
    return str(encoded)


# =========================================================
# Tile inference
# =========================================================
@torch.no_grad()
def forward_tiled(model, x, scale=4, tile_size=256, overlap=32):
    """
    x: [1, 3, H, W]
    return: [1, 3, H*scale, W*scale]
    """
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
# TTA / self-ensemble inference
# =========================================================
@torch.no_grad()
def forward_x4_tta(model, x, scale=4, use_tile=True, tile_size=256, overlap=32):
    """
    x4 TTA:
    original, horizontal flip, vertical flip, horizontal+vertical flip
    """
    outs = []

    # 1. original
    y = forward_once(model, x, scale, use_tile, tile_size, overlap)
    outs.append(y)

    # 2. horizontal flip
    x_aug = torch.flip(x, dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3])
    outs.append(y)

    # 3. vertical flip
    x_aug = torch.flip(x, dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2])
    outs.append(y)

    # 4. horizontal + vertical flip
    x_aug = torch.flip(x, dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3])
    outs.append(y)

    sr = torch.stack(outs, dim=0).mean(dim=0)
    return sr


@torch.no_grad()
def forward_x8_tta(model, x, scale=4, use_tile=True, tile_size=256, overlap=32):
    """
    x8 TTA:
    original / flips / transpose variants.
    較慢，但通常最穩。
    """
    outs = []

    # 1. original
    y = forward_once(model, x, scale, use_tile, tile_size, overlap)
    outs.append(y)

    # 2. horizontal flip
    x_aug = torch.flip(x, dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3])
    outs.append(y)

    # 3. vertical flip
    x_aug = torch.flip(x, dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2])
    outs.append(y)

    # 4. horizontal + vertical flip
    x_aug = torch.flip(x, dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3])
    outs.append(y)

    # 5. transpose
    x_aug = x.transpose(2, 3)
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = y.transpose(2, 3)
    outs.append(y)

    # 6. transpose + horizontal flip
    x_aug = torch.flip(x.transpose(2, 3), dims=[3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[3]).transpose(2, 3)
    outs.append(y)

    # 7. transpose + vertical flip
    x_aug = torch.flip(x.transpose(2, 3), dims=[2])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2]).transpose(2, 3)
    outs.append(y)

    # 8. transpose + horizontal + vertical flip
    x_aug = torch.flip(x.transpose(2, 3), dims=[2, 3])
    y = forward_once(model, x_aug, scale, use_tile, tile_size, overlap)
    y = torch.flip(y, dims=[2, 3]).transpose(2, 3)
    outs.append(y)

    sr = torch.stack(outs, dim=0).mean(dim=0)
    return sr


@torch.no_grad()
def infer_sr(model, x, scale=4):
    """
    Unified inference function.
    """
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

    raise ValueError(f"Unsupported TTA_MODE: {TTA_MODE}. Use 'none', 'x4', or 'x8'.")


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

    sample = pd.read_csv(SAMPLE_SUBMISSION)

    print("Sample submission:")
    print(sample.head())
    print("Rows:", len(sample))

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    model_name = ckpt["model_name"]
    scale = int(ckpt["scale"])

    print("Model:", model_name)
    print("Scale:", scale)
    print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
    print("Best PSNR:", ckpt.get("best_psnr", "unknown"))

    model = build_model(model_name, scale)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    rows = []

    with torch.no_grad():
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Inference"):
            idx = int(row["id"])
            filename = str(row["filename"])

            img_path = Path(TEST_LR_DIR) / filename

            if not img_path.exists():
                raise FileNotFoundError(f"Missing test image: {img_path}")

            # Model input uses RGB
            img = Image.open(img_path).convert("RGB")
            x = pil_to_tensor_rgb(img).to(DEVICE)

            sr = infer_sr(model, x, scale=scale)
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