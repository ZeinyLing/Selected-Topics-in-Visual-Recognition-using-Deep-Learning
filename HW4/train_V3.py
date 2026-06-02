import os
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Config
# ============================================================

DATA_ROOT = "./hw4_realse_dataset"

TRAIN_DEGRADED_DIR = os.path.join(DATA_ROOT, "train", "degraded")
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train", "clean")

SAVE_DIR = "./outputs_promptir_V3"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_promptir_v3.pth")

LOG_CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")
LOSS_FIG_PATH = os.path.join(SAVE_DIR, "loss_curve.png")
PSNR_FIG_PATH = os.path.join(SAVE_DIR, "psnr_curve.png")

IMG_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 2

LR = 8e-5
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MIXED_LOSS = False
MSE_WEIGHT = 0.0


# ============================================================
# Utils
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image(path):
    return Image.open(path).convert("RGB")


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def calculate_psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target, reduction="mean")

    if mse.item() == 0:
        return 100.0

    psnr = 20 * torch.log10(
        torch.tensor(1.0, device=pred.device) / torch.sqrt(mse + eps)
    )

    return psnr.item()


# ============================================================
# Dataset
# ============================================================

class RestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, img_size=256, augment=True):
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.img_size = img_size
        self.augment = augment

        self.pairs = []

        degraded_files = sorted(list(self.degraded_dir.glob("*.png")))

        for deg_path in degraded_files:
            name = deg_path.name

            if name.startswith("rain-"):
                idx = name.replace("rain-", "").replace(".png", "")
                clean_name = f"rain_clean-{idx}.png"

            elif name.startswith("snow-"):
                idx = name.replace("snow-", "").replace(".png", "")
                clean_name = f"snow_clean-{idx}.png"

            else:
                continue

            clean_path = self.clean_dir / clean_name

            if clean_path.exists():
                self.pairs.append((deg_path, clean_path))
            else:
                print(f"[Warning] Missing clean image: {clean_path}")

        print(f"Loaded image pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def random_crop(self, degraded, clean):
        w, h = degraded.size

        if w < self.img_size or h < self.img_size:
            new_w = max(w, self.img_size)
            new_h = max(h, self.img_size)

            degraded = degraded.resize((new_w, new_h), Image.BICUBIC)
            clean = clean.resize((new_w, new_h), Image.BICUBIC)

            w, h = degraded.size

        left = random.randint(0, w - self.img_size)
        top = random.randint(0, h - self.img_size)

        degraded = degraded.crop(
            (left, top, left + self.img_size, top + self.img_size)
        )
        clean = clean.crop(
            (left, top, left + self.img_size, top + self.img_size)
        )

        return degraded, clean

    def resize_pair(self, degraded, clean):
        degraded = degraded.resize((self.img_size, self.img_size), Image.BICUBIC)
        clean = clean.resize((self.img_size, self.img_size), Image.BICUBIC)
        return degraded, clean

    def augment_pair(self, degraded, clean):
        if random.random() < 0.5:
            degraded = degraded.transpose(Image.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            degraded = degraded.transpose(Image.FLIP_TOP_BOTTOM)
            clean = clean.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            degraded = degraded.rotate(angle)
            clean = clean.rotate(angle)

        return degraded, clean

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]

        degraded = read_image(deg_path)
        clean = read_image(clean_path)

        if self.augment:
            degraded, clean = self.random_crop(degraded, clean)
            degraded, clean = self.augment_pair(degraded, clean)
        else:
            degraded, clean = self.resize_pair(degraded, clean)

        degraded = pil_to_tensor(degraded)
        clean = pil_to_tensor(clean)

        return degraded, clean


# ============================================================
# Strong PromptIR Model
# ============================================================

class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim, layer_norm_type="WithBias"):
        super().__init__()

        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.body(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class PromptGenBlock(nn.Module):
    def __init__(self, dim, prompt_dim=64):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.prompt = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        prompt = self.prompt(self.pool(x))
        prompt_feature = x * prompt

        out = torch.cat([x, prompt_feature], dim=1)
        out = self.fuse(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias
        )

        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=bias
        )

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias
        )

        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)

        return out


class PromptTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        use_prompt=True
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(dim, layer_norm_type)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            bias=bias
        )

        self.norm2 = LayerNorm2d(dim, layer_norm_type)

        self.ffn = FeedForward(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias
        )

        self.use_prompt = use_prompt

        if use_prompt:
            self.prompt = PromptGenBlock(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        if self.use_prompt:
            x = x + self.prompt(x)

        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, inp_channels=3, embed_dim=48, bias=False):
        super().__init__()

        self.proj = nn.Conv2d(
            inp_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                dim,
                dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                dim,
                dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# ============================================================
# V2.2 Modules: Detail + Frequency Refinement
# ============================================================

class DetailRefineBlock(nn.Module):
    """
    Detail refinement block.
    Helps recover local texture and edge details.
    """

    def __init__(self, dim):
        super().__init__()

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        self.edge = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        local = self.local(x)

        edge = x - F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )
        edge = self.edge(edge)

        out = torch.cat([local, edge], dim=1)
        out = self.fuse(out)

        return x + out


class FrequencyEnhanceBlock(nn.Module):
    """
    Frequency-aware enhancement block.
    Uses high-frequency residual decomposition.
    """

    def __init__(self, dim):
        super().__init__()

        self.high_freq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )

        high = x - low
        high = self.high_freq(high)

        out = torch.cat([x, high], dim=1)
        out = self.fuse(out)

        return x + out


class PromptIRStrong(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias"
    ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(
            inp_channels=inp_channels,
            embed_dim=dim,
            bias=bias
        )

        self.encoder_level1 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 4,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 8,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(dim * 8)

        self.reduce_chan_level3 = nn.Conv2d(
            dim * 8,
            dim * 4,
            kernel_size=1,
            bias=bias
        )

        self.decoder_level3 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 4,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4)

        self.reduce_chan_level2 = nn.Conv2d(
            dim * 4,
            dim * 2,
            kernel_size=1,
            bias=bias
        )

        self.decoder_level2 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)

        self.decoder_level1 = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                PromptTransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    use_prompt=True
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        # V2.2: detail + frequency refinement before output head
        self.detail_refine = DetailRefineBlock(dim * 2)
        self.freq_refine = FrequencyEnhanceBlock(dim * 2)

        self.output = nn.Conv2d(
            dim * 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        # Important stabilization:
        # At the beginning, model behaves like identity mapping.
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()

        factor = 8

        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x, h, w

    def forward(self, inp_img):
        inp_img, original_h, original_w = self.check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat(
            [inp_dec_level3, out_enc_level3],
            dim=1
        )
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2],
            dim=1
        )
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1
        )

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        # V2.2: detail + frequency refinement
        out_dec_level1 = self.detail_refine(out_dec_level1)
        out_dec_level1 = self.freq_refine(out_dec_level1)

        out = self.output(out_dec_level1)

        # Global residual learning
        out = out + inp_img

        out = out[:, :, :original_h, :original_w]

        return out


# ============================================================
# Loss
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


def restoration_loss(pred, target, criterion):
    if USE_MIXED_LOSS:
        return criterion(pred, target) + MSE_WEIGHT * F.mse_loss(pred, target)
    else:
        return criterion(pred, target)


# ============================================================
# Save Training Log and Curves
# ============================================================

def save_training_log(history):
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_psnr",
        "val_loss",
        "val_psnr"
    ]

    with open(LOG_CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in history:
            writer.writerow(row)


def save_training_curves(history):
    if len(history) == 0:
        return

    epochs = [x["epoch"] for x in history]

    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]

    train_psnr = [x["train_psnr"] for x in history]
    val_psnr = [x["val_psnr"] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_FIG_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_psnr, label="Train PSNR")
    plt.plot(epochs, val_psnr, label="Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Training and Validation PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PSNR_FIG_PATH, dpi=300)
    plt.close()


# ============================================================
# Train / Valid
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_psnr = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for degraded, clean in pbar:
        degraded = degraded.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        restored = model(degraded)

        # Do not clamp before loss.
        loss = restoration_loss(restored, clean, criterion)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # Clamp only for metric.
        restored_for_metric = torch.clamp(restored, 0, 1)
        psnr = calculate_psnr(restored_for_metric, clean)

        total_loss += loss.item()
        total_psnr += psnr

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "psnr": f"{psnr:.2f}"
        })

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)

    return avg_loss, avg_psnr


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0

    pbar = tqdm(loader, desc="Valid", leave=False)

    for degraded, clean in pbar:
        degraded = degraded.to(DEVICE)
        clean = clean.to(DEVICE)

        restored = model(degraded)

        # Do not clamp before loss.
        loss = restoration_loss(restored, clean, criterion)

        # Clamp only for metric.
        restored_for_metric = torch.clamp(restored, 0, 1)
        psnr = calculate_psnr(restored_for_metric, clean)

        total_loss += loss.item()
        total_psnr += psnr

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "psnr": f"{psnr:.2f}"
        })

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)

    return avg_loss, avg_psnr


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    full_dataset = RestorationDataset(
        degraded_dir=TRAIN_DEGRADED_DIR,
        clean_dir=TRAIN_CLEAN_DIR,
        img_size=IMG_SIZE,
        augment=True
    )

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    train_dataset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_base_dataset = RestorationDataset(
        degraded_dir=TRAIN_DEGRADED_DIR,
        clean_dir=TRAIN_CLEAN_DIR,
        img_size=IMG_SIZE,
        augment=False
    )

    val_dataset = torch.utils.data.Subset(
        val_base_dataset,
        val_subset.indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = PromptIRStrong(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    ).to(DEVICE)

    criterion = CharbonnierLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    best_psnr = -1.0
    history = []

    print("=" * 60)
    print("Training Strong PromptIR V2.2 - Detail + Frequency")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Train size:", len(train_dataset))
    print("Valid size:", len(val_dataset))
    print("Image size:", IMG_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("LR:", LR)
    print("USE_MIXED_LOSS:", USE_MIXED_LOSS)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_psnr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr
        })

        save_training_log(history)
        save_training_curves(history)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Train PSNR: {train_psnr:.2f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_psnr": best_psnr,
                    "img_size": IMG_SIZE,
                    "dim": 48,
                    "num_blocks": [4, 6, 6, 8],
                    "num_refinement_blocks": 4,
                    "heads": [1, 2, 4, 8],
                    "ffn_expansion_factor": 2.66,
                    "lr": LR,
                    "use_mixed_loss": USE_MIXED_LOSS,
                    "model_name": "PromptIRStrong_V22_DetailFreq"
                },
                BEST_MODEL_PATH
            )

            print(f"Saved best model: {BEST_MODEL_PATH}")
            print(f"Best PSNR: {best_psnr:.2f}")

    print("=" * 60)
    print("Training finished")
    print(f"Best PSNR: {best_psnr:.2f}")
    print(f"Saved model: {BEST_MODEL_PATH}")
    print(f"Saved log: {LOG_CSV_PATH}")
    print(f"Saved loss curve: {LOSS_FIG_PATH}")
    print(f"Saved PSNR curve: {PSNR_FIG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()