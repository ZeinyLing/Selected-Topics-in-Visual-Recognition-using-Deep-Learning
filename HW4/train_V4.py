import os
import csv
import copy
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

SAVE_DIR = "./outputs_promptir_V4"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_promptir_v4.pth")

LOG_CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")
LOSS_FIG_PATH = os.path.join(SAVE_DIR, "loss_curve.png")
PSNR_FIG_PATH = os.path.join(SAVE_DIR, "psnr_curve.png")

IMG_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 2

LR = 5e-5
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MIXED_LOSS = False
MSE_WEIGHT = 0.05

USE_EMA = True
EMA_DECAY = 0.999


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
        # Safer augmentation for rain/snow restoration.
        # Avoid 90/180/270 rotations because rain streak direction can be meaningful.
        if random.random() < 0.5:
            degraded = degraded.transpose(Image.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

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
# Restormer-style Transformer Block
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

        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias"
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# PromptIR V4.1 Modules
# ============================================================

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class PromptGenBlockV4(nn.Module):
    def __init__(
        self,
        features_input_dim,
        num_prompt_components,
        prompt_channel_dim,
        base_prompt_hw,
        bias=False
    ):
        super().__init__()

        if isinstance(base_prompt_hw, int):
            base_prompt_h, base_prompt_w = base_prompt_hw, base_prompt_hw
        else:
            base_prompt_h, base_prompt_w = base_prompt_hw

        self.prompt_components = nn.Parameter(
            torch.randn(
                1,
                num_prompt_components,
                prompt_channel_dim,
                base_prompt_h,
                base_prompt_w
            ) * 0.02
        )

        self.weight_generator = nn.Linear(
            features_input_dim,
            num_prompt_components
        )

        self.final_conv = nn.Conv2d(
            prompt_channel_dim,
            prompt_channel_dim,
            kernel_size=3,
            padding=1,
            bias=bias
        )

    def forward(self, decoder_features):
        b, _, h, w = decoder_features.shape

        pooled_features = F.adaptive_avg_pool2d(
            decoder_features,
            (1, 1)
        ).view(b, -1)

        prompt_weights = F.softmax(
            self.weight_generator(pooled_features),
            dim=1
        )

        weighted_prompts = (
            prompt_weights.view(b, -1, 1, 1, 1)
            * self.prompt_components
        )

        summed_prompt = torch.sum(weighted_prompts, dim=1)

        interpolated_prompt = F.interpolate(
            summed_prompt,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        output_prompt = self.final_conv(interpolated_prompt)

        return output_prompt


class PromptInteractionBlockV4(nn.Module):
    def __init__(
        self,
        feature_dim,
        prompt_dim,
        num_transformer_heads,
        ffn_expansion_factor,
        bias
    ):
        super().__init__()

        concat_dim = feature_dim + prompt_dim

        self.transformer = TransformerBlock(
            dim=concat_dim,
            num_heads=num_transformer_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias
        )

        self.channel_adjust_conv = nn.Conv2d(
            concat_dim,
            feature_dim,
            kernel_size=1,
            bias=bias
        )

        # Zero-start interaction for stability.
        self.gamma = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))

    def forward(self, features, prompt):
        x = torch.cat([features, prompt], dim=1)
        x = self.transformer(x)
        x = self.channel_adjust_conv(x)

        return features + x * self.gamma


class SkipFusionGate(nn.Module):
    def __init__(self, dec_dim, skip_dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(dec_dim + skip_dim, skip_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(skip_dim, skip_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, dec, skip):
        g = self.gate(torch.cat([dec, skip], dim=1))
        skip = skip * g
        return torch.cat([dec, skip], dim=1)


class DetailRefineBlock(nn.Module):
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


# ============================================================
# PromptIR V4.1 Model
# ============================================================

class PromptIR_V41(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_dim=48,
        num_blocks_per_level=(3, 4, 4, 6),
        num_refinement_blocks=4,
        num_prompt_components=5,
        pg_prompt_dim_map=(256, 128, 64),
        pg_base_hw_map=(16, 32, 64),
        backbone_num_attn_heads=(1, 2, 4, 8),
        prompt_interaction_num_attn_heads=8,
        ffn_expansion_factor=2.66,
        bias=False
    ):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)

        if isinstance(backbone_num_attn_heads, int):
            backbone_num_attn_heads = [backbone_num_attn_heads] * self.num_levels

        self.initial_conv = nn.Conv2d(
            in_channels,
            base_dim,
            kernel_size=3,
            padding=1,
            bias=bias
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_skip_dims = []

        current_dim = base_dim

        for i in range(self.num_levels):
            self.encoder_skip_dims.append(current_dim)

            blocks = nn.Sequential(*[
                TransformerBlock(
                    dim=current_dim,
                    num_heads=backbone_num_attn_heads[i],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias
                )
                for _ in range(num_blocks_per_level[i])
            ])

            self.encoder_blocks.append(blocks)

            if i < self.num_levels - 1:
                self.downsamples.append(
                    Downsample(current_dim, current_dim * 2)
                )
                current_dim *= 2

        # Decoder
        self.upsamples = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.prompt_gens = nn.ModuleList()
        self.prompt_interactions = nn.ModuleList()

        for i in range(self.num_levels - 1):
            upsample_out_dim = current_dim // 2

            self.upsamples.append(
                Upsample(current_dim, upsample_out_dim)
            )

            skip_dim_idx = self.num_levels - 2 - i
            skip_dim = self.encoder_skip_dims[skip_dim_idx]

            self.skip_gates.append(
                SkipFusionGate(
                    dec_dim=upsample_out_dim,
                    skip_dim=skip_dim
                )
            )

            merged_dim = upsample_out_dim + skip_dim
            target_dim = upsample_out_dim

            self.merge_convs.append(
                nn.Conv2d(
                    merged_dim,
                    target_dim,
                    kernel_size=1,
                    bias=bias
                )
            )

            current_dim = target_dim

            decoder_stage = nn.Sequential(*[
                TransformerBlock(
                    dim=current_dim,
                    num_heads=backbone_num_attn_heads[skip_dim_idx],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias
                )
                for _ in range(num_blocks_per_level[skip_dim_idx])
            ])

            self.decoder_blocks.append(decoder_stage)

            prompt_dim = pg_prompt_dim_map[i]
            prompt_hw = pg_base_hw_map[i]

            self.prompt_gens.append(
                PromptGenBlockV4(
                    features_input_dim=current_dim,
                    num_prompt_components=num_prompt_components,
                    prompt_channel_dim=prompt_dim,
                    base_prompt_hw=prompt_hw,
                    bias=bias
                )
            )

            self.prompt_interactions.append(
                PromptInteractionBlockV4(
                    feature_dim=current_dim,
                    prompt_dim=prompt_dim,
                    num_transformer_heads=prompt_interaction_num_attn_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias
                )
            )

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=base_dim,
                num_heads=backbone_num_attn_heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias
            )
            for _ in range(num_refinement_blocks)
        ])

        self.detail_refine = DetailRefineBlock(base_dim)
        self.freq_refine = FrequencyEnhanceBlock(base_dim)

        self.final_conv = nn.Conv2d(
            base_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias
        )

        # Identity-like start.
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()

        factor = 8

        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode="reflect"
        )

        return x, h, w

    def forward(self, x_inp):
        x_inp, original_h, original_w = self.check_image_size(x_inp)

        inp = x_inp
        skip_connections = []

        x = self.initial_conv(x_inp)

        # Encoder
        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)

            if i < self.num_levels - 1:
                skip_connections.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for i in range(self.num_levels - 1):
            x = self.upsamples[i](x)

            skip = skip_connections.pop()

            x = self.skip_gates[i](x, skip)
            x = self.merge_convs[i](x)
            x = self.decoder_blocks[i](x)

            prompt = self.prompt_gens[i](x)
            x = self.prompt_interactions[i](x, prompt)

        x = self.refinement(x)

        x = self.detail_refine(x)
        x = self.freq_refine(x)

        out = self.final_conv(x)
        out = out + inp

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
    return criterion(pred, target)


# ============================================================
# EMA
# ============================================================

def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    return ema_model


@torch.no_grad()
def update_ema_model(model, ema_model, decay=0.999):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()

    for key in ema_state.keys():
        if key in model_state:
            ema_state[key].copy_(
                ema_state[key] * decay + model_state[key] * (1.0 - decay)
            )


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
        "val_psnr",
        "best_psnr"
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
    plt.plot(epochs, val_loss, label="Validation Loss EMA" if USE_EMA else "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PromptIR V4.1 Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_FIG_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_psnr, label="Train PSNR")
    plt.plot(epochs, val_psnr, label="Validation PSNR EMA" if USE_EMA else "Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PromptIR V4.1 Training and Validation PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PSNR_FIG_PATH, dpi=300)
    plt.close()


# ============================================================
# Train / Valid
# ============================================================

def train_one_epoch(model, ema_model, loader, optimizer, criterion):
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

        if USE_EMA:
            update_ema_model(model, ema_model, decay=EMA_DECAY)

        # Clamp only for metric.
        restored_for_metric = torch.clamp(restored, 0, 1)
        psnr = calculate_psnr(restored_for_metric, clean)

        total_loss += loss.item()
        total_psnr += psnr

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "psnr": f"{psnr:.2f}"
        })

    return total_loss / len(loader), total_psnr / len(loader)


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

    return total_loss / len(loader), total_psnr / len(loader)


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

    model = PromptIR_V41(
        in_channels=3,
        out_channels=3,
        base_dim=48,
        num_blocks_per_level=(3, 4, 4, 6),
        num_refinement_blocks=4,
        num_prompt_components=5,
        pg_prompt_dim_map=(256, 128, 64),
        pg_base_hw_map=(16, 32, 64),
        backbone_num_attn_heads=(1, 2, 4, 8),
        prompt_interaction_num_attn_heads=8,
        ffn_expansion_factor=2.66,
        bias=False
    ).to(DEVICE)

    ema_model = None
    if USE_EMA:
        ema_model = create_ema_model(model)

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
    print("Training PromptIR V4.1 Enhanced")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Train size:", len(train_dataset))
    print("Valid size:", len(val_dataset))
    print("Image size:", IMG_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("LR:", LR)
    print("USE_EMA:", USE_EMA)
    print("EMA_DECAY:", EMA_DECAY)
    print("USE_MIXED_LOSS:", USE_MIXED_LOSS)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_psnr = train_one_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            criterion
        )

        eval_model = ema_model if USE_EMA else model

        val_loss, val_psnr = validate(
            eval_model,
            val_loader,
            criterion
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_model = ema_model if USE_EMA else model

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": save_model.state_dict(),
                    "best_psnr": best_psnr,
                    "img_size": IMG_SIZE,
                    "base_dim": 48,
                    "num_blocks_per_level": (3, 4, 4, 6),
                    "num_refinement_blocks": 4,
                    "num_prompt_components": 5,
                    "pg_prompt_dim_map": (256, 128, 64),
                    "pg_base_hw_map": (16, 32, 64),
                    "backbone_num_attn_heads": (1, 2, 4, 8),
                    "prompt_interaction_num_attn_heads": 8,
                    "ffn_expansion_factor": 2.66,
                    "lr": LR,
                    "use_ema": USE_EMA,
                    "ema_decay": EMA_DECAY,
                    "use_mixed_loss": USE_MIXED_LOSS,
                    "mse_weight": MSE_WEIGHT,
                    "model_name": "PromptIR_V41_DynamicPrompt_GatedSkip_DetailFreq"
                },
                BEST_MODEL_PATH
            )

            print(f"Saved best model: {BEST_MODEL_PATH}")
            print(f"Best PSNR: {best_psnr:.2f}")

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "best_psnr": best_psnr
        })

        save_training_log(history)
        save_training_curves(history)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Train PSNR: {train_psnr:.2f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f} | "
            f"Best: {best_psnr:.2f}"
        )

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