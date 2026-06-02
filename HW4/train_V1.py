import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

DATA_ROOT = "./hw4_realse_dataset"

TRAIN_DEGRADED_DIR = os.path.join(DATA_ROOT, "train", "degraded")
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train", "clean")

SAVE_DIR = "./outputs_promptir"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_promptir.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")
LOSS_FIG_PATH = os.path.join(SAVE_DIR, "loss_curve.png")
PSNR_FIG_PATH = os.path.join(SAVE_DIR, "psnr_curve.png")

IMG_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 2

LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        print(f"Loaded pairs: {len(self.pairs)}")

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
# Model: PromptIR-like
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.norm(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PromptBlock(nn.Module):
    def __init__(self, dim, prompt_dim=64):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.prompt = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        prompt = self.prompt(self.pool(x))
        x = x * prompt
        x = self.fuse(x)
        return x


class PromptIRBlock(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()

        hidden_dim = dim * expansion

        self.norm1 = LayerNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)

        self.dwconv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_dim * 2
        )

        self.gate = SimpleGate()
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.prompt = PromptBlock(dim)

        self.norm2 = LayerNorm2d(dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.conv1(out)
        out = self.dwconv(out)
        out = self.gate(out)
        out = self.conv2(out)

        out = out + self.prompt(out)

        x = identity + out

        identity = x

        out = self.norm2(x)
        out = self.ffn(out)

        x = identity + out

        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class PromptIRLite(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2, 3, 4, 4],
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1)

        self.encoder_level1 = nn.Sequential(
            *[PromptIRBlock(dim) for _ in range(num_blocks[0])]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[PromptIRBlock(dim * 2) for _ in range(num_blocks[1])]
        )

        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[PromptIRBlock(dim * 4) for _ in range(num_blocks[2])]
        )

        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[PromptIRBlock(dim * 8) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)

        self.decoder_level3 = nn.Sequential(
            *[PromptIRBlock(dim * 4) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)

        self.decoder_level2 = nn.Sequential(
            *[PromptIRBlock(dim * 2) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.decoder_level1 = nn.Sequential(
            *[PromptIRBlock(dim) for _ in range(num_blocks[0])]
        )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def check_image_size(self, x):
        _, _, h, w = x.size()

        factor = 8

        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x, h, w

    def forward(self, x):
        x, original_h, original_w = self.check_image_size(x)

        inp = x

        x1 = self.patch_embed(x)
        x1 = self.encoder_level1(x1)

        x2 = self.down1_2(x1)
        x2 = self.encoder_level2(x2)

        x3 = self.down2_3(x2)
        x3 = self.encoder_level3(x3)

        x4 = self.down3_4(x3)
        x4 = self.latent(x4)

        y3 = self.up4_3(x4)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.reduce3(y3)
        y3 = self.decoder_level3(y3)

        y2 = self.up3_2(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.reduce2(y2)
        y2 = self.decoder_level2(y2)

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.reduce1(y1)
        y1 = self.decoder_level1(y1)

        out = self.output(y1)

        # residual learning
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


# ============================================================
# Train / Valid
# ============================================================
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

    # Loss curve
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

    # PSNR curve
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
        restored = torch.clamp(restored, 0, 1)

        loss = criterion(restored, clean)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        psnr = calculate_psnr(restored, clean)

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
        restored = torch.clamp(restored, 0, 1)

        loss = criterion(restored, clean)
        psnr = calculate_psnr(restored, clean)

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

    model = PromptIRLite(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2, 3, 4, 4]
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
    print("Device:", DEVICE)
    print("Train size:", len(train_dataset))
    print("Valid size:", len(val_dataset))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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
                    "num_blocks": [2, 3, 4, 4]
                },
                BEST_MODEL_PATH
            )

            print(f"Saved best model: {BEST_MODEL_PATH}")
            print(f"Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()