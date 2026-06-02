import os
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from super_image import MsrnModel, EdsrModel, DrlnModel


# =========================================================
# Config: 直接改這裡
# =========================================================
LR_DIR = "./data_sr/train/lr"
HR_DIR = "./data_sr/train/hr"
OUT_DIR = "./outputs_super_image_drln_retrain_vnew2"

MODEL_NAME = "drln"   # "msrn", "edsr", "drln"
SCALE = 4


# 單階段重訓設定
PATCH_SIZE = 128
EPOCHS = 150
BATCH_SIZE = 8
NUM_WORKERS = 8
LR = 9e-5
VAL_RATIO = 0.02
SEED = 42

# 讓同一張圖在一個 epoch 內看到更多 random crop
DATASET_REPEAT = 3

# 每個 epoch 最多抽多少 patch
MAX_TRAIN_SAMPLES_PER_EPOCH = 200000
MAX_VAL_SAMPLES = 1000



'''
# 單階段重訓設定
PATCH_SIZE = 64
EPOCHS = 40
BATCH_SIZE = 8
NUM_WORKERS = 8
LR = 8e-5
VAL_RATIO = 0.02
SEED = 42

# 讓同一張圖在一個 epoch 內看到更多 random crop
DATASET_REPEAT = 3

# 每個 epoch 最多抽多少 patch
MAX_TRAIN_SAMPLES_PER_EPOCH = 80000
MAX_VAL_SAMPLES = 500
'''
SAVE_EVERY = 1
USE_AMP = True

# EMA
USE_EMA = True
EMA_DECAY = 0.999

# Loss weight
W_L1 = 0.5
W_CHARB = 0.5

# 如果 OOM，改：
# PATCH_SIZE = 48
# BATCH_SIZE = 4
# NUM_WORKERS = 4
# MAX_TRAIN_SAMPLES_PER_EPOCH = 30000


# =========================================================
# Utils
# =========================================================
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder):
    folder = Path(folder)
    files = []

    for ext in IMG_EXTS:
        files += list(folder.glob(f"*{ext}"))
        files += list(folder.glob(f"*{ext.upper()}"))

    return sorted(files)


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def calc_psnr(sr, hr):
    sr = torch.clamp(sr, 0, 1)
    hr = torch.clamp(hr, 0, 1)

    mse = F.mse_loss(sr, hr)

    if mse.item() == 0:
        return 99.0

    psnr = 10.0 * torch.log10(torch.tensor(1.0, device=sr.device) / mse)
    return psnr.item()


def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


def build_model(model_name, scale):
    if model_name == "msrn":
        model = MsrnModel.from_pretrained("eugenesiow/msrn", scale=scale)

    elif model_name == "edsr":
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)

    elif model_name == "drln":
        model = DrlnModel.from_pretrained("eugenesiow/drln", scale=scale)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        model_state = model.state_dict()
        ema_state = ema_model.state_dict()

        for k in ema_state.keys():
            if ema_state[k].dtype.is_floating_point:
                ema_state[k].mul_(decay).add_(model_state[k].detach(), alpha=1.0 - decay)
            else:
                ema_state[k].copy_(model_state[k])


# =========================================================
# Dataset
# =========================================================
class PairedSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, patch_size=64, train=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.train = train

        lr_files = list_images(self.lr_dir)
        hr_files = list_images(self.hr_dir)

        if len(lr_files) == 0:
            raise RuntimeError(f"No LR images found in {self.lr_dir}")

        if len(hr_files) == 0:
            raise RuntimeError(f"No HR images found in {self.hr_dir}")

        hr_map = {p.stem: p for p in hr_files}

        self.pairs = []

        for lr_path in lr_files:
            key = lr_path.stem

            possible_keys = [
                key,
                key.replace("_lr", "").replace("_LR", ""),
                key.replace("-lr", "").replace("-LR", ""),
                key.replace("lr", "hr"),
                key.replace("LR", "HR"),
            ]

            matched_hr = None

            for k in possible_keys:
                if k in hr_map:
                    matched_hr = hr_map[k]
                    break

            if matched_hr is not None:
                self.pairs.append((lr_path, matched_hr))

        if len(self.pairs) == 0:
            raise RuntimeError("No LR-HR image pairs found. Check LR_DIR / HR_DIR / filenames.")

        print(f"Found LR images: {len(lr_files)}")
        print(f"Found HR images: {len(hr_files)}")
        print(f"Matched pairs: {len(self.pairs)}")

    def __len__(self):
        if self.train:
            return len(self.pairs) * DATASET_REPEAT
        return len(self.pairs)

    def random_crop_pair(self, lr_img, hr_img):
        w_lr, h_lr = lr_img.size
        ps = self.patch_size

        if w_lr < ps or h_lr < ps:
            lr_img = lr_img.resize(
                (max(w_lr, ps), max(h_lr, ps)),
                Image.BICUBIC
            )
            w_lr, h_lr = lr_img.size

        target_hr_size = (w_lr * self.scale, h_lr * self.scale)

        if hr_img.size != target_hr_size:
            hr_img = hr_img.resize(target_hr_size, Image.BICUBIC)

        x = random.randint(0, w_lr - ps)
        y = random.randint(0, h_lr - ps)

        lr_crop = lr_img.crop((x, y, x + ps, y + ps))

        x_hr = x * self.scale
        y_hr = y * self.scale
        ps_hr = ps * self.scale

        hr_crop = hr_img.crop((x_hr, y_hr, x_hr + ps_hr, y_hr + ps_hr))

        return lr_crop, hr_crop

    def center_crop_pair(self, lr_img, hr_img):
        w_lr, h_lr = lr_img.size
        ps = min(self.patch_size, w_lr, h_lr)

        target_hr_size = (w_lr * self.scale, h_lr * self.scale)

        if hr_img.size != target_hr_size:
            hr_img = hr_img.resize(target_hr_size, Image.BICUBIC)

        x = max(0, (w_lr - ps) // 2)
        y = max(0, (h_lr - ps) // 2)

        lr_crop = lr_img.crop((x, y, x + ps, y + ps))

        x_hr = x * self.scale
        y_hr = y * self.scale
        ps_hr = ps * self.scale

        hr_crop = hr_img.crop((x_hr, y_hr, x_hr + ps_hr, y_hr + ps_hr))

        return lr_crop, hr_crop

    def augment(self, lr_img, hr_img):
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.ROTATE_90)
            hr_img = hr_img.transpose(Image.ROTATE_90)

        return lr_img, hr_img

    def __getitem__(self, idx):
        idx = idx % len(self.pairs)

        lr_path, hr_path = self.pairs[idx]

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        if self.train:
            lr_img, hr_img = self.random_crop_pair(lr_img, hr_img)
            lr_img, hr_img = self.augment(lr_img, hr_img)
        else:
            lr_img, hr_img = self.center_crop_pair(lr_img, hr_img)

        lr_tensor = pil_to_tensor(lr_img)
        hr_tensor = pil_to_tensor(hr_img)

        return lr_tensor, hr_tensor


# =========================================================
# Train / Valid
# =========================================================
def train_one_epoch(model, ema_model, loader, optimizer, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_charb = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for lr_img, hr_img in pbar:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                sr = model(lr_img)

                loss_l1 = F.l1_loss(sr, hr_img)
                loss_charb = charbonnier_loss(sr, hr_img)
                loss = W_L1 * loss_l1 + W_CHARB * loss_charb

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            sr = model(lr_img)

            loss_l1 = F.l1_loss(sr, hr_img)
            loss_charb = charbonnier_loss(sr, hr_img)
            loss = W_L1 * loss_l1 + W_CHARB * loss_charb

            loss.backward()
            optimizer.step()

        if USE_EMA and ema_model is not None:
            update_ema(ema_model, model, decay=EMA_DECAY)

        total_loss += loss.item()
        total_l1 += loss_l1.item()
        total_charb += loss_charb.item()

        pbar.set_postfix(
            loss=loss.item(),
            l1=loss_l1.item(),
            charb=loss_charb.item(),
        )

    n = len(loader)

    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "charb": total_charb / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_charb = 0.0
    total_psnr = 0.0

    pbar = tqdm(loader, desc="Valid", leave=False)

    for lr_img, hr_img in pbar:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        sr = model(lr_img)

        loss_l1 = F.l1_loss(sr, hr_img)
        loss_charb = charbonnier_loss(sr, hr_img)
        loss = W_L1 * loss_l1 + W_CHARB * loss_charb

        psnr = calc_psnr(sr, hr_img)

        total_loss += loss.item()
        total_l1 += loss_l1.item()
        total_charb += loss_charb.item()
        total_psnr += psnr

        pbar.set_postfix(
            loss=loss.item(),
            l1=loss_l1.item(),
            psnr=psnr,
        )

    n = len(loader)

    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "charb": total_charb / n,
        "psnr": total_psnr / n,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_name": MODEL_NAME,
        "scale": SCALE,
        "patch_size": PATCH_SIZE,
        "data_repeat": DATASET_REPEAT,
        "use_ema": USE_EMA,
    }

    torch.save(ckpt, path)


def make_train_loader(train_set):
    if len(train_set) > MAX_TRAIN_SAMPLES_PER_EPOCH:
        train_indices = random.sample(
            range(len(train_set)),
            MAX_TRAIN_SAMPLES_PER_EPOCH
        )
        epoch_train_set = Subset(train_set, train_indices)
    else:
        epoch_train_set = train_set

    loader = DataLoader(
        epoch_train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    return loader, len(epoch_train_set)


# =========================================================
# Main
# =========================================================
def main():
    seed_everything(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    full_dataset = PairedSRDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        train=True,
    )

    val_size = max(1, int(len(full_dataset) * VAL_RATIO))
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    if len(val_set) > MAX_VAL_SAMPLES:
        val_indices = random.sample(range(len(val_set)), MAX_VAL_SAMPLES)
        val_set = Subset(val_set, val_indices)

    print("Train size:", len(train_set))
    print("Val size:", len(val_set))

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    model = build_model(MODEL_NAME, SCALE)
    model = model.to(device)

    ema_model = None
    if USE_EMA:
        ema_model = deepcopy(model).to(device)
        ema_model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {MODEL_NAME}")
    print(f"Scale: x{SCALE}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Use EMA: {USE_EMA}")
    print(f"EMA decay: {EMA_DECAY}")
    print(f"Loss: {W_L1} * L1 + {W_CHARB} * Charbonnier")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.05,
    )

    scaler = None
    if USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_psnr = -1.0

    log_path = Path(OUT_DIR) / "train_log.csv"

    with open(log_path, "w") as f:
        f.write(
            "epoch,lr,epoch_train_samples,"
            "train_loss,train_l1,train_charb,"
            "val_loss,val_l1,val_charb,val_psnr,best_psnr\n"
        )

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loader, epoch_train_samples = make_train_loader(train_set)

        train_log = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        eval_model = ema_model if USE_EMA and ema_model is not None else model

        val_log = validate(
            model=eval_model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        if epoch % SAVE_EVERY == 0:
            save_model = eval_model if USE_EMA and ema_model is not None else model
            save_checkpoint(
                Path(OUT_DIR) / "last.pth",
                save_model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
            )

        if val_log["psnr"] > best_psnr:
            best_psnr = val_log["psnr"]

            save_model = eval_model if USE_EMA and ema_model is not None else model
            save_checkpoint(
                Path(OUT_DIR) / "best.pth",
                save_model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
            )

            print(f"Saved best model: {Path(OUT_DIR) / 'best.pth'}")

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.6e} | "
            f"Samples: {epoch_train_samples} | "
            f"Train Loss: {train_log['loss']:.6f} | "
            f"Train L1: {train_log['l1']:.6f} | "
            f"Val Loss: {val_log['loss']:.6f} | "
            f"Val L1: {val_log['l1']:.6f} | "
            f"Val PSNR: {val_log['psnr']:.4f} | "
            f"Best PSNR: {best_psnr:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{current_lr},{epoch_train_samples},"
                f"{train_log['loss']},{train_log['l1']},{train_log['charb']},"
                f"{val_log['loss']},{val_log['l1']},{val_log['charb']},"
                f"{val_log['psnr']},{best_psnr}\n"
            )

    print("=" * 80)
    print("Training finished.")
    print("Best PSNR:", best_psnr)
    print("Best model:", Path(OUT_DIR) / "best.pth")
    print("=" * 80)


if __name__ == "__main__":
    main()