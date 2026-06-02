import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from train_V2 import PromptIRStrong


# ============================================================
# Config
# ============================================================

DATA_ROOT = "./hw4_realse_dataset"

TEST_DEGRADED_DIR = os.path.join(DATA_ROOT, "test", "degraded")

SAVE_DIR = "./outputs_promptir_V2"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_promptir_v2.pth")
PRED_NPZ_PATH = os.path.join(SAVE_DIR, "pred.npz")

RESTORED_DIR = os.path.join(SAVE_DIR, "restored_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utils
# ============================================================

def read_image(path):
    return Image.open(path).convert("RGB")


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_to_uint8_chw(x):
    """
    Input:
        x: torch.Tensor, shape [3, H, W], range 0~1

    Output:
        np.ndarray, shape [3, H, W], dtype uint8
    """
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).round().astype(np.uint8)

    return x


# ============================================================
# Test Dataset
# ============================================================

class TestDataset(Dataset):
    def __init__(self, degraded_dir):
        self.degraded_dir = Path(degraded_dir)

        self.files = sorted(
            list(self.degraded_dir.glob("*.png")),
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )

        print(f"Loaded test images: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        img = read_image(path)
        tensor = pil_to_tensor(img)

        return tensor, path.name


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)

    test_dataset = TestDataset(TEST_DEGRADED_DIR)

    model = PromptIRStrong(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    ).to(DEVICE)

    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 60)
    print("Inference Strong PromptIR")
    print("=" * 60)
    print(f"Loaded model from: {BEST_MODEL_PATH}")
    print(f"Best validation PSNR: {ckpt.get('best_psnr', 'N/A')}")
    print("Device:", DEVICE)
    print("=" * 60)

    pred_dict = {}

    for img_tensor, filename in tqdm(test_dataset, desc="Inference"):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        restored = model(img_tensor)
        restored = torch.clamp(restored, 0, 1)

        restored_chw = tensor_to_uint8_chw(restored[0])

        # for pred.npz
        pred_dict[filename] = restored_chw

        # save restored image for visual checking
        restored_hwc = np.transpose(restored_chw, (1, 2, 0))
        Image.fromarray(restored_hwc).save(
            os.path.join(RESTORED_DIR, filename)
        )

    np.savez(PRED_NPZ_PATH, **pred_dict)

    print(f"Saved prediction file: {PRED_NPZ_PATH}")
    print(f"Saved restored images to: {RESTORED_DIR}")
    print(f"Total images: {len(pred_dict)}")

    loaded = np.load(PRED_NPZ_PATH)
    keys = list(loaded.keys())

    print("=" * 60)
    print("Check pred.npz")
    print("Example keys:", keys[:5])

    first_key = keys[0]
    print("First key:", first_key)
    print("Shape:", loaded[first_key].shape)
    print("Dtype:", loaded[first_key].dtype)
    print("Min:", loaded[first_key].min())
    print("Max:", loaded[first_key].max())
    print("=" * 60)


if __name__ == "__main__":
    main()