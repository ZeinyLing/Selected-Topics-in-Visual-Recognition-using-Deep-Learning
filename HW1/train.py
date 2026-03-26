import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ======================
# SEED
# ======================

SEED = 69

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================
# CONFIG
# ======================

TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"

IMG_SIZE = 600
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "./outputs_se_resnet50_69"
CURVE_DIR = os.path.join(SAVE_DIR, "curves")
CM_DIR = os.path.join(SAVE_DIR, "confusion_matrix")
MODEL_DIR = os.path.join(SAVE_DIR, "models")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_se_seed69.pth")

# ======================
# NORMALIZE
# ======================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ======================
# AUGMENTATION
# ======================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.15)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ======================
# DATASET
# ======================

train_set = ImageFolder(TRAIN_DIR, transform=train_transform)
val_set = ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,

)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,

)

print("Train:", len(train_set))
print("Val:", len(val_set))

# ======================
# SAVE CLASS ORDER
# ======================

classes = train_set.classes

with open(os.path.join(SAVE_DIR, "classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

# ======================
# SE BLOCK
# ======================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_dim = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ======================
# SE-RESNET50
# ======================

class SEResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = self.layer4(x)
        x = self.se4(x)

        x = self.pool(x).flatten(1)
        x = self.fc(x)

        return x

# ======================
# TTA
# ======================

def tta_predict(model, img):
    img_flip = torch.flip(img, dims=[3])
    out1 = model(img)
    out2 = model(img_flip)
    out = (out1 + out2) / 2
    return out.argmax(1)

# ======================
# MODEL
# ======================

model = SEResNet50(NUM_CLASSES).to(DEVICE)

# ======================
# PARAMETER COUNT
# ======================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ======================
# LOSS
# ======================

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ======================
# OPTIMIZER
# ======================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ======================
# EARLY STOPPING
# ======================

best_acc = 0.0
patience = 15
counter = 0

# ======================
# HISTORY
# ======================

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# ======================
# PLOT FUNCTION
# ======================

def save_training_curves(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, save_dir):
    epochs_range = range(1, len(train_loss_hist) + 1)

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss_hist, label="Train Loss")
    plt.plot(epochs_range, val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_acc_hist, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_hist, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

# ======================
# TRAIN LOOP
# ======================

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()

    train_correct = 0
    train_total = 0
    train_loss_sum = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)

        batch_size_now = y.size(0)
        train_loss_sum += loss.item() * batch_size_now
        train_correct += (pred == y).sum().item()
        train_total += batch_size_now

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ----- VALIDATION -----
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            pred = out.argmax(1)

            batch_size_now = y.size(0)
            val_loss_sum += loss.item() * batch_size_now
            val_correct += (pred == y).sum().item()
            val_total += batch_size_now

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    scheduler.step()

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # 每個 epoch 都更新 curve
    save_training_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        CURVE_DIR
    )

    # ----- SAVE BEST MODEL -----
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved best model to {BEST_MODEL_PATH}")


print("Best Val Acc:", best_acc)

# ======================
# SAVE HISTORY
# ======================

history = {
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "train_acc": train_acc_history,
    "val_acc": val_acc_history,
    "best_val_acc": best_acc,
    "total_params": total_params,
    "trainable_params": trainable_params
}

with open(os.path.join(SAVE_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(history, f)

# ======================
# LOAD BEST MODEL
# ======================

print("Loading best model for confusion matrix...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# CONFUSION MATRIX
# ======================

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        pred = out.argmax(1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

# 存原始 confusion matrix 數值
np.save(os.path.join(CM_DIR, "confusion_matrix.npy"), cm)

# ======================
# FULL CONFUSION MATRIX
# ======================

fig, ax = plt.subplots(figsize=(20, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    colorbar=False,
    values_format="d"
)
plt.title("Confusion Matrix (Full Labels)")
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "confusion_matrix_full.png"), dpi=300)
plt.close()

# ======================
# SIMPLIFIED CONFUSION MATRIX
# 不顯示類別名稱，只顯示 index 或乾脆不顯示 tick
# ======================

fig, ax = plt.subplots(figsize=(16, 16))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix (Simplified)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# 類別太多，不顯示每個類別名稱
ax.set_xticks([])
ax.set_yticks([])

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "confusion_matrix_simplified.png"), dpi=300)
plt.close()

# ======================
# NORMALIZED CONFUSION MATRIX
# ======================

cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)

fig, ax = plt.subplots(figsize=(16, 16))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
ax.set_title("Normalized Confusion Matrix (Simplified)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([])
ax.set_yticks([])

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(CM_DIR, "confusion_matrix_normalized.png"), dpi=300)
plt.close()

print("All outputs saved to:", SAVE_DIR)
print("Curve files:")
print(" -", os.path.join(CURVE_DIR, "loss_curve.png"))
print(" -", os.path.join(CURVE_DIR, "accuracy_curve.png"))
print("Confusion matrix files:")
print(" -", os.path.join(CM_DIR, "confusion_matrix_full.png"))
print(" -", os.path.join(CM_DIR, "confusion_matrix_simplified.png"))
print(" -", os.path.join(CM_DIR, "confusion_matrix_normalized.png"))