import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import pickle

# =========================
# CONFIG
# =========================

TEST_DIR = "./data/test"

MODEL_PATHS = [
    "best_model_se_seed42.pkl",
    "best_model_se_seed69.pkl",
    "best_model_se_seed50.pkl",
]

IMG_SIZE = 600
BATCH_SIZE = 32
NUM_CLASSES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# NORMALIZE
# =========================

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
])

# =========================
# LOAD CLASS ORDER
# =========================

with open("classes.pkl","rb") as f:
    classes = pickle.load(f)

print("Classes:",len(classes))

# =========================
# DATASET
# =========================

class TestDataset(Dataset):

    def __init__(self,root,transform):

        self.root = root
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".jpg",".png",".jpeg",".bmp",".tif"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        name = self.images[idx]
        path = os.path.join(self.root,name)

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img,name

dataset = TestDataset(TEST_DIR,transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:",len(dataset))

# =========================
# SE BLOCK
# =========================

class SEBlock(nn.Module):

    def __init__(self,channels,reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels,channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels),
            nn.Sigmoid()
        )

    def forward(self,x):

        b,c,_,_ = x.shape

        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)

        return x * y

# =========================
# MODEL
# =========================

class SEResNet50(nn.Module):

    def __init__(self,num_classes):

        super().__init__()

        base = torchvision.models.resnet50(weights=None)

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
            nn.Linear(2048,num_classes)
        )

    def forward(self,x):

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

# =========================
# LOAD MODELS
# =========================

models = []

for path in MODEL_PATHS:

    model = SEResNet50(NUM_CLASSES).to(DEVICE)

    state = torch.load(path,map_location=DEVICE)
    model.load_state_dict(state)

    model.eval()

    models.append(model)

print("Loaded models:",len(models))

# =========================
# SOFT VOTING INFERENCE
# =========================

results = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():

    for imgs,names in loader:

        imgs = imgs.to(DEVICE)

        prob_sum = 0

        for model in models:

            outputs = model(imgs)

            probs = softmax(outputs)

            prob_sum += probs

        prob_avg = prob_sum / len(models)

        preds = torch.argmax(prob_avg,1).cpu().numpy()

        for name,p in zip(names,preds):

            results.append([name[:-4],classes[p]])

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results,columns=["image_name","pred_label"])

df.to_csv("submission_softvoting.csv",index=False)

print("Saved submission_softvoting.csv")
