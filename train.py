# train_resnet50_with_gradcam_lime.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from PIL import Image
import random

# ─── interpretability imports ───
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import lime
from lime import lime_image

# ─── CONFIG ───
DATA_DIR    = r'C:\Users\chitt\Downloads\train'
BATCH_SIZE  = 16
VAL_RATIO   = 0.20
EPOCHS      = 12
LR          = 1e-4
IMG_SIZE    = 224
NUM_WORKERS = 0
SEED        = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ─── TRANSFORMS ───
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── DATASET + SPLIT ───
full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)

val_size   = int(VAL_RATIO * len(full_ds))
train_size = len(full_ds) - val_size

train_ds, val_ds = random_split(
    full_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

class_names = full_ds.classes
num_classes = len(class_names)

# ─── MODEL ─── ResNet-50 ───
model = models.resnet50(weights='DEFAULT')          # or weights=models.ResNet50_Weights.IMAGENET1K_V2
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

device = next(model.parameters()).device

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─── GRAD-CAM TARGET LAYER for ResNet-50 ───
# Last conv block before the avg pool → layer4[-1]
target_layers = [model.layer4[-1]]

# ─── TRAIN / VAL FUNCTIONS ───
def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return total_loss / len(train_loader), correct / total


def validate_epoch():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    return total_loss / len(val_loader), correct / total, preds_all, labels_all


# ─── TRAINING LOOP ───
history = {'tl':[], 'ta':[], 'vl':[], 'va':[]}

for ep in range(1, EPOCHS + 1):
    tl, ta = train_epoch()
    vl, va, preds, truths = validate_epoch()
    scheduler.step()

    history['tl'].append(tl)
    history['ta'].append(ta)
    history['vl'].append(vl)
    history['va'].append(va)

    print(f"[{ep:02d}/{EPOCHS}]  train loss: {tl:.4f}  acc: {ta:.4f}  |  val loss: {vl:.4f}  acc: {va:.4f}")

# ─── PLOT ───
plt.figure(figsize=(10,4))
plt.subplot(121); plt.plot(history['tl'], label='train'); plt.plot(history['vl'], label='val'); plt.title('loss'); plt.legend()
plt.subplot(122); plt.plot(history['ta'], label='train'); plt.plot(history['va'], label='val'); plt.title('acc'); plt.legend()
plt.tight_layout()
plt.show()

print("\nClassification Report (val):")
print(classification_report(truths, preds, target_names=class_names, digits=4))

torch.save(model.state_dict(), "plant_disease_resnet50.pth")
print("Model saved → plant_disease_resnet50.pth")

# ─── GRAD-CAM EXAMPLE ───
cam = GradCAM(model=model, target_layers=target_layers)

img_batch, lbl_batch = next(iter(val_loader))
img = img_batch[0:1].to(device)
true_cls = lbl_batch[0].item()

grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(true_cls)])
grayscale_cam = grayscale_cam[0]

img_np = img[0].cpu().permute(1,2,0).numpy()
img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
img_np = np.clip(img_np, 0, 1)

vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.figure(figsize=(9,4))
plt.subplot(121); plt.imshow(img_np); plt.title(f"Orig - {class_names[true_cls]}"); plt.axis('off')
plt.subplot(122); plt.imshow(vis); plt.title("Grad-CAM"); plt.axis('off')
plt.show()

# ─── LIME EXAMPLE ───
def classifier_fn(images):
    tensor = torch.from_numpy(images).permute(0,3,1,2).float().to(device)
    tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(tensor)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1).cpu().numpy()

explainer = lime_image.LimeImageExplainer()

idx = 0
img_path = val_ds.dataset.imgs[val_ds.indices[idx]][0]
true_label = val_ds[idx][1]
pil_img = Image.open(img_path).convert('RGB')

explanation = explainer.explain_instance(
    np.array(pil_img),
    classifier_fn,
    top_labels=5,
    hide_color=0,
    num_samples=1200
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=8,
    hide_rest=False
)

plt.figure(figsize=(9,4))
plt.subplot(121); plt.imshow(np.array(pil_img)); plt.title(f"Orig - {class_names[true_label]}"); plt.axis('off')
plt.subplot(122); plt.imshow(temp); plt.title("LIME"); plt.axis('off')
plt.show()

print("Done.")