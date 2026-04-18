"""
Glomerulus Multi-Label Classifier Training
ResNet50 fine-tuning — PyTorch MPS (Apple Silicon)

Classes: Normal, Adhesion, Thickening GBM, Fibrinoid necrosis,
         Hypercellularity, Fibrosis, Crescent, Sclerosis
Input:   600x600 px RGB patches (NAS)
Output:  best_model.pth + training_log.csv (NAS models/)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

# ── PARAMETERS ────────────────────────────────────────────────
PATCHES_DIR = "/Users/antonino/Desktop/MLGlom/patches"
LABELS_FILE = "/Users/antonino/Desktop/MLGlom/labels/labels.csv"
OUTPUT_DIR  = "/Users/antonino/Desktop/MLGlom/models"

CLASSES = [
    "Normal", "Adhesion", "Thickening GBM", "Fibrinoid necrosis",
    "Hypercellularity", "Fibrosis", "Crescent", "Sclerosis"
]
EXCLUDE = ["Double glomerulus", "Not a glom"]

BATCH_SIZE   = 16
NUM_EPOCHS   = 50
LR           = 1e-4
LR_BACKBONE  = 1e-5
WEIGHT_DECAY = 1e-4
IMG_SIZE     = 224     # ResNet standard input
VAL_RATIO    = 0.15
SEED         = 42
NUM_WORKERS  = 0       # MPS: keep 0 to avoid multiprocessing issues
# ──────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠ Using CPU")

# ── Load and filter labels ─────────────────────────────────────
print("\n→ Loading labels...")
df = pd.read_csv(LABELS_FILE)
df = df[df['labels'].notna() & (df['labels'] != '') & (df['labels'] != 'nan')]

# Exclude special classes
df = df[~df['labels'].str.contains('|'.join(EXCLUDE), na=False)]

# Verify patch files exist
df['path'] = df['patch'].apply(lambda x: os.path.join(PATCHES_DIR, x))
df = df[df['path'].apply(os.path.exists)]
print(f"✓ {len(df)} valid patches after filtering")

# Parse multi-label into binary vectors
def encode_labels(label_str):
    parts = [p.strip() for p in label_str.split('|')]
    return [1 if cls in parts else 0 for cls in CLASSES]

df['label_vec'] = df['labels'].apply(encode_labels)

# Class distribution
print("\n=== Class distribution ===")
counts = Counter()
for vec in df['label_vec']:
    for i, v in enumerate(vec):
        if v: counts[CLASSES[i]] += 1
for cls in CLASSES:
    n = counts[cls]
    print(f"  {cls:<22} {n:>5}  {'█'*(n//20)}")

# ── Train/Val split ────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=VAL_RATIO,
                                    random_state=SEED, shuffle=True)
print(f"\n→ Train: {len(train_df)} | Val: {len(val_df)}")

# ── Dataset ───────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class GlomDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['path']).convert('RGB')
        img   = self.transform(img)
        label = torch.tensor(row['label_vec'], dtype=torch.float32)
        return img, label

train_ds = GlomDataset(train_df, train_transforms)
val_ds   = GlomDataset(val_df,   val_transforms)

# Weighted sampler — oversample rare classes (Hypercellularity)
# Weight each sample by inverse frequency of its rarest class
def sample_weights(dataframe):
    freq = np.array([counts[cls] for cls in CLASSES], dtype=float)
    freq = freq / freq.sum()
    weights = []
    for vec in dataframe['label_vec']:
        v = np.array(vec)
        active = freq[v == 1]
        w = 1.0 / active.min() if active.size > 0 else 1.0
        weights.append(w)
    return weights

w = sample_weights(train_df)
sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

# ── Model ─────────────────────────────────────────────────────
print("\n→ Building ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze backbone, train only classifier first
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(CLASSES))
)
model = model.to(device)

# ── Loss — weighted BCE for class imbalance ────────────────────
total = len(df)
pos_weight = torch.tensor(
    [(total - counts[cls]) / max(counts[cls], 1) for cls in CLASSES],
    dtype=torch.float32
).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── Optimizer — two learning rates ────────────────────────────
optimizer = optim.AdamW([
    {'params': model.fc.parameters(), 'lr': LR},
], weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ── Training loop ─────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    bin_preds  = (all_preds > 0.5).astype(int)
    f1_macro   = f1_score(all_labels, bin_preds, average='macro', zero_division=0)
    f1_each    = f1_score(all_labels, bin_preds, average=None, zero_division=0)
    try:
        map_score = average_precision_score(all_labels, all_preds, average='macro')
    except Exception:
        map_score = 0.0
    return val_loss / len(loader), f1_macro, f1_each, map_score

log_rows  = []
best_f1   = 0.0
best_path = os.path.join(OUTPUT_DIR, "best_model.pth")

print(f"\n{'='*60}")
print(f"Training {NUM_EPOCHS} epochs | Batch {BATCH_SIZE} | Device: {device}")
print(f"{'='*60}\n")

UNFREEZE_EPOCH = 10   # Unfreeze backbone after warmup

for epoch in range(NUM_EPOCHS):
    # Unfreeze backbone at epoch UNFREEZE_EPOCH
    if epoch == UNFREEZE_EPOCH:
        print(f"\n→ Epoch {epoch}: Unfreezing backbone with lr={LR_BACKBONE}")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([
            {'params': model.fc.parameters(),        'lr': LR},
            {'params': [p for n, p in model.named_parameters()
                        if not n.startswith('fc')],  'lr': LR_BACKBONE},
        ], weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH)

    # Train
    model.train()
    train_loss = 0.0
    t0 = time.time()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_loader)

    # Validate
    val_loss, f1_macro, f1_each, map_score = evaluate(val_loader)
    elapsed = time.time() - t0

    # Save best
    if f1_macro > best_f1:
        best_f1 = f1_macro
        torch.save({
            'epoch':      epoch,
            'model_state_dict': model.state_dict(),
            'f1_macro':   f1_macro,
            'map_score':  map_score,
            'classes':    CLASSES,
        }, best_path)
        flag = " ← best"
    else:
        flag = ""

    print(f"Epoch {epoch:>3}/{NUM_EPOCHS} | "
          f"train={train_loss:.4f} val={val_loss:.4f} | "
          f"F1={f1_macro:.4f} mAP={map_score:.4f} | "
          f"{elapsed:.0f}s{flag}")

    # Per-class F1 every 10 epochs
    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        print("  Per-class F1:")
        for cls, f1 in zip(CLASSES, f1_each):
            print(f"    {cls:<22} {f1:.3f}")

    log_rows.append({
        'epoch': epoch, 'train_loss': round(train_loss, 4),
        'val_loss': round(val_loss, 4), 'f1_macro': round(f1_macro, 4),
        'map_score': round(map_score, 4),
        **{f'f1_{cls.replace(" ","_")}': round(f, 3) for cls, f in zip(CLASSES, f1_each)}
    })

# Save log
log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
pd.DataFrame(log_rows).to_csv(log_path, index=False)

print(f"\n{'='*60}")
print(f"✓ Training complete")
print(f"✓ Best model: {best_path}  (F1={best_f1:.4f})")
print(f"✓ Training log: {log_path}")
print(f"{'='*60}")

# Save class list and metadata
meta = {
    "classes": CLASSES,
    "img_size": IMG_SIZE,
    "threshold": 0.5,
    "n_train": len(train_df),
    "n_val": len(val_df),
    "best_f1": round(best_f1, 4),
}
with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Model metadata: {OUTPUT_DIR}/model_meta.json")
