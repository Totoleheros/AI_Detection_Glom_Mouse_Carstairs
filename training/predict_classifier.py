"""
Glomerulus Classifier — Inference
Runs best_model.pth on all patches → CSV + GeoJSON per slide for QuPath

Output:
  MLGlom/results/results_per_glom.csv    — one row per glomerulus
  MLGlom/results/results_per_slide.csv   — aggregated per slide
  MLGlom/results/geojson/SLIDE_classified.geojson  — for QuPath import
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models, transforms

# ── PARAMETERS ────────────────────────────────────────────────
PATCHES_DIR   = "/Users/antonino/Desktop/MLGlom/patches"
METADATA_FILE = "/Users/antonino/Desktop/MLGlom/patches_metadata.json"
MODEL_PATH    = "/Users/antonino/Desktop/MLGlom/models/best_model.pth"
GEOJSON_DIR   = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet"
OUTPUT_DIR    = "/Users/antonino/Desktop/MLGlom/results"
THRESHOLD     = 0.5    # probability threshold for positive class
BATCH_SIZE    = 32
# ─────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUTPUT_DIR, "geojson")).mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Device: {device}")

# ── Load model ────────────────────────────────────────────────
print("→ Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASSES    = checkpoint['classes']
N_CLASSES  = len(CLASSES)
print(f"✓ Classes: {CLASSES}")
print(f"✓ Best F1 at training: {checkpoint.get('f1_macro', '?'):.4f}")

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, N_CLASSES)
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("✓ Model loaded")

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load patches ───────────────────────────────────────────────
patches = sorted([f for f in os.listdir(PATCHES_DIR) if f.endswith('.png')])
print(f"→ {len(patches)} patches found")

# Load metadata (polygon coords)
with open(METADATA_FILE) as f:
    metadata = json.load(f)

# ── Inference ─────────────────────────────────────────────────
print("→ Running inference...")
all_results = []

for batch_start in range(0, len(patches), BATCH_SIZE):
    batch_names = patches[batch_start:batch_start + BATCH_SIZE]
    batch_imgs  = []

    for name in batch_names:
        path = os.path.join(PATCHES_DIR, name)
        try:
            img = Image.open(path).convert('RGB')
            batch_imgs.append(transform(img))
        except Exception as e:
            print(f"  ⚠ Error loading {name}: {e}")
            batch_imgs.append(torch.zeros(3, 224, 224))

    batch_tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        logits = model(batch_tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()

    for i, name in enumerate(batch_names):
        p       = probs[i]
        pred    = (p >= THRESHOLD).astype(int)
        classes_positive = [CLASSES[j] for j in range(N_CLASSES) if pred[j]]
        label_str = '|'.join(classes_positive) if classes_positive else 'Unclassified'

        meta = metadata.get(name, {})
        row  = {
            'patch':   name,
            'slide':   meta.get('slide', name.rsplit('_', 1)[0]),
            'index':   meta.get('index', -1),
            'classes': label_str,
            'n_classes': len(classes_positive),
        }
        for j, cls in enumerate(CLASSES):
            row[f'prob_{cls.replace(" ","_")}'] = round(float(p[j]), 4)
        all_results.append(row)

    if (batch_start // BATCH_SIZE) % 20 == 0:
        print(f"  {batch_start + len(batch_names)}/{len(patches)} patches processed...")

print(f"✓ Inference complete — {len(all_results)} patches")

# ── Save per-glom results ──────────────────────────────────────
df = pd.DataFrame(all_results)
glom_path = os.path.join(OUTPUT_DIR, "results_per_glom.csv")
df.to_csv(glom_path, index=False)
print(f"✓ Per-glom: {glom_path}")

# ── Per-slide statistics ───────────────────────────────────────
slide_stats = []
for slide, sdf in df.groupby('slide'):
    n_total = len(sdf)
    row = {'slide': slide, 'n_glom': n_total}

    for cls in CLASSES:
        col = f'prob_{cls.replace(" ","_")}'
        # Count positives
        n_pos = (sdf[col] >= THRESHOLD).sum()
        # Count pure (only this class)
        n_pure = sum(
            1 for _, r in sdf.iterrows()
            if r['classes'] == cls
        )
        # Count mixed (this class + others)
        n_mixed = sum(
            1 for _, r in sdf.iterrows()
            if cls in r['classes'].split('|') and len(r['classes'].split('|')) > 1
        )
        row[f'n_{cls.replace(" ","_")}'] = int(n_pos)
        row[f'pure_{cls.replace(" ","_")}'] = int(n_pure)
        row[f'mixed_{cls.replace(" ","_")}'] = int(n_mixed)
        row[f'pct_{cls.replace(" ","_")}'] = round(100 * n_pos / n_total, 1)

    row['n_unclassified'] = int((sdf['classes'] == 'Unclassified').sum())
    row['n_single_label'] = int((sdf['n_classes'] == 1).sum())
    row['n_multi_label']  = int((sdf['n_classes'] > 1).sum())
    slide_stats.append(row)

slide_df = pd.DataFrame(slide_stats)
slide_path = os.path.join(OUTPUT_DIR, "results_per_slide.csv")
slide_df.to_csv(slide_path, index=False)
print(f"✓ Per-slide summary: {slide_path}")

# Per-slide individual CSVs
slide_csv_dir = os.path.join(OUTPUT_DIR, "per_slide_csv")
Path(slide_csv_dir).mkdir(parents=True, exist_ok=True)
for slide, sdf in df.groupby('slide'):
    out = os.path.join(slide_csv_dir, f"{slide}_results.csv")
    sdf.to_csv(out, index=False)
print(f"✓ Per-slide CSVs: {slide_csv_dir}/")

# ── GeoJSON export for QuPath ──────────────────────────────────
# Color map per primary class (most probable)
CLASS_COLORS = {
    "Normal":             [100, 200, 100],   # green
    "Adhesion":           [255, 165, 0],     # orange
    "Thickening GBM":     [100, 180, 255],   # light blue
    "Fibrinoid necrosis": [220, 50, 50],     # red
    "Hypercellularity":   [180, 0, 220],     # purple
    "Fibrosis":           [200, 140, 60],    # brown
    "Crescent":           [255, 60, 120],    # pink
    "Sclerosis":          [120, 120, 120],   # grey
    "Unclassified":       [200, 200, 200],   # light grey
}

print("→ Generating GeoJSON files...")

for slide, sdf in df.groupby('slide'):
    # Load original detection GeoJSON
    geojson_src = os.path.join(GEOJSON_DIR, f"{slide}_detections.geojson")
    if not os.path.exists(geojson_src):
        print(f"  ⚠ No source GeoJSON for {slide} — skipping")
        continue

    with open(geojson_src) as f:
        geojson = json.load(f)

    features = geojson.get('features', [])

    # Build lookup by index
    idx_to_result = {}
    for _, row in sdf.iterrows():
        idx_to_result[int(row['index'])] = row

    new_features = []
    for feat in features:
        props = feat.get('properties', {})
        # GeoJSON glom index (1-based, from patch name like LysM_01_0042.png)
        name  = props.get('name', '')
        try:
            glom_idx = int(name.split('_')[-1])
        except Exception:
            glom_idx = -1

        result = idx_to_result.get(glom_idx, None)

        if result is not None:
            classes_str  = result['classes']
            prob_cols    = {cls: float(result[f'prob_{cls.replace(" ","_")}'])
                           for cls in CLASSES}
            # Primary class = highest probability among positives
            if classes_str != 'Unclassified':
                primary = max(classes_str.split('|'),
                              key=lambda c: prob_cols.get(c, 0))
            else:
                primary = 'Unclassified'

            color = CLASS_COLORS.get(primary, [200, 200, 200])

            props['classification']  = {'name': primary, 'color': color}
            props['glom_classes']    = classes_str
            props['n_classes']       = int(result['n_classes'])
            # Add all probabilities as measurements
            for cls in CLASSES:
                props[f'prob_{cls.replace(" ","_")}'] = prob_cols[cls]
        else:
            props['classification'] = {'name': 'No patch', 'color': [150, 150, 150]}
            props['glom_classes']   = 'No patch'

        feat['properties'] = props
        new_features.append(feat)

    geojson['features'] = new_features
    out_path = os.path.join(OUTPUT_DIR, "geojson", f"{slide}_classified.geojson")
    with open(out_path, 'w') as f:
        json.dump(geojson, f)
    print(f"  ✓ {slide}: {len(new_features)} glomeruli → {out_path}")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"✓ Inference complete — {len(df)} glomeruli")
print(f"\n=== Global class distribution ===")
for cls in CLASSES:
    col = f'prob_{cls.replace(" ","_")}'
    n   = (df[col] >= THRESHOLD).sum()
    pct = 100 * n / len(df)
    print(f"  {cls:<22} {n:>5}  ({pct:.1f}%)")

print(f"\n=== Label complexity ===")
print(f"  Single class:  {(df['n_classes']==1).sum()}")
print(f"  2 classes:     {(df['n_classes']==2).sum()}")
print(f"  3+ classes:    {(df['n_classes']>=3).sum()}")
print(f"  Unclassified:  {(df['classes']=='Unclassified').sum()}")

print(f"\n✓ Results: {OUTPUT_DIR}")
print(f"  → results_per_glom.csv")
print(f"  → results_per_slide.csv")
print(f"  → geojson/SLIDE_classified.geojson (import in QuPath)")
print(f"{'='*60}")
