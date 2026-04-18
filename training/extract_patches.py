"""
Glomerulus patch extractor — Round 4
Extracts 600x600px patches from full-resolution images
Saves polygon contours in local coordinates for overlay display
Output: NAS /Team1/MLGlom/patches/ + patches_metadata.json
"""

import os
import json
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ── PARAMETERS ────────────────────────────────────────────────
IMAGE_DIR     = "/Users/antonino/Desktop/Export pics"
GEOJSON_DIR   = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet"
OUTPUT_DIR    = "/Volumes/External DATA/Team1/MLGlom/patches"
METADATA_FILE = "/Volumes/External DATA/Team1/MLGlom/patches_metadata.json"
PATCH_SIZE    = 600
# ─────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

image_paths = sorted(glob(os.path.join(IMAGE_DIR, "LysM_*.jpg"))) + \
              sorted(glob(os.path.join(IMAGE_DIR, "Kidney*.jpg")))

print(f"→ {len(image_paths)} images found")
print(f"→ Patch size: {PATCH_SIZE}×{PATCH_SIZE} px\n")

total_patches = 0
total_skipped = 0
metadata = {}   # patch_name → local polygon coords

for image_path in image_paths:
    image_name   = Path(image_path).stem
    geojson_path = os.path.join(GEOJSON_DIR, f"{image_name}_detections.geojson")

    if not os.path.exists(geojson_path):
        print(f"⚠ No GeoJSON for {image_name} — skipping")
        continue

    print(f"→ Processing: {image_name}")

    img  = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    with open(geojson_path) as f:
        features = json.load(f)["features"]

    print(f"   {len(features)} glomeruli | Image: {W}×{H}")

    slide_count   = 0
    slide_skipped = 0

    for i, feature in enumerate(features):
        coords = feature["geometry"]["coordinates"][0]
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        half = PATCH_SIZE // 2
        x0 = max(0, min(cx - half, W - PATCH_SIZE))
        y0 = max(0, min(cy - half, H - PATCH_SIZE))
        x1 = x0 + PATCH_SIZE
        y1 = y0 + PATCH_SIZE

        patch = img[y0:y1, x0:x1]

        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            slide_skipped += 1
            continue

        patch_name = f"{image_name}_{i+1:04d}.png"
        Image.fromarray(patch).save(os.path.join(OUTPUT_DIR, patch_name))

        # Convert polygon to local patch coordinates
        local_polygon = [
            [round(pt[0] - x0, 1), round(pt[1] - y0, 1)]
            for pt in coords
        ]

        metadata[patch_name] = {
            "slide":   image_name,
            "index":   i + 1,
            "polygon": local_polygon,   # local coords for canvas overlay
            "cx_local": cx - x0,        # centroid x in patch
            "cy_local": cy - y0,        # centroid y in patch
        }

        slide_count += 1

    print(f"   ✓ {slide_count} patches | {slide_skipped} skipped\n")
    total_patches += slide_count
    total_skipped += slide_skipped

# Save metadata
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f)

print(f"✓ Total: {total_patches} patches → {OUTPUT_DIR}")
print(f"✓ Metadata: {METADATA_FILE}")
print(f"  ({total_skipped} skipped — border glomeruli)")
