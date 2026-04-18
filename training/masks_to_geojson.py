"""
Convert nnU-Net binary prediction masks to GeoJSON polygons for QuPath import.
Handles both LysM and Kidney slides.
"""

import os, json
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image
from skimage import measure

PRED_DIR    = "/Users/antonino/QuPath/nnunet_data/predictions"
OUTPUT_DIR  = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet"
DOWNSAMPLE  = 4.0
MIN_AREA_PX = 500   # pixels² at downsampled resolution (~90 µm diameter)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
mask_files = sorted(glob(os.path.join(PRED_DIR, "*.png")))
print(f"→ {len(mask_files)} masks found")

for mask_path in mask_files:
    image_name  = Path(mask_path).stem
    output_json = os.path.join(OUTPUT_DIR, f"{image_name}_detections.geojson")
    mask        = np.array(Image.open(mask_path))
    labeled     = measure.label(mask > 0)
    regions     = measure.regionprops(labeled)
    features    = []
    skipped     = 0

    for region in regions:
        if region.area < MIN_AREA_PX:
            skipped += 1
            continue
        contours = measure.find_contours(labeled == region.label, 0.5)
        if not contours:
            skipped += 1
            continue
        contour = max(contours, key=len)
        pts = [[float(pt[1])*DOWNSAMPLE, float(pt[0])*DOWNSAMPLE] for pt in contour]
        pts.append(pts[0])
        if len(pts) < 4:
            skipped += 1
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [pts]},
            "properties": {
                "objectType": "detection",
                "classification": {"name": "Glomerulus", "color": [200, 0, 0]},
                "name": f"Glomerulus_{len(features)+1:04d}",
                "area_px": round(region.area * DOWNSAMPLE**2, 1),
            }
        })

    with open(output_json, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    print(f"✓ {image_name}: {len(features)} glomeruli ({skipped} skipped)")

print(f"\n✓ All masks converted → {OUTPUT_DIR}")
