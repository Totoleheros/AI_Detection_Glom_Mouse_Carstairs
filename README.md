# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + Custom StarDist + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px)

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (Healthy vs Pathological)

**Target environment:** Mac M1 Max, QuPath 0.7.0-arm64, Python via Miniforge

---

## Abandoned Approaches (and Why)

| Approach | Reason for Abandonment |
|---|---|
| StarDist with pre-trained model `he_heavy_augment.pb` | Detects individual nuclei, not whole glomeruli |
| BioImage.IO Model Zoo | No glomerulus/kidney model available |
| Cellpose | Designed for individual cells — not suited for ~200 µm complex structures |
| Random Forest Pixel Classifier alone | Performance ceiling insufficient for a robust long-term pipeline |
| Color deconvolution (Visual Stain Editor) | Not applicable to Carstairs staining (designed for H&E and H-DAB only) |
| StarDist inference via ONNX in QuPath 0.7.0 | OpenCV 4.11 cannot handle dynamic tensor shapes; DJL backend also failed |
| Direct full-image prediction (no tiling) | OOM on GPU: 469M pixel image exceeds Metal allocator capacity |
| n_tiles StarDist native with PATCH_SIZE=128 | Model trained on 128px patches could not generalize — detected intra-glomerular substructures |
| Absolute intensity threshold for Bowman's capsule | Images have different global brightness — a fixed threshold fails across images |
| Relative percentile threshold (p65) | Carstairs staining has very high green channel values — p65 = 208/255, filters everything |
| Circularity filter | StarDist outputs are intrinsically star-convex — all pass circularity filter |

---

## Root Causes and Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| 41 detections on training image | PATCH_SIZE=128 < glomerulus diameter (200px) — model learned fragments | Re-export with padding=150, retrain with PATCH_SIZE=320 |
| 1145 detections on LysM_01 | Model detects inflammatory infiltrates (similar round dense morphology) | Bowman's capsule contrast ratio filter (ring/interior intensity ratio) |
| Contrast filter not adaptive | Absolute intensity threshold fails across images with different staining | Relative contrast ratio: ring_mean / interior_mean > threshold |

---

## Retained Architecture

```
102 manual QuPath annotations (ground truth)
            ↓
     Export patches PNG + masks
     padding=150 → patch size 420–598 px
     (Groovy script: export_training_patches.groovy)
            ↓
  Custom StarDist training
  PATCH_SIZE=(320,320), EPOCHS=150
  (Python, TensorFlow Metal, M1 Max GPU, ~65 min)
            ↓
  Python inference with tiling (n_tiles ~33×58)
  + Post-processing filters:
    - Size filter: 10,000–80,000 px²
    - Bowman contrast ratio: ring/interior > threshold
  (detect_glomeruli.py)
            ↓
  GeoJSON export → QuPath import
  (Groovy script: import_geojson.groovy)
            ↓
  Healthy/Pathological classification
  (QuPath Object Classifier)
```

---

## PART 1 — Environment Setup

### 1.1 QuPath

- Download **QuPath 0.7.0-arm64**: https://qupath.github.io
- Install in `/Applications/`
- Verify: `Help → About QuPath` → Architecture = `aarch64`

### 1.2 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew --version
```

Expected output:
```
Homebrew 5.0.16
```

### 1.3 Miniforge (native Apple Silicon conda)

```bash
brew install miniforge
conda init bash
```

⚠️ **Fully close Terminal (Cmd+Q) and reopen it.**

```bash
conda --version   # → 26.1.1
```

### 1.4 Dedicated Python environment

```bash
conda create -n stardist-glom python=3.10 -y
```

⚠️ **Fully close Terminal (Cmd+Q) and reopen it.**

```bash
conda activate stardist-glom
python --version   # → Python 3.10.20
```

### 1.5 Python dependencies

```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools
pip install "numpy<2"        # CRITICAL: gputools upgrades numpy to 2.x which breaks TensorFlow
pip install tf2onnx onnx     # Optional: for ONNX export attempts
pip install shapely
pip install opencv-python-headless
pip install "numpy<2"        # Re-run after opencv install — it may re-upgrade numpy
```

> ⚠️ Always verify numpy version after any new pip install:
> `python -c "import numpy; print(numpy.__version__)"` must show `1.26.x`

Verification:
```bash
python -c "import numpy; print(numpy.__version__); import tensorflow as tf; print(tf.__version__); import stardist; print('StarDist OK')"
```

Expected output:
```
1.26.4
2.16.2
StarDist OK
```

---

## PART 2 — QuPath Project Setup

### 2.1 Project creation

- Launch QuPath → `File → New Project`
- Name: `GlomAndreMarc`
- Location: `/Users/antonino/Desktop/GlomAndreMarc`
- Import images from `/Users/antonino/Desktop/Export pics/` (`LysM_01.jpg` → `LysM_11.jpg`)

### 2.2 Manual annotations (ground truth)

- Open `LysM_01.jpg`
- Brush tool (**B**) → annotate 102 glomeruli as class `Glomerulus`
- Additional classes: `Cortex Tissue` (26), `Medulla Tissue` (16), `White` (9)
- Total: 153 annotations

> These 102 annotations = training dataset for custom StarDist model.
> **Do not delete** — also used for Healthy/Pathological classification.

### 2.3 Training patch export

**`Automate`** → **`Project scripts`** → `export_training_patches.groovy`

```groovy
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.geom.AffineTransform

def outputDir  = "/Users/antonino/QuPath/training_data"
def className  = "Glomerulus"
def padding    = 150          // 150px margin — ensures full glomerulus + capsule captured
def downsample = 1.0

new File("${outputDir}/images").mkdirs()
new File("${outputDir}/masks").mkdirs()

def imageData = QP.getCurrentImageData()
def server    = imageData.getServer()

def annotations = QP.getAnnotationObjects().findAll {
    it.getPathClass() != null &&
    it.getPathClass().getName() == className
}

println "→ ${annotations.size()} '${className}' annotations found"
if (annotations.isEmpty()) { println "⚠ No annotations found."; return }

int count = 0
annotations.eachWithIndex { annotation, idx ->
    def roi = annotation.getROI()
    int x = Math.max(0, (int)(roi.getBoundsX() - padding))
    int y = Math.max(0, (int)(roi.getBoundsY() - padding))
    int w = Math.min(server.getWidth()  - x, (int)(roi.getBoundsWidth()  + 2 * padding))
    int h = Math.min(server.getHeight() - y, (int)(roi.getBoundsHeight() + 2 * padding))

    def region   = RegionRequest.createInstance(server.getPath(), downsample, x, y, w, h)
    def imgPatch = server.readRegion(region)
    ImageIO.write(imgPatch, "PNG", new File("${outputDir}/images/glom_${String.format('%03d', idx)}.png"))

    def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    def g2d  = mask.createGraphics()
    g2d.setColor(Color.BLACK); g2d.fillRect(0, 0, w, h)
    g2d.setColor(Color.WHITE)
    def t = new AffineTransform(); t.translate(-x as double, -y as double)
    g2d.fill(t.createTransformedShape(roi.getShape()))
    g2d.dispose()
    ImageIO.write(mask, "PNG", new File("${outputDir}/masks/glom_${String.format('%03d', idx)}.png"))

    count++
    if (count % 10 == 0) println "  ${count}/${annotations.size()} exported..."
}
println "✓ Export complete: ${count} pairs in ${outputDir}"
```

Expected result: **102 pairs**, patch size min 420×397 px, max 598×625 px

---

## PART 3 — Custom StarDist Training

### 3.1 Training script

```bash
mkdir -p /Users/antonino/QuPath/training
nano /Users/antonino/QuPath/training/train_stardist_glom.py
```

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from glob import glob
from skimage.io import imread as sk_imread
from skimage.transform import resize as sk_resize
from csbdeep.utils import normalize
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

DATA_DIR   = "/Users/antonino/QuPath/training_data"
MODEL_DIR  = "/Users/antonino/QuPath/models"
MODEL_NAME = "glomerulus_carstairs"
PATCH_SIZE = (320, 320)
N_RAYS     = 32
EPOCHS     = 150

img_paths  = sorted(glob(os.path.join(DATA_DIR, "images", "*.png")))
mask_paths = sorted(glob(os.path.join(DATA_DIR, "masks",  "*.png")))

assert len(img_paths) == len(mask_paths)
print(f"→ {len(img_paths)} images found")

images, masks = [], []
for ip, mp in zip(img_paths, mask_paths):
    img = sk_imread(ip)
    img = img[:, :, 1].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    msk = sk_imread(mp)
    msk = msk[:, :, 0] if msk.ndim == 3 else msk
    msk = (msk > 128).astype(np.uint16)
    msk = fill_label_holes(msk)
    images.append(img)
    masks.append(msk)

sizes = [img.shape for img in images]
print(f"→ Min: {min(s[0] for s in sizes)}×{min(s[1] for s in sizes)} | Max: {max(s[0] for s in sizes)}×{max(s[1] for s in sizes)}")

images_r, masks_r = [], []
for img, msk in zip(images, masks):
    if img.shape[0] < PATCH_SIZE[0] or img.shape[1] < PATCH_SIZE[1]:
        img = sk_resize(img, PATCH_SIZE, preserve_range=True).astype(np.float32)
        msk = sk_resize(msk, PATCH_SIZE, preserve_range=True, order=0).astype(np.uint16)
    images_r.append(img); masks_r.append(msk)

n_val   = max(1, int(len(images_r) * 0.2))
n_train = len(images_r) - n_val
X_train = [normalize(x, 1, 99) for x in images_r[:n_train]]
Y_train = masks_r[:n_train]
X_val   = [normalize(x, 1, 99) for x in images_r[n_train:]]
Y_val   = masks_r[n_train:]
print(f"→ Train: {n_train} | Validation: {n_val}")

conf = Config2D(
    n_rays=N_RAYS, grid=(2,2), n_channel_in=1,
    train_patch_size=PATCH_SIZE, train_epochs=EPOCHS,
    train_steps_per_epoch=100, train_batch_size=2, use_gpu=True,
)

os.makedirs(MODEL_DIR, exist_ok=True)
model = StarDist2D(conf, name=MODEL_NAME, basedir=MODEL_DIR)
print(f"→ Training {EPOCHS} epochs...")

model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=None)
model.optimize_thresholds(X_val, Y_val)

print(f"✓ Training complete: {MODEL_DIR}/{MODEL_NAME}")
```

### 3.2 Launch

```bash
cd /Users/antonino/QuPath/training
python train_stardist_glom.py
```

Expected: ~65 min, final optimized thresholds `prob_thresh=0.793594, nms_thresh=0.3`

### 3.3 Output

```
/Users/antonino/QuPath/models/glomerulus_carstairs/
    ├── config.json
    ├── thresholds.json      ← prob_thresh=0.793594, nms_thresh=0.3
    ├── weights_best.h5
    ├── weights_last.h5
    └── logs/
```

---

## PART 4 — Inference and GeoJSON Export

> **Why Python inference instead of QuPath native?**
> QuPath 0.7.0 uses OpenCV 4.11 + DJL. OpenCV 4.11 rejects ONNX models with dynamic shapes.
> DJL could not load the StarDist .h5 model either.
> Solution: run inference in Python with Metal GPU, export as GeoJSON, import into QuPath natively.

### 4.1 Detection script

```bash
nano /Users/antonino/QuPath/training/detect_glomeruli.py
```

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from glob import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from shapely.geometry import Polygon
from shapely.affinity import scale
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import cv2

# ── PARAMETERS ────────────────────────────────────────────────
IMAGE_DIR          = "/Users/antonino/Desktop/Export pics"
MODEL_DIR          = "/Users/antonino/QuPath/models"
MODEL_NAME         = "glomerulus_carstairs"
OUTPUT_DIR         = "/Users/antonino/Desktop/GlomAndreMarc/detections"
PROB_THRESH        = 0.793
NMS_THRESH         = 0.3
MIN_AREA_PX        = 10000
MAX_AREA_PX        = 80000
BOWMAN_RING_FACTOR = 1.3
MIN_CONTRAST_RATIO = 1.10   # ring_mean / interior_mean — calibrated on LysM_01
# Contrast ratio distribution on LysM_01: min=0.775 mean=1.029 max=1.317
# True glomeruli expected at ratio > 1.10
# ─────────────────────────────────────────────────────────────

def coords_to_polygon(coords):
    y_arr, x_arr = coords[0], coords[1]
    pts = [[float(x_arr[j]), float(y_arr[j])] for j in range(len(y_arr))]
    pts.append(pts[0])
    return pts

def contrast_ratio(img_green, poly, factor, img_shape):
    H, W  = img_shape
    outer = scale(poly, xfact=factor, yfact=factor)
    mask_i = np.zeros((H, W), dtype=np.uint8)
    mask_o = np.zeros((H, W), dtype=np.uint8)

    def to_cv2(p):
        return np.array([[int(x), int(y)] for x, y in p.exterior.coords], dtype=np.int32)

    try:
        cv2.fillPoly(mask_i, [to_cv2(poly)],  1)
        cv2.fillPoly(mask_o, [to_cv2(outer)], 1)
    except Exception:
        return 0

    interior = mask_i > 0
    ring     = (mask_o - mask_i) > 0

    if interior.sum() == 0 or ring.sum() == 0:
        return 0

    interior_mean = float(img_green[interior].mean())
    if interior_mean == 0:
        return 0

    return float(img_green[ring].mean()) / interior_mean

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = StarDist2D(None, name=MODEL_NAME, basedir=MODEL_DIR)

image_paths = sorted(glob(os.path.join(IMAGE_DIR, "LysM_*.jpg")))
print(f"→ {len(image_paths)} images found\n")

for image_path in image_paths:
    image_name  = os.path.splitext(os.path.basename(image_path))[0]
    output_json = os.path.join(OUTPUT_DIR, f"{image_name}_detections.geojson")
    print(f"→ Processing: {image_name}")

    img       = np.array(Image.open(image_path))
    H, W      = img.shape[:2]
    img_green = img[:, :, 1].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    img_norm  = normalize(img_green, 1, 99)

    n_ty = max(1, int(np.ceil(H / 500)))
    n_tx = max(1, int(np.ceil(W / 500)))

    labels, details = model.predict_instances(
        img_norm, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH, n_tiles=(n_ty, n_tx)
    )

    coords_list = details['coord']
    print(f"   {len(coords_list)} raw detections")

    features = []
    skip_area = skip_contrast = 0
    ratios = []

    for coords in coords_list:
        pts  = coords_to_polygon(coords)
        poly = Polygon(pts)
        area = poly.area

        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            skip_area += 1
            continue

        ratio = contrast_ratio(img_green, poly, BOWMAN_RING_FACTOR, (H, W))
        ratios.append(ratio)

        if ratio < MIN_CONTRAST_RATIO:
            skip_contrast += 1
            continue

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [pts]},
            "properties": {
                "objectType": "detection",
                "classification": {"name": "Glomerulus", "color": [200, 0, 0]},
                "name":           f"Glomerulus_{len(features)+1:04d}",
                "area_px":        round(area, 1),
                "contrast_ratio": round(ratio, 3),
            }
        })

    if ratios:
        print(f"   Contrast: min={min(ratios):.3f} mean={np.mean(ratios):.3f} max={max(ratios):.3f}")
    print(f"   Filtered: {skip_area} by area | {skip_contrast} by contrast")
    print(f"   {len(features)} glomeruli retained")

    with open(output_json, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    print(f"   ✓ Saved: {output_json}\n")

print("✓ All images processed")
print(f"✓ GeoJSON files in: {OUTPUT_DIR}")
```

### 4.2 Launch

```bash
cd /Users/antonino/QuPath/training
python detect_glomeruli.py
```

### 4.3 Import into QuPath

For each image, in QuPath Script Editor:

```groovy
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections/LysM_01_detections.geojson"
def imageData   = getCurrentImageData()

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def root        = JsonParser.parseString(json).getAsJsonObject()
def featuresArr = root.getAsJsonArray("features")

println "→ ${featuresArr.size()} features in GeoJSON"

def objects = GsonTools.getInstance().fromJson(
    featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType()
)

println "→ ${objects.size()} objects parsed"
imageData.getHierarchy().addObjects(objects)
fireHierarchyUpdate()
println "✓ Objects imported"
```

---

## PART 5 — Healthy/Pathological Classification

*(To be completed after detection validation)*

---

## Technical Notes

- Average glomerulus diameter: **~200 px** (measured: 204 px)
- Training patch size (padding=150): **min 420×397 px, max 598×625 px**
- Training PATCH_SIZE: **320×320** (must exceed glomerulus diameter)
- Optimized detection thresholds: **prob_thresh=0.793594, nms_thresh=0.3**
- Calibration: **~1 µm/pixel** (estimated — no JPEG metadata)
- Staining: **Carstairs** — color deconvolution not applicable
- Optimal channel: **Green (index 1)** — best contrast for Carstairs
- Glomeruli: star-convex, ~200 µm, dense mauve interior + bright Bowman's capsule ring
- Contrast ratio (ring/interior green intensity): glomeruli > 1.10, infiltrates ≈ 1.00–1.05
- GPU: Apple M1 Max Metal, 32 GB unified memory
- Images: 11 files, ~469M pixels (28928×16240 or 28800×16240 px)
- numpy must stay at **1.26.4** — any pip install may silently upgrade it
