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

---

## Retained Architecture

```
102 manual QuPath annotations (ground truth)
            ↓
     Export patches PNG + masks
     (Groovy script: export_training_patches.groovy)
            ↓
  Custom StarDist training
  (Python, TensorFlow Metal, M1 Max GPU, ~45 min)
            ↓
  Model "glomerulus_carstairs"
            ↓
  Automated detection in QuPath
  (StarDist brightfield script, green channel)
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

Expected output:
```
modified      /Users/<username>/.bash_profile
==> For changes to take effect, close and re-open your current shell. <==
```

⚠️ **Fully close Terminal (Cmd+Q) and reopen it.**

```bash
conda --version
```

Expected output:
```
conda 26.1.1
```

> **Note:** If your default shell is zsh, replace `conda init bash` with `conda init zsh`.
> To check your shell: `echo $SHELL`

### 1.4 Dedicated Python environment

```bash
conda create -n stardist-glom python=3.10 -y
```

⚠️ **Fully close Terminal (Cmd+Q) and reopen it.**

```bash
conda activate stardist-glom
python --version
```

Expected output:
```
(stardist-glom) user$ python --version
Python 3.10.20
```

> The `(stardist-glom)` prefix in the prompt confirms the environment is active.

### 1.5 StarDist + TensorFlow Metal installation

```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools
pip install "numpy<2"   # Required downgrade — gputools forces numpy 2.x which breaks TensorFlow
```

> ⚠️ Order matters: install `gputools` THEN downgrade numpy.
> The `reikna` conflict warning displayed is harmless for this pipeline.

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
- Name the project `GlomAndreMarc`
- Import images (`LysM_01.jpg`, etc.)

### 2.2 Manual annotations (ground truth)

- 102 glomeruli manually annotated using the Brush tool (class `Glomerulus`)
- Additional classes: `Cortex Tissue` (26), `Medulla Tissue` (16), `White` (9)
- Total: 153 annotations

> These 102 annotations form the training dataset for the custom StarDist model.  
> **Do not delete** — they will also be used for Healthy/Pathological classification.

### 2.3 Training patch export

In QuPath: **`Automate`** → **`Script editor`**

Paste the script below → **`File → Save As`** → name it `export_training_patches.groovy`

```groovy
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.geom.AffineTransform

def outputDir  = "/Users/antonino/QuPath/training_data"
def className  = "Glomerulus"
def padding    = 64
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

if (annotations.isEmpty()) {
    println "⚠ No annotations found. Check the class name."
    return
}

int count = 0

annotations.eachWithIndex { annotation, idx ->
    def roi = annotation.getROI()
    int x = Math.max(0, (int)(roi.getBoundsX() - padding))
    int y = Math.max(0, (int)(roi.getBoundsY() - padding))
    int w = Math.min(server.getWidth()  - x, (int)(roi.getBoundsWidth()  + 2 * padding))
    int h = Math.min(server.getHeight() - y, (int)(roi.getBoundsHeight() + 2 * padding))

    def region   = RegionRequest.createInstance(server.getPath(), downsample, x, y, w, h)
    def imgPatch = server.readRegion(region)
    def imgFile  = new File("${outputDir}/images/glom_${String.format('%03d', idx)}.png")
    ImageIO.write(imgPatch, "PNG", imgFile)

    def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    def g2d  = mask.createGraphics()
    g2d.setColor(Color.BLACK)
    g2d.fillRect(0, 0, w, h)
    g2d.setColor(Color.WHITE)
    def transform        = new AffineTransform()
    transform.translate(-x as double, -y as double)
    def transformedShape = transform.createTransformedShape(roi.getShape())
    g2d.fill(transformedShape)
    g2d.dispose()

    def maskFile = new File("${outputDir}/masks/glom_${String.format('%03d', idx)}.png")
    ImageIO.write(mask, "PNG", maskFile)

    count++
    if (count % 10 == 0) println "  ${count}/${annotations.size()} exported..."
}

println "✓ Export complete: ${count} image/mask pairs in ${outputDir}"
```

Expected QuPath console output:
```
INFO: → 102 'Glomerulus' annotations found
INFO:   10/102 exported...
[...]
INFO: ✓ Export complete: 102 image/mask pairs in /Users/antonino/QuPath/training_data
```

Terminal verification:
```bash
ls /Users/antonino/QuPath/training_data/images/ | wc -l   # → 102
ls /Users/antonino/QuPath/training_data/masks/  | wc -l   # → 102
```

Generated structure:
```
/Users/antonino/QuPath/training_data/
    ├── images/
    │   ├── glom_000.png  (minimum observed size: 248×225 px)
    │   └── ... (102 files)
    └── masks/
        ├── glom_000.png
        └── ... (102 files)
```

---

## PART 3 — Custom StarDist Training

### 3.1 Training script

Create the folder and script:
```bash
mkdir -p /Users/antonino/QuPath/training
nano /Users/antonino/QuPath/training/train_stardist_glom.py
```

Full script content:

```python
import numpy as np
import os
from glob import glob
from skimage.io import imread as sk_imread
from csbdeep.utils import normalize
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ── PARAMETERS ────────────────────────────────────────────────
DATA_DIR   = "/Users/antonino/QuPath/training_data"
MODEL_DIR  = "/Users/antonino/QuPath/models"
MODEL_NAME = "glomerulus_carstairs"
PATCH_SIZE = (128, 128)   # Reduced to 128 — some patches are as small as 248×225 px
N_RAYS     = 32
EPOCHS     = 100
# ─────────────────────────────────────────────────────────────

img_paths  = sorted(glob(os.path.join(DATA_DIR, "images", "*.png")))
mask_paths = sorted(glob(os.path.join(DATA_DIR, "masks",  "*.png")))

print(f"\n→ {len(img_paths)} images found")
print(f"→ {len(mask_paths)} masks found")

assert len(img_paths) == len(mask_paths), "Image and mask counts differ!"

images = []
masks  = []

for ip, mp in zip(img_paths, mask_paths):
    img = sk_imread(ip)
    if img.ndim == 3:
        img = img[:, :, 1].astype(np.float32)   # Green channel (optimal for Carstairs)
    else:
        img = img.astype(np.float32)

    msk = sk_imread(mp)
    if msk.ndim == 3:
        msk = msk[:, :, 0]
    msk = (msk > 128).astype(np.uint16)
    msk = fill_label_holes(msk)

    images.append(img)
    masks.append(msk)

sizes = [img.shape for img in images]
min_h = min(s[0] for s in sizes)
min_w = min(s[1] for s in sizes)
print(f"→ Images loaded. Min size: {min_h}×{min_w} | Max: {max(s[0] for s in sizes)}×{max(s[1] for s in sizes)}")

# Filter images too small for patch_size
images_ok = []
masks_ok  = []
skipped   = 0
for img, msk in zip(images, masks):
    if img.shape[0] >= PATCH_SIZE[0] and img.shape[1] >= PATCH_SIZE[1]:
        images_ok.append(img)
        masks_ok.append(msk)
    else:
        skipped += 1

if skipped > 0:
    print(f"⚠ {skipped} images skipped (too small for PATCH_SIZE {PATCH_SIZE})")

print(f"→ {len(images_ok)} images retained for training")

n_val   = max(1, int(len(images_ok) * 0.2))
n_train = len(images_ok) - n_val

X_train = images_ok[:n_train]
Y_train = masks_ok[:n_train]
X_val   = images_ok[n_train:]
Y_val   = masks_ok[n_train:]

print(f"→ Train: {n_train} | Validation: {n_val}")

X_train = [normalize(x, 1, 99) for x in X_train]
X_val   = [normalize(x, 1, 99) for x in X_val]

conf = Config2D(
    n_rays                = N_RAYS,
    grid                  = (2, 2),
    n_channel_in          = 1,
    train_patch_size      = PATCH_SIZE,
    train_epochs          = EPOCHS,
    train_steps_per_epoch = 100,
    train_batch_size      = 4,
    use_gpu               = True,
)

os.makedirs(MODEL_DIR, exist_ok=True)
model = StarDist2D(conf, name=MODEL_NAME, basedir=MODEL_DIR)

print(f"\n→ Starting training ({EPOCHS} epochs)...")
print(f"→ Model will be saved to: {MODEL_DIR}/{MODEL_NAME}")
print("─" * 50)

model.train(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    augmenter       = None,
)

print("\n→ Optimizing detection thresholds...")
model.optimize_thresholds(X_val, Y_val)

print(f"\n✓ Training complete.")
print(f"✓ Model available at: {MODEL_DIR}/{MODEL_NAME}")
print(f"✓ For QuPath: use the full folder (not a .pb file)")
```

### 3.2 Launch

```bash
cd /Users/antonino/QuPath/training
python train_stardist_glom.py
```

Expected output at startup:
```
TensorFlow: 2.16.2
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
→ 102 images found
→ 102 masks found
→ Images loaded. Min size: 248×225 | Max: 426×453
→ 102 images retained for training
→ Train: 82 | Validation: 20
Metal device set to: Apple M1 Max
Epoch 1/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 27s 160ms/step - loss: 8.89 ...
```

> Estimated duration: ~45 minutes on M1 Max (27s/epoch × 100 epochs)  
> Do not close the Terminal during training.

### 3.3 Expected output

```
/Users/antonino/QuPath/models/glomerulus_carstairs/
    ├── config.json
    ├── thresholds.json
    └── weights_best.h5
```

---

## PART 4 — Detection in QuPath

*(To be completed after training)*

---

## PART 5 — Healthy/Pathological Classification

*(To be completed)*

---

## Technical Notes

- Average glomerulus diameter: **204 pixels**
- Minimum exported patch size: **248×225 px** → PATCH_SIZE reduced to 128×128
- Estimated calibration: **1 µm/pixel**
- Staining: **Carstairs** (multi-chromatic — standard color deconvolution does not apply)
- Optimal preprocessing channel: **Green channel (index 1)**
- Glomeruli: star-convex structures ~200 µm, dense mauve on pink background
- GPU Metal confirmed active during training: Apple M1 Max, 32 GB
