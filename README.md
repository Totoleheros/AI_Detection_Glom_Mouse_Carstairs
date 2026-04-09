# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + nnU-Net v2 + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px)

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (Healthy vs Pathological)

**Target environment:** Mac M1 Max, QuPath 0.7.0-arm64, Python via Miniforge

---

## Why nnU-Net and not StarDist

StarDist was the initial approach but was abandoned after thorough testing. The fundamental issue is architectural: StarDist is a **shape detector** (star-convex objects), not a **semantic segmenter**. It cannot distinguish glomeruli from inflammatory infiltrates or transverse tubular sections, all of which share a similar round, dense morphology in Carstairs staining. All post-hoc filters tested (size, circularity, Bowman's capsule contrast ratio) failed to robustly discriminate these structures across images with different staining intensities.

nnU-Net is a **semantic segmentation** framework based on U-Net architecture. It learns what a glomerulus looks like in context, not just its shape. It natively handles background learning from unannotated regions, is self-configuring, and has published results on renal glomeruli segmentation.

---

## Abandoned Approaches (full history)

| Approach | Reason for Abandonment |
|---|---|
| StarDist pre-trained `he_heavy_augment.pb` | Detects individual nuclei, not whole glomeruli |
| BioImage.IO Model Zoo | No glomerulus/kidney model available |
| Cellpose | Designed for individual cells — not suited for ~200 µm structures |
| Random Forest Pixel Classifier alone | Insufficient performance ceiling for a robust pipeline |
| Color deconvolution (Visual Stain Editor) | Not applicable to Carstairs (designed for H&E and H-DAB only) |
| StarDist ONNX inference in QuPath 0.7.0 | OpenCV 4.11 rejects dynamic tensor shapes; DJL backend also failed |
| Direct full-image StarDist prediction | OOM on GPU: 469M pixel image exceeds Metal allocator capacity |
| StarDist with PATCH_SIZE=128 | Model trained on fragments — detected intra-glomerular substructures |
| Absolute intensity threshold (Bowman ring) | Fails across images with different global staining brightness |
| Relative percentile threshold (p65) | Carstairs green channel too uniformly bright — p65 = 208/255 |
| Circularity filter | StarDist outputs are intrinsically star-convex — all pass |
| Contrast ratio filter (ring/interior) | Detects tissue edges and infiltrates, not true Bowman's capsule |

---

## Final Architecture

```
Manual annotations in QuPath
  LysM_01: 193 glomeruli (exhaustive)
  LysM_02: 165 glomeruli (exhaustive)
  LysM_03: 158 glomeruli (exhaustive)
  Total:   516 glomeruli across 3 slides
            ↓
  Export full-slide images + binary masks
  (downsample ×4 → ~7200×4060 px)
  (Groovy script: export_nnunet.groovy)
            ↓
  Dataset preparation (nnU-Net format)
  Green channel extraction, binary mask conversion
  (Python: prepare_nnunet.py)
            ↓
  nnU-Net v2 training
  2D U-Net, patch 896×1792, MPS GPU (Apple Metal)
  (nnUNetv2_train 1 2d 0 --npz -device mps)
            ↓
  nnU-Net inference → segmentation masks
            ↓
  Mask → GeoJSON conversion → QuPath import
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
brew --version   # → 5.0.16
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

> To check your shell: `echo $SHELL`. If zsh, use `conda init zsh` instead.

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
pip install shapely
pip install opencv-python-headless
pip install "numpy<2"        # Re-run after opencv — it may re-upgrade numpy
pip install nnunetv2
pip install "numpy<2"        # Re-run after nnunetv2 — torch may upgrade numpy
```

> ⚠️ After ANY pip install, verify: `python -c "import numpy; print(numpy.__version__)"`
> Must show `1.26.x`. If it shows `2.x`, run `pip install "numpy<2"` again.

Final verification:
```bash
python -c "import numpy; print(numpy.__version__); import tensorflow as tf; print(tf.__version__); import stardist; import nnunetv2; print('All OK')"
```

Expected:
```
1.26.4
2.16.2
All OK
```

### 1.6 nnU-Net environment variables

Add to `~/.bash_profile`:

```bash
echo '' >> ~/.bash_profile
echo '# nnU-Net paths' >> ~/.bash_profile
echo 'export nnUNet_raw="/Users/antonino/QuPath/nnunet_data/nnUNet_raw"' >> ~/.bash_profile
echo 'export nnUNet_preprocessed="/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed"' >> ~/.bash_profile
echo 'export nnUNet_results="/Users/antonino/QuPath/nnunet_data/nnUNet_results"' >> ~/.bash_profile
source ~/.bash_profile
```

Verify:
```bash
echo $nnUNet_raw
# → /Users/antonino/QuPath/nnunet_data/nnUNet_raw
```

---

## PART 2 — QuPath Project Setup

### 2.1 Project

- Launch QuPath → `File → New Project`
- Name: `GlomAndreMarc`
- Location: `/Users/antonino/Desktop/GlomAndreMarc`
- Images location: `/Users/antonino/Desktop/Export pics/` (`LysM_01.jpg` → `LysM_11.jpg`)

### 2.2 Annotation rules for nnU-Net

> **Critical rule:** Every visible glomerulus in the annotated area MUST be annotated.
> An unannotated glomerulus = the model learns it is background.

- Tool: Brush (**B**)
- Class: `Glomerulus`
- Precision: paint generously to include Bowman's capsule — slight overlap is fine, missing part of a glomerulus is not
- Coverage: exhaustive within each slide — do not annotate a subset

Final annotation counts:
- LysM_01: **193 glomeruli**
- LysM_02: **165 glomeruli**
- LysM_03: **158 glomeruli**
- **Total: 516 glomeruli**

### 2.3 Export for nnU-Net

In QuPath: **`Automate`** → **`Script editor`** → save as `export_nnunet.groovy`

Run once per slide (open the slide first, then run):

```groovy
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.geom.AffineTransform

def outputDir  = "/Users/antonino/QuPath/nnunet_data/raw"
def className  = "Glomerulus"
def downsample = 4.0   // ×4 reduction: 28928px → ~7232px

def imageData  = QP.getCurrentImageData()
def server     = imageData.getServer()
def imageName  = server.getMetadata().getName().replaceAll("\\.[^.]+\$", "")

new File("${outputDir}/images").mkdirs()
new File("${outputDir}/masks").mkdirs()

int W = (int)(server.getWidth()  / downsample)
int H = (int)(server.getHeight() / downsample)
println "→ ${imageName} | Output: ${W}×${H}"

def region   = RegionRequest.createInstance(server.getPath(), downsample,
               0, 0, server.getWidth(), server.getHeight())
def imgPatch = server.readRegion(region)
ImageIO.write(imgPatch, "PNG", new File("${outputDir}/images/${imageName}_0000.png"))

def mask = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY)
def g2d  = mask.createGraphics()
g2d.setColor(Color.BLACK); g2d.fillRect(0, 0, W, H)
g2d.setColor(Color.WHITE)

def annotations = QP.getAnnotationObjects().findAll {
    it.getPathClass()?.getName() == className
}
println "→ ${annotations.size()} '${className}' annotations"

annotations.each { annotation ->
    def t = new AffineTransform()
    t.scale(1.0/downsample, 1.0/downsample)
    g2d.fill(t.createTransformedShape(annotation.getROI().getShape()))
}
g2d.dispose()

ImageIO.write(mask, "PNG", new File("${outputDir}/masks/${imageName}.png"))
println "✓ Export complete: ${imageName}"
```

Expected output per slide:
```
→ LysM_01 | Output: 7232×4060
→ 193 'Glomerulus' annotations
✓ Export complete: LysM_01
```

---

## PART 3 — nnU-Net Dataset Preparation

### 3.1 Prepare dataset structure

```bash
nano /Users/antonino/QuPath/training/prepare_nnunet.py
```

```python
import os, json, numpy as np
from PIL import Image
from pathlib import Path

RAW_DIR      = "/Users/antonino/QuPath/nnunet_data/raw"
NNUNET_DIR   = "/Users/antonino/QuPath/nnunet_data"
DATASET_ID   = 1
DATASET_NAME = "GlomCarstairs"

dataset_name = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
base         = Path(NNUNET_DIR) / "nnUNet_raw" / dataset_name
tr_images    = base / "imagesTr"
tr_labels    = base / "labelsTr"
tr_images.mkdir(parents=True, exist_ok=True)
tr_labels.mkdir(parents=True, exist_ok=True)

print(f"→ Preparing: {dataset_name}")
image_files = sorted(Path(RAW_DIR, "images").glob("*.png"))
cases = []

for img_path in image_files:
    case_name = img_path.stem.replace("_0000", "")
    cases.append(case_name)

    # Extract green channel (single channel for nnU-Net)
    img_green = np.array(Image.open(img_path).convert("RGB"))[:, :, 1]
    Image.fromarray(img_green).save(tr_images / img_path.name)
    print(f"   Image: {img_path.name} → shape={img_green.shape}")

    # Binary mask: 0=background, 1=glomerulus
    mask = np.array(Image.open(Path(RAW_DIR, "masks", f"{case_name}.png")).convert("L"))
    mask_bin = (mask > 128).astype(np.uint8)
    Image.fromarray(mask_bin).save(tr_labels / f"{case_name}.png")
    print(f"   Mask:  {case_name}.png → glom pixels: {mask_bin.sum():,}")

dataset_json = {
    "channel_names": {"0": "Green"},
    "labels": {"background": 0, "Glomerulus": 1},
    "numTraining": len(cases),
    "file_ending": ".png",
    "overwrite_image_reader_writer": "NaturalImage2DIO"
}
with open(base / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

for p in ["nnUNet_preprocessed", "nnUNet_results"]:
    (Path(NNUNET_DIR) / p).mkdir(parents=True, exist_ok=True)

print(f"\n✓ Dataset ready: {base}")
print(f"→ Cases: {cases}")
```

```bash
python /Users/antonino/QuPath/training/prepare_nnunet.py
```

### 3.2 Create manual cross-validation split

With only 3 images, nnU-Net's default 5-fold split fails. Create a manual split:

```bash
nano /Users/antonino/QuPath/training/create_split.py
```

```python
import json
from pathlib import Path

split = [
    {"train": ["LysM_01", "LysM_02"], "val": ["LysM_03"]},
    {"train": ["LysM_01", "LysM_03"], "val": ["LysM_02"]},
    {"train": ["LysM_02", "LysM_03"], "val": ["LysM_01"]},
    {"train": ["LysM_01", "LysM_02"], "val": ["LysM_03"]},
    {"train": ["LysM_01", "LysM_03"], "val": ["LysM_02"]}
]

out = Path("/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed/Dataset001_GlomCarstairs/splits_final.json")
with open(out, "w") as f:
    json.dump(split, f, indent=2)
print(f"✓ Split file written: {out}")
```

```bash
python /Users/antonino/QuPath/training/create_split.py
```

### 3.3 Preprocess dataset

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Expected output:
```
verify_dataset_integrity Done.
If you didn't see any error messages then your dataset is most likely OK!
Preprocessing cases: 100%|█████| 3/3
```

Auto-selected configuration:
- Architecture: 2D PlainConvUNet, 9 stages
- Patch size: 896×1792 px
- Batch size: 2

---

## PART 4 — nnU-Net Training

```bash
conda activate stardist-glom
nnUNetv2_train 1 2d 0 --npz -device mps
```

Parameters:
- `1` = Dataset ID
- `2d` = 2D configuration
- `0` = fold 0 (2 train / 1 validation)
- `--npz` = save validation predictions
- `-device mps` = Apple Metal GPU (mandatory on M1/M2/M3)

Expected output at start:
```
Using device: mps
This split has 2 training and 1 validation cases.
Epoch 0
Current learning rate: 0.01
```

> Default: 1000 epochs. Results saved to:
> `/Users/antonino/QuPath/nnunet_data/nnUNet_results/Dataset001_GlomCarstairs/`

---

## PART 5 — Inference and QuPath Import

*(To be completed after training)*

---

## PART 6 — Healthy/Pathological Classification

*(To be completed after detection validation)*

---

## Technical Notes

- Glomerulus diameter: **~200 px** at full resolution (~204 px measured)
- Export downsample: **×4** → images ~7200×4060 px for nnU-Net
- Annotation exhaustiveness: **mandatory** — every visible glomerulus must be labeled
- Staining: **Carstairs** — color deconvolution not applicable
- Optimal channel: **Green (index 1)** — best contrast for Carstairs
- GPU backend: **Apple MPS** (Metal) — use `-device mps` flag, not cuda
- numpy must stay at **1.26.4** — verify after every pip install
- Images: 11 slides, ~469M pixels each (28928×16240 or 28800×16240 px)
- Dataset: 3 annotated slides, 516 glomeruli total, split 2 train / 1 val
