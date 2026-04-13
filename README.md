# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + nnU-Net v2 + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px)

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (Normal / Inflammatory infiltrate / Mesangial expansion / Crescent / Global sclerosis / Segmental sclerosis / Ischemic collapse / Thrombosis)

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

## Model Training History

> Validation Dice is the key performance metric.
> Dice = 1.0 is perfect segmentation. Published models on renal glomeruli: 0.80–0.92.

| Round | Slides | Glomeruli | Val Dice | EMA Dice | Notes |
|---|---|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 | 0.738 | LysM_01–03, manual annotations only |
| Round 2 | 6 | 1199 | **0.821** | **0.861** | +3 slides corrected from Round 1 predictions |
| Round 3 | 11 | **2239** | TBD | TBD | All slides, exhaustive annotations |

### Round 1 details
- Training slides: LysM_01 (193), LysM_02 (165), LysM_03 (158)
- Validation slide: LysM_03
- Epoch 0 Dice: 0.595 | Final val Dice: 0.765
- Duration: ~10h on M1 Max (100 epochs × ~370s)

### Round 2 details
- Added slides: LysM_04 (218), LysM_05 (175), LysM_06 (290) — nnU-Net predictions + manual correction
- Validation slide: LysM_06
- Epoch 0 Dice: 0.628 (+0.033 vs Round 1) | Epoch 1 Dice: 0.726 (+0.077)
- Final val Dice: **0.821** | EMA Dice: **0.861**
- Duration: ~10h on M1 Max

### Round 3 details (in progress)
- All 11 slides with exhaustive manual annotations
- LysM_01: 192 | LysM_02: 165 | LysM_03: 158 | LysM_04: 218 | LysM_05: 175
- LysM_06: 290 | LysM_07: 218 | LysM_08: 185 | LysM_09: 219 | LysM_10: 210 | LysM_11: 209
- Fold 0: 9 train / 2 val (LysM_01, LysM_02)
- Status: 🔄 Training in progress

---

## Full Pipeline Architecture

```
PHASE 1 — DETECTION (nnU-Net)
──────────────────────────────
Exhaustive manual annotations in QuPath
            ↓
  export_nnunet.groovy (QuPath)
  prepare_nnunet.py
  create_split.py
  nnUNetv2_plan_and_preprocess
  nnUNetv2_train -device mps
            ↓
  nnUNetv2_predict → binary masks
  masks_to_geojson.py → GeoJSON
  import in QuPath → review & correct
            ↓
  Iterative: add corrected slides → retrain

PHASE 2 — CLASSIFICATION (planned)
────────────────────────────────────
Detected glomeruli → patch extraction
            ↓
  GUI sorting tool (local web app)
  Classes: Normal / Inflammatory infiltrate /
           Mesangial expansion / Crescent /
           Global sclerosis / Segmental sclerosis /
           Ischemic collapse / Thrombosis
            ↓
  Train classifier (PyTorch ResNet, MPS GPU)
  Auto-classify in QuPath via GeoJSON
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

Install in this exact order — numpy must stay at 1.26.x throughout:

```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools
pip install "numpy<2"
pip install shapely
pip install opencv-python-headless
pip install "numpy<2"
pip install nnunetv2
pip install "numpy<2"
```

> ⚠️ After ANY pip install, verify numpy:
> `python -c "import numpy; print(numpy.__version__)"`
> Must show `1.26.x`. If `2.x` appears: `pip install "numpy<2"`

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

```bash
echo '' >> ~/.bash_profile
echo '# nnU-Net paths' >> ~/.bash_profile
echo 'export nnUNet_raw="/Users/antonino/QuPath/nnunet_data/nnUNet_raw"' >> ~/.bash_profile
echo 'export nnUNet_preprocessed="/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed"' >> ~/.bash_profile
echo 'export nnUNet_results="/Users/antonino/QuPath/nnunet_data/nnUNet_results"' >> ~/.bash_profile
source ~/.bash_profile
```

> Reload in each new Terminal: `source ~/.bash_profile` then `conda activate stardist-glom`

---

## PART 2 — QuPath Project Setup

### 2.1 Project

- Launch QuPath → `File → New Project`
- Name: `GlomAndreMarc`
- Location: `/Users/antonino/Desktop/GlomAndreMarc`
- Images: `/Users/antonino/Desktop/Export pics/` (`LysM_01.jpg` → `LysM_11.jpg`)

### 2.2 Annotation rules (critical)

> Every visible glomerulus on each slide MUST be annotated.
> Unannotated glomerulus = model learns it is background = training error.

- Tool: Brush (**B**), class: `Glomerulus`
- Include Bowman's capsule — slight overshoot acceptable, undershoot is not
- Coverage: **exhaustive** — no exceptions

### 2.3 Export for nnU-Net (`export_nnunet.groovy`)

Run once per slide with that slide open in QuPath.

```groovy
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.geom.AffineTransform

def outputDir  = "/Users/antonino/QuPath/nnunet_data/raw"
def className  = "Glomerulus"
def downsample = 4.0

def imageData  = QP.getCurrentImageData()
def server     = imageData.getServer()
def imageName  = server.getMetadata().getName().replaceAll("\\.[^.]+\$", "")

new File("${outputDir}/images").mkdirs()
new File("${outputDir}/masks").mkdirs()

int W = (int)(server.getWidth()  / downsample)
int H = (int)(server.getHeight() / downsample)
println "→ ${imageName} | Output: ${W}×${H}"

def region = RegionRequest.createInstance(server.getPath(), downsample,
             0, 0, server.getWidth(), server.getHeight())
ImageIO.write(server.readRegion(region), "PNG",
    new File("${outputDir}/images/${imageName}_0000.png"))

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

### 2.4 Import predictions for correction (`import_nnunet_predictions.groovy`)

Change `geojsonPath` for each slide. Clears all existing objects first.

```groovy
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_04_detections.geojson"
def imageData   = getCurrentImageData()
def hierarchy   = imageData.getHierarchy()

hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def root        = JsonParser.parseString(json).getAsJsonObject()
def featuresArr = root.getAsJsonArray("features")

def objects = GsonTools.getInstance().fromJson(
    featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType()
)

def newAnnotations = objects.collect { obj ->
    qupath.lib.objects.PathObjects.createAnnotationObject(
        obj.getROI(), getPathClass("Glomerulus"))
}

hierarchy.addObjects(newAnnotations)
fireHierarchyUpdate()
println "✓ ${newAnnotations.size()} annotations imported"
```

After import: add missed glomeruli (Brush **B**), delete false positives (**Delete** key).

---

## PART 3 — nnU-Net Dataset Preparation

### 3.1 prepare_nnunet.py

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

image_files = sorted(Path(RAW_DIR, "images").glob("*.png"))
cases = []

for img_path in image_files:
    case_name = img_path.stem.replace("_0000", "")
    cases.append(case_name)
    img_green = np.array(Image.open(img_path).convert("RGB"))[:, :, 1]
    Image.fromarray(img_green).save(tr_images / img_path.name)
    mask = np.array(Image.open(Path(RAW_DIR, "masks", f"{case_name}.png")).convert("L"))
    mask_bin = (mask > 128).astype(np.uint8)
    Image.fromarray(mask_bin).save(tr_labels / f"{case_name}.png")
    print(f"   {case_name}: shape={img_green.shape} | glom px: {mask_bin.sum():,}")

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

print(f"\n✓ {len(cases)} cases: {cases}")
```

### 3.2 create_split.py (11 slides version)

```python
import json
from pathlib import Path

slides = ["LysM_01", "LysM_02", "LysM_03", "LysM_04", "LysM_05",
          "LysM_06", "LysM_07", "LysM_08", "LysM_09", "LysM_10", "LysM_11"]

split = [
    {"train": [s for s in slides if s not in ["LysM_01", "LysM_02"]], "val": ["LysM_01", "LysM_02"]},
    {"train": [s for s in slides if s not in ["LysM_03", "LysM_04"]], "val": ["LysM_03", "LysM_04"]},
    {"train": [s for s in slides if s not in ["LysM_05", "LysM_06"]], "val": ["LysM_05", "LysM_06"]},
    {"train": [s for s in slides if s not in ["LysM_07", "LysM_08"]], "val": ["LysM_07", "LysM_08"]},
    {"train": [s for s in slides if s not in ["LysM_09", "LysM_10"]], "val": ["LysM_09", "LysM_10"]},
]

out = Path("/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed/Dataset001_GlomCarstairs/splits_final.json")
with open(out, "w") as f:
    json.dump(split, f, indent=2)

print(f"✓ Split written: {out}")
for i, fold in enumerate(split):
    print(f"   Fold {i}: train={len(fold['train'])} slides | val={fold['val']}")
```

### 3.3 Full training sequence

```bash
# 1. Prepare dataset
python /Users/antonino/QuPath/training/prepare_nnunet.py

# 2. Preprocess
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 3. Create split
python /Users/antonino/QuPath/training/create_split.py

# 4. Train (fold 0, 100 epochs, MPS GPU)
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs

# To resume after interruption:
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs --c
```

> ⚠️ Interrupt safely with **Ctrl+C** at any time.
> Resume with `--c` flag — picks up from last checkpoint automatically.

---

## PART 4 — Inference on New Images

### 4.1 Prepare images

```bash
python /Users/antonino/QuPath/training/prepare_predict.py
```

```python
import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob

Image.MAX_IMAGE_PIXELS = None
INPUT_DIR  = "/Users/antonino/Desktop/Export pics"
OUTPUT_DIR = "/Users/antonino/QuPath/nnunet_data/predict_input"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for p in sorted(glob(f"{INPUT_DIR}/LysM_*.jpg")):
    name = Path(p).stem
    img  = np.array(Image.open(p).convert("RGB"))
    img_small = img[::4, ::4, 1].astype(np.uint8)
    Image.fromarray(img_small).save(f"{OUTPUT_DIR}/{name}_0000.png")
    print(f"✓ {name}: {img_small.shape}")
```

### 4.2 Predict

```bash
source ~/.bash_profile && conda activate stardist-glom

nnUNetv2_predict \
  -i /Users/antonino/QuPath/nnunet_data/predict_input \
  -o /Users/antonino/QuPath/nnunet_data/predictions \
  -d 1 -c 2d -tr nnUNetTrainer_100epochs -f 0 -device mps
```

### 4.3 Convert to GeoJSON

```bash
python /Users/antonino/QuPath/training/masks_to_geojson.py
```

```python
import os, json
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image
from skimage import measure

PRED_DIR    = "/Users/antonino/QuPath/nnunet_data/predictions"
OUTPUT_DIR  = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet"
DOWNSAMPLE  = 4.0
MIN_AREA_PX = 500

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for mask_path in sorted(glob(os.path.join(PRED_DIR, "LysM_*.png"))):
    image_name  = Path(mask_path).stem
    output_json = os.path.join(OUTPUT_DIR, f"{image_name}_detections.geojson")
    mask        = np.array(Image.open(mask_path))
    labeled     = measure.label(mask > 0)
    regions     = measure.regionprops(labeled)
    features    = []

    for region in regions:
        if region.area < MIN_AREA_PX:
            continue
        contours = measure.find_contours(labeled == region.label, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        pts = [[float(pt[1])*DOWNSAMPLE, float(pt[0])*DOWNSAMPLE] for pt in contour]
        pts.append(pts[0])
        if len(pts) < 4:
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
    print(f"✓ {image_name}: {len(features)} glomeruli")
```

### 4.4 Import into QuPath

```groovy
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

// ← Change filename for each slide
def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_01_detections.geojson"
def imageData   = getCurrentImageData()
def hierarchy   = imageData.getHierarchy()

hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def root        = JsonParser.parseString(json).getAsJsonObject()
def featuresArr = root.getAsJsonArray("features")

def objects = GsonTools.getInstance().fromJson(
    featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType()
)

def newAnnotations = objects.collect { obj ->
    qupath.lib.objects.PathObjects.createAnnotationObject(
        obj.getROI(), getPathClass("Glomerulus"))
}

hierarchy.addObjects(newAnnotations)
fireHierarchyUpdate()
println "✓ ${newAnnotations.size()} glomeruli imported"
```

---

## PART 5 — Pathological Classification (Planned)

### Classification scheme

Designed for inflammatory/autoimmune and ischemia-reperfusion context:

| Class | Lesion | Typical context |
|---|---|---|
| Normal | Preserved architecture | Reference |
| Inflammatory infiltrate | Intra-glomerular leukocyte infiltration | Autoimmune, vasculitis |
| Mesangial expansion | Mesangial matrix enlargement ± hypercellularity | IgA, lupus |
| Crescent | Epithelial/fibrous crescent | Rapidly progressive GN |
| Global sclerosis | Obsolescent glomerulus | End-stage all GN |
| Segmental sclerosis | Segmental fibrosis | Post-inflammatory FSGS |
| Ischemic collapse | Flocculus collapse | Ischemia-reperfusion |
| Thrombosis | Intracapillary thrombus | Vasculitis, severe I-R |

### Planned technical stack

- Patch extraction: Python + PIL (crop each glomerulus from full-res image)
- GUI: local web app (Flask or React) — grid display with click-to-classify
- Classifier: PyTorch ResNet (transfer learning), MPS GPU
- Output: GeoJSON with classification property → import in QuPath

---

## Technical Notes

| Parameter | Value |
|---|---|
| Glomerulus diameter | ~200 px at full resolution |
| Export downsample | ×4 → ~7200×4060 px |
| Annotation rule | Exhaustive per slide — no exceptions |
| Staining | Carstairs — color deconvolution not applicable |
| Optimal channel | Green (index 1) |
| GPU backend | Apple MPS (Metal) — always `-device mps` |
| PyTorch | 2.11.0 |
| TensorFlow | 2.16.2 |
| nnU-Net | 2.7.0 |
| numpy | Must stay at 1.26.4 — verify after every pip install |
| Image size | ~469M pixels (28928×16240 or 28800×16240 px) |
