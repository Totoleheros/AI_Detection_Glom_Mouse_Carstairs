# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + nnU-Net v2 + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px) and TIFF

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (10 classes)

**Target environment:** Mac M1 Max, QuPath 0.7.0-arm64, Python via Miniforge  
**Classification interface:** iMac 2012 (Intel i5, 8GB, macOS Catalina) running Flask  
**Storage:** Synology NAS (`/Volumes/External DATA/Team1/MLGlom/`)

---

## Why nnU-Net and not StarDist

StarDist was abandoned after thorough testing. It is a **shape detector** (star-convex objects), not a **semantic segmenter**. It cannot distinguish glomeruli from inflammatory infiltrates sharing similar round morphology. All post-hoc filters tested failed across images with different staining intensities.

nnU-Net learns what a glomerulus looks like **in context**, handles background learning natively, and is self-configuring.

---

## Abandoned Approaches

| Approach | Reason |
|---|---|
| StarDist pre-trained `he_heavy_augment.pb` | Detects nuclei, not whole glomeruli |
| BioImage.IO Model Zoo | No glomerulus model available |
| Cellpose | Designed for individual cells |
| Random Forest Pixel Classifier | Insufficient performance |
| Color deconvolution | Not applicable to Carstairs staining |
| StarDist ONNX in QuPath 0.7.0 | OpenCV 4.11 rejects dynamic tensor shapes |
| Direct full-image StarDist prediction | OOM: 469M pixel image |
| StarDist PATCH_SIZE=128 | Detected intra-glomerular substructures |
| Intensity/circularity/contrast filters | Failed across different staining batches |

---

## Model Training History

| Round | Slides | Glomeruli | Val Dice | Val slides | Notes |
|---|---|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 | LysM_03 | Manual annotations |
| Round 2 | 6 | 1,199 | 0.821 | LysM_06 | +3 corrected from predictions |
| Round 3 | 11 | 2,239 | 0.803 | LysM_01+02 | All LysM, stricter validation |
| Round 4 | 16 | **3,084** | **0.824** | LysM_01+02 | +5 Kidney WT (normal reins) |

> Dice 0.824 on the same validation criterion as Round 3 (+0.021) — the Kidney WT slides added morphological diversity and inter-batch robustness.

### Dataset composition (Round 4)

| Slide | Glomeruli | Type |
|---|---|---|
| LysM_01 | 220 | Pathological (inflammatory/autoimmune) |
| LysM_02 | 184 | Pathological |
| LysM_03 | 163 | Pathological |
| LysM_04 | 218 | Pathological |
| LysM_05 | 187 | Pathological |
| LysM_06 | 302 | Pathological |
| LysM_07 | 228 | Pathological |
| LysM_08 | 197 | Pathological |
| LysM_09 | 223 | Pathological |
| LysM_10 | 218 | Pathological |
| LysM_11 | 216 | Pathological |
| Kidney 3 WT | 197 | **Normal** |
| Kidney 4 WTa | 158 | **Normal** |
| Kidney 4 WTb | 122 | **Normal** |
| Kidney 5 WTa | 127 | **Normal** |
| Kidney 5 WTb | 124 | **Normal** |
| **Total** | **3,084** | |

### Checkpoints
- Round 2 backup: `fold_0_round2_backup/`
- Round 3: `checkpoint_best.pth` (Dice 0.803)
- Round 4: `checkpoint_best.pth` (Dice 0.824) ← current

---

## Classification Scheme

Multi-label — a glomerulus can have multiple classes simultaneously.  
`Double glomerulus` and `Not a glom` are **exclusive** (auto-deselect others).

| # | Class | Key | Description |
|---|---|---|---|
| 1 | Normal | `1` | Preserved architecture |
| 2 | Adhesion | `2` | Capsular synechia |
| 3 | Thickening GBM | `3` | GBM thickening |
| 4 | Fibrinoid necrosis | `4` | Fibrinoid necrosis |
| 5 | Hypercellularity | `5` | Leukocyte infiltration / proliferation |
| 6 | Fibrosis | `6` | Glomerular/interstitial fibrosis |
| 7 | Crescent | `7` | Epithelial/fibrous crescent |
| 8 | Sclerosis | `8` | Global or segmental sclerosis |
| 9 | Double glomerulus | `9` | Two contiguous glomeruli — manual correction needed |
| 10 | Not a glom | `0` | False positive from nnU-Net detector |

---

## Full Pipeline Architecture

```
PHASE 1 — DETECTION (nnU-Net, M1 Max)
──────────────────────────────────────
Exhaustive manual annotations in QuPath
  → export_nnunet.groovy (QuPath)
  → prepare_nnunet.py + create_split.py
  → nnUNetv2_plan_and_preprocess
  → nnUNetv2_train -device mps
  → nnUNetv2_predict → binary masks
  → masks_to_geojson.py → GeoJSON
  → import in QuPath → correct → retrain (active learning)

PHASE 2 — PATCH EXTRACTION (M1 Max → NAS)
───────────────────────────────────────────
  → extract_patches.py
  → 600×600 px PNG per glomerulus
  → NAS: /Team1/MLGlom/patches/

PHASE 3 — CLASSIFICATION (iMac Flask server)
─────────────────────────────────────────────
  → python3 ~/GlomClassifier/app.py
  → http://iMac_IP:5000
  → 10-class multi-label interface
  → NAS: /Team1/MLGlom/labels/labels.csv

PHASE 4 — CLASSIFIER TRAINING (M1 Max, planned)
─────────────────────────────────────────────────
  → ResNet fine-tuning (PyTorch MPS)
  → Auto-classification → GeoJSON → QuPath
```

---

## PART 1 — Environment Setup

### 1.1 QuPath (M1 Max)
- **QuPath 0.7.0-arm64**: https://qupath.github.io
- Verify: `Help → About QuPath` → Architecture = `aarch64`

### QuPath memory fix (if crashing)
```bash
# Remove corrupted preferences
rm "/Users/antonino/Library/Preferences/qupath.plist"
rm "/Users/antonino/Library/Application Support/CrashReporter/QuPath*.plist"

# Pre-generate thumbnails to prevent crash on large images
conda activate stardist-glom
python3 << 'EOF'
from PIL import Image
import os, json
Image.MAX_IMAGE_PIXELS = None
src = "/Users/antonino/Desktop/Export pics"
data = "/Users/antonino/Desktop/GlomAndreMarc/data"
with open("/Users/antonino/Desktop/GlomAndreMarc/project.qpproj") as f:
    project = json.load(f)
for i, entry in enumerate(project.get('images', []), 1):
    name = entry.get('imageName', '')
    for ext in ['', '.jpg', '.tiff']:
        p = f"{src}/{name}"
        if os.path.exists(p): break
    thumb = f"{data}/{i}/thumbnail.jpg"
    if os.path.exists(p) and not os.path.exists(thumb):
        img = Image.open(p); img.thumbnail((500,500))
        img.save(thumb, "JPEG"); print(f"✓ {name}")
EOF
```

### 1.2 Python environment (M1 Max)

```bash
brew install miniforge
conda init bash
# Close and reopen Terminal
conda create -n stardist-glom python=3.10 -y
# Close and reopen Terminal
conda activate stardist-glom
```

> ⚠️ Always: `conda activate stardist-glom` AFTER `source ~/.bash_profile`

```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools && pip install "numpy<2"
pip install shapely
pip install opencv-python-headless && pip install "numpy<2"
pip install nnunetv2 && pip install "numpy<2"
```

> After ANY pip install: `python -c "import numpy; print(numpy.__version__)"`
> Must show `1.26.x`. If `2.x`: `pip install "numpy<2"`

### 1.3 nnU-Net environment variables

```bash
echo 'export nnUNet_raw="/Users/antonino/QuPath/nnunet_data/nnUNet_raw"' >> ~/.bash_profile
echo 'export nnUNet_preprocessed="/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed"' >> ~/.bash_profile
echo 'export nnUNet_results="/Users/antonino/QuPath/nnunet_data/nnUNet_results"' >> ~/.bash_profile
```

### 1.4 Flask server (iMac — macOS Catalina)

```bash
# Install Miniconda (x86_64)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Miniconda3-latest-MacOSX-x86_64.sh
# Close and reopen Terminal
python3 -m ensurepip --upgrade
python3 -m pip install flask pillow pandas
# Start server
cd ~ && python3 ~/GlomClassifier/app.py
```

---

## PART 2 — QuPath Annotation Rules

> **Every visible glomerulus on each slide MUST be annotated.**
> Unannotated glomerulus = model learns it is background.

- Tool: Brush (**B**), class: `Glomerulus`
- Include Bowman's capsule — overshoot acceptable, undershoot is not
- Coverage: **exhaustive** — no exceptions

---

## PART 3 — QuPath Scripts

### export_nnunet.groovy
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
println "→ Image: ${imageName} | Output size: ${W}×${H}"

def region = RegionRequest.createInstance(server.getPath(), downsample,
             0, 0, server.getWidth(), server.getHeight())
ImageIO.write(server.readRegion(region), "PNG",
    new File("${outputDir}/images/${imageName}_0000.png"))
println "→ Image saved: ${outputDir}/images/${imageName}_0000.png"

def mask = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY)
def g2d  = mask.createGraphics()
g2d.setColor(Color.BLACK); g2d.fillRect(0, 0, W, H)
g2d.setColor(Color.WHITE)

def annotations = QP.getAnnotationObjects().findAll {
    it.getPathClass()?.getName() == className
}
println "→ ${annotations.size()} '${className}' annotations found"

annotations.each { annotation ->
    def t = new AffineTransform()
    t.scale(1.0/downsample, 1.0/downsample)
    g2d.fill(t.createTransformedShape(annotation.getROI().getShape()))
}
g2d.dispose()

ImageIO.write(mask, "PNG", new File("${outputDir}/masks/${imageName}.png"))
println "→ Mask saved: ${outputDir}/masks/${imageName}.png"
println "✓ Export complete for ${imageName}"
```

### import_nnunet_predictions.groovy
```groovy
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

// ← Change for each slide
def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_01_detections.geojson"
def hierarchy   = getCurrentImageData().getHierarchy()

hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def featuresArr = JsonParser.parseString(json).getAsJsonObject().getAsJsonArray("features")
def objects     = GsonTools.getInstance().fromJson(featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType())

hierarchy.addObjects(objects.collect {
    qupath.lib.objects.PathObjects.createAnnotationObject(it.getROI(), getPathClass("Glomerulus"))
})
fireHierarchyUpdate()
println "✓ ${objects.size()} annotations imported"
```

---

## PART 4 — nnU-Net Training Sequence

```bash
# 1. Prepare dataset (after export from QuPath)
python /Users/antonino/QuPath/training/prepare_nnunet.py

# 2. Preprocess
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 3. Create split
python /Users/antonino/QuPath/training/create_split.py

# 4. Train
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs

# Resume after interruption
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs --c
```

> ⚠️ If `--c` finds `checkpoint_final.pth` and skips training:
> `mv checkpoint_final.pth checkpoint_final_backup.pth` then rerun with `--c`

---

## PART 5 — Inference

```bash
# Prepare all images (LysM + Kidney)
python -c "
import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob
Image.MAX_IMAGE_PIXELS = None
INPUT_DIR  = '/Users/antonino/Desktop/Export pics'
OUTPUT_DIR = '/Users/antonino/QuPath/nnunet_data/predict_input'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
files = sorted(glob(f'{INPUT_DIR}/LysM_*.jpg')) + sorted(glob(f'{INPUT_DIR}/Kidney*.jpg'))
for p in files:
    name = Path(p).stem
    img  = np.array(Image.open(p).convert('RGB'))
    img_small = img[::4, ::4, 1].astype(np.uint8)
    Image.fromarray(img_small).save(f'{OUTPUT_DIR}/{name}_0000.png')
    print(f'✓ {name}: {img_small.shape}')
"

# Predict
nnUNetv2_predict \
  -i /Users/antonino/QuPath/nnunet_data/predict_input \
  -o /Users/antonino/QuPath/nnunet_data/predictions \
  -d 1 -c 2d -tr nnUNetTrainer_100epochs -f 0 -device mps

# Convert to GeoJSON
python /Users/antonino/QuPath/training/masks_to_geojson.py
```

### Notes on image formats
- LysM: 28928×16240 px JPEG — QuPath compatible
- Kidney WT: 33120×18304 px — must use TIFF pyramidal for QuPath:
```bash
python -c "
import numpy as np; from PIL import Image; import tifffile
Image.MAX_IMAGE_PIXELS = None
for name in ['Kidney 3 WT', 'Kidney 4 WTa', 'Kidney 4 WTb', 'Kidney 5 WTa', 'Kidney 5 WTb']:
    img = np.array(Image.open(f'/Users/antonino/Desktop/Export pics/{name}.jpg').convert('RGB'))
    tifffile.imwrite(f'/Users/antonino/Desktop/Export pics/{name}.tiff',
                     img, tile=(512,512), compression='jpeg', subfiletype=1)
    print(f'✓ {name}.tiff')
"
```

---

## PART 6 — Patch Extraction

```bash
conda activate stardist-glom
python /Users/antonino/QuPath/training/extract_patches.py
```

- Patch size: **600×600 px** (full resolution, centered on glomerulus)
- Output: NAS `/Team1/MLGlom/patches/`
- Naming: `LysM_01_0001.png`, `Kidney 3 WT_0001.png`

---

## PART 7 — Classification Interface (Flask on iMac)

```bash
cd ~ && python3 ~/GlomClassifier/app.py
```

Access: `http://iMac_IP:5000`

**Keyboard shortcuts:** `1`–`9`, `0` = classes | `←``→` = save & navigate | `Space` = zoom

**CSV output:** `/Team1/MLGlom/labels/labels.csv`
```
patch,labels,user
LysM_01_0001.png,Hypercellularity|Adhesion,Marie
LysM_01_0002.png,Normal,Marie
Kidney 3 WT_0001.png,Normal,Marie
```

---

## PART 8 — Classifier Training (Planned)

- Architecture: ResNet (transfer learning, PyTorch MPS)
- Input: 600×600 px RGB patches
- `Not a glom` → excluded, flagged for nnU-Net Round 5
- `Double glomerulus` → excluded, flagged for manual correction in QuPath

---

## QuPath Project Management Notes

> ⚠️ The project file `project.qpproj` can become corrupted if QuPath is open
> when the file is modified externally. Always quit QuPath (Cmd+Q) before
> running any Python scripts that modify `project.qpproj`.

### Removing images from project (without losing annotations)
```bash
python3 -c "
import json
with open('/Users/antonino/Desktop/GlomAndreMarc/project.qpproj') as f: d=json.load(f)
d['images'] = [img for img in d.get('images',[]) if 'Kidney' not in img.get('imageName','')]
with open('/Users/antonino/Desktop/GlomAndreMarc/project.qpproj','w') as f: json.dump(d,f,indent=2)
print('Done:', [img.get('imageName') for img in d['images']])
"
```

### Re-linking annotations to correct data folder
```bash
python3 -c "
import json
with open('/Users/antonino/Desktop/GlomAndreMarc/project.qpproj') as f: d=json.load(f)
for img in d.get('images',[]):
    if 'Kidney 3 WT' in img.get('imageName',''):
        img['entryID'] = 21  # folder with actual annotations (299K data.qpdata)
        print('Fixed:', img['imageName'])
with open('/Users/antonino/Desktop/GlomAndreMarc/project.qpproj','w') as f: json.dump(d,f,indent=2)
"
```

---

## Technical Notes

| Parameter | Value |
|---|---|
| Glomerulus diameter | ~200 px at full resolution |
| Export downsample | ×4 → ~7200×4060 px (LysM) / ~8300×4600 px (Kidney) |
| Classification patch size | 600×600 px full resolution |
| Annotation rule | Exhaustive per slide — no exceptions |
| Staining | Carstairs — color deconvolution not applicable |
| Optimal channel | Green (index 1) |
| GPU (M1 Max) | Apple MPS — always `-device mps` |
| numpy | Must stay at 1.26.4 |
| PyTorch | 2.11.0 | TensorFlow | 2.16.2 | nnU-Net | 2.7.0 |
