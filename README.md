# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + nnU-Net v2 + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px)

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (10 classes, see below)

**Target environment:** Mac M1 Max, QuPath 0.7.0-arm64, Python via Miniforge  
**Classification interface:** iMac 2012 (Intel i5, 8GB, macOS Catalina) running Flask server  
**Storage:** Synology NAS (`/Volumes/External DATA/Team1/MLGlom/`)

---

## Why nnU-Net and not StarDist

StarDist was the initial approach but was abandoned after thorough testing. The fundamental issue is architectural: StarDist is a **shape detector** (star-convex objects), not a **semantic segmenter**. It cannot distinguish glomeruli from inflammatory infiltrates or transverse tubular sections, all of which share a similar round, dense morphology in Carstairs staining. All post-hoc filters tested (size, circularity, Bowman's capsule contrast ratio) failed to robustly discriminate these structures across images with different staining intensities.

nnU-Net is a **semantic segmentation** framework based on U-Net architecture. It learns what a glomerulus looks like in context, not just its shape.

---

## Abandoned Approaches (full history)

| Approach | Reason for Abandonment |
|---|---|
| StarDist pre-trained `he_heavy_augment.pb` | Detects individual nuclei, not whole glomeruli |
| BioImage.IO Model Zoo | No glomerulus/kidney model available |
| Cellpose | Designed for individual cells — not suited for ~200 µm structures |
| Random Forest Pixel Classifier alone | Insufficient performance ceiling |
| Color deconvolution (Visual Stain Editor) | Not applicable to Carstairs staining |
| StarDist ONNX inference in QuPath 0.7.0 | OpenCV 4.11 rejects dynamic tensor shapes |
| Direct full-image StarDist prediction | OOM on GPU: 469M pixel image |
| StarDist PATCH_SIZE=128 | Detected intra-glomerular substructures |
| Absolute/relative intensity threshold | Fails across images with different staining |
| Circularity filter | StarDist outputs all pass |
| Contrast ratio filter (ring/interior) | Detects tissue edges, not Bowman's capsule |

---

## Model Training History

| Round | Slides | Glomeruli | Val Dice | Notes |
|---|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 | LysM_01–03, manual annotations |
| Round 2 | 6 | 1,199 | **0.821** | +3 slides corrected from predictions |
| Round 3 | 11 | **2,239** | 🔄 in progress | All LysM slides, exhaustive annotations |

### Round 3 progress
- Epoch 57/100 | EMA Dice: 0.781 | Expected final Dice: >0.85
- Fold 0: 9 train / 2 val (LysM_01, LysM_02)
- Epoch time: ~410s on M1 Max

### Backup
Round 2 checkpoint saved at:
`nnUNet_results/Dataset001_GlomCarstairs/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0_round2_backup`

---

## Classification Scheme

Designed for inflammatory/autoimmune and ischemia-reperfusion context.  
**Multi-label** — a single glomerulus can have multiple classes simultaneously.  
**Exclusive classes** — `Double glomerulus` and `Not a glom` are mutually exclusive with all other classes.

| # | Class | Key | Description |
|---|---|---|---|
| 1 | Normal | `1` | Preserved architecture |
| 2 | Adhesion | `2` | Capsular synechia |
| 3 | Thickening GBM | `3` | Glomerular basement membrane thickening |
| 4 | Fibrinoid necrosis | `4` | Fibrinoid necrosis |
| 5 | Hypercellularity | `5` | Intra-glomerular leukocyte infiltration / proliferation |
| 6 | Fibrosis | `6` | Glomerular/interstitial fibrosis |
| 7 | Crescent | `7` | Epithelial/fibrous crescent |
| 8 | Sclerosis | `8` | Global or segmental sclerosis |
| — | — | — | — |
| 9 | Double glomerulus | `9` | Two contiguous glomeruli not separated by detector — manual correction needed |
| 10 | Not a glom | `0` | False positive from nnU-Net detector — used as hard negative for Round 4 |

---

## Full Pipeline Architecture

```
PHASE 1 — DETECTION (nnU-Net, M1 Max)
──────────────────────────────────────
Exhaustive manual annotations in QuPath
  → export_nnunet.groovy
  → prepare_nnunet.py + create_split.py
  → nnUNetv2_plan_and_preprocess
  → nnUNetv2_train -device mps -tr nnUNetTrainer_100epochs
  → nnUNetv2_predict → binary masks
  → masks_to_geojson.py → GeoJSON
  → import in QuPath → correct → retrain (active learning)

PHASE 2 — PATCH EXTRACTION (M1 Max → NAS)
───────────────────────────────────────────
  → extract_patches.py
  → 600×600 px PNG per glomerulus
  → saved to NAS: /Team1/MLGlom/patches/

PHASE 3 — CLASSIFICATION (iMac Flask server)
─────────────────────────────────────────────
  → python3 ~/GlomClassifier/app.py
  → http://iMac_IP:5000 from any Mac on network
  → 10-class multi-label interface
  → labels saved to NAS: /Team1/MLGlom/labels/labels.csv

PHASE 4 — CLASSIFIER TRAINING (M1 Max)
─────────────────────────────────────────────
  → ResNet fine-tuning on labeled patches (PyTorch, MPS)
  → Auto-classification of all detected glomeruli
  → Results exported as GeoJSON → QuPath
```

---

## PART 1 — Environment Setup

### 1.1 QuPath (M1 Max)
- **QuPath 0.7.0-arm64**: https://qupath.github.io
- Verify: `Help → About QuPath` → Architecture = `aarch64`

### 1.2 Python environment (M1 Max)

```bash
brew install miniforge
conda init bash
# Fully close Terminal (Cmd+Q) and reopen
conda create -n stardist-glom python=3.10 -y
# Fully close Terminal (Cmd+Q) and reopen
conda activate stardist-glom
```

Install dependencies in this exact order:
```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools && pip install "numpy<2"
pip install shapely
pip install opencv-python-headless && pip install "numpy<2"
pip install nnunetv2 && pip install "numpy<2"
```

> ⚠️ After ANY pip install: `python -c "import numpy; print(numpy.__version__)"`
> Must show `1.26.x`. If `2.x`: `pip install "numpy<2"`

### 1.3 nnU-Net environment variables (M1 Max)

```bash
echo 'export nnUNet_raw="/Users/antonino/QuPath/nnunet_data/nnUNet_raw"' >> ~/.bash_profile
echo 'export nnUNet_preprocessed="/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed"' >> ~/.bash_profile
echo 'export nnUNet_results="/Users/antonino/QuPath/nnunet_data/nnUNet_results"' >> ~/.bash_profile
source ~/.bash_profile
```

### 1.4 Flask server (iMac — macOS Catalina 10.15.8)

```bash
# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Miniconda3-latest-MacOSX-x86_64.sh
# Close and reopen Terminal
conda --version   # → 25.7.0

# Install Flask dependencies
python3 -m ensurepip --upgrade
python3 -m pip install flask pillow pandas
```

Start server:
```bash
cd ~
python3 ~/GlomClassifier/app.py
```

Access from any Mac on the network: `http://iMac_IP:5000`

### 1.5 NAS structure

```
/Volumes/External DATA/Team1/MLGlom/
    ├── patches/     ← 600×600 px PNG patches (one per glomerulus)
    ├── labels/
    │   └── labels.csv   ← classification labels
    └── models/      ← trained classifier
```

---

## PART 2 — QuPath Annotation Rules

> Every visible glomerulus on each slide MUST be annotated.
> Unannotated glomerulus = model learns it is background.

- Tool: Brush (**B**), class: `Glomerulus`
- Include Bowman's capsule — slight overshoot acceptable
- Coverage: **exhaustive** per slide

### Annotation counts (Round 3)

| Slide | Glomeruli | Source |
|---|---|---|
| LysM_01 | 192 | Manual |
| LysM_02 | 165 | Manual |
| LysM_03 | 158 | Manual |
| LysM_04 | 218 | nnU-Net + correction |
| LysM_05 | 175 | nnU-Net + correction |
| LysM_06 | 290 | nnU-Net + correction |
| LysM_07 | 218 | Manual |
| LysM_08 | 185 | Manual |
| LysM_09 | 219 | Manual |
| LysM_10 | 210 | Manual |
| LysM_11 | 209 | Manual |
| **Total** | **2,239** | |

---

## PART 3 — QuPath Scripts

### export_nnunet.groovy
Run once per slide (open slide first):
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
QP.getAnnotationObjects().findAll { it.getPathClass()?.getName() == className }.each {
    def t = new AffineTransform(); t.scale(1.0/downsample, 1.0/downsample)
    g2d.fill(t.createTransformedShape(it.getROI().getShape()))
}
g2d.dispose()
ImageIO.write(mask, "PNG", new File("${outputDir}/masks/${imageName}.png"))
println "✓ Export complete: ${imageName}"
```

### import_nnunet_predictions.groovy
Change `geojsonPath` for each slide:
```groovy
import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_04_detections.geojson"
def hierarchy   = getCurrentImageData().getHierarchy()

hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def featuresArr = JsonParser.parseString(json).getAsJsonObject().getAsJsonArray("features")

def objects = GsonTools.getInstance().fromJson(featuresArr,
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
# 1. Prepare dataset
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

Available epoch counts: 5, 10, 20, 50, **100**, 250, 500, 750, 1000

---

## PART 5 — Inference on New Images

```bash
# Prepare images (downsample ×4 + green channel)
python /Users/antonino/QuPath/training/prepare_predict.py

# Predict
nnUNetv2_predict \
  -i /Users/antonino/QuPath/nnunet_data/predict_input \
  -o /Users/antonino/QuPath/nnunet_data/predictions \
  -d 1 -c 2d -tr nnUNetTrainer_100epochs -f 0 -device mps

# Convert to GeoJSON
python /Users/antonino/QuPath/training/masks_to_geojson.py
```

### Notes on image formats
- LysM slides: 28928×16240 px JPEG — compatible with QuPath directly
- Kidney slides: 33120×18304 px — must be converted to pyramidal TIFF for QuPath:
```bash
python -c "
import numpy as np; from PIL import Image; import tifffile
Image.MAX_IMAGE_PIXELS = None
img = np.array(Image.open('Kidney 3 WT.jpg').convert('RGB'))
tifffile.imwrite('Kidney 3 WT.tiff', img, tile=(512,512), compression='jpeg', subfiletype=1)
"
```

---

## PART 6 — Patch Extraction

```bash
# On M1 Max — after Round 3 complete
conda activate stardist-glom
python /Users/antonino/QuPath/training/extract_patches.py
```

Parameters:
- Patch size: **600×600 px** (full resolution, centered on glomerulus centroid)
- Output: NAS `/Team1/MLGlom/patches/`
- Naming: `LysM_01_0001.png` (slide + glomerulus index)

---

## PART 7 — Classification Interface (Flask)

### Features
- Login screen with username (for traceability in CSV)
- 10-class multi-label classification
- `Double glomerulus` and `Not a glom` are **exclusive** classes (auto-deselect others)
- Keyboard shortcuts: `1`–`9`, `0` for classes; `←``→` save & navigate; `Space` zoom
- Thumbnail strip with color coding: green = classified, orange = special class
- Progress bar
- Auto-advances to next unclassified patch

### Start server
```bash
cd ~
python3 ~/GlomClassifier/app.py
```

### CSV output format
```
patch,labels,user
LysM_01_0001.png,Hypercellularity|Adhesion,Marie
LysM_01_0002.png,Normal,Marie
LysM_01_0003.png,Not a glom,Marie
```

---

## PART 8 — Classifier Training (Planned)

- Architecture: ResNet (transfer learning, PyTorch MPS)
- Input: 600×600 px RGB patches
- Labels: multi-label from CSV
- `Not a glom` patches → hard negatives for nnU-Net Round 4
- `Double glomerulus` patches → excluded from training, flagged for manual correction in QuPath

---

## Technical Notes

| Parameter | Value |
|---|---|
| Glomerulus diameter | ~200 px at full resolution |
| Export downsample | ×4 → ~7200×4060 px |
| Classification patch size | 600×600 px (full resolution) |
| Annotation rule | Exhaustive per slide — no exceptions |
| Staining | Carstairs — color deconvolution not applicable |
| Optimal channel | Green (index 1) |
| GPU (M1 Max) | Apple MPS — always `-device mps` |
| GPU (training TF) | Apple Metal via tensorflow-metal |
| numpy | Must stay at 1.26.4 — verify after every pip install |
| PyTorch | 2.11.0 | TensorFlow | 2.16.2 | nnU-Net | 2.7.0 |
