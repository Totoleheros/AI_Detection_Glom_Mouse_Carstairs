# Glomerulus Detection and Classification Pipeline
**QuPath 0.7.0 + nnU-Net v2 + Apple Silicon (M1 Max)**  
Staining: Carstairs | Images: Brightfield JPEG (~28928 × 16240 px) and TIFF

---

## Context

Semi-automated pipeline for:
1. Detecting glomeruli on kidney sections (Brightfield, Carstairs staining)
2. Classifying them by pathological status (10 classes)
3. Exporting results to QuPath + quantitative CSV output

**Hardware (development):**
- Detection/training: MacBook Pro M1 Max (32 GB unified memory)
- Classification server: iMac 2012 (Intel i5 2.7 GHz, 8 GB, macOS Catalina 10.15.8)
- Storage: Synology NAS (`/Volumes/External DATA/Team1/MLGlom/`)

**Hardware (planned deployment):**
- Mac Studio M4 / M4 Max — dedicated inference + Flask server

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

## Model Training History — Detection (nnU-Net)

| Round | Slides | Glomeruli | Val Dice | Val slides | Notes |
|---|---|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 | LysM_03 | Manual annotations only |
| Round 2 | 6 | 1,199 | 0.821 | LysM_06 | +3 slides corrected from predictions |
| Round 3 | 11 | 2,239 | 0.803 | LysM_01+02 | All LysM slides |
| Round 4 | 16 | **3,084** | **0.824** | LysM_01+02 | +5 Kidney WT normal slides |

### Dataset composition (Round 4)

| Slide | Glomeruli | Type |
|---|---|---|
| LysM_01–11 | 2,356 | Pathological (inflammatory/autoimmune) |
| Kidney 3 WT | 197 | Normal |
| Kidney 4 WTa | 158 | Normal |
| Kidney 4 WTb | 122 | Normal |
| Kidney 5 WTa | 127 | Normal |
| Kidney 5 WTb | 124 | Normal |
| **Total** | **3,084** | |

### Round 4 detection recall

| Dataset | Annotated | Detected | Recall |
|---|---|---|---|
| LysM (pathological) | 2,356 | 2,285 | 97.0% |
| Kidney WT (normal) | 728 | 719 | 98.8% |
| **Total** | **3,084** | **3,004** | **97.4%** |

### Checkpoints
- Round 2 backup: `fold_0_round2_backup/`
- Round 4 current: `checkpoint_best.pth` (Val Dice 0.824)

---

## Classifier Training History — Classification (ResNet50)

| Round | Patches | Macro F1 | mAP | Notes |
|---|---|---|---|---|
| Round 1 | 2,780 | **0.672** | **0.722** | 50 epochs, threshold 0.5 |

### Round 1 per-class F1

| Class | F1 | Training examples |
|---|---|---|
| Fibrosis | 0.891 | 858 |
| Crescent | 0.888 | 855 |
| Normal | 0.786 | 1,105 |
| Fibrinoid necrosis | 0.737 | 433 |
| Thickening GBM | 0.712 | 740 |
| Sclerosis | 0.632 | 323 |
| Adhesion | 0.490 | 468 |
| Hypercellularity | 0.211 | 76 ← priority for Round 2 |

> `best_model.pth` saved at epoch 45. Target for Round 2: n≥150 Hypercellularity examples.

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
| 6 | Fibrosis | `6` | Glomerular or interstitial fibrosis |
| 7 | Crescent | `7` | Epithelial or fibrous crescent |
| 8 | Sclerosis | `8` | Global or segmental sclerosis |
| 9 | Double glomerulus | `9` | Two contiguous glomeruli — manual correction needed |
| 10 | Not a glom | `0` | False positive from nnU-Net detector |

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
  → masks_to_geojson.py → GeoJSON polygons
  → import in QuPath → correct → retrain (active learning)

PHASE 2 — PATCH EXTRACTION (M1 Max → NAS/local)
─────────────────────────────────────────────────
  → extract_patches.py
  → 600×600 px PNG per glomerulus
  → patches_metadata.json (polygon in local coords for overlay)
  → /Desktop/MLGlom/patches/

PHASE 3 — MANUAL CLASSIFICATION (iMac Flask)
─────────────────────────────────────────────
  → python3 ~/GlomClassifier/app.py
  → http://iMac_IP:5000
  → 10-class multi-label, shuffle seed=42
  → Segmentation overlay (yellow dashed polygon)
  → /Desktop/MLGlom/labels/labels.csv

PHASE 4 — CLASSIFIER TRAINING (M1 Max)
────────────────────────────────────────
  → train_classifier.py
  → ResNet50 fine-tuning, PyTorch MPS, 50 epochs
  → /Desktop/MLGlom/models/best_model.pth

PHASE 5 — AUTOMATED INFERENCE (M1 Max)
────────────────────────────────────────
  → predict_classifier.py
  → results_per_glom.csv + results_per_slide.csv
  → per_slide_csv/SLIDE_results.csv
  → geojson/SLIDE_classified.geojson (→ QuPath import)

PHASE 6 — HUMAN VALIDATION (iMac Flask)
─────────────────────────────────────────
  → python3 ~/GlomClassifier/validation.py
  → http://iMac_IP:5000  — model predictions pre-checked
  → http://iMac_IP:5000/results — live dashboard (auto-refresh 30s)
  → /Desktop/MLGlom/labels/validated_labels.csv
```

---

## PART 1 — Environment Setup

### 1.1 Python environment (M1 Max — Miniforge)

```bash
brew install miniforge
conda init bash
# Close and reopen Terminal
conda create -n stardist-glom python=3.10 -y
# Close and reopen Terminal
conda activate stardist-glom

pip install tensorflow-macos tensorflow-metal
pip install stardist numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools && pip install "numpy<2"
pip install shapely opencv-python-headless && pip install "numpy<2"
pip install nnunetv2 && pip install "numpy<2"
pip install tifffile scikit-learn torchvision
```

> ⚠️ Always: `source ~/.bash_profile` THEN `conda activate stardist-glom`
> After ANY pip install: verify numpy → `python -c "import numpy; print(numpy.__version__)"` → must be `1.26.x`

### 1.2 nnU-Net environment variables

```bash
echo 'export nnUNet_raw="/Users/antonino/QuPath/nnunet_data/nnUNet_raw"' >> ~/.bash_profile
echo 'export nnUNet_preprocessed="/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed"' >> ~/.bash_profile
echo 'export nnUNet_results="/Users/antonino/QuPath/nnunet_data/nnUNet_results"' >> ~/.bash_profile
```

### 1.3 Flask server (iMac — macOS Catalina)

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Miniconda3-latest-MacOSX-x86_64.sh
# Close and reopen Terminal
python3 -m ensurepip --upgrade
python3 -m pip install flask pillow pandas

# Manual classification
cd ~ && python3 ~/GlomClassifier/app.py

# Validation mode (with model predictions)
cd ~ && python3 ~/GlomClassifier/validation.py
```

### 1.4 File structure

```
/Users/antonino/Desktop/MLGlom/
    ├── patches/                    ← 600×600 px PNG (3,004 files)
    ├── patches_metadata.json       ← polygon coords per patch
    ├── labels/
    │   ├── labels.csv             ← manual classification
    │   └── validated_labels.csv   ← human validation of model predictions
    ├── models/
    │   ├── best_model.pth         ← ResNet50 best checkpoint
    │   ├── model_meta.json        ← classes, threshold, metadata
    │   └── training_log.csv       ← per-epoch metrics
    └── results/
        ├── results_per_glom.csv   ← per-patch predictions + probabilities
        ├── results_per_slide.csv  ← aggregated per slide
        ├── per_slide_csv/         ← one CSV per slide
        └── geojson/               ← SLIDE_classified.geojson for QuPath
```

---

## PART 2 — QuPath Setup

> **Every visible glomerulus MUST be annotated exhaustively.**
> Unannotated glomerulus = background = training error.

### QuPath crash fix (large images)

```bash
rm "/Users/antonino/Library/Preferences/qupath.plist"
rm "/Users/antonino/Library/Application Support/CrashReporter/QuPath*.plist"
# Pre-generate thumbnails
conda activate stardist-glom && python3 << 'EOF'
from PIL import Image; import os, json; Image.MAX_IMAGE_PIXELS = None
src="/Users/antonino/Desktop/Export pics"; data="/Users/antonino/Desktop/GlomAndreMarc/data"
with open("/Users/antonino/Desktop/GlomAndreMarc/project.qpproj") as f: project=json.load(f)
for i,entry in enumerate(project.get('images',[]),1):
    name=entry.get('imageName',''); p=f"{src}/{name}"; thumb=f"{data}/{i}/thumbnail.jpg"
    if os.path.exists(p) and not os.path.exists(thumb):
        img=Image.open(p); img.thumbnail((500,500)); img.save(thumb,"JPEG"); print(f"✓ {name}")
EOF
```

### Kidney images — pyramidal TIFF conversion (required for QuPath)

```bash
conda activate stardist-glom
python -c "
import numpy as np; from PIL import Image; import tifffile; Image.MAX_IMAGE_PIXELS = None
for name in ['Kidney 3 WT','Kidney 4 WTa','Kidney 4 WTb','Kidney 5 WTa','Kidney 5 WTb']:
    img = np.array(Image.open(f'/Users/antonino/Desktop/Export pics/{name}.jpg').convert('RGB'))
    tifffile.imwrite(f'/Users/antonino/Desktop/Export pics/{name}.tiff',
                     img, tile=(512,512), compression='jpeg', subfiletype=1)
    print(f'✓ {name}.tiff')
"
```

---

## PART 3 — QuPath Scripts

### export_nnunet.groovy
```groovy
import qupath.lib.regions.RegionRequest; import qupath.lib.scripting.QP
import javax.imageio.ImageIO; import java.awt.image.BufferedImage
import java.awt.Color; import java.awt.geom.AffineTransform
def outputDir="/Users/antonino/QuPath/nnunet_data/raw", className="Glomerulus"; def downsample=4.0
def imageData=QP.getCurrentImageData(); def server=imageData.getServer()
def imageName=server.getMetadata().getName().replaceAll("\\.[^.]+\$","")
new File("${outputDir}/images").mkdirs(); new File("${outputDir}/masks").mkdirs()
int W=(int)(server.getWidth()/downsample), H=(int)(server.getHeight()/downsample)
println "→ Image: ${imageName} | Output size: ${W}×${H}"
def region=RegionRequest.createInstance(server.getPath(),downsample,0,0,server.getWidth(),server.getHeight())
ImageIO.write(server.readRegion(region),"PNG",new File("${outputDir}/images/${imageName}_0000.png"))
def mask=new BufferedImage(W,H,BufferedImage.TYPE_BYTE_GRAY); def g2d=mask.createGraphics()
g2d.setColor(Color.BLACK); g2d.fillRect(0,0,W,H); g2d.setColor(Color.WHITE)
def annotations=QP.getAnnotationObjects().findAll{it.getPathClass()?.getName()==className}
println "→ ${annotations.size()} '${className}' annotations found"
annotations.each{def t=new AffineTransform();t.scale(1.0/downsample,1.0/downsample);g2d.fill(t.createTransformedShape(it.getROI().getShape()))}
g2d.dispose(); ImageIO.write(mask,"PNG",new File("${outputDir}/masks/${imageName}.png"))
println "✓ Export complete for ${imageName}"
```

### import_nnunet_predictions.groovy
```groovy
import qupath.lib.io.GsonTools; import java.nio.file.Files, Paths; import com.google.gson.JsonParser
def geojsonPath="/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_01_detections.geojson"
def hierarchy=getCurrentImageData().getHierarchy()
hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null).findAll{!it.isRootObject()},true)
def json=new String(Files.readAllBytes(Paths.get(geojsonPath)))
def featuresArr=JsonParser.parseString(json).getAsJsonObject().getAsJsonArray("features")
def objects=GsonTools.getInstance().fromJson(featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType())
hierarchy.addObjects(objects.collect{qupath.lib.objects.PathObjects.createAnnotationObject(it.getROI(),getPathClass("Glomerulus"))})
fireHierarchyUpdate(); println "✓ ${objects.size()} annotations imported"
```

### import_nnunet_classification.groovy (with probability measurements)
```groovy
import qupath.lib.io.GsonTools; import java.nio.file.Files, Paths; import com.google.gson.JsonParser
def geojsonPath="/Users/antonino/Desktop/MLGlom/results/geojson/LysM_01_classified.geojson"
def hierarchy=getCurrentImageData().getHierarchy()
hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null).findAll{!it.isRootObject()},true)
def json=new String(Files.readAllBytes(Paths.get(geojsonPath)))
def root=JsonParser.parseString(json).getAsJsonObject(); def featuresArr=root.getAsJsonArray("features")
def newAnnotations=[]
featuresArr.each{featElem->
    def feat=featElem.getAsJsonObject(); def props=feat.getAsJsonObject("properties")
    def singleFC="""{"type":"FeatureCollection","features":[${featElem.toString()}]}"""
    def objects=GsonTools.getInstance().fromJson(
        JsonParser.parseString(singleFC).getAsJsonObject().getAsJsonArray("features"),
        new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType())
    if(!objects) return; def roi=objects[0].getROI()
    def className="Unclassified"
    if(props.has("classification")){def cls=props.getAsJsonObject("classification");if(cls.has("name"))className=cls.get("name").getAsString()}
    def annotation=qupath.lib.objects.PathObjects.createAnnotationObject(roi,getPathClass(className))
    if(props.has("glom_classes")) annotation.setName(props.get("glom_classes").getAsString())
    def ml=annotation.getMeasurementList()
    ["prob_Normal","prob_Adhesion","prob_Thickening_GBM","prob_Fibrinoid_necrosis",
     "prob_Hypercellularity","prob_Fibrosis","prob_Crescent","prob_Sclerosis","n_classes"].each{key->
        if(props.has(key)) ml.put(key,props.get(key).getAsDouble())}
    ml.close(); newAnnotations<<annotation}
hierarchy.addObjects(newAnnotations); fireHierarchyUpdate()
println "✓ ${newAnnotations.size()} classified glomeruli imported with measurements"
```

---

## PART 4 — nnU-Net Training Sequence

```bash
source ~/.bash_profile && conda activate stardist-glom

python /Users/antonino/QuPath/training/prepare_nnunet.py
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
python /Users/antonino/QuPath/training/create_split.py
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs

# Resume after interruption
nnUNetv2_train 1 2d 0 --npz -device mps -tr nnUNetTrainer_100epochs --c
```

> ⚠️ If `--c` skips training: `mv checkpoint_final.pth checkpoint_final_backup.pth` then rerun

---

## PART 5 — Inference (nnU-Net)

```bash
# Prepare all images
python -c "
import numpy as np; from PIL import Image; from pathlib import Path; from glob import glob
Image.MAX_IMAGE_PIXELS = None
INPUT='/Users/antonino/Desktop/Export pics'; OUTPUT='/Users/antonino/QuPath/nnunet_data/predict_input'
Path(OUTPUT).mkdir(parents=True,exist_ok=True)
for p in sorted(glob(f'{INPUT}/LysM_*.jpg'))+sorted(glob(f'{INPUT}/Kidney*.jpg')):
    name=Path(p).stem; img=np.array(Image.open(p).convert('RGB'))
    Image.fromarray(img[::4,::4,1]).save(f'{OUTPUT}/{name}_0000.png'); print(f'✓ {name}')
"

nnUNetv2_predict \
  -i /Users/antonino/QuPath/nnunet_data/predict_input \
  -o /Users/antonino/QuPath/nnunet_data/predictions \
  -d 1 -c 2d -tr nnUNetTrainer_100epochs -f 0 -device mps

python /Users/antonino/QuPath/training/masks_to_geojson.py
```

---

## PART 6 — Patch Extraction

```bash
conda activate stardist-glom
python /Users/antonino/QuPath/training/extract_patches.py
```

- Patch size: **600×600 px** full resolution
- Output: `/Desktop/MLGlom/patches/` + `patches_metadata.json`
- Naming: `LysM_01_0001.png`, `Kidney 3 WT_0001.png`

---

## PART 7 — Manual Classification (Flask — iMac)

```bash
cd ~ && python3 ~/GlomClassifier/app.py
# → http://iMac_IP:5000
```

- Shuffle seed=42 (reproducible inter-slide mixing)
- Segmentation overlay: yellow dashed polygon
- Saves to: `/Desktop/MLGlom/labels/labels.csv`
- Keyboard: `1`–`9`,`0`=classes | `←``→`=save & navigate | `Space`=zoom

---

## PART 8 — Classifier Training (ResNet50)

```bash
conda activate stardist-glom
pip install scikit-learn torchvision
python /Users/antonino/QuPath/training/train_classifier.py
```

- ResNet50 pretrained ImageNet → multi-label BCE loss
- Phase 1 (0–9): backbone frozen, lr=1e-4
- Phase 2 (10–50): backbone unfrozen, lr=1e-5/1e-4
- WeightedRandomSampler + pos_weight for class imbalance
- Output: `/Desktop/MLGlom/models/best_model.pth`

---

## PART 9 — Automated Inference (ResNet)

```bash
conda activate stardist-glom
python /Users/antonino/QuPath/training/predict_classifier.py
```

Output:
```
/Desktop/MLGlom/results/
    ├── results_per_glom.csv       ← patch, slide, classes, prob_* per class
    ├── results_per_slide.csv      ← aggregated stats per slide
    ├── per_slide_csv/             ← one CSV per slide
    └── geojson/
        └── SLIDE_classified.geojson  ← import in QuPath (colors + measurements)
```

Color coding in QuPath: green=Normal, orange=Adhesion, blue=Thickening GBM,
red=Fibrinoid necrosis, purple=Hypercellularity, dark orange=Fibrosis,
pink=Crescent, grey=Sclerosis

---

## PART 10 — Human Validation (Flask — iMac)

```bash
cd ~ && python3 ~/GlomClassifier/validation.py
# → Validation:  http://iMac_IP:5000
# → Live results: http://iMac_IP:5000/results
```

- Model predictions pre-checked (blue dot = model predicted this class)
- Saves `confirmed` or `corrected` status to `validated_labels.csv`
- `/results` dashboard: auto-refresh 30s, dual radar (model vs human) per slide
- Thumbnail colors: blue=predicted/unseen, green=confirmed, orange=corrected

---

## PART 11 — Planned Deployment (Mac Studio)

Target hardware: **Mac Studio M4 / M4 Max** (dedicated, always-on server)

```
USER (any workstation on network)
  → upload scan JPEG/TIFF via browser
MAC STUDIO (Flask server + inference)
  → nnU-Net detection (~30-40s/slide on M4 Max)
  → patch extraction
  → ResNet classification (~8 min / 3000 patches)
  → results dashboard + export PDF/CSV
  → manual amendment interface
NAS
  → scan storage + results archive
```

Zero dependency on QuPath in deployment. Full pipeline triggered by upload.

---

## Technical Notes

| Parameter | Value |
|---|---|
| Glomerulus diameter | ~200 px at full resolution |
| Export downsample | ×4 → ~7200×4060 px (LysM) |
| Classification patch size | 600×600 px full resolution |
| Annotation rule | Exhaustive per slide — no exceptions |
| Optimal channel for nnU-Net | Green (index 1) |
| GPU backend | Apple MPS (`-device mps`) |
| numpy | Must stay at 1.26.4 |
| PyTorch | 2.11.0 | TensorFlow | 2.16.2 | nnU-Net | 2.7.0 |
| ResNet threshold | 0.5 (adjustable in predict_classifier.py) |
