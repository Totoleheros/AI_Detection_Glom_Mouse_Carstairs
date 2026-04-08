# Pipeline de détection et classification de glomérules rénaux
**QuPath 0.7.0 + StarDist custom + Apple Silicon (M1 Max)**  
Coloration : Carstairs | Images : Brightfield JPEG (~28928 × 16240 px)

---

## Contexte

Pipeline semi-automatique pour :
1. Détecter les glomérules sur des coupes de rein (Brightfield, Carstairs)
2. Les classifier selon leur état pathologique (Sain vs Pathologique)

**Environnement cible :** Mac M1 Max, QuPath 0.7.0-arm64, Python via Miniforge

---

## Voies abandonnées (et pourquoi)

| Approche | Raison d'abandon |
|---|---|
| StarDist avec modèle pré-entraîné `he_heavy_augment.pb` | Détecte des noyaux individuels, pas des glomérules entiers |
| BioImage.IO Model Zoo | Aucun modèle glomérule/kidney disponible |
| Cellpose | Conçu pour cellules individuelles — pas adapté à des structures de ~200 µm complexes |
| Pixel Classifier Random Forest seul | Plafond de performance insuffisant pour un pipeline durable |
| Déconvolution couleur (Visual Stain Editor) | Non applicable à la coloration Carstairs (conçu pour H&E et H-DAB uniquement) |

---

## Architecture retenue

```
102 annotations manuelles QuPath (ground truth)
            ↓
     Export patches PNG + masques
     (script Groovy : export_training_patches.groovy)
            ↓
  Entraînement StarDist custom
  (Python, TensorFlow Metal, GPU M1 Max, ~45 min)
            ↓
  Modèle "glomerulus_carstairs"
            ↓
  Détection automatique dans QuPath
  (StarDist brightfield script, canal vert)
            ↓
  Classification Sain/Pathologique
  (Object Classifier QuPath)
```

---

## PARTIE 1 — Installation de l'environnement

### 1.1 QuPath

- Télécharger **QuPath 0.7.0-arm64** : https://qupath.github.io
- Installer dans `/Applications/`
- Vérifier : `Help → About QuPath` → Architecture = `aarch64`

### 1.2 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew --version
```

Résultat attendu :
```
Homebrew 5.0.16
```

### 1.3 Miniforge (conda natif Apple Silicon)

```bash
brew install miniforge
conda init bash
```

Résultat attendu :
```
modified      /Users/<username>/.bash_profile
==> For changes to take effect, close and re-open your current shell. <==
```

⚠️ **Fermer le Terminal complètement (Cmd+Q) et le réouvrir.**

```bash
conda --version
```

Résultat attendu :
```
conda 26.1.1
```

> **Note :** Si tu utilises zsh comme shell par défaut, remplace `conda init bash` par `conda init zsh`.
> Pour vérifier ton shell : `echo $SHELL`

### 1.4 Environnement Python dédié

```bash
conda create -n stardist-glom python=3.10 -y
```

⚠️ **Fermer le Terminal complètement (Cmd+Q) et le réouvrir.**

```bash
conda activate stardist-glom
python --version
```

Résultat attendu :
```
(stardist-glom) user$ python --version
Python 3.10.20
```

> Le préfixe `(stardist-glom)` dans le prompt confirme que l'environnement est actif.

### 1.5 Installation StarDist + TensorFlow Metal

```bash
pip install tensorflow-macos tensorflow-metal
pip install stardist
pip install numpy matplotlib tifffile scikit-image csbdeep jupyter
pip install gputools
pip install "numpy<2"   # Rétrogradation nécessaire — gputools force numpy 2.x qui casse TF
```

> ⚠️ L'ordre est important : installer `gputools` PUIS rétrograder numpy.
> Le conflit `reikna` affiché est sans conséquence pour ce pipeline.

Vérification :
```bash
python -c "import numpy; print(numpy.__version__); import tensorflow as tf; print(tf.__version__); import stardist; print('StarDist OK')"
```

Résultat attendu :
```
1.26.4
2.16.2
StarDist OK
```

---

## PARTIE 2 — Préparation du projet QuPath

### 2.1 Création du projet

- Lancer QuPath → `File → New Project`
- Nommer le projet `GlomAndreMarc`
- Importer les images (`LysM_01.jpg`, etc.)

### 2.2 Annotations manuelles (ground truth)

- 102 glomérules annotés manuellement avec l'outil Brush (classe `Glomerulus`)
- Classes supplémentaires : `Cortex Tissue` (26), `Medulla Tissue` (16), `White` (9)
- Total : 153 annotations

> Ces 102 annotations constituent le dataset d'entraînement du modèle StarDist custom.  
> **Ne pas supprimer** — elles serviront aussi à la classification Sain/Pathologique.

### 2.3 Export des patches d'entraînement

Dans QuPath : **`Automate`** → **`Script editor`**

Coller le script suivant → **`File → Save As`** → nommer `export_training_patches.groovy`

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

println "→ ${annotations.size()} annotations '${className}' trouvées"

if (annotations.isEmpty()) {
    println "⚠ Aucune annotation trouvée. Vérifie le nom de classe."
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
    if (count % 10 == 0) println "  ${count}/${annotations.size()} exportés..."
}

println "✓ Export terminé : ${count} paires image/masque dans ${outputDir}"
```

Résultat attendu dans la console QuPath :
```
INFO: → 102 annotations 'Glomerulus' trouvées
INFO:   10/102 exportés...
[...]
INFO: ✓ Export terminé : 102 paires image/masque dans /Users/antonino/QuPath/training_data
```

Vérification dans le Terminal :
```bash
ls /Users/antonino/QuPath/training_data/images/ | wc -l   # → 102
ls /Users/antonino/QuPath/training_data/masks/  | wc -l   # → 102
```

Structure générée :
```
/Users/antonino/QuPath/training_data/
    ├── images/
    │   ├── glom_000.png  (taille min observée : 248×225 px)
    │   └── ... (102 fichiers)
    └── masks/
        ├── glom_000.png
        └── ... (102 fichiers)
```

---

## PARTIE 3 — Entraînement StarDist custom

### 3.1 Script d'entraînement

Créer le dossier et le script :
```bash
mkdir -p /Users/antonino/QuPath/training
nano /Users/antonino/QuPath/training/train_stardist_glom.py
```

Contenu complet du script :

```python
import numpy as np
import os
from glob import glob
from skimage.io import imread as sk_imread
from csbdeep.utils import normalize
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D
import tensorflow as tf

print(f"TensorFlow : {tf.__version__}")
print(f"GPU disponible : {tf.config.list_physical_devices('GPU')}")

# ── PARAMÈTRES ────────────────────────────────────────────────
DATA_DIR   = "/Users/antonino/QuPath/training_data"
MODEL_DIR  = "/Users/antonino/QuPath/models"
MODEL_NAME = "glomerulus_carstairs"
PATCH_SIZE = (128, 128)   # Réduit à 128 car certains patches font 248×225 minimum
N_RAYS     = 32
EPOCHS     = 100
# ─────────────────────────────────────────────────────────────

img_paths  = sorted(glob(os.path.join(DATA_DIR, "images", "*.png")))
mask_paths = sorted(glob(os.path.join(DATA_DIR, "masks",  "*.png")))

print(f"\n→ {len(img_paths)} images trouvées")
print(f"→ {len(mask_paths)} masques trouvés")

assert len(img_paths) == len(mask_paths), "Nombre d'images et masques différent !"

images = []
masks  = []

for ip, mp in zip(img_paths, mask_paths):
    img = sk_imread(ip)
    if img.ndim == 3:
        img = img[:, :, 1].astype(np.float32)   # Canal vert (optimal pour Carstairs)
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
print(f"→ Images chargées. Taille min : {min_h}×{min_w} | max : {max(s[0] for s in sizes)}×{max(s[1] for s in sizes)}")

# Filtre les images trop petites pour le patch_size
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
    print(f"⚠ {skipped} images ignorées car trop petites pour PATCH_SIZE {PATCH_SIZE}")

print(f"→ {len(images_ok)} images retenues pour l'entraînement")

n_val   = max(1, int(len(images_ok) * 0.2))
n_train = len(images_ok) - n_val

X_train = images_ok[:n_train]
Y_train = masks_ok[:n_train]
X_val   = images_ok[n_train:]
Y_val   = masks_ok[n_train:]

print(f"→ Train : {n_train} | Validation : {n_val}")

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

print(f"\n→ Début de l'entraînement ({EPOCHS} epochs)...")
print(f"→ Modèle sauvegardé dans : {MODEL_DIR}/{MODEL_NAME}")
print("─" * 50)

model.train(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    augmenter       = None,
)

print("\n→ Optimisation du seuil de détection...")
model.optimize_thresholds(X_val, Y_val)

print(f"\n✓ Entraînement terminé.")
print(f"✓ Modèle disponible dans : {MODEL_DIR}/{MODEL_NAME}")
print(f"✓ Pour QuPath : utilise le dossier complet (pas un fichier .pb)")
```

### 3.2 Lancement

```bash
cd /Users/antonino/QuPath/training
python train_stardist_glom.py
```

Résultat attendu au démarrage :
```
TensorFlow : 2.16.2
GPU disponible : [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
→ 102 images trouvées
→ 102 masques trouvés
→ Images chargées. Taille min : 248×225 | max : 426×453
→ 102 images retenues pour l'entraînement
→ Train : 82 | Validation : 20
Metal device set to: Apple M1 Max
Epoch 1/100
100/100 ━━━━━━━━━━━━━━━━━━━━ 27s 160ms/step - loss: 8.89 ...
```

> Durée estimée : ~45 minutes sur M1 Max (27s/epoch × 100 epochs)  
> Ne pas fermer le Terminal pendant l'entraînement.

### 3.3 Résultat attendu

```
/Users/antonino/QuPath/models/glomerulus_carstairs/
    ├── config.json
    ├── thresholds.json
    └── weights_best.h5
```

---

## PARTIE 4 — Détection dans QuPath

*(Section à compléter après fin de l'entraînement)*

---

## PARTIE 5 — Classification Sain/Pathologique

*(Section à compléter)*

---

## Notes techniques

- Diamètre moyen d'un glomérule : **204 pixels**
- Taille min des patches exportés : **248×225 px** → PATCH_SIZE réduit à 128×128
- Calibration estimée : **1 µm/pixel**
- Coloration : **Carstairs** (multi-chromatique — la déconvolution standard ne s'applique pas)
- Canal optimal pour prétraitement : **Canal vert (index 1)**
- Glomérules : structures star-convexes ~200 µm, mauve dense sur fond rose
- GPU Metal confirmé actif pendant l'entraînement : Apple M1 Max, 32 GB
