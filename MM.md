# Materials and Methods
## Automated Detection and Classification of Glomeruli in Kidney Sections

---

### 1. Image Acquisition

Kidney cryosections were stained using the Carstairs method and digitized as brightfield images. Whole-slide images were acquired in JPEG format at two resolutions: 28,928 × 16,240 pixels (LysM series, n=11) and 33,120 × 18,304 pixels (Kidney WT series, n=5). A total of 16 whole-slide images were analyzed, corresponding to sections from two biological models: a murine model of inflammatory/autoimmune nephropathy (LysM-Cre lineage, n=11) and wild-type control kidneys with normal renal histology (Kidney WT, n=5). The LysM and Kidney WT sections were prepared in separate staining batches, introducing inter-batch variability in staining intensity. The estimated resolution was approximately 1 µm/pixel, consistent with the expected murine glomerular diameter of approximately 200 µm (~200 pixels at full resolution).

---

### 2. Image Analysis Software

All image visualization, manual annotation, and object import/export were performed using **QuPath 0.7.0** (arm64 build for Apple Silicon; https://qupath.github.io) on a MacBook Pro with an Apple M1 Max processor (32 GB unified memory). Deep learning model training and inference were performed using Python 3.10 within a dedicated conda environment (Miniforge, Apple Silicon native), with TensorFlow 2.16.2 and PyTorch 2.11.0 as backends, both leveraging the Apple Metal Performance Shaders (MPS) GPU framework.

The glomerulus classification interface and the human validation interface were deployed as local Flask web applications (Python 3.11, macOS Catalina 10.15.8) on a dedicated iMac (Intel Core i5 2.7 GHz, 8 GB RAM), accessible from any workstation on the local network. Classified patches and labels were stored on a Synology NAS, with local copies on the MacBook Pro for model training.

---

### 3. Manual Annotation

Glomeruli were manually annotated on all 16 whole-slide images using the QuPath Brush annotation tool. Each annotation encompassed the glomerular tuft including Bowman's capsule. A critical constraint of the nnU-Net training framework was strictly enforced: **all visible glomeruli within each slide were annotated exhaustively**, as any unannotated glomerulus would be incorrectly interpreted as background during model training. A total of **3,084 glomeruli** were annotated across 16 slides (LysM: 2,356 glomeruli, range 163–302; Kidney WT: 728 glomeruli, range 122–197). Annotations were exported as paired full-slide images and binary segmentation masks using a custom Groovy script within QuPath, with a spatial downsampling factor of 4×.

---

### 4. Deep Learning Model for Glomerulus Detection

#### 4.1 Model Architecture

Glomerulus segmentation was performed using **nnU-Net v2** (version 2.7.0), a self-configuring deep learning framework for biomedical image segmentation [Isensee et al., Nature Methods, 2021]. The framework automatically selected a 2D U-Net with 9 encoding stages, feature maps ranging from 32 to 512 channels, instance normalization, and LeakyReLU activations. The patch size was automatically set to 896 × 1,792 pixels with a batch size of 2. Z-score intensity normalization was applied to the input channel.

#### 4.2 Input Preprocessing

The green channel (index 1 of the RGB image) was extracted from each downsampled whole-slide image, as it provided optimal contrast between glomerular structures and surrounding tissue in Carstairs-stained sections.

#### 4.3 Iterative Active Learning Strategy

Model training followed an **active learning iterative strategy** over four rounds:

**Round 1** — Initial training on 3 manually annotated slides (total: 516 glomeruli). Validation Dice: **0.765**.

**Round 2** — Three additional slides predicted by Round 1, corrected by expert review, added to training (6 slides, 1,199 glomeruli). Validation Dice: **0.821** (+7.3%).

**Round 3** — All 11 LysM slides annotated exhaustively (2,239 glomeruli; 9 training / 2 validation). Validation Dice: **0.803** (stricter 2-slide criterion).

**Round 4** — Five wild-type kidney slides (Kidney WT, distinct staining batch) added for morphological and inter-batch diversity. Total: 16 slides, 3,084 glomeruli. Validation Dice: **0.824** (+2.6% vs Round 3 on same criterion).

#### 4.4 Training Parameters

All training runs used: 100 epochs, polynomial learning rate decay from 0.01, batch size 2, patch size 896 × 1,792 pixels, Apple MPS GPU (~370–420 s/epoch on M1 Max). A manual slide-level cross-validation split was defined to prevent data leakage.

#### 4.5 Inference and Post-processing

Inference was performed using `nnUNetv2_predict` with `checkpoint_best.pth` (fold 0). Binary segmentation masks were post-processed using connected component labeling (scikit-image) with a minimum area threshold of 500 pixels² at downsampled resolution (~90 µm diameter equivalent). Detected glomeruli were exported as GeoJSON polygon files.

---

### 5. Glomerulus Detection Performance

| Training Round | Slides (n) | Glomeruli (n) | Validation Dice |
|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 |
| Round 2 | 6 | 1,199 | 0.821 |
| Round 3 | 11 | 2,239 | 0.803 |
| Round 4 | 16 | **3,084** | **0.824** |

Detection recall on final Round 4 model:

| Dataset | Annotated | Detected | Recall |
|---|---|---|---|
| LysM (pathological) | 2,356 | 2,285 | 97.0% |
| Kidney WT (normal) | 728 | 719 | 98.8% |
| **Total** | **3,084** | **3,004** | **97.4%** |

---

### 6. Glomerulus Classification

#### 6.1 Patch Extraction

For each detected glomerulus, a 600 × 600 pixel patch was extracted from the full-resolution image, centered on the glomerulus centroid as computed from the GeoJSON polygon. Border glomeruli where the full patch could not be extracted were excluded. A total of 3,004 patches were extracted across all 16 slides. A metadata JSON file stored the polygon coordinates in local patch coordinates, enabling display of the segmentation contour overlay in the classification interface. Patches were shuffled with a fixed random seed (seed=42) to ensure inter-slide mixing and prevent order-dependent bias.

#### 6.2 Manual Classification

Manual classification was performed using a custom local Flask web application deployed on a dedicated iMac. The interface displayed each glomerulus patch at full resolution with the nnU-Net segmentation contour overlaid as a dashed yellow line. Navigation between patches automatically triggered label saving. A total of **2,922 patches** were manually classified by one expert observer.

The following 10 morphological categories were used:

| Class | N patches | Description |
|---|---|---|
| Normal | 1,105 | Preserved architecture |
| Fibrosis | 858 | Glomerular or periglomerular fibrosis |
| Crescent | 855 | Epithelial or fibrous crescent |
| Thickening GBM | 740 | GBM thickening |
| Adhesion | 468 | Capsular synechia |
| Fibrinoid necrosis | 433 | Fibrinoid necrosis |
| Sclerosis | 323 | Global or segmental sclerosis |
| Hypercellularity | 76 | Leukocyte infiltration / proliferation |
| Double glomerulus | 99 | Two contiguous glomeruli (excluded from training) |
| Not a glom | 43 | False positive from detector (excluded from training) |

Classification was performed as a **multi-label** task. Of the 2,780 patches eligible for classifier training, 326 received a single class label, 107 two classes, 49 three classes, and 39 four or more classes.

#### 6.3 Automated Classifier Training

A **ResNet50** convolutional neural network (pretrained on ImageNet) was fine-tuned for multi-label glomerulus classification using PyTorch 2.11.0 with Apple MPS GPU acceleration. The classification head was replaced by a two-layer fully connected network (Dropout 0.4 → Linear(2048→256) → ReLU → Dropout 0.3 → Linear(256→8 classes)).

Training followed a two-phase strategy: backbone frozen for the first 10 epochs (warmup, head lr=1×10⁻⁴), then unfrozen with differentiated learning rates (head: 1×10⁻⁴; backbone: 1×10⁻⁵). Total: 50 epochs, batch size 16, AdamW optimizer, cosine annealing learning rate schedule. Input images resized to 224×224 px with ImageNet normalization. Data augmentation: random horizontal/vertical flips, 90° rotations, color jitter.

Class imbalance was addressed by: (1) `BCEWithLogitsLoss` with inverse-frequency positive weights; (2) `WeightedRandomSampler` oversampling patches containing the rarest active class. Dataset split: 85% training (n=2,363) / 15% validation (n=417), stratified random sampling (seed=42). Best checkpoint selected on macro-averaged F1 score.

#### 6.4 Classifier Performance — Round 1

| Class | F1 score | Training examples |
|---|---|---|
| Fibrosis | 0.891 | 858 |
| Crescent | 0.888 | 855 |
| Normal | 0.786 | 1,105 |
| Fibrinoid necrosis | 0.737 | 433 |
| Thickening GBM | 0.712 | 740 |
| Sclerosis | 0.632 | 323 |
| Adhesion | 0.490 | 468 |
| Hypercellularity | 0.211 | 76 |
| **Macro F1** | **0.672** | |
| **mAP** | **0.722** | |

Best checkpoint at epoch 45. Classes with strong morphological distinctiveness (Crescent, Fibrosis) achieved near-excellent discrimination. Hypercellularity showed limited performance due to small training set (n=76) and visual overlap with Fibrinoid necrosis. A second round of targeted manual labeling (target: n≥150) is planned before Round 2 classifier training.

#### 6.5 Automated Inference and Export

The trained ResNet50 classifier was applied to all 3,004 detected glomerulus patches using `predict_classifier.py`. For each patch, class probabilities were computed using sigmoid activation; classes with probability ≥0.5 were assigned as positive. Results were exported as: (1) a per-glomerulus CSV with all class probabilities; (2) per-slide aggregated statistics; (3) GeoJSON files for QuPath import, with color-coded annotations based on the primary class (highest positive probability) and per-class probabilities stored as QuPath measurements.

#### 6.6 Global Inference Results (Round 1 Classifier, Threshold 0.5)

| Class | Positive glomeruli | % of total |
|---|---|---|
| Thickening GBM | 1,324 | 44.1% |
| Adhesion | 1,133 | 37.7% |
| Normal | 1,032 | 34.4% |
| Crescent | 972 | 32.4% |
| Fibrosis | 930 | 31.0% |
| Sclerosis | 744 | 24.8% |
| Fibrinoid necrosis | 728 | 24.2% |
| Hypercellularity | 635 | 21.1% |

Kidney WT slides showed near-exclusive Normal classification (96–100%), confirming model biological validity. LysM slides showed heterogeneous pathological profiles consistent with inflammatory/autoimmune nephropathy.

#### 6.7 Human Validation Interface

To enable expert review and correction of automated predictions, a dedicated validation Flask application (`validation.py`) was developed. The interface pre-loaded model predictions as initial class selections for each patch. The expert could confirm predictions (registered as "confirmed") or modify them (registered as "corrected"), with all interactions saved to `validated_labels.csv` with timestamp and user identifier.

A live results dashboard (`/results` route, auto-refresh every 30 seconds) displayed for each slide: overall validation progress, proportion of confirmed vs corrected patches, and dual radar charts comparing model predictions (all patches) against human-validated labels (validated patches only). This real-time feedback was designed to motivate and inform the validation process.

---

### 7. Planned Next Steps

- Targeted manual labeling of additional Hypercellularity patches (target: n≥150)
- Classifier Round 2 retraining with expanded dataset
- Deployment on dedicated Mac Studio (M4 / M4 Max) as always-on server
- Development of upload-based pipeline: scan upload → automatic detection → classification → results dashboard → manual amendment interface
- Quantitative analysis: per-slide lesion burden, pure vs. mixed lesion proportions, correlation with biological phenotype

---

### 8. Software Availability

All scripts for annotation export, dataset preparation, model training, inference, patch extraction, manual classification, automated classification, human validation, and results visualization are available at: [GitHub repository URL]

**Key dependencies:** QuPath 0.7.0, nnU-Net v2.7.0, Python 3.10/3.11, PyTorch 2.11.0, TensorFlow 2.16.2, torchvision, scikit-learn, scikit-image, shapely, Flask 3.1, pandas.

---

### References

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18(2):203–211.

Bankhead P, et al. QuPath: Open source software for digital pathology image analysis. *Scientific Reports*. 2017;7(1):16878.

He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *Proceedings of the IEEE CVPR*. 2016:770–778.
