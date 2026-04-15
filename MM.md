# Materials and Methods
## Automated Detection and Classification of Glomeruli in Kidney Sections

---

### 1. Image Acquisition

Kidney cryosections were stained using the Carstairs method and digitized as brightfield images. Whole-slide images were acquired in JPEG format at two resolutions: 28,928 × 16,240 pixels (LysM series, n=11) and 33,120 × 18,304 pixels (Kidney WT series, n=5). A total of 16 whole-slide images were analyzed, corresponding to sections from two biological models: a murine model of inflammatory/autoimmune nephropathy (LysM-Cre lineage, n=11) and wild-type control kidneys with normal renal histology (Kidney WT, n=5). The LysM and Kidney WT sections were prepared in separate staining batches, introducing inter-batch variability in staining intensity. No pixel size calibration metadata was embedded in the image files; the estimated resolution was approximately 1 µm/pixel, consistent with the expected murine glomerular diameter of approximately 200 µm (~200 pixels at full resolution).

---

### 2. Image Analysis Software

All image visualization, manual annotation, and object import/export were performed using **QuPath 0.7.0** (arm64 build for Apple Silicon; https://qupath.github.io) on a MacBook Pro with an Apple M1 Max processor (32 GB unified memory). Deep learning model training and inference were performed using Python 3.10 within a dedicated conda environment (Miniforge, Apple Silicon native), with TensorFlow 2.16.2 and PyTorch 2.11.0 as backends, both leveraging the Apple Metal Performance Shaders (MPS) GPU framework. The glomerulus classification interface was deployed as a local Flask web application (Python 3.11, macOS Catalina 10.15.8) on a dedicated iMac (Intel Core i5 2.7 GHz, 8 GB RAM), accessible from any workstation on the local network. Classified patches and labels were stored on a Synology NAS accessible via SMB.

---

### 3. Manual Annotation

Glomeruli were manually annotated on all 16 whole-slide images using the QuPath Brush annotation tool. Each annotation encompassed the glomerular tuft including Bowman's capsule. A critical constraint of the nnU-Net training framework was strictly enforced: **all visible glomeruli within each slide were annotated exhaustively**, as any unannotated glomerulus would be incorrectly interpreted as background during model training. A total of **3,084 glomeruli** were annotated across 16 slides (LysM: 2,356 glomeruli across 11 slides, range 163–302; Kidney WT: 728 glomeruli across 5 slides, range 122–197). Annotations were exported as paired full-slide images and binary segmentation masks using a custom Groovy script within QuPath, with a spatial downsampling factor of 4× (output size: ~7,200 × 4,060 px for LysM; ~8,300 × 4,600 px for Kidney WT).

---

### 4. Deep Learning Model for Glomerulus Detection

#### 4.1 Model Architecture

Glomerulus segmentation was performed using **nnU-Net v2** (version 2.7.0), a self-configuring deep learning framework for biomedical image segmentation [Isensee et al., Nature Methods, 2021]. The framework automatically determined the optimal training configuration from the dataset fingerprint, selecting a 2D U-Net with 9 encoding stages, feature maps ranging from 32 to 512 channels, instance normalization, and LeakyReLU activations. The patch size was automatically set to 896 × 1,792 pixels with a batch size of 2. Z-score intensity normalization was applied to the input channel.

#### 4.2 Input Preprocessing

The green channel (index 1 of the RGB image) was extracted from each downsampled whole-slide image, as it provided optimal contrast between glomerular structures and surrounding tissue in Carstairs-stained sections. This single-channel input was used for all training, validation, and inference steps.

#### 4.3 Iterative Active Learning Strategy

Model training followed an **active learning iterative strategy** over four rounds, progressively expanding the training dataset through model-assisted annotation correction:

**Round 1** — Initial training on 3 manually annotated slides (LysM_01: 193, LysM_02: 165, LysM_03: 158 glomeruli; total: 516). A leave-one-out cross-validation split was defined. Training was performed for 100 epochs using the `nnUNetTrainer_100epochs` variant. The final mean validation Dice coefficient was **0.765**.

**Round 2** — Three additional slides (LysM_04, LysM_05, LysM_06) were predicted using the Round 1 model, imported into QuPath as annotations, manually corrected by expert review (addition of missed glomeruli, deletion of false positives), and added to the training dataset. Retraining on 6 slides (1,199 glomeruli; 5 training / 1 validation) yielded a mean validation Dice of **0.821** (+7.3% vs Round 1).

**Round 3** — All 11 LysM slides were annotated exhaustively and used for training (2,239 glomeruli; 9 training / 2 validation). Mean validation Dice: **0.803** (on the stricter 2-slide validation set LysM_01 + LysM_02).

**Round 4** — Five wild-type kidney slides with normal renal histology (Kidney WT, distinct staining batch) were added to the training dataset, providing morphologically normal glomeruli and inter-batch diversity. Total dataset: 16 slides, 3,084 glomeruli (9 training / 2 validation, LysM_01 + LysM_02). Mean validation Dice: **0.824** (+2.6% vs Round 3 on the same validation criterion).

#### 4.4 Training Parameters

All training runs used the following parameters: 100 epochs, polynomial learning rate decay from 0.01, batch size 2, patch size 896 × 1,792 pixels, Apple MPS GPU acceleration (~370–420 s/epoch on M1 Max). A manual cross-validation split was defined based on slide-level leave-two-out rotation to prevent data leakage between training and validation at the slide level.

#### 4.5 Inference and Post-processing

For inference on new whole-slide images, each image was preprocessed identically to the training pipeline (4× downsampling, green channel extraction). Prediction was performed using `nnUNetv2_predict` with the best checkpoint (`checkpoint_best.pth`, fold 0). Binary segmentation masks were post-processed using connected component labeling (scikit-image) to extract individual glomerulus instances, with a minimum area threshold of 500 pixels² at downsampled resolution (equivalent to ~8,000 pixels² at full resolution, corresponding to ~90 µm diameter) applied to remove spurious detections. Detected glomeruli were exported as GeoJSON polygon files and imported into QuPath for visualization and downstream classification.

---

### 5. Model Performance

| Training Round | Slides (n) | Glomeruli (n) | Validation Dice | Validation slides |
|---|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 | LysM_03 |
| Round 2 | 6 | 1,199 | 0.821 | LysM_06 |
| Round 3 | 11 | 2,239 | 0.803 | LysM_01, LysM_02 |
| Round 4 | 16 | **3,084** | **0.824** | LysM_01, LysM_02 |

Detection recall on the final Round 4 model (comparing detections to exhaustive manual annotations):

| Dataset | Annotated | Detected | Recall |
|---|---|---|---|
| LysM (pathological) | 2,356 | 2,285 | 97.0% |
| Kidney WT (normal) | 728 | 719 | 98.8% |
| **Total** | **3,084** | **3,004** | **97.4%** |

The Dice coefficient was computed between the predicted binary mask and the manual annotation mask on held-out validation slides not used during training. The consistently high recall across both pathological and normal datasets, despite differing staining batches and glomerular morphology, demonstrates robust generalization of the Round 4 model.

---

### 6. Glomerulus Classification

#### 6.1 Patch Extraction

For each detected glomerulus, a 600 × 600 pixel patch was extracted from the full-resolution image, centered on the glomerulus centroid as computed from the GeoJSON polygon. Border glomeruli where the full patch could not be extracted were excluded. A total of approximately 3,000 patches were extracted across all 16 slides. Patches were stored as RGB PNG files on a network-attached storage server. A metadata JSON file stored the polygon coordinates in local patch coordinates, enabling downstream display of the segmentation overlay in the classification interface.

Patches were shuffled with a fixed random seed (seed=42) prior to classification to ensure inter-slide mixing and prevent order-dependent classification bias.

#### 6.2 Classification Scheme

Detected glomeruli were classified into the following morphological categories, selected based on the biological context of inflammatory/autoimmune nephropathy (LysM model) and normal renal histology (Kidney WT):

| Class | Description |
|---|---|
| Normal | Preserved glomerular architecture |
| Adhesion | Capsular synechia (glomerulotubular adhesion) |
| Thickening GBM | Glomerular basement membrane thickening |
| Fibrinoid necrosis | Fibrinoid necrosis of the capillary tuft |
| Hypercellularity | Intra-glomerular leukocyte infiltration or cellular proliferation |
| Fibrosis | Glomerular or periglomerular fibrosis |
| Crescent | Epithelial or fibrous crescent formation |
| Sclerosis | Global or segmental glomerulosclerosis |
| Double glomerulus | Two contiguous glomeruli not resolved by the detector (excluded from classifier training; flagged for manual correction) |
| Not a glom | False positive from the nnU-Net detector (excluded from classifier training) |

Classification was performed as a **multi-label** task: a single glomerulus could receive multiple non-exclusive classes simultaneously (e.g., Hypercellularity + Adhesion). The classes Double glomerulus and Not a glom were treated as exclusive and mutually incompatible with all pathological classes.

#### 6.3 Classification Interface

Manual classification was performed using a custom local web application (Flask 3.1, Python 3.11) deployed on a dedicated iMac and accessible from any workstation on the local network. The interface displayed each glomerulus patch at full resolution with the nnU-Net segmentation contour overlaid in yellow, enabling assessment of the detection quality alongside the morphological classification. Navigation was performed via keyboard shortcuts or thumbnail clicking; all navigation events triggered automatic label saving to a centralized CSV file on the NAS before loading the next patch.

---

### 7. Planned: Automated Classifier Training

Following manual classification of a representative subset of glomeruli (~150–200 patches covering all classes), a convolutional neural network classifier (ResNet architecture, transfer learning, PyTorch 2.11.0, Apple MPS GPU) will be trained on the labeled patches for automatic multi-label classification of the full glomerulus dataset. False positive patches (Not a glom) will additionally serve as hard negatives for a potential Round 5 of nnU-Net detector retraining.

---

### 8. Software Availability

All scripts used for annotation export, dataset preparation, model training, inference, patch extraction, and classification are available at: [GitHub repository URL]

**Key dependencies:** QuPath 0.7.0, nnU-Net v2.7.0, Python 3.10, PyTorch 2.11.0, TensorFlow 2.16.2, scikit-image, shapely, Flask 3.1, pandas.

---

### References

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18(2):203–211.

Bankhead P, et al. QuPath: Open source software for digital pathology image analysis. *Scientific Reports*. 2017;7(1):16878.
