# Materials and Methods
## Automated Detection and Classification of Glomeruli in Kidney Sections

---

### 1. Image Acquisition

Kidney cryosections were stained using the Carstairs method and digitized as brightfield images (JPEG format, 28,928 × 16,240 pixels or 28,800 × 16,240 pixels). A total of 11 whole-slide images were acquired (LysM_01 through LysM_11), corresponding to sections from a murine model of inflammatory/autoimmune nephropathy and ischemia-reperfusion injury. No pixel size calibration metadata was available in the JPEG files; the estimated resolution was approximately 1 µm/pixel based on the measured diameter of individual glomeruli (~200 pixels, consistent with the expected ~200 µm diameter of murine glomeruli).

---

### 2. Image Analysis Software

All image visualization, manual annotation, and object import/export were performed using **QuPath 0.7.0** (arm64 build for Apple Silicon; https://qupath.github.io) running on a MacBook Pro with an Apple M1 Max processor (32 GB unified memory). Deep learning model training and inference were performed using Python 3.10 within a dedicated conda environment (Miniforge, Apple Silicon native), with TensorFlow 2.16.2 and PyTorch 2.11.0 as backends, both leveraging the Apple Metal Performance Shaders (MPS) GPU framework for hardware acceleration.

---

### 3. Manual Annotation

Glomeruli were manually annotated on all 11 whole-slide images using the QuPath Brush annotation tool. Each annotation encompassed the glomerular tuft including Bowman's capsule. A critical constraint of the nnU-Net training framework was strictly enforced: **all visible glomeruli within each slide were annotated exhaustively**, as any unannotated glomerulus would be incorrectly interpreted as background during model training. A total of **2,239 glomeruli** were annotated across 11 slides (range: 158–290 per slide; mean: 203.5 ± 35.4). Annotations were exported as paired full-slide images and binary segmentation masks using a custom Groovy script within QuPath, with a spatial downsampling factor of 4× applied to reduce computational requirements (output image size: ~7,200 × 4,060 pixels).

---

### 4. Deep Learning Model for Glomerulus Detection

#### 4.1 Model Architecture

Glomerulus segmentation was performed using **nnU-Net v2** (version 2.7.0), a self-configuring deep learning framework for biomedical image segmentation based on the U-Net architecture [Isensee et al., Nature Methods, 2021]. The framework automatically determined the optimal training configuration from the dataset fingerprint, selecting a 2D U-Net with 9 encoding stages, feature maps ranging from 32 to 512 channels, instance normalization, and LeakyReLU activations. The patch size was automatically set to 896 × 1,792 pixels with a batch size of 2.

#### 4.2 Input Preprocessing

Prior to model training and inference, the green channel (index 1 of the RGB image) was extracted from each downsampled whole-slide image, as it provided optimal contrast between the glomerular structures and surrounding tissue in Carstairs-stained sections. Intensity normalization was performed using Z-score normalization as determined by the nnU-Net automatic configuration.

#### 4.3 Iterative Training Strategy

Model training followed an **active learning iterative strategy** over three rounds:

**Round 1** — Initial training on 3 manually annotated slides (LysM_01: 193, LysM_02: 165, LysM_03: 158 glomeruli; total: 516). A manual cross-validation split was defined (2 training / 1 validation) due to the limited number of cases. Training was performed for 100 epochs using the `nnUNetTrainer_100epochs` variant. The final mean validation Dice coefficient was **0.765**.

**Round 2** — Three additional slides (LysM_04, LysM_05, LysM_06) were predicted using the Round 1 model, imported into QuPath as annotations, manually corrected by an expert (addition of missed glomeruli, deletion of false positives), and added to the training dataset. Retraining on 6 slides (1,199 glomeruli total; 5 training / 1 validation) yielded a mean validation Dice of **0.821**, representing a 7.3% improvement over Round 1.

**Round 3** — All 11 slides were fully annotated exhaustively and used for training (2,239 glomeruli total; 9 training / 2 validation). Training is ongoing.

#### 4.4 Training Parameters

All training runs used the following parameters: 100 epochs, learning rate initialized at 0.01 with polynomial decay, batch size 2, patch size 896 × 1,792 pixels, MPS (Apple Metal) GPU acceleration. Training duration was approximately 10 hours per round on the M1 Max processor (~370 seconds per epoch).

#### 4.5 Inference

For inference on new whole-slide images, each image was preprocessed identically to the training pipeline (4× downsampling, green channel extraction). Prediction was performed using `nnUNetv2_predict` with the best checkpoint (`checkpoint_best.pth`). Binary segmentation masks were post-processed using connected component labeling (scikit-image) to extract individual glomerulus instances, with a minimum area threshold of 500 pixels² (equivalent to ~8,000 pixels² at full resolution) applied to remove spurious detections. Detected glomeruli were exported as GeoJSON polygon files and imported into QuPath for visualization and downstream classification.

---

### 5. Model Performance

| Training Round | Slides (n) | Glomeruli (n) | Validation Dice |
|---|---|---|---|
| Round 1 | 3 | 516 | 0.765 |
| Round 2 | 6 | 1,199 | **0.821** |
| Round 3 | 11 | 2,239 | In progress |

The Dice coefficient was computed between the predicted binary mask and the manual annotation mask on held-out validation slides not used during training.

---

### 6. Glomerulus Classification (Planned)

Detected glomeruli will be classified into the following morphological categories, selected based on the biological context of inflammatory/autoimmune nephropathy and ischemia-reperfusion injury: Normal, Inflammatory infiltrate, Mesangial expansion, Crescent formation, Global sclerosis, Segmental sclerosis, Ischemic collapse, and Intracapillary thrombosis. Classification will be performed using a convolutional neural network (ResNet architecture) trained on labeled glomerular patches extracted from full-resolution images. Ground truth labels will be assigned using a dedicated graphical annotation interface displaying individual glomerulus crops, allowing expert-guided classification.

---

### 7. Software Availability

All scripts used for annotation export, dataset preparation, model training, inference, and QuPath integration are available at: [GitHub repository URL to be added]

**Key dependencies:**
- QuPath 0.7.0: https://qupath.github.io
- nnU-Net v2: https://github.com/MIC-DKFZ/nnUNet
- Python 3.10, PyTorch 2.11.0, TensorFlow 2.16.2
- scikit-image, shapely, numpy 1.26.4

---

### References

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18(2):203-211. https://doi.org/10.1038/s41592-020-01008-z

Bankhead P, et al. QuPath: Open source software for digital pathology image analysis. *Scientific Reports*. 2017;7(1):16878. https://doi.org/10.1038/s41598-017-17204-5
