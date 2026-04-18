"""
Prepare nnU-Net dataset from QuPath exports.
Extracts green channel, converts masks to binary.
Run after export_nnunet.groovy on all slides.
"""

import os, json
import numpy as np
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

print(f"\n✓ {len(cases)} cases prepared")
print(f"✓ Cases: {cases}")
