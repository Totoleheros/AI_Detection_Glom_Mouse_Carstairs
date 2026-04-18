"""
Create slide-level cross-validation split for nnU-Net.
Prevents data leakage by separating whole slides into train/val sets.
Update slides list when adding new slides.
"""

import json
from pathlib import Path

# All slides — update when adding new ones
slides = [
    "LysM_01", "LysM_02", "LysM_03", "LysM_04", "LysM_05",
    "LysM_06", "LysM_07", "LysM_08", "LysM_09", "LysM_10", "LysM_11",
    "Kidney 3 WT", "Kidney 4 WTa", "Kidney 4 WTb",
    "Kidney 5 WTa", "Kidney 5 WTb"
]

# 5-fold: rotate 2 LysM + 1 Kidney as validation
split = [
    {"train": [s for s in slides if s not in ["LysM_01","LysM_02","Kidney 3 WT"]],
     "val":   ["LysM_01","LysM_02","Kidney 3 WT"]},
    {"train": [s for s in slides if s not in ["LysM_03","LysM_04","Kidney 4 WTa"]],
     "val":   ["LysM_03","LysM_04","Kidney 4 WTa"]},
    {"train": [s for s in slides if s not in ["LysM_05","LysM_06","Kidney 4 WTb"]],
     "val":   ["LysM_05","LysM_06","Kidney 4 WTb"]},
    {"train": [s for s in slides if s not in ["LysM_07","LysM_08","Kidney 5 WTa"]],
     "val":   ["LysM_07","LysM_08","Kidney 5 WTa"]},
    {"train": [s for s in slides if s not in ["LysM_09","LysM_10","Kidney 5 WTb"]],
     "val":   ["LysM_09","LysM_10","Kidney 5 WTb"]},
]

out = Path("/Users/antonino/QuPath/nnunet_data/nnUNet_preprocessed/Dataset001_GlomCarstairs/splits_final.json")
with open(out, "w") as f:
    json.dump(split, f, indent=2)

print(f"✓ Split written: {out}")
for i, fold in enumerate(split):
    print(f"   Fold {i}: train={len(fold['train'])} | val={fold['val']}")
