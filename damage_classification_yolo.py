from ultralytics import YOLO
from pathlib import Path

import os
import random
import shutil
from pathlib import Path
from typing import List, Set, Dict  # <-- use typing generics for <3.9

# ----------------------------
# 1) CONFIG — change paths if needed
# ----------------------------
DAMAGED_ROOT = Path(r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\car_integrity_cls")
CARCOMP_EXTERNAL_ROOT = Path(r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\archive (2)\Car parts\External")
NEW_ROOT = Path(r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\car_integrity_binary")

RNG_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------------
# 2) Prepare destination folders
# ----------------------------
splits = ["train", "val", "test"]
classes = ["damaged", "undamaged"]

for split in splits:
    for cls in classes:
        out_dir = NEW_ROOT / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 3) Helpers
# ----------------------------
def list_images(root: Path) -> List[Path]:
    if not root.exists():
        raise RuntimeError(f"Path not found: {root}")
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def copy_many(paths: List[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, out_dir / p.name)

# ----------------------------
# 4) Gather DAMAGED per split
# ----------------------------
damaged_counts: Dict[str, int] = {}
damaged_paths: Dict[str, List[Path]] = {}
for split in splits:
    ddir = DAMAGED_ROOT / split / "damaged"
    files = list_images(ddir)
    if not files:
        raise RuntimeError(f"No damaged images found in: {ddir}")
    damaged_paths[split] = files
    damaged_counts[split] = len(files)

print("Damaged counts:", damaged_counts)  # expect {'train': 2816, 'val': 810, 'test': 374}

# ----------------------------
# 5) UNDAMAGED candidates (CarComp External)
# ----------------------------
carcomp_all = list_images(CARCOMP_EXTERNAL_ROOT)
if not carcomp_all:
    raise RuntimeError(f"No images found under: {CARCOMP_EXTERNAL_ROOT}")
print(f"Found {len(carcomp_all)} candidate undamaged images in CarComp External.")

# De-duplicate by filename
unique_by_name: Dict[str, Path] = {}
for p in carcomp_all:
    key = p.name.lower()
    if key not in unique_by_name:
        unique_by_name[key] = p
carcomp_unique: List[Path] = list(unique_by_name.values())
print(f"Unique by filename: {len(carcomp_unique)}")

# ----------------------------
# 6) Copy DAMAGED into new dataset
# ----------------------------
for split in splits:
    out_dir = NEW_ROOT / split / "damaged"
    copy_many(damaged_paths[split], out_dir)
    print(f"Copied {len(damaged_paths[split])} damaged -> {out_dir}")

# ----------------------------
# 7) Sample equal UNDAMAGED per split and copy
# ----------------------------
random.seed(RNG_SEED)

def sample_undamaged(n: int, used: Set[str]) -> List[Path]:
    pool = [p for p in carcomp_unique if p.name.lower() not in used]
    if len(pool) < n:
        pool = carcomp_unique  # fallback if we run out of unique names
    if len(pool) < n:
        raise RuntimeError(f"Not enough undamaged images to sample {n}. Only {len(pool)} available.")
    return random.sample(pool, n)

used_names: Set[str] = set()
for split in splits:
    need = damaged_counts[split]
    chosen = sample_undamaged(need, used_names)
    for p in chosen:
        used_names.add(p.name.lower())
    out_dir = NEW_ROOT / split / "undamaged"
    copy_many(chosen, out_dir)
    print(f"Copied {len(chosen)} undamaged -> {out_dir}")

# ----------------------------
# 8) Write YOLOv8 classification YAML
# ----------------------------
yaml_text = f"""# Binary car integrity classification (damaged vs undamaged)
path: {NEW_ROOT.as_posix()}
train: train
val: val
test: test
names:
  0: damaged
  1: undamaged
"""
(NEW_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
print(f"Wrote YAML -> {(NEW_ROOT / 'data.yaml')}")
print("Done.")

DATA_DIR = Path(r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\car_integrity_binary")

model = YOLO("yolov8s-cls.pt")

# Train
model.train(
    data=str(DATA_DIR),   # <-- folder, not YAML
    epochs=30,
    imgsz=224,
    batch=64,
    device=0,
    workers=0
)

# Validate on test split
model.val(
    data=str(DATA_DIR),
    split="test"
)

# Predict a few
model.predict(
    source=str(DATA_DIR / "val" / "undamaged"),
    save=True
)
