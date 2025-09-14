from ultralytics import YOLO
from pathlib import Path
import json
import shutil

# ==== EDIT THESE ====
CARDD_COCO_ROOT = Path(r"C:/Users/azamat.shmitov/Downloads/indrive decentrathon/CarDD_release/CarDD_release/CarDD_COCO")
OUT_YOLO_ROOT   = Path(r"C:/Users/azamat.shmitov/Downloads/indrive decentrathon/cardd_yolo")
# ====================

# Choose correct split dir names if val is 'valid2017'
coco_img_dirs = {
    "train": "train2017",
    "val":   "val2017" if (CARDD_COCO_ROOT / "val2017").exists() else "valid2017",
    "test":  "test2017",
}
# Choose correct annotation filenames for val
val_json_name = "instances_val2017.json"
if not (CARDD_COCO_ROOT / "annotations" / val_json_name).exists():
    # common alternates
    if (CARDD_COCO_ROOT / "annotations" / "instances_valid2017.json").exists():
        val_json_name = "instances_valid2017.json"
    elif (CARDD_COCO_ROOT / "annotations" / "instances_validation2017.json").exists():
        val_json_name = "instances_validation2017.json"

coco_jsons = {
    "train": CARDD_COCO_ROOT / "annotations" / "instances_train2017.json",
    "val":   CARDD_COCO_ROOT / "annotations" / val_json_name,
    "test":  CARDD_COCO_ROOT / "annotations" / "instances_test2017.json",
}

# Make YOLO tree
for split in ("train", "val", "test"):
    (OUT_YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Class list (index order must match yaml)
names = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
name2idx = {n: i for i, n in enumerate(names)}

def coco_to_yolo_bbox(x, y, w, h, img_w, img_h):
    # COCO: [x_min, y_min, width, height]
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    ww = w / img_w
    hh = h / img_h
    # clip
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    ww = min(max(ww, 0.0), 1.0)
    hh = min(max(hh, 0.0), 1.0)
    return xc, yc, ww, hh

for split in ("train", "val", "test"):
    json_path = coco_jsons[split]
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category id -> yolo idx
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    # (If category names differ in capitalization/spacing, we map by name)
    id2idx = {}
    for cid, cname in id2name.items():
        if cname not in name2idx:
            raise KeyError(f"Category name '{cname}' not in expected list: {names}")
        id2idx[cid] = name2idx[cname]

    imgs = {im["id"]: im for im in coco["images"]}

    anns_by_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    src_img_dir = CARDD_COCO_ROOT / coco_img_dirs[split]
    out_img_dir = OUT_YOLO_ROOT / "images" / split
    out_lbl_dir = OUT_YOLO_ROOT / "labels" / split

    count_imgs = 0
    for img_id, im in imgs.items():
        fn = im["file_name"]
        src = src_img_dir / fn
        if not src.exists():
            # try to find by recursive search as fallback
            matches = list(src_img_dir.rglob(fn))
            if matches:
                src = matches[0]
            else:
                print(f"[WARN] Missing image: {src}")
                continue

        dst = out_img_dir / fn
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        W, H = im["width"], im["height"]
        label_lines = []
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"]
            if cid not in id2idx:
                continue
            cls = id2idx[cid]
            x, y, w, h = a["bbox"]
            xc, yc, ww, hh = coco_to_yolo_bbox(x, y, w, h, W, H)
            if ww <= 0 or hh <= 0:
                continue
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        lbl_path = out_lbl_dir / (Path(fn).stem + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))  # <-- complete line (no syntax error)

        count_imgs += 1

    print(f"[OK] {split}: wrote {count_imgs} images to {out_img_dir} and labels to {out_lbl_dir}")

# Write data.yaml
yaml_text = f"""# CarDD for YOLOv8 (converted from COCO)
path: {OUT_YOLO_ROOT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
"""
(OUT_YOLO_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
print(f"[READY] data.yaml -> {(OUT_YOLO_ROOT/'data.yaml').as_posix()}")

# Quick summary
for split in ("train", "val", "test"):
    ni = len(list((OUT_YOLO_ROOT / "images" / split).glob("*.jpg")))
    nl = len(list((OUT_YOLO_ROOT / "labels" / split).glob("*.txt")))
    print(f"{split}: {ni} images, {nl} labels")

data_yaml = r"C:/Users/azamat.shmitov/Downloads/indrive decentrathon/cardd_yolo/data.yaml"

model = YOLO("yolov8s.pt")
model.train(
    data=data_yaml,
    imgsz=1280,
    epochs=100,
    batch=-1,
    device=0,
    workers=8,
    mosaic=0.1,
    mixup=0.0,
    translate=0.05,
    scale=0.5,
    degrees=0.0,
    shear=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    patience=20
)
