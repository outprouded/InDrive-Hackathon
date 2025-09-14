import os
from pathlib import Path
from io import BytesIO
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import timm
from torchvision import models, transforms

# ======================
# CONFIG — EDIT PATHS
# ======================
CFG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "damage_cls": {
        "yolov8s_cls": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\damage_class_yolo\best.pt",
        "vit": {
            "weights": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\damage_class_vis\vit_2025-09-14_12_01_19.pth",
            "timm_arch": "vit_base_patch16_224",
            "tv_arch": "vit_b_16",
            "num_classes": 2,
            "class_names": ["damaged", "undamaged"]
        }
    },
    "damage_det": {
        "yolov8s_det": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\damage_detection\best.pt"
    },
    "det_names": ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"],
    "clean_cls": {
        "yolov8l_cls": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\cleanliness_class\best.pt"
    },
    "clean_class_names": ["clean", "dirty"],
    "severity_cls": {
        "yolov8l_cls": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\dirtyness_class_yolo\best.pt",
        "vgg": {
            "weights": r"C:\Users\azamat.shmitov\Downloads\indrive decentrathon\dirtyness_class_vgg\vgg16_best.pth",
            "arch": "vgg16",
            "num_classes": 3,
            "class_names": ["slightly_dirty", "dirty", "very_dirty"]
        }
    }
}

# ----------------------
# Loaders (strict, no silent fallback)
# ----------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model_strict(weight_path: str, expected_num_classes: Optional[int] = None):
    """Load YOLO weights and verify class count (no fallback, no name override)."""
    p = Path(weight_path)
    if not p.exists():
        raise FileNotFoundError(f"Model weights not found: {p}")
    model = YOLO(str(p))
    names = model.names
    if isinstance(names, dict):
        n_classes = len(names)
    elif names is not None:
        n_classes = len(list(names))
    else:
        n_classes = None
    if expected_num_classes is not None and n_classes is not None and n_classes != expected_num_classes:
        raise RuntimeError(f"Loaded YOLO model has {n_classes} classes. Expected {expected_num_classes}. Path: {p}")
    return model

@st.cache_resource(show_spinner=False)
def load_vit_from_ckpt(weights_path: str, timm_arch: str, tv_arch: str, num_classes: int, device: str):
    """Loads ViT from ckpt, detecting whether ckpt is torchvision-ViT or timm-ViT."""
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    keys = list(state.keys())

    def _is_torchvision(keys_list):
        first = " ".join(keys_list[:50])
        return ("conv_proj" in first) or any(k.startswith("encoder.") for k in keys_list) or ("heads.head.weight" in keys_list)

    if _is_torchvision(keys):
        if not hasattr(models, tv_arch):
            raise ValueError(f"Unknown torchvision ViT arch '{tv_arch}'. Try 'vit_b_16', 'vit_b_32', etc.")
        model = getattr(models, tv_arch)(weights=None)
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feats, num_classes)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        src = "torchvision"
    else:
        model = timm.create_model(timm_arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        src = "timm"
    return model, src

@st.cache_resource(show_spinner=False)
def load_vgg_strict(ckpt_path: str, arch: str, num_classes: int, device: str):
    """Build exact torchvision VGG variant and load checkpoint STRICTLY."""
    if not hasattr(models, arch):
        raise ValueError(f"Unknown VGG arch '{arch}'. Use: vgg16, vgg16_bn, vgg19, vgg19_bn")
    model = getattr(models, arch)(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    model = model.to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load VGG weights strictly.\n- Ensure CFG['severity_cls']['vgg']['arch'] matches training arch.\n- Ensure num_classes matches your trained head.\nOriginal error:\n{e}")
    return model

# ----------------------
# Inference helpers
# ----------------------
IMAGENET_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def pil_from_bytes(file) -> Image.Image:
    return Image.open(BytesIO(file.read())).convert("RGB")

def torch_predict_single(model, img_pil: Image.Image, class_names: List[str]):
    t = IMAGENET_TFMS(img_pil).unsqueeze(0).to(CFG["device"])
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), dict(zip(class_names, probs.tolist()))

def yolo_cls_predict(yolo_model: YOLO, img: Image.Image, imgsz=224):
    ydev = 0 if torch.cuda.is_available() else "cpu"
    r = yolo_model.predict(source=img, imgsz=imgsz, device=ydev, verbose=False)[0]
    idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    probs = r.probs.data.cpu().numpy()
    names = yolo_model.names
    names_list = [names[i] for i in range(len(probs))] if isinstance(names, dict) else list(names)
    return names_list[idx], conf, dict(zip(names_list, probs.tolist()))

def yolo_det_predict(yolo_model: YOLO, img: Image.Image, imgsz=1280, conf=0.25, iou=0.7):
    ydev = 0 if torch.cuda.is_available() else "cpu"
    r = yolo_model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, device=ydev, verbose=False)[0]
    anno = Image.fromarray(r.plot())
    names = yolo_model.names
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            dets.append({
                "label": names[int(c)] if isinstance(names, dict) else str(int(c)),
                "conf": float(s),
                "xyxy": [float(x) for x in b.tolist()]
            })
    return anno, dets

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Car Integrity Pipeline", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved design
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0055aa;
    }
    .stSlider .st-bd {
        background-color: #e6f0fa;
        border-radius: 8px;
    }
    .stSelectbox .st-bm {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .stFileUploader .st-ds {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .stTextInput .st-ds {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .sidebar .stMarkdown {
        font-size: 14px;
        color: #333333;
    }
    .sidebar .stSelectbox, .sidebar .stSlider, .sidebar .stCheckbox {
        margin-bottom: 15px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛠️ Car Integrity Analysis")
st.markdown("**Advanced vehicle inspection pipeline for damage detection, cleanliness classification, and severity assessment.**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("Model Selection", expanded=True):
        stageA_choice = st.selectbox("Damage Classifier", ["YOLOv8s-cls", "ViT (visual transformer)"], index=0)
        stageD_choice = st.selectbox("Severity Classifier", ["YOLOv8l-cls", "VGG (custom)"], index=0)

    with st.expander("Decision Thresholds", expanded=True):
        thr_damage = st.slider("Trigger detection if 'Damaged' confidence ≥", 0.0, 1.0, 0.50, 0.01)
        thr_dirty = st.slider("Trigger severity if 'Dirty' confidence ≥", 0.0, 1.0, 0.50, 0.01)
        det_conf = st.slider("Detection Confidence", 0.05, 0.9, 0.25, 0.01)
        det_iou = st.slider("Detection IoU (NMS)", 0.30, 0.90, 0.70, 0.01)

    with st.expander("Display Options", expanded=True):
        show_probs = st.checkbox("Show Probability Tables", value=True)
        show_dets = st.checkbox("Show Detection JSON", value=False)

    st.markdown("---")
    st.caption(f"**Device:** {CFG['device']}")
    st.markdown("**Logic:** Always classify *damage* and *cleanliness*. If *damaged* → run detection. If *dirty* → run severity.")

# Image inputs
st.subheader("📤 Upload Images or Specify Folder")
col_up, col_dir = st.columns([3, 2])
with col_up:
    files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
with col_dir:
    folder = st.text_input("Or enter a folder path", value="", placeholder="e.g., C:/images/")

images = []
if files:
    for f in files:
        try:
            images.append(("uploaded:" + f.name, pil_from_bytes(f)))
        except Exception:
            st.warning(f"Could not read {f.name}", icon="⚠️")
elif folder and Path(folder).exists():
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            try:
                images.append((str(p), Image.open(p).convert("RGB")))
            except Exception:
                pass

if not images:
    st.info("Please upload images or provide a valid folder path to proceed.", icon="ℹ️")
    st.stop()

# Load models
with st.spinner("Loading models..."):
    if stageA_choice == "YOLOv8s-cls":
        dmg_cls_model = load_yolo_model_strict(CFG["damage_cls"]["yolov8s_cls"], expected_num_classes=2)
        dmg_cls_kind = "yolo"
    else:
        vit_cfg = CFG["damage_cls"]["vit"]
        dmg_cls_model, vit_src = load_vit_from_ckpt(
            vit_cfg["weights"], vit_cfg["timm_arch"], vit_cfg["tv_arch"], vit_cfg["num_classes"], CFG["device"]
        )
        dmg_cls_kind = "vit"
        st.sidebar.info(f"Loaded ViT from: {vit_src}")

    det_model = load_yolo_model_strict(CFG["damage_det"]["yolov8s_det"], expected_num_classes=6)
    clean_model = load_yolo_model_strict(CFG["clean_cls"]["yolov8l_cls"], expected_num_classes=2)

    if stageD_choice == "YOLOv8l-cls":
        sev_model = load_yolo_model_strict(CFG["severity_cls"]["yolov8l_cls"], expected_num_classes=3)
        sev_kind = "yolo"
    else:
        vgg_cfg = CFG["severity_cls"]["vgg"]
        sev_model = load_vgg_strict(vgg_cfg["weights"], vgg_cfg["arch"], vgg_cfg["num_classes"], CFG["device"])
        sev_kind = "vgg"

st.success("Models loaded successfully!", icon="✅")

# Process images with progress bar
st.subheader("📊 Analysis Results")
progress_bar = st.progress(0)
for idx, (name, img) in enumerate(images, 1):
    progress_bar.progress(idx / len(images))
    st.markdown(f"#### Image {idx}: `{name}`")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Input Image", use_container_width=True)

    # Damage classification
    with st.spinner("Running damage classification..."):
        if dmg_cls_kind == "yolo":
            dmg_label, dmg_conf, dmg_probs = yolo_cls_predict(dmg_cls_model, img, imgsz=224)
        else:
            dmg_label, dmg_conf, dmg_probs = torch_predict_single(
                dmg_cls_model, img, CFG["damage_cls"]["vit"]["class_names"]
            )

    # Cleanliness classification
    with st.spinner("Running cleanliness classification..."):
        cl_label, cl_conf, cl_probs = yolo_cls_predict(clean_model, img, imgsz=224)

    route_text = f"**Damage:** {dmg_label} ({dmg_conf:.3f})  |  **Cleanliness:** {cl_label} ({cl_conf:.3f})"
    anno, dets, sev_out = None, None, None

    # Damage detection
    if dmg_label.lower().startswith("damag") and dmg_conf >= thr_damage:
        with st.spinner("Running damage detection..."):
            anno, dets = yolo_det_predict(det_model, img, imgsz=1280, conf=det_conf, iou=det_iou)
        route_text += "  →  **Detection ran**"

    # Severity classification
    if cl_label.lower() == "dirty" and cl_conf >= thr_dirty:
        with st.spinner("Running severity classification..."):
            if sev_kind == "yolo":
                s_label, s_conf, s_probs = yolo_cls_predict(sev_model, img, imgsz=224)
            else:
                s_label, s_conf, s_probs = torch_predict_single(
                    sev_model, img, CFG["severity_cls"]["vgg"]["class_names"]
                )
            sev_out = (s_label, s_conf, s_probs)
            route_text += f"  →  **Severity:** {s_label} ({s_conf:.3f})"

    with c2:
        st.markdown(route_text)
        if show_probs:
            with st.expander("Probability Details", expanded=False):
                st.markdown("**Damage Probabilities:**")
                st.dataframe({k: [f"{v:.3f}"] for k, v in dmg_probs.items()})
                st.markdown("**Cleanliness Probabilities:**")
                st.dataframe({k: [f"{v:.3f}"] for k, v in cl_probs.items()})
                if sev_out is not None:
                    st.markdown("**Severity Probabilities:**")
                    st.dataframe({k: [f"{v:.3f}"] for k, v in sev_out[2].items()})

        if anno is not None:
            st.image(anno, caption="Damage Detection Output", use_container_width=True)
            if show_dets:
                with st.expander("Detection Details (JSON)", expanded=False):
                    st.json(dets or [])

    st.markdown("---")

progress_bar.empty()
st.markdown("**Analysis complete!** 🎉")