# yolov8_cls_full.py

from ultralytics import YOLO
import torch
import os
import shutil

def main():
    # Check GPU availability
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Training will run on CPU, which is much slower!")


    dataset_path = r"C:\Users\user\Desktop\Decentrathon\projectvenv\Datasets\larger_dirty_clean"

    # Verify dataset structure
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"⚠️ {split} folder not found: {split_path}")

    # Load YOLOv8 classification model
    model = YOLO("yolov8l-cls.pt")  # choose size: n/s/m/l

    # Train
    model.train(
        data=dataset_path,
        epochs=40,
        imgsz=224,       # 224 is standard for classification
        batch=32,
        device=0,
        augment=True
    )

    # Validate on val set
    print("\n🔎 Evaluating on validation set...")
    val_metrics = model.val(split="val")
    print(val_metrics)

    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_metrics = model.val(split="test")
    print(test_metrics)

    # Auto-save trained model
    best_model_path = model.ckpt_path  # path to best.pt
    dst = r"C:\Users\user\Desktop\Decentrathon\projectvenv\saved_models\saved_yolo_clean_dirty_binary\best.pt"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(best_model_path, dst)
    print(f"✅ Model automatically saved to: {dst}")

    # Export (optional)
    model.export(format="onnx")

if __name__ == "__main__":
    main()
