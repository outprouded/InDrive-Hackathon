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

    # Dataset root
    dataset_root = r"C:\Users\user\Desktop\Decentrathon\projectvenv\Datasets\levels_of_dirty"

    # Verify dataset structure
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            print(f"⚠️ {split} folder not found: {split_path}")

    # 1. Load YOLOv8 classification model (choose size depending on VRAM)
    model = YOLO("yolov8l-cls.pt")  # use 's' or 'm' for 3060, 'l' might OOM

    # 2. Train
    model.train(
        data=dataset_root,   # dataset root containing train/val/test
        epochs=100,
        imgsz=224,           # keep train/val consistent
        batch=32,            # adjust if OOM (try 8 or 4 if needed)
        device=0             # GPU
    )

    # 3. Validate (will automatically use test set if available)
    results = model.val(
        data=dataset_root,
        imgsz=224,
        batch=32,
        device=0
    )
    print("Validation Results:", results)

    # 4. Save trained model (best.pt)
    run_dir = model.trainer.save_dir   # training run folder
    best_model_path = os.path.join(run_dir, "weights", "best.pt")
    dst = r"C:\Users\user\Desktop\Decentrathon\projectvenv\saved_models\best.pt"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(best_model_path, dst)
    print(f"✅ Model automatically saved to: {dst}")

    # 5. Export model (optional)
    model.export(format="onnx")   # or 'torchscript', 'engine', etc.

if __name__ == "__main__":
    main()
