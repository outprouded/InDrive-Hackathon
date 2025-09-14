import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



def main():
    # --- Check GPU ---
    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Dataset paths ---
    dataset_root = r"D:\studying\university\Decentrathon\hackathon\levels_of_dirty\levels_of_dirty"
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")

    # --- Data transforms ---
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Datasets ---
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, transform=data_transforms["val"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
        for x in ["train", "val", "test"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # --- Load pretrained VGG16 ---
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False  # freeze feature extractor

    # Replace classifier head
    model.classifier[6] = nn.Linear(4096, num_classes)

    model = model.to(device)

    # --- Loss + Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    # --- Training loop ---
    epochs = 50
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # --- Save best model ---
    model.load_state_dict(best_model_wts)
    save_dir = r"D:\studying\university\Decentrathon\hackathon\runs\vgg16"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vgg16_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Best model saved to {save_path} with val acc {best_acc:.4f}")

    # --- Evaluate on test set ---
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"📊 Test Accuracy: {acc:.4f}")

    print("\n--- Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Save report to file
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (VGG16)")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
