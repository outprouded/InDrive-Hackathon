import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def validate():
    """Inference function to validate model quality"""

    # Load Model
    model = models.vit_b_16(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("model/vit.pth", map_location="cpu"))

    # Path to dataset
    dataset_path = "dataset\\car_integrity_binary"

    # Verify dataset structure
    for split in ["val", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"⚠️ {split} folder not found: {split_path}")

    # Standard preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    test_dataset = datasets.ImageFolder(
        root=dataset_path + "\\test", transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Prepare environment
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Turn model into evaluation mode
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            predicted = 1 - predicted.cpu().numpy()
            all_preds.extend(predicted)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = (
        torch.tensor(all_preds) == torch.tensor(all_targets)
    ).sum().item() / len(all_targets)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # --- Avg Loss, Accuracy ---
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # --- Precision, Recall, F1 ---
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))


def main():
    """Main file to train and evaluate model"""
    # Check GPU availability
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Training will run on CPU, which is much slower!")

    # Path to dataset
    dataset_path = "dataset\\car_integrity_binary"

    # Verify dataset structure
    for split in ["train"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"⚠️ {split} folder not found: {split_path}")

    # Load ViT classification model
    model = models.vit_b_16(pretrained=True)

    # Standard preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load data
    train_dataset = datasets.ImageFolder(
        root=dataset_path + "\\train", transform=transform
    )
    val_dataset = datasets.ImageFolder(root=dataset_path + "\\val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Freeze feature extractor (optional for transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    features = train_dataset.classes
    model.heads.head = nn.Linear(model.heads.head.in_features, len(features))

    # Prepare environment
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.head.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train
    num_epochs = 77
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Train Accuracy: {100 * correct / total:.2f}%"
        )

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save model
    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    torch.save(model.state_dict(), f"model/model_{time}.pth")

    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    validate()


if __name__ == "__main__":
    main()
