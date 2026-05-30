import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODEL_PATH = PROJECT_ROOT / "model_weights" / "best_cnn_5class.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Confusion Matrix - 5 Class")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_5class.png", dpi=300)
    plt.close()


def save_binary_confusion_matrix(binary_cm):
    classes = ["Real", "Fake"]

    plt.figure(figsize=(6, 5))
    plt.imshow(binary_cm)
    plt.title("Binary Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    for i in range(binary_cm.shape[0]):
        for j in range(binary_cm.shape[1]):
            plt.text(j, i, str(binary_cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_binary.png", dpi=300)
    plt.close()


def calculate_binary_metrics(y_true, y_pred, class_names):
    real_idx = class_names.index("0_real")

    y_true_binary = [0 if y == real_idx else 1 for y in y_true]
    y_pred_binary = [0 if y == real_idx else 1 for y in y_pred]

    binary_cm = confusion_matrix(y_true_binary, y_pred_binary)
    binary_report = classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=["Real", "Fake"],
        digits=4
    )

    binary_acc = accuracy_score(y_true_binary, y_pred_binary)

    return binary_cm, binary_report, binary_acc


def main():
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    class_names = test_dataset.classes

    print("Classes:", class_names)
    print("Class mapping:", test_dataset.class_to_idx)
    print("Test images:", len(test_dataset))

    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    test_acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    binary_cm, binary_report, binary_acc = calculate_binary_metrics(
        y_true,
        y_pred,
        class_names
    )

    save_confusion_matrix(y_true, y_pred, class_names)
    save_binary_confusion_matrix(binary_cm)

    print("\n5-Class Test Accuracy:", test_acc)
    print("\n5-Class Classification Report:\n", report)

    print("\nBinary Real/Fake Accuracy:", binary_acc)
    print("\nBinary Real/Fake Report:\n", binary_report)

    with open(OUTPUT_DIR / "classification_report_5class.txt", "w") as f:
        f.write(f"5-Class Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    with open(OUTPUT_DIR / "classification_report_binary.txt", "w") as f:
        f.write(f"Binary Real/Fake Accuracy: {binary_acc:.4f}\n\n")
        f.write(binary_report)

    history = {
        "classes": class_names,
        "class_mapping": test_dataset.class_to_idx,
        "test_accuracy_5class": test_acc,
        "test_accuracy_binary": binary_acc
    }

    with open(OUTPUT_DIR / "evaluation_saved_model.json", "w") as f:
        json.dump(history, f, indent=4)

    np.save(OUTPUT_DIR / "test_true_labels.npy", np.array(y_true))
    np.save(OUTPUT_DIR / "test_pred_labels.npy", np.array(y_pred))
    np.save(OUTPUT_DIR / "test_pred_probs.npy", np.array(y_probs))

    print("\nSaved evaluation outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()