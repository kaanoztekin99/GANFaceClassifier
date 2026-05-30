import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "model_weights"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

NUM_CLASSES = len(train_dataset.classes)


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


model = SimpleCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def train_one_epoch():
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(train_dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(loader, dataset):
    model.eval()
    total_loss = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds, all_probs


def save_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()


def save_accuracy_curve(train_accs, val_accs):
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
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


def calculate_binary_metrics(y_true, y_pred, class_names):
    real_idx = class_names.index("real")

    y_true_binary = [0 if y == real_idx else 1 for y in y_true]
    y_pred_binary = [0 if y == real_idx else 1 for y in y_pred]

    binary_cm = confusion_matrix(y_true_binary, y_pred_binary)
    binary_report = classification_report(
        y_true_binary,
        y_pred_binary,
        target_names=["Real", "Fake"],
        digits=4
    )

    return binary_cm, binary_report


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


def main():
    print(f"Using device: {DEVICE}")
    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    best_val_loss = float("inf")
    early_stop_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc, _, _, _ = evaluate(val_loader, val_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_cnn_5class.pth")
        else:
            early_stop_counter += 1

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Patience: {early_stop_counter}/{PATIENCE}"
        )

        if early_stop_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    torch.save(model.state_dict(), MODEL_DIR / "last_cnn_5class.pth")

    save_loss_curve(train_losses, val_losses)
    save_accuracy_curve(train_accs, val_accs)

    model.load_state_dict(torch.load(MODEL_DIR / "best_cnn_5class.pth", map_location=DEVICE))

    test_loss, test_acc, y_true, y_pred, y_probs = evaluate(test_loader, test_dataset)

    class_names = test_dataset.classes

    save_confusion_matrix(y_true, y_pred, class_names)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    binary_cm, binary_report = calculate_binary_metrics(
        y_true,
        y_pred,
        class_names
    )

    save_binary_confusion_matrix(binary_cm)

    print("\nTest Loss:", test_loss)
    print("5-Class Test Accuracy:", test_acc)
    print("\n5-Class Classification Report:\n", report)
    print("\nBinary Real/Fake Report:\n", binary_report)

    with open(OUTPUT_DIR / "classification_report_5class.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"5-Class Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    with open(OUTPUT_DIR / "classification_report_binary.txt", "w") as f:
        f.write(binary_report)

    history = {
        "classes": class_names,
        "class_mapping": train_dataset.class_to_idx,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accs,
        "val_accuracy": val_accs,
        "test_loss": test_loss,
        "test_accuracy_5class": test_acc,
        "early_stopping_patience": PATIENCE,
        "best_val_loss": best_val_loss
    }

    with open(OUTPUT_DIR / "training_history_5class.json", "w") as f:
        json.dump(history, f, indent=4)

    np.save(OUTPUT_DIR / "test_true_labels.npy", np.array(y_true))
    np.save(OUTPUT_DIR / "test_pred_labels.npy", np.array(y_pred))
    np.save(OUTPUT_DIR / "test_pred_probs.npy", np.array(y_probs))

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("Saved model weights to:", MODEL_DIR)


if __name__ == "__main__":
    main()