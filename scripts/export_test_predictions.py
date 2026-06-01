import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODEL_PATH = PROJECT_ROOT / "model_weights" / "best_cnn_5class.pth"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test_predictions.json"

IMG_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "0_real",
    "1_dcgan",
    "2_progan",
    "3_stylegan2",
    "4_stylegan3"
]

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


def clean_label(label):
    return label.split("_", 1)[1]


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

    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = {}

    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(images.size(0)):
                image_path, true_label_idx = dataset.samples[sample_index]

                image_path = Path(image_path)

                relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()

                pred_idx = preds[i].item()
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[i][pred_idx].item() * 100

                true_class = CLASS_NAMES[true_label_idx]

                if pred_class == "0_real":
                    display_label = "REAL"
                    source = "Real"
                else:
                    display_label = "FAKE"
                    source = clean_label(pred_class).upper()

                results[relative_path] = {
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "display_label": display_label,
                    "source": source,
                    "confidence": round(confidence, 2),
                    "probabilities": {
                        CLASS_NAMES[j]: round(probs[i][j].item() * 100, 2)
                        for j in range(len(CLASS_NAMES))
                    }
                }

                sample_index += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()