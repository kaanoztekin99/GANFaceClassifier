import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "model_weights" / "best_cnn_5class.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224

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


def get_display_text(predicted_class, confidence):
    clean = clean_label(predicted_class)

    if clean == "real":
        return f"REAL ({confidence:.2f}%)"

    return f"FAKE - {clean.upper()} ({confidence:.2f}%)"


def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx].item() * 100

    return image, predicted_class, confidence, probabilities.cpu().numpy()


def save_annotated_image(image, image_path, predicted_class, confidence):
    image = image.copy().convert("RGBA")

    width, height = image.size

    clean = clean_label(predicted_class)

    if clean == "real":
        lines = [
            "REAL",
            f"{confidence:.2f}%"
        ]
    else:
        lines = [
            "FAKE",
            clean.upper(),
            f"{confidence:.2f}%"
        ]

    font_size = max(14, int(width * 0.035))
    small_font_size = max(12, int(width * 0.025))

    try:
        font_main = ImageFont.truetype("Arial.ttf", font_size)
        font_sub = ImageFont.truetype("Arial.ttf", small_font_size)
    except:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    padding = max(6, int(width * 0.015))
    margin = max(6, int(width * 0.015))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text_width = 0
    text_height = 0

    for i, line in enumerate(lines):
        font = font_main if i == 0 else font_sub

        bbox = draw.textbbox((0, 0), line, font=font)

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        text_width = max(text_width, w)
        text_height += h + padding // 2

    box = [
        margin,
        margin,
        margin + text_width + padding * 2,
        margin + text_height + padding * 2
    ]

    draw.rectangle(
        box,
        fill=(0, 0, 0, 170)
    )

    current_y = margin + padding

    for i, line in enumerate(lines):
        font = font_main if i == 0 else font_sub

        draw.text(
            (margin + padding, current_y),
            line,
            fill=(255, 255, 255, 255),
            font=font
        )

        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]

        current_y += line_height + padding // 2

    image = Image.alpha_composite(
        image,
        overlay
    ).convert("RGB")

    output_path = OUTPUT_DIR / f"predicted_{Path(image_path).name}"

    image.save(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an image is real or GAN-generated."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image."
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image, predicted_class, confidence, probabilities = predict_image(image_path)

    output_path = save_annotated_image(
        image,
        image_path,
        predicted_class,
        confidence
    )

    print("\nPrediction Result")
    print("-----------------")
    print("Input image:", image_path)
    print("Predicted class:", predicted_class)
    print("Display label:", get_display_text(predicted_class, confidence))
    print("Confidence:", f"{confidence:.2f}%")
    print("\nClass probabilities:")

    for class_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {prob * 100:.2f}%")

    print("\nAnnotated image saved to:")
    print(output_path)


if __name__ == "__main__":
    main()