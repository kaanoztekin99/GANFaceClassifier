import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "model_weights" / "best_cnn_5class.pth"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224

CLASS_NAMES = [
    "0_real",
    "1_dcgan",
    "2_progan",
    "3_stylegan2",
    "4_stylegan3"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - cam.min()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def load_model():
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(resized_image)

    return input_tensor, original_np


def create_heatmap_overlay(original_np, cam):
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(
        original_bgr,
        0.55,
        heatmap,
        0.45,
        0
    )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay_rgb


def generate_gradcam_for_image(model, image_path):
    input_tensor, original_np = load_image(image_path)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    target_layer = model.features[18]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor, pred_idx)
    gradcam.close()

    overlay = create_heatmap_overlay(original_np, cam)

    output_name = f"{image_path.stem}_gradcam.png"
    output_path = OUTPUT_DIR / output_name

    Image.fromarray(overlay).save(output_path)

    return output_path, pred_class, confidence


def collect_test_images():
    image_paths = []

    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional single image path. If omitted, all test images are processed."
    )

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("Model path:", MODEL_PATH)
    print("Model exists:", MODEL_PATH.exists())

    model = load_model()

    if args.image:
        image_path = Path(args.image)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        output_path, pred_class, confidence = generate_gradcam_for_image(
            model,
            image_path
        )

        print("\nGrad-CAM Result")
        print("----------------")
        print("Input image:", image_path)
        print("Predicted class:", pred_class)
        print("Confidence:", f"{confidence:.2f}%")
        print("Saved to:", output_path)

    else:
        image_paths = collect_test_images()

        print(f"\nFound {len(image_paths)} test images.")
        print("Generating Grad-CAM heatmaps...\n")

        for idx, image_path in enumerate(image_paths, start=1):
            output_path, pred_class, confidence = generate_gradcam_for_image(
                model,
                image_path
            )

            print(
                f"[{idx}/{len(image_paths)}] "
                f"{image_path.name} -> {output_path.name} "
                f"({pred_class}, {confidence:.2f}%)"
            )

        print("\nAll Grad-CAM heatmaps saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()