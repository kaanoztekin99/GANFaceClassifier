import argparse
import random
import shutil
from pathlib import Path


def select_celeba_real(source_dir, output_dir, num_images=2000, seed=42):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png"}

    images = [
        p for p in source_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if len(images) < num_images:
        raise ValueError(
            f"Not enough images. Found {len(images)}, requested {num_images}."
        )

    random.seed(seed)
    selected_images = random.sample(images, num_images)

    for idx, img_path in enumerate(selected_images):
        new_name = f"real_{idx:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, output_dir / new_name)

    print(f"Copied {num_images} real CelebA images to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select real images from CelebA dataset."
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to CelebA img_align_celeba folder."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/celebA/real",
        help="Output folder for selected real images."
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=2000,
        help="Number of real images to select."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    select_celeba_real(
        source_dir=args.source,
        output_dir=args.output,
        num_images=args.num_images,
        seed=args.seed
    )