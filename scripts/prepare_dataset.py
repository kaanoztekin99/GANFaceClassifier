import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_BASE = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_BASE = PROJECT_ROOT / "dataset"

CLASS_DIRS = {
    "0_real": RAW_BASE / "celeba" / "selected",
    "1_dcgan": RAW_BASE / "dcgan" / "fake_generated_images_dcgan",
    "2_progan": RAW_BASE / "progan" / "fake_generated_images_progan",
    "3_stylegan2": RAW_BASE / "stylegan2" / "fake_generated_images_stylegan2",
    "4_stylegan3": RAW_BASE / "stylegan3" / "fake_generated_images_stylegan3",
}

SAMPLES_PER_CLASS = 500

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(folder):
    folder = Path(folder)
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def clear_split_folders():
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_BASE / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

        for class_name in CLASS_DIRS.keys():
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def split_images(images):
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(images, target_dir, class_name):
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(images):
        new_name = f"{class_name}_{idx:05d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, target_dir / new_name)


def main():
    random.seed(SEED)

    clear_split_folders()

    print("Preparing 5-class dataset...\n")

    summary = {}

    for class_name, class_dir in CLASS_DIRS.items():
        images = collect_images(class_dir)

        if len(images) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"{class_name}: only {len(images)} images found, "
                f"but {SAMPLES_PER_CLASS} required."
            )

        selected_images = random.sample(images, SAMPLES_PER_CLASS)

        train_imgs, val_imgs, test_imgs = split_images(selected_images)

        copy_images(
            train_imgs,
            OUTPUT_BASE / "train" / class_name,
            class_name
        )

        copy_images(
            val_imgs,
            OUTPUT_BASE / "val" / class_name,
            class_name
        )

        copy_images(
            test_imgs,
            OUTPUT_BASE / "test" / class_name,
            class_name
        )

        summary[class_name] = {
            "total": len(selected_images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }

        print(
            f"{class_name}: "
            f"train={len(train_imgs)}, "
            f"val={len(val_imgs)}, "
            f"test={len(test_imgs)}"
        )

    print("\nDataset prepared successfully.")
    print("\nSummary:")

    for class_name, info in summary.items():
        print(
            f"{class_name}: total={info['total']} | "
            f"train={info['train']} | "
            f"val={info['val']} | "
            f"test={info['test']}"
        )


if __name__ == "__main__":
    main()