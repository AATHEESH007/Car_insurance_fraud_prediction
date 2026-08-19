#!/usr/bin/env python3
"""
Offline Data Augmentation Script for Imbalanced Dataset.

Augments images in the minority 'Fraud' class (in backend/Data/train/Fraud)
to balance the class distribution against 'Non-Fraud' images.

Techniques applied per image:
  - Random Horizontal Flip
  - Random Rotation (-25° to +25°)
  - Random Brightness & Contrast Adjustment
  - Random Color Saturation
  - Random Slight Blur / Sharpness Filtering

Usage:
    python backend/Data/augment_fraud_data.py --help
    python backend/Data/augment_fraud_data.py --target-count 2000
"""

import argparse
import random
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_random_augmentations(img: Image.Image) -> Image.Image:
    """
    Apply a randomized sequence of visual augmentations to a PIL Image.
    
    Returns:
        Image.Image: Augmented image copy.
    """
    # Ensure RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 1. Random Horizontal Flip (50% probability)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # 2. Random Rotation (-25 to +25 degrees)
    angle = random.uniform(-25.0, 25.0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))

    # 3. Random Brightness Adjustment (0.75x to 1.25x)
    brightness_factor = random.uniform(0.75, 1.25)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 4. Random Contrast Adjustment (0.8x to 1.2x)
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 5. Random Color Saturation Adjustment (0.8x to 1.2x)
    color_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Color(img).enhance(color_factor)

    # 6. Random Sharpen / Blur (20% chance)
    rand_val = random.random()
    if rand_val < 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.0)))
    elif rand_val < 0.2:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=130))

    return img


def augment_fraud_class(
    data_dir: Path,
    split: str = "train",
    target_class: str = "Fraud",
    target_count: int = 2000,
    match_majority: bool = False,
    dry_run: bool = False,
    seed: int = 42,
) -> None:
    """
    Augment images in target_class folder up to target_count images.
    """
    random.seed(seed)

    target_dir = data_dir / split / target_class
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    # Gather existing image files (excluding previously generated aug_ files if re-running)
    all_files = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    original_files = [p for p in all_files if not p.name.startswith("aug_")]
    existing_aug_files = [p for p in all_files if p.name.startswith("aug_")]

    if not original_files:
        print(f"Error: No original image files found in '{target_dir}'")
        return

    # Check majority class count if requested
    if match_majority:
        majority_dir = data_dir / split / "Non-Fraud"
        if majority_dir.exists():
            majority_count = sum(
                1 for p in majority_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
            target_count = majority_count
            print(f"Matching majority class ('Non-Fraud') image count: {majority_count}")

    current_total = len(all_files)
    num_to_generate = max(0, target_count - current_total)

    print("=" * 65)
    print("         DATASET FRAUD CLASS OFFLINE AUGMENTATION")
    print("=" * 65)
    print(f"Dataset Directory  : {data_dir.resolve()}")
    print(f"Split / Class      : {split} / {target_class}")
    print(f"Original Images    : {len(original_files)}")
    print(f"Existing Augmented : {len(existing_aug_files)}")
    print(f"Current Total      : {current_total}")
    print(f"Target Total       : {target_count}")
    print(f"New to Generate    : {num_to_generate}")
    print(f"Dry Run Mode       : {dry_run}")
    print("=" * 65)

    if num_to_generate <= 0:
        print(f"Target count of {target_count} already reached or exceeded. No augmentations needed.")
        return

    if dry_run:
        print(f"[DRY RUN] Would generate {num_to_generate} new augmented images in '{target_dir.relative_to(data_dir)}'.")
        return

    print(f"\nGenerating {num_to_generate} augmented images...")

    generated_count = 0
    # Determine starting index for new aug files
    start_index = len(existing_aug_files) + 1

    while generated_count < num_to_generate:
        # Pick a random original image as source
        src_path = random.choice(original_files)
        try:
            with Image.open(src_path) as img:
                aug_img = apply_random_augmentations(img)

                aug_filename = f"aug_{(start_index + generated_count):05d}_{src_path.stem}.jpg"
                save_path = target_dir / aug_filename

                # Save as high-quality JPEG
                aug_img.save(save_path, format="JPEG", quality=95)
                generated_count += 1

                if generated_count % 100 == 0 or generated_count == num_to_generate:
                    print(f"  Progress: {generated_count}/{num_to_generate} augmented images generated...")

        except Exception as e:
            print(f"Warning: Failed to process image '{src_path.name}': {e}")
            continue

    print("\n" + "=" * 65)
    print(f"Augmentation complete! Generated {generated_count} new images.")
    final_total = len([
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ])
    print(f"Final image count in '{split}/{target_class}': {final_total}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Augment minority Fraud class images in dataset split to balance class distribution."
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help=f"Dataset root directory containing train and test folders (default: {script_dir})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split directory to augment (default: train)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="Fraud",
        help="Target class subfolder to augment (default: Fraud)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=2000,
        help="Target total image count to reach in target class subfolder (default: 2000)",
    )
    parser.add_argument(
        "--match-majority",
        action="store_true",
        help="Automatically set target-count to match the number of images in 'Non-Fraud' class",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview augmentation counts without creating image files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations (default: 42)",
    )

    args = parser.parse_args()
    augment_fraud_class(
        data_dir=args.data_dir,
        split=args.split,
        target_class=args.target_class,
        target_count=args.target_count,
        match_majority=args.match_majority,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
