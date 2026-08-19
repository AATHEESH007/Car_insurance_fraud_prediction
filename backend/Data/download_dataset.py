#!/usr/bin/env python3
"""
Script to download and organize the Kaggle Car Insurance Fraud Detection dataset.
Dataset URL: https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection

This script fetches the dataset using `kagglehub` and structures it into:
  - train/
      - Fraud/
      - Non-Fraud/
  - test/
      - Fraud/
      - Non-Fraud/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure kagglehub is installed
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub package...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub


DATASET_HANDLE = "pacificrm/car-insurance-fraud-detection"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_splits_root(base_path: Path) -> Path:
    """Recursively find the directory that contains both 'train' and 'test' folders."""
    for root, dirs, _ in os.walk(base_path):
        root_path = Path(root)
        if "train" in dirs and "test" in dirs:
            return root_path
    raise FileNotFoundError(
        f"Could not locate 'train' and 'test' directories under downloaded path: {base_path}"
    )


def count_images(directory: Path) -> int:
    """Count valid image files in a directory recursively."""
    if not directory.exists():
        return 0
    return sum(
        1
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def copy_dataset_splits(src_root: Path, dst_root: Path, clean: bool = False, dry_run: bool = False):
    """Copy train and test splits into dst_root while keeping class structure."""
    splits = ["train", "test"]
    classes = ["Fraud", "Non-Fraud"]

    print(f"\nSource splits root: {src_root}")
    print(f"Target directory:   {dst_root.resolve()}\n")

    summary = {}

    for split in splits:
        src_split_dir = src_root / split
        dst_split_dir = dst_root / split

        for cls_name in classes:
            src_cls_dir = src_split_dir / cls_name
            dst_cls_dir = dst_split_dir / cls_name

            if not src_cls_dir.exists():
                print(f"[WARNING] Expected class directory not found: {src_cls_dir}")
                continue

            if clean and dst_cls_dir.exists() and not dry_run:
                print(f"Cleaning existing directory: {dst_cls_dir}")
                shutil.rmtree(dst_cls_dir)

            if not dry_run:
                dst_cls_dir.mkdir(parents=True, exist_ok=True)

            copied_count = 0
            for file_path in src_cls_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    target_file = dst_cls_dir / file_path.name
                    if not dry_run:
                        shutil.copy2(file_path, target_file)
                    copied_count += 1

            key = f"{split}/{cls_name}"
            summary[key] = copied_count
            print(f"[{'DRY-RUN ' if dry_run else ''}COPIED] {key}: {copied_count} images")

    print("\n--- Summary ---")
    total_images = sum(summary.values())
    for split_cls, count in summary.items():
        print(f"  {split_cls:<20}: {count:>5} images")
    print(f"  Total Images        : {total_images:>5}")
    print("----------------")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Kaggle car insurance fraud dataset."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help=f"Target directory to store dataset splits (default: {script_dir})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing dataset files in destination before copying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the copy process without writing files",
    )

    args = parser.parse_args()

    print(f"Downloading Kaggle dataset '{DATASET_HANDLE}' via kagglehub...")
    cache_path_str = kagglehub.dataset_download(DATASET_HANDLE)
    cache_path = Path(cache_path_str)

    print(f"Dataset downloaded to cache: {cache_path}")
    splits_root = find_splits_root(cache_path)

    copy_dataset_splits(
        src_root=splits_root,
        dst_root=args.output_dir,
        clean=args.clean,
        dry_run=args.dry_run,
    )
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
