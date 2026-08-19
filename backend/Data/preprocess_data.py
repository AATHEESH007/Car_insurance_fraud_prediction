#!/usr/bin/env python3
"""
Preprocess dataset to remove data leakage and deduplicate image files.
Detects duplicate images (via raw byte MD5 and perceptual image content hashing)
between train and test splits, removing any leaked test images so the test set
contains strictly unique, unseen samples.
"""

import argparse
import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute raw binary MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def compute_image_dhash(file_path: Path, hash_size: int = 8) -> str:
    """
    Compute perceptual difference hash (dhash) of an image file.
    Insensitive to JPEG re-compression, metadata differences, or minor scaling.
    """
    try:
        with Image.open(file_path) as img:
            img_l = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img_l.getdata())
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(pixel_left > pixel_right)

            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2 ** (index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].zfill(2))
                    decimal_value = 0
            return "".join(hex_string)
    except Exception as e:
        # Fallback to file md5 if image cannot be opened
        return compute_file_md5(file_path)


def scan_dataset(data_dir: Path, use_perceptual: bool = True):
    """
    Scan data_dir for train and test splits, computing hashes for all images.
    Returns:
      hash_to_files: dict[hash -> list of file_info dicts]
    """
    splits = ["train", "test"]
    classes = ["Fraud", "Non-Fraud"]

    hash_to_files = defaultdict(list)
    total_scanned = 0

    for split in splits:
        for cls_name in classes:
            dir_path = data_dir / split / cls_name
            if not dir_path.exists():
                continue

            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    file_hash = compute_image_dhash(file_path) if use_perceptual else compute_file_md5(file_path)
                    hash_to_files[file_hash].append(
                        {
                            "split": split,
                            "class": cls_name,
                            "path": file_path,
                            "filename": file_path.name,
                        }
                    )
                    total_scanned += 1

    return hash_to_files, total_scanned


def deduplicate_dataset(data_dir: Path, dry_run: bool = False, verbose: bool = False, use_perceptual: bool = True):
    """
    Find duplicate images across train/test splits.
    Resolves data leakage by ensuring any image in train is NOT in test,
    and moves/consolidates duplicate test images into train.
    """
    print(f"\nScanning dataset in '{data_dir.resolve()}' for duplicate images...")
    hash_type = "Perceptual Image Hash (dhash)" if use_perceptual else "Binary File MD5"
    print(f"Hashing method: {hash_type}")

    hash_to_files, total_scanned = scan_dataset(data_dir, use_perceptual=use_perceptual)

    print(f"Scanned {total_scanned} total image files.")
    print(f"Found {len(hash_to_files)} unique image hashes.\n")

    train_removed = 0
    test_removed = 0
    test_moved_to_train = 0
    label_mismatches = 0

    for file_hash, file_list in hash_to_files.items():
        if len(file_list) <= 1:
            continue

        train_files = [f for f in file_list if f["split"] == "train"]
        test_files = [f for f in file_list if f["split"] == "test"]

        # Case 1: Image exists in BOTH train and test (DATA LEAKAGE)
        if train_files and test_files:
            # We keep the copy in train. Remove all test copies to eliminate data leakage.
            for test_item in test_files:
                if test_item["class"] != train_files[0]["class"]:
                    label_mismatches += 1
                    if verbose:
                        print(
                            f"[LABEL MISMATCH] Hash {file_hash[:8]}: "
                            f"train class '{train_files[0]['class']}', test class '{test_item['class']}'"
                        )

                if verbose:
                    print(
                        f"[LEAKAGE REMOVED] Removing test copy: {test_item['path'].relative_to(data_dir)} "
                        f"(train copy exists: {train_files[0]['path'].relative_to(data_dir)})"
                    )

                if not dry_run:
                    test_item["path"].unlink()
                test_removed += 1

            # Deduplicate multiple train copies if present
            if len(train_files) > 1:
                for extra_train in train_files[1:]:
                    if verbose:
                        print(f"[TRAIN DUP REMOVED] Removing duplicate train copy: {extra_train['path'].relative_to(data_dir)}")
                    if not dry_run:
                        extra_train["path"].unlink()
                    train_removed += 1

        # Case 2: Image exists ONLY in test, but has duplicate copies within test
        elif not train_files and len(test_files) > 1:
            primary_test = test_files[0]
            # Move primary test copy to train split
            target_train_dir = data_dir / "train" / primary_test["class"]
            target_train_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_train_dir / primary_test["filename"]

            if target_path.exists() and target_path != primary_test["path"]:
                target_path = target_train_dir / f"from_test_{primary_test['filename']}"

            if verbose:
                print(f"[TEST DUP MOVED TO TRAIN] {primary_test['path'].relative_to(data_dir)} -> {target_path.relative_to(data_dir)}")

            if not dry_run:
                shutil.move(str(primary_test["path"]), str(target_path))
            test_moved_to_train += 1

            # Delete remaining test duplicate copies
            for extra_test in test_files[1:]:
                if verbose:
                    print(f"[TEST DUP REMOVED] Removing extra test copy: {extra_test['path'].relative_to(data_dir)}")
                if not dry_run:
                    extra_test["path"].unlink()
                test_removed += 1

        # Case 3: Image exists ONLY in train, but has duplicate copies within train
        elif len(train_files) > 1 and not test_files:
            for extra_train in train_files[1:]:
                if verbose:
                    print(f"[TRAIN DUP REMOVED] Removing duplicate train copy: {extra_train['path'].relative_to(data_dir)}")
                if not dry_run:
                    extra_train["path"].unlink()
                train_removed += 1

    print("--- Deduplication Action Summary ---")
    print(f"  Mode                     : {'DRY RUN (No files modified)' if dry_run else 'EXECUTED'}")
    print(f"  Leaked Test Files Removed: {test_removed}")
    print(f"  Test Files Moved to Train: {test_moved_to_train}")
    print(f"  Duplicate Train Files    : {train_removed}")
    print(f"  Label Mismatches Noted   : {label_mismatches}")

    # Post-clean summary
    print("\n--- Final Image Counts Per Split ---")
    splits = ["train", "test"]
    classes = ["Fraud", "Non-Fraud"]
    final_total = 0
    for split in splits:
        for cls_name in classes:
            dir_path = data_dir / split / cls_name
            count = 0
            if dir_path.exists():
                count = sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS)
            print(f"  {split}/{cls_name:<12}: {count:>5} images")
            final_total += count
    print(f"  Total Clean Images      : {final_total:>5}")
    print("------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset: detect and remove data leakage between train/test splits using image hashing."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help=f"Dataset directory containing train and test folders (default: {script_dir})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deduplication without deleting or moving files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed log of every duplicate file action",
    )
    parser.add_argument(
        "--exact-md5-only",
        action="store_true",
        help="Use exact binary file MD5 hash instead of perceptual image hashing",
    )

    args = parser.parse_args()
    deduplicate_dataset(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        use_perceptual=not args.exact_md5_only,
    )


if __name__ == "__main__":
    main()
