#!/usr/bin/env python3

import os
import random
import shutil
import sys

# Configuration
SEED = 42  # For reproducible splits
TRAIN_RATIO = 0.80  # 80% train, 20% validation
SOURCE_DIR = "processed_data"
OUTPUT_DIR = "Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def split_dataset(source_dir, output_dir, train_ratio, seed):
    """Split dataset from source_dir into train/val with deterministic seed."""
    random.seed(seed)

    # Validate source directory exists
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {os.path.abspath(source_dir)}")
        sys.exit(1)

    train_root = os.path.join(output_dir, "train")
    val_root = os.path.join(output_dir, "val")

    # Clean up existing output directories (safe for reruns)
    if os.path.exists(train_root):
        shutil.rmtree(train_root)
        print(f"[INFO] Deleted existing directory: {train_root}")

    if os.path.exists(val_root):
        shutil.rmtree(val_root)
        print(f"[INFO] Deleted existing directory: {val_root}")

    class_names = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    if not class_names:
        print(f"[ERROR] No class sub-directories found in {source_dir}")
        sys.exit(1)

    print()
    print("=" * 65)
    print(f"  Source directory : {os.path.abspath(source_dir)}")
    print(f"  Output directory : {os.path.abspath(output_dir)}")
    print(f"  Train ratio      : {int(train_ratio * 100)}%")
    print(f"  Val ratio        : {int((1 - train_ratio) * 100)}%")
    print(f"  Random seed      : {seed}")
    print(f"  Classes detected : {len(class_names)} -> {class_names}")
    print("=" * 65)
    print()

    total_train = 0
    total_val = 0
    skipped_files = 0

    for class_name in class_names:
        class_source = os.path.join(source_dir, class_name)

        all_files = sorted(os.listdir(class_source))
        images = [
            f for f in all_files
            if os.path.isfile(os.path.join(class_source, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        non_images = [
            f for f in all_files
            if os.path.isfile(os.path.join(class_source, f))
            and os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS
        ]
        skipped_files += len(non_images)

        if not images:
            print(f"  [WARN] {class_name:12s} — no images found, skipping.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_dir = os.path.join(train_root, class_name)
        val_dir = os.path.join(val_root, class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(class_source, img), os.path.join(train_dir, img))

        for img in val_images:
            shutil.copy2(os.path.join(class_source, img), os.path.join(val_dir, img))

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"  {class_name:12s}  |  total: {len(images):5d}  "
            f"|  train: {len(train_images):5d}  |  val: {len(val_images):5d}"
        )

    print()
    print("-" * 65)
    print(f"  {'TOTAL':12s}  |  total: {total_train + total_val:5d}  "
          f"|  train: {total_train:5d}  |  val: {total_val:5d}")
    if skipped_files > 0:
        print(f"  Non-image files skipped: {skipped_files}")
    print("-" * 65)
    print()

    verify_train = 0
    verify_val = 0
    verify_classes = set()

    if os.path.isdir(train_root):
        for cls in os.listdir(train_root):
            cls_path = os.path.join(train_root, cls)
            if os.path.isdir(cls_path):
                verify_classes.add(cls)
                verify_train += len([
                    f for f in os.listdir(cls_path)
                    if os.path.isfile(os.path.join(cls_path, f))
                ])

    if os.path.isdir(val_root):
        for cls in os.listdir(val_root):
            cls_path = os.path.join(val_root, cls)
            if os.path.isdir(cls_path):
                verify_classes.add(cls)
                verify_val += len([
                    f for f in os.listdir(cls_path)
                    if os.path.isfile(os.path.join(cls_path, f))
                ])

    print("=" * 65)
    print("  VERIFICATION")
    print("=" * 65)
    print(f"  Total train images : {verify_train}")
    print(f"  Total val images   : {verify_val}")
    print(f"  Detected classes   : {len(verify_classes)} -> {sorted(verify_classes)}")
    print(f"  Dataset location   : {os.path.abspath(output_dir)}")
    print("=" * 65)
    print()

    if verify_train == total_train and verify_val == total_val:
        print("[SUCCESS] Dataset split completed successfully. Ready for training.")
    else:
        print("[WARNING] Verification mismatch — please inspect the Dataset folder.")


if __name__ == "__main__":
    split_dataset(SOURCE_DIR, OUTPUT_DIR, TRAIN_RATIO, SEED)
