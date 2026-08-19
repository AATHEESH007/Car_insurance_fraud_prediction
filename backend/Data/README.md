# Dataset Directory & Processing Guide

This folder contains dataset management scripts and organized splits for training and evaluating the **Car Insurance Fraud Detection** model (EfficientNetV2-S).

## Source Dataset
- **Kaggle Dataset**: [pacificrm/car-insurance-fraud-detection](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection)

## Directory Structure
```
backend/Data/
├── download_dataset.py   # Downloads raw dataset from Kaggle
├── preprocess_data.py    # MD5 deduplication & data leakage removal
├── augment_fraud_data.py # Offline augmentation for minority Fraud class
├── README.md             # Dataset documentation
├── train/
│   ├── Fraud/            # Training images for fraudulent claims
│   └── Non-Fraud/        # Training images for non-fraudulent claims
└── test/
    ├── Fraud/            # Test images for fraudulent claims (leakage free)
    └── Non-Fraud/        # Test images for non-fraudulent claims (leakage free)
```

## Workflows

### 1. Download Dataset
Downloads raw Kaggle dataset splits into `backend/Data`:

```bash
python backend/Data/download_dataset.py
```

### 2. Preprocess & Eliminate Data Leakage
Runs hashing across all images in `train` and `test` splits to detect duplicate images and ensure zero data leakage into the evaluation set.

```bash
python backend/Data/preprocess_data.py
```

### 3. Augment Minority Class ('Fraud')
Augments the ~194 original `train/Fraud` images to generate synthetic variations (rotations, flips, brightness/contrast adjustments, color jitter, and subtle filtering) to address severe class imbalance.

```bash
# Augment train/Fraud up to 2,000 images
python backend/Data/augment_fraud_data.py --target-count 2000

# Or match majority Non-Fraud class image count (~4,643 images)
python backend/Data/augment_fraud_data.py --match-majority
```

#### Augmentation Options
- `--target-count` : Target number of total images in `Fraud` subfolder (default: `2000`).
- `--match-majority`: Automatically sets target count to match `Non-Fraud` count (~4,643).
- `--split`          : Dataset split to target (`train`, `test`, `val`).
- `--dry-run`        : Preview image counts without writing files.
- `--seed`           : Random seed for reproducible visual transformations.
