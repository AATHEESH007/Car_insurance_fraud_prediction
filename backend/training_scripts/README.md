# Training Scripts for Car Insurance Fraud Detection

This directory contains model training scripts for the **Car Insurance Fraud Detection** system.

## Overview

The primary training script is [`train_efficientnetv2.py`](file:///c:/Users/Anto%20Rahul/Desktop/cognizant/car_insurance_fraud_data/Insurance-Fraud-Detection/backend/training_scripts/train_efficientnetv2.py), which builds and fine-tunes EfficientNetV2 (`s`, `m`, or `l`) on the preprocessed image dataset using transfer learning and extensive anti-overfitting techniques.

---

## Overfitting Reduction Techniques Implemented

1. **Early Stopping**: Monitors validation loss or macro F1-score and halts training when performance stops improving (configurable via `--patience` and `--monitor-metric`).
2. **Dropout Regularization**: Injects configurable dropout (`--dropout`, default `0.3`) into the classifier head before the linear output layer.
3. **Data Augmentation**: Uses torchvision transforms to prevent feature memorization:
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomRotation(degrees=15)`
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
   - `RandomErasing` (cutout-like patch erasure, `--random-erasing-prob 0.1`)
4. **Weight Decay (L2 Regularization)**: Configurable `weight_decay` (`1e-2`) in the `AdamW` optimizer to penalize large weights.
5. **Label Smoothing**: Softens target targets using `nn.CrossEntropyLoss(label_smoothing=0.1)` to prevent classifier overconfidence.
6. **Learning Rate Scheduling**: Supports `ReduceLROnPlateau` and `CosineAnnealingLR` to lower learning rate when validation loss plateaus.
7. **Gradient Clipping**: Prevents exploding gradients during fine-tuning using `clip_grad_norm_`.
8. **Transfer Learning / Backbone Freezing**: Freezes backbone feature extraction layers for initial epochs (`--freeze-epochs 3`) so the classifier head warms up before fine-tuning pre-trained weights.

---

## Usage Instructions

### Basic Training Command
From the project root:

```bash
# Using the backend virtual environment python
backend\.venv\Scripts\python.exe backend/training_scripts/train_efficientnetv2.py
```

### Customizing Hyperparameters

```bash
backend\.venv\Scripts\python.exe backend/training_scripts/train_efficientnetv2.py \
  --variant s \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --dropout 0.4 \
  --weight-decay 0.01 \
  --label-smoothing 0.1 \
  --patience 5 \
  --freeze-epochs 3 \
  --scheduler plateau
```

---

## Command Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` | `str` | `backend/Data` | Path to dataset containing `train` and `test` split folders |
| `--output-dir` | `str` | `backend/model/weights` | Directory to save best model checkpoint and history JSON |
| `--variant` | `str` | `"s"` | EfficientNetV2 variant (`s`, `m`, `l`) |
| `--epochs` | `int` | `30` | Maximum number of training epochs |
| `--batch-size` | `int` | `32` | DataLoader mini-batch size |
| `--lr` | `float` | `1e-4` | Learning rate for AdamW optimizer |
| `--dropout` | `float` | `0.3` | Dropout probability in classifier head |
| `--weight-decay` | `float` | `1e-2` | Weight decay / L2 regularization parameter |
| `--label-smoothing`| `float` | `0.1` | Cross-Entropy label smoothing parameter |
| `--patience` | `int` | `7` | Early stopping patience (number of non-improving epochs) |
| `--monitor-metric` | `str` | `"val_loss"` | Early stopping metric (`val_loss` or `val_f1`) |
| `--freeze-epochs` | `int` | `3` | Number of initial epochs to freeze backbone parameters |
| `--scheduler` | `str` | `"plateau"` | Learning rate scheduler type (`plateau` or `cosine`) |

---

## Saved Checkpoints & Output Format

Upon training completion or when a new best validation metric is reached, a checkpoint file is saved to:
`backend/model/weights/best_efficientnetv2_s.pth`

The saved checkpoint dictionary contains:
- `model_state_dict`: Full model weights (compatible with [`backend/model/architecture.py`](file:///c:/Users/Anto%20Rahul/Desktop/cognizant/car_insurance_fraud_data/Insurance-Fraud-Detection/backend/model/architecture.py))
- `optimizer_state_dict`: AdamW optimizer state
- `epoch`: Best epoch number
- `val_loss`, `val_acc`, `val_f1`: Evaluation metrics
- `class_names`: Categorical class labels (`['Fraud', 'Non-Fraud']`)
- `hyperparameters`: All CLI hyperparameter arguments used
