#!/usr/bin/env python3
"""
Model Evaluation Script for Car Insurance Fraud Detection.

Loads a trained EfficientNetV2 checkpoint and computes comprehensive metrics on the test dataset:
  - Matthews Correlation Coefficient (MCC)
  - Accuracy & Balanced Accuracy
  - Precision, Recall (Sensitivity), Specificity, F1-Score (Macro, Weighted, Per-Class)
  - ROC-AUC (Area Under ROC Curve)
  - Detailed Confusion Matrix (TP, TN, FP, FN breakdown)

Usage:
    backend\.venv\Scripts\python.exe backend/eval.py
    backend\.venv\Scripts\python.exe backend/eval.py --checkpoint backend/model/weights/best_efficientnetv2_s.pth

    # If the checkpoint was saved with EMA (training_scripts/train_efficientnetv2.py, ema-decay > 0),
    # you can compare EMA vs raw trained weights:
    backend\.venv\Scripts\python.exe backend/eval.py --use-raw-weights
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)


# ---------------------------------------------------------------------------
# Metric Calculation Utilities (Pure PyTorch & NumPy)
# ---------------------------------------------------------------------------
def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve (ROC-AUC) using trapezoidal integration.
    """
    desc_indices = np.argsort(-y_probs)
    y_true_sorted = y_true[desc_indices]

    num_pos = np.sum(y_true_sorted == 1)
    num_neg = np.sum(y_true_sorted == 0)

    if num_pos == 0 or num_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos)
        fpr_list.append(fp / num_neg)

    # Pure Python trapezoidal integration for ROC-AUC
    auc = 0.0
    for i in range(1, len(tpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        mean_y = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc += mean_y * dx

    return float(abs(auc))


def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Returns value between -1.0 and +1.0.
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list
) -> Dict[str, Any]:
    """
    Run evaluation loop over dataloader and calculate comprehensive metrics.
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_logits.append(outputs.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    test_loss = running_loss / total_samples if total_samples > 0 else 0.0

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    num_classes = len(class_names)
    total = len(y_true)
    accuracy = float((y_pred == y_true).sum() / total) if total > 0 else 0.0

    # Per-class & Binary Confusion Matrix Metrics
    per_class_metrics = {}

    # Identify index for 'Fraud' class if present, default to 0
    fraud_class_idx = 0
    for idx, name in enumerate(class_names):
        if "fraud" in name.lower() and "non" not in name.lower():
            fraud_class_idx = idx
            break

    # Calculate TP, TN, FP, FN for binary / target class (Fraud vs Non-Fraud)
    binary_true = (y_true == fraud_class_idx).astype(int)
    binary_pred = (y_pred == fraud_class_idx).astype(int)
    binary_prob = y_prob[:, fraud_class_idx]

    tp = int(np.sum((binary_true == 1) & (binary_pred == 1)))
    tn = int(np.sum((binary_true == 0) & (binary_pred == 0)))
    fp = int(np.sum((binary_true == 0) & (binary_pred == 1)))
    fn = int(np.sum((binary_true == 1) & (binary_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = float((recall + specificity) / 2.0)
    mcc = compute_mcc(tp, tn, fp, fn)
    roc_auc = compute_roc_auc(binary_true, binary_prob)

    # Detailed per-class metrics dictionary
    for c_idx, c_name in enumerate(class_names):
        c_true = (y_true == c_idx).astype(int)
        c_pred = (y_pred == c_idx).astype(int)
        c_tp = int(np.sum((c_true == 1) & (c_pred == 1)))
        c_tn = int(np.sum((c_true == 0) & (c_pred == 0)))
        c_fp = int(np.sum((c_true == 0) & (c_pred == 1)))
        c_fn = int(np.sum((c_true == 1) & (c_pred == 0)))

        c_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
        c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = float(2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0

        per_class_metrics[c_name] = {
            "samples": int(np.sum(c_true)),
            "true_positives": c_tp,
            "true_negatives": c_tn,
            "false_positives": c_fp,
            "false_negatives": c_fn,
            "precision": c_prec,
            "recall": c_rec,
            "f1_score": c_f1,
        }

    metrics = {
        "test_loss": float(test_loss),
        "total_samples": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "target_class": class_names[fraud_class_idx],
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        },
        "binary_metrics": {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "mcc": mcc,
            "roc_auc": roc_auc,
        },
        "per_class_metrics": per_class_metrics
    }

    return metrics


# ---------------------------------------------------------------------------
# CLI & Main Logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained EfficientNetV2 model on test dataset with MCC and comprehensive metrics."
    )

    script_dir = Path(__file__).resolve().parent
    default_checkpoint = script_dir / "model" / "weights" / "best_efficientnetv2_s.pth"
    default_data_dir = script_dir / "Data"
    default_output_json = script_dir / "model" / "weights" / "evaluation_report.json"

    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint), help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir), help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--output-json", type=str, default=str(default_output_json), help="Path to save output JSON metrics report")
    parser.add_argument("--device", type=str, default=None, help="Execution device override ('cuda', 'cpu')")
    parser.add_argument(
        "--use-raw-weights",
        action="store_true",
        default=False,
        help=(
            "If the checkpoint contains an EMA-smoothed 'model_state_dict' AND a "
            "'raw_model_state_dict' (saved by training_scripts/train_efficientnetv2.py "
            "with --ema-decay > 0), load the raw (non-EMA) trained weights instead. "
            "Useful to sanity-check whether EMA averaging under-trained relative to "
            "the actual optimized weights."
        ),
    )

    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        print("Please train the model first using: backend\\.venv\\Scripts\\python.exe backend/training_scripts/train_efficientnetv2.py")
        return

    data_dir = Path(args.data_dir).resolve()
    test_dir = data_dir / "test"
    if not test_dir.exists():
        test_dir = data_dir / "val"
    if not test_dir.exists():
        test_dir = data_dir / "train"

    print("=" * 70)
    print("           EFFICIENTNET-V2 MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Checkpoint Path  : {checkpoint_path}")
    print(f"Test Data Folder : {test_dir}")
    print("=" * 70)

    # Test Image Transformations
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    test_dataset = ImageFolder(root=str(test_dir), transform=test_transform)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(test_dataset)} test samples across {num_classes} classes: {class_names}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Load Checkpoint & Construct Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_hyperparams = checkpoint.get("hyperparameters", {})
    variant = saved_hyperparams.get("variant", "s").lower()

    if variant == "s":
        model = efficientnet_v2_s(weights=None)
    elif variant == "m":
        model = efficientnet_v2_m(weights=None)
    else:
        model = efficientnet_v2_l(weights=None)

    in_features = model.classifier[1].in_features
    dropout_rate = saved_hyperparams.get("dropout", 0.3)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    # ------------------------------------------------------------------
    # Choose which weights to load. Checkpoints saved by the EMA-enabled
    # training script store TWO sets of weights:
    #   "model_state_dict"     -> EMA-smoothed weights (used by default)
    #   "raw_model_state_dict" -> the actual optimizer-trained weights
    # If --use-raw-weights is passed and the raw weights exist, load those
    # instead. Otherwise fall back to the normal behavior.
    # ------------------------------------------------------------------
    if args.use_raw_weights and "raw_model_state_dict" in checkpoint:
        print("Loading RAW (non-EMA) trained weights: 'raw_model_state_dict'")
        state_dict = checkpoint["raw_model_state_dict"]
    else:
        if args.use_raw_weights:
            print("Warning: --use-raw-weights was set but checkpoint has no "
                  "'raw_model_state_dict' key. Falling back to 'model_state_dict'.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)
    model.to(device)

    # Execute Evaluation
    metrics = evaluate_model(model, test_loader, device, class_names)

    # Display Pretty Formatted Output
    cm = metrics["confusion_matrix"]
    bm = metrics["binary_metrics"]

    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Test Samples        : {metrics['total_samples']}")
    print(f"Test Loss                 : {metrics['test_loss']:.4f}")
    print(f"Accuracy                  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy         : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"MCC (Matthews Corr Coeff) : {metrics['mcc']:.4f}  <-- Key Imbalance Metric")
    print(f"ROC-AUC                   : {metrics['roc_auc']:.4f}")
    print("-" * 70)
    print("Confusion Matrix Breakdown:")
    print(f"  True Positives  (TP - Fraud predicted as Fraud)     : {cm['TP']}")
    print(f"  True Negatives  (TN - Non-Fraud predicted as Non-Fraud): {cm['TN']}")
    print(f"  False Positives (FP - Non-Fraud predicted as Fraud) : {cm['FP']}")
    print(f"  False Negatives (FN - Fraud predicted as Non-Fraud) : {cm['FN']}")
    print("-" * 70)
    print("Binary Classification Metrics ('Fraud' Class):")
    print(f"  Precision               : {bm['precision']:.4f}")
    print(f"  Recall (Sensitivity)    : {bm['recall_sensitivity']:.4f}")
    print(f"  Specificity             : {bm['specificity']:.4f}")
    print(f"  F1-Score                : {bm['f1_score']:.4f}")
    print("=" * 70)

    print("\nPer-Class Breakdown:")
    for cls_name, cls_m in metrics["per_class_metrics"].items():
        print(
            f"  Class [{cls_name:<10}] | Samples: {cls_m['samples']:<4} | "
            f"Prec: {cls_m['precision']:.4f} | Rec: {cls_m['recall']:.4f} | F1: {cls_m['f1_score']:.4f}"
        )
    print("=" * 70)

    # Save output JSON report
    output_json_path = Path(args.output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved full evaluation report to: '{output_json_path}'")


if __name__ == "__main__":
    main()