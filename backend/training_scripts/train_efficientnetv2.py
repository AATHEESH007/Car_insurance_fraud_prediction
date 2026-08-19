#!/usr/bin/env python3
"""
EfficientNetV2 Training Script for Car Insurance Fraud Detection.

Uses built-in PyTorch and Torchvision functions for training, data transformations,
loss functions, optimizers, learning rate scheduling, and early stopping.

Includes anti-overfitting techniques:
  - Built-in nn.Dropout in classifier head
  - Inline Early Stopping based on validation loss
  - Built-in torchvision transforms (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing)
  - L2 Regularization (Weight Decay) in AdamW optimizer
  - Label Smoothing in nn.CrossEntropyLoss
  - Learning Rate Scheduling (ReduceLROnPlateau / CosineAnnealingLR)
  - Automatic Mixed Precision (torch.amp) for GPU acceleration

Usage:
    backend\.venv\Scripts\python.exe backend/training_scripts/train_efficientnetv2.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2 using built-in PyTorch functions with Early Stopping and Dropout."
    )

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "Data"
    default_output_dir = script_dir.parent / "model" / "weights"

    # Dataset & Paths
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir), help="Dataset root folder")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir), help="Output folder for checkpoints")

    # Hyperparameters
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "l"], help="EfficientNetV2 variant (s, m, l)")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")

    # Overfitting Reduction & Regularization
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 weight decay for AdamW")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing parameter")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--random-erasing-prob", type=float, default=0.1, help="Random erasing probability")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"], help="LR scheduler")

    # System & Hardware
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda', 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Reproducibility via built-in functions
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Built-in Device Selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("      EFFICIENTNET-V2 FRAUD DETECTION MODEL TRAINING SCRIPT")
    print("=" * 70)
    print(f"Device                   : {device}")
    print(f"EfficientNetV2 Variant   : {args.variant.upper()}")
    print(f"Pretrained Weights       : {not args.no_pretrained}")
    print(f"Batch Size               : {args.batch_size}")
    print(f"Learning Rate            : {args.lr}")
    print(f"Weight Decay (L2 Reg)    : {args.weight_decay}")
    print(f"Dropout Rate             : {args.dropout}")
    print(f"Label Smoothing          : {args.label_smoothing}")
    print(f"Early Stopping Patience  : {args.patience}")
    print("=" * 70)

    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    # Built-in Data Transformations (torchvision.transforms)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=args.random_erasing_prob, scale=(0.02, 0.2), value="random"),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Built-in Datasets & DataLoaders
    full_train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    if val_dir.exists():
        val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)
        train_dataset = full_train_dataset
    elif test_dir.exists():
        val_dataset = ImageFolder(root=str(test_dir), transform=val_transform)
        train_dataset = full_train_dataset
    else:
        raw_train = ImageFolder(root=str(train_dir), transform=train_transform)
        raw_val = ImageFolder(root=str(train_dir), transform=val_transform)
        val_size = int(len(raw_train) * args.val_split)
        train_size = len(raw_train) - val_size
        train_indices, val_indices = random_split(
            range(len(raw_train)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataset = torch.utils.data.Subset(raw_train, train_indices)
        val_dataset = torch.utils.data.Subset(raw_val, val_indices)

    print(f"Dataset split -> Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Built-in EfficientNetV2 Model Construction
    variant = args.variant.lower()
    if variant == "s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_s(weights=weights)
    elif variant == "m":
        weights = EfficientNet_V2_M_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_m(weights=weights)
    else:
        weights = EfficientNet_V2_L_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_l(weights=weights)

    # Built-in Dropout & Classifier Head Replacement
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=args.dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)

    # Built-in Loss, Optimizer, and LR Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Built-in AMP GradScaler
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.use_amp) else None

    # Setup Output Checkpoint Directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / f"best_efficientnetv2_{args.variant}.pth"

    # Inline Early Stopping State Variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting Training Loop...")
    start_time = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # -------------------------------------------------------------
        # Training Loop
        # -------------------------------------------------------------
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # -------------------------------------------------------------
        # Validation Loop
        # -------------------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Built-in LR Scheduler Step
        if args.scheduler == "plateau":
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # -------------------------------------------------------------
        # Inline Early Stopping & Checkpoint Saving
        # -------------------------------------------------------------
        if epoch_val_loss < (best_val_loss - 1e-4):
            print(f"  --> Validation loss improved ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving checkpoint.")
            best_val_loss = epoch_val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "class_names": class_names,
                "hyperparameters": vars(args),
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1
            print(f"  --> No validation loss improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Stopping training at epoch {epoch}.")
                break

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Best model saved to: {best_model_path}")

    # Save training history using built-in json module
    with open(output_dir / f"training_history_{args.variant}.json", "w") as f:
        json.dump(history, f, indent=2)
    print("=" * 70)


if __name__ == "__main__":
    main()
