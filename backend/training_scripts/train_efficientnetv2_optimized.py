#!/usr/bin/env python3
"""
EfficientNetV2 Training Script for Car Insurance Fraud Detection (v2 - imbalance-aware).

Built on top of the original script, with additions specifically aimed at:
  (a) class imbalance (minority "fraud" class), and
  (b) overfitting on a limited number of unique fraud images even after augmentation.

New / changed vs the original script:
  - WeightedRandomSampler so each training batch is roughly class-balanced,
    instead of relying on raw folder counts.
  - Optional class-weighted CrossEntropyLoss (off by default; don't usually
    stack with the sampler, but available via --use-class-weights).
  - Mixup / CutMix augmentation (strongest lever here against memorization of
    the small fraud set) - via torchvision.transforms.v2 built-ins.
  - Stochastic depth (native torchvision EfficientNetV2 arg) for extra
    regularization inside the backbone.
  - Progressive unfreezing: backbone frozen for --freeze-epochs, then
    unfrozen with a lower LR than the classifier head (differential LR).
  - Exponential Moving Average (EMA) of model weights via torch.optim.swa_utils.AveragedModel.
  - Early stopping + checkpointing on validation macro-F1 (fallback: val
    loss), not raw accuracy/loss, because those are misleading under
    imbalance.
  - Per-epoch precision/recall/F1/AUC by class, plus a final classification
    report and confusion matrix.
  - Checkpoints are NEVER overwritten: first run saves as
    best_efficientnetv2_<variant>.pth, subsequent runs auto-version as
    best_efficientnetv2_<variant>_2.0.pth, _3.0.pth, etc.

Usage:
    backend\\.venv\\Scripts\\python.exe backend/training_scripts/train_efficientnetv2.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from torchvision.datasets import ImageFolder
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2 with imbalance-aware sampling and strong anti-overfitting measures."
    )

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "Data"
    default_output_dir = script_dir.parent / "model" / "weights"

    # Dataset & Paths
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))

    # Hyperparameters
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "l"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for classifier head")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="Backbone LR = lr * this, once unfrozen")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.15)

    # Imbalance handling
    parser.add_argument("--use-weighted-sampler", action="store_true", default=True,
                         help="Balance classes per batch via WeightedRandomSampler")
    parser.add_argument("--use-class-weights", action="store_true", default=False,
                         help="Also weight the loss by inverse class frequency (usually redundant with sampler)")

    # Overfitting reduction & regularization
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout in classifier head")
    parser.add_argument("--weight-decay", type=float, default=2e-2, help="L2 weight decay for AdamW")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    parser.add_argument("--random-erasing-prob", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine"])
    parser.add_argument("--stochastic-depth", type=float, default=0.2, help="Stochastic depth prob in backbone")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Epochs to keep backbone frozen (0 disables)")
    parser.add_argument("--use-mixup", action="store_true", default=True)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--early-stop-metric", type=str, default="f1_macro",
                         choices=["f1_macro", "val_loss", "auc"])

    # System & hardware
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.early_stop_metric in ("auc", "f1_macro") and not SKLEARN_AVAILABLE:
        raise RuntimeError(
            f"scikit-learn is required for --early-stop-metric {args.early_stop_metric}. "
            "Install it with: pip install scikit-learn "
            "(or pass --early-stop-metric val_loss to avoid the dependency)."
        )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("   EFFICIENTNET-V2 FRAUD DETECTION - IMBALANCE-AWARE TRAINING")
    print("=" * 70)
    print(f"Device                   : {device}")
    print(f"Variant                  : {args.variant.upper()}")
    print(f"Pretrained               : {not args.no_pretrained}")
    print(f"Batch Size               : {args.batch_size}")
    print(f"Head LR / Backbone LR    : {args.lr} / {args.lr * args.backbone_lr_mult}")
    print(f"Weight Decay             : {args.weight_decay}")
    print(f"Dropout                  : {args.dropout}")
    print(f"Stochastic Depth         : {args.stochastic_depth}")
    print(f"Weighted Sampler         : {args.use_weighted_sampler}")
    print(f"Class-weighted Loss      : {args.use_class_weights}")
    print(f"Mixup/CutMix             : {args.use_mixup}")
    print(f"Freeze Epochs (backbone) : {args.freeze_epochs}")
    print(f"EMA Decay                : {args.ema_decay}")
    print(f"Early Stop Metric        : {args.early_stop_metric} (patience={args.patience})")
    print("=" * 70)

    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=args.random_erasing_prob, scale=(0.02, 0.2), value="random"),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    full_train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    if val_dir.exists():
        val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)
        train_dataset = full_train_dataset
        train_targets = [s[1] for s in full_train_dataset.samples]
    elif test_dir.exists():
        val_dataset = ImageFolder(root=str(test_dir), transform=val_transform)
        train_dataset = full_train_dataset
        train_targets = [s[1] for s in full_train_dataset.samples]
    else:
        raw_train = ImageFolder(root=str(train_dir), transform=train_transform)
        raw_val = ImageFolder(root=str(train_dir), transform=val_transform)
        val_size = int(len(raw_train) * args.val_split)
        train_size = len(raw_train) - val_size
        train_indices, val_indices = random_split(
            range(len(raw_train)), [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataset = torch.utils.data.Subset(raw_train, train_indices)
        val_dataset = torch.utils.data.Subset(raw_val, val_indices)
        all_targets = np.array([s[1] for s in raw_train.samples])
        train_targets = all_targets[list(train_indices.indices)].tolist()

    print(f"Dataset split -> Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ------------------------------------------------------------------
    # Imbalance handling: per-sample weights -> WeightedRandomSampler
    # ------------------------------------------------------------------
    class_counts = np.bincount(train_targets, minlength=num_classes)
    print(f"Train class counts: {dict(zip(class_names, class_counts.tolist()))}")

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        class_weights_inv = 1.0 / np.clip(class_counts, 1, None)
        sample_weights = class_weights_inv[train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model: EfficientNetV2 + dropout head + stochastic depth
    # ------------------------------------------------------------------
    variant = args.variant.lower()
    model_kwargs = {"stochastic_depth_prob": args.stochastic_depth}
    if variant == "s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_s(weights=weights, **model_kwargs)
    elif variant == "m":
        weights = EfficientNet_V2_M_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_m(weights=weights, **model_kwargs)
    else:
        weights = EfficientNet_V2_L_Weights.DEFAULT if not args.no_pretrained else None
        model = efficientnet_v2_l(weights=weights, **model_kwargs)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=args.dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model.to(device)

    # Progressive unfreezing: freeze everything except the classifier head initially
    if args.freeze_epochs > 0 and not args.no_pretrained:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad_(False)
        print(f"Backbone frozen for the first {args.freeze_epochs} epoch(s).")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    class_weight_tensor = None
    if args.use_class_weights:
        inv_freq = 1.0 / np.clip(class_counts, 1, None)
        inv_freq = inv_freq / inv_freq.sum() * num_classes
        class_weight_tensor = torch.tensor(inv_freq, dtype=torch.float32, device=device)
        print(f"Class weights (loss): {class_weight_tensor.tolist()}")

    hard_criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing, weight=class_weight_tensor
    )
    # nn.CrossEntropyLoss accepts probability/soft targets directly (PyTorch >= 1.10),
    # so the same criterion works for both hard labels and Mixup/CutMix soft labels.

    # Built-in Mixup/CutMix (torchvision.transforms.v2, requires torchvision >= 0.16)
    mixup_cutmix_transform = None
    if args.use_mixup:
        cutmix = transforms_v2.CutMix(num_classes=num_classes, alpha=args.cutmix_alpha)
        mixup = transforms_v2.MixUp(num_classes=num_classes, alpha=args.mixup_alpha)
        mixup_cutmix_transform = transforms_v2.RandomChoice([cutmix, mixup])

    # ------------------------------------------------------------------
    # Optimizer with differential LR (head vs backbone) + scheduler
    # ------------------------------------------------------------------
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.use_amp) else None

    # Built-in EMA via torch.optim.swa_utils.AveragedModel with a custom avg_fn.
    # This is the officially documented way to do EMA in PyTorch (see AveragedModel docs);
    # avg_fn is a small formula, not a full custom implementation.
    ema_model = None
    if args.ema_decay > 0:
        def ema_avg_fn(averaged_param, current_param, num_averaged):
            return args.ema_decay * averaged_param + (1 - args.ema_decay) * current_param
        ema_model = AveragedModel(model, avg_fn=ema_avg_fn)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_next_checkpoint_path(out_dir: Path, variant: str) -> Path:
        """Never overwrite an existing checkpoint. The first run for a given
        variant uses the plain name; each subsequent run gets a new
        '_<version>.0' suffix (2.0, 3.0, 4.0, ...) so every trained model is
        kept on disk."""
        base_path = out_dir / f"best_efficientnetv2_{variant}.pth"
        if not base_path.exists():
            return base_path

        version = 2
        while True:
            candidate = out_dir / f"best_efficientnetv2_{variant}_{version}.0.pth"
            if not candidate.exists():
                return candidate
            version += 1

    best_model_path = get_next_checkpoint_path(output_dir, args.variant)
    print(f"Checkpoint for this run will be saved to: {best_model_path}")

    # Early stopping state
    best_metric = float("inf") if args.early_stop_metric == "val_loss" else -float("inf")
    patience_counter = 0

    def metric_improved(current, best):
        if args.early_stop_metric == "val_loss":
            return current < best - 1e-4
        return current > best + 1e-4

    print("\nStarting Training Loop...")
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1_macro": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone once warm-up is over
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad_(True)
            print(f"  --> Backbone unfrozen at epoch {epoch} (LR mult {args.backbone_lr_mult}).")

        # ---------------- Training ----------------
        model.train()
        running_train_loss = 0.0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            use_mix = args.use_mixup and mixup_cutmix_transform is not None and random.random() < args.mixup_prob
            if use_mix:
                inputs, soft_targets = mixup_cutmix_transform(inputs, targets)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    if use_mix:
                        loss = hard_criterion(outputs, soft_targets)
                    else:
                        loss = hard_criterion(outputs, targets)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if use_mix:
                    loss = hard_criterion(outputs, soft_targets)
                else:
                    loss = hard_criterion(outputs, targets)
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(model)

            running_train_loss += loss.item() * inputs.size(0)
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / train_total if train_total > 0 else 0.0

        # ---------------- Validation (use EMA weights if available) ----------------
        eval_model = ema_model.module if ema_model is not None else model
        eval_model.eval()

        running_val_loss = 0.0
        val_total = 0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = eval_model(inputs)
                loss = hard_criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_total += targets.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0.0
        val_acc = float(np.mean(np.array(all_preds) == np.array(all_targets))) if val_total else 0.0

        val_f1_macro, val_auc = 0.0, 0.0
        if SKLEARN_AVAILABLE and val_total > 0:
            _, _, f1_per_class, _ = precision_recall_fscore_support(
                all_targets, all_preds, labels=list(range(num_classes)), zero_division=0
            )
            val_f1_macro = float(np.mean(f1_per_class))
            try:
                if num_classes == 2:
                    val_auc = float(roc_auc_score(all_targets, np.array(all_probs)[:, 1]))
                else:
                    val_auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr"))
            except ValueError:
                val_auc = 0.0

        if args.scheduler == "plateau":
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_auc"].append(val_auc)

        epoch_time = time.time() - epoch_start
        head_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1(macro): {val_f1_macro:.4f} AUC: {val_auc:.4f} | "
            f"HeadLR: {head_lr:.6f}"
        )

        # ---------------- Early stopping / checkpointing ----------------
        current_metric = {"val_loss": epoch_val_loss, "f1_macro": val_f1_macro, "auc": val_auc}[args.early_stop_metric]

        if metric_improved(current_metric, best_metric):
            print(f"  --> {args.early_stop_metric} improved ({best_metric:.4f} -> {current_metric:.4f}). Saving checkpoint.")
            best_metric = current_metric
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (ema_model.module.state_dict() if ema_model is not None else model.state_dict()),
                "raw_model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "val_auc": val_auc,
                "class_names": class_names,
                "hyperparameters": vars(args),
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1
            print(f"  --> No {args.early_stop_metric} improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Stopping training at epoch {epoch}.")
                break

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Best model saved to: {best_model_path} (best {args.early_stop_metric} = {best_metric:.4f})")

    # ---------------- Final report on best checkpoint ----------------
    if SKLEARN_AVAILABLE and val_total > 0:
        print("\nFinal validation classification report (last epoch's predictions):")
        print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(all_targets, all_preds))

    history_path = output_dir / (best_model_path.stem.replace("best_efficientnetv2", "training_history") + ".json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()