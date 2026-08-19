#!/usr/bin/env python3
"""
Grad-CAM Visualization Script for Car Insurance Fraud Detection.

Loads a trained EfficientNetV2 checkpoint (from training_scripts/train_efficientnetv2.py)
and generates Grad-CAM heatmaps showing which regions of an image the model
is "looking at" when it predicts Fraud / Non-Fraud. Useful for:
  - Sanity-checking whether the model is learning meaningful visual cues
    (e.g. damage location) rather than spurious background artifacts.
  - Explaining individual predictions (to a reviewer, in a report, etc.).

Uses the standard 'pytorch-grad-cam' library (pip install grad-cam) rather
than a hand-rolled hook implementation.

Usage:
    # Single image
    backend\\.venv\\Scripts\\python.exe backend/gradcam_visualize.py --image path/to/image.jpg

    # Sample N images per class from the test set
    backend\\.venv\\Scripts\\python.exe backend/gradcam_visualize.py --num-per-class 5

    # Use a specific checkpoint / method
    backend\\.venv\\Scripts\\python.exe backend/gradcam_visualize.py --checkpoint backend/model/weights/best_efficientnetv2_s_2.0.pth --method gradcam++
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "xgradcam": XGradCAM,
}


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyperparameters", {})
    variant = hyperparams.get("variant", "s").lower()
    dropout_rate = hyperparams.get("dropout", 0.3)
    class_names = checkpoint.get("class_names", ["Fraud", "Non-Fraud"])
    num_classes = len(class_names)

    if variant == "s":
        model = efficientnet_v2_s(weights=None)
    elif variant == "m":
        model = efficientnet_v2_m(weights=None)
    else:
        model = efficientnet_v2_l(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Last conv block (Conv2dNormActivation) before global pooling -- the
    # standard Grad-CAM target layer for EfficientNet-style architectures.
    target_layers = [model.features[-1]]

    return model, target_layers, class_names


def run_gradcam_on_image(
    image_path: Path,
    model,
    target_layers,
    class_names,
    device,
    method_name: str,
    img_size: int,
    output_dir: Path,
    target_class_idx: int = None,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    # Separate 0-1 (unnormalized) version for overlaying the heatmap on.
    display_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = normalize_transform(pil_img).unsqueeze(0).to(device)
    rgb_img = display_transform(pil_img).permute(1, 2, 0).numpy().astype(np.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probs).item())

    cam_class_idx = target_class_idx if target_class_idx is not None else predicted_idx
    targets = [ClassifierOutputTarget(cam_class_idx)]

    cam_algorithm = METHODS[method_name]
    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_label = class_names[predicted_idx]
    cam_label = class_names[cam_class_idx]
    confidence = float(probs[predicted_idx])

    out_name = (
        f"{image_path.parent.name}_{image_path.stem}_pred-{predicted_label}_{confidence:.2f}"
        f"_cam-{cam_label}_{method_name}.png"
    )
    out_path = output_dir / out_name
    Image.fromarray(visualization).save(out_path)

    print(
        f"  {image_path.name:<40} | Predicted: {predicted_label} ({confidence:.2%}) | "
        f"CAM target: {cam_label} | saved -> {out_path.name}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for a trained EfficientNetV2 fraud detection model."
    )

    script_dir = Path(__file__).resolve().parent
    default_checkpoint = script_dir / "model" / "weights" / "best_efficientnetv2_s.pth"
    default_data_dir = script_dir / "Data"
    default_output_dir = script_dir / "model" / "gradcam_outputs"

    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint),
                         help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a single image to visualize. If omitted, samples from --data-dir/test.")
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir), help="Dataset root directory")
    parser.add_argument("--num-per-class", type=int, default=5,
                         help="If no --image given, how many images per class to sample from the test set")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir),
                         help="Where to save heatmap overlay images")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--method", type=str, default="gradcam", choices=list(METHODS.keys()),
                         help="Grad-CAM variant to use")
    parser.add_argument("--target-class", type=str, default=None,
                         help="Force CAM for a specific class name (default: model's own predicted class)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at '{checkpoint_path}'")
        return

    output_dir = Path(args.output_dir).resolve()

    print("=" * 70)
    print("           GRAD-CAM VISUALIZATION - FRAUD DETECTION MODEL")
    print("=" * 70)
    print(f"Device      : {device}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Method      : {args.method}")
    print(f"Output Dir  : {output_dir}")
    print("=" * 70)

    model, target_layers, class_names = load_model(checkpoint_path, device)
    print(f"Loaded model. Classes: {class_names}")

    target_class_idx = None
    if args.target_class is not None:
        if args.target_class not in class_names:
            raise ValueError(f"--target-class '{args.target_class}' not in {class_names}")
        target_class_idx = class_names.index(args.target_class)

    if args.image:
        image_paths = [Path(args.image).resolve()]
    else:
        data_dir = Path(args.data_dir).resolve()
        test_dir = data_dir / "test"
        if not test_dir.exists():
            test_dir = data_dir / "val"
        if not test_dir.exists():
            raise FileNotFoundError(f"No test/val directory found under {data_dir}")

        dataset = ImageFolder(root=str(test_dir))
        by_class = {name: [] for name in dataset.classes}
        for path, label_idx in dataset.samples:
            by_class[dataset.classes[label_idx]].append(Path(path))

        image_paths = []
        for cls_name, paths in by_class.items():
            random.shuffle(paths)
            image_paths.extend(paths[: args.num_per_class])

        print(f"Sampling {args.num_per_class} image(s) per class from: {test_dir}")

    print(f"\nGenerating Grad-CAM for {len(image_paths)} image(s)...\n")
    for image_path in image_paths:
        run_gradcam_on_image(
            image_path=image_path,
            model=model,
            target_layers=target_layers,
            class_names=class_names,
            device=device,
            method_name=args.method,
            img_size=args.img_size,
            output_dir=output_dir,
            target_class_idx=target_class_idx,
        )

    print("\n" + "=" * 70)
    print(f"Done. Heatmaps saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()