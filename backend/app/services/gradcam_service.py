"""
GradCAM service for fraud detection explainability.

Reuses the already-loaded model from model_service (no double loading).
Generates a Grad-CAM heatmap overlay for a claim image and saves it to
  uploads/gradcam/<claim_id>.png

Public API:
    generate_gradcam(image_path, claim_id, upload_dir) -> str
        Returns the relative URL path to the saved heatmap PNG.
"""
import logging
import os
import sys

import numpy as np
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

IMG_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms():
    normalize = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    display = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return normalize, display


def generate_gradcam(image_path: str, claim_id: str, upload_dir: str) -> str:
    """
    Generate a Grad-CAM heatmap overlay for the given image.

    Parameters
    ----------
    image_path : str
        Absolute path to the claim vehicle image on disk.
    claim_id : str
        Used to name the output file (gradcam/<claim_id>.png).
    upload_dir : str
        Root uploads directory (e.g. backend/uploads).

    Returns
    -------
    str
        URL path to the saved heatmap: /api/v1/uploads/gradcam/<claim_id>.png
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Reuse the already-loaded model — avoids reloading weights
    from app.services.model_service import get_model_and_layers
    model, target_layers, device, class_to_idx = get_model_and_layers()

    if model is None or target_layers is None:
        raise RuntimeError("ML model is not loaded. Cannot generate GradCAM.")

    normalize_transform, display_transform = _build_transforms()

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = normalize_transform(pil_img).unsqueeze(0).to(device)
    rgb_img = display_transform(pil_img).permute(1, 2, 0).numpy().astype(np.float32)

    # Determine the predicted class index for CAM targeting
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probs).item())

    targets = [ClassifierOutputTarget(predicted_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay heatmap on the display image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save to uploads/gradcam/<claim_id>.png
    gradcam_dir = os.path.join(upload_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    out_filename = f"{claim_id}.png"
    out_path = os.path.join(gradcam_dir, out_filename)
    Image.fromarray(visualization).save(out_path)

    logger.info("GradCAM saved for claim %s -> %s", claim_id, out_path)

    return f"/api/v1/uploads/gradcam/{out_filename}"
