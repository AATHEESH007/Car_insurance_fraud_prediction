"""
ML inference service. Model is loaded once at app startup.
Preprocessing exactly mirrors the training/evaluation pipeline:
  Resize(384, 384) -> ToTensor -> Normalize(ImageNet mean/std)
"""
import logging
import sys
import os

logger = logging.getLogger(__name__)

_model = None
_device = None
_class_to_idx = None
_inference_transform = None
_target_layers = None

IMG_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_path: str):
    global _model, _device, _class_to_idx, _inference_transform, _target_layers

    import torch

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model")))
    from architecture import build_model, load_checkpoint

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading EfficientNetV2-S model on %s from %s", _device, model_path)

    _model = build_model()
    checkpoint = load_checkpoint(_model, model_path, _device)

    _class_to_idx = checkpoint.get("class_to_idx", {"Fraud": 0, "Non-Fraud": 1})
    _model.to(_device)
    _model.eval()
    _inference_transform = _build_transform()
    _target_layers = [_model.features[-1]]

    logger.info("Model loaded. class_to_idx=%s", _class_to_idx)


def _ensure_model_loaded():
    global _model
    if _model is not None:
        return
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "weights", "best_efficientnetv2_s.pth"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "weights", "best_efficientnetv2_s.pth"),
        "backend/model/weights/best_efficientnetv2_s.pth",
        "model/weights/best_efficientnetv2_s.pth",
    ]
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and os.path.getsize(abs_p) > 1000:
            try:
                load_model(abs_p)
                return
            except Exception as e:
                logger.error("Auto-load failed for %s: %s", abs_p, e)


def is_model_loaded() -> bool:
    _ensure_model_loaded()
    return _model is not None


def get_model_and_layers():
    """Return (model, target_layers, device, class_to_idx) for GradCAM usage."""
    _ensure_model_loaded()
    return _model, _target_layers, _device, _class_to_idx


def predict(image, config) -> dict:
    import torch
    import torch.nn.functional as F

    _ensure_model_loaded()
    if _model is None:
        raise RuntimeError("Model is not loaded.")

    tensor = _inference_transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)

    probs = F.softmax(logits, dim=1).squeeze(0)

    idx_to_class = {v: k for k, v in _class_to_idx.items()}
    fraud_idx = _class_to_idx.get("Fraud", 0)
    non_fraud_idx = _class_to_idx.get("Non-Fraud", 1)

    fraud_prob = float(probs[fraud_idx].item())
    non_fraud_prob = float(probs[non_fraud_idx].item())

    predicted_idx = int(torch.argmax(probs).item())
    prediction_label = idx_to_class.get(predicted_idx, "Unknown")

    risk_level = _classify_risk(fraud_prob, config)
    recommendation = _get_recommendation(risk_level)

    return {
        "prediction": prediction_label,
        "fraud_probability": round(fraud_prob, 4),
        "non_fraud_probability": round(non_fraud_prob, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def _classify_risk(fraud_prob: float, config) -> str:
    high_threshold = float(getattr(config, "FRAUD_HIGH_THRESHOLD", 0.70))
    medium_threshold = float(getattr(config, "FRAUD_MEDIUM_THRESHOLD", 0.40))
    if fraud_prob >= high_threshold:
        return "HIGH"
    if fraud_prob >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def _get_recommendation(risk_level: str) -> str:
    recommendations = {
        "HIGH": "Manual review recommended. High fraud probability detected.",
        "MEDIUM": "Additional verification recommended.",
        "LOW": "Claim appears legitimate. Standard processing.",
    }
    return recommendations.get(risk_level, "Review required.")
