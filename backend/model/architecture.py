"""
EfficientNetV2-S architecture matching the training repository exactly.
Weights: model.classifier[1] = nn.Linear(in_features, num_classes)
Trained on: Fraud (class 0), Non-Fraud (class 1) - alphabetical directory ordering.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


NUM_CLASSES = 2
CLASS_NAMES = ["Fraud", "Non-Fraud"]


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
