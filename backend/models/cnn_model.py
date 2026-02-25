from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet18 backbone used as a frozen feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.feature_dim = feature_dim

        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess_spectrogram(self, spec: np.ndarray) -> torch.Tensor:
        """
        Convert a spectrogram (H x W) into a 3-channel normalized tensor.
        Values are min-max scaled before conversion to an image.
        """
        spec_min = float(np.min(spec))
        spec_max = float(np.max(spec))
        if spec_max - spec_min < 1e-6:
            norm = np.zeros_like(spec, dtype=np.float32)
        else:
            norm = (spec - spec_min) / (spec_max - spec_min)

        img = np.stack([norm, norm, norm], axis=-1)  # H x W x 3
        pil_img = Image.fromarray((img * 255).astype("uint8"))
        tensor = self._transform(pil_img).unsqueeze(0)
        return tensor

    def forward(self, spec: np.ndarray) -> torch.Tensor:
        """
        Run a forward pass through ResNet18 to obtain a feature vector.
        """
        self.backbone.eval()
        with torch.no_grad():
            x = self.preprocess_spectrogram(spec)
            features = self.backbone(x)
        return features.squeeze(0)


class TypeIIDetectorHead(nn.Module):
    """
    Simple linear classification head on top of the CNN features
    producing a Type II burst probability via sigmoid.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.fc(features)
        prob = self.activation(logits)
        return prob.squeeze(-1)


def build_cnn_detector() -> Tuple[ResNetFeatureExtractor, TypeIIDetectorHead]:
    extractor = ResNetFeatureExtractor()
    head = TypeIIDetectorHead(feature_dim=extractor.feature_dim)
    return extractor, head

