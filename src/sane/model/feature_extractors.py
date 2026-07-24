"""
Modular feature extractor for dataset encoding.

Provides a common interface for extracting per-image features from dataset images.
Used by the DeepSets dataset encoder.

Available extractors:
    - ConvFeatureExtractor: Lightweight CNN (same as original DeepSets ImagePhi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import logging
logger = logging.getLogger(__name__)


class ConvFeatureExtractor(nn.Module):
    """Lightweight CNN feature extractor (identical to original DeepSets ImagePhi).

    3 × (Conv2d → BN → ReLU → MaxPool → Dropout) → AdaptiveAvgPool → Linear.

    Shape:
        Input:  (B, C, H, W)
        Output: (B, output_dim)
    """

    def __init__(self, input_channels: int = 3, hidden_dims: List[int] = [64, 128, 256], output_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = input_channels
        for hd in hidden_dims:
            layers += [nn.Conv2d(in_ch, hd, 3, padding=1), nn.BatchNorm2d(hd), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout)]
            in_ch = hd
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).flatten(1)
        return F.relu(self.fc(x))


def build_feature_extractor(name: str = "conv", **kwargs) -> nn.Module:
    """Factory for feature extractors.

    Args:
        name: "conv"
        **kwargs: forwarded to the constructor

    Returns:
        A feature extractor module with an `output_dim` attribute.
    """
    if name == "conv":
        return ConvFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown feature extractor: {name}. Only 'conv' is supported.")
