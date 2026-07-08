"""
Modular feature extractors for dataset encoding.

Provides a common interface for extracting per-image features from dataset images.
Used by both DeepSets and Set Transformer dataset encoders.

Available extractors:
    - ConvFeatureExtractor: Lightweight CNN (same as original DeepSets ImagePhi)
    - ResNetFeatureExtractor: torchvision ResNet18, optionally pretrained on ImageNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import logging
logger = logging.getLogger(__name__)


class ConvFeatureExtractor(nn.Module):
    """Lightweight CNN feature extractor (identical to original DeepSets ImagePhi).
    
    3 × (Conv2d → BN → ReLU → MaxPool → Dropout) → AdaptiveAvgPool → Linear.
    
    Shape:
        Input:  (B, C, H, W)
        Output: (B, output_dim)
    """

    def __init__(self, input_channels: int = 3, hidden_dims: List[int] = [64, 128, 256],
                 output_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = input_channels
        for hd in hidden_dims:
            layers += [nn.Conv2d(in_ch, hd, 3, padding=1), nn.BatchNorm2d(hd),
                       nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout)]
            in_ch = hd
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).flatten(1)
        return F.relu(self.fc(x))


class ResNetFeatureExtractor(nn.Module):
    """ResNet-18 feature extractor, optionally pretrained on ImageNet.
    
    Uses torchvision ResNet18 with the classification head removed.
    Adds an optional projection to match a desired output_dim.
    
    Shape:
        Input:  (B, C, H, W)   — C must be 3 (RGB)
        Output: (B, output_dim)
    """

    def __init__(self, output_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=pretrained)

        # Remove classification head; keep everything up to avgpool
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        resnet_dim = 512  # ResNet-18 outputs 512-d features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("ResNet backbone frozen (pretrained features only)")

        self.proj = nn.Linear(resnet_dim, output_dim) if resnet_dim != output_dim else nn.Identity()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(1)  # (B, 512)
        return F.relu(self.proj(x))


def build_feature_extractor(name: str = "conv", **kwargs) -> nn.Module:
    """Factory for feature extractors.
    
    Args:
        name: "conv" or "resnet"
        **kwargs: forwarded to the constructor
    
    Returns:
        A feature extractor module with an `output_dim` attribute.
    """
    if name == "conv":
        return ConvFeatureExtractor(**kwargs)
    elif name == "resnet":
        return ResNetFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown feature extractor: {name}. Choose 'conv' or 'resnet'.")
