"""
Unified dataset encoder: feature_extractor + set_encoder + classifier.

This module is a **drop-in replacement** for the original ``DeepSetsClassifier``
and exposes the same interface expected by ``DatasetAlignmentLoss``:

    .forward(x)       → logits   (B, num_classes)
    .encode(x)        → embedding (B, embedding_dim)
    .embedding_dim    → int

Supported configuration:

    ┌──────────────────────┬─────────────────────────────┐
    │ feature_extractor    │ set_encoder                 │
    ├──────────────────────┼─────────────────────────────┤
    │ conv   (orig DeepSets│ deepsets  (sum/mean/max)    │
    └──────────────────────┴─────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from sane.model.feature_extractors import build_feature_extractor

import logging
logger = logging.getLogger(__name__)


# Thin DeepSets aggregator that works on *pre-extracted* features
class DeepSetsAggregator(nn.Module):
    """ρ-only DeepSets: aggregate pre-extracted per-element features → set embedding.

    φ is handled externally by the feature extractor.
    
    Shape:
        Input:  (B, n, feat_dim)
        Output: (B, output_dim)
    """
    def __init__(self, feat_dim: int, hidden_dims=(512, 512), output_dim: int = 256,
                 aggregation: str = "sum", dropout: float = 0.3, **kwargs):
        super().__init__()
        self.aggregation = aggregation
        layers = []
        in_d = feat_dim
        for hd in hidden_dims:
            layers += [nn.Linear(in_d, hd), nn.BatchNorm1d(hd), nn.ReLU(True), nn.Dropout(dropout)]
            in_d = hd
        layers.append(nn.Linear(in_d, output_dim))
        self.rho = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """x: (B, n, feat_dim) → (B, output_dim)"""
        if self.aggregation == "sum":
            agg = x.sum(dim=1)
        elif self.aggregation == "mean":
            agg = x.mean(dim=1)
        elif self.aggregation == "max":
            agg = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        return self.rho(agg)


# Unified DatasetEncoder
class DatasetEncoder(nn.Module):
    """Modular dataset encoder = feature_extractor + set_encoder + classifier.

    Fully backward-compatible with ``DeepSetsClassifier``: exposes
    ``.forward()``, ``.encode()``, and ``.embedding_dim``.
    """

    def __init__(
        self,
        num_classes: int,
        # Feature extractor
        feature_extractor: str = "conv",       # "conv"
        fe_kwargs: Optional[Dict[str, Any]] = None,
        # Set encoder
        set_encoder: str = "deepsets",         # "deepsets"
        se_kwargs: Optional[Dict[str, Any]] = None,
        # Embedding
        embedding_dim: int = 256,
        normalize_embeddings: bool = False,
        embedding_scale: float = 1.0,
        # Conv feature extractor defaults (only used when feature_extractor="conv")
        input_channels: int = 3,
    ):
        super().__init__()
        fe_kwargs = dict(fe_kwargs or {})
        se_kwargs = dict(se_kwargs or {})

        # Feature extractor
        if feature_extractor == "conv":
            fe_kwargs.setdefault("input_channels", input_channels)
            fe_kwargs.setdefault("output_dim", 256)
        self.feature_extractor = build_feature_extractor(feature_extractor, **fe_kwargs)
        feat_dim = self.feature_extractor.output_dim

        # Set encoder
        se_kwargs["feat_dim"] = feat_dim
        se_kwargs.setdefault("output_dim", embedding_dim)
        if set_encoder == "deepsets":
            self.set_encoder = DeepSetsAggregator(**se_kwargs)
        else:
            raise ValueError(f"Unknown set_encoder: {set_encoder}")

        self.embedding_dim = self.set_encoder.output_dim
        self.normalize_embeddings = normalize_embeddings
        self.embedding_scale = embedding_scale

        # Classifier head
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

        logger.info(f"DatasetEncoder: feature_extractor={feature_extractor}, "
                     f"set_encoder={set_encoder}, embedding_dim={self.embedding_dim}, "
                     f"num_classes={num_classes}")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n, C, H, W) → (B, n, feat_dim)"""
        B, n = x.shape[:2]
        feats = self.feature_extractor(x.reshape(B * n, *x.shape[2:]))  # (B*n, feat_dim)
        return feats.view(B, n, -1)

    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1) * self.embedding_scale
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: input → logits.

        Args:
            x: (B, set_size, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feats = self._extract_features(x)
        emb = self.set_encoder(feats)
        emb = self._normalize(emb)
        return self.classifier(emb)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings without classification head."""
        feats = self._extract_features(x)
        emb = self.set_encoder(feats)
        return self._normalize(emb)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for ``get_embeddings`` (expected by ``DatasetAlignmentLoss``)."""
        return self.get_embeddings(x)


# Presets — single-string shortcuts for grid search
ENCODER_PRESETS: Dict[str, Dict[str, Any]] = {
    # Original DeepSets: Conv CNN + sum aggregation (same as legacy ImagePhi)
    "deepsets_conv": {
        "feature_extractor": "conv",
        "fe_kwargs": {"hidden_dims": [64, 128, 256], "output_dim": 256, "dropout": 0.3},
        "set_encoder": "deepsets",
        "se_kwargs": {"hidden_dims": [512, 512], "aggregation": "sum", "dropout": 0.3},
    },
}


# Factory from config dict
def build_dataset_encoder(config: Dict[str, Any]) -> DatasetEncoder:
    """Build a ``DatasetEncoder`` from a flat config dictionary.

    Expected keys (all optional except ``num_classes``):
        num_classes, feature_extractor, fe_kwargs, set_encoder, se_kwargs,
        embedding_dim, normalize_embeddings, embedding_scale, input_channels
    """
    return DatasetEncoder(
        num_classes=config["num_classes"],
        feature_extractor=config.get("feature_extractor", "conv"),
        fe_kwargs=config.get("fe_kwargs", None),
        set_encoder=config.get("set_encoder", "deepsets"),
        se_kwargs=config.get("se_kwargs", None),
        embedding_dim=config.get("embedding_dim", 256),
        normalize_embeddings=config.get("normalize_embeddings", False),
        embedding_scale=config.get("embedding_scale", 1.0),
        input_channels=config.get("input_channels", 3),
    )
