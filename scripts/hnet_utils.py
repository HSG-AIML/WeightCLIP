#!/usr/bin/env python3
"""Shared utilities, architecture, and constants for the ResNet18Slim hypernetwork baseline.

constants, helpers (set_seed, mean_std, mask_logits, normalize_dataset_name),
DatasetTensorCache, Text2ModelStructuredStateHyperNetwork, build_dataset_encoder_for_hnet.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
INTERNAL_SANE_DIR = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, INTERNAL_SANE_DIR / "src", INTERNAL_SANE_DIR):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))


def _find_text2model_dir() -> Path:
    """Walk up from this script until we find a sibling Text2Model directory."""
    for parent in [INTERNAL_SANE_DIR, *INTERNAL_SANE_DIR.parents]:
        candidate = parent / "Text2Model"
        if (candidate / "models.py").is_file(): return candidate
    raise FileNotFoundError(f"Could not find Text2Model/models.py walking up from {INTERNAL_SANE_DIR}")


TEXT2MODEL_DIR = _find_text2model_dir()
TEXT2MODEL_MODELS_PATH = TEXT2MODEL_DIR / "models.py"

_T2M_SPEC = importlib.util.spec_from_file_location("text2model_models", TEXT2MODEL_MODELS_PATH)
if _T2M_SPEC is None or _T2M_SPEC.loader is None: raise ImportError(f"Could not load Text2Model module from {TEXT2MODEL_MODELS_PATH}")
_t2m = importlib.util.module_from_spec(_T2M_SPEC)
_T2M_SPEC.loader.exec_module(_t2m)
Text2ModelEVLayer = _t2m.EVLayer


from sane.model.dataset_encoder import ENCODER_PRESETS, build_dataset_encoder
from sane.model.position import FunctionalSinusoidalPositionEmbeddings, LearnedPositionEmbeddings

from ood_utils import get_train_test_loaders
from tans_utils import get_num_classes as get_resnet_num_classes, get_zoo_path as get_resnet_zoo_path, load_dataset_pt


# ---- constants ------------------------------------------------------------

RESNET_METATRAIN_DATASETS = ["land-cover", "cactus-aerial", "ct-images", "lego-vs-generic", "cassava-leaf", "asl", "artworks", "blood-cells", "casting", "breakhis"]


# ---- helpers --------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals: return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def normalize_dataset_name(name: str) -> str: return name.lower().replace("_", "-")


def infer_target_classes(train_datasets: Iterable[str]) -> int:
    return max(get_resnet_num_classes(ds, architecture="resnet") for ds in train_datasets)


def mask_logits(logits: torch.Tensor, num_classes: int, enabled: bool = True) -> torch.Tensor:
    """Set logits past `num_classes` to -inf so they never win argmax."""
    if not enabled or logits.shape[-1] <= num_classes: return logits
    out = logits.clone()
    out[..., num_classes:] = -1e9
    return out


class DatasetTensorCache:
    """Caches tensorised train splits so we can sample image subsets cheaply."""

    def __init__(self, max_cache_samples: Optional[int] = None):
        self.max_cache_samples = max_cache_samples
        self._cache: Dict[str, torch.Tensor] = {}

    def _materialise(self, dataset) -> torch.Tensor:
        if hasattr(dataset, "tensors") and dataset.tensors:
            data = dataset.tensors[0]
            if self.max_cache_samples and len(data) > self.max_cache_samples:
                data = data[torch.randperm(len(data))[: self.max_cache_samples]]
            return data.clone()
        limit = min(len(dataset), self.max_cache_samples) if self.max_cache_samples else len(dataset)
        return torch.stack([dataset[i][0] for i in range(limit)])

    def get(self, dataset_name: str) -> torch.Tensor:
        key = normalize_dataset_name(dataset_name)
        if key in self._cache: return self._cache[key]
        trainset, _ = load_dataset_pt(key, architecture="resnet")
        if trainset is None:
            root = get_resnet_zoo_path(key, architecture="resnet")
            train_loader, _ = get_train_test_loaders(key, batch_size=8, num_workers=0, root_dir=root)
            if train_loader is None: raise FileNotFoundError(f"Could not load train split for {dataset_name}")
            trainset = train_loader.dataset
        self._cache[key] = self._materialise(trainset)
        return self._cache[key]


# ---- hypernetwork ---------------------------------------------------------

class Text2ModelStructuredStateHyperNetwork(nn.Module):
    """EVLayer stack that maps one dataset embedding to a sparse-token state dict.

    Two EVLayer blocks encode the dataset embedding into a hidden vector; the hidden
    vector is broadcast over `L` state-dict-token positions and a learned positional
    embedding is added; three more EVLayer blocks decode each position into a
    `token_dim`-wide token.
    """

    def __init__(self, input_dim: int, anchor_mask: torch.Tensor, anchor_pos: torch.Tensor, token_dim: int, hidden_dim: int = 120, token_window_size: int = 256, position_embedding: str = "learned"):
        super().__init__()
        self.input_dim, self.hidden_dim, self.token_dim = int(input_dim), int(hidden_dim), int(token_dim)
        self.token_window_size = int(token_window_size)

        self.Hs1 = Text2ModelEVLayer(input_dim, hidden_dim)
        self.Hs2 = Text2ModelEVLayer(hidden_dim, hidden_dim)
        self.token_H1 = Text2ModelEVLayer(hidden_dim, hidden_dim)
        self.token_H2 = Text2ModelEVLayer(hidden_dim, hidden_dim)
        self.token_head = Text2ModelEVLayer(hidden_dim, token_dim)

        max_positions = [int(anchor_pos[:, j].max().item()) + 1 for j in range(anchor_pos.shape[1])]
        if position_embedding == "learned":
            self.position_embedding = LearnedPositionEmbeddings(hidden_dim, max_positions=max_positions)
        elif position_embedding == "sinusoidal":
            self.position_embedding = FunctionalSinusoidalPositionEmbeddings(hidden_dim, num_pos_dims=int(anchor_pos.shape[1]))
        else: raise ValueError(f"Unknown position embedding '{position_embedding}'")

        self.register_buffer("anchor_mask", anchor_mask.to(dtype=torch.bool), persistent=False)
        self.register_buffer("anchor_mask_float", anchor_mask.to(dtype=torch.float32), persistent=False)
        self.register_buffer("anchor_pos", anchor_pos.to(dtype=torch.long), persistent=False)

    def _decode_token_windows(self, token_inputs: torch.Tensor) -> torch.Tensor:
        window = token_inputs.shape[0] if self.token_window_size <= 0 else self.token_window_size
        chunks: List[torch.Tensor] = []
        for chunk in torch.split(token_inputs, window, dim=0):
            chunk_list = [chunk[i : i + 1] for i in range(chunk.shape[0])]
            feat = [F.relu(z) for z in self.token_H1(chunk_list)]
            feat = [F.relu(z) for z in self.token_H2(feat)]
            chunks.append(torch.cat(self.token_head(feat), dim=0))
        return torch.cat(chunks, dim=0)

    def forward(self, dataset_embedding: torch.Tensor) -> torch.Tensor:
        if dataset_embedding.dim() == 1: embedding_list = [dataset_embedding.unsqueeze(0)]
        elif dataset_embedding.dim() == 2 and dataset_embedding.shape[0] == 1: embedding_list = [dataset_embedding]
        else: raise ValueError(f"Expected a single dataset embedding, got shape {tuple(dataset_embedding.shape)}")

        features = [F.relu(z) for z in self.Hs1(embedding_list)]
        features = [F.relu(z) for z in self.Hs2(features)]
        global_feature = features[0].expand(self.anchor_pos.shape[0], -1)
        token_inputs = self.position_embedding(global_feature.unsqueeze(0), self.anchor_pos.unsqueeze(0)).squeeze(0)
        return self._decode_token_windows(token_inputs) * self.anchor_mask_float


# ---- dataset encoder ------------------------------------------------------

def build_dataset_encoder_for_hnet(args, device: torch.device) -> Tuple[nn.Module, Dict]:
    """DeepSets-style dataset encoder with a dataset-ID classifier head for warm-start."""
    if args.dataset_encoder_preset not in ENCODER_PRESETS:
        raise ValueError(f"Unknown dataset encoder preset '{args.dataset_encoder_preset}'. Available: {sorted(ENCODER_PRESETS.keys())}")
    cfg = dict(ENCODER_PRESETS[args.dataset_encoder_preset])
    cfg["num_classes"] = int(len(args.train_datasets))
    cfg.setdefault("embedding_dim", int(args.dataset_embedding_dim))
    model = build_dataset_encoder(cfg).to(device)
    model.train()
    return model, {"preset": args.dataset_encoder_preset, "config": cfg}
