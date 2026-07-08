#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

_repo_root = Path(__file__).resolve().parent.parent
for _p in (_repo_root / "src", _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sane.data.datasets.zoo_dataset_models import ResNet18Slim

from tans_common import load_dataset_pt


# ---- loss & metrics --------------------------------------------------

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=20, margin=0.2):
        super().__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, model_embs, query_embs):
        scores = torch.mm(model_embs, query_embs.t())
        diag = scores.diag()
        scores = scores - torch.diag(diag)
        hard_q = torch.sort(scores, 0, descending=True).values[: self.nmax]
        hard_m = torch.sort(scores, 1, descending=True).values[:, : self.nmax]
        loss_q = torch.clamp(hard_q + (self.margin - diag).view(1, -1), min=0).sum()
        loss_m = torch.clamp(hard_m + (self.margin - diag).view(-1, 1), min=0).sum()
        return loss_q + loss_m


def compute_recall(q_embs, m_embs):
    if isinstance(q_embs, torch.Tensor):
        q_embs = q_embs.detach().cpu().numpy()
    if isinstance(m_embs, torch.Tensor):
        m_embs = m_embs.detach().cpu().numpy()
    n = q_embs.shape[0]
    ranks = np.array([
        np.where(np.argsort(np.dot(q_embs[i : i + 1], m_embs.T).flatten())[::-1] == i)[0][0]
        for i in range(n)
    ])
    recalls = {k: 100.0 * (ranks < k).mean() for k in [1, 5, 10, 50, 100]}
    return recalls, float(np.median(ranks) + 1), float(ranks.mean() + 1)


# ---- encoders --------------------------------------------------------

class QueryEncoder(nn.Module):
    def __init__(self, n_dims=128, input_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_dims)

    def forward(self, feat_list):
        return torch.cat([F.normalize(self.fc(feats).mean(0, keepdim=True), dim=1) for feats in feat_list],dim=0)


class ModelEncoder(nn.Module):
    def __init__(self, n_dims=128, functional_dim=None):
        super().__init__()
        self.n_dims = n_dims
        self.functional_dim = functional_dim
        if functional_dim is not None:
            self.fc = nn.Linear(functional_dim, n_dims)

    def forward(self, functional_embs):
        return F.normalize(self.fc(F.normalize(functional_embs, dim=-1)), dim=1)


class PerformancePredictor(nn.Module):
    def __init__(self, n_dims=128):
        super().__init__()
        self.fc = nn.Linear(n_dims * 2, 1)

    def forward(self, q_embs, m_embs):
        return torch.sigmoid(self.fc(torch.cat([q_embs, m_embs], dim=1)))


# ---- functional embeddings -------------------------------------------

class FunctionalEmbeddingGenerator:
    def __init__(self, n_noise_samples=16, device="cuda", seed=42, architecture=None):
        self.device = device
        torch.manual_seed(seed)
        noise = torch.randn(n_noise_samples, 3, 32, 32, device=device)
        self.fixed_noise = torch.clamp(noise, -3, 3) / 3

    @torch.no_grad()
    def extract_one(self, model, use_features=True):
        model.eval()
        model.to(self.device)
        x = self.fixed_noise
        if use_features:
            if isinstance(model, ResNet18Slim) or hasattr(model, "layer4"):
                x = model.act1(model.bn1(model.conv1(x)))
                for layer in (model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool):
                    x = layer(x)
                return x.flatten()
            for layer in model.module_list[:-1]:
                x = layer(x)
            return x.flatten()
        return F.softmax(model(x), dim=-1).flatten()

    def extract_batch(self, zoo_models, use_features=True):
        embs = [self.extract_one(m, use_features) for m in zoo_models]
        max_d = max(e.shape[0] for e in embs)
        return torch.stack([F.pad(e, (0, max_d - e.shape[0])) for e in embs])


class ImageFeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def extract(self, images):
        if images.min() < 0:
            images = (images + 1) / 2
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return self.net(self.tf(images.to(self.device))).flatten(1)


def query_features_for_dataset(dataset_name, extractor, n_images=100, seed=None, architecture=None):
    trainset, _ = load_dataset_pt(dataset_name, architecture)
    if trainset is None:
        return None
    rng = random if seed is None else random.Random(seed)
    idx = rng.sample(range(len(trainset)), min(n_images, len(trainset)))
    return extractor.extract(torch.stack([trainset[i][0] for i in idx]))
