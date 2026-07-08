#!/usr/bin/env python3
"""
TANS Utilities for CNN3 and ResNet18 Model Zoos

- HardNegativeContrastiveLoss: Exact implementation from TANS
- compute_recall: Exact implementation from TANS
- Encoders: Adapted for CNN3/ResNet (functional embedding instead of OFA topology)
- Data loading utilities for both CNN3 and ResNet18Slim model zoos

Original TANS: https://github.com/wyjeong/TANS
"""

import os
import sys
import glob
import json
import random
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Add paths for imports
script_dir = Path(__file__).resolve().parent
internal_sane_dir = script_dir.parent
sys.path.insert(0, str(internal_sane_dir / "src"))

from sane.data.datasets.zoo_dataset_models import ResNet18Slim
from ood_utils import canonicalize_model_type, create_model_from_state_dict, get_model_config_cached, get_num_classes as shared_get_num_classes, get_zoo_path as shared_get_zoo_path, load_dataset_from_root_dir


# Configuration

_FIXED_DATASET_SPLITS = {
    'cnn3': {
        'metatrain': (
            'proptit-aif-homework', 'cactus-aerial', 'dl2020', 'four-shapes', 'e4040-assignment',
            'car-classification', 'simpsons-characters', 'simpsons-challenge', 'breakhis', 'mushrooms',
            'artworks', 'numta-bengali', 'lego-bricks', 'ads5035', 'day3-kaggle',
            'ct-images', 'land-cover', 'casting', 'asl', 'cassava-leaf',
        ),
        'metatest': (
            'colorectal-histology', 'covid19', 'cifar10', 'speed-limit-signs',
            'honeybee-pollen', 'real-or-drawing',
        ),
    },
    'resnet18slim': {
        'metatrain': (
            'cactus-aerial', 'lego-vs-generic', 'asl', 'breakhis', 'artworks',
            'blood-cells', 'ct-images', 'land-cover', 'casting', 'cassava-leaf',
        ),
        'metatest': (
            'colorectal-histology', 'covid19', 'cifar10', 'speed-limit-signs',
            'honeybee-pollen', 'real-or-drawing',
        ),
    },
}


def _resolve_architecture(architecture=None):
    return canonicalize_model_type('cnn3' if architecture is None else architecture)


def _get_fixed_dataset_split(split, architecture=None):
    family = _resolve_architecture(architecture)
    if family not in _FIXED_DATASET_SPLITS:
        raise ValueError(f'Unsupported architecture for fixed TANS split: {family}')
    return _FIXED_DATASET_SPLITS[family][split]


def get_metatrain_datasets(architecture=None):
    return _get_fixed_dataset_split('metatrain', architecture)


def get_metatest_datasets(architecture=None):
    return _get_fixed_dataset_split('metatest', architecture)


# TANS Loss Function (Exact reproduction from TANS/retrieval/loss.py)

class HardNegativeContrastiveLoss(nn.Module):
    """Hard Negative Contrastive Loss (exact TANS implementation).
    
    From TANS paper: Uses hard negative mining with margin-based ranking loss.
    """
    def __init__(self, nmax=1, margin=0.2, contrast=True):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.contrast = contrast

    def forward(self, m, q, matched=None):
        """
        Args:
            m: [N, dim] model embeddings
            q: [N, dim] query embeddings
        Returns:
            loss: scalar
        """
        scores = torch.mm(m, q.t())  # (N, N)
        diag = scores.diag()  # (N,)
        
        # Zero out diagonal
        scores = scores - torch.diag(scores.diag())
        
        # Sort the score matrix in the query dimension
        sorted_query, _ = torch.sort(scores, 0, descending=True)
        # Sort the score matrix in the model dimension
        sorted_model, _ = torch.sort(scores, 1, descending=True)
        
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :]  # (nmax, N)
        max_m = sorted_model[:, :self.nmax]  # (N, nmax)
        
        neg_q = torch.sum(torch.clamp(
            max_q + (self.margin - diag).view(1, -1).expand_as(max_q), min=0))
        neg_m = torch.sum(torch.clamp(
            max_m + (self.margin - diag).view(-1, 1).expand_as(max_m), min=0))

        if self.contrast:
            loss = neg_m + neg_q
        else:
            loss = neg_m

        return loss

# TANS Metrics (Exact reproduction from TANS/retrieval/measure.py)

def compute_recall(query_embs, model_embs, npts=None):
    """Compute recall metrics (exact TANS implementation).
    
    Args:
        query_embs: [N, dim] query embeddings
        model_embs: [N, dim] model embeddings
        npts: number of points (defaults to N)
    
    Returns:
        recalls: dict with R@1, R@5, R@10, R@50, R@100
        medr: median rank
        meanr: mean rank
    """
    if isinstance(query_embs, torch.Tensor):
        query_embs = query_embs.cpu().numpy()
    if isinstance(model_embs, torch.Tensor):
        model_embs = model_embs.cpu().numpy()
    
    if npts is None:
        npts = query_embs.shape[0]
    
    index_list = []
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    for index in range(npts):
        # Get query embedding
        query_emb = query_embs[index].reshape(1, query_embs.shape[1])
        # Compute scores
        d = np.dot(query_emb, model_embs.T).flatten()
        sorted_index_lst = np.argsort(d)[::-1]
        index_list.append(sorted_index_lst[0])
        # Score
        rank = np.where(sorted_index_lst == index)[0][0]
        ranks[index] = rank
        top1[index] = sorted_index_lst[0]
    
    recalls = {}
    for v in [1, 5, 10, 50, 100]:
        recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
    
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    return recalls, medr, meanr


# TANS Encoders (Adapted for CNN3)

class QueryEncoder(nn.Module):
    """Dataset Query Encoder (TANS-style).
    
    Original TANS: Takes 512-dim ResNet18 features per image,
    projects to n_dims, averages over images, L2 normalizes.
    
    Exact reproduction of TANS/retrieval/nets.py QueryEncoder.
    """
    def __init__(self, n_dims=128, input_dim=512):
        super(QueryEncoder, self).__init__()
        self.n_dims = n_dims
        self.fc = nn.Linear(input_dim, n_dims)
    
    def forward(self, D):
        """
        Args:
            D: List of [n_images, 512] tensors (one per dataset)
        Returns:
            q: [batch_size, n_dims] L2-normalized query embeddings
        """
        q = []
        for d in D:
            _q = self.fc(d)  # [n_images, n_dims]
            _q = torch.mean(_q, 0)  # [n_dims]
            _q = self.l2norm(_q.unsqueeze(0))  # [1, n_dims]
            q.append(_q)
        q = torch.stack(q).squeeze(1)  # [batch, n_dims]
        return q
    
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x


class ModelEncoder(nn.Module):
    """Model Encoder using Functional Embedding (TANS-style, adapted for CNN3/ResNet).
    
    Original TANS: topology (45-dim) + functional_emb (1536-dim) -> normalize -> project -> L2 norm
    
    Our adaptation: Since all CNN3/ResNet models have identical topology, we only use
    functional embedding (fixed Gaussian noise -> network output).
    
    TANS uses functional embedding to embed networks based on input-output pairs by
    feeding fixed Gaussian random noise into each network and obtaining the output.
    """
    def __init__(self, n_dims=128, functional_dim=None):
        super(ModelEncoder, self).__init__()
        self.n_dims = n_dims
        self.functional_dim = functional_dim
        if functional_dim is not None:
            self.fc = nn.Linear(functional_dim, n_dims)
    
    def set_functional_dim(self, functional_dim):
        """Set functional dimension after initialization."""
        self.functional_dim = functional_dim
        self.fc = nn.Linear(functional_dim, self.n_dims)
    
    def forward(self, v_f):
        """
        Args:
            v_f: [batch_size, functional_dim] functional embeddings
        Returns:
            m: [batch_size, n_dims] L2-normalized model embeddings
        """
        # Normalize input (like TANS does with concat of topology + functional)
        m = F.normalize(v_f, dim=-1)
        m = self.fc(m)
        m = self.l2norm(m)
        return m
    
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x


class PerformancePredictor(nn.Module):
    """Accuracy predictor from concatenated embeddings (exact TANS implementation)."""
    def __init__(self, n_dims=128):
        super(PerformancePredictor, self).__init__()
        self.fc = nn.Linear(n_dims * 2, 1)
    
    def forward(self, q, m):
        """Predict accuracy from query and model embeddings."""
        p = torch.cat([q, m], dim=1)
        p = torch.sigmoid(self.fc(p))
        return p


# Functional Embedding Generator (TANS core idea)

class FunctionalEmbeddingGenerator:
    """Generate functional embeddings for CNN3 or ResNet18Slim models (TANS-style).

    Original TANS consumes a precomputed 1536-d functional vector from its OFA zoo.
    In this repo, we synthesize an analogous per-model functional signal by feeding
    one fixed bank of Gaussian noise images through each checkpoint and flattening
    either penultimate features or output probabilities across all samples.

    The resulting dimensionality depends on architecture, width multiplier,
    n_noise_samples, and use_features; it is not forced to equal 1536.
    """
    
    def __init__(self, n_noise_samples=16, device='cuda', seed=42, architecture=None):
        """
        Args:
            n_noise_samples: Number of fixed Gaussian noise images
            device: torch device
            seed: Random seed for reproducible noise
            architecture: 'cnn3' or 'resnet'
        """
        self.device = device
        self.n_noise_samples = n_noise_samples
        self.architecture = canonicalize_model_type('cnn3' if architecture is None else architecture)
        
        # Generate FIXED Gaussian noise (same for all models - CRITICAL for TANS)
        # Both CNN3 and ResNet18Slim expect 32x32 RGB images
        torch.manual_seed(seed)
        self.fixed_noise = torch.randn(n_noise_samples, 3, 32, 32, device=device)
        # Normalize to [-1, 1] range like training data
        self.fixed_noise = torch.clamp(self.fixed_noise, -3, 3) / 3
    
    @torch.no_grad()
    def extract(self, model, use_features=True):
        """Extract functional embedding from a model.
        
        Args:
            model: CNN3 or ResNet18Slim model instance
            use_features: If True, use pre-logit features (more stable across different n_classes)
        
        Returns:
            embedding: [functional_dim] tensor
        """
        model.eval()
        model = model.to(self.device)
        
        # Detect architecture based on model type
        is_resnet = isinstance(model, ResNet18Slim) or hasattr(model, 'layer4')
        
        if use_features:
            if is_resnet:
                # ResNet18Slim: extract features before fc layer
                x = self.fixed_noise
                x = model.act1(model.bn1(model.conv1(x)))
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)  # [n_samples, c4] where c4 = int(512 * width_mult)
                embedding = x.flatten()
            else:
                # CNN3: conv layers -> flatten -> fc(60,20) -> act -> dropout -> fc(20, n_classes)
                # We want the 20-dim features after the penultimate fc
                x = self.fixed_noise
                for i, layer in enumerate(model.module_list[:-1]):  # All except last fc
                    x = layer(x)
                # x is now [n_samples, 20] (pre-logit features)
                embedding = x.flatten()
        else:
            # Use full outputs (logits) - dimension varies by n_classes
            outputs = model(self.fixed_noise)  # [n_samples, n_classes]
            # Apply softmax for more stable representation
            outputs = F.softmax(outputs, dim=-1)
            embedding = outputs.flatten()
        
        return embedding
    
    def extract_batch(self, models, use_features=True):
        """Extract functional embeddings for multiple models.
        
        Args:
            models: List of CNN3 or ResNet18Slim model instances
            use_features: Use pre-logit features (recommended)
        
        Returns:
            embeddings: [n_models, functional_dim] tensor
        """
        embeddings = []
        for model in models:
            emb = self.extract(model, use_features=use_features)
            embeddings.append(emb)
        
        # Handle variable output dimensions (different n_classes) by padding
        max_dim = max(e.shape[0] for e in embeddings)
        padded = []
        for e in embeddings:
            if e.shape[0] < max_dim:
                e = F.pad(e, (0, max_dim - e.shape[0]))
            padded.append(e)
        
        return torch.stack(padded)


# Image Feature Extractor (pretrained ResNet18 like TANS)

class ImageFeatureExtractor:
    """Extract features from images using pretrained ResNet18 (TANS-style).
    
    TANS uses ResNet18 pretrained on ImageNet to get 512-dim features per image.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC layer to get 512-dim features
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract(self, images):
        """
        Args:
            images: [N, C, H, W] tensor (normalized to [-1, 1] or [0, 1])
        Returns:
            features: [N, 512] tensor
        """
        # Denormalize from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Ensure RGB (handle grayscale)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Apply ImageNet transform
        images = self.transform(images.to(self.device))
        
        # Extract features
        features = self.model(images)
        return features.view(features.size(0), -1)  # [N, 512]


# Data Loading Utilities

def get_zoo_path(dataset_name, architecture=None):
    """Get zoo path for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        architecture: 'cnn3' or 'resnet'
    
    Returns:
        Path to the zoo directory, or None if not found
    """
    return shared_get_zoo_path(dataset_name, model_type=_resolve_architecture(architecture))


def load_dataset_pt(dataset_name, architecture=None):
    """Load train/test data from zoo's dataset.pt
    
    Args:
        dataset_name: Name of the dataset
        architecture: 'cnn3' or 'resnet'
    """
    zoo_path = get_zoo_path(dataset_name, architecture=architecture)
    return (None, None) if zoo_path is None else load_dataset_from_root_dir(zoo_path)


def get_num_classes(dataset_name, architecture=None):
    """Get number of classes for a dataset from its zoo.
    
    Args:
        dataset_name: Name of the dataset
        architecture: 'cnn3' or 'resnet'
    """
    return shared_get_num_classes(dataset_name, model_type=_resolve_architecture(architecture))


def load_models_from_zoo(dataset_name, max_models=100, device='cpu', start_idx=0, architecture=None):
    """Load models from a dataset's zoo (supports CNN3 and ResNet18Slim).
    
    Args:
        dataset_name: Name of the dataset
        max_models: Maximum number of models to load
        device: Device to load models to
        start_idx: Starting index in the sorted checkpoint list (for train/val splits)
        architecture: 'cnn3' or 'resnet'
    
    Returns:
        models: List of model instances (CNN3 or ResNet18Slim)
        accuracies: List of accuracies
        state_dicts: List of state dicts
    """
    architecture = _resolve_architecture(architecture)
    zoo_path = get_zoo_path(dataset_name, architecture=architecture)
    if zoo_path is None:
        return None, None, None
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(glob.glob(str(zoo_path / 'NN_tune_trainable_seed=*')))
    
    n_classes = get_num_classes(dataset_name, architecture=architecture)
    model_config = get_model_config_cached(zoo_path)
    models = []
    accuracies = []
    state_dicts = []
    
    # Select subset based on start_idx and max_models
    end_idx = start_idx + max_models
    for ckpt_dir in checkpoint_dirs[start_idx:end_idx]:
        # Get the last checkpoint
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_*/checkpoints')))
        if not ckpt_files:
            continue
        
        last_ckpt = ckpt_files[-1]
        try:
            state_dict = torch.load(last_ckpt, map_location=device, weights_only=False)

            model = create_model_from_state_dict(state_dict, num_classes=n_classes, config=model_config, device=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            models.append(model)
            state_dicts.append(state_dict)
            
            # Try to get accuracy from progress file
            progress_file = Path(ckpt_dir) / 'progress.csv'
            acc = 0.9  # Default
            if progress_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(progress_file)
                    if 'val_accuracy' in df.columns:
                        acc = df['val_accuracy'].iloc[-1]
                    elif 'val_acc' in df.columns:
                        acc = df['val_acc'].iloc[-1]
                    elif 'accuracy' in df.columns:
                        acc = df['accuracy'].iloc[-1]
                except:
                    pass
            accuracies.append(acc)
            
        except Exception as e:
            continue
    
    if not models:
        return None, None, None
    
    return models, accuracies, state_dicts

def extract_query_features(dataset_name, feature_extractor, n_query_images=100, seed: Optional[int] = None, architecture=None):
    """Extract query features for a dataset (TANS-style).
    
    Args:
        dataset_name: Name of the dataset
        feature_extractor: ImageFeatureExtractor instance
        n_query_images: Number of images to sample
        seed: Random seed for reproducibility
        architecture: 'cnn3' or 'resnet'
    
    Returns ResNet18 features for a random sample of images.
    """
    trainset, _ = load_dataset_pt(dataset_name, architecture=architecture)
    if trainset is None:
        return None
    
    # Sample images (optionally using a local RNG for deterministic splits)
    n_samples = min(n_query_images, len(trainset))
    rng = random if seed is None else random.Random(int(seed))
    indices = rng.sample(range(len(trainset)), n_samples)
    
    images = torch.stack([trainset[i][0] for i in indices])
    features = feature_extractor.extract(images)
    
    return features  # [n_samples, 512]


def extract_functional_embeddings(models, func_emb_generator, use_features=True):
    """Extract functional embeddings for a list of models."""
    return func_emb_generator.extract_batch(models, use_features=use_features)


# File I/O Utilities (from TANS/misc/utils.py)

def torch_save(base_dir, filename, data):
    """Save torch data to file."""
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)
    print(f'File saved ({fpath})')


def torch_load(fpath):
    """Load torch data from file."""
    return torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)


def f_write(filepath, filename, data):
    """Write JSON data to file."""
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile, indent=2)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
