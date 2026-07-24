import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict, Tuple
from pathlib import Path
import logging
import __main__

from sane.data.datasets import DatasetPtImageSetSampler
from sane.loss.alignment import CachedDataset, GrayscaleToRGB
from sane.loss.logical_datasets import infer_logical_dataset_name

# Ensure shims are available in __main__ even when this module is imported standalone
__main__.CachedDataset = CachedDataset
__main__.GrayscaleToRGB = GrayscaleToRGB

logger = logging.getLogger(__name__)


def get_logical_dataset_name(zoo_path: str) -> str:
    return infer_logical_dataset_name(zoo_path)


def build_logical_dataset_registry(zoo_paths: list[str]) -> dict[str, int]:
    registry: dict[str, int] = {}
    for zoo_path in zoo_paths:
        name = get_logical_dataset_name(zoo_path)
        if name not in registry:
            registry[name] = len(registry)
    return registry


def collapse_batch_to_logical_datasets(dataset_indices: torch.Tensor, zoo_paths: list[str]) -> tuple[list[str], torch.Tensor]:
    logical_name_to_pos: Dict[str, int] = {}
    raw_idx_to_pos: Dict[int, int] = {}
    logical_zoo_paths = []
    for raw in dataset_indices.unique():
        ri = raw.item()
        if ri >= len(zoo_paths):
            continue
        name = get_logical_dataset_name(zoo_paths[ri])
        if name not in logical_name_to_pos:
            logical_name_to_pos[name] = len(logical_zoo_paths)
            logical_zoo_paths.append(zoo_paths[ri])
        raw_idx_to_pos[ri] = logical_name_to_pos[name]

    model_pos = torch.tensor([raw_idx_to_pos[i.item()] for i in dataset_indices], device=dataset_indices.device, dtype=torch.long)
    return logical_zoo_paths, model_pos




class DatasetEncoderAuxiliaryLosses:
    """Reusable helper for dataset-encoder classification term."""

    def __init__(
        self,
        dataset_encoder: nn.Module,
        logical_dataset_to_class_id: dict[str, int],
        *,
        set_size: int = 32,
        image_resize: Optional[Any] = None,
        split: str = "trainset",
        classification_weight: float = 1.0,
        target_channels: int = 3,
    ):
        self.dataset_encoder = dataset_encoder
        self.logical_dataset_to_class_id = logical_dataset_to_class_id
        self.classification_weight = classification_weight
        resize = self._parse_image_resize(image_resize)
        target_hw = (32, 32) if resize is None else resize
        self.sampler = DatasetPtImageSetSampler(
            split=split,
            set_size=set_size,
            target_image_size=target_hw,
            target_channels=target_channels,
        )

    def _parse_image_resize(self, image_resize: Optional[Any]) -> Optional[Tuple[int, int]]:
        if image_resize is None:
            return None
        if isinstance(image_resize, int):
            if image_resize <= 0:
                raise ValueError(f"image_resize must be positive, got {image_resize}")
            return (image_resize, image_resize)
        if isinstance(image_resize, (list, tuple)):
            if len(image_resize) == 1:
                s = int(image_resize[0])
                if s <= 0:
                    raise ValueError(f"image_resize must be positive, got {image_resize}")
                return (s, s)
            if len(image_resize) == 2:
                h, w = int(image_resize[0]), int(image_resize[1])
                if h <= 0 or w <= 0:
                    raise ValueError(f"image_resize values must be positive, got {image_resize}")
                return (h, w)
        raise ValueError(f"image_resize must be None, int, or 1/2-item sequence, got {image_resize!r}")

    def _build_batch(self, dataset_indices: torch.Tensor, zoo_paths: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        logical_zoo_paths, _ = collapse_batch_to_logical_datasets(dataset_indices, zoo_paths)
        image_sets = []
        labels = []
        for zoo_path in logical_zoo_paths:
            logical_name = get_logical_dataset_name(zoo_path)
            class_id = self.logical_dataset_to_class_id[logical_name]
            image_sets.append(self.sampler.sample_image_set(zoo_path, device=device))
            labels.append(class_id)

        if len(image_sets) < 2:
            return torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)

        set_size = max(imgs.shape[0] for imgs in image_sets)
        padded = []
        for imgs in image_sets:
            if imgs.shape[0] < set_size:
                reps = (set_size + imgs.shape[0] - 1) // imgs.shape[0]
                imgs = imgs.repeat(reps, 1, 1, 1)[:set_size]
            padded.append(imgs)

        batch = torch.stack(padded)
        labels_t = torch.tensor(labels, device=device, dtype=torch.long)
        return batch, labels_t

    def compute(self, dataset_indices: torch.Tensor, zoo_paths: list[str], *, training: bool) -> Dict[str, torch.Tensor]:
        device = dataset_indices.device
        zero = torch.tensor(0.0, device=device)
        batch, labels = self._build_batch(dataset_indices, zoo_paths, device)
        if batch.numel() == 0:
            return {'deepsets_cls_loss': zero, 'deepsets_cls_acc': zero}

        logits = self.dataset_encoder(batch)
        cls_loss = F.cross_entropy(logits, labels)
        cls_loss = self.classification_weight * cls_loss
        acc = (logits.argmax(dim=1) == labels).float().mean() * 100.0

        return {'deepsets_cls_loss': cls_loss, 'deepsets_cls_acc': acc}


class DeepSetsClassificationLoss(nn.Module):
    """Dataset-identity classification loss for the DeepSets encoder.

    Trains the dataset encoder to discriminate between logical datasets by
    running its full forward pass (encoder + classifier head) on sampled
    image sets and computing cross-entropy against logical dataset labels.

    Intended to run alongside DatasetAlignmentLoss. The training code is
    responsible for combining the two losses with appropriate weights:

        total = alignment_loss + cls_weight * deepsets_cls_loss

    Args:
        deepsets_encoder: Dataset encoder with .forward(x) -> [B, num_classes].
            Input x: [B, set_size, C, H, W].
        weight: Scalar weight applied before returning the loss. Default 1.0.
        set_size: Images sampled per dataset from dataset.pt. Default 32.
        image_resize: Optional (H, W) to resize dataset images. Accepts int
            (square), (H, W) tuple, or None.
    """

    def __init__(self, deepsets_encoder: nn.Module, weight: float = 1.0, set_size: int = 32, image_resize: Optional[Any] = None):
        super().__init__()
        self.deepsets_encoder = deepsets_encoder
        self.weight = weight
        self.set_size = set_size
        self.image_resize = self._parse_image_resize(image_resize)

    # Dataset loading helpers (mirror of DatasetAlignmentLoss)

    def _get_logical_dataset_name(self, zoo_path: str) -> str:
        return get_logical_dataset_name(zoo_path)

    def _load_dataset(self, zoo_path: str):
        if zoo_path in self._dataset_cache:
            return self._dataset_cache[zoo_path]
        pt = Path(zoo_path) / "dataset.pt"
        if not pt.exists():
            logger.warning("dataset.pt not found in %s", zoo_path)
            self._dataset_cache[zoo_path] = None
            return None
        d = torch.load(str(pt), weights_only=False)
        dataset = d.get('trainset') or d.get('testset')
        if dataset is None:
            logger.warning("No trainset/testset in %s", pt)
        self._dataset_cache[zoo_path] = dataset
        return dataset

    def _parse_image_resize(self, image_resize: Optional[Any]) -> Optional[Tuple[int, int]]:
        if image_resize is None:
            return None
        if isinstance(image_resize, int):
            if image_resize <= 0:
                raise ValueError(f"image_resize must be positive, got {image_resize}")
            return (image_resize, image_resize)
        if isinstance(image_resize, (list, tuple)):
            if len(image_resize) == 1:
                s = int(image_resize[0])
                if s <= 0:
                    raise ValueError(f"image_resize must be positive, got {image_resize}")
                return (s, s)
            if len(image_resize) == 2:
                h, w = int(image_resize[0]), int(image_resize[1])
                if h <= 0 or w <= 0:
                    raise ValueError(f"image_resize values must be positive, got {image_resize}")
                return (h, w)
        raise ValueError(f"image_resize must be None, int, or 1/2-item sequence, got {image_resize!r}")

    def _normalize_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if self.image_resize is not None and tuple(images.shape[-2:]) != self.image_resize:
            if not torch.is_floating_point(images):
                images = images.float()
            images = F.interpolate(images, size=self.image_resize, mode="bilinear", align_corners=False)
        return images

    def _sample_images(self, dataset, device: torch.device) -> torch.Tensor:
        """Sample set_size random images -> [n, C, H, W] on device."""
        total = len(dataset)
        indices = torch.randperm(total)[:min(self.set_size, total)]
        if hasattr(dataset, 'tensors'):
            images = dataset.tensors[0][indices]
        else:
            images = torch.stack([dataset[i.item()][0] for i in indices])
        return self._normalize_image_batch(images).to(device)

    # Forward

    def forward(self, dataset_indices: torch.Tensor, zoo_paths: list) -> Dict[str, torch.Tensor]:
        """Compute dataset-identity classification loss for DeepSets.

        Args:
            dataset_indices: [B] integer indices into zoo_paths (one per model).
            zoo_paths: list of zoo root paths.

        Returns dict with:
            'deepsets_cls_loss' - weighted cross-entropy scalar.
            'deepsets_cls_acc'  - % datasets classified correctly (no-grad).
        """
        logical_zoo_paths, _ = collapse_batch_to_logical_datasets(dataset_indices, zoo_paths)
        registry = {get_logical_dataset_name(path): idx for idx, path in enumerate(logical_zoo_paths)}
        helper = DatasetEncoderAuxiliaryLosses(
            dataset_encoder=self.deepsets_encoder,
            logical_dataset_to_class_id=registry,
            set_size=self.set_size,
            image_resize=self.image_resize,
            classification_weight=self.weight,
        )
        out = helper.compute(dataset_indices, zoo_paths, training=False)
        return {
            'deepsets_cls_loss': out['deepsets_cls_loss'],
            'deepsets_cls_acc': out['deepsets_cls_acc'].item() if isinstance(out['deepsets_cls_acc'], torch.Tensor) else out['deepsets_cls_acc'],
        }
