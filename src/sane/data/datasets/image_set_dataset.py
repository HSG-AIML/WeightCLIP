"""Dataset that yields *sets* of images for dataset-identity classification.

Each sample is a tuple ``(images, label)`` where:
    - ``images`` is a ``(set_size, C, H, W)`` float tensor
    - ``label``  is an integer class index (== dataset identity)

Zoo roots are supplied as a list of ``(root_dir, label)`` pairs.  The
``dataset.pt`` file inside each root is expected to contain a dict with at
least a ``trainset`` or ``testset`` key pointing to a standard PyTorch
``Dataset`` whose items are ``(image_tensor, class_label)``.

Set sizes can be fixed or randomly drawn from an inclusive ``[min, max]``
range each call.  Images are sampled *with replacement* so that any valid
set size is always achievable regardless of zoo size.
All images are converted to a user-specified target ``(C, H, W)``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)


class CachedDataset(Dataset):
    """Compatibility shim for old pickled ``dataset.pt`` files.

    Some historic zoo datasets were serialized from a script where
    ``CachedDataset`` lived in ``__main__``. Registering this class under the
    current ``__main__`` module lets ``torch.load`` unpickle those files.
    """

    def __init__(self, dataset=None, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []

        if dataset is not None:
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                self.data.append(img)
                self.targets.append(label)

            self.data = torch.stack(self.data)
            self.targets = torch.tensor(self.targets)

    def __len__(self):
        if hasattr(self, "targets") and self.targets is not None:
            return len(self.targets)
        return 0

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def _register_pickle_compat_types() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CachedDataset"):
        setattr(main_module, "CachedDataset", CachedDataset)


def resolve_dataset_pt_path(root_dir_or_dataset_pt: str | Path) -> Path:
    path = Path(root_dir_or_dataset_pt)
    return path if path.name == "dataset.pt" else path / "dataset.pt"


class DatasetPtImageSetSampler:
    """Lazy sampler for image sets stored in zoo ``dataset.pt`` files.

    This is the shared low-level utility used by the standalone dataset-encoder
    path and by joint training helpers that need to sample standardized image
    sets from model zoos on demand.
    """

    def __init__(
        self,
        split: str = "trainset",
        set_size: Union[int, Tuple[int, int]] = 16,
        target_image_size: Tuple[int, int] = (32, 32),
        target_channels: int = 3,
        transform=None,
    ):
        self.split = split
        self.set_size = set_size
        self.target_hw = tuple(target_image_size)
        self.target_channels = target_channels
        self.transform = transform
        self.image_shape = (self.target_channels, self.target_hw[0], self.target_hw[1])
        self._pools: dict[str, List[torch.Tensor]] = {}

    def _normalize_layout(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 2:
            return img.unsqueeze(0)

        if img.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {tuple(img.shape)}")

        if img.shape[0] in (1, 3, 4):
            return img
        if img.shape[-1] in (1, 3, 4):
            return img.permute(2, 0, 1)

        raise ValueError(f"Could not infer channel dimension for image shape {tuple(img.shape)}")

    def _convert_channels(self, img: torch.Tensor) -> torch.Tensor:
        current_channels = img.shape[0]
        if current_channels == self.target_channels:
            return img
        if self.target_channels == 1:
            return img.mean(dim=0, keepdim=True)
        if current_channels == 1:
            return img.repeat(self.target_channels, 1, 1)
        if current_channels > self.target_channels:
            return img[:self.target_channels]

        repeats = (self.target_channels + current_channels - 1) // current_channels
        return img.repeat(repeats, 1, 1)[:self.target_channels]

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        if (img.shape[-2], img.shape[-1]) == self.target_hw:
            return img
        return F.interpolate(
            img.unsqueeze(0),
            size=self.target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        img = self._normalize_layout(img).float()
        img = self._convert_channels(img)
        img = self._resize_image(img)
        return img.contiguous()

    def load_image_pool(self, root_dir: str) -> List[torch.Tensor]:
        pt_path = resolve_dataset_pt_path(root_dir)
        cache_key = str(pt_path)

        if cache_key in self._pools:
            return self._pools[cache_key]

        if not pt_path.exists():
            logger.error(f"dataset.pt not found at {pt_path}")
            self._pools[cache_key] = []
            return self._pools[cache_key]

        try:
            _register_pickle_compat_types()
            data = torch.load(str(pt_path), weights_only=False)
        except Exception as exc:
            logger.error(f"Failed to torch.load {pt_path}: {exc}")
            self._pools[cache_key] = []
            return self._pools[cache_key]

        dataset = data.get(self.split)
        if dataset is None:
            fallback = "testset" if self.split == "trainset" else "trainset"
            dataset = data.get(fallback)
            if dataset is not None:
                logger.info(f"{self.split} missing in {pt_path.parent}; using {fallback}")
        if dataset is None:
            logger.error(f"Neither trainset nor testset found in {pt_path}")
            self._pools[cache_key] = []
            return self._pools[cache_key]

        images: List[torch.Tensor] = []
        for item in dataset:
            img = item[0] if isinstance(item, (tuple, list)) else item
            images.append(self._prepare_image(img))

        self._pools[cache_key] = images
        return images

    def _sample_set_size(self, set_size: Union[int, Tuple[int, int], None] = None) -> int:
        value = self.set_size if set_size is None else set_size
        if isinstance(value, int):
            return value
        lo, hi = value
        return random.randint(lo, hi)

    def sample_image_set(
        self,
        root_dir: str,
        *,
        device: torch.device | None = None,
        set_size: Union[int, Tuple[int, int], None] = None,
    ) -> torch.Tensor:
        images = self.load_image_pool(root_dir)
        if not images:
            raise RuntimeError(f"No images available for zoo root {root_dir}")

        chosen = random.choices(images, k=self._sample_set_size(set_size))
        if self.transform is not None:
            chosen = [self.transform(img) for img in chosen]

        stacked = torch.stack(chosen, dim=0)
        if device is not None:
            stacked = stacked.to(device)
        return stacked


class ImageSetDataset(Dataset):
    """Dataset of image sets for dataset-identity classification.

    Args:
        zoo_roots_and_labels: List of ``(root_dir, label)`` pairs.
            ``root_dir`` is the path that contains ``dataset.pt``.
        split: Which key to read from ``dataset.pt`` (``"trainset"`` or
            ``"testset"``).  Falls back to the other key if the requested
            one is absent.
        set_size: Fixed set size (int) **or** an inclusive ``(min, max)``
            tuple for random sizing.  Default: ``16``.
        num_sets_per_zoo: How many sets to expose per zoo per epoch.
            Default: ``500``.
        target_image_size: Target spatial size as ``(height, width)``.
        target_channels: Target number of channels for every image.
        transform: Optional callable applied to each individual image
            *before* stacking into a set.  Useful for augmentation.
    """

    def __init__(
        self,
        zoo_roots_and_labels: List[Tuple[str, int]],
        split: str = "trainset",
        set_size: Union[int, Tuple[int, int]] = 16,
        num_sets_per_zoo: int = 500,
        target_image_size: Tuple[int, int] = (32, 32),
        target_channels: int = 3,
        transform=None,
    ):
        super().__init__()
        self._sampler = DatasetPtImageSetSampler(
            split=split,
            set_size=set_size,
            target_image_size=target_image_size,
            target_channels=target_channels,
            transform=transform,
        )
        self.set_size = set_size
        self.num_sets_per_zoo = num_sets_per_zoo
        self.target_hw = tuple(target_image_size)
        self.target_channels = target_channels
        self.transform = transform
        self.image_shape = self._sampler.image_shape

        # Load all zoo image pools
        self._pools: List[Tuple[List[torch.Tensor], int]] = []
        for root_dir, label in zoo_roots_and_labels:
            images = self._sampler.load_image_pool(root_dir)
            if len(images) == 0:
                logger.warning(f"No images loaded from {root_dir} (split={split}); skipping.")
                continue
            self._pools.append((images, label))
            logger.info(f"Loaded {len(images)} images from {root_dir} (label={label})")

        if not self._pools:
            raise RuntimeError("No valid zoo root directories could be loaded.")

        # Build an index: each entry is (pool_idx, dummy) for __len__
        self._index: List[int] = []
        for pool_idx in range(len(self._pools)):
            self._index.extend([pool_idx] * num_sets_per_zoo)

        logger.info(f"Canonical image shape set to C,H,W={self.image_shape}")

    # ------------------------------------------------------------------
    def _normalize_layout(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 2:
            return img.unsqueeze(0)

        if img.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {tuple(img.shape)}")

        if img.shape[0] in (1, 3, 4):
            return img
        if img.shape[-1] in (1, 3, 4):
            return img.permute(2, 0, 1)

        raise ValueError(f"Could not infer channel dimension for image shape {tuple(img.shape)}")

    def _convert_channels(self, img: torch.Tensor) -> torch.Tensor:
        current_channels = img.shape[0]
        if current_channels == self.target_channels:
            return img
        if self.target_channels == 1:
            return img.mean(dim=0, keepdim=True)
        if current_channels == 1:
            return img.repeat(self.target_channels, 1, 1)
        if current_channels > self.target_channels:
            return img[:self.target_channels]

        repeats = (self.target_channels + current_channels - 1) // current_channels
        return img.repeat(repeats, 1, 1)[:self.target_channels]

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        if (img.shape[-2], img.shape[-1]) == self.target_hw:
            return img
        return F.interpolate(
            img.unsqueeze(0),
            size=self.target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _prepare_image(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        img = self._normalize_layout(img).float()
        img = self._convert_channels(img)
        img = self._resize_image(img)
        return img.contiguous()

    def _load_images(self, root_dir: str, preferred_split: str) -> List[torch.Tensor]:
        """Load all images from ``dataset.pt`` inside *root_dir*."""
        pt_path = Path(root_dir) / "dataset.pt"
        if not pt_path.exists():
            logger.error(f"dataset.pt not found at {pt_path}")
            return []
        try:
            _register_pickle_compat_types()
            data = torch.load(str(pt_path), weights_only=False)
        except Exception as exc:
            logger.error(f"Failed to torch.load {pt_path}: {exc}")
            return []

        # Prefer requested split, fall back to the other
        dataset = data.get(preferred_split)
        if dataset is None:
            fallback = "testset" if preferred_split == "trainset" else "trainset"
            dataset = data.get(fallback)
            if dataset is not None:
                logger.info(f"{preferred_split} missing in {root_dir}; using {fallback}")
        if dataset is None:
            logger.error(f"Neither trainset nor testset found in {pt_path}")
            return []

        images: List[torch.Tensor] = []
        for item in dataset:
            img = item[0] if isinstance(item, (tuple, list)) else item
            images.append(self._prepare_image(img))
        return images

    # ------------------------------------------------------------------
    def _sample_set_size(self) -> int:
        if isinstance(self.set_size, int):
            return self.set_size
        lo, hi = self.set_size
        return random.randint(lo, hi)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pool_idx = self._index[idx]
        images, label = self._pools[pool_idx]
        n = self._sampler._sample_set_size()
        # Sample with replacement
        chosen = random.choices(images, k=n)
        if self.transform is not None:
            chosen = [self.transform(img) for img in chosen]
        return torch.stack(chosen, dim=0), label
