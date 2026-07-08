import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

from typing_extensions import Literal


@dataclass
class AugmentationConfig:
    permutation_spec: str
    num_permutations: int
    align: bool


@dataclass
class ZooDatasetConfig:
    root_dir: str
    epoch_idx: List[int]
    drop_nan: bool
    max_absolute_weight: Optional[float]
    max_checkpoints: Optional[int]
    augmentations: Optional[AugmentationConfig]
    name: Optional[str] = None


@dataclass
class TokenizerConfig:
    tokenizer_class: Literal["sparse", "dense"]
    mode: Literal["full_model", "weights_only"]
    tokensize: int
    device: str
    padding: Literal["zero", "mean", "gaussian", "reflect", "replicate", "circular"] = "zero"


@dataclass
class CachedWindowedDatasetConfig:
    window_size: int
    num_windows_per_model: int | Literal["auto"]
    pad_windows: bool
    window_distribution_strategy: Literal["consecutive", "distributed"]
    allow_overlapping_windows: bool
    cache_dir: Optional[str]
    pad_to_max_length: bool = False
    include_zoo_metadata: bool = False
    data_config_for_hash: dict[str, Any] | None = None
    num_windows_per_layer: int | None = None
    cache_mode: Literal["per_model", "per_window"] = "per_window"
    global_cache: bool = True
    read_only: bool = False

@dataclass
class SplitConfig:
    train: float | None = None
    val: float | None = None
    test: float | None = None
    train_datasets: List[str] | None = None
    val_datasets: List[str] | None = None
    test_datasets: List[str] | None = None


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    prefetch_factor: int


_T = TypeVar("_T")

import logging

logger = logging.getLogger(__name__)


def from_dict(cls: Type[_T], d: Dict[str, Any], **overrides: Any) -> _T:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    names = {f.name for f in dataclasses.fields(cls)}
    all_keys = {**d, **overrides}
    ignored_keys = [k for k in all_keys if k not in names]
    if ignored_keys:
        logger.debug("Ignoring unknown keys for %s: %s", cls.__name__, ignored_keys)
    return cls(**{k: v for k, v in all_keys.items() if k in names})



@dataclass
class DatasetFactoryConfig:
    checkpoints_dataset: ZooDatasetConfig | List[ZooDatasetConfig]
    tokenizer: TokenizerConfig
    tokens_dataset: CachedWindowedDatasetConfig
    split: SplitConfig
    dataloader: DataLoaderConfig
    splitter: str = "RandomSplitter"
