import logging
from collections.abc import Sequence

from torch.utils.data import DataLoader, Subset

from sane.data import datasets
from sane.data.checkpoint.checkpoint import Checkpoint
from sane.data.checkpoint.checkpoint_augmentation import (
    CheckpointAligner,
    CheckpointAugmentation,
    CheckpointAugmentationPipeline,
    PermutationAugmentation,
)
from sane.data.datasets import (CheckpointsDataset, CombinedCheckpointsDataset, ZooDataset)
from sane.data.splitter import DatasetNameSplitter, RandomSplitter
from sane.data.tokenizers import DenseTokenizer, SparseTokenizer
from sane.utils import git_re_basin

from sane.data.config_schema import (
    AugmentationConfig,
    CachedWindowedDatasetConfig,
    DataLoaderConfig,
    DatasetFactoryConfig,
    SplitConfig,
    TokenizerConfig,
    ZooDatasetConfig,
)

logger = logging.getLogger(__name__)


def _ref_checkpoint(dataset: CheckpointsDataset | Subset[CheckpointsDataset]) -> Checkpoint:
    reference_checkpoint = dataset[0]

    if isinstance(reference_checkpoint, Checkpoint):
        return reference_checkpoint

    return reference_checkpoint[0]


class DatasetFactory:
    def build(self, config: DatasetFactoryConfig) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        checkpoints_dataset = self._load_checkpoints_dataset(config.checkpoints_dataset)

        # We split checkpoints and not windows. Benefit: Windows from a single checkpoint will not be distributed across multiple splits. Downside: If models are not of equal size, split ratios cannot be guararanteed.
        # Further consideration: If multiple epochs are used, checkpoints of the same model in similar epochs may be distributed across splits.
        splits = self._split_checkpoints(config.split, checkpoints_dataset, config.splitter)

        # Tokenizer is built as part of the tokens_dataset, as it may get different reference_statedicts.
        tokens_datasets = [
            self._build_tokens_dataset(
                config.tokens_dataset,
                config.tokenizer,
                ds,
                split_name,
                checkpoints_dataset.name,
            )
            for split_name, ds in splits
            if ds is not None
        ]

        loaders = self._make_dataloaders(config.dataloader, tokens_datasets)

        # print stats
        split_sizes = {name: len(ds) for name, ds in splits if ds is not None}
        tok = config.tokenizer
        ds_cfg = config.tokens_dataset
        dl_cfg = config.dataloader
        logger.info(
            "Built dataloaders\n"
            "  splits:     %s\n"
            "  tokenizer:  class=%s, mode=%s, tokensize=%d, padding=%s \n"
            "  windows:    size=%d, per_model=%s, overlapping=%s, strategy=%s\n"
            "  dataloader: batch=%d, workers=%d, drop_last=%s",
            split_sizes,
            tok.tokenizer_class, tok.mode, tok.tokensize, tok.padding,
            ds_cfg.window_size, ds_cfg.num_windows_per_model, ds_cfg.allow_overlapping_windows, ds_cfg.window_distribution_strategy,
            dl_cfg.batch_size, dl_cfg.num_workers, dl_cfg.drop_last,
        )
        return loaders

    def _load_checkpoints_dataset(self, config: ZooDatasetConfig | Sequence[ZooDatasetConfig]) -> CheckpointsDataset:
        if isinstance(config, ZooDatasetConfig):
            return self._load_single_checkpoints_dataset(config)
        return CombinedCheckpointsDataset(datasets=[self._load_single_checkpoints_dataset(c) for c in config])

    def _load_single_checkpoints_dataset(self, config: ZooDatasetConfig) -> CheckpointsDataset:
        dataset = ZooDataset(
            root_dir=config.root_dir,
            epoch_idx=config.epoch_idx,
            drop_nan=config.drop_nan,
            max_absolute_weight=config.max_absolute_weight,
            max_checkpoints=config.max_checkpoints,
        )
        dataset.name = config.name or dataset.name

        if config.augmentations is not None:
            try:
                dataset.augmentations = self._load_augmentations(config.augmentations, reference_checkpoint=_ref_checkpoint(dataset))
            except Exception as e:
                raise ValueError("Cannot load augmentations.") from e

        return dataset

    def _load_augmentations(self, config: AugmentationConfig, reference_checkpoint: Checkpoint) -> CheckpointAugmentation | None:

        permutation_spec = getattr(git_re_basin, config.permutation_spec)()
        augmentations = []

        if config.align:
            augmentations.append(CheckpointAligner(permutation_spec=permutation_spec, reference_checkpoint=reference_checkpoint))
        if config.num_permutations > 1:
            augmentations.append(PermutationAugmentation(permutation_spec=permutation_spec, num_permutations=config.num_permutations))

        if not augmentations:
            return None
        return CheckpointAugmentationPipeline(augmentations=augmentations)

    def _split_checkpoints(
        self, config: SplitConfig, dataset: CheckpointsDataset, splitter_name: str = "RandomSplitter"
    ) -> tuple[
        tuple[str, Subset[CheckpointsDataset]],
        tuple[str, Subset[CheckpointsDataset] | None],
        tuple[str, Subset[CheckpointsDataset] | None],
    ]:
        if splitter_name == "RandomSplitter":
            if config.train is None:
                raise ValueError("Train split must be specified.")

            active_splits = [s for s in [config.train, config.val, config.test] if s is not None and s > 0]
            active_names = [
                name
                for name, s in [
                    ("train", config.train),
                    ("val", config.val),
                    ("test", config.test),
                ]
                if s is not None and s > 0
            ]
            splits = RandomSplitter().split(dataset, active_splits)
            split_map = dict(zip(active_names, splits))
            full_split_map = (dict(zip(["train", "val", "test"], [None, None, None])) | split_map)
            return tuple((k, v) for k, v in full_split_map.items())  # type: ignore

        if splitter_name == "DatasetNameSplitter":
            train_subset, val_subset, test_subset = DatasetNameSplitter().split(
                dataset,
                {
                    "train": config.train_datasets,
                    "val": config.val_datasets,
                    "test": config.test_datasets,
                },
            )
            return (("train", train_subset), ("val", val_subset), ("test", test_subset))

        raise ValueError(f"Unknown splitter {splitter_name!r}")

    def _build_tokens_dataset(
        self,
        config: CachedWindowedDatasetConfig,
        tokenizer_config: TokenizerConfig,
        subset: Subset[CheckpointsDataset],
        split_name: str,
        dataset_name: str,
    ):
        checkpoints_dataset = subset
        tokenizer = self._build_tokenizer(tokenizer_config, _ref_checkpoint(checkpoints_dataset))

        return datasets.CachedWindowedDataset(
            checkpoints_dataset=checkpoints_dataset,
            tokenizer=tokenizer,
            split_name=split_name,
            dataset_name=dataset_name,
            window_size=config.window_size,
            num_windows_per_model=config.num_windows_per_model,
            num_windows_per_layer=config.num_windows_per_layer,
            pad_windows=config.pad_windows,
            pad_to_max_length=config.pad_to_max_length,
            allow_overlapping_windows=config.allow_overlapping_windows,
            include_zoo_metadata=config.include_zoo_metadata,
            window_distribution_strategy=config.window_distribution_strategy,
            cache_dir=config.cache_dir,
            cache_mode=config.cache_mode,
            global_cache=config.global_cache,
            read_only=config.read_only,
            data_config_for_hash= config.data_config_for_hash,
        )

    def _build_tokenizer(self, config: TokenizerConfig, reference_checkpoint: Checkpoint):
        cls = {"dense": DenseTokenizer, "sparse": SparseTokenizer}[config.tokenizer_class]
        return cls(
            mode=config.mode,
            tokensize=config.tokensize,
            device=config.device,
            reference_statedict=reference_checkpoint.model.state_dict(),
            padding=config.padding,
        )

    def _make_dataloaders(
        self,
        config: DataLoaderConfig,
        tokens_datasets: list[datasets.CachedWindowedDataset],
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:

        kwargs: dict = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": config.drop_last,
        }
        if config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = config.prefetch_factor

        shuffles = [True, False, False]
        loaders = {ds.split_name: DataLoader(ds, shuffle=s, **kwargs) for ds, s in zip(tokens_datasets, shuffles)}

        return loaders.get("train"), loaders.get("val", None), loaders.get("test", None)  # type: ignore
