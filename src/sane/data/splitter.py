from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence
from torch.utils.data import random_split, Subset

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset


class Splitter(ABC):
    """ The dataset splitter mutator base class. """
    def __call__(self, *args, **kwargs)-> List[Subset[CheckpointsDataset]]:
        return self.split(*args, **kwargs)

    @abstractmethod
    def split(self, dataset: CheckpointsDataset)-> List[Subset[CheckpointsDataset]]:
        pass

class RandomSplitter(Splitter):
    """ A mutator class for splitting a dataset using ratios.
    
    Using this splitter is, for most cases, a bad choice. Since the splitting is done at random, it can lead to leakage between train and test sets.
    For example, if the dataset contains checkpoints of the same model, the train and test sets may contain checkpoints of the same model.
    It can also be the case the multiple checkpoints trained with the same configuration but different seed are split into train and test sets, albeit not being significantly different.
    Use this mutator only if you are aware of the risks and limitations.

    Args:
        dataset (CheckpointsDataset): The dataset to split.
        ratios (Sequence[int | float]): The ratios to split the dataset into. The sum of the ratios must be equal to 1.
    """
    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        return random_split(dataset, ratios)


def _normalize_dataset_identifier(name: str) -> str:
    return name.lower().replace("_", "-")


def _logical_dataset_name_from_root(root_dir: str | None) -> str | None:
    if root_dir is None:
        return None
    path = Path(root_dir)
    if len(path.parts) >= 3:
        return _normalize_dataset_identifier(path.parent.parent.name)
    return _normalize_dataset_identifier(path.name)


class DatasetNameSplitter(Splitter):
    """Split a combined checkpoints dataset by underlying dataset identity.

    Names in the config can refer either to the checkpoint-dataset config name
    (for example ``cassava_leaf_cnn3``) or to the logical dataset name derived
    from the zoo root path (for example ``cassava-leaf``).
    Any dataset not explicitly assigned to ``val`` or ``test`` defaults to
    ``train``.
    """

    def _dataset_aliases(self, dataset: CheckpointsDataset, fallback_name: str) -> set[str]:
        aliases = {_normalize_dataset_identifier(fallback_name)}
        name = getattr(dataset, "name", None)
        if isinstance(name, str):
            aliases.add(_normalize_dataset_identifier(name))
        root_dir = getattr(dataset, "root_dir", None)
        logical_name = _logical_dataset_name_from_root(root_dir)
        if logical_name is not None:
            aliases.add(logical_name)
        return aliases

    def _dataset_spans(self, dataset: CheckpointsDataset):
        datasets = getattr(dataset, "datasets", None)
        cumulative_sizes = getattr(dataset, "_cumulative_sizes", None)
        if datasets is not None and cumulative_sizes is not None:
            start = 0
            for idx, (child, end) in enumerate(zip(datasets, cumulative_sizes)):
                yield child, start, end, f"dataset_{idx}"
                start = end
            return

        yield dataset, 0, len(dataset), getattr(dataset, "name", "dataset_0")

    def split(self, dataset: CheckpointsDataset, split_names: dict[str, Sequence[str] | None]) -> List[Subset[CheckpointsDataset] | None]:
        normalized = {split: {_normalize_dataset_identifier(name) for name in (names or [])} for split, names in split_names.items()}

        overlap = (
            (normalized["train"] & normalized["val"])
            | (normalized["train"] & normalized["test"])
            | (normalized["val"] & normalized["test"])
        )
        if overlap:
            raise ValueError(f"Dataset split names must be disjoint across train/val/test, got overlap: {sorted(overlap)}")

        split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        unmatched_aliases: list[tuple[str, set[str]]] = []
        for child, start, end, fallback_name in self._dataset_spans(dataset):
            aliases = self._dataset_aliases(child, fallback_name)
            matched = [split for split, names in normalized.items() if split != "train" and aliases & names]
            if len(matched) > 1:
                raise ValueError(f"Dataset {fallback_name} matched multiple splits via aliases {sorted(aliases)}: {matched}")

            if matched:
                target_split = matched[0]
            else:
                target_split = "train"
                unmatched_aliases.append((fallback_name, aliases))

            split_indices[target_split].extend(range(start, end))

        if not split_indices["train"]:
            raise ValueError("DatasetNameSplitter produced an empty train split.")

        return [
            Subset(dataset, split_indices["train"]),
            Subset(dataset, split_indices["val"]) if split_indices["val"] else None,
            Subset(dataset, split_indices["test"]) if split_indices["test"] else None,
        ]