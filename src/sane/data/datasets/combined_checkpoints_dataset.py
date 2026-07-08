from typing import List, Sequence
from sane.data.datasets.checkpoints_dataset import CheckpointsDataset, Checkpoint


class CombinedCheckpointsDataset(CheckpointsDataset):
    """Combines multiple CheckpointsDataset instances into a single dataset.

    This dataset allows combining checkpoints from multiple sources (e.g., different model zoos)
    into a single unified dataset. The datasets are concatenated sequentially, meaning all
    checkpoints from the first dataset come first, then all from the second, and so on.

    Each underlying dataset can have its own augmentations, which are preserved when accessing
    checkpoints through the combined dataset.

    Args:
        datasets (List[CheckpointsDataset]): A list of CheckpointsDataset instances to combine.

    Returns:
        Each __getitem__ call returns a checkpoint (or list of checkpoints if augmentations are enabled).
        The checkpoint(s) will have `zoo_root_dir` populated when available.

    Example:
        >>> zoo1 = ZooDataset(root_dir="/path/to/zoo1", epoch_idx=[25])
        >>> zoo2 = ZooDataset(root_dir="/path/to/zoo2", epoch_idx=[25])
        >>> combined = CombinedCheckpointsDataset(datasets=[zoo1, zoo2])
        >>> len(combined)  # Total checkpoints from both zoos
        >>> checkpoint = combined[0]  # Get first checkpoint from zoo1
    """

    def __init__(self, datasets: List[CheckpointsDataset]) -> None:
        """Initialize the combined dataset.

        Args:
            datasets (List[CheckpointsDataset]): A list of CheckpointsDataset instances to combine.

        Raises:
            ValueError: If the datasets list is empty or contains non-CheckpointsDataset objects.
        """
        if not datasets:
            raise ValueError("Cannot create CombinedCheckpointsDataset with an empty list of datasets.")

        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, CheckpointsDataset):
                raise ValueError(
                    f"All datasets must be instances of CheckpointsDataset, "
                    f"but dataset at index {i} is of type {type(dataset).__name__}."
                )

        # Don't call super().__init__() with checkpoints/augmentations since we manage them differently
        # Instead, directly initialize as a Dataset
        super(CheckpointsDataset, self).__init__()

        self.name = self.__class__.__name__
        self.datasets: List[CheckpointsDataset] = datasets
        self._cumulative_sizes = self._compute_cumulative_sizes()

        # Note: We don't set self.checkpoints or self.augmentations because each underlying
        # dataset manages its own checkpoints and augmentations

    def _compute_cumulative_sizes(self) -> List[int]:
        """Compute cumulative sizes for efficient index mapping.

        Returns:
            List[int]: Cumulative sizes where cumulative_sizes[i] is the total number of
                      checkpoints in datasets[0:i+1].
        """
        cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __len__(self) -> int:
        """Return the total number of checkpoints across all datasets.

        Returns:
            int: Total number of checkpoints.
        """
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> Checkpoint | Sequence[Checkpoint]:
        """Get a checkpoint from the combined dataset.

        Args:
            index (int): Index of the checkpoint to retrieve.

        Returns:
            Union[Checkpoint, List[Checkpoint]]: The checkpoint at the specified index.
                If the underlying dataset uses augmentations, this may return a list of checkpoints.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        # Find which dataset contains this index
        dataset_idx = 0
        for i, cumulative_size in enumerate(self._cumulative_sizes):
            if index < cumulative_size:
                dataset_idx = i
                break

        # Compute the local index within the target dataset
        if dataset_idx == 0:
            local_index = index
        else:
            local_index = index - self._cumulative_sizes[dataset_idx - 1]

        # Retrieve the checkpoint from the appropriate dataset
        checkpoint_or_list = self.datasets[dataset_idx][local_index]

        # Attach origin metadata when possible
        zoo_root_dir = getattr(self.datasets[dataset_idx], 'root_dir', None)
        if isinstance(checkpoint_or_list, list):
            for ckpt in checkpoint_or_list:
                if isinstance(ckpt, Checkpoint):
                    ckpt.zoo_root_dir = zoo_root_dir
        elif isinstance(checkpoint_or_list, Checkpoint):
            checkpoint_or_list.zoo_root_dir = zoo_root_dir

        return checkpoint_or_list