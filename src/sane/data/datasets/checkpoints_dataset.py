from torch.utils.data import Dataset
from typing import Optional, List, Sequence

from sane.data.checkpoint import Checkpoint
from sane.data.checkpoint.checkpoint_augmentation import CheckpointAugmentation

class CheckpointsDataset(Dataset[Checkpoint | Sequence[Checkpoint]]):
    """The base dataset class for datasets of checkpoints.

    This base class handles a list of checkpoints, where each checkpoint is an instance of the `Checkpoint` class.
    It can be extended to create specific datasets that handle checkpoints from different sources, such as model zoos or Hugging Face."""

    def __init__(self, checkpoints: List[Checkpoint] | None = None, augmentations: Optional[CheckpointAugmentation] = None) -> None:
        """
        Args:
            checkpoints (Optional[List[Checkpoint]]): A list of checkpoints to initialize the dataset with. Defaults to None.
            augmentations (Optional[CheckpointAugmentation]): A function/transform to apply to the dataset. Default: None
        """
        super(CheckpointsDataset, self).__init__()
        self.name = self.__class__.__name__
        self.checkpoints = checkpoints
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.checkpoints) if self.checkpoints else 0

    def __getitem__(self, index: int) -> Checkpoint | Sequence[Checkpoint]:
        if self.checkpoints is None:
            raise IndexError("Dataset is empty.")
        checkpoint = self.checkpoints[index]
        # If we use augmentations, we return a list of augmented checkpoints
        return self.augmentations(checkpoint) if self.augmentations is not None else checkpoint
