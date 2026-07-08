from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sane.data.checkpoint.checkpoint import Checkpoint
from sane.data.datasets.checkpoints_dataset import CheckpointsDataset
from sane.data.datasets.combined_checkpoints_dataset import CombinedCheckpointsDataset
from sane.data.datasets.windowed_dataset import WindowedDataset


class _StubCheckpointsDataset(CheckpointsDataset):
    def __init__(self, root_dir: str, size: int = 2) -> None:
        checkpoints = [
            Checkpoint(model=nn.Linear(2, 2), metadata={"idx": idx})
            for idx in range(size)
        ]
        super().__init__(checkpoints=checkpoints)
        self.root_dir = root_dir


class _StubTokenizer:
    tokensize = 4
    mode = "full_model"

    def tokenize(self, state_dict):
        del state_dict
        tokens = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        position = torch.stack(
            [
                torch.arange(3, dtype=torch.long),
                torch.zeros(3, dtype=torch.long),
            ],
            dim=-1,
        )
        return tokens, mask, position


def test_windowed_dataset_includes_zoo_metadata_for_subset_of_combined_dataset(tmp_path: Path):
    zoo_a = str(tmp_path / "zoo_a")
    zoo_b = str(tmp_path / "zoo_b")

    combined = CombinedCheckpointsDataset(
        [
            _StubCheckpointsDataset(zoo_a, size=2),
            _StubCheckpointsDataset(zoo_b, size=2),
        ]
    )
    subset = Subset(combined, [1, 2])
    dataset = WindowedDataset(
        checkpoints_dataset=subset,
        tokenizer=_StubTokenizer(),
        window_size=3,
        num_windows_per_model=1,
        include_zoo_metadata=True,
    )

    sample = dataset[0]
    assert sample["zoo_root_dir"] == zoo_a
    assert sample["checkpoint_idx"].item() == 0

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert tuple(batch["tokens"].shape) == (2, 3, 4)
    assert tuple(batch["position"].shape) == (2, 3, 2)
    assert batch["checkpoint_idx"].tolist() == [0, 1]
    assert list(batch["zoo_root_dir"]) == [zoo_a, zoo_b]
