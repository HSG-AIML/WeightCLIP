"""
Test multiprocessing behavior of CachedWindowedDataset (per_window mode).

With preload-at-init, all windows are written to disk before any DataLoader
worker is spawned, so workers only ever read from cache — no lazy generation
races can occur.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from sane.data.datasets.cached_windowed_dataset import CachedWindowedDataset
from sane.data.checkpoint import Checkpoint
from sane.data.datasets import CheckpointsDataset
from sane.data.tokenizers.dense import DenseTokenizer


def _make_models(n: int):
    return [
        torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
        for _ in range(n)
    ]


def _make_dataset(checkpoints_dataset, tmp_path, **kwargs) -> CachedWindowedDataset:
    tokenizer = DenseTokenizer(tokensize=32, mode="layer_wise")
    return CachedWindowedDataset(
        checkpoints_dataset,
        tokenizer,
        cache_mode="per_window",
        cache_dir=tmp_path,
        window_size=25,
        num_windows_per_model=2,
        dataset_name="test_model",
        split_name="train",
        **kwargs,
    )


class TestMultiprocessingCache:

    def test_multiprocessing_dataloader(self, tmp_path):
        """Dataset must survive DataLoader with num_workers > 0.

        Because the cache is fully written before the DataLoader is created,
        worker processes only read from disk — no generation races can occur.
        """
        checkpoints = [Checkpoint(m) for m in _make_models(5)]
        checkpoints_dataset = CheckpointsDataset(checkpoints)

        dataset = _make_dataset(checkpoints_dataset, tmp_path)

        dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

        batches = list(dataloader)

        total_windows = len(checkpoints) * 2  # 5 * 2
        total_samples = sum(b[0].shape[0] for b in batches)
        assert total_samples == total_windows

        for tokens, mask, position in batches:
            assert tokens.shape[-1] == 32  # tokensize

    def test_second_instance_loads_metadata(self, tmp_path):
        """A new dataset instance pointing at the same cache dir must read
        the existing metadata and skip regeneration entirely.
        """
        checkpoints = [Checkpoint(m) for m in _make_models(3)]
        checkpoints_dataset = CheckpointsDataset(checkpoints)

        # First instance — preloads everything
        dataset1 = _make_dataset(checkpoints_dataset, tmp_path)
        assert set(range(3)) == dataset1._cached_checkpoints

        # Second instance — should reload metadata and have all checkpoints present
        dataset2 = _make_dataset(checkpoints_dataset, tmp_path)
        assert set(range(3)) == dataset2._cached_checkpoints

        # Data must be consistent between instances
        assert torch.equal(dataset1[0][0], dataset2[0][0])

    def test_concurrent_instances_return_same_data(self, tmp_path):
        """Two instances sharing a cache directory must return identical tensors."""
        checkpoints = [Checkpoint(m) for m in _make_models(2)]
        checkpoints_dataset = CheckpointsDataset(checkpoints)

        ds1 = _make_dataset(checkpoints_dataset, tmp_path)
        ds2 = _make_dataset(checkpoints_dataset, tmp_path)

        for idx in range(len(ds1)):
            r1 = ds1[idx]
            r2 = ds2[idx]
            assert torch.equal(r1[0], r2[0])
            assert torch.equal(r1[1], r2[1])
            assert torch.equal(r1[2], r2[2])

    def test_all_windows_cached_before_dataloader_created(self, tmp_path):
        """After __init__, all window cache files must already exist on disk."""
        checkpoints = [Checkpoint(m) for m in _make_models(3)]
        checkpoints_dataset = CheckpointsDataset(checkpoints)

        dataset = _make_dataset(checkpoints_dataset, tmp_path)

        # Every expected window key must be present in the cache
        for checkpoint_idx in range(len(checkpoints)):
            for window_idx in range(dataset._windows_per_checkpoint):
                key = dataset._get_window_cache_key(checkpoint_idx, window_idx)
                assert dataset._disk_cache.contains(key), (
                    f"Window {window_idx} of checkpoint {checkpoint_idx} missing after init"
                )