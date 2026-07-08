"""
Integration tests for CachedWindowedDataset with disk caching.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

from sane.data.checkpoint import Checkpoint
from sane.data.datasets import CachedWindowedDataset
from sane.data.datasets.checkpoints_dataset import CheckpointsDataset
from sane.data.tokenizers import SANETokenizer


class MockSANETokenizer(SANETokenizer):
    def __init__(self, tokensize=16, mode="layer_wise"):
        super().__init__(tokensize=tokensize, mode=mode)

    def flatten(self, statedict):
        return [v.flatten() for v in statedict.values()]

    def slice_by_layers(self, flattened):
        results = []
        for i, layer in enumerate(flattened):
            n = max(1, layer.shape[0] // self.tokensize)
            tokens = torch.randn(n, self.tokensize)
            mask = torch.ones(n, self.tokensize, dtype=torch.bool)
            position = torch.stack([
                torch.full((n,), i),
                torch.zeros(n, dtype=torch.long),
                torch.arange(n),
            ], dim=1)
            results.append((tokens, mask, position))
        return results

    def unslice(self, tokens_input, mask=None, position=None):
        return []

    def rebuild_state_dict(self, flattened_weights, reference_statedict=None):
        return {}


def _make_checkpoint():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    return Checkpoint(model=model, metadata={"test": True})


def _make_checkpoints_dataset(n=3):
    return CheckpointsDataset(checkpoints=[_make_checkpoint() for _ in range(n)])


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def checkpoints_dataset():
    return _make_checkpoints_dataset(n=3)


@pytest.fixture
def tokenizer():
    return MockSANETokenizer(tokensize=16, mode="layer_wise")


def _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir, **kwargs):
    defaults = dict(
        checkpoints_dataset=checkpoints_dataset,
        tokenizer=tokenizer,
        window_size=8,
        num_windows_per_model=2,
        cache_dir=temp_cache_dir,
    )
    defaults.update(kwargs)
    return CachedWindowedDataset(**defaults)


class TestCachedWindowedDatasetDiskCache:
    def test_disk_cache_enabled(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset = _make_dataset(
            checkpoints_dataset, tokenizer, temp_cache_dir
        )

        assert str(temp_cache_dir.resolve()) in str(dataset._disk_cache.cache_dir.resolve())
        assert len(dataset) > 0

        window = dataset[0]
        assert isinstance(window, tuple)
        assert len(window) == 3
        assert all(isinstance(t, torch.Tensor) for t in window)

    def test_cache_key_generation(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir)

        key0a = dataset._get_cache_key(0)
        key0b = dataset._get_cache_key(0)
        key1 = dataset._get_cache_key(1)

        assert key0a == key0b
        assert key0a != key1
        assert isinstance(key0a, str)

    def test_cache_persistence(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset1 = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir)
        window1 = dataset1[0]

        dataset2 = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir)
        window2 = dataset2[0]

        assert torch.equal(window1[0], window2[0])
        assert torch.equal(window1[1], window2[1])
        assert torch.equal(window1[2], window2[2])

    def test_cache_invalidation(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset1 = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir, window_size=8)
        dataset2 = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir, window_size=10)

        assert dataset1._disk_cache.cache_dir != dataset2._disk_cache.cache_dir

    def test_cache_info_and_management(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir)

        dataset[0]
        dataset[1]

        assert dataset._disk_cache.cache_dir.exists()

        dataset.clear_cache()
        dataset.preload_cache()

        assert isinstance(dataset[0], tuple)

    def test_preload_cache(self, temp_cache_dir, checkpoints_dataset, tokenizer):
        dataset = _make_dataset(checkpoints_dataset, tokenizer, temp_cache_dir)

        key0 = dataset._get_cache_key(0)
        key1 = dataset._get_cache_key(1)
        key2 = dataset._get_cache_key(2)

        # All checkpoints are preloaded on init
        assert dataset._disk_cache.contains(key0)
        assert dataset._disk_cache.contains(key1)
        assert dataset._disk_cache.contains(key2)

        dataset.clear_cache()
        dataset.preload_cache([0, 1])

        assert dataset._disk_cache.contains(key0)
        assert dataset._disk_cache.contains(key1)
        assert not dataset._disk_cache.contains(key2)

    def test_full_model_tokenizer_mode(self, temp_cache_dir, checkpoints_dataset):
        tok = MockSANETokenizer(tokensize=16, mode="full_model")
        dataset = _make_dataset(checkpoints_dataset, tok, temp_cache_dir)

        window = dataset[0]
        assert isinstance(window, tuple)
        assert len(window) == 3

    def test_large_dataset_disk_cache(self, temp_cache_dir, tokenizer):
        large_ds = _make_checkpoints_dataset(n=20)
        dataset = _make_dataset(large_ds, tokenizer, temp_cache_dir, window_size=6, num_windows_per_model=3)

        dataset_len = len(dataset)
        step = max(1, dataset_len // 10)
        accessed = []
        for i in range(0, min(dataset_len, 50), step):
            assert isinstance(dataset[i], tuple)
            accessed.append(i)

        for i in accessed[:5]:
            assert isinstance(dataset[i], tuple)

        stats = dataset._disk_cache.get_stats()
        assert stats["disk_cache_files"] > 0
        assert stats["hits"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])