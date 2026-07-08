import pytest
import torch
from pathlib import Path
from unittest.mock import Mock

from sane.data.datasets import CachedWindowedDataset
from sane.data.checkpoint import Checkpoint
from sane.data.tokenizers.dense import DenseTokenizer
from sane.data.cache import (
    list_all_cache_configs,
    print_cache_summary,
    clear_cache_config,
    clear_old_caches,
    get_cache_size
)


def create_mock_model():
    model = Mock()
    model.state_dict = Mock(return_value={
        'layer1.weight': torch.randn(32, 64),
        'layer2.weight': torch.randn(32, 64),
    })
    return model


class TestCacheConfigDirectories:

    def test_cache_directory_uses_config_hash(self, tmp_path):
        model = create_mock_model()
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = model

        checkpoints_dataset = Mock()
        checkpoints_dataset.__len__ = Mock(return_value=1)
        checkpoints_dataset.__getitem__ = Mock(return_value=checkpoint)

        tokenizer = DenseTokenizer(tokensize=32)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=50,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )

        _ = dataset[0]

        config_hash = dataset._cache_key_suffix
        expected_dir = tmp_path / f"default_{config_hash}" / "checkpoint" / "default"

        assert expected_dir.exists()
        assert expected_dir.is_dir()

        cache_files = list(expected_dir.glob("cache_*.pt"))
        assert len(cache_files) > 0

    def test_different_configs_create_different_directories(self, tmp_path):
        model = create_mock_model()
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = model

        checkpoints_dataset = Mock()
        checkpoints_dataset.__len__ = Mock(return_value=1)
        checkpoints_dataset.__getitem__ = Mock(return_value=checkpoint)

        tokenizer = DenseTokenizer(tokensize=32)

        dataset1 = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=50,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )
        _ = dataset1[0]
        config_hash1 = dataset1._cache_key_suffix

        dataset2 = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=100,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )
        _ = dataset2[0]
        config_hash2 = dataset2._cache_key_suffix

        assert config_hash1 != config_hash2

        assert (tmp_path / f"default_{config_hash1}" / "checkpoint" / "default").exists()
        assert (tmp_path / f"default_{config_hash2}" / "checkpoint" / "default").exists()

    def test_list_cache_configs(self, tmp_path):
        model = create_mock_model()
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = model

        checkpoints_dataset = Mock()
        checkpoints_dataset.__len__ = Mock(return_value=1)
        checkpoints_dataset.__getitem__ = Mock(return_value=checkpoint)

        tokenizer = DenseTokenizer(tokensize=32)

        dataset1 = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=50,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )
        _ = dataset1[0]

        dataset2 = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=100,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )
        _ = dataset2[0]

        configs = list_all_cache_configs(tmp_path)

        assert len(configs) == 2
        assert all(c["num_files"] > 0 for c in configs)
        for config in configs:
            assert "config_hash" in config
            assert "cache_dir" in config
            assert "num_files" in config
            assert "total_size_mb" in config
            assert "total_size_gb" in config


class TestCacheUtilityFunctions:

    def test_list_all_cache_configs(self, tmp_path):
        config1_dir = tmp_path / "config_hash_1"
        config1_dir.mkdir()
        (config1_dir / "cache_0.pt").write_text("test")

        config2_dir = tmp_path / "config_hash_2"
        config2_dir.mkdir()
        (config2_dir / "cache_0.pt").write_text("test")

        configs = list_all_cache_configs(tmp_path)

        assert len(configs) == 2
        assert all("config_hash" in c for c in configs)
        assert all("num_files" in c for c in configs)
        assert all("total_size_mb" in c for c in configs)

    def test_get_cache_size(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        (cache_dir / "cache_0.pt").write_bytes(b"x" * 1000)
        (cache_dir / "cache_1.pt").write_bytes(b"x" * 2000)

        size = get_cache_size(cache_dir)

        assert "bytes" in size
        assert "mb" in size
        assert "gb" in size
        assert size["bytes"] == 3000
        assert size["mb"] == pytest.approx(0.003, rel=0.01)

    def test_clear_cache_config(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        (cache_dir / "cache_0.pt").write_text("test")
        (cache_dir / "cache_1.pt").write_text("test")

        assert cache_dir.exists()

        result = clear_cache_config(cache_dir)

        assert result is True
        assert not cache_dir.exists()

    def test_clear_cache_config_dry_run(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        (cache_dir / "cache_0.pt").write_text("test")

        result = clear_cache_config(cache_dir, dry_run=True)

        assert result is True
        assert cache_dir.exists()

    def test_clear_old_caches(self, tmp_path):
        for i, size in enumerate([1000, 2000, 500]):
            cache_dir = tmp_path / f"config_{i}"
            cache_dir.mkdir()
            (cache_dir / "cache_0.pt").write_bytes(b"x" * size)

        removed = clear_old_caches(tmp_path, keep_latest_n=1)

        assert removed == 2
        configs = list_all_cache_configs(tmp_path)
        assert len(configs) == 1

    def test_clear_old_caches_dry_run(self, tmp_path):
        for i in range(2):
            cache_dir = tmp_path / f"config_{i}"
            cache_dir.mkdir()
            (cache_dir / "cache_0.pt").write_text("test")

        removed = clear_old_caches(tmp_path, keep_latest_n=1, dry_run=True)

        assert removed == 1
        configs = list_all_cache_configs(tmp_path)
        assert len(configs) == 2