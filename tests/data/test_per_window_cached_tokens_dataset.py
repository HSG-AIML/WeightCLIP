import json
import pytest
import torch
from copy import deepcopy
from unittest.mock import Mock

from sane.data.datasets.cached_windowed_dataset import CachedWindowedDataset
from sane.data.checkpoint import Checkpoint
from sane.data.datasets import CheckpointsDataset
from sane.data.tokenizers.dense import DenseTokenizer
from sane.data.checkpoint.checkpoint_augmentation import CheckpointAugmentation


def create_mock_model():
    model = Mock()
    model.state_dict = Mock(return_value={
        'layer1.weight': torch.randn(32, 64),
        'layer2.weight': torch.randn(32, 64),
    })
    return model

def create_mock_checkpoints_dataset(num_checkpoints: int = 2):
    checkpoints_dataset = Mock()
    checkpoints_dataset.__len__ = Mock(return_value=num_checkpoints)
    checkpoints = []
    for _ in range(num_checkpoints):
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = create_mock_model()
        checkpoints.append(checkpoint)
    checkpoints_dataset.__getitem__ = Mock(side_effect=lambda idx: checkpoints[idx])
    return checkpoints_dataset


def create_sample_layer_data(num_tokens: int = 100, tokensize: int = 32, num_layers: int = 4):
    layer_data = []
    tokens_per_layer = num_tokens // num_layers
    global_token_idx = 0
    for layer_idx in range(num_layers):
        current_layer_tokens = (
            num_tokens - global_token_idx
            if layer_idx == num_layers - 1
            else tokens_per_layer
        )
        layer_tokens = torch.randn(current_layer_tokens, tokensize)
        layer_mask = torch.ones(current_layer_tokens, tokensize, dtype=torch.bool)
        layer_position = torch.zeros(current_layer_tokens, 3, dtype=torch.long)
        layer_position[:, 0] = torch.arange(global_token_idx, global_token_idx + current_layer_tokens)
        layer_position[:, 1] = layer_idx
        layer_position[:, 2] = torch.arange(current_layer_tokens)
        layer_data.append((layer_tokens, layer_mask, layer_position))
        global_token_idx += current_layer_tokens
    return layer_data


def make_dataset(checkpoints_dataset, tokenizer, tmp_path, **kwargs):
    """Convenience wrapper — always per_window cache mode."""
    return CachedWindowedDataset(
        checkpoints_dataset,
        tokenizer,
        cache_mode="per_window",
        cache_dir=tmp_path,
        **kwargs,
    )


class TestCachedWindowedDatasetPerWindow:

    def test_init_basic(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset.checkpoints_dataset is checkpoints_dataset
        assert dataset.tokenizer is tokenizer
        assert dataset.window_size == 50
        assert dataset.num_windows_per_model == 2
        assert dataset.num_windows_per_layer is None
        assert dataset._cache_mode == "per_window"

    def test_cache_directory_uses_window_subdir(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        # Structure: {dataset}_{hash}/window/{split}/
        parts = dataset._disk_cache.cache_dir.relative_to(tmp_path).parts
        assert len(parts) == 3
        assert parts[0].startswith("default_")
        assert parts[1] == "window"
        assert parts[2] == "default"

    def test_cache_directory_with_dataset_and_split_names(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            dataset_name="mnist_cnn", split_name="train",
        )

        parts = dataset._disk_cache.cache_dir.relative_to(tmp_path).parts
        assert parts[0].startswith("mnist_cnn_")
        assert parts[1] == "window"
        assert parts[2] == "train"

    def test_len_fixed_windows(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=3)
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert len(dataset) == 3 * 2

    def test_map_index(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=3)
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset._map_index(0) == (0, 0)
        assert dataset._map_index(1) == (0, 1)
        assert dataset._map_index(2) == (1, 0)
        assert dataset._map_index(3) == (1, 1)
        assert dataset._map_index(4) == (2, 0)
        assert dataset._map_index(5) == (2, 1)

        with pytest.raises(IndexError):
            dataset._map_index(6)

    def test_window_cache_key_format(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset._get_window_cache_key(0, 0) == "checkpoint_0_window_0"
        assert dataset._get_window_cache_key(1, 3) == "checkpoint_1_window_3"

    def test_metadata_path(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset._get_metadata_path().name == "cache_metadata.json"

    def test_windows_cached_individually(self, tmp_path):
        """Each window should be its own cache file, not all windows in one entry."""
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
        )

        # Preload runs at init, so windows should already be on disk
        cache_dir = dataset._disk_cache.cache_dir
        window_files = list(cache_dir.glob("cache_checkpoint_0_window_*.pt"))
        assert len(window_files) > 0

    def test_getitem_returns_single_window_tuple(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        result = dataset[0]

        assert isinstance(result, tuple)
        assert len(result) == 3
        tokens, mask, position = result
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(position, torch.Tensor)
        assert tokens.shape[1] == 32
        assert mask.shape[1] == 32
        assert position.shape[1] == 3
        assert tokens.shape[0] <= 50

    def test_metadata_written_after_preload(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
        )

        # All checkpoints preloaded at init
        assert 0 in dataset._cached_checkpoints
        assert 1 in dataset._cached_checkpoints

        metadata_path = dataset._get_metadata_path()
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "checkpoints" in metadata
        assert "config" in metadata
        assert "0" in metadata["checkpoints"]
        assert "num_windows" in metadata["checkpoints"]["0"]
        assert "cached_at" in metadata["checkpoints"]["0"]

    def test_metadata_recovery_skips_retokenization(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            dataset_name="mnist_cnn", split_name="train",
        )

        baseline_tokenizer = Mock()
        baseline_tokenizer.mode = "layer_wise"
        baseline_tokenizer.tokensize = 32
        baseline_tokenizer.tokenize = Mock(return_value=layer_data)
        _ = make_dataset(
            checkpoints_dataset, baseline_tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            dataset_name="mnist_cnn", split_name="train",
        )
        baseline_calls = baseline_tokenizer.tokenize.call_count

        metadata_path = dataset._get_metadata_path()
        metadata_path.write_text(metadata_path.read_text() + "\n{\"extra\": true}\n")

        reloaded_tokenizer = Mock()
        reloaded_tokenizer.mode = "layer_wise"
        reloaded_tokenizer.tokensize = 32
        reloaded_tokenizer.tokenize = Mock(return_value=layer_data)

        reloaded_dataset = make_dataset(
            checkpoints_dataset, reloaded_tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            dataset_name="mnist_cnn", split_name="train",
        )

        assert reloaded_dataset._cached_checkpoints == {0, 1}
        assert reloaded_tokenizer.tokenize.call_count == baseline_calls

    def test_no_retokenize_on_repeated_access(self, tmp_path):
        """After preload, repeated item access must not call tokenize again."""
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
        )

        call_count_after_init = tokenizer.tokenize.call_count

        for i in range(len(dataset)):
            _ = dataset[i]

        assert tokenizer.tokenize.call_count == call_count_after_init

    def test_clear_cache(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=2)
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=2,
        )

        assert 0 in dataset._cached_checkpoints
        assert dataset._get_metadata_path().exists()

        dataset.clear_cache()

        assert len(dataset._cached_checkpoints) == 0
        assert not dataset._get_metadata_path().exists()
        assert not dataset._disk_cache.contains("checkpoint_0_window_0")

    def test_metadata_loaded_by_new_instance(self, tmp_path):
        """A second dataset instance pointing at the same cache dir loads existing metadata."""
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        # First instance — preloads everything at init
        make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
        )

        tokenizer.tokenize.reset_mock()

        # Second instance — should load metadata and skip regeneration
        dataset2 = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
        )

        assert 0 in dataset2._cached_checkpoints
        # preload_cache skips everything already in _cached_checkpoints
        assert tokenizer.tokenize.call_count == 0

    def test_read_only_does_not_warm_or_write_cache(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        tokenizer.padding = "zero"
        tokenizer.tokenize = Mock(return_value=create_sample_layer_data(num_tokens=100, num_layers=4))

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            data_config_for_hash={"window_size": 25},
            read_only=True,
        )

        config_file = tmp_path / f"default_{dataset._cache_key_suffix}" / "data_config.yaml"
        assert tokenizer.tokenize.call_count == 0
        assert not dataset._get_metadata_path().exists()
        assert not config_file.exists()

        with pytest.raises(RuntimeError, match="read-only mode"):
            _ = dataset[0]

        assert tokenizer.tokenize.call_count == 0

    def test_read_only_uses_existing_cache_without_retokenizing(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        writer_tokenizer = Mock()
        writer_tokenizer.mode = "layer_wise"
        writer_tokenizer.tokensize = 32
        writer_tokenizer.padding = "zero"
        writer_tokenizer.tokenize = Mock(return_value=create_sample_layer_data(num_tokens=100, num_layers=4))

        _ = make_dataset(
            checkpoints_dataset, writer_tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            dataset_name="mnist_cnn", split_name="train",
        )

        reader_tokenizer = Mock()
        reader_tokenizer.mode = "layer_wise"
        reader_tokenizer.tokensize = 32
        reader_tokenizer.padding = "zero"
        reader_tokenizer.tokenize = Mock(side_effect=AssertionError("read-only cache should not tokenize"))

        dataset = make_dataset(
            checkpoints_dataset, reader_tokenizer, tmp_path,
            window_size=25, num_windows_per_model=4,
            dataset_name="mnist_cnn", split_name="train",
            read_only=True,
        )

        _ = dataset[0]

        assert reader_tokenizer.tokenize.call_count == 0


class TestCacheInvalidation:

    def test_cache_key_differs_on_augmentation_config_change(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        ds1 = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            data_config_for_hash={"augmentations": {"num_permutations": 10}},
        )
        ds2 = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            data_config_for_hash={"augmentations": {"num_permutations": 12}},
        )

        assert ds1._cache_key_suffix != ds2._cache_key_suffix

    def test_data_config_yaml_written(self, tmp_path):
        import yaml

        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)
        data_config = {"augmentations": {"num_permutations": 10}, "window_size": 50}

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            data_config_for_hash=data_config,
        )

        config_file = tmp_path / f"default_{dataset._cache_key_suffix}" / "data_config.yaml"
        assert config_file.exists()
        assert yaml.safe_load(config_file.read_text()) == data_config


class TestCachedWindowedDatasetPerWindowIntegration:

    def test_integration_with_real_tokenizer(self, tmp_path):
        model = create_mock_model()
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = model

        checkpoints_dataset = Mock()
        checkpoints_dataset.__len__ = Mock(return_value=2)
        checkpoints_dataset.__getitem__ = Mock(return_value=checkpoint)

        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=3,
            allow_overlapping_windows=True,
        )

        assert len(dataset) > 2

        for i in range(min(6, len(dataset))):
            result = dataset[i]
            assert isinstance(result, tuple)
            assert len(result) == 3
            tokens, mask, position = result
            assert tokens.shape[1] == 32
            assert mask.shape[1] == 32
            assert position.shape[1] == 3
            assert tokens.shape[0] > 0

    def test_deterministic_indexing(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=2,
        )

        w1 = dataset[0]
        w2 = dataset[0]

        assert torch.equal(w1[0], w2[0])
        assert torch.equal(w1[1], w2[1])
        assert torch.equal(w1[2], w2[2])

    def test_with_augmentations(self, tmp_path):

        class MockAugmentation(CheckpointAugmentation):
            def __init__(self, n_views=2):
                super().__init__()
                self.n_views = n_views

            def forward(self, x):
                return [deepcopy(x) for _ in range(self.n_views)]

        N_VIEWS = 2

        model = create_mock_model()
        checkpoint = Checkpoint(model)
        checkpoints_dataset = CheckpointsDataset([checkpoint], augmentations=MockAugmentation(n_views=N_VIEWS))
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=3,
            allow_overlapping_windows=True,
        )

        for i in range(min(3, len(dataset))):
            result = dataset[i]
            assert isinstance(result, tuple)
            assert len(result) == 3
            tokens, mask, position = result
            if tokens.dim() == 3:
                assert tokens.shape[0] == N_VIEWS
                assert mask.shape[0] == N_VIEWS
                assert position.shape[0] == N_VIEWS
            else:
                assert tokens.dim() == 2

    def test_pad_to_max_length_uses_window_size_for_single_view(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=1)
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=1000, num_windows_per_model=1,
            pad_windows=False, pad_to_max_length=True,
        )

        tokens, mask, position = dataset[0]
        assert dataset.max_sequence_length is not None
        assert tokens.shape[0] == dataset.max_sequence_length
        assert mask.shape[0] == dataset.max_sequence_length
        assert position.shape[0] == dataset.max_sequence_length
        valid_len = int(mask.any(dim=1).sum().item())
        assert valid_len < dataset.max_sequence_length
        assert torch.equal(tokens[valid_len:], torch.zeros_like(tokens[valid_len:]))
        assert not mask[valid_len:].any()
        assert torch.equal(
            position[valid_len:, 0],
            torch.arange(valid_len, dataset.max_sequence_length, dtype=position.dtype),
        )

    def test_pad_to_max_length_uses_window_size_for_augmented_views(self, tmp_path):

        class MockAugmentation(CheckpointAugmentation):
            def __init__(self, n_views=2):
                super().__init__()
                self.n_views = n_views

            def forward(self, x):
                return [deepcopy(x) for _ in range(self.n_views)]

        checkpoint = Checkpoint(create_mock_model())
        checkpoints_dataset = CheckpointsDataset([checkpoint], augmentations=MockAugmentation(n_views=2))
        tokenizer = DenseTokenizer(tokensize=32)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=1000, num_windows_per_model=1,
            pad_windows=False, pad_to_max_length=True,
        )

        tokens, mask, position = dataset[0]
        assert tokens.dim() == 3
        assert dataset.max_sequence_length is not None
        assert tokens.shape[1] == dataset.max_sequence_length
        assert mask.shape[1] == dataset.max_sequence_length
        assert position.shape[1] == dataset.max_sequence_length
        valid_len = int(mask[0].any(dim=1).sum().item())
        assert valid_len < dataset.max_sequence_length
        assert torch.equal(tokens[:, valid_len:], torch.zeros_like(tokens[:, valid_len:]))
        assert not mask[:, valid_len:].any()

    def test_disk_io_reads_only_one_window(self, tmp_path):
        """Accessing window N should only deserialise window N, not all windows."""
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=1)
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=1000, num_layers=10)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model="auto",
        )

        stats_before = dataset._disk_cache.get_stats()
        hits_before = stats_before["hits"]

        for i in range(min(10, len(dataset))):
            _ = dataset[i]

        stats_after = dataset._disk_cache.get_stats()
        assert stats_after["hits"] > hits_before