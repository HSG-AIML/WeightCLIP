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
    for i in range(num_checkpoints):
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
    """Convenience wrapper — always per_model cache mode."""
    return CachedWindowedDataset(
        checkpoints_dataset,
        tokenizer,
        cache_mode="per_model",
        cache_dir=tmp_path,
        **kwargs,
    )


class TestCachedWindowedDatasetPerModel:

    def test_init_basic(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset.checkpoints_dataset is checkpoints_dataset
        assert dataset.tokenizer is tokenizer
        assert dataset.window_size == 50
        assert dataset.num_windows_per_model == 2
        assert dataset.num_windows_per_layer is None
        assert dataset._cache_mode == "per_model"

    def test_cache_directory_uses_checkpoint_subdir(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        # Structure: {dataset}_{hash}/checkpoint/{split}/
        parts = dataset._disk_cache.cache_dir.relative_to(tmp_path).parts
        assert len(parts) == 3
        assert parts[0].startswith("default_")
        assert parts[1] == "checkpoint"
        assert parts[2] == "default"

    def test_cache_directory_with_dataset_and_split_names(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            dataset_name="mnist_cnn", split_name="train",
        )

        parts = dataset._disk_cache.cache_dir.relative_to(tmp_path).parts
        assert parts[0].startswith("mnist_cnn_")
        assert parts[1] == "checkpoint"
        assert parts[2] == "train"

    def test_init_validation_warns_on_conflicting_window_params(self, tmp_path, caplog):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2, num_windows_per_layer=1,
        )

        assert any("per-layer windowing will take precedence" in r.message for r in caplog.records)

    def test_len_fixed_windows(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=3)
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert len(dataset) == 3 * 2

    def test_len_auto_windows(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=2)
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model="auto",
        )

        # 100 tokens / 25 window_size = 4 windows per checkpoint
        assert len(dataset) == 2 * 4

    def test_map_index(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=3)
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

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

    def test_cache_key_format(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        assert dataset._get_cache_key(0) == "checkpoint_0"
        assert dataset._get_cache_key(1) == "checkpoint_1"
        assert dataset._get_cache_key(0) == dataset._get_cache_key(0)

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

    def test_getitem_different_windows_same_checkpoint(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=3,
        )

        windows = [dataset[i] for i in range(3)]
        for w in windows:
            assert isinstance(w, tuple)
            tokens, _, _ = w
            assert tokens.shape[0] <= 25
            assert tokens.shape[1] == 32

    def test_getitem_different_checkpoints(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset(num_checkpoints=2)
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        tokenizer.tokenize = Mock(side_effect=lambda _: create_sample_layer_data(num_tokens=50, num_layers=2))

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=25, num_windows_per_model=1,
        )

        w0 = dataset[0]
        w1 = dataset[1]

        assert isinstance(w0, tuple)
        assert isinstance(w1, tuple)
        # one tokenize call per checkpoint (preload caches both at init)
        assert tokenizer.tokenize.call_count == 2

    def test_cache_preloaded_at_init(self, tmp_path):
        """All checkpoints should be cached after __init__ — no extra tokenize calls on access."""
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

        call_count_after_init = tokenizer.tokenize.call_count

        # All accesses should be cache hits — no new tokenize calls
        for i in range(len(dataset)):
            _ = dataset[i]

        assert tokenizer.tokenize.call_count == call_count_after_init

    def test_per_layer_windowing(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=100, num_layers=4)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=10, num_windows_per_layer=1,
        )

        result = dataset[0]
        assert isinstance(result, tuple)
        tokens, _, _ = result
        assert tokens.shape[0] <= 10
        assert tokens.shape[1] == 32

    def test_edge_case_short_sequence(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=10, num_layers=1)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
        )

        result = dataset[0]
        assert isinstance(result, tuple)
        tokens, _, _ = result
        assert tokens.shape[0] <= 10
        assert tokens.shape[1] == 32

    def test_edge_case_single_token(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = Mock()
        tokenizer.mode = "layer_wise"
        tokenizer.tokensize = 32
        layer_data = create_sample_layer_data(num_tokens=1, num_layers=1)
        tokenizer.tokenize = Mock(return_value=layer_data)

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=10, num_windows_per_model=1,
        )

        result = dataset[0]
        assert isinstance(result, tuple)
        tokens, _, _ = result
        assert tokens.shape[0] == 1
        assert tokens.shape[1] == 32

    def test_cache_key_invalidation_on_config_change(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        ds1 = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=8, num_windows_per_model=2)
        ds2 = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=10, num_windows_per_model=2)

        assert ds1._disk_cache.cache_dir != ds2._disk_cache.cache_dir

    def test_cache_invalidation_on_data_config_change(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

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
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)
        data_config = {"augmentations": {"num_permutations": 10}, "window_size": 50}

        dataset = make_dataset(
            checkpoints_dataset, tokenizer, tmp_path,
            window_size=50, num_windows_per_model=2,
            data_config_for_hash=data_config,
        )

        config_file = tmp_path / f"default_{dataset._cache_key_suffix}" / "data_config.yaml"
        assert config_file.exists()
        assert yaml.safe_load(config_file.read_text()) == data_config

    def test_cache_persistence_across_instances(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        ds1 = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=8, num_windows_per_model=2)
        w1 = ds1[0]

        ds2 = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=8, num_windows_per_model=2)
        w2 = ds2[0]

        assert torch.equal(w1[0], w2[0])
        assert torch.equal(w1[1], w2[1])
        assert torch.equal(w1[2], w2[2])

    def test_clear_cache(self, tmp_path):
        checkpoints_dataset = create_mock_checkpoints_dataset()
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=8, num_windows_per_model=2)

        assert dataset._disk_cache.contains(dataset._get_cache_key(0))

        dataset.clear_cache()

        assert not dataset._disk_cache.contains(dataset._get_cache_key(0))


class TestCachedWindowedDatasetPerModelIntegration:

    def test_integration_with_real_tokenizer(self, tmp_path):
        model = create_mock_model()
        checkpoint = Mock(spec=Checkpoint)
        checkpoint.model = model

        checkpoints_dataset = Mock()
        checkpoints_dataset.__len__ = Mock(return_value=2)
        checkpoints_dataset.__getitem__ = Mock(return_value=checkpoint)

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

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
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = make_dataset(checkpoints_dataset, tokenizer, tmp_path, window_size=25, num_windows_per_model=2)

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
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

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