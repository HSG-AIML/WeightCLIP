import pytest
import torch
import torch.nn as nn
from typing import List, Tuple, Any

from sane.data.checkpoint import Checkpoint
from sane.data.datasets import CachedWindowedDataset, CheckpointsDataset
from sane.data.tokenizers import SANETokenizer
from sane.data.tokenizers.dense import DenseTokenizer


def _make_multi_layer_model():
    return nn.Sequential(
        nn.Linear(10, 8),
        nn.Linear(8, 6),
        nn.Linear(6, 4),
        nn.Linear(4, 2),
    )


def _make_dataset(tmp_path, tokenizer, **kwargs):
    model = _make_multi_layer_model()
    checkpoint = Checkpoint(model=model, metadata={})
    ds = CheckpointsDataset([checkpoint])
    defaults = dict(window_size=8, num_windows_per_model=1, cache_dir=tmp_path)
    defaults.update(kwargs)
    return CachedWindowedDataset(ds, tokenizer, **defaults)


class FixedOutputTokenizer(SANETokenizer):
    """Tokenizer that returns a fixed pre-built output, for precise windowing tests."""

    def __init__(self, tokens, mask, positions):
        super().__init__(tokensize=tokens.shape[1], mode="full_model")
        self._tokens = tokens
        self._mask = mask
        self._positions = positions

    def flatten(self, statedict):
        return []

    def slice_by_layers(self, flattened):
        return [(self._tokens, self._mask, self._positions)]

    def unslice(self, tokens_input, mask=None, position=None):
        return []

    def rebuild_state_dict(self, flattened_weights, reference_statedict=None):
        return {}

    # Override tokenize directly since slice_by_layers would wrap in a list
    def tokenize(self, statedict):
        return (self._tokens, self._mask, self._positions)


class TestFullModelWindowing:
    def test_cached_windowed_dataset_full_model_windowing_padded(self, tmp_path):
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        dataset = _make_dataset(tmp_path, tokenizer, window_size=10, num_windows_per_model=2, pad_windows=True)

        windows = dataset.__getmodel__(0)
        assert len(windows) >= 1
        for tokens, mask, position in windows:
            assert tokens.shape[0] == 10
            assert mask.shape[0] == 10
            assert position.shape[0] == 10

    def test_cached_windowed_dataset_full_model_windowing(self, tmp_path):
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        dataset = _make_dataset(tmp_path, tokenizer, window_size=8, num_windows_per_model=1, pad_windows=True)

        assert len(dataset) == 1
        tokens, mask, position = dataset[0]
        assert tokens.shape[0] == 8
        assert mask.shape[0] == 8
        assert position.shape[0] == 8

    def test_cross_layer_windowing_behavior(self, tmp_path):
        """Windows in full_model mode should span multiple layers."""
        n, d = 6, 32
        tokens = torch.randn(n, d)
        mask = torch.ones(n, d, dtype=torch.bool)
        positions = torch.tensor([
            [0, 0, 0],
            [1, 0, 1],
            [2, 1, 0],
            [3, 1, 1],
            [4, 2, 0],
            [5, 2, 1],
        ], dtype=torch.int32)

        tokenizer = FixedOutputTokenizer(tokens, mask, positions)
        checkpoint = Checkpoint(model=_make_multi_layer_model(), metadata={})
        ds = CheckpointsDataset([checkpoint])
        dataset = CachedWindowedDataset(
            ds, tokenizer, window_size=4, num_windows_per_model=1, pad_windows=False, cache_dir=tmp_path
        )

        win_tokens, win_mask, win_position = dataset[0]
        assert win_tokens.shape[0] == 4

        unique_layers = torch.unique(win_position[:, 1]).tolist()
        assert len(unique_layers) >= 2, f"Expected cross-layer window, got layers: {unique_layers}"

    def test_full_model_vs_layer_wise_windowing_difference(self, tmp_path):
        model = _make_multi_layer_model()
        checkpoint = Checkpoint(model=model, metadata={})
        ds = CheckpointsDataset([checkpoint])

        tok_layer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise")
        dataset_layer = CachedWindowedDataset(ds, tok_layer, window_size=5, num_windows_per_model=1, cache_dir=tmp_path / "layer")

        tok_full = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        dataset_full = CachedWindowedDataset(ds, tok_full, window_size=5, num_windows_per_model=1, cache_dir=tmp_path / "full")

        assert len(dataset_layer) >= 1
        assert len(dataset_full) >= 1

    def test_no_mode_validation_errors(self, tmp_path):
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        # Should not raise
        _make_dataset(tmp_path, tokenizer, window_size=5, num_windows_per_model=1)

    def test_consistent_tokenizer_output(self):
        model = _make_multi_layer_model()
        state_dict = model.state_dict()

        tok_layer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise")
        result_layer = tok_layer.tokenize(state_dict)
        assert isinstance(result_layer, list)
        assert len(result_layer) > 1
        for tokens, mask, position in result_layer:
            assert isinstance(tokens, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert isinstance(position, torch.Tensor)

        tok_full = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        result_full = tok_full.tokenize(state_dict)
        assert isinstance(result_full, tuple)
        assert len(result_full) == 3
        tokens_full, mask_full, position_full = result_full
        assert isinstance(tokens_full, torch.Tensor)
        assert isinstance(mask_full, torch.Tensor)
        assert isinstance(position_full, torch.Tensor)

    def test_per_layer_windowing_validation_with_full_model(self, tmp_path):
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model")
        checkpoint = Checkpoint(model=_make_multi_layer_model(), metadata={})
        ds = CheckpointsDataset([checkpoint])

        with pytest.raises(ValueError, match="Per-layer windowing \\(num_windows_per_layer\\) cannot be used with tokenizer mode 'full_model'"):
            CachedWindowedDataset(ds, tokenizer, window_size=5, num_windows_per_layer=2, cache_dir=tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])