import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from sane.data.checkpoint import Checkpoint
from sane.data.datasets import CachedWindowedDataset, CheckpointsDataset
from sane.data.tokenizers.dense import DenseTokenizer


def create_mock_model():
    """Create a mock model for testing."""
    return nn.Sequential(
        nn.Linear(10, 8),  # Smaller sizes for predictable token counts
        nn.Linear(8, 6),
        nn.Linear(6, 4)
    )


class TestAutoModePadding:
    """Test auto mode with padding functionality."""
    
    def test_auto_mode_without_padding_baseline(self, tmp_path):
        """Test auto mode without padding (baseline behavior)."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=5,
            num_windows_per_model="auto",
            pad_windows=False,
            cache_dir=tmp_path,
        )

        # Verify that all windows are exactly window_size or smaller (no padding)
        for i in range(len(dataset)):
            window_tokens, window_mask, _ = dataset[i]
            assert window_tokens.shape[0] <= 5
            valid_count = window_mask.any(dim=1).sum().item()
            assert valid_count == window_tokens.shape[0]
    
    def test_auto_mode_with_padding_includes_remaining_tokens(self, tmp_path):
        """Test that auto mode with padding includes remaining tokens."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset_padded = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=5,
            num_windows_per_model="auto",
            pad_windows=True,
            cache_dir=tmp_path / "padded",
        )

        dataset_no_pad = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=5,
            num_windows_per_model="auto",
            pad_windows=False,
            cache_dir=tmp_path / "no_pad",
        )

        total_tokens_padded = 0
        total_valid_tokens_padded = 0
        for i in range(len(dataset_padded)):
            window_tokens, window_mask, _ = dataset_padded[i]
            assert window_tokens.shape[0] == 5
            total_tokens_padded += window_tokens.shape[0]
            total_valid_tokens_padded += window_mask.any(dim=1).sum().item()

        total_tokens_no_pad = 0
        total_valid_tokens_no_pad = 0
        for i in range(len(dataset_no_pad)):
            window_tokens, window_mask, _ = dataset_no_pad[i]
            total_tokens_no_pad += window_tokens.shape[0]
            total_valid_tokens_no_pad += window_mask.any(dim=1).sum().item()

        assert total_tokens_padded >= total_tokens_no_pad
        assert total_valid_tokens_padded >= total_valid_tokens_no_pad
    
    def test_auto_mode_padding_exact_multiple(self, tmp_path):
        """Test auto mode with padding when layer length is exact multiple of window_size."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        window_size = 2

        dataset_padded = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=window_size,
            num_windows_per_model="auto",
            pad_windows=True,
            cache_dir=tmp_path / "padded",
        )

        dataset_no_pad = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=window_size,
            num_windows_per_model="auto",
            pad_windows=False,
            cache_dir=tmp_path / "no_pad",
        )

        for i in range(len(dataset_padded)):
            window_tokens, _, _ = dataset_padded[i]
            assert window_tokens.shape[0] == window_size
    
    def test_auto_mode_padding_cached_windowed_dataset(self, tmp_path):
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=4,
            num_windows_per_model="auto",
            pad_windows=True,
            cache_dir=tmp_path,
        )
        
        # Check that all individual windows are exactly window_size
        dataset_size = len(dataset)
        for i in range(min(dataset_size, 5)):  # Check first few windows
            tokens, mask, position = dataset[i]
            assert tokens.shape[0] == 4  # Should be exactly window_size
            
            # Count valid tokens
            valid_count = mask.any(dim=1).sum().item()
            assert valid_count <= 4  # Can't have more valid tokens than window_size
    
    def test_auto_mode_padding_full_model_mode(self, tmp_path):
        """Test auto mode with padding in full_model tokenizer mode."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="full_model", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=6,
            num_windows_per_model="auto",
            pad_windows=True,
            cache_dir=tmp_path,
        )

        for i in range(len(dataset)):
            window_tokens, _, _ = dataset[i]
            assert window_tokens.shape[0] == 6
    
    def test_auto_mode_padding_edge_case_single_token_remainder(self, tmp_path):
        """Test edge case where only one token remains after complete windows."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        window_size = 7

        dataset_padded = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=window_size,
            num_windows_per_model="auto",
            pad_windows=True,
            cache_dir=tmp_path / "padded",
        )

        dataset_no_pad = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=window_size,
            num_windows_per_model="auto",
            pad_windows=False,
            cache_dir=tmp_path / "no_pad",
        )

        for i in range(len(dataset_padded)):
            window_tokens, _, _ = dataset_padded[i]
            assert window_tokens.shape[0] == window_size

        total_covered_padded = len(dataset_padded) * window_size
        total_covered_no_pad = sum(dataset_no_pad[i][0].shape[0] for i in range(len(dataset_no_pad)))
        assert total_covered_padded >= total_covered_no_pad
    
    def test_auto_mode_padding_backward_compatibility(self, tmp_path):
        """Test that pad_windows=False is the default and produces consistent results."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset_default = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=5,
            num_windows_per_model="auto",
            allow_overlapping_windows=False,
            cache_dir=tmp_path / "default",
            # pad_windows defaults to False
        )

        dataset_explicit = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=5,
            num_windows_per_model="auto",
            pad_windows=False,
            allow_overlapping_windows=False,
            cache_dir=tmp_path / "explicit",
        )

        assert len(dataset_default) == len(dataset_explicit)

        for i in range(len(dataset_default)):
            tokens1, mask1, pos1 = dataset_default[i]
            tokens2, mask2, pos2 = dataset_explicit[i]
            assert torch.equal(tokens1, tokens2)
            assert torch.equal(mask1, mask2)
            assert torch.equal(pos1, pos2)
    
    def test_auto_mode_includes_more_tokens_with_padding(self, tmp_path):
        """Verify that auto mode with padding includes more tokens than without padding."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset_with_padding = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=7,
            num_windows_per_model="auto",
            pad_windows=True,
            allow_overlapping_windows=False,
            cache_dir=tmp_path / "padded",
        )

        dataset_without_padding = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=7,
            num_windows_per_model="auto",
            pad_windows=False,
            allow_overlapping_windows=False,
            cache_dir=tmp_path / "no_pad",
        )

        total_valid_with_padding = sum(
            dataset_with_padding[i][1].any(dim=1).sum().item()
            for i in range(len(dataset_with_padding))
        )
        total_valid_without_padding = sum(
            dataset_without_padding[i][1].any(dim=1).sum().item()
            for i in range(len(dataset_without_padding))
        )

        assert total_valid_with_padding >= total_valid_without_padding

        for i in range(len(dataset_with_padding)):
            tokens, _, _ = dataset_with_padding[i]
            assert tokens.shape[0] == 7

        assert len(dataset_with_padding) >= len(dataset_without_padding)
    
    def test_auto_mode_non_overlapping_with_padding_coverage(self, tmp_path):
        """Test that auto mode with non-overlapping windows and padding covers all tokens."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=16, device="cpu", mode="layer_wise", reference_statedict=None)

        tokenized_result = tokenizer.tokenize(checkpoint.model.state_dict())

        window_size = 7

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=window_size,
            num_windows_per_model="auto",
            pad_windows=True,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        layer_coverage = {}
        original_layer_lengths = {}

        for layer_tokens, layer_mask, layer_position in tokenized_result:
            layer_idx = layer_position[0, 1].item()
            original_layer_lengths[layer_idx] = layer_tokens.shape[0]

        for i in range(len(dataset)):
            window_tokens, window_mask, window_position = dataset[i]
            layer_idx = window_position[0, 1].item()
            if layer_idx not in layer_coverage:
                layer_coverage[layer_idx] = set()

            valid_mask = window_mask.any(dim=1)
            for j, is_valid in enumerate(valid_mask):
                if is_valid:
                    layer_relative_pos = window_position[j, 2].item()
                    layer_coverage[layer_idx].add(layer_relative_pos)

        for layer_idx, original_length in original_layer_lengths.items():
            if layer_idx in layer_coverage:
                covered_positions = layer_coverage[layer_idx]
                expected_positions = set(range(original_length))
                missing_positions = expected_positions - covered_positions
                assert len(missing_positions) == 0, \
                    f"Layer {layer_idx}: Missing coverage for {len(missing_positions)} tokens at positions {missing_positions}"