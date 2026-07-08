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
        nn.Linear(16, 8),
        nn.Linear(8, 4),
        nn.Linear(4, 2)
    )


class TestNonOverlappingWindows:
    """Test non-overlapping window functionality."""
    
    def test_non_overlapping_windowed_dataset(self, tmp_path):
        """Test CachedWindowedDataset with non-overlapping windows."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            num_windows_per_model=3,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        # Collect all windows for the single checkpoint
        layer_windows = {}
        for i in range(len(dataset)):
            window_tokens, window_mask, window_position = dataset[i]
            if len(window_tokens) > 0:
                layer_idx = window_position[0, 1].item()
                start_pos = window_position[0, 0].item()
                if layer_idx not in layer_windows:
                    layer_windows[layer_idx] = []
                layer_windows[layer_idx].append(start_pos)

        # Check no overlaps within each layer
        for layer_idx, starts in layer_windows.items():
            if len(starts) > 1:
                starts.sort()
                for i in range(1, len(starts)):
                    gap = starts[i] - starts[i-1]
                    assert gap >= 10, f"Layer {layer_idx} windows overlap: gap {gap} < window_size 10"
    
    def test_non_overlapping_cached_windowed_dataset(self, tmp_path):
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])
        
        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)
        
        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=8,
            num_windows_per_model=2,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )
        
        # Collect all windows from same checkpoint to group by layer
        all_windows = []
        dataset_size = len(dataset)
        for i in range(min(dataset_size, 4)):  # Check first few windows
            tokens, mask, position = dataset[i]
            layer_idx = position[0, 1].item()
            start_pos = position[0, 0].item()
            all_windows.append((layer_idx, start_pos))
        
        # Group by layer and check non-overlapping within each layer
        layer_windows = {}
        for layer_idx, start_pos in all_windows:
            if layer_idx not in layer_windows:
                layer_windows[layer_idx] = []
            layer_windows[layer_idx].append(start_pos)
        
        # Verify no overlaps within each layer
        for layer_idx, starts in layer_windows.items():
            if len(starts) > 1:
                starts.sort()
                for i in range(1, len(starts)):
                    gap = starts[i] - starts[i-1]
                    assert gap >= 8, f"Layer {layer_idx} windows overlap: gap {gap} < window_size 8"
    
    def test_get_window_starts_non_overlapping(self, tmp_path):
        """Test _get_window_starts method with non-overlapping setting."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        # Test with different sequence lengths and window counts
        starts = dataset._get_window_starts(50, 3)
        expected = [0, 10, 20]  # Non-overlapping windows
        assert starts == expected

        starts = dataset._get_window_starts(25, 2)
        expected = [0, 10]  # Only 2 windows can fit non-overlapping
        assert starts == expected

        starts = dataset._get_window_starts(15, 1)
        expected = [0]  # Single window at start
        assert starts == expected
    
    def test_get_window_starts_overlapping_vs_non_overlapping(self, tmp_path):
        """Test difference between overlapping and non-overlapping modes."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset_overlap = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            allow_overlapping_windows=True,
            cache_dir=tmp_path / "overlap",
        )

        dataset_no_overlap = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            allow_overlapping_windows=False,
            cache_dir=tmp_path / "no_overlap",
        )

        total_tokens = 50
        num_windows = 3

        starts_overlap = dataset_overlap._get_window_starts(total_tokens, num_windows)
        starts_no_overlap = dataset_no_overlap._get_window_starts(total_tokens, num_windows)

        assert starts_no_overlap == [0, 10, 20]
        assert len(starts_overlap) == 3
        assert all(0 <= start <= 40 for start in starts_overlap)
    
    def test_warning_when_too_many_non_overlapping_windows(self, tmp_path):
        """Test warning when requesting more non-overlapping windows than possible."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        with pytest.warns(UserWarning, match="Requested 5 non-overlapping windows"):
            starts = dataset._get_window_starts(25, 5)

        assert starts == [0, 10]
    
    def test_backward_compatibility_default_non_overlapping(self, tmp_path):
        """Test that default behavior is non-overlapping."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=10,
            num_windows_per_model=2,
            cache_dir=tmp_path,
        )

        assert dataset.allow_overlapping_windows == False
    
    def test_non_overlapping_full_model_mode(self, tmp_path):
        """Test non-overlapping windows with full_model tokenizer mode."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=8,
            num_windows_per_model=2,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        actual_windows = dataset.__getmodel__(0)
        if len(actual_windows) >= 2:
            _, _, pos1 = actual_windows[0]
            _, _, pos2 = actual_windows[1]
            gap = abs(pos2[0, 0].item() - pos1[0, 0].item())
            assert gap >= 8, f"Windows overlap in full_model mode: gap {gap} < window_size 8"
        else:
            assert len(actual_windows) >= 1
    
    def test_edge_case_window_size_equals_sequence_length(self, tmp_path):
        """Test edge case where window size equals sequence length."""
        model = create_mock_model()
        checkpoint = Checkpoint(model=model, metadata={})
        checkpoints_dataset = CheckpointsDataset([checkpoint])

        tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

        dataset = CachedWindowedDataset(
            checkpoints_dataset,
            tokenizer,
            window_size=50,  # Large window size
            num_windows_per_model=2,
            allow_overlapping_windows=False,
            cache_dir=tmp_path,
        )

        # At least one window per checkpoint
        assert len(dataset) >= 1