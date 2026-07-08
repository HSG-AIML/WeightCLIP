"""Tests for max_checkpoints parameter in checkpoint parsing."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from sane.data.checkpoint import CheckpointInfo, parseCheckpointInfo


def test_parse_checkpoint_info_no_limit():
    """Test parseCheckpointInfo without max_checkpoints limit."""
    # Create a temporary directory structure in memory
    with patch('os.listdir') as mock_listdir, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch('os.path.basename', side_effect=lambda p: p.split('/')[-1]):

        # Mock directory structure: 3 models, each with 2 epochs
        root_files = ['model1', 'model2', 'model3']
        mock_listdir.side_effect = lambda path: {
            '/test/zoo': root_files,
            '/test/zoo/model1': ['params.json', 'result.json', 'epoch_0', 'epoch_1'],
            '/test/zoo/model2': ['params.json', 'result.json', 'epoch_0', 'epoch_1'],
            '/test/zoo/model3': ['params.json', 'result.json', 'epoch_0', 'epoch_1'],
        }.get(path, [])

        mock_isdir.side_effect = lambda path: not path.endswith('.json')

        # Parse without limit
        result = parseCheckpointInfo('/test/zoo', epoch_idx=[0], checkpoint_filter=None, max_checkpoints=None)

        # Should return 3 checkpoints (one per model)
        assert len(result) == 3


def test_parse_checkpoint_info_with_limit():
    """Test parseCheckpointInfo with max_checkpoints limit."""
    with patch('os.listdir') as mock_listdir, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch('os.path.basename', side_effect=lambda p: p.split('/')[-1]):

        # Mock directory structure: 5 models, each with 1 epoch
        root_files = ['model1', 'model2', 'model3', 'model4', 'model5']
        mock_listdir.side_effect = lambda path: {
            '/test/zoo': root_files,
            '/test/zoo/model1': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model2': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model3': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model4': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model5': ['params.json', 'result.json', 'epoch_0'],
        }.get(path, [])

        mock_isdir.side_effect = lambda path: not path.endswith('.json')

        # Parse with limit of 3
        result = parseCheckpointInfo('/test/zoo', epoch_idx=[0], checkpoint_filter=None, max_checkpoints=3)

        # Should return only 3 checkpoints
        assert len(result) == 3


def test_parse_checkpoint_info_limit_one():
    """Test parseCheckpointInfo with max_checkpoints=1."""
    with patch('os.listdir') as mock_listdir, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch('os.path.basename', side_effect=lambda p: p.split('/')[-1]):

        root_files = ['model1', 'model2']
        mock_listdir.side_effect = lambda path: {
            '/test/zoo': root_files,
            '/test/zoo/model1': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model2': ['params.json', 'result.json', 'epoch_0'],
        }.get(path, [])

        mock_isdir.side_effect = lambda path: not path.endswith('.json')

        # Parse with limit of 1
        result = parseCheckpointInfo('/test/zoo', epoch_idx=[0], checkpoint_filter=None, max_checkpoints=1)

        # Should return only 1 checkpoint
        assert len(result) == 1


def test_parse_checkpoint_info_limit_greater_than_available():
    """Test parseCheckpointInfo when max_checkpoints is greater than available checkpoints."""
    with patch('os.listdir') as mock_listdir, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch('os.path.basename', side_effect=lambda p: p.split('/')[-1]):

        # Mock directory structure: only 2 models
        root_files = ['model1', 'model2']
        mock_listdir.side_effect = lambda path: {
            '/test/zoo': root_files,
            '/test/zoo/model1': ['params.json', 'result.json', 'epoch_0'],
            '/test/zoo/model2': ['params.json', 'result.json', 'epoch_0'],
        }.get(path, [])

        mock_isdir.side_effect = lambda path: not path.endswith('.json')

        # Parse with limit of 10 (more than available)
        result = parseCheckpointInfo('/test/zoo', epoch_idx=[0], checkpoint_filter=None, max_checkpoints=10)

        # Should return all 2 available checkpoints
        assert len(result) == 2


def test_parse_checkpoint_info_matches_exact_epoch_numbers():
    """Positive epoch_idx should match checkpoint directory numbers when present."""
    with patch('os.listdir') as mock_listdir, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', side_effect=lambda *args: '/'.join(args)), \
         patch('os.path.basename', side_effect=lambda p: p.split('/')[-1]):

        mock_listdir.side_effect = lambda path: {
            '/test/zoo': ['model1'],
            '/test/zoo/model1': [
                'params.json',
                'result.json',
                'checkpoint_000020',
                'checkpoint_000021',
                'checkpoint_000024',
            ],
        }.get(path, [])

        mock_isdir.side_effect = lambda path: not path.endswith('.json')

        result = parseCheckpointInfo('/test/zoo', epoch_idx=[20, 24], checkpoint_filter=None, max_checkpoints=None)

        assert [info.epoch_idx for info in result] == [20, 24]
        assert result[0].state_dict_path == '/test/zoo/model1/checkpoint_000020/checkpoints'
        assert result[1].state_dict_path == '/test/zoo/model1/checkpoint_000024/checkpoints'
