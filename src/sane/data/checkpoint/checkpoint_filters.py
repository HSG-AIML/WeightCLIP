from typing import List, Optional
import torch
from sane.data.checkpoint.checkpoint_info import CheckpointInfo
from sane.data.datasets.zoo_dataset_models import get_model

import logging
logger = logging.getLogger(__name__)

class CheckpointFilter:
    r"""Base class for filtering checkpoints.

    This class defines the interface for all checkpoint filters. Subclasses should
    implement the `__call__` method to define the filtering logic.

    Args:
        checkpoint_info (CheckpointInfo): An object containing information about the checkpoint.

    Returns:
        bool: ``True`` if the checkpoint passes the filter, ``False`` otherwise.
    """
    def __call__(self, checkpoint_info: CheckpointInfo) -> bool:
        return True

class AllCheckpointFilter(CheckpointFilter):
    r"""Combines multiple checkpoint filters using a logical AND.

    This filter applies all provided filters to a checkpoint and returns ``True``
    only if all filters return ``True``.

    Args:
        filters (List[CheckpointFilter]): A list of checkpoint filters to combine.

    Shape:
        - Input: A `CheckpointInfo` object.
        - Output: ``True`` if all filters return ``True``, ``False`` otherwise.
    """
    def __init__(self, filters: List[CheckpointFilter]):
        self.filters = filters

    def __call__(self, checkpoint_info: CheckpointInfo) -> bool:
        return all(f(checkpoint_info) for f in self.filters)

class WeightMagnitudeCheckpointFilter(CheckpointFilter):
    r"""Filters checkpoints based on the magnitude of their weights.

    This filter checks the state dictionary of a checkpoint for weights that exceed
    a specified maximum absolute value or contain NaN values. If either condition
    is met, the checkpoint is filtered out.

    Args:
        max_absolute_weight (float, optional): The maximum allowed absolute value for any weight.
            If ``None``, this check is skipped. Default: ``None``
        drop_nan (bool, optional): If ``True``, checkpoints containing NaN values are filtered out.
            Default: ``True``

    Shape:
        - Input: A `CheckpointInfo` object.
        - Output: ``True`` if the checkpoint passes all checks, ``False`` otherwise.
    """
    def __init__(self, max_absolute_weight: Optional[float] = None, drop_nan: bool = True):
        self.max_absolute_weight = max_absolute_weight
        self.drop_nan = drop_nan

    def __call__(self, checkpoint_info: CheckpointInfo) -> bool:
        try:
            state_dict = torch.load(checkpoint_info.state_dict_path, map_location='cpu')
        except FileNotFoundError as e:
            logger.warning(f"Could not load state_dict from {checkpoint_info.state_dict_path}: {e}")
            return False
        for name, param in state_dict.items():
            if self.drop_nan and torch.isnan(param).any():
                return False
            if self.max_absolute_weight is not None and torch.abs(param).max() > self.max_absolute_weight:
                return False
        return True
