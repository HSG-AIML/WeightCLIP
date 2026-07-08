import os
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

from sane.utils.generic_utils import resolve_abs_path

from typing import Dict, List, Optional, Union
from collections.abc import Callable

@dataclass
class CheckpointInfo():
    """
    A dataclass for retaining checkpoint related information.

    Attributes:
        epoch_idx (int): the epoch index of the checkpoint.
        state_dict_path (str): path to the pytorch state_dict of the checkpoint.
        model_config_path (str): path to the model configuration file.
        eval_results_path (str): path to the evaluation results file. 
    """
    epoch_idx: int
    state_dict_path: str
    model_config_path: Optional[str] = None
    eval_results_path: Optional[str] = None


def parseCheckpointInfo(root_dir:str, epoch_idx:List[int], checkpoint_filter: Optional[Callable[[CheckpointInfo], bool]] = None, max_checkpoints: Optional[int] = None) -> List[CheckpointInfo]:
    """
    Parses the model zoo directory structure creating a CheckpointInfo instance per checkpoint.

    Args:
        root_dir (str): The path to a model zoo directory.
        epoch_idx (int|List[int]): The epoch index/indices of the checkpoint(s) of a model.
        checkpoint_filter (Callable[CheckpointInfo, bool], optional): A function to filter the checkpoints. This function should return True for checkpoints to be kept, False otherwise.
        max_checkpoints (int, optional): Maximum number of checkpoints to load. If None, all checkpoints are loaded. Default: None.

    Returns:
        List[CheckpointInfo]: A list of CheckpointInfo instances, one per checkpoint.
    """

    root_dir = resolve_abs_path(root_dir)

    checkpoint_paths_list = []
    for fname in os.listdir(root_dir):
        # ModelZoo contents (1st level)

        path = os.path.join(root_dir, fname)
        if not os.path.isdir(path):
            # path points to a zoo meta file
            # (1) config.json
            # (2) dataset.pt
            # These files are not stored/processed for now.
            continue

        model_dir = path
        model_config = None
        eval_results = None
        epoch_ckpt_list = []
        for sub_fname in os.listdir(model_dir):
            # Model contents (2nd level)

            sub_path = os.path.join(model_dir, sub_fname)
            if not os.path.isdir(sub_path):
                # sub_path points to a model meta file
                # (1) params.json
                # (2) results.json
                filename = os.path.basename(sub_path)
                if filename == "params.json":
                    model_config = sub_path
                elif filename == "result.json":
                    eval_results = sub_path
                else:
                    # ignore other model meta files.
                    pass

                continue
            
            epoch_ckpt_list.append(sub_path)

        # If we fail to load metadata, directory was not a model directory.
        if len(epoch_ckpt_list) == 0:
            continue
        
        def epoch_number(path: str) -> int | None:
            basename = os.path.basename(path)
            try:
                return int(basename.split('_')[-1])
            except ValueError:
                return None

        def epoch_key(path: str) -> int:
            parsed = epoch_number(path)
            return parsed if parsed is not None else float('inf')

        epoch_ckpt_list.sort(key=epoch_key)
        available_epochs = [epoch_number(path) for path in epoch_ckpt_list]
        epoch_to_path = {
            epoch: path
            for path in epoch_ckpt_list
            if (epoch := epoch_number(path)) is not None
        }

        for e in epoch_idx:
            if e >= 0 and e in epoch_to_path:
                state_dict_dir = epoch_to_path[e]
                epoch_number = e
            else:
                try:
                    state_dict_dir = epoch_ckpt_list[e]
                except IndexError as error:
                    raise IndexError(
                        f"Got IndexError trying to get epoch {e} for subpath {model_dir} "
                        f"from checkpoints list with {len(epoch_ckpt_list)} elements. "
                        f"Available checkpoint epochs: {available_epochs}"
                    ) from error
                epoch_number = e if e >= 0 else e + len(epoch_ckpt_list)
            state_dict = os.path.join(state_dict_dir, 'checkpoints')
            # checkpoints_training is not stored/processed for now.

            checkpoint_info = CheckpointInfo(
                epoch_idx=epoch_number,
                state_dict_path=state_dict,
                model_config_path=model_config,
                eval_results_path=eval_results
            )

            if checkpoint_filter is None or checkpoint_filter(checkpoint_info):
                checkpoint_paths_list.append(checkpoint_info)

    # Apply max_checkpoints limit if specified
    if max_checkpoints is not None and max_checkpoints > 0:
        if max_checkpoints > len(checkpoint_paths_list):
            logger.warning(f"Requested max_checkpoints={max_checkpoints} is greater than available checkpoints={len(checkpoint_paths_list)}. Returning all available checkpoints.")
        else:
            checkpoint_paths_list = checkpoint_paths_list[:max_checkpoints]

    return checkpoint_paths_list