import torch
from typing import List, Optional, Union
import logging

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset
from sane.data.checkpoint import Checkpoint, parseCheckpointInfo
from sane.data.checkpoint.checkpoint_augmentation import CheckpointAugmentation
from sane.data.checkpoint.checkpoint_filters import WeightMagnitudeCheckpointFilter
from sane.data.datasets.zoo_dataset_models import get_model
from sane.utils.generic_utils import parse_json, parse_jsonl


logger = logging.getLogger(__name__)


def _unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        nested_state = state_dict.get(key)
        if isinstance(nested_state, dict):
            state_dict = nested_state
            break

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict


def _highest_index_weight_out_dim(state_dict: dict[str, torch.Tensor], prefix: str) -> Optional[int]:
    best_match = None
    for key, value in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue

        parts = key.split(".")
        if len(parts) < 3:
            continue

        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue

        if best_match is None or layer_idx > best_match[0]:
            best_match = (layer_idx, int(value.shape[0]))

    return None if best_match is None else best_match[1]


def _infer_classifier_out_dim(state_dict: dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("fc.weight", "classifier.weight", "heads.head.weight", "head.weight"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[0])

    for prefix in ("module_list.", "layers.", "classifier."):
        out_dim = _highest_index_weight_out_dim(state_dict, prefix)
        if out_dim is not None:
            return out_dim

    return None


def _reconcile_model_output_dim(model_config: dict, state_dict: dict[str, torch.Tensor]) -> dict:
    out_dim = _infer_classifier_out_dim(state_dict)
    if out_dim is None:
        return model_config

    configured_out_dim = model_config.get("model::o_dim")
    if configured_out_dim is not None and int(configured_out_dim) == out_dim:
        return model_config

    resolved_config = dict(model_config)
    # if configured_out_dim is not None:
    #     logger.warning(
    #         "Overriding model::o_dim from %s to %s for %s because checkpoint weights disagree with params.json.",
    #         configured_out_dim,
    #         out_dim,
    #         resolved_config.get("model::type", "<unknown>"),
    #     )
    resolved_config["model::o_dim"] = out_dim
    return resolved_config

class ZooDataset(CheckpointsDataset):
    """
    ZooDataset class supports ModelZoos compiled by the AIML lab of
    the University of St. Gallen.

    Usage:
        To instantiate a ZooDataset a path to a model zoo's root directory is required.
        Additionally, the epoch index(s) can be specified to include the checkpoint(s) 
        of a model of the specified epoch.

        As this class inherits from the standard Pytorch Dataset class both __init__ and 
        __get_item__ methods are supported.

    Indicative model zoo directory structure:

    zoo_root_dir/
    ├── model_id_dir/
    │   ├── checkpoint_<1>/
    │   ├── ...
    │   ├── checkpoint_<epoch_no>/
    │       ├── checkpoints # model's state_dict
    │       ├── checkpoints_training [optional]
    │   ├── ...
    │   ├── checkpoint_<N>/
    │   ├── params.json # model's config dictionary (specifies the architecture, optimizer and scheduler)
    │   ├── result.json # model's evaluation results - contains a line for each epoch (JSONL format)
    ├── model_id_dir/
    │   ├── checkpoint_<1>/
    │   ├── ...
    │   ├── checkpoint_<epoch_no>/
    │       ├── checkpoints # model's state_dict
    │       ├── checkpoints_training [optional]
    │   ├── ...
    │   ├── checkpoint_<N>/
    │   ├── params.json # model's config dictionary (specifies the architecture, optimizer and scheduler)
    │   ├── result.json # model's evaluation results - contains a line for each epoch (JSONL format)
    ...
    ├── config.json # grid search dictionary
    ├── dataset.pt # image dataset used for training serialized 
    """
    def __init__(self, root_dir:str, epoch_idx:Union[int, List[int]]=-1, drop_nan:bool=False, max_absolute_weight:Optional[float]=None, max_checkpoints:Optional[int]=None, augmentations:Optional[CheckpointAugmentation]=None):
        """
        Args:
        root_dir (str): Model zoo directory path.
        epoch_idx (int|List[int]): The epoch index/indices of the checkpoint(s) of a model.
        drop_nan (bool, optional): If True, checkpoints containing NaN values are filtered out. Default: False
        max_absolute_weight (float, optional): The maximum allowed absolute value for any weight. If None, this check is skipped. Default: None
        max_checkpoints (int, optional): Maximum number of checkpoints to load. If None, all checkpoints are loaded. Note that this number is taken after loading multiple epochs, it is not to be multiplied by the number of epochs loaded. Default: None
        augmentations (CheckpointAugmentation, optional): A function/transform to apply to the dataset. Default: None
        """
        super().__init__()
        if isinstance(epoch_idx, int):
            epoch_idx = [epoch_idx]

        if drop_nan or max_absolute_weight is not None:
            checkpoint_filter = WeightMagnitudeCheckpointFilter(
                max_absolute_weight=max_absolute_weight,
                drop_nan=drop_nan
            )
        else:
            checkpoint_filter = None

        self.__ckpt_info_list = parseCheckpointInfo(root_dir, epoch_idx, checkpoint_filter, max_checkpoints)
        self.epoch_idx = epoch_idx
        self.augmentations = augmentations
        self.root_dir = root_dir  # Store root_dir for accessing dataset.pt

    def __getitem__(self, index: int) -> Union[Checkpoint, List[Checkpoint]]:
        ckpt_info = self.__ckpt_info_list[index]
        
        model_config = parse_json(ckpt_info.model_config_path)
        eval_results = parse_jsonl(ckpt_info.eval_results_path)[ckpt_info.epoch_idx]

        state_dict = torch.load(ckpt_info.state_dict_path, map_location='cpu')
        state_dict = _unwrap_state_dict(state_dict)
        model_config = _reconcile_model_output_dim(model_config, state_dict)
        model = get_model(model_config)
        model.load_state_dict(state_dict)

        checkpoint = Checkpoint(
              model
            , metadata = {'model_config': model_config, 'eval_results': eval_results}
        )

        # If we use augmentations, we return a list of augmented checkpoints
        return self.augmentations(checkpoint) if self.augmentations is not None else checkpoint

    def __len__(self) -> int:
        return len(self.__ckpt_info_list)