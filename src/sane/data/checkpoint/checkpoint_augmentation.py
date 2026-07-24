import torch
import torch.nn as nn
from typing import OrderedDict, Any, List, Tuple, Optional
from copy import deepcopy
from warnings import warn

from sane.data.checkpoint.checkpoint import Checkpoint
from sane.utils.git_re_basin import (PermutationSpec, weight_matching, apply_permutation)

__all__ = ["CheckpointAugmentation", "CheckpointAugmentationPipeline", "PermutationAugmentation", "CheckpointAligner"]

class CheckpointAugmentation(nn.Module):
    """Base class for preprocessing augmentation methods.
    Such augmentations are typically done during preprocessing, taking as input a checkpoint and returning a list of augmented checkpoints.

    This class serves as a base for implementing various preprocessing and augmentation techniques.
    Subclasses should override the `forward` method to implement specific preprocessing or augmentation logic.

    Args:
        None
    """
    def __init__(self):
        """Initializes the CheckpointAugmentation class."""
        super(CheckpointAugmentation, self).__init__()
        # This class is a placeholder for preprocessing and augmentation methods.
        # It can be extended with specific preprocessing or augmentation techniques as needed.
        pass

    def forward(self, checkpoint: Checkpoint) -> List[Checkpoint]:
        """Applies preprocessing and augmentation to the input checkpoint.

        Args:
            checkpoint (Checkpoint): The input checkpoint to augment.

        Returns:
            List[Checkpoint]: A list of augmented checkpoints, with the original or canonical version being at index 0.
        """
        # Placeholder for actual preprocessing and augmentation logic
        # This should be implemented in subclasses or extended as needed
        return [checkpoint]

class CheckpointAugmentationPipeline(CheckpointAugmentation):
    """A pipeline for applying a sequence of preprocessing augmentations.

    This class allows for the composition of multiple preprocessing augmentations.
    Each one is applied in sequence to all checkpoints output by the previous step.

    Args:
        augmentations (List[CheckpointAugmentation]): A list of preprocessing augmentations to apply.
        suppress_warnings (bool): If True, suppresses warnings about the order of augmentations.
    """
    def __init__(self, augmentations: List[CheckpointAugmentation], suppress_warnings: bool = False):
        """Initializes the CheckpointAugmentationPipeline class with a list of augmentations.

        Args:
            augmentations (List[CheckpointAugmentation]): A list of preprocessing and augmentation methods to apply.
            suppress_warnings (bool): If True, suppresses warnings about the order of augmentations.
        """
        super(CheckpointAugmentationPipeline, self).__init__()
        self.augmentations = nn.ModuleList(augmentations)

        if not suppress_warnings:
            # Check if alignment is performed after permutation augmentation
            has_permutations = False
            for augmentation in self.augmentations:
                if isinstance(augmentation, PermutationAugmentation):
                    has_permutations = True
                elif isinstance(augmentation, CheckpointAligner) and has_permutations:
                    warn((
                        "CheckpointAligner is used after PermutationAugmentation in the current pipeline. "
                        "This may lead to unexpected results, as the permuted checkpoints will also be aligned to the reference checkpoint. "
                        "Consider reordering the augmentations or using a different approach if this is not intended.\n"
                        "Current augmentations: {}".format(
                            [type(a).__name__ for a in self.augmentations]
                        )
                    ))

    def forward(self, checkpoint: OrderedDict[str, Any]) -> List[OrderedDict[str, Any]]:
        """Applies a sequence of preprocessing augmentations to the input checkpoint.

        Args:
            checkpoint (OrderedDict[str, Any]): The input checkpoint to augment.

        Returns:
            List[OrderedDict[str, Any]]: A list of augmented checkpoints, with the original or canonical version being at index 0.
        """
        in_chkpt = [checkpoint]
        for augmentation in self.augmentations:
            out_chkpt = []
            for chkpt in in_chkpt:
                out_chkpt.extend(augmentation(chkpt))
            in_chkpt = out_chkpt
        return out_chkpt

class PermutationAugmentation(CheckpointAugmentation):
    """An augmentation that applies a permutation to the weights of a checkpoint.

    Args:
        permutation_spec (PermutationSpec): The specification for the permutation to apply.
        permutation_number (int): The number of permutations to generate and apply.
    """

    def __init__(self, permutation_spec: PermutationSpec, num_permutations: int):
        """Initializes the PermutationAugmentation class with a given permutation specification and number of permutations.

        Args:
            permutation_spec (PermutationSpec): The specification for the permutation to apply.
            num_permutations (int): The number of permutations to generate and apply.
        """
        super(PermutationAugmentation, self).__init__()
        self.permutation_spec = permutation_spec
        self.num_permutations = num_permutations

    def forward(self, checkpoint: Checkpoint) -> List[Checkpoint]:
        """Applies the specified permutation to the input checkpoint. The original checkpoint is included as the first element in the output list.

        Args:
            checkpoint (Checkpoint): The input checkpoint to augment.

        Returns:
            List[Checkpoint]: A list containing the original checkpoint and the permuted checkpoints.
        """
        state_dict = checkpoint.model.state_dict()

        # Get reference checkpoint
        # Find permutation of model to itself as reference
        reference_permutation = weight_matching(ps=self.permutation_spec, params_a=state_dict, params_b=state_dict)

        # Compute random permutations
        permutation_dicts = []
        for _ in range(self.num_permutations):
            perm = deepcopy(reference_permutation)
            for key in perm.keys():
                # Get permuted indices for current layer
                perm[key] = torch.randperm(perm[key].shape[0]).float()
            # Append to list of permutation dicts
            permutation_dicts.append(perm)

        # Apply permutations on checkpoints
        checkpoints = [checkpoint]
        for perm_dict in permutation_dicts:
            # Copy reference checkpoint
            index_check = deepcopy(state_dict)
            # Apply permutation on checkpoint
            index_check_perm = apply_permutation(ps=self.permutation_spec, perm=perm_dict, params=index_check)

            chkpt = deepcopy(checkpoint)
            chkpt.model.load_state_dict(index_check_perm)
            checkpoints.append(chkpt)

        return checkpoints


class CheckpointAligner(CheckpointAugmentation):
    """Aligns the weights of a checkpoint to a reference checkpoint using Git Re-Basin.

    Args:
        permutation_spec (PermutationSpec): The specification for the permutation to apply.
        reference_checkpoint (OrderedDict[str, Any]): The reference checkpoint to align to.
    """

    def __init__(self, permutation_spec: PermutationSpec, reference_checkpoint: Checkpoint):
        """Initializes the CheckpointAligner class with a given permutation specification and reference checkpoint.

        Args:
            permutation_spec (PermutationSpec): The specification for the permutation to apply.
            reference_checkpoint (Checkpoint): The reference checkpoint to align to.
        """
        super(CheckpointAligner, self).__init__()
        self.permutation_spec = permutation_spec
        self.reference_checkpoint = reference_checkpoint

    def forward(self, checkpoint: Checkpoint) -> List[Checkpoint]:
        """Aligns the input checkpoint to the reference checkpoint using the specified permutation.

        Args:
            checkpoint (Checkpoint): The input checkpoint to align.

        Returns:
            List[Checkpoint]: A list containing a unique element that is the aligned checkpoint.
        """
        # Find permutation of model to itself as reference
        perm = weight_matching(
            ps=self.permutation_spec, params_a=self.reference_checkpoint.model.state_dict(), params_b=checkpoint.model.state_dict()
        )
        
        # Apply permutation on checkpoint
        aligned_checkpoint = apply_permutation(ps=self.permutation_spec, perm=perm, params=checkpoint.model.state_dict())
        
        out_chkpt = deepcopy(checkpoint)
        out_chkpt.model.load_state_dict(aligned_checkpoint)

        return [out_chkpt]