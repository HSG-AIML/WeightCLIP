import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Optional, Union, List, Tuple
from copy import deepcopy
from .base import SANETokenizer, StateDict

class DenseTokenizer(SANETokenizer):
    """A class to tokenize and detokenize model checkpoints following the dense variant.

    This class provides methods to tokenize and detokenize state dictionaries.
    The tokenization process involves flattening and slicing the checkpoint,
    while the detokenization process involves unslicing and rebuilding the state dictionary.

    The tokenizer supports two output modes:
    - "layer_wise": Returns a list of tuples, one per layer (default behavior)
    - "full_model": Returns a single tuple with concatenated tensors for all layers
    """

    def flatten(self, statedict: StateDict) -> List[torch.Tensor]:
        """Flattens the input checkpoint's weights, layer-wise.

        Args:
            statedict (StateDict): The input state dictionary to flatten.

        Returns:
            List[torch.Tensor]: A list of flattened tensors from the checkpoint, each element corresponding to a layer.
        """
        flattened_weights = []
        for key in statedict:
            if "weight" in key:
                w = statedict[key]
                if key.replace("weight", "bias") in statedict:
                    b = statedict[key.replace("weight", "bias")]
                    w = torch.cat([
                        w.view(statedict[key].shape[0], -1),
                        b.unsqueeze(1)
                        ], dim=1)
                flattened_weights.append(w.view(-1))
        return flattened_weights

    def slice_by_layers(self, flattened_weights: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Slices the flattened weights into tokens, returning layer-structured data.

        Args:
            flattened_weights (List[torch.Tensor]): The flattened weights to slice.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: A list of tuples, one per layer, each containing:
                - tokens (torch.Tensor): The sliced tokens for this layer.
                - mask (torch.Tensor): A mask indicating the valid tokens for this layer.
                - position (torch.Tensor): The positions of the tokens for this layer.
        """
        layer_results = []
        global_token_idx = 0

        for layer_idx, w in enumerate(flattened_weights):
            n_tokens_layer = w.numel() // self.tokensize
            n_weights_remaining = w.numel() % self.tokensize
            if n_weights_remaining > 0:
                n_tokens_layer += 1

            padding_width = n_tokens_layer * self.tokensize
            pad_right = padding_width - w.numel()

            w = w.to(self.device)

            # Pad if needed
            w_pad = self.pad(w, pad_right)

            # Create the token tensor for this layer
            tokens = w_pad.view(n_tokens_layer, self.tokensize).to(dtype=torch.float32, device=self.device)

            # Create the mask tensor for this layer
            mask = torch.zeros(padding_width, dtype=torch.bool, device=self.device)
            mask[:w.numel()] = True
            mask = mask.view(n_tokens_layer, self.tokensize)

            # Create the position tensor for this layer
            layer_positions = []
            for token_in_layer_idx in range(n_tokens_layer):
                layer_positions.append([global_token_idx, layer_idx, token_in_layer_idx])
                global_token_idx += 1
            position = torch.tensor(layer_positions, dtype=torch.int32, device=self.device)

            layer_results.append((tokens, mask, position))

        return layer_results

    def unslice(self, tokens_input: Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], mask: Optional[torch.Tensor] = None, position: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Unslices the input tokens, supporting both layer-wise and full model formats.

        Args:
            tokens_input (Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Either a single tensor (full model format) or a list of tuples (layer-wise format).
            mask (torch.Tensor, optional): The mask indicating the valid tokens. Required for full model format.
            position (torch.Tensor, optional): The positions of the tokens. Required for full model format.

        Returns:
            List[torch.Tensor]: A list of unsliced tensors corresponding to the original layers.
        """
        # Handle layer-wise input format
        if isinstance(tokens_input, list):
            flattened_weights = []
            for layer_tokens, layer_mask, _ in tokens_input:
                # Flatten the tokens and mask for this layer
                layer_tokens_flat = layer_tokens.flatten()
                layer_mask_flat = layer_mask.flatten()
                # Extract valid tokens
                flattened_weights.append(layer_tokens_flat[layer_mask_flat])
            return flattened_weights

        # Handle full model format (concatenated tensors)
        if mask is None or position is None:
            raise ValueError("For full model format, both mask and position tensors are required")

        tokens = tokens_input
        flattened_weights = []
        for layer_idx in range(position[:, 1].max().item() + 1):
            # Get the tokens for the current layer
            layer_tokens = tokens[position[:, 1] == layer_idx]
            layer_mask = mask[position[:, 1] == layer_idx]
            # Flatten the tokens and mask
            layer_tokens = layer_tokens.flatten()
            layer_mask = layer_mask.flatten()
            # Create a tensor for the current layer
            flattened_weights.append(layer_tokens[layer_mask])
        return flattened_weights

    def rebuild_state_dict(self, flattened_weights: List[torch.Tensor], reference_statedict: Optional[StateDict] = None) -> StateDict:
        """Rebuilds the state dictionary from flattened weights.

        Args:
            flattened_weights (List[torch.Tensor]): The flattened weights to rebuild the state dictionary from.
            reference_statedict (StateDict, optional): A reference state dictionary to use for rebuilding. Default is None.

        Returns:
            StateDict: The rebuilt state dictionary.
        """
        if reference_statedict is None:
            reference_statedict = self.reference_statedict
        if reference_statedict is None:
            raise ValueError("No reference checkpoint provided for rebuilding state dict. Please provide a reference checkpoint or set the reference_checkpoint attribute of the tokenizer.")

        checkpoint = deepcopy(reference_statedict)
        for key in checkpoint:
            if "weight" in key:
                w = flattened_weights.pop(0).view(checkpoint[key].shape[0], -1)
                if key.replace("weight", "bias") in checkpoint:
                    b = w[:, -1]
                    checkpoint[key.replace("weight", "bias")] = b
                    w = w[:, :-1]
                checkpoint[key] = w.view(checkpoint[key].shape)
        return checkpoint