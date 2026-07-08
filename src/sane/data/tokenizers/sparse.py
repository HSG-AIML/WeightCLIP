import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Optional, Union, List, Tuple
from copy import deepcopy
from .base import SANETokenizer, StateDict

class SparseTokenizer(SANETokenizer):
    """A class to tokenize and detokenize model checkpoints following the non-dense (sparse) variant.

    This class provides methods to tokenize and detokenize state dictionaries.
    The tokenization process involves flattening and slicing the checkpoint,
    while the detokenization process involves unslicing and rebuilding the state dictionary.

    The tokenizer supports two output modes:
    - "layer_wise": Returns a list of tuples, one per layer (default behavior)
    - "full_model": Returns a single tuple with concatenated tensors for all layers
    """

    def flatten(self, statedict: StateDict) -> List[torch.Tensor]:
        """
        Flattens the input checkpoint's weights into 2D matrices.

        Args:
            statedict (StateDict): The input state dictionary to flatten.

        Returns:
            List[torch.Tensor]: A list of 2D tensors (per processed layer) in the OrderedDict iteration order.
        """

        flattened_weights: List[torch.Tensor] = []
        for key in statedict:
            if "weight" in key:
                w = statedict[key].reshape(statedict[key].shape[0], -1)
                if key.replace("weight", "bias") in statedict:
                    b = statedict[key.replace("weight", "bias")].unsqueeze(1)
                    w = torch.cat([w, b], dim=1)
                flattened_weights.append(w)
        return flattened_weights

    def slice_by_layers(self, flattened_weights: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Splits each layer's 2D weight matrix into fixed-size tokens (sparse variant).

        Each input tensor w has shape [C_out, F(+1?)], where each row corresponds to
        one output channel's flattened weights (and optional bias). The method pads
        each row to a multiple of `tokensize`, reshapes into tokens, and builds
        corresponding masks and positional indices.

        Args:
            flattened_weights (List[torch.Tensor]): List of 2D tensors, one per layer.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: A list where each element
            corresponds to one layer and contains:
                - tokens: Tensor of shape [num_tokens_layer, tokensize]
                - mask: Boolean tensor marking valid weights (same shape as tokens)
                - position: Tensor of shape [num_tokens_layer, 3] with columns
                [global_token_id, layer_id, channel_id]
        """
        layer_results = []
        global_token_idx = 0

        for layer_idx, w in enumerate(flattened_weights):
            if w.dim() != 2:
                raise ValueError(f"Expected a 2D tensor, got {w.shape}")

            channel, feature = int(w.shape[0]), int(w.shape[1])

            token_factor = feature // self.tokensize + (1 if (feature % self.tokensize) else 0)
            padding_width = token_factor * self.tokensize
            pad_right = padding_width - feature

            w = w.to(self.device)

            # Pad if needed
            w_pad = self.pad(w, pad_right)

            # Create the token tensor
            tokens = w_pad.view(channel * token_factor, self.tokensize)

            # Create the mask tensor
            mask = torch.zeros((channel, padding_width), dtype=torch.bool, device=self.device)
            mask[:, :feature] = True
            mask = mask.view(channel * token_factor, self.tokensize)

            pos = []
            for c in range(channel):
                for _ in range(token_factor):
                    pos.append([global_token_idx, layer_idx, c])
                    global_token_idx += 1
            position = torch.tensor(pos, dtype=torch.int32, device=self.device)

            layer_results.append((tokens, mask, position))

        return layer_results

    def unslice(self, tokens_input: Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], mask: Optional[torch.Tensor] = None, position: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Unslices the input tokens, supporting both layer-wise and full model formats (sparse variant).

        In the sparse tokenizer, each layer is reconstructed as a 2D tensor where each
        row corresponds to one output channel's flattened weights (and possibly bias).

        Args:
            tokens_input (Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Either a single tensor (full model format) or a list of tuples (layer-wise format).
            mask (torch.Tensor, optional): A boolean mask indicating which token elements are valid.
                Required when using the full model format.
            position (torch.Tensor, optional): A tensor encoding each token's (global_id, layer_id, channel_id).
                Required when using the full model format.

        Returns:
            List[torch.Tensor]: A list of 2D tensors (one per layer), each with shape [C_out, F(+1?)],
            where C_out is the number of output channels and F(+1?) represents the flattened features
            (plus one column if bias was appended during flattening).
        """
        # Handle layer-wise input format
        if isinstance(tokens_input, list):
            layer_weights = []
            for layer_tokens, layer_mask, layer_pos in tokens_input:
                # Remove all fully padded tokens
                valid = layer_mask.any(dim=-1)
                layer_tokens = layer_tokens[valid]
                layer_mask = layer_mask[valid]
                layer_pos = layer_pos[valid]
                rows = []
                for c in range(layer_pos[:, 2].max().item() + 1):
                    rows_c = layer_tokens[layer_pos[:, 2] == c]
                    mask_c = layer_mask[layer_pos[:, 2] == c]
                    flattened_tokens = rows_c.flatten()
                    flattened_mask = mask_c.flatten()
                    # Extract valid tokens
                    rows.append(flattened_tokens[flattened_mask])
                layer_weights.append(torch.stack(rows, dim=0))
            return layer_weights

        # Handle full model format (concatenated tensors)
        if mask is None or position is None:
            raise ValueError("For full model format, both mask and position tensors are required")

        # Remove all fully padded tokens
        valid = mask.any(dim=-1)
        tokens_input = tokens_input[valid]
        mask = mask[valid]
        position = position[valid]

        layer_weights = []
        for layer_idx in range(position[:, 1].max().item() + 1):
            # Get the tokens for the current layer
            layer_tokens = tokens_input[position[:, 1] == layer_idx]
            layer_mask = mask[position[:, 1] == layer_idx]
            layer_pos = position[position[:, 1] == layer_idx]
            rows = []
            # Group by one channel at a time
            for c in range(layer_pos[:, 2].max().item() + 1):
                channel_tokens = layer_tokens[layer_pos[:, 2] == c]
                channel_mask = layer_mask[layer_pos[:, 2] == c]
                flattened_tokens = channel_tokens.flatten()
                flattened_mask = channel_mask.flatten()
                rows.append(flattened_tokens[flattened_mask])
            # Create a tensor for the current layer
            layer_weights.append(torch.stack(rows, dim=0))
        return layer_weights

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
                # Slice off bias and reshape per channel
                w = flattened_weights.pop(0)
                if key.replace("weight", "bias") in checkpoint:
                    b = w[:, -1]
                    w = w[:, :-1]
                    checkpoint[key.replace("weight", "bias")] = b
                checkpoint[key] = w.view(checkpoint[key].shape)
        return checkpoint
