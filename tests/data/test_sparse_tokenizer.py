import torch
from copy import deepcopy
from sane.data.tokenizers import SparseTokenizer

def generate_mlp_checkpoint():
    """Generates a simple checkpoint for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2)
    )
    checkpoint = model.state_dict()
    return checkpoint


def generate_reference_checkpoint(checkpoint):
    """Generates a reference checkpoint full of 0s, following the same architecture of some other checkpoint."""
    # For reference in rebuild
    reference_checkpoint = deepcopy(checkpoint)
    # We zero all weights to ensure weights are not used in the rebuild
    for key in reference_checkpoint:
            if 'weight' in key or 'bias' in key:
                reference_checkpoint[key].fill_(0)
    # Verify reference checkpoint is different from the original checkpoint
    assert not all(torch.equal(checkpoint[key], reference_checkpoint[key]) for key in checkpoint)
    return reference_checkpoint


def test_detokenize_tokenize_equality():
    """Test that detokenize(tokenize(x)) == x for full_model mode."""

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()
    reference_checkpoint = generate_reference_checkpoint(checkpoint)

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="full_model", reference_statedict=reference_checkpoint)

    # Tokenize the checkpoint using the full model method
    tokens, mask, position = tokenizer.tokenize(checkpoint)

    # Detokenize the tokens
    detokenized_checkpoint = tokenizer.detokenize(tokens, mask, position, reference_statedict=reference_checkpoint)
    # Check that detokenize(tokenize(x)) == x
    for key in checkpoint:
        assert torch.equal(checkpoint[key], detokenized_checkpoint[key])


def test_tokenize_shape():
    """Test that the shape of the layer-structured tokenized output is correct."""

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()
    n_weights = sum(t.numel() for t in checkpoint.values())

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=checkpoint)

    # Tokenize the checkpoint (returns layer-structured data)
    layer_data = tokenizer.tokenize(checkpoint)
    
    # Should return a list of tuples
    assert isinstance(layer_data, list)
    assert len(layer_data) == 2  # Two layers in the MLP
    
    # Check each layer
    total_valid_tokens = 0
    total_tokens = 0
    for layer_tokens, layer_mask, layer_position in layer_data:
        # Check shapes
        assert layer_tokens.dim() == 2
        assert layer_tokens.shape[1] == tokenizer.tokensize  # Tokens should match the specified size
        assert layer_mask.shape == layer_tokens.shape
        assert layer_position.shape[0] == layer_tokens.shape[0]
        assert layer_position.shape[1] == 3  # [token_idx, layer_idx, token_in_layer_idx]
        
        # Count valid tokens
        total_valid_tokens += layer_mask.sum().item()
        total_tokens += layer_tokens.shape[0]

    # Mask should have as many true values as there are weights in the checkpoint
    assert total_valid_tokens == n_weights


def test_tokenize_positions():
    """Test that the positions of the tokens are correctly generated in layer-structured output."""

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=checkpoint)

    # Tokenize the checkpoint (layer-structured)
    layer_data = tokenizer.tokenize(checkpoint)
    
    global_token_idx = 0
    for layer_idx, (layer_tokens, layer_mask, layer_position) in enumerate(layer_data):
        # Check that positions are correctly generated for each layer
        num_tokens_in_layer = layer_tokens.shape[0]
        
        # First dimension should be global token index (continuing from previous layers)
        expected_global_indices = torch.arange(global_token_idx, global_token_idx + num_tokens_in_layer)
        assert torch.all(layer_position[:, 0] == expected_global_indices)
        
        # Second dimension should be layer index (same for all tokens in this layer)
        assert torch.all(layer_position[:, 1] == layer_idx)
        
        # Third dimension should be token index within the layer
        expected_local_indices = torch.arange(num_tokens_in_layer)
        assert torch.all(layer_position[:, 2] == expected_local_indices)
        
        global_token_idx += num_tokens_in_layer


def test_unslice_slice_equality():
    """Test that unslice(slice(x)) == x for full model format."""
    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=checkpoint)

    # Flatten the checkpoint
    flattened_weights = tokenizer.flatten(checkpoint)

    # Slice the flattened weights
    tokens, mask, position = tokenizer.slice(flattened_weights)

    # Unslice the tokens
    unsliced_weights = tokenizer.unslice(tokens, mask, position)

    # Check that unslice(slice(x)) == x
    for i, w in enumerate(flattened_weights):
        assert torch.all(unsliced_weights[i] == w)


def test_unslice_slice_by_layers_equality():
    """Test that unslice(slice_by_layers(x)) == x for layer-wise format."""

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=checkpoint)

    # Flatten the checkpoint
    flattened_weights = tokenizer.flatten(checkpoint)

    # Slice the flattened weights by layers
    layer_data = tokenizer.slice_by_layers(flattened_weights)

    # Unslice the tokens
    unsliced_weights = tokenizer.unslice(layer_data)

    # Check that unslice(slice_by_layers(x)) == x
    for i, w in enumerate(flattened_weights):
        assert torch.all(unsliced_weights[i] == w)


def test_rebuild_flatten_equality():
    """Test that rebuild(flatten(x)) == x."""
    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()
    reference_checkpoint = generate_reference_checkpoint(checkpoint)

    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=reference_checkpoint)

    # Flatten the checkpoint
    flattened_weights = tokenizer.flatten(checkpoint)

    # Rebuild the flattened weights
    rebuilt_checkpoint = tokenizer.rebuild_state_dict(flattened_weights, reference_statedict=reference_checkpoint)

    # Check that rebuild(flatten(x)) == x
    for key in checkpoint:
        assert torch.equal(checkpoint[key], rebuilt_checkpoint[key])


def test_flatten_shape():
    """Test that the shape of the flattened output is correct."""
    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()

    # Flatten the checkpoint
    flattened_weights = tokenizer.flatten(checkpoint)

    # Check that the flattened weights are a list of tensors
    assert isinstance(flattened_weights, list)
    assert all(isinstance(t, torch.Tensor) for t in flattened_weights)

    # Check that the total number of elements matches the original checkpoint
    n_weights = sum(t.numel() for t in checkpoint.values())
    assert sum(t.numel() for t in flattened_weights) == n_weights
    assert len(flattened_weights) == 2 # Check that the number of flattened weights is equal to the number of layers

    # Check shapes: (sparse = 2D: [C_out, F(+1 bias)])
    assert flattened_weights[0].shape == (8, 17)   # 16 weights + 1 bias per row
    assert flattened_weights[1].shape == (2, 9)    # 8 weights + 1 bias per row

    # Check total elements
    assert flattened_weights[0].numel() == 16*8+8  # First layer: 16 inputs, 8 outputs, plus bias
    assert flattened_weights[1].numel() == 8*2+2   # Second layer: 8 inputs, 2 outputs, plus bias


def test_tokenize_layer_vs_full_model_compatibility():
    """Test that layer-wise and full_model modes produce equivalent results."""
    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()
    
    # Initialize tokenizers with different modes
    tokenizer_layer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)
    tokenizer_full = SparseTokenizer(tokensize=32, device="cpu", mode="full_model", reference_statedict=None)
    
    # Get layer-structured output
    layer_data = tokenizer_layer.tokenize(checkpoint)
    
    # Get full model output
    tokens_full, mask_full, position_full = tokenizer_full.tokenize(checkpoint)
    
    # Concatenate layer data
    tokens_concat = torch.cat([layer_tokens for layer_tokens, _, _ in layer_data], dim=0)
    mask_concat = torch.cat([layer_mask for _, layer_mask, _ in layer_data], dim=0)
    position_concat = torch.cat([layer_position for _, _, layer_position in layer_data], dim=0)
    
    # Should be identical
    assert torch.equal(tokens_concat, tokens_full)
    assert torch.equal(mask_concat, mask_full)
    assert torch.equal(position_concat, position_full)


def test_detokenize_tokenize_layer_wise_equality():
    """Test that detokenize(tokenize(x)) == x for layer_wise mode."""
    # Initialize the tokenizer
    tokenizer = SparseTokenizer(tokensize=32, device="cpu", mode="layer_wise", reference_statedict=None)

    # Create a simple checkpoint for testing
    checkpoint = generate_mlp_checkpoint()
    reference_checkpoint = generate_reference_checkpoint(checkpoint)

    # Tokenize the checkpoint using layer-wise mode
    layer_data = tokenizer.tokenize(checkpoint)

    # Detokenize the tokens
    detokenized_checkpoint = tokenizer.detokenize(layer_data, reference_statedict=reference_checkpoint)

    # Check that detokenize(tokenize(x)) == x
    for key in checkpoint:
        assert torch.equal(checkpoint[key], detokenized_checkpoint[key])

