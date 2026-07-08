import torch
import torch.nn as nn
import math

from sane.data.tokenizers.dense import DenseTokenizer
from sane.model.transformer import SANETransformerEncoder

def generate_tokens():
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2)
    )
    checkpoint = model.state_dict()

    tokenizer = DenseTokenizer(tokensize=32, device="cpu", mode="full_model", reference_statedict=None)
    tokens, mask, position = tokenizer.tokenize(checkpoint)

    return tokens.unsqueeze(0), mask.unsqueeze(0), position.unsqueeze(0)

def test_transformer_encoder():
    """Test the SANETransformerEncoder."""
    embedding_dim = 16
    tokens, _, position = generate_tokens()
    
    encoder = SANETransformerEncoder(
        input_dim = tokens.shape[-1],
        output_dim = embedding_dim,
        d_model = 20,
        nhead = 2,
        num_layers = 2,
        position_embedding = 'sinusoidal'
    )

    # Forward pass
    output = encoder(tokens, position)

    # Check the output shape
    assert output.shape == (tokens.shape[0], tokens.shape[1], embedding_dim), f"Expected output shape {(tokens.shape[0], tokens.shape[1], embedding_dim)}, but got {output.shape}"

    # Check that the transformer is learnable
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) > 0, "TransformerEncoder should have learnable parameters."

