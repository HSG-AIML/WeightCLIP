import torch
import torch.nn as nn

from sane.model.translator import MLPTranslator

def test_mlp_translator():
    """Test the MLPTranslator."""
    n_tokens = 16
    latent_dim = 8

    output_dim = 10  # Example output dimension
    output_n_tokens = 5  # Example number of output tokens

    translator = MLPTranslator(
        input_dim = latent_dim,
        input_n_tokens = n_tokens,
        output_dim = output_dim,
        output_n_tokens = output_n_tokens,
        hidden_layers_dim= (32,),
        activation = 'relu'
    )
    
    # Create a dummy tensor of shape (B, N, L)
    B = 32
    z = torch.randn(B, n_tokens, latent_dim)
    
    # Forward pass
    output = translator(z)
    
    # Check the output shape
    assert output.shape == (B, output_n_tokens, output_dim), f"Expected output shape {(B, output_n_tokens, output_dim)}, but got {output.shape}"
    
    # Check that the output is a linear transformation of the input
    assert output.dtype == torch.float32, "Output should be of type float32."

def test_mlp_translator_construction():
    """Test the construction of MLPTranslator."""
    n_tokens = 16
    latent_dim = 8
    hidden_layers_dim = (32, 64)

    translator = MLPTranslator(
        input_dim = latent_dim,
        input_n_tokens = n_tokens,
        hidden_layers_dim= hidden_layers_dim,
        activation = 'leaky_relu'
    )
    
    # Check the attributes
    assert translator.output_dim == latent_dim, "output_dim attribute mismatch."
    assert translator.output_n_tokens == n_tokens, "output_n_tokens attribute mismatch."
    
    # Check the structure of the MLP
    assert isinstance(translator.head, torch.nn.Sequential), "head should be a Sequential module."
    assert isinstance(translator.head[0], torch.nn.Linear), "First layer should be a Linear layer."
    assert translator.head[0].in_features == n_tokens * latent_dim, "First layer input features should match n_tokens * latent_dim."
    assert translator.head[0].out_features == hidden_layers_dim[0], "First layer output features should match the first hidden layer dimension."
    assert isinstance(translator.head[1], torch.nn.LayerNorm), "Second layer should be a LayerNorm layer."
    assert isinstance(translator.head[2], torch.nn.LeakyReLU), "Third layer should be a LeakyReLU activation."
    assert isinstance(translator.head[3], torch.nn.Linear), "Fourth layer should be a Linear layer."
    assert translator.head[3].in_features == hidden_layers_dim[0], "Fourth layer input features should match the first hidden layer dimension."
    assert translator.head[3].out_features == hidden_layers_dim[1], "Fourth layer output features should match the second hidden layer dimension."
    assert isinstance(translator.head[-1], torch.nn.Linear), "Last layer should be a Linear layer."
    assert translator.head[-1].in_features == hidden_layers_dim[-1], "Last hidden layer input features should match the last hidden layer dimension."
    assert translator.head[-1].out_features == n_tokens * latent_dim, "Last layer output features should match projection_dim."