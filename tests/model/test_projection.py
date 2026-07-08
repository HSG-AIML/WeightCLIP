import torch

from sane.model.projection import AverageProjectionHead, MLPProjectionHead

def test_average_projection_head():
    """Test the AverageProjectionHead."""
    embedder = AverageProjectionHead()
    
    # Create a dummy tensor of shape (B, N, L)
    B, N, L = 32, 16, 8
    z = torch.randn(B, N, L)
    
    # Forward pass
    output = embedder(z)
    
    # Check the output shape
    assert output.shape == (B, L), f"Expected output shape {(B, L)}, but got {output.shape}"
    
    # Check that the output is the mean across tokens
    expected_output = z.mean(dim=1)
    assert torch.allclose(output, expected_output), "Output does not match expected mean across tokens."

def test_mlp_projection_head():
    """Test the MLPProjectionHead."""
    n_tokens = 16
    latent_dim = 8
    projection_dim = 20
    projector = MLPProjectionHead(n_tokens, latent_dim, projection_dim)
    
    # Create a dummy tensor of shape (B, N, L)
    B = 32
    z = torch.randn(B, n_tokens, latent_dim)
    
    # Forward pass
    output = projector(z)
    
    # Check the output shape
    assert output.shape == (B, projection_dim), f"Expected output shape {(B, projection_dim)}, but got {output.shape}"
    
    # Check that the output is a linear transformation of the input
    assert output.dtype == torch.float32, "Output should be of type float32."

def test_mlp_projection_head_construction():
    """Test the construction of MLPProjectionHead."""
    n_tokens = 16
    latent_dim = 8
    projection_dim = 20
    hidden_layers_dim = (32, 64)
    activation = 'leaky_relu'
    
    projector = MLPProjectionHead(
        n_tokens=n_tokens,
        latent_dim=latent_dim,
        projection_dim=projection_dim,
        hidden_layers_dim=hidden_layers_dim,
        activation=activation
    )
    
    # Check the attributes
    assert projector.n_tokens == n_tokens, "n_tokens attribute mismatch."
    assert projector.latent_dim == latent_dim, "latent_dim attribute mismatch."
    assert projector.projection_dim == projection_dim, "projection_dim attribute mismatch."
    
    # Check the structure of the MLP
    assert isinstance(projector.head, torch.nn.Sequential), "head should be a Sequential module."
    assert isinstance(projector.head[0], torch.nn.Linear), "First layer should be a Linear layer."
    assert projector.head[0].in_features == n_tokens * latent_dim, "First layer input features should match n_tokens * latent_dim."
    assert projector.head[0].out_features == hidden_layers_dim[0], "First layer output features should match the first hidden layer dimension."
    assert isinstance(projector.head[1], torch.nn.LayerNorm), "Second layer should be a LayerNorm layer."
    assert isinstance(projector.head[2], torch.nn.LeakyReLU), "Third layer should be a LeakyReLU activation."
    assert isinstance(projector.head[3], torch.nn.Linear), "Fourth layer should be a Linear layer."
    assert projector.head[3].in_features == hidden_layers_dim[0], "Fourth layer input features should match the first hidden layer dimension."
    assert projector.head[3].out_features == hidden_layers_dim[1], "Fourth layer output features should match the second hidden layer dimension."
    assert isinstance(projector.head[-1], torch.nn.Linear), "Last layer should be a Linear layer."
    assert projector.head[-1].in_features == hidden_layers_dim[-1], "Last hidden layer input features should match the last hidden layer dimension."
    assert projector.head[-1].out_features == projection_dim, "Last layer output features should match projection_dim."
