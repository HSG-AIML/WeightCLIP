import torch
import torch.nn as nn

from sane.data.tokenizers.dense import DenseTokenizer
from sane.model.position import LearnedPositionEmbeddings, FunctionalSinusoidalPositionEmbeddings

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

def test_learned_position_embeddings():
    """Test the LearnedPositionEmbeddings."""
    embedding_dim = 16
    tokens, _, position = generate_tokens()

    projection = nn.Linear(tokens.shape[-1], embedding_dim)
    pos_embedder = LearnedPositionEmbeddings(embedding_dim)
    
    # Forward pass
    input = projection(tokens)
    output = pos_embedder(input, position)
    
    # Check the output shape
    assert output.shape == (tokens.shape[0], tokens.shape[1], embedding_dim), f"Expected output shape {(tokens.shape[0], tokens.shape[1], embedding_dim)}, but got {output.shape}"

    # Check that output is different from input
    assert not torch.allclose(input, output), "Output should not be equal to input tokens."

    # Check that the embeddings are learnable
    assert sum(p.numel() for p in pos_embedder.parameters() if p.requires_grad) > 0, "LearnedPositionEmbeddings should have learnable parameters."

def test_functional_sinusoidal_position_embeddings():
    """Test the FunctionalSinusoidalPositionEmbeddings."""
    embedding_dim = 16
    tokens, _, position = generate_tokens()

    projection = nn.Linear(tokens.shape[-1], embedding_dim)
    pos_embedder = FunctionalSinusoidalPositionEmbeddings(embedding_dim)
    
    # Forward pass
    input = projection(tokens)
    output = pos_embedder(input, position)

    # Check the output shape
    assert output.shape == (tokens.shape[0], tokens.shape[1], embedding_dim), f"Expected output shape {(tokens.shape[0], tokens.shape[1], embedding_dim)}, but got {output.shape}"

    # Check that output is different from input
    assert not torch.allclose(input, output), "Output should not be equal to input tokens."

    # Check that the embeddings are not learnable
    assert sum(p.numel() for p in pos_embedder.parameters() if p.requires_grad) == 0, "FunctionalSinusoidalPositionEmbeddings should not have learnable parameters."

def test_functional_sinusoidal_position_embeddings_vectorised_implementation():
    x, _, positions = generate_tokens()
    pos_embedder = FunctionalSinusoidalPositionEmbeddings(x.shape[-1])

    def get_sinusoidal_embedding_original(positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_pos_dims = positions.shape
        emb = torch.zeros(batch_size, seq_len, pos_embedder.embedding_dim, device=positions.device, dtype=torch.float32)
        for i in range(num_pos_dims):
            position_indices = positions[:, :, i].unsqueeze(-1).float()
            sin_part = torch.sin(position_indices * pos_embedder.div_term)
            cos_part = torch.cos(position_indices * pos_embedder.div_term)
            emb[:, :, 0::2] += sin_part
            emb[:, :, 1::2] += cos_part
        return emb


    # Get embeddings from both functions
    emb_orig = get_sinusoidal_embedding_original(positions)
    emb_vect = pos_embedder._get_sinusoidal_embedding(positions)

    print(emb_orig.shape)
    print(torch.max(emb_orig - emb_vect))

    # Check if the outputs are equal
    assert torch.allclose(emb_orig, emb_vect), "The outputs of the two functions are not equal"