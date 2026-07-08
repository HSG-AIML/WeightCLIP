import torch
import torch.nn as nn

from sane.model.autoencoder import SANEAutoEncoder
from sane.model.transformer import SANETransformerEncoder
from sane.data.tokenizers.dense import DenseTokenizer


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


def test_autoencoder_without_translator():
    """Test the SANEAutoEncoder."""
    embedding_dim = 16
    projection_dim = 8
    d_model = 20
    nhead = 2
    num_layers = 2

    tokens, _, position = generate_tokens()

    config = {
        'input_dim': tokens.shape[-1],
        'n_tokens': tokens.shape[1],
        'latent_dim': embedding_dim,
        'encoder': {
            'architecture': 'transformer',
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'position_embedding': 'sinusoidal',
        },
        'decoder': {
            'architecture': 'transformer',
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'position_embedding': 'sinusoidal',
        },
        'embedder': {
            'architecture': 'average',
        },
        'projector': {
            'architecture': 'mlp',
            'projection_dim': projection_dim,
            'hidden_layers_dim': (16, 16, 8),
        },
        'device': 'cpu'
    }

    model = SANEAutoEncoder.from_config(config)
    model.eval()

    out = model(tokens, position)

    # Check the output shapes
    assert 'z' in out, f"Output should contain 'z', but contains {list(out.keys())}."
    assert out['z'].shape == (tokens.shape[0], tokens.shape[1], embedding_dim), f"Expected shape {(tokens.shape[0], tokens.shape[1], embedding_dim)}, but got {out['z'].shape}."

    assert 'x_hat' in out, f"Output should contain 'x_hat', but contains {list(out.keys())}."
    assert out['x_hat'].shape == (tokens.shape[0], tokens.shape[1], tokens.shape[-1]), f"Expected shape {(tokens.shape[0], tokens.shape[1], tokens.shape[-1])}, but got {out['x_hat'].shape}."

    assert 'z_p' in out, f"Output should contain 'z_p', but contains {list(out.keys())}."
    assert out['z_p'].shape == (tokens.shape[0], projection_dim), f"Expected shape {(tokens.shape[0], projection_dim)}, but got {out['z_p'].shape}."

    assert 'z_tilde' not in out, "Output should not contain 'z_tilde' when translator is None."

    # Check the number of learnable parameters
    encoder = SANETransformerEncoder(input_dim=tokens.shape[-1], output_dim=embedding_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, position_embedding='sinusoidal')
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == sum(p.numel() for p in model.encoder.parameters() if p.requires_grad), "Encoder parameters do not match."

    decoder = SANETransformerEncoder(input_dim=embedding_dim, output_dim=tokens.shape[-1], d_model=d_model, nhead=nhead, num_layers=num_layers, position_embedding='sinusoidal')
    assert sum(p.numel() for p in decoder.parameters() if p.requires_grad) == sum(p.numel() for p in model.decoder.parameters() if p.requires_grad), "Decoder parameters do not match."

    # Check the generation of embeddings
    embeddings = model.forward_embeddings(tokens, position)
    assert embeddings.shape == (tokens.shape[0], embedding_dim), f"Expected shape {(tokens.shape[0], embedding_dim)}, but got {embeddings.shape}."


def test_autoencoder_with_translator():
    """Test the SANEAutoEncoder with a translator."""
    latent_dim = 16
    latent_dim_translator = 8
    n_tokens_translator = 16

    d_model = 20
    nhead = 2
    num_layers = 2

    tokens, _, position = generate_tokens()

    config = {
        'input_dim': tokens.shape[-1],
        'n_tokens': tokens.shape[1],
        'latent_dim': latent_dim,
        'encoder': {
            'architecture': 'transformer',
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'position_embedding': 'sinusoidal',
        },
        'decoder': {
            'architecture': 'transformer',
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'position_embedding': 'sinusoidal',
        },
        'embedder': {
            'architecture': 'average',
        },
        'translator': {
            'architecture': 'mlp',
            'output_dim': latent_dim_translator,
            'output_n_tokens': n_tokens_translator,
            'hidden_layers_dim': (32, 64),
        },
        'device': 'cpu'
    }

    model = SANEAutoEncoder.from_config(config)
    model.eval()

    out = model(tokens, position)

    # Check the output shapes
    assert out['z'].shape == (tokens.shape[0], tokens.shape[1], latent_dim), f"Expected shape {(tokens.shape[0], tokens.shape[1], latent_dim)}, but got {out['z'].shape}."
    
    assert out['x_hat'].shape == (tokens.shape[0], n_tokens_translator, tokens.shape[-1]), f"Expected shape {(tokens.shape[0], tokens.shape[1], tokens.shape[-1])}, but got {out['x_hat'].shape}."

    assert out['z_tilde'].shape == (tokens.shape[0], n_tokens_translator, latent_dim_translator), f"Expected shape {(tokens.shape[0], n_tokens_translator, latent_dim_translator)}, but got {out['z_tilde'].shape}."

    assert 'z_p' not in out, "Output should not contain 'z_p' when projector is None."

    # Check the generation of embeddings
    embeddings = model.forward_embeddings(tokens, position)
    assert embeddings.shape == (tokens.shape[0], latent_dim), f"Expected shape {(tokens.shape[0], latent_dim)}, but got {embeddings.shape}."