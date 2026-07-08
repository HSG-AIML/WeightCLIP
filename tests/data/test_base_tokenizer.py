import pytest
import torch
from collections import OrderedDict
from sane.data.tokenizers.base import SANETokenizer

class DummySANETokenizer(SANETokenizer):
    """Concrete implementation for testing SANETokenizer abstract base class."""

    def flatten(self, checkpoint):
        # Flatten each tensor in the checkpoint
        return [v.flatten() for v in checkpoint.values()]

    def slice_by_layers(self, flattened):
        # Each flattened tensor becomes a token, mask (all ones), and position (arange)
        result = []
        for i, tensor in enumerate(flattened):
            tokens = tensor.unsqueeze(1)
            mask = torch.ones(tokens.shape[0], 1)
            pos = torch.arange(tokens.shape[0]).unsqueeze(1)
            result.append((tokens, mask, pos))
        return result

    def unslice(self, tokens_input, mask=None, position=None):
        # Just return dummy tensors for testing
        if isinstance(tokens_input, list):
            # layer_wise format
            return [t.squeeze(1) for (t, _, _) in tokens_input]
        else:
            # full_model format
            return [tokens_input.squeeze(1)]

    def rebuild_state_dict(self, flattened, reference_statedict=None):
        # Rebuild with dummy keys
        od = OrderedDict()
        for i, tensor in enumerate(flattened):
            od[f'layer{i}'] = tensor
        return od

@pytest.fixture
def checkpoint():
    # Create a dummy checkpoint with two layers
    return OrderedDict({
        'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        'layer2.bias': torch.tensor([5.0, 6.0])
    })

def test_init_valid_modes():
    # Should not raise
    DummySANETokenizer(tokensize=2, device='cpu', mode='layer_wise', reference_statedict=None)
    DummySANETokenizer(tokensize=2, device='cpu', mode='full_model', reference_statedict=None)

def test_init_invalid_mode():
    with pytest.raises(ValueError):
        DummySANETokenizer(tokensize=2, device='cpu', mode='invalid', reference_statedict=None)

def test_tokenize_layer_wise(checkpoint):
    tokenizer = DummySANETokenizer(tokensize=2, device='cpu', mode='layer_wise', reference_statedict=None)
    result = tokenizer.tokenize(checkpoint)
    assert isinstance(result, list)
    assert len(result) == 2
    for tokens, mask, pos in result:
        assert tokens.shape[1] == 1
        assert mask.shape == tokens.shape
        assert pos.shape == tokens.shape

def test_tokenize_full_model(checkpoint):
    tokenizer = DummySANETokenizer(tokensize=2, device='cpu', mode='full_model', reference_statedict=None)
    tokens, mask, pos = tokenizer.tokenize(checkpoint)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[1] == 1
    assert mask.shape == tokens.shape
    assert pos.shape == tokens.shape

def test_slice_concatenates(checkpoint):
    tokenizer = DummySANETokenizer(tokensize=2, device='cpu', mode='full_model', reference_statedict=None)
    flattened = tokenizer.flatten(checkpoint)
    tokens, mask, pos = tokenizer.slice(flattened)
    # Should concatenate all tokens from both layers
    expected_len = sum(t.numel() for t in flattened)
    assert tokens.shape[0] == expected_len
    assert mask.shape[0] == expected_len
    assert pos.shape[0] == expected_len

def test_detokenize_layer_wise(checkpoint):
    tokenizer = DummySANETokenizer(tokensize=2, device='cpu', mode='layer_wise', reference_statedict=None)
    result = tokenizer.tokenize(checkpoint)
    state_dict = tokenizer.detokenize(result)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == {'layer0', 'layer1'}

def test_detokenize_full_model(checkpoint):
    tokenizer = DummySANETokenizer(tokensize=2, device='cpu', mode='full_model', reference_statedict=None)
    tokens, mask, pos = tokenizer.tokenize(checkpoint)
    state_dict = tokenizer.detokenize(tokens, mask, pos)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == {'layer0'}