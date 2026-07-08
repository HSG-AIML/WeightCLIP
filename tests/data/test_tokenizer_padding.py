import pytest
import torch
from copy import deepcopy
from collections import OrderedDict

from sane.data.tokenizers import DenseTokenizer, SparseTokenizer
from sane.data.tokenizers.base import SANETokenizer


class DummyPadTokenizer(SANETokenizer):
    def flatten(self, statedict):
        return []

    def slice_by_layers(self, flattened_weights):
        return []

    def unslice(self, tokens_input, mask=None, position=None):
        return []

    def rebuild_state_dict(self, flattened_weights, reference_statedict=None):
        return {}


def generate_checkpoint():
    return OrderedDict(
        {
            "layer0.weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "layer0.bias": torch.tensor([7.0, 8.0]),
            "layer1.weight": torch.tensor([[9.0, 10.0]]),
            "layer1.bias": torch.tensor([11.0]),
        }
    )


def generate_reference_checkpoint(checkpoint):
    reference_checkpoint = deepcopy(checkpoint)
    for key in reference_checkpoint:
        if "weight" in key or "bias" in key:
            reference_checkpoint[key].fill_(0)
    assert not all(torch.equal(checkpoint[key], reference_checkpoint[key]) for key in checkpoint)
    return reference_checkpoint


def make_dummy_tokenizer(padding="zero"):
    return DummyPadTokenizer(tokensize=5, device="cpu", mode="layer_wise", padding=padding)


def test_padding_default_is_zero():
    tokenizer = DummyPadTokenizer(tokensize=5, device="cpu", mode="layer_wise")
    assert tokenizer.padding == "zero"


def test_zero_padding_appends_zeros():
    tokenizer = make_dummy_tokenizer("zero")
    w = torch.tensor([1.0, 2.0, 3.0])

    w_pad = tokenizer.pad(w, 2)

    assert torch.equal(w_pad, torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]))


def test_mean_padding_appends_mean_value():
    tokenizer = make_dummy_tokenizer("mean")
    w = torch.tensor([1.0, 3.0, 5.0])

    w_pad = tokenizer.pad(w, 2)

    assert torch.equal(w_pad, torch.tensor([1.0, 3.0, 5.0, 3.0, 3.0]))


def test_mean_padding_is_row_wise_for_2d_inputs():
    tokenizer = make_dummy_tokenizer("mean")
    w = torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    w_pad = tokenizer.pad(w, 2)

    expected = torch.tensor([[1.0, 3.0, 2.0, 2.0], [2.0, 6.0, 4.0, 4.0]])
    assert torch.equal(w_pad, expected)


def test_gaussian_padding_matches_seeded_samples():
    tokenizer = make_dummy_tokenizer("gaussian")
    w = torch.tensor([1.0, 3.0, 5.0])

    torch.manual_seed(123)
    w_pad = tokenizer.pad(w, 2)

    torch.manual_seed(123)
    expected_padding = torch.randn(2) * w.std(unbiased=False) + w.mean()
    expected = torch.cat([w, expected_padding], dim=0)

    assert torch.allclose(w_pad, expected)


def test_gaussian_padding_is_row_wise_for_2d_inputs():
    tokenizer = make_dummy_tokenizer("gaussian")
    w = torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    torch.manual_seed(321)
    w_pad = tokenizer.pad(w, 2)

    torch.manual_seed(321)
    mean = w.mean(dim=-1, keepdim=True)
    std = w.std(dim=-1, keepdim=True, unbiased=False)
    expected_padding = torch.randn(2, 2) * std + mean
    expected = torch.cat([w, expected_padding], dim=-1)

    assert torch.allclose(w_pad, expected)


def test_reflect_padding_handles_large_padding_width():
    tokenizer = make_dummy_tokenizer("reflect")
    w = torch.tensor([1.0, 2.0, 3.0])

    w_pad = tokenizer.pad(w, 5)

    expected = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0])
    assert torch.equal(w_pad, expected)


def test_reflect_padding_falls_back_to_zero_for_single_value(caplog):
    tokenizer = make_dummy_tokenizer("reflect")
    w = torch.tensor([5.0])

    with caplog.at_level("WARNING"):
        w_pad = tokenizer.pad(w, 3)

    assert "Falling back to zero padding" in caplog.text
    assert torch.equal(w_pad, torch.tensor([5.0, 0.0, 0.0, 0.0]))


def test_replicate_padding_handles_large_padding_width():
    tokenizer = make_dummy_tokenizer("replicate")
    w = torch.tensor([1.0, 2.0, 3.0])

    w_pad = tokenizer.pad(w, 5)

    expected = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    assert torch.equal(w_pad, expected)


def test_circular_padding_handles_large_padding_width():
    tokenizer = make_dummy_tokenizer("circular")
    w = torch.tensor([1.0, 2.0, 3.0])

    w_pad = tokenizer.pad(w, 5)

    expected = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
    assert torch.equal(w_pad, expected)


def test_invalid_padding_mode_raises_value_error():
    tokenizer = make_dummy_tokenizer("not_a_padding_mode")
    w = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Unsupported padding mode"):
        tokenizer.pad(w, 1)


@pytest.mark.parametrize("tokenizer_cls", [DenseTokenizer, SparseTokenizer])
@pytest.mark.parametrize("mode", ["layer_wise", "full_model"])
@pytest.mark.parametrize("padding", ["zero", "mean", "gaussian", "reflect", "replicate", "circular"])
def test_tokenize_detokenize_roundtrip_is_padding_invariant(tokenizer_cls, mode, padding):
    checkpoint = generate_checkpoint()
    reference_checkpoint = generate_reference_checkpoint(checkpoint)

    tokenizer = tokenizer_cls(
        tokensize=5,
        device="cpu",
        mode=mode,
        reference_statedict=reference_checkpoint,
        padding=padding,
    )

    if padding == "gaussian":
        torch.manual_seed(0)

    tokenized = tokenizer.tokenize(checkpoint)

    if mode == "layer_wise":
        detokenized_checkpoint = tokenizer.detokenize(
            tokenized,
            reference_statedict=reference_checkpoint,
        )
    else:
        tokens, mask, position = tokenized
        detokenized_checkpoint = tokenizer.detokenize(
            tokens,
            mask,
            position,
            reference_statedict=reference_checkpoint,
        )

    for key in checkpoint:
        assert torch.equal(checkpoint[key], detokenized_checkpoint[key])
