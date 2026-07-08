import torch
import torch.nn as nn

from sane.loss import GammaContrastReconLoss

def test_gamma_contrast_recon_loss():
    loss_fn = GammaContrastReconLoss()

    input_tokens = torch.randn(16, 20, 32)
    output_tokens = torch.randn(16, 20, 32)

    input_proj = torch.randn(16, 32)
    output_proj = torch.randn(16, 32)

    tensors_same = {
        'x': input_tokens, 
        'x_hat': input_tokens,
        'z_p_i': input_proj,
        'z_p_j': input_proj
    }

    loss_same = loss_fn(tensors_same)

    tensors_diff = {
        'x': input_tokens, 
        'x_hat': output_tokens,
        'z_p_i': input_proj,
        'z_p_j': output_proj
    }

    loss_diff = loss_fn(tensors_diff)

    # Check that the loss is not zero
    assert loss_diff['loss'].item() > 0, "Loss should be greater than zero for different inputs."
    assert loss_diff[loss_fn.name].item() > 0, "Gamma contrast reconstruction loss should be greater than zero for different inputs."
    assert loss_diff['loss_mse_reconstruction'].item() > 0, "MSE reconstruction loss should be greater than zero for different inputs."
    assert torch.allclose(loss_same['loss_mse_reconstruction'], torch.tensor(0.0)), "MSE reconstruction loss should be zero when inputs are the same."
    assert loss_diff['loss_contrastive'].item() > 0, "Contrastive loss should be greater than zero for different inputs."
    
    # Check that the gamma contrast term is computed correctly
    assert loss_diff['loss'].item() > loss_same['loss'].item(), "Total loss should be greater when inputs are different than when inputs are the same, but are respectively {} and {}.".format(loss_diff['loss'].item(), loss_same['loss'].item())

def test_gamma_contrast_recon_loss_gamma_zero():
    loss_fn = GammaContrastReconLoss(gamma=0.0)

    input_tokens = torch.randn(16, 20, 32)
    output_tokens = torch.randn(16, 20, 32)

    input_proj = torch.randn(16, 32)
    output_proj = torch.randn(16, 32)

    tensors = {
        'x': input_tokens, 
        'x_hat': output_tokens,
        'z_p_i': input_proj,
        'z_p_j': output_proj
    }

    loss = loss_fn(tensors)

    assert torch.allclose(loss['loss_contrastive'], torch.tensor(0.0)), "Contrastive loss should be zero when gamma is zero."
    assert torch.allclose(loss['loss'], loss['loss_mse_reconstruction']), "Total loss should equal MSE reconstruction loss when gamma is zero."

def test_gamma_contrast_recon_loss_gamma_one():
    loss_fn = GammaContrastReconLoss(gamma=1.0)

    input_tokens = torch.randn(16, 20, 32)
    output_tokens = torch.randn(16, 20, 32)

    input_proj = torch.randn(16, 32)
    output_proj = torch.randn(16, 32)

    tensors = {
        'x': input_tokens, 
        'x_hat': output_tokens,
        'z_p_i': input_proj,
        'z_p_j': output_proj
    }

    loss = loss_fn(tensors)

    assert torch.allclose(loss['loss_mse_reconstruction'], torch.tensor(0.0)), "MSE reconstruction loss should be zero when gamma is one."
    assert torch.allclose(loss['loss'], loss['loss_contrastive']), "Total loss should equal contrastive loss when gamma is one."