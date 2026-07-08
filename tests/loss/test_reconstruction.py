import torch
import torch.nn as nn

from sane.loss import MSEReconstructionLoss

def test_mse_reconstruction_loss_on_same_input():
    loss_fn = MSEReconstructionLoss()
    input_tensor = torch.randn(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    tensors = {'x': input_tensor, 'x_hat': input_tensor, 'mask': mask}
    loss = loss_fn(tensors)
    assert torch.allclose(loss['loss'], torch.tensor(0.0)), "Loss should be zero when input and output are the same."
    assert torch.allclose(loss[loss_fn.name], torch.tensor(0.0)), "MSE should be zero when input and output are the same."
    assert torch.allclose(loss[loss_fn.name + '_rsq'], torch.tensor(1.0)), "R-squared score should be 1 when input and output are the same."

def test_mse_reconstruction_loss_on_different_input():
    loss_fn = MSEReconstructionLoss()
    input_tensor = torch.randn(3, 5)
    output_tensor = torch.randn(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    tensors = {'x': input_tensor, 'x_hat': output_tensor, 'mask': mask}
    loss = loss_fn(tensors)
    
    # Check that the loss is not zero
    assert loss['loss'].item() > 0, "Loss should be greater than zero for different inputs."
    assert loss[loss_fn.name].item() > 0, "MSE should be greater than zero for different inputs."
    
    # Check that R-squared score is less than 1
    assert loss[loss_fn.name + '_rsq'].item() < 1, "R-squared score should be less than 1 for different inputs."

def test_mse_reconstruction_loss_on_inputs_different_only_where_masked():
    loss_fn = MSEReconstructionLoss()
    input_tensor = torch.randn(3, 5)
    output_tensor = input_tensor.clone()
    output_tensor[-1, -1] += 1.0  # Change one element
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[-1, -1] = 0  # Mask the changed element
    tensors = {'x': input_tensor, 'x_hat': output_tensor, 'mask': mask}
    
    loss = loss_fn(tensors)
    
    # Check that the loss is zero since the masked element is not considered
    assert torch.allclose(loss['loss'], torch.tensor(0.0)), "Loss should be zero when masked elements are not considered."
    assert torch.allclose(loss[loss_fn.name], torch.tensor(0.0)), "MSE should be zero when masked elements are not considered."
    assert torch.allclose(loss[loss_fn.name + '_rsq'], torch.tensor(1.0)), "R-squared score should be 1 when masked elements are not considered."