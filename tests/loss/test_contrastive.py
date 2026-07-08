import torch
import torch.nn as nn

from sane.loss import NTXentLoss

def test_ntxent_loss():
    loss_fn = NTXentLoss()
    input_tensor = torch.randn(16, 32)
    output_tensor = torch.randn(16, 32)

    loss_same = loss_fn({
        'z_p_i': input_tensor,
        'z_p_j': input_tensor
    })

    loss = loss_fn({
        'z_p_i': input_tensor,
        'z_p_j': output_tensor
    })

    assert loss['loss'].item() > 0., "Loss should not be zero when both inputs are different but is {}.".format(loss['loss'].item())
    assert loss[loss_fn.name] > 0., "Loss should not be zero when both inputs are different but is {}.".format(loss[loss_fn.name].item())

    assert loss['loss'].item() > loss_same['loss'].item(), "Loss should be greater when inputs are different than when inputs are the same, but are respectively {} and {}.".format(loss['loss'].item(), loss_same['loss'].item())
