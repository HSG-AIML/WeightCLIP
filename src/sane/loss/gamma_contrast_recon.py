from sane.loss.loss import CompositeLoss
from sane.loss.reconstruction import MSEReconstructionLoss
from sane.loss.contrastive import NTXentLoss

class GammaContrastReconLoss(CompositeLoss):
    def __init__(
        self, 
        name: str = "loss_gamma_contrast_recon",
        gamma: float = 0.05,
        temperature: float = 0.1,
        reduction: str = "mean",
        normalize: str = 'token',
        std_eps: float = 1e-6
    ):
        """
        A simple composite loss function that combines NT-Xent contrastive loss and MSE reconstruction loss.
        The NT-Xent loss is weighted by a gamma factor, and the MSE reconstruction loss by (1 - gamma).

        Args:
            name (str): The name of the loss function. Default is "loss_gamma_contrast_recon".
            gamma (float): The gamma value used to weight the contrastive loss. Default is 0.05.
            temperature (float): The temperature for the contrastive loss. Default is 0.1.
            reduction (str): Specifies the reduction to apply to the output of the MSE reconstruction loss: 'none' | 'mean' | 'sum'. Default is 'mean'.
            normalize (bool): What normalization strategy to use in the MSE reconstruction loss. Can be None, 'token', 'window' or 'batch'. Default: 'token'
            std_eps (float): A small value added to the standard deviation to avoid division by zero in the MSE reconstruction loss. Default is 1e-6.
        """
        losses = [NTXentLoss(temperature=temperature), MSEReconstructionLoss(reduction=reduction, normalize=normalize, std_eps=std_eps)]
        ratios = [gamma]
        super(GammaContrastReconLoss, self).__init__(name, losses, ratios)