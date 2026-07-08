import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from sane.loss.loss import SANELoss

class NTXentLoss(SANELoss):
    r"""
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent Loss) for contrastive learning.

    This loss function is used in contrastive learning to maximize the agreement between
    differently augmented views of the same data example while minimizing the agreement
    between augmented views of different data examples.

    Args:
        name (str, optional): Name of the loss function. Default: ``"loss_contrastive"``
        temperature (float, optional): Scaling factor for the similarity scores. Default: ``0.1``

    Attributes:
        temperature (float): Scaling factor for the similarity scores.
        criterion (nn.CrossEntropyLoss): Cross entropy loss function.
        similarity_f (nn.CosineSimilarity): Cosine similarity function.
        mask (torch.Tensor): Mask to exclude correlated samples from the similarity matrix.

    Shape:
        - Input: Dict[str, torch.Tensor] with keys ``'z_p_i'`` and ``'z_p_j'``, each of shape :math:`(B, D)` where
          :math:`B` is the batch size and :math:`D` is the dimension of the representations.
        - Output: Dict[str, torch.Tensor] with keys ``'loss'`` and the name of the loss function, each containing a scalar loss value.

    Examples:
        >>> loss_fn = NTXentLoss()
        >>> z_p_i = torch.randn(32, 128)
        >>> z_p_j = torch.randn(32, 128)
        >>> tensors = {'z_p_i': z_p_i, 'z_p_j': z_p_j}
        >>> loss = loss_fn(tensors)
    """
    def __init__(self, name: str = "loss_contrastive", temperature: float = 0.1) -> None:
        super(NTXentLoss, self).__init__(name)
        self.temperature = temperature
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f: nn.CosineSimilarity = nn.CosineSimilarity(dim=2)
        self.mask: Optional[torch.Tensor] = None

    def __mask_correlated_samples(self, batch_size: int) -> torch.Tensor:
        r"""
        Creates a mask to exclude correlated samples from the similarity matrix.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Mask of shape :math:`(2 \times \text{batch_size}, 2 \times \text{batch_size})`.
        """
        N = 2 * batch_size
        if self.mask is not None and self.mask.shape[0] == N:
            return self.mask

        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        self.mask = mask
        return mask

    def forward(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Computes the NT-Xent loss.

        Args:
            tensors (Dict[str, torch.Tensor]): Dictionary containing the tensors ``'z_p_i'`` and ``'z_p_j'``.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss values with keys ``'loss'`` and the name of the loss function.
        """
        z_i = tensors['z_p_i']  # shape: (B, D)
        z_j = tensors['z_p_j']  # shape: (B, D)
        batch_size = z_i.shape[0]

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        mask = self.__mask_correlated_samples(batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return {'loss': loss, self.name: loss}
