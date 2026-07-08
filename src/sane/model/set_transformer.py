import torch
import torch.nn as nn
from typing import Optional


class MAB(nn.Module):
    """Multi-head Attention Block:  LayerNorm(Q + MHA(Q,K,K)) → LayerNorm(· + FF(·))"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask)
        h = self.norm1(Q + h)
        return self.norm2(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block — O(nM) attention via M inducing points."""

    def __init__(self, d_model: int, nhead: int, n_inducing: int = 32, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, nhead, dropout)
        self.mab2 = MAB(d_model, nhead, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        I = self.I.expand(X.size(0), -1, -1)
        H = self.mab1(I, X, key_padding_mask)   # (B, M, d)
        return self.mab2(X, H)                   # (B, n, d)


class PMA(nn.Module):
    """Pooling by Multihead Attention — collapses a set to k seed vectors."""

    def __init__(self, d_model: int, nhead: int, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.mab = MAB(d_model, nhead, dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        S = self.S.expand(Z.size(0), -1, -1)
        return self.mab(S, Z)                    # (B, k, d)


class SetTransformerEncoder(nn.Module):
    """Set Transformer encoder (and optional decoder head).

    Default (encoder-only): projection → ISAB×L → PMA(k=1) → output projection,
    producing a single set embedding of shape (B, output_dim).

    Optional decoder mode (to match the original `scripts/models_set_deep.py` SetTransformer):
    projection → ISAB×L → PMA(k=num_outputs) → SAB → SAB → Linear,
    producing (B, num_outputs, dim_output).
    """
 
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        n_inducing: int = 32,
        output_dim: int = 256,
        dropout: float = 0.0,
        *,
        use_decoder: bool = False,
        num_outputs: int = 1,
        dim_output: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.isabs = nn.ModuleList([ISAB(d_model, nhead, n_inducing, dropout) for _ in range(num_layers)])
        self.use_decoder = bool(use_decoder)

        if self.use_decoder:
            if num_outputs < 1:
                raise ValueError(f"num_outputs must be >= 1, got {num_outputs}")
            if dim_output is None:
                raise ValueError("dim_output must be set when use_decoder=True")
            self.pma = PMA(d_model, nhead, k=num_outputs, dropout=dropout)
            self.sab1 = MAB(d_model, nhead, dropout)
            self.sab2 = MAB(d_model, nhead, dropout)
            self.output_proj = nn.Linear(d_model, dim_output)
            self.output_dim = dim_output
            self.num_outputs = num_outputs
        else:
            self.pma = PMA(d_model, nhead, k=1, dropout=dropout)
            self.output_proj = nn.Linear(d_model, output_dim)
            self.output_dim = output_dim
            self.num_outputs = 1

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, n, feat_dim)
            key_padding_mask: (B, n), True = ignore
        Returns:
            - if use_decoder=False: (B, output_dim)
            - if use_decoder=True: (B, num_outputs, dim_output)
        """
        x = self.input_proj(x)
        for isab in self.isabs:
            x = isab(x, key_padding_mask)

        z = self.pma(x)
        if self.use_decoder:
            z = self.sab1(z, z)
            z = self.sab2(z, z)
            return self.output_proj(z)

        z = z.squeeze(1)  # (B, d_model)
        return self.output_proj(z)
