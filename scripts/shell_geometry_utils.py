#!/usr/bin/env python3
from __future__ import annotations
import torch


def compute_sphere_center(z: torch.Tensor, center_mode: str = "origin") -> torch.Tensor | None:
    if center_mode == "origin":
        return None
    if center_mode == "mean":
        z = z.unsqueeze(0) if z.dim() == 2 else z
        return z.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown center mode: {center_mode}")


def compute_per_token_norms(z: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
    z = z.unsqueeze(0) if z.dim() == 2 else z
    return (z if center is None else z - center).norm(dim=-1)


def mean_token_norm(z: torch.Tensor, center_mode: str = "origin") -> float:
    return float(compute_per_token_norms(z, compute_sphere_center(z, center_mode)).mean().item())


def project_to_shell(z: torch.Tensor, target_norms: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
    zc = z if center is None else z - center
    out = zc * (target_norms / zc.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return out if center is None else out + center


def scale_toward_radius(z: torch.Tensor, target_radius: float, alpha: float, center: torch.Tensor | None = None) -> torch.Tensor:
    zc = z if center is None else z - center
    current = zc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = torch.full_like(current, float(target_radius))
    out = zc * (((1.0 - float(alpha)) * current + float(alpha) * target) / current)
    return out if center is None else out + center
