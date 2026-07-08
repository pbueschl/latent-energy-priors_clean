"""Latent standardization helpers shared by post-hoc priors."""

from __future__ import annotations

from typing import Tuple

import torch


def compute_latent_standardizer(
    latents: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-dimension mean/std for train-latent standardization."""
    latents = latents.detach().float()
    mean = latents.mean(dim=0)
    std = latents.std(dim=0, unbiased=False).clamp_min(float(eps))
    return mean, std


def standardize_latents(
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a saved train-latent standardizer."""
    mean = mean.to(device=latents.device, dtype=latents.dtype)
    std = std.to(device=latents.device, dtype=latents.dtype).clamp_min(float(eps))
    return (latents - mean) / std
