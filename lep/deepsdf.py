"""Small DeepSDF-style autodecoder components."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


def clamp_sdf(sdf: torch.Tensor, clamp: Optional[float] = 0.1) -> torch.Tensor:
    """Clamp SDF targets/predictions to the narrow band used by DeepSDF."""
    if clamp is None or clamp <= 0:
        return sdf
    return torch.clamp(sdf, min=-float(clamp), max=float(clamp))


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "softplus":
        return nn.Softplus(beta=100)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _flatten_inputs(
    z: torch.Tensor, xyz: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape [B, L] or [L], got {tuple(z.shape)}")

    if xyz.ndim == 2:
        if xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz last dimension 3, got {tuple(xyz.shape)}")
        if z.shape[0] == 1:
            z_flat = z.expand(xyz.shape[0], -1)
        elif z.shape[0] == xyz.shape[0]:
            z_flat = z
        else:
            raise ValueError(
                "For xyz [N, 3], z must be [1, L], [L], or [N, L]; "
                f"got z {tuple(z.shape)} and xyz {tuple(xyz.shape)}"
            )
        return z_flat, xyz, (xyz.shape[0],)

    if xyz.ndim == 3:
        if xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz last dimension 3, got {tuple(xyz.shape)}")
        batch, count, _ = xyz.shape
        if z.shape[0] == 1:
            z = z.expand(batch, -1)
        elif z.shape[0] != batch:
            raise ValueError(
                "For xyz [B, N, 3], z must be [B, L] or [1, L]; "
                f"got z {tuple(z.shape)} and xyz {tuple(xyz.shape)}"
            )
        z_flat = z[:, None, :].expand(batch, count, z.shape[-1]).reshape(batch * count, -1)
        xyz_flat = xyz.reshape(batch * count, 3)
        return z_flat, xyz_flat, (batch, count)

    raise ValueError(f"Expected xyz with shape [N, 3] or [B, N, 3], got {tuple(xyz.shape)}")


class LatentDecoder(nn.Module):
    """Latent-conditioned MLP mapping `(z, x, y, z)` to an SDF value."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layers: Optional[Sequence[int]] = None,
        activation: str = "relu",
        xyz_dim: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must include at least one hidden layer and one output layer")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.skip_layers = tuple(int(i) for i in (skip_layers or ()))
        self.activation_name = activation
        self.xyz_dim = int(xyz_dim)

        input_dim = self.latent_dim + self.xyz_dim
        layers = []
        current_dim = input_dim
        for layer_idx in range(self.num_layers):
            if layer_idx in self.skip_layers and layer_idx != 0:
                current_dim += input_dim
            output_dim = self.hidden_dim if layer_idx < self.num_layers - 1 else 1
            layers.append(nn.Linear(current_dim, output_dim))
            current_dim = self.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.activation = _activation(activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, z: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        z_flat, xyz_flat, out_shape = _flatten_inputs(z, xyz)
        h0 = torch.cat([z_flat, xyz_flat], dim=-1)
        h = h0
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.skip_layers and layer_idx != 0:
                h = torch.cat([h, h0], dim=-1) / math.sqrt(2.0)
            h = layer(h)
            if layer_idx < len(self.layers) - 1:
                h = self.activation(h)
        return h.squeeze(-1).reshape(*out_shape)

    def config(self) -> Dict[str, object]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "skip_layers": list(self.skip_layers),
            "activation": self.activation_name,
            "xyz_dim": self.xyz_dim,
        }


class DeepSDFAutodecoder(nn.Module):
    """Decoder plus one optimized latent vector per train shape."""

    def __init__(
        self,
        num_shapes: int,
        latent_dim: int,
        decoder: Optional[LatentDecoder] = None,
        latent_init_std: float = 0.01,
        decoder_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.decoder = decoder or LatentDecoder(
            latent_dim=self.latent_dim, **(decoder_kwargs or {})
        )
        self.latent_codes = nn.Embedding(int(num_shapes), self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=float(latent_init_std))

    def forward(self, shape_indices: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        shape_indices = shape_indices.long()
        z = self.latent_codes(shape_indices)
        return self.decoder(z, xyz)

    def latent_l2(self, shape_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if shape_indices is None:
            z = self.latent_codes.weight
        else:
            z = self.latent_codes(shape_indices.long())
        return z.pow(2).sum(dim=-1).mean()

    def export_latents(self) -> torch.Tensor:
        return self.latent_codes.weight.detach().cpu().clone()


def make_decoder_from_config(config: Dict[str, object]) -> LatentDecoder:
    """Build a decoder from either a top-level or nested `deepsdf` config."""
    cfg = dict(config.get("deepsdf", config)) if isinstance(config, dict) else {}
    return LatentDecoder(
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 8)),
        skip_layers=cfg.get("skip_layers", ()),
        activation=str(cfg.get("activation", "relu")),
        xyz_dim=int(cfg.get("xyz_dim", 3)),
    )

