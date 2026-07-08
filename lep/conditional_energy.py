"""Observation-conditioned latent energy priors for sparse SDF observations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .latent_standardization import compute_latent_standardizer, standardize_latents
from .train_deepsdf import _torch_load, find_sdf_file, load_sdf_npz


CONDITIONAL_TRAINING_FIELDS = [
    "epoch",
    "loss",
    "em_loss",
    "rank_loss",
    "rank_context_loss",
    "rank_context_weight",
    "rank_context_margin",
    "langevin_loss",
    "energy_l2",
    "grad_norm",
    "target_norm",
    "energy_pos_mean",
    "energy_wrong_mean",
    "energy_wrong_context_mean",
    "energy_interp_mean",
    "energy_neg_init_mean",
    "energy_neg_langevin_mean",
    "num_valid_negatives",
    "neg_nearest_train_standardized_l2_mean",
    "neg_standardized_norm_mean",
    "observation_points",
]


class SparseSDFObservationEncoder(nn.Module):
    """DeepSets encoder for unordered sparse SDF observations."""

    SUPPORTED_POOLING_MODES = ("mean", "mean_max_logn")

    def __init__(
        self,
        point_hidden_size: int = 128,
        context_dim: int = 128,
        context_hidden_size: Optional[int] = None,
        pooling_mode: str = "mean_max_logn",
        observation_pooling: Optional[str] = None,
        logn_reference_points: int = 128,
    ) -> None:
        super().__init__()
        if point_hidden_size <= 0:
            raise ValueError("point_hidden_size must be positive")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        context_hidden_size = point_hidden_size if context_hidden_size is None else int(context_hidden_size)
        if context_hidden_size <= 0:
            raise ValueError("context_hidden_size must be positive")
        pooling_mode = str(observation_pooling or pooling_mode)
        if pooling_mode not in self.SUPPORTED_POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {self.SUPPORTED_POOLING_MODES}")
        if int(logn_reference_points) <= 1:
            raise ValueError("logn_reference_points must be greater than 1")

        pooled_dim = int(point_hidden_size) if pooling_mode == "mean" else 2 * int(point_hidden_size) + 1
        self.point_mlp = nn.Sequential(
            nn.Linear(4, int(point_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(point_hidden_size), int(point_hidden_size)),
            nn.SiLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(pooled_dim, int(context_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(context_hidden_size), int(context_dim)),
        )
        self.point_hidden_size = int(point_hidden_size)
        self.context_dim = int(context_dim)
        self.context_hidden_size = int(context_hidden_size)
        self.pooling_mode = str(pooling_mode)
        self.logn_reference_points = int(logn_reference_points)

    def _observation_tensor(self, points: torch.Tensor, sdf: Optional[torch.Tensor]) -> torch.Tensor:
        if sdf is None:
            observation = points
            if observation.ndim == 2:
                observation = observation.unsqueeze(0)
            if observation.ndim != 3 or observation.shape[-1] != 4:
                raise ValueError("combined observations must have shape [B, N, 4] or [N, 4]")
            return observation.float()

        obs_points = points
        obs_sdf = sdf
        if obs_points.ndim == 2:
            obs_points = obs_points.unsqueeze(0)
        if obs_sdf.ndim == 1:
            obs_sdf = obs_sdf.unsqueeze(0)
        if obs_sdf.ndim == 3 and obs_sdf.shape[-1] == 1:
            obs_sdf = obs_sdf.squeeze(-1)
        if obs_points.ndim != 3 or obs_points.shape[-1] != 3:
            raise ValueError("points must have shape [B, N, 3] or [N, 3]")
        if obs_sdf.ndim != 2 or obs_sdf.shape != obs_points.shape[:2]:
            raise ValueError("sdf must have shape [B, N] or [N] matching points")
        return torch.cat((obs_points.float(), obs_sdf.float().unsqueeze(-1)), dim=-1)

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "mean":
            return features.mean(dim=1)
        if self.pooling_mode == "mean_max_logn":
            mean_features = features.mean(dim=1)
            max_features = features.max(dim=1).values
            count = torch.as_tensor(float(features.shape[1]), dtype=features.dtype, device=features.device)
            reference = torch.as_tensor(
                float(self.logn_reference_points), dtype=features.dtype, device=features.device
            )
            log_count = (torch.log(count) / torch.log(reference)).expand(features.shape[0], 1)
            return torch.cat((mean_features, max_features, log_count), dim=-1)
        raise RuntimeError(f"Unsupported pooling mode: {self.pooling_mode}")

    def forward(self, points: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        observation = self._observation_tensor(points, sdf)
        if int(observation.shape[1]) <= 0:
            raise ValueError("observation must contain at least one point")
        features = self.point_mlp(observation.reshape(-1, 4)).reshape(
            int(observation.shape[0]), int(observation.shape[1]), -1
        )
        return self.context_mlp(self._pool(features))

    def config(self) -> Dict[str, object]:
        return {
            "point_hidden_size": self.point_hidden_size,
            "context_dim": self.context_dim,
            "context_hidden_size": self.context_hidden_size,
            "pooling_mode": self.pooling_mode,
            "observation_pooling": self.pooling_mode,
            "logn_reference_points": self.logn_reference_points,
        }


class ConditionalEnergyMLP(nn.Module):
    """Scalar energy ``E(z_tilde, c(O))`` for standardized latents."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if int(context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_layers) < 2:
            raise ValueError("num_layers must include hidden layers plus scalar output")

        layers: List[nn.Module] = []
        current_dim = int(latent_dim) + int(context_dim)
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

    def forward(self, z_tilde: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if z_tilde.ndim == 1:
            z_tilde = z_tilde.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if z_tilde.ndim != 2 or z_tilde.shape[-1] != self.latent_dim:
            raise ValueError(f"z_tilde must have shape [B, {self.latent_dim}]")
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"context must have shape [B, {self.context_dim}]")
        if z_tilde.shape[0] != context.shape[0]:
            if z_tilde.shape[0] == 1:
                z_tilde = z_tilde.expand(context.shape[0], -1)
            elif context.shape[0] == 1:
                context = context.expand(z_tilde.shape[0], -1)
            else:
                raise ValueError("z_tilde and context batch sizes must match, or one batch size must be 1")
        return self.net(torch.cat((z_tilde, context.to(device=z_tilde.device, dtype=z_tilde.dtype)), dim=-1)).squeeze(-1)

    def config(self) -> Dict[str, object]:
        return {
            "latent_dim": self.latent_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


class ConditionalLatentEnergyPrior(nn.Module):
    """Conditional energy prior over raw decoder latents and sparse observations."""

    def __init__(
        self,
        observation_encoder: SparseSDFObservationEncoder,
        energy_model: ConditionalEnergyMLP,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if int(latent_mean.numel()) != int(energy_model.latent_dim):
            raise ValueError("latent_mean dimension must match energy latent_dim")
        if int(latent_std.numel()) != int(energy_model.latent_dim):
            raise ValueError("latent_std dimension must match energy latent_dim")
        if observation_encoder.context_dim != energy_model.context_dim:
            raise ValueError("encoder context_dim must match energy context_dim")
        self.observation_encoder = observation_encoder
        self.energy_model = energy_model
        self.register_buffer("latent_mean", latent_mean.detach().float().view(1, -1).clone())
        self.register_buffer("latent_std", latent_std.detach().float().view(1, -1).clamp_min(1e-6).clone())

    @property
    def latent_dim(self) -> int:
        return int(self.energy_model.latent_dim)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        return standardize_latents(z, self.latent_mean, self.latent_std)

    def context(self, points: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.observation_encoder(points, sdf)

    def energy(self, z: torch.Tensor, points: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_tilde = self.standardize(z)
        context = self.context(points, sdf).to(device=z_tilde.device, dtype=z_tilde.dtype)
        return self.energy_model(z_tilde, context)

    def forward(self, z: torch.Tensor, points: torch.Tensor, sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.energy(z, points, sdf)


def _nearest_train_distances(z: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(z.detach(), reference.detach())
    return distances.min(dim=1).values


def _sample_negative_initialization(
    z_data: torch.Tensor,
    noise_scales: Sequence[float],
    data_noise: float,
) -> torch.Tensor:
    if torch.rand((), device=z_data.device).item() < 0.5:
        scale_index = int(torch.randint(0, len(noise_scales), (), device=z_data.device).item())
        return torch.randn_like(z_data) * float(noise_scales[scale_index])
    perm = torch.randperm(z_data.shape[0], device=z_data.device)
    return z_data[perm] + torch.randn_like(z_data) * float(data_noise)


def _clamp_norm(z: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return z * scale


def run_conditional_langevin_negatives(
    prior: ConditionalLatentEnergyPrior,
    z_neg_init: torch.Tensor,
    context: torch.Tensor,
    steps: int,
    step_size: float,
    noise_scale: float,
    clamp_norm: float,
) -> torch.Tensor:
    """Run Langevin descent on ``E(z, c)`` for fixed contexts in standardized latent space."""
    z = _clamp_norm(z_neg_init.detach(), float(clamp_norm))
    context = context.detach()
    noise_multiplier = float(noise_scale) * math.sqrt(2.0 * float(step_size))
    for _ in range(int(steps)):
        z = z.detach().requires_grad_(True)
        energy = prior.energy_model(z, context).sum()
        grad = torch.autograd.grad(energy, z, create_graph=False)[0]
        z = z - float(step_size) * grad
        if noise_multiplier > 0.0:
            z = z + noise_multiplier * torch.randn_like(z)
        z = _clamp_norm(z.detach(), float(clamp_norm))
    return z.detach()


def conditional_energy_training_step(
    prior: ConditionalLatentEnergyPrior,
    optimizer: torch.optim.Optimizer,
    z_data: torch.Tensor,
    observation_points: torch.Tensor,
    observation_sdf: torch.Tensor,
    z_reference: torch.Tensor,
    *,
    ot_weight: float = 1.0,
    rank_weight: float = 1.0,
    rank_margin: float = 1.0,
    rank_context_weight: float = 1.0,
    rank_context_margin: Optional[float] = None,
    langevin_weight: float = 1.0,
    langevin_margin: float = 10.0,
    energy_l2_weight: float = 1e-5,
    negative_noise_scales: Sequence[float] = (1.0, 3.0, 6.0, 10.0),
    negative_data_noise: float = 0.5,
    langevin_steps: int = 20,
    langevin_step_size: float = 1e-3,
    langevin_noise_scale: float = 1.0,
    negative_clamp_norm: float = 150.0,
    min_negative_train_distance: float = 2.0,
) -> Dict[str, float]:
    """One optimization step for Conditional Energy Matching."""
    if z_data.ndim != 2:
        raise ValueError("z_data must have shape [B, D]")
    if z_data.shape[0] <= 0:
        raise ValueError("z_data must be non-empty")
    if z_data.shape[0] == 1 and (rank_weight > 0.0 or rank_context_weight > 0.0):
        raise ValueError("rank losses need batch size > 1")
    resolved_context_margin = float(rank_margin if rank_context_margin is None else rank_context_margin)

    z_data = z_data.detach()
    context = prior.context(observation_points, observation_sdf).to(device=z_data.device, dtype=z_data.dtype)

    z0 = torch.randn_like(z_data)
    t = torch.rand((z_data.shape[0], 1), dtype=z_data.dtype, device=z_data.device)
    z_t = ((1.0 - t) * z0 + t * z_data).detach().requires_grad_(True)
    energy_interp = prior.energy_model(z_t, context).sum()
    grad = torch.autograd.grad(energy_interp, z_t, create_graph=True)[0]
    target = -(z_data - z0)
    em_loss = torch.mean((grad - target) ** 2)

    z_wrong = torch.roll(z_data, shifts=1, dims=0)
    context_wrong = torch.roll(context, shifts=1, dims=0)
    e_pos = prior.energy_model(z_data, context)
    e_wrong = prior.energy_model(z_wrong, context)
    e_wrong_context = prior.energy_model(z_data, context_wrong)
    rank_loss = F.softplus(e_pos + float(rank_margin) - e_wrong).mean()
    rank_context_loss = F.softplus(e_pos + resolved_context_margin - e_wrong_context).mean()
    energy_l2_terms = [e_pos.pow(2).mean(), e_wrong.pow(2).mean(), e_wrong_context.pow(2).mean()]

    z_neg_init = None
    z_neg = None
    neg_nearest = None
    valid_negative_mask = None
    langevin_loss = e_pos.new_tensor(0.0)
    if float(langevin_weight) > 0.0:
        z_neg_init = _sample_negative_initialization(
            z_data,
            noise_scales=tuple(float(value) for value in negative_noise_scales),
            data_noise=float(negative_data_noise),
        )
        z_neg = run_conditional_langevin_negatives(
            prior=prior,
            z_neg_init=z_neg_init,
            context=context,
            steps=int(langevin_steps),
            step_size=float(langevin_step_size),
            noise_scale=float(langevin_noise_scale),
            clamp_norm=float(negative_clamp_norm),
        )
        reference = z_reference.to(device=z_data.device, dtype=z_data.dtype)
        neg_nearest = _nearest_train_distances(z_neg, reference)
        valid_negative_mask = neg_nearest >= float(min_negative_train_distance)
        e_neg_all = prior.energy_model(z_neg, context)
        if bool(valid_negative_mask.any().item()):
            e_neg = e_neg_all[valid_negative_mask]
            langevin_loss = F.softplus(e_pos.mean() + float(langevin_margin) - e_neg).mean()
            energy_l2_terms.append(e_neg.pow(2).mean())

    energy_l2 = torch.stack(energy_l2_terms).mean()
    loss = (
        float(ot_weight) * em_loss
        + float(rank_weight) * rank_loss
        + float(rank_context_weight) * rank_context_loss
        + float(langevin_weight) * langevin_loss
        + float(energy_l2_weight) * energy_l2
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        return {
            "loss": float(loss.detach().cpu()),
            "em_loss": float(em_loss.detach().cpu()),
            "rank_loss": float(rank_loss.detach().cpu()),
            "rank_context_loss": float(rank_context_loss.detach().cpu()),
            "rank_context_weight": float(rank_context_weight),
            "rank_context_margin": float(resolved_context_margin),
            "langevin_loss": float(langevin_loss.detach().cpu()),
            "energy_l2": float(energy_l2.detach().cpu()),
            "grad_norm": float(grad.detach().norm(dim=1).mean().cpu()),
            "target_norm": float(target.detach().norm(dim=1).mean().cpu()),
            "energy_pos_mean": float(e_pos.detach().mean().cpu()),
            "energy_wrong_mean": float(e_wrong.detach().mean().cpu()),
            "energy_wrong_context_mean": float(e_wrong_context.detach().mean().cpu()),
            "energy_interp_mean": float(prior.energy_model(z_t.detach(), context).mean().cpu()),
            "energy_neg_init_mean": (
                float(prior.energy_model(z_neg_init.detach(), context).mean().cpu())
                if z_neg_init is not None
                else 0.0
            ),
            "energy_neg_langevin_mean": (
                float(prior.energy_model(z_neg.detach(), context).mean().cpu()) if z_neg is not None else 0.0
            ),
            "num_valid_negatives": (
                float(valid_negative_mask.sum().cpu()) if valid_negative_mask is not None else 0.0
            ),
            "neg_nearest_train_standardized_l2_mean": (
                float(neg_nearest.mean().cpu()) if neg_nearest is not None else 0.0
            ),
            "neg_standardized_norm_mean": float(z_neg.norm(dim=1).mean().cpu()) if z_neg is not None else 0.0,
            "observation_points": float(observation_points.shape[1]),
        }


def load_deepsdf_train_latents(checkpoint_path: str) -> Tuple[torch.Tensor, List[str], Dict[str, object]]:
    checkpoint = _torch_load(Path(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"DeepSDF checkpoint must be a dict: {checkpoint_path}")
    if "train_latents" not in checkpoint or "train_shape_ids" not in checkpoint:
        raise KeyError(f"{checkpoint_path} must contain train_latents and train_shape_ids")
    latents = torch.as_tensor(checkpoint["train_latents"], dtype=torch.float32)
    shape_ids = [str(shape_id) for shape_id in checkpoint["train_shape_ids"]]
    if latents.ndim != 2 or latents.shape[0] != len(shape_ids):
        raise ValueError("train_latents must be [N, D] and match train_shape_ids")
    return latents, shape_ids, checkpoint


def _sample_training_batch(
    train_shape_ids: Sequence[str],
    train_latents: torch.Tensor,
    sdf_root: str,
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    observation_counts: Sequence[int],
    rng: np.random.Generator,
    device: str,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    obs_count = int(observation_counts[int(rng.integers(0, len(observation_counts)))])
    positions = rng.choice(
        len(train_shape_ids),
        size=int(batch_size),
        replace=int(batch_size) > len(train_shape_ids),
    )
    points_batch: List[np.ndarray] = []
    sdf_batch: List[np.ndarray] = []
    latent_batch: List[torch.Tensor] = []
    for position in positions:
        shape_id = str(train_shape_ids[int(position)])
        if shape_id not in cache:
            cache[shape_id] = load_sdf_npz(str(find_sdf_file(sdf_root, shape_id)))
        points, sdf = cache[shape_id]
        replace = int(obs_count) > int(points.shape[0])
        sample_indices = rng.choice(points.shape[0], size=obs_count, replace=replace)
        points_batch.append(points[sample_indices])
        sdf_batch.append(sdf[sample_indices])
        latent_batch.append(train_latents[int(position)])

    raw_z = torch.stack(latent_batch, dim=0)
    z_data = standardize_latents(raw_z, z_mean, z_std).to(device)
    obs_points = torch.as_tensor(np.stack(points_batch), dtype=torch.float32, device=device)
    obs_sdf = torch.as_tensor(np.stack(sdf_batch), dtype=torch.float32, device=device)
    return z_data, obs_points, obs_sdf, obs_count


def train_conditional_energy(
    config: Dict[str, object],
    checkpoint: str,
    sdf_root: str,
    out_dir: str,
    device: str = "cpu",
    epochs_override: Optional[int] = None,
) -> Path:
    """Train a conditional energy prior from a frozen DeepSDF checkpoint."""
    cfg = dict(config.get("conditional_energy", config))
    if epochs_override is not None:
        cfg["epochs"] = int(epochs_override)

    train_latents, train_shape_ids, checkpoint_payload = load_deepsdf_train_latents(checkpoint)
    latent_mean, latent_std = compute_latent_standardizer(train_latents, eps=float(cfg.get("standardize_eps", 1e-6)))
    z_reference = standardize_latents(train_latents, latent_mean, latent_std).to(device)
    latent_dim = int(train_latents.shape[1])

    observation_encoder = SparseSDFObservationEncoder(
        point_hidden_size=int(cfg.get("point_hidden_size", 128)),
        context_dim=int(cfg.get("context_dim", 128)),
        context_hidden_size=int(cfg.get("context_hidden_size", cfg.get("point_hidden_size", 128))),
        pooling_mode=str(cfg.get("observation_pooling", cfg.get("pooling_mode", "mean_max_logn"))),
        logn_reference_points=int(cfg.get("logn_reference_points", 128)),
    )
    energy_model = ConditionalEnergyMLP(
        latent_dim=latent_dim,
        context_dim=observation_encoder.context_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 4)),
    )
    prior = ConditionalLatentEnergyPrior(
        observation_encoder=observation_encoder,
        energy_model=energy_model,
        latent_mean=latent_mean,
        latent_std=latent_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-6)),
    )

    observation_counts = [int(value) for value in cfg.get("observation_points", [16, 32, 64, 128])]
    if not observation_counts or any(value <= 0 for value in observation_counts):
        raise ValueError("conditional_energy.observation_points must be a non-empty list of positive ints")
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 2000))
    seed = int(config.get("seed", cfg.get("seed", 13)))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_row: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        z_data, obs_points, obs_sdf, obs_count = _sample_training_batch(
            train_shape_ids=train_shape_ids,
            train_latents=train_latents,
            sdf_root=sdf_root,
            cache=cache,
            batch_size=batch_size,
            observation_counts=observation_counts,
            rng=rng,
            device=device,
            z_mean=latent_mean,
            z_std=latent_std,
        )
        row = conditional_energy_training_step(
            prior=prior,
            optimizer=optimizer,
            z_data=z_data,
            observation_points=obs_points,
            observation_sdf=obs_sdf,
            z_reference=z_reference,
            ot_weight=float(cfg.get("ot_weight", 1.0)),
            rank_weight=float(cfg.get("rank_weight", 1.0)),
            rank_margin=float(cfg.get("rank_margin", 1.0)),
            rank_context_weight=float(cfg.get("rank_context_weight", 1.0)),
            rank_context_margin=cfg.get("rank_context_margin", None),
            langevin_weight=float(cfg.get("langevin_weight", 1.0)),
            langevin_margin=float(cfg.get("langevin_margin", 10.0)),
            energy_l2_weight=float(cfg.get("energy_l2_weight", 1e-5)),
            negative_noise_scales=tuple(float(value) for value in cfg.get("negative_noise_scales", [1.0, 3.0, 6.0, 10.0])),
            negative_data_noise=float(cfg.get("negative_data_noise", 0.5)),
            langevin_steps=int(cfg.get("langevin_steps", 20)),
            langevin_step_size=float(cfg.get("langevin_step_size", 1e-3)),
            langevin_noise_scale=float(cfg.get("langevin_noise_scale", 1.0)),
            negative_clamp_norm=float(cfg.get("negative_clamp_norm", 150.0)),
            min_negative_train_distance=float(cfg.get("min_negative_train_distance", 2.0)),
        )
        row["epoch"] = float(epoch)
        row["observation_points"] = float(obs_count)
        history.append(row)
        if float(row["loss"]) < best_loss:
            best_loss = float(row["loss"])
            best_epoch = int(epoch)
            best_row = dict(row)
            best_state = {key: value.detach().cpu().clone() for key, value in prior.state_dict().items()}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "conditional_energy": cfg,
        "checkpoint": checkpoint,
        "sdf_root": sdf_root,
        "latent_dim": latent_dim,
        "num_train_latents": int(train_latents.shape[0]),
        "train_shape_ids": train_shape_ids,
        "deepsdf_config": checkpoint_payload.get("config", {}),
        "encoder_config": observation_encoder.config(),
        "energy_config": energy_model.config(),
        "latent_stats": {
            "standardized": True,
            "latent_mean": latent_mean.cpu().tolist(),
            "latent_std": latent_std.cpu().tolist(),
            "standardize_eps": float(cfg.get("standardize_eps", 1e-6)),
        },
    }
    checkpoint_path = out_path / "conditional_energy_prior.pt"
    save_conditional_energy_prior(str(checkpoint_path), prior.cpu(), resolved_config, history)
    if best_state is not None:
        prior.load_state_dict(best_state)
        save_conditional_energy_prior(
            str(out_path / "conditional_energy_prior_best_loss.pt"),
            prior.cpu(),
            {**resolved_config, "selected_checkpoint": "best_training_loss", "best_epoch": best_epoch},
            history,
        )
    with (out_path / "conditional_energy_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONDITIONAL_TRAINING_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(out_path / "conditional_energy_prior_best_loss.pt"),
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "best_row": best_row,
                "num_train_latents": int(train_latents.shape[0]),
                "latent_dim": latent_dim,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return checkpoint_path


def save_conditional_energy_prior(
    path: str,
    prior: ConditionalLatentEnergyPrior,
    config: Dict[str, object],
    history: Optional[List[Dict[str, float]]] = None,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation_encoder_state_dict": prior.observation_encoder.state_dict(),
            "energy_state_dict": prior.energy_model.state_dict(),
            "encoder_config": prior.observation_encoder.config(),
            "energy_config": prior.energy_model.config(),
            "latent_stats": {
                "standardized": True,
                "latent_mean": prior.latent_mean.detach().cpu().reshape(-1).tolist(),
                "latent_std": prior.latent_std.detach().cpu().reshape(-1).tolist(),
            },
            "config": config,
            "history": history or [],
        },
        path_obj,
    )


def load_conditional_energy_prior(
    path: str,
    map_location: Optional[str] = None,
) -> Tuple[ConditionalLatentEnergyPrior, Dict[str, object]]:
    checkpoint = _torch_load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Conditional energy checkpoint must be a dict: {path}")
    encoder = SparseSDFObservationEncoder(**checkpoint["encoder_config"])
    energy_model = ConditionalEnergyMLP(**checkpoint["energy_config"])
    stats = checkpoint.get("latent_stats", {})
    latent_mean = torch.as_tensor(stats["latent_mean"], dtype=torch.float32).view(1, -1)
    latent_std = torch.as_tensor(stats["latent_std"], dtype=torch.float32).view(1, -1)
    prior = ConditionalLatentEnergyPrior(
        observation_encoder=encoder,
        energy_model=energy_model,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    prior.observation_encoder.load_state_dict(checkpoint["observation_encoder_state_dict"])
    prior.energy_model.load_state_dict(checkpoint["energy_state_dict"])
    prior.eval()
    return prior, checkpoint
