"""Post-hoc Gaussian mixture latent priors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .conditional_energy import load_deepsdf_train_latents
from .train_deepsdf import _torch_load


SUPPORTED_COVARIANCE_TYPES = ("diag", "full")


class GaussianMixtureLatentPrior(nn.Module):
    """Differentiable torch evaluator for a fitted global latent GMM."""

    def __init__(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        covariance_type: str = "diag",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        covariance_type = str(covariance_type).lower()
        if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}, got {covariance_type!r}"
            )
        weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        means = torch.as_tensor(means, dtype=torch.float32)
        covariances = torch.as_tensor(covariances, dtype=torch.float32)
        if weights.ndim != 1 or means.ndim != 2 or weights.shape[0] != means.shape[0]:
            raise ValueError("weights must be [K] and means must be [K, D]")
        if torch.any(weights <= 0):
            raise ValueError("GMM weights must be positive")
        if covariance_type == "diag":
            if covariances.shape != means.shape:
                raise ValueError("diag covariances must have shape [K, D]")
            if torch.any(covariances <= 0):
                raise ValueError("diag covariances must be positive")
        else:
            expected = (int(means.shape[0]), int(means.shape[1]), int(means.shape[1]))
            if tuple(covariances.shape) != expected:
                raise ValueError(f"full covariances must have shape {expected}")

        self.covariance_type = covariance_type
        self.latent_dim = int(means.shape[1])
        self.num_components = int(means.shape[0])
        self.eps = float(eps)
        self.register_buffer("weights", weights / weights.sum())
        self.register_buffer("means", means)
        self.register_buffer("covariances", covariances)

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Return per-component log probabilities with shape ``[B, K]``."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have shape [B, {self.latent_dim}]")
        z = z.to(device=self.means.device, dtype=self.means.dtype)
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)
        const = float(self.latent_dim) * math.log(2.0 * math.pi)
        if self.covariance_type == "diag":
            variances = self.covariances.clamp_min(self.eps)
            log_det = torch.log(variances).sum(dim=-1)
            quad = diff.pow(2).div(variances.unsqueeze(0)).sum(dim=-1)
        else:
            covariances = self.covariances + torch.eye(
                self.latent_dim,
                dtype=self.covariances.dtype,
                device=self.covariances.device,
            ).unsqueeze(0) * self.eps
            chol = torch.linalg.cholesky(covariances)
            diff_t = diff.permute(1, 2, 0)
            solved = torch.linalg.solve_triangular(chol, diff_t, upper=False)
            quad = solved.pow(2).sum(dim=1).transpose(0, 1)
            log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        return -0.5 * (const + log_det.unsqueeze(0) + quad)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_weights = torch.log(self.weights.clamp_min(self.eps)).to(
            device=self.means.device,
            dtype=self.means.dtype,
        )
        return torch.logsumexp(self.component_log_prob(z) + log_weights.unsqueeze(0), dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.log_prob(z)


def save_gmm_prior(
    path: str,
    prior: GaussianMixtureLatentPrior,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prior_type": "gaussian_mixture_latent_prior",
            "weights": prior.weights.detach().cpu(),
            "means": prior.means.detach().cpu(),
            "covariances": prior.covariances.detach().cpu(),
            "covariance_type": prior.covariance_type,
            "latent_dim": prior.latent_dim,
            "num_components": prior.num_components,
            "metadata": metadata or {},
        },
        path_obj,
    )


def load_gmm_prior(
    path: str,
    map_location: Optional[str] = None,
) -> Tuple[GaussianMixtureLatentPrior, Dict[str, object]]:
    checkpoint = _torch_load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"GMM prior checkpoint must be a dict: {path}")
    prior = GaussianMixtureLatentPrior(
        weights=torch.as_tensor(checkpoint["weights"], dtype=torch.float32),
        means=torch.as_tensor(checkpoint["means"], dtype=torch.float32),
        covariances=torch.as_tensor(checkpoint["covariances"], dtype=torch.float32),
        covariance_type=str(checkpoint.get("covariance_type", "diag")),
    )
    prior.eval()
    return prior, checkpoint


def _sklearn_gaussian_mixture_or_none():
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        return None
    return GaussianMixture


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.sum(np.exp(values - max_values), axis=axis)
    )


def _fit_diag_gmm_numpy(
    latents: torch.Tensor,
    components: int,
    reg_covar: float,
    seed: int,
    max_iter: int,
    tol: float = 1e-4,
) -> Tuple[GaussianMixtureLatentPrior, Dict[str, object]]:
    """Small EM fallback for diagonal-covariance GMMs when sklearn is unavailable."""

    x = latents.detach().cpu().numpy().astype(np.float64, copy=False)
    n, latent_dim = x.shape
    k = int(components)
    rng = np.random.default_rng(int(seed))
    init_indices = rng.choice(n, size=k, replace=False)
    means = x[init_indices].copy()
    global_var = np.var(x, axis=0) + float(reg_covar)
    covariances = np.tile(global_var[None, :], (k, 1))
    weights = np.full(k, 1.0 / float(k), dtype=np.float64)
    lower_bound = -np.inf
    converged = False
    n_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        diff = x[:, None, :] - means[None, :, :]
        variances = np.maximum(covariances, float(reg_covar))
        log_det = np.log(variances).sum(axis=1)
        quad = np.square(diff / np.sqrt(variances[None, :, :])).sum(axis=2)
        log_components = (
            np.log(np.maximum(weights, 1e-12))[None, :]
            - 0.5 * (latent_dim * np.log(2.0 * np.pi) + log_det[None, :] + quad)
        )
        log_norm = _logsumexp_np(log_components, axis=1)
        new_lower_bound = float(np.mean(log_norm))
        responsibilities = np.exp(log_components - log_norm[:, None])
        nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
        weights = nk / float(n)
        means = (responsibilities.T @ x) / nk[:, None]
        centered = x[:, None, :] - means[None, :, :]
        covariances = (responsibilities[:, :, None] * np.square(centered)).sum(axis=0) / nk[:, None]
        covariances = np.maximum(covariances, float(reg_covar))
        n_iter = iteration
        if abs(new_lower_bound - lower_bound) < float(tol):
            lower_bound = new_lower_bound
            converged = True
            break
        lower_bound = new_lower_bound

    prior = GaussianMixtureLatentPrior(
        weights=torch.as_tensor(weights, dtype=torch.float32),
        means=torch.as_tensor(means, dtype=torch.float32),
        covariances=torch.as_tensor(covariances, dtype=torch.float32),
        covariance_type="diag",
    )
    summary = {
        "components": int(components),
        "covariance_type": "diag",
        "reg_covar": float(reg_covar),
        "seed": int(seed),
        "max_iter": int(max_iter),
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "lower_bound": float(lower_bound),
        "latent_dim": int(latents.shape[1]),
        "num_train_latents": int(latents.shape[0]),
        "fitter": "numpy_em_fallback",
    }
    return prior, summary


def fit_gmm_prior_from_latents(
    latents: torch.Tensor,
    components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-6,
    seed: int = 13,
    max_iter: int = 200,
) -> Tuple[GaussianMixtureLatentPrior, Dict[str, object]]:
    covariance_type = str(covariance_type).lower()
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"covariance_type must be one of {SUPPORTED_COVARIANCE_TYPES}")
    latents = torch.as_tensor(latents, dtype=torch.float32)
    if latents.ndim != 2:
        raise ValueError("latents must have shape [N, D]")
    if int(components) <= 0:
        raise ValueError("components must be positive")
    if int(components) > int(latents.shape[0]):
        raise ValueError("components cannot exceed the number of train latents")

    GaussianMixture = _sklearn_gaussian_mixture_or_none()
    if GaussianMixture is None:
        if covariance_type != "diag":
            raise RuntimeError(
                "Training a full-covariance GMM prior requires scikit-learn. "
                "Use --covariance-type diag for the built-in fallback, or install sklearn."
            )
        return _fit_diag_gmm_numpy(
            latents,
            components=components,
            reg_covar=reg_covar,
            seed=seed,
            max_iter=max_iter,
        )

    gmm = GaussianMixture(
        n_components=int(components),
        covariance_type=covariance_type,
        reg_covar=float(reg_covar),
        random_state=int(seed),
        max_iter=int(max_iter),
    )
    gmm.fit(latents.detach().cpu().numpy())
    prior = GaussianMixtureLatentPrior(
        weights=torch.as_tensor(gmm.weights_, dtype=torch.float32),
        means=torch.as_tensor(gmm.means_, dtype=torch.float32),
        covariances=torch.as_tensor(gmm.covariances_, dtype=torch.float32),
        covariance_type=covariance_type,
    )
    summary = {
        "components": int(components),
        "covariance_type": covariance_type,
        "reg_covar": float(reg_covar),
        "seed": int(seed),
        "max_iter": int(max_iter),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
        "latent_dim": int(latents.shape[1]),
        "num_train_latents": int(latents.shape[0]),
        "fitter": "sklearn",
    }
    return prior, summary


def train_gmm_prior(
    checkpoint: str,
    out: str,
    components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-6,
    seed: int = 13,
    max_iter: int = 200,
) -> Path:
    train_latents, _train_shape_ids, checkpoint_payload = load_deepsdf_train_latents(checkpoint)
    prior, summary = fit_gmm_prior_from_latents(
        train_latents,
        components=components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        seed=seed,
        max_iter=max_iter,
    )
    out_path = Path(out)
    metadata = {
        **summary,
        "checkpoint": checkpoint,
        "deepsdf_config": checkpoint_payload.get("config", {}),
    }
    save_gmm_prior(str(out_path), prior, metadata=metadata)
    with out_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(out_path), **metadata}, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a post-hoc GMM latent prior from train latents.")
    parser.add_argument("--checkpoint", required=True, help="DeepSDF checkpoint with train_latents.")
    parser.add_argument("--out", required=True, help="Output GMM prior checkpoint, e.g. gmm_prior.pt.")
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--covariance-type", default="diag", choices=SUPPORTED_COVARIANCE_TYPES)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        path = train_gmm_prior(
            checkpoint=args.checkpoint,
            out=args.out,
            components=args.components,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            seed=args.seed,
            max_iter=args.max_iter,
        )
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps({"checkpoint": str(path)}, indent=2))


if __name__ == "__main__":
    main()
