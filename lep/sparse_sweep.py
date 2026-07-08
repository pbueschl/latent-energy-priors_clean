"""Sparse held-out latent inference sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .conditional_energy import load_conditional_energy_prior
from .deepsdf import LatentDecoder, clamp_sdf
from .latent_gmm_prior import GaussianMixtureLatentPrior, load_gmm_prior
from .preprocess import _normalize_mesh, _require_trimesh, _signed_distance
from .shapenet import discover_meshes, load_split, mesh_map
from .train_deepsdf import _torch_load, find_sdf_file, load_sdf_npz


METHODS = (
    "no_prior",
    "l2",
    "conditional_energy",
    "l2_conditional_energy",
    "conditional_energy_shuffled_context",
    "l2_conditional_energy_shuffled_context",
    "gmm",
    "l2_gmm",
)
CONDITIONAL_ENERGY_METHODS = (
    "conditional_energy",
    "l2_conditional_energy",
    "conditional_energy_shuffled_context",
    "l2_conditional_energy_shuffled_context",
)
GMM_METHODS = (
    "gmm",
    "l2_gmm",
)
L2_METHODS = (
    "l2",
    "l2_conditional_energy",
    "l2_conditional_energy_shuffled_context",
    "l2_gmm",
)
SHUFFLED_CONTEXT_METHODS = (
    "conditional_energy_shuffled_context",
    "l2_conditional_energy_shuffled_context",
)
GRID_METRIC_FIELDS = (
    "grid_mean_abs_sdf_error",
    "grid_rmse_sdf_error",
    "grid_mean_abs_clipped_sdf_error",
    "grid_rmse_clipped_sdf_error",
    "eval_dice",
    "eval_iou",
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "pred_positive_fraction",
    "gt_positive_fraction",
    "mesh_is_watertight",
)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    method: str
    lambda_l2: float
    lambda_conditional_energy: float
    lambda_gmm_prior: float


@dataclass(frozen=True)
class GridMetricsConfig:
    enabled: bool
    mesh_root: str
    synset: str
    mesh_layout: str
    grid_size: int
    bounds: float
    sdf_clamp: Optional[float]
    threshold: float
    steps: Tuple[int, ...]
    batch_points: int
    gt_batch_points: int
    allow_non_watertight_gt: bool


def _load_config(path: Optional[str]) -> Dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolved_sdf_clamp(sdf_clamp: Optional[float]) -> Optional[float]:
    if sdf_clamp is None:
        return None
    value = float(sdf_clamp)
    return value if value > 0 else None


def _clip_sdf_np(sdf: np.ndarray, sdf_clamp: Optional[float]) -> np.ndarray:
    values = np.asarray(sdf, dtype=np.float32)
    resolved = _resolved_sdf_clamp(sdf_clamp)
    if resolved is None:
        return values.astype(np.float32, copy=True)
    return np.clip(values, -resolved, resolved).astype(np.float32, copy=False)


def _grid_points(grid_size: int, bounds: float) -> np.ndarray:
    axis = np.linspace(-float(bounds), float(bounds), int(grid_size), dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(np.float32, copy=False)


def _decode_sdf_grid(
    decoder: LatentDecoder,
    z: torch.Tensor,
    points: torch.Tensor,
    grid_size: int,
    batch_points: int,
) -> np.ndarray:
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        z_eval = z.detach()
        for start in range(0, points.shape[0], int(batch_points)):
            chunk = points[start : start + int(batch_points)]
            chunks.append(decoder(z_eval, chunk).reshape(-1).detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32).reshape(
        int(grid_size), int(grid_size), int(grid_size)
    )


def _gt_sdf_grid(mesh, points: np.ndarray, grid_size: int, gt_batch_points: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, points.shape[0], int(gt_batch_points)):
        chunk = points[start : start + int(gt_batch_points)]
        sdf = np.asarray(_signed_distance(mesh, chunk), dtype=np.float32).reshape(-1)
        if sdf.shape[0] != chunk.shape[0]:
            raise RuntimeError(
                f"_signed_distance returned {sdf.shape[0]} values for {chunk.shape[0]} points"
            )
        chunks.append(sdf)
    flat = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
    if flat.shape[0] != points.shape[0]:
        raise RuntimeError(f"GT SDF returned {flat.shape[0]} values for {points.shape[0]} points")
    return flat.astype(np.float32, copy=False).reshape(
        int(grid_size), int(grid_size), int(grid_size)
    )


def _binary_grid_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_mask).astype(bool, copy=False)
    gt = np.asarray(gt_mask).astype(bool, copy=False)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT mask shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn
    both_empty = pred_sum == 0 and gt_sum == 0
    return {
        "eval_dice": float(1.0 if both_empty else (2.0 * tp) / float(pred_sum + gt_sum)),
        "eval_iou": float(1.0 if union == 0 else tp / float(union)),
        "eval_accuracy": float((tp + tn) / float(pred.size)),
        "eval_precision": float(
            1.0 if both_empty else (tp / float(pred_sum) if pred_sum else 0.0)
        ),
        "eval_recall": float(1.0 if both_empty else (tp / float(gt_sum) if gt_sum else 0.0)),
        "pred_positive_fraction": float(pred.mean()),
        "gt_positive_fraction": float(gt.mean()),
    }


def _grid_sdf_metrics(
    pred_sdf: np.ndarray,
    gt_sdf: np.ndarray,
    sdf_clamp: Optional[float],
    threshold: float,
) -> Dict[str, float]:
    pred = np.asarray(pred_sdf, dtype=np.float32)
    gt = np.asarray(gt_sdf, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT SDF shape mismatch: {pred.shape} vs {gt.shape}")
    pred_clip = _clip_sdf_np(pred, sdf_clamp)
    gt_clip = _clip_sdf_np(gt, sdf_clamp)
    diff = pred - gt
    clipped_diff = pred_clip - gt_clip
    pred_mask = pred < float(threshold)
    gt_mask = gt < float(threshold)
    row = {
        "grid_mean_abs_sdf_error": float(np.mean(np.abs(diff))),
        "grid_rmse_sdf_error": float(np.sqrt(np.mean(np.square(diff)))),
        "grid_mean_abs_clipped_sdf_error": float(np.mean(np.abs(clipped_diff))),
        "grid_rmse_clipped_sdf_error": float(np.sqrt(np.mean(np.square(clipped_diff)))),
    }
    row.update(_binary_grid_metrics(pred_mask, gt_mask))
    return row


def _infer_method(
    name: str,
    lambda_l2: float,
    lambda_conditional_energy: float,
    lambda_gmm_prior: float = 0.0,
) -> str:
    name_lower = name.lower()
    if name_lower in METHODS:
        return name_lower
    if "l2_gmm" in name_lower or "gmm_l2" in name_lower:
        return "l2_gmm"
    if "gmm" in name_lower:
        return "gmm"
    is_shuffled = "shuffled_context" in name_lower or "shuffle_context" in name_lower
    if "l2_conditional" in name_lower or "conditional_l2" in name_lower:
        if is_shuffled:
            return "l2_conditional_energy_shuffled_context"
        return "l2_conditional_energy"
    if "conditional" in name_lower or "_cond" in name_lower or name_lower.startswith("cond"):
        if is_shuffled:
            return "conditional_energy_shuffled_context"
        return "conditional_energy"
    if "l2_energy" in name_lower or name_lower.startswith("energy"):
        raise ValueError(
            "Unconditional energy variants were removed from this paper artifact; "
            "use conditional_energy or l2_conditional_energy."
        )
    if name_lower.startswith("l2") or "_l2" in name_lower:
        return "l2"
    has_l2 = float(lambda_l2) > 0
    has_conditional = float(lambda_conditional_energy) > 0
    has_gmm = float(lambda_gmm_prior) > 0
    if has_l2 and has_gmm:
        return "l2_gmm"
    if has_gmm:
        return "gmm"
    if has_l2 and has_conditional:
        return "l2_conditional_energy"
    if has_conditional:
        return "conditional_energy"
    if has_l2:
        return "l2"
    return "no_prior"


def resolve_sweep_variants(
    sweep_cfg: Dict[str, object],
    methods_override: Optional[Sequence[str]] = None,
) -> List[SweepVariant]:
    """Resolve sparse-sweep methods or per-variant lambda settings."""
    if "lambda_energy" in sweep_cfg:
        raise ValueError(
            "sparse_sweep.lambda_energy is no longer supported; "
            "use lambda_conditional_energy for observation-conditioned priors."
        )
    global_l2 = float(sweep_cfg.get("lambda_l2", 1e-4))
    global_conditional_energy = float(sweep_cfg.get("lambda_conditional_energy", 1e-2))
    global_gmm_prior = float(
        sweep_cfg.get(
            "lambda_gmm_prior",
            sweep_cfg.get("lambda_gmm", sweep_cfg.get("gmm_weight", 1e-2)),
        )
    )
    if methods_override is not None:
        method_names = [str(method) for method in methods_override]
        for method in method_names:
            if method not in METHODS:
                raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
        return [
            SweepVariant(
                name=method,
                method=method,
                lambda_l2=global_l2,
                lambda_conditional_energy=global_conditional_energy,
                lambda_gmm_prior=global_gmm_prior,
            )
            for method in method_names
        ]

    raw_variants = sweep_cfg.get("variants")
    if raw_variants is not None:
        if not isinstance(raw_variants, list):
            raise ValueError("sparse_sweep.variants must be a list of mappings")
        variants: List[SweepVariant] = []
        for index, raw in enumerate(raw_variants):
            if not isinstance(raw, dict):
                raise ValueError(f"sparse_sweep.variants[{index}] must be a mapping")
            if "name" not in raw:
                raise ValueError(f"sparse_sweep.variants[{index}] is missing required field 'name'")
            name = str(raw["name"])
            if "lambda_energy" in raw:
                raise ValueError(
                    f"sparse_sweep.variants[{index}] uses lambda_energy, which is no longer "
                    "supported; use lambda_conditional_energy."
                )
            lambda_l2 = float(raw.get("lambda_l2", global_l2))
            lambda_conditional_energy = float(
                raw.get(
                    "lambda_conditional_energy",
                    raw.get("conditional_energy_weight", global_conditional_energy),
                )
            )
            lambda_gmm_prior = float(
                raw.get(
                    "lambda_gmm_prior",
                    raw.get("lambda_gmm", raw.get("gmm_weight", global_gmm_prior)),
                )
            )
            method = str(
                raw.get("method")
                or _infer_method(name, lambda_l2, lambda_conditional_energy, lambda_gmm_prior)
            )
            if method not in METHODS:
                raise ValueError(f"Unknown method {method!r} for variant {name!r}; expected one of {METHODS}")
            variants.append(
                SweepVariant(
                    name=name,
                    method=method,
                    lambda_l2=lambda_l2,
                    lambda_conditional_energy=lambda_conditional_energy,
                    lambda_gmm_prior=lambda_gmm_prior,
                )
            )
        return variants

    method_names = [str(method) for method in sweep_cfg.get("methods", ["no_prior", "l2"])]
    return [
        SweepVariant(
            name=method,
            method=method,
            lambda_l2=global_l2,
            lambda_conditional_energy=global_conditional_energy,
            lambda_gmm_prior=global_gmm_prior,
        )
        for method in method_names
    ]


def resolve_grid_metrics_config(
    sweep_cfg: Dict[str, object],
    deepsdf_cfg: Dict[str, object],
    steps: Sequence[int],
) -> Optional[GridMetricsConfig]:
    raw_cfg = sweep_cfg.get("grid_metrics", {})
    if not raw_cfg:
        return None
    if not isinstance(raw_cfg, dict):
        raise ValueError("sparse_sweep.grid_metrics must be a mapping")
    if not bool(raw_cfg.get("enabled", False)):
        return None
    mesh_root = str(raw_cfg.get("mesh_root", "")).strip()
    if not mesh_root:
        raise ValueError("sparse_sweep.grid_metrics.mesh_root is required when enabled")

    raw_steps = raw_cfg.get("steps", [max(int(step) for step in steps)])
    metric_steps = tuple(sorted({int(step) for step in raw_steps}))
    if not metric_steps:
        raise ValueError("sparse_sweep.grid_metrics.steps cannot be empty")

    batch_points = int(raw_cfg.get("batch_points", 262144))
    gt_batch_points = int(raw_cfg.get("gt_batch_points", batch_points))
    if batch_points <= 0 or gt_batch_points <= 0:
        raise ValueError("sparse_sweep.grid_metrics batch sizes must be positive")

    return GridMetricsConfig(
        enabled=True,
        mesh_root=mesh_root,
        synset=str(raw_cfg.get("synset", "medshape")),
        mesh_layout=str(raw_cfg.get("mesh_layout", "auto")),
        grid_size=int(raw_cfg.get("grid_size", 64)),
        bounds=float(raw_cfg.get("bounds", 1.15)),
        sdf_clamp=_resolved_sdf_clamp(raw_cfg.get("sdf_clamp", deepsdf_cfg.get("sdf_clamp", 0.1))),
        threshold=float(raw_cfg.get("threshold", 0.0)),
        steps=metric_steps,
        batch_points=batch_points,
        gt_batch_points=gt_batch_points,
        allow_non_watertight_gt=bool(raw_cfg.get("allow_non_watertight_gt", False)),
    )


class SparseGridMetricEvaluator:
    def __init__(
        self,
        decoder: LatentDecoder,
        mesh_path: str,
        config: GridMetricsConfig,
        device: str,
    ) -> None:
        if int(config.grid_size) <= 0:
            raise ValueError(f"grid_size must be positive, got {config.grid_size}")
        trimesh = _require_trimesh()
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh.is_empty:
            raise ValueError(f"Empty mesh: {mesh_path}")
        mesh, _, _ = _normalize_mesh(mesh)
        mesh_is_watertight = bool(getattr(mesh, "is_watertight", False))
        if not mesh_is_watertight and not config.allow_non_watertight_gt:
            raise ValueError(
                f"Mesh is not watertight for grid metrics: {mesh_path}. "
                "Set sparse_sweep.grid_metrics.allow_non_watertight_gt=true for exploratory runs."
            )

        points_np = _grid_points(config.grid_size, config.bounds)
        self.decoder = decoder
        self.points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        self.grid_size = int(config.grid_size)
        self.batch_points = int(config.batch_points)
        self.sdf_clamp = config.sdf_clamp
        self.threshold = float(config.threshold)
        self.mesh_is_watertight = mesh_is_watertight
        self.gt_sdf = _gt_sdf_grid(
            mesh=mesh,
            points=points_np,
            grid_size=self.grid_size,
            gt_batch_points=int(config.gt_batch_points),
        )

    def __call__(self, z: torch.Tensor) -> Dict[str, object]:
        pred_sdf = _decode_sdf_grid(
            self.decoder,
            z,
            self.points,
            grid_size=self.grid_size,
            batch_points=self.batch_points,
        )
        row: Dict[str, object] = _grid_sdf_metrics(
            pred_sdf=pred_sdf,
            gt_sdf=self.gt_sdf,
            sdf_clamp=self.sdf_clamp,
            threshold=self.threshold,
        )
        row["mesh_is_watertight"] = self.mesh_is_watertight
        return row


def load_decoder_checkpoint(path: str, device: str = "cpu") -> Tuple[LatentDecoder, Dict[str, object]]:
    checkpoint = _torch_load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Decoder checkpoint must be a dict: {path}")
    decoder = LatentDecoder(**checkpoint["decoder_config"]).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, checkpoint


def sample_disjoint_observation_eval(
    points: np.ndarray,
    sdf: np.ndarray,
    obs_count: int,
    eval_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample sparse observations and held-out eval points with no overlap when possible."""
    if points.shape[0] != sdf.shape[0]:
        raise ValueError(f"Point/SDF count mismatch: {points.shape[0]} vs {sdf.shape[0]}")
    if points.shape[0] == 0:
        raise ValueError("Cannot sample from an empty SDF array")

    n = points.shape[0]
    obs_replace = int(obs_count) > n
    obs_indices = rng.choice(n, size=int(obs_count), replace=obs_replace)
    observed = np.unique(obs_indices)
    remaining = np.setdiff1d(np.arange(n), observed, assume_unique=False)
    if remaining.size > 0:
        eval_replace = int(eval_count) > remaining.size
        eval_indices = rng.choice(remaining, size=int(eval_count), replace=eval_replace)
    else:
        eval_indices = rng.choice(n, size=int(eval_count), replace=True)
    return (
        points[obs_indices],
        sdf[obs_indices],
        points[eval_indices],
        sdf[eval_indices],
        obs_indices,
        eval_indices,
    )


def shuffled_context_shape_id(shape_ids: Sequence[str], shape_index: int, repeat: int) -> Tuple[str, bool]:
    """Return a deterministic non-self context shape id when the split has alternatives."""
    if not shape_ids:
        raise ValueError("shape_ids cannot be empty")
    if len(shape_ids) == 1:
        return str(shape_ids[int(shape_index)]), False
    context_index = (int(shape_index) + int(repeat) + 1) % len(shape_ids)
    if context_index == int(shape_index):
        context_index = (context_index + 1) % len(shape_ids)
    return str(shape_ids[context_index]), True


def _context_rng(seed: int, shape_index: int, repeat: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(shape_index), int(repeat), 7919])


def _evaluate(
    decoder: LatentDecoder,
    z: torch.Tensor,
    obs_xyz: torch.Tensor,
    obs_sdf: torch.Tensor,
    eval_xyz: torch.Tensor,
    eval_sdf: torch.Tensor,
    sdf_clamp: Optional[float],
    conditional_energy_model: Optional[torch.nn.Module] = None,
    conditional_context_xyz: Optional[torch.Tensor] = None,
    conditional_context_sdf: Optional[torch.Tensor] = None,
    gmm_prior: Optional[GaussianMixtureLatentPrior] = None,
) -> Dict[str, float]:
    pred_obs = decoder(z, obs_xyz).reshape_as(obs_sdf)
    pred_eval = decoder(z, eval_xyz).reshape_as(eval_sdf)
    row = {
        "obs_l1": float(F.l1_loss(clamp_sdf(pred_obs, sdf_clamp), clamp_sdf(obs_sdf, sdf_clamp)).cpu()),
        "eval_l1": float(F.l1_loss(clamp_sdf(pred_eval, sdf_clamp), clamp_sdf(eval_sdf, sdf_clamp)).cpu()),
        "latent_norm": float(z.norm(dim=-1).mean().cpu()),
        "conditional_energy": "",
        "gmm_log_prob": "",
        "weighted_gmm_loss": "",
    }
    if conditional_energy_model is not None:
        context_xyz = obs_xyz if conditional_context_xyz is None else conditional_context_xyz
        context_sdf = obs_sdf if conditional_context_sdf is None else conditional_context_sdf
        row["conditional_energy"] = float(
            conditional_energy_model.energy(z, context_xyz, context_sdf).mean().cpu()
        )
    if gmm_prior is not None:
        log_prob = gmm_prior.log_prob(z).mean()
        row["gmm_log_prob"] = float(log_prob.cpu())
    return row


def infer_one(
    decoder: LatentDecoder,
    obs_points: np.ndarray,
    obs_sdf_np: np.ndarray,
    eval_points: np.ndarray,
    eval_sdf_np: np.ndarray,
    method: str,
    steps: Sequence[int],
    lr: float = 0.01,
    lambda_l2: float = 1e-4,
    lambda_conditional_energy: float = 1e-2,
    lambda_gmm_prior: float = 1e-2,
    sdf_clamp: Optional[float] = 0.1,
    conditional_energy_model: Optional[torch.nn.Module] = None,
    conditional_context_points: Optional[np.ndarray] = None,
    conditional_context_sdf_np: Optional[np.ndarray] = None,
    gmm_prior: Optional[GaussianMixtureLatentPrior] = None,
    device: str = "cpu",
    grid_metric_evaluator: Optional[Callable[[torch.Tensor], Dict[str, object]]] = None,
    grid_metric_steps: Optional[Sequence[int]] = None,
) -> List[Dict[str, object]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    if method in CONDITIONAL_ENERGY_METHODS and conditional_energy_model is None:
        raise ValueError(f"Method {method!r} requires --conditional-energy-checkpoint")
    if method in GMM_METHODS and gmm_prior is None:
        raise ValueError(f"Method {method!r} requires --gmm-prior-checkpoint")

    obs_xyz = torch.as_tensor(obs_points, dtype=torch.float32, device=device)
    obs_sdf = torch.as_tensor(obs_sdf_np.reshape(-1), dtype=torch.float32, device=device)
    context_points_np = obs_points if conditional_context_points is None else conditional_context_points
    context_sdf_np = obs_sdf_np if conditional_context_sdf_np is None else conditional_context_sdf_np
    conditional_context_xyz = torch.as_tensor(context_points_np, dtype=torch.float32, device=device)
    conditional_context_sdf = torch.as_tensor(context_sdf_np.reshape(-1), dtype=torch.float32, device=device)
    eval_xyz = torch.as_tensor(eval_points, dtype=torch.float32, device=device)
    eval_sdf = torch.as_tensor(eval_sdf_np.reshape(-1), dtype=torch.float32, device=device)
    z = torch.zeros(1, decoder.latent_dim, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=float(lr))
    step_set = sorted(set(int(step) for step in steps))
    grid_step_set = set(int(step) for step in (grid_metric_steps or []))
    max_step = max(step_set)
    rows: List[Dict[str, object]] = []

    if 0 in step_set:
        with torch.no_grad():
            row = _evaluate(
                decoder,
                z,
                obs_xyz,
                obs_sdf,
                eval_xyz,
                eval_sdf,
                sdf_clamp,
                conditional_energy_model,
                conditional_context_xyz,
                conditional_context_sdf,
                gmm_prior,
            )
            if gmm_prior is not None and method in GMM_METHODS:
                gmm_log_prob = gmm_prior.log_prob(z).mean()
                row["weighted_gmm_loss"] = float((-float(lambda_gmm_prior) * gmm_log_prob).cpu())
            if grid_metric_evaluator is not None and 0 in grid_step_set:
                row.update(grid_metric_evaluator(z))
        row["step"] = 0
        rows.append(row)

    for step in range(1, max_step + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = decoder(z, obs_xyz).reshape_as(obs_sdf)
        obs_loss = F.l1_loss(clamp_sdf(pred, sdf_clamp), clamp_sdf(obs_sdf, sdf_clamp))
        loss = obs_loss
        if method in L2_METHODS:
            loss = loss + float(lambda_l2) * z.pow(2).sum(dim=-1).mean()
        if method in CONDITIONAL_ENERGY_METHODS:
            assert conditional_energy_model is not None
            conditional_energy = conditional_energy_model.energy(
                z,
                conditional_context_xyz,
                conditional_context_sdf,
            )
            loss = loss + float(lambda_conditional_energy) * conditional_energy.mean()
        if method in GMM_METHODS:
            assert gmm_prior is not None
            gmm_log_prob = gmm_prior.log_prob(z)
            loss = loss - float(lambda_gmm_prior) * gmm_log_prob.mean()
        loss.backward()
        optimizer.step()

        if step in step_set:
            with torch.no_grad():
                row = _evaluate(
                    decoder,
                    z,
                    obs_xyz,
                    obs_sdf,
                    eval_xyz,
                    eval_sdf,
                    sdf_clamp,
                    conditional_energy_model,
                    conditional_context_xyz,
                    conditional_context_sdf,
                    gmm_prior,
                )
                if gmm_prior is not None and method in GMM_METHODS:
                    gmm_log_prob = gmm_prior.log_prob(z).mean()
                    row["weighted_gmm_loss"] = float((-float(lambda_gmm_prior) * gmm_log_prob).cpu())
                if grid_metric_evaluator is not None and step in grid_step_set:
                    row.update(grid_metric_evaluator(z))
            row["step"] = step
            rows.append(row)
    return rows


def run_sparse_sweep(
    config: Dict[str, object],
    sdf_root: str,
    split_file: str,
    checkpoint: str,
    out: str,
    conditional_energy_checkpoint: Optional[str] = None,
    gmm_prior_checkpoint: Optional[str] = None,
    device: str = "cpu",
    split_name: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    steps: Optional[Sequence[int]] = None,
) -> List[Dict[str, object]]:
    sweep_cfg = dict(config.get("sparse_sweep", {}))
    deepsdf_cfg = dict(config.get("deepsdf", {}))
    split = load_split(split_file)
    chosen_split = split_name or str(sweep_cfg.get("split", "test"))
    shape_ids = [str(shape_id) for shape_id in split.get(chosen_split, [])]
    if not shape_ids:
        raise ValueError(f"Split {chosen_split!r} has no shape ids in {split_file}")

    variants = resolve_sweep_variants(sweep_cfg, methods_override=methods)
    step_list = sorted(int(step) for step in (steps or sweep_cfg.get("steps", [0, 50, 100])))
    grid_metrics_cfg = resolve_grid_metrics_config(sweep_cfg, deepsdf_cfg, step_list)
    decoder, _ = load_decoder_checkpoint(checkpoint, device=device)
    conditional_energy_model = None
    gmm_prior = None
    needs_conditional_energy = any(variant.method in CONDITIONAL_ENERGY_METHODS for variant in variants)
    needs_gmm_prior = any(variant.method in GMM_METHODS for variant in variants)
    if conditional_energy_checkpoint is not None:
        conditional_energy_model, _conditional_state = load_conditional_energy_prior(
            conditional_energy_checkpoint,
            map_location=device,
        )
        conditional_energy_model = conditional_energy_model.to(device)
        conditional_energy_model.eval()
        for param in conditional_energy_model.parameters():
            param.requires_grad_(False)
    elif needs_conditional_energy:
        raise ValueError(
            "A conditional energy checkpoint is required for variants using "
            "conditional energy methods"
        )
    if gmm_prior_checkpoint is not None:
        gmm_prior, _gmm_state = load_gmm_prior(gmm_prior_checkpoint, map_location=device)
        gmm_prior = gmm_prior.to(device)
        gmm_prior.eval()
        for param in gmm_prior.parameters():
            param.requires_grad_(False)
    elif needs_gmm_prior:
        raise ValueError("A GMM prior checkpoint is required for variants using GMM methods")

    seed = int(config.get("seed", 13))
    rng = np.random.default_rng(seed)
    obs_count = int(sweep_cfg.get("observation_points", 512))
    eval_count = int(sweep_cfg.get("eval_points", 8192))
    repeats = int(sweep_cfg.get("repeats", 1))
    lr = float(sweep_cfg.get("lr", 0.01))
    sdf_band = deepsdf_cfg.get("sdf_clamp", 0.1)
    mesh_records = {}
    if grid_metrics_cfg is not None:
        mesh_records = mesh_map(
            discover_meshes(
                grid_metrics_cfg.mesh_root,
                synset=grid_metrics_cfg.synset,
                layout=grid_metrics_cfg.mesh_layout,
            )
        )

    all_rows: List[Dict[str, object]] = []
    needs_shuffled_context = any(variant.method in SHUFFLED_CONTEXT_METHODS for variant in variants)
    for shape_index, shape_id in enumerate(tqdm(shape_ids, desc="sparse_sweep")):
        points, sdf = load_sdf_npz(str(find_sdf_file(sdf_root, shape_id)))
        grid_metric_evaluator = None
        if grid_metrics_cfg is not None:
            record = mesh_records.get(shape_id)
            if record is None:
                raise KeyError(
                    f"No mesh found for shape id {shape_id!r} under "
                    f"{grid_metrics_cfg.mesh_root!r} with synset {grid_metrics_cfg.synset!r}."
                )
            grid_metric_evaluator = SparseGridMetricEvaluator(
                decoder=decoder,
                mesh_path=record.mesh_path,
                config=grid_metrics_cfg,
                device=device,
            )
        for repeat in range(repeats):
            obs_points, obs_sdf_np, eval_points, eval_sdf_np, _, _ = sample_disjoint_observation_eval(
                points,
                sdf,
                obs_count,
                eval_count,
                rng,
            )
            shuffled_shape_id = shape_id
            shuffled_context_points = obs_points
            shuffled_context_sdf_np = obs_sdf_np
            context_is_shuffled = False
            if needs_shuffled_context:
                shuffled_shape_id, context_is_shuffled = shuffled_context_shape_id(
                    shape_ids,
                    shape_index,
                    repeat,
                )
                if context_is_shuffled:
                    context_points_all, context_sdf_all = load_sdf_npz(
                        str(find_sdf_file(sdf_root, shuffled_shape_id))
                    )
                    shuffled_context_points, shuffled_context_sdf_np, _, _, _, _ = (
                        sample_disjoint_observation_eval(
                            context_points_all,
                            context_sdf_all,
                            obs_count,
                            eval_count=1,
                            rng=_context_rng(seed, shape_index, repeat),
                        )
                    )
            for variant in variants:
                variant_uses_shuffled_context = variant.method in SHUFFLED_CONTEXT_METHODS
                context_shape_id = shuffled_shape_id if variant_uses_shuffled_context else shape_id
                method_rows = infer_one(
                    decoder=decoder,
                    obs_points=obs_points,
                    obs_sdf_np=obs_sdf_np,
                    eval_points=eval_points,
                    eval_sdf_np=eval_sdf_np,
                    method=variant.method,
                    steps=step_list,
                    lr=lr,
                    lambda_l2=variant.lambda_l2,
                    lambda_conditional_energy=variant.lambda_conditional_energy,
                    lambda_gmm_prior=variant.lambda_gmm_prior,
                    sdf_clamp=sdf_band,
                    conditional_energy_model=conditional_energy_model,
                    conditional_context_points=(
                        shuffled_context_points if variant_uses_shuffled_context else obs_points
                    ),
                    conditional_context_sdf_np=(
                        shuffled_context_sdf_np if variant_uses_shuffled_context else obs_sdf_np
                    ),
                    gmm_prior=gmm_prior,
                    device=device,
                    grid_metric_evaluator=grid_metric_evaluator,
                    grid_metric_steps=grid_metrics_cfg.steps if grid_metrics_cfg is not None else None,
                )
                for row in method_rows:
                    all_rows.append(
                        {
                            "shape_id": shape_id,
                            "context_shape_id": context_shape_id,
                            "context_is_shuffled": bool(
                                variant_uses_shuffled_context and context_is_shuffled
                            ),
                            "split": chosen_split,
                            "repeat": repeat,
                            "observation_points": obs_count,
                            "variant": variant.name,
                            "method": variant.method,
                            "lambda_l2": variant.lambda_l2,
                            "lambda_conditional_energy": variant.lambda_conditional_energy,
                            "lambda_gmm_prior": variant.lambda_gmm_prior,
                            **row,
                        }
                    )

    fieldnames = [
        "shape_id",
        "context_shape_id",
        "context_is_shuffled",
        "split",
        "repeat",
        "observation_points",
        "variant",
        "method",
        "lambda_l2",
        "lambda_conditional_energy",
        "lambda_gmm_prior",
        "step",
        "obs_l1",
        "eval_l1",
        "latent_norm",
        "conditional_energy",
        "gmm_log_prob",
        "weighted_gmm_loss",
    ]
    fieldnames.extend(GRID_METRIC_FIELDS)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return all_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run sparse latent inference step sweeps.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sdf-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conditional-energy-checkpoint", default=None)
    parser.add_argument("--gmm-prior-checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated methods.")
    parser.add_argument("--steps", default=None, help="Comma-separated step list, e.g. 0,25,100.")
    args = parser.parse_args(argv)

    rows = run_sparse_sweep(
        config=_load_config(args.config),
        sdf_root=args.sdf_root,
        split_file=args.split_file,
        checkpoint=args.checkpoint,
        out=args.out,
        conditional_energy_checkpoint=args.conditional_energy_checkpoint,
        gmm_prior_checkpoint=args.gmm_prior_checkpoint,
        device=args.device,
        split_name=args.split,
        methods=_parse_str_list(args.methods),
        steps=_parse_int_list(args.steps),
    )
    print(json.dumps({"out": args.out, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
