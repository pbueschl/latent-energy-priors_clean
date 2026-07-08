"""Quantitative QC for DeepSDF autodecoder grids against mesh SDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .preprocess import _normalize_mesh, _require_trimesh, _signed_distance
from .shapenet import CHAIR_SYNSET, ShapeRecord, discover_meshes, mesh_map
from .sparse_sweep import load_decoder_checkpoint


QC_CSV_FIELDS = [
    "shape_id",
    "train_index",
    "mesh_path",
    "npz",
    "pred_sdf_tiff",
    "gt_sdf_tiff",
    "pred_sdf_clipped_tiff",
    "gt_sdf_clipped_tiff",
    "pred_mask_tiff",
    "gt_mask_tiff",
    "grid_size",
    "bounds",
    "sdf_clamp",
    "threshold",
    "mean_abs_sdf_error",
    "rmse_sdf_error",
    "mean_abs_clipped_sdf_error",
    "rmse_clipped_sdf_error",
    "dice",
    "iou",
    "accuracy",
    "precision",
    "recall",
    "pred_positive_fraction",
    "gt_positive_fraction",
    "mean_pred_sdf",
    "mean_gt_sdf",
    "mean_pred_sdf_inside",
    "mean_pred_sdf_outside",
    "mean_gt_sdf_inside",
    "mean_gt_sdf_outside",
    "tp",
    "fp",
    "tn",
    "fn",
    "mesh_is_watertight",
]


def _torch_load(path: Path, map_location: Optional[str] = None) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_int_list(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_checkpoint_latents(
    path: str,
    device: str,
) -> Tuple[torch.Tensor, List[str], Dict[str, object]]:
    checkpoint = _torch_load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    if "train_latents" in checkpoint:
        latents = torch.as_tensor(checkpoint["train_latents"], dtype=torch.float32, device=device)
    elif "latent_codes_state_dict" in checkpoint:
        weight = checkpoint["latent_codes_state_dict"]["weight"]
        latents = torch.as_tensor(weight, dtype=torch.float32, device=device)
    else:
        raise KeyError(f"{path} must contain train_latents or latent_codes_state_dict")
    shape_ids = [str(shape_id) for shape_id in checkpoint.get("train_shape_ids", [])]
    if shape_ids and len(shape_ids) != latents.shape[0]:
        raise ValueError(
            f"Checkpoint train_shape_ids length {len(shape_ids)} does not match "
            f"latents {latents.shape[0]}"
        )
    if not shape_ids:
        shape_ids = [str(index) for index in range(latents.shape[0])]
    return latents, shape_ids, checkpoint


def _resolve_indices(
    indices: Sequence[int],
    shape_ids: Sequence[str],
    requested_shape_ids: Sequence[str],
) -> List[int]:
    selected = list(dict.fromkeys(int(index) for index in indices))
    if requested_shape_ids:
        lookup = {shape_id: index for index, shape_id in enumerate(shape_ids)}
        missing = [shape_id for shape_id in requested_shape_ids if shape_id not in lookup]
        if missing:
            raise KeyError(f"Shape ids not found in checkpoint: {missing}")
        selected.extend(lookup[shape_id] for shape_id in requested_shape_ids)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("Pass at least one --indices or --shape-ids value")
    for index in selected:
        if index < 0 or index >= len(shape_ids):
            raise IndexError(f"Latent index {index} out of range [0, {len(shape_ids)})")
    return selected


def _grid_points(grid_size: int, bounds: float) -> Tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-float(bounds), float(bounds), int(grid_size), dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return axis, grid.astype(np.float32, copy=False)


def _decode_values_grid(
    decoder: torch.nn.Module,
    z: torch.Tensor,
    points: torch.Tensor,
    grid_size: int,
    batch_points: int,
) -> np.ndarray:
    values_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], int(batch_points)):
            chunk = points[start : start + int(batch_points)]
            values_chunks.append(decoder(z, chunk).reshape(-1).detach().cpu())
    return torch.cat(values_chunks, dim=0).numpy().astype(np.float32).reshape(
        int(grid_size), int(grid_size), int(grid_size)
    )


def _safe_shape_id(shape_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in shape_id)


def _load_normalized_mesh(mesh_path: str):
    trimesh = _require_trimesh()
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    return _normalize_mesh(mesh)


def _check_watertight_gt(mesh, allow_non_watertight_gt: bool) -> None:
    watertight = bool(getattr(mesh, "is_watertight", False))
    if not watertight and not allow_non_watertight_gt:
        raise ValueError(
            "Mesh is not watertight; grid QC ground truth would be unreliable. "
            "Pass --allow-non-watertight-gt to override."
        )


def _discover_mesh_records(mesh_root: str, synset: str) -> Dict[str, ShapeRecord]:
    return mesh_map(discover_meshes(mesh_root, synset=synset))


def _require_tifffile():
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TIFF export was requested, but tifffile is not installed. "
            'Install with `pip install -e ".[tiff]"` or pass --no-tiff.'
        ) from exc
    return tifffile


def _mean_numeric(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and row.get(key) != ""]
    return float(np.mean(values)) if values else None


def _resolved_sdf_clamp(sdf_clamp: Optional[float]) -> Optional[float]:
    if sdf_clamp is None:
        return None
    value = float(sdf_clamp)
    return value if value > 0 else None


def _clip_sdf(sdf: np.ndarray, sdf_clamp: Optional[float]) -> np.ndarray:
    values = np.asarray(sdf, dtype=np.float32)
    resolved = _resolved_sdf_clamp(sdf_clamp)
    if resolved is None:
        return values.astype(np.float32, copy=True)
    return np.clip(values, -resolved, resolved).astype(np.float32, copy=False)


def _binary_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, object]:
    pred = np.asarray(pred_mask).astype(bool, copy=False)
    gt = np.asarray(gt_mask).astype(bool, copy=False)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT mask shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.size == 0:
        raise ValueError("Cannot compute SDF grid metrics for empty masks")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    both_empty = pred_sum == 0 and gt_sum == 0
    dice = 1.0 if both_empty else (2.0 * tp) / float(pred_sum + gt_sum)
    iou = 1.0 if union == 0 else tp / float(union)
    accuracy = (tp + tn) / float(pred.size)
    precision = 1.0 if both_empty else (tp / float(pred_sum) if pred_sum else 0.0)
    recall = 1.0 if both_empty else (tp / float(gt_sum) if gt_sum else 0.0)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "pred_positive_fraction": float(pred.mean()),
        "gt_positive_fraction": float(gt.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def sdf_grid_metrics(
    pred_sdf: np.ndarray,
    gt_sdf: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_sdf_clipped: Optional[np.ndarray] = None,
    gt_sdf_clipped: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute SDF reconstruction and threshold-mask metrics for same-shaped grids."""
    pred = np.asarray(pred_sdf, dtype=np.float32)
    gt = np.asarray(gt_sdf, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT SDF shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.size == 0:
        raise ValueError("Cannot compute SDF grid metrics for empty grids")

    pred_clip = pred if pred_sdf_clipped is None else np.asarray(pred_sdf_clipped, dtype=np.float32)
    gt_clip = gt if gt_sdf_clipped is None else np.asarray(gt_sdf_clipped, dtype=np.float32)
    if pred_clip.shape != pred.shape or gt_clip.shape != gt.shape:
        raise ValueError(
            "Clipped SDF grid shape mismatch: "
            f"pred {pred_clip.shape}, gt {gt_clip.shape}, expected {pred.shape}"
        )

    diff = pred - gt
    clipped_diff = pred_clip - gt_clip
    gt_inside = np.asarray(gt_mask).astype(bool, copy=False)
    if gt_inside.shape != pred.shape:
        raise ValueError(f"GT mask shape mismatch: {gt_inside.shape} vs {pred.shape}")

    row: Dict[str, object] = {
        "mean_abs_sdf_error": float(np.mean(np.abs(diff))),
        "rmse_sdf_error": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_abs_clipped_sdf_error": float(np.mean(np.abs(clipped_diff))),
        "rmse_clipped_sdf_error": float(np.sqrt(np.mean(np.square(clipped_diff)))),
        "mean_pred_sdf": float(pred.mean()),
        "mean_gt_sdf": float(gt.mean()),
        "mean_pred_sdf_inside": float(pred[gt_inside].mean()) if np.any(gt_inside) else None,
        "mean_pred_sdf_outside": float(pred[~gt_inside].mean()) if np.any(~gt_inside) else None,
        "mean_gt_sdf_inside": float(gt[gt_inside].mean()) if np.any(gt_inside) else None,
        "mean_gt_sdf_outside": float(gt[~gt_inside].mean()) if np.any(~gt_inside) else None,
    }
    row.update(_binary_mask_metrics(pred_mask, gt_mask))
    return row


def _decode_sdf_grid(
    decoder: torch.nn.Module,
    z: torch.Tensor,
    points: torch.Tensor,
    grid_size: int,
    batch_points: int,
) -> np.ndarray:
    return _decode_values_grid(decoder, z, points, grid_size, batch_points)


def _gt_sdf_grid(mesh, points: np.ndarray, grid_size: int, gt_batch_points: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, points.shape[0], int(gt_batch_points)):
        chunk = points[start : start + int(gt_batch_points)]
        sdf = np.asarray(_signed_distance(mesh, chunk), dtype=np.float32).reshape(-1)
        if sdf.shape[0] != chunk.shape[0]:
            raise RuntimeError(
                f"_signed_distance returned {sdf.shape[0]} values for "
                f"{chunk.shape[0]} points"
            )
        chunks.append(sdf)
    flat = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
    if flat.shape[0] != points.shape[0]:
        raise RuntimeError(f"GT SDF returned {flat.shape[0]} values for {points.shape[0]} points")
    return flat.astype(np.float32, copy=False).reshape(
        int(grid_size), int(grid_size), int(grid_size)
    )


def _write_tiff_outputs(
    stem: Path,
    pred_sdf: np.ndarray,
    gt_sdf: np.ndarray,
    pred_sdf_clipped: np.ndarray,
    gt_sdf_clipped: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, str]:
    tifffile = _require_tifffile()
    paths = {
        "pred_sdf_tiff": str(stem.with_name(f"{stem.name}_pred_sdf.tif")),
        "gt_sdf_tiff": str(stem.with_name(f"{stem.name}_gt_sdf.tif")),
        "pred_sdf_clipped_tiff": str(stem.with_name(f"{stem.name}_pred_sdf_clipped.tif")),
        "gt_sdf_clipped_tiff": str(stem.with_name(f"{stem.name}_gt_sdf_clipped.tif")),
        "pred_mask_tiff": str(stem.with_name(f"{stem.name}_pred_mask.tif")),
        "gt_mask_tiff": str(stem.with_name(f"{stem.name}_gt_mask.tif")),
    }
    tifffile.imwrite(paths["pred_sdf_tiff"], pred_sdf.astype(np.float32, copy=False))
    tifffile.imwrite(paths["gt_sdf_tiff"], gt_sdf.astype(np.float32, copy=False))
    tifffile.imwrite(
        paths["pred_sdf_clipped_tiff"],
        pred_sdf_clipped.astype(np.float32, copy=False),
    )
    tifffile.imwrite(
        paths["gt_sdf_clipped_tiff"],
        gt_sdf_clipped.astype(np.float32, copy=False),
    )
    tifffile.imwrite(paths["pred_mask_tiff"], pred_mask.astype(np.uint8, copy=False))
    tifffile.imwrite(paths["gt_mask_tiff"], gt_mask.astype(np.uint8, copy=False))
    return paths


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QC_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in QC_CSV_FIELDS})


def _summarize_rows(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    csv_path: Path,
    json_path: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    metric_keys = [
        "mean_abs_sdf_error",
        "rmse_sdf_error",
        "mean_abs_clipped_sdf_error",
        "rmse_clipped_sdf_error",
        "dice",
        "iou",
        "accuracy",
        "precision",
        "recall",
        "pred_positive_fraction",
        "gt_positive_fraction",
        "mean_pred_sdf",
        "mean_gt_sdf",
        "mean_pred_sdf_inside",
        "mean_pred_sdf_outside",
        "mean_gt_sdf_inside",
        "mean_gt_sdf_outside",
    ]
    aggregate = {f"mean_{key}": _mean_numeric(rows, key) for key in metric_keys}
    return {
        "out": str(out_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "count": len(rows),
        "config": config,
        "aggregate": aggregate,
        "rows": list(rows),
    }


def run_sdf_grid_qc(
    checkpoint: str,
    mesh_root: str,
    out_dir: str,
    indices: Optional[Sequence[int]] = None,
    shape_ids: Optional[Sequence[str]] = None,
    synset: str = CHAIR_SYNSET,
    grid_size: int = 64,
    bounds: float = 1.15,
    sdf_clamp: Optional[float] = 0.1,
    threshold: float = 0.0,
    batch_points: int = 262144,
    gt_batch_points: Optional[int] = None,
    device: str = "cpu",
    write_tiff: bool = False,
    allow_non_watertight_gt: bool = False,
) -> Dict[str, object]:
    """Decode train latents and compare them to mesh-derived SDF grids."""
    if int(grid_size) <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if int(batch_points) <= 0:
        raise ValueError(f"batch_points must be positive, got {batch_points}")
    resolved_gt_batch_points = int(gt_batch_points) if gt_batch_points is not None else int(batch_points)
    if resolved_gt_batch_points <= 0:
        raise ValueError(f"gt_batch_points must be positive, got {resolved_gt_batch_points}")
    resolved_sdf_clamp = _resolved_sdf_clamp(sdf_clamp)

    decoder, _ = load_decoder_checkpoint(checkpoint, device=device)
    latents, train_shape_ids, _ = _load_checkpoint_latents(checkpoint, device=device)
    selected = _resolve_indices(indices or [], train_shape_ids, shape_ids or [])
    records = _discover_mesh_records(mesh_root, synset=synset)
    axis, points_np = _grid_points(int(grid_size), float(bounds))
    points = torch.as_tensor(points_np, dtype=torch.float32, device=device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for index in selected:
        shape_id = train_shape_ids[index]
        record = records.get(shape_id)
        if record is None:
            raise KeyError(
                f"No mesh found for checkpoint shape id {shape_id!r} under "
                f"{mesh_root!r} with synset {synset!r}."
            )

        mesh, center, scale = _load_normalized_mesh(record.mesh_path)
        mesh_is_watertight = bool(getattr(mesh, "is_watertight", False))
        _check_watertight_gt(mesh, allow_non_watertight_gt)
        gt_sdf = _gt_sdf_grid(
            mesh=mesh,
            points=points_np,
            grid_size=int(grid_size),
            gt_batch_points=resolved_gt_batch_points,
        )
        z = latents[index : index + 1]
        pred_sdf = _decode_sdf_grid(decoder, z, points, int(grid_size), int(batch_points))
        pred_sdf_clipped = _clip_sdf(pred_sdf, resolved_sdf_clamp)
        gt_sdf_clipped = _clip_sdf(gt_sdf, resolved_sdf_clamp)
        pred_mask = (pred_sdf < float(threshold)).astype(np.uint8)
        gt_mask = (gt_sdf < float(threshold)).astype(np.uint8)

        stem = out_path / f"{index:04d}_{_safe_shape_id(shape_id)}"
        npz_path = stem.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            pred_sdf_clipped=pred_sdf_clipped,
            gt_sdf_clipped=gt_sdf_clipped,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            grid_axis=axis,
            bounds=np.float32(bounds),
            sdf_clamp=np.asarray(
                np.nan if resolved_sdf_clamp is None else resolved_sdf_clamp,
                dtype=np.float32,
            ),
            sdf_clamp_enabled=np.asarray(resolved_sdf_clamp is not None),
            threshold=np.float32(threshold),
            shape_id=np.asarray(shape_id),
            train_index=np.int64(index),
            synset=np.asarray(synset),
            mesh_path=np.asarray(record.mesh_path),
            mesh_center=np.asarray(center, dtype=np.float32),
            mesh_scale=np.float32(scale),
            mesh_is_watertight=np.asarray(mesh_is_watertight),
            gt_batch_points=np.int64(resolved_gt_batch_points),
        )

        tiff_paths = {
            "pred_sdf_tiff": "",
            "gt_sdf_tiff": "",
            "pred_sdf_clipped_tiff": "",
            "gt_sdf_clipped_tiff": "",
            "pred_mask_tiff": "",
            "gt_mask_tiff": "",
        }
        if write_tiff:
            tiff_paths = _write_tiff_outputs(
                stem=stem,
                pred_sdf=pred_sdf,
                gt_sdf=gt_sdf,
                pred_sdf_clipped=pred_sdf_clipped,
                gt_sdf_clipped=gt_sdf_clipped,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

        row: Dict[str, object] = {
            "shape_id": shape_id,
            "train_index": int(index),
            "mesh_path": record.mesh_path,
            "npz": str(npz_path),
            "grid_size": int(grid_size),
            "bounds": float(bounds),
            "sdf_clamp": resolved_sdf_clamp,
            "threshold": float(threshold),
            "mesh_is_watertight": mesh_is_watertight,
            **tiff_paths,
        }
        row.update(
            sdf_grid_metrics(
                pred_sdf=pred_sdf,
                gt_sdf=gt_sdf,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                pred_sdf_clipped=pred_sdf_clipped,
                gt_sdf_clipped=gt_sdf_clipped,
            )
        )
        rows.append(row)

    csv_path = out_path / "sdf_grid_qc.csv"
    json_path = out_path / "sdf_grid_qc_summary.json"
    config = {
        "checkpoint": checkpoint,
        "mesh_root": mesh_root,
        "synset": synset,
        "indices": [int(index) for index in selected],
        "shape_ids": [train_shape_ids[index] for index in selected],
        "grid_size": int(grid_size),
        "bounds": float(bounds),
        "sdf_clamp": resolved_sdf_clamp,
        "threshold": float(threshold),
        "batch_points": int(batch_points),
        "gt_batch_points": int(resolved_gt_batch_points),
        "device": device,
        "write_tiff": bool(write_tiff),
        "allow_non_watertight_gt": bool(allow_non_watertight_gt),
    }
    summary = _summarize_rows(rows, out_path, csv_path, json_path, config)
    _write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare decoded DeepSDF grids against mesh-derived GT SDF grids."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", default=None, help="Comma-separated train latent indices.")
    parser.add_argument("--shape-ids", default=None, help="Comma-separated train shape ids.")
    parser.add_argument("--synset", default=CHAIR_SYNSET)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--bounds", type=float, default=1.15)
    parser.add_argument(
        "--sdf-clamp",
        type=float,
        default=0.1,
        help="Symmetric clamp for clipped SDF outputs/metrics. Use 0 to disable.",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-points", type=int, default=262144)
    parser.add_argument(
        "--gt-batch-points",
        type=int,
        default=None,
        help="Points per GT signed-distance chunk. Defaults to --batch-points.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--write-tiff",
        dest="write_tiff",
        action="store_true",
        default=False,
        help="Write pred/GT SDF, clipped SDF, and mask TIFF stacks.",
    )
    parser.add_argument(
        "--no-tiff",
        dest="write_tiff",
        action="store_false",
        help="Do not write TIFF stacks.",
    )
    parser.add_argument(
        "--allow-non-watertight-gt",
        action="store_true",
        help="Allow mesh-derived GT SDF even when the mesh is not watertight.",
    )
    args = parser.parse_args(argv)
    summary = run_sdf_grid_qc(
        checkpoint=args.checkpoint,
        mesh_root=args.mesh_root,
        out_dir=args.out,
        indices=_parse_int_list(args.indices),
        shape_ids=_parse_str_list(args.shape_ids),
        synset=args.synset,
        grid_size=args.grid_size,
        bounds=args.bounds,
        sdf_clamp=args.sdf_clamp,
        threshold=args.threshold,
        batch_points=args.batch_points,
        gt_batch_points=args.gt_batch_points,
        device=args.device,
        write_tiff=args.write_tiff,
        allow_non_watertight_gt=args.allow_non_watertight_gt,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
