"""Mesh-to-SDF preprocessing entry point with lazy optional dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .shapenet import CHAIR_SYNSET, MESH_LAYOUTS, discover_meshes, load_split, mesh_map


@dataclass(frozen=True)
class PreprocessTask:
    shape_id: str
    mesh_path: str
    out_path: str
    seed: int
    samples: int
    surface_samples: int
    bbox_scale: float
    sdf_clip: Optional[float]
    overwrite: bool


@dataclass(frozen=True)
class PreprocessResult:
    shape_id: str
    mesh_path: str
    out_path: str
    status: str
    seed: int
    error_type: str = ""
    error_message: str = ""


def _require_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Mesh preprocessing requires trimesh. Install with "
            '`pip install -e ".[preprocess]"`.'
        ) from exc
    return trimesh


def _signed_distance(mesh, points: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    try:
        import point_cloud_utils as pcu  # type: ignore

        result = pcu.signed_distance_to_mesh(points.astype(np.float64), vertices, faces)
        return np.asarray(result[0] if isinstance(result, tuple) else result, dtype=np.float32)
    except ImportError:
        pass

    try:
        from trimesh.proximity import signed_distance  # type: ignore

        return np.asarray(signed_distance(mesh, points), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Could not compute signed distances. Install point-cloud-utils, or "
            "install trimesh with the proximity dependencies needed by your platform."
        ) from exc


def _normalize_mesh(mesh):
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float(np.max(bounds[1] - bounds[0]) / 2.0)
    if scale <= 0:
        raise ValueError("Mesh has degenerate bounds")
    mesh = mesh.copy()
    mesh.vertices = (np.asarray(mesh.vertices) - center[None, :]) / scale
    return mesh, center.astype(np.float32), np.float32(scale)


def sample_sdf_for_mesh(
    mesh_path: str,
    samples: int = 50000,
    surface_samples: int = 50000,
    bbox_scale: float = 1.15,
    sdf_clip: Optional[float] = 0.1,
    seed: int = 13,
) -> dict:
    """Sample SDF values near the surface and in a bounding cube."""
    trimesh = _require_trimesh()
    rng = np.random.default_rng(seed)
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    mesh, center, scale = _normalize_mesh(mesh)

    uniform = rng.uniform(
        low=-float(bbox_scale),
        high=float(bbox_scale),
        size=(int(samples), 3),
    ).astype(np.float32)

    surface_points = np.empty((0, 3), dtype=np.float32)
    if surface_samples > 0:
        surface_points, _ = trimesh.sample.sample_surface(mesh, int(surface_samples))
        surface_points = np.asarray(surface_points, dtype=np.float32)
        noise_scale = rng.choice([0.005, 0.02, 0.05], size=(surface_points.shape[0], 1))
        surface_points = surface_points + rng.normal(size=surface_points.shape).astype(np.float32) * noise_scale

    points = np.concatenate([uniform, surface_points], axis=0).astype(np.float32)
    sdf = _signed_distance(mesh, points).reshape(-1).astype(np.float32)
    if sdf_clip is not None and sdf_clip > 0:
        sdf = np.clip(sdf, -float(sdf_clip), float(sdf_clip))
    return {
        "points": points,
        "sdf": sdf,
        "mesh_center": center,
        "mesh_scale": scale,
    }


def _shape_ids_for_split(split: Dict[str, object], split_name: str) -> List[str]:
    if split_name == "all":
        return [
            str(shape_id)
            for key in ("train", "val", "test")
            for shape_id in list(split.get(key, []))
        ]
    return [str(shape_id) for shape_id in list(split.get(split_name, []))]


def build_preprocess_tasks(
    shape_ids: Sequence[str],
    records: Dict[str, object],
    out_dir: str,
    samples: int = 50000,
    surface_samples: int = 50000,
    bbox_scale: float = 1.15,
    sdf_clip: Optional[float] = 0.1,
    seed: int = 13,
    overwrite: bool = False,
) -> Tuple[List[PreprocessTask], List[PreprocessResult], List[PreprocessResult]]:
    """Build deterministic one-shape tasks plus skipped/missing results."""
    out_path = Path(out_dir)
    tasks: List[PreprocessTask] = []
    skipped: List[PreprocessResult] = []
    failed: List[PreprocessResult] = []
    for offset, shape_id in enumerate(shape_ids):
        shape_id = str(shape_id)
        target = out_path / f"{shape_id}.npz"
        task_seed = int(seed) + offset
        record = records.get(shape_id)
        if record is None:
            failed.append(
                PreprocessResult(
                    shape_id=shape_id,
                    mesh_path="",
                    out_path=str(target),
                    status="failed",
                    seed=task_seed,
                    error_type="FileNotFoundError",
                    error_message=f"No mesh found for shape id {shape_id!r}",
                )
            )
            continue
        mesh_path = str(getattr(record, "mesh_path"))
        if target.exists() and not overwrite:
            skipped.append(
                PreprocessResult(
                    shape_id=shape_id,
                    mesh_path=mesh_path,
                    out_path=str(target),
                    status="skipped",
                    seed=task_seed,
                )
            )
            continue
        tasks.append(
            PreprocessTask(
                shape_id=shape_id,
                mesh_path=mesh_path,
                out_path=str(target),
                seed=task_seed,
                samples=int(samples),
                surface_samples=int(surface_samples),
                bbox_scale=float(bbox_scale),
                sdf_clip=sdf_clip,
                overwrite=bool(overwrite),
            )
        )
    return tasks, skipped, failed


def _process_one_shape(task: PreprocessTask) -> PreprocessResult:
    target = Path(task.out_path)
    try:
        if target.exists() and not task.overwrite:
            return PreprocessResult(
                shape_id=task.shape_id,
                mesh_path=task.mesh_path,
                out_path=task.out_path,
                status="skipped",
                seed=task.seed,
            )
        data = sample_sdf_for_mesh(
            task.mesh_path,
            samples=task.samples,
            surface_samples=task.surface_samples,
            bbox_scale=task.bbox_scale,
            sdf_clip=task.sdf_clip,
            seed=task.seed,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            points=data["points"],
            sdf=data["sdf"],
            shape_id=np.asarray(task.shape_id),
            mesh_center=data["mesh_center"],
            mesh_scale=data["mesh_scale"],
        )
        return PreprocessResult(
            shape_id=task.shape_id,
            mesh_path=task.mesh_path,
            out_path=task.out_path,
            status="processed",
            seed=task.seed,
        )
    except Exception as exc:
        return PreprocessResult(
            shape_id=task.shape_id,
            mesh_path=task.mesh_path,
            out_path=task.out_path,
            status="failed",
            seed=task.seed,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _write_failure_log(path: str, failures: Sequence[PreprocessResult]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shape_id", "mesh_path", "error_type", "error_message"],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "shape_id": failure.shape_id,
                    "mesh_path": failure.mesh_path,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                }
            )


def _raise_first_failure(failures: Sequence[PreprocessResult]) -> None:
    if not failures:
        return
    failure = failures[0]
    raise RuntimeError(
        f"Preprocessing failed for {failure.shape_id}: "
        f"{failure.error_type}: {failure.error_message}"
    )


def _run_preprocess_tasks(
    tasks: Sequence[PreprocessTask],
    workers: int,
    continue_on_error: bool,
) -> List[PreprocessResult]:
    if not tasks:
        return []
    if workers <= 1:
        results: List[PreprocessResult] = []
        for task in tqdm(tasks, desc="preprocess"):
            result = _process_one_shape(task)
            results.append(result)
            if result.status == "failed" and not continue_on_error:
                break
        return results

    results = []
    executor = ProcessPoolExecutor(max_workers=int(workers))
    try:
        futures = [executor.submit(_process_one_shape, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
            result = future.result()
            results.append(result)
            if result.status == "failed" and not continue_on_error:
                for pending in futures:
                    pending.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=not continue_on_error)
    return results


def _summarize_results(
    results: Sequence[PreprocessResult],
    out_dir: str,
    workers: int,
    failure_log: str,
) -> Dict[str, object]:
    summary = {
        "out": out_dir,
        "workers": int(workers),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failure_log": failure_log,
    }
    for result in results:
        if result.status == "processed":
            summary["processed"] = int(summary["processed"]) + 1
        elif result.status == "skipped":
            summary["skipped"] = int(summary["skipped"]) + 1
        elif result.status == "failed":
            summary["failed"] = int(summary["failed"]) + 1
    summary["total"] = int(summary["processed"]) + int(summary["skipped"]) + int(summary["failed"])
    return summary


def preprocess_split(
    mesh_root: str,
    split_file: str,
    out_dir: str,
    split_name: str = "all",
    synset: str = CHAIR_SYNSET,
    mesh_layout: str = "auto",
    samples: int = 50000,
    surface_samples: int = 50000,
    bbox_scale: float = 1.15,
    sdf_clip: Optional[float] = 0.1,
    seed: int = 13,
    overwrite: bool = False,
    workers: int = 1,
    failure_log: Optional[str] = None,
    continue_on_error: bool = False,
) -> Dict[str, object]:
    split = load_split(split_file)
    records = mesh_map(
        discover_meshes(
            mesh_root,
            synset=str(split.get("synset", synset)),
            layout=str(split.get("mesh_layout", mesh_layout)),
        )
    )
    shape_ids = _shape_ids_for_split(split, split_name)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    failure_log = failure_log or str(out_path / "preprocess_failures.csv")
    tasks, skipped, missing_failures = build_preprocess_tasks(
        shape_ids=shape_ids,
        records=records,
        out_dir=str(out_path),
        samples=samples,
        surface_samples=surface_samples,
        bbox_scale=bbox_scale,
        sdf_clip=sdf_clip,
        seed=seed,
        overwrite=overwrite,
    )
    task_results = _run_preprocess_tasks(
        tasks,
        workers=int(workers),
        continue_on_error=continue_on_error,
    )
    results = [*skipped, *missing_failures, *task_results]
    failures = [result for result in results if result.status == "failed"]
    _write_failure_log(failure_log, failures)
    summary = _summarize_results(results, str(out_path), int(workers), failure_log)
    if failures and not continue_on_error:
        _raise_first_failure(failures)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess ShapeNet meshes into SDF .npz files.")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--synset", default=CHAIR_SYNSET)
    parser.add_argument("--mesh-layout", default="auto", choices=MESH_LAYOUTS)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--surface-samples", type=int, default=50000)
    parser.add_argument("--bbox-scale", type=float, default=1.15)
    parser.add_argument("--sdf-clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--failure-log", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    summary = preprocess_split(
        mesh_root=args.mesh_root,
        split_file=args.split_file,
        out_dir=args.out,
        split_name=args.split,
        synset=args.synset,
        mesh_layout=args.mesh_layout,
        samples=args.samples,
        surface_samples=args.surface_samples,
        bbox_scale=args.bbox_scale,
        sdf_clip=args.sdf_clip,
        seed=args.seed,
        overwrite=args.overwrite,
        workers=args.workers,
        failure_log=args.failure_log,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
