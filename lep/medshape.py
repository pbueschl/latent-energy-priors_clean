"""Helpers for public MedShape-style diverse SDF experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .shapenet import MESH_LAYOUTS, discover_meshes, load_split, save_split


DEFAULT_MEDSHAPE_SYNSET = "medshape"
MANIFEST_REQUIRED_COLUMNS = ("class", "filename", "shape_id", "url")
OBSERVATION_COLUMNS = (
    "observation_points",
    "num_observations",
    "num_observation_points",
    "obs_count",
)
AGGREGATE_METRICS = (
    "obs_l1",
    "eval_l1",
    "obs_bce",
    "eval_bce",
    "eval_acc",
    "eval_iou",
    "eval_dice",
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "grid_mean_abs_sdf_error",
    "grid_rmse_sdf_error",
    "grid_mean_abs_clipped_sdf_error",
    "grid_rmse_clipped_sdf_error",
    "latent_norm",
    "conditional_energy",
)


@dataclass(frozen=True)
class ManifestRecord:
    class_label: str
    filename: str
    shape_id: str
    url: str


def _clean_cell(row: Dict[str, str], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def read_manifest(path: str) -> List[ManifestRecord]:
    """Read a MedShape-style manifest CSV."""
    records: List[ManifestRecord] = []
    with Path(path).expanduser().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Manifest {path} is missing required columns: {missing}")
        seen: Dict[str, ManifestRecord] = {}
        for row in reader:
            filename = _clean_cell(row, "filename")
            shape_id = _clean_cell(row, "shape_id") or Path(filename).stem
            record = ManifestRecord(
                class_label=_clean_cell(row, "class"),
                filename=filename,
                shape_id=shape_id,
                url=_clean_cell(row, "url"),
            )
            if not record.class_label:
                raise ValueError(f"Manifest row for shape {shape_id!r} has an empty class")
            if not record.filename:
                raise ValueError(f"Manifest row for shape {shape_id!r} has an empty filename")
            previous = seen.get(record.shape_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate shape_id {record.shape_id!r} in manifest for "
                    f"classes {previous.class_label!r} and {record.class_label!r}"
                )
            seen[record.shape_id] = record
            records.append(record)
    if not records:
        raise ValueError(f"Manifest {path} contains no records")
    return records


def _scan_sdf_ids(sdf_root: str) -> set:
    root = Path(sdf_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"SDF root does not exist: {root}")
    shape_ids = set()
    for path in sorted(root.rglob("*.npz")):
        if path.name == "samples.npz":
            shape_ids.add(path.parent.name)
        else:
            shape_ids.add(path.stem)
    return shape_ids


def _available_id_sets(
    mesh_root: Optional[str],
    sdf_root: Optional[str],
    synset: str,
    mesh_layout: str,
) -> Dict[str, set]:
    available: Dict[str, set] = {}
    if mesh_root is not None:
        available["mesh"] = {
            record.shape_id
            for record in discover_meshes(mesh_root, synset=synset, layout=mesh_layout)
        }
    if sdf_root is not None:
        available["sdf"] = _scan_sdf_ids(sdf_root)
    return available


def _filter_available_records(
    records: Sequence[ManifestRecord],
    available: Dict[str, set],
) -> Tuple[Dict[str, List[ManifestRecord]], Dict[str, List[Dict[str, str]]]]:
    by_class: Dict[str, List[ManifestRecord]] = defaultdict(list)
    missing_by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        missing_sources = [
            source for source, ids in available.items() if record.shape_id not in ids
        ]
        if missing_sources:
            missing_by_class[record.class_label].append(
                {
                    "shape_id": record.shape_id,
                    "filename": record.filename,
                    "missing": ",".join(missing_sources),
                }
            )
        else:
            by_class[record.class_label].append(record)
    return dict(by_class), dict(missing_by_class)


def _split_counts(n: int, train_fraction: float, val_fraction: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _split_selected_records(
    records_by_class: Dict[str, List[ManifestRecord]],
    seed: int,
    per_class_target: Optional[int],
    train_fraction: float,
    val_fraction: float,
) -> Tuple[Dict[str, List[str]], Dict[str, List[ManifestRecord]], Dict[str, Dict[str, int]]]:
    if not records_by_class:
        raise ValueError("No manifest records are available after filtering")

    classes = sorted(records_by_class)
    if per_class_target is None:
        target = min(len(records_by_class[class_label]) for class_label in classes)
    else:
        target = int(per_class_target)
    if target <= 0:
        raise ValueError(f"per_class_target must be positive after filtering, got {target}")

    rng = random.Random(int(seed))
    split = {"train": [], "val": [], "test": []}
    selected_by_class: Dict[str, List[ManifestRecord]] = {}
    counts_by_class: Dict[str, Dict[str, int]] = {}
    for class_label in classes:
        records = sorted(records_by_class[class_label], key=lambda record: record.shape_id)
        rng.shuffle(records)
        selected = records[: min(target, len(records))]
        selected_by_class[class_label] = selected
        n_train, n_val, n_test = _split_counts(
            len(selected),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        train = selected[:n_train]
        val = selected[n_train : n_train + n_val]
        test = selected[n_train + n_val : n_train + n_val + n_test]
        split["train"].extend(record.shape_id for record in train)
        split["val"].extend(record.shape_id for record in val)
        split["test"].extend(record.shape_id for record in test)
        counts_by_class[class_label] = {
            "available": len(records),
            "selected": len(selected),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }
    return split, selected_by_class, counts_by_class


def build_balanced_split_from_manifest(
    manifest_csv: str,
    synset: str = DEFAULT_MEDSHAPE_SYNSET,
    seed: int = 13,
    per_class_target: Optional[int] = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    mesh_root: Optional[str] = None,
    sdf_root: Optional[str] = None,
    mesh_layout: str = "class_subdirs",
) -> Dict[str, object]:
    """Build a class-balanced train/val/test split from a MedShape manifest."""
    if mesh_layout not in MESH_LAYOUTS:
        raise ValueError(f"Unknown mesh layout {mesh_layout!r}; expected one of {MESH_LAYOUTS}")
    records = read_manifest(manifest_csv)
    available_sets = _available_id_sets(
        mesh_root=mesh_root,
        sdf_root=sdf_root,
        synset=synset,
        mesh_layout=mesh_layout,
    )
    records_by_class, missing_by_class = _filter_available_records(records, available_sets)
    split_ids, selected_by_class, counts_by_class = _split_selected_records(
        records_by_class,
        seed=seed,
        per_class_target=per_class_target,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    selected_records = [
        record
        for class_label in sorted(selected_by_class)
        for record in selected_by_class[class_label]
    ]
    selected_shape_ids = {record.shape_id for record in selected_records}
    classes = sorted({record.class_label for record in records})
    target_for_report = (
        int(per_class_target)
        if per_class_target is not None
        else min(counts["available"] for counts in counts_by_class.values())
    )
    replacement_needed = {
        class_label: max(0, target_for_report - counts_by_class.get(class_label, {}).get("available", 0))
        for class_label in classes
    }
    split = {
        "synset": synset,
        "mesh_layout": mesh_layout,
        "seed": int(seed),
        "source": "medshape_manifest",
        "source_manifest_name": Path(manifest_csv).name,
        "classes": classes,
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
        "class_map": {record.shape_id: record.class_label for record in selected_records},
        "filename_map": {record.shape_id: record.filename for record in selected_records},
        "url_map": {record.shape_id: record.url for record in selected_records},
        "counts": {
            "total_selected": len(selected_shape_ids),
            "by_class": counts_by_class,
        },
        "availability": {
            "mesh_root_checked": mesh_root is not None,
            "sdf_root_checked": sdf_root is not None,
            "manifest_count": len(records),
            "available_count": sum(count["available"] for count in counts_by_class.values()),
        },
        "missing_by_class": missing_by_class,
        "replacement_needed_by_class": replacement_needed,
    }
    return split


def _class_counts(split: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    class_map = {str(k): str(v) for k, v in dict(split.get("class_map", {})).items()}
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name in ("train", "val", "test"):
        for shape_id in list(split.get(split_name, [])):
            class_label = class_map.get(str(shape_id), "UNKNOWN")
            counts[class_label][split_name] += 1
    return dict(counts)


def write_balanced_split(split: Dict[str, object], out: str) -> Dict[str, object]:
    save_split(split, out)
    return {
        "out": out,
        "counts": _class_counts(split),
        "replacement_needed_by_class": split.get("replacement_needed_by_class", {}),
        "missing_by_class": split.get("missing_by_class", {}),
    }


def _load_class_map(
    manifest_csv: Optional[str] = None,
    split_file: Optional[str] = None,
) -> Dict[str, str]:
    class_map: Dict[str, str] = {}
    if split_file is not None:
        split = load_split(split_file)
        class_map.update({str(k): str(v) for k, v in dict(split.get("class_map", {})).items()})
    if manifest_csv is not None:
        class_map.update({record.shape_id: record.class_label for record in read_manifest(manifest_csv)})
    return class_map


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _float_or_none(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    return float(value)


def _detect_observation_column(fieldnames: Sequence[str]) -> Optional[str]:
    for column in OBSERVATION_COLUMNS:
        if column in fieldnames:
            return column
    return None


def _aggregate_rows(
    rows: Sequence[Dict[str, str]],
    class_map: Dict[str, str],
    metrics: Sequence[str],
    observation_column: Optional[str],
    include_class: bool,
    missing_class_label: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    group_values: Dict[Tuple[object, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[object, ...], int] = defaultdict(int)
    missing_shape_ids = set()

    for row in rows:
        shape_id = str(row.get("shape_id", "") or "")
        class_label = class_map.get(shape_id)
        if class_label is None:
            class_label = missing_class_label
            if shape_id:
                missing_shape_ids.add(shape_id)
        method = row.get("method", "") or ""
        variant = row.get("variant", "") or method
        method = method or variant
        step = int(float(row.get("step", 0)))
        key_parts: List[object] = []
        if include_class:
            key_parts.append(class_label)
        key_parts.extend([variant, method])
        if observation_column is not None:
            key_parts.append(row.get(observation_column, ""))
        key_parts.append(step)
        key = tuple(key_parts)
        counts[key] += 1
        for metric in metrics:
            value = _float_or_none(row.get(metric, ""))
            if value is not None:
                group_values[key][metric].append(value)

    out_rows: List[Dict[str, object]] = []
    for key in sorted(group_values.keys()):
        offset = 0
        out: Dict[str, object] = {}
        if include_class:
            out["class"] = key[0]
            offset = 1
        else:
            out["class"] = "ALL"
        out["variant"] = key[offset]
        out["method"] = key[offset + 1]
        metric_offset = offset + 2
        if observation_column is not None:
            out[observation_column] = key[metric_offset]
            metric_offset += 1
        out["step"] = key[metric_offset]
        out["n"] = counts[key]
        for metric in metrics:
            values = group_values[key].get(metric, [])
            if values:
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_std"] = _std(values)
        out_rows.append(out)
    return out_rows, sorted(missing_shape_ids)


def _write_aggregate_csv(
    rows: Sequence[Dict[str, object]],
    out: str,
    metrics: Sequence[str],
    observation_column: Optional[str],
) -> None:
    fieldnames = ["class", "variant", "method"]
    if observation_column is not None:
        fieldnames.append(observation_column)
    fieldnames.extend(["step", "n"])
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_sparse_sweep_by_class(
    sweep_csv: str,
    out_class_csv: str,
    manifest_csv: Optional[str] = None,
    split_file: Optional[str] = None,
    out_overall_csv: Optional[str] = None,
    summary_json: Optional[str] = None,
    missing_class_label: str = "UNKNOWN",
) -> Dict[str, object]:
    """Aggregate sparse-sweep metrics by manifest/split class labels."""
    class_map = _load_class_map(manifest_csv=manifest_csv, split_file=split_file)
    with Path(sweep_csv).expanduser().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    observation_column = _detect_observation_column(fieldnames)
    metrics = [metric for metric in AGGREGATE_METRICS if metric in fieldnames]
    if not metrics:
        raise ValueError(f"No aggregate metric columns found in {sweep_csv}")

    class_rows, missing_shape_ids = _aggregate_rows(
        rows=rows,
        class_map=class_map,
        metrics=metrics,
        observation_column=observation_column,
        include_class=True,
        missing_class_label=missing_class_label,
    )
    _write_aggregate_csv(class_rows, out_class_csv, metrics, observation_column)

    overall_rows: List[Dict[str, object]] = []
    if out_overall_csv is not None:
        overall_rows, _ = _aggregate_rows(
            rows=rows,
            class_map=class_map,
            metrics=metrics,
            observation_column=observation_column,
            include_class=False,
            missing_class_label=missing_class_label,
        )
        _write_aggregate_csv(overall_rows, out_overall_csv, metrics, observation_column)

    summary = {
        "sweep_csv": sweep_csv,
        "out_class_csv": out_class_csv,
        "out_overall_csv": out_overall_csv or "",
        "summary_json": summary_json or "",
        "rows": len(rows),
        "class_groups": len(class_rows),
        "overall_groups": len(overall_rows),
        "metrics": metrics,
        "observation_column": observation_column or "",
        "missing_class_label": missing_class_label,
        "missing_class_shape_ids": missing_shape_ids,
    }
    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    return summary


def split_main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create a balanced MedShape split from a manifest CSV.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mesh-root", default=None)
    parser.add_argument("--sdf-root", default=None)
    parser.add_argument("--mesh-layout", default="class_subdirs", choices=MESH_LAYOUTS)
    parser.add_argument("--synset", default=DEFAULT_MEDSHAPE_SYNSET)
    parser.add_argument("--per-class-target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    split = build_balanced_split_from_manifest(
        manifest_csv=args.manifest_csv,
        synset=args.synset,
        seed=args.seed,
        per_class_target=args.per_class_target,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        mesh_root=args.mesh_root,
        sdf_root=args.sdf_root,
        mesh_layout=args.mesh_layout,
    )
    summary = write_balanced_split(split, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


def aggregate_main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate sparse-sweep metrics by class.")
    parser.add_argument("sweep_csv")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--out-class", required=True)
    parser.add_argument("--out-overall", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--missing-class-label", default="UNKNOWN")
    args = parser.parse_args(argv)
    summary = aggregate_sparse_sweep_by_class(
        sweep_csv=args.sweep_csv,
        out_class_csv=args.out_class,
        manifest_csv=args.manifest_csv,
        split_file=args.split_file,
        out_overall_csv=args.out_overall,
        summary_json=args.summary_json,
        missing_class_label=args.missing_class_label,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "ManifestRecord",
    "aggregate_sparse_sweep_by_class",
    "build_balanced_split_from_manifest",
    "read_manifest",
    "write_balanced_split",
]
