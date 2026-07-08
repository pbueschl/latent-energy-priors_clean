"""ShapeNet chair discovery and split helpers."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


CHAIR_SYNSET = "03001627"
DEFAULT_MESH_FILENAME = "models/model_normalized.obj"
MESH_EXTENSIONS = (".obj", ".ply", ".off", ".stl")
MESH_LAYOUTS = ("auto", "shapenet", "flat", "class_subdirs")


@dataclass(frozen=True)
class ShapeRecord:
    synset: str
    shape_id: str
    mesh_path: str


def _synset_root(mesh_root: Path, synset: str) -> Path:
    if mesh_root.name == synset:
        return mesh_root
    candidate = mesh_root / synset
    return candidate if candidate.exists() else mesh_root


def _is_mesh_file(path: Path, extensions: Sequence[str]) -> bool:
    suffixes = {ext.lower() for ext in extensions}
    return path.is_file() and path.suffix.lower() in suffixes


def _add_mesh_record(
    records: List[ShapeRecord],
    seen: Dict[str, str],
    synset: str,
    shape_id: str,
    mesh_path: Path,
) -> None:
    if shape_id in seen:
        if seen[shape_id] != str(mesh_path):
            raise ValueError(
                f"Duplicate shape id {shape_id!r} for meshes {seen[shape_id]!r} "
                f"and {str(mesh_path)!r}. Use unique filename stems or manifest shape ids."
            )
        return
    records.append(ShapeRecord(synset=synset, shape_id=shape_id, mesh_path=str(mesh_path)))
    seen[shape_id] = str(mesh_path)


def _discover_shapenet_layout(
    root: Path,
    synset: str,
    mesh_filename: str,
    extensions: Sequence[str],
) -> List[ShapeRecord]:
    records: List[ShapeRecord] = []
    seen: Dict[str, str] = {}
    for shape_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        expected = shape_dir / mesh_filename
        mesh_path: Optional[Path] = expected if expected.exists() else None
        if mesh_path is None:
            for ext in extensions:
                matches = sorted(shape_dir.rglob(f"*{ext}"))
                if matches:
                    mesh_path = matches[0]
                    break
        if mesh_path is None:
            continue
        _add_mesh_record(records, seen, synset, shape_dir.name, mesh_path)
    return records


def _discover_flat_layout(
    root: Path,
    synset: str,
    extensions: Sequence[str],
) -> List[ShapeRecord]:
    records: List[ShapeRecord] = []
    seen: Dict[str, str] = {}
    for mesh_path in sorted(p for p in root.iterdir() if _is_mesh_file(p, extensions)):
        _add_mesh_record(records, seen, synset, mesh_path.stem, mesh_path)
    return records


def _discover_class_subdirs_layout(
    root: Path,
    synset: str,
    extensions: Sequence[str],
) -> List[ShapeRecord]:
    records: List[ShapeRecord] = []
    seen: Dict[str, str] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mesh_path in sorted(p for p in class_dir.rglob("*") if _is_mesh_file(p, extensions)):
            _add_mesh_record(records, seen, synset, mesh_path.stem, mesh_path)
    return records


def _discover_recursive_fallback(
    root: Path,
    synset: str,
    extensions: Sequence[str],
) -> List[ShapeRecord]:
    records: List[ShapeRecord] = []
    seen = set()
    for ext in extensions:
        for mesh_path in sorted(root.rglob(f"*{ext}")):
            shape_id = mesh_path.parent.parent.name if mesh_path.parent.name == "models" else mesh_path.stem
            if shape_id in seen:
                continue
            records.append(ShapeRecord(synset=synset, shape_id=shape_id, mesh_path=str(mesh_path)))
            seen.add(shape_id)
    return records


def discover_meshes(
    mesh_root: str,
    synset: str = CHAIR_SYNSET,
    mesh_filename: str = DEFAULT_MESH_FILENAME,
    extensions: Sequence[str] = MESH_EXTENSIONS,
    layout: str = "auto",
) -> List[ShapeRecord]:
    """Discover meshes without baking paths into split files.

    `layout="auto"` preserves the historical ShapeNet behavior: direct
    subdirectories are treated as shape directories, with a recursive fallback
    only if no such records are found. Use `layout="class_subdirs"` for public
    MedShape-style roots such as `mesh_root/skull_mri/001_skull.stl`, where
    each mesh file is a separate shape and the shape id is the filename stem.
    """
    if layout not in MESH_LAYOUTS:
        raise ValueError(f"Unknown mesh layout {layout!r}; expected one of {MESH_LAYOUTS}")
    root = _synset_root(Path(mesh_root).expanduser(), synset)
    if not root.exists():
        raise FileNotFoundError(f"Mesh root does not exist: {root}")

    if layout == "flat":
        return _discover_flat_layout(root, synset, extensions)
    if layout == "class_subdirs":
        return _discover_class_subdirs_layout(root, synset, extensions)
    if layout == "shapenet":
        return _discover_shapenet_layout(root, synset, mesh_filename, extensions)

    records = _discover_shapenet_layout(root, synset, mesh_filename, extensions)
    if records:
        return records
    return _discover_recursive_fallback(root, synset, extensions)


def create_split(
    shape_ids: Iterable[str],
    synset: str = CHAIR_SYNSET,
    seed: int = 13,
    limit: Optional[int] = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Dict[str, object]:
    ids = sorted(dict.fromkeys(str(shape_id) for shape_id in shape_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if limit is not None:
        ids = ids[: int(limit)]
    n = len(ids)
    if n == 0:
        raise ValueError("Cannot create a split from zero shape ids")

    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val

    return {
        "synset": synset,
        "seed": seed,
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val : n_train + n_val + n_test],
    }


def save_split(split: Dict[str, object], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)
        f.write("\n")


def load_split(path: str) -> Dict[str, object]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def mesh_map(records: Sequence[ShapeRecord]) -> Dict[str, ShapeRecord]:
    return {record.shape_id: record for record in records}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create ShapeNet chair split JSON.")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synset", default=CHAIR_SYNSET)
    parser.add_argument("--mesh-layout", default="auto", choices=MESH_LAYOUTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)

    records = discover_meshes(args.mesh_root, synset=args.synset, layout=args.mesh_layout)
    split = create_split(
        [record.shape_id for record in records],
        synset=args.synset,
        seed=args.seed,
        limit=args.limit,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    save_split(split, args.out)
    split_counts = {key: len(split[key]) for key in ("train", "val", "test")}
    print(json.dumps({"out": args.out, "counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
