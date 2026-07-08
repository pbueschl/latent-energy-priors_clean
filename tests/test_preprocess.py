import csv
import json

import numpy as np

from lep import preprocess
from lep.shapenet import ShapeRecord


def test_build_preprocess_tasks_deterministic_seeds_and_skip(tmp_path):
    out_dir = tmp_path / "sdf"
    out_dir.mkdir()
    (out_dir / "shape_a.npz").write_bytes(b"existing")
    records = {
        shape_id: ShapeRecord("03001627", shape_id, f"meshes/{shape_id}.obj")
        for shape_id in ("shape_a", "shape_b", "shape_c")
    }

    tasks, skipped, failed = preprocess.build_preprocess_tasks(
        ["shape_a", "shape_b", "shape_c"],
        records,
        str(out_dir),
        seed=100,
        overwrite=False,
    )
    overwrite_tasks, overwrite_skipped, _ = preprocess.build_preprocess_tasks(
        ["shape_a", "shape_b", "shape_c"],
        records,
        str(out_dir),
        seed=100,
        overwrite=True,
    )

    assert failed == []
    assert [result.shape_id for result in skipped] == ["shape_a"]
    assert skipped[0].seed == 100
    assert [task.shape_id for task in tasks] == ["shape_b", "shape_c"]
    assert [task.seed for task in tasks] == [101, 102]
    assert [task.out_path for task in tasks] == [
        str(out_dir / "shape_b.npz"),
        str(out_dir / "shape_c.npz"),
    ]
    assert overwrite_skipped == []
    assert [task.seed for task in overwrite_tasks] == [100, 101, 102]


def test_preprocess_split_continue_on_error_writes_failure_log(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "03001627", "train": ["good", "bad"], "val": [], "test": []}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "sdf"
    failure_log = tmp_path / "failures.csv"

    def fake_discover_meshes(mesh_root, synset, layout="auto"):
        return [
            ShapeRecord(synset, "good", "good.obj"),
            ShapeRecord(synset, "bad", "bad.obj"),
        ]

    def fake_sample_sdf_for_mesh(mesh_path, **kwargs):
        if mesh_path == "bad.obj":
            raise ValueError("broken mesh")
        return {
            "points": np.zeros((2, 3), dtype=np.float32),
            "sdf": np.zeros(2, dtype=np.float32),
            "mesh_center": np.zeros(3, dtype=np.float32),
            "mesh_scale": np.float32(1.0),
        }

    monkeypatch.setattr(preprocess, "discover_meshes", fake_discover_meshes)
    monkeypatch.setattr(preprocess, "sample_sdf_for_mesh", fake_sample_sdf_for_mesh)

    summary = preprocess.preprocess_split(
        mesh_root="unused",
        split_file=str(split_file),
        out_dir=str(out_dir),
        samples=2,
        surface_samples=0,
        workers=1,
        failure_log=str(failure_log),
        continue_on_error=True,
    )

    assert summary["processed"] == 1
    assert summary["failed"] == 1
    assert (out_dir / "good.npz").exists()
    with failure_log.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "shape_id": "bad",
            "mesh_path": "bad.obj",
            "error_type": "ValueError",
            "error_message": "broken mesh",
        }
    ]


def test_preprocess_split_raises_without_continue_on_error(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "03001627", "train": ["bad"], "val": [], "test": []}),
        encoding="utf-8",
    )
    failure_log = tmp_path / "failures.csv"

    monkeypatch.setattr(
        preprocess,
        "discover_meshes",
        lambda mesh_root, synset, layout="auto": [ShapeRecord(synset, "bad", "bad.obj")],
    )
    monkeypatch.setattr(
        preprocess,
        "sample_sdf_for_mesh",
        lambda mesh_path, **kwargs: (_ for _ in ()).throw(ValueError("broken mesh")),
    )

    try:
        preprocess.preprocess_split(
            mesh_root="unused",
            split_file=str(split_file),
            out_dir=str(tmp_path / "sdf"),
            workers=1,
            failure_log=str(failure_log),
            continue_on_error=False,
        )
    except RuntimeError as exc:
        assert "Preprocessing failed for bad" in str(exc)
    else:
        raise AssertionError("Expected preprocessing to fail without --continue-on-error")

    with failure_log.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["shape_id"] == "bad"
