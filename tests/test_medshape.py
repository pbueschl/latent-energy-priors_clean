import csv
import json

from lep.medshape import aggregate_sparse_sweep_by_class, build_balanced_split_from_manifest


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_balanced_split_from_manifest_with_mesh_availability(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = []
    for class_label, count in [("skull_mri", 4), ("brain", 3)]:
        for index in range(count):
            shape_id = f"{class_label}_{index:03d}"
            rows.append(
                {
                    "class": class_label,
                    "filename": f"{shape_id}.stl",
                    "shape_id": shape_id,
                    "url": f"https://example.org/{shape_id}.stl",
                }
            )
    _write_csv(manifest, ["class", "filename", "shape_id", "url"], rows)

    mesh_root = tmp_path / "meshes"
    for row in rows:
        if row["shape_id"] == "skull_mri_003":
            continue
        class_dir = mesh_root / row["class"]
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / row["filename"]).write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")

    split = build_balanced_split_from_manifest(
        manifest_csv=str(manifest),
        mesh_root=str(mesh_root),
        synset="medshape",
        mesh_layout="class_subdirs",
        per_class_target=4,
        seed=7,
    )

    assert split["mesh_layout"] == "class_subdirs"
    assert set(split["classes"]) == {"brain", "skull_mri"}
    assert split["counts"]["by_class"]["brain"]["selected"] == 3
    assert split["counts"]["by_class"]["skull_mri"]["selected"] == 3
    assert split["replacement_needed_by_class"] == {"brain": 1, "skull_mri": 1}
    assert split["missing_by_class"]["skull_mri"][0]["shape_id"] == "skull_mri_003"
    assert split["missing_by_class"]["skull_mri"][0]["missing"] == "mesh"
    assert len(split["train"]) == 2
    assert len(split["val"]) == 2
    assert len(split["test"]) == 2
    assert set(split["class_map"].values()) == {"brain", "skull_mri"}
    assert "skull_mri_003" not in split["class_map"]


def test_aggregate_sparse_sweep_by_class_with_unknown_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        ["class", "filename", "shape_id", "url"],
        [
            {"class": "brain", "filename": "brain_001.stl", "shape_id": "brain_001", "url": ""},
            {"class": "skull", "filename": "skull_001.stl", "shape_id": "skull_001", "url": ""},
        ],
    )
    sweep = tmp_path / "sparse.csv"
    _write_csv(
        sweep,
        [
            "shape_id",
            "variant",
            "method",
            "observation_points",
            "step",
            "obs_l1",
            "eval_l1",
            "latent_norm",
            "conditional_energy",
        ],
        [
            {
                "shape_id": "brain_001",
                "variant": "no_prior",
                "method": "no_prior",
                "observation_points": 8,
                "step": 10,
                "obs_l1": 1.0,
                "eval_l1": 2.0,
                "latent_norm": 3.0,
                "conditional_energy": "",
            },
            {
                "shape_id": "brain_001",
                "variant": "no_prior",
                "method": "no_prior",
                "observation_points": 8,
                "step": 10,
                "obs_l1": 3.0,
                "eval_l1": 4.0,
                "latent_norm": 5.0,
                "conditional_energy": "",
            },
            {
                "shape_id": "missing_001",
                "variant": "l2_conditional",
                "method": "l2_conditional_energy",
                "observation_points": 8,
                "step": 10,
                "obs_l1": 2.0,
                "eval_l1": 1.0,
                "latent_norm": 4.0,
                "conditional_energy": 0.5,
            },
        ],
    )

    out_class = tmp_path / "class_aggregate.csv"
    out_overall = tmp_path / "overall_aggregate.csv"
    summary_json = tmp_path / "summary.json"
    summary = aggregate_sparse_sweep_by_class(
        sweep_csv=str(sweep),
        manifest_csv=str(manifest),
        out_class_csv=str(out_class),
        out_overall_csv=str(out_overall),
        summary_json=str(summary_json),
    )

    assert summary["observation_column"] == "observation_points"
    assert summary["missing_class_shape_ids"] == ["missing_001"]

    with out_class.open("r", encoding="utf-8") as f:
        class_rows = list(csv.DictReader(f))
    by_key = {(row["class"], row["variant"]): row for row in class_rows}
    assert by_key[("brain", "no_prior")]["n"] == "2"
    assert float(by_key[("brain", "no_prior")]["eval_l1_mean"]) == 3.0
    assert by_key[("UNKNOWN", "l2_conditional")]["n"] == "1"
    assert by_key[("UNKNOWN", "l2_conditional")]["observation_points"] == "8"

    with out_overall.open("r", encoding="utf-8") as f:
        overall_rows = list(csv.DictReader(f))
    assert {row["class"] for row in overall_rows} == {"ALL"}

    with summary_json.open("r", encoding="utf-8") as f:
        saved_summary = json.load(f)
    assert saved_summary["missing_class_shape_ids"] == ["missing_001"]
