import csv
import json

import numpy as np
import pytest
import torch

from lep import sdf_grid_qc
from lep.deepsdf import LatentDecoder
from lep.sdf_grid_qc import run_sdf_grid_qc, sdf_grid_metrics
from lep.shapenet import ShapeRecord


def test_sdf_grid_metrics_sdf_and_binary_masks():
    pred_sdf = np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    gt_sdf = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    pred_mask = pred_sdf < 0
    gt_mask = gt_sdf < 0
    pred_sdf_clipped = np.clip(pred_sdf, -0.5, 0.5)
    gt_sdf_clipped = np.clip(gt_sdf, -0.5, 0.5)

    metrics = sdf_grid_metrics(
        pred_sdf,
        gt_sdf,
        pred_mask,
        gt_mask,
        pred_sdf_clipped=pred_sdf_clipped,
        gt_sdf_clipped=gt_sdf_clipped,
    )

    assert metrics["mean_abs_sdf_error"] == pytest.approx(1.0)
    assert metrics["rmse_sdf_error"] == pytest.approx(np.sqrt(2.0))
    assert metrics["mean_abs_clipped_sdf_error"] == pytest.approx(0.5)
    assert metrics["rmse_clipped_sdf_error"] == pytest.approx(np.sqrt(0.5))
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["pred_positive_fraction"] == pytest.approx(0.5)
    assert metrics["gt_positive_fraction"] == pytest.approx(0.5)
    assert metrics["mean_pred_sdf_inside"] == pytest.approx(0.0)
    assert metrics["mean_pred_sdf_outside"] == pytest.approx(0.0)
    assert metrics["mean_gt_sdf_inside"] == pytest.approx(-1.0)
    assert metrics["mean_gt_sdf_outside"] == pytest.approx(1.0)


def test_sdf_grid_qc_writes_npz_csv_json_without_tiff(tmp_path, monkeypatch):
    torch.manual_seed(0)
    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["shape_a"],
            "config": {},
        },
        checkpoint,
    )

    class FakeMesh:
        is_watertight = True

    def fake_discover_mesh_records(mesh_root, synset):
        assert mesh_root == "unused_mesh_root"
        return {"shape_a": ShapeRecord(synset=synset, shape_id="shape_a", mesh_path="shape_a.obj")}

    def fake_load_normalized_mesh(mesh_path):
        assert mesh_path == "shape_a.obj"
        return FakeMesh(), np.zeros(3, dtype=np.float32), np.float32(1.0)

    gt_chunk_sizes = []

    def fake_signed_distance(mesh, points):
        gt_chunk_sizes.append(points.shape[0])
        return np.linalg.norm(points, axis=1).astype(np.float32) - np.float32(0.75)

    monkeypatch.setattr(sdf_grid_qc, "_discover_mesh_records", fake_discover_mesh_records)
    monkeypatch.setattr(sdf_grid_qc, "_load_normalized_mesh", fake_load_normalized_mesh)
    monkeypatch.setattr(sdf_grid_qc, "_signed_distance", fake_signed_distance)

    out_dir = tmp_path / "qc"
    summary = run_sdf_grid_qc(
        checkpoint=str(checkpoint),
        mesh_root="unused_mesh_root",
        out_dir=str(out_dir),
        shape_ids=["shape_a"],
        grid_size=5,
        bounds=1.0,
        sdf_clamp=0.25,
        threshold=0.0,
        batch_points=16,
        gt_batch_points=17,
        device="cpu",
        write_tiff=False,
    )

    assert summary["count"] == 1
    assert summary["config"]["shape_ids"] == ["shape_a"]
    assert summary["config"]["sdf_clamp"] == pytest.approx(0.25)
    assert summary["config"]["gt_batch_points"] == 17
    assert summary["aggregate"]["mean_mean_abs_sdf_error"] == pytest.approx(
        summary["rows"][0]["mean_abs_sdf_error"]
    )
    assert max(gt_chunk_sizes) <= 17
    assert sum(gt_chunk_sizes) == 5**3

    row = summary["rows"][0]
    assert row["shape_id"] == "shape_a"
    assert row["pred_sdf_tiff"] == ""
    assert row["gt_sdf_tiff"] == ""
    assert row["pred_sdf_clipped_tiff"] == ""
    assert row["gt_sdf_clipped_tiff"] == ""
    assert row["pred_mask_tiff"] == ""
    assert row["gt_mask_tiff"] == ""
    assert not list(out_dir.glob("*.tif"))

    with np.load(row["npz"]) as data:
        assert data["pred_sdf"].shape == (5, 5, 5)
        assert data["gt_sdf"].shape == (5, 5, 5)
        assert data["pred_sdf_clipped"].shape == (5, 5, 5)
        assert data["gt_sdf_clipped"].shape == (5, 5, 5)
        assert data["pred_mask"].shape == (5, 5, 5)
        assert data["gt_mask"].shape == (5, 5, 5)
        assert data["grid_axis"].shape == (5,)
        assert data["shape_id"].item() == "shape_a"
        assert data["mesh_is_watertight"].item()
        assert data["sdf_clamp"].item() == pytest.approx(0.25)
        assert data["sdf_clamp_enabled"].item()
        assert data["threshold"].item() == pytest.approx(0.0)
        assert data["gt_batch_points"].item() == 17

    with (out_dir / "sdf_grid_qc.csv").open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 1
    assert csv_rows[0]["shape_id"] == "shape_a"
    assert "mean_abs_sdf_error" in csv_rows[0]
    assert "dice" in csv_rows[0]

    with (out_dir / "sdf_grid_qc_summary.json").open("r", encoding="utf-8") as f:
        saved_summary = json.load(f)
    assert saved_summary["count"] == 1
    assert saved_summary["rows"][0]["shape_id"] == "shape_a"


def test_sdf_grid_qc_rejects_non_watertight_gt(tmp_path, monkeypatch):
    torch.manual_seed(0)
    decoder = LatentDecoder(latent_dim=2, hidden_dim=4, num_layers=2)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 2),
            "train_shape_ids": ["shape_a"],
        },
        checkpoint,
    )

    class NonWatertightMesh:
        is_watertight = False

    monkeypatch.setattr(
        sdf_grid_qc,
        "_discover_mesh_records",
        lambda mesh_root, synset: {
            "shape_a": ShapeRecord(synset=synset, shape_id="shape_a", mesh_path="shape_a.obj")
        },
    )
    monkeypatch.setattr(
        sdf_grid_qc,
        "_load_normalized_mesh",
        lambda mesh_path: (NonWatertightMesh(), np.zeros(3, dtype=np.float32), np.float32(1.0)),
    )

    with pytest.raises(ValueError, match="not watertight"):
        run_sdf_grid_qc(
            checkpoint=str(checkpoint),
            mesh_root="unused_mesh_root",
            out_dir=str(tmp_path / "qc"),
            indices=[0],
            grid_size=3,
            batch_points=8,
            device="cpu",
            write_tiff=False,
        )
