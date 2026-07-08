import csv
import json

import numpy as np
import torch

from lep.conditional_energy import (
    ConditionalEnergyMLP,
    ConditionalLatentEnergyPrior,
    SparseSDFObservationEncoder,
    save_conditional_energy_prior,
)
from lep.deepsdf import LatentDecoder
from lep.latent_gmm_prior import GaussianMixtureLatentPrior, save_gmm_prior
from lep.sparse_sweep import (
    infer_one,
    resolve_sweep_variants,
    run_sparse_sweep,
    sample_disjoint_observation_eval,
)


def test_disjoint_observation_eval_sampling_no_overlap():
    rng = np.random.default_rng(0)
    points = np.arange(90, dtype=np.float32).reshape(30, 3)
    sdf = np.arange(30, dtype=np.float32)
    _, _, _, _, obs_indices, eval_indices = sample_disjoint_observation_eval(
        points,
        sdf,
        obs_count=5,
        eval_count=20,
        rng=rng,
    )

    assert set(obs_indices.tolist()).isdisjoint(set(eval_indices.tolist()))


def test_sparse_sweep_synthetic(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    points = rng.normal(size=(64, 3)).astype(np.float32)
    sdf = (np.linalg.norm(points, axis=1) - 0.75).astype(np.float32)
    np.savez_compressed(sdf_root / "shape_a.npz", points=points, sdf=sdf)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": [], "val": [], "test": ["shape_a"]}),
        encoding="utf-8",
    )

    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["train_dummy"],
            "config": {},
        },
        checkpoint,
    )

    out = tmp_path / "sweep.csv"
    rows = run_sparse_sweep(
        config={
            "seed": 0,
            "deepsdf": {"sdf_clamp": 0.1},
            "sparse_sweep": {
                "methods": ["no_prior", "l2"],
                "steps": [0, 2],
                "observation_points": 8,
                "eval_points": 16,
                "repeats": 1,
            },
        },
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        checkpoint=str(checkpoint),
        out=str(out),
        device="cpu",
    )

    assert out.exists()
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"no_prior", "l2"}
    assert {row["variant"] for row in rows} == {"no_prior", "l2"}
    assert all(row["variant"] == row["method"] for row in rows)
    assert {row["observation_points"] for row in rows} == {8}
    with out.open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["observation_points"] == "8"


def test_sparse_sweep_variant_output(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    points = rng.normal(size=(64, 3)).astype(np.float32)
    sdf = (np.linalg.norm(points, axis=1) - 0.75).astype(np.float32)
    np.savez_compressed(sdf_root / "shape_a.npz", points=points, sdf=sdf)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": [], "val": [], "test": ["shape_a"]}),
        encoding="utf-8",
    )

    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["train_dummy"],
            "config": {},
        },
        checkpoint,
    )

    out = tmp_path / "sweep.csv"
    rows = run_sparse_sweep(
        config={
            "seed": 0,
            "deepsdf": {"sdf_clamp": 0.1},
            "sparse_sweep": {
                "variants": [
                    {"name": "no_prior", "method": "no_prior", "lambda_l2": 0.0},
                    {"name": "l2_lam_1e_4", "method": "l2", "lambda_l2": 1e-4},
                ],
                "steps": [0, 2],
                "observation_points": 8,
                "eval_points": 16,
                "repeats": 1,
            },
        },
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        checkpoint=str(checkpoint),
        out=str(out),
        device="cpu",
    )

    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {"no_prior", "l2_lam_1e_4"}
    assert {row["method"] for row in rows} == {"no_prior", "l2"}
    assert {row["lambda_l2"] for row in rows if row["variant"] == "l2_lam_1e_4"} == {1e-4}
    assert {row["observation_points"] for row in rows} == {8}


def test_sparse_sweep_conditional_energy_variant(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    points = rng.normal(size=(64, 3)).astype(np.float32)
    sdf = (np.linalg.norm(points, axis=1) - 0.75).astype(np.float32)
    np.savez_compressed(sdf_root / "shape_a.npz", points=points, sdf=sdf)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": [], "val": [], "test": ["shape_a"]}),
        encoding="utf-8",
    )

    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["train_dummy"],
            "config": {},
        },
        checkpoint,
    )

    encoder = SparseSDFObservationEncoder(point_hidden_size=8, context_dim=6)
    energy = ConditionalEnergyMLP(latent_dim=4, context_dim=6, hidden_dim=8, num_layers=2)
    prior = ConditionalLatentEnergyPrior(
        observation_encoder=encoder,
        energy_model=energy,
        latent_mean=torch.zeros(1, 4),
        latent_std=torch.ones(1, 4),
    )
    conditional_checkpoint = tmp_path / "conditional_energy.pt"
    save_conditional_energy_prior(str(conditional_checkpoint), prior, config={}, history=[])

    out = tmp_path / "sweep.csv"
    rows = run_sparse_sweep(
        config={
            "seed": 0,
            "deepsdf": {"sdf_clamp": 0.1},
            "sparse_sweep": {
                "variants": [
                    {
                        "name": "l2_conditional_lam_1e_4_1e_3",
                        "method": "l2_conditional_energy",
                        "lambda_l2": 1e-4,
                        "lambda_conditional_energy": 1e-3,
                    },
                ],
                "steps": [0, 2],
                "observation_points": 8,
                "eval_points": 16,
                "repeats": 1,
            },
        },
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        checkpoint=str(checkpoint),
        out=str(out),
        conditional_energy_checkpoint=str(conditional_checkpoint),
        device="cpu",
    )

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"l2_conditional_energy"}
    assert all(row["conditional_energy"] != "" for row in rows)
    with out.open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["lambda_conditional_energy"] == "0.001"
    assert csv_rows[0]["conditional_energy"] != ""


def test_sparse_sweep_gmm_variants(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    points = rng.normal(size=(64, 3)).astype(np.float32)
    sdf = (np.linalg.norm(points, axis=1) - 0.75).astype(np.float32)
    np.savez_compressed(sdf_root / "shape_a.npz", points=points, sdf=sdf)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": [], "val": [], "test": ["shape_a"]}),
        encoding="utf-8",
    )

    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["train_dummy"],
            "config": {},
        },
        checkpoint,
    )
    gmm_prior = GaussianMixtureLatentPrior(
        weights=torch.tensor([0.5, 0.5]),
        means=torch.tensor([[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        covariances=torch.ones(2, 4),
        covariance_type="diag",
    )
    gmm_checkpoint = tmp_path / "gmm_prior.pt"
    save_gmm_prior(str(gmm_checkpoint), gmm_prior, metadata={"components": 2})

    out = tmp_path / "sweep.csv"
    rows = run_sparse_sweep(
        config={
            "seed": 0,
            "deepsdf": {"sdf_clamp": 0.1},
            "sparse_sweep": {
                "variants": [
                    {"name": "gmm_lam_1e_2", "method": "gmm", "lambda_gmm_prior": 1e-2},
                    {
                        "name": "l2_gmm_lam_1e_4_1e_2",
                        "method": "l2_gmm",
                        "lambda_l2": 1e-4,
                        "lambda_gmm_prior": 1e-2,
                    },
                ],
                "steps": [0, 2],
                "observation_points": 8,
                "eval_points": 16,
                "repeats": 1,
            },
        },
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        checkpoint=str(checkpoint),
        out=str(out),
        gmm_prior_checkpoint=str(gmm_checkpoint),
        device="cpu",
    )

    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"gmm", "l2_gmm"}
    assert all(row["gmm_log_prob"] != "" for row in rows)
    assert all(row["weighted_gmm_loss"] != "" for row in rows)
    with out.open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["lambda_gmm_prior"] == "0.01"
    assert csv_rows[0]["gmm_log_prob"] != ""
    assert csv_rows[0]["weighted_gmm_loss"] != ""


def test_sparse_sweep_shuffled_context_variant_records_context_shape(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    for shape_id, offset in (("shape_a", 0.0), ("shape_b", 0.5)):
        points = (rng.normal(size=(64, 3)) + offset).astype(np.float32)
        sdf = (np.linalg.norm(points, axis=1) - 0.75).astype(np.float32)
        np.savez_compressed(sdf_root / f"{shape_id}.npz", points=points, sdf=sdf)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": [], "val": [], "test": ["shape_a", "shape_b"]}),
        encoding="utf-8",
    )

    decoder = LatentDecoder(latent_dim=4, hidden_dim=8, num_layers=3)
    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "epoch": 0,
            "decoder_state_dict": decoder.state_dict(),
            "decoder_config": decoder.config(),
            "train_latents": torch.zeros(1, 4),
            "train_shape_ids": ["train_dummy"],
            "config": {},
        },
        checkpoint,
    )

    encoder = SparseSDFObservationEncoder(point_hidden_size=8, context_dim=6)
    energy = ConditionalEnergyMLP(latent_dim=4, context_dim=6, hidden_dim=8, num_layers=2)
    prior = ConditionalLatentEnergyPrior(
        observation_encoder=encoder,
        energy_model=energy,
        latent_mean=torch.zeros(1, 4),
        latent_std=torch.ones(1, 4),
    )
    conditional_checkpoint = tmp_path / "conditional_energy.pt"
    save_conditional_energy_prior(str(conditional_checkpoint), prior, config={}, history=[])

    out = tmp_path / "sweep.csv"
    rows = run_sparse_sweep(
        config={
            "seed": 0,
            "deepsdf": {"sdf_clamp": 0.1},
            "sparse_sweep": {
                "variants": [
                    {
                        "name": "conditional_shuffled",
                        "method": "conditional_energy_shuffled_context",
                        "lambda_l2": 0.0,
                        "lambda_conditional_energy": 1e-3,
                    },
                    {
                        "name": "l2_conditional_shuffled",
                        "method": "l2_conditional_energy_shuffled_context",
                        "lambda_l2": 1e-4,
                        "lambda_conditional_energy": 1e-3,
                    },
                ],
                "steps": [0],
                "observation_points": 8,
                "eval_points": 16,
                "repeats": 1,
            },
        },
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        checkpoint=str(checkpoint),
        out=str(out),
        conditional_energy_checkpoint=str(conditional_checkpoint),
        device="cpu",
    )

    assert len(rows) == 4
    assert {row["method"] for row in rows} == {
        "conditional_energy_shuffled_context",
        "l2_conditional_energy_shuffled_context",
    }
    assert all(row["context_is_shuffled"] for row in rows)
    assert all(row["context_shape_id"] != row["shape_id"] for row in rows)
    assert all(row["conditional_energy"] != "" for row in rows)
    with out.open("r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["context_is_shuffled"] == "True"
    assert csv_rows[0]["context_shape_id"] != csv_rows[0]["shape_id"]


def test_resolve_sweep_variants_and_methods_override():
    variants = resolve_sweep_variants(
        {
            "lambda_l2": 1e-4,
            "variants": [
                {
                    "name": "l2_conditional_l2_1e_4_cond_1e_3",
                    "lambda_l2": 1e-4,
                    "lambda_conditional_energy": 1e-3,
                },
                {
                    "name": "conditional_shuffled_context_lam_1e_3",
                    "lambda_l2": 0.0,
                    "lambda_conditional_energy": 1e-3,
                },
                {
                    "name": "l2_gmm_lam_1e_4_1e_2",
                    "lambda_l2": 1e-4,
                    "lambda_gmm_prior": 1e-2,
                },
            ],
        }
    )
    override = resolve_sweep_variants(
        {
            "lambda_l2": 1e-4,
            "variants": [{"name": "ignored", "method": "conditional_energy"}],
        },
        methods_override=["no_prior", "conditional_energy_shuffled_context"],
    )

    assert [(variant.name, variant.method) for variant in variants] == [
        ("l2_conditional_l2_1e_4_cond_1e_3", "l2_conditional_energy"),
        ("conditional_shuffled_context_lam_1e_3", "conditional_energy_shuffled_context"),
        ("l2_gmm_lam_1e_4_1e_2", "l2_gmm"),
    ]
    assert variants[0].lambda_l2 == 1e-4
    assert variants[0].lambda_conditional_energy == 1e-3
    assert variants[2].lambda_gmm_prior == 1e-2
    assert [(variant.name, variant.method) for variant in override] == [
        ("no_prior", "no_prior"),
        ("conditional_energy_shuffled_context", "conditional_energy_shuffled_context"),
    ]
