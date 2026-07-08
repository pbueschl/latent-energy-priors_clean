import json

import numpy as np
import torch

from lep.conditional_energy import (
    ConditionalEnergyMLP,
    ConditionalLatentEnergyPrior,
    SparseSDFObservationEncoder,
    load_conditional_energy_prior,
    save_conditional_energy_prior,
    train_conditional_energy,
)


def test_sparse_observation_encoder_accepts_variable_point_counts():
    torch.manual_seed(0)
    encoder = SparseSDFObservationEncoder(point_hidden_size=8, context_dim=6, pooling_mode="mean_max_logn")
    points_8 = torch.randn(2, 8, 3)
    sdf_8 = torch.randn(2, 8)
    points_16 = torch.randn(2, 16, 3)
    sdf_16 = torch.randn(2, 16)

    assert encoder(points_8, sdf_8).shape == (2, 6)
    assert encoder(points_16, sdf_16).shape == (2, 6)


def test_conditional_energy_save_load_alias_config(tmp_path):
    torch.manual_seed(1)
    encoder = SparseSDFObservationEncoder(point_hidden_size=8, context_dim=6)
    energy = ConditionalEnergyMLP(latent_dim=3, context_dim=6, hidden_dim=8, num_layers=2)
    prior = ConditionalLatentEnergyPrior(
        observation_encoder=encoder,
        energy_model=energy,
        latent_mean=torch.zeros(1, 3),
        latent_std=torch.ones(1, 3),
    )
    path = tmp_path / "conditional.pt"
    save_conditional_energy_prior(str(path), prior, config={"conditional_energy": {}}, history=[])
    loaded, checkpoint = load_conditional_energy_prior(str(path))

    z = torch.randn(2, 3)
    points = torch.randn(2, 5, 3)
    sdf = torch.randn(2, 5)
    with torch.no_grad():
        assert torch.allclose(prior.energy(z, points, sdf), loaded.energy(z, points, sdf))
    assert checkpoint["encoder_config"]["observation_pooling"] == "mean_max_logn"


def test_train_conditional_energy_from_synthetic_deepsdf_checkpoint(tmp_path):
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    shape_ids = [f"shape_{idx}" for idx in range(4)]
    for shape_id in shape_ids:
        points = rng.normal(size=(32, 3)).astype(np.float32)
        sdf = (np.linalg.norm(points, axis=1) - 0.5).astype(np.float32)
        np.savez_compressed(sdf_root / f"{shape_id}.npz", points=points, sdf=sdf)

    checkpoint = tmp_path / "deepsdf.pt"
    torch.save(
        {
            "decoder_config": {"latent_dim": 3, "hidden_dim": 8, "num_layers": 3},
            "decoder_state_dict": {},
            "train_latents": torch.randn(4, 3),
            "train_shape_ids": shape_ids,
            "config": {"deepsdf": {"latent_dim": 3}},
        },
        checkpoint,
    )

    out_dir = tmp_path / "cond_energy"
    checkpoint_path = train_conditional_energy(
        config={
            "seed": 2,
            "conditional_energy": {
                "point_hidden_size": 8,
                "context_dim": 6,
                "hidden_dim": 8,
                "num_layers": 2,
                "epochs": 2,
                "batch_size": 4,
                "observation_points": [4, 8],
                "langevin_steps": 1,
                "negative_noise_scales": [1.0],
                "min_negative_train_distance": 0.0,
            },
        },
        checkpoint=str(checkpoint),
        sdf_root=str(sdf_root),
        out_dir=str(out_dir),
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert (out_dir / "conditional_energy_prior_best_loss.pt").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["latent_dim"] == 3
    assert summary["num_train_latents"] == 4
