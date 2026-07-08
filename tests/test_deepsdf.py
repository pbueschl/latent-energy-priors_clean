import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lep.deepsdf import DeepSDFAutodecoder, LatentDecoder, clamp_sdf
from lep.train_deepsdf import _latent_regularization_terms, _torch_load, train_deepsdf


def test_decoder_forward_shapes():
    decoder = LatentDecoder(latent_dim=8, hidden_dim=16, num_layers=4, skip_layers=[2])
    z = torch.randn(2, 8)
    xyz = torch.randn(2, 5, 3)
    out = decoder(z, xyz)
    assert out.shape == (2, 5)

    single = decoder(z[:1], xyz[0])
    assert single.shape == (5,)


def test_tiny_autodecoder_step():
    torch.manual_seed(0)
    model = DeepSDFAutodecoder(
        num_shapes=3,
        latent_dim=4,
        decoder_kwargs={"hidden_dim": 16, "num_layers": 3},
    )
    shape_indices = torch.tensor([0, 1])
    xyz = torch.randn(2, 6, 3)
    target = torch.tanh(xyz.sum(dim=-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pred = model(shape_indices, xyz)
    loss = F.l1_loss(clamp_sdf(pred, 0.1), clamp_sdf(target, 0.1)) + 1e-4 * model.latent_l2()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_norm_hinge_regularizer_penalty_behavior():
    z = torch.tensor([[3.0, 4.0], [0.6, 0.8], [0.0, 0.0]])
    weighted_reg, latent_reg, latent_l2, latent_norm = _latent_regularization_terms(
        z,
        latent_regularizer="norm_hinge",
        latent_l2_weight=0.1,
        latent_hinge_radius=2.0,
        latent_hinge_weight=0.5,
    )

    assert latent_reg.item() == pytest.approx(3.0)
    assert weighted_reg.item() == pytest.approx(1.5)
    assert latent_l2.item() == pytest.approx((25.0 + 1.0 + 0.0) / 3.0)
    assert latent_norm.item() == pytest.approx((5.0 + 1.0 + 0.0) / 3.0)


def _write_tiny_sdf_dataset(tmp_path):
    rng = np.random.default_rng(0)
    sdf_root = tmp_path / "sdf"
    sdf_root.mkdir()
    shape_ids = ["shape_a", "shape_b", "shape_c"]
    for index, shape_id in enumerate(shape_ids):
        points = rng.normal(size=(24, 3)).astype(np.float32)
        sdf = (points.sum(axis=1) * 0.05 + index * 0.01).astype(np.float32)
        np.savez_compressed(sdf_root / f"{shape_id}.npz", points=points, sdf=sdf)
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"synset": "synthetic", "train": shape_ids, "val": [], "test": []}),
        encoding="utf-8",
    )
    return sdf_root, split_file, shape_ids


def _tiny_train_config(num_workers):
    return {
        "seed": 0,
        "deepsdf": {
            "latent_dim": 4,
            "hidden_dim": 8,
            "num_layers": 3,
            "latent_init_std": 0.01,
            "sdf_clamp": 0.1,
            "batch_shapes": 2,
            "samples_per_shape": 6,
            "epochs": 1,
            "lr_decoder": 0.001,
            "lr_latents": 0.001,
            "latent_l2_weight": 0.0001,
            "save_every": 0,
            "num_workers": num_workers,
        },
    }


def test_train_deepsdf_manual_and_dataloader_paths(tmp_path):
    sdf_root, split_file, shape_ids = _write_tiny_sdf_dataset(tmp_path)
    for num_workers in (0, 1):
        out_dir = tmp_path / f"out_workers_{num_workers}"
        checkpoint_path = train_deepsdf(
            config=_tiny_train_config(num_workers),
            sdf_root=str(sdf_root),
            split_file=str(split_file),
            out_dir=str(out_dir),
            device="cpu",
        )
        checkpoint = _torch_load(checkpoint_path, map_location="cpu")
        latents = _torch_load(out_dir / "train_latents.pt", map_location="cpu")

        assert checkpoint_path.exists()
        assert checkpoint["train_latents"].shape == (len(shape_ids), 4)
        assert latents["train_latents"].shape == (len(shape_ids), 4)
        assert checkpoint["config"]["deepsdf"]["num_workers"] == num_workers


def test_train_deepsdf_latent_perturbations_export_finite_latents(tmp_path):
    sdf_root, split_file, shape_ids = _write_tiny_sdf_dataset(tmp_path)
    config = _tiny_train_config(num_workers=0)
    config["deepsdf"].update(
        {
            "latent_regularizer": "norm_hinge",
            "latent_hinge_radius": 0.02,
            "latent_hinge_weight": 0.001,
            "latent_noise_std": 0.05,
            "latent_dropout_p": 0.5,
        }
    )
    out_dir = tmp_path / "out_perturbed"

    checkpoint_path = train_deepsdf(
        config=config,
        sdf_root=str(sdf_root),
        split_file=str(split_file),
        out_dir=str(out_dir),
        device="cpu",
    )
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    latents = _torch_load(out_dir / "train_latents.pt", map_location="cpu")

    assert checkpoint["train_latents"].shape == (len(shape_ids), 4)
    assert latents["train_latents"].shape == (len(shape_ids), 4)
    assert torch.isfinite(checkpoint["train_latents"]).all().item()
    assert torch.isfinite(latents["train_latents"]).all().item()
    assert checkpoint["config"]["deepsdf"]["latent_regularizer"] == "norm_hinge"
    assert checkpoint["config"]["deepsdf"]["latent_noise_std"] == pytest.approx(0.05)
    assert checkpoint["config"]["deepsdf"]["latent_dropout_p"] == pytest.approx(0.5)

    with (out_dir / "train_history.csv").open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "latent_reg" in header
    assert "latent_norm" in header
