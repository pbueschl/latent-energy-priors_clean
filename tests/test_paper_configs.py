from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_config(name):
    with (ROOT / "configs" / "paper" / name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_paper_configs_use_relative_paths_and_expected_variants():
    deepsdf_cfg = _load_config("medshape_diverse_deepsdf.yaml")
    conditional_cfg = _load_config("medshape_diverse_conditional_energy.yaml")
    conditional_val_cfg = _load_config("medshape_diverse_conditional_validation_grid.yaml")
    final_cfg = _load_config("medshape_diverse_final_eval.yaml")
    val_cfg = _load_config("medshape_diverse_l2_validation_grid.yaml")

    for cfg in (deepsdf_cfg, conditional_cfg, conditional_val_cfg, final_cfg, val_cfg):
        text_paths = [
            str(value)
            for section in ("data", "artifacts")
            for value in dict(cfg.get(section, {})).values()
        ]
        assert all(not path.startswith("/") for path in text_paths)

    assert deepsdf_cfg["experiment"] == "medshape_diverse_sdf_6x250"
    assert deepsdf_cfg["deepsdf"]["latent_dim"] == 128
    assert deepsdf_cfg["deepsdf"]["batch_shapes"] == 128
    assert deepsdf_cfg["deepsdf"]["samples_per_shape"] == 4096
    assert deepsdf_cfg["deepsdf"]["num_workers"] == 0
    assert deepsdf_cfg["deepsdf"]["epochs"] == 2000
    assert deepsdf_cfg["deepsdf"]["latent_regularizer"] == "l2"

    assert conditional_cfg["conditional_energy"]["observation_points"] == [16, 64, 128]
    assert conditional_cfg["conditional_energy"]["batch_size"] == 512
    assert conditional_cfg["conditional_energy"]["langevin_steps"] == 20
    assert conditional_cfg["conditional_energy"]["epochs"] == 5000

    final_variants = {
        variant["name"]: variant["method"]
        for variant in final_cfg["sparse_sweep"]["variants"]
    }
    assert final_variants == {
        "no_prior": "no_prior",
        "l2_lam_1e_3": "l2",
        "conditional_lam_1e_5": "conditional_energy",
        "l2_conditional_l2_1e_3_cond_3e_5": "l2_conditional_energy",
    }

    l2_values = [
        float(variant["lambda_l2"])
        for variant in val_cfg["sparse_sweep"]["variants"]
        if variant["method"] == "l2"
    ]
    assert l2_values == [1e-5, 1e-4, 1e-3]

    conditional_values = [
        float(variant["lambda_conditional_energy"])
        for variant in conditional_val_cfg["sparse_sweep"]["variants"]
        if variant["method"] == "conditional_energy"
    ]
    assert conditional_values == [1e-5, 3e-5, 1e-4]

    assert val_cfg["sparse_sweep"]["observation_points"] == 16
    assert final_cfg["sparse_sweep"]["observation_points"] == 16
    assert final_cfg["sparse_sweep"]["steps"] == [
        0,
        25,
        50,
        100,
        250,
        500,
        1000,
        1500,
        2000,
    ]
    final_by_name = {
        variant["name"]: variant for variant in final_cfg["sparse_sweep"]["variants"]
    }
    assert final_by_name["l2_lam_1e_3"]["lambda_l2"] == 1e-3
    assert final_by_name["conditional_lam_1e_5"]["lambda_conditional_energy"] == 1e-5
    assert (
        final_by_name["l2_conditional_l2_1e_3_cond_3e_5"]["lambda_conditional_energy"]
        == 3e-5
    )
    grid_metrics = final_cfg["sparse_sweep"]["grid_metrics"]
    assert grid_metrics == {
        "enabled": True,
        "mesh_root": "data/raw/medshape_diverse",
        "synset": "medshape",
        "mesh_layout": "class_subdirs",
        "grid_size": 64,
        "bounds": 1.15,
        "sdf_clamp": 0.1,
        "threshold": 0.0,
        "steps": [250, 500, 1000, 1500, 2000],
        "batch_points": 262144,
        "gt_batch_points": 262144,
        "allow_non_watertight_gt": True,
    }
