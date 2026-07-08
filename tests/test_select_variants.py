import csv

from lep.select_variants import select_variants, write_selection_outputs


def _write_sparse_csv(path, rows):
    fieldnames = [
        "shape_id",
        "repeat",
        "variant",
        "method",
        "step",
        "obs_l1",
        "eval_l1",
        "latent_norm",
        "conditional_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_variants_picks_best_global_and_family(tmp_path):
    csv_path = tmp_path / "val_sweep.csv"
    rows = []
    for shape_id, offset in [("a", 0.0), ("b", 0.05)]:
        rows.extend(
            [
                {"shape_id": shape_id, "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 10, "obs_l1": 1.0, "eval_l1": 1.0 + offset, "latent_norm": 1.0, "conditional_energy": ""},
                {"shape_id": shape_id, "repeat": 0, "variant": "l2_lam_1e_4", "method": "l2", "step": 10, "obs_l1": 0.9, "eval_l1": 0.8 + offset, "latent_norm": 1.0, "conditional_energy": ""},
                {"shape_id": shape_id, "repeat": 0, "variant": "conditional_lam_1e_5", "method": "conditional_energy", "step": 10, "obs_l1": 0.9, "eval_l1": 0.85 + offset, "latent_norm": 1.0, "conditional_energy": 0.1},
                {"shape_id": shape_id, "repeat": 0, "variant": "l2_conditional_good", "method": "l2_conditional_energy", "step": 10, "obs_l1": 0.8, "eval_l1": 0.7 + offset, "latent_norm": 1.0, "conditional_energy": 0.1},
                {"shape_id": shape_id, "repeat": 0, "variant": "l2_conditional_bad", "method": "l2_conditional_energy", "step": 10, "obs_l1": 0.9, "eval_l1": 0.9 + offset, "latent_norm": 1.0, "conditional_energy": 0.1},
            ]
        )
    _write_sparse_csv(csv_path, rows)

    selection = select_variants(str(csv_path), metric="eval_l1")
    write_selection_outputs(
        selection,
        out_json=str(tmp_path / "selection.json"),
        out_csv=str(tmp_path / "ranking.csv"),
    )

    assert selection["step"] == 10
    assert selection["ranking"][0]["variant"] == "l2_conditional_good"
    assert selection["selected"]["l2"]["variant"] == "l2_lam_1e_4"
    assert selection["selected"]["conditional_energy"]["variant"] == "conditional_lam_1e_5"
    assert selection["selected"]["l2_conditional_energy"]["variant"] == "l2_conditional_good"
    assert (tmp_path / "selection.json").exists()
    assert (tmp_path / "ranking.csv").exists()


def test_select_variants_groups_conditional_energy_families(tmp_path):
    csv_path = tmp_path / "val_conditional_sweep.csv"
    rows = [
        {"shape_id": "a", "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 10, "obs_l1": 1.0, "eval_l1": 1.0, "latent_norm": 1.0, "conditional_energy": ""},
        {"shape_id": "a", "repeat": 0, "variant": "conditional_lam_1e_4", "method": "conditional_energy", "step": 10, "obs_l1": 0.8, "eval_l1": 0.7, "latent_norm": 1.0, "conditional_energy": ""},
        {"shape_id": "a", "repeat": 0, "variant": "l2_conditional_l2_1e_4_cond_1e_4", "method": "l2_conditional_energy", "step": 10, "obs_l1": 0.75, "eval_l1": 0.65, "latent_norm": 1.0, "conditional_energy": ""},
    ]
    _write_sparse_csv(csv_path, rows)

    selection = select_variants(str(csv_path), metric="eval_l1")

    assert selection["selected"]["conditional_energy"]["variant"] == "conditional_lam_1e_4"
    assert (
        selection["selected"]["l2_conditional_energy"]["variant"]
        == "l2_conditional_l2_1e_4_cond_1e_4"
    )
