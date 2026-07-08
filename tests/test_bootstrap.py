import csv

from lep.bootstrap import paired_bootstrap, write_bootstrap_outputs


def test_paired_bootstrap_positive_for_lower_is_better_improvement(tmp_path):
    csv_path = tmp_path / "test_sweep.csv"
    rows = [
        {"shape_id": "a", "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 50, "eval_l1": 1.0},
        {"shape_id": "a", "repeat": 0, "variant": "l2_conditional", "method": "l2_conditional_energy", "step": 50, "eval_l1": 0.8},
        {"shape_id": "b", "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 50, "eval_l1": 1.2},
        {"shape_id": "b", "repeat": 0, "variant": "l2_conditional", "method": "l2_conditional_energy", "step": 50, "eval_l1": 1.0},
        {"shape_id": "c", "repeat": 0, "variant": "l2_conditional", "method": "l2_conditional_energy", "step": 50, "eval_l1": 0.5},
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shape_id", "repeat", "variant", "method", "step", "eval_l1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    result = paired_bootstrap(
        str(csv_path),
        baseline_variant="no_prior",
        variants=["l2_conditional"],
        metric="eval_l1",
        num_bootstrap=100,
        seed=7,
    )
    write_bootstrap_outputs(
        result,
        out_json=str(tmp_path / "bootstrap.json"),
        out_csv=str(tmp_path / "bootstrap.csv"),
    )
    row = result["results"][0]

    assert row["num_pairs"] == 2
    assert abs(row["mean_effect"] - 0.2) < 1e-12
    assert row["ci_low"] > 0
    assert (tmp_path / "bootstrap.json").exists()
    assert (tmp_path / "bootstrap.csv").exists()
