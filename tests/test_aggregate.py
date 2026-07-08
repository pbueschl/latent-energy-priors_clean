import csv

from lep.aggregate import aggregate_csvs


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_groups_by_variant_when_present(tmp_path):
    path = tmp_path / "sweep.csv"
    _write_csv(
        path,
        ["variant", "method", "step", "obs_l1", "eval_l1", "latent_norm", "conditional_energy"],
        [
            {"variant": "l2_a", "method": "l2", "step": 10, "obs_l1": 1.0, "eval_l1": 2.0, "latent_norm": 3.0, "conditional_energy": ""},
            {"variant": "l2_b", "method": "l2", "step": 10, "obs_l1": 3.0, "eval_l1": 4.0, "latent_norm": 5.0, "conditional_energy": ""},
            {"variant": "l2_a", "method": "l2", "step": 10, "obs_l1": 5.0, "eval_l1": 6.0, "latent_norm": 7.0, "conditional_energy": ""},
        ],
    )

    rows = aggregate_csvs([str(path)])
    by_variant = {row["variant"]: row for row in rows}

    assert set(by_variant) == {"l2_a", "l2_b"}
    assert by_variant["l2_a"]["method"] == "l2"
    assert by_variant["l2_a"]["n"] == 2
    assert by_variant["l2_a"]["obs_l1_mean"] == 3.0


def test_aggregate_old_csv_sets_variant_to_method(tmp_path):
    path = tmp_path / "old_sweep.csv"
    _write_csv(
        path,
        ["method", "step", "obs_l1", "eval_l1", "latent_norm", "conditional_energy"],
        [
            {"method": "l2", "step": 0, "obs_l1": 1.0, "eval_l1": 2.0, "latent_norm": 3.0, "conditional_energy": ""},
            {"method": "l2", "step": 0, "obs_l1": 3.0, "eval_l1": 4.0, "latent_norm": 5.0, "conditional_energy": ""},
        ],
    )

    rows = aggregate_csvs([str(path)])

    assert rows[0]["variant"] == "l2"
    assert rows[0]["method"] == "l2"
    assert rows[0]["n"] == 2
    assert rows[0]["eval_l1_mean"] == 3.0
