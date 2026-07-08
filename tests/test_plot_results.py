import builtins
import csv

from lep import plot_results


def test_plot_missing_matplotlib_has_clear_message(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        plot_results._require_matplotlib()
    except RuntimeError as exc:
        assert ".[viz]" in str(exc)
    else:
        raise AssertionError("Expected missing matplotlib to raise RuntimeError")


def test_plot_tiny_csv_if_matplotlib_available(tmp_path):
    try:
        plot_results._require_matplotlib()
    except RuntimeError:
        return

    csv_path = tmp_path / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shape_id", "repeat", "variant", "method", "step", "eval_l1"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"shape_id": "a", "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 0, "eval_l1": 1.0},
                {"shape_id": "a", "repeat": 0, "variant": "no_prior", "method": "no_prior", "step": 10, "eval_l1": 0.9},
                {"shape_id": "a", "repeat": 0, "variant": "l2_conditional", "method": "l2_conditional_energy", "step": 0, "eval_l1": 1.0},
                {"shape_id": "a", "repeat": 0, "variant": "l2_conditional", "method": "l2_conditional_energy", "step": 10, "eval_l1": 0.7},
            ]
        )

    outputs = plot_results.plot_results(
        str(csv_path),
        out_dir=str(tmp_path / "plots"),
        metrics=["eval_l1"],
        baseline_variant="no_prior",
    )

    assert outputs
    assert all((tmp_path / "plots" / output.split("/")[-1]).exists() for output in outputs)
