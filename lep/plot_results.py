"""Plot sparse-sweep curves and final-step improvements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .select_variants import read_metric_means


METRICS = (
    "obs_l1",
    "eval_l1",
    "grid_mean_abs_sdf_error",
    "grid_rmse_sdf_error",
    "grid_mean_abs_clipped_sdf_error",
    "grid_rmse_clipped_sdf_error",
    "eval_dice",
    "eval_iou",
    "eval_accuracy",
    "latent_norm",
    "conditional_energy",
)
LOWER_IS_BETTER = (
    "obs_l1",
    "eval_l1",
    "grid_mean_abs_sdf_error",
    "grid_rmse_sdf_error",
    "grid_mean_abs_clipped_sdf_error",
    "grid_rmse_clipped_sdf_error",
)


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Plotting requires matplotlib. Install it with `pip install -e ".[viz]"`.'
        ) from exc
    return plt


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _metric_rows_by_metric(
    path: str,
    metrics: Sequence[str],
    max_step: Optional[int],
    variants: Optional[Iterable[str]],
) -> Dict[str, List[Dict[str, object]]]:
    return {
        metric: read_metric_means(path, metric=metric, max_step=max_step, variants=variants)
        for metric in metrics
    }


def _available_metrics(rows_by_metric: Dict[str, List[Dict[str, object]]]) -> List[str]:
    return [metric for metric, rows in rows_by_metric.items() if rows]


def _plot_step_curves(
    plt,
    rows_by_metric: Dict[str, List[Dict[str, object]]],
    metrics: Sequence[str],
    title: Optional[str],
    out_path: Path,
) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3, 2.8 * n)), squeeze=False)
    for axis, metric in zip(axes[:, 0], metrics):
        by_variant: Dict[str, List[Dict[str, object]]] = {}
        for row in rows_by_metric[metric]:
            by_variant.setdefault(str(row["variant"]), []).append(row)
        for variant, rows in sorted(by_variant.items()):
            rows = sorted(rows, key=lambda row: int(row["step"]))
            axis.plot(
                [int(row["step"]) for row in rows],
                [float(row["value"]) for row in rows],
                marker="o",
                linewidth=1.5,
                label=variant,
            )
        axis.set_xlabel("optimization step")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_improvements(
    plt,
    rows_by_metric: Dict[str, List[Dict[str, object]]],
    metrics: Sequence[str],
    baseline_variant: str,
    title: Optional[str],
    out_path: Path,
) -> bool:
    improvement_metrics = [metric for metric in metrics if metric in LOWER_IS_BETTER and rows_by_metric[metric]]
    if not improvement_metrics:
        return False
    fig, axes = plt.subplots(
        len(improvement_metrics),
        1,
        figsize=(8, max(3, 2.8 * len(improvement_metrics))),
        squeeze=False,
    )
    plotted = False
    for axis, metric in zip(axes[:, 0], improvement_metrics):
        rows = rows_by_metric[metric]
        final_step = max(int(row["step"]) for row in rows)
        final_rows = [row for row in rows if int(row["step"]) == final_step]
        baseline_rows = [row for row in final_rows if row["variant"] == baseline_variant]
        if not baseline_rows:
            axis.set_visible(False)
            continue
        baseline_value = float(baseline_rows[0]["value"])
        variants = []
        effects = []
        for row in final_rows:
            variant = str(row["variant"])
            if variant == baseline_variant:
                continue
            variants.append(variant)
            effects.append(baseline_value - float(row["value"]))
        axis.bar(variants, effects)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(f"{metric} improvement")
        axis.set_title(f"{metric}, step {final_step}: {baseline_variant} minus variant")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.25)
        plotted = True
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if plotted:
        fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return plotted


def plot_results(
    csv_path: str,
    out_dir: str,
    metrics: Sequence[str] = METRICS,
    max_step: Optional[int] = None,
    variants: Optional[Sequence[str]] = None,
    baseline_variant: str = "no_prior",
    title: Optional[str] = None,
) -> List[str]:
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}; expected subset of {METRICS}")
    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows_by_metric = _metric_rows_by_metric(csv_path, metrics, max_step=max_step, variants=variants)
    available = _available_metrics(rows_by_metric)
    if not available:
        raise ValueError(f"No plottable rows found in {csv_path}")
    outputs: List[str] = []
    curves_path = out_path / "step_curves.png"
    _plot_step_curves(plt, rows_by_metric, available, title, curves_path)
    outputs.append(str(curves_path))
    improvements_path = out_path / "final_improvements.png"
    if _plot_improvements(plt, rows_by_metric, available, baseline_variant, title, improvements_path):
        outputs.append(str(improvements_path))
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot sparse-sweep curves and final improvements.")
    parser.add_argument("csv")
    parser.add_argument("--metrics", default="obs_l1,eval_l1,latent_norm,conditional_energy")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--variants", default=None, help="Comma-separated variant subset.")
    parser.add_argument("--baseline-variant", default="no_prior")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    outputs = plot_results(
        args.csv,
        out_dir=args.out_dir,
        metrics=_parse_csv_list(args.metrics) or list(METRICS),
        max_step=args.max_step,
        variants=_parse_csv_list(args.variants),
        baseline_variant=args.baseline_variant,
        title=args.title,
    )
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
