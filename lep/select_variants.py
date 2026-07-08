"""Validation-based sparse-sweep variant selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FAMILIES = (
    "no_prior",
    "l2",
    "conditional_energy",
    "l2_conditional_energy",
)


def _as_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    return float(value)


def _row_variant(row: Dict[str, str]) -> str:
    return row.get("variant") or row.get("method") or "unknown"


def _row_method(row: Dict[str, str], variant: str) -> str:
    return row.get("method") or variant


def _metric_value(row: Dict[str, str], metric: str) -> Optional[float]:
    if row.get(f"{metric}_mean", "") not in ("", None):
        return float(row[f"{metric}_mean"])
    return _as_float(row.get(metric, ""))


def infer_family(variant: str, method: str = "") -> str:
    method = method or variant
    method_lower = method.lower()
    variant_lower = variant.lower()
    if method_lower in FAMILIES:
        return method_lower
    if "l2_conditional" in variant_lower or "conditional_l2" in variant_lower:
        return "l2_conditional_energy"
    if "conditional" in variant_lower or "_cond" in variant_lower or variant_lower.startswith("cond"):
        return "conditional_energy"
    if variant_lower == "no_prior" or variant_lower.startswith("no_prior"):
        return "no_prior"
    if variant_lower.startswith("l2") or "_l2" in variant_lower:
        return "l2"
    return method_lower if method_lower in FAMILIES else "unknown"


def read_metric_means(
    path: str,
    metric: str,
    max_step: Optional[int] = None,
    variants: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """Read raw or aggregate sparse-sweep CSV and return mean rows."""
    keep_variants = set(variants or [])
    grouped: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    counts: Dict[Tuple[str, str, int], int] = defaultdict(int)
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = _row_variant(row)
            if keep_variants and variant not in keep_variants:
                continue
            step = int(float(row["step"]))
            if max_step is not None and step > max_step:
                continue
            value = _metric_value(row, metric)
            if value is None:
                continue
            method = _row_method(row, variant)
            key = (variant, method, step)
            grouped[key].append(value)
            counts[key] += int(float(row.get("n", "1") or 1))

    rows: List[Dict[str, object]] = []
    for (variant, method, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        rows.append(
            {
                "variant": variant,
                "method": method,
                "family": infer_family(variant, method),
                "step": step,
                "metric": metric,
                "value": sum(values) / len(values),
                "n": counts[(variant, method, step)],
            }
        )
    return rows


def select_variants(
    path: str,
    metric: str = "eval_l1",
    step: Optional[int] = None,
    lower_is_better: bool = True,
) -> Dict[str, object]:
    rows = read_metric_means(path, metric)
    if not rows:
        raise ValueError(f"No rows found for metric {metric!r} in {path}")
    selection_step = int(step) if step is not None else max(int(row["step"]) for row in rows)
    step_rows = [row for row in rows if int(row["step"]) == selection_step]
    if not step_rows:
        raise ValueError(f"No rows found for metric {metric!r} at step {selection_step}")

    ranking = sorted(step_rows, key=lambda row: float(row["value"]), reverse=not lower_is_better)
    ranked_rows: List[Dict[str, object]] = []
    selected: Dict[str, Dict[str, object]] = {}
    for rank, row in enumerate(ranking, start=1):
        compact = {
            "rank": rank,
            "variant": row["variant"],
            "method": row["method"],
            "family": row["family"],
            "step": selection_step,
            "metric": metric,
            "value": row["value"],
            "n": row["n"],
        }
        ranked_rows.append(compact)
        selected.setdefault(str(row["family"]), compact)

    return {
        "source_file": path,
        "metric": metric,
        "step": selection_step,
        "lower_is_better": lower_is_better,
        "selected": selected,
        "ranking": ranked_rows,
    }


def write_selection_outputs(selection: Dict[str, object], out_json: str, out_csv: str) -> None:
    out_json_path = Path(out_json)
    out_csv_path = Path(out_csv)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, sort_keys=True)
        f.write("\n")
    ranking = list(selection["ranking"])
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "variant", "method", "family", "step", "metric", "value", "n"],
        )
        writer.writeheader()
        writer.writerows(ranking)


def _default_outputs(path: str) -> Tuple[str, str]:
    base = Path(path)
    return (
        str(base.with_name(f"{base.stem}_selection.json")),
        str(base.with_name(f"{base.stem}_ranking.csv")),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Select sparse-sweep variants on validation metrics.")
    parser.add_argument("csv")
    parser.add_argument("--metric", default="eval_l1")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)

    default_json, default_csv = _default_outputs(args.csv)
    selection = select_variants(
        args.csv,
        metric=args.metric,
        step=args.step,
        lower_is_better=not args.higher_is_better,
    )
    write_selection_outputs(
        selection,
        out_json=args.out_json or default_json,
        out_csv=args.out_csv or default_csv,
    )
    print(json.dumps({"out_json": args.out_json or default_json, "out_csv": args.out_csv or default_csv}, indent=2))


if __name__ == "__main__":
    main()
