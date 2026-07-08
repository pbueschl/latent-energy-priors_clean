"""Paired bootstrap confidence intervals for sparse-sweep variants."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Pair = Tuple[str, str]


def _row_variant(row: Dict[str, str]) -> str:
    return row.get("variant") or row.get("method") or "unknown"


def _pair_key(row: Dict[str, str]) -> Pair:
    return (row.get("shape_id", ""), row.get("repeat", ""))


def _select_step(rows: Sequence[Dict[str, str]], step: Optional[int]) -> int:
    if step is not None:
        return int(step)
    if not rows:
        raise ValueError("No rows available for bootstrap")
    return max(int(float(row["step"])) for row in rows)


def _read_raw_rows(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _paired_values(
    rows: Sequence[Dict[str, str]],
    metric: str,
    step: int,
) -> Dict[str, Dict[Pair, float]]:
    values: Dict[str, Dict[Pair, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(float(row["step"])) != step:
            continue
        value = row.get(metric, "")
        if value in ("", None):
            continue
        values[_row_variant(row)][_pair_key(row)].append(float(value))
    return {
        variant: {pair: sum(pair_values) / len(pair_values) for pair, pair_values in pair_map.items()}
        for variant, pair_map in values.items()
    }


def paired_bootstrap(
    path: str,
    baseline_variant: str,
    variants: Optional[Iterable[str]] = None,
    metric: str = "eval_l1",
    step: Optional[int] = None,
    num_bootstrap: int = 1000,
    seed: int = 123,
    lower_is_better: bool = True,
) -> Dict[str, object]:
    rows = _read_raw_rows(path)
    selected_step = _select_step(rows, step)
    values = _paired_values(rows, metric=metric, step=selected_step)
    if baseline_variant not in values:
        raise ValueError(f"Baseline variant {baseline_variant!r} not found at step {selected_step}")
    selected_variants = list(variants or [variant for variant in values if variant != baseline_variant])
    rng = np.random.default_rng(seed)
    out_rows: List[Dict[str, object]] = []
    baseline = values[baseline_variant]

    for variant in selected_variants:
        if variant == baseline_variant:
            continue
        if variant not in values:
            raise ValueError(f"Variant {variant!r} not found at step {selected_step}")
        shared_pairs = sorted(set(baseline).intersection(values[variant]))
        if not shared_pairs:
            raise ValueError(f"No paired rows for baseline {baseline_variant!r} and variant {variant!r}")
        effects = np.asarray(
            [
                baseline[pair] - values[variant][pair]
                if lower_is_better
                else values[variant][pair] - baseline[pair]
                for pair in shared_pairs
            ],
            dtype=np.float64,
        )
        if effects.shape[0] == 1:
            boot_means = np.repeat(effects.mean(), int(num_bootstrap))
        else:
            indices = rng.integers(0, effects.shape[0], size=(int(num_bootstrap), effects.shape[0]))
            boot_means = effects[indices].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        out_rows.append(
            {
                "baseline": baseline_variant,
                "variant": variant,
                "metric": metric,
                "step": selected_step,
                "higher_is_better": not lower_is_better,
                "mean_effect": float(effects.mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "num_pairs": int(effects.shape[0]),
                "num_bootstrap": int(num_bootstrap),
                "seed": int(seed),
            }
        )

    return {
        "source_file": path,
        "baseline": baseline_variant,
        "metric": metric,
        "step": selected_step,
        "lower_is_better": lower_is_better,
        "num_bootstrap": int(num_bootstrap),
        "seed": int(seed),
        "results": out_rows,
    }


def write_bootstrap_outputs(result: Dict[str, object], out_json: str, out_csv: str) -> None:
    out_json_path = Path(out_json)
    out_csv_path = Path(out_csv)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "baseline",
                "variant",
                "metric",
                "step",
                "higher_is_better",
                "mean_effect",
                "ci_low",
                "ci_high",
                "num_pairs",
                "num_bootstrap",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(result["results"])


def _default_outputs(path: str, baseline: str) -> Tuple[str, str]:
    base = Path(path)
    stem = f"{base.stem}_bootstrap_vs_{baseline}"
    return (
        str(base.with_name(f"{stem}.json")),
        str(base.with_name(f"{stem}.csv")),
    )


def _parse_variants(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for sparse-sweep variants.")
    parser.add_argument("csv")
    parser.add_argument("--baseline-variant", default="no_prior")
    parser.add_argument("--variants", default=None, help="Comma-separated variants to compare.")
    parser.add_argument("--metric", default="eval_l1")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)

    default_json, default_csv = _default_outputs(args.csv, args.baseline_variant)
    result = paired_bootstrap(
        args.csv,
        baseline_variant=args.baseline_variant,
        variants=_parse_variants(args.variants),
        metric=args.metric,
        step=args.step,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        lower_is_better=not args.higher_is_better,
    )
    write_bootstrap_outputs(
        result,
        out_json=args.out_json or default_json,
        out_csv=args.out_csv or default_csv,
    )
    print(json.dumps({"out_json": args.out_json or default_json, "out_csv": args.out_csv or default_csv}, indent=2))


if __name__ == "__main__":
    main()

