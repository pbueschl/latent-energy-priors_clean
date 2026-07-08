"""Aggregate sparse-sweep CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


METRICS = ("obs_l1", "eval_l1", "latent_norm", "conditional_energy")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_csvs(paths: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[str, int], int] = defaultdict(int)
    methods: Dict[Tuple[str, int], set] = defaultdict(set)
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            has_variant = bool(reader.fieldnames and "variant" in reader.fieldnames)
            for row in reader:
                method = row.get("method", "")
                variant = row.get("variant", "") if has_variant else ""
                if not variant:
                    variant = method
                key = (variant, int(row["step"]))
                counts[key] += 1
                groups[key]
                if method:
                    methods[key].add(method)
                for metric in METRICS:
                    value = row.get(metric, "")
                    if value not in ("", None):
                        groups[key][metric].append(float(value))

    rows: List[Dict[str, object]] = []
    for variant, step in sorted(groups.keys(), key=lambda item: (item[0], item[1])):
        key = (variant, step)
        method_values = sorted(methods[key])
        method = method_values[0] if len(method_values) == 1 else "|".join(method_values)
        out: Dict[str, object] = {
            "variant": variant,
            "method": method or variant,
            "step": step,
            "n": counts[key],
        }
        for metric in METRICS:
            values = groups[key].get(metric, [])
            if values:
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_std"] = _std(values)
        rows.append(out)
    return rows


def write_aggregate(rows: Sequence[Dict[str, object]], out: str) -> None:
    fieldnames = ["variant", "method", "step", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate sparse sweep CSVs.")
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    write_aggregate(aggregate_csvs(args.csvs), args.out)


if __name__ == "__main__":
    main()
