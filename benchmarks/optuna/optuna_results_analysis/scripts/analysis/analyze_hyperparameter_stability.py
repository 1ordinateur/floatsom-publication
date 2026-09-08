#!/usr/bin/env python3
"""
Run hyperparameter stability analysis across topology pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.hyperparameter_stability import (
    StabilityMetric,
    build_hyperparameter_stability_report,
)


def _parse_topology_pairs(raw_pairs: List[str]) -> Tuple[Tuple[str, str], ...]:
    parsed: List[Tuple[str, str]] = []
    for item in raw_pairs:
        if ":" not in item:
            raise ValueError(f"Invalid topology pair '{item}'. Use format like hexagonal:mst")
        left, right = item.split(":", 1)
        left = left.strip().lower()
        right = right.strip().lower()
        if not left or not right:
            raise ValueError(f"Invalid topology pair '{item}'. Use format like hexagonal:mst")
        parsed.append((left, right))
    return tuple(parsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter stability analysis")
    parser.add_argument("--input", required=True, help="Seed-preserving CSV export to analyze")
    parser.add_argument("--output-dir", default="results", help="Output directory for CSV/markdown")
    parser.add_argument(
        "--metric",
        default="quantization_error_holdout",
        help="Metric column for selecting tuned trials (default: quantization_error_holdout)",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat metric as higher-is-better when selecting tuned trials",
    )
    parser.add_argument(
        "--topology-pairs",
        nargs="+",
        default=["hexagonal:mst", "hexagonal:rng"],
        help="Topology pairs to compare (format: topo_a:topo_b)",
    )
    parser.add_argument(
        "--markdown-name",
        default="HYPERPARAMETER_STABILITY.md",
        help="Markdown filename for the summary report",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    topology_pairs = _parse_topology_pairs(args.topology_pairs)

    metric = StabilityMetric(
        column=args.metric,
        label=args.metric.replace("_", " ").title(),
        slug=args.metric.strip().lower().replace(" ", "_"),
        higher_is_better=args.higher_is_better,
    )

    if metric.column not in df.columns:
        raise ValueError(f"Missing required metric column in input CSV: {metric.column}")

    output_path = Path(args.output_dir)
    report_path = build_hyperparameter_stability_report(
        df=df,
        metric=metric,
        output_dir=output_path,
        topology_pairs=topology_pairs,
        markdown_filename=args.markdown_name,
    )
    print(f"Saved hyperparameter stability report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
