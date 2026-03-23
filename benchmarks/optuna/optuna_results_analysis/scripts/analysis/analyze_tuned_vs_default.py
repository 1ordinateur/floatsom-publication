#!/usr/bin/env python3
"""
Run tuned-vs-default paired t-test analysis on seed-preserving Optuna exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.default_benchmark_adapter import (
    load_default_runs_csv,
)
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.tuned_vs_default import (
    TunedDefaultMetric,
    build_tuned_vs_default_report,
    build_tuned_vs_external_default_report,
    diagnose_external_default_pairing_overlap,
)


def _metric_label(column: str) -> str:
    return column.replace("_", " ").title()


def _metric_slug(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tuned-vs-default paired t-test analysis")
    parser.add_argument("--input", required=True, help="Seed-preserving CSV export to analyze")
    parser.add_argument(
        "--defaults-input",
        default=None,
        help="Optional true-default run CSV. If provided, pairs tuned top-1 against explicit defaults.",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory for CSV/markdown")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["quantization_error_holdout", "quantization_error_train"],
        help="Metric columns to analyze (default: quantization_error_holdout quantization_error_train)",
    )
    parser.add_argument(
        "--higher-is-better",
        nargs="*",
        default=[],
        help="Metric columns where higher values indicate better performance",
    )
    parser.add_argument(
        "--markdown-name",
        default="TUNED_VS_DEFAULT_PAIRED_TTEST.md",
        help="Markdown filename for the summary report",
    )
    parser.add_argument(
        "--split-policy",
        choices=["both", "holdout", "train"],
        default="both",
        help="Split policy for explicit true-default pairing (default: both)",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    higher_is_better = {name.strip() for name in args.higher_is_better}

    metrics = [
        TunedDefaultMetric(
            column=metric,
            label=_metric_label(metric),
            slug=_metric_slug(metric),
            higher_is_better=metric in higher_is_better,
        )
        for metric in args.metrics
    ]

    missing = [metric.column for metric in metrics if metric.column not in df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns in input CSV: {missing}")

    output_path = Path(args.output_dir)
    if args.defaults_input:
        default_df = load_default_runs_csv(args.defaults_input, split_policy=args.split_policy)
        overlap = diagnose_external_default_pairing_overlap(
            tuned_df=df,
            default_df=default_df,
            split_policy=args.split_policy,
        )
        print(
            "Pairing overlap diagnostics: "
            f"join_keys={overlap['join_keys']}, "
            f"tuned_unique_keys={overlap['tuned_unique_keys']}, "
            f"default_unique_keys={overlap['default_unique_keys']}, "
            f"matched_unique_keys={overlap['matched_unique_keys']}"
        )
        if int(overlap["matched_unique_keys"]) == 0:
            per_key = overlap.get("per_key", {})
            seed_diag = per_key.get("pair_seed", {})
            raise ValueError(
                "No overlapping pairing keys between tuned and default rows. "
                f"split_policy={args.split_policy}, "
                f"pair_seed overlap={seed_diag.get('overlap_unique', 0)} "
                f"(tuned={seed_diag.get('tuned_unique', 0)}, default={seed_diag.get('default_unique', 0)}). "
                "Regenerate defaults with matching seeds via "
                "floatsom/benchmarks/optuna/run_matched_default_floatsom_batch.py."
            )

        report_path = build_tuned_vs_external_default_report(
            tuned_df=df,
            default_df=default_df,
            metrics=metrics,
            output_dir=output_path,
            markdown_filename=args.markdown_name,
            split_policy=args.split_policy,
        )
    else:
        report_path = build_tuned_vs_default_report(
            df=df,
            metrics=metrics,
            output_dir=output_path,
            markdown_filename=args.markdown_name,
        )
    print(f"Saved tuned-vs-default report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
