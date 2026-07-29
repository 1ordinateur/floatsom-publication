#!/usr/bin/env python3
"""Build the matched default-aweSOM versus untuned-FloatSOM-hex table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("iris", "wine", "digits", "breast_cancer", "olivetti_faces")
METRICS: Tuple[Tuple[str, str], ...] = (
    ("balanced_qe_raw", "balanced_qe"),
    ("quantization_error_holdout", "holdout_qe"),
    ("quantization_error_train", "train_qe"),
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awesom-root", required=True)
    parser.add_argument("--floatsom-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _discover(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)


def load_matched_runs(awesom_root: Path, floatsom_root: Path) -> pd.DataFrame:
    awe = _discover(awesom_root, "awesom_default_runs.csv")
    awe["implementation"] = "aweSOM default"

    float_frame = _discover(floatsom_root, "matched_untuned_floatsom_hex_runs.csv")
    float_frame = float_frame[
        (float_frame["architecture"].astype(str).str.lower() == "hexagonal")
        & (float_frame["sampling_method"].astype(str).str.lower() == "full")
        & (float_frame["status"].astype(str).str.lower() == "ok")
    ].copy()
    float_frame["implementation"] = "FloatSOM hex untuned"

    columns = ["implementation", "dataset", "seed"] + [metric for metric, _ in METRICS]
    combined = pd.concat([awe[columns], float_frame[columns]], ignore_index=True)
    combined = combined[combined["dataset"].astype(str).isin(DATASETS)].copy()
    combined["seed"] = pd.to_numeric(combined["seed"], errors="raise").astype(int)
    for metric, _ in METRICS:
        combined[metric] = pd.to_numeric(combined[metric], errors="raise")

    counts = combined.groupby(["implementation", "dataset"]).size()
    expected = pd.MultiIndex.from_product(
        [["aweSOM default", "FloatSOM hex untuned"], DATASETS],
        names=["implementation", "dataset"],
    )
    counts = counts.reindex(expected, fill_value=0)
    invalid = counts[counts != 10]
    if not invalid.empty:
        raise ValueError(f"Expected 10 matched runs per implementation/dataset; got {invalid.to_dict()}")
    if combined.duplicated(["implementation", "dataset", "seed"]).any():
        raise ValueError("Duplicate implementation/dataset/seed rows detected.")
    return combined


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half_width = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half_width, mean + half_width


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def build_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        awe = combined[
            (combined["implementation"] == "aweSOM default")
            & (combined["dataset"] == dataset)
        ].set_index("seed")
        floatsom = combined[
            (combined["implementation"] == "FloatSOM hex untuned")
            & (combined["dataset"] == dataset)
        ].set_index("seed")
        matched = awe.join(floatsom, lsuffix="_awe", rsuffix="_floatsom", how="inner")
        if len(matched) != 10:
            raise ValueError(f"Expected 10 matched seeds for {dataset}; got {len(matched)}")

        for metric, label in METRICS:
            awe_values = matched[f"{metric}_awe"].to_numpy(dtype=float)
            float_values = matched[f"{metric}_floatsom"].to_numpy(dtype=float)
            differences = float_values - awe_values
            improvements = 100.0 * (awe_values - float_values) / awe_values
            awe_mean, awe_low, awe_high = _mean_ci(awe_values)
            float_mean, float_low, float_high = _mean_ci(float_values)
            diff_mean, diff_low, diff_high = _mean_ci(differences)
            improvement_mean, improvement_low, improvement_high = _mean_ci(improvements)
            test = stats.ttest_rel(float_values, awe_values, alternative="two-sided")
            rows.append(
                {
                    "dataset": dataset,
                    "metric": label,
                    "n_pairs": len(matched),
                    "awesom_default_mean": awe_mean,
                    "awesom_default_ci_low": awe_low,
                    "awesom_default_ci_high": awe_high,
                    "floatsom_hex_untuned_mean": float_mean,
                    "floatsom_hex_untuned_ci_low": float_low,
                    "floatsom_hex_untuned_ci_high": float_high,
                    "floatsom_minus_awesom_mean": diff_mean,
                    "floatsom_minus_awesom_ci_low": diff_low,
                    "floatsom_minus_awesom_ci_high": diff_high,
                    "floatsom_improvement_pct_mean": improvement_mean,
                    "floatsom_improvement_pct_ci_low": improvement_low,
                    "floatsom_improvement_pct_ci_high": improvement_high,
                    "paired_t_p_raw": float(test.pvalue),
                }
            )
    table = pd.DataFrame(rows)
    table["paired_t_p_holm"] = np.nan
    for metric, _ in METRICS:
        label = dict(METRICS)[metric]
        mask = table["metric"] == label
        table.loc[mask, "paired_t_p_holm"] = _holm_adjust(
            table.loc[mask, "paired_t_p_raw"].to_numpy(dtype=float)
        )
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    combined = load_matched_runs(
        Path(args.awesom_root).expanduser().resolve(),
        Path(args.floatsom_root).expanduser().resolve(),
    )
    table = build_table(combined)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, sep="\t" if output.suffix.lower() == ".tsv" else ",", index=False)
    temporary.replace(output)
    print(f"Wrote supplementary table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
