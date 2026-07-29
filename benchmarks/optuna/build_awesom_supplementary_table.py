#!/usr/bin/env python3
"""Build one supplementary quality table from aweSOM and FloatSOM Optuna CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = (
    "iris",
    "wine",
    "digits",
    "breast_cancer",
    "olivetti_faces",
)
TRIALS_FILENAME = "awesom_optuna_trials.csv"


METRICS: Tuple[Tuple[str, str], ...] = (
    ("balanced_qe_raw", "qe_balanced"),
    ("quantization_error_holdout", "qe_holdout"),
    ("quantization_error_train", "qe_train"),
)
TOP_K = 5


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awesom-results-root", required=True)
    parser.add_argument("--floatsom-processed-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args(argv)


def discover_awesom_trials(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob(TRIALS_FILENAME))
    if not paths:
        raise FileNotFoundError(f"No {TRIALS_FILENAME} files found under {root}")
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[
        combined["dataset"].astype(str).isin(DATASETS)
        & (combined["state"].astype(str).str.lower() == "complete")
    ].copy()
    for metric, _ in METRICS:
        combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    return combined.dropna(subset=[metric for metric, _ in METRICS])


def _normalize_floatsom(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "seed",
        "trial_number",
        "architecture",
        "balanced_qe_raw",
        "quantization_error_holdout",
        "quantization_error_train",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FloatSOM CSV is missing columns: {missing}")
    out = frame.copy()
    out = out[out["dataset"].astype(str).isin(DATASETS)]
    if "sampling_method_final" in out.columns:
        out = out[out["sampling_method_final"].astype(str).str.lower() == "full"]
    elif "sampling_method" in out.columns:
        out = out[out["sampling_method"].astype(str).str.lower() == "full"]
    if "processing_type" in out.columns:
        out = out[out["processing_type"].astype(str).str.lower() == "batch"]
    out["implementation"] = "FloatSOM"
    out["architecture"] = out["architecture"].astype(str).str.lower()
    out = out[out["architecture"].isin(["hexagonal", "mst", "rng"])]
    for metric, _ in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.dropna(subset=[metric for metric, _ in METRICS])


def select_top_k_units(
    frame: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Return one top-k median record per implementation/dataset/seed/topology."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    working = frame.copy()
    if "implementation" not in working:
        raise ValueError("Input frame must include implementation.")
    if "architecture" not in working:
        raise ValueError("Input frame must include architecture.")
    keys = ["implementation", "architecture", "dataset", "seed"]
    selected = (
        working.sort_values(keys + ["balanced_qe_raw", "trial_number"])
        .groupby(keys, dropna=False, group_keys=False)
        .head(int(top_k))
    )
    aggregations: Dict[str, str] = {
        metric: "median" for metric, _ in METRICS
    }
    aggregations["trial_number"] = "count"
    units = (
        selected.groupby(keys, dropna=False)
        .agg(aggregations)
        .reset_index()
        .rename(columns={"trial_number": "retained_trials"})
    )
    units["seed"] = pd.to_numeric(units["seed"], errors="coerce").astype("Int64")
    return units


def _mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, float("nan"), float("nan")
    sem = float(stats.sem(array))
    critical = float(stats.t.ppf(0.975, df=array.size - 1))
    return mean, mean - critical * sem, mean + critical * sem


def summarize_units(units: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    keys = ["implementation", "architecture", "dataset"]
    for (implementation, architecture, dataset), group in units.groupby(keys, dropna=False):
        row: Dict[str, object] = {
            "dataset": dataset,
            "implementation": implementation,
            "topology": architecture,
            "n_seeds": int(group["seed"].nunique()),
            "top_k": int(group["retained_trials"].max()),
        }
        for metric, slug in METRICS:
            mean, low, high = _mean_ci(group[metric])
            row[f"{slug}_mean"] = mean
            row[f"{slug}_ci_low"] = low
            row[f"{slug}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def add_paired_effects(summary: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Add FloatSOM improvement relative to aweSOM on matched seed units."""
    output = summary.copy()
    output["matched_seed_count_vs_awesom"] = np.nan
    output["balanced_qe_improvement_vs_awesom_pct"] = np.nan
    output["balanced_qe_improvement_ci_low_pct"] = np.nan
    output["balanced_qe_improvement_ci_high_pct"] = np.nan

    awe = units[
        (units["implementation"] == "aweSOM")
        & (units["architecture"] == "rectangular_chebyshev")
    ][["dataset", "seed", "balanced_qe_raw"]].rename(
        columns={"balanced_qe_raw": "awesom_qe"}
    )
    for index, row in output.iterrows():
        if row["implementation"] != "FloatSOM":
            continue
        candidate = units[
            (units["implementation"] == "FloatSOM")
            & (units["architecture"] == row["topology"])
            & (units["dataset"] == row["dataset"])
        ][["dataset", "seed", "balanced_qe_raw"]].rename(
            columns={"balanced_qe_raw": "floatsom_qe"}
        )
        matched = candidate.merge(awe, on=["dataset", "seed"], how="inner")
        if matched.empty:
            continue
        effects = 100.0 * (
            matched["awesom_qe"].to_numpy(dtype=float)
            - matched["floatsom_qe"].to_numpy(dtype=float)
        ) / matched["awesom_qe"].to_numpy(dtype=float)
        mean, low, high = _mean_ci(effects)
        output.at[index, "matched_seed_count_vs_awesom"] = int(len(matched))
        output.at[index, "balanced_qe_improvement_vs_awesom_pct"] = mean
        output.at[index, "balanced_qe_improvement_ci_low_pct"] = low
        output.at[index, "balanced_qe_improvement_ci_high_pct"] = high
    return output


def build_table(
    awesom: pd.DataFrame,
    *,
    floatsom: Optional[pd.DataFrame] = None,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    awe = awesom.copy()
    awe["implementation"] = "aweSOM"
    awe["architecture"] = "rectangular_chebyshev"
    frames = [awe]
    if floatsom is not None:
        frames.append(_normalize_floatsom(floatsom))
    units = select_top_k_units(pd.concat(frames, ignore_index=True, sort=False), top_k=top_k)
    summary = summarize_units(units)
    if floatsom is not None:
        summary = add_paired_effects(summary, units)
    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    implementation_order = {
        ("aweSOM", "rectangular_chebyshev"): 0,
        ("FloatSOM", "hexagonal"): 1,
        ("FloatSOM", "mst"): 2,
        ("FloatSOM", "rng"): 3,
    }
    summary["_dataset_order"] = summary["dataset"].map(order)
    summary["_method_order"] = [
        implementation_order.get((str(implementation), str(topology)), 99)
        for implementation, topology in zip(
            summary["implementation"],
            summary["topology"],
        )
    ]
    return (
        summary.sort_values(["_dataset_order", "_method_order"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    awesom = discover_awesom_trials(
        Path(args.awesom_results_root).expanduser().resolve()
    )
    floatsom = (
        pd.read_csv(Path(args.floatsom_processed_csv).expanduser().resolve())
        if args.floatsom_processed_csv
        else None
    )
    table = build_table(awesom, floatsom=floatsom, top_k=int(args.top_k))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.suffix.lower() == ".tsv":
        table.to_csv(temporary, sep="\t", index=False)
    else:
        table.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"Wrote supplementary table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
