"""
Adapter utilities for true-default benchmark run CSVs.

Converts external default-run CSVs into canonical pairing columns used by
default-aware tuned-vs-default analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _normalize_batch_mode(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return "full_batch"
    if text in {"full", "fullbatch"}:
        return "full_batch"
    if text in {"mini", "mini_batch"}:
        return "minibatch"
    return text


def load_default_runs_csv(
    path: str | Path,
    *,
    split_policy: str = "both",
) -> pd.DataFrame:
    """
    Load a default benchmark CSV and return canonical pairing columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Default runs CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Default runs CSV is empty: {csv_path}")

    dataset_col = _resolve_column(df, ("dataset",))
    topology_col = _resolve_column(df, ("architecture", "config_topology_type", "param_topology_type", "map_type"))
    sampling_col = _resolve_column(df, ("sampling_method", "sampling_method_final", "sampling_method_parsed"))
    seed_col = _resolve_column(df, ("seed", "config_seed", "random_seed", "seed_name", "config_seed_name"))
    processing_col = _resolve_column(df, ("processing_type", "algorithm", "method", "config_processing_method"))
    batch_mode_col = _resolve_column(df, ("batch_mode", "config_batch_mode", "param_batch_mode"))
    split_col = _resolve_column(df, ("evaluation_split", "config_evaluation_split", "split"))
    trial_col = _resolve_column(df, ("trial_number", "trial_id", "number"))
    method_col = _resolve_column(df, ("method",))

    missing = [
        name
        for name, value in (
            ("dataset", dataset_col),
            ("topology", topology_col),
            ("sampling", sampling_col),
            ("seed", seed_col),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Default runs CSV missing required columns: {missing}")

    working = df.copy()
    if method_col:
        method_values = _normalize_text(working[method_col])
        if (method_values == "floatsom").any():
            working = working[method_values == "floatsom"].copy()

    working["pair_dataset"] = _normalize_text(working[dataset_col])
    working["pair_topology"] = _normalize_text(working[topology_col])
    working["pair_sampling"] = _normalize_text(working[sampling_col])
    working["pair_seed"] = _normalize_text(working[seed_col])

    if processing_col:
        working["pair_processing"] = _normalize_text(working[processing_col])
    else:
        working["pair_processing"] = "batch"

    if batch_mode_col:
        working["pair_batch_mode"] = working[batch_mode_col].map(_normalize_batch_mode)
    else:
        working["pair_batch_mode"] = "full_batch"

    if split_col:
        working["pair_split"] = _normalize_text(working[split_col])
    else:
        working["pair_split"] = str(split_policy).lower().strip()
    if split_policy:
        working = working[working["pair_split"] == str(split_policy).lower().strip()].copy()

    if trial_col:
        working["default_trial_number"] = pd.to_numeric(working[trial_col], errors="coerce")
    else:
        working["default_trial_number"] = 0.0

    if (
        "balanced_qe_raw" not in working.columns
        and "quantization_error_holdout" in working.columns
        and "quantization_error_train" in working.columns
    ):
        qe_holdout = pd.to_numeric(working["quantization_error_holdout"], errors="coerce")
        qe_train = pd.to_numeric(working["quantization_error_train"], errors="coerce")
        working["balanced_qe_raw"] = (qe_holdout + qe_train) / 2.0

    if working.empty:
        raise ValueError("No default rows remained after filtering (method/split).")

    key_cols = [
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
        "pair_batch_mode",
        "pair_topology",
        "pair_seed",
        "pair_split",
    ]
    dup_counts = working.groupby(key_cols, dropna=False).size().reset_index(name="count")
    duplicates = dup_counts[dup_counts["count"] > 1]
    if not duplicates.empty:
        raise ValueError(
            "Default runs CSV contains duplicate rows for matched keys. "
            "Expected one default row per dataset/seed/topology/sampling/batch/split."
        )

    metrics = [
        col
        for col in ("quantization_error_holdout", "quantization_error_train", "balanced_qe_raw")
        if col in working.columns
    ]
    out_cols = key_cols + ["default_trial_number"] + metrics
    out = working[out_cols].copy()

    for metric_col in metrics:
        out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
