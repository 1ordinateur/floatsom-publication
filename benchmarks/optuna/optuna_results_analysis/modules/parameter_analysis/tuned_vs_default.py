"""
Paired tuned-vs-default analysis for seed-preserving Optuna exports.

Pairs default trials (lowest trial_number) with tuned trials (best metric value)
within matched dataset/topology/sampling/algorithm/seed keys and performs
paired t-tests with dataset, dataset-type, and global summaries.

Also supports pairing tuned top-1 rows against explicit true-default CSV rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.dataset_groups import dataset_group, ordered_dataset_labels
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.default_benchmark_adapter import load_default_runs_csv


@dataclass(frozen=True)
class TunedDefaultMetric:
    """Metric configuration for tuned-vs-default analysis."""

    column: str
    label: str
    slug: str
    higher_is_better: bool = False


@dataclass(frozen=True)
class TunedDefaultConfig:
    """Column resolution config for tuned-vs-default analysis."""

    dataset_col_candidates: Tuple[str, ...] = ("dataset",)
    processing_col_candidates: Tuple[str, ...] = ("processing_type", "algorithm")
    sampling_col_candidates: Tuple[str, ...] = (
        "sampling_method",
        "sampling_method_parsed",
        "sampling_method_final",
    )
    topology_col_candidates: Tuple[str, ...] = (
        "config_topology_type",
        "param_topology_type",
        "architecture",
        "map_type",
    )
    seed_col_candidates: Tuple[str, ...] = (
        "seed",
        "config_seed",
        "random_seed",
        "seed_name",
        "config_seed_name",
    )
    trial_number_col_candidates: Tuple[str, ...] = ("trial_number", "trial_id", "number")
    batch_mode_col_candidates: Tuple[str, ...] = (
        "pair_batch_mode",
        "batch_mode",
        "config_batch_mode",
        "param_batch_mode",
    )
    split_col_candidates: Tuple[str, ...] = (
        "evaluation_split",
        "config_evaluation_split",
        "split",
    )


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


def _resolve_pairing_columns(
    df: pd.DataFrame,
    config: TunedDefaultConfig,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Optional[str]]]:
    dataset_col = _resolve_column(df, config.dataset_col_candidates)
    processing_col = _resolve_column(df, config.processing_col_candidates)
    sampling_col = _resolve_column(df, config.sampling_col_candidates)
    topology_col = _resolve_column(df, config.topology_col_candidates)
    seed_col = _resolve_column(df, config.seed_col_candidates)
    trial_col = _resolve_column(df, config.trial_number_col_candidates)
    batch_mode_col = _resolve_column(df, config.batch_mode_col_candidates)
    split_col = _resolve_column(df, config.split_col_candidates)

    missing = [
        name
        for name, value in (
            ("dataset", dataset_col),
            ("processing", processing_col),
            ("sampling", sampling_col),
            ("topology", topology_col),
            ("seed", seed_col),
            ("trial_number", trial_col),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required pairing columns: {missing}")

    working = df.copy()
    working["pair_dataset"] = _normalize_text(working[dataset_col])
    working["pair_processing"] = _normalize_text(working[processing_col])
    working["pair_sampling"] = _normalize_text(working[sampling_col])
    working["pair_topology"] = _normalize_text(working[topology_col])
    working["pair_seed"] = _normalize_text(working[seed_col])
    working["pair_trial_number"] = pd.to_numeric(working[trial_col], errors="coerce")
    if batch_mode_col:
        working["pair_batch_mode"] = working[batch_mode_col].map(_normalize_batch_mode)
    else:
        working["pair_batch_mode"] = np.nan
    is_batch = working["pair_processing"] == "batch"
    working.loc[is_batch & working["pair_batch_mode"].isna(), "pair_batch_mode"] = "full_batch"
    working.loc[~is_batch, "pair_batch_mode"] = "all"
    working["pair_batch_mode"] = working["pair_batch_mode"].fillna("full_batch")

    if split_col:
        working["pair_split"] = _normalize_text(working[split_col])

    key_cols = [
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
        "pair_batch_mode",
        "pair_topology",
        "pair_seed",
    ]
    if split_col:
        key_cols.append("pair_split")

    selected = {
        "dataset": dataset_col,
        "processing": processing_col,
        "sampling": sampling_col,
        "topology": topology_col,
        "seed": seed_col,
        "trial_number": trial_col,
        "batch_mode": batch_mode_col,
        "split": split_col,
    }
    return working, key_cols, selected


def _apply_split_policy(df: pd.DataFrame, split_value: str = "both") -> pd.DataFrame:
    if "pair_split" not in df.columns:
        return df
    target = str(split_value).lower().strip()
    split_series = df["pair_split"].astype(str).str.lower().str.strip()
    has_target = (split_series == target).any()
    if has_target:
        return df[split_series == target].copy()

    # Backward-compat path: some prepared tuned exports synthesize pair_split="all"
    # when no explicit split metadata exists. In that case, avoid dropping all rows.
    synthetic_markers = {"all", "", "nan", "none"}
    present_markers = set(split_series.dropna().unique().tolist())
    if present_markers and present_markers.issubset(synthetic_markers):
        return df.copy()
    return df.iloc[0:0].copy()


def _key_diagnostics(
    tuned_keys: pd.DataFrame,
    default_keys: pd.DataFrame,
    join_keys: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tuned_unique = tuned_keys[list(join_keys)].drop_duplicates().copy()
    default_unique = default_keys[list(join_keys)].drop_duplicates().copy()
    merged = tuned_unique.merge(default_unique, on=list(join_keys), how="outer", indicator=True)
    tuned_only = merged[merged["_merge"] == "left_only"].drop(columns="_merge").copy()
    default_only = merged[merged["_merge"] == "right_only"].drop(columns="_merge").copy()
    return tuned_only, default_only


def _normalized_value_set(series: pd.Series) -> set[str]:
    values = series.astype(str).str.strip().str.lower()
    return {value for value in values if value not in {"", "nan", "none"}}


def _preview_values(values: set[str], limit: int = 8) -> str:
    ordered = sorted(values)
    if len(ordered) <= limit:
        return ", ".join(ordered) if ordered else "<none>"
    shown = ", ".join(ordered[:limit])
    return f"{shown}, ... (+{len(ordered) - limit} more)"


def _build_no_overlap_message(
    *,
    join_keys: Sequence[str],
    tuned_unique: pd.DataFrame,
    default_unique: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "No overlapping pairing keys between tuned and default rows.",
        f"Join keys: {list(join_keys)}",
        f"Unique tuned keys: {int(len(tuned_unique))}",
        f"Unique default keys: {int(len(default_unique))}",
    ]

    focus_keys = [
        key
        for key in ("pair_seed", "pair_dataset", "pair_sampling", "pair_batch_mode", "pair_topology", "pair_split")
        if key in join_keys
    ]
    for key in focus_keys:
        tuned_values = _normalized_value_set(tuned_unique[key])
        default_values = _normalized_value_set(default_unique[key])
        overlap_values = tuned_values.intersection(default_values)
        lines.append(
            f"{key}: tuned={len(tuned_values)}, default={len(default_values)}, overlap={len(overlap_values)}"
        )
        if key == "pair_seed":
            lines.append(f"  tuned seeds: {_preview_values(tuned_values)}")
            lines.append(f"  default seeds: {_preview_values(default_values)}")

    lines.append(
        "Regenerate matched defaults with the same seed set as the tuned export "
        "(see floatsom/benchmarks/optuna/run_matched_default_floatsom_batch.py)."
    )
    return "\n".join(lines)


def diagnose_external_default_pairing_overlap(
    *,
    tuned_df: pd.DataFrame,
    default_df: pd.DataFrame,
    split_policy: str = "both",
    config: Optional[TunedDefaultConfig] = None,
) -> Dict[str, object]:
    """Return key-overlap diagnostics for tuned-vs-default pairing."""
    resolved_config = config or TunedDefaultConfig()
    tuned_working, key_cols, _ = _resolve_pairing_columns(tuned_df, resolved_config)
    tuned_working = _apply_split_policy(tuned_working, split_value=split_policy)

    defaults = default_df.copy()
    join_keys = [col for col in key_cols if col in defaults.columns]
    missing_keys = [col for col in key_cols if col not in join_keys]
    if missing_keys:
        raise ValueError(f"Default dataframe missing required pairing keys: {missing_keys}")

    if tuned_working.empty:
        return {
            "join_keys": list(join_keys),
            "tuned_unique_keys": 0,
            "default_unique_keys": int(len(defaults[join_keys].drop_duplicates())),
            "matched_unique_keys": 0,
            "overlap_ratio_tuned": 0.0,
            "overlap_ratio_default": 0.0,
            "per_key": {},
        }

    tuned_unique = tuned_working[join_keys].drop_duplicates().copy()
    default_unique = defaults[join_keys].drop_duplicates().copy()
    matched_unique = tuned_unique.merge(default_unique, on=list(join_keys), how="inner")

    per_key: Dict[str, Dict[str, object]] = {}
    for key in join_keys:
        tuned_values = _normalized_value_set(tuned_unique[key])
        default_values = _normalized_value_set(default_unique[key])
        overlap_values = tuned_values.intersection(default_values)
        per_key[key] = {
            "tuned_unique": int(len(tuned_values)),
            "default_unique": int(len(default_values)),
            "overlap_unique": int(len(overlap_values)),
            "tuned_preview": _preview_values(tuned_values),
            "default_preview": _preview_values(default_values),
        }

    tuned_count = int(len(tuned_unique))
    default_count = int(len(default_unique))
    matched_count = int(len(matched_unique))
    return {
        "join_keys": list(join_keys),
        "tuned_unique_keys": tuned_count,
        "default_unique_keys": default_count,
        "matched_unique_keys": matched_count,
        "overlap_ratio_tuned": float(matched_count) / float(tuned_count) if tuned_count else 0.0,
        "overlap_ratio_default": float(matched_count) / float(default_count) if default_count else 0.0,
        "per_key": per_key,
    }


def _select_default_trials(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    base = df.dropna(subset=["pair_trial_number"]).copy()
    if base.empty:
        return pd.DataFrame()

    ordered = base.sort_values("pair_trial_number", ascending=True)
    default_rows = ordered.groupby(list(key_cols), dropna=False).head(1)
    return default_rows


def _select_best_trials(
    df: pd.DataFrame,
    metric_col: str,
    key_cols: Sequence[str],
    higher_is_better: bool,
) -> pd.DataFrame:
    base = df.dropna(subset=[metric_col]).copy()
    if base.empty:
        return pd.DataFrame()

    ascending = not higher_is_better
    order_cols = [metric_col]
    if "pair_trial_number" in base.columns:
        order_cols.append("pair_trial_number")

    base = base.sort_values(order_cols, ascending=[ascending] * len(order_cols))
    best_rows = base.groupby(list(key_cols), dropna=False).head(1)
    return best_rows


def build_metric_pairs(
    df: pd.DataFrame,
    metric: TunedDefaultMetric,
    config: Optional[TunedDefaultConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Return paired default/tuned rows for a given metric."""
    resolved_config = config or TunedDefaultConfig()
    working, key_cols, selected = _resolve_pairing_columns(df, resolved_config)

    if metric.column not in working.columns:
        raise KeyError(f"Metric column '{metric.column}' not found in dataframe.")

    default_rows = _select_default_trials(working, key_cols)
    tuned_rows = _select_best_trials(working, metric.column, key_cols, metric.higher_is_better)
    if default_rows.empty or tuned_rows.empty:
        return pd.DataFrame(), selected

    merged = default_rows.merge(
        tuned_rows,
        on=list(key_cols),
        suffixes=("_default", "_tuned"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), selected

    merged = merged.rename(
        columns={
            f"{metric.column}_default": "default_value",
            f"{metric.column}_tuned": "tuned_value",
            "pair_trial_number_default": "default_trial_number",
            "pair_trial_number_tuned": "tuned_trial_number",
        }
    )

    merged["delta_tuned_minus_default"] = merged["tuned_value"] - merged["default_value"]
    denom = merged["default_value"].abs()
    merged["pct_improvement"] = np.where(
        denom > 0,
        (merged["default_value"] - merged["tuned_value"]) / denom * 100.0,
        np.nan,
    )
    if metric.higher_is_better:
        merged["pct_improvement"] = np.where(
            denom > 0,
            (merged["tuned_value"] - merged["default_value"]) / denom * 100.0,
            np.nan,
        )

    return merged, selected


def build_metric_pairs_against_defaults(
    tuned_df: pd.DataFrame,
    default_df: pd.DataFrame,
    metric: TunedDefaultMetric,
    config: Optional[TunedDefaultConfig] = None,
    split_policy: str = "both",
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], pd.DataFrame, pd.DataFrame]:
    """
    Pair tuned top-1 trials against an explicit true-default dataframe.
    """
    resolved_config = config or TunedDefaultConfig()
    tuned_working, key_cols, selected = _resolve_pairing_columns(tuned_df, resolved_config)
    tuned_working = _apply_split_policy(tuned_working, split_value=split_policy)
    if tuned_working.empty:
        return pd.DataFrame(), selected, pd.DataFrame(), pd.DataFrame()
    if metric.column not in tuned_working.columns:
        raise KeyError(f"Metric column '{metric.column}' not found in tuned dataframe.")
    if metric.column not in default_df.columns:
        raise KeyError(f"Metric column '{metric.column}' not found in default dataframe.")

    tuned_best = _select_best_trials(tuned_working, metric.column, key_cols, metric.higher_is_better)
    if tuned_best.empty:
        return pd.DataFrame(), selected, pd.DataFrame(), pd.DataFrame()

    defaults = default_df.copy()
    join_keys = [col for col in key_cols if col in defaults.columns]
    missing_keys = [col for col in key_cols if col not in join_keys]
    if missing_keys:
        raise ValueError(f"Default dataframe missing required pairing keys: {missing_keys}")

    tuned_only, default_only = _key_diagnostics(tuned_best, defaults, join_keys=join_keys)
    tuned_unique = tuned_best[join_keys].drop_duplicates().copy()
    default_unique = defaults[join_keys].drop_duplicates().copy()
    overlap_unique = tuned_unique.merge(default_unique, on=list(join_keys), how="inner")
    if overlap_unique.empty:
        raise ValueError(
            _build_no_overlap_message(
                join_keys=join_keys,
                tuned_unique=tuned_unique,
                default_unique=default_unique,
            )
        )

    default_cols = join_keys + [metric.column]
    if "default_trial_number" in defaults.columns:
        default_cols.append("default_trial_number")
    default_metric = defaults[default_cols].rename(columns={metric.column: "default_value"}).copy()

    tuned_metric = tuned_best[join_keys + [metric.column, "pair_trial_number"]].rename(
        columns={
            metric.column: "tuned_value",
            "pair_trial_number": "tuned_trial_number",
        }
    )

    merged = default_metric.merge(tuned_metric, on=join_keys, how="inner")
    if merged.empty:
        return pd.DataFrame(), selected, tuned_only, default_only

    merged["delta_tuned_minus_default"] = merged["tuned_value"] - merged["default_value"]
    denom = merged["default_value"].abs()
    merged["pct_improvement"] = np.where(
        denom > 0,
        (merged["default_value"] - merged["tuned_value"]) / denom * 100.0,
        np.nan,
    )
    if metric.higher_is_better:
        merged["pct_improvement"] = np.where(
            denom > 0,
            (merged["tuned_value"] - merged["default_value"]) / denom * 100.0,
            np.nan,
        )
    return merged, selected, tuned_only, default_only


def _paired_ttest_summary(
    default_values: np.ndarray,
    tuned_values: np.ndarray,
    higher_is_better: bool,
) -> Dict[str, float]:
    valid = np.isfinite(default_values) & np.isfinite(tuned_values)
    default_values = default_values[valid]
    tuned_values = tuned_values[valid]
    if default_values.size == 0:
        return {
            "pairs": 0,
            "mean_default": np.nan,
            "mean_tuned": np.nan,
            "mean_delta": np.nan,
            "std_delta": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "tuned_better_rate": np.nan,
        }

    deltas = tuned_values - default_values
    n_pairs = int(deltas.size)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n_pairs > 1 else np.nan

    t_stat = np.nan
    p_value = np.nan
    if n_pairs > 1 and np.isfinite(std_delta) and std_delta > 0:
        t_stat, p_value = stats.ttest_rel(tuned_values, default_values, nan_policy="omit")

    ci_low = np.nan
    ci_high = np.nan
    if n_pairs > 1:
        if np.isfinite(std_delta) and std_delta > 0:
            t_crit = stats.t.ppf(0.975, n_pairs - 1)
            margin = t_crit * std_delta / np.sqrt(n_pairs)
            ci_low = mean_delta - margin
            ci_high = mean_delta + margin
        else:
            ci_low = mean_delta
            ci_high = mean_delta

    if higher_is_better:
        tuned_better_rate = float(np.mean(tuned_values > default_values))
    else:
        tuned_better_rate = float(np.mean(tuned_values < default_values))

    return {
        "pairs": n_pairs,
        "mean_default": float(np.mean(default_values)),
        "mean_tuned": float(np.mean(tuned_values)),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "tuned_better_rate": tuned_better_rate,
    }


def _pct_improvement_summary(pct_values: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(pct_values)
    pct_values = pct_values[valid]
    if pct_values.size == 0:
        return {
            "pairs": 0,
            "median_pct": np.nan,
            "mean_pct": np.nan,
            "std_pct": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    n_pairs = int(pct_values.size)
    median_pct = float(np.median(pct_values))
    mean_pct = float(np.mean(pct_values))
    std_pct = float(np.std(pct_values, ddof=1)) if n_pairs > 1 else np.nan

    t_stat = np.nan
    p_value = np.nan
    if n_pairs > 1 and np.isfinite(std_pct) and std_pct > 0:
        t_stat, p_value = stats.ttest_1samp(pct_values, popmean=0.0, nan_policy="omit")

    ci_low = np.nan
    ci_high = np.nan
    if n_pairs > 1:
        if np.isfinite(std_pct) and std_pct > 0:
            t_crit = stats.t.ppf(0.975, n_pairs - 1)
            margin = t_crit * std_pct / np.sqrt(n_pairs)
            ci_low = mean_pct - margin
            ci_high = mean_pct + margin
        else:
            ci_low = mean_pct
            ci_high = mean_pct

    return {
        "pairs": n_pairs,
        "median_pct": median_pct,
        "mean_pct": mean_pct,
        "std_pct": std_pct,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
    }


def summarize_paired_value_table(
    pairs: pd.DataFrame,
    group_col: str,
    *,
    higher_is_better: bool,
    value_a_col: str = "tuned_value",
    value_b_col: str = "default_value",
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    required_cols = [group_col, value_a_col, value_b_col]
    missing = [column for column in required_cols if column not in pairs.columns]
    if missing:
        raise ValueError(
            "Paired-value summary missing required columns: "
            f"{missing}"
        )

    for group_value, subset in pairs.groupby(group_col, dropna=False):
        stats_dict = _paired_ttest_summary(
            subset[value_b_col].to_numpy(dtype=float),
            subset[value_a_col].to_numpy(dtype=float),
            higher_is_better=higher_is_better,
        )
        pct_stats = _pct_improvement_summary(
            subset["pct_improvement"].to_numpy(dtype=float)
        ) if "pct_improvement" in subset.columns else {
            "median_pct": np.nan,
            "mean_pct": np.nan,
            "std_pct": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
        record: Dict[str, object] = {
            group_col: group_value,
            "pairs": stats_dict["pairs"],
            "mean_default": stats_dict["mean_default"],
            "mean_tuned": stats_dict["mean_tuned"],
            "mean_delta": stats_dict["mean_delta"],
            "std_delta": stats_dict["std_delta"],
            "t_stat": stats_dict["t_stat"],
            "p_value": stats_dict["p_value"],
            "ci_low": stats_dict["ci_low"],
            "ci_high": stats_dict["ci_high"],
            "tuned_better_rate": stats_dict["tuned_better_rate"],
            "median_pct_improvement": pct_stats["median_pct"],
            "mean_pct_improvement": pct_stats["mean_pct"],
            "mean_pct": pct_stats["mean_pct"],
            "std_pct": pct_stats["std_pct"],
            "pct_t_stat": pct_stats["t_stat"],
            "pct_p_value": pct_stats["p_value"],
            "ci_low_pct": pct_stats["ci_low"],
            "ci_high_pct": pct_stats["ci_high"],
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def summarize_paired_value_global_row(
    pairs: pd.DataFrame,
    group_col: str,
    group_value: object,
    *,
    higher_is_better: bool,
    value_a_col: str = "tuned_value",
    value_b_col: str = "default_value",
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    global_pairs = pairs.copy()
    global_pairs[group_col] = group_value
    return summarize_paired_value_table(
        global_pairs,
        group_col,
        higher_is_better=higher_is_better,
        value_a_col=value_a_col,
        value_b_col=value_b_col,
    )


def _summary_table(
    pairs: pd.DataFrame,
    group_col: str,
    metric: TunedDefaultMetric,
) -> pd.DataFrame:
    return summarize_paired_value_table(
        pairs,
        group_col,
        higher_is_better=metric.higher_is_better,
        value_a_col="tuned_value",
        value_b_col="default_value",
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No qualifying rows._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def build_tuned_vs_default_report(
    df: pd.DataFrame,
    metrics: Sequence[TunedDefaultMetric],
    output_dir: Path,
    markdown_filename: str = "TUNED_VS_DEFAULT_PAIRED_TTEST.md",
    config: Optional[TunedDefaultConfig] = None,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build tuned-vs-default report – input dataframe is empty.")

    tables_dir = Path(output_dir) / "tables"
    summary_dir = Path(output_dir) / "summary_outputs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    markdown_sections: List[str] = []
    markdown_sections.append("# Tuned vs Default Paired t-test")
    markdown_sections.append("")
    markdown_sections.append(
        "Pairs default trials (lowest trial number) with tuned trials (best metric value) "
        "within matched dataset/topology/sampling/algorithm/seed groups."
    )
    markdown_sections.append("")

    for metric in metrics:
        pairs, selected = build_metric_pairs(df, metric, config=config)

        markdown_sections.append(f"## Metric: {metric.label}")
        markdown_sections.append("")
        if pairs.empty:
            markdown_sections.append("_No paired rows available for this metric._")
            markdown_sections.append("")
            continue

        dataset_summary = _summary_table(pairs, "pair_dataset", metric)
        dataset_summary = dataset_summary.rename(columns={"pair_dataset": "dataset"})
        ordered = ordered_dataset_labels(dataset_summary["dataset"].tolist())
        if ordered:
            dataset_summary["dataset"] = pd.Categorical(
                dataset_summary["dataset"], categories=ordered, ordered=True
            )
            dataset_summary = dataset_summary.sort_values("dataset")

        dataset_type = pairs["pair_dataset"].map(dataset_group)
        pairs_with_type = pairs.assign(dataset_type=dataset_type)
        dataset_type_summary = _summary_table(pairs_with_type, "dataset_type", metric)

        global_summary = summarize_paired_value_global_row(
            pairs,
            "global_label",
            "GLOBAL",
            higher_is_better=metric.higher_is_better,
            value_a_col="tuned_value",
            value_b_col="default_value",
        )

        pairs_path = tables_dir / f"tuned_vs_default_pairs_{metric.slug}.csv"
        dataset_path = tables_dir / f"tuned_vs_default_dataset_summary_{metric.slug}.csv"
        dataset_type_path = tables_dir / f"tuned_vs_default_dataset_type_summary_{metric.slug}.csv"
        global_path = tables_dir / f"tuned_vs_default_global_summary_{metric.slug}.csv"

        _write_csv(pairs, pairs_path)
        _write_csv(dataset_summary, dataset_path)
        _write_csv(dataset_type_summary, dataset_type_path)
        _write_csv(global_summary, global_path)

        markdown_sections.append("### Pairing Columns")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(pd.DataFrame([
            {
                "dataset": selected.get("dataset"),
                "processing": selected.get("processing"),
                "sampling": selected.get("sampling"),
                "topology": selected.get("topology"),
                "seed": selected.get("seed"),
                "trial_number": selected.get("trial_number"),
                "split": selected.get("split"),
            }
        ])))
        markdown_sections.append("")

        markdown_sections.append("### Dataset Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(dataset_summary))
        markdown_sections.append("")

        markdown_sections.append("### Dataset-Type Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(dataset_type_summary))
        markdown_sections.append("")

        markdown_sections.append("### Global Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(global_summary))
        markdown_sections.append("")

    markdown_path = summary_dir / markdown_filename
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")
    return markdown_path


def build_tuned_vs_external_default_report(
    tuned_df: pd.DataFrame,
    default_df: pd.DataFrame,
    metrics: Sequence[TunedDefaultMetric],
    output_dir: Path,
    markdown_filename: str = "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST.md",
    split_policy: str = "both",
    config: Optional[TunedDefaultConfig] = None,
) -> Path:
    if tuned_df.empty:
        raise ValueError("Cannot build tuned-vs-default report – tuned dataframe is empty.")
    if default_df.empty:
        raise ValueError("Cannot build tuned-vs-default report – default dataframe is empty.")

    tables_dir = Path(output_dir) / "tables"
    diagnostics_dir = Path(output_dir) / "diagnostics"
    summary_dir = Path(output_dir) / "summary_outputs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    markdown_sections: List[str] = []
    markdown_sections.append("# Tuned vs True Default (Paired t-test)")
    markdown_sections.append("")
    markdown_sections.append(
        "Pairs tuned top-1 trials against explicit true-default runs within matched "
        "dataset/topology/sampling/batch_mode/seed keys."
    )
    markdown_sections.append("")
    markdown_sections.append(f"Split policy: `{split_policy}`")
    markdown_sections.append("")

    for metric in metrics:
        pairs, selected, tuned_only, default_only = build_metric_pairs_against_defaults(
            tuned_df=tuned_df,
            default_df=default_df,
            metric=metric,
            config=config,
            split_policy=split_policy,
        )
        markdown_sections.append(f"## Metric: {metric.label}")
        markdown_sections.append("")
        if pairs.empty:
            markdown_sections.append("_No paired rows available for this metric._")
            markdown_sections.append("")
            continue

        dataset_summary = _summary_table(pairs, "pair_dataset", metric).rename(columns={"pair_dataset": "dataset"})
        ordered = ordered_dataset_labels(dataset_summary["dataset"].tolist())
        if ordered:
            dataset_summary["dataset"] = pd.Categorical(
                dataset_summary["dataset"], categories=ordered, ordered=True
            )
            dataset_summary = dataset_summary.sort_values("dataset")

        dataset_type = pairs["pair_dataset"].map(dataset_group)
        dataset_type_summary = _summary_table(pairs.assign(dataset_type=dataset_type), "dataset_type", metric)
        global_summary = summarize_paired_value_global_row(
            pairs,
            "global_label",
            "GLOBAL",
            higher_is_better=metric.higher_is_better,
            value_a_col="tuned_value",
            value_b_col="default_value",
        )

        pairs_path = tables_dir / f"tuned_vs_default_pairs_{metric.slug}.csv"
        dataset_path = tables_dir / f"tuned_vs_default_dataset_summary_{metric.slug}.csv"
        dataset_type_path = tables_dir / f"tuned_vs_default_dataset_type_summary_{metric.slug}.csv"
        global_path = tables_dir / f"tuned_vs_default_global_summary_{metric.slug}.csv"
        tuned_only_path = diagnostics_dir / f"tuned_only_keys_{metric.slug}.csv"
        default_only_path = diagnostics_dir / f"default_only_keys_{metric.slug}.csv"

        _write_csv(pairs, pairs_path)
        _write_csv(dataset_summary, dataset_path)
        _write_csv(dataset_type_summary, dataset_type_path)
        _write_csv(global_summary, global_path)
        _write_csv(tuned_only, tuned_only_path)
        _write_csv(default_only, default_only_path)

        markdown_sections.append("### Pairing Columns")
        markdown_sections.append("")
        markdown_sections.append(
            _dataframe_to_markdown(
                pd.DataFrame(
                    [
                        {
                            "dataset": selected.get("dataset"),
                            "processing": selected.get("processing"),
                            "sampling": selected.get("sampling"),
                            "batch_mode": selected.get("batch_mode"),
                            "topology": selected.get("topology"),
                            "seed": selected.get("seed"),
                            "trial_number": selected.get("trial_number"),
                            "split": selected.get("split"),
                        }
                    ]
                )
            )
        )
        markdown_sections.append("")
        markdown_sections.append(
            f"Dropped keys (tuned-only): {int(len(tuned_only))}, dropped keys (default-only): {int(len(default_only))}"
        )
        markdown_sections.append("")

        markdown_sections.append("### Dataset Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(dataset_summary))
        markdown_sections.append("")

        markdown_sections.append("### Dataset-Type Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(dataset_type_summary))
        markdown_sections.append("")

        markdown_sections.append("### Global Summary")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(global_summary))
        markdown_sections.append("")

    markdown_path = summary_dir / markdown_filename
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")
    return markdown_path


def build_tuned_vs_external_default_report_from_csv(
    tuned_df: pd.DataFrame,
    default_csv_path: str | Path,
    metrics: Sequence[TunedDefaultMetric],
    output_dir: Path,
    markdown_filename: str = "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST.md",
    split_policy: str = "both",
    config: Optional[TunedDefaultConfig] = None,
) -> Path:
    default_df = load_default_runs_csv(default_csv_path, split_policy=split_policy)
    return build_tuned_vs_external_default_report(
        tuned_df=tuned_df,
        default_df=default_df,
        metrics=metrics,
        output_dir=output_dir,
        markdown_filename=markdown_filename,
        split_policy=split_policy,
        config=config,
    )
