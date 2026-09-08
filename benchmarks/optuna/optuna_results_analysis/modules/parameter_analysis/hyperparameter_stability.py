"""
Hyperparameter stability analysis across topology pairs (hex vs MST/RNG).

Uses seed-preserving Optuna exports to select tuned trials per seed, then
compares tuned parameter values between topology pairs within matched
dataset/sampling/algorithm/seed keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis.dataset_groups import dataset_group, ordered_dataset_labels


@dataclass(frozen=True)
class StabilityMetric:
    """Metric configuration for selecting tuned trials."""

    column: str
    label: str
    slug: str
    higher_is_better: bool = False


@dataclass(frozen=True)
class StabilityConfig:
    """Column resolution config for hyperparameter stability analysis."""

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
    split_col_candidates: Tuple[str, ...] = (
        "evaluation_split",
        "config_evaluation_split",
        "split",
    )
    parameter_prefix: str = "param_"


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _resolve_pairing_columns(
    df: pd.DataFrame,
    config: StabilityConfig,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Optional[str]]]:
    dataset_col = _resolve_column(df, config.dataset_col_candidates)
    processing_col = _resolve_column(df, config.processing_col_candidates)
    sampling_col = _resolve_column(df, config.sampling_col_candidates)
    topology_col = _resolve_column(df, config.topology_col_candidates)
    seed_col = _resolve_column(df, config.seed_col_candidates)
    trial_col = _resolve_column(df, config.trial_number_col_candidates)
    split_col = _resolve_column(df, config.split_col_candidates)

    missing = [
        name
        for name, value in (
            ("dataset", dataset_col),
            ("processing", processing_col),
            ("sampling", sampling_col),
            ("topology", topology_col),
            ("seed", seed_col),
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

    if trial_col:
        working["pair_trial_number"] = pd.to_numeric(working[trial_col], errors="coerce")

    if split_col:
        working["pair_split"] = _normalize_text(working[split_col])

    key_cols = [
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
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
        "split": split_col,
    }
    return working, key_cols, selected


def _detect_parameter_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null == 0:
        return False
    return coerced.notna().sum() == non_null


def _split_param_columns(df: pd.DataFrame, param_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for column in param_columns:
        if column not in df.columns:
            continue
        series = df[column]
        if _is_numeric_column(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def _select_best_trials(
    df: pd.DataFrame,
    metric: StabilityMetric,
    key_cols: Sequence[str],
) -> pd.DataFrame:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' not found in dataframe.")

    base = df.dropna(subset=[metric.column]).copy()
    if base.empty:
        return pd.DataFrame()

    ascending = not metric.higher_is_better
    order_cols = [metric.column]
    if "pair_trial_number" in base.columns:
        order_cols.append("pair_trial_number")

    base = base.sort_values(order_cols, ascending=[ascending] * len(order_cols))
    best_rows = base.groupby(list(key_cols) + ["pair_topology"], dropna=False).head(1)
    return best_rows


def _compute_param_differences(
    pairs: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    epsilon = 1e-12
    numeric_abs_frames: List[pd.Series] = []
    numeric_rel_frames: List[pd.Series] = []
    numeric_counts: List[pd.Series] = []
    details: Dict[str, pd.Series] = {}

    for column in numeric_cols:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        if col_a not in pairs.columns or col_b not in pairs.columns:
            continue
        a = pd.to_numeric(pairs[col_a], errors="coerce")
        b = pd.to_numeric(pairs[col_b], errors="coerce")
        valid = a.notna() & b.notna()
        diff = (a - b).abs()
        delta = a - b
        denom = np.maximum(np.maximum(a.abs(), b.abs()), epsilon)
        rel = diff / denom
        diff[~valid] = np.nan
        delta[~valid] = np.nan
        rel[~valid] = np.nan
        numeric_abs_frames.append(diff)
        numeric_rel_frames.append(rel)
        numeric_counts.append(valid.astype(float))
        details[f"{column}_delta_left_minus_right"] = delta
        details[f"{column}_abs_diff"] = diff
        details[f"{column}_rel_diff"] = rel

    categorical_frames: List[pd.Series] = []
    categorical_counts: List[pd.Series] = []
    for column in categorical_cols:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        if col_a not in pairs.columns or col_b not in pairs.columns:
            continue
        a = pairs[col_a]
        b = pairs[col_b]
        valid = a.notna() & b.notna()
        mismatch = pd.Series(np.nan, index=pairs.index)
        mismatch.loc[valid] = (a[valid].astype(str) != b[valid].astype(str)).astype(float)
        categorical_frames.append(mismatch)
        categorical_counts.append(valid.astype(float))
        details[f"{column}_mismatch"] = mismatch

    numeric_abs_mean = (
        pd.concat(numeric_abs_frames, axis=1).mean(axis=1, skipna=True)
        if numeric_abs_frames
        else pd.Series(np.nan, index=pairs.index)
    )
    numeric_rel_mean = (
        pd.concat(numeric_rel_frames, axis=1).mean(axis=1, skipna=True)
        if numeric_rel_frames
        else pd.Series(np.nan, index=pairs.index)
    )
    categorical_mismatch = (
        pd.concat(categorical_frames, axis=1).mean(axis=1, skipna=True)
        if categorical_frames
        else pd.Series(np.nan, index=pairs.index)
    )

    numeric_used = (
        pd.concat(numeric_counts, axis=1).sum(axis=1, skipna=True)
        if numeric_counts
        else pd.Series(0.0, index=pairs.index)
    )
    categorical_used = (
        pd.concat(categorical_counts, axis=1).sum(axis=1, skipna=True)
        if categorical_counts
        else pd.Series(0.0, index=pairs.index)
    )

    data: Dict[str, pd.Series] = {
        "numeric_stability_abs_diff_mean": numeric_abs_mean,
        "numeric_stability_rel_diff_mean": numeric_rel_mean,
        "categorical_stability_mismatch_rate": categorical_mismatch,
        "numeric_params_compared": numeric_used,
        "categorical_params_compared": categorical_used,
    }
    data.update(details)
    return pd.DataFrame(data)


def build_topology_param_pairs(
    best_df: pd.DataFrame,
    key_cols: Sequence[str],
    topology_pair: Tuple[str, str],
    param_columns: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    topology_a, topology_b = topology_pair
    df_a = best_df[best_df["pair_topology"] == topology_a].copy()
    df_b = best_df[best_df["pair_topology"] == topology_b].copy()
    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    pairs = df_a.merge(df_b, on=list(key_cols), suffixes=("_a", "_b"), how="inner")
    if pairs.empty:
        return pd.DataFrame()

    for column in param_columns:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        if col_a not in pairs.columns:
            pairs[col_a] = np.nan
        if col_b not in pairs.columns:
            pairs[col_b] = np.nan

    diffs = _compute_param_differences(pairs, numeric_cols, categorical_cols)
    result = pd.concat([pairs, diffs], axis=1)
    result["topology_a"] = topology_a
    result["topology_b"] = topology_b
    result["pair_label"] = f"{topology_a}_vs_{topology_b}"
    return result


def build_within_topology_param_pairs(
    best_df: pd.DataFrame,
    key_cols: Sequence[str],
    topology: str,
    param_columns: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    topology_rows = best_df[best_df["pair_topology"] == topology].copy()
    if topology_rows.empty:
        return pd.DataFrame()

    key_without_seed = [column for column in key_cols if column != "pair_seed"]
    if not key_without_seed:
        return pd.DataFrame()

    pairs = topology_rows.merge(topology_rows, on=list(key_without_seed), suffixes=("_a", "_b"), how="inner")
    if pairs.empty:
        return pd.DataFrame()

    if "pair_seed_a" not in pairs.columns or "pair_seed_b" not in pairs.columns:
        return pd.DataFrame()

    seed_a = pairs["pair_seed_a"].astype(str)
    seed_b = pairs["pair_seed_b"].astype(str)
    pairs = pairs[seed_a < seed_b].copy()
    if pairs.empty:
        return pd.DataFrame()

    for column in param_columns:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        if col_a not in pairs.columns:
            pairs[col_a] = np.nan
        if col_b not in pairs.columns:
            pairs[col_b] = np.nan

    diffs = _compute_param_differences(pairs, numeric_cols, categorical_cols)
    result = pd.concat([pairs, diffs], axis=1)
    result["pair_topology"] = topology
    result["pair_label"] = f"{topology}_within_topology_seed_pairs"
    return result


def build_topology_stability_pairs(
    best_df: pd.DataFrame,
    key_cols: Sequence[str],
    topologies: Sequence[str],
    param_columns: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for topology in topologies:
        frame = build_within_topology_param_pairs(
            best_df=best_df,
            key_cols=key_cols,
            topology=topology,
            param_columns=param_columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _lower_better_winner(
    left_value: object,
    right_value: object,
    *,
    left_label: str,
    right_label: str,
) -> str:
    if pd.isna(left_value) or pd.isna(right_value):
        return "insufficient_data"

    left_float = float(left_value)
    right_float = float(right_value)
    if np.isclose(left_float, right_float, rtol=1e-9, atol=1e-12):
        return "tie"
    return left_label if left_float < right_float else right_label


def _parameter_topology_stability_table(
    topology_stability_pairs: pd.DataFrame,
    group_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    """Per-parameter internal stability per topology and grouping scope."""
    if topology_stability_pairs.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = topology_stability_pairs.groupby([group_col, "pair_topology"], dropna=False)
    for (group_value, topology), subset in grouped:
        for column in numeric_cols:
            rel_col = f"{column}_rel_diff"
            if rel_col not in subset.columns:
                continue
            rel = pd.to_numeric(subset[rel_col], errors="coerce")
            valid = rel.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    group_col: group_value,
                    "topology": topology,
                    "parameter": column,
                    "parameter_type": "numeric",
                    "seed_pairs": int(valid.sum()),
                    "stability_score_mean": float(rel[valid].mean()),
                    "stability_score_median": float(rel[valid].median()),
                }
            )

        for column in categorical_cols:
            mismatch_col = f"{column}_mismatch"
            if mismatch_col not in subset.columns:
                continue
            mismatch = pd.to_numeric(subset[mismatch_col], errors="coerce")
            valid = mismatch.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    group_col: group_value,
                    "topology": topology,
                    "parameter": column,
                    "parameter_type": "categorical",
                    "seed_pairs": int(valid.sum()),
                    "stability_score_mean": float(mismatch[valid].mean()),
                    "stability_score_median": float(mismatch[valid].median()),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _parameter_stability_comparison_table(
    parameter_topology_stability: pd.DataFrame,
    group_col: str,
    *,
    topology_left: str,
    topology_right: str,
) -> pd.DataFrame:
    """Compare per-parameter internal-stability scores between two topologies.

    Lower stability score is better (more stable).
    """
    if parameter_topology_stability.empty:
        return pd.DataFrame()

    select_cols = [
        group_col,
        "topology",
        "parameter",
        "parameter_type",
        "seed_pairs",
        "stability_score_mean",
        "stability_score_median",
    ]
    available = [col for col in select_cols if col in parameter_topology_stability.columns]
    if not {"topology", "parameter", "stability_score_mean"}.issubset(set(available)):
        return pd.DataFrame()

    left = parameter_topology_stability[parameter_topology_stability["topology"] == topology_left][available].copy()
    right = parameter_topology_stability[parameter_topology_stability["topology"] == topology_right][available].copy()
    if left.empty or right.empty:
        return pd.DataFrame()

    left = left.rename(
        columns={
            "seed_pairs": "seed_pairs_left",
            "stability_score_mean": "stability_score_mean_left",
            "stability_score_median": "stability_score_median_left",
        }
    ).drop(columns=["topology"], errors="ignore")
    right = right.rename(
        columns={
            "seed_pairs": "seed_pairs_right",
            "stability_score_mean": "stability_score_mean_right",
            "stability_score_median": "stability_score_median_right",
        }
    ).drop(columns=["topology"], errors="ignore")

    join_cols = [column for column in [group_col, "parameter", "parameter_type"] if column in left.columns and column in right.columns]
    if len(join_cols) < 2:
        return pd.DataFrame()

    merged = left.merge(right, on=join_cols, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged.insert(1, "topology_left", topology_left)
    merged.insert(2, "topology_right", topology_right)
    merged["delta_left_minus_right"] = (
        pd.to_numeric(merged["stability_score_mean_left"], errors="coerce")
        - pd.to_numeric(merged["stability_score_mean_right"], errors="coerce")
    )
    merged["winner"] = merged.apply(
        lambda row: _lower_better_winner(
            row.get("stability_score_mean_left"),
            row.get("stability_score_mean_right"),
            left_label=topology_left,
            right_label=topology_right,
        ),
        axis=1,
    )

    overall_rows: List[Dict[str, object]] = []
    for group_value, subset in merged.groupby(group_col, dropna=False):
        scored = subset.dropna(subset=["stability_score_mean_left", "stability_score_mean_right"])
        if scored.empty:
            continue
        left_mean = float(pd.to_numeric(scored["stability_score_mean_left"], errors="coerce").mean())
        right_mean = float(pd.to_numeric(scored["stability_score_mean_right"], errors="coerce").mean())
        overall_rows.append(
            {
                group_col: group_value,
                "topology_left": topology_left,
                "topology_right": topology_right,
                "parameter": "OVERALL",
                "parameter_type": "overall",
                "seed_pairs_left": float(pd.to_numeric(scored["seed_pairs_left"], errors="coerce").mean()),
                "seed_pairs_right": float(pd.to_numeric(scored["seed_pairs_right"], errors="coerce").mean()),
                "stability_score_mean_left": left_mean,
                "stability_score_mean_right": right_mean,
                "stability_score_median_left": float(
                    pd.to_numeric(scored["stability_score_median_left"], errors="coerce").mean()
                ),
                "stability_score_median_right": float(
                    pd.to_numeric(scored["stability_score_median_right"], errors="coerce").mean()
                ),
                "delta_left_minus_right": left_mean - right_mean,
                "winner": _lower_better_winner(
                    left_mean,
                    right_mean,
                    left_label=topology_left,
                    right_label=topology_right,
                ),
                "parameters_compared": int(scored["parameter"].nunique()),
            }
        )

    if overall_rows:
        merged["parameters_compared"] = np.nan
        merged = pd.concat([merged, pd.DataFrame(overall_rows)], ignore_index=True)

    return merged


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def stability_table_filename(scope: str, topology_a: str, topology_b: str, metric_slug: str) -> str:
    scope_map = {
        "dataset": "ds",
        "dataset_type": "dst",
        "global": "g",
    }
    scope_token = scope_map.get(str(scope).strip().lower())
    if scope_token is None:
        raise ValueError(f"Unsupported stability table scope: {scope!r}")
    return f"hps_{scope_token}_{topology_a}vs{topology_b}_{metric_slug}.csv"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No qualifying rows._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def build_hyperparameter_stability_report(
    df: pd.DataFrame,
    metric: StabilityMetric,
    output_dir: Path,
    topology_pairs: Sequence[Tuple[str, str]] = (("hexagonal", "mst"), ("hexagonal", "rng")),
    markdown_filename: str = "HYPERPARAMETER_STABILITY.md",
    config: Optional[StabilityConfig] = None,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build hyperparameter stability report – input dataframe is empty.")

    resolved_config = config or StabilityConfig()
    working, key_cols, selected = _resolve_pairing_columns(df, resolved_config)

    param_columns = _detect_parameter_columns(working, resolved_config.parameter_prefix)
    if not param_columns:
        raise ValueError("No parameter columns detected – expected columns prefixed with 'param_'.")

    numeric_cols, categorical_cols = _split_param_columns(working, param_columns)

    best_rows = _select_best_trials(working, metric, key_cols)
    if best_rows.empty:
        raise ValueError("No tuned trials available after filtering for metric values.")

    tables_dir = Path(output_dir) / "tables"
    summary_dir = Path(output_dir) / "summary_outputs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    markdown_sections: List[str] = []
    markdown_sections.append("# Hyperparameter Stability (Topology Pairs)")
    markdown_sections.append("")
    markdown_sections.append(
        "Compares per-parameter internal seed-to-seed stability between topology pairs "
        "within matched dataset/sampling/algorithm groups. Lower scores are better. "
        "Winner and delta are computed from mean stability scores."
    )
    markdown_sections.append("")
    markdown_sections.append("## Selection Metric")
    markdown_sections.append("")
    markdown_sections.append(
        _dataframe_to_markdown(
            pd.DataFrame(
                [
                    {
                        "metric": metric.column,
                        "label": metric.label,
                        "higher_is_better": metric.higher_is_better,
                    }
                ]
            )
        )
    )
    markdown_sections.append("")
    markdown_sections.append("## Pairing Columns")
    markdown_sections.append("")
    markdown_sections.append(
        _dataframe_to_markdown(
            pd.DataFrame(
                [
                    {
                        "dataset": selected.get("dataset"),
                        "processing": selected.get("processing"),
                        "sampling": selected.get("sampling"),
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

    for topology_pair in topology_pairs:
        topology_a, topology_b = topology_pair
        pair_label = f"{topology_a} vs {topology_b}"
        markdown_sections.append(f"## Pair: {pair_label}")
        markdown_sections.append("")

        pairs = build_topology_param_pairs(
            best_df=best_rows,
            key_cols=key_cols,
            topology_pair=(topology_a, topology_b),
            param_columns=param_columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        if pairs.empty:
            markdown_sections.append("_No matched pairs available for this topology comparison._")
            markdown_sections.append("")
            continue

        topology_stability_pairs = build_topology_stability_pairs(
            best_df=best_rows,
            key_cols=key_cols,
            topologies=[topology_a, topology_b],
            param_columns=param_columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        parameter_topology_dataset = _parameter_topology_stability_table(
            topology_stability_pairs=topology_stability_pairs,
            group_col="pair_dataset",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        parameter_dataset_summary = _parameter_stability_comparison_table(
            parameter_topology_stability=parameter_topology_dataset,
            group_col="pair_dataset",
            topology_left=topology_a,
            topology_right=topology_b,
        ).rename(columns={"pair_dataset": "dataset"})
        ordered: List[str] = []
        if "dataset" in parameter_dataset_summary.columns:
            ordered = ordered_dataset_labels(parameter_dataset_summary["dataset"].tolist())
        if not parameter_dataset_summary.empty and ordered:
            parameter_dataset_summary["dataset"] = pd.Categorical(
                parameter_dataset_summary["dataset"], categories=ordered, ordered=True
            )
            parameter_dataset_summary = parameter_dataset_summary.sort_values(
                ["dataset", "parameter_type", "parameter"]
            )

        topology_param_dataset_type = topology_stability_pairs["pair_dataset"].map(dataset_group)
        parameter_topology_dataset_type = _parameter_topology_stability_table(
            topology_stability_pairs=topology_stability_pairs.assign(dataset_type=topology_param_dataset_type),
            group_col="dataset_type",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        parameter_dataset_type_summary = _parameter_stability_comparison_table(
            parameter_topology_stability=parameter_topology_dataset_type,
            group_col="dataset_type",
            topology_left=topology_a,
            topology_right=topology_b,
        )
        parameter_topology_global = _parameter_topology_stability_table(
            topology_stability_pairs=topology_stability_pairs.assign(global_label="GLOBAL"),
            group_col="global_label",
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        parameter_global_summary = _parameter_stability_comparison_table(
            parameter_topology_stability=parameter_topology_global,
            group_col="global_label",
            topology_left=topology_a,
            topology_right=topology_b,
        )

        parameter_dataset_path = tables_dir / stability_table_filename(
            "dataset", topology_a, topology_b, metric.slug
        )
        parameter_dataset_type_path = tables_dir / stability_table_filename(
            "dataset_type", topology_a, topology_b, metric.slug
        )
        parameter_global_path = tables_dir / stability_table_filename(
            "global", topology_a, topology_b, metric.slug
        )
        _write_csv(parameter_dataset_summary, parameter_dataset_path)
        _write_csv(parameter_dataset_type_summary, parameter_dataset_type_path)
        _write_csv(parameter_global_summary, parameter_global_path)

        markdown_sections.append("### Per-Parameter Stability Comparison (Dataset, lower is better)")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(parameter_dataset_summary))
        markdown_sections.append("")

        markdown_sections.append("### Per-Parameter Stability Comparison (Dataset Type, lower is better)")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(parameter_dataset_type_summary))
        markdown_sections.append("")

        markdown_sections.append("### Per-Parameter Stability Comparison (Global, lower is better)")
        markdown_sections.append("")
        markdown_sections.append(_dataframe_to_markdown(parameter_global_summary))
        markdown_sections.append("")

    markdown_path = summary_dir / markdown_filename
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")
    return markdown_path
