"""
Paired analysis utilities for algorithm and topology comparisons.

This module creates matched units where only the comparison dimension varies,
computes paired deltas per metric, and writes dedicated markdown reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups


METRIC_LABELS: Dict[str, str] = {
    "quantization_error_holdout_normalized": "QE Holdout (Normalized)",
    "quantization_error_train_normalized": "QE Train (Normalized)",
    "balanced_qe_normalized": "Balanced QE (Normalized)",
    "Normalized_Overall_Score": "Overall Score (Normalized)",
}


@dataclass(frozen=True)
class PairingConfig:
    """Configuration for a paired comparison report."""

    report_title: str = "Algorithm Comparison Paired: Batch vs Colours"
    comparison_values: Tuple[str, str] = ("batch", "colors")
    comparison_labels: Tuple[str, str] = ("Batch", "Colours")
    comparison_column_candidates: Tuple[str, ...] = ("algorithm", "processing_type")
    optional_key_column_candidates: Tuple[Tuple[str, ...], ...] = (
        ("architecture", "map_type", "config_topology_type"),
        ("sampling_method_parsed", "sampling_method_final", "sampling_method"),
        ("evaluation_split", "config_evaluation_split", "split"),
        ("seed", "config_seed", "random_seed", "seed_name", "config_seed_name"),
    )
    metric_columns: Tuple[str, ...] = (
        "quantization_error_holdout_normalized",
        "quantization_error_train_normalized",
        "balanced_qe_normalized",
        "Normalized_Overall_Score",
    )
    dataset_column_candidates: Tuple[str, ...] = ("dataset",)
    top_k: int = 5
    # Deprecated compatibility knob retained for caller stability.
    bootstrap_iterations: int = 2000
    random_seed: int = 42
    max_detail_rows: int = 10


def _algorithm_pairing_config() -> PairingConfig:
    return PairingConfig(
        report_title="Algorithm Comparison Paired: Batch vs Colours",
        comparison_values=("batch", "colors"),
        comparison_labels=("Batch", "Colours"),
        comparison_column_candidates=("algorithm", "processing_type"),
        optional_key_column_candidates=(
            ("architecture", "map_type", "config_topology_type"),
            ("sampling_method_parsed", "sampling_method_final", "sampling_method"),
            ("evaluation_split", "config_evaluation_split", "split"),
            ("seed", "config_seed", "random_seed", "seed_name", "config_seed_name"),
        ),
        metric_columns=(
            "quantization_error_holdout_normalized",
            "quantization_error_train_normalized",
            "balanced_qe_normalized",
            "Normalized_Overall_Score",
        ),
    )


def _topology_pairing_config() -> PairingConfig:
    return PairingConfig(
        report_title="Topology Comparison Paired: MST vs Hexagonal",
        comparison_values=("mst", "hexagonal"),
        comparison_labels=("MST", "Hexagonal"),
        comparison_column_candidates=("architecture", "map_type", "config_topology_type"),
        optional_key_column_candidates=(
            ("algorithm", "processing_type"),
            ("sampling_method_parsed", "sampling_method_final", "sampling_method"),
            ("evaluation_split", "config_evaluation_split", "split"),
            ("seed", "config_seed", "random_seed", "seed_name", "config_seed_name"),
        ),
        metric_columns=(
            "quantization_error_holdout_normalized",
            "quantization_error_train_normalized",
            "balanced_qe_normalized",
        ),
    )


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    try:
        # Preserve pre-formatted string values (e.g., small deltas/p-values)
        # instead of re-parsing them as floats in tabulate.
        return df.to_markdown(index=False, disable_numparse=True)
    except ImportError:
        return df.to_string(index=False)


def _wilcoxon_two_sided_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                nonzero,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                mode="auto",
            ).pvalue
        )
    except ValueError:
        return float("nan")


@lru_cache(maxsize=128)
def _wilcoxon_rank_sum_cdf(n_nonzero: int) -> np.ndarray:
    """Exact CDF of one-sample Wilcoxon W+ for ranks 1..n_nonzero."""
    max_rank_sum = int(n_nonzero * (n_nonzero + 1) // 2)
    probs = np.zeros(max_rank_sum + 1, dtype=float)
    probs[0] = 1.0
    for rank in range(1, int(n_nonzero) + 1):
        updated = np.zeros_like(probs)
        updated += 0.5 * probs
        updated[rank:] += 0.5 * probs[:-rank]
        probs = updated
    return np.cumsum(probs)


@lru_cache(maxsize=256)
def _wilcoxon_two_sided_cutoff(n_nonzero: int, alpha: float) -> int:
    """Largest lower-tail cutoff c with two-sided size <= alpha."""
    if n_nonzero <= 0 or alpha <= 0.0:
        return -1
    max_rank_sum = int(n_nonzero * (n_nonzero + 1) // 2)
    half_rank_sum = int(max_rank_sum // 2)
    tail_alpha = float(alpha) / 2.0
    strict_threshold = np.nextafter(tail_alpha, float("-inf"))
    cdf = _wilcoxon_rank_sum_cdf(int(n_nonzero))
    accepted = np.flatnonzero(cdf[: half_rank_sum + 1] <= strict_threshold)
    if accepted.size == 0:
        return -1
    return int(accepted[-1])


def _walsh_averages(nonzero: np.ndarray) -> np.ndarray:
    pairwise = (nonzero[:, None] + nonzero[None, :]) / 2.0
    return pairwise[np.triu_indices(nonzero.size)].astype(float, copy=False)


def _wilcoxon_location_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")

    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 0.0, 0.0, 0.0
    if nonzero.size == 1:
        point = float(nonzero[0])
        return point, point, point

    walsh = _walsh_averages(nonzero)
    count = int(walsh.size)
    cutoff = _wilcoxon_two_sided_cutoff(int(nonzero.size), float(alpha))
    if cutoff < 0:
        lower_idx = 0
        upper_idx = count - 1
    else:
        lower_idx = min(max(cutoff, 0), count - 1)
        upper_idx = min(max(count - cutoff - 1, lower_idx), count - 1)

    middle = count // 2
    if count % 2 == 1:
        partition_idx = sorted({lower_idx, upper_idx, middle})
        selected = np.partition(walsh, partition_idx)
        location = float(selected[middle])
    else:
        middle_low = middle - 1
        partition_idx = sorted({lower_idx, upper_idx, middle_low, middle})
        selected = np.partition(walsh, partition_idx)
        location = float((selected[middle_low] + selected[middle]) / 2.0)

    ci_low = float(selected[lower_idx])
    ci_high = float(selected[upper_idx])
    return location, ci_low, ci_high


def _compute_delta_stats(
    deltas: np.ndarray,
    bootstrap_iterations: int,
    seed: int,
) -> Dict[str, float]:
    # Retained for API compatibility with existing call sites.
    _ = bootstrap_iterations, seed
    finite = deltas[np.isfinite(deltas)]
    n_pairs = int(finite.size)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "group_a_wins": 0,
            "group_b_wins": 0,
            "ties": 0,
            "group_a_win_rate": float("nan"),
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "median_ci_low": float("nan"),
            "median_ci_high": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "effect_size_signed": float("nan"),
        }

    group_a_wins = int(np.sum(finite < 0))
    group_b_wins = int(np.sum(finite > 0))
    ties = int(n_pairs - group_a_wins - group_b_wins)

    wilcoxon_p_value = _wilcoxon_two_sided_pvalue(finite)

    sign_denominator = group_a_wins + group_b_wins
    effect_size_signed = (
        float(group_a_wins - group_b_wins) / float(sign_denominator)
        if sign_denominator > 0
        else float("nan")
    )

    location_shift, ci_low, ci_high = _wilcoxon_location_ci(finite)

    return {
        "n_pairs": n_pairs,
        "group_a_wins": group_a_wins,
        "group_b_wins": group_b_wins,
        "ties": ties,
        "group_a_win_rate": float(group_a_wins) / float(n_pairs) if n_pairs > 0 else float("nan"),
        "median_delta": location_shift,
        "mean_delta": float(np.mean(finite)),
        "median_ci_low": ci_low,
        "median_ci_high": ci_high,
        "wilcoxon_p_value": wilcoxon_p_value,
        "effect_size_signed": effect_size_signed,
    }


def _format_number(
    value: float,
    digits: int = 4,
    scientific: bool = False,
) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)

    # For tiny non-zero values, prefer scientific notation so signal is visible.
    if scientific and number != 0.0 and abs(number) < (10.0 ** (-digits)):
        return f"{number:.2e}"

    text = f"{number:.{digits}f}"
    # Avoid displaying negative zero after rounding.
    if text.startswith("-0.") and float(text) == 0.0:
        return text[1:]
    return text


def _metric_label(metric_col: str) -> str:
    return METRIC_LABELS.get(metric_col, metric_col.replace("_", " ").title())


def _is_qe_metric(metric_col: str) -> bool:
    return metric_col.startswith("quantization_error") or metric_col.startswith("balanced_qe")


def _slug(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _order_dataset_summary_rows(df: pd.DataFrame, dataset_label_col: str) -> pd.DataFrame:
    if df.empty or dataset_label_col not in df.columns:
        return df
    order = dataset_groups.ordered_dataset_labels(df[dataset_label_col].astype(str).tolist())
    order_map = {label: idx for idx, label in enumerate(order)}
    ordered = df.copy()
    ordered["__order__"] = ordered[dataset_label_col].map(order_map)
    ordered = ordered.sort_values("__order__").drop(columns="__order__")
    return ordered


def _append_group_summary_rows(
    dataset_summary_df: pd.DataFrame,
    dataset_label_col: str,
    metric_is_qe: bool,
    win_rate_col: str,
    delta_col: str,
    effect_col: str,
    pct_pair_col: str,
    pct_worse_col: str,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    if dataset_summary_df.empty or dataset_label_col not in dataset_summary_df.columns:
        return dataset_summary_df

    working = dataset_summary_df.copy()
    working["__dataset_group__"] = working[dataset_label_col].map(dataset_groups.dataset_group)

    rows: List[Dict[str, float | str]] = []
    for group_key in dataset_groups.SUMMARY_GROUPS:
        if group_key == "overall":
            group_df = working
        else:
            group_df = working[working["__dataset_group__"] == group_key]
        if group_df.empty:
            continue

        delta_values = pd.to_numeric(group_df[delta_col], errors="coerce").to_numpy(dtype=float)
        stats_dict = _compute_delta_stats(
            delta_values,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        if stats_dict["n_pairs"] <= 0:
            continue

        row: Dict[str, float | str] = {
            dataset_label_col: dataset_groups.SUMMARY_LABELS[group_key],
            "Pairs": int(stats_dict["n_pairs"]),
            win_rate_col: stats_dict["group_a_win_rate"],
            delta_col: stats_dict["median_delta"],
            "Wilcoxon p": stats_dict["wilcoxon_p_value"],
            effect_col: stats_dict["effect_size_signed"],
        }
        if metric_is_qe:
            pct_pair_values = pd.to_numeric(group_df[pct_pair_col], errors="coerce").to_numpy(dtype=float)
            pct_worse_values = pd.to_numeric(group_df[pct_worse_col], errors="coerce").to_numpy(dtype=float)
            pct_pair_stats = _compute_delta_stats(
                pct_pair_values,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            pct_worse_stats = _compute_delta_stats(
                pct_worse_values,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            row[pct_pair_col] = pct_pair_stats["median_delta"]
            row[pct_worse_col] = pct_worse_stats["median_delta"]

        rows.append(row)

    if not rows:
        return dataset_summary_df

    return pd.concat([dataset_summary_df, pd.DataFrame(rows)], ignore_index=True)


def _select_pairing_columns(
    df: pd.DataFrame,
    cfg: PairingConfig,
) -> Tuple[Optional[str], List[str], Dict[str, Optional[str]]]:
    comparison_col = _resolve_column(df, cfg.comparison_column_candidates)
    dataset_col = _resolve_column(df, cfg.dataset_column_candidates)
    if dataset_col is None:
        return comparison_col, [], {}

    key_cols = [dataset_col]
    selected: Dict[str, Optional[str]] = {
        "dataset": dataset_col,
        "comparison": comparison_col,
    }

    for optional_candidates in cfg.optional_key_column_candidates:
        resolved = _resolve_column(df, optional_candidates)
        selected[optional_candidates[0]] = resolved
        if resolved and resolved not in key_cols and resolved != comparison_col:
            key_cols.append(resolved)

    return comparison_col, key_cols, selected


def _build_pairs(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    comparison_col: str,
    metric_col: str,
    top_k: int,
    comparison_values: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working = df.dropna(subset=list(key_cols) + [comparison_col, metric_col]).copy()
    working[comparison_col] = working[comparison_col].astype(str).str.lower().str.strip()
    group_a, group_b = comparison_values[0].lower(), comparison_values[1].lower()
    working = working[working[comparison_col].isin([group_a, group_b])]

    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = list(key_cols) + [comparison_col]
    top_rows = (
        working.sort_values(metric_col, ascending=True)
        .groupby(group_cols, dropna=False)
        .head(top_k)
        .copy()
    )

    aggregated_top = (
        top_rows.groupby(group_cols, dropna=False)
        .agg(
            paired_score=(metric_col, "median"),
            trials_considered=(metric_col, "size"),
        )
        .reset_index()
    )

    worst_rows = (
        working.sort_values(metric_col, ascending=False)
        .groupby(group_cols, dropna=False)
        .head(top_k)
        .copy()
    )
    aggregated_worst = (
        worst_rows.groupby(group_cols, dropna=False)
        .agg(
            paired_score_worst=(metric_col, "median"),
            trials_considered_worst=(metric_col, "size"),
        )
        .reset_index()
    )
    aggregated = aggregated_top.merge(aggregated_worst, on=group_cols, how="left")

    group_a_df = aggregated[aggregated[comparison_col] == group_a].copy()
    group_b_df = aggregated[aggregated[comparison_col] == group_b].copy()

    group_a_df = group_a_df.rename(
        columns={
            "paired_score": "score_group_a",
            "trials_considered": "trials_group_a",
            "paired_score_worst": "worst_score_group_a",
            "trials_considered_worst": "worst_trials_group_a",
        }
    ).drop(columns=[comparison_col])
    group_b_df = group_b_df.rename(
        columns={
            "paired_score": "score_group_b",
            "trials_considered": "trials_group_b",
            "paired_score_worst": "worst_score_group_b",
            "trials_considered_worst": "worst_trials_group_b",
        }
    ).drop(columns=[comparison_col])

    pairs = group_a_df.merge(group_b_df, on=list(key_cols), how="inner")
    if not pairs.empty:
        pairs["delta_group_a_minus_group_b"] = pairs["score_group_a"] - pairs["score_group_b"]
        improvement_numerator = pairs["score_group_b"] - pairs["score_group_a"]
        pair_ref = np.maximum(np.abs(pairs["score_group_a"]), np.abs(pairs["score_group_b"]))
        worse5_ref = np.maximum(np.abs(pairs["worst_score_group_a"]), np.abs(pairs["worst_score_group_b"]))
        pairs["pct_improvement_pair_ref"] = np.where(
            pair_ref > 0,
            (improvement_numerator / pair_ref) * 100.0,
            np.nan,
        )
        pairs["pct_improvement_worse5_ref"] = np.where(
            worse5_ref > 0,
            (improvement_numerator / worse5_ref) * 100.0,
            np.nan,
        )

    return pairs, aggregated


def _compute_coverage(
    aggregated: pd.DataFrame,
    key_cols: Sequence[str],
    comparison_col: str,
    comparison_values: Tuple[str, str],
    comparison_labels: Tuple[str, str],
) -> pd.DataFrame:
    label_a, label_b = comparison_labels
    if aggregated.empty:
        return pd.DataFrame(
            [
                {"Metric": "Total unique units", "Value": 0},
                {"Metric": f"Matched units ({label_a} + {label_b})", "Value": 0},
                {"Metric": f"{label_a}-only units", "Value": 0},
                {"Metric": f"{label_b}-only units", "Value": 0},
            ]
        )

    value_a, value_b = comparison_values[0].lower(), comparison_values[1].lower()
    unit_values = (
        aggregated.groupby(list(key_cols), dropna=False)[comparison_col]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="comparison_values")
    )
    total_units = int(len(unit_values))
    matched_units = int(
        unit_values["comparison_values"].apply(lambda values: value_a in values and value_b in values).sum()
    )
    group_a_only_units = int(
        unit_values["comparison_values"].apply(lambda values: value_a in values and value_b not in values).sum()
    )
    group_b_only_units = int(
        unit_values["comparison_values"].apply(lambda values: value_b in values and value_a not in values).sum()
    )

    return pd.DataFrame(
        [
            {"Metric": "Total unique units", "Value": total_units},
            {"Metric": f"Matched units ({label_a} + {label_b})", "Value": matched_units},
            {"Metric": f"{label_a}-only units", "Value": group_a_only_units},
            {"Metric": f"{label_b}-only units", "Value": group_b_only_units},
        ]
    )


def _prepare_metric_columns(df: pd.DataFrame, cfg: PairingConfig) -> Tuple[pd.DataFrame, List[str]]:
    working = df.copy()

    if (
        "balanced_qe_normalized" not in working.columns
        and "quantization_error_train_normalized" in working.columns
        and "quantization_error_holdout_normalized" in working.columns
    ):
        working["balanced_qe_normalized"] = working[
            ["quantization_error_train_normalized", "quantization_error_holdout_normalized"]
        ].mean(axis=1)

    available = [metric for metric in cfg.metric_columns if metric in working.columns]
    return working, available


def _generate_paired_analysis_report(
    df: pd.DataFrame,
    output_file: Path | str,
    config: PairingConfig,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_a, label_b = config.comparison_labels
    value_a, value_b = config.comparison_values[0].lower(), config.comparison_values[1].lower()
    win_rate_col = f"{label_a} Win Rate"
    delta_col = f"Wilcoxon Shift ({label_a}-{label_b})"
    mean_delta_col = f"Mean Delta ({label_a}-{label_b})"
    delta_col_short = "Wilcoxon Shift (A-B)"
    delta_ci_col = "Wilcoxon Shift 95% CI"
    effect_col = "Effect Size (Signed)"

    lines: List[str] = []
    lines.append(f"# {config.report_title}")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")

    comparison_col, key_cols, selected = _select_pairing_columns(df, config)
    if comparison_col is None:
        lines.append("Unable to run paired analysis: comparison column not found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path
    if not key_cols:
        lines.append("Unable to run paired analysis: dataset column not found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    working_df, metric_cols = _prepare_metric_columns(df, config)
    if not metric_cols:
        lines.append("Unable to run paired analysis: no required metric columns were found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Comparison: `{label_a}` vs `{label_b}`")
    lines.append(f"- Comparison column: `{comparison_col}`")
    lines.append("- Metrics tested:")
    for metric_col in metric_cols:
        lines.append(f"  - `{metric_col}` ({_metric_label(metric_col)})")
    lines.append(f"- Robust score per unit/group: median of top-{config.top_k} trials")
    lines.append(f"- Pairing columns: `{', '.join(key_cols)}`")
    if selected.get("seed") is None:
        lines.append("- Seed column: not available (condition-level pairing)")
    lines.append("")

    dataset_col = key_cols[0]
    dataset_values = dataset_groups.ordered_dataset_labels(working_df[dataset_col].dropna().unique())
    global_summary_rows: List[Dict[str, str]] = []

    for metric_col in metric_cols:
        metric_label = _metric_label(metric_col)
        metric_is_qe = _is_qe_metric(metric_col)
        pct_pair_col = f"% Improvement ({label_a} over {label_b}, pair-ref)"
        pct_worse_col = f"% Improvement ({label_a} over {label_b}, worse-5 ref)"
        pct_pair_ci_col = "% Improvement Wilcoxon 95% CI (pair-ref)"
        pct_worse_ci_col = "% Improvement Wilcoxon 95% CI (worse-5 ref)"
        lines.append(f"## Metric: {metric_label}")
        lines.append("")

        pairs, aggregated = _build_pairs(
            df=working_df,
            key_cols=key_cols,
            comparison_col=comparison_col,
            metric_col=metric_col,
            top_k=config.top_k,
            comparison_values=(value_a, value_b),
        )

        lines.append("### Pair Coverage")
        lines.append("")
        coverage_df = _compute_coverage(
            aggregated=aggregated,
            key_cols=key_cols,
            comparison_col=comparison_col,
            comparison_values=(value_a, value_b),
            comparison_labels=(label_a, label_b),
        )
        lines.append(_to_markdown(coverage_df))
        lines.append("")

        lines.append("### Dataset-Stratified Results")
        lines.append("")
        dataset_summary_records: List[Dict[str, float]] = []
        for dataset in dataset_values:
            lines.append(f"#### Dataset: {dataset}")
            lines.append("")
            dataset_pairs = pairs[pairs[dataset_col] == dataset].copy() if not pairs.empty else pd.DataFrame()

            if dataset_pairs.empty:
                lines.append(f"No matched {label_a}/{label_b} pairs for this dataset.")
                lines.append("")
                continue

            stats_dict = _compute_delta_stats(
                dataset_pairs["delta_group_a_minus_group_b"].to_numpy(dtype=float),
                bootstrap_iterations=config.bootstrap_iterations,
                seed=config.random_seed,
            )
            pct_pair_stats: Dict[str, float] | None = None
            pct_worse_stats: Dict[str, float] | None = None
            if metric_is_qe:
                pct_pair_stats = _compute_delta_stats(
                    dataset_pairs["pct_improvement_pair_ref"].to_numpy(dtype=float),
                    bootstrap_iterations=config.bootstrap_iterations,
                    seed=config.random_seed,
                )
                pct_worse_stats = _compute_delta_stats(
                    dataset_pairs["pct_improvement_worse5_ref"].to_numpy(dtype=float),
                    bootstrap_iterations=config.bootstrap_iterations,
                    seed=config.random_seed,
                )

            summary_record: Dict[str, float | str] = {
                "Dataset": dataset,
                "Pairs": int(stats_dict["n_pairs"]),
                win_rate_col: stats_dict["group_a_win_rate"],
                delta_col: stats_dict["median_delta"],
                "Wilcoxon p": stats_dict["wilcoxon_p_value"],
                effect_col: stats_dict["effect_size_signed"],
            }
            if metric_is_qe and pct_pair_stats is not None and pct_worse_stats is not None:
                summary_record[pct_pair_col] = pct_pair_stats["median_delta"]
                summary_record[pct_worse_col] = pct_worse_stats["median_delta"]
            dataset_summary_records.append(summary_record)

            stats_row: Dict[str, str | int] = {
                "Pairs": int(stats_dict["n_pairs"]),
                f"{label_a} Wins": int(stats_dict["group_a_wins"]),
                f"{label_b} Wins": int(stats_dict["group_b_wins"]),
                "Ties": int(stats_dict["ties"]),
                win_rate_col: _format_number(stats_dict["group_a_win_rate"], digits=4),
                delta_col_short: _format_number(
                    stats_dict["median_delta"], digits=6, scientific=True
                ),
                delta_ci_col: (
                    f"[{_format_number(stats_dict['median_ci_low'], digits=6, scientific=True)}, "
                    f"{_format_number(stats_dict['median_ci_high'], digits=6, scientific=True)}]"
                ),
                mean_delta_col: _format_number(
                    stats_dict["mean_delta"], digits=6, scientific=True
                ),
                "Wilcoxon p": _format_number(
                    stats_dict["wilcoxon_p_value"], digits=6, scientific=True
                ),
                effect_col: _format_number(stats_dict["effect_size_signed"]),
            }
            if metric_is_qe and pct_pair_stats is not None and pct_worse_stats is not None:
                stats_row[pct_pair_col] = _format_number(
                    pct_pair_stats["median_delta"], digits=3, scientific=True
                )
                stats_row[pct_pair_ci_col] = (
                    f"[{_format_number(pct_pair_stats['median_ci_low'], digits=3, scientific=True)}, "
                    f"{_format_number(pct_pair_stats['median_ci_high'], digits=3, scientific=True)}]"
                )
                stats_row[pct_worse_col] = _format_number(
                    pct_worse_stats["median_delta"], digits=3, scientific=True
                )
                stats_row[pct_worse_ci_col] = (
                    f"[{_format_number(pct_worse_stats['median_ci_low'], digits=3, scientific=True)}, "
                    f"{_format_number(pct_worse_stats['median_ci_high'], digits=3, scientific=True)}]"
                )
            stats_table = pd.DataFrame([stats_row])
            lines.append(_to_markdown(stats_table))
            lines.append("")

            group_a_slug = _slug(label_a)
            group_b_slug = _slug(label_b)
            detail_cols = [col for col in key_cols if col != dataset_col] + [
                "score_group_a",
                "score_group_b",
                "worst_score_group_a",
                "worst_score_group_b",
                "delta_group_a_minus_group_b",
                "pct_improvement_pair_ref",
                "pct_improvement_worse5_ref",
                "trials_group_a",
                "trials_group_b",
                "worst_trials_group_a",
                "worst_trials_group_b",
            ]
            detail_cols = [col for col in detail_cols if col in dataset_pairs.columns]
            detail_df = (
                dataset_pairs[detail_cols]
                .sort_values("delta_group_a_minus_group_b", ascending=True)
                .head(config.max_detail_rows)
                .rename(
                    columns={
                        "score_group_a": f"score_{group_a_slug}",
                        "score_group_b": f"score_{group_b_slug}",
                        "worst_score_group_a": f"worst5_score_{group_a_slug}",
                        "worst_score_group_b": f"worst5_score_{group_b_slug}",
                        "delta_group_a_minus_group_b": f"delta_{group_a_slug}_minus_{group_b_slug}",
                        "pct_improvement_pair_ref": "pct_improvement_pair_ref",
                        "pct_improvement_worse5_ref": "pct_improvement_worse5_ref",
                        "trials_group_a": f"trials_{group_a_slug}",
                        "trials_group_b": f"trials_{group_b_slug}",
                        "worst_trials_group_a": f"worst5_trials_{group_a_slug}",
                        "worst_trials_group_b": f"worst5_trials_{group_b_slug}",
                    }
                )
            )
            lines.append(
                f"Top {config.max_detail_rows} paired units by delta (most {label_a}-favouring first):"
            )
            lines.append("")
            lines.append(_to_markdown(detail_df))
            lines.append("")

        if dataset_summary_records:
            lines.append("### Dataset Summary Table")
            lines.append("")
            dataset_summary_df = pd.DataFrame(dataset_summary_records)
            dataset_summary_df = _append_group_summary_rows(
                dataset_summary_df=dataset_summary_df,
                dataset_label_col="Dataset",
                metric_is_qe=metric_is_qe,
                win_rate_col=win_rate_col,
                delta_col=delta_col,
                effect_col=effect_col,
                pct_pair_col=pct_pair_col,
                pct_worse_col=pct_worse_col,
                bootstrap_iterations=config.bootstrap_iterations,
                seed=config.random_seed,
            )
            dataset_summary_df[win_rate_col] = dataset_summary_df[win_rate_col].map(
                lambda x: _format_number(x, digits=4)
            )
            dataset_summary_df[delta_col] = dataset_summary_df[delta_col].map(
                lambda x: _format_number(x, digits=6, scientific=True)
            )
            dataset_summary_df["Wilcoxon p"] = dataset_summary_df["Wilcoxon p"].map(
                lambda x: _format_number(x, digits=6, scientific=True)
            )
            dataset_summary_df[effect_col] = dataset_summary_df[effect_col].map(_format_number)
            if metric_is_qe and pct_pair_col in dataset_summary_df.columns:
                dataset_summary_df[pct_pair_col] = dataset_summary_df[pct_pair_col].map(
                    lambda x: _format_number(x, digits=3, scientific=True)
                )
            if metric_is_qe and pct_worse_col in dataset_summary_df.columns:
                dataset_summary_df[pct_worse_col] = dataset_summary_df[pct_worse_col].map(
                    lambda x: _format_number(x, digits=3, scientific=True)
                )
            dataset_summary_df = _order_dataset_summary_rows(dataset_summary_df, dataset_label_col="Dataset")
            lines.append(_to_markdown(dataset_summary_df))
            lines.append("")

        if pairs.empty:
            global_summary_rows.append(
                {
                    "Metric": metric_label,
                    "Pairs": "0",
                    f"{label_a} Wins": "0",
                    f"{label_b} Wins": "0",
                    "Ties": "0",
                    win_rate_col: "N/A",
                    delta_col: "N/A",
                    delta_ci_col: "N/A",
                    "Wilcoxon p": "N/A",
                    effect_col: "N/A",
                    pct_pair_col: "N/A" if metric_is_qe else "N/A",
                    pct_worse_col: "N/A" if metric_is_qe else "N/A",
                }
            )
            lines.append("---")
            lines.append("")
            continue

        global_stats = _compute_delta_stats(
            pairs["delta_group_a_minus_group_b"].to_numpy(dtype=float),
            bootstrap_iterations=config.bootstrap_iterations,
            seed=config.random_seed,
        )
        global_summary_rows.append(
            {
                "Metric": metric_label,
                "Pairs": str(int(global_stats["n_pairs"])),
                f"{label_a} Wins": str(int(global_stats["group_a_wins"])),
                f"{label_b} Wins": str(int(global_stats["group_b_wins"])),
                "Ties": str(int(global_stats["ties"])),
                win_rate_col: _format_number(global_stats["group_a_win_rate"], digits=4),
                delta_col: _format_number(
                    global_stats["median_delta"], digits=6, scientific=True
                ),
                delta_ci_col: (
                    f"[{_format_number(global_stats['median_ci_low'], digits=6, scientific=True)}, "
                    f"{_format_number(global_stats['median_ci_high'], digits=6, scientific=True)}]"
                ),
                "Wilcoxon p": _format_number(
                    global_stats["wilcoxon_p_value"], digits=6, scientific=True
                ),
                effect_col: _format_number(global_stats["effect_size_signed"]),
                pct_pair_col: "N/A",
                pct_worse_col: "N/A",
            }
        )
        if metric_is_qe:
            global_pct_pair_stats = _compute_delta_stats(
                pairs["pct_improvement_pair_ref"].to_numpy(dtype=float),
                bootstrap_iterations=config.bootstrap_iterations,
                seed=config.random_seed,
            )
            global_pct_worse_stats = _compute_delta_stats(
                pairs["pct_improvement_worse5_ref"].to_numpy(dtype=float),
                bootstrap_iterations=config.bootstrap_iterations,
                seed=config.random_seed,
            )
            global_summary_rows[-1][pct_pair_col] = _format_number(
                global_pct_pair_stats["median_delta"], digits=3, scientific=True
            )
            global_summary_rows[-1][pct_worse_col] = _format_number(
                global_pct_worse_stats["median_delta"], digits=3, scientific=True
            )
        lines.append("---")
        lines.append("")

    lines.append("## Global Results")
    lines.append("")
    if global_summary_rows:
        global_summary_df = pd.DataFrame(global_summary_rows)
        lines.append(_to_markdown(global_summary_df))
        lines.append("")
        lines.append(
            f"Interpretation note: negative delta means {label_a} has lower (better) score than {label_b}."
        )
        lines.append(
            f"QE interpretation note: positive % improvement means {label_a} has lower (better) QE than {label_b}."
        )
        lines.append("")
    else:
        lines.append("No matched pairs available for global analysis.")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_algorithm_paired_analysis_report(
    df: pd.DataFrame,
    output_file: Path | str,
    config: PairingConfig | None = None,
) -> Path:
    """Generate Batch-vs-Colours paired report."""
    return _generate_paired_analysis_report(df=df, output_file=output_file, config=config or _algorithm_pairing_config())


def generate_topology_paired_analysis_report(
    df: pd.DataFrame,
    output_file: Path | str,
    config: PairingConfig | None = None,
) -> Path:
    """Generate MST-vs-Hexagonal paired report."""
    return _generate_paired_analysis_report(df=df, output_file=output_file, config=config or _topology_pairing_config())


def generate_paired_analysis_report(
    df: pd.DataFrame,
    output_file: Path | str,
    config: PairingConfig | None = None,
) -> Path:
    """
    Backward-compatible entry point.

    Defaults to algorithm comparison (Batch vs Colours).
    """
    return generate_algorithm_paired_analysis_report(df=df, output_file=output_file, config=config)
