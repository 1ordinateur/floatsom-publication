from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from matplotlib import ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups

from .constants import *

def _ordered_dataset_labels(dataset_values: Sequence[str]) -> List[str]:
    return dataset_groups.ordered_dataset_labels(dataset_values)

def _is_global_dataset_label(value: object) -> bool:
    return dataset_groups.is_global_dataset_label(value)

def _dataset_group_key(value: object) -> str:
    return dataset_groups.dataset_group(value)

def _attach_dataset_indices(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    sample_size_col: str = "sample_size",
    index_col: str = "dataset_index",
) -> pd.DataFrame:
    if df.empty or dataset_col not in df.columns:
        return df.copy()

    working = df.copy()
    dataset_series = working[dataset_col].astype(str).str.strip()
    working[dataset_col] = dataset_series
    metadata = pd.DataFrame(
        {
            dataset_col: dataset_series,
            sample_size_col: (
                pd.to_numeric(working[sample_size_col], errors="coerce")
                if sample_size_col in working.columns
                else np.nan
            ),
        }
    )
    metadata = metadata[metadata[dataset_col] != ""].copy()
    metadata = metadata[~metadata[dataset_col].map(_is_global_dataset_label)].copy()
    if metadata.empty:
        working[index_col] = pd.Series(pd.NA, index=working.index, dtype="Int64")
        return working

    metadata = metadata.sort_values(
        [sample_size_col, dataset_col],
        ascending=[True, True],
        na_position="last",
    ).drop_duplicates(subset=[dataset_col], keep="first").reset_index(drop=True)
    metadata[index_col] = pd.Series(np.arange(1, len(metadata) + 1), dtype="Int64")
    index_lookup = dict(zip(metadata[dataset_col], metadata[index_col]))
    working[index_col] = working[dataset_col].map(index_lookup).astype("Int64")
    return working

def _dataset_group_rank(group: str) -> int:
    if group in dataset_groups.DATASET_GROUP_ORDER:
        return dataset_groups.DATASET_GROUP_ORDER.index(group)
    return len(dataset_groups.DATASET_GROUP_ORDER)

def _dataset_group_boundaries(dataset_order: Sequence[str]) -> List[Tuple[str, str]]:
    if not dataset_order:
        return []
    groups = [_dataset_group_key(label) for label in dataset_order]
    boundaries: List[Tuple[str, str]] = []
    for idx in range(len(dataset_order) - 1):
        if groups[idx] != groups[idx + 1]:
            boundaries.append((dataset_order[idx], dataset_order[idx + 1]))
        # Within the global-summary block, draw a stronger separator before GLOBAL_OVERALL.
        if (
            str(dataset_order[idx]).strip().upper() in {dataset_groups.GLOBAL_SYNTHETIC_LABEL, dataset_groups.GLOBAL_REAL_LABEL}
            and str(dataset_order[idx + 1]).strip().upper() == dataset_groups.GLOBAL_OVERALL_LABEL
        ):
            boundaries.append((dataset_order[idx], dataset_order[idx + 1]))
    return boundaries

def _draw_dataset_group_separators(
    ax: Any,
    dataset_order: Sequence[str],
    y_positions: Dict[str, float],
    *,
    color: str = DATASET_GROUP_SEPARATOR_COLOR,
    linewidth: float = DATASET_GROUP_SEPARATOR_LINEWIDTH,
    alpha: float = DATASET_GROUP_SEPARATOR_ALPHA,
    zorder: float = 0.6,
) -> None:
    for upper_label, lower_label in _dataset_group_boundaries(dataset_order):
        if upper_label not in y_positions or lower_label not in y_positions:
            continue
        y_value = (float(y_positions[upper_label]) + float(y_positions[lower_label])) / 2.0
        is_double = (
            str(lower_label).strip().upper() == dataset_groups.GLOBAL_OVERALL_LABEL
            and str(upper_label).strip().upper() in {dataset_groups.GLOBAL_SYNTHETIC_LABEL, dataset_groups.GLOBAL_REAL_LABEL}
        )
        if not is_double:
            ax.axhline(y_value, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
            continue

        # Double separator: two close lines.
        delta = 0.07
        ax.axhline(y_value - delta, color=color, linewidth=linewidth * 1.25, alpha=alpha, zorder=zorder)
        ax.axhline(y_value + delta, color=color, linewidth=linewidth * 1.25, alpha=alpha, zorder=zorder)

def _extract_per_row_stats(summary_df: pd.DataFrame, dataset_col: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    ordered_columns = [
        dataset_col,
        "n_pairs",
        "p_value",
        "q_value",
        "effect_size_signed",
        "median_pct",
        "mean_pct",
        "ci_low_pct",
        "ci_high_pct",
        "comparison_key",
        "comparison_label",
    ]
    available = [column for column in ordered_columns if column in summary_df.columns]
    if not available:
        return pd.DataFrame()
    stats_df = summary_df[available].copy()
    if dataset_col in stats_df.columns:
        stats_df = stats_df.rename(columns={dataset_col: "dataset"})
    return stats_df

def _write_per_row_stats_table(
    summary_df: pd.DataFrame,
    dataset_col: str,
    output_path: Optional[Path],
) -> Optional[Path]:
    if output_path is None:
        return None
    stats_df = _extract_per_row_stats(summary_df, dataset_col=dataset_col)
    if stats_df.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    return output_path

def _architecture_label(value: str) -> str:
    text = str(value).strip().lower()
    if text == "hexagonal":
        return "Hexagonal"
    if text == "mst":
        return "MST"
    if text == "rng":
        return "RNG"
    return text.replace("_", " ").title() if text else "Unknown"

def _method_color_for_algorithm_label(label: object) -> Optional[str]:
    """Resolve canonical method color from an algorithm/batch-mode label."""
    text = str(label).strip().lower()
    if not text:
        return None
    if text == "colors" or text.startswith("colors "):
        return METHOD_BASE_COLORS["colors"]
    if text == "minibatch":
        return METHOD_BASE_COLORS["minibatch"]
    if text.startswith("batch") and "mini" in text:
        return METHOD_BASE_COLORS["minibatch"]
    if text.startswith("batch"):
        return METHOD_BASE_COLORS["batch"]
    return METHOD_BASE_COLORS.get(text)

def _build_algorithm_batch_mode_color_map(labels: Sequence[str]) -> Dict[str, Any]:
    """Build deterministic colors for strata labels, preferring canonical method colors."""
    color_map: Dict[str, Any] = {}
    unknown_labels: List[str] = []
    unique_labels = sorted({str(label) for label in labels})

    for label in unique_labels:
        canonical_color = _method_color_for_algorithm_label(label)
        if canonical_color is not None:
            color_map[label] = canonical_color
        else:
            unknown_labels.append(label)

    if unknown_labels:
        if sns is not None:
            fallback_palette = sns.color_palette("colorblind", n_colors=len(unknown_labels) + 3)
        else:
            fallback_palette = plt.cm.tab10(np.linspace(0, 1, len(unknown_labels) + 3))

        used_colors = {mcolors.to_hex(color) for color in color_map.values()}
        fallback_idx = 0
        for label in unknown_labels:
            while fallback_idx < len(fallback_palette):
                candidate = mcolors.to_hex(fallback_palette[fallback_idx])
                fallback_idx += 1
                if candidate not in used_colors:
                    color_map[label] = candidate
                    used_colors.add(candidate)
                    break
            if label not in color_map:
                color_map[label] = mcolors.to_hex(plt.cm.tab10(0))

    return color_map

def _comparison_series_color(compare_col: str, value_a: str) -> Any:
    """Pick a semantic color for curves tied to the primary compared value."""
    compare_key = str(compare_col).strip().lower()
    value_key = str(value_a).strip().lower()
    if compare_key == "architecture":
        return TOPOLOGY_BASE_COLORS.get(value_key, DEFAULT_SERIES_COLOR)
    method_color = _method_color_for_algorithm_label(value_key)
    return method_color if method_color is not None else DEFAULT_SERIES_COLOR

def _darken_color(color: Any, factor: float = 0.55) -> str:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    clamped_factor = min(1.0, max(0.0, float(factor)))
    darkened = np.clip(rgb * clamped_factor, 0.0, 1.0)
    return mcolors.to_hex(darkened)

def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None

def _normalize_batch_mode(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "pooled" in text:
        return "pooled_batch_modes"
    if "mini" in text:
        return "minibatch"
    if "full" in text or "batch" in text:
        return "full_batch"
    return text.replace(" ", "_")

def _batch_mode_label(value: str) -> str:
    text = str(value).strip().lower()
    if text == "pooled_batch_modes":
        return "Pooled batch modes"
    if text == "minibatch":
        return "Mini-batch"
    if text == "full_batch":
        return "Full batch"
    return text.replace("_", " ").title() if text else "Unknown"

def _topology_stratum_label(algorithm: object, batch_mode: object) -> str:
    if algorithm is None or pd.isna(algorithm):
        return "Unknown"
    algo_text = str(algorithm).strip().lower()
    if algo_text == "batch":
        normalized_mode = _normalize_batch_mode(batch_mode) or "full_batch"
        return f"Batch ({_batch_mode_label(normalized_mode)})"
    if algo_text == "colors":
        return "Colors"
    return algo_text.replace("_", " ").title() if algo_text else "Unknown"

def _slug(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

def _format_prob(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"

def _format_axis_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    value = float(value)
    abs_value = abs(value)

    if abs_value > 0 and (abs_value >= 1e4 or abs_value < 1e-3):
        return f"{value:.2e}"
    if abs_value >= 100:
        text = f"{value:.1f}"
    elif abs_value >= 10:
        text = f"{value:.2f}"
    elif abs_value >= 1:
        text = f"{value:.3f}"
    else:
        text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")

def _pvalue_to_stars(value: float) -> str:
    if value is None or pd.isna(value):
        return "ns"
    value = float(value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"

def _row_significance_value(row: pd.Series) -> float:
    q_raw = row.get("q_value", np.nan)
    if q_raw is not None and pd.notna(q_raw):
        try:
            q_value = float(q_raw)
        except (TypeError, ValueError):
            q_value = float("nan")
        if np.isfinite(q_value):
            return q_value

    p_raw = row.get("p_value", np.nan)
    if p_raw is None or pd.isna(p_raw):
        return float("nan")
    try:
        p_value = float(p_raw)
    except (TypeError, ValueError):
        return float("nan")
    return p_value if np.isfinite(p_value) else float("nan")

def _row_is_significant(row: pd.Series, alpha: float) -> bool:
    value = _row_significance_value(row)
    return np.isfinite(value) and value < alpha

def _paired_t_two_sided_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0

    std = float(np.std(finite, ddof=1))
    mean = float(np.mean(finite))
    if not np.isfinite(std) or std == 0.0:
        return 1.0 if np.isclose(mean, 0.0) else 0.0

    try:
        test = stats.ttest_1samp(finite, popmean=0.0, alternative="two-sided")
    except TypeError:
        test = stats.ttest_1samp(finite, popmean=0.0)
    p_value = float(test.pvalue)
    if np.isfinite(p_value):
        return p_value
    return 1.0 if np.isclose(mean, 0.0) else 0.0

def _wilcoxon_two_sided_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")

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

def _paired_t_mean_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean_value = float(np.mean(finite))
    if n == 1:
        return mean_value, mean_value, mean_value

    std = float(np.std(finite, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return mean_value, mean_value, mean_value

    t_crit = float(stats.t.ppf(1.0 - float(alpha) / 2.0, n - 1))
    if not np.isfinite(t_crit):
        return mean_value, mean_value, mean_value

    margin = t_crit * std / np.sqrt(float(n))
    return mean_value, mean_value - margin, mean_value + margin

def _signed_test_stats(delta_a_minus_b: np.ndarray, test_method: str = "paired_t") -> Dict[str, float]:
    finite = delta_a_minus_b[np.isfinite(delta_a_minus_b)]
    n_pairs = int(finite.size)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "wins_a": 0,
            "wins_b": 0,
            "ties": 0,
            "win_rate_a": float("nan"),
            "p_value": float("nan"),
            "effect_size_signed": float("nan"),
        }

    wins_a = int(np.sum(finite > 0))
    wins_b = int(np.sum(finite < 0))
    ties = int(n_pairs - wins_a - wins_b)

    if test_method == "paired_t":
        p_value = _paired_t_two_sided_pvalue(finite)
    elif test_method == "wilcoxon":
        p_value = _wilcoxon_two_sided_pvalue(finite)
    else:
        raise ValueError(f"Unsupported signed-test method: {test_method}")

    denom = wins_a + wins_b
    effect_size_signed = float(wins_a - wins_b) / float(denom) if denom > 0 else float("nan")

    return {
        "n_pairs": n_pairs,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": float(wins_a) / float(n_pairs),
        "p_value": p_value,
        "effect_size_signed": effect_size_signed,
    }

def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if not valid.any():
        return q

    p_valid = p[valid].astype(float)
    order = np.argsort(p_valid.values)
    ranked = p_valid.values[order]
    m = float(len(ranked))
    q_ranked = np.empty_like(ranked)

    prev = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        rank = float(i + 1)
        value = min(prev, ranked[i] * m / rank)
        q_ranked[i] = value
        prev = value

    restored = np.empty_like(q_ranked)
    restored[order] = q_ranked
    q.loc[valid] = restored
    return q

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    known_topologies = {"hexagonal", "mst", "rng", "grid"}
    known_sampling_modes = set(CANONICAL_SAMPLING_MODES)

    if "dataset" not in working.columns:
        raise ValueError("Input file must contain a 'dataset' column.")

    if "algorithm" not in working.columns and "processing_type" in working.columns:
        working["algorithm"] = working["processing_type"]

    architecture_col = _resolve_column(working, ["config_topology_type", "architecture", "map_type"])
    if architecture_col and architecture_col != "architecture":
        working["architecture"] = working[architecture_col]

    if "algorithm" in working.columns:
        working["algorithm"] = working["algorithm"].astype(str).str.lower().str.strip()
    if "architecture" in working.columns:
        working["architecture"] = working["architecture"].astype(str).str.lower().str.strip()
        if "config_topology_type" in working.columns:
            config_topology = working["config_topology_type"].astype(str).str.lower().str.strip()
            invalid_topology = (~working["architecture"].isin(known_topologies)) & (config_topology.isin(known_topologies))
            if invalid_topology.any():
                working.loc[invalid_topology, "architecture"] = config_topology[invalid_topology]

    sampling_col = _resolve_column(
        working,
        ["sampling_method_final", "sampling_method", "config_sampling_method", "sampling_method_parsed"],
    )
    seed_col = _resolve_column(working, ["seed", "config_seed", "random_seed", "seed_name", "config_seed_name"])
    split_col = _resolve_column(working, ["evaluation_split", "config_evaluation_split", "split"])
    batch_mode_col = _resolve_column(working, ["pair_batch_mode", "param_batch_mode", "batch_mode", "config_batch_mode"])

    working["pair_sampling"] = working[sampling_col] if sampling_col else "all"
    working["pair_sampling"] = working["pair_sampling"].astype(str).str.lower().str.strip()
    if "config_sampling_method" in working.columns:
        config_sampling = working["config_sampling_method"].astype(str).str.lower().str.strip()
        invalid_sampling = (~working["pair_sampling"].isin(known_sampling_modes)) & (config_sampling.isin(known_sampling_modes))
        if invalid_sampling.any():
            working.loc[invalid_sampling, "pair_sampling"] = config_sampling[invalid_sampling]
    working["pair_seed"] = working[seed_col] if seed_col else np.nan
    # When split metadata is missing, default-aware tuned-vs-default analysis expects "both".
    working["pair_split"] = working[split_col] if split_col else "both"
    if batch_mode_col:
        working["pair_batch_mode"] = working[batch_mode_col].apply(_normalize_batch_mode)
    else:
        working["pair_batch_mode"] = np.nan

    if "algorithm" in working.columns:
        is_batch = working["algorithm"].astype(str).str.lower().str.strip() == "batch"
        working.loc[is_batch & working["pair_batch_mode"].isna(), "pair_batch_mode"] = "full_batch"
        working.loc[~is_batch, "pair_batch_mode"] = "all"
    else:
        working["pair_batch_mode"] = working["pair_batch_mode"].fillna("full_batch")

    if (
        "balanced_qe_raw" not in working.columns
        and "quantization_error_train" in working.columns
        and "quantization_error_holdout" in working.columns
    ):
        working["balanced_qe_raw"] = working[["quantization_error_train", "quantization_error_holdout"]].mean(axis=1)

    if (
        "balanced_qe_normalized" not in working.columns
        and "quantization_error_train_normalized" in working.columns
        and "quantization_error_holdout_normalized" in working.columns
    ):
        working["balanced_qe_normalized"] = working[
            ["quantization_error_train_normalized", "quantization_error_holdout_normalized"]
        ].mean(axis=1)

    if (
        "distortion_measure_holdout" not in working.columns
        and "distortion_measure_train" in working.columns
    ):
        working["distortion_measure_holdout"] = working["distortion_measure_train"]
    if (
        "distortion_measure_holdout_normalized" not in working.columns
        and "distortion_measure_train_normalized" in working.columns
    ):
        working["distortion_measure_holdout_normalized"] = working["distortion_measure_train_normalized"]

    return working

def _exclude_minibatch_rows_for_reporting(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    working = df.copy()
    if "algorithm" not in working.columns or "pair_batch_mode" not in working.columns:
        return working, 0

    algorithm = working["algorithm"].astype(str).str.lower().str.strip()
    batch_mode = working["pair_batch_mode"].astype(str).str.lower().str.strip()
    minibatch_mask = (algorithm == "batch") & (batch_mode == "minibatch")
    excluded_rows = int(minibatch_mask.sum())
    if excluded_rows == 0:
        return working, 0
    return working.loc[~minibatch_mask].copy(), excluded_rows

def _resolve_metric_columns(df: pd.DataFrame, requested: Iterable[str], strict_mode: bool) -> List[str]:
    fallback = {
        "quantization_error_holdout_normalized": "quantization_error_holdout",
        "quantization_error_train_normalized": "quantization_error_train",
        "distortion_measure_holdout_normalized": "distortion_measure_holdout",
        "balanced_qe_normalized": "balanced_qe_raw",
        "distortion_measure_holdout": "distortion_measure_train",
    }
    resolved: List[str] = []
    for metric in requested:
        if metric in df.columns:
            resolved.append(metric)
            continue

        alt = fallback.get(metric)
        if alt and alt in df.columns:
            if strict_mode:
                raise ValueError(
                    f"Requested metric '{metric}' is missing and strict mode is enabled; "
                    f"fallback to '{alt}' is disallowed."
                )
            resolved.append(alt)
            continue

        if strict_mode:
            raise ValueError(f"Requested metric '{metric}' not found in input data.")

    return list(dict.fromkeys(resolved))

def _validate_required_pairing_columns(df: pd.DataFrame, required_cols: Sequence[str], context: str) -> None:
    missing = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
    if missing:
        raise ValueError(
            f"Strict pairing check failed for {context}: missing required pairing columns: {missing}."
        )

def _prepare_pair_working_frame(
    df: pd.DataFrame,
    metric: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    key_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], str, str]:
    needed_cols = list(key_cols) + [compare_col, metric]
    working = df.dropna(subset=needed_cols).copy()
    if working.empty:
        return pd.DataFrame(), [], value_a.lower(), value_b.lower()

    value_a_norm = value_a.lower()
    value_b_norm = value_b.lower()
    working[PAIR_COMPARE_KEY] = working[compare_col].astype(str).str.lower().str.strip()
    working = working[working[PAIR_COMPARE_KEY].isin([value_a_norm, value_b_norm])]
    if working.empty:
        return pd.DataFrame(), [], value_a_norm, value_b_norm

    group_cols = list(key_cols) + [PAIR_COMPARE_KEY]
    return working, group_cols, value_a_norm, value_b_norm

def _build_pairs_from_aggregated(
    aggregated: pd.DataFrame,
    key_cols: Sequence[str],
    value_a: str,
    value_b: str,
    reference_value: Optional[str] = None,
) -> pd.DataFrame:
    if aggregated.empty:
        return pd.DataFrame()

    a_df = aggregated[aggregated[PAIR_COMPARE_KEY] == value_a].copy()
    b_df = aggregated[aggregated[PAIR_COMPARE_KEY] == value_b].copy()
    if a_df.empty or b_df.empty:
        return pd.DataFrame()

    a_df = a_df.rename(columns={"score": "score_a", "trials": "trials_a"}).drop(columns=[PAIR_COMPARE_KEY])
    b_df = b_df.rename(columns={"score": "score_b", "trials": "trials_b"}).drop(columns=[PAIR_COMPARE_KEY])

    pairs = a_df.merge(b_df, on=list(key_cols), how="inner")
    if pairs.empty:
        return pairs

    pairs["delta_a_minus_b"] = pairs["score_a"] - pairs["score_b"]
    improvement_numerator = pairs["score_b"] - pairs["score_a"]
    denom = np.maximum(np.abs(pairs["score_a"]), np.abs(pairs["score_b"]))
    pairs["pct_improvement_a_over_b"] = np.where(
        denom > 0,
        (improvement_numerator / denom) * 100.0,
        np.nan,
    )

    baseline_ref = np.abs(pairs["score_b"])
    pairs["pct_improvement_a_over_b_baseline_ref"] = np.where(
        baseline_ref > 0,
        (improvement_numerator / baseline_ref) * 100.0,
        np.nan,
    )

    if reference_value is None:
        pairs["pct_improvement_a_over_b_reference"] = pairs["pct_improvement_a_over_b_baseline_ref"]
        pairs["reference_score"] = pairs["score_b"]
        return pairs

    reference_norm = str(reference_value).strip().lower()
    ref_df = aggregated[aggregated[PAIR_COMPARE_KEY] == reference_norm].copy()
    if ref_df.empty:
        pairs["pct_improvement_a_over_b_reference"] = np.nan
        pairs["reference_score"] = np.nan
        return pairs

    ref_df = ref_df.rename(columns={"score": "reference_score"}).drop(columns=[PAIR_COMPARE_KEY, "trials"], errors="ignore")
    pairs = pairs.merge(ref_df, on=list(key_cols), how="left")
    reference_abs = np.abs(pd.to_numeric(pairs["reference_score"], errors="coerce"))
    pairs["pct_improvement_a_over_b_reference"] = np.where(
        reference_abs > 0,
        (improvement_numerator / reference_abs) * 100.0,
        np.nan,
    )
    return pairs

def _build_pairs(
    df: pd.DataFrame,
    metric: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    key_cols: Sequence[str],
    top_k: int,
    reference_value: Optional[str] = None,
) -> pd.DataFrame:
    working, group_cols, value_a_norm, value_b_norm = _prepare_pair_working_frame(
        df=df,
        metric=metric,
        compare_col=compare_col,
        value_a=value_a,
        value_b=value_b,
        key_cols=key_cols,
    )
    if working.empty:
        return pd.DataFrame()

    top_rows = (
        working.sort_values(metric, ascending=True)
        .groupby(group_cols, dropna=False)
        .head(top_k)
    )

    aggregated = (
        top_rows.groupby(group_cols, dropna=False)
        .agg(score=(metric, "median"), trials=(metric, "size"))
        .reset_index()
    )
    return _build_pairs_from_aggregated(
        aggregated=aggregated,
        key_cols=key_cols,
        value_a=value_a_norm,
        value_b=value_b_norm,
        reference_value=reference_value,
    )

def _summarize_pairs(
    pairs: pd.DataFrame,
    group_cols: Sequence[str],
    bootstrap_iterations: int,
    seed: int,
    pct_col: str = "pct_improvement_a_over_b",
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> pd.DataFrame:
    _ = bootstrap_iterations, seed
    if pairs.empty:
        return pd.DataFrame(columns=[*group_cols, "n_pairs"])
    if pct_col not in pairs.columns:
        raise ValueError(f"Requested pct column not available in pairs: {pct_col}")

    rows: List[Dict[str, object]] = []
    grouped = pairs.groupby(list(group_cols), dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row: Dict[str, object] = {group_cols[i]: group_key[i] for i in range(len(group_cols))}
        pct = group_df[pct_col].to_numpy(dtype=float)
        delta = group_df["delta_a_minus_b"].to_numpy(dtype=float)
        signed = _signed_test_stats(pct, test_method=test_method)
        if location_ci_method == "paired_t":
            location_pct, ci_low, ci_high = _paired_t_mean_ci(pct)
        elif location_ci_method == "wilcoxon":
            location_pct, ci_low, ci_high = _wilcoxon_location_ci(pct)
        else:
            raise ValueError(f"Unsupported location/CI method: {location_ci_method}")

        row.update(
            {
                "n_pairs": int(signed["n_pairs"]),
                "wins_a": int(signed["wins_a"]),
                "wins_b": int(signed["wins_b"]),
                "ties": int(signed["ties"]),
                "win_rate_a": signed["win_rate_a"],
                "median_delta_raw": float(np.nanmean(delta)),
                "median_pct": location_pct,
                "mean_pct": float(np.nanmean(pct)),
                "ci_low_pct": ci_low,
                "ci_high_pct": ci_high,
                "p_value": signed["p_value"],
                "effect_size_signed": signed["effect_size_signed"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)

def _compute_group_summary_row(
    dataset_rows: pd.DataFrame,
    dataset_col: str,
    label: str,
    *,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> Dict[str, object]:
    dataset_delta = pd.to_numeric(dataset_rows["median_delta_raw"], errors="coerce").to_numpy(dtype=float)
    dataset_pct = pd.to_numeric(dataset_rows["median_pct"], errors="coerce").to_numpy(dtype=float)
    dataset_delta_finite = dataset_delta[np.isfinite(dataset_delta)]
    dataset_pct_finite = dataset_pct[np.isfinite(dataset_pct)]

    signed = _signed_test_stats(dataset_pct_finite, test_method=test_method)
    if location_ci_method == "paired_t":
        location_pct, ci_low, ci_high = _paired_t_mean_ci(dataset_pct_finite)
    elif location_ci_method == "wilcoxon":
        location_pct, ci_low, ci_high = _wilcoxon_location_ci(dataset_pct_finite)
    else:
        raise ValueError(f"Unsupported location/CI method: {location_ci_method}")
    mean_delta = float(np.mean(dataset_delta_finite)) if dataset_delta_finite.size else float("nan")
    mean_pct = location_pct

    return {
        dataset_col: label,
        "n_pairs": int(signed["n_pairs"]),
        "wins_a": int(signed["wins_a"]),
        "wins_b": int(signed["wins_b"]),
        "ties": int(signed["ties"]),
        "win_rate_a": signed["win_rate_a"],
        "median_delta_raw": mean_delta,
        "median_pct": location_pct,
        "mean_pct": mean_pct,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "p_value": signed["p_value"],
        "q_value": np.nan,
        "effect_size_signed": signed["effect_size_signed"],
    }

def _build_group_summary_rows(
    summary_df: pd.DataFrame,
    dataset_col: str,
    *,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    if "median_delta_raw" not in summary_df.columns:
        raise ValueError("GLOBAL group aggregation requires 'median_delta_raw' in summary rows.")
    if "median_pct" not in summary_df.columns:
        raise ValueError("GLOBAL group aggregation requires 'median_pct' in summary rows.")

    working = summary_df.copy()
    working["__dataset_group__"] = working[dataset_col].map(_dataset_group_key)

    rows: List[Dict[str, object]] = []
    # Publication output keeps only GLOBAL_OVERALL in aggregated summary rows.
    for group_key in ("overall",):
        if group_key == "overall":
            group_rows = working
        else:
            group_rows = working[working["__dataset_group__"] == group_key]
        if group_rows.empty:
            continue
        label = dataset_groups.SUMMARY_LABELS[group_key]
        rows.append(
            _compute_group_summary_row(
                group_rows,
                dataset_col=dataset_col,
                label=label,
                test_method=test_method,
                location_ci_method=location_ci_method,
            )
        )

    return pd.DataFrame(rows)

def _compute_pairwise_global_summary_row(
    pairs: pd.DataFrame,
    dataset_col: str,
    pct_col: str,
    *,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> Optional[Dict[str, object]]:
    if pairs.empty:
        return None
    if pct_col not in pairs.columns:
        raise ValueError(f"Requested pct column not available in pairs: {pct_col}")

    working = pairs.copy()
    pct = working[pct_col].to_numpy(dtype=float)
    delta = working["delta_a_minus_b"].to_numpy(dtype=float)
    signed = _signed_test_stats(pct, test_method=test_method)
    if location_ci_method == "paired_t":
        location_pct, ci_low, ci_high = _paired_t_mean_ci(pct)
    elif location_ci_method == "wilcoxon":
        location_pct, ci_low, ci_high = _wilcoxon_location_ci(pct)
    else:
        raise ValueError(f"Unsupported location/CI method: {location_ci_method}")

    return {
        dataset_col: dataset_groups.GLOBAL_OVERALL_LABEL,
        "n_pairs": int(signed["n_pairs"]),
        "wins_a": int(signed["wins_a"]),
        "wins_b": int(signed["wins_b"]),
        "ties": int(signed["ties"]),
        "win_rate_a": signed["win_rate_a"],
        "median_delta_raw": float(np.nanmean(delta)),
        "median_pct": location_pct,
        "mean_pct": float(np.nanmean(pct)),
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "p_value": signed["p_value"],
        "q_value": np.nan,
        "effect_size_signed": signed["effect_size_signed"],
    }

def _add_global_row(
    summary_df: pd.DataFrame,
    pairs: pd.DataFrame,
    dataset_col: str,
    bootstrap_iterations: int,
    seed: int,
    pct_col: str = "pct_improvement_a_over_b",
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> pd.DataFrame:
    _ = bootstrap_iterations, seed

    if summary_df.empty:
        return summary_df

    is_global = summary_df[dataset_col].map(_is_global_dataset_label)
    dataset_rows = summary_df[~is_global].copy()
    if dataset_rows.empty:
        return summary_df

    global_row = _compute_pairwise_global_summary_row(
        pairs=pairs,
        dataset_col=dataset_col,
        pct_col=pct_col,
        test_method=test_method,
        location_ci_method=location_ci_method,
    )
    if global_row is None:
        return summary_df

    group_rows = pd.DataFrame([global_row])
    for column in dataset_rows.columns:
        if column not in group_rows.columns:
            group_rows[column] = np.nan
    group_rows = group_rows[dataset_rows.columns]
    return pd.concat([dataset_rows, group_rows], ignore_index=True)

def _add_group_summary_rows_by_columns(
    summary_df: pd.DataFrame,
    dataset_col: str,
    group_cols: Sequence[str],
    *,
    pairs: Optional[pd.DataFrame] = None,
    pct_col: str = "pct_improvement_a_over_b",
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    if not group_cols:
        return summary_df

    is_global = summary_df[dataset_col].map(_is_global_dataset_label)
    dataset_rows = summary_df[~is_global].copy()
    if dataset_rows.empty:
        return summary_df

    grouped_rows: List[pd.DataFrame] = []
    if pairs is not None and not pairs.empty:
        grouped_pairs = pairs.groupby(list(group_cols), dropna=False)
        for group_key, group_pairs in grouped_pairs:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_summary_row = _compute_pairwise_global_summary_row(
                pairs=group_pairs,
                dataset_col=dataset_col,
                pct_col=pct_col,
                test_method=test_method,
                location_ci_method=location_ci_method,
            )
            if group_summary_row is None:
                continue
            group_summary_rows = pd.DataFrame([group_summary_row])
            for idx, col in enumerate(group_cols):
                group_summary_rows[col] = group_key[idx]
            grouped_rows.append(group_summary_rows)
    else:
        grouped = dataset_rows.groupby(list(group_cols), dropna=False)
        for group_key, group_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_summary_rows = _build_group_summary_rows(
                group_df,
                dataset_col=dataset_col,
                test_method=test_method,
                location_ci_method=location_ci_method,
            )
            if group_summary_rows.empty:
                continue
            for idx, col in enumerate(group_cols):
                group_summary_rows[col] = group_key[idx]
            grouped_rows.append(group_summary_rows)

    if not grouped_rows:
        return summary_df

    group_rows = pd.concat(grouped_rows, ignore_index=True)
    for column in dataset_rows.columns:
        if column not in group_rows.columns:
            group_rows[column] = np.nan
    group_rows = group_rows[dataset_rows.columns]
    return pd.concat([dataset_rows, group_rows], ignore_index=True)

def _apply_q_values(summary_df: pd.DataFrame, dataset_col: str) -> pd.DataFrame:
    out = summary_df.copy()
    out["q_value"] = np.nan
    mask = ~out[dataset_col].map(_is_global_dataset_label)
    out.loc[mask, "q_value"] = _benjamini_hochberg(out.loc[mask, "p_value"])
    return out

def _validate_expected_pairs(summary_df: pd.DataFrame, dataset_col: str, expected_pairs: Optional[int], context: str) -> None:
    if expected_pairs is None:
        return

    dataset_rows = summary_df[~summary_df[dataset_col].map(_is_global_dataset_label)].copy()
    if dataset_rows.empty:
        return

    bad = dataset_rows[dataset_rows["n_pairs"] != int(expected_pairs)]
    if not bad.empty:
        observed = bad[[dataset_col, "n_pairs"]].to_dict(orient="records")
        raise ValueError(
            f"Expected {expected_pairs} pairs per dataset for {context}, but observed mismatches: {observed}"
        )

def _pair_identifier(pairs: pd.DataFrame) -> pd.Series:
    parts: List[pd.Series] = []
    for col in ["pair_sampling", "pair_seed", "pair_split"]:
        if col in pairs.columns:
            parts.append(pairs[col].astype(str))
    if not parts:
        return pd.Series(["pair"] * len(pairs), index=pairs.index)

    identifier = parts[0]
    for extra in parts[1:]:
        identifier = identifier + "|" + extra
    return identifier

def _build_sensitivity_summary(
    df: pd.DataFrame,
    metric: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    key_cols: Sequence[str],
    top_k_values: Sequence[int],
    bootstrap_iterations: int,
    seed: int,
    pct_col: str = "pct_improvement_a_over_b_reference",
    reference_value: Optional[str] = None,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
    pair_sampling_modes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    _ = bootstrap_iterations, seed
    rows: List[Dict[str, float]] = []

    working, group_cols, value_a_norm, value_b_norm = _prepare_pair_working_frame(
        df=df,
        metric=metric,
        compare_col=compare_col,
        value_a=value_a,
        value_b=value_b,
        key_cols=key_cols,
    )
    if working.empty:
        return pd.DataFrame(rows)

    ranked = working.sort_values(metric, ascending=True).copy()
    ranked[PAIR_RANK_KEY] = ranked.groupby(group_cols, dropna=False).cumcount() + 1

    resolved_top_k_values = sorted({int(top_k) for top_k in top_k_values if int(top_k) > 0})
    for top_k in resolved_top_k_values:
        top_rows = ranked[ranked[PAIR_RANK_KEY] <= top_k]
        if top_rows.empty:
            continue

        aggregated = (
            top_rows.groupby(group_cols, dropna=False)
            .agg(score=(metric, "median"), trials=(metric, "size"))
            .reset_index()
        )
        pairs = _build_pairs_from_aggregated(
            aggregated=aggregated,
            key_cols=key_cols,
            value_a=value_a_norm,
            value_b=value_b_norm,
            reference_value=reference_value,
        )
        if pair_sampling_modes is not None:
            pairs = _filter_pairs_to_sampling_modes(pairs, pair_sampling_modes)
        if pairs.empty:
            continue

        global_row = _compute_pairwise_global_summary_row(
            pairs=pairs,
            dataset_col="dataset",
            pct_col=pct_col,
            test_method=test_method,
            location_ci_method=location_ci_method,
        )
        if global_row is None:
            continue
        rows.append(
            {
                "top_k": int(top_k),
                "n_pairs": int(global_row["n_pairs"]),
                "median_pct": float(global_row["median_pct"]),
                "mean_pct": float(global_row["mean_pct"]),
                "ci_low_pct": float(global_row["ci_low_pct"]),
                "ci_high_pct": float(global_row["ci_high_pct"]),
                "p_value": float(global_row["p_value"]),
                "effect_size_signed": float(global_row["effect_size_signed"]),
            }
        )

    return pd.DataFrame(rows)

def _build_topology_sensitivity_series(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    metric: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    key_cols: Sequence[str],
    top_k_values: Sequence[int],
    bootstrap_iterations: int,
    seed: int,
    pct_col: str = "pct_improvement_a_over_b_reference",
    reference_value: Optional[str] = None,
    exclude_algorithm_batch_mode_labels: Optional[Set[str]] = None,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
    pair_sampling_modes: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]], Optional[str]]:
    if df.empty or pairs.empty or "algorithm_batch_mode" not in pairs.columns:
        return {}, {}, None
    if "algorithm" not in pairs.columns or "algorithm" not in df.columns:
        return {}, {}, None

    combo_cols = [col for col in ["algorithm", "pair_batch_mode"] if col in pairs.columns and col in df.columns]
    if "pair_batch_mode" in combo_cols:
        if pairs["pair_batch_mode"].isna().all() or df["pair_batch_mode"].isna().all():
            combo_cols = ["algorithm"]
    if not combo_cols:
        combo_cols = ["algorithm"]

    labels = sorted(
        {
            str(label)
            for label in pairs["algorithm_batch_mode"].dropna().astype(str).tolist()
            if str(label).strip()
        }
    )
    excluded_labels = {
        str(label).strip().lower()
        for label in (exclude_algorithm_batch_mode_labels or set())
        if str(label).strip()
    }
    if excluded_labels:
        labels = [label for label in labels if str(label).strip().lower() not in excluded_labels]
    if not labels:
        return {}, {}, None

    normalized_df_cols: Dict[str, pd.Series] = {}
    for column in combo_cols:
        normalized_df_cols[column] = (
            df[column]
            .where(df[column].notna(), "")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    sensitivity_series_by_stratum: Dict[str, pd.DataFrame] = {}
    for label in labels:
        pair_slice = pairs[pairs["algorithm_batch_mode"].astype(str) == label]
        if pair_slice.empty:
            continue
        combo_values = pair_slice[combo_cols].drop_duplicates()
        if combo_values.empty:
            continue

        stratum_mask = pd.Series(False, index=df.index)
        for _, combo_row in combo_values.iterrows():
            combo_mask = pd.Series(True, index=df.index)
            for column in combo_cols:
                value = combo_row[column]
                normalized_value = "" if pd.isna(value) else str(value).strip().lower()
                combo_mask &= normalized_df_cols[column] == normalized_value
            stratum_mask |= combo_mask
        if not bool(stratum_mask.any()):
            continue

        stratum_df = df[stratum_mask].copy()
        sensitivity_df = _build_sensitivity_summary(
            df=stratum_df,
            metric=metric,
            compare_col=compare_col,
            value_a=value_a,
            value_b=value_b,
            key_cols=key_cols,
            top_k_values=top_k_values,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col=pct_col,
            reference_value=reference_value,
            test_method=test_method,
            location_ci_method=location_ci_method,
            pair_sampling_modes=pair_sampling_modes,
        )
        if sensitivity_df.empty:
            continue
        sensitivity_series_by_stratum[label] = sensitivity_df

    if not sensitivity_series_by_stratum:
        return {}, {}, None

    ordered_labels = list(sensitivity_series_by_stratum.keys())
    color_map = _build_algorithm_batch_mode_color_map(ordered_labels)
    mode_styles: Dict[str, Dict[str, Any]] = {}
    for idx, label in enumerate(ordered_labels):
        mode_styles[label] = {
            "label": label,
            "color": color_map.get(label, DEFAULT_SERIES_COLOR),
            "marker": TOPOLOGY_SENSITIVITY_MARKERS[idx % len(TOPOLOGY_SENSITIVITY_MARKERS)],
        }

    label_counts = pairs["algorithm_batch_mode"].astype(str).value_counts(dropna=False)
    primary_mode = ordered_labels[0]
    primary_count = int(label_counts.get(primary_mode, 0))
    for label in ordered_labels[1:]:
        current_count = int(label_counts.get(label, 0))
        if current_count > primary_count:
            primary_mode = label
            primary_count = current_count

    return sensitivity_series_by_stratum, mode_styles, primary_mode

def _build_topology_algorithm_mode_strata(
    pairs: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
    pct_col: str,
    test_method: str = "paired_t",
    location_ci_method: str = "paired_t",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs_with_strata = pairs.copy()
    pairs_with_strata["algorithm_batch_mode"] = pairs_with_strata.apply(
        lambda row: _topology_stratum_label(
            row.get("algorithm"),
            row.get("pair_batch_mode"),
        ),
        axis=1,
    )
    strata = _summarize_pairs(
        pairs=pairs_with_strata,
        group_cols=["dataset", "algorithm_batch_mode"],
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col=pct_col,
        test_method=test_method,
        location_ci_method=location_ci_method,
    )
    if not strata.empty and "algorithm_batch_mode" in strata.columns:
        strata = _add_group_summary_rows_by_columns(
            summary_df=strata,
            dataset_col="dataset",
            group_cols=["algorithm_batch_mode"],
            pairs=pairs_with_strata,
            pct_col=pct_col,
            test_method=test_method,
            location_ci_method=location_ci_method,
        )
    return pairs_with_strata, strata

def _filter_pairs_to_sampling_modes(
    pairs: pd.DataFrame,
    sampling_modes: Sequence[str],
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()

    working = pairs.copy()
    if "pair_sampling" not in working.columns:
        return working

    allowed_modes = {
        str(mode).strip().lower()
        for mode in sampling_modes
        if str(mode).strip()
    }
    if not allowed_modes:
        return working

    pair_sampling = working["pair_sampling"].astype(str).str.lower().str.strip()
    return working[pair_sampling.isin(allowed_modes)].copy()

def _sampling_mode_display(value: str) -> str:
    text = str(value).strip().lower()
    if text == "full":
        return "Full"
    if text == "random":
        return "Random"
    if text == "hdsssom":
        return "HDSSOM"
    return text.replace("_", " ").title() if text else "Unknown"

def _resolve_dataset_sample_sizes(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
) -> pd.Series:
    if dataset_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    dataset_series = df[dataset_col].astype(str).str.strip().str.lower()
    fallback = dataset_series.map(DATASET_SAMPLE_SIZE_FALLBACK).astype(float)

    best_candidate: Optional[pd.Series] = None
    best_valid_count = -1
    for column in DATASET_SAMPLE_SIZE_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        candidate = candidate.where(candidate > 0)
        valid_count = int(candidate.notna().sum())
        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_candidate = candidate

    if best_candidate is None or best_valid_count <= 0:
        return fallback
    return best_candidate.fillna(fallback)

def _ordered_sampling_modes(values: Iterable[str]) -> List[str]:
    canonical = list(CANONICAL_SAMPLING_MODES)
    seen = {str(v).strip().lower() for v in values if str(v).strip()}
    ordered = [mode for mode in canonical if mode in seen]
    remainder = sorted(mode for mode in seen if mode not in canonical)
    return ordered + remainder

def _canonical_sampling_modes(values: Iterable[str]) -> List[str]:
    allowed = set(CANONICAL_SAMPLING_MODES)
    return [mode for mode in _ordered_sampling_modes(values) if mode in allowed]

def _sampling_comparison_scope_slug(
    available_modes: Iterable[str],
    *,
    focus_comparison_key: Optional[str] = None,
) -> str:
    focus_key = str(focus_comparison_key or "").strip().lower()
    focus_scope_map = {
        "full_vs_random": "full_random_only",
        "full_vs_hdsssom": "full_hdsssom_only",
        "random_vs_hdsssom": "random_hdsssom_only",
    }
    if focus_key:
        return focus_scope_map.get(focus_key, "all_sampling_modes")

    ordered_modes = _ordered_sampling_modes(available_modes)
    if ordered_modes == ["full", "random"]:
        return "full_random_only"
    if ordered_modes == ["full", "hdsssom"]:
        return "full_hdsssom_only"
    if ordered_modes == ["random", "hdsssom"]:
        return "random_hdsssom_only"
    return "all_sampling_modes"

def _sampling_comparison_specs(
    available_modes: Iterable[str],
) -> List[Tuple[str, str, str, str, str]]:
    available = {str(value).strip().lower() for value in available_modes if str(value).strip()}
    comparison_specs: List[Tuple[str, str, str, str, str]] = []
    for comparison_key, comparison_label, value_a, value_b, reference_value in [
        ("full_vs_random", "Full vs Random", "full", "random", "full"),
        ("full_vs_hdsssom", "Full vs HDSSOM", "full", "hdsssom", "full"),
        ("random_vs_hdsssom", "Random vs HDSSOM", "random", "hdsssom", "random"),
    ]:
        if value_a in available and value_b in available:
            comparison_specs.append((comparison_key, comparison_label, value_a, value_b, reference_value))
    return comparison_specs

def _ordered_topology_modes(values: Iterable[str]) -> List[str]:
    canonical = ["hexagonal", "mst", "rng", "grid"]
    seen = {str(v).strip().lower() for v in values if str(v).strip()}
    ordered = [mode for mode in canonical if mode in seen]
    remainder = sorted(mode for mode in seen if mode not in canonical)
    return ordered + remainder

def _qe_panel_label(metric_name: str, fallback_display: str) -> str:
    metric_key = str(metric_name).strip().lower()
    if metric_key in {"balanced_qe_raw", "balanced_qe_normalized"}:
        return "Balanced QE"
    if metric_key in {"quantization_error_holdout", "quantization_error_holdout_normalized"}:
        return "Holdout QE"
    if metric_key in {"quantization_error_train", "quantization_error_train_normalized"}:
        return "Train QE"
    return str(fallback_display)

def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No qualifying rows._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)

def _select_groupwise_best_rows(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_col: str,
    trial_col: Optional[str],
) -> pd.DataFrame:
    required_cols = list(group_cols) + [metric_col]
    working = df.dropna(subset=required_cols).copy()
    if working.empty:
        return pd.DataFrame()

    if trial_col and trial_col in working.columns:
        working["_trial_rank"] = pd.to_numeric(working[trial_col], errors="coerce").fillna(np.inf)
    else:
        working["_trial_rank"] = np.inf

    sort_cols = [metric_col, "_trial_rank"]
    sort_ascending = [True, True]
    if "pair_seed" in working.columns:
        working["_seed_rank"] = working["pair_seed"].astype(str).str.lower().str.strip()
        sort_cols.append("_seed_rank")
        sort_ascending.append(True)

    ranked = working.sort_values(sort_cols, ascending=sort_ascending, kind="mergesort")
    best_rows = ranked.groupby(list(group_cols), dropna=False).head(1).copy()
    return best_rows.drop(columns=["_trial_rank", "_seed_rank"], errors="ignore")

def _is_numeric_param_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null == 0:
        return False
    return coerced.notna().sum() == non_null

def _split_parameter_columns_for_aggregation(
    df: pd.DataFrame,
    param_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for column in param_cols:
        if column not in df.columns:
            continue
        if _is_numeric_param_series(df[column]):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols

def _mode_aggregate(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    norm = valid.astype(str).str.strip().str.lower()
    counts = norm.value_counts(dropna=False)
    if counts.empty:
        return np.nan
    top_count = int(counts.max())
    top_norm_values = sorted([value for value, count in counts.items() if int(count) == top_count])
    winner_norm = top_norm_values[0]
    winners = valid[norm == winner_norm]
    if winners.empty:
        return np.nan
    unique_winners = list(dict.fromkeys(winners.tolist()))
    unique_winners = sorted(unique_winners, key=lambda value: str(value).strip().lower())
    return unique_winners[0]

def _aggregate_best_config_by_groups(
    best_rows: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_col: str,
    numeric_param_cols: Sequence[str],
    categorical_param_cols: Sequence[str],
    mode_meta_cols: Sequence[str],
) -> pd.DataFrame:
    required = list(group_cols) + [metric_col]
    working = best_rows.dropna(subset=required).copy()
    if working.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for group_values, subset in working.groupby(list(group_cols), dropna=False):
        values_tuple = group_values if isinstance(group_values, tuple) else (group_values,)
        row: Dict[str, object] = {column: value for column, value in zip(group_cols, values_tuple)}
        metric_values = pd.to_numeric(subset[metric_col], errors="coerce")
        row[metric_col] = float(metric_values.mean()) if metric_values.notna().any() else np.nan
        row[f"median_{metric_col}"] = float(metric_values.median()) if metric_values.notna().any() else np.nan
        row["rows_aggregated"] = int(len(subset))

        for column in mode_meta_cols:
            if column in subset.columns:
                row[column] = _mode_aggregate(subset[column])

        for column in numeric_param_cols:
            if column not in subset.columns:
                row[column] = np.nan
                continue
            values = pd.to_numeric(subset[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else np.nan

        for column in categorical_param_cols:
            if column not in subset.columns:
                row[column] = np.nan
                continue
            row[column] = _mode_aggregate(subset[column])

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def _to_json_compatible_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        cast = float(value)
        return cast if np.isfinite(cast) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value

def _build_defaults_by_sampling_topology_runner_payload(
    *,
    overall_table: pd.DataFrame,
    include_none_values: bool = False,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    param_cols = sorted([column for column in overall_table.columns if column.startswith("param_")])
    defaults: Dict[str, Dict[str, Dict[str, object]]] = {}

    for _, row in overall_table.iterrows():
        sampling_mode = str(row.get("sampling_mode", "")).strip().lower()
        topology = str(row.get("topology", "")).strip().lower()
        if not sampling_mode or not topology:
            continue

        params: Dict[str, object] = {}
        for param_col in param_cols:
            param_key = str(param_col).removeprefix("param_")
            value = _to_json_compatible_scalar(row.get(param_col))
            if value is None and not include_none_values:
                continue
            params[param_key] = value
        defaults.setdefault(sampling_mode, {})[topology] = params

    return defaults

def _build_defaults_by_sampling_topology_metadata_payload(
    *,
    overall_table: pd.DataFrame,
    metric_col: str,
    dataset_size_filter: str,
    source_rows_after_filter: Optional[int] = None,
) -> Dict[str, object]:
    best_metric_col = f"best_{metric_col}"
    median_metric_col = f"median_{metric_col}"
    param_cols = sorted([column for column in overall_table.columns if column.startswith("param_")])
    metadata_cols = [
        column
        for column in ["algorithm", "batch_mode", best_metric_col, median_metric_col, "rows_aggregated"]
        if column in overall_table.columns
    ]

    defaults: Dict[str, Dict[str, Dict[str, object]]] = {}
    for _, row in overall_table.iterrows():
        sampling_mode = str(row.get("sampling_mode", "")).strip().lower()
        topology = str(row.get("topology", "")).strip().lower()
        if not sampling_mode or not topology:
            continue

        params: Dict[str, object] = {}
        for param_col in param_cols:
            param_key = str(param_col).removeprefix("param_")
            params[param_key] = _to_json_compatible_scalar(row.get(param_col))

        metadata: Dict[str, object] = {}
        for metadata_col in metadata_cols:
            metadata[metadata_col] = _to_json_compatible_scalar(row.get(metadata_col))

        entry = {
            "sampling_mode": sampling_mode,
            "topology": topology,
            "params": params,
            "metadata": metadata,
        }
        defaults.setdefault(sampling_mode, {})[topology] = entry

    payload: Dict[str, object] = {
        "metric": metric_col,
        "stratification": ["sampling_mode", "topology"],
        "dataset_size_filter": dataset_size_filter,
        "defaults": defaults,
    }
    if source_rows_after_filter is not None:
        payload["source_rows_after_filter"] = int(source_rows_after_filter)
    return payload

def _run_optimal_parameter_reporting(
    *,
    tuned_df: pd.DataFrame,
    run_output_dir: Path,
    generated_files: List[str],
    metric_col: str = OPTIMAL_PARAMETER_DEFAULT_METRIC,
) -> Dict[str, object]:
    if metric_col not in tuned_df.columns:
        raise ValueError(
            f"Optimal-parameter report requires metric column '{metric_col}', "
            f"but available columns do not include it."
        )

    required_cols = ["dataset", "architecture", "pair_sampling"]
    missing = [col for col in required_cols if col not in tuned_df.columns]
    if missing:
        raise ValueError(f"Optimal-parameter report requires columns: {missing}")

    working, excluded_minibatch_rows = _filter_full_batch_rows_for_stability(tuned_df)
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working["pair_sampling"] = working["pair_sampling"].astype(str).str.lower().str.strip()
    working["architecture"] = working["architecture"].astype(str).str.lower().str.strip()
    working = working[~working["dataset"].map(_is_global_dataset_label)].copy()
    working = working.dropna(subset=["dataset", "pair_sampling", "architecture", metric_col])
    if working.empty:
        raise ValueError("No qualifying tuned rows remain for optimal-parameter reporting.")

    trial_col = _resolve_column(working, ["trial_number", "trial_id", "number", "pair_trial_number"])
    param_cols = sorted([col for col in working.columns if col.startswith("param_")])
    numeric_param_cols, categorical_param_cols = _split_parameter_columns_for_aggregation(working, param_cols)
    metric_slug = _slug(metric_col)

    unit_group_cols = ["pair_sampling", "dataset", "architecture"]
    if "pair_seed" in working.columns and not working["pair_seed"].isna().all():
        unit_group_cols.append("pair_seed")

    best_unit_rows = _select_groupwise_best_rows(
        working,
        group_cols=unit_group_cols,
        metric_col=metric_col,
        trial_col=trial_col,
    )
    if best_unit_rows.empty:
        raise ValueError("Unable to determine per-unit best rows from tuned data.")

    if "algorithm" not in best_unit_rows.columns:
        best_unit_rows["algorithm"] = "n/a"
    if "pair_batch_mode" not in best_unit_rows.columns:
        best_unit_rows["pair_batch_mode"] = "n/a"

    mode_meta_cols = [column for column in ["algorithm", "pair_batch_mode"] if column in best_unit_rows.columns]
    dataset_table = _aggregate_best_config_by_groups(
        best_rows=best_unit_rows,
        group_cols=["pair_sampling", "dataset", "architecture"],
        metric_col=metric_col,
        numeric_param_cols=numeric_param_cols,
        categorical_param_cols=categorical_param_cols,
        mode_meta_cols=mode_meta_cols,
    )
    best_unit_rows["dataset_type"] = best_unit_rows["dataset"].map(_dataset_group_key)
    dataset_type_table = _aggregate_best_config_by_groups(
        best_rows=best_unit_rows,
        group_cols=["pair_sampling", "dataset_type", "architecture"],
        metric_col=metric_col,
        numeric_param_cols=numeric_param_cols,
        categorical_param_cols=categorical_param_cols,
        mode_meta_cols=mode_meta_cols,
    )
    overall_table = _aggregate_best_config_by_groups(
        best_rows=best_unit_rows,
        group_cols=["pair_sampling", "architecture"],
        metric_col=metric_col,
        numeric_param_cols=numeric_param_cols,
        categorical_param_cols=categorical_param_cols,
        mode_meta_cols=mode_meta_cols,
    )
    if dataset_table.empty or overall_table.empty:
        raise ValueError("Unable to aggregate optimal parameter rows from tuned data.")

    # Additional defaults variant: exclude datasets with fewer than 1000 samples.
    best_unit_rows_large_only = best_unit_rows.copy()
    best_unit_rows_large_only["_dataset_sample_size"] = _resolve_dataset_sample_sizes(
        best_unit_rows_large_only,
        dataset_col="dataset",
    )
    best_unit_rows_large_only = best_unit_rows_large_only[
        pd.to_numeric(best_unit_rows_large_only["_dataset_sample_size"], errors="coerce") >= 1000
    ].copy()
    overall_table_large_only = _aggregate_best_config_by_groups(
        best_rows=best_unit_rows_large_only,
        group_cols=["pair_sampling", "architecture"],
        metric_col=metric_col,
        numeric_param_cols=numeric_param_cols,
        categorical_param_cols=categorical_param_cols,
        mode_meta_cols=mode_meta_cols,
    )

    rename_map = {
        "pair_sampling": "sampling_mode",
        "architecture": "topology",
        "pair_batch_mode": "batch_mode",
        metric_col: f"best_{metric_col}",
    }
    dataset_table = dataset_table.rename(columns=rename_map)
    dataset_type_table = dataset_type_table.rename(columns=rename_map)
    overall_table = overall_table.rename(columns=rename_map)
    overall_table_large_only = overall_table_large_only.rename(columns=rename_map)

    dataset_order = _ordered_dataset_labels(dataset_table["dataset"].tolist())
    sampling_order = _ordered_sampling_modes(dataset_table["sampling_mode"].tolist())
    topology_order = _ordered_topology_modes(dataset_table["topology"].tolist())
    dataset_rank = {name: idx for idx, name in enumerate(dataset_order)}
    sampling_rank = {name: idx for idx, name in enumerate(sampling_order)}
    topology_rank = {name: idx for idx, name in enumerate(topology_order)}

    dataset_table["__dataset_rank"] = dataset_table["dataset"].map(lambda x: dataset_rank.get(x, len(dataset_rank)))
    dataset_table["__sampling_rank"] = dataset_table["sampling_mode"].map(
        lambda x: sampling_rank.get(x, len(sampling_rank))
    )
    dataset_table["__topology_rank"] = dataset_table["topology"].map(lambda x: topology_rank.get(x, len(topology_rank)))
    dataset_table = dataset_table.sort_values(
        ["__sampling_rank", "__dataset_rank", "__topology_rank", f"best_{metric_col}"],
        ascending=[True, True, True, True],
    ).drop(columns=["__dataset_rank", "__sampling_rank", "__topology_rank"])

    if not dataset_type_table.empty:
        dataset_type_rank = {name: _dataset_group_rank(name) for name in dataset_type_table["dataset_type"].tolist()}
        dataset_type_table["__sampling_rank"] = dataset_type_table["sampling_mode"].map(
            lambda value: sampling_rank.get(value, len(sampling_rank))
        )
        dataset_type_table["__dataset_type_rank"] = dataset_type_table["dataset_type"].map(
            lambda value: dataset_type_rank.get(value, len(dataset_type_rank))
        )
        dataset_type_table["__topology_rank"] = dataset_type_table["topology"].map(
            lambda value: topology_rank.get(value, len(topology_rank))
        )
        dataset_type_table = dataset_type_table.sort_values(
            ["__sampling_rank", "__dataset_type_rank", "__topology_rank", f"best_{metric_col}"],
            ascending=[True, True, True, True],
        ).drop(columns=["__sampling_rank", "__dataset_type_rank", "__topology_rank"])

    overall_sampling_order = _ordered_sampling_modes(overall_table["sampling_mode"].tolist())
    overall_topology_order = _ordered_topology_modes(overall_table["topology"].tolist())
    overall_sampling_rank = {name: idx for idx, name in enumerate(overall_sampling_order)}
    overall_topology_rank = {name: idx for idx, name in enumerate(overall_topology_order)}
    overall_table["__sampling_rank"] = overall_table["sampling_mode"].map(
        lambda x: overall_sampling_rank.get(x, len(overall_sampling_rank))
    )
    overall_table["__topology_rank"] = overall_table["topology"].map(
        lambda x: overall_topology_rank.get(x, len(overall_topology_rank))
    )
    overall_table = overall_table.sort_values(
        ["__sampling_rank", "__topology_rank", f"best_{metric_col}"],
        ascending=[True, True, True],
    ).drop(columns=["__sampling_rank", "__topology_rank"])
    if not overall_table_large_only.empty:
        overall_table_large_only["__sampling_rank"] = overall_table_large_only["sampling_mode"].map(
            lambda x: overall_sampling_rank.get(x, len(overall_sampling_rank))
        )
        overall_table_large_only["__topology_rank"] = overall_table_large_only["topology"].map(
            lambda x: overall_topology_rank.get(x, len(overall_topology_rank))
        )
        overall_table_large_only = overall_table_large_only.sort_values(
            ["__sampling_rank", "__topology_rank", f"best_{metric_col}"],
            ascending=[True, True, True],
        ).drop(columns=["__sampling_rank", "__topology_rank"])

    report_dir = run_output_dir / "optimal_parameters"
    tables_dir = report_dir / "tables"
    summary_dir = report_dir / "summary_outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = tables_dir / f"optimal_parameters_by_sampling_dataset_topology_{metric_slug}.csv"
    dataset_type_csv = tables_dir / f"optimal_parameters_by_sampling_dataset_type_topology_{metric_slug}.csv"
    overall_csv = tables_dir / f"optimal_parameters_overall_sampling_topology_{metric_slug}.csv"
    overall_csv_min1000 = tables_dir / f"optimal_parameters_overall_sampling_topology_min1000_{metric_slug}.csv"
    defaults_json = summary_dir / f"defaults_by_sampling_topology_{metric_slug}.json"
    defaults_json_min1000 = summary_dir / f"defaults_by_sampling_topology_min1000_{metric_slug}.json"
    defaults_metadata_json = summary_dir / f"defaults_by_sampling_topology_metadata_{metric_slug}.json"
    defaults_metadata_json_min1000 = summary_dir / f"defaults_by_sampling_topology_metadata_min1000_{metric_slug}.json"
    markdown_path = summary_dir / OPTIMAL_PARAMETER_MARKDOWN_FILENAME
    dataset_table.to_csv(dataset_csv, index=False)
    dataset_type_table.to_csv(dataset_type_csv, index=False)
    overall_table.to_csv(overall_csv, index=False)
    overall_table_large_only.to_csv(overall_csv_min1000, index=False)
    defaults_payload = _build_defaults_by_sampling_topology_runner_payload(
        overall_table=overall_table,
    )
    defaults_json.write_text(json.dumps(defaults_payload, indent=2), encoding="utf-8")
    defaults_payload_min1000 = _build_defaults_by_sampling_topology_runner_payload(
        overall_table=overall_table_large_only,
    )
    defaults_json_min1000.write_text(json.dumps(defaults_payload_min1000, indent=2), encoding="utf-8")
    defaults_metadata_payload = _build_defaults_by_sampling_topology_metadata_payload(
        overall_table=overall_table,
        metric_col=metric_col,
        dataset_size_filter="none",
    )
    defaults_metadata_json.write_text(json.dumps(defaults_metadata_payload, indent=2), encoding="utf-8")
    defaults_metadata_payload_min1000 = _build_defaults_by_sampling_topology_metadata_payload(
        overall_table=overall_table_large_only,
        metric_col=metric_col,
        dataset_size_filter="n_samples >= 1000",
        source_rows_after_filter=int(len(best_unit_rows_large_only)),
    )
    defaults_metadata_json_min1000.write_text(json.dumps(defaults_metadata_payload_min1000, indent=2), encoding="utf-8")

    markdown_lines: List[str] = []
    markdown_lines.append("# Optimal Parameter Tables")
    markdown_lines.append("")
    markdown_lines.append(
        "Selection rule: first, select best trial per unit "
        "`(sampling, dataset, topology, seed)` by minimum "
        f"`{metric_col}` (ties: lowest trial number, then seed). "
        "Then aggregate to report configs: numeric `param_*` use mean; categorical `param_*` use mode "
        "(ties broken deterministically by alphabetical value). Rows with `pair_batch_mode=minibatch` are excluded."
    )
    markdown_lines.append("")
    markdown_lines.append("## Per-Sampling Best by Dataset and Topology")
    markdown_lines.append("")
    if not param_cols:
        markdown_lines.append("_No `param_` columns were detected; metadata-only rows are shown._")
        markdown_lines.append("")

    for sampling_mode in _ordered_sampling_modes(dataset_table["sampling_mode"].tolist()):
        subset = dataset_table[dataset_table["sampling_mode"] == sampling_mode].copy()
        if subset.empty:
            continue
        markdown_lines.append(f"### Sampling: {_sampling_mode_display(sampling_mode)}")
        markdown_lines.append("")
        markdown_lines.append(_dataframe_to_markdown_table(subset.drop(columns=["sampling_mode"])))
        markdown_lines.append("")

    markdown_lines.append("## Per-Sampling Best by Dataset Type and Topology")
    markdown_lines.append("")
    markdown_lines.append(_dataframe_to_markdown_table(dataset_type_table))
    markdown_lines.append("")

    markdown_lines.append("## Overall Best (Stratified by Topology and Sampling Method)")
    markdown_lines.append("")
    markdown_lines.append(_dataframe_to_markdown_table(overall_table))
    markdown_lines.append("")
    markdown_lines.append("## Machine-Usable Defaults JSON")
    markdown_lines.append("")
    markdown_lines.append(
        f"- `{defaults_json.resolve()}` (runner-compatible for `--fixed-params-by-sampling-topology-json`)"
    )
    markdown_lines.append(
        f"- `{defaults_json_min1000.resolve()}` (runner-compatible; datasets with `n_samples >= 1000` only)"
    )
    markdown_lines.append(f"- `{defaults_metadata_json.resolve()}` (metadata-rich export)")
    markdown_lines.append(
        f"- `{defaults_metadata_json_min1000.resolve()}` (metadata-rich export; datasets with `n_samples >= 1000` only)"
    )
    markdown_lines.append("")
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    generated_files.extend(
        [
            str(dataset_csv.resolve()),
            str(dataset_type_csv.resolve()),
            str(overall_csv.resolve()),
            str(overall_csv_min1000.resolve()),
            str(defaults_json.resolve()),
            str(defaults_json_min1000.resolve()),
            str(defaults_metadata_json.resolve()),
            str(defaults_metadata_json_min1000.resolve()),
            str(markdown_path.resolve()),
        ]
    )

    return {
        "enabled": True,
        "metric": metric_col,
        "rows_total": int(len(tuned_df)),
        "rows_used": int(len(working)),
        "rows_excluded_minibatch": int(excluded_minibatch_rows),
        "output_dir": str(report_dir.resolve()),
        "dataset_topology_table_csv": str(dataset_csv.resolve()),
        "dataset_type_topology_table_csv": str(dataset_type_csv.resolve()),
        "overall_table_csv": str(overall_csv.resolve()),
        "overall_table_min1000_csv": str(overall_csv_min1000.resolve()),
        "defaults_json": str(defaults_json.resolve()),
        "defaults_min1000_json": str(defaults_json_min1000.resolve()),
        "defaults_metadata_json": str(defaults_metadata_json.resolve()),
        "defaults_metadata_min1000_json": str(defaults_metadata_json_min1000.resolve()),
        "markdown_report": str(markdown_path.resolve()),
    }

def _build_sampling_mode_outcome_summary(
    df: pd.DataFrame,
    metric: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    reference_value: Optional[str],
    key_cols: Sequence[str],
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    pct_col: str = "pct_improvement_a_over_b_reference",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    pair_rows: List[pd.DataFrame] = []

    sampling_values = _ordered_sampling_modes(df["pair_sampling"].astype(str).dropna().unique().tolist())
    for sampling_mode in sampling_values:
        subset = df[df["pair_sampling"].astype(str).str.lower().str.strip() == sampling_mode].copy()
        if subset.empty:
            continue

        pairs = _build_pairs(
            df=subset,
            metric=metric,
            compare_col=compare_col,
            value_a=value_a,
            value_b=value_b,
            key_cols=key_cols,
            top_k=top_k,
            reference_value=reference_value,
        )
        if pairs.empty:
            continue

        summary = _summarize_pairs(
            pairs=pairs,
            group_cols=["dataset"],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col=pct_col,
        )
        summary = _add_global_row(
            summary_df=summary,
            pairs=pairs,
            dataset_col="dataset",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col=pct_col,
        )
        global_rows = summary[summary["dataset"] == dataset_groups.GLOBAL_OVERALL_LABEL]
        if global_rows.empty:
            continue
        global_row = global_rows.iloc[0]

        rows.append(
            {
                "pair_sampling": sampling_mode,
                "sampling_label": _sampling_mode_display(sampling_mode),
                "n_pairs": int(global_row["n_pairs"]),
                "median_pct": float(global_row["median_pct"]),
                "ci_low_pct": float(global_row["ci_low_pct"]),
                "ci_high_pct": float(global_row["ci_high_pct"]),
                "p_value": float(global_row["p_value"]),
                "effect_size_signed": float(global_row["effect_size_signed"]),
            }
        )

        pair_key_cols = [col for col in key_cols if col in pairs.columns]
        pair_frame = pairs[pair_key_cols + [pct_col]].copy()
        pair_frame["pair_sampling"] = sampling_mode
        pair_rows.append(pair_frame)

    summary_df = pd.DataFrame(rows)
    if not pair_rows:
        return summary_df, pd.DataFrame()

    combined_pairs = pd.concat(pair_rows, ignore_index=True)
    compare_key_cols = [
        col
        for col in combined_pairs.columns
        if col not in {"pair_sampling", pct_col}
    ]

    pairwise_rows: List[Dict[str, object]] = []
    present_modes = _ordered_sampling_modes(combined_pairs["pair_sampling"].astype(str).unique().tolist())
    for idx_a in range(len(present_modes)):
        for idx_b in range(idx_a + 1, len(present_modes)):
            mode_a = present_modes[idx_a]
            mode_b = present_modes[idx_b]
            a_df = (
                combined_pairs[combined_pairs["pair_sampling"] == mode_a][compare_key_cols + [pct_col]]
                .rename(columns={pct_col: "outcome_a"})
            )
            b_df = (
                combined_pairs[combined_pairs["pair_sampling"] == mode_b][compare_key_cols + [pct_col]]
                .rename(columns={pct_col: "outcome_b"})
            )
            merged = a_df.merge(b_df, on=compare_key_cols, how="inner")
            if merged.empty:
                continue

            delta = merged["outcome_a"].to_numpy(dtype=float) - merged["outcome_b"].to_numpy(dtype=float)
            location_delta, ci_low, ci_high = _paired_t_mean_ci(delta)
            signed = _signed_test_stats(delta)
            pairwise_rows.append(
                {
                    "sampling_a": mode_a,
                    "sampling_b": mode_b,
                    "sampling_a_label": _sampling_mode_display(mode_a),
                    "sampling_b_label": _sampling_mode_display(mode_b),
                    "n_pairs": int(signed["n_pairs"]),
                    "median_delta_outcome_pct_points": location_delta,
                    "mean_delta_outcome_pct_points": float(np.nanmean(delta)),
                    "ci_low_delta": float(ci_low),
                    "ci_high_delta": float(ci_high),
                    "p_value": float(signed["p_value"]),
                    "effect_size_signed": float(signed["effect_size_signed"]),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    return summary_df, pairwise_df

def _format_stability_parameter_label(parameter: object) -> str:
    text = str(parameter).strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    if lowered.startswith("overall"):
        return "Overall (4 params)"
    compact_labels = {
        "param_initial_radius": "Initial Radius",
        "param_initialization_method": "Init Method",
        "param_radius_decay_type": "Decay Type",
        "param_use_momentum": "Use Momentum",
    }
    if lowered in compact_labels:
        return compact_labels[lowered]
    if lowered.startswith("param_"):
        text = text[len("param_") :]
    return text.replace("_", " ").title()

def _build_selected_parameter_stability_table(
    parameter_global_df: pd.DataFrame,
    selected_parameters: Sequence[str],
    *,
    topology_left: str,
    topology_right: str,
) -> pd.DataFrame:
    if parameter_global_df.empty:
        return pd.DataFrame()
    if not selected_parameters:
        return pd.DataFrame()

    required_cols = [
        "parameter",
        "parameter_type",
        "seed_pairs_left",
        "seed_pairs_right",
        "stability_score_mean_left",
        "stability_score_mean_right",
        "stability_score_median_left",
        "stability_score_median_right",
    ]
    missing = [col for col in required_cols if col not in parameter_global_df.columns]
    if missing:
        return pd.DataFrame()

    selected = parameter_global_df[
        parameter_global_df["parameter"].astype(str).isin(list(selected_parameters))
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["parameter"] = selected["parameter"].astype(str)
    selected["_parameter_order"] = selected["parameter"].map(
        {name: idx for idx, name in enumerate(selected_parameters)}
    )
    selected = selected.sort_values(["_parameter_order", "parameter"]).drop(columns=["_parameter_order"])
    selected = selected.drop_duplicates(subset=["parameter"], keep="first")

    for column in [
        "seed_pairs_left",
        "seed_pairs_right",
        "stability_score_mean_left",
        "stability_score_mean_right",
        "stability_score_median_left",
        "stability_score_median_right",
    ]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected["delta_left_minus_right"] = (
        selected["stability_score_mean_left"] - selected["stability_score_mean_right"]
    )
    selected["winner"] = selected.apply(
        lambda row: (
            "insufficient_data"
            if pd.isna(row["stability_score_mean_left"]) or pd.isna(row["stability_score_mean_right"])
            else (
                "tie"
                if np.isclose(
                    float(row["stability_score_mean_left"]),
                    float(row["stability_score_mean_right"]),
                    rtol=1e-9,
                    atol=1e-12,
                )
                else (
                    topology_left
                    if float(row["stability_score_mean_left"]) < float(row["stability_score_mean_right"])
                    else topology_right
                )
            )
        ),
        axis=1,
    )
    selected["parameters_compared"] = np.nan

    mean_left = float(pd.to_numeric(selected["stability_score_mean_left"], errors="coerce").mean())
    mean_right = float(pd.to_numeric(selected["stability_score_mean_right"], errors="coerce").mean())
    overall_winner = (
        "insufficient_data"
        if not np.isfinite(mean_left) or not np.isfinite(mean_right)
        else (
            "tie"
            if np.isclose(mean_left, mean_right, rtol=1e-9, atol=1e-12)
            else (topology_left if mean_left < mean_right else topology_right)
        )
    )
    overall_row = pd.DataFrame(
        [
            {
                "global_label": "GLOBAL",
                "topology_left": topology_left,
                "topology_right": topology_right,
                "parameter": "OVERALL_SELECTED",
                "parameter_type": "overall",
                "seed_pairs_left": float(
                    pd.to_numeric(selected["seed_pairs_left"], errors="coerce").mean()
                ),
                "seed_pairs_right": float(
                    pd.to_numeric(selected["seed_pairs_right"], errors="coerce").mean()
                ),
                "stability_score_mean_left": mean_left,
                "stability_score_mean_right": mean_right,
                "stability_score_median_left": float(
                    pd.to_numeric(selected["stability_score_median_left"], errors="coerce").mean()
                ),
                "stability_score_median_right": float(
                    pd.to_numeric(selected["stability_score_median_right"], errors="coerce").mean()
                ),
                "delta_left_minus_right": mean_left - mean_right,
                "winner": overall_winner,
                "parameters_compared": int(selected["parameter"].nunique()),
            }
        ]
    )

    available_cols = [
        "global_label",
        "topology_left",
        "topology_right",
        "parameter",
        "parameter_type",
        "seed_pairs_left",
        "seed_pairs_right",
        "stability_score_mean_left",
        "stability_score_mean_right",
        "stability_score_median_left",
        "stability_score_median_right",
        "delta_left_minus_right",
        "winner",
        "parameters_compared",
    ]
    aligned_selected = selected[[col for col in available_cols if col in selected.columns]].copy()
    aligned_overall = overall_row[[col for col in available_cols if col in overall_row.columns]].copy()
    return pd.concat([aligned_selected, aligned_overall], ignore_index=True)

def _build_selected_parameter_stability_three_topology_table(
    parameter_global_hex_mst_df: pd.DataFrame,
    parameter_global_hex_rng_df: pd.DataFrame,
    selected_parameters: Sequence[str],
) -> pd.DataFrame:
    if parameter_global_hex_mst_df.empty or parameter_global_hex_rng_df.empty:
        return pd.DataFrame()
    if not selected_parameters:
        return pd.DataFrame()

    required_cols = [
        "parameter",
        "stability_score_mean_left",
        "stability_score_mean_right",
    ]
    missing_hex_mst = [col for col in required_cols if col not in parameter_global_hex_mst_df.columns]
    missing_hex_rng = [col for col in required_cols if col not in parameter_global_hex_rng_df.columns]
    if missing_hex_mst or missing_hex_rng:
        raise ValueError(
            "Selected three-topology stability table requires columns "
            f"{required_cols}; missing hex_vs_mst={missing_hex_mst}, missing hex_vs_rng={missing_hex_rng}"
        )

    hex_mst = parameter_global_hex_mst_df.copy()
    hex_rng = parameter_global_hex_rng_df.copy()
    hex_mst = hex_mst[~hex_mst["parameter"].astype(str).str.lower().str.startswith("overall")].copy()
    hex_rng = hex_rng[~hex_rng["parameter"].astype(str).str.lower().str.startswith("overall")].copy()
    hex_mst["parameter"] = hex_mst["parameter"].astype(str).str.strip()
    hex_rng["parameter"] = hex_rng["parameter"].astype(str).str.strip()
    hex_mst = hex_mst.drop_duplicates(subset=["parameter"], keep="first")
    hex_rng = hex_rng.drop_duplicates(subset=["parameter"], keep="first")

    parameter_lookup_hex_mst = {
        str(row["parameter"]).strip(): row for _, row in hex_mst.iterrows()
    }
    parameter_lookup_hex_rng = {
        str(row["parameter"]).strip(): row for _, row in hex_rng.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for parameter in selected_parameters:
        parameter_key = str(parameter).strip()
        row_hex_mst = parameter_lookup_hex_mst.get(parameter_key)
        row_hex_rng = parameter_lookup_hex_rng.get(parameter_key)
        if row_hex_mst is None or row_hex_rng is None:
            continue

        hex_candidates = [
            pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_left")]), errors="coerce").iloc[0],
            pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_left")]), errors="coerce").iloc[0],
        ]
        hex_values = [float(v) for v in hex_candidates if pd.notna(v)]
        hex_mean = float(np.mean(hex_values)) if hex_values else float("nan")
        mst_mean = float(
            pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_right")]), errors="coerce").iloc[0]
        )
        rng_mean = float(
            pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_right")]), errors="coerce").iloc[0]
        )

        if not (np.isfinite(hex_mean) or np.isfinite(mst_mean) or np.isfinite(rng_mean)):
            continue
        rows.append(
            {
                "parameter": parameter_key,
                "parameter_label": _format_stability_parameter_label(parameter_key),
                "is_overall": False,
                "stability_score_mean_hexagonal": hex_mean,
                "stability_score_mean_mst": mst_mean,
                "stability_score_mean_rng": rng_mean,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    overall_row = pd.DataFrame(
        [
            {
                "parameter": "OVERALL_SELECTED",
                "parameter_label": _format_stability_parameter_label("OVERALL_SELECTED"),
                "is_overall": True,
                "stability_score_mean_hexagonal": float(
                    pd.to_numeric(out["stability_score_mean_hexagonal"], errors="coerce").mean()
                ),
                "stability_score_mean_mst": float(
                    pd.to_numeric(out["stability_score_mean_mst"], errors="coerce").mean()
                ),
                "stability_score_mean_rng": float(
                    pd.to_numeric(out["stability_score_mean_rng"], errors="coerce").mean()
                ),
            }
        ]
    )
    out = pd.concat([out, overall_row], ignore_index=True)
    return out

def _build_dataset_selected_parameter_stability_three_topology_table(
    parameter_dataset_hex_mst_df: pd.DataFrame,
    parameter_dataset_hex_rng_df: pd.DataFrame,
    selected_parameters: Sequence[str],
) -> pd.DataFrame:
    if parameter_dataset_hex_mst_df.empty or parameter_dataset_hex_rng_df.empty:
        return pd.DataFrame()
    if not selected_parameters:
        return pd.DataFrame()

    required_cols = [
        "dataset",
        "parameter",
        "stability_score_mean_left",
        "stability_score_mean_right",
    ]
    missing_hex_mst = [col for col in required_cols if col not in parameter_dataset_hex_mst_df.columns]
    missing_hex_rng = [col for col in required_cols if col not in parameter_dataset_hex_rng_df.columns]
    if missing_hex_mst or missing_hex_rng:
        raise ValueError(
            "Dataset selected-parameter three-topology stability table requires columns "
            f"{required_cols}; missing hex_vs_mst={missing_hex_mst}, missing hex_vs_rng={missing_hex_rng}"
        )

    selected_keys = [str(parameter).strip() for parameter in selected_parameters if str(parameter).strip()]
    if not selected_keys:
        return pd.DataFrame()
    selected_lookup = set(selected_keys)

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["dataset"] = prepared["dataset"].astype(str).str.strip()
        prepared["parameter"] = prepared["parameter"].astype(str).str.strip()
        prepared = prepared[~prepared["dataset"].map(_is_global_dataset_label)].copy()
        prepared = prepared[prepared["dataset"] != ""].copy()
        prepared = prepared[prepared["parameter"].isin(selected_lookup)].copy()
        prepared = prepared.drop_duplicates(subset=["dataset", "parameter"], keep="first")
        return prepared

    hex_mst = _prepare(parameter_dataset_hex_mst_df)
    hex_rng = _prepare(parameter_dataset_hex_rng_df)
    if hex_mst.empty or hex_rng.empty:
        return pd.DataFrame()

    shared_datasets = set(hex_mst["dataset"].astype(str).tolist()) & set(hex_rng["dataset"].astype(str).tolist())
    if not shared_datasets:
        return pd.DataFrame()

    ordered_datasets = _ordered_dataset_labels(list(shared_datasets))
    ordered_dataset_lookup = set(ordered_datasets)
    dataset_order = ordered_datasets + sorted(dataset for dataset in shared_datasets if dataset not in ordered_dataset_lookup)

    hex_mst_lookup = {
        (str(row["dataset"]).strip(), str(row["parameter"]).strip()): row
        for _, row in hex_mst.iterrows()
    }
    hex_rng_lookup = {
        (str(row["dataset"]).strip(), str(row["parameter"]).strip()): row
        for _, row in hex_rng.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for dataset in dataset_order:
        per_parameter_rows: List[Tuple[float, float, float]] = []
        for parameter in selected_keys:
            row_hex_mst = hex_mst_lookup.get((dataset, parameter))
            row_hex_rng = hex_rng_lookup.get((dataset, parameter))
            if row_hex_mst is None or row_hex_rng is None:
                continue

            hex_candidates = [
                pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_left")]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_left")]), errors="coerce").iloc[0],
            ]
            hex_values = [float(value) for value in hex_candidates if pd.notna(value)]
            hex_mean = float(np.mean(hex_values)) if hex_values else float("nan")
            mst_mean = float(
                pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_right")]), errors="coerce").iloc[0]
            )
            rng_mean = float(
                pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_right")]), errors="coerce").iloc[0]
            )
            if not (np.isfinite(hex_mean) or np.isfinite(mst_mean) or np.isfinite(rng_mean)):
                continue
            per_parameter_rows.append((hex_mean, mst_mean, rng_mean))

        if not per_parameter_rows:
            continue

        rows.append(
            {
                "dataset": dataset,
                "n_selected_parameters": int(len(per_parameter_rows)),
                "stability_score_mean_hexagonal": float(np.mean([row[0] for row in per_parameter_rows])),
                "stability_score_mean_mst": float(np.mean([row[1] for row in per_parameter_rows])),
                "stability_score_mean_rng": float(np.mean([row[2] for row in per_parameter_rows])),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def _dataset_type_display_label(value: object) -> str:
    key = str(value).strip().lower()
    if key == "synthetic":
        return "Synthetic"
    if key == "real":
        return "Non-synthetic"
    if not key:
        return "Unknown"
    return str(value).strip()

def _build_dataset_type_stability_three_topology_table(
    parameter_dataset_type_hex_mst_df: pd.DataFrame,
    parameter_dataset_type_hex_rng_df: pd.DataFrame,
) -> pd.DataFrame:
    if parameter_dataset_type_hex_mst_df.empty or parameter_dataset_type_hex_rng_df.empty:
        return pd.DataFrame()

    required_cols = [
        "dataset_type",
        "parameter",
        "stability_score_mean_left",
        "stability_score_mean_right",
    ]
    missing_hex_mst = [col for col in required_cols if col not in parameter_dataset_type_hex_mst_df.columns]
    missing_hex_rng = [col for col in required_cols if col not in parameter_dataset_type_hex_rng_df.columns]
    if missing_hex_mst or missing_hex_rng:
        raise ValueError(
            "Dataset-type three-topology stability table requires columns "
            f"{required_cols}; missing hex_vs_mst={missing_hex_mst}, missing hex_vs_rng={missing_hex_rng}"
        )

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["dataset_type"] = prepared["dataset_type"].astype(str).str.strip().str.lower()
        prepared["parameter"] = prepared["parameter"].astype(str).str.strip().str.upper()
        prepared = prepared[prepared["parameter"] == "OVERALL"].copy()
        prepared = prepared.dropna(subset=["dataset_type"])
        prepared = prepared[prepared["dataset_type"] != ""].copy()
        prepared = prepared.drop_duplicates(subset=["dataset_type"], keep="first")
        return prepared

    hex_mst = _prepare(parameter_dataset_type_hex_mst_df)
    hex_rng = _prepare(parameter_dataset_type_hex_rng_df)
    if hex_mst.empty or hex_rng.empty:
        return pd.DataFrame()

    dataset_types = [key for key in ["synthetic", "real"] if key in set(hex_mst["dataset_type"]) and key in set(hex_rng["dataset_type"])]
    if not dataset_types:
        return pd.DataFrame()

    hex_mst_lookup = {str(row["dataset_type"]).strip().lower(): row for _, row in hex_mst.iterrows()}
    hex_rng_lookup = {str(row["dataset_type"]).strip().lower(): row for _, row in hex_rng.iterrows()}

    rows: List[Dict[str, object]] = []
    for dataset_type in dataset_types:
        row_hex_mst = hex_mst_lookup.get(dataset_type)
        row_hex_rng = hex_rng_lookup.get(dataset_type)
        if row_hex_mst is None or row_hex_rng is None:
            continue

        hex_candidates = [
            pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_left")]), errors="coerce").iloc[0],
            pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_left")]), errors="coerce").iloc[0],
        ]
        hex_values = [float(v) for v in hex_candidates if pd.notna(v)]
        hex_mean = float(np.mean(hex_values)) if hex_values else float("nan")
        mst_mean = float(
            pd.to_numeric(pd.Series([row_hex_mst.get("stability_score_mean_right")]), errors="coerce").iloc[0]
        )
        rng_mean = float(
            pd.to_numeric(pd.Series([row_hex_rng.get("stability_score_mean_right")]), errors="coerce").iloc[0]
        )
        if not (np.isfinite(hex_mean) or np.isfinite(mst_mean) or np.isfinite(rng_mean)):
            continue

        rows.append(
            {
                "parameter": f"DATASET_TYPE_{dataset_type.upper()}",
                "parameter_label": _dataset_type_display_label(dataset_type),
                "is_overall": False,
                "stability_score_mean_hexagonal": hex_mean,
                "stability_score_mean_mst": mst_mean,
                "stability_score_mean_rng": rng_mean,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    overall_row = pd.DataFrame(
        [
            {
                "parameter": "OVERALL_DATASET_TYPE",
                "parameter_label": "Overall (Synthetic + Non-synthetic)",
                "is_overall": True,
                "stability_score_mean_hexagonal": float(
                    pd.to_numeric(out["stability_score_mean_hexagonal"], errors="coerce").mean()
                ),
                "stability_score_mean_mst": float(
                    pd.to_numeric(out["stability_score_mean_mst"], errors="coerce").mean()
                ),
                "stability_score_mean_rng": float(
                    pd.to_numeric(out["stability_score_mean_rng"], errors="coerce").mean()
                ),
            }
        ]
    )
    out = pd.concat([out, overall_row], ignore_index=True)
    return out

def _filter_full_batch_rows_for_stability(tuned_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if tuned_df.empty:
        return tuned_df.copy(), 0
    if "pair_batch_mode" not in tuned_df.columns:
        return tuned_df.copy(), 0
    batch_mode = tuned_df["pair_batch_mode"].astype(str).str.lower().str.strip()
    mask = batch_mode != "minibatch"
    removed = int((~mask).sum())
    return tuned_df.loc[mask].copy(), removed

def _prepare_tuned_vs_default_dataset_summary_for_forest(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    required_cols = ["dataset", "pairs", "mean_pct", "ci_low_pct", "ci_high_pct", "pct_p_value"]
    missing = [col for col in required_cols if col not in summary_df.columns]
    if missing:
        raise ValueError(
            f"Tuned-vs-default dataset summary missing required columns: {missing}"
        )

    working = summary_df.copy()
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working["n_pairs"] = pd.to_numeric(working["pairs"], errors="coerce")
    working["mean_pct"] = pd.to_numeric(working["mean_pct"], errors="coerce")
    working["median_pct"] = working["mean_pct"]
    working["ci_low_pct"] = pd.to_numeric(working["ci_low_pct"], errors="coerce")
    working["ci_high_pct"] = pd.to_numeric(working["ci_high_pct"], errors="coerce")
    working["p_value"] = pd.to_numeric(working["pct_p_value"], errors="coerce")

    # Keep rows plottable even when CI is unavailable (e.g., single pair).
    for ci_col in ["ci_low_pct", "ci_high_pct"]:
        working[ci_col] = working[ci_col].where(
            np.isfinite(pd.to_numeric(working[ci_col], errors="coerce")),
            working["median_pct"],
        )
    low = np.minimum(
        pd.to_numeric(working["ci_low_pct"], errors="coerce"),
        pd.to_numeric(working["ci_high_pct"], errors="coerce"),
    )
    high = np.maximum(
        pd.to_numeric(working["ci_low_pct"], errors="coerce"),
        pd.to_numeric(working["ci_high_pct"], errors="coerce"),
    )
    working["ci_low_pct"] = low
    working["ci_high_pct"] = high

    plot_cols = [
        "dataset",
        "n_pairs",
        "median_pct",
        "mean_pct",
        "ci_low_pct",
        "ci_high_pct",
        "p_value",
    ]
    plot_df = working[plot_cols].copy()
    plot_df = plot_df.dropna(subset=["dataset", "median_pct", "ci_low_pct", "ci_high_pct"])
    if plot_df.empty:
        return plot_df
    return _apply_q_values(plot_df, dataset_col="dataset")

def _prepare_sampling_dataset_summary_for_forest(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    required_cols = ["dataset", "n_pairs", "median_pct", "ci_low_pct", "ci_high_pct", "p_value"]
    missing = [col for col in required_cols if col not in summary_df.columns]
    if missing:
        raise ValueError(f"Sampling dataset summary missing required columns: {missing}")

    working = summary_df.copy()
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working["n_pairs"] = pd.to_numeric(working["n_pairs"], errors="coerce")
    working["median_pct"] = pd.to_numeric(working["median_pct"], errors="coerce")
    working["ci_low_pct"] = pd.to_numeric(working["ci_low_pct"], errors="coerce")
    working["ci_high_pct"] = pd.to_numeric(working["ci_high_pct"], errors="coerce")
    working["p_value"] = pd.to_numeric(working["p_value"], errors="coerce")

    # Keep rows plottable even when CI is unavailable (e.g., single pair).
    for ci_col in ["ci_low_pct", "ci_high_pct"]:
        working[ci_col] = working[ci_col].where(
            np.isfinite(pd.to_numeric(working[ci_col], errors="coerce")),
            working["median_pct"],
        )
    low = np.minimum(
        pd.to_numeric(working["ci_low_pct"], errors="coerce"),
        pd.to_numeric(working["ci_high_pct"], errors="coerce"),
    )
    high = np.maximum(
        pd.to_numeric(working["ci_low_pct"], errors="coerce"),
        pd.to_numeric(working["ci_high_pct"], errors="coerce"),
    )
    working["ci_low_pct"] = low
    working["ci_high_pct"] = high

    plot_cols = ["dataset", "n_pairs", "median_pct", "ci_low_pct", "ci_high_pct", "p_value"]
    plot_df = working[plot_cols].copy()
    plot_df = plot_df.dropna(subset=["dataset", "median_pct", "ci_low_pct", "ci_high_pct"])
    if plot_df.empty:
        return plot_df
    return _apply_q_values(plot_df, dataset_col="dataset")

def _merge_tuned_vs_default_global_row(
    dataset_summary_df: pd.DataFrame,
    global_summary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if dataset_summary_df.empty:
        return dataset_summary_df.copy()

    dataset_rows = dataset_summary_df.copy()
    if "dataset" not in dataset_rows.columns:
        return dataset_rows

    dataset_rows["dataset"] = dataset_rows["dataset"].astype(str).str.strip()
    dataset_rows = dataset_rows[~dataset_rows["dataset"].map(_is_global_dataset_label)].copy()

    if global_summary_df is None or global_summary_df.empty:
        return dataset_rows

    global_rows = global_summary_df.copy()
    if "dataset" not in global_rows.columns:
        if "global_label" in global_rows.columns:
            global_rows = global_rows.rename(columns={"global_label": "dataset"})
        else:
            return dataset_rows
    global_rows["dataset"] = global_rows["dataset"].astype(str).str.strip()
    global_rows.loc[
        global_rows["dataset"].str.upper() == "GLOBAL",
        "dataset",
    ] = dataset_groups.GLOBAL_OVERALL_LABEL
    global_rows = global_rows[global_rows["dataset"].map(_is_global_dataset_label)].copy()
    if global_rows.empty:
        return dataset_rows

    overall_rows = global_rows[
        global_rows["dataset"].astype(str).str.upper() == dataset_groups.GLOBAL_OVERALL_LABEL
    ].copy()
    chosen_global_row = overall_rows.iloc[[0]].copy() if not overall_rows.empty else global_rows.iloc[[0]].copy()
    chosen_global_row["dataset"] = dataset_groups.GLOBAL_OVERALL_LABEL

    for column in dataset_rows.columns:
        if column not in chosen_global_row.columns:
            chosen_global_row[column] = np.nan
    chosen_global_row = chosen_global_row[dataset_rows.columns]
    return pd.concat([dataset_rows, chosen_global_row], ignore_index=True)

def _normalize_source_path_for_compare(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text

__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
