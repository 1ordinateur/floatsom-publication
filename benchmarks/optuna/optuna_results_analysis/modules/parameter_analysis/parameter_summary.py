#!/usr/bin/env python3
"""
Utilities for constructing parameter summary tables based on top-performing Optuna trials.

The workflow:
1. Rank scenarios within each dataset/algorithm/sampling bucket by a target metric.
2. Retain the top-k rows (lower scores are better) per dataset and collapse their parameter
   values using the median for numeric fields and the mode for categorical ones.
3. For the overall view, identify the single best row per dataset (still keeping lower-is-better)
   within an algorithm/sampling group, then aggregate the parameter values across those dataset
   winners.
4. Persist the resulting tables and emit a markdown report tailored for downstream review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups


@dataclass(frozen=True)
class MetricConfig:
    """Configuration describing how to aggregate a metric."""

    column: str                     # Column name containing the metric values.
    label: str                      # Human-readable label for markdown tables.
    slug: str                       # Short identifier for filenames/column names.


def _resolve_numeric_value(series: pd.Series) -> Optional[float]:
    """Return the series median while preserving integer types when appropriate."""
    numeric = series.dropna()
    if numeric.empty:
        return None

    median_value = float(numeric.median())
    dtype = numeric.dtype

    if pd.api.types.is_integer_dtype(dtype):
        # Preserve the integer nature when the median ends up on an exact integer.
        rounded = round(median_value)
        return float(rounded) if rounded != median_value else int(rounded)

    return median_value


def _resolve_mode_value(series: pd.Series) -> Optional[object]:
    """Return the most common value within the series (ties resolved deterministically)."""
    values = series.dropna()
    if values.empty:
        return None

    counts = values.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    if not candidates:
        return None

    # Resolve ties deterministically by string representation.
    if len(candidates) == 1:
        return candidates[0]

    return sorted(candidates, key=lambda x: str(x))[0]


def _detect_parameter_columns(df: pd.DataFrame, prefix: str = "param_") -> List[str]:
    """Identify parameter columns that should be aggregated."""
    return [col for col in df.columns if col.startswith(prefix)]


def _split_numeric_and_categorical(columns: Iterable[str], df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Partition parameter columns into numeric and categorical groups."""
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def _dataset_group_label(group: str) -> str:
    if group == "synthetic":
        return "Synthetic"
    if group == "real":
        return "Real"
    if group == "overall":
        return "Overall"
    return str(group).title() if group else "N/A"


def _aggregate_parameters(top_rows: pd.DataFrame, numeric_columns: Sequence[str], categorical_columns: Sequence[str]) -> dict:
    """Aggregate the parameter values within the provided rows."""
    aggregated: dict = {}

    for column in numeric_columns:
        value = _resolve_numeric_value(top_rows[column])
        aggregated[column] = value

    for column in categorical_columns:
        value = _resolve_mode_value(top_rows[column])
        aggregated[column] = value

    return aggregated


def _summarize_group(
    top_rows: pd.DataFrame,
    metric: MetricConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    dataset_values: Optional[pd.Series] = None,
    architecture_col: Optional[str] = None,
) -> dict:
    """Create a summary row containing parameter aggregates and helpful metadata."""
    summary = _aggregate_parameters(top_rows, numeric_columns, categorical_columns)
    summary["entries_considered"] = int(len(top_rows))
    summary[f"best_{metric.slug}"] = float(top_rows[metric.column].min())
    summary[f"median_{metric.slug}"] = float(top_rows[metric.column].median())

    if architecture_col and architecture_col in top_rows.columns:
        summary["architecture_mode"] = _resolve_mode_value(top_rows[architecture_col])

    if dataset_values is not None:
        unique_datasets = dataset_groups.ordered_dataset_labels(dataset_values.dropna().unique())
        summary["datasets_included"] = ", ".join(unique_datasets)
    return summary


def _per_dataset_tables(
    df: pd.DataFrame,
    metric: MetricConfig,
    dataset_col: str,
    algorithm_col: str,
    sampling_col: str,
    architecture_col: Optional[str],
    top_k: int,
    param_columns: Sequence[str],
) -> pd.DataFrame:
    """Return per-dataset aggregated parameter tables."""
    usable = df.dropna(subset=[metric.column]).copy()
    if usable.empty:
        return pd.DataFrame()

    numeric_cols, categorical_cols = _split_numeric_and_categorical(param_columns, usable)
    grouping_cols = [dataset_col, algorithm_col, sampling_col]
    records: List[dict] = []

    grouped = usable.groupby(grouping_cols, dropna=False)
    for keys, group in grouped:
        sorted_group = group.sort_values(metric.column, ascending=True)
        top_rows = sorted_group.head(top_k)
        if top_rows.empty:
            continue

        summary = _summarize_group(
            top_rows=top_rows,
            metric=metric,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            architecture_col=architecture_col,
        )

        summary[dataset_col], summary[algorithm_col], summary[sampling_col] = keys
        records.append(summary)

    if not records:
        return pd.DataFrame()

    df_summary = pd.DataFrame(records)
    ordering = [dataset_col, algorithm_col, sampling_col, "entries_considered"]
    metric_cols = [f"best_{metric.slug}", f"median_{metric.slug}"]
    if architecture_col:
        ordering.append("architecture_mode")
    ordering.extend(metric_cols)
    ordering.extend(param_columns)

    existing_cols = [col for col in ordering if col in df_summary.columns]
    df_summary = df_summary[existing_cols]
    dataset_order = dataset_groups.ordered_dataset_labels(df_summary[dataset_col].astype(str).tolist())
    order_map = {label: idx for idx, label in enumerate(dataset_order)}
    df_summary["__dataset_order__"] = df_summary[dataset_col].map(order_map)
    df_summary = df_summary.sort_values(["__dataset_order__", algorithm_col, sampling_col]).drop(columns="__dataset_order__")
    return df_summary


def _cross_dataset_tables(
    df: pd.DataFrame,
    metric: MetricConfig,
    dataset_col: str,
    algorithm_col: str,
    sampling_col: str,
    architecture_col: Optional[str],
    param_columns: Sequence[str],
) -> pd.DataFrame:
    """Return cross-dataset aggregated parameter tables.

    The workflow keeps the single best trial per dataset (within an algorithm/sampling
    group) based on the provided metric, then aggregates the resulting parameter values
    across datasets. Numeric parameters report the median of the dataset-level winners,
    while categorical parameters report their mode.
    """
    usable = df.dropna(subset=[metric.column]).copy()
    if usable.empty:
        return pd.DataFrame()

    numeric_cols, categorical_cols = _split_numeric_and_categorical(param_columns, usable)
    records: List[dict] = []

    grouped = usable.groupby([algorithm_col, sampling_col], dropna=False)
    for (algorithm_value, sampling_value), group in grouped:
        # Identify the single best trial per dataset for this algorithm/sampling pair.
        sorted_group = group.sort_values(metric.column, ascending=True)
        best_per_dataset = sorted_group.groupby(dataset_col, dropna=False).head(1)
        if best_per_dataset.empty:
            continue

        summary = _summarize_group(
            top_rows=best_per_dataset,
            metric=metric,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            dataset_values=best_per_dataset[dataset_col],
            architecture_col=architecture_col,
        )

        summary[algorithm_col] = algorithm_value
        summary[sampling_col] = sampling_value
        summary["datasets_considered"] = int(best_per_dataset[dataset_col].nunique())
        best_dataset = best_per_dataset.sort_values(metric.column, ascending=True).iloc[0][dataset_col]
        summary["best_dataset"] = best_dataset
        records.append(summary)

    if not records:
        return pd.DataFrame()

    df_summary = pd.DataFrame(records)
    ordering = [
        algorithm_col,
        sampling_col,
        "best_dataset",
        "datasets_included",
        "datasets_considered",
        "entries_considered",
    ]
    if architecture_col:
        ordering.append("architecture_mode")
    ordering.extend([f"best_{metric.slug}", f"median_{metric.slug}"])
    ordering.extend(param_columns)

    existing_cols = [col for col in ordering if col in df_summary.columns]
    return df_summary[existing_cols].sort_values([algorithm_col, sampling_col])


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as markdown, handling empty inputs gracefully."""
    if df.empty:
        return "_No qualifying rows._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Fall back to a simpler representation when tabulate is unavailable.
        return df.to_string(index=False)


def _write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe to CSV when it contains rows."""
    if dataframe.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def build_parameter_summary(
    df: pd.DataFrame,
    metrics: Sequence[MetricConfig],
    output_dir: Path,
    architecture_label: Optional[str] = None,
    dataset_col: str = "dataset",
    algorithm_col: str = "algorithm",
    sampling_col: str = "sampling_method_parsed",
    architecture_col: str = "architecture",
    per_dataset_top_k: int = 10,
    parameter_prefix: str = "param_",
    markdown_filename: str = "PARAMETER_SUMMARY.md",
) -> Path:
    """
    Build per-dataset and cross-dataset parameter summaries for the provided metrics.

    Returns
    -------
    Path
        Location of the generated markdown report.
    """
    if df.empty:
        raise ValueError("Cannot build parameter summary – input dataframe is empty.")

    if dataset_col not in df.columns:
        raise KeyError(f"Column '{dataset_col}' not found in dataframe.")
    if algorithm_col not in df.columns:
        raise KeyError(f"Column '{algorithm_col}' not found in dataframe.")
    if sampling_col not in df.columns:
        raise KeyError(f"Column '{sampling_col}' not found in dataframe.")

    if architecture_col not in df.columns:
        architecture_col = None

    param_columns = _detect_parameter_columns(df, prefix=parameter_prefix)
    if not param_columns:
        raise ValueError("No parameter columns detected – expected columns prefixed with 'param_'.")

    tables_dir = Path(output_dir) / "tables"
    summary_dir = Path(output_dir) / "summary_outputs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    architecture_suffix = ""
    if architecture_label:
        architecture_suffix = f"_{architecture_label.lower()}"

    markdown_sections: List[str] = []
    header_label = architecture_label if architecture_label else "Unified"
    markdown_sections.append(f"# Parameter Summary – {header_label}")
    markdown_sections.append("")
    markdown_sections.append(
        "Per-dataset tables retain the top-k trials (lower is better) before aggregating "
        "parameters via median/mode. Cross-dataset tables first select the best trial in "
        "each dataset for a given algorithm/sampling pair and then aggregate those winners."
    )
    markdown_sections.append("")

    for metric in metrics:
        if metric.column not in df.columns:
            # Skip metrics absent in the dataframe for robustness.
            continue

        per_dataset_df = _per_dataset_tables(
            df=df,
            metric=metric,
            dataset_col=dataset_col,
            algorithm_col=algorithm_col,
            sampling_col=sampling_col,
            architecture_col=architecture_col,
            top_k=per_dataset_top_k,
            param_columns=param_columns,
        )

        cross_dataset_df = _cross_dataset_tables(
            df=df,
            metric=metric,
            dataset_col=dataset_col,
            algorithm_col=algorithm_col,
            sampling_col=sampling_col,
            architecture_col=architecture_col,
            param_columns=param_columns,
        )

        # Persist CSV outputs for downstream consumption.
        dataset_csv = tables_dir / f"parameters_{metric.slug}_per_dataset{architecture_suffix}.csv"
        overall_csv = tables_dir / f"parameters_{metric.slug}_cross_dataset{architecture_suffix}.csv"
        _write_csv(per_dataset_df, dataset_csv)
        _write_csv(cross_dataset_df, overall_csv)

        # Append markdown sections.
        markdown_sections.append(f"## Metric: {metric.label}")
        markdown_sections.append("")
        markdown_sections.append("### Per Dataset (Top 10 Trials)")
        markdown_sections.append(_dataframe_to_markdown(per_dataset_df))
        markdown_sections.append("")
        markdown_sections.append("### Cross Dataset (Best Per Dataset Aggregation)")
        markdown_sections.append(_dataframe_to_markdown(cross_dataset_df))
        markdown_sections.append("")

    markdown_path = summary_dir / markdown_filename
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")
    return markdown_path
