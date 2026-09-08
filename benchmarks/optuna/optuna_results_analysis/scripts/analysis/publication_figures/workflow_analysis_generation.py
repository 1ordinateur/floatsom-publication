from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from collections import OrderedDict
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

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.default_benchmark_adapter import (
    load_default_runs_csv,
)
from floatsom_benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.hyperparameter_stability import (
    StabilityMetric,
    build_hyperparameter_stability_report,
    stability_table_filename,
)
from floatsom_benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.tuned_vs_default import (
    TunedDefaultMetric,
    build_tuned_vs_external_default_report,
    summarize_paired_value_table,
    summarize_paired_value_global_row,
)

from .constants import *
from .helpers import *
from .plots import *
from .composition import *
from .publication_tripanel_outcomes import (
    PublicationOutcomePanelInput,
    compose_publication_outcome_tripanel,
    render_publication_outcome_panels,
)

def _direction_banner_text(left_label: str, right_label: str) -> str:
    return f"\u2190 Favours {left_label} | Favours {right_label} \u2192"


TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN = (0.08 + PUBLICATION_SAFE_LEFT_MARGIN) / 2.0
TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN = 0.09


def _humanize_comparison_label(value: str) -> str:
    tokens = [token for token in str(value).replace("_", " ").strip().split() if token]
    if not tokens:
        return ""
    token_map = {
        "xpysom": "XPySOM",
        "floatsom": "FloatSOM",
        "mst": "MST",
        "rng": "RNG",
        "hexagonal": "Hexagonal",
        "default": "Default",
        "tuned": "Tuned",
        "untuned": "Untuned",
        "true": "True",
    }
    return " ".join(token_map.get(token.lower(), token.title()) for token in tokens)


def _resolve_dataset_dimension_counts(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
) -> pd.Series:
    if dataset_col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")

    dataset_series = df[dataset_col].astype(str).str.strip().str.lower()
    fallback = dataset_series.map(DATASET_DIMENSION_COUNT_FALLBACK).astype("Int64").astype("object")

    best_candidate: Optional[pd.Series] = None
    best_valid_count = -1
    for column in DATASET_DIMENSION_COUNT_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce").where(lambda series: series > 0)
        valid_count = int(candidate.notna().sum())
        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_candidate = candidate
    if best_candidate is None or best_valid_count <= 0:
        return fallback
    return best_candidate.round().astype("Int64").astype("object").fillna(fallback)


def _first_resolved_dimension_count(series: pd.Series) -> object:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return pd.NA
    return int(round(float(valid.iloc[0])))


def _coerce_nonfinite_numeric_columns_to_nan(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite_mask = np.isfinite(numeric.astype(float))
        df[column] = numeric.where(finite_mask, np.nan)
    return df


def _resolve_effective_stability_sample_sizes(
    df: pd.DataFrame,
    *,
    sampling_mode: str,
    dataset_col: str = "dataset",
) -> pd.Series:
    sample_sizes = pd.to_numeric(
        _resolve_dataset_sample_sizes(df, dataset_col=dataset_col),
        errors="coerce",
    )
    sample_sizes = sample_sizes.where(sample_sizes > 0)

    if "n_train_samples" in df.columns:
        n_train_samples = pd.to_numeric(df["n_train_samples"], errors="coerce").where(lambda series: series > 0)
    else:
        # Optuna objectives run on the 70% train split before any sampler-specific
        # subsampling is applied.
        n_train_samples = np.floor(sample_sizes * 0.7)
        n_train_samples = pd.Series(n_train_samples, index=df.index, dtype=float).where(lambda series: series > 0)

    effective_sizes = n_train_samples.copy()
    sampling_mode_key = str(sampling_mode).strip().lower()
    if sampling_mode_key == "random":
        target_proportion = None
        for column in ("target_proportion", "param_target_proportion", "config_target_proportion"):
            if column not in df.columns:
                continue
            candidate = pd.to_numeric(df[column], errors="coerce").where(
                lambda series: (series > 0.0) & (series <= 1.0)
            )
            if candidate.notna().any():
                target_proportion = candidate
                break
        if target_proportion is None:
            # SamplingConfig.target_proportion defaults to 0.1 in the benchmark pipeline.
            target_proportion = pd.Series(0.1, index=df.index, dtype=float)
        effective_sizes = np.floor(effective_sizes * target_proportion)
        effective_sizes = pd.Series(effective_sizes, index=df.index, dtype=float)

    return effective_sizes.where(effective_sizes > 0)


def _analyze_topology_pair_basic(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    strict_pairing: bool,
    expected_pairs_per_dataset: Optional[int],
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
    value_a: str = "hexagonal",
    value_b: str = "mst",
    pair_slug: str = "hex_vs_mst",
    reference_value: str = "hexagonal",
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if "architecture" not in df.columns:
        return

    required_architectures = {str(value_a).strip().lower(), str(value_b).strip().lower(), str(reference_value).strip().lower()}
    subset = df[df["architecture"].isin(list(required_architectures))].copy()
    if subset.empty:
        return

    required_cols = ["dataset", "algorithm", "pair_sampling", "pair_seed"]
    if strict_pairing:
        _validate_required_pairing_columns(
            subset,
            required_cols,
            context=f"{pair_slug} (topology pairwise main)",
        )

    key_cols = ["dataset", "algorithm", "pair_sampling", "pair_seed", "pair_split"]
    if "pair_batch_mode" in subset.columns and not subset["pair_batch_mode"].isna().all():
        key_cols.insert(2, "pair_batch_mode")
    key_cols = [col for col in key_cols if col in subset.columns and not subset[col].isna().all()]

    pairs = _build_pairs(
        df=subset,
        metric=metric,
        compare_col="architecture",
        value_a=value_a,
        value_b=value_b,
        key_cols=key_cols,
        top_k=top_k,
        reference_value=reference_value,
    )
    if pairs.empty:
        return

    summary_pairs = _filter_pairs_to_sampling_modes(pairs, ["full"])
    if summary_pairs.empty:
        return

    summary = _summarize_pairs(
        pairs=summary_pairs,
        group_cols=["dataset"],
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col="pct_improvement_a_over_b_reference",
    )
    summary = _apply_q_values(summary, dataset_col="dataset")
    summary = _add_global_row(
        summary_df=summary,
        pairs=summary_pairs,
        dataset_col="dataset",
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col="pct_improvement_a_over_b_reference",
    )
    _validate_expected_pairs(
        summary,
        dataset_col="dataset",
        expected_pairs=expected_pairs_per_dataset,
        context=f"{pair_slug} (topology pairwise main)",
    )
    summary_pairs, strata = _build_topology_algorithm_mode_strata(
        summary_pairs,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col="pct_improvement_a_over_b_reference",
    )

    metric_slug = _slug(metric)
    pair_title = f"{_architecture_label(value_a)} vs {_architecture_label(value_b)}"
    reference_label = _architecture_label(reference_value)
    reference_slug = _slug(reference_value)
    show_forest_dataset_labels = False
    show_strata_dataset_labels = show_forest_dataset_labels
    show_sensitivity_y_axis_label = show_strata_dataset_labels
    sensitivity_y_axis_label = "Δ QE Mean (%)"
    strata_left_margin = (
        PUBLICATION_SAFE_LEFT_MARGIN
        if show_strata_dataset_labels
        else PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS
    )
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "tables" / f"{pair_slug}_main_{metric_slug}.csv"
    strata_path = output_dir / "tables" / f"{pair_slug}_strata_algorithm_mode_main_{metric_slug}.csv"
    pairs_path = output_dir / "tables" / f"{pair_slug}_pairs_main_{metric_slug}.csv"
    stats_table_path = output_dir / "tables" / f"{pair_slug}_stats_main_{metric_slug}.csv"
    fig_path = output_dir / "figures" / f"fig_{pair_slug}_main_{metric_slug}.svg"
    strata_fig_path = output_dir / "figures" / f"fig_{pair_slug}_by_algorithm_mode_main_{metric_slug}.svg"
    summary.to_csv(table_path, index=False)
    strata.to_csv(strata_path, index=False)
    summary_pairs.to_csv(pairs_path, index=False)
    _plot_dataset_forest_publication(
        summary_df=summary,
        title=f"{pair_title} ({metric_label}, Batch Modes Strata (Main))",
        subtitle=(
            f"Paired mean % improvement of {_architecture_label(value_a)} over {_architecture_label(value_b)} "
            f"with 95% paired t-test CI (relative to |{reference_label}|)"
        ),
        output_path=fig_path,
        dpi=dpi,
        alpha=alpha,
        inline_legends=inline_legends,
        legend_registry=legend_registry,
        stats_table_output_path=stats_table_path,
        x_axis_label="",
        show_y_tick_labels=show_forest_dataset_labels,
        **_tripanel_forest_panel_style(show_dataset_labels=show_forest_dataset_labels),
    )
    if not strata.empty and "algorithm_batch_mode" in strata.columns:
        _plot_topology_algorithm_strata(
            strata_df=strata,
            dataset_col="dataset",
            algorithm_col="algorithm_batch_mode",
            title=f"{pair_title} by Algorithm + Batch Mode ({metric_label}, Batch Modes Strata (Main))",
            subtitle="Dataset-level paired % improvement, stratified by algorithm/batch-mode",
            output_path=strata_fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            x_axis_label="",
            show_y_tick_labels=show_strata_dataset_labels,
            left_margin_override=strata_left_margin,
            use_tight_bbox=False,
            **_tripanel_topology_strata_text_style(show_dataset_labels=show_strata_dataset_labels),
        )
    generated_files.extend(
        [
            str(table_path),
            str(strata_path),
            str(pairs_path),
            str(stats_table_path),
            str(fig_path),
            str(strata_fig_path),
        ]
    )

    sensitivity_df = _build_sensitivity_summary(
        df=subset,
        metric=metric,
        compare_col="architecture",
        value_a=value_a,
        value_b=value_b,
        key_cols=key_cols,
        top_k_values=DEFAULT_SENSITIVITY_TOP_K,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col="pct_improvement_a_over_b_reference",
        reference_value=reference_value,
        test_method="paired_t",
        location_ci_method="paired_t",
        pair_sampling_modes=["full"],
    )
    sensitivity_table_path = output_dir / "tables" / f"{pair_slug}_sensitivity_main_{metric_slug}.csv"
    sensitivity_fig_path = output_dir / "figures" / f"fig_{pair_slug}_sensitivity_main_{metric_slug}.svg"
    sensitivity_df.to_csv(sensitivity_table_path, index=False)
    (
        topology_sensitivity_series,
        topology_sensitivity_styles,
        topology_sensitivity_primary_mode,
    ) = _build_topology_sensitivity_series(
        df=subset,
        pairs=pairs,
        metric=metric,
        compare_col="architecture",
        value_a=value_a,
        value_b=value_b,
        key_cols=key_cols,
        top_k_values=DEFAULT_SENSITIVITY_TOP_K,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        pct_col="pct_improvement_a_over_b_reference",
        reference_value=reference_value,
        exclude_algorithm_batch_mode_labels={"colors"},
        test_method="paired_t",
        location_ci_method="paired_t",
        pair_sampling_modes=["full"],
    )
    if len(topology_sensitivity_series) >= 2:
        _plot_sensitivity_curve_multiseries(
            sensitivity_series_by_mode=topology_sensitivity_series,
            mode_styles=topology_sensitivity_styles,
            primary_mode=topology_sensitivity_primary_mode,
            title=f"{pair_title}: Sensitivity to Top-k ({metric_label}, Batch Modes Strata (Main))",
            subtitle=(
                "Global pooled paired mean % improvement "
                f"by algorithm/batch-mode stratum with dotted 95% CI bounds (relative to |{reference_label}|)"
            ),
            output_path=sensitivity_fig_path,
            dpi=dpi,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            x_axis_label="",
            y_axis_label=sensitivity_y_axis_label,
            show_y_axis_label=show_sensitivity_y_axis_label,
            use_tight_bbox=False,
            left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
            bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
        )
    else:
        _plot_sensitivity_curve(
            sensitivity_df=sensitivity_df,
            title=f"{pair_title}: Sensitivity to Top-k ({metric_label}, Batch Modes Strata (Main))",
            subtitle=f"Global pooled paired mean % improvement with 95% CI (relative to |{reference_label}|)",
            output_path=sensitivity_fig_path,
            dpi=dpi,
            series_color=_comparison_series_color(compare_col="architecture", value_a=value_a),
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            x_axis_label="",
            y_axis_label=sensitivity_y_axis_label,
            show_y_axis_label=show_sensitivity_y_axis_label,
            use_tight_bbox=False,
            left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
            bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
        )
    generated_files.extend([str(sensitivity_table_path), str(sensitivity_fig_path)])

    diagnostics[f"{diagnostics_prefix}topology_pair.{pair_slug}.{metric_slug}"] = {
        "pairing_keys": key_cols,
        "pair": [value_a, value_b],
        "reference_mode": f"absolute_{reference_slug}",
        "reference_value": reference_value,
        "strata_file": str(strata_path),
        "strata_figure": str(strata_fig_path),
        "uses_stratified_sensitivity": len(topology_sensitivity_series) >= 2,
        "sensitivity_strata": list(topology_sensitivity_series.keys()),
        "per_dataset_pairs": summary[~summary["dataset"].map(_is_global_dataset_label)][["dataset", "n_pairs"]].to_dict(
            orient="records"
        ),
    }

def _resolve_hex_comparison_scheme(
    available_algorithms: Set[str],
    batch_mode_values: Sequence[str],
) -> str:
    normalized_algorithms = {str(value).strip().lower() for value in available_algorithms if str(value).strip()}
    normalized_batch_modes = {str(value).strip().lower() for value in batch_mode_values if str(value).strip()}
    has_colors = "colors" in normalized_algorithms
    has_batch = "batch" in normalized_algorithms
    has_full_batch = "full_batch" in normalized_batch_modes
    has_minibatch = "minibatch" in normalized_batch_modes
    if has_colors and has_batch:
        return "colors_vs_batch"
    if has_batch and has_full_batch and has_minibatch:
        return "full_vs_minibatch"
    return "unsupported"

def _analyze_hex_algorithms(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    strict_pairing: bool,
    expected_pairs_per_dataset: Optional[int],
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if "architecture" not in df.columns or "algorithm" not in df.columns:
        return

    subset = df[df["architecture"] == "hexagonal"].copy()
    if subset.empty:
        return

    batch_subset = subset[subset["algorithm"] == "batch"].copy()
    if batch_subset.empty:
        return

    if "pair_batch_mode" in batch_subset.columns:
        batch_mode_values = (
            batch_subset["pair_batch_mode"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
            .tolist()
        )
    else:
        batch_mode_values = ["full_batch"]

    batch_mode_values = sorted(value for value in batch_mode_values if value)
    if not batch_mode_values:
        batch_mode_values = ["full_batch"]

    available_algorithms = set(
        subset["algorithm"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .tolist()
    )
    compare_scheme = _resolve_hex_comparison_scheme(
        available_algorithms=available_algorithms,
        batch_mode_values=batch_mode_values,
    )
    if compare_scheme in {"unsupported", "colors_vs_batch"}:
        return

    mode_configs: List[Tuple[str, str, bool]] = [
        (batch_mode, _batch_mode_label(batch_mode), False)
        for batch_mode in batch_mode_values
    ]
    if compare_scheme == "full_vs_minibatch":
        mode_configs = [
            ("full_batch", _batch_mode_label("full_batch"), False),
            ("minibatch", _batch_mode_label("minibatch"), False),
        ]

    if compare_scheme == "colors_vs_batch":
        sensitivity_mode_styles: Dict[str, Dict[str, Any]] = HEX_SENSITIVITY_MODE_STYLES
    else:
        sensitivity_mode_styles = {
            "full_batch": {
                "label": "Full Batch vs Mini-batch",
                "color": METHOD_BASE_COLORS["batch"],
                "marker": "s",
            },
            "minibatch": {
                "label": "Mini-batch vs Full Batch",
                "color": METHOD_BASE_COLORS["minibatch"],
                "marker": "^",
            },
        }

    required_cols = ["dataset", "pair_sampling", "pair_seed"]
    key_cols_base = ["dataset", "pair_sampling", "pair_seed", "pair_split"]
    key_cols_base = [col for col in key_cols_base if col in subset.columns and not subset[col].isna().all()]
    sensitivity_series_by_mode: Dict[str, pd.DataFrame] = {}
    sensitivity_plot_meta_by_mode: Dict[str, Dict[str, Any]] = {}
    sensitivity_primary_label_by_mode: Dict[str, str] = {}
    sensitivity_pair_title_by_mode: Dict[str, str] = {}

    for mode_key, mode_display, pooled_batch_modes in mode_configs:
        mode_slug = _slug(mode_key)
        if compare_scheme == "colors_vs_batch":
            mode_subset = subset[
                (subset["algorithm"] == "colors")
                | ((subset["algorithm"] == "batch") & (subset["pair_batch_mode"] == mode_key))
            ].copy()
            compare_col = "algorithm"
            value_a = "colors"
            value_b = "batch"
            reference_value = "batch"
            value_a_label = "Colors"
            value_b_label = "Batch"
            pair_title = "Colors vs Batch"
            percent_beaten_col = "colors_beats_batch"
        else:
            mode_subset = subset[subset["algorithm"] == "batch"].copy()
            mode_subset = mode_subset[mode_subset["pair_batch_mode"].isin(["full_batch", "minibatch"])].copy()
            compare_col = "pair_batch_mode"
            if mode_key == "minibatch":
                value_a = "minibatch"
                value_b = "full_batch"
                reference_value = "minibatch"
                value_a_label = "Mini-batch"
                value_b_label = "Full Batch"
                pair_title = "Mini-batch vs Full Batch"
                percent_beaten_col = "minibatch_beats_full_batch"
            else:
                value_a = "full_batch"
                value_b = "minibatch"
                reference_value = "full_batch"
                value_a_label = "Full Batch"
                value_b_label = "Mini-batch"
                pair_title = "Full Batch vs Mini-batch"
                percent_beaten_col = "full_batch_beats_minibatch"
        if mode_subset.empty:
            continue

        if strict_pairing:
            _validate_required_pairing_columns(
                mode_subset,
                required_cols,
                context=f"hex {pair_title.lower()} ({mode_display})",
            )

        key_cols = list(key_cols_base)
        pairs = _build_pairs(
            df=mode_subset,
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
            pct_col="pct_improvement_a_over_b_reference",
        )
        summary = _apply_q_values(summary, dataset_col="dataset")
        summary = _add_global_row(
            summary_df=summary,
            pairs=pairs,
            dataset_col="dataset",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
        )
        expected_pairs_for_mode = expected_pairs_per_dataset
        _validate_expected_pairs(
            summary,
            dataset_col="dataset",
            expected_pairs=expected_pairs_for_mode,
            context=f"hex {pair_title.lower()} ({mode_display})",
        )

        metric_slug = _slug(metric)
        percent_beaten_df = (
            pairs.assign(**{percent_beaten_col: (pairs["score_a"] < pairs["score_b"]).astype(float)})
            .groupby("dataset", dropna=False)
            .agg(
                n_pairs=(percent_beaten_col, "size"),
                percent_beaten=(percent_beaten_col, lambda x: float(np.mean(x)) * 100.0),
            )
            .reset_index()
        )
        percent_beaten_table_path = (
            output_dir / "tables" / f"hex_colors_vs_batch_percent_beaten_{mode_slug}_{metric_slug}.csv"
        )
        percent_beaten_fig_path = (
            output_dir / "figures" / f"fig_hex_colors_vs_batch_percent_beaten_{mode_slug}_{metric_slug}.svg"
        )
        percent_beaten_df.to_csv(percent_beaten_table_path, index=False)
        _plot_dataset_percent_beaten(
            dataset_df=percent_beaten_df,
            title=f"Hexagonal: Percent Beaten ({pair_title}, {mode_display}, {metric_label})",
            subtitle=f"Dataset-wise win percentage for {value_a_label} (lower is better objective)",
            output_path=percent_beaten_fig_path,
            dpi=dpi,
        )

        table_path = output_dir / "tables" / f"hex_colors_vs_batch_{mode_slug}_{metric_slug}.csv"
        pairs_path = output_dir / "tables" / f"hex_colors_vs_batch_pairs_{mode_slug}_{metric_slug}.csv"
        stats_table_path = output_dir / "tables" / f"hex_colors_vs_batch_stats_{mode_slug}_{metric_slug}.csv"
        fig_path = output_dir / "figures" / f"fig_hex_colors_vs_batch_{mode_slug}_{metric_slug}.svg"

        summary.to_csv(table_path, index=False)
        pairs.to_csv(pairs_path, index=False)
        generated_files.extend(
            [
                str(table_path),
                str(pairs_path),
                str(stats_table_path),
                str(fig_path),
                str(percent_beaten_table_path),
                str(percent_beaten_fig_path),
            ]
        )

        _plot_dataset_forest(
            summary_df=summary,
            dataset_col="dataset",
            title=f"Hexagonal: {pair_title} ({mode_display}, {metric_label})",
            subtitle=(
                f"Paired mean % improvement of {value_a_label} over {value_b_label} "
                f"with 95% paired t-test CI (relative to |{value_b_label if compare_scheme == 'colors_vs_batch' else value_a_label}|)"
            ),
            output_path=fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            stats_table_output_path=stats_table_path,
        )

        dumbbell_path = output_dir / "figures" / f"fig_hex_colors_vs_batch_dumbbell_{mode_slug}_{metric_slug}.svg"
        dumbbell_table_path = output_dir / "tables" / f"hex_colors_vs_batch_dumbbell_{mode_slug}_{metric_slug}.csv"
        dumbbell_df = pairs.copy()
        dumbbell_df["pair_id"] = _pair_identifier(dumbbell_df)
        dumbbell_df.to_csv(dumbbell_table_path, index=False)
        _plot_hex_dumbbell(
            pairs=pairs,
            metric_label=metric_label,
            title=f"Hexagonal: Pairwise Dumbbell ({pair_title}, {mode_display}, {metric_label})",
            subtitle="Each line is one paired unit (sampling × seed × split)",
            output_path=dumbbell_path,
            dpi=dpi,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
        )
        generated_files.extend([str(dumbbell_table_path), str(dumbbell_path)])

        sensitivity_df = _build_sensitivity_summary(
            df=mode_subset,
            metric=metric,
            compare_col=compare_col,
            value_a=value_a,
            value_b=value_b,
            key_cols=key_cols,
            top_k_values=DEFAULT_SENSITIVITY_TOP_K,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
            reference_value=reference_value,
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        sensitivity_table_path = output_dir / "tables" / f"hex_colors_vs_batch_sensitivity_{mode_slug}_{metric_slug}.csv"
        sensitivity_fig_path = output_dir / "figures" / f"fig_hex_colors_vs_batch_sensitivity_{mode_slug}_{metric_slug}.svg"
        sensitivity_df.to_csv(sensitivity_table_path, index=False)
        sensitivity_series_by_mode[mode_key] = sensitivity_df.copy()
        sensitivity_plot_meta_by_mode[mode_key] = {
            "mode_display": mode_display,
            "sensitivity_fig_path": sensitivity_fig_path,
        }
        sensitivity_primary_label_by_mode[mode_key] = value_a_label
        sensitivity_pair_title_by_mode[mode_key] = pair_title
        generated_files.append(str(sensitivity_table_path))

        diagnostics[f"{diagnostics_prefix}hex_{mode_slug}_{metric_slug}"] = {
            "batch_mode": mode_key,
            "batch_mode_display": mode_display,
            "pooled_batch_modes": bool(pooled_batch_modes),
            "reference_mode": f"absolute_{_slug(reference_value)}",
            "comparison_scheme": compare_scheme,
            "compare_col": compare_col,
            "compare_pair": [value_a, value_b],
            "pairing_keys": key_cols,
            "per_dataset_pairs": summary[~summary["dataset"].map(_is_global_dataset_label)][
                ["dataset", "n_pairs"]
            ].to_dict(orient="records"),
        }

    ordered_modes_for_plot = [mode_key for mode_key, _, _ in mode_configs if mode_key in sensitivity_plot_meta_by_mode]
    for mode_key in ordered_modes_for_plot:
        mode_meta = sensitivity_plot_meta_by_mode[mode_key]
        sensitivity_pair_title = sensitivity_pair_title_by_mode.get(mode_key, "Colors vs Batch")
        sensitivity_primary_label = sensitivity_primary_label_by_mode.get(mode_key, "Batch")
        _plot_sensitivity_curve_multiseries(
            sensitivity_series_by_mode=sensitivity_series_by_mode,
            mode_styles=sensitivity_mode_styles,
            primary_mode=mode_key,
            title=f"Hexagonal: Sensitivity to Top-k ({sensitivity_pair_title}, {mode_meta['mode_display']}, {metric_label})",
            subtitle=(
                "Global pooled paired mean % improvement "
                f"with dotted 95% CI bounds (relative to |{sensitivity_primary_label}|)"
            ),
            output_path=Path(mode_meta["sensitivity_fig_path"]),
            dpi=dpi,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
        )
        generated_files.append(str(mode_meta["sensitivity_fig_path"]))

def _render_sampling_qe_vs_sample_size_regression(
    *,
    pairs_df: pd.DataFrame,
    metric_label: str,
    output_figure_path: Path,
    output_table_path: Path,
    output_stats_path: Path,
    dpi: int,
    x_axis_label: str = "Sample size (log10)",
    y_axis_label: str = "QE difference (%)",
    left_margin_override: Optional[float] = None,
    bottom_margin_override: Optional[float] = None,
    use_tight_bbox: bool = True,
) -> Dict[str, object]:
    required_cols = {"dataset", "pct_improvement_a_over_b_reference"}
    if pairs_df.empty or not required_cols.issubset(set(pairs_df.columns)):
        return {
            "generated": False,
            "reason": "Missing required pair columns for regression.",
        }

    working = pairs_df.copy()
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working = working[~working["dataset"].map(_is_global_dataset_label)].copy()
    if working.empty:
        return {
            "generated": False,
            "reason": "No non-global dataset rows available for regression.",
        }

    working["sample_size"] = _resolve_dataset_sample_sizes(working, dataset_col="dataset")
    working["dimension_count"] = _resolve_dataset_dimension_counts(working, dataset_col="dataset")
    working["qe_diff_pct"] = pd.to_numeric(working["pct_improvement_a_over_b_reference"], errors="coerce")
    working = working.dropna(subset=["sample_size", "qe_diff_pct"])
    working = working[working["sample_size"] > 0].copy()
    if working.empty:
        return {
            "generated": False,
            "reason": "No rows with resolved sample_size and QE difference.",
        }

    dataset_summary = (
        working.groupby("dataset", dropna=False)
        .agg(
            qe_diff_pct=("qe_diff_pct", "median"),
            n_pair_units=("qe_diff_pct", "size"),
            sample_size=("sample_size", "median"),
            dimension_count=("dimension_count", _first_resolved_dimension_count),
        )
        .reset_index()
    )
    dataset_summary["dataset_type"] = dataset_summary["dataset"].map(_dataset_group_key)
    dataset_summary["sample_size"] = pd.to_numeric(dataset_summary["sample_size"], errors="coerce").round().astype("Int64")
    dataset_summary["dimension_count"] = pd.to_numeric(
        dataset_summary["dimension_count"],
        errors="coerce",
    ).round().astype("Int64")
    dataset_summary["dimension_count"] = dataset_summary["dimension_count"].astype(object)
    dataset_summary["dimension_count"] = dataset_summary["dimension_count"].where(
        dataset_summary["dimension_count"].notna(),
        "NA",
    )
    dataset_summary["log10_sample_size"] = np.log10(dataset_summary["sample_size"].astype(float))
    dataset_summary = _coerce_nonfinite_numeric_columns_to_nan(
        dataset_summary,
        columns=["qe_diff_pct", "sample_size", "log10_sample_size"],
    ).dropna(
        subset=["qe_diff_pct", "sample_size", "log10_sample_size"]
    )
    if dataset_summary.empty:
        return {
            "generated": False,
            "reason": "No dataset-level rows available after regression preprocessing.",
        }

    dataset_summary = dataset_summary.sort_values("sample_size", ascending=True).reset_index(drop=True)
    dataset_summary = _attach_dataset_indices(
        dataset_summary,
        dataset_col="dataset",
        sample_size_col="sample_size",
    )
    ordered_columns = [
        "dataset_index",
        "dataset",
        "qe_diff_pct",
        "n_pair_units",
        "sample_size",
        "dimension_count",
        "dataset_type",
        "log10_sample_size",
    ]
    dataset_summary = dataset_summary[
        [column for column in ordered_columns if column in dataset_summary.columns]
    ].copy()
    x = dataset_summary["log10_sample_size"].to_numpy(dtype=float)
    y = dataset_summary["qe_diff_pct"].to_numpy(dtype=float)

    r_value = float("nan")
    p_value = float("nan")
    slope = float("nan")
    intercept = float("nan")
    model_name = "exp_nonnegative"
    model_intercept = float("nan")
    model_slope = float("nan")
    if len(dataset_summary) >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        r_value, p_value = stats.pearsonr(x, y)
    # Use non-negative exponential-form fit so the trend does not go below zero.
    if len(dataset_summary) >= 2 and np.nanstd(x) > 0:
        y_for_fit = np.maximum(y, 0.0) + 1e-6
        model_slope, model_intercept = np.polyfit(x, np.log(y_for_fit), 1)
        slope = model_slope
        intercept = model_intercept

    stats_df = pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset_summary)),
                "pearson_r": r_value,
                "pearson_r_p_value": p_value,
                "slope": slope,
                "intercept": intercept,
                "model": model_name,
                "model_log_intercept": model_intercept,
                "model_log_slope": model_slope,
            }
        ]
    )

    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary.to_csv(output_table_path, index=False)
    stats_df.to_csv(output_stats_path, index=False)

    _set_plot_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.4, 7.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    # De-overlap exact duplicate points with a deterministic x-offset.
    dataset_summary["x_plot"] = dataset_summary["log10_sample_size"].astype(float)
    dataset_summary["y_plot"] = dataset_summary["qe_diff_pct"].astype(float)
    duplicate_groups = dataset_summary.groupby(["x_plot", "y_plot"], dropna=False, sort=False).indices
    for _, index_array in duplicate_groups.items():
        indices = list(index_array)
        if len(indices) <= 1:
            continue
        offsets = np.linspace(-0.03, 0.03, len(indices))
        for local_idx, row_idx in enumerate(indices):
            dataset_summary.at[row_idx, "x_plot"] = float(dataset_summary.at[row_idx, "x_plot"]) + float(offsets[local_idx])

    # Additional spread for dense near-overlap clusters.
    y_values = dataset_summary["y_plot"].to_numpy(dtype=float)
    y_span = float(np.nanmax(y_values) - np.nanmin(y_values)) if len(dataset_summary) > 1 else 1.0
    x_bin_width = 0.16
    y_bin_width = max(0.9, 0.04 * max(1.0, y_span))
    dataset_summary["_x_bin"] = np.round(dataset_summary["x_plot"] / x_bin_width) * x_bin_width
    dataset_summary["_y_bin"] = np.round(dataset_summary["y_plot"] / y_bin_width) * y_bin_width
    near_groups = dataset_summary.groupby(["_x_bin", "_y_bin"], dropna=False, sort=False).indices
    for _, index_array in near_groups.items():
        indices = list(index_array)
        if len(indices) <= 1:
            continue
        offsets = np.linspace(-0.05, 0.05, len(indices))
        for local_idx, row_idx in enumerate(indices):
            dataset_summary.at[row_idx, "x_plot"] = float(dataset_summary.at[row_idx, "x_plot"]) + float(offsets[local_idx])

    ax.scatter(
        dataset_summary["x_plot"],
        dataset_summary["y_plot"],
        s=_scaled_scatter(92),
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        zorder=2,
    )
    for _, row in dataset_summary.iterrows():
        dataset_index = row.get("dataset_index", pd.NA)
        if pd.isna(dataset_index):
            continue
        ax.annotate(
            str(int(dataset_index)),
            (float(row["x_plot"]), float(row["y_plot"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=17.6,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#1f1f1f",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=3,
            clip_on=False,
        )

    if np.isfinite(model_slope) and np.isfinite(model_intercept):
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = np.exp(model_intercept + model_slope * x_line) - 1e-6
        y_line = np.clip(y_line, 0.0, None)
        ax.plot(
            x_line,
            y_line,
            color="#2ca02c",
            linewidth=2.1,
            linestyle="-",
            alpha=0.9,
            zorder=1,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
    ax.set_xlabel(str(x_axis_label))
    ax.set_ylabel(str(y_axis_label))
    y_top = float(np.nanmax(np.maximum(dataset_summary["y_plot"].to_numpy(dtype=float), 0.0))) if len(dataset_summary) else 1.0
    y_top = max(1.0, 1.08 * y_top)
    y_bottom = min(-0.6, -0.06 * y_top)
    x_vals = dataset_summary["x_plot"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_vals))
    x_max = float(np.nanmax(x_vals))
    x_span = max(0.5, x_max - x_min)
    x_pad = max(0.12, 0.07 * x_span)

    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    subplot_left = float(left_margin_override) if left_margin_override is not None else 0.12
    subplot_bottom = float(bottom_margin_override) if bottom_margin_override is not None else 0.10
    fig.subplots_adjust(left=subplot_left, right=0.985, bottom=subplot_bottom, top=0.985)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.16, "transparent": True}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_figure_path, **save_kwargs)
    plt.close(fig)

    return {
        "generated": True,
        "figure": str(output_figure_path.resolve()),
        "table": str(output_table_path.resolve()),
        "stats_table": str(output_stats_path.resolve()),
        "pearson_r": r_value,
        "pearson_r_p_value": p_value,
        "n_datasets": int(len(dataset_summary)),
    }

def _render_selected_parameter_stability_vs_sample_size_regression(
    *,
    dataset_summary_df: pd.DataFrame,
    sampling_mode: str,
    output_figure_path: Path,
    output_table_path: Path,
    output_stats_path: Path,
    dpi: int,
    show_mode_title: bool = True,
    x_axis_label: str = "Sample size (log10)",
    y_axis_label: str = "Stability Score",
    left_margin_override: Optional[float] = None,
    bottom_margin_override: Optional[float] = None,
    figure_width_override: Optional[float] = None,
    figure_height_override: Optional[float] = None,
    use_tight_bbox: bool = True,
) -> Dict[str, object]:
    topology_columns = {
        "hexagonal": "stability_score_mean_hexagonal",
        "mst": "stability_score_mean_mst",
        "rng": "stability_score_mean_rng",
    }
    required_cols = {"dataset", *topology_columns.values()}
    if dataset_summary_df.empty or not required_cols.issubset(set(dataset_summary_df.columns)):
        return {
            "generated": False,
            "reason": "Missing required dataset stability columns for size regression.",
        }

    working = dataset_summary_df.copy()
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working = working[~working["dataset"].map(_is_global_dataset_label)].copy()
    if working.empty:
        return {
            "generated": False,
            "reason": "No non-global dataset rows available for stability regression.",
        }

    working["sample_size"] = _resolve_effective_stability_sample_sizes(
        working,
        sampling_mode=sampling_mode,
        dataset_col="dataset",
    )
    working = working.dropna(subset=["sample_size"])
    working = working[working["sample_size"] > 0].copy()
    if working.empty:
        return {
            "generated": False,
            "reason": "No rows with resolved sample_size for stability regression.",
        }

    agg_spec: Dict[str, Tuple[str, str]] = {
        "sample_size": ("sample_size", "median"),
    }
    for col in topology_columns.values():
        agg_spec[col] = (col, "median")
    if "n_selected_parameters" in working.columns:
        agg_spec["n_selected_parameters"] = ("n_selected_parameters", "median")

    dataset_summary = working.groupby("dataset", dropna=False).agg(**agg_spec).reset_index()
    dataset_summary["log10_sample_size"] = np.log10(pd.to_numeric(dataset_summary["sample_size"], errors="coerce"))
    topology_value_columns = list(topology_columns.values())
    dataset_summary = _coerce_nonfinite_numeric_columns_to_nan(
        dataset_summary,
        columns=["sample_size", "log10_sample_size", *topology_value_columns],
    )
    dataset_summary = dataset_summary.dropna(subset=["sample_size", "log10_sample_size"])
    dataset_summary = dataset_summary[
        dataset_summary[topology_value_columns].notna().any(axis=1)
    ].copy()
    if dataset_summary.empty:
        return {
            "generated": False,
            "reason": "No dataset-level stability rows available after preprocessing.",
        }

    dataset_summary = dataset_summary.sort_values("sample_size", ascending=True).reset_index(drop=True)
    dataset_summary["dataset_type"] = dataset_summary["dataset"].map(_dataset_group_key)

    stats_rows: List[Dict[str, object]] = []
    for topology, col in topology_columns.items():
        topology_df = dataset_summary[["dataset", "log10_sample_size", col]].copy()
        topology_df[col] = pd.to_numeric(topology_df[col], errors="coerce")
        topology_df = topology_df.dropna(subset=["log10_sample_size", col])
        topology_df = _coerce_nonfinite_numeric_columns_to_nan(
            topology_df,
            columns=["log10_sample_size", col],
        ).dropna(subset=["log10_sample_size", col])

        r_value = float("nan")
        p_value = float("nan")
        slope = float("nan")
        intercept = float("nan")
        model_intercept = float("nan")
        model_slope = float("nan")
        model_name = "exp_nonnegative"

        if len(topology_df) >= 2:
            x = topology_df["log10_sample_size"].to_numpy(dtype=float)
            y = topology_df[col].to_numpy(dtype=float)
            if np.nanstd(x) > 0 and np.nanstd(y) > 0:
                r_value, p_value = stats.pearsonr(x, y)
            if np.nanstd(x) > 0:
                y_for_fit = np.maximum(y, 0.0) + 1e-6
                model_slope, model_intercept = np.polyfit(x, np.log(y_for_fit), 1)
                slope = model_slope
                intercept = model_intercept

        stats_rows.append(
            {
                "sampling_mode": sampling_mode,
                "topology": topology,
                "n_datasets": int(len(topology_df)),
                "pearson_r": r_value,
                "pearson_r_p_value": p_value,
                "slope": slope,
                "intercept": intercept,
                "model": model_name,
                "model_log_intercept": model_intercept,
                "model_log_slope": model_slope,
            }
        )

    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary.to_csv(output_table_path, index=False)
    pd.DataFrame(stats_rows).to_csv(output_stats_path, index=False)

    _set_plot_theme(style="whitegrid", context="talk")
    fig_width = float(figure_width_override) if figure_width_override is not None else 10.6
    fig_height = float(figure_height_override) if figure_height_override is not None else 7.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if len(topology_columns) == 1:
        topology_offsets = {next(iter(topology_columns)): 0.0}
    else:
        offset_values = np.linspace(-0.028, 0.028, len(topology_columns))
        topology_offsets = {
            topology: float(offset_values[idx])
            for idx, topology in enumerate(topology_columns)
        }
    topology_colors = {
        topology: TOPOLOGY_BASE_COLORS.get(topology, "#4C78A8")
        for topology in topology_columns
    }
    x_values = dataset_summary["log10_sample_size"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))

    all_y_values: List[float] = []
    for topology, col in topology_columns.items():
        topology_df = dataset_summary[["dataset", "log10_sample_size", col]].copy()
        topology_df[col] = pd.to_numeric(topology_df[col], errors="coerce")
        topology_df = topology_df.dropna(subset=["log10_sample_size", col])
        if topology_df.empty:
            continue

        y = topology_df[col].to_numpy(dtype=float)
        x = topology_df["log10_sample_size"].to_numpy(dtype=float)
        all_y_values.extend(y.tolist())

        ax.scatter(
            x + float(topology_offsets[topology]),
            y,
            s=_scaled_scatter(108),
            color=topology_colors[topology],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
        )

        if len(topology_df) >= 2 and np.nanstd(x) > 0:
            y_for_fit = np.maximum(y, 0.0) + 1e-6
            model_slope, model_intercept = np.polyfit(x, np.log(y_for_fit), 1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            y_line = np.exp(model_intercept + model_slope * x_line) - 1e-6
            y_line = np.clip(y_line, 0.0, None)
            ax.plot(
                x_line,
                y_line,
                color=topology_colors[topology],
                linewidth=2.2,
                linestyle="-",
                alpha=0.92,
                zorder=2,
            )

    if not all_y_values:
        plt.close(fig)
        return {
            "generated": False,
            "reason": "No finite topology stability values available for plotting.",
        }

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55, zorder=1)
    ax.set_xlabel(str(x_axis_label), fontsize=_scaled_font(13.2))
    ax.set_ylabel(str(y_axis_label), fontsize=_scaled_font(13.2))
    if show_mode_title:
        ax.set_title(_sampling_mode_display(sampling_mode), pad=10, fontsize=_scaled_font(14.8), loc="left")

    y_min = float(np.nanmin(all_y_values))
    y_max = float(np.nanmax(all_y_values))
    y_span = max(0.08, y_max - y_min)
    y_bottom = max(-0.02, y_min - 0.12 * y_span)
    y_top = min(1.06, y_max + 0.14 * y_span)
    ax.set_ylim(y_bottom, y_top)

    x_span = max(0.5, x_max - x_min)
    x_pad = max(0.05, 0.04 * x_span)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    ax.tick_params(axis="x", labelsize=_scaled_font(11.8))
    ax.tick_params(axis="y", labelsize=_scaled_font(11.8))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: _format_axis_tick(value)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: _format_axis_tick(value)))
    ax.grid(False, axis="y")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8, alpha=0.85)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)

    subplot_left = float(left_margin_override) if left_margin_override is not None else 0.12
    subplot_bottom = float(bottom_margin_override) if bottom_margin_override is not None else 0.10
    fig.subplots_adjust(left=subplot_left, right=0.985, bottom=subplot_bottom, top=0.985)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.16}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_figure_path, **save_kwargs)
    plt.close(fig)

    return {
        "generated": True,
        "figure": str(output_figure_path.resolve()),
        "table": str(output_table_path.resolve()),
        "stats_table": str(output_stats_path.resolve()),
        "n_datasets": int(len(dataset_summary)),
    }

def _analyze_sampling_mode_outcomes(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
    inline_legends: bool = False,
    comparison_focus_key: Optional[str] = None,
) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    metric_slug = _slug(metric)

    if "pair_sampling" not in df.columns:
        return

    working = df.copy()
    working["pair_sampling"] = working["pair_sampling"].astype(str).str.lower().str.strip()
    available_modes_global = _canonical_sampling_modes(working["pair_sampling"].dropna().tolist())
    focus_key = str(comparison_focus_key or "").strip().lower()
    comparison_scope = _sampling_comparison_scope_slug(
        available_modes_global,
        focus_comparison_key=focus_key or None,
    )
    comparison_specs = _sampling_comparison_specs(available_modes_global)
    if focus_key:
        comparison_specs = [spec for spec in comparison_specs if spec[0] == focus_key]
    if not comparison_specs:
        diagnostics[f"{diagnostics_prefix}sampling_mode_comparison_{metric_slug}"] = {
            "panels": {},
            "combined_figure": "",
            "combined_figure_no_stack": "",
            "main_hex_full_batch_figure": "",
            "main_hex_full_batch_comparison_figures": {},
            "legend_key": "",
            "available_sampling_modes": available_modes_global,
            "comparison_scope": comparison_scope,
            "comparison_keys": [],
            "comparison_focus_key": focus_key,
        }
        return

    architecture_specs: List[Tuple[str, str]] = [
        ("hexagonal", "Hexagonal"),
        ("mst", "MST"),
    ]
    cohort_specs: List[Tuple[str, str, str, Optional[str], bool]] = [
        ("full_batch", "Full Batch", "batch", "full_batch", False),
        ("colors", "Colors", "colors", None, False),
    ]
    diagnostics_payload: Dict[str, object] = {
        "panels": {},
        "combined_figure": "",
        "combined_figure_no_stack": "",
        "main_hex_full_batch_figure": "",
        "main_hex_full_batch_comparison_figures": {},
        "legend_key": "",
        "qe_vs_sample_size_regression_figure": "",
        "qe_vs_sample_size_regression_table": "",
        "qe_vs_sample_size_regression_stats_table": "",
        "qe_vs_sample_size_regression_r": float("nan"),
        "qe_vs_sample_size_regression_p_value": float("nan"),
        "qe_vs_sample_size_regression_n": 0,
        "available_sampling_modes": available_modes_global,
        "comparison_scope": comparison_scope,
        "comparison_keys": [comparison_key for comparison_key, _, _, _, _ in comparison_specs],
        "comparison_focus_key": focus_key,
    }
    sampling_legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    panel_summary_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    regression_pairs_full_vs_random: Optional[pd.DataFrame] = None

    for architecture_name, architecture_label in architecture_specs:
        arch_subset = working[working["architecture"] == architecture_name].copy()
        if arch_subset.empty:
            continue

        for (
            cohort_slug,
            cohort_label,
            algorithm_name,
            required_batch_mode,
            include_batch_mode_in_pairing,
        ) in cohort_specs:
            cohort_subset = arch_subset[arch_subset["algorithm"] == algorithm_name].copy()
            if cohort_subset.empty:
                continue
            if (
                algorithm_name == "batch"
                and required_batch_mode is not None
                and "pair_batch_mode" in cohort_subset.columns
                and not cohort_subset["pair_batch_mode"].isna().all()
            ):
                cohort_subset = cohort_subset[
                    cohort_subset["pair_batch_mode"].astype(str).str.lower().str.strip() == required_batch_mode
                ].copy()
                if cohort_subset.empty:
                    continue

            key_cols = [col for col in ["dataset", "pair_seed", "pair_split"] if col in cohort_subset.columns]
            if (
                algorithm_name == "batch"
                and include_batch_mode_in_pairing
                and "pair_batch_mode" in cohort_subset.columns
                and not cohort_subset["pair_batch_mode"].isna().all()
            ):
                key_cols.insert(1, "pair_batch_mode")

            panel_key = (architecture_name, cohort_slug)
            panel_rows: List[pd.DataFrame] = []
            panel_diag_key = f"{architecture_name}.{cohort_slug}"
            panel_diag: Dict[str, Any] = {
                "architecture": architecture_label,
                "cohort": cohort_label,
                "pairing_keys": key_cols,
                "comparisons": {},
            }

            for comparison_key, comparison_label, value_a, value_b, reference_value in comparison_specs:
                pairs = _build_pairs(
                    df=cohort_subset,
                    metric=metric,
                    compare_col="pair_sampling",
                    value_a=value_a,
                    value_b=value_b,
                    key_cols=key_cols,
                    top_k=top_k,
                    reference_value=reference_value,
                )
                if pairs.empty:
                    continue
                if (
                    architecture_name == "hexagonal"
                    and cohort_slug == "full_batch"
                    and comparison_key == "full_vs_random"
                ):
                    regression_pairs_full_vs_random = pairs.copy()

                summary = _summarize_pairs(
                    pairs=pairs,
                    group_cols=["dataset"],
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                    pct_col="pct_improvement_a_over_b_reference",
                )
                summary = _apply_q_values(summary, dataset_col="dataset")
                summary = _add_global_row(
                    summary_df=summary,
                    pairs=pairs,
                    dataset_col="dataset",
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                    pct_col="pct_improvement_a_over_b_reference",
                )
                summary["comparison_key"] = comparison_key
                summary["comparison_label"] = comparison_label
                panel_rows.append(summary)

                arch_slug = _slug(architecture_name)
                if comparison_scope == "all_sampling_modes":
                    summary_name = (
                        f"sampling_mode_outcome_{arch_slug}_{cohort_slug}_{comparison_key}_{metric_slug}.csv"
                    )
                    pairs_name = (
                        f"sampling_mode_pairs_{arch_slug}_{cohort_slug}_{comparison_key}_{metric_slug}.csv"
                    )
                else:
                    summary_name = (
                        f"sampling_mode_outcome_{arch_slug}_{cohort_slug}_{comparison_scope}_{comparison_key}_{metric_slug}.csv"
                    )
                    pairs_name = (
                        f"sampling_mode_pairs_{arch_slug}_{cohort_slug}_{comparison_scope}_{comparison_key}_{metric_slug}.csv"
                    )
                summary_path = output_dir / "tables" / summary_name
                pairs_path = output_dir / "tables" / pairs_name
                summary.to_csv(summary_path, index=False)
                pairs.to_csv(pairs_path, index=False)
                generated_files.extend([str(summary_path), str(pairs_path)])
                panel_diag["comparisons"][comparison_key] = {
                    "label": comparison_label,
                    "summary_table": str(summary_path),
                    "pairs_table": str(pairs_path),
                    "reference_value": reference_value,
                }

            if panel_rows:
                panel_summary_frames[panel_key] = pd.concat(panel_rows, ignore_index=True)
                diagnostics_payload["panels"][panel_diag_key] = panel_diag

    if panel_summary_frames:
        if comparison_scope == "all_sampling_modes":
            combined_fig_name = f"fig_sampling_mode_disaggregated_hex_mst_{metric_slug}.svg"
        else:
            combined_fig_name = f"fig_sampling_mode_disaggregated_hex_mst_{comparison_scope}_{metric_slug}.svg"
        combined_fig_path = output_dir / "figures" / combined_fig_name
        _plot_sampling_mode_disaggregated_grid(
            panel_data=panel_summary_frames,
            title=f"Sampling Comparison (Hexagonal and MST, {metric_label})",
            output_path=combined_fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=None,
        )
        generated_files.append(str(combined_fig_path))
        diagnostics_payload["combined_figure"] = str(combined_fig_path)

        if comparison_scope == "all_sampling_modes":
            no_stack_fig_name = f"fig_sampling_mode_disaggregated_no_stack_hex_mst_{metric_slug}.svg"
        else:
            no_stack_fig_name = (
                f"fig_sampling_mode_disaggregated_no_stack_hex_mst_{comparison_scope}_{metric_slug}.svg"
            )
        no_stack_fig_path = output_dir / "figures" / no_stack_fig_name
        _plot_sampling_mode_disaggregated_no_stack_grid(
            panel_data=panel_summary_frames,
            title=f"Sampling Comparison (Hexagonal and MST, {metric_label}) - No Stacking",
            output_path=no_stack_fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=None,
        )
        generated_files.append(str(no_stack_fig_path))
        diagnostics_payload["combined_figure_no_stack"] = str(no_stack_fig_path)

        if metric in {"balanced_qe_raw", "balanced_qe_normalized"}:
            comparison_sides: Dict[str, Tuple[str, str, str]] = {
                "full_vs_random": ("Full", "Random", "Full"),
                "full_vs_hdsssom": ("Full", "HDSSOM", "Full"),
                "random_vs_hdsssom": ("Random", "HDSSOM", "Random"),
            }
            for cohort_slug, cohort_title, diag_figure_key, diag_compare_key in [
                (
                    "full_batch",
                    "Full Batch",
                    "main_hex_full_batch_figure",
                    "main_hex_full_batch_comparison_figures",
                ),
            ]:
                main_panel_key = ("hexagonal", cohort_slug)
                main_panel_df = panel_summary_frames.get(main_panel_key, pd.DataFrame()).copy()
                if main_panel_df.empty:
                    continue
                if comparison_scope == "all_sampling_modes":
                    main_fig_name = f"fig_sampling_mode_hexagonal_{cohort_slug}_main_{metric_slug}.svg"
                else:
                    main_fig_name = (
                        f"fig_sampling_mode_hexagonal_{cohort_slug}_{comparison_scope}_main_{metric_slug}.svg"
                    )
                main_fig_path = output_dir / "figures" / main_fig_name
                _plot_sampling_mode_main_hex_batch_panel(
                    summary_df=main_panel_df,
                    title=f"Sampling Comparison (Hexagonal / {cohort_title}, {metric_label})",
                    output_path=main_fig_path,
                    dpi=dpi,
                    alpha=alpha,
                    inline_legends=inline_legends,
                    legend_registry=None,
                )
                generated_files.append(str(main_fig_path))
                diagnostics_payload[diag_figure_key] = str(main_fig_path)

                comparison_figures: Dict[str, str] = {}
                for comparison_key, _, _, _, _ in comparison_specs:
                    comparison_summary = (
                        main_panel_df[main_panel_df["comparison_key"].astype(str) == comparison_key].copy()
                    )
                    if comparison_summary.empty:
                        continue

                    side_a, side_b, reference_side = comparison_sides[comparison_key]
                    if comparison_scope == "all_sampling_modes":
                        comparison_fig_name = (
                            f"fig_sampling_mode_hexagonal_{cohort_slug}_{comparison_key}_{metric_slug}.svg"
                        )
                        stats_table_name = (
                            f"sampling_mode_stats_hexagonal_{cohort_slug}_{comparison_key}_{metric_slug}.csv"
                        )
                    else:
                        comparison_fig_name = (
                            f"fig_sampling_mode_hexagonal_{cohort_slug}_{comparison_scope}_{comparison_key}_{metric_slug}.svg"
                        )
                        stats_table_name = (
                            f"sampling_mode_stats_hexagonal_{cohort_slug}_{comparison_scope}_{comparison_key}_{metric_slug}.csv"
                        )
                    comparison_fig_path = output_dir / "figures" / comparison_fig_name
                    stats_table_path = output_dir / "tables" / stats_table_name
                    comparison_show_labels = comparison_key == "full_vs_random"
                    _plot_dataset_forest_publication(
                        summary_df=comparison_summary,
                        title=f"Hexagonal: Sampling Comparison ({cohort_title}, {metric_label})",
                        subtitle=(
                            f"Paired mean % improvement of {side_a} over {side_b} "
                            f"with 95% paired t-test CI (relative to |{reference_side}|)"
                        ),
                        output_path=comparison_fig_path,
                        dpi=dpi,
                        alpha=alpha,
                        inline_legends=inline_legends,
                        legend_registry=sampling_legend_registry,
                        stats_table_output_path=stats_table_path,
                        show_y_tick_labels=comparison_show_labels,
                        x_axis_label="QE change (%)",
                        **_tripanel_forest_panel_style(show_dataset_labels=comparison_show_labels),
                    )
                    generated_files.extend([str(comparison_fig_path), str(stats_table_path)])
                    comparison_figures[comparison_key] = str(comparison_fig_path)

                diagnostics_payload[diag_compare_key] = comparison_figures

        regression_metrics = {
            "balanced_qe_raw",
            "balanced_qe_normalized",
            "quantization_error_holdout",
            "quantization_error_holdout_normalized",
            "quantization_error_train",
            "quantization_error_train_normalized",
        }
        if regression_pairs_full_vs_random is not None and metric in regression_metrics:
            if comparison_scope == "all_sampling_modes":
                regression_figure_name = (
                    f"fig_sampling_mode_hexagonal_full_batch_full_vs_random_qe_vs_sample_size_regression_{metric_slug}.svg"
                )
                regression_table_name = (
                    f"sampling_mode_hexagonal_full_batch_full_vs_random_qe_vs_sample_size_regression_{metric_slug}.csv"
                )
                regression_stats_name = (
                    f"sampling_mode_hexagonal_full_batch_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv"
                )
            else:
                regression_figure_name = (
                    f"fig_sampling_mode_hexagonal_full_batch_{comparison_scope}_full_vs_random_qe_vs_sample_size_regression_{metric_slug}.svg"
                )
                regression_table_name = (
                    f"sampling_mode_hexagonal_full_batch_{comparison_scope}_full_vs_random_qe_vs_sample_size_regression_{metric_slug}.csv"
                )
                regression_stats_name = (
                    f"sampling_mode_hexagonal_full_batch_{comparison_scope}_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv"
                )

            regression_result = _render_sampling_qe_vs_sample_size_regression(
                pairs_df=regression_pairs_full_vs_random,
                metric_label=metric_label,
                output_figure_path=output_dir / "figures" / regression_figure_name,
                output_table_path=output_dir / "tables" / regression_table_name,
                output_stats_path=output_dir / "tables" / regression_stats_name,
                dpi=dpi,
                x_axis_label="",
                y_axis_label="",
                left_margin_override=0.08,
                bottom_margin_override=0.10,
                use_tight_bbox=True,
            )
            if bool(regression_result.get("generated")):
                diagnostics_payload["qe_vs_sample_size_regression_figure"] = str(regression_result["figure"])
                diagnostics_payload["qe_vs_sample_size_regression_table"] = str(regression_result["table"])
                diagnostics_payload["qe_vs_sample_size_regression_stats_table"] = str(regression_result["stats_table"])
                diagnostics_payload["qe_vs_sample_size_regression_r"] = float(
                    regression_result.get("pearson_r", float("nan"))
                )
                diagnostics_payload["qe_vs_sample_size_regression_p_value"] = float(
                    regression_result.get("pearson_r_p_value", float("nan"))
                )
                diagnostics_payload["qe_vs_sample_size_regression_n"] = int(
                    regression_result.get("n_datasets", 0)
                )
                generated_files.extend(
                    [
                        str(regression_result["figure"]),
                        str(regression_result["table"]),
                        str(regression_result["stats_table"]),
                    ]
                )

        if not inline_legends:
            legend_key_path = output_dir / "figures" / "legend_key.svg"
            if _render_legend_key(legend_registry=sampling_legend_registry, output_path=legend_key_path, dpi=dpi):
                generated_files.append(str(legend_key_path))
                diagnostics_payload["legend_key"] = str(legend_key_path)

    diagnostics[f"{diagnostics_prefix}sampling_mode_comparison_{metric_slug}"] = diagnostics_payload

def _infer_hex_comparison_scheme_from_diagnostics(
    diagnostics: Dict[str, object],
    *,
    diagnostics_prefix: str = "",
) -> str:
    prefix = str(diagnostics_prefix)
    for key, payload in diagnostics.items():
        if prefix and not str(key).startswith(prefix):
            continue
        if not isinstance(payload, dict):
            continue
        scheme = payload.get("comparison_scheme")
        if scheme in {"colors_vs_batch", "full_vs_minibatch"}:
            return str(scheme)
    return "colors_vs_batch"

def _infer_sampling_comparison_scope_from_diagnostics(
    diagnostics: Dict[str, object],
    *,
    diagnostics_prefix: str = "",
) -> str:
    prefix = str(diagnostics_prefix)
    for key, payload in diagnostics.items():
        if prefix and not str(key).startswith(prefix):
            continue
        if not isinstance(payload, dict):
            continue
        scope = payload.get("comparison_scope")
        if isinstance(scope, str) and scope.strip():
            return scope.strip()
    return "all_sampling_modes"

def _infer_sampling_modes_from_diagnostics(
    diagnostics: Dict[str, object],
    *,
    diagnostics_prefix: str = "",
) -> List[str]:
    prefix = str(diagnostics_prefix)
    for key, payload in diagnostics.items():
        if prefix and not str(key).startswith(prefix):
            continue
        if not isinstance(payload, dict):
            continue
        available_modes = payload.get("available_sampling_modes")
        if isinstance(available_modes, (list, tuple, set)):
            ordered = _ordered_sampling_modes(available_modes)
            if ordered:
                return ordered
    return []

def _infer_sampling_regression_figure_from_diagnostics(
    diagnostics: Dict[str, object],
    *,
    diagnostics_prefix: str = "",
    metric_slug: str,
) -> Optional[Path]:
    key = f"{diagnostics_prefix}sampling_comparison.sampling_mode_comparison_{metric_slug}"
    payload = diagnostics.get(key)
    if isinstance(payload, dict):
        figure_text = str(payload.get("qe_vs_sample_size_regression_figure", "")).strip()
        if figure_text:
            candidate = Path(figure_text)
            if candidate.exists():
                return candidate
    return None


def _infer_sampling_regression_table_from_diagnostics(
    diagnostics: Dict[str, object],
    *,
    diagnostics_prefix: str = "",
    metric_slug: str,
) -> Optional[Path]:
    key = f"{diagnostics_prefix}sampling_comparison.sampling_mode_comparison_{metric_slug}"
    payload = diagnostics.get(key)
    if isinstance(payload, dict):
        table_text = str(payload.get("qe_vs_sample_size_regression_table", "")).strip()
        if table_text:
            candidate = Path(table_text)
            if candidate.exists():
                return candidate
    return None

def _compose_sampling_stratified_publication_figures(
    suite_output_dir: Path,
    suite_metrics_list: Sequence[str],
    sampling_dirs: Dict[str, Path],
    full_suite: bool,
    alpha: float,
    dpi: int,
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
    sampling_primary_comparison_key: Optional[str] = None,
) -> None:
    publication_dir = suite_output_dir / "publication_figures"
    publication_dir.mkdir(parents=True, exist_ok=True)
    hex_compare_scheme = _infer_hex_comparison_scheme_from_diagnostics(
        diagnostics=diagnostics,
        diagnostics_prefix=diagnostics_prefix,
    )
    sampling_comparison_scope = _infer_sampling_comparison_scope_from_diagnostics(
        diagnostics=diagnostics,
        diagnostics_prefix=f"{diagnostics_prefix}sampling_comparison.",
    )
    if hex_compare_scheme == "full_vs_minibatch":
        hex_pair_title = "Full Batch"
        hex_sensitivity_legend_specs: List[Tuple[str, Any, str]] = [
            ("Full Batch vs Mini-batch", METHOD_BASE_COLORS["batch"], "s"),
        ]
    else:
        hex_pair_title = "Colors vs Batch (Full Batch)"
        hex_sensitivity_legend_specs = [
            ("Colors vs Batch (Full Batch)", METHOD_BASE_COLORS["batch"], "s"),
        ]
    topology_sensitivity_legend_specs: List[Tuple[str, Any, str]] = [
        ("Batch (Full batch)", METHOD_BASE_COLORS["batch"], "s"),
    ]

    def _pick_metric(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in suite_metrics_list:
                return candidate
        return None

    canonical_modes = [mode for mode in ["full", "random", "hdsssom"] if mode in sampling_dirs]
    if not canonical_modes:
        canonical_modes = _infer_sampling_modes_from_diagnostics(
            diagnostics=diagnostics,
            diagnostics_prefix=f"{diagnostics_prefix}sampling_comparison.",
        )
    if full_suite and len(canonical_modes) < 2:
        diagnostics[f"{diagnostics_prefix}publication_figures"] = {
            "generated": False,
            "reason": "Fewer than two sampling blocks available for sampling-comparison publication figures.",
            "available_modes": canonical_modes,
        }
        return

    diagnostics_root: Dict[str, object] = {
        "generated": True,
        "full_suite": bool(full_suite),
        "figures": {},
        "sampling_comparison_scope": sampling_comparison_scope,
        "available_sampling_modes": canonical_modes,
    }
    letters = [chr(ord("A") + idx) for idx in range(26)]
    figure_1_legend = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_1_sampling_comparison.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=True,
        include_sensitivity=False,
        compact=True,
    )
    figure_2_forest_legend = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_2_sampling_comparison.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=True,
        include_sensitivity=False,
        compact=False,
    )
    figure_2_regression_legend = _render_sample_size_regression_legend_key(
        output_path=publication_dir / "legend_figure_2_sample_size_regression.svg",
        dpi=dpi,
        compact=True,
    )
    figure_3_forest_legend = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_3_topology.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=True,
        include_sensitivity=False,
        compact=False,
    )
    figure_3_sensitivity_legend = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_3_topology_sensitivity.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=False,
        include_sensitivity=True,
        sensitivity_color=_comparison_series_color(compare_col="architecture", value_a="hexagonal"),
        compact=True,
    )
    figure_3_sensitivity_legend_hex_rng = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_3_topology_sensitivity_hex_rng.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=False,
        include_sensitivity=True,
        sensitivity_color=_comparison_series_color(compare_col="architecture", value_a="hexagonal"),
        compact=True,
    )
    figure_3_sensitivity_legend_mst_rng = _render_composite_legend_key(
        output_path=publication_dir / "legend_figure_3_topology_sensitivity_mst_rng.svg",
        dpi=dpi,
        alpha=alpha,
        include_forest=False,
        include_sensitivity=True,
        sensitivity_color=_comparison_series_color(compare_col="architecture", value_a="mst"),
        compact=True,
    )
    supplementary_outcomes_legend: Optional[Path] = None
    supplementary_sensitivity_legend: Optional[Path] = None
    if full_suite:
        supplementary_outcomes_legend = _render_composite_legend_key(
            output_path=publication_dir / "legend_supp_outcomes.svg",
            dpi=dpi,
            alpha=alpha,
            include_forest=True,
            include_sensitivity=False,
        )
        supplementary_sensitivity_legend = _render_composite_legend_key(
            output_path=publication_dir / "legend_supp_sensitivity.svg",
            dpi=dpi,
            alpha=alpha,
            include_forest=False,
            include_sensitivity=True,
            sensitivity_series_specs=hex_sensitivity_legend_specs,
            sensitivity_dotted_bounds=True,
        )
    for legend_path in [
        figure_1_legend,
        figure_2_forest_legend,
        figure_2_regression_legend,
        figure_3_forest_legend,
        figure_3_sensitivity_legend,
        figure_3_sensitivity_legend_hex_rng,
        figure_3_sensitivity_legend_mst_rng,
        supplementary_outcomes_legend,
        supplementary_sensitivity_legend,
    ]:
        if legend_path is not None:
            generated_files.append(str(legend_path))

    balanced_metric = _pick_metric(["balanced_qe_raw", "balanced_qe_normalized"])
    holdout_metric = _pick_metric(["quantization_error_holdout", "quantization_error_holdout_normalized"])
    train_metric = _pick_metric(["quantization_error_train", "quantization_error_train_normalized"])
    distortion_metric = _pick_metric(["distortion_measure_holdout", "distortion_measure_holdout_normalized"])

    figure_3_metrics: List[Tuple[str, str, str]] = []
    for metric_name in [balanced_metric, holdout_metric, train_metric]:
        if metric_name is None:
            continue
        figure_3_metrics.append(
            (
                metric_name,
                _qe_panel_label(metric_name, METRIC_LABELS.get(metric_name, metric_name)),
                _slug(metric_name),
            )
        )

    comparison_specs = _sampling_comparison_specs(canonical_modes)
    manuscript_comparison_keys: List[str] = []
    if any(spec[0] == "full_vs_random" for spec in comparison_specs):
        manuscript_comparison_keys.append("full_vs_random")
    if any(spec[0] == "full_vs_hdsssom" for spec in comparison_specs):
        manuscript_comparison_keys.append("full_vs_hdsssom")
    if not manuscript_comparison_keys:
        preferred_comparison_key = str(sampling_primary_comparison_key or "full_vs_random").strip().lower()
        primary_comparison = next((spec for spec in comparison_specs if spec[0] == preferred_comparison_key), None)
        if primary_comparison is None and comparison_specs:
            primary_comparison = comparison_specs[0]
        if primary_comparison is not None:
            manuscript_comparison_keys.append(str(primary_comparison[0]))

    if manuscript_comparison_keys and figure_3_metrics:
        from .workflow_default_aware import _render_sampling_comparison_publication_figure

        comparison_payloads: Dict[str, Dict[str, object]] = {}
        diagnostic_keys = {
            "full_vs_random": "figure_2",
            "full_vs_hdsssom": "supp_figure_s4",
        }
        for comparison_key in manuscript_comparison_keys:
            comparison_spec = next((spec for spec in comparison_specs if spec[0] == comparison_key), None)
            if comparison_spec is None:
                continue
            _, comparison_label, _, _, _ = comparison_spec
            figure_3_comparison_scope = _sampling_comparison_scope_slug(
                canonical_modes,
                focus_comparison_key=comparison_key,
            )
            figure_3_payload = _render_sampling_comparison_publication_figure(
                sampling_comparison_dir=suite_output_dir / "sampling_comparison",
                publication_dir=publication_dir,
                metric_specs=figure_3_metrics,
                comparison_key=comparison_key,
                comparison_label=comparison_label,
                comparison_scope=figure_3_comparison_scope,
                generated_files=generated_files,
                dpi=int(dpi),
                alpha=float(alpha),
                legend_path=figure_2_forest_legend,
                secondary_legend_path=figure_2_regression_legend,
                diagnostics=diagnostics,
                diagnostics_prefix=diagnostics_prefix,
            )
            comparison_payloads[comparison_key] = figure_3_payload
            diagnostic_key = diagnostic_keys.get(comparison_key, comparison_key)
            if figure_3_payload.get("generated"):
                diagnostics_root["figures"][diagnostic_key] = str(
                    figure_3_payload.get("combined_figure", "")
                )
        diagnostics_root["sampling_publication"] = comparison_payloads

    if balanced_metric is not None and holdout_metric is not None and train_metric is not None:
        full_only_topology_fig_dir = suite_output_dir / "sampling" / "full" / "topology_main" / "figures"
        full_only_topology_table_dir = suite_output_dir / "sampling" / "full" / "topology_main" / "tables"
        topology_metric_columns: List[Tuple[str, str]] = [
            (balanced_metric, METRIC_LABELS.get(balanced_metric, balanced_metric)),
            (holdout_metric, METRIC_LABELS.get(holdout_metric, holdout_metric)),
            (train_metric, METRIC_LABELS.get(train_metric, train_metric)),
        ]
        topology_figure_specs: List[Dict[str, object]] = [
            {
                "slug": "hex_mst",
                "output_filename": "figure_4_topology_hex_mst_metrics_full_only.svg",
                "title_prefix": "Figure 4",
                "comparison_label": "Hexagonal vs MST",
                "outcome_template": "fig_hex_vs_mst_main_{metric_slug}.svg",
                "table_template": "hex_vs_mst_main_{metric_slug}.csv",
                "sensitivity_template": "fig_hex_vs_mst_sensitivity_main_{metric_slug}.svg",
                "sensitivity_legend": figure_3_sensitivity_legend,
                "diagnostic_key": "figure_4",
                "sensitivity_output_filename": "supp_figure_s6_topology_hex_mst_sensitivity_full_only.svg",
                "sensitivity_title_prefix": "Supplementary Figure S6",
                "sensitivity_diagnostic_key": "supp_figure_s6_hex_mst_sensitivity",
            },
            {
                "slug": "hex_rng",
                "output_filename": "figure_5_topology_hex_rng_metrics_full_only.svg",
                "title_prefix": "Figure 5",
                "comparison_label": "Hexagonal vs RNG",
                "outcome_template": "fig_hex_vs_rng_main_{metric_slug}.svg",
                "table_template": "hex_vs_rng_main_{metric_slug}.csv",
                "sensitivity_template": "fig_hex_vs_rng_sensitivity_main_{metric_slug}.svg",
                "sensitivity_legend": figure_3_sensitivity_legend_hex_rng,
                "diagnostic_key": "figure_5",
                "sensitivity_output_filename": "supp_figure_s7_topology_hex_rng_sensitivity_full_only.svg",
                "sensitivity_title_prefix": "Supplementary Figure S7",
                "sensitivity_diagnostic_key": "supp_figure_s7_hex_rng_sensitivity",
            },
            {
                "slug": "mst_rng",
                "output_filename": "supp_figure_topology_mst_rng_metrics_full_only.svg",
                "title_prefix": "Supplementary Figure S5",
                "comparison_label": "MST vs RNG",
                "outcome_template": "fig_mst_vs_rng_main_{metric_slug}.svg",
                "table_template": "mst_vs_rng_main_{metric_slug}.csv",
                "sensitivity_template": "fig_mst_vs_rng_sensitivity_main_{metric_slug}.svg",
                "sensitivity_legend": figure_3_sensitivity_legend_mst_rng,
                "diagnostic_key": "supp_figure_s5_mst_rng",
                "sensitivity_output_filename": "supp_figure_s8_topology_mst_rng_sensitivity_full_only.svg",
                "sensitivity_title_prefix": "Supplementary Figure S8",
                "sensitivity_diagnostic_key": "supp_figure_s8_mst_rng_sensitivity",
            },
        ]

        for spec in topology_figure_specs:
            slug = str(spec["slug"])
            output_filename = str(spec["output_filename"])
            title_prefix = str(spec["title_prefix"])
            diagnostic_key = str(spec["diagnostic_key"])
            comparison_label = str(spec["comparison_label"])
            outcome_template = str(spec["outcome_template"])
            table_template = str(spec["table_template"])
            sensitivity_template = str(spec["sensitivity_template"])
            sensitivity_legend = spec.get("sensitivity_legend")
            sensitivity_output_filename = str(spec["sensitivity_output_filename"])
            sensitivity_title_prefix = str(spec["sensitivity_title_prefix"])
            sensitivity_diagnostic_key = str(spec["sensitivity_diagnostic_key"])
            top_row_dataset_labels: List[str] = []

            outcome_panel_inputs: List[PublicationOutcomePanelInput] = []
            outcome_row: List[Tuple[str, str, Path]] = []
            sensitivity_row: List[Tuple[str, str, Path]] = []
            for idx, (metric_name, metric_display) in enumerate(topology_metric_columns):
                metric_slug = _slug(metric_name)
                panel_metric_label = _qe_panel_label(metric_name, metric_display)
                table_path = full_only_topology_table_dir / table_template.format(metric_slug=metric_slug)
                publication_panel_path = publication_dir / f"panel_{slug}_{metric_slug}.svg"
                if table_path.exists():
                    table_df = pd.read_csv(table_path)
                    outcome_panel_inputs.append(
                        PublicationOutcomePanelInput(
                            panel_letter=letters[idx],
                            metric_label=panel_metric_label,
                            metric_slug=metric_slug,
                            plot_df=table_df,
                            output_path=publication_panel_path,
                            title=f"{panel_metric_label}",
                            subtitle=comparison_label,
                            empty_message="Topology summary exists but has no plottable rows.",
                        )
                    )
                sensitivity_row.append(
                    (
                        letters[idx + 3],
                        panel_metric_label,
                        full_only_topology_fig_dir / sensitivity_template.format(metric_slug=metric_slug),
                    )
                )

            render_result = render_publication_outcome_panels(
                panel_inputs=outcome_panel_inputs,
                dpi=int(dpi),
                alpha=float(alpha),
                generated_files=generated_files,
            )
            outcome_row = list(render_result["panel_specs"])
            top_row_dataset_labels = list(render_result["top_row_dataset_labels"])

            direction_banner = _direction_banner_text(
                left_label=comparison_label.split(" vs ", 1)[1],
                right_label=comparison_label.split(" vs ", 1)[0],
            )

            output_path = publication_dir / output_filename
            if compose_publication_outcome_tripanel(
                panel_specs=outcome_row,
                figure_title=f"{title_prefix}: {comparison_label} Across QE Metrics",
                output_path=output_path,
                dpi=dpi,
                legend_path=figure_3_forest_legend,
                direction_label=direction_banner,
                generated_files=generated_files,
                external_y_labels=top_row_dataset_labels,
            ):
                diagnostics_root["figures"][diagnostic_key] = str(output_path)
                diagnostics_root["figures"][f"topology_{slug}"] = str(output_path)

            sensitivity_output_path = publication_dir / sensitivity_output_filename
            if _compose_three_panel_stats_figure(
                panel_rows=[sensitivity_row],
                title=f"{sensitivity_title_prefix}: {comparison_label} Sensitivity Across QE Metrics",
                output_path=sensitivity_output_path,
                dpi=dpi,
                legend_path=sensitivity_legend if isinstance(sensitivity_legend, Path) else None,
                direction_labels_by_row=[direction_banner],
                shared_y_labels_by_row=["Δ QE Mean (%)"],
                shared_x_labels_by_row=["Top-k Trials Used per Paired Unit"],
            ):
                generated_files.append(str(sensitivity_output_path))
                diagnostics_root["figures"][sensitivity_diagnostic_key] = str(sensitivity_output_path)
                diagnostics_root["figures"][f"topology_{slug}_sensitivity"] = str(sensitivity_output_path)

    diagnostics[f"{diagnostics_prefix}publication_figures"] = diagnostics_root

def _analyze_topology(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    strict_pairing: bool,
    expected_pairs_per_dataset: Optional[int],
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
    figure_scope: str = TOPOLOGY_SCOPE_MAIN,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if "architecture" not in df.columns:
        return

    subset = df[df["architecture"].isin(["mst", "hexagonal"])].copy()
    if subset.empty:
        return
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    required_cols = ["dataset", "algorithm", "pair_sampling", "pair_seed"]
    if strict_pairing:
        _validate_required_pairing_columns(subset, required_cols, context="hex-vs-mst")
    # Publication topology composites provide shared external row labels, so the
    # source panels should not embed per-panel dataset labels.
    show_forest_dataset_labels = False
    show_strata_dataset_labels = False
    show_sensitivity_y_axis_label = False
    sensitivity_y_axis_label = "Δ QE Mean (%)"
    # Keep panel plotting area consistent across A/B/C metric columns in Figure 4/5 composites.
    strata_left_margin = (
        PUBLICATION_SAFE_LEFT_MARGIN
        if show_strata_dataset_labels
        else PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS
    )
    batch_mode_values: List[str] = []
    if "pair_batch_mode" in subset.columns:
        batch_subset = subset[subset["algorithm"].astype(str).str.lower().str.strip() == "batch"].copy()
        if not batch_subset.empty:
            batch_mode_values = sorted(
                value
                for value in (
                    batch_subset["pair_batch_mode"]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .unique()
                    .tolist()
                )
                if value
            )

    if figure_scope == TOPOLOGY_SCOPE_MAIN:
        mode_configs: List[Tuple[str, str, str]] = [
            ("all_batch_modes_main", "All Batch Modes (Main)", "unpooled"),
        ]
    elif figure_scope == TOPOLOGY_SCOPE_SUPPLEMENTARY:
        mode_configs = [
            ("all_batch_modes", "All Batch Modes (Supplementary)", "unpooled"),
        ]
    else:
        raise ValueError(f"Unknown topology figure_scope: {figure_scope}")

    for mode_slug, mode_display, mode_kind in mode_configs:
        pooled_batch_modes = mode_kind == "pooled"
        mode_subset = subset.copy()
        if mode_kind == "full_batch_only" and "pair_batch_mode" in mode_subset.columns:
            is_batch = mode_subset["algorithm"].astype(str).str.lower().str.strip() == "batch"
            mode_subset = mode_subset[(~is_batch) | (mode_subset["pair_batch_mode"] == "full_batch")].copy()
            if mode_subset.empty:
                continue

        key_cols = ["dataset", "algorithm", "pair_sampling", "pair_seed", "pair_split"]
        if "pair_batch_mode" in mode_subset.columns and not mode_subset["pair_batch_mode"].isna().all():
            key_cols.insert(2, "pair_batch_mode")
        key_cols = [col for col in key_cols if col in mode_subset.columns and not mode_subset[col].isna().all()]

        pairs = _build_pairs(
            df=mode_subset,
            metric=metric,
            compare_col="architecture",
            value_a="hexagonal",
            value_b="mst",
            key_cols=key_cols,
            top_k=top_k,
            reference_value="hexagonal",
        )
        if pairs.empty:
            continue

        summary_pairs = _filter_pairs_to_sampling_modes(pairs, ["full"])
        if summary_pairs.empty:
            continue

        summary_pairs, strata = _build_topology_algorithm_mode_strata(
            summary_pairs,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
            test_method="paired_t",
            location_ci_method="paired_t",
        )

        summary = _summarize_pairs(
            pairs=summary_pairs,
            group_cols=["dataset"],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        summary = _apply_q_values(summary, dataset_col="dataset")
        summary = _add_global_row(
            summary_df=summary,
            pairs=summary_pairs,
            dataset_col="dataset",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        expected_pairs_for_mode = expected_pairs_per_dataset if figure_scope == TOPOLOGY_SCOPE_MAIN else None
        _validate_expected_pairs(
            summary,
            dataset_col="dataset",
            expected_pairs=expected_pairs_for_mode,
            context=f"hex-vs-mst ({mode_display})",
        )

        metric_slug = _slug(metric)
        mode_file_suffix = "_main" if figure_scope == TOPOLOGY_SCOPE_MAIN else "_supplementary"
        mode_title_suffix = f", {mode_display}"
        table_path = output_dir / "tables" / f"hex_vs_mst{mode_file_suffix}_{metric_slug}.csv"
        strata_path = output_dir / "tables" / f"hex_vs_mst_strata_algorithm_mode{mode_file_suffix}_{metric_slug}.csv"
        pairs_path = output_dir / "tables" / f"hex_vs_mst_pairs{mode_file_suffix}_{metric_slug}.csv"
        stats_table_path = output_dir / "tables" / f"hex_vs_mst_stats{mode_file_suffix}_{metric_slug}.csv"
        fig_path = output_dir / "figures" / f"fig_hex_vs_mst{mode_file_suffix}_{metric_slug}.svg"
        strata_fig_path = output_dir / "figures" / f"fig_hex_vs_mst_by_algorithm_mode{mode_file_suffix}_{metric_slug}.svg"

        summary.to_csv(table_path, index=False)
        strata.to_csv(strata_path, index=False)
        summary_pairs.to_csv(pairs_path, index=False)

        _plot_dataset_forest_publication(
            summary_df=summary,
            title=f"Hexagonal vs MST ({metric_label}{mode_title_suffix})",
            subtitle="Paired t mean % improvement of Hexagonal over MST using denominator |Hexagonal| with 95% paired t-test CI",
            output_path=fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            stats_table_output_path=stats_table_path,
            x_axis_label="",
            show_y_tick_labels=show_forest_dataset_labels,
            **_tripanel_forest_panel_style(show_dataset_labels=show_forest_dataset_labels),
        )

        if not strata.empty and "algorithm_batch_mode" in strata.columns:
            _plot_topology_algorithm_strata(
                strata_df=strata,
                dataset_col="dataset",
                algorithm_col="algorithm_batch_mode",
                title=f"Hexagonal vs MST by Algorithm + Batch Mode ({metric_label}{mode_title_suffix})",
                subtitle="Dataset-level paired t mean % improvement, stratified by algorithm/batch-mode",
                output_path=strata_fig_path,
                dpi=dpi,
                alpha=alpha,
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                show_y_tick_labels=show_strata_dataset_labels,
                left_margin_override=strata_left_margin,
                x_axis_label="",
                use_tight_bbox=False,
                **_tripanel_topology_strata_text_style(show_dataset_labels=show_strata_dataset_labels),
            )

        generated_files.extend(
            [str(table_path), str(strata_path), str(pairs_path), str(stats_table_path), str(fig_path), str(strata_fig_path)]
        )

        summary_baseline = _summarize_pairs(
            pairs=summary_pairs,
            group_cols=["dataset"],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_baseline_ref",
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        summary_baseline = _apply_q_values(summary_baseline, dataset_col="dataset")
        summary_baseline = _add_global_row(
            summary_df=summary_baseline,
            pairs=summary_pairs,
            dataset_col="dataset",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_baseline_ref",
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        _, strata_baseline = _build_topology_algorithm_mode_strata(
            summary_pairs,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_baseline_ref",
            test_method="paired_t",
            location_ci_method="paired_t",
        )
        baseline_table_path = output_dir / "tables" / f"hex_vs_mst_baseline_ref{mode_file_suffix}_{metric_slug}.csv"
        baseline_strata_path = (
            output_dir / "tables" / f"hex_vs_mst_strata_algorithm_mode_baseline_ref{mode_file_suffix}_{metric_slug}.csv"
        )
        baseline_stats_table_path = output_dir / "tables" / f"hex_vs_mst_stats_baseline_ref{mode_file_suffix}_{metric_slug}.csv"
        baseline_fig_path = output_dir / "figures" / f"fig_hex_vs_mst_baseline_ref{mode_file_suffix}_{metric_slug}.svg"
        baseline_strata_fig_path = (
            output_dir / "figures" / f"fig_hex_vs_mst_by_algorithm_mode_baseline_ref{mode_file_suffix}_{metric_slug}.svg"
        )
        summary_baseline.to_csv(baseline_table_path, index=False)
        strata_baseline.to_csv(baseline_strata_path, index=False)
        _plot_dataset_forest_publication(
            summary_df=summary_baseline,
            title=f"Hexagonal vs MST ({metric_label}, Baseline Reference{mode_title_suffix})",
            subtitle="Paired t mean % improvement of Hexagonal over MST using denominator |MST| with 95% paired t-test CI",
            output_path=baseline_fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            stats_table_output_path=baseline_stats_table_path,
            x_axis_label="",
            show_y_tick_labels=show_forest_dataset_labels,
            **_tripanel_forest_panel_style(show_dataset_labels=show_forest_dataset_labels),
        )
        if not strata_baseline.empty and "algorithm_batch_mode" in strata_baseline.columns:
            _plot_topology_algorithm_strata(
                strata_df=strata_baseline,
                dataset_col="dataset",
                algorithm_col="algorithm_batch_mode",
                title=f"Hexagonal vs MST by Algorithm + Batch Mode ({metric_label}, Baseline Reference{mode_title_suffix})",
                subtitle="Dataset-level paired t mean % improvement (baseline ref), stratified by algorithm/batch-mode",
                output_path=baseline_strata_fig_path,
                dpi=dpi,
                alpha=alpha,
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                show_y_tick_labels=show_strata_dataset_labels,
                left_margin_override=strata_left_margin,
                x_axis_label="",
                use_tight_bbox=False,
                **_tripanel_topology_strata_text_style(show_dataset_labels=show_strata_dataset_labels),
            )
        generated_files.extend(
            [
                str(baseline_table_path),
                str(baseline_strata_path),
                str(baseline_stats_table_path),
                str(baseline_fig_path),
                str(baseline_strata_fig_path),
            ]
        )

        # Topology interaction heatmap by dataset x algorithm/batch-mode stratum
        interaction_df = (
            summary_pairs.assign(hex_beats_mst=(summary_pairs["score_a"] < summary_pairs["score_b"]).astype(float))
            .groupby(["dataset", "algorithm_batch_mode"], dropna=False)
            .agg(
                n_pairs=("hex_beats_mst", "size"),
                beats_rate_pct=("hex_beats_mst", lambda x: float(np.mean(x)) * 100.0),
            )
            .reset_index()
        )
        interaction_table_path = (
            output_dir / "tables" / f"hex_vs_mst_interaction_algorithm_mode{mode_file_suffix}_{metric_slug}.csv"
        )
        interaction_fig_path = (
            output_dir / "figures" / f"fig_hex_vs_mst_interaction_algorithm_mode{mode_file_suffix}_{metric_slug}.svg"
        )
        interaction_df.to_csv(interaction_table_path, index=False)
        _plot_heatmap(
            heat_df=interaction_df,
            index_col="dataset",
            column_col="algorithm_batch_mode",
            value_col="beats_rate_pct",
            title=f"Hexagonal vs MST: Win Rate by Algorithm + Batch Mode ({metric_label}{mode_title_suffix})",
            subtitle="Percent of paired units where Hexagonal beats MST (lower score)",
            output_path=interaction_fig_path,
            dpi=dpi,
            cmap="Greens",
        )
        generated_files.extend([str(interaction_table_path), str(interaction_fig_path)])

        # Sensitivity curve
        sensitivity_df = _build_sensitivity_summary(
            df=mode_subset,
            metric=metric,
            compare_col="architecture",
            value_a="hexagonal",
            value_b="mst",
            key_cols=key_cols,
            top_k_values=DEFAULT_SENSITIVITY_TOP_K,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            reference_value="hexagonal",
            test_method="paired_t",
            location_ci_method="paired_t",
            pair_sampling_modes=["full"],
        )
        sensitivity_table_path = output_dir / "tables" / f"hex_vs_mst_sensitivity{mode_file_suffix}_{metric_slug}.csv"
        sensitivity_fig_path = output_dir / "figures" / f"fig_hex_vs_mst_sensitivity{mode_file_suffix}_{metric_slug}.svg"
        sensitivity_df.to_csv(sensitivity_table_path, index=False)
        (
            topology_sensitivity_series,
            topology_sensitivity_styles,
            topology_sensitivity_primary_mode,
        ) = _build_topology_sensitivity_series(
            df=mode_subset,
            pairs=pairs,
            metric=metric,
            compare_col="architecture",
            value_a="hexagonal",
            value_b="mst",
            key_cols=key_cols,
            top_k_values=DEFAULT_SENSITIVITY_TOP_K,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_reference",
            reference_value="hexagonal",
            exclude_algorithm_batch_mode_labels={"colors"},
            test_method="paired_t",
            location_ci_method="paired_t",
            pair_sampling_modes=["full"],
        )
        if len(topology_sensitivity_series) >= 2:
            _plot_sensitivity_curve_multiseries(
                sensitivity_series_by_mode=topology_sensitivity_series,
                mode_styles=topology_sensitivity_styles,
                primary_mode=topology_sensitivity_primary_mode,
                title=f"Hexagonal vs MST: Sensitivity to Top-k ({metric_label}{mode_title_suffix})",
                subtitle=(
                    "Global pooled paired mean % improvement by algorithm/batch-mode stratum "
                    "using denominator |Hexagonal| with dotted 95% CI bounds"
                ),
                output_path=sensitivity_fig_path,
                dpi=dpi,
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                x_axis_label="",
                y_axis_label=sensitivity_y_axis_label,
                show_y_axis_label=show_sensitivity_y_axis_label,
                use_tight_bbox=False,
                left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
                bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
            )
        else:
            _plot_sensitivity_curve(
                sensitivity_df=sensitivity_df,
                title=f"Hexagonal vs MST: Sensitivity to Top-k ({metric_label}{mode_title_suffix})",
                subtitle="Global pooled paired mean % improvement using denominator |Hexagonal|",
                output_path=sensitivity_fig_path,
                dpi=dpi,
                series_color=_comparison_series_color(compare_col="architecture", value_a="hexagonal"),
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                x_axis_label="",
                y_axis_label=sensitivity_y_axis_label,
                show_y_axis_label=show_sensitivity_y_axis_label,
                use_tight_bbox=False,
                left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
                bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
            )
        generated_files.extend([str(sensitivity_table_path), str(sensitivity_fig_path)])

        sensitivity_baseline_df = _build_sensitivity_summary(
            df=mode_subset,
            metric=metric,
            compare_col="architecture",
            value_a="hexagonal",
            value_b="mst",
            key_cols=key_cols,
            top_k_values=DEFAULT_SENSITIVITY_TOP_K,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_baseline_ref",
            test_method="paired_t",
            location_ci_method="paired_t",
            pair_sampling_modes=["full"],
        )
        sensitivity_baseline_table_path = (
            output_dir / "tables" / f"hex_vs_mst_sensitivity_baseline_ref{mode_file_suffix}_{metric_slug}.csv"
        )
        sensitivity_baseline_fig_path = (
            output_dir / "figures" / f"fig_hex_vs_mst_sensitivity_baseline_ref{mode_file_suffix}_{metric_slug}.svg"
        )
        sensitivity_baseline_df.to_csv(sensitivity_baseline_table_path, index=False)
        (
            topology_sensitivity_baseline_series,
            topology_sensitivity_baseline_styles,
            topology_sensitivity_baseline_primary_mode,
        ) = _build_topology_sensitivity_series(
            df=mode_subset,
            pairs=pairs,
            metric=metric,
            compare_col="architecture",
            value_a="hexagonal",
            value_b="mst",
            key_cols=key_cols,
            top_k_values=DEFAULT_SENSITIVITY_TOP_K,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            pct_col="pct_improvement_a_over_b_baseline_ref",
            exclude_algorithm_batch_mode_labels={"colors"},
            test_method="paired_t",
            location_ci_method="paired_t",
            pair_sampling_modes=["full"],
        )
        if len(topology_sensitivity_baseline_series) >= 2:
            _plot_sensitivity_curve_multiseries(
                sensitivity_series_by_mode=topology_sensitivity_baseline_series,
                mode_styles=topology_sensitivity_baseline_styles,
                primary_mode=topology_sensitivity_baseline_primary_mode,
                title=f"Hexagonal vs MST: Sensitivity to Top-k ({metric_label}, Baseline Reference{mode_title_suffix})",
                subtitle=(
                    "Global pooled paired mean % improvement by algorithm/batch-mode stratum "
                    "using denominator |MST| with dotted 95% CI bounds"
                ),
                output_path=sensitivity_baseline_fig_path,
                dpi=dpi,
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                x_axis_label="",
                y_axis_label=sensitivity_y_axis_label,
                show_y_axis_label=show_sensitivity_y_axis_label,
                use_tight_bbox=False,
                left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
                bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
            )
        else:
            _plot_sensitivity_curve(
                sensitivity_df=sensitivity_baseline_df,
                title=f"Hexagonal vs MST: Sensitivity to Top-k ({metric_label}, Baseline Reference{mode_title_suffix})",
                subtitle="Global pooled paired mean % improvement using denominator |MST|",
                output_path=sensitivity_baseline_fig_path,
                dpi=dpi,
                series_color=_comparison_series_color(compare_col="architecture", value_a="hexagonal"),
                inline_legends=inline_legends,
                legend_registry=legend_registry,
                x_axis_label="",
                y_axis_label=sensitivity_y_axis_label,
                show_y_axis_label=show_sensitivity_y_axis_label,
                use_tight_bbox=False,
                left_margin_override=TOPOLOGY_SENSITIVITY_PLOT_LEFT_MARGIN,
                bottom_margin_override=TOPOLOGY_SENSITIVITY_PLOT_BOTTOM_MARGIN,
            )
        generated_files.extend([str(sensitivity_baseline_table_path), str(sensitivity_baseline_fig_path)])

        # P-value panel
        pvalue_table_path = output_dir / "tables" / f"hex_vs_mst_pvalue_panel{mode_file_suffix}_{metric_slug}.csv"
        pvalue_stats_table_path = output_dir / "tables" / f"hex_vs_mst_pvalue_stats{mode_file_suffix}_{metric_slug}.csv"
        pvalue_fig_path = output_dir / "figures" / f"fig_hex_vs_mst_pvalue_panel{mode_file_suffix}_{metric_slug}.svg"
        pvalue_df = summary[~summary["dataset"].map(_is_global_dataset_label)].copy()
        pvalue_df["minus_log10_p"] = -np.log10(np.clip(pvalue_df["p_value"].astype(float), 1e-300, 1.0))
        pvalue_df.to_csv(pvalue_table_path, index=False)
        _plot_pvalue_panel(
            summary_df=summary,
            dataset_col="dataset",
            title=f"Hexagonal vs MST: Dataset Significance Panel ({metric_label}{mode_title_suffix})",
            subtitle="Paired t-test p-values by dataset",
            output_path=pvalue_fig_path,
            dpi=dpi,
            alpha=alpha,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
            stats_table_output_path=pvalue_stats_table_path,
        )
        generated_files.extend([str(pvalue_table_path), str(pvalue_stats_table_path), str(pvalue_fig_path)])

        diagnostics_key = f"{diagnostics_prefix}topology_{figure_scope}_{metric_slug}"
        diagnostics[diagnostics_key] = {
            "pairing_keys": key_cols,
            "figure_scope": figure_scope,
            "batch_mode_scope": mode_slug,
            "batch_mode_scope_label": mode_display,
            "pooled_batch_modes": bool(pooled_batch_modes),
            "uses_stratified_sensitivity": len(topology_sensitivity_series) >= 2,
            "sensitivity_strata": list(topology_sensitivity_series.keys()),
            "per_dataset_pairs": summary[~summary["dataset"].map(_is_global_dataset_label)][
                ["dataset", "n_pairs"]
            ].to_dict(orient="records"),
        }
__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
