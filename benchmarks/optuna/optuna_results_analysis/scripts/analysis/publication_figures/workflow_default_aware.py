from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from collections import OrderedDict
from datetime import UTC, datetime
from functools import lru_cache
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

from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.default_benchmark_adapter import (
    load_default_runs_csv,
)
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.hyperparameter_stability import (
    StabilityMetric,
    build_hyperparameter_stability_report,
    stability_table_filename,
)
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.tuned_vs_default import (
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
    CANONICAL_PUBLICATION_QE_CHANGE_LABEL,
    PublicationOutcomePanelInput,
    compose_publication_outcome_tripanel,
    render_publication_outcome_panels,
)

from .workflow_analysis_generation import (
    _direction_banner_text,
    _infer_sampling_regression_table_from_diagnostics,
    _humanize_comparison_label,
    _render_selected_parameter_stability_vs_sample_size_regression,
)


def _figure_3_dataset_metadata_table_name(comparison_scope: str) -> str:
    if comparison_scope == "all_sampling_modes":
        return "figure_3_full_vs_random_dataset_metadata.csv"
    return f"figure_3_{comparison_scope}_full_vs_random_dataset_metadata.csv"


def _write_figure_3_dataset_metadata_table(
    *,
    regression_table_paths: Sequence[Path],
    output_path: Path,
) -> Optional[Path]:
    metadata_frames: List[pd.DataFrame] = []
    required_columns = ["dataset", "dimension_count", "sample_size", "dataset_type"]
    for table_path in regression_table_paths:
        if not table_path.exists():
            continue
        table_df = pd.read_csv(table_path)
        if not set(required_columns).issubset(table_df.columns):
            continue
        available_columns = [column for column in ["dataset_index", *required_columns] if column in table_df.columns]
        metadata_frames.append(table_df[available_columns].copy())
    if not metadata_frames:
        return None

    metadata_df = pd.concat(metadata_frames, ignore_index=True)
    metadata_df["dataset"] = metadata_df["dataset"].astype(str).str.strip()
    metadata_df = metadata_df[metadata_df["dataset"] != ""].copy()
    metadata_df["sample_size"] = pd.to_numeric(metadata_df["sample_size"], errors="coerce").round().astype("Int64")
    metadata_df["dimension_count"] = pd.to_numeric(
        metadata_df["dimension_count"],
        errors="coerce",
    ).round().astype("Int64")
    metadata_df["dataset_type"] = metadata_df["dataset_type"].astype(str).str.strip()
    metadata_df = metadata_df.sort_values(["sample_size", "dataset"], ascending=[True, True], na_position="last")
    metadata_df = metadata_df.drop_duplicates(subset=["dataset"], keep="first").reset_index(drop=True)
    metadata_df = _attach_dataset_indices(
        metadata_df,
        dataset_col="dataset",
        sample_size_col="sample_size",
    )
    metadata_df["dimension_count"] = metadata_df["dimension_count"].astype(object)
    metadata_df["dimension_count"] = metadata_df["dimension_count"].where(
        metadata_df["dimension_count"].notna(),
        "NA",
    )
    metadata_df = metadata_df[
        ["dataset_index", "dataset", "dimension_count", "sample_size", "dataset_type"]
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)
    return output_path


def _render_default_aware_stability_publication_figures(
    *,
    stability_dir: Path,
    metric_specs: Sequence[Tuple[str, str, str]],
    selected_parameters: Sequence[str],
    output_dir: Path,
    generated_files: List[str],
    dpi: int,
    manuscript_only: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, str] = {}
    sampling_mode_figures: Dict[str, Dict[str, str]] = {}
    sampling_mode_size_figures: Dict[str, Dict[str, str]] = {}
    tables: Dict[str, str] = {}
    missing_or_empty: Dict[str, str] = {}

    for _, metric_label, metric_slug in metric_specs:
        sampling_tables: Dict[str, pd.DataFrame] = {}
        sampling_dataset_tables: Dict[str, pd.DataFrame] = {}
        combined_rows: List[pd.DataFrame] = []
        combined_dataset_rows: List[pd.DataFrame] = []
        for sampling_mode in DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES:
            sampling_tables_dir = stability_dir / metric_slug / f"sampling_{sampling_mode}" / "tables"
            table_hex_mst_path = sampling_tables_dir / stability_table_filename(
                "global", "hexagonal", "mst", metric_slug
            )
            table_hex_rng_path = sampling_tables_dir / stability_table_filename(
                "global", "hexagonal", "rng", metric_slug
            )
            dataset_hex_mst_path = sampling_tables_dir / stability_table_filename(
                "dataset", "hexagonal", "mst", metric_slug
            )
            dataset_hex_rng_path = sampling_tables_dir / stability_table_filename(
                "dataset", "hexagonal", "rng", metric_slug
            )
            if (
                not table_hex_mst_path.exists()
                or not table_hex_rng_path.exists()
                or not dataset_hex_mst_path.exists()
                or not dataset_hex_rng_path.exists()
            ):
                missing_or_empty[f"{metric_slug}.{sampling_mode}"] = (
                    "Missing sampling-specific stability tables: "
                    f"hex_vs_mst={table_hex_mst_path}, hex_vs_rng={table_hex_rng_path}, "
                    f"dataset_hex_vs_mst={dataset_hex_mst_path}, dataset_hex_vs_rng={dataset_hex_rng_path}"
                )
                continue

            selected_table = _build_selected_parameter_stability_three_topology_table(
                parameter_global_hex_mst_df=pd.read_csv(table_hex_mst_path),
                parameter_global_hex_rng_df=pd.read_csv(table_hex_rng_path),
                selected_parameters=selected_parameters,
            )
            if selected_table.empty:
                missing_or_empty[f"{metric_slug}.{sampling_mode}"] = (
                    "No selected parameter rows found in sampling-specific stability tables."
                )
                continue
            selected_table.insert(0, "sampling_mode", sampling_mode)
            sampling_tables[sampling_mode] = selected_table
            combined_rows.append(selected_table)

            dataset_selected_table = _build_dataset_selected_parameter_stability_three_topology_table(
                parameter_dataset_hex_mst_df=pd.read_csv(dataset_hex_mst_path),
                parameter_dataset_hex_rng_df=pd.read_csv(dataset_hex_rng_path),
                selected_parameters=selected_parameters,
            )
            if dataset_selected_table.empty:
                missing_or_empty[f"{metric_slug}.{sampling_mode}.dataset"] = (
                    "No dataset-level selected stability rows found in sampling-specific stability tables."
                )
            else:
                dataset_selected_table.insert(0, "sampling_mode", sampling_mode)
                sampling_dataset_tables[sampling_mode] = dataset_selected_table
                combined_dataset_rows.append(dataset_selected_table)

        if not sampling_tables:
            missing_or_empty[metric_slug] = "No sampling strata available for selected-parameter stability publication."
            continue

        if not manuscript_only:
            selected_table_path = (
                output_dir
                / f"table_stability_selected_params_hexagonal_mst_rng_full_vs_random_{metric_slug}.csv"
            )
            combined_table = pd.concat(combined_rows, ignore_index=True)
            combined_table.to_csv(selected_table_path, index=False)
            generated_files.append(str(selected_table_path.resolve()))
            tables[metric_slug] = str(selected_table_path.resolve())

            if combined_dataset_rows:
                dataset_table_path = (
                    output_dir
                    / f"table_stability_selected_params_vs_sample_size_full_vs_random_{metric_slug}.csv"
                )
                pd.concat(combined_dataset_rows, ignore_index=True).to_csv(dataset_table_path, index=False)
                generated_files.append(str(dataset_table_path.resolve()))

            figure_path = (
                output_dir
                / f"fig_stability_selected_params_hexagonal_mst_rng_full_vs_random_{metric_slug}.svg"
            )
            _plot_selected_parameter_stability_stratified_figure(
                sampling_tables=sampling_tables,
                output_path=figure_path,
                title="Selected Hyperparameter Stability",
                subtitle=f"{metric_label}: Full vs Random sampling, with Hexagonal/MST/RNG",
                dpi=dpi,
            )
            if not figure_path.exists():
                missing_or_empty[metric_slug] = f"Figure was not generated for {metric_slug}."
                continue
            generated_files.append(str(figure_path.resolve()))
            figures[metric_slug] = str(figure_path.resolve())

        mode_specific_paths: Dict[str, str] = {}
        for sampling_mode in DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES:
            mode_table = sampling_tables.get(sampling_mode)
            if mode_table is None or mode_table.empty:
                missing_or_empty[f"{metric_slug}.sampling_mode.{sampling_mode}"] = (
                    f"No selected-parameter stability rows available for sampling mode '{sampling_mode}'."
                )
                continue
            mode_figure_path = (
                output_dir
                / f"fig_stability_selected_params_hexagonal_mst_rng_{sampling_mode}_{metric_slug}.svg"
            )
            _plot_selected_parameter_stability_stratified_figure(
                sampling_tables={sampling_mode: mode_table},
                output_path=mode_figure_path,
                title=f"Selected Hyperparameter Stability ({_sampling_mode_display(sampling_mode)})",
                subtitle=f"{metric_label}: Hexagonal/MST/RNG",
                dpi=dpi,
                show_titles=False,
                show_mode_titles=False,
                show_shared_y_label=False,
                show_shared_x_label=False,
                x_axis_label="Stability Score (Lower Is Better)",
                figure_width_override=10.8,
                figure_height_override=5.8,
                left_margin_override=0.045,
                bottom_margin_override=0.055,
                use_tight_bbox=True,
            )
            if mode_figure_path.exists():
                generated_files.append(str(mode_figure_path.resolve()))
                mode_specific_paths[sampling_mode] = str(mode_figure_path.resolve())
            else:
                missing_or_empty[f"{metric_slug}.sampling_mode.{sampling_mode}"] = (
                    f"Sampling-mode stability figure was not generated for {metric_slug} ({sampling_mode})."
                )
        if mode_specific_paths:
            sampling_mode_figures[metric_slug] = mode_specific_paths

        mode_specific_size_paths: Dict[str, str] = {}
        for sampling_mode in DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES:
            mode_dataset_table = sampling_dataset_tables.get(sampling_mode)
            if mode_dataset_table is None or mode_dataset_table.empty:
                missing_or_empty[f"{metric_slug}.sampling_mode_size.{sampling_mode}"] = (
                    f"No dataset-level selected stability rows available for sampling mode '{sampling_mode}'."
                )
                continue

            mode_dataset_table_path = output_dir / f"table_stability_selected_params_vs_sample_size_{sampling_mode}_{metric_slug}.csv"
            mode_dataset_stats_path = output_dir / f"stats_stability_selected_params_vs_sample_size_{sampling_mode}_{metric_slug}.csv"
            mode_size_figure_path = output_dir / f"fig_stability_selected_params_vs_sample_size_{sampling_mode}_{metric_slug}.svg"
            regression_result = _render_selected_parameter_stability_vs_sample_size_regression(
                dataset_summary_df=mode_dataset_table,
                sampling_mode=sampling_mode,
                output_figure_path=mode_size_figure_path,
                output_table_path=mode_dataset_table_path,
                output_stats_path=mode_dataset_stats_path,
                dpi=dpi,
                show_mode_title=False,
                x_axis_label="Samples used per run (log10)",
                y_axis_label="",
                left_margin_override=0.09,
                bottom_margin_override=0.09,
                figure_width_override=10.8,
                figure_height_override=7.2,
                use_tight_bbox=False,
            )
            if regression_result.get("generated"):
                generated_files.extend(
                    [
                        str(mode_dataset_table_path.resolve()),
                        str(mode_dataset_stats_path.resolve()),
                        str(mode_size_figure_path.resolve()),
                    ]
                )
                mode_specific_size_paths[sampling_mode] = str(mode_size_figure_path.resolve())
            else:
                missing_or_empty[f"{metric_slug}.sampling_mode_size.{sampling_mode}"] = str(
                    regression_result.get("reason", "Size-regression stability figure was not generated.")
                )
        if mode_specific_size_paths:
            sampling_mode_size_figures[metric_slug] = mode_specific_size_paths

    legend_output_path = output_dir / "legend_stability_selected_params_hexagonal_mst_rng.svg"
    legend_path = _render_selected_parameter_stability_legend(
        output_path=legend_output_path,
        topology_values=DEFAULT_AWARE_STABILITY_PUBLICATION_TOPOLOGIES,
        dpi=int(dpi),
    )
    if legend_path is not None and legend_path.exists():
        generated_files.append(str(legend_path.resolve()))

    combined_figure_path = output_dir / "fig_stability_selected_params_full_vs_random_all_metrics.svg"
    combined_generated = False
    if not manuscript_only:
        panel_specs: List[Tuple[str, str, Path]] = []
        panel_letters = [chr(ord("A") + idx) for idx in range(26)]
        for idx, (_, metric_label, metric_slug) in enumerate(metric_specs):
            metric_figure_path = figures.get(metric_slug, "")
            if not metric_figure_path:
                continue
            panel_specs.append((panel_letters[idx], metric_label, Path(metric_figure_path)))
        if panel_specs:
            combined_generated = _compose_panel_matrix_from_images(
                panel_rows=[panel_specs],
                title="Selected Hyperparameter Stability by Sampling (Full vs Random)",
                output_path=combined_figure_path,
                dpi=int(dpi),
                legend_path=legend_path if legend_path is not None and legend_path.exists() else None,
                legend_band_height_override=PUBLICATION_LEGEND_ROW_HEIGHT,
                legend_gap_override=1.0,
                margin_bottom_override=4.0,
                gap_x_override=6.0,
            )
            if combined_generated and combined_figure_path.exists():
                generated_files.append(str(combined_figure_path.resolve()))

    figure_7_path = output_dir / DEFAULT_AWARE_STABILITY_FIGURE7_ASSET_FILENAME
    figure_7_generated = False
    mode_panels = sampling_mode_figures.get(DEFAULT_AWARE_STABILITY_FIGURE7_METRIC_SLUG, {})
    mode_size_panels = sampling_mode_size_figures.get(DEFAULT_AWARE_STABILITY_FIGURE7_METRIC_SLUG, {})
    panel_a_path_text = mode_panels.get("full", "")
    panel_b_path_text = mode_panels.get("random", "")
    panel_c_path_text = mode_size_panels.get("random", "")
    if panel_a_path_text and panel_b_path_text and panel_c_path_text:
        figure_7_generated = _compose_three_panel_stats_figure(
            panel_rows=[
                [
                    (
                        "A",
                        "Full sampling",
                        Path(panel_a_path_text),
                    ),
                    (
                        "B",
                        "Random sampling",
                        Path(panel_b_path_text),
                    ),
                    (
                        "C",
                        "Random sampling vs sample size",
                        Path(panel_c_path_text),
                    ),
                ],
            ],
            title="Figure 7: Hyperparameter Stability",
            output_path=figure_7_path,
            dpi=int(dpi),
            legend_path=legend_path if legend_path is not None and legend_path.exists() else None,
            legend_band_height_override=150,
            legend_gap_override=10.0,
            margin_bottom_override=4.0,
            gap_x_override=8.0,
            panel_width_override=760.0,
            panel_height_override=500.0,
        )
        if figure_7_generated and figure_7_path.exists():
            generated_files.append(str(figure_7_path.resolve()))
    else:
        missing_or_empty["figure_7"] = (
            "Figure 7 composition skipped because one or more tripanel panels are missing "
            f"(A={panel_a_path_text!r}, B={panel_b_path_text!r}, C={panel_c_path_text!r})."
        )

    return {
        "generated": bool(figures),
        "sampling_modes": list(DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES),
        "topologies": list(DEFAULT_AWARE_STABILITY_PUBLICATION_TOPOLOGIES),
        "selected_parameters": list(selected_parameters),
        "figures": figures,
        "sampling_mode_figures": sampling_mode_figures,
        "sampling_mode_size_figures": sampling_mode_size_figures,
        "combined_figure": str(combined_figure_path.resolve()) if combined_generated and combined_figure_path.exists() else "",
        "figure_7": str(figure_7_path.resolve()) if figure_7_generated and figure_7_path.exists() else "",
        "tables": tables,
        "legend": str(legend_path.resolve()) if legend_path is not None and legend_path.exists() else "",
        "missing_or_empty": missing_or_empty,
    }

def _render_tuned_vs_default_publication_figure(
    *,
    tuned_vs_default_dir: Path,
    metric_specs: Sequence[Tuple[str, str, str]],
    generated_files: List[str],
    dpi: int,
    alpha: float,
    combined_figure_title: str,
) -> Dict[str, object]:
    tables_dir = tuned_vs_default_dir / "tables"
    figures_dir = tuned_vs_default_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel_inputs: List[PublicationOutcomePanelInput] = []
    panel_specs: List[Tuple[str, str, Path]] = []
    per_metric_figures: Dict[str, str] = {}
    missing_or_empty: Dict[str, str] = {}
    top_row_dataset_labels: List[str] = []
    letters = [chr(ord("A") + idx) for idx in range(26)]

    for idx, (_, metric_label, metric_slug) in enumerate(metric_specs):
        dataset_summary_path = tables_dir / f"tuned_vs_default_dataset_summary_{metric_slug}.csv"
        global_summary_path = tables_dir / f"tuned_vs_default_global_summary_{metric_slug}.csv"
        if not dataset_summary_path.exists():
            missing_or_empty[metric_slug] = f"Missing dataset summary table: {dataset_summary_path}"
            continue

        summary_df = pd.read_csv(dataset_summary_path)
        global_summary_df = pd.read_csv(global_summary_path) if global_summary_path.exists() else pd.DataFrame()
        summary_df = _merge_tuned_vs_default_global_row(summary_df, global_summary_df)
        plot_df = _prepare_tuned_vs_default_dataset_summary_for_forest(summary_df)
        if plot_df.empty:
            missing_or_empty[metric_slug] = "Dataset summary exists but has no plottable rows."
            continue

        figure_path = figures_dir / f"fig_tuned_vs_true_default_{metric_slug}.svg"
        stats_path = tables_dir / f"tuned_vs_true_default_stats_{metric_slug}.csv"
        panel_inputs.append(
            PublicationOutcomePanelInput(
                panel_letter=letters[idx],
                metric_label=metric_label,
                metric_slug=metric_slug,
                plot_df=plot_df,
                output_path=figure_path,
                title=f"Tuned Configuration vs Untuned Reference ({metric_label})",
                subtitle=(
                    "Paired effect with 95% paired t-test CI. "
                    "Positive means tuned configuration outperforms untuned reference; global pools all matched pairs."
                ),
                stats_table_output_path=stats_path,
            )
        )

    render_result = render_publication_outcome_panels(
        panel_inputs=panel_inputs,
        dpi=int(dpi),
        alpha=float(alpha),
        generated_files=generated_files,
    )
    panel_specs = list(render_result["panel_specs"])
    per_metric_figures = dict(render_result["per_metric_figures"])
    top_row_dataset_labels = list(render_result["top_row_dataset_labels"])
    missing_or_empty.update(render_result["missing_or_empty"])

    combined_figure = figures_dir / "fig_tuned_vs_true_default_metrics.svg"
    legend_path = _render_composite_legend_key(
        output_path=figures_dir / "legend_tuned_vs_true_default.svg",
        dpi=int(dpi),
        alpha=float(alpha),
        include_forest=True,
        include_sensitivity=False,
        compact=False,
    )
    if legend_path is not None and legend_path.exists():
        generated_files.append(str(legend_path.resolve()))

    generated = False
    if panel_specs:
        generated = compose_publication_outcome_tripanel(
            panel_specs=panel_specs,
            figure_title=combined_figure_title,
            output_path=combined_figure,
            dpi=int(dpi),
            legend_path=legend_path if legend_path is not None and legend_path.exists() else None,
            direction_label=_direction_banner_text(left_label="Untuned Reference", right_label="Tuned Configuration"),
            generated_files=generated_files,
            external_y_labels=top_row_dataset_labels,
            shared_x_label=CANONICAL_PUBLICATION_QE_CHANGE_LABEL,
        )

    return {
        "generated": bool(generated),
        "output_dir": str(figures_dir.resolve()),
        "combined_figure": str(combined_figure.resolve()) if generated and combined_figure.exists() else "",
        "per_metric_figures": per_metric_figures,
        "missing_or_empty": missing_or_empty,
        "legend": str(legend_path.resolve()) if legend_path is not None and legend_path.exists() else "",
    }


def _normalize_xpysom_default_runs_frame(
    df: pd.DataFrame,
    *,
    split_policy: str = "both",
) -> pd.DataFrame:
    dataset_col = _resolve_column(df, ("dataset",))
    seed_col = _resolve_column(df, ("seed", "config_seed", "random_seed"))
    topology_col = _resolve_column(df, ("comparison_topology", "architecture", "config_topology_type"))
    sampling_col = _resolve_column(df, ("sampling_method",))
    batch_mode_col = _resolve_column(df, ("batch_mode",))
    processing_col = _resolve_column(df, ("algorithm", "processing_type"))
    method_col = _resolve_column(df, ("method",))
    required_metric_cols = [
        col
        for col in ("quantization_error_train", "quantization_error_holdout", "balanced_qe_raw")
        if col in df.columns
    ]
    if dataset_col is None or seed_col is None or topology_col is None or sampling_col is None:
        raise ValueError(
            "XPySOM default runs CSV must include dataset, seed, topology, and sampling columns."
        )
    if len(required_metric_cols) < 3:
        raise ValueError(
            "XPySOM default runs CSV must include quantization_error_train, "
            "quantization_error_holdout, and balanced_qe_raw."
        )

    working = df.copy()
    if method_col is not None:
        method_values = working[method_col].astype(str).str.lower().str.strip()
        if (method_values == "xpysom").any():
            working = working[method_values == "xpysom"].copy()

    processing_values = (
        working[processing_col].astype(str).str.lower().str.strip()
        if processing_col is not None
        else pd.Series("batch", index=working.index, dtype=object)
    )
    batch_mode_values = (
        working[batch_mode_col].apply(_normalize_batch_mode)
        if batch_mode_col is not None
        else pd.Series("full_batch", index=working.index, dtype=object)
    )
    split_value = str(split_policy).lower().strip()
    out = pd.DataFrame(
        {
            "pair_dataset": working[dataset_col].astype(str).str.lower().str.strip(),
            "pair_processing": processing_values,
            "pair_sampling": working[sampling_col].astype(str).str.lower().str.strip(),
            "pair_batch_mode": batch_mode_values,
            "pair_topology": working[topology_col].astype(str).str.lower().str.strip(),
            "pair_seed": working[seed_col].astype(str).str.lower().str.strip(),
            "pair_split": split_value,
            "default_trial_number": 0.0,
            "quantization_error_train": pd.to_numeric(
                working["quantization_error_train"], errors="coerce"
            ),
            "quantization_error_holdout": pd.to_numeric(
                working["quantization_error_holdout"], errors="coerce"
            ),
            "balanced_qe_raw": pd.to_numeric(working["balanced_qe_raw"], errors="coerce"),
        }
    )
    if "train_time_s" in working.columns:
        out["train_time_s"] = pd.to_numeric(working["train_time_s"], errors="coerce")
    out = out.dropna(
        subset=[
            "pair_seed",
            "quantization_error_train",
            "quantization_error_holdout",
            "balanced_qe_raw",
        ]
    ).copy()
    out["pair_processing"] = out["pair_processing"].astype(str).str.lower().str.strip()
    out["pair_batch_mode"] = out["pair_batch_mode"].apply(_normalize_batch_mode)
    out["dataset"] = out["pair_dataset"]
    if out.empty:
        raise ValueError("XPySOM default runs data produced no usable rows after normalization.")

    key_cols = [
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
        "pair_batch_mode",
        "pair_topology",
        "pair_seed",
        "pair_split",
    ]
    dup_counts = out.groupby(key_cols, dropna=False).size().reset_index(name="count")
    duplicates = dup_counts[dup_counts["count"] > 1]
    if not duplicates.empty:
        raise ValueError(
            "XPySOM default runs data contains duplicate rows for matched keys. "
            f"Example keys: {duplicates.head(5).to_dict(orient='records')}"
        )
    return out.reset_index(drop=True)


def _load_xpysom_default_runs_csv(path: str | Path) -> pd.DataFrame:
    input_path = Path(path).expanduser().resolve()
    csv_path = input_path
    if input_path.is_dir():
        candidate = input_path / "xpysom_batch_topology_sweep_runs.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                "XPySOM benchmark directory did not contain the expected combined runs CSV: "
                f"{candidate}"
            )
        csv_path = candidate
    if not csv_path.exists():
        raise FileNotFoundError(f"XPySOM default runs CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"XPySOM default runs CSV is empty: {csv_path}")
    return _normalize_xpysom_default_runs_frame(df)


def _load_explicit_xpysom_topology_benchmark_csv(
    path: str | Path,
    *,
    expected_topology: str,
) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Explicit XPySOM topology benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Explicit XPySOM topology benchmark CSV is empty: {csv_path}")

    required_cols = {
        "dataset",
        "seed",
        "method",
        "train_time_s",
        "quantization_error_train",
        "quantization_error_holdout",
        "balanced_qe_raw",
        "sampling_method",
        "algorithm",
        "batch_mode",
        "architecture",
        "comparison_topology",
        "train_test_split",
    }
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(
            f"Explicit XPySOM topology benchmark CSV is missing required columns {missing_cols}: {csv_path}"
        )

    topology_key = str(expected_topology).strip().lower()
    working = df.copy()
    working["method"] = working["method"].astype(str).str.lower().str.strip()
    working["dataset"] = working["dataset"].astype(str).str.strip()
    working["architecture"] = working["architecture"].astype(str).str.lower().str.strip()
    working["comparison_topology"] = working["comparison_topology"].astype(str).str.lower().str.strip()
    working["sampling_method"] = working["sampling_method"].astype(str).str.lower().str.strip()
    working["algorithm"] = working["algorithm"].astype(str).str.lower().str.strip()
    working["batch_mode"] = working["batch_mode"].apply(_normalize_batch_mode).astype(str).str.lower().str.strip()
    working["seed"] = pd.to_numeric(working["seed"], errors="coerce")
    if bool(working["seed"].isna().any()):
        raise ValueError(f"Explicit XPySOM topology benchmark CSV has non-numeric seed values: {csv_path}")
    working["seed"] = working["seed"].astype(int)

    working = working[
        (working["comparison_topology"] == topology_key)
        & (working["sampling_method"] == "full")
        & (working["algorithm"] == "batch")
        & (working["batch_mode"] == "full_batch")
    ].copy()
    if working.empty:
        raise ValueError(
            "Explicit XPySOM topology benchmark CSV produced no "
            f"{topology_key}/full/batch/full_batch rows: {csv_path}"
        )

    methods = {value for value in working["method"].tolist() if value}
    required_methods = {"floatsom", "xpysom"}
    if not required_methods.issubset(methods):
        raise ValueError(
            "Explicit XPySOM topology benchmark CSV must contain both floatsom and xpysom rows "
            f"for topology '{topology_key}'. Found methods={sorted(methods)} in {csv_path}"
        )

    dup_counts = working.groupby(["dataset", "seed", "method"], dropna=False).size()
    if bool((dup_counts > 1).any()):
        raise ValueError(
            "Explicit XPySOM topology benchmark CSV contains duplicate dataset/seed/method rows "
            f"for topology '{topology_key}': {csv_path}"
        )

    numeric_cols = [
        "training_epochs",
        "train_time_s",
        "quantization_error_train",
        "quantization_error_holdout",
        "balanced_qe_raw",
        "train_test_split",
    ]
    for column in numeric_cols:
        if column not in working.columns:
            continue
        working[column] = pd.to_numeric(working[column], errors="coerce")

    return working.sort_values(["dataset", "seed", "method"]).reset_index(drop=True)


def _render_xpysom_default_calibration_publication_figures(
    *,
    topology_csvs: Dict[str, str | Path | None],
    output_dir: Path,
    generated_files: List[str],
    manuscript_only: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = ("mst", "rng", "hexagonal")
    figures: Dict[str, str] = {}
    tables: Dict[str, str] = {}
    stats_tables: Dict[str, str] = {}
    missing_or_empty: Dict[str, str] = {}
    output_dirs: Dict[str, str] = {}
    runtime_sources: Dict[str, str] = {}
    resolved_csvs: Dict[str, str] = {}

    for topology in topologies:
        csv_path_value = topology_csvs.get(topology)
        if not csv_path_value:
            missing_or_empty[topology] = f"No explicit XPySOM benchmark CSV provided for topology '{topology}'."
            continue
        try:
            combined_runs_df = _load_explicit_xpysom_topology_benchmark_csv(
                csv_path_value,
                expected_topology=topology,
            )
        except Exception as exc:
            missing_or_empty[topology] = str(exc)
            continue

        topology_output_dir = output_dir / topology
        topology_output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[topology] = str(topology_output_dir.resolve())
        resolved_csvs[topology] = str(Path(csv_path_value).expanduser().resolve())

        runs_csv_path = topology_output_dir / f"xpysom_{topology}_default_calibration_runs.csv"
        combined_runs_df.to_csv(runs_csv_path, index=False)
        if not manuscript_only:
            generated_files.append(str(runs_csv_path.resolve()))

        summary_df = _xpysom_publication_build_summary_table(combined_runs_df)
        if summary_df.empty:
            missing_or_empty[topology] = (
                f"Combined FloatSOM-default/XPySOM rows produced no summary table for topology '{topology}'."
            )
            continue
        summary_df = summary_df[
            [
                "dataset_index",
                "dataset",
                "metric",
                "split",
                "wins_floatsom",
                "wins_xpysom",
                "ties",
                "win_rate_floatsom",
                "median_delta_raw",
                "median_pct_improvement",
                "mean_pct_improvement",
                "ci_low_pct",
                "ci_high_pct",
                "p_value",
                "n_pairs",
                "notes",
                "effect_size_signed",
            ]
        ]
        summary_tsv_path = topology_output_dir / _xpysom_publication_table_filename_for_topology(topology)
        summary_df.to_csv(summary_tsv_path, sep="\t", index=False)
        if not manuscript_only:
            generated_files.append(str(summary_tsv_path.resolve()))

        figure_path = topology_output_dir / _xpysom_publication_figure_filename_for_topology(topology)
        stats_path = topology_output_dir / _xpysom_publication_forest_stats_filename_for_topology(topology)
        _xpysom_publication_generate_calibration_figure(
            runs_df=combined_runs_df,
            topology=topology,
            output_path=figure_path,
            stats_output_path=stats_path,
        )
        generated_files.append(str(figure_path.resolve()))
        if stats_path.exists():
            generated_files.append(str(stats_path.resolve()))

        figures[topology] = str(figure_path.resolve())
        tables[topology] = str(summary_tsv_path.resolve())
        stats_tables[topology] = str(stats_path.resolve()) if stats_path.exists() else ""
        runtime_sources[topology] = "paired_csv" if bool(combined_runs_df["train_time_s"].notna().any()) else "missing"

    return {
        "enabled": True,
        "output_dir": str(output_dir.resolve()),
        "topology_csvs": resolved_csvs,
        "topologies": list(topologies),
        "figures": figures,
        "tables": tables,
        "stats_tables": stats_tables,
        "topology_output_dirs": output_dirs,
        "runtime_sources": runtime_sources,
        "missing_or_empty": missing_or_empty,
    }


def _normalize_xpysom_topology_tripanel_runs_frame(
    df: pd.DataFrame,
    *,
    expected_topology: str,
) -> pd.DataFrame:
    working = df.copy()
    topology_key = str(expected_topology).strip().lower()
    working["comparison_topology"] = (
        working["comparison_topology"].astype(str).str.lower().str.strip()
    )
    working["architecture"] = working["comparison_topology"]
    working["evaluation_split"] = "both"
    working = working[working["comparison_topology"] == topology_key].copy()
    if working.empty:
        raise ValueError(
            "Explicit XPySOM topology benchmark CSV produced no rows for tripanel rendering "
            f"after filtering topology '{topology_key}'."
        )
    return _canonicalize_matched_parameter_runs(working)


def _xpysom_tripanel_figure_filename_for_topology(topology: str) -> str:
    return f"supp_fig_xpysom_default_vs_tuned_{str(topology)}_metrics.svg"


def _render_xpysom_topology_tripanel_publication_figures(
    *,
    topology_csvs: Dict[str, str | Path | None],
    output_dir: Path,
    generated_files: List[str],
    dpi: int,
    alpha: float,
    manuscript_only: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = ("hexagonal", "mst")
    figures: Dict[str, str] = {}
    output_dirs: Dict[str, str] = {}
    resolved_csvs: Dict[str, str] = {}
    missing_or_empty: Dict[str, str] = {}
    per_topology_outputs: Dict[str, Dict[str, object]] = {}

    metric_specs: List[Tuple[str, str, str]] = [
        ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ("quantization_error_holdout", "Holdout QE", "qe_holdout"),
        ("quantization_error_train", "Train QE", "qe_train"),
    ]
    topology_label_map = {
        "hexagonal": "Hexagonal",
        "mst": "MST",
    }

    for topology in topologies:
        csv_path_value = topology_csvs.get(topology)
        if not csv_path_value:
            missing_or_empty[topology] = (
                f"No explicit XPySOM tripanel CSV provided for topology '{topology}'."
            )
            continue
        try:
            combined_runs_df = _load_explicit_xpysom_topology_benchmark_csv(
                csv_path_value,
                expected_topology=topology,
            )
            normalized_runs_df = _normalize_xpysom_topology_tripanel_runs_frame(
                combined_runs_df,
                expected_topology=topology,
            )
        except Exception as exc:
            missing_or_empty[topology] = str(exc)
            continue

        topology_output_dir = output_dir / topology
        topology_output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[topology] = str(topology_output_dir.resolve())
        resolved_csvs[topology] = str(Path(csv_path_value).expanduser().resolve())

        runs_csv_path = topology_output_dir / f"xpysom_{topology}_tripanel_runs.csv"
        normalized_runs_df.to_csv(runs_csv_path, index=False)
        if not manuscript_only:
            generated_files.append(str(runs_csv_path.resolve()))

        topology_label = topology_label_map[topology]
        topology_figure_label_map = {
            "hexagonal": "Supplementary Figure S13",
            "mst": "Supplementary Figure S14",
        }
        comparison_key = f"xpysom_default_vs_tuned_{topology}"
        comparison_output = _render_matched_parameter_tripanel_figure(
            output_dir=topology_output_dir,
            generated_files=generated_files,
            metric_specs=metric_specs,
            figure_title=(
                f"{topology_figure_label_map[topology]}: XPySOM (Default) v FloatSOM {topology_label}"
            ),
            panel_title=f"XPySOM (Default) v Tuned FloatSOM {topology_label}",
            x_axis_label=(
                f"QE % Change (positive means tuned FloatSOM {topology_label} is better)"
            ),
            subtitle=(
                "Paired effect with 95% paired t-test CI. "
                f"Positive means tuned FloatSOM {topology_label} outperforms default XPySOM; "
                "global pools all matched dataset/seed pairs."
            ),
            comparison_key=comparison_key,
            df_a=normalized_runs_df,
            df_b=normalized_runs_df,
            selectors_a={"pair_topology": topology, "method": "floatsom"},
            selectors_b={"pair_topology": topology, "method": "xpysom"},
            label_a=f"tuned_floatsom_{topology}",
            label_b="xpysom_default",
            dpi=int(dpi),
            alpha=float(alpha),
        )
        if not bool(comparison_output.get("generated")):
            missing_or_empty[topology] = (
                "Tripanel figure generation produced no combined figure for topology "
                f"'{topology}'."
            )
            continue

        combined_figure_text = str(comparison_output.get("combined_figure", "")).strip()
        if not combined_figure_text:
            missing_or_empty[topology] = (
                f"Tripanel figure output for topology '{topology}' is missing a combined figure path."
            )
            continue

        combined_figure_path = Path(combined_figure_text).resolve()
        if not combined_figure_path.exists():
            missing_or_empty[topology] = (
                f"Tripanel figure output for topology '{topology}' was not written: {combined_figure_path}"
            )
            continue

        figure_alias_path = topology_output_dir / _xpysom_tripanel_figure_filename_for_topology(topology)
        shutil.copy2(combined_figure_path, figure_alias_path)
        generated_files.append(str(figure_alias_path.resolve()))

        figures[topology] = str(figure_alias_path.resolve())
        per_topology_outputs[topology] = comparison_output

    return {
        "enabled": True,
        "output_dir": str(output_dir.resolve()),
        "topology_csvs": resolved_csvs,
        "topologies": list(topologies),
        "figures": figures,
        "topology_output_dirs": output_dirs,
        "per_topology_outputs": per_topology_outputs,
        "missing_or_empty": missing_or_empty,
    }


def _normalize_matched_rng_full_runs_frame(df: pd.DataFrame) -> pd.DataFrame:
    topology_col = _resolve_column(df, ("pair_topology", "architecture", "config_topology_type"))
    sampling_col = _resolve_column(df, ("pair_sampling", "sampling_method", "config_sampling_method"))
    dataset_col = _resolve_column(df, ("dataset", "pair_dataset"))
    seed_col = _resolve_column(df, ("seed", "config_seed", "random_seed", "pair_seed"))
    if topology_col is None or sampling_col is None or dataset_col is None or seed_col is None:
        raise ValueError(
            "Matched RNG runs data must include dataset, seed, topology, and sampling columns."
        )

    working = df.copy()
    topology_values = working[topology_col].astype(str).str.lower().str.strip()
    sampling_values = working[sampling_col].astype(str).str.lower().str.strip()
    working = working[(topology_values == "rng") & (sampling_values == "full")].copy()
    if working.empty:
        raise ValueError(
            "Matched RNG runs data produced no full-sampling RNG rows for publication panels A-C."
        )

    if "balanced_qe_raw" not in working.columns:
        if {"quantization_error_holdout", "quantization_error_train"}.issubset(working.columns):
            qe_holdout = pd.to_numeric(working["quantization_error_holdout"], errors="coerce")
            qe_train = pd.to_numeric(working["quantization_error_train"], errors="coerce")
            working["balanced_qe_raw"] = (qe_holdout + qe_train) / 2.0
        else:
            raise ValueError(
                "Matched RNG runs CSV must include balanced_qe_raw or both train/holdout QE columns."
            )

    out = pd.DataFrame(
        {
            "dataset": working[dataset_col].astype(str).str.strip(),
            "seed": pd.to_numeric(working[seed_col], errors="coerce"),
            "method": "floatsom",
            "quantization_error_train": pd.to_numeric(
                working["quantization_error_train"], errors="coerce"
            ),
            "quantization_error_holdout": pd.to_numeric(
                working["quantization_error_holdout"], errors="coerce"
            ),
            "balanced_qe_raw": pd.to_numeric(working["balanced_qe_raw"], errors="coerce"),
        }
    )
    out = out.dropna(
        subset=[
            "seed",
            "quantization_error_train",
            "quantization_error_holdout",
            "balanced_qe_raw",
        ]
    ).copy()
    out["seed"] = out["seed"].astype(int)
    if out.empty:
        raise ValueError("Matched RNG runs data produced no usable rows after normalization.")
    return out


def _load_matched_rng_full_runs_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Matched RNG runs CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Matched RNG runs CSV is empty: {csv_path}")
    return df


def _prepare_xpysom_publication_default_rows(df: pd.DataFrame) -> pd.DataFrame:
    selectors = {
        "pair_topology": "rng",
        "pair_sampling": "full",
        "pair_processing": "batch",
        "pair_batch_mode": "full_batch",
        "pair_split": "both",
    }
    filtered = _filter_matched_parameter_rows(df, selectors)
    if filtered.empty:
        raise ValueError(
            "XPySOM default runs data produced no RNG/full/batch/full_batch rows for Figure 11 panels A-C."
        )
    return filtered.copy()


def _prepare_xpysom_publication_rng_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    working = _canonicalize_matched_parameter_runs(df)
    selectors = {
        "pair_topology": "rng",
        "pair_sampling": "full",
        "pair_processing": "batch",
        "pair_batch_mode": "full_batch",
        "pair_split": "both",
    }
    filtered = _filter_matched_parameter_rows(working, selectors)
    if filtered.empty:
        raise ValueError(
            "Matched RNG benchmark data produced no RNG/full/batch/full_batch rows for Figure 11 panels A-C."
        )
    required_metric_cols = [
        metric_col
        for metric_col in ("balanced_qe_raw", "quantization_error_holdout", "quantization_error_train")
        if metric_col not in filtered.columns
    ]
    if required_metric_cols:
        raise ValueError(
            "Matched RNG benchmark data is missing required QE columns for Figure 11 panels A-C: "
            f"{required_metric_cols}"
        )
    return filtered.copy()


def _load_scaling_results_by_topology(path: str | Path) -> Dict[str, Dict[str, Dict[Any, Any]]]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Scaling CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Scaling CSV is empty: {csv_path}")

    key_cols = {"topology", "method", "sample_size", "gpu_count"}
    missing_key_cols = sorted(key_cols.difference(df.columns))
    if missing_key_cols:
        raise ValueError(
            f"Scaling CSV missing required columns {missing_key_cols}. Expected columns to include {sorted(key_cols)}"
        )

    has_raw_runtime = "train_time_s" in df.columns
    has_aggregated_runtime = {
        "train_time_mean_s",
        "train_time_std_s",
        "n_runs",
    }.issubset(df.columns)
    if not has_raw_runtime and not has_aggregated_runtime:
        raise ValueError(
            "Scaling CSV must contain either raw runtime column 'train_time_s' or aggregated runtime columns "
            "['train_time_mean_s', 'train_time_std_s', 'n_runs']."
        )

    results_by_topology: Dict[str, Dict[str, Dict[Any, Any]]] = {}
    grouped = df.copy().assign(
        topology=lambda frame: frame["topology"].astype(str).str.lower().str.strip(),
        method=lambda frame: frame["method"].astype(str).str.lower().str.strip(),
        sample_size=lambda frame: pd.to_numeric(frame["sample_size"], errors="coerce"),
        gpu_count=lambda frame: pd.to_numeric(frame["gpu_count"], errors="coerce"),
    )

    if has_raw_runtime:
        grouped = grouped.assign(
            train_time_s=lambda frame: pd.to_numeric(frame["train_time_s"], errors="coerce"),
        )
        if "status" in grouped.columns:
            grouped = grouped[
                grouped["status"].astype(str).str.strip().str.lower().eq("success")
            ].copy()
        grouped = grouped.dropna(subset=["sample_size", "gpu_count", "train_time_s"])
    else:
        grouped = grouped.assign(
            train_time_mean_s=lambda frame: pd.to_numeric(frame["train_time_mean_s"], errors="coerce"),
            train_time_std_s=lambda frame: pd.to_numeric(frame["train_time_std_s"], errors="coerce"),
            n_runs=lambda frame: pd.to_numeric(frame["n_runs"], errors="coerce"),
        ).dropna(subset=["sample_size", "gpu_count", "train_time_mean_s"])
    if grouped.empty:
        raise ValueError("Scaling CSV produced no usable rows after normalization.")

    for (topology, method, sample_size, gpu_count), subset in grouped.groupby(
        ["topology", "method", "sample_size", "gpu_count"], dropna=False
    ):
        topology_key = str(topology)
        method_key = str(method)
        sample_key = int(sample_size)
        gpu_key = int(gpu_count)
        if has_raw_runtime:
            times = subset["train_time_s"].to_numpy(dtype=float)
            stats_payload = {
                "times": [float(value) for value in times.tolist()],
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "count": int(times.size),
            }
        else:
            row = subset.iloc[0]
            stats_payload = {
                "times": [],
                "mean": float(row["train_time_mean_s"]),
                "std": float(row["train_time_std_s"]) if pd.notna(row["train_time_std_s"]) else 0.0,
                "count": int(row["n_runs"]) if pd.notna(row["n_runs"]) else 1,
            }
        results_by_topology.setdefault(topology_key, {}).setdefault(method_key, {}).setdefault(sample_key, {})[
            gpu_key
        ] = stats_payload
    return results_by_topology


def _exclude_scaling_sample_sizes(
    results_by_topology: Dict[str, Dict[str, Dict[Any, Any]]],
    *,
    excluded_sample_sizes: Set[int],
) -> Dict[str, Dict[str, Dict[Any, Any]]]:
    if not excluded_sample_sizes:
        return results_by_topology

    filtered_results: Dict[str, Dict[str, Dict[Any, Any]]] = {}
    for topology, method_map in results_by_topology.items():
        filtered_method_map: Dict[str, Dict[Any, Any]] = {}
        for method, sample_map in method_map.items():
            filtered_sample_map = {
                sample_size: gpu_payload
                for sample_size, gpu_payload in sample_map.items()
                if int(sample_size) not in excluded_sample_sizes
            }
            if filtered_sample_map:
                filtered_method_map[method] = filtered_sample_map
        if filtered_method_map:
            filtered_results[topology] = filtered_method_map
    return filtered_results


def _xpysom_publication_wilcoxon_two_sided_pvalue(values: np.ndarray) -> float:
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


def _xpysom_publication_paired_pct_improvements(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    baseline = np.abs(np.asarray(values_b, dtype=float))
    pct = np.where(
        baseline > 0.0,
        ((np.asarray(values_b, dtype=float) - np.asarray(values_a, dtype=float)) / baseline) * 100.0,
        np.nan,
    )
    pct = np.asarray(pct, dtype=float)
    return pct[np.isfinite(pct)]


def _xpysom_publication_ordered_dataset_labels(dataset_values: Sequence[str]) -> List[str]:
    names = [str(value) for value in dataset_values if pd.notna(value)]
    if not names:
        return []
    unique_names = set(names)
    ordered = [name for name in CANONICAL_DATASET_ORDER if name in unique_names]
    extras = sorted(name for name in unique_names if name not in CANONICAL_DATASET_ORDER and name != "GLOBAL")
    ordered.extend(extras)
    if "GLOBAL" in unique_names:
        ordered.append("GLOBAL")
    return ordered


def _xpysom_publication_pvalue_to_stars(value: float) -> str:
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


def _xpysom_publication_row_is_significant(row: pd.Series, alpha: float) -> bool:
    p_raw = row.get("p_value", np.nan)
    if p_raw is None or pd.isna(p_raw):
        return False
    try:
        p_value = float(p_raw)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(p_value):
        return False
    return p_value < alpha


@lru_cache(maxsize=128)
def _xpysom_publication_wilcoxon_rank_sum_cdf(n_nonzero: int) -> np.ndarray:
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
def _xpysom_publication_wilcoxon_two_sided_cutoff(n_nonzero: int, alpha: float) -> int:
    if n_nonzero <= 0 or alpha <= 0.0:
        return -1
    max_rank_sum = int(n_nonzero * (n_nonzero + 1) // 2)
    half_rank_sum = int(max_rank_sum // 2)
    tail_alpha = float(alpha) / 2.0
    strict_threshold = np.nextafter(tail_alpha, float("-inf"))
    cdf = _xpysom_publication_wilcoxon_rank_sum_cdf(int(n_nonzero))
    accepted = np.flatnonzero(cdf[: half_rank_sum + 1] <= strict_threshold)
    if accepted.size == 0:
        return -1
    return int(accepted[-1])


def _xpysom_publication_walsh_averages(nonzero: np.ndarray) -> np.ndarray:
    pairwise = (nonzero[:, None] + nonzero[None, :]) / 2.0
    return pairwise[np.triu_indices(nonzero.size)].astype(float, copy=False)


def _xpysom_publication_wilcoxon_location_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
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

    walsh = _xpysom_publication_walsh_averages(nonzero)
    count = int(walsh.size)
    cutoff = _xpysom_publication_wilcoxon_two_sided_cutoff(int(nonzero.size), float(alpha))
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


def _xpysom_publication_wilcoxon_location_shift(values: np.ndarray) -> float:
    return float(_xpysom_publication_wilcoxon_location_ci(values)[0])


def _xpysom_publication_paired_pct_improvement_summary_generic(
    runs_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    win_label_a: str,
    win_label_b: str,
    win_rate_label_a: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pooled_pct: List[float] = []
    pooled_delta_raw: List[float] = []
    value_a_str = str(value_a)
    value_b_str = str(value_b)
    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(index="seed", columns=compare_col, values=metric_col, aggfunc="first")
        if pivot.empty:
            continue
        pivot.columns = pivot.columns.astype(str)
        required_cols = {value_a_str, value_b_str}
        if not required_cols.issubset(set(pivot.columns.tolist())):
            continue

        paired = pivot[[value_a_str, value_b_str]].dropna()
        if paired.empty:
            continue

        values_a = paired[value_a_str].to_numpy(dtype=float)
        values_b = paired[value_b_str].to_numpy(dtype=float)
        deltas = values_a - values_b
        pct = _xpysom_publication_paired_pct_improvements(values_a, values_b)
        if pct.size == 0:
            continue

        pooled_pct.extend(float(value) for value in pct.tolist())
        pooled_delta_raw.extend(float(value) for value in deltas[np.isfinite(deltas)].tolist())
        location_pct, ci_low_pct, ci_high_pct = _xpysom_publication_wilcoxon_location_ci(pct)
        location_raw = _xpysom_publication_wilcoxon_location_shift(deltas)
        wins_a = int(np.sum(pct > 0.0))
        wins_b = int(np.sum(pct < 0.0))
        ties = int(len(pct) - wins_a - wins_b)
        win_rate_a = float(wins_a) / float(len(pct)) if len(pct) > 0 else float("nan")
        denom = wins_a + wins_b
        effect_size_signed = float(wins_a - wins_b) / float(denom) if denom > 0 else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "metric": metric_label,
                "n_pairs": int(pct.size),
                win_label_a: wins_a,
                win_label_b: wins_b,
                "ties": ties,
                win_rate_label_a: win_rate_a,
                "median_delta_raw": float(location_raw),
                "median_pct": float(location_pct),
                "mean_pct": float(np.nanmean(pct)),
                "ci_low_pct": float(ci_low_pct),
                "ci_high_pct": float(ci_high_pct),
                "p_value": _xpysom_publication_wilcoxon_two_sided_pvalue(pct),
                "effect_size_signed": effect_size_signed,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "metric",
                "n_pairs",
                win_label_a,
                win_label_b,
                "ties",
                win_rate_label_a,
                "median_delta_raw",
                "median_pct",
                "mean_pct",
                "ci_low_pct",
                "ci_high_pct",
                "p_value",
                "effect_size_signed",
            ]
        )

    summary_df = pd.DataFrame(rows)
    pooled_pct_array = np.asarray(pooled_pct, dtype=float)
    pooled_pct_array = pooled_pct_array[np.isfinite(pooled_pct_array)]
    pooled_delta_array = np.asarray(pooled_delta_raw, dtype=float)
    pooled_delta_array = pooled_delta_array[np.isfinite(pooled_delta_array)]
    if pooled_pct_array.size > 0:
        location_pct, ci_low_pct, ci_high_pct = _xpysom_publication_wilcoxon_location_ci(pooled_pct_array)
        wins_a = int(np.sum(pooled_pct_array > 0.0))
        wins_b = int(np.sum(pooled_pct_array < 0.0))
        ties = int(pooled_pct_array.size - wins_a - wins_b)
        win_rate_a = float(wins_a) / float(pooled_pct_array.size)
        denom = wins_a + wins_b
        effect_size_signed = float(wins_a - wins_b) / float(denom) if denom > 0 else float("nan")
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "dataset": "GLOBAL",
                            "metric": metric_label,
                            "n_pairs": int(pooled_pct_array.size),
                            win_label_a: wins_a,
                            win_label_b: wins_b,
                            "ties": ties,
                            win_rate_label_a: win_rate_a,
                            "median_delta_raw": (
                                float(_xpysom_publication_wilcoxon_location_shift(pooled_delta_array))
                                if pooled_delta_array.size > 0
                                else np.nan
                            ),
                            "median_pct": float(location_pct),
                            "mean_pct": float(np.mean(pooled_pct_array)),
                            "ci_low_pct": float(ci_low_pct),
                            "ci_high_pct": float(ci_high_pct),
                            "p_value": _xpysom_publication_wilcoxon_two_sided_pvalue(pooled_pct_array),
                            "effect_size_signed": effect_size_signed,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return summary_df


def _xpysom_publication_plot_dataset_forest_axis(
    ax: object,
    summary_df: pd.DataFrame,
    panel_title: str,
    alpha: float = 0.05,
    x_label: str | None = None,
    show_y_tick_labels: bool = True,
    show_x_label: bool = False,
) -> None:
    if summary_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = summary_df.copy()
    order = _xpysom_publication_ordered_dataset_labels(plot_df["dataset"].astype(str).tolist())
    order_map = {name: idx for idx, name in enumerate(order)}
    plot_df["plot_order"] = plot_df["dataset"].map(order_map)
    plot_df = plot_df.sort_values("plot_order", ascending=True).reset_index(drop=True)
    plot_df["y"] = np.arange(len(plot_df))[::-1]
    plot_df["is_global"] = plot_df["dataset"].astype(str) == "GLOBAL"
    plot_df["is_significant"] = plot_df.apply(
        lambda row: _xpysom_publication_row_is_significant(row, alpha=alpha),
        axis=1,
    )
    plot_df["sig_color"] = np.where(plot_df["is_significant"], "#d7301f", "#7a7a7a")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    for _, row in plot_df.iterrows():
        ax.hlines(
            y=float(row["y"]),
            xmin=float(row["ci_low_pct"]),
            xmax=float(row["ci_high_pct"]),
            color=str(row["sig_color"]),
            linewidth=2.0,
            alpha=0.9,
        )

    nonglobal = plot_df[~plot_df["is_global"]]
    if not nonglobal.empty:
        ax.scatter(
            nonglobal["median_pct"].to_numpy(dtype=float),
            nonglobal["y"].to_numpy(dtype=float),
            c=nonglobal["sig_color"].tolist(),
            s=85.0,
            marker="o",
            zorder=3,
        )

    global_rows = plot_df[plot_df["is_global"]]
    if not global_rows.empty:
        ax.scatter(
            global_rows["median_pct"].to_numpy(dtype=float),
            global_rows["y"].to_numpy(dtype=float),
            c=global_rows["sig_color"].tolist(),
            s=95.0,
            marker="D",
            zorder=4,
        )

    for _, row in plot_df.iterrows():
        if not bool(row["is_significant"]):
            continue
        stars = _xpysom_publication_pvalue_to_stars(float(row["p_value"]))
        if stars == "ns":
            continue
        ax.annotate(
            stars,
            (float(row["median_pct"]), float(row["y"])),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=14.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=str(row["sig_color"]),
            clip_on=False,
        )

    x_bounds = plot_df[["ci_low_pct", "ci_high_pct", "median_pct"]].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_bounds))
    x_max = float(np.nanmax(x_bounds))
    span = max(1.0, x_max - x_min)
    ax.set_xlim(min(-0.5, x_min - 0.08 * span), max(0.5, x_max + 0.08 * span))
    ax.set_yticks(plot_df["y"].to_numpy(dtype=float))
    if show_y_tick_labels:
        ax.set_yticklabels(plot_df["dataset"].astype(str).tolist())
    else:
        ax.set_yticklabels([""] * len(plot_df))
    ax.tick_params(axis="x", labelsize=16.0)
    ax.tick_params(axis="y", labelsize=15.0, length=0)
    if show_x_label:
        ax.set_xlabel(
            x_label
            if x_label is not None
            else "Median % Improvement vs XPySOM (positive means FloatSOM is better)",
            fontsize=15.0,
        )
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.25)


def _render_xpysom_style_forest_panel(
    *,
    summary_df: pd.DataFrame,
    output_path: Path,
    show_y_tick_labels: bool,
    x_axis_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    _xpysom_publication_plot_dataset_forest_axis(
        ax,
        summary_df,
        panel_title="",
        alpha=0.05,
        x_label=x_axis_label,
        show_y_tick_labels=show_y_tick_labels,
        show_x_label=True,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _render_placeholder_panel(
    *,
    output_path: Path,
    panel_title: str,
    detail_lines: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f7f7")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(1.2)

    ax.text(
        0.5,
        0.62,
        panel_title,
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.45,
        "\n".join(str(line) for line in detail_lines),
        ha="center",
        va="center",
        fontsize=17,
        color="#444444",
        linespacing=1.35,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _xpysom_publication_resolve_dataset_sample_sizes(dataset_values: Sequence[str]) -> np.ndarray:
    sample_sizes: List[float] = []
    for dataset in dataset_values:
        dataset_key = str(dataset).strip().lower()
        sample_size = DATASET_SAMPLE_SIZE_FALLBACK.get(dataset_key)
        sample_sizes.append(float(sample_size) if sample_size is not None else np.nan)
    return np.asarray(sample_sizes, dtype=float)


def _xpysom_publication_dataset_index_lookup(dataset_values: Sequence[str]) -> Dict[str, int]:
    dataset_names = sorted(
        {
            str(value).strip()
            for value in dataset_values
            if pd.notna(value) and str(value).strip() and str(value).strip().upper() != "GLOBAL"
        }
    )
    if not dataset_names:
        return {}

    metadata = pd.DataFrame({"dataset": dataset_names})
    metadata["sample_size"] = _xpysom_publication_resolve_dataset_sample_sizes(metadata["dataset"].tolist())
    metadata = metadata.sort_values(["sample_size", "dataset"], ascending=[True, True], na_position="last")
    metadata = metadata.reset_index(drop=True)
    metadata["dataset_index"] = np.arange(1, len(metadata) + 1, dtype=int)
    return {
        str(row["dataset"]): int(row["dataset_index"])
        for _, row in metadata.iterrows()
    }


def _xpysom_publication_build_runtime_delta_dataset_summary(
    runs_df: pd.DataFrame,
    *,
    compare_col: str,
    value_a: str,
    value_b: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    value_a_str = str(value_a)
    value_b_str = str(value_b)
    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(index="seed", columns=compare_col, values="train_time_s", aggfunc="first")
        if pivot.empty:
            continue
        pivot.columns = pivot.columns.astype(str)
        if not {value_a_str, value_b_str}.issubset(set(pivot.columns.tolist())):
            continue

        paired = pivot[[value_a_str, value_b_str]].dropna()
        if paired.empty:
            continue

        deltas = paired[value_a_str].to_numpy(dtype=float) - paired[value_b_str].to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue

        rows.append(
            {
                "dataset": str(dataset),
                "n_pairs": int(deltas.size),
                "sample_size": float(_xpysom_publication_resolve_dataset_sample_sizes([dataset])[0]),
                "median_delta_s": float(np.median(deltas)),
                "mean_delta_s": float(np.mean(deltas)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["dataset", "n_pairs", "sample_size", "median_delta_s", "mean_delta_s"])

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_size", "median_delta_s"])
    if summary_df.empty:
        return summary_df
    summary_df = summary_df.sort_values("sample_size", ascending=True).reset_index(drop=True)
    dataset_index_lookup = _xpysom_publication_dataset_index_lookup(summary_df["dataset"].tolist())
    summary_df["dataset_index"] = summary_df["dataset"].map(dataset_index_lookup).astype("Int64")
    return summary_df[
        ["dataset_index", "dataset", "n_pairs", "sample_size", "median_delta_s", "mean_delta_s"]
    ].copy()


def _xpysom_publication_plot_runtime_delta_vs_dataset_size_axis(
    ax: object,
    runtime_df: pd.DataFrame,
    *,
    panel_title: str,
) -> None:
    if runtime_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = runtime_df.copy()
    plot_df["sample_size"] = pd.to_numeric(plot_df["sample_size"], errors="coerce")
    plot_df["median_delta_s"] = pd.to_numeric(plot_df["median_delta_s"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_size", "median_delta_s"])
    if plot_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = plot_df.sort_values("sample_size", ascending=True).reset_index(drop=True)
    delta_values = plot_df["median_delta_s"].to_numpy(dtype=float)
    point_colors = np.where(delta_values <= 0.0, "#1b9e77", "#d95f02")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.scatter(
        plot_df["sample_size"].to_numpy(dtype=float),
        delta_values,
        c=point_colors.tolist(),
        s=88.0,
        edgecolors="white",
        linewidths=0.9,
        zorder=2,
    )
    for _, row in plot_df.iterrows():
        dataset_index = row.get("dataset_index", pd.NA)
        if pd.isna(dataset_index):
            continue
        ax.annotate(
            str(int(dataset_index)),
            (float(row["sample_size"]), float(row["median_delta_s"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#1f1f1f",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=3,
            clip_on=False,
        )

    max_abs_delta = float(np.nanmax(np.abs(delta_values))) if len(delta_values) else 1.0
    linthresh = max(1e-3, min(0.25, max_abs_delta * 0.12))
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
    ax.set_xscale("log", base=10)

    x_values = plot_df["sample_size"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    ax.set_xlim(x_min * 0.92, x_max * 1.08)
    ax.set_xlabel("Dataset size (samples)", fontsize=15.0)
    ax.set_ylabel("Runtime delta (FloatSOM - baseline, s)", fontsize=15.0)
    ax.tick_params(axis="x", labelsize=14.0)
    ax.tick_params(axis="y", labelsize=14.0)
    ax.grid(True, axis="both", alpha=0.22)
    ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold", pad=12)


def _xpysom_publication_generate_pairwise_calibration_figure(
    runs_df: pd.DataFrame,
    *,
    output_path: Path,
    stats_output_path: Path,
    figure_label: str,
    figure_caption: str,
) -> pd.DataFrame:
    panel_specs = [
        ("balanced_qe_raw", "A  Balanced QE"),
        ("quantization_error_holdout", "B  Holdout QE"),
        ("quantization_error_train", "C  Train QE"),
    ]

    panel_stats: List[pd.DataFrame] = []
    for metric_col, panel_title in panel_specs:
        panel_df = _xpysom_publication_paired_pct_improvement_summary_generic(
            runs_df,
            metric_col=metric_col,
            metric_label=panel_title,
            compare_col="method",
            value_a="floatsom",
            value_b="xpysom",
            win_label_a="wins_floatsom",
            win_label_b="wins_xpysom",
            win_rate_label_a="win_rate_floatsom",
        )
        panel_stats.append(panel_df)

    runtime_df = _xpysom_publication_build_runtime_delta_dataset_summary(
        runs_df,
        compare_col="method",
        value_a="floatsom",
        value_b="xpysom",
    )

    all_stats = pd.concat(panel_stats, ignore_index=True) if panel_stats else pd.DataFrame()
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(stats_output_path, index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(22.0, 16.0),
        gridspec_kw={"hspace": 0.27, "wspace": 0.14},
    )
    flat_axes = axes.flatten()
    for idx, (_, panel_title) in enumerate(panel_specs):
        panel_df = all_stats[all_stats["metric"] == panel_title].copy()
        _xpysom_publication_plot_dataset_forest_axis(
            flat_axes[idx],
            panel_df,
            panel_title=panel_title,
            alpha=0.05,
            x_label="Median % Improvement vs XPySOM (positive means FloatSOM is better)",
            show_y_tick_labels=idx in (0, 2),
            show_x_label=False,
        )
    _xpysom_publication_plot_runtime_delta_vs_dataset_size_axis(
        flat_axes[3],
        runtime_df,
        panel_title="D  Runtime Delta vs Dataset Size",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7301f", markeredgecolor="#d7301f", markersize=12, label="Significant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7a7a7a", markeredgecolor="#7a7a7a", markersize=12, label="Non-significant"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#7a7a7a", markeredgecolor="#7a7a7a", markersize=12, label="GLOBAL"),
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="95% CI"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero effect"),
    ]
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.992))

    top_left = axes[0, 0].get_position()
    top_right = axes[0, 1].get_position()
    title_y = min(0.999, float(top_left.y1) + 0.032)
    fig.text(
        0.02,
        title_y,
        f"{figure_label}: {figure_caption}",
        ha="left",
        va="bottom",
        fontsize=24,
        fontweight="bold",
    )
    top_x_center = (top_left.x0 + top_right.x1) / 2.0
    row_xlabel_offset = 0.030
    fig.text(
        top_x_center,
        top_left.y0 - row_xlabel_offset,
        "Median % Improvement vs XPySOM (positive means FloatSOM is better)",
        ha="center",
        va="top",
        fontsize=16,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.001),
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return all_stats


def _xpysom_publication_generate_calibration_figure(
    runs_df: pd.DataFrame,
    *,
    topology: str,
    output_path: Path,
    stats_output_path: Path,
) -> pd.DataFrame:
    topology_key = str(topology).strip().lower()
    topology_label_map = {
        "mst": "MST",
        "rng": "RNG",
        "hexagonal": "Hexagonal",
    }
    figure_label_map = {
        "mst": "Supplementary Figure S1",
        "rng": "Supplementary Figure S2",
        "hexagonal": "Supplementary Figure S3",
    }
    topology_label = topology_label_map.get(topology_key, str(topology).replace("_", " ").title())
    figure_label = figure_label_map.get(topology_key, "Supplementary Figure")
    return _xpysom_publication_generate_pairwise_calibration_figure(
        runs_df,
        output_path=output_path,
        stats_output_path=stats_output_path,
        figure_label=figure_label,
        figure_caption=f"FloatSOM vs XPySOM Calibration (Default Settings, {topology_label})",
    )


def _xpysom_publication_paired_metric_rows(
    runs_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_name: str,
    split_name: str,
    strict_pairing: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pooled_pct: List[float] = []
    pooled_delta_raw: List[float] = []

    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(index="seed", columns="method", values=metric_col, aggfunc="first")
        required_cols = {"floatsom", "xpysom"}
        if not required_cols.issubset(set(pivot.columns.astype(str).tolist())):
            if strict_pairing:
                raise ValueError(
                    f"Missing paired methods for dataset '{dataset}' metric '{metric_col}'. "
                    f"Expected methods: {sorted(required_cols)}. Found: {list(pivot.columns)}"
                )
            continue
        paired = pivot[["floatsom", "xpysom"]].dropna()
        if paired.empty:
            continue

        floatsom_values = paired["floatsom"].to_numpy(dtype=float)
        xpysom_values = paired["xpysom"].to_numpy(dtype=float)
        deltas = floatsom_values - xpysom_values
        pct = _xpysom_publication_paired_pct_improvements(floatsom_values, xpysom_values)
        if pct.size == 0:
            continue
        pooled_pct.extend(float(value) for value in pct.tolist())
        pooled_delta_raw.extend(float(value) for value in deltas[np.isfinite(deltas)].tolist())
        location_pct, ci_low_pct, ci_high_pct = _xpysom_publication_wilcoxon_location_ci(pct)
        location_raw = _xpysom_publication_wilcoxon_location_shift(deltas)

        wins_floatsom = int(np.sum(pct > 0.0))
        wins_xpysom = int(np.sum(pct < 0.0))
        ties = int(len(pct) - wins_floatsom - wins_xpysom)
        win_rate_floatsom = float(wins_floatsom) / float(len(pct))
        denom = wins_floatsom + wins_xpysom
        effect_size_signed = float(wins_floatsom - wins_xpysom) / float(denom) if denom > 0 else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "metric": metric_name,
                "split": split_name,
                "wins_floatsom": wins_floatsom,
                "wins_xpysom": wins_xpysom,
                "ties": ties,
                "win_rate_floatsom": win_rate_floatsom,
                "median_delta_raw": float(location_raw),
                "median_pct_improvement": float(location_pct),
                "mean_pct_improvement": float(np.mean(pct)),
                "ci_low_pct": float(ci_low_pct),
                "ci_high_pct": float(ci_high_pct),
                "p_value": _xpysom_publication_wilcoxon_two_sided_pvalue(pct),
                "n_pairs": int(len(pct)),
                "effect_size_signed": effect_size_signed,
                "notes": "positive means FloatSOM is better",
            }
        )

    if rows:
        pooled_values = np.asarray(pooled_pct, dtype=float)
        pooled_values = pooled_values[np.isfinite(pooled_values)]
        pooled_raw_values = np.asarray(pooled_delta_raw, dtype=float)
        pooled_raw_values = pooled_raw_values[np.isfinite(pooled_raw_values)]
        macro_location, macro_ci_low, macro_ci_high = _xpysom_publication_wilcoxon_location_ci(pooled_values)
        wins_floatsom = int(np.sum(pooled_values > 0.0))
        wins_xpysom = int(np.sum(pooled_values < 0.0))
        ties = int(len(pooled_values) - wins_floatsom - wins_xpysom)
        win_rate_floatsom = (
            float(wins_floatsom) / float(len(pooled_values)) if len(pooled_values) > 0 else float("nan")
        )
        denom = wins_floatsom + wins_xpysom
        effect_size_signed = float(wins_floatsom - wins_xpysom) / float(denom) if denom > 0 else float("nan")
        rows.append(
            {
                "dataset": "GLOBAL",
                "metric": metric_name,
                "split": split_name,
                "wins_floatsom": wins_floatsom,
                "wins_xpysom": wins_xpysom,
                "ties": ties,
                "win_rate_floatsom": win_rate_floatsom,
                "median_delta_raw": (
                    float(_xpysom_publication_wilcoxon_location_shift(pooled_raw_values))
                    if pooled_raw_values.size > 0
                    else np.nan
                ),
                "median_pct_improvement": float(macro_location) if pooled_values.size > 0 else np.nan,
                "mean_pct_improvement": float(np.mean(pooled_values)) if pooled_values.size > 0 else np.nan,
                "ci_low_pct": float(macro_ci_low) if pooled_values.size > 0 else np.nan,
                "ci_high_pct": float(macro_ci_high) if pooled_values.size > 0 else np.nan,
                "p_value": _xpysom_publication_wilcoxon_two_sided_pvalue(pooled_values),
                "n_pairs": int(len(pooled_values)),
                "effect_size_signed": effect_size_signed,
                "notes": "pooled paired samples across datasets",
            }
        )

    return rows


def _xpysom_publication_build_summary_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("quantization_error_train", "quantization_error", "train", True),
        ("quantization_error_holdout", "quantization_error", "holdout", True),
        ("balanced_qe_raw", "balanced_qe", "both", True),
        ("train_time_s", "train_time", "train", False),
    ]
    rows: List[Dict[str, object]] = []
    for metric_col, metric_name, split_name, strict_pairing in metric_specs:
        rows.extend(
            _xpysom_publication_paired_metric_rows(
                runs_df,
                metric_col=metric_col,
                metric_name=metric_name,
                split_name=split_name,
                strict_pairing=strict_pairing,
            )
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    dataset_index_lookup = _xpysom_publication_dataset_index_lookup(summary_df["dataset"].tolist())
    summary_df["dataset_index"] = summary_df["dataset"].map(dataset_index_lookup).astype("Int64")
    return summary_df


def _xpysom_publication_table_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return "supp_xpysom_calibration_qe_hexagonal.tsv"
    return f"supp_xpysom_calibration_qe_{str(topology)}.tsv"


def _xpysom_publication_figure_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return "supp_fig_xpysom_calibration_qe_hexagonal.svg"
    return f"supp_fig_xpysom_calibration_qe_{str(topology)}.svg"


def _xpysom_publication_forest_stats_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return "xpysom_calibration_forest_stats.csv"
    return f"xpysom_calibration_forest_stats_{str(topology)}.csv"


def _render_xpysom_rng_scaling_publication_figure(
    *,
    xpysom_default_runs_file: str | Path,
    matched_rng_runs_file: str | Path,
    scaling_csv_file: str | Path | None,
    output_dir: Path,
    generated_files: List[str],
    dpi: int,
    manuscript_only: bool = False,
) -> Dict[str, object]:
    from floatsom.benchmarks.visualization import plot_topology_comparison_scaling_benchmark

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    xpysom_df = _prepare_xpysom_publication_default_rows(
        _load_xpysom_default_runs_csv(xpysom_default_runs_file)
    )
    rng_full_df = _prepare_xpysom_publication_rng_benchmark_rows(
        _load_matched_rng_full_runs_csv(matched_rng_runs_file)
    )
    ac_panel_dir = output_dir / "ac_pairing"
    ac_panel_dir.mkdir(parents=True, exist_ok=True)
    ac_tables_dir = ac_panel_dir / "tables"
    ac_tables_dir.mkdir(parents=True, exist_ok=True)
    metric_configs = [
        ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw", True, "panel_a_balanced_qe.svg"),
        (
            "quantization_error_holdout",
            "Holdout QE",
            "quantization_error_holdout",
            False,
            "panel_b_holdout_qe.svg",
        ),
        (
            "quantization_error_train",
            "Train QE",
            "quantization_error_train",
            True,
            "panel_c_train_qe.svg",
        ),
    ]
    pairing_report_path = ac_panel_dir / "XPYSOM_VS_FLOATSOM_RNG_KEYED_PAIRS.md"
    pairing_report_path.write_text(
        "\n".join(
            [
                "# XPySOM Default vs FloatSOM RNG Keyed Pairing",
                "",
                "Panels A-C compare the two benchmark CSVs directly by shared canonical run keys.",
                "",
                "Join keys:",
                "- pair_dataset",
                "- pair_processing",
                "- pair_sampling",
                "- pair_batch_mode",
                "- pair_seed",
                "- pair_split",
                "",
                f"- XPySOM benchmark CSV: {Path(xpysom_default_runs_file).expanduser().resolve()}",
                f"- FloatSOM RNG benchmark CSV: {Path(matched_rng_runs_file).expanduser().resolve()}",
                "",
                "No Optuna trial-number pairing or best-trial selection is used in this figure.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not manuscript_only:
        generated_files.append(str(pairing_report_path.resolve()))
    panel_paths: List[Tuple[str, str, Path]] = []
    summary_stats: List[pd.DataFrame] = []
    for panel_letter, (metric_col, panel_label, metric_slug, show_y_labels, filename) in zip(
        ["A", "B", "C"], metric_configs
    ):
        pairs_df = _build_matched_parameter_pairs(
            df_a=rng_full_df,
            df_b=xpysom_df,
            metric=metric_col,
            selectors_a={},
            selectors_b={},
            label_a="floatsom_rng",
            label_b="xpysom_default",
        )
        if pairs_df.empty:
            raise ValueError(
                f"XPySOM-vs-RNG publication pairing produced no matched keyed rows for metric '{metric_slug}'."
            )
        dataset_summary_df = summarize_paired_value_table(
            pairs_df,
            "pair_dataset",
            higher_is_better=False,
        ).rename(columns={"pair_dataset": "dataset"})
        global_summary_df = summarize_paired_value_global_row(
            pairs_df,
            "global_label",
            "GLOBAL",
            higher_is_better=False,
            value_a_col="tuned_value",
            value_b_col="default_value",
        )
        summary_df = _merge_tuned_vs_default_global_row(dataset_summary_df, global_summary_df)
        plot_df = _prepare_tuned_vs_default_dataset_summary_for_forest(summary_df)
        if plot_df.empty:
            raise ValueError(
                f"XPySOM-vs-RNG publication pairing produced no plottable rows for metric '{metric_slug}'."
            )
        pairs_path = ac_tables_dir / f"xpysom_rng_keyed_pairs_{metric_slug}.csv"
        dataset_summary_path = ac_tables_dir / f"xpysom_rng_dataset_summary_{metric_slug}.csv"
        global_summary_path = ac_tables_dir / f"xpysom_rng_global_summary_{metric_slug}.csv"
        pairs_df.to_csv(pairs_path, index=False)
        dataset_summary_df.to_csv(dataset_summary_path, index=False)
        global_summary_df.to_csv(global_summary_path, index=False)
        if not manuscript_only:
            generated_files.append(str(pairs_path.resolve()))
            generated_files.append(str(dataset_summary_path.resolve()))
            generated_files.append(str(global_summary_path.resolve()))
        summary_df = plot_df.copy()
        summary_df["panel"] = panel_letter
        summary_df["metric_col"] = metric_col
        summary_stats.append(summary_df)
        panel_path = figures_dir / filename
        panel_stats_path = None if manuscript_only else tables_dir / f"xpysom_rng_panel_stats_{metric_slug}.csv"
        _plot_dataset_forest_publication(
            summary_df=plot_df,
            title="",
            subtitle="",
            output_path=panel_path,
            dpi=int(dpi),
            alpha=0.05,
            inline_legends=False,
            stats_table_output_path=panel_stats_path,
            x_axis_label="QE change (%)",
            show_y_tick_labels=bool(show_y_labels),
            **_tripanel_forest_panel_style(show_dataset_labels=bool(show_y_labels)),
        )
        generated_files.append(str(panel_path.resolve()))
        if panel_stats_path is not None and panel_stats_path.exists():
            generated_files.append(str(panel_stats_path.resolve()))
        panel_paths.append((panel_letter, panel_label, panel_path))

    summary_stats_path = tables_dir / "xpysom_rng_full_summary_stats.csv"
    if not manuscript_only:
        pd.concat(summary_stats, axis=0, ignore_index=True).to_csv(summary_stats_path, index=False)
        generated_files.append(str(summary_stats_path.resolve()))

    scaling_panel_dir = output_dir / "scaling_panel"
    scaling_panel_dir.mkdir(parents=True, exist_ok=True)
    scaling_panel_path = scaling_panel_dir / "sample_scaling_performance.svg"
    scaling_placeholder = False
    if scaling_csv_file is not None and str(scaling_csv_file).strip():
        scaling_results = _load_scaling_results_by_topology(scaling_csv_file)
        if not scaling_results:
            raise ValueError("Scaling CSV produced no plottable rows for the Figure 11 scaling panel.")
        plot_topology_comparison_scaling_benchmark(
            results_by_topology=scaling_results,
            output_dir=str(scaling_panel_dir),
            title="",
            axis_mode="sample",
            comparison_gpu_count=4,
            show_speedup=False,
            show_error_bars=True,
            show_legend=True,
            methods_to_plot=["default", "full", "random"],
            emit_shared_legend=False,
            include_performance=True,
            topology_color_overrides={
                "xpysom": "#1f77b4",
                "floatsom": "#9467bd",
            },
            method_linestyle_overrides={
                "default": "-",
                "full": "-",
                "random": ":",
            },
            sample_right_pad_ratio=0.08,
            sample_min_upper_bound=1.1e9,
            bottom_legend_fontsize_scale=1.08,
            bottom_legend_markerscale=1.05,
        )
        if not scaling_panel_path.exists():
            raise FileNotFoundError(
                f"Scaling panel was not generated from scaling CSV: {scaling_panel_path}"
            )
    else:
        scaling_placeholder = True
        _render_placeholder_panel(
            output_path=scaling_panel_path,
            panel_title="Scaling Runtime Placeholder",
            detail_lines=[
                "Panel D requires the dedicated XPySOM vs FloatSOM RNG scaling CSV.",
                "Panels A-C are derived from the shared XPySOM benchmark CSV",
                "and the matched FloatSOM RNG benchmark CSV available today.",
            ],
        )
    generated_files.append(str(scaling_panel_path.resolve()))

    combined_figure_path = figures_dir / "fig_xpysom_rng_default_vs_scaling.svg"
    combined_generated = _compose_panel_matrix_from_images(
        panel_rows=[
            [
                panel_paths[0],
                panel_paths[1],
            ],
            [
                panel_paths[2],
                ("D", "Scaling Runtime", scaling_panel_path),
            ],
        ],
        title="Figure 11: XPySOM (Default) v FloatSOM RNG",
        output_path=combined_figure_path,
        dpi=int(dpi),
    )
    if combined_generated and combined_figure_path.exists():
        generated_files.append(str(combined_figure_path.resolve()))

    metadata = {
        "enabled": True,
        "output_dir": str(output_dir.resolve()),
        "xpysom_default_runs_file": str(Path(xpysom_default_runs_file).expanduser().resolve()),
        "matched_rng_runs_file": str(Path(matched_rng_runs_file).expanduser().resolve()),
        "scaling_csv_file": (
            str(Path(scaling_csv_file).expanduser().resolve())
            if scaling_csv_file is not None and str(scaling_csv_file).strip()
            else ""
        ),
        "combined_figure": (
            str(combined_figure_path.resolve())
            if combined_generated and combined_figure_path.exists()
            else ""
        ),
        "summary_stats_csv": (
            str(summary_stats_path.resolve()) if not manuscript_only else ""
        ),
        "scaling_panel": str(scaling_panel_path.resolve()),
        "scaling_placeholder": bool(scaling_placeholder),
        "excluded_sample_sizes": [],
    }
    if not manuscript_only:
        metadata_path = output_dir / "XPYSOM_RNG_PUBLICATION_METADATA.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        generated_files.append(str(metadata_path.resolve()))
        metadata["metadata_json"] = str(metadata_path.resolve())
    else:
        metadata["metadata_json"] = ""
    return metadata

def _render_sampling_comparison_publication_figure(
    *,
    sampling_comparison_dir: Path,
    publication_dir: Path,
    metric_specs: Sequence[Tuple[str, str, str]],
    comparison_key: str,
    comparison_label: str,
    comparison_scope: str,
    generated_files: List[str],
    dpi: int,
    alpha: float,
    legend_path: Optional[Path],
    secondary_legend_path: Optional[Path],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str,
) -> Dict[str, object]:
    tables_dir = sampling_comparison_dir / "tables"
    include_regression_row = comparison_key == "full_vs_random"
    title_prefix = "Figure 2" if comparison_key == "full_vs_random" else "Supplementary Figure S4"
    top_panel_inputs: List[PublicationOutcomePanelInput] = []
    top_panel_specs: List[Tuple[str, str, Path]] = []
    regression_panel_specs: List[Tuple[str, str, Path]] = []
    per_metric_figures: Dict[str, str] = {}
    per_metric_regressions: Dict[str, str] = {}
    regression_table_paths: List[Path] = []
    missing_or_empty: Dict[str, str] = {}
    top_row_dataset_labels: List[str] = []
    letters = [chr(ord("A") + idx) for idx in range(26)]

    for idx, (_, metric_label, metric_slug) in enumerate(metric_specs):
        if comparison_scope == "all_sampling_modes":
            summary_name = f"sampling_mode_outcome_hexagonal_full_batch_{comparison_key}_{metric_slug}.csv"
        else:
            summary_name = (
                f"sampling_mode_outcome_hexagonal_full_batch_{comparison_scope}_{comparison_key}_{metric_slug}.csv"
            )
        summary_path = tables_dir / summary_name
        if not summary_path.exists():
            missing_or_empty[metric_slug] = f"Missing sampling summary table: {summary_path}"
            continue

        summary_df = pd.read_csv(summary_path)
        plot_df = _prepare_sampling_dataset_summary_for_forest(summary_df)
        if plot_df.empty:
            missing_or_empty[metric_slug] = "Sampling summary exists but has no plottable rows."
            continue

        if comparison_scope == "all_sampling_modes":
            figure_name = f"figure_3_panel_{comparison_key}_{metric_slug}.svg"
        else:
            figure_name = f"figure_3_panel_{comparison_scope}_{comparison_key}_{metric_slug}.svg"
        figure_path = publication_dir / figure_name
        top_panel_inputs.append(
            PublicationOutcomePanelInput(
                panel_letter=letters[idx],
                metric_label=metric_label,
                metric_slug=metric_slug,
                plot_df=plot_df,
                output_path=figure_path,
                title=f"{metric_label}",
                subtitle=comparison_label,
            )
        )

        if include_regression_row:
            from .workflow_analysis_generation import _infer_sampling_regression_figure_from_diagnostics

            regression_path = _infer_sampling_regression_figure_from_diagnostics(
                diagnostics=diagnostics,
                diagnostics_prefix=diagnostics_prefix,
                metric_slug=metric_slug,
            )
            if regression_path is None or not regression_path.exists():
                missing_or_empty[f"{metric_slug}_regression"] = (
                    "Missing regression panel output for this metric."
                )
                continue
            letter_idx = idx + len(metric_specs)
            if letter_idx >= len(letters):
                letter_idx = len(letters) - 1
            regression_panel_specs.append((letters[letter_idx], metric_label, regression_path))
            per_metric_regressions[metric_slug] = str(regression_path.resolve())
            regression_table_path = _infer_sampling_regression_table_from_diagnostics(
                diagnostics=diagnostics,
                diagnostics_prefix=diagnostics_prefix,
                metric_slug=metric_slug,
            )
            if regression_table_path is not None and regression_table_path.exists():
                regression_table_paths.append(regression_table_path)

    render_result = render_publication_outcome_panels(
        panel_inputs=top_panel_inputs,
        dpi=int(dpi),
        alpha=float(alpha),
        generated_files=generated_files,
    )
    top_panel_specs = list(render_result["panel_specs"])
    per_metric_figures = dict(render_result["per_metric_figures"])
    top_row_dataset_labels = list(render_result["top_row_dataset_labels"])
    missing_or_empty.update(render_result["missing_or_empty"])

    if comparison_scope == "all_sampling_modes":
        combined_name = "figure_3_algorithm_sampling_stratified_all_metrics.svg"
    else:
        combined_name = f"figure_3_algorithm_sampling_stratified_{comparison_scope}_all_metrics.svg"
    combined_figure = publication_dir / combined_name
    dataset_metadata_table = (
        _write_figure_3_dataset_metadata_table(
            regression_table_paths=regression_table_paths,
            output_path=tables_dir / _figure_3_dataset_metadata_table_name(comparison_scope),
        )
        if include_regression_row
        else None
    )
    if dataset_metadata_table is not None and dataset_metadata_table.exists():
        generated_files.append(str(dataset_metadata_table.resolve()))

    generated = False
    if include_regression_row:
        if (
            top_panel_specs
            and len(top_panel_specs) == len(metric_specs)
            and len(regression_panel_specs) == len(metric_specs)
        ):
            generated = _compose_six_panel_stats_figure(
                panel_rows=[top_panel_specs, regression_panel_specs],
                title=(
                    f"{title_prefix}: {comparison_label} Outcomes and Dataset-Size Regression "
                    "(Hexagonal / Full Batch)"
                ),
                output_path=combined_figure,
                dpi=int(dpi),
                legend_path=legend_path,
                legend_after_row=1,
                legend_band_height_override=PUBLICATION_LEGEND_ROW_HEIGHT,
                legend_gap_override=10.0,
                secondary_legend_path=secondary_legend_path,
                secondary_legend_band_height_override=PUBLICATION_LEGEND_ROW_HEIGHT,
                secondary_legend_gap_override=8.0,
                margin_bottom_override=132.0,
                direction_labels_by_row=[
                    _direction_banner_text(
                        left_label=comparison_label.split(" vs ", 1)[1],
                        right_label=comparison_label.split(" vs ", 1)[0],
                    ),
                    _direction_banner_text(
                        left_label=comparison_label.split(" vs ", 1)[1],
                        right_label=comparison_label.split(" vs ", 1)[0],
                    ),
                ],
                direction_label_positions_by_row=["top_center", "left_vertical"],
                shared_y_labels_by_row=["", "QE difference (%)"],
                shared_x_labels_by_row=[CANONICAL_PUBLICATION_QE_CHANGE_LABEL, "Sample size (log10)"],
                external_y_labels_by_row=[top_row_dataset_labels, None] if top_row_dataset_labels else None,
                left_annotation_gutter_extra_override=20.0,
            )
            if generated and combined_figure.exists():
                generated_files.append(str(combined_figure.resolve()))
        else:
            missing_or_empty["combined"] = (
                f"Expected {len(metric_specs)} top and {len(metric_specs)} regression panels for Figure 2, "
                f"generated {len(top_panel_specs)} and {len(regression_panel_specs)}."
            )
    else:
        if top_panel_specs and len(top_panel_specs) == len(metric_specs):
            generated = compose_publication_outcome_tripanel(
                panel_specs=top_panel_specs,
                figure_title=f"{title_prefix}: {comparison_label} Outcomes (Hexagonal / Full Batch)",
                output_path=combined_figure,
                dpi=int(dpi),
                legend_path=legend_path,
                direction_label=_direction_banner_text(
                    left_label=comparison_label.split(" vs ", 1)[1],
                    right_label=comparison_label.split(" vs ", 1)[0],
                ),
                generated_files=generated_files,
                external_y_labels=top_row_dataset_labels,
            )
        else:
            missing_or_empty["combined"] = (
                f"Expected {len(metric_specs)} top panels for Figure 2, generated {len(top_panel_specs)}."
            )

    return {
        "generated": bool(generated),
        "combined_figure": str(combined_figure.resolve()) if generated and combined_figure.exists() else "",
        "per_metric_figures": per_metric_figures,
        "per_metric_regressions": per_metric_regressions,
        "dataset_metadata_table": (
            str(dataset_metadata_table.resolve())
            if dataset_metadata_table is not None and dataset_metadata_table.exists()
            else ""
        ),
        "missing_or_empty": missing_or_empty,
    }


def _canonicalize_matched_parameter_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    canonical_cols = {
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
        "pair_batch_mode",
        "pair_topology",
        "pair_seed",
        "pair_split",
    }
    if not canonical_cols.issubset(set(working.columns)):
        working = _prepare_dataframe(working)
        if "dataset" not in working.columns:
            raise ValueError("Matched-parameter dataframe is missing a dataset column.")
        if "architecture" not in working.columns:
            raise ValueError("Matched-parameter dataframe is missing an architecture/topology column.")
        working["pair_dataset"] = working["dataset"].astype(str).str.lower().str.strip()
        if "algorithm" in working.columns:
            working["pair_processing"] = working["algorithm"].astype(str).str.lower().str.strip()
        else:
            working["pair_processing"] = "batch"
        working["pair_topology"] = working["architecture"].astype(str).str.lower().str.strip()
        working["pair_sampling"] = working["pair_sampling"].astype(str).str.lower().str.strip()
        working["pair_seed"] = working["pair_seed"].astype(str).str.lower().str.strip()
        working["pair_split"] = working["pair_split"].astype(str).str.lower().str.strip()
        if "pair_batch_mode" in working.columns:
            working["pair_batch_mode"] = (
                working["pair_batch_mode"].apply(_normalize_batch_mode).astype(str).str.lower().str.strip()
            )
        else:
            working["pair_batch_mode"] = "full_batch"
    else:
        working["pair_dataset"] = working["pair_dataset"].astype(str).str.lower().str.strip()
        working["pair_processing"] = working["pair_processing"].astype(str).str.lower().str.strip()
        working["pair_sampling"] = working["pair_sampling"].astype(str).str.lower().str.strip()
        working["pair_batch_mode"] = working["pair_batch_mode"].apply(_normalize_batch_mode)
        working["pair_topology"] = working["pair_topology"].astype(str).str.lower().str.strip()
        working["pair_seed"] = working["pair_seed"].astype(str).str.lower().str.strip()
        working["pair_split"] = working["pair_split"].astype(str).str.lower().str.strip()

    if "dataset" not in working.columns:
        working["dataset"] = working["pair_dataset"]
    else:
        working["dataset"] = working["dataset"].astype(str).str.strip()
    return working


def _filter_matched_parameter_rows(
    df: pd.DataFrame,
    selectors: Dict[str, str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    for column, value in selectors.items():
        if column not in working.columns:
            raise ValueError(f"Matched-parameter selector column '{column}' is missing from dataframe.")
        target = str(value).strip().lower()
        series = working[column].astype(str).str.lower().str.strip()
        working = working[series == target].copy()
        if working.empty:
            return working
    return working


def _build_matched_parameter_pairs(
    *,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    selectors_a: Dict[str, str],
    selectors_b: Dict[str, str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    left = _filter_matched_parameter_rows(df_a, selectors_a)
    right = _filter_matched_parameter_rows(df_b, selectors_b)
    if left.empty or right.empty:
        return pd.DataFrame()

    if metric not in left.columns or metric not in right.columns:
        raise ValueError(
            f"Matched-parameter comparison metric '{metric}' is missing from one or both inputs."
        )

    join_keys = [
        "pair_dataset",
        "pair_processing",
        "pair_sampling",
        "pair_batch_mode",
        "pair_seed",
        "pair_split",
    ]
    missing_left = [column for column in join_keys if column not in left.columns]
    missing_right = [column for column in join_keys if column not in right.columns]
    if missing_left or missing_right:
        raise ValueError(
            "Matched-parameter comparison is missing join keys. "
            f"left_missing={missing_left}, right_missing={missing_right}"
        )

    left_dup_counts = left.groupby(join_keys, dropna=False).size()
    right_dup_counts = right.groupby(join_keys, dropna=False).size()
    if bool((left_dup_counts > 1).any()) or bool((right_dup_counts > 1).any()):
        raise ValueError(
            "Matched-parameter comparison found duplicate rows for the pairing keys. "
            f"comparison_a={label_a}, comparison_b={label_b}"
        )

    left_pairs = left[join_keys + ["dataset", "pair_topology", metric]].rename(
        columns={
            "dataset": "dataset_a",
            "pair_topology": "pair_topology_a",
            metric: "tuned_value",
        }
    )
    right_pairs = right[join_keys + ["dataset", "pair_topology", metric]].rename(
        columns={
            "dataset": "dataset_b",
            "pair_topology": "pair_topology_b",
            metric: "default_value",
        }
    )
    merged = left_pairs.merge(right_pairs, on=join_keys, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["dataset"] = (
        merged["dataset_a"]
        .where(merged["dataset_a"].notna(), merged["dataset_b"])
        .astype(str)
        .str.strip()
    )
    merged["comparison_a"] = str(label_a)
    merged["comparison_b"] = str(label_b)
    merged["delta_tuned_minus_default"] = merged["tuned_value"] - merged["default_value"]
    denom = merged["default_value"].abs()
    merged["pct_improvement"] = np.where(
        denom > 0,
        (merged["default_value"] - merged["tuned_value"]) / denom * 100.0,
        np.nan,
    )
    return merged


def _render_matched_parameter_tripanel_figure(
    *,
    output_dir: Path,
    generated_files: List[str],
    metric_specs: Sequence[Tuple[str, str, str]],
    figure_title: str,
    panel_title: str,
    x_axis_label: str,
    subtitle: str,
    comparison_key: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    selectors_a: Dict[str, str],
    selectors_b: Dict[str, str],
    label_a: str,
    label_b: str,
    dpi: int,
    alpha: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel_specs: List[Tuple[str, str, Path]] = []
    per_metric_figures: Dict[str, str] = {}
    table_outputs: Dict[str, Dict[str, str]] = {}
    missing_or_empty: Dict[str, str] = {}
    top_row_dataset_labels: List[str] = []
    letters = [chr(ord("A") + idx) for idx in range(26)]

    for idx, (metric, metric_label, metric_slug) in enumerate(metric_specs):
        pairs = _build_matched_parameter_pairs(
            df_a=df_a,
            df_b=df_b,
            metric=metric,
            selectors_a=selectors_a,
            selectors_b=selectors_b,
            label_a=label_a,
            label_b=label_b,
        )
        if pairs.empty:
            missing_or_empty[metric_slug] = "No matched parameter pairs available for this comparison."
            continue

        dataset_summary = summarize_paired_value_table(
            pairs,
            "pair_dataset",
            higher_is_better=False,
        ).rename(columns={"pair_dataset": "dataset"})
        global_summary = summarize_paired_value_global_row(
            pairs,
            "global_label",
            "GLOBAL",
            higher_is_better=False,
            value_a_col="tuned_value",
            value_b_col="default_value",
        )
        summary_df = _merge_tuned_vs_default_global_row(dataset_summary, global_summary)
        plot_df = _prepare_tuned_vs_default_dataset_summary_for_forest(summary_df)
        if plot_df.empty:
            missing_or_empty[metric_slug] = "Matched parameter summary exists but has no plottable rows."
            continue

        pairs_path = tables_dir / f"{comparison_key}_pairs_{metric_slug}.csv"
        dataset_summary_path = tables_dir / f"{comparison_key}_dataset_summary_{metric_slug}.csv"
        global_summary_path = tables_dir / f"{comparison_key}_global_summary_{metric_slug}.csv"
        stats_path = tables_dir / f"{comparison_key}_stats_{metric_slug}.csv"
        figure_path = figures_dir / f"fig_{comparison_key}_{metric_slug}.svg"

        pairs.to_csv(pairs_path, index=False)
        dataset_summary.to_csv(dataset_summary_path, index=False)
        global_summary.to_csv(global_summary_path, index=False)
        generated_files.append(str(pairs_path.resolve()))
        generated_files.append(str(dataset_summary_path.resolve()))
        generated_files.append(str(global_summary_path.resolve()))

        if not top_row_dataset_labels:
            top_row_dataset_labels = _ordered_dataset_labels(plot_df["dataset"].astype(str).tolist())
        _plot_dataset_forest_publication(
            summary_df=plot_df,
            title=f"{panel_title} ({metric_label})",
            subtitle=subtitle,
            output_path=figure_path,
            dpi=int(dpi),
            alpha=float(alpha),
            inline_legends=False,
            stats_table_output_path=stats_path,
            x_axis_label="",
            show_y_tick_labels=False,
            **_tripanel_forest_panel_style(show_dataset_labels=False),
        )
        if not figure_path.exists():
            missing_or_empty[metric_slug] = f"Figure was not generated for {metric_slug}."
            continue

        generated_files.append(str(figure_path.resolve()))
        if stats_path.exists():
            generated_files.append(str(stats_path.resolve()))

        per_metric_figures[metric_slug] = str(figure_path.resolve())
        table_outputs[metric_slug] = {
            "pairs": str(pairs_path.resolve()),
            "dataset_summary": str(dataset_summary_path.resolve()),
            "global_summary": str(global_summary_path.resolve()),
            "stats": str(stats_path.resolve()) if stats_path.exists() else "",
        }
        panel_specs.append((letters[idx], metric_label, figure_path))

    legend_path = _render_composite_legend_key(
        output_path=figures_dir / f"legend_{comparison_key}.svg",
        dpi=int(dpi),
        alpha=float(alpha),
        include_forest=True,
        include_sensitivity=False,
        compact=False,
    )
    if legend_path is not None and legend_path.exists():
        generated_files.append(str(legend_path.resolve()))

    combined_figure = figures_dir / f"fig_{comparison_key}_metrics.svg"
    generated = False
    if panel_specs:
        generated = _compose_three_panel_stats_figure(
            panel_rows=[panel_specs],
            title=figure_title,
            output_path=combined_figure,
            dpi=int(dpi),
            legend_path=legend_path if legend_path is not None and legend_path.exists() else None,
            legend_band_height_override=PUBLICATION_LEGEND_ROW_HEIGHT,
            margin_bottom_override=36.0,
            direction_labels_by_row=[
                _direction_banner_text(
                    left_label=_humanize_comparison_label(label_b),
                    right_label=_humanize_comparison_label(label_a),
                )
            ],
            shared_x_labels_by_row=[x_axis_label],
            external_y_labels_by_row=[top_row_dataset_labels] if top_row_dataset_labels else None,
        )
        if generated and combined_figure.exists():
            generated_files.append(str(combined_figure.resolve()))

    return {
        "generated": bool(generated),
        "output_dir": str(output_dir.resolve()),
        "combined_figure": str(combined_figure.resolve()) if generated and combined_figure.exists() else "",
        "per_metric_figures": per_metric_figures,
        "tables": table_outputs,
        "legend": str(legend_path.resolve()) if legend_path is not None and legend_path.exists() else "",
        "missing_or_empty": missing_or_empty,
        "comparison_a": str(label_a),
        "comparison_b": str(label_b),
        "selectors_a": dict(selectors_a),
        "selectors_b": dict(selectors_b),
    }


def _render_matched_parameter_publication_figures(
    *,
    tuned_df: pd.DataFrame,
    defaults_df: pd.DataFrame,
    output_dir: Path,
    generated_files: List[str],
    dpi: int,
    alpha: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_specs: List[Tuple[str, str, str]] = [
        ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ("quantization_error_holdout", "Holdout QE", "qe_holdout"),
        ("quantization_error_train", "Train QE", "qe_train"),
    ]
    tuned_runs = _canonicalize_matched_parameter_runs(tuned_df)
    default_runs = _canonicalize_matched_parameter_runs(defaults_df)

    comparison_specs: List[Dict[str, object]] = [
        {
            "key": "matched_tuned_vs_untuned_hex",
            "figure_title": "Tuned vs Untuned FloatSOM Hex Across QE Metrics",
            "panel_title": "Tuned vs Untuned FloatSOM Hex",
            "x_axis_label": "QE % Change (positive means tuned FloatSOM Hex is better)",
            "subtitle": (
                "Paired effect with 95% paired t-test CI. Positive means tuned FloatSOM Hex outperforms "
                "untuned FloatSOM Hex; global pools all matched dataset/sampling/seed pairs."
            ),
            "df_a": tuned_runs,
            "df_b": default_runs,
            "selectors_a": {"pair_topology": "hexagonal"},
            "selectors_b": {"pair_topology": "hexagonal"},
            "label_a": "tuned_hex",
            "label_b": "untuned_hex",
        },
        {
            "key": "matched_tuned_vs_untuned_rng",
            "figure_title": "Tuned vs Untuned FloatSOM RNG Across QE Metrics",
            "panel_title": "Tuned vs Untuned FloatSOM RNG",
            "x_axis_label": "QE % Change (positive means tuned FloatSOM RNG is better)",
            "subtitle": (
                "Paired effect with 95% paired t-test CI. Positive means tuned FloatSOM RNG outperforms "
                "untuned FloatSOM RNG; global pools all matched dataset/sampling/seed pairs."
            ),
            "df_a": tuned_runs,
            "df_b": default_runs,
            "selectors_a": {"pair_topology": "rng"},
            "selectors_b": {"pair_topology": "rng"},
            "label_a": "tuned_rng",
            "label_b": "untuned_rng",
        },
        {
            "key": "matched_tuned_hex_vs_tuned_rng",
            "figure_title": "Tuned FloatSOM Hex vs Tuned FloatSOM RNG Across QE Metrics",
            "panel_title": "Tuned FloatSOM Hex vs Tuned FloatSOM RNG",
            "x_axis_label": "QE % Change (positive means tuned FloatSOM Hex is better)",
            "subtitle": (
                "Paired effect with 95% paired t-test CI. Positive means tuned FloatSOM Hex outperforms "
                "tuned FloatSOM RNG; global pools all matched dataset/sampling/seed pairs."
            ),
            "df_a": tuned_runs,
            "df_b": tuned_runs,
            "selectors_a": {"pair_topology": "hexagonal"},
            "selectors_b": {"pair_topology": "rng"},
            "label_a": "tuned_hex",
            "label_b": "tuned_rng",
        },
        {
            "key": "matched_untuned_hex_vs_tuned_rng",
            "figure_title": "Untuned FloatSOM Hex vs Tuned FloatSOM RNG Across QE Metrics",
            "panel_title": "Untuned FloatSOM Hex vs Tuned FloatSOM RNG",
            "x_axis_label": "QE % Change (positive means untuned FloatSOM Hex is better)",
            "subtitle": (
                "Paired effect with 95% paired t-test CI. Positive means untuned FloatSOM Hex outperforms "
                "tuned FloatSOM RNG; global pools all matched dataset/sampling/seed pairs."
            ),
            "df_a": default_runs,
            "df_b": tuned_runs,
            "selectors_a": {"pair_topology": "hexagonal"},
            "selectors_b": {"pair_topology": "rng"},
            "label_a": "untuned_hex",
            "label_b": "tuned_rng",
        },
    ]

    outputs: Dict[str, Dict[str, object]] = {}
    for spec in comparison_specs:
        comparison_key = str(spec["key"])
        outputs[comparison_key] = _render_matched_parameter_tripanel_figure(
            output_dir=output_dir / comparison_key,
            generated_files=generated_files,
            metric_specs=metric_specs,
            figure_title=str(spec["figure_title"]),
            panel_title=str(spec["panel_title"]),
            x_axis_label=str(spec["x_axis_label"]),
            subtitle=str(spec["subtitle"]),
            comparison_key=comparison_key,
            df_a=spec["df_a"],
            df_b=spec["df_b"],
            selectors_a=dict(spec["selectors_a"]),
            selectors_b=dict(spec["selectors_b"]),
            label_a=str(spec["label_a"]),
            label_b=str(spec["label_b"]),
            dpi=int(dpi),
            alpha=float(alpha),
        )

    return {
        "generated": any(bool(value.get("generated")) for value in outputs.values()),
        "output_dir": str(output_dir.resolve()),
        "figures": outputs,
    }


def _build_default_aware_pairing_variants(
    *,
    selected_df: pd.DataFrame,
    selected_source: str,
    base_df: pd.DataFrame,
    base_source: str,
    include_selected_topology_variants: bool,
    include_base_variant: bool,
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = [
        {
            "key": "selected_source",
            "label": "Selected tuned source",
            "source": str(selected_source),
            "df": selected_df,
            "output_subdir": "tuned_vs_default",
            "markdown_filename": "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST.md",
            "figure_title": "Figure 6: Tuned Configuration vs Untuned Reference Across QE Metrics",
        }
    ]
    if include_selected_topology_variants and "pair_topology" in selected_df.columns:
        topology_series = selected_df["pair_topology"].astype(str).str.lower().str.strip()
        for topology_name, topology_slug, topology_label in (
            ("hexagonal", "hex", "Hexagonal"),
            ("mst", "mst", "MST"),
            ("rng", "rng", "RNG"),
        ):
            subset = selected_df[topology_series == topology_name].copy()
            if subset.empty:
                continue
            variants.append(
                {
                    "key": f"selected_source_{topology_slug}",
                    "label": f"Selected tuned source ({topology_label} only)",
                    "source": str(selected_source),
                    "df": subset,
                    "output_subdir": f"tuned_vs_default_{topology_slug}",
                    "markdown_filename": (
                        f"TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST_{topology_slug.upper()}.md"
                    ),
                    "figure_title": (
                        f"Supplementary Figure S{9 if topology_slug == 'hex' else 10 if topology_slug == 'mst' else 11}: "
                        f"Tuned Configuration vs Untuned Reference Across QE Metrics ({topology_label})"
                    ),
                }
            )
    if (
        include_base_variant
        and _normalize_source_path_for_compare(selected_source) != _normalize_source_path_for_compare(base_source)
    ):
        variants.append(
            {
                "key": "data_file_source",
                "label": "Data-file tuned source",
                "source": str(base_source),
                "df": base_df,
                "output_subdir": "tuned_vs_default_data_file",
                "markdown_filename": "TUNED_VS_TRUE_DEFAULT_PAIRED_TTEST_DATA_FILE.md",
                "figure_title": "Figure 6: Tuned Configuration vs Untuned Reference Across QE Metrics",
            }
        )
    return variants

def _run_tuned_vs_default_variant(
    *,
    variant: Dict[str, object],
    defaults_df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str, str]],
    default_aware_dir: Path,
    generated_files: List[str],
    dpi: int,
    alpha: float,
) -> Dict[str, object]:
    pairing_df = variant["df"]
    if not isinstance(pairing_df, pd.DataFrame):
        raise ValueError("Default-aware tuned-vs-default variant is missing a dataframe payload.")
    output_subdir = str(variant.get("output_subdir", "")).strip()
    if not output_subdir:
        raise ValueError("Default-aware tuned-vs-default variant is missing output_subdir.")
    markdown_filename = str(variant.get("markdown_filename", "")).strip()
    if not markdown_filename:
        raise ValueError("Default-aware tuned-vs-default variant is missing markdown_filename.")

    missing_tuned = [metric for metric, _, _ in metric_specs if metric not in pairing_df.columns]
    if missing_tuned:
        raise ValueError(
            "Missing tuned metric columns required for default-aware analysis "
            f"(variant={variant.get('key')}): {missing_tuned}"
        )

    variant_dir = default_aware_dir / output_subdir
    tuned_vs_default_metrics = [
        TunedDefaultMetric(column=metric, label=label, slug=slug, higher_is_better=False)
        for metric, label, slug in metric_specs
    ]
    report_path = build_tuned_vs_external_default_report(
        tuned_df=pairing_df,
        default_df=defaults_df,
        metrics=tuned_vs_default_metrics,
        output_dir=variant_dir,
        markdown_filename=markdown_filename,
        split_policy="both",
    )
    generated_files.append(str(report_path.resolve()))
    publication = _render_tuned_vs_default_publication_figure(
        tuned_vs_default_dir=variant_dir,
        metric_specs=metric_specs,
        generated_files=generated_files,
        dpi=int(dpi),
        alpha=float(alpha),
        combined_figure_title=str(
            variant.get("figure_title", "Figure 6: Tuned Configuration vs Untuned Reference Across QE Metrics")
        ),
    )
    return {
        "key": str(variant.get("key", "")),
        "label": str(variant.get("label", "")),
        "source": str(variant.get("source", "")),
        "rows_total": int(len(pairing_df)),
        "output_dir": str(variant_dir.resolve()),
        "report": str(report_path.resolve()),
        "publication": publication,
    }

def _run_default_aware_analysis(
    *,
    tuned_df: pd.DataFrame,
    tuned_vs_default_df: Optional[pd.DataFrame],
    tuned_vs_default_source: str,
    tuned_vs_default_base_source: str,
    default_runs_file: Path,
    run_output_dir: Path,
    output_subdir: str,
    generated_files: List[str],
    dpi: int,
    alpha: float,
    manuscript_only: bool = False,
) -> Dict[str, object]:
    default_aware_dir = run_output_dir / output_subdir
    default_aware_dir.mkdir(parents=True, exist_ok=True)
    pairing_df_source = tuned_vs_default_df if tuned_vs_default_df is not None else tuned_df
    pairing_df = _canonicalize_matched_parameter_runs(pairing_df_source)

    defaults_df = load_default_runs_csv(default_runs_file, split_policy="both")
    metric_specs: List[Tuple[str, str, str]] = [
        ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ("quantization_error_holdout", "QE Holdout", "qe_holdout"),
        ("quantization_error_train", "QE Train", "qe_train"),
    ]

    missing_default = [metric for metric, _, _ in metric_specs if metric not in defaults_df.columns]
    if missing_default:
        raise ValueError(f"Missing default metric columns required for default-aware analysis: {missing_default}")

    tuned_vs_default_variants = _build_default_aware_pairing_variants(
        selected_df=pairing_df,
        selected_source=tuned_vs_default_source,
        base_df=tuned_df,
        base_source=tuned_vs_default_base_source,
        include_selected_topology_variants="pair_topology" in pairing_df.columns,
        include_base_variant=tuned_vs_default_df is None,
    )
    if manuscript_only:
        allowed_variant_keys = {
            "selected_source",
            "selected_source_hex",
            "selected_source_mst",
            "selected_source_rng",
        }
        tuned_vs_default_variants = [
            variant
            for variant in tuned_vs_default_variants
            if str(variant.get("key", "")).strip() in allowed_variant_keys
        ]
    tuned_vs_default_variant_outputs: List[Dict[str, object]] = []
    for variant in tuned_vs_default_variants:
        tuned_vs_default_variant_outputs.append(
            _run_tuned_vs_default_variant(
                variant=variant,
                defaults_df=defaults_df,
                metric_specs=metric_specs,
                default_aware_dir=default_aware_dir,
                generated_files=generated_files,
                dpi=int(dpi),
                alpha=float(alpha),
            )
        )
    primary_tuned_vs_default = tuned_vs_default_variant_outputs[0]

    stability_dir = default_aware_dir / "hyperparameter_stability"
    stability_df, excluded_minibatch_rows = _filter_full_batch_rows_for_stability(tuned_df)
    if stability_df.empty:
        raise ValueError("No rows remain for hyperparameter stability after excluding minibatch runs.")
    topology_pairs = (("hexagonal", "mst"), ("hexagonal", "rng"))
    stability_reports: Dict[str, str] = {}
    stability_reports_sampling: Dict[str, Dict[str, str]] = {}
    for metric, label, slug in metric_specs:
        metric_output_dir = stability_dir / slug
        report_path = build_hyperparameter_stability_report(
            df=stability_df,
            metric=StabilityMetric(
                column=metric,
                label=label,
                slug=slug,
                higher_is_better=False,
            ),
            output_dir=metric_output_dir,
            topology_pairs=topology_pairs,
            markdown_filename=f"HYPERPARAMETER_STABILITY_{slug.upper()}.md",
        )
        stability_reports[metric] = str(report_path.resolve())
        if not manuscript_only:
            generated_files.append(str(report_path.resolve()))

        if "pair_sampling" in stability_df.columns:
            sampling_series = stability_df["pair_sampling"].astype(str).str.lower().str.strip()
            per_sampling_reports: Dict[str, str] = {}
            for sampling_mode in DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES:
                sampling_subset = stability_df[sampling_series == sampling_mode].copy()
                if sampling_subset.empty:
                    continue
                sampling_output_dir = metric_output_dir / f"sampling_{sampling_mode}"
                sampling_report_path = build_hyperparameter_stability_report(
                    df=sampling_subset,
                    metric=StabilityMetric(
                        column=metric,
                        label=label,
                        slug=slug,
                        higher_is_better=False,
                    ),
                    output_dir=sampling_output_dir,
                    topology_pairs=topology_pairs,
                    markdown_filename=f"HYPERPARAMETER_STABILITY_{slug.upper()}_{sampling_mode.upper()}.md",
                )
                per_sampling_reports[sampling_mode] = str(sampling_report_path.resolve())
                if not manuscript_only:
                    generated_files.append(str(sampling_report_path.resolve()))
            if per_sampling_reports:
                stability_reports_sampling[metric] = per_sampling_reports

    default_aware_publication_dir = default_aware_dir / "publication_figures"
    selected_stability_publication = _render_default_aware_stability_publication_figures(
        stability_dir=stability_dir,
        metric_specs=metric_specs,
        selected_parameters=DEFAULT_AWARE_STABILITY_PUBLICATION_PARAMETERS,
        output_dir=default_aware_publication_dir,
        generated_files=generated_files,
        dpi=int(dpi),
        manuscript_only=manuscript_only,
    )
    matched_parameter_publication: Dict[str, object] = {
        "generated": False,
        "reason": "Matched-parameter publication figures require an explicit tuned-runs CSV override.",
    }
    if tuned_vs_default_df is not None and not manuscript_only:
        matched_parameter_publication = _render_matched_parameter_publication_figures(
            tuned_df=pairing_df,
            defaults_df=defaults_df,
            output_dir=default_aware_dir / "matched_parameter_publication_figures",
            generated_files=generated_files,
            dpi=int(dpi),
            alpha=float(alpha),
        )

    summary = {
        "enabled": True,
        "split_policy": "both",
        "default_runs_file": str(default_runs_file.resolve()),
        "output_dir": str(default_aware_dir.resolve()),
        "tuned_vs_default_source": str(primary_tuned_vs_default.get("source", "")),
        "tuned_vs_default_base_source": str(tuned_vs_default_base_source),
        "tuned_vs_default_rows_total": int(primary_tuned_vs_default.get("rows_total", 0)),
        "tuned_vs_default_report": str(primary_tuned_vs_default.get("report", "")),
        "tuned_vs_default_publication": primary_tuned_vs_default.get("publication", {}),
        "tuned_vs_default_variants": tuned_vs_default_variant_outputs,
        "stability_rows_total": int(len(tuned_df)),
        "stability_rows_used": int(len(stability_df)),
        "stability_rows_excluded_minibatch": int(excluded_minibatch_rows),
        "stability_reports": stability_reports,
        "stability_reports_sampling": stability_reports_sampling,
        "metrics": [metric for metric, _, _ in metric_specs],
        "topology_pairs": [list(pair) for pair in topology_pairs],
        "selected_stability_publication": selected_stability_publication,
        "matched_parameter_publication": matched_parameter_publication,
    }
    if not manuscript_only:
        summary_path = default_aware_dir / "default_aware_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        generated_files.append(str(summary_path.resolve()))
        summary["summary_json"] = str(summary_path.resolve())
    else:
        summary["summary_json"] = ""
    return summary

def _resolve_default_aware_paths_from_manifest(
    manifest_path_value: str | Path,
) -> Tuple[Path, Optional[str], str]:
    manifest_path = Path(manifest_path_value)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Default-aware manifest does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise IsADirectoryError(
            f"Default-aware manifest path is not a file: {manifest_path}. Pass a JSON file path."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Default-aware manifest must be a JSON object: {manifest_path}")

    default_runs_value = payload.get("default_runs_file")
    tuned_runs_value = payload.get("default_aware_tuned_runs_file")
    if not default_runs_value:
        raise ValueError(
            "Default-aware manifest is missing required key 'default_runs_file'. "
            f"Manifest: {manifest_path}"
        )

    default_runs_path = Path(str(default_runs_value))
    tuned_runs_path_str = str(tuned_runs_value) if tuned_runs_value else None
    return default_runs_path, tuned_runs_path_str, str(manifest_path.resolve())

def _resolve_default_aware_pairing_input(
    *,
    base_df: pd.DataFrame,
    data_file: Path,
    override_csv_path: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    if not override_csv_path:
        return base_df.copy(), str(data_file.resolve())

    override_path = Path(override_csv_path)
    if not override_path.exists():
        raise FileNotFoundError(f"Default-aware tuned-runs CSV does not exist: {override_path}")
    if not override_path.is_file():
        raise IsADirectoryError(
            f"Default-aware tuned-runs path is not a file: {override_path}. "
            "Pass a CSV file path."
        )

    override_df = pd.read_csv(override_path)
    override_df = _prepare_dataframe(override_df)
    override_df, _ = _exclude_minibatch_rows_for_reporting(override_df)
    return override_df, str(override_path.resolve())
__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
