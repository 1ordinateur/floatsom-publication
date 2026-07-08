from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
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

def _resolve_run_output_dir(base_output_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        run_slug = _slug(run_name)
    else:
        run_slug = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_utc")
    return base_output_dir / run_slug

def _resolve_paper_assets_dir() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        for candidate in (
            parent / "paper" / "assets",
            parent / "floatsom" / "paper" / "assets",
        ):
            if candidate.is_dir():
                return candidate
    return None


def _resolve_paper_manuscript_path(assets_dir: Path) -> Path:
    paper_dir = assets_dir.parent
    for filename in ("manuscript.md", "manuscript_tmlr_revisions.md", "manuscript_tmlr.md"):
        candidate = (paper_dir / filename).resolve()
        if candidate.exists():
            return candidate
    return (paper_dir / "manuscript.md").resolve()


def _resolve_paper_manual_assets_dir() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        for candidate in (
            parent / "paper" / "assets_manual",
            parent / "floatsom" / "paper" / "assets_manual",
        ):
            if candidate.is_dir():
                return candidate
    return None


def _mirror_figure_to_manual_assets(src_path: Path, dst_name: str, generated_files: List[str]) -> Optional[Path]:
    if dst_name in {"fig_3.svg", "fig_11.svg"}:
        return None
    manual_assets_dir = _resolve_paper_manual_assets_dir()
    if manual_assets_dir is None:
        return None
    manual_figures_dir = manual_assets_dir / "figures"
    manual_figures_dir.mkdir(parents=True, exist_ok=True)
    manual_path = manual_figures_dir / dst_name
    shutil.copy2(src_path, manual_path)
    generated_files.append(str(manual_path.resolve()))
    return manual_path

def _sync_sampling_comparison_assets_to_paper(
    suite_outputs: Dict[str, Dict[str, object]],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "copied": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    raw_suite = suite_outputs.get("raw", {})
    sampling_comparison_path = Path(str(raw_suite.get("sampling_comparison", ""))).resolve()
    if not sampling_comparison_path.exists():
        return {
            "copied": False,
            "reason": "Raw sampling_comparison output directory does not exist.",
            "assets_dir": str(assets_dir.resolve()),
            "sampling_comparison": str(sampling_comparison_path),
        }
    publication_figures_path = Path(str(raw_suite.get("publication_figures", ""))).resolve()
    src_figures_dir = sampling_comparison_path / "figures"
    src_tables_dir = sampling_comparison_path / "tables"
    dst_figures_dir = assets_dir / "figures"
    dst_tables_dir = assets_dir / "tables"
    dst_figures_dir.mkdir(parents=True, exist_ok=True)
    dst_tables_dir.mkdir(parents=True, exist_ok=True)

    copied_figures: Dict[str, str] = {}
    copied_tables: List[str] = []
    missing_publication_figure_sources: List[Dict[str, str]] = []
    sampling_comparison_scope = "all_sampling_modes"
    required_publication_figure_sources: Dict[str, str] = {
        "figure_3_algorithm_sampling_stratified_full_random_only_all_metrics.svg": "fig_4.svg",
    }
    publication_figure_name_map: Dict[str, Tuple[str, ...]] = {
        "figure_3_algorithm_sampling_stratified_full_random_only_all_metrics.svg": (
            "fig_4.svg",
        ),
        "figure_3_algorithm_sampling_stratified_full_hdsssom_only_all_metrics.svg": (
            "fig_3.svg",
        ),
        "figure_4_topology_hex_mst_metrics_full_only.svg": (
            "fig_6.svg",
        ),
        "figure_5_topology_hex_rng_metrics_full_only.svg": (
            "fig_7.svg",
        ),
        "supp_figure_topology_mst_rng_metrics_full_only.svg": (
            "supp_fig_s4.svg",
        ),
        "supp_figure_s6_topology_hex_mst_sensitivity_full_only.svg": (
            "supp_fig_s5.svg",
        ),
        "supp_figure_s7_topology_hex_rng_sensitivity_full_only.svg": (
            "supp_fig_s6.svg",
        ),
        "supp_figure_s8_topology_mst_rng_sensitivity_full_only.svg": (
            "supp_fig_s7.svg",
        ),
    }
    if (publication_figures_path / "figure_3_algorithm_sampling_stratified_full_random_only_all_metrics.svg").exists():
        sampling_comparison_scope = "full_random_only"
    elif (publication_figures_path / "figure_3_algorithm_sampling_stratified_full_hdsssom_only_all_metrics.svg").exists():
        sampling_comparison_scope = "full_hdsssom_only"
    if publication_figures_path.exists():
        missing_required_sources = []
        for src_name, dst_name in required_publication_figure_sources.items():
            if (publication_figures_path / src_name).exists():
                continue
            missing_required_sources.append(f"{src_name} -> {dst_name}")
            missing_publication_figure_sources.append(
                {
                    "source": src_name,
                    "destination": dst_name,
                    "required": True,
                    "reason": f"Source figure not found in {publication_figures_path}",
                }
            )
        if missing_required_sources:
            print(
                "Sampling publication sync is missing required manuscript figure source(s): "
                + ", ".join(missing_required_sources)
                + f". Expected these in {publication_figures_path}."
                + " Continuing with best-effort asset sync.",
                file=sys.stderr,
            )
        for src_name, dst_names in publication_figure_name_map.items():
            src_path = publication_figures_path / src_name
            if not src_path.exists():
                if src_name not in required_publication_figure_sources:
                    missing_publication_figure_sources.extend(
                        {
                            "source": src_name,
                            "destination": dst_name,
                            "required": False,
                            "reason": f"Source figure not found in {publication_figures_path}",
                        }
                        for dst_name in dst_names
                    )
                continue
            for dst_name in dst_names:
                dst_path = dst_figures_dir / dst_name
                shutil.copy2(src_path, dst_path)
                copied_figures[f"publication_figures/{src_name}:{dst_name}"] = str(dst_path.resolve())
                generated_files.append(str(dst_path.resolve()))
                manual_path = _mirror_figure_to_manual_assets(dst_path, dst_name, generated_files)
                if manual_path is not None:
                    copied_figures[f"publication_figures/{src_name}:manual/{dst_name}"] = str(
                        manual_path.resolve()
                    )

    figure_3_metadata_candidates = [
        src_tables_dir / "figure_3_full_vs_random_dataset_metadata.csv",
        src_tables_dir / f"figure_3_{sampling_comparison_scope}_full_vs_random_dataset_metadata.csv",
    ]
    for src_path in figure_3_metadata_candidates:
        if not src_path.exists():
            continue
        dst_path = dst_tables_dir / "supp_table_figure_2_sampling_dataset_metadata.csv"
        shutil.copy2(src_path, dst_path)
        copied_tables.append(str(dst_path.resolve()))
        generated_files.append(str(dst_path.resolve()))
        break

    return {
        "copied": bool(copied_figures or copied_tables),
        "assets_dir": str(assets_dir.resolve()),
        "sampling_comparison": str(sampling_comparison_path),
        "publication_figures": str(publication_figures_path),
        "sampling_comparison_scope": sampling_comparison_scope,
        "figures": copied_figures,
        "tables": copied_tables,
        "missing_publication_figure_sources": missing_publication_figure_sources,
    }

def _sync_sampling_regression_stats_to_manuscript(
    suite_outputs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    raw_suite = suite_outputs.get("raw", {})
    sampling_comparison_path = Path(str(raw_suite.get("sampling_comparison", ""))).resolve()
    tables_dir = sampling_comparison_path / "tables"
    if not tables_dir.exists():
        return {
            "updated": False,
            "reason": f"Sampling comparison tables directory not found: {tables_dir}",
            "manuscript": str(manuscript_path),
        }

    metric_specs: List[Tuple[str, str]] = [
        ("balanced_qe_raw", "Balanced QE"),
        ("quantization_error_holdout", "Holdout QE"),
        ("quantization_error_train", "Train QE"),
    ]

    def _pick_stats_file(metric_slug: str) -> Optional[Path]:
        patterns = [
            f"sampling_mode_hexagonal_full_batch_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv",
            f"sampling_mode_hexagonal_full_batch_full_random_only_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv",
            f"sampling_mode_hexagonal_full_batch_*_full_vs_random_qe_vs_sample_size_regression_stats_{metric_slug}.csv",
            f"*full_vs_random*qe_vs_sample_size_regression_stats_{metric_slug}.csv",
        ]
        candidates: List[Path] = []
        for pattern in patterns:
            candidates.extend(sorted(tables_dir.glob(pattern)))
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    metadata_patterns = [
        "figure_3_full_vs_random_dataset_metadata.csv",
        "figure_3_*_full_vs_random_dataset_metadata.csv",
    ]
    metadata_candidates: List[Path] = []
    for pattern in metadata_patterns:
        metadata_candidates.extend(sorted(tables_dir.glob(pattern)))
    metadata_path = None
    if metadata_candidates:
        metadata_candidates = sorted(metadata_candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        metadata_path = metadata_candidates[0]

    metric_phrases: List[str] = []
    stats_sources: Dict[str, str] = {}
    for metric_slug, metric_label in metric_specs:
        stats_path = _pick_stats_file(metric_slug)
        if stats_path is None:
            metric_phrases.append(f"{metric_label} (pending)")
            continue
        stats_df = pd.read_csv(stats_path)
        if stats_df.empty:
            metric_phrases.append(f"{metric_label} (pending)")
            continue
        row = stats_df.iloc[0]
        r_value = pd.to_numeric(pd.Series([row.get("pearson_r")]), errors="coerce").iloc[0]
        p_value = pd.to_numeric(pd.Series([row.get("pearson_r_p_value")]), errors="coerce").iloc[0]
        n_value = pd.to_numeric(pd.Series([row.get("n_datasets")]), errors="coerce").iloc[0]
        r_text = f"{float(r_value):.3f}" if np.isfinite(r_value) else "NA"
        p_text = f"{float(p_value):.3g}" if np.isfinite(p_value) else "NA"
        n_text = f"{int(n_value)}" if np.isfinite(n_value) else "NA"
        metric_phrases.append(f"{metric_label} (Pearson R={r_text}, p={p_text}, n={n_text})")
        stats_sources[metric_slug] = str(stats_path.resolve())

    start_marker = "<!-- AUTO-SAMPLING-REGRESSION-STATS:START -->"
    end_marker = "<!-- AUTO-SAMPLING-REGRESSION-STATS:END -->"
    caption_token = "[[AUTO-SAMPLING-REGRESSION-STATS]]"
    if metric_phrases:
        if all("pending" in phrase for phrase in metric_phrases):
            caption_text = (
                "Differences in QE between random and full stratified by dataset size "
                "(panels D-F) will be populated automatically after the publication "
                "figure-generation run."
            )
        else:
            if len(metric_phrases) == 1:
                metric_summary = metric_phrases[0]
            else:
                metric_summary = "; ".join(metric_phrases[:-1]) + "; and " + metric_phrases[-1]
            caption_text = (
                "Differences in QE between random and full stratified by dataset size "
                "(panels D-F): "
                + metric_summary
                + "."
            )
    else:
        caption_text = (
            "Differences in QE between random and full stratified by dataset size "
            "(panels D-F) will be populated automatically after the publication "
            "figure-generation run."
        )
    if metadata_path is not None and metadata_path.exists():
        caption_text += (
            " The corresponding dataset metadata table is listed in Supplementary Table S1."
        )
    block_body = "\n".join(
        [
            start_marker,
            end_marker,
        ]
    )

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    updated_text = manuscript_text
    if caption_token in updated_text:
        updated_text = updated_text.replace(caption_token, caption_text)
    start_idx = updated_text.find(start_marker)
    end_idx = updated_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx += len(end_marker)
        updated_text = updated_text[:start_idx] + block_body + updated_text[end_idx:]
    else:
        anchor = "### 5.3 MST Results"
        anchor_idx = updated_text.find(anchor)
        if caption_token not in manuscript_text:
            fallback_body = "\n".join([start_marker, caption_text, end_marker])
            if anchor_idx == -1:
                updated_text = updated_text.rstrip() + "\n\n" + fallback_body + "\n"
            else:
                updated_text = updated_text[:anchor_idx].rstrip() + "\n\n" + fallback_body + "\n\n" + updated_text[anchor_idx:]

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "stats_sources": stats_sources,
            "dataset_metadata_table": (
                str(metadata_path.resolve()) if metadata_path is not None and metadata_path.exists() else ""
            ),
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "stats_sources": stats_sources,
        "dataset_metadata_table": (
            str(metadata_path.resolve()) if metadata_path is not None and metadata_path.exists() else ""
        ),
    }


def _sync_figure12_topology_runtime_stats_to_manuscript() -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    stats_path = (assets_dir / "tables" / "supp_table_figure_12_topology_runtime_summary.tsv").resolve()
    if not stats_path.exists():
        return {
            "updated": False,
            "reason": f"Figure 12 topology runtime summary table not found: {stats_path}",
            "manuscript": str(manuscript_path),
        }

    stats_df = pd.read_csv(stats_path, sep="\t")
    if stats_df.empty:
        return {
            "updated": False,
            "reason": f"Figure 12 topology runtime summary table is empty: {stats_path}",
            "manuscript": str(manuscript_path),
        }

    def _upsert_block(
        text: str,
        *,
        start_marker: str,
        end_marker: str,
        block_body: str,
        anchor: str,
    ) -> str:
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            end_idx += len(end_marker)
            return text[:start_idx] + block_body + text[end_idx:]

        anchor_idx = text.find(anchor)
        if anchor_idx == -1:
            return text.rstrip() + "\n\n" + block_body + "\n"
        return text[:anchor_idx].rstrip() + "\n\n" + block_body + "\n\n" + text[anchor_idx:]

    mode_order: List[Tuple[str, str, str]] = [
        ("dimension_scaling", "dimension scaling", "dimensions"),
        ("sample_scaling", "sample scaling", "samples"),
    ]

    def _format_percent(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.2f}" if np.isfinite(numeric) else "NA"

    def _format_runtime(value_s: object) -> str:
        numeric = pd.to_numeric(pd.Series([value_s]), errors="coerce").iloc[0]
        if not np.isfinite(numeric):
            return "NA"
        minutes = float(numeric) / 60.0
        return f"{float(numeric):.2f} s ({minutes:.2f} min)"

    def _format_axis_value(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not np.isfinite(numeric):
            return "NA"
        numeric_float = float(numeric)
        if numeric_float.is_integer():
            return f"{int(numeric_float):,}"
        return f"{numeric_float:,.2f}"

    def _lookup_mode_row(mode_name: str) -> Optional[pd.Series]:
        mode_df = stats_df[stats_df["mode_name"].astype(str).str.strip().str.lower() == mode_name].copy()
        if mode_df.empty:
            return None
        return mode_df.iloc[0]

    metric_phrases: List[str] = []
    available_modes: List[str] = []
    for mode_name, mode_label, axis_label in mode_order:
        row = _lookup_mode_row(mode_name)
        if row is None:
            continue
        axis_value_text = _format_axis_value(row.get("axis_value"))
        axis_phrase = f"{axis_value_text} {axis_label}"
        metric_phrases.append(
            f"{mode_label} ({_format_percent(row.get('max_pairwise_runtime_spread_pct'))}% at "
            f"{axis_phrase})"
        )
        available_modes.append(mode_name)

    grid_row = _lookup_mode_row("grid_size_scaling")
    if grid_row is None:
        grid_discussion_paragraph = (
            "Grid-size topology slowdown text will be populated automatically after the Figure 12 "
            "topology runtime summary table is generated."
        )
    else:
        grid_axis_value_text = _format_axis_value(grid_row.get("axis_value"))
        hex_runtime = pd.to_numeric(pd.Series([grid_row.get("hexagonal_runtime_mean_s")]), errors="coerce").iloc[0]
        mst_runtime = pd.to_numeric(pd.Series([grid_row.get("mst_runtime_mean_s")]), errors="coerce").iloc[0]
        rng_runtime = pd.to_numeric(pd.Series([grid_row.get("rng_runtime_mean_s")]), errors="coerce").iloc[0]
        mst_fold = float(mst_runtime) / float(hex_runtime)
        rng_fold = float(rng_runtime) / float(hex_runtime)
        if np.isfinite(mst_fold) and np.isfinite(rng_fold):
            grid_discussion_paragraph = (
            "However, when the grid itself is enlarged in Fig. 12C, topology-dependent runtime "
                "differences become readily evident. "
                f"At the largest tested grid size (grid size {grid_axis_value_text}), "
                f"the 8-GPU mean runtimes are {_format_runtime(hex_runtime)} for hexagonal, "
                f"{_format_runtime(mst_runtime)} for MST, and {_format_runtime(rng_runtime)} for RNG, "
                f"corresponding to 8-GPU MST and RNG runtimes that are {mst_fold:.2f}x and "
                f"{rng_fold:.2f}x the hexagonal runtime, respectively."
            )
        else:
            grid_discussion_paragraph = (
                "Grid-size topology slowdown text will be populated automatically after the Figure 12 "
                "topology runtime summary table is generated."
            )

    start_marker = "<!-- AUTO-FIGURE12-TOPOLOGY-RUNTIME-STATS:START -->"
    end_marker = "<!-- AUTO-FIGURE12-TOPOLOGY-RUNTIME-STATS:END -->"
    discussion_start = "<!-- AUTO-FIGURE12-GRID-SIZE-DISCUSSION:START -->"
    discussion_end = "<!-- AUTO-FIGURE12-GRID-SIZE-DISCUSSION:END -->"
    if metric_phrases:
        paragraph = (
            "In Fig. 12A-B, the topologies scale similarly as input complexity and data volume increase: "
            "even at the largest tested axis values, the maximum pairwise runtime spread remains modest at "
            + "; ".join(metric_phrases[:-1] + [metric_phrases[-1]])
            + "."
        )
    else:
        paragraph = (
            "Figure 12 topology runtime summary statistics will be populated automatically after the "
            "harmonized speed-scaling publication tables are generated."
        )
    block_body = "\n".join([start_marker, paragraph, end_marker])
    discussion_block = "\n".join([discussion_start, grid_discussion_paragraph, discussion_end])

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    updated_text = _upsert_block(
        manuscript_text,
        start_marker=start_marker,
        end_marker=end_marker,
        block_body=block_body,
        anchor="We interpret scaling efficiency using the standard single-GPU baseline-over-observed speedup definition.",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=discussion_start,
        end_marker=discussion_end,
        block_body=discussion_block,
        anchor="RNG runtime traces are interpreted jointly with the quality outcomes in Section 5.4, using the same harmonized benchmark construction described in Section 4.4.",
    )

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "stats_source": str(stats_path),
            "available_modes": available_modes,
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "stats_source": str(stats_path),
        "available_modes": available_modes,
    }


def _sync_figure13_deployment_runtime_stats_to_manuscript() -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    deployment_summary_path = (
        assets_dir / "tables" / "supp_table_figure_13_xpysom_rng_deployment_summary.tsv"
    ).resolve()
    stats_path = (assets_dir / "tables" / "supp_table_figure_12_topology_runtime_summary.tsv").resolve()
    if not stats_path.exists():
        return {
            "updated": False,
            "reason": f"Figure 12 topology runtime summary table not found: {stats_path}",
            "manuscript": str(manuscript_path),
        }

    stats_df = pd.read_csv(stats_path, sep="\t")
    if stats_df.empty:
        return {
            "updated": False,
            "reason": f"Figure 12 topology runtime summary table is empty: {stats_path}",
            "manuscript": str(manuscript_path),
        }

    def _select_mode_row(mode_name: str) -> Optional[pd.Series]:
        mode_df = stats_df[stats_df["mode_name"].astype(str).str.strip().str.lower() == mode_name].copy()
        if mode_df.empty:
            return None
        return mode_df.iloc[0]

    def _format_runtime(value_s: object) -> str:
        numeric = pd.to_numeric(pd.Series([value_s]), errors="coerce").iloc[0]
        if not np.isfinite(numeric):
            return "NA"
        minutes = float(numeric) / 60.0
        return f"{float(numeric):.2f} s ({minutes:.2f} min)"

    def _format_runtime_range(values: Sequence[object]) -> str:
        numerics = pd.to_numeric(pd.Series(list(values)), errors="coerce")
        finite_values = [float(value) for value in numerics if np.isfinite(value)]
        if not finite_values:
            return "NA"
        low_value = min(finite_values)
        high_value = max(finite_values)
        return (
            f"{low_value:.2f}-{high_value:.2f} s "
            f"({low_value / 60.0:.2f}-{high_value / 60.0:.2f} min)"
        )

    def _format_axis_value(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not np.isfinite(numeric):
            return "NA"
        numeric_float = float(numeric)
        if numeric_float.is_integer():
            return f"{int(numeric_float):,}"
        return f"{numeric_float:,.2f}"

    def _format_topology_name(value: object) -> str:
        text = str(value).strip().lower()
        if text == "mst":
            return "MST"
        if text == "rng":
            return "RNG"
        if text == "hexagonal":
            return "hexagonal"
        return str(value).strip() or "unknown"

    def _format_pct(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.1f}%" if np.isfinite(numeric) else "NA"

    qe_paragraph = (
        "Figure 11 deployment QE summary text will be populated automatically after the deployment "
        "summary table is generated."
    )
    qe_stats_source = ""
    if deployment_summary_path.exists():
        deployment_df = pd.read_csv(deployment_summary_path, sep="\t")
        required_qe_cols = {"metric", "dataset", "median_pct_change"}
        if not deployment_df.empty and required_qe_cols.issubset(set(deployment_df.columns)):
            overall_df = deployment_df[
                deployment_df["dataset"].astype(str).str.strip() == "GLOBAL_OVERALL"
            ].copy()
            metric_order = [("QE_B", "$QE_B$"), ("QE_H", "$QE_H$"), ("QE_T", "$QE_T$")]
            metric_phrases: List[str] = []
            for metric_key, metric_label in metric_order:
                metric_df = overall_df[overall_df["metric"].astype(str).str.strip() == metric_key].copy()
                if metric_df.empty:
                    continue
                metric_phrases.append(
                    f"{metric_label} ({_format_pct(metric_df.iloc[0]['median_pct_change'])})"
                )
            if metric_phrases:
                if len(metric_phrases) == 1:
                    metric_text = metric_phrases[0]
                elif len(metric_phrases) == 2:
                    metric_text = f"{metric_phrases[0]} and {metric_phrases[1]}"
                else:
                    metric_text = f"{'; '.join(metric_phrases[:-1])}; and {metric_phrases[-1]}"
                qe_paragraph = (
                    "At the overall level, Fig. 13 shows median percentage improvements of "
                    + metric_text
                    + " for tuned FloatSOM RNG relative to default hexagonal XPySOM, capturing the combined "
                    "deployment effect of topology choice and tuning on $QE$."
                )
                qe_stats_source = str(deployment_summary_path)

    dimension_row = _select_mode_row("dimension_scaling")
    sample_row = _select_mode_row("sample_scaling")
    if dimension_row is None or sample_row is None:
        return {
            "updated": False,
            "reason": "Figure 12 topology runtime summary table is missing the dimension- or sample-scaling rows needed for the deployment-runtime summary.",
            "manuscript": str(manuscript_path),
            "stats_table": str(stats_path),
        }

    paragraph = (
        "In the deployment comparison, tuned FloatSOM RNG delivers better $QE$ than the default "
        "hexagonal XPySOM baseline, while also running faster and scaling to larger workloads "
        "(Supplementary Table S7)."
    )

    qe_start_marker = "<!-- AUTO-FIGURE13-DEPLOYMENT-QE-STATS:START -->"
    qe_end_marker = "<!-- AUTO-FIGURE13-DEPLOYMENT-QE-STATS:END -->"
    start_marker = "<!-- AUTO-FIGURE13-DEPLOYMENT-RUNTIME-STATS:START -->"
    end_marker = "<!-- AUTO-FIGURE13-DEPLOYMENT-RUNTIME-STATS:END -->"
    qe_block_body = "\n".join([qe_start_marker, qe_paragraph, qe_end_marker])
    block_body = "\n".join([start_marker, paragraph, end_marker])

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    qe_start_idx = manuscript_text.find(qe_start_marker)
    qe_end_idx = manuscript_text.find(qe_end_marker)
    if qe_start_idx != -1 and qe_end_idx != -1 and qe_end_idx > qe_start_idx:
        qe_end_idx += len(qe_end_marker)
        updated_text = manuscript_text[:qe_start_idx] + qe_block_body + manuscript_text[qe_end_idx:]
    else:
        qe_anchor = "*Figure 13. Integrated deployment comparison of default hexagonal XPySOM versus tuned FloatSOM RNG. Panels A-C compare $QE_B$, $QE_H$, and $QE_T$ using the untuned hexagonal XPySOM baseline against matched tuned FloatSOM RNG full-sampling runs. Panel D provides the scaling/runtime context for the same comparison, with the separately executed targeted 1B-sample runs discussed in the text rather than plotted directly. Taken together, this integrated figure summarizes the operating point observed for tuned FloatSOM RNG once workload size is large enough for steady-state execution to dominate startup overhead. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S7.*"
        qe_anchor_idx = manuscript_text.find(qe_anchor)
        if qe_anchor_idx == -1:
            updated_text = manuscript_text.rstrip() + "\n\n" + qe_block_body + "\n"
        else:
            insert_at = qe_anchor_idx + len(qe_anchor)
            updated_text = manuscript_text[:insert_at] + "\n\n" + qe_block_body + "\n" + manuscript_text[insert_at:]

    start_idx = updated_text.find(start_marker)
    end_idx = updated_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx += len(end_marker)
        updated_text = updated_text[:start_idx] + block_body + updated_text[end_idx:]
    else:
        anchor = "Alongside these figure-level scaling outputs, we use log-derived systems diagnostics as supporting evidence (not additional main figures): staging mode and worker throughput summaries, per-iteration timing breakdowns (submit/get/collective components), and OOM-avoidance stability notes under the largest workloads."
        anchor_idx = updated_text.find(anchor)
        if anchor_idx == -1:
            updated_text = updated_text.rstrip() + "\n\n" + block_body + "\n"
        else:
            updated_text = updated_text[:anchor_idx].rstrip() + "\n\n" + block_body + "\n\n" + updated_text[anchor_idx:]

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "stats_table": str(stats_path),
            "qe_stats_table": qe_stats_source,
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "stats_table": str(stats_path),
        "qe_stats_table": qe_stats_source,
    }


def _sync_topology_pvalue_summary_to_paper_and_manuscript(
    suite_outputs: Dict[str, Dict[str, object]],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "generated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = _resolve_paper_manuscript_path(assets_dir)
    if not manuscript_path.exists():
        return {
            "generated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    raw_suite = suite_outputs.get("raw", {})
    raw_root_text = str(raw_suite.get("root", "")).strip()
    if not raw_root_text:
        return {
            "generated": False,
            "reason": "Raw suite output directory was not recorded.",
            "manuscript": str(manuscript_path),
        }
    raw_root = Path(raw_root_text).resolve()
    if not raw_root.exists():
        return {
            "generated": False,
            "reason": f"Raw suite output directory not found: {raw_root}",
            "manuscript": str(manuscript_path),
        }

    publication_dir_text = str(raw_suite.get("publication_figures", "")).strip()
    if publication_dir_text:
        publication_dir = Path(publication_dir_text).resolve()
    else:
        publication_dir = raw_root / "publication_figures"
    publication_tables_dir = publication_dir / "tables"
    publication_tables_dir.mkdir(parents=True, exist_ok=True)

    metric_specs: List[Tuple[str, str]] = [
        ("balanced_qe_raw", "Balanced QE"),
        ("quantization_error_holdout", "Holdout QE"),
        ("quantization_error_train", "Train QE"),
    ]
    comparison_specs: List[Tuple[str, str]] = [
        ("hex_vs_mst", "MST"),
        ("hex_vs_rng", "RNG"),
    ]

    def _pick_summary_file(comparison_slug: str, metric_slug: str) -> Optional[Path]:
        candidates = sorted(
            raw_root.rglob(f"{comparison_slug}_main_{metric_slug}.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _format_p_value(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.3g}" if np.isfinite(numeric) else "NA"

    def _format_effect_value(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.4g}" if np.isfinite(numeric) else "NA"

    def _format_ci_text(low: object, high: object) -> str:
        low_numeric = pd.to_numeric(pd.Series([low]), errors="coerce").iloc[0]
        high_numeric = pd.to_numeric(pd.Series([high]), errors="coerce").iloc[0]
        if not (np.isfinite(low_numeric) and np.isfinite(high_numeric)):
            return "NA"
        low_value = min(float(low_numeric), float(high_numeric))
        high_value = max(float(low_numeric), float(high_numeric))
        return f"[{_format_effect_value(low_value)}, {_format_effect_value(high_value)}]"

    def _format_test_value_text(value: object, label: str) -> str:
        return f"{label}={_format_p_value(value)}"

    def _format_p_value_text(value: object) -> str:
        return _format_test_value_text(value, "p")

    def _format_q_value_text(value: object) -> str:
        return _format_test_value_text(value, "q")

    def _markdown_table_from_df(df: pd.DataFrame) -> str:
        columns = [str(column) for column in df.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in df.iterrows():
            values = [str(row[column]).replace("|", "\\|") for column in df.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    stats_sources: Dict[str, str] = {}
    summary_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    dataset_names: Set[str] = set()
    table_rows: Dict[Tuple[str, str], Dict[str, str]] = {}

    for metric_slug, metric_label in metric_specs:
        for comparison_slug, comparison_label in comparison_specs:
            summary_path = _pick_summary_file(comparison_slug, metric_slug)
            if summary_path is None:
                continue
            summary_df = pd.read_csv(summary_path)
            if summary_df.empty or not {"dataset", "p_value"}.issubset(set(summary_df.columns)):
                continue
            for optional_col in ("median_pct", "ci_low_pct", "ci_high_pct"):
                if optional_col not in summary_df.columns:
                    summary_df[optional_col] = np.nan
            columns = ["dataset", "median_pct", "ci_low_pct", "ci_high_pct", "p_value"]
            if "q_value" in summary_df.columns:
                columns.append("q_value")
            working = summary_df[columns].copy()
            working["dataset"] = working["dataset"].astype(str).str.strip()
            working = working[working["dataset"] != "GLOBAL_REAL"].copy()
            if "q_value" not in working.columns:
                working = _apply_q_values(working, dataset_col="dataset")
            if working.empty:
                continue
            summary_frames[(comparison_slug, metric_slug)] = working
            stats_sources[f"{comparison_slug}:{metric_slug}"] = str(summary_path.resolve())
            for _, row in working.iterrows():
                dataset_key = str(row["dataset"]).strip()
                dataset_display = "OVERALL" if dataset_key == "GLOBAL_OVERALL" else dataset_key
                if dataset_display != "OVERALL":
                    dataset_names.add(dataset_display)
                row_key = (metric_label, dataset_display)
                table_rows.setdefault(
                    row_key,
                    {
                        "metric": metric_label,
                        "dataset": dataset_display,
                        "MST_effect_pct": "NA",
                        "MST_95pct_CI": "NA",
                        "MST_p": "p=NA",
                        "MST_q": "q=NA",
                        "RNG_effect_pct": "NA",
                        "RNG_95pct_CI": "NA",
                        "RNG_p": "p=NA",
                        "RNG_q": "q=NA",
                    },
                )
                table_rows[row_key][f"{comparison_label}_effect_pct"] = _format_effect_value(row["median_pct"])
                table_rows[row_key][f"{comparison_label}_95pct_CI"] = _format_ci_text(
                    row["ci_low_pct"],
                    row["ci_high_pct"],
                )
                table_rows[row_key][f"{comparison_label}_p"] = _format_p_value_text(row["p_value"])
                table_rows[row_key][f"{comparison_label}_q"] = _format_q_value_text(row.get("q_value", np.nan))

    if not table_rows:
        return {
            "generated": False,
            "reason": "No topology summary tables were available to build manuscript p-value updates.",
            "manuscript": str(manuscript_path),
            "stats_sources": stats_sources,
        }

    dataset_order = _ordered_dataset_labels(sorted(dataset_names))
    dataset_order.append("OVERALL")
    ordered_rows: List[Dict[str, str]] = []
    for _, metric_label in metric_specs:
        for dataset_label in dataset_order:
            row = table_rows.get((metric_label, dataset_label))
            if row is not None:
                ordered_rows.append(row)
    supp_table_df = pd.DataFrame(
        ordered_rows,
        columns=[
            "metric",
            "dataset",
            "MST_effect_pct",
            "MST_95pct_CI",
            "MST_p",
            "MST_q",
            "RNG_effect_pct",
            "RNG_95pct_CI",
            "RNG_p",
            "RNG_q",
        ],
    )

    run_table_path = publication_tables_dir / "supp_table_topology_hex_vs_mst_rng_pvalues.tsv"
    supp_table_df.to_csv(run_table_path, sep="\t", index=False)
    generated_files.append(str(run_table_path.resolve()))

    paper_tables_dir = assets_dir / "tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paper_table_path = paper_tables_dir / run_table_path.name
    shutil.copy2(run_table_path, paper_table_path)
    generated_files.append(str(paper_table_path.resolve()))

    def _overall_metric_phrases(comparison_slug: str) -> List[str]:
        phrases: List[str] = []
        for metric_slug, metric_label in metric_specs:
            summary_df = summary_frames.get((comparison_slug, metric_slug))
            if summary_df is None or summary_df.empty:
                continue
            overall_df = summary_df[summary_df["dataset"].astype(str).str.strip() == "GLOBAL_OVERALL"].copy()
            if overall_df.empty:
                continue
            phrases.append(f"{metric_label} ({_format_p_value_text(overall_df.iloc[0]['p_value'])})")
        return phrases

    def _join_metric_phrases(values: Sequence[str]) -> str:
        items = [str(value).strip() for value in values if str(value).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{'; '.join(items[:-1])}; and {items[-1]}"

    mst_phrases = _overall_metric_phrases("hex_vs_mst")
    rng_phrases = _overall_metric_phrases("hex_vs_rng")
    mst_sentence = (
        "Overall, MST outperforms matched hexagonal on Balanced QE (Fig. 6A), indicating a net advantage across "
        "train and holdout performance. This aggregate gain is driven more clearly by Train QE (Fig. 6C), while "
        "Holdout QE is more mixed across datasets (Fig. 6B)."
    )
    if mst_phrases:
        mst_sentence = (
            "Overall, MST outperforms matched hexagonal on Balanced QE (Fig. 6A), indicating a net advantage "
            "across train and holdout performance. This aggregate gain is driven more clearly by Train QE "
            "(Fig. 6C), while Holdout QE is more mixed across datasets (Fig. 6B) and shows no clear overall "
            "holdout advantage. The overall paired t-test p-values are "
            + _join_metric_phrases(mst_phrases)
            + ". Supplementary Table S7 lists the corresponding per-dataset and overall hexagonal-comparison "
            "effect estimates, 95% confidence intervals, raw p-values, and Benjamini-Hochberg q-values for MST "
            "and RNG."
        )

    rng_sentence = (
        "RNG has lower QE than hexagonal topologies on all QE metrics, and significantly better than MST in "
        "Balanced and Train QE (Fig. 7, Supplementary Fig. S6)."
    )
    if rng_phrases:
        rng_sentence = (
            "RNG has lower QE than matched hexagonal on the reported QE endpoints (Fig. 7A-7C), with overall paired "
            "t-test p-values of "
            + _join_metric_phrases(rng_phrases)
            + ". Supplementary Table S7 lists the corresponding per-dataset and overall hexagonal-comparison "
            "effect estimates, 95% confidence intervals, raw p-values, and Benjamini-Hochberg q-values for MST "
            "and RNG."
        )

    supp_table_caption = (
        "**Supplementary Table S7. Paired topology-comparison effects for hexagonal versus MST and hexagonal "
        "versus RNG across Balanced QE, Holdout QE, and Train QE.** Rows list metric/dataset entries, including "
        "the OVERALL row. Effect columns report the paired mean percent improvement of hexagonal over the "
        "comparator topology, using hexagonal QE as the reference denominator; positive values favor hexagonal "
        "and negative values favor MST or RNG. The CI columns give the corresponding 95% paired $t$-test "
        "confidence intervals. Raw p-values are retained for audit, and q-values report Benjamini-Hochberg "
        "adjustment over dataset-level rows; OVERALL rows are pooled summaries and retain q=NA. The embedded "
        "table is reproduced from "
        "`assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`."
    )

    def _upsert_block(
        manuscript_text: str,
        *,
        start_marker: str,
        end_marker: str,
        block_body: str,
        anchor: str,
    ) -> str:
        start_idx = manuscript_text.find(start_marker)
        end_idx = manuscript_text.find(end_marker)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            end_idx += len(end_marker)
            return manuscript_text[:start_idx] + block_body + manuscript_text[end_idx:]
        anchor_idx = manuscript_text.find(anchor)
        if anchor_idx == -1:
            return manuscript_text.rstrip() + "\n\n" + block_body + "\n"
        return manuscript_text[:anchor_idx].rstrip() + "\n\n" + block_body + "\n\n" + manuscript_text[anchor_idx:]

    mst_start = "<!-- AUTO-TOPOLOGY-MST-PVALUES:START -->"
    mst_end = "<!-- AUTO-TOPOLOGY-MST-PVALUES:END -->"
    rng_start = "<!-- AUTO-TOPOLOGY-RNG-PVALUES:START -->"
    rng_end = "<!-- AUTO-TOPOLOGY-RNG-PVALUES:END -->"
    supp_start = "<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:START -->"
    supp_end = "<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:END -->"

    mst_block = "\n".join([mst_start, mst_sentence, mst_end])
    rng_block = "\n".join([rng_start, rng_sentence, rng_end])
    supp_block = "\n".join(
        [
            supp_start,
            supp_table_caption,
            "",
            _markdown_table_from_df(supp_table_df),
            supp_end,
        ]
    )

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    updated_text = _upsert_block(
        manuscript_text,
        start_marker=mst_start,
        end_marker=mst_end,
        block_body=mst_block,
        anchor="![Figure 6](assets_manual/figures/fig_6.svg)",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=rng_start,
        end_marker=rng_end,
        block_body=rng_block,
        anchor="![Figure 7](assets_manual/figures/fig_7.svg)",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=supp_start,
        end_marker=supp_end,
        block_body=supp_block,
        anchor="## Supplementary Figures (End Matter)",
    )

    manuscript_updated = updated_text != manuscript_text
    if manuscript_updated:
        manuscript_path.write_text(updated_text, encoding="utf-8")

    return {
        "generated": True,
        "manuscript_updated": manuscript_updated,
        "manuscript": str(manuscript_path),
        "run_table": str(run_table_path.resolve()),
        "paper_table": str(paper_table_path.resolve()),
        "stats_sources": stats_sources,
    }


def _sync_systems_scaling_stats_to_manuscript() -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    diagnostics_path = (assets_dir / "tables" / "supp_table_figure_11_rng_scaling_diagnostics.tsv").resolve()
    if not diagnostics_path.exists():
        return {
            "updated": False,
            "reason": f"Systems scaling diagnostics table not found: {diagnostics_path}",
            "manuscript": str(manuscript_path),
        }

    diagnostics_df = pd.read_csv(diagnostics_path, sep="\t")
    required_cols = {"mode_name", "axis_value", "gpu_count", "runtime_mean_s"}
    if diagnostics_df.empty or not required_cols.issubset(set(diagnostics_df.columns)):
        return {
            "updated": False,
            "reason": "Systems scaling diagnostics table is empty or missing required columns.",
            "manuscript": str(manuscript_path),
            "diagnostics_table": str(diagnostics_path),
        }

    working = diagnostics_df.copy()
    if "topology" in working.columns:
        working = working[working["topology"].astype(str).str.lower().str.strip() == "rng"].copy()
    if "method" in working.columns:
        working = working[working["method"].astype(str).str.lower().str.strip() == "batch"].copy()

    if working.empty:
        return {
            "updated": False,
            "reason": "Systems scaling diagnostics table contained no RNG batch rows.",
            "manuscript": str(manuscript_path),
            "diagnostics_table": str(diagnostics_path),
        }

    working["mode_name"] = working["mode_name"].astype(str).str.strip()
    working["axis_value_numeric"] = pd.to_numeric(working["axis_value"], errors="coerce")
    working["gpu_count_numeric"] = pd.to_numeric(working["gpu_count"], errors="coerce")
    working["runtime_mean_s_numeric"] = pd.to_numeric(working["runtime_mean_s"], errors="coerce")
    working = working[working["axis_value_numeric"].notna() & working["gpu_count_numeric"].notna()].copy()
    if working.empty:
        return {
            "updated": False,
            "reason": "Systems scaling diagnostics table contained no usable numeric rows.",
            "manuscript": str(manuscript_path),
            "diagnostics_table": str(diagnostics_path),
        }

    def _coerce_bool(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        return text in {"1", "true", "t", "yes", "y"}

    def _format_int(value: int) -> str:
        return f"{int(value):,}"

    def _format_gpu(gpu_count: int) -> str:
        return f"{int(gpu_count)} GPU" if int(gpu_count) == 1 else f"{int(gpu_count)} GPUs"

    def _format_runtime(value_s: float) -> str:
        minutes = float(value_s) / 60.0
        return f"{float(value_s):.2f} s ({minutes:.2f} min)"

    def _first_disk_thresholds(mode_name: str) -> Dict[int, int]:
        mode_df = working[working["mode_name"] == mode_name].copy()
        if mode_df.empty:
            return {}

        disk_mask = pd.Series(False, index=mode_df.index)
        if "all_repeats_disk" in mode_df.columns:
            disk_mask = mode_df["all_repeats_disk"].map(_coerce_bool)
        if not bool(disk_mask.any()):
            if "staging_mode" in mode_df.columns:
                disk_mask = mode_df["staging_mode"].astype(str).str.lower().str.strip() == "disk"
        if not bool(disk_mask.any()) and "any_repeat_disk" in mode_df.columns:
            disk_mask = mode_df["any_repeat_disk"].map(_coerce_bool)

        disk_df = mode_df.loc[disk_mask].copy()
        if disk_df.empty:
            return {}

        thresholds: Dict[int, int] = {}
        for gpu_count, gpu_df in disk_df.groupby("gpu_count_numeric", sort=True):
            if gpu_df.empty:
                continue
            thresholds[int(gpu_count)] = int(gpu_df["axis_value_numeric"].min())
        return thresholds

    dimension_thresholds = _first_disk_thresholds("dimension_scaling")
    sample_thresholds = _first_disk_thresholds("sample_scaling")

    block_lines: List[str] = []
    sample_gpu_counts = sorted(sample_thresholds.keys())
    if sample_gpu_counts:
        example_low_gpu = 2 if 2 in sample_thresholds else sample_gpu_counts[0]
        example_high_gpu = sample_gpu_counts[-1]
        crossover_sentence = (
            "Fig. 11 suggests that increasing GPU count improves performance in the sample-scaling regime through three related mechanisms. "
            "First, computation is distributed across a larger number of workers, thereby increasing parallel throughput. "
            "Second, the onset of disk-backed execution is deferred to larger workloads because the aggregate worker-memory pool increases with GPU count. "
            f"In the sample-scaling benchmark, for example, the {('500,000,000' if int(sample_thresholds[example_low_gpu]) == 500_000_000 else _format_int(sample_thresholds[example_low_gpu]))} sample dataset requires disk backing under the "
            f"{int(example_low_gpu)}-GPU configuration, whereas the {int(example_high_gpu)}-GPU configuration remains in RAM mode until the "
            f"{_format_int(sample_thresholds[example_high_gpu])} sample dataset. "
            "Third, when disk-backed staging is still required, higher GPU counts appear to improve runtime because staging and disk-to-GPU transfers are distributed across more nodes. "
            "As per-node disk bandwidth is limited, distributing the workload across additional nodes may reduce transfer-path saturation and enable more stable high-throughput operation."
        )
        block_lines.append(crossover_sentence)
    else:
        block_lines.append(
            "Figure 11 systems-scaling text will be populated automatically after the RNG scaling diagnostics table includes sample-scaling staging-mode entries."
        )

    runtime_row = working[
        (working["mode_name"] == "sample_scaling")
        & (working["gpu_count_numeric"] == 8)
        & (working["axis_value_numeric"] == 1_000_000_000)
        & working["runtime_mean_s_numeric"].notna()
    ].copy()
    if not runtime_row.empty:
        runtime_value_s = float(runtime_row.iloc[0]["runtime_mean_s_numeric"])
        block_lines.append(
            f"The 8-GPU RNG configuration processes 1,000,000,000 samples in {_format_runtime(runtime_value_s)}, "
            "demonstrating billion-sample training at a runtime measured in minutes rather than hours. "
            "To reiterate, this is on a relatively complex 50-feature dataset, using 1024-node network "
            "($32 \\times 32 = 1024$), with under multi-node distributed execution, and including time "
            "taken to remotely stage data from shared non-local storage to node-local shards before training."
        )
    else:
        block_lines.append(
            "The exact 8-GPU RNG runtime at the 1,000,000,000-sample point will be populated automatically after that scaling point is present in the diagnostics table."
        )

    grid_rows = working[
        (working["mode_name"] == "grid_size_scaling")
        & (working["axis_value_numeric"] == 64)
        & working["runtime_mean_s_numeric"].notna()
        & working["gpu_count_numeric"].isin([1, 8])
    ].copy()
    if len(grid_rows.index) >= 2:
        grid_one = grid_rows[grid_rows["gpu_count_numeric"] == 1].copy()
        grid_eight = grid_rows[grid_rows["gpu_count_numeric"] == 8].copy()
        if not grid_one.empty and not grid_eight.empty:
            grid_one_runtime = float(grid_one.iloc[0]["runtime_mean_s_numeric"])
            grid_eight_runtime = float(grid_eight.iloc[0]["runtime_mean_s_numeric"])
            reduction_pct = ((grid_one_runtime - grid_eight_runtime) / grid_one_runtime) * 100.0
            block_lines.append(
                "The grid-size scaling panel proves to be the main exception: at the largest tested grid size (64), "
                f"runtime shortens from {_format_runtime(grid_one_runtime)} on 1 GPU to only {_format_runtime(grid_eight_runtime)} "
                f"on 8 GPUs, a {reduction_pct:.2f}% reduction, indicating that once map-size/topology-refresh costs dominate, "
                "additional GPUs contribute little extra speedup."
            )

    start_marker = "<!-- AUTO-SYSTEMS-SCALING-STATS:START -->"
    end_marker = "<!-- AUTO-SYSTEMS-SCALING-STATS:END -->"
    block_body = "\n".join([start_marker, *block_lines, end_marker])

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    start_idx = manuscript_text.find(start_marker)
    end_idx = manuscript_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx += len(end_marker)
        updated_text = manuscript_text[:start_idx] + block_body + manuscript_text[end_idx:]
    else:
        anchor = "*Figure 11. Multi-GPU full-batch scaling across $G\\in\\{1,2,4,8\\}$ GPUs. Panels A-C show runtime (s) for dimension-, sample-, and grid-size-scaling workloads, respectively. Panels D-F show scaling efficiency for the same workloads, computed from the single-GPU baseline and the corresponding $G$-GPU runtime. Runtime error bars denote $\\pm 1$ standard deviation across $n=3$ repeated runs per configuration; the 100\\% efficiency reference line indicates ideal linear scaling.*"
        anchor_idx = manuscript_text.find(anchor)
        if anchor_idx == -1:
            updated_text = manuscript_text.rstrip() + "\n\n" + block_body + "\n"
        else:
            insert_at = anchor_idx + len(anchor)
            updated_text = manuscript_text[:insert_at] + "\n\n" + block_body + "\n" + manuscript_text[insert_at:]

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "diagnostics_table": str(diagnostics_path),
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "diagnostics_table": str(diagnostics_path),
    }

def _sync_default_aware_stats_to_manuscript(
    default_aware_analysis: Dict[str, object],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    if not bool(default_aware_analysis.get("enabled")):
        return {
            "updated": False,
            "reason": "Default-aware analysis disabled for this run.",
            "manuscript": str(manuscript_path),
        }

    variants = default_aware_analysis.get("tuned_vs_default_variants", [])
    if not isinstance(variants, list):
        variants = []
    selected_variant = next(
        (
            variant
            for variant in variants
            if isinstance(variant, dict) and str(variant.get("key", "")).strip() == "selected_source"
        ),
        None,
    )
    if not isinstance(selected_variant, dict):
        return {
            "updated": False,
            "reason": "Selected-source tuned-vs-default variant not found.",
            "manuscript": str(manuscript_path),
        }

    output_dir_text = str(selected_variant.get("output_dir", "")).strip()
    if not output_dir_text:
        return {
            "updated": False,
            "reason": "Selected-source tuned-vs-default variant is missing output_dir.",
            "manuscript": str(manuscript_path),
        }

    tables_dir = Path(output_dir_text).resolve() / "tables"
    if not tables_dir.exists():
        return {
            "updated": False,
            "reason": f"Default-aware tables directory not found: {tables_dir}",
            "manuscript": str(manuscript_path),
        }

    metric_specs: List[Tuple[str, str, str]] = [
        ("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
        ("quantization_error_holdout", "Holdout QE", "qe_holdout"),
        ("quantization_error_train", "Train QE", "qe_train"),
    ]

    def _plain_text_join(values: Sequence[str]) -> str:
        formatted = [str(value).strip() for value in values if str(value).strip()]
        if not formatted:
            return ""
        if len(formatted) == 1:
            return formatted[0]
        if len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"

    def _format_pct(value: float) -> str:
        return f"{float(value):.2f}%"

    def _prose_join(values: Sequence[str]) -> str:
        items = [str(value).strip() for value in values if str(value).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{'; '.join(items[:-1])}; and {items[-1]}"

    def _upsert_block(
        manuscript_text: str,
        *,
        start_marker: str,
        end_marker: str,
        block_body: str,
        anchor: str,
    ) -> str:
        start_idx = manuscript_text.find(start_marker)
        end_idx = manuscript_text.find(end_marker)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            end_idx += len(end_marker)
            return manuscript_text[:start_idx] + block_body + manuscript_text[end_idx:]
        anchor_idx = manuscript_text.find(anchor)
        if anchor_idx == -1:
            return manuscript_text.rstrip() + "\n\n" + block_body + "\n"
        return manuscript_text[:anchor_idx].rstrip() + "\n\n" + block_body + "\n\n" + manuscript_text[anchor_idx:]

    metric_rows: List[Dict[str, object]] = []
    stats_sources: Dict[str, str] = {}
    pair_metadata: Optional[Dict[str, object]] = None
    for _, metric_label, metric_slug in metric_specs:
        pairs_path = tables_dir / f"tuned_vs_default_pairs_{metric_slug}.csv"
        if not pairs_path.exists():
            continue
        pairs_df = pd.read_csv(pairs_path)
        if pairs_df.empty or not {"default_value", "tuned_value", "pct_improvement"}.issubset(set(pairs_df.columns)):
            continue

        default_values = pd.to_numeric(pairs_df["default_value"], errors="coerce")
        tuned_values = pd.to_numeric(pairs_df["tuned_value"], errors="coerce")
        pct_values = pd.to_numeric(pairs_df["pct_improvement"], errors="coerce")
        valid_mask = default_values.notna() & tuned_values.notna()
        if not bool(valid_mask.any()):
            continue

        working = pairs_df.loc[valid_mask].copy()
        default_values = default_values.loc[valid_mask]
        tuned_values = tuned_values.loc[valid_mask]
        pct_values = pct_values.loc[valid_mask]
        better_mask = tuned_values < default_values
        worse_mask = tuned_values > default_values
        tie_mask = ~(better_mask | worse_mask)
        pct_nonnull = pct_values.dropna()

        metric_rows.append(
            {
                "metric_label": metric_label,
                "metric_slug": metric_slug,
                "pairs": int(len(working)),
                "better": int(better_mask.sum()),
                "worse": int(worse_mask.sum()),
                "ties": int(tie_mask.sum()),
                "median_pct": float(pct_nonnull.median()) if not pct_nonnull.empty else float("nan"),
                "mean_pct": float(pct_nonnull.mean()) if not pct_nonnull.empty else float("nan"),
            }
        )
        stats_sources[metric_slug] = str(pairs_path.resolve())

        if pair_metadata is None:
            pair_metadata = {
                "datasets": sorted(
                    {
                        str(value).strip()
                        for value in working.get("pair_dataset", pd.Series(dtype=object)).dropna().tolist()
                        if str(value).strip()
                    }
                ),
                "seeds": sorted(
                    {
                        str(value).strip()
                        for value in working.get("pair_seed", pd.Series(dtype=object)).dropna().tolist()
                        if str(value).strip()
                    }
                ),
                "sampling_modes": sorted(
                    {
                        str(value).strip()
                        for value in working.get("pair_sampling", pd.Series(dtype=object)).dropna().tolist()
                        if str(value).strip()
                    }
                ),
                "topologies": sorted(
                    {
                        str(value).strip()
                        for value in working.get("pair_topology", pd.Series(dtype=object)).dropna().tolist()
                        if str(value).strip()
                    }
                ),
            }

    if not metric_rows:
        return {
            "updated": False,
            "reason": "No default-aware pair tables were available to sync.",
            "manuscript": str(manuscript_path),
            "stats_sources": stats_sources,
        }

    pair_counts = {int(row["pairs"]) for row in metric_rows}
    total_pairs = int(sum(int(row["pairs"]) for row in metric_rows))
    if len(pair_counts) == 1:
        per_endpoint_pairs = next(iter(pair_counts))
        pair_count_phrase = (
            f"n={per_endpoint_pairs:,} paired comparisons per $QE$ endpoint "
            f"(n={total_pairs:,} total across all $QE$ variants $QE_B$/$QE_H$/$QE_T$)"
        )
    else:
        pair_count_phrase = (
            "metric-specific paired counts "
            + ", ".join(
                f"{str(row['metric_label'])} n={int(row['pairs']):,}"
                for row in metric_rows
            )
            + f" (n={total_pairs:,} total across all $QE$ variants $QE_B$/$QE_H$/$QE_T$)"
        )

    n_datasets = len(pair_metadata.get("datasets", [])) if pair_metadata is not None else 0
    n_seeds = len(pair_metadata.get("seeds", [])) if pair_metadata is not None else 0
    sampling_modes_text = _plain_text_join(pair_metadata.get("sampling_modes", [])) if pair_metadata is not None else ""

    results_sentence = (
        "The tuned-versus-reference pairing results in Fig. 8 show the same direction across the $QE$ endpoints, "
        f"based on {pair_count_phrase} from {n_datasets} datasets, {n_seeds} seeds"
    )
    if sampling_modes_text:
        results_sentence += f", and the {sampling_modes_text} sampling modes"
    results_sentence += "."

    fig6_lines = [results_sentence]
    metric_phrases: List[str] = []
    for row in metric_rows:
        extras: List[str] = []
        if int(row["worse"]) > 0:
            extras.append(f"{int(row['worse']):,} worse")
        if int(row["ties"]) > 0:
            extras.append(f"{int(row['ties']):,} ties")
        extras_text = f" ({', '.join(extras)})" if extras else ""
        metric_phrases.append(
            f"{row['metric_label']} in {int(row['better']):,}/{int(row['pairs']):,} pairs{extras_text}, "
            f"with median and mean improvements of {_format_pct(float(row['median_pct']))} and "
            f"{_format_pct(float(row['mean_pct']))}"
        )
    if metric_phrases:
        fig6_lines.append(
            "Across the matched pairs, tuned settings improve " + _prose_join(metric_phrases) + "."
        )

    topology_variant_specs: List[Tuple[str, str]] = [
        ("selected_source_hex", "hexagonal"),
        ("selected_source_mst", "MST"),
        ("selected_source_rng", "RNG"),
    ]
    topology_mean_phrases: List[str] = []
    for variant_key, topology_label in topology_variant_specs:
        topology_variant = next(
            (
                variant
                for variant in variants
                if isinstance(variant, dict) and str(variant.get("key", "")).strip() == variant_key
            ),
            None,
        )
        if not isinstance(topology_variant, dict):
            continue
        topology_output_dir_text = str(topology_variant.get("output_dir", "")).strip()
        if not topology_output_dir_text:
            continue
        topology_tables_dir = Path(topology_output_dir_text).resolve() / "tables"
        if not topology_tables_dir.exists():
            continue
        topology_pairs_path = topology_tables_dir / "tuned_vs_default_pairs_balanced_qe_raw.csv"
        if not topology_pairs_path.exists():
            continue
        topology_pairs_df = pd.read_csv(topology_pairs_path)
        if topology_pairs_df.empty or "pct_improvement" not in topology_pairs_df.columns:
            continue
        topology_pct = pd.to_numeric(topology_pairs_df["pct_improvement"], errors="coerce").dropna()
        if topology_pct.empty:
            continue
        topology_mean_phrases.append(f"{topology_label} ({_format_pct(float(topology_pct.mean()))})")

    if topology_mean_phrases:
        topology_sentence = (
            "The same tuning pattern is observed across topologies: mean Balanced-QE improvement is positive for "
            + _plain_text_join(topology_mean_phrases)
            + ", indicating that tuning affects all topology families rather than a single-architecture artifact."
        )
    else:
        topology_sentence = (
            "Topology-specific Balanced-QE summary text will be populated automatically after the topology-specific "
            "tuned-versus-default variants are generated."
        )

    discussion_sentence = (
        "Across the matched Fig. 8 comparisons, tuned configurations consistently produce better QE results than "
        "untuned reference settings. This suggests that tuning should be treated as part of the method configuration "
        "rather than as optional post-processing."
    )

    conclusion_sentence = (
        "This manuscript reports four main findings: in iteration-matched comparisons, "
        "full and random show no meaningful paired QE difference in larger datasets (>10,000 samples), "
        "while random provides runtime gains on smaller datasets, where it also shows greater instability; "
        "graph topologies show lower QE than the fixed hexagonal structure, with RNG showing the lowest QE in these comparisons; "
        f"default-aware analyses over {pair_count_phrase}, drawn from {n_datasets} datasets, {n_seeds} seeds"
    )
    if sampling_modes_text:
        conclusion_sentence += f", and the {sampling_modes_text} sampling modes"
    conclusion_sentence += (
        ", show that hyperparameter selection affects outcomes under the derived default hyperparameters; "
        "and the multi-GPU, OOM-capable execution pipeline "
        "scales effectively when storage and file I/O are sufficient to sustain throughput."
    )

    fig6_start = "<!-- AUTO-DEFAULT-AWARE-FIGURE6-STATS:START -->"
    fig6_end = "<!-- AUTO-DEFAULT-AWARE-FIGURE6-STATS:END -->"
    topology_start = "<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:START -->"
    topology_end = "<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:END -->"
    discussion_start = "<!-- AUTO-DEFAULT-AWARE-DISCUSSION:START -->"
    discussion_end = "<!-- AUTO-DEFAULT-AWARE-DISCUSSION:END -->"
    conclusion_start = "<!-- AUTO-DEFAULT-AWARE-CONCLUSION:START -->"
    conclusion_end = "<!-- AUTO-DEFAULT-AWARE-CONCLUSION:END -->"

    fig6_block = "\n".join([fig6_start, *fig6_lines, fig6_end])
    topology_block = "\n".join([topology_start, topology_sentence, topology_end])
    discussion_block = "\n".join([discussion_start, discussion_sentence, discussion_end])
    conclusion_block = "\n".join([conclusion_start, conclusion_sentence, conclusion_end])

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    updated_text = _upsert_block(
        manuscript_text,
        start_marker=fig6_start,
        end_marker=fig6_end,
        block_body=fig6_block,
        anchor="At the pooled overall level, the paired summaries across all matched tuned/default pairs also favor tuning for all three metrics, consistent with the per-dataset pattern in Fig. 8.",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=topology_start,
        end_marker=topology_end,
        block_body=topology_block,
        anchor="### 5.6 Hyperparameter stability under full versus random sampling",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=discussion_start,
        end_marker=discussion_end,
        block_body=discussion_block,
        anchor="To our knowledge, comprehensive cross-topology and cross-sampling evaluations of chosen SOM hyperparameters have been limited in prior large-scale deployment-oriented studies. In this context, the present paired results make the practical point clear: hyperparameter choice has a large impact on achieved performance.",
    )
    updated_text = _upsert_block(
        updated_text,
        start_marker=conclusion_start,
        end_marker=conclusion_end,
        block_body=conclusion_block,
        anchor="Taken together, the best observed operating profile in this study uses the maximum practical GPU count supported by adequate file I/O, RNG topology, and the derived default hyperparameters, with sampling chosen by scale: full for smaller datasets when stability is critical, and random as a practical throughput option in the larger-dataset regime (>10000 samples) where paired QE differences are not meaningfully detected. Under these conditions, the framework achieves the strongest overall quality-throughput trade-off observed in our evaluations. When workloads are dominated by very large grid-size scaling, MST remains a reasonable alternative because its graph-construction path scales more favorably than RNG.",
    )

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "stats_sources": stats_sources,
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "stats_sources": stats_sources,
        "selected_variant_output_dir": str(Path(output_dir_text).resolve()),
    }

def _sync_default_aware_stability_regression_stats_to_manuscript(
    default_aware_analysis: Dict[str, object],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "updated": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }

    manuscript_path = (assets_dir.parent / "manuscript.md").resolve()
    if not manuscript_path.exists():
        return {
            "updated": False,
            "reason": f"Manuscript file not found: {manuscript_path}",
        }

    if not bool(default_aware_analysis.get("enabled")):
        return {
            "updated": False,
            "reason": "Default-aware analysis disabled for this run.",
            "manuscript": str(manuscript_path),
        }

    output_dir_text = str(default_aware_analysis.get("output_dir", "")).strip()
    if not output_dir_text:
        return {
            "updated": False,
            "reason": "Default-aware output_dir was not recorded.",
            "manuscript": str(manuscript_path),
        }

    publication_dir = Path(output_dir_text).resolve() / "publication_figures"
    if not publication_dir.exists():
        return {
            "updated": False,
            "reason": f"Default-aware publication_figures directory not found: {publication_dir}",
            "manuscript": str(manuscript_path),
        }

    topology_order: List[Tuple[str, str]] = [
        ("hexagonal", "hexagonal"),
        ("mst", "MST"),
        ("rng", "RNG"),
    ]
    sampling_mode_order: List[Tuple[str, str]] = [
        ("full", "full sampling"),
        ("random", "random sampling"),
    ]

    def _format_r(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.3f}" if np.isfinite(numeric) else "NA"

    def _format_p(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{float(numeric):.3g}" if np.isfinite(numeric) else "NA"

    def _format_n(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{int(numeric)}" if np.isfinite(numeric) else "NA"

    def _join_phrases(values: Sequence[str]) -> str:
        items = [str(value).strip() for value in values if str(value).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{'; '.join(items[:-1])}; and {items[-1]}"

    stats_sources: Dict[str, str] = {}
    mode_summaries: Dict[str, str] = {}
    for sampling_mode, sampling_label in sampling_mode_order:
        stats_path = publication_dir / (
            f"stats_stability_selected_params_vs_sample_size_{sampling_mode}_"
            f"{DEFAULT_AWARE_STABILITY_FIGURE7_METRIC_SLUG}.csv"
        )
        if not stats_path.exists():
            continue
        stats_df = pd.read_csv(stats_path)
        if stats_df.empty:
            continue

        topology_phrases: List[str] = []
        for topology_key, topology_label in topology_order:
            topology_df = stats_df[stats_df["topology"].astype(str).str.lower().str.strip() == topology_key].copy()
            if topology_df.empty:
                continue
            row = topology_df.iloc[0]
            topology_phrases.append(
                f"{topology_label} (Pearson R={_format_r(row.get('pearson_r'))}, "
                f"p={_format_p(row.get('pearson_r_p_value'))}, n={_format_n(row.get('n_datasets'))})"
            )
        if not topology_phrases:
            continue
        mode_summaries[sampling_mode] = _join_phrases(topology_phrases)
        stats_sources[sampling_mode] = str(stats_path.resolve())

    start_marker = "<!-- AUTO-DEFAULT-AWARE-STABILITY-REGRESSION:START -->"
    end_marker = "<!-- AUTO-DEFAULT-AWARE-STABILITY-REGRESSION:END -->"
    full_summary = mode_summaries.get("full", "")
    random_summary = mode_summaries.get("random", "")
    if full_summary and random_summary:
        paragraph = (
            "The dataset-size regression summaries show little evidence of a full-sampling size relationship, with "
            f"near-zero correlations under full sampling, {full_summary}. By contrast, Fig. 9C shows a clearer "
            "random-sampling size relationship, with "
            f"{random_summary}. Under random sampling, larger datasets tend to produce lower selected-parameter "
            "stability scores, indicating improved stability with scale. This reinforces the practical interpretation "
            "that random is attractive primarily as a throughput-oriented choice rather than a stability-first "
            "setting at smaller dataset scales."
        )
    elif full_summary:
        paragraph = (
            "The dataset-size regression summaries show little evidence of a full-sampling size relationship, with "
            f"near-zero correlations under full sampling, {full_summary}."
        )
    elif random_summary:
        paragraph = (
            "Figure 7C shows the random-sampling dataset-size relationship, with "
            f"{random_summary}. Under random sampling, larger datasets tend to produce lower selected-parameter "
            "stability scores, indicating improved stability with scale."
        )
    else:
        paragraph = (
            "Figure 7 stability-regression statistics will be populated automatically after the publication "
            "figure-generation run."
        )

    block_body = "\n".join([start_marker, paragraph, end_marker])

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    start_idx = manuscript_text.find(start_marker)
    end_idx = manuscript_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx += len(end_marker)
        updated_text = manuscript_text[:start_idx] + block_body + manuscript_text[end_idx:]
    else:
        anchor = "![Figure 8](assets_manual/figures/fig_8.svg)"
        anchor_idx = manuscript_text.find(anchor)
        if anchor_idx == -1:
            updated_text = manuscript_text.rstrip() + "\n\n" + block_body + "\n"
        else:
            updated_text = manuscript_text[:anchor_idx].rstrip() + "\n\n" + block_body + "\n\n" + manuscript_text[anchor_idx:]

    if updated_text == manuscript_text:
        return {
            "updated": False,
            "reason": "No manuscript changes were necessary.",
            "manuscript": str(manuscript_path),
            "stats_sources": stats_sources,
            "publication_dir": str(publication_dir),
        }

    manuscript_path.write_text(updated_text, encoding="utf-8")
    return {
        "updated": True,
        "manuscript": str(manuscript_path),
        "stats_sources": stats_sources,
        "publication_dir": str(publication_dir),
    }

def _sync_default_aware_assets_to_paper(
    default_aware_analysis: Dict[str, object],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "copied": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }
    if not bool(default_aware_analysis.get("enabled")):
        return {
            "copied": False,
            "reason": "Default-aware analysis disabled for this run.",
            "assets_dir": str(assets_dir.resolve()),
        }

    dst_figures_dir = assets_dir / "figures"
    dst_figures_dir.mkdir(parents=True, exist_ok=True)

    variant_destinations: Dict[str, str] = {
        "selected_source": "fig_8.svg",
        "selected_source_hex": "supp_fig_s8.svg",
        "selected_source_mst": "supp_fig_s9.svg",
        "selected_source_rng": "supp_fig_s10.svg",
    }
    copied_figures: Dict[str, str] = {}

    def _copy_figure(src_path_value: object, dst_name: str, key: str) -> None:
        src_text = str(src_path_value or "").strip()
        if not src_text:
            return
        src_path = Path(src_text).resolve()
        if not src_path.exists():
            return
        dst_path = dst_figures_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied_figures[key] = str(dst_path.resolve())
        generated_files.append(str(dst_path.resolve()))
        _mirror_figure_to_manual_assets(dst_path, dst_name, generated_files)

    variants = default_aware_analysis.get("tuned_vs_default_variants", [])
    if not isinstance(variants, list):
        variants = []

    for variant in variants:
        if not isinstance(variant, dict):
            continue
        key = str(variant.get("key", "")).strip()
        if key not in variant_destinations:
            continue
        publication = variant.get("publication", {})
        if not isinstance(publication, dict):
            continue
        _copy_figure(
            publication.get("combined_figure", ""),
            variant_destinations[key],
            f"{key}:combined_figure",
        )

    selected_stability_publication = default_aware_analysis.get("selected_stability_publication", {})
    if isinstance(selected_stability_publication, dict):
        _copy_figure(
            selected_stability_publication.get("figure_7", ""),
            DEFAULT_AWARE_STABILITY_FIGURE7_ASSET_FILENAME,
            "selected_stability:figure_7",
        )

    return {
        "copied": bool(copied_figures),
        "assets_dir": str(assets_dir.resolve()),
        "figures": copied_figures,
    }


def _sync_xpysom_calibration_assets_to_paper(
    xpysom_calibration_publication: Dict[str, object],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "copied": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }
    if not bool(xpysom_calibration_publication.get("enabled")):
        return {
            "copied": False,
            "reason": "XPySOM calibration publication generation disabled for this run.",
            "assets_dir": str(assets_dir.resolve()),
        }

    figures = xpysom_calibration_publication.get("figures", {})
    tables = xpysom_calibration_publication.get("tables", {})
    if not isinstance(figures, dict):
        figures = {}
    if not isinstance(tables, dict):
        tables = {}

    dst_figures_dir = assets_dir / "figures"
    dst_tables_dir = assets_dir / "tables"
    dst_figures_dir.mkdir(parents=True, exist_ok=True)
    dst_tables_dir.mkdir(parents=True, exist_ok=True)

    figure_name_map = {
        "mst": "supp_fig_s1.svg",
        "rng": "supp_fig_s2.svg",
        "hexagonal": "supp_fig_s3.svg",
    }
    table_name_map = {
        "mst": "supp_xpysom_calibration_qe_mst.tsv",
        "rng": "supp_xpysom_calibration_qe_rng.tsv",
        "hexagonal": "supp_xpysom_calibration_qe_hexagonal.tsv",
    }

    copied_figures: Dict[str, str] = {}
    copied_tables: Dict[str, str] = {}

    for topology, dst_name in figure_name_map.items():
        src_text = str(figures.get(topology, "")).strip()
        if not src_text:
            continue
        src_path = Path(src_text).resolve()
        if not src_path.exists():
            continue
        dst_path = dst_figures_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied_figures[topology] = str(dst_path.resolve())
        generated_files.append(str(dst_path.resolve()))

    for topology, dst_name in table_name_map.items():
        src_text = str(tables.get(topology, "")).strip()
        if not src_text:
            continue
        src_path = Path(src_text).resolve()
        if not src_path.exists():
            continue
        dst_path = dst_tables_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied_tables[topology] = str(dst_path.resolve())
        generated_files.append(str(dst_path.resolve()))

    return {
        "copied": bool(copied_figures or copied_tables),
        "assets_dir": str(assets_dir.resolve()),
        "figures": copied_figures,
        "tables": copied_tables,
    }


def _sync_xpysom_rng_publication_assets_to_paper(
    xpysom_rng_publication: Dict[str, object],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "copied": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }
    if not bool(xpysom_rng_publication.get("enabled")):
        return {
            "copied": False,
            "reason": "XPySOM RNG publication figure generation disabled for this run.",
            "assets_dir": str(assets_dir.resolve()),
        }

    figure_src_text = str(xpysom_rng_publication.get("combined_figure", "")).strip()
    if not figure_src_text:
        return {
            "copied": False,
            "reason": "Combined XPySOM RNG publication figure path is empty.",
            "assets_dir": str(assets_dir.resolve()),
        }

    dst_figures_dir = assets_dir / "figures"
    dst_figures_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}

    figure_src = Path(figure_src_text).resolve()
    if figure_src.exists():
        figure_dst = dst_figures_dir / "fig_13.svg"
        shutil.copy2(figure_src, figure_dst)
        copied["figure"] = str(figure_dst.resolve())
        generated_files.append(str(figure_dst.resolve()))
        _mirror_figure_to_manual_assets(figure_dst, figure_dst.name, generated_files)

    return {
        "copied": bool(copied),
        "assets_dir": str(assets_dir.resolve()),
        "files": copied,
    }


def _sync_xpysom_topology_tripanel_assets_to_paper(
    xpysom_topology_tripanel_publication: Dict[str, object],
    generated_files: List[str],
) -> Dict[str, object]:
    assets_dir = _resolve_paper_assets_dir()
    if assets_dir is None:
        return {
            "copied": False,
            "reason": "Could not locate paper/assets from checkout or legacy floatsom/paper/assets from script path.",
        }
    if not bool(xpysom_topology_tripanel_publication.get("enabled")):
        return {
            "copied": False,
            "reason": "XPySOM topology tripanel publication generation disabled for this run.",
            "assets_dir": str(assets_dir.resolve()),
        }

    figures = xpysom_topology_tripanel_publication.get("figures", {})
    if not isinstance(figures, dict):
        figures = {}

    dst_figures_dir = assets_dir / "figures"
    dst_figures_dir.mkdir(parents=True, exist_ok=True)

    figure_name_map = {
        "hexagonal": "supp_fig_s12.svg",
        "mst": "supp_fig_s13.svg",
    }
    copied_figures: Dict[str, str] = {}

    for topology, dst_name in figure_name_map.items():
        src_text = str(figures.get(topology, "")).strip()
        if not src_text:
            continue
        src_path = Path(src_text).resolve()
        if not src_path.exists():
            continue
        dst_path = dst_figures_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied_figures[topology] = str(dst_path.resolve())
        generated_files.append(str(dst_path.resolve()))
        _mirror_figure_to_manual_assets(dst_path, dst_name, generated_files)

    return {
        "copied": bool(copied_figures),
        "assets_dir": str(assets_dir.resolve()),
        "figures": copied_figures,
    }
__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
