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

from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups

from .constants import *
from .helpers import *

def _scaled_font(size: float) -> float:
    return float(size) * TEXT_SIZE_SCALE

def _scaled_marker(size: float) -> float:
    return float(size) * MARKER_SIZE_SCALE

def _scaled_scatter(size: float) -> float:
    return float(size) * SCATTER_SIZE_SCALE

def _set_plot_theme(style: str = "whitegrid", context: str = "talk") -> None:
    sns.set_theme(
        style=style,
        context=context,
        font_scale=TEXT_SIZE_SCALE,
        rc={
            "axes.labelpad": AXIS_LABEL_PADDING,
            "grid.color": "black",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
        },
    )

def _apply_optional_titles(
    fig: Any,
    ax: Any,
    title: str,
    subtitle: str,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if not show_titles:
        return
    fig.suptitle(title, fontsize=_scaled_font(16), x=0.01, ha="left", y=0.995)
    ax.set_title(subtitle, fontsize=_scaled_font(11.5), loc="left", pad=8, color="#404040")

def _publication_left_margin(*, show_y_tick_labels: bool, left_margin_override: Optional[float]) -> float:
    if left_margin_override is not None:
        return float(left_margin_override)
    return PUBLICATION_SAFE_LEFT_MARGIN if show_y_tick_labels else PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS

def _publication_panel_margins(
    *,
    inline_legends: bool,
    show_titles: bool,
    show_y_tick_labels: bool,
    x_axis_label: str,
    left_margin_override: Optional[float],
) -> Dict[str, float]:
    # Keep x-tick labels visible even when a shared composite x-label is used
    # and per-panel x_axis_label is intentionally blank.
    has_x_axis_label = bool(str(x_axis_label).strip())
    bottom = 0.14 if inline_legends or has_x_axis_label else 0.08
    return {
        "left": _publication_left_margin(
            show_y_tick_labels=show_y_tick_labels,
            left_margin_override=left_margin_override,
        ),
        "right": 0.98 if inline_legends else 0.996,
        "bottom": bottom,
        "top": 0.92 if show_titles else 0.955,
    }

def _publication_panel_tight_rect(*, inline_legends: bool, show_titles: bool, x_axis_label: str) -> List[float]:
    has_x_axis_label = bool(str(x_axis_label).strip())
    if inline_legends:
        return [0.0, 0.21, 0.93, 0.90]
    return [
        0.0,
        0.14 if has_x_axis_label else 0.08,
        0.996,
        0.90 if show_titles else 0.955,
    ]

def _tripanel_forest_panel_style(*, show_dataset_labels: bool) -> Dict[str, float]:
    return {
        "x_axis_label_font_size": TRIPANEL_FOREST_X_AXIS_LABEL_FONT_SIZE,
        "x_tick_label_font_size": TRIPANEL_FOREST_X_TICK_LABEL_FONT_SIZE,
        "y_tick_label_font_size": (
            TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE
            if show_dataset_labels
            else TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE_NO_LABELS
        ),
        "left_margin_override": (
            PUBLICATION_SAFE_LEFT_MARGIN
            if show_dataset_labels
            else PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS
        ),
        "figure_width_override": (
            PUBLICATION_PANEL_FIG_WIDTH_WITH_Y_LABELS
            if show_dataset_labels
            else PUBLICATION_PANEL_FIG_WIDTH
        ),
    }

def _tripanel_topology_strata_text_style(*, show_dataset_labels: bool) -> Dict[str, float]:
    return {
        "x_axis_label_font_size": TRIPANEL_FOREST_X_AXIS_LABEL_FONT_SIZE,
        "x_tick_label_font_size": TRIPANEL_FOREST_X_TICK_LABEL_FONT_SIZE,
        "y_tick_label_font_size": (
            TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE
            if show_dataset_labels
            else TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE_NO_LABELS
        ),
    }

def _legend_registry_add_items(
    legend_registry: Dict[str, Dict[str, Any]],
    section_id: str,
    section_title: str,
    items: Sequence[Tuple[str, Dict[str, Any]]],
) -> None:
    section = legend_registry.setdefault(
        section_id,
        {
            "title": section_title,
            "items": OrderedDict(),
        },
    )
    if not section.get("title"):
        section["title"] = section_title

    section_items = section["items"]
    for kind, kwargs in items:
        label = str(kwargs.get("label", "")).strip()
        if not label or label in section_items:
            continue
        section_items[label] = {
            "kind": str(kind),
            "kwargs": dict(kwargs),
        }

def _publication_forest_legend_item_specs(*, alpha: float) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markersize": 11,
                "label": f"Significant (q < {alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markersize": 11,
                "label": f"Non-Significant (q >= {alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "marker": "D",
                "color": "none",
                "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markersize": 10,
                "label": "Global Summary",
            },
        ),
        (
            "line2d",
            {
                "color": "#4d4d4d",
                "linewidth": 2.8,
                "label": "95% CI Paired t-Test",
            },
        ),
        (
            "line2d",
            {
                "marker": "|",
                "linestyle": "None",
                "color": "black",
                "markersize": 18,
                "markeredgewidth": 2.2,
                "label": "Zero Effect",
            },
        ),
    ]

COMPOSITE_LEGEND_VISUAL_SCALE = 1.0


def _build_legend_handle_from_spec(spec: Dict[str, Any], *, scale: float = 1.0) -> Any:
    kind = str(spec.get("kind", "line2d")).lower().strip()
    kwargs = dict(spec.get("kwargs", {}))
    if "markersize" in kwargs and kwargs["markersize"] is not None:
        kwargs["markersize"] = _scaled_marker(float(kwargs["markersize"]) * float(scale))
    if "linewidth" in kwargs and kwargs["linewidth"] is not None:
        kwargs["linewidth"] = float(kwargs["linewidth"]) * 1.2 * float(scale)
    if kind == "patch":
        return Patch(**kwargs)
    return Line2D([0], [0], **kwargs)

def _render_legend_key(
    legend_registry: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int,
    *,
    compact: bool = False,
    font_scale_override: float = 1.0,
) -> bool:
    sections = [
        section
        for section in legend_registry.values()
        if section.get("items")
    ]
    if not sections:
        return False

    n_sections = len(sections)
    legend_font_scale = max(0.1, float(font_scale_override)) * COMPOSITE_LEGEND_VISUAL_SCALE
    if n_sections == 1:
        section_specs = list(sections[0]["items"].values())
        legend_handles = [
            _build_legend_handle_from_spec(spec, scale=COMPOSITE_LEGEND_VISUAL_SCALE)
            for spec in section_specs
        ]
        legend_labels = [str(spec.get("kwargs", {}).get("label", "")).strip() for spec in section_specs]
        n_items = max(1, len(legend_handles))
        ncol = min(5, n_items)
        nrows = int(np.ceil(n_items / ncol))
        fig_width = max(
            13.0 * COMPOSITE_LEGEND_VISUAL_SCALE,
            (4.0 * ncol) * COMPOSITE_LEGEND_VISUAL_SCALE,
        )
        fig_height = max(
            2.2 * COMPOSITE_LEGEND_VISUAL_SCALE,
            (1.0 * nrows + 0.9) * COMPOSITE_LEGEND_VISUAL_SCALE,
        )
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.axis("off")

        legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="center",
            ncol=ncol,
            frameon=False,
            fontsize=_scaled_font(12.0 * legend_font_scale),
            borderpad=1.0 * COMPOSITE_LEGEND_VISUAL_SCALE,
            labelspacing=0.85 * COMPOSITE_LEGEND_VISUAL_SCALE,
            handletextpad=0.8 * COMPOSITE_LEGEND_VISUAL_SCALE,
            handlelength=2.4 * COMPOSITE_LEGEND_VISUAL_SCALE,
            columnspacing=2.0 * COMPOSITE_LEGEND_VISUAL_SCALE,
        )
    else:
        fig_width = max(22.0 * COMPOSITE_LEGEND_VISUAL_SCALE, (11.0 * n_sections) * COMPOSITE_LEGEND_VISUAL_SCALE)
        fig_height = max(2.1 * COMPOSITE_LEGEND_VISUAL_SCALE, (1.2 + 0.45 * n_sections) * COMPOSITE_LEGEND_VISUAL_SCALE)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.axis("off")

        def _blank_handle() -> Line2D:
            return Line2D([], [], linestyle="None", marker="", color="none")

        y_positions = np.linspace(0.72, 0.28, num=n_sections)
        for section_idx, section in enumerate(sections):
            section_specs = list(section["items"].values())
            section_handles: List[Any] = [_blank_handle()]
            section_labels: List[str] = [str(section.get("title", "")).strip()]
            for spec in section_specs:
                section_handles.append(
                    _build_legend_handle_from_spec(spec, scale=COMPOSITE_LEGEND_VISUAL_SCALE)
                )
                section_labels.append(str(spec.get("kwargs", {}).get("label", "")).strip())

            legend = fig.legend(
                handles=section_handles,
                labels=section_labels,
                loc="center",
                bbox_to_anchor=(0.5, float(y_positions[section_idx])),
                ncol=max(1, len(section_handles)),
                frameon=False,
                fontsize=_scaled_font(11.0 * legend_font_scale),
                borderpad=0.35 * COMPOSITE_LEGEND_VISUAL_SCALE,
                labelspacing=0.55 * COMPOSITE_LEGEND_VISUAL_SCALE,
                handletextpad=0.65 * COMPOSITE_LEGEND_VISUAL_SCALE,
                handlelength=2.2 * COMPOSITE_LEGEND_VISUAL_SCALE,
                columnspacing=1.35 * COMPOSITE_LEGEND_VISUAL_SCALE,
            )
            if legend is not None:
                texts = legend.get_texts()
                if texts:
                    texts[0].set_fontweight("bold")
                    texts[0].set_fontsize(_scaled_font(11.8 * legend_font_scale))
                    texts[0].set_color("#1f1f1f")

    fig.tight_layout(pad=0.5 * COMPOSITE_LEGEND_VISUAL_SCALE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=(0.06 if compact else 0.16) * COMPOSITE_LEGEND_VISUAL_SCALE,
    )
    plt.close(fig)
    return True

def _render_composite_legend_key(
    output_path: Path,
    *,
    dpi: int,
    alpha: float,
    include_forest: bool,
    include_sensitivity: bool,
    sensitivity_color: Any = METHOD_BASE_COLORS["colors"],
    sensitivity_series_specs: Optional[Sequence[Tuple[str, Any, str]]] = None,
    sensitivity_dotted_bounds: bool = False,
    compact: bool = False,
    font_scale_override: float = 1.0,
) -> Optional[Path]:
    legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()

    if include_forest:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="forest_semantics",
            section_title="Forest Plot Semantics",
            items=_publication_forest_legend_item_specs(alpha=alpha),
        )

    if include_sensitivity:
        sensitivity_items: List[Tuple[str, Dict[str, Any]]] = []
        if sensitivity_series_specs:
            for series_label, series_color, series_marker in sensitivity_series_specs:
                sensitivity_items.append(
                    (
                        "line2d",
                        {
                            "color": series_color,
                            "marker": series_marker,
                            "markersize": 10,
                            "linewidth": 3.0,
                            "label": str(series_label),
                        },
                    )
                )
        else:
            sensitivity_items.append(
                (
                    "line2d",
                    {
                        "color": sensitivity_color,
                        "marker": "o",
                        "markersize": 10,
                        "linewidth": 3.0,
                        "label": "Paired mean improvement",
                    },
                )
            )

        if sensitivity_dotted_bounds:
            sensitivity_items.append(
                (
                    "line2d",
                    {
                        "color": "#4f4f4f",
                        "linestyle": ":",
                        "linewidth": 2.2,
                        "label": "Dotted bounds: 95% CI",
                    },
                )
            )
        else:
            sensitivity_items.append(
                (
                    "patch",
                    {
                        "facecolor": sensitivity_color,
                        "edgecolor": sensitivity_color,
                        "alpha": 0.18,
                        "label": "95% CI",
                    },
                )
            )

        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sensitivity_semantics",
            section_title="Sensitivity Curve Semantics",
            items=sensitivity_items,
        )

    if _render_legend_key(
        legend_registry=legend_registry,
        output_path=output_path,
        dpi=dpi,
        compact=compact,
        font_scale_override=font_scale_override,
    ):
        return output_path
    return None

def _render_sample_size_regression_legend_key(
    output_path: Path,
    *,
    dpi: int,
    compact: bool = True,
    font_scale_override: float = 1.0,
) -> Optional[Path]:
    legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    _legend_registry_add_items(
        legend_registry=legend_registry,
        section_id="sample_size_regression_semantics",
        section_title="Dataset-Size Regression Semantics",
        items=[
            (
                "line2d",
                {
                    "marker": "o",
                    "color": "none",
                    "markerfacecolor": "#1f77b4",
                    "markeredgecolor": "#1f77b4",
                    "markersize": 10,
                    "label": "Dataset median QE difference",
                },
            ),
            (
                "line2d",
                {
                    "color": "#2ca02c",
                    "linewidth": 2.6,
                    "label": "Fitted trend",
                },
            ),
        ],
    )
    if _render_legend_key(
        legend_registry=legend_registry,
        output_path=output_path,
        dpi=dpi,
        compact=compact,
        font_scale_override=font_scale_override,
    ):
        return output_path
    return None

def _plot_dataset_forest(
    summary_df: pd.DataFrame,
    dataset_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
    stats_table_output_path: Optional[Path] = None,
    x_axis_label: str = "Mean % Improvement (positive means first group is better)",
    show_y_tick_labels: bool = True,
    x_axis_label_font_size: Optional[float] = None,
    x_tick_label_font_size: Optional[float] = None,
    y_tick_label_font_size: Optional[float] = None,
    left_margin_override: Optional[float] = None,
    figure_width_override: Optional[float] = None,
    use_tight_bbox: bool = True,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    order = _ordered_dataset_labels(plot_df[dataset_col].astype(str).tolist())
    order_map = {name: idx for idx, name in enumerate(order)}
    plot_df["plot_order"] = plot_df[dataset_col].map(order_map)
    plot_df = plot_df.sort_values("plot_order", ascending=True).reset_index(drop=True)
    plot_df["y"] = np.arange(len(plot_df))[::-1]
    plot_df["is_global"] = plot_df[dataset_col].map(_is_global_dataset_label)
    plot_df["is_significant"] = plot_df.apply(
        lambda row: _row_is_significant(row, alpha=alpha),
        axis=1,
    )
    plot_df["sig_color"] = np.where(plot_df["is_significant"], "#d7301f", "#7a7a7a")

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(6, 0.65 * len(plot_df) + 2.0)
    fig_width = (
        float(figure_width_override)
        if figure_width_override is not None
        else PUBLICATION_PANEL_FIG_WIDTH
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axvline(0.0, color="black", linestyle="-", linewidth=2.2, alpha=0.98, zorder=0)
    y_positions = {name: float(len(order) - 1 - idx) for idx, name in enumerate(order)}
    _draw_dataset_group_separators(ax, order, y_positions)

    for _, row in plot_df.iterrows():
        ax.hlines(
            y=row["y"],
            xmin=row["ci_low_pct"],
            xmax=row["ci_high_pct"],
            color=row["sig_color"],
            linewidth=2.6,
            alpha=0.9,
        )

    scatter_palette = {True: "#d7301f", False: "#7a7a7a"}
    nonglobal = plot_df[~plot_df["is_global"]]
    if not nonglobal.empty:
        sns.scatterplot(
            data=nonglobal,
            x="median_pct",
            y="y",
            hue="is_significant",
            palette=scatter_palette,
            s=_scaled_scatter(124),
            legend=False,
            ax=ax,
        )
        for _, row in nonglobal.iterrows():
            if not bool(row["is_significant"]):
                continue
            stars = _pvalue_to_stars(_row_significance_value(row))
            if stars == "ns":
                continue
            ax.annotate(
                stars,
                (float(row["median_pct"]), float(row["y"])),
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=_scaled_font(13.0),
                fontweight="bold",
                ha="center",
                va="bottom",
                color=row["sig_color"],
                clip_on=False,
            )

    global_rows = plot_df[plot_df["is_global"]]
    if not global_rows.empty:
        for _, row in global_rows.iterrows():
            ax.scatter(
                row["median_pct"],
                row["y"],
                color=row["sig_color"],
                s=_scaled_scatter(156),
                marker="D",
                zorder=3,
            )
            if bool(row["is_significant"]):
                stars = _pvalue_to_stars(_row_significance_value(row))
                if stars != "ns":
                    ax.annotate(
                        stars,
                        (float(row["median_pct"]), float(row["y"])),
                        xytext=(0, 9),
                        textcoords="offset points",
                        fontsize=_scaled_font(13.0),
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                        color=row["sig_color"],
                        clip_on=False,
                    )

    x_min = float(np.nanmin(plot_df[["ci_low_pct", "median_pct"]].to_numpy(dtype=float)))
    x_max = float(np.nanmax(plot_df[["ci_high_pct", "median_pct"]].to_numpy(dtype=float)))
    span = max(1.0, x_max - x_min)
    left = x_min - 0.08 * span
    right = x_max + PUBLICATION_XLIM_RIGHT_PAD_FRACTION * span
    ax.set_xlim(left, right)

    ax.set_yticks(plot_df["y"])
    if show_y_tick_labels:
        ax.set_yticklabels(plot_df[dataset_col].astype(str))
    else:
        ax.set_yticklabels([""] * len(plot_df))
    if x_axis_label_font_size is not None:
        ax.set_xlabel(x_axis_label, fontsize=_scaled_font(x_axis_label_font_size))
    else:
        ax.set_xlabel(x_axis_label)
    if x_tick_label_font_size is not None:
        ax.tick_params(axis="x", labelsize=_scaled_font(x_tick_label_font_size))
    if y_tick_label_font_size is not None:
        ax.tick_params(axis="y", labelsize=_scaled_font(y_tick_label_font_size))
    ax.set_ylabel("")
    ax.grid(axis="x", color="black", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", color="black", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    legend_item_specs = _publication_forest_legend_item_specs(alpha=alpha)
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.00),
            ncol=3,
            frameon=True,
            fontsize=_scaled_font(9.5),
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="forest_semantics",
            section_title="Forest Plot Semantics",
            items=legend_item_specs,
        )

    panel_margins = _publication_panel_margins(
        inline_legends=inline_legends,
        show_titles=show_titles,
        show_y_tick_labels=show_y_tick_labels,
        x_axis_label=x_axis_label,
        left_margin_override=left_margin_override,
    )
    fig.subplots_adjust(
        left=panel_margins["left"],
        right=panel_margins["right"],
        bottom=panel_margins["bottom"],
        top=panel_margins["top"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.2}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
    _write_per_row_stats_table(summary_df, dataset_col=dataset_col, output_path=stats_table_output_path)

def _plot_topology_algorithm_strata(
    strata_df: pd.DataFrame,
    dataset_col: str,
    algorithm_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    x_axis_label: str = "Mean % Improvement (positive means first group is better)",
    show_y_tick_labels: bool = True,
    left_margin_override: Optional[float] = None,
    use_tight_bbox: bool = True,
    x_axis_label_font_size: Optional[float] = None,
    x_tick_label_font_size: Optional[float] = None,
    y_tick_label_font_size: Optional[float] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if strata_df.empty:
        return

    plot_df = strata_df.copy()
    dataset_order = _ordered_dataset_labels(plot_df[dataset_col].astype(str).tolist())
    dataset_map = {name: idx for idx, name in enumerate(dataset_order)}
    plot_df["base_y"] = plot_df[dataset_col].map(dataset_map).astype(float)

    algorithms = sorted(plot_df[algorithm_col].astype(str).unique())
    if len(algorithms) <= 1:
        # Keep single-stratum rows exactly centered on the dataset tick.
        offset_map = {str(algorithms[0]): 0.0} if algorithms else {}
    else:
        offsets = np.linspace(-0.18, 0.18, num=len(algorithms))
        offset_map = {str(algorithms[i]): float(offsets[i]) for i in range(len(algorithms))}
    plot_df["y"] = plot_df["base_y"] + plot_df[algorithm_col].map(offset_map)

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(6, 0.65 * len(dataset_order) + 2.0)
    fig, ax = plt.subplots(figsize=(PUBLICATION_PANEL_FIG_WIDTH, fig_height))
    ax.axvline(0.0, color="black", linestyle="-", linewidth=2.2, alpha=0.98, zorder=0)
    _draw_dataset_group_separators(ax, dataset_order, dataset_map)

    marker_cycle: Tuple[str, ...] = ("o", "s", "^", "v", "P", "X", "<", ">", "h", "8")
    marker_map = {
        str(algorithm): marker_cycle[idx % len(marker_cycle)]
        for idx, algorithm in enumerate(algorithms)
    }

    for _, row in plot_df.iterrows():
        is_significant = _row_is_significant(row, alpha=alpha)
        color = SAMPLING_SIGNIFICANT_COLOR if is_significant else SAMPLING_NON_SIGNIFICANT_COLOR
        ax.hlines(
            y=row["y"],
            xmin=row["ci_low_pct"],
            xmax=row["ci_high_pct"],
            color=color,
            linewidth=2.34,
            alpha=0.9,
            zorder=1,
        )
        is_global = _is_global_dataset_label(row.get(dataset_col))
        marker_style = "D" if is_global else marker_map.get(str(row[algorithm_col]), "o")
        marker_size = _scaled_scatter(114) if not is_global else _scaled_scatter(153)
        ax.scatter(
            float(row["median_pct"]),
            float(row["y"]),
            color=color,
            edgecolors=_darken_color(color, factor=0.70),
            linewidths=1.05,
            marker=marker_style,
            s=marker_size,
            zorder=2,
        )

    algorithm_item_specs = [
        (
            "line2d",
            {
                "marker": marker_map.get(str(algorithm), "o"),
                "linestyle": "none",
                "color": "#3f3f3f",
                "markerfacecolor": "white",
                "markeredgecolor": "#3f3f3f",
                "markeredgewidth": 1.0,
                "markersize": 7.2,
                "label": str(algorithm),
            },
        )
        for algorithm in algorithms
    ]

    x_min = float(np.nanmin(plot_df[["ci_low_pct", "median_pct"]].to_numpy(dtype=float)))
    x_max = float(np.nanmax(plot_df[["ci_high_pct", "median_pct"]].to_numpy(dtype=float)))
    span = max(1.0, x_max - x_min)
    for _, row in plot_df.iterrows():
        if not _row_is_significant(row, alpha=alpha):
            continue
        stars = _pvalue_to_stars(_row_significance_value(row))
        if stars == "ns":
            continue
        ax.annotate(
            stars,
            (float(row["median_pct"]), float(row["y"])),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=_scaled_font(12.5),
            fontweight="bold",
            ha="center",
            va="bottom",
            color=SAMPLING_SIGNIFICANT_COLOR,
            clip_on=False,
        )

    ax.set_xlim(x_min - 0.08 * span, x_max + PUBLICATION_XLIM_RIGHT_PAD_FRACTION * span)
    ax.set_yticks([dataset_map[name] for name in dataset_order])
    if show_y_tick_labels:
        ax.set_yticklabels(dataset_order)
    else:
        ax.set_yticklabels([""] * len(dataset_order))
    # Match Figure 2 ordering convention (alphabetical datasets from top to bottom).
    ax.invert_yaxis()
    if x_axis_label_font_size is not None:
        ax.set_xlabel(x_axis_label, fontsize=_scaled_font(x_axis_label_font_size))
    else:
        ax.set_xlabel(x_axis_label, fontsize=_scaled_font(16.0))
    ax.set_ylabel("")
    if x_tick_label_font_size is not None:
        ax.tick_params(axis="x", labelsize=_scaled_font(x_tick_label_font_size))
    else:
        ax.tick_params(axis="x", labelsize=_scaled_font(13.8))
    if y_tick_label_font_size is not None:
        ax.tick_params(axis="y", labelsize=_scaled_font(y_tick_label_font_size))
    else:
        ax.tick_params(axis="y", labelsize=_scaled_font(15.0))
    ax.grid(axis="x", color="black", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", color="black", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    significance_item_specs = [
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Significant (q/p<{alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Non-significant (q/p>={alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "D",
                "color": "#4f4f4f",
                "markerfacecolor": "#4f4f4f",
                "markeredgecolor": "#4f4f4f",
                "markersize": 7.2,
                "label": "GLOBAL group summary",
            },
        ),
    ]

    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in significance_item_specs],
            title="Significance",
            frameon=True,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            fontsize=_scaled_font(8.5),
        )
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in algorithm_item_specs],
            title="Stratum Markers",
            frameon=True,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.00),
            ncol=min(4, max(1, len(algorithm_item_specs))),
            fontsize=_scaled_font(8.5),
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="topology_strata_significance",
            section_title="Strata Significance",
            items=significance_item_specs,
        )
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="topology_strata_markers",
            section_title="Stratum Markers",
            items=algorithm_item_specs,
        )

    tight_rect = _publication_panel_tight_rect(
        inline_legends=inline_legends,
        show_titles=show_titles,
        x_axis_label=x_axis_label,
    )
    fig.tight_layout(rect=tight_rect)
    panel_margins = _publication_panel_margins(
        inline_legends=inline_legends,
        show_titles=show_titles,
        show_y_tick_labels=show_y_tick_labels,
        x_axis_label=x_axis_label,
        left_margin_override=left_margin_override,
    )
    fig.subplots_adjust(
        left=panel_margins["left"],
        right=panel_margins["right"],
        bottom=panel_margins["bottom"],
        top=panel_margins["top"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.2}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

def _plot_hex_dumbbell(
    pairs: pd.DataFrame,
    metric_label: str,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if pairs.empty:
        return

    plot_df = pairs.copy()
    plot_df["pair_id"] = _pair_identifier(plot_df)
    datasets = sorted(plot_df["dataset"].astype(str).unique())
    n_datasets = len(datasets)

    ncols = 2
    nrows = int(np.ceil(n_datasets / ncols))

    _set_plot_theme(style="whitegrid", context="talk")
    # Give each dataset panel enough vertical room for paired-unit lines.
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.5, max(10.0, 3.6 * nrows)),
        sharex=False,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, dataset in enumerate(datasets):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        ds = plot_df[plot_df["dataset"] == dataset].copy()
        ds = ds.sort_values("score_b", ascending=True).reset_index(drop=True)
        ds["y"] = np.arange(len(ds))

        for _, r in ds.iterrows():
            ax.hlines(
                y=r["y"],
                xmin=min(r["score_b"], r["score_a"]),
                xmax=max(r["score_b"], r["score_a"]),
                color="#9b9b9b",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.scatter(ds["score_b"], ds["y"], s=_scaled_scatter(34), color=METHOD_BASE_COLORS["batch"], alpha=0.9, label="Batch")
        ax.scatter(ds["score_a"], ds["y"], s=_scaled_scatter(34), color=METHOD_BASE_COLORS["colors"], alpha=0.9, label="Colors")
        ax.set_title(f"{dataset} (n={len(ds)})", fontsize=_scaled_font(10.5), loc="left")
        ax.set_xlabel(metric_label)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_axis_tick(x)))

    total_axes = nrows * ncols
    for idx in range(n_datasets, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    legend_item_specs = [
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": METHOD_BASE_COLORS["batch"],
                "markeredgecolor": METHOD_BASE_COLORS["batch"],
                "markersize": 7,
                "label": "Batch",
            },
        ),
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": METHOD_BASE_COLORS["colors"],
                "markeredgecolor": METHOD_BASE_COLORS["colors"],
                "markersize": 7,
                "label": "Colors",
            },
        ),
        (
            "line2d",
            {
                "color": "#9b9b9b",
                "linewidth": 1.5,
                "label": "Paired unit line",
            },
        ),
    ]
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=True,
            fontsize=_scaled_font(9.5),
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="dumbbell_semantics",
            section_title="Dumbbell Plot Semantics",
            items=legend_item_specs,
        )

    if show_titles:
        fig.suptitle(title, fontsize=_scaled_font(16), x=0.01, ha="left", y=0.996)
        fig.text(0.01, 0.965, subtitle, fontsize=_scaled_font(11), color="#404040", ha="left")
    # Larger hspace prevents row overlap when many datasets are present.
    tight_rect = [0.0, 0.10, 1.0, 0.94] if inline_legends else [0.0, 0.05, 1.0, 0.94]
    fig.tight_layout(rect=tight_rect, h_pad=1.4, w_pad=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def _plot_dataset_percent_beaten(
    dataset_df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if dataset_df.empty:
        return

    plot_df = dataset_df.copy()
    plot_df["dataset"] = plot_df["dataset"].astype(str)
    plot_df["dataset_group"] = plot_df["dataset"].map(_dataset_group_key)
    plot_df["group_order"] = plot_df["dataset_group"].map(_dataset_group_rank)
    plot_df = plot_df.sort_values(
        ["group_order", "percent_beaten", "dataset"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(5.8, 0.55 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    ax.barh(
        y=np.arange(len(plot_df)),
        width=plot_df["percent_beaten"].to_numpy(dtype=float),
        color="#4C78A8",
        alpha=0.85,
    )
    y_positions = {
        str(dataset): float(idx)
        for idx, dataset in enumerate(plot_df["dataset"].astype(str).tolist())
    }
    _draw_dataset_group_separators(ax, plot_df["dataset"].astype(str).tolist(), y_positions, zorder=1.2)
    ax.axvline(50.0, color="#505050", linestyle="--", linewidth=1.1, alpha=0.85)

    for idx, row in plot_df.iterrows():
        x = float(row["percent_beaten"])
        n_pairs = int(row["n_pairs"])
        label_x = min(98.0, x + 1.2)
        ax.text(
            label_x,
            float(idx),
            f"{x:.1f}% (n={n_pairs})",
            va="center",
            ha="left",
            fontsize=_scaled_font(9.8),
            color="#303030",
        )

    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["dataset"])
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Percent of paired units where Colors beats Batch")
    ax.set_ylabel("")
    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

def _plot_heatmap(
    heat_df: pd.DataFrame,
    index_col: str,
    column_col: str,
    value_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    vmin: float = 0.0,
    vmax: float = 100.0,
    cmap: str = "RdYlBu_r",
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if heat_df.empty:
        return

    matrix = heat_df.pivot(index=index_col, columns=column_col, values=value_col)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    _set_plot_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.6 * matrix.shape[0] + 2)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="#f0f0f0",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Win rate (%)"},
    )

    ax.set_xlabel(column_col.replace("_", " ").title())
    if str(index_col).lower() == "dataset":
        ax.set_ylabel("")
    else:
        ax.set_ylabel(index_col.replace("_", " ").title())
    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def _plot_sensitivity_curve(
    sensitivity_df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    series_color: Any = DEFAULT_SERIES_COLOR,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    x_axis_label: str = "Top-k trials used per paired unit",
    y_axis_label: str = "Global dataset-average paired mean % improvement",
    show_y_axis_label: bool = True,
    use_tight_bbox: bool = True,
    left_margin_override: Optional[float] = None,
    bottom_margin_override: Optional[float] = None,
    figure_width_override: Optional[float] = None,
    axes_width_scale_override: Optional[float] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if sensitivity_df.empty:
        return

    plot_df = sensitivity_df.copy().sort_values("top_k")

    _set_plot_theme(style="whitegrid", context="talk")
    fig_width = float(figure_width_override) if figure_width_override is not None else 9.5
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    ax.fill_between(
        plot_df["top_k"].to_numpy(dtype=float),
        plot_df["ci_low_pct"].to_numpy(dtype=float),
        plot_df["ci_high_pct"].to_numpy(dtype=float),
        alpha=0.18,
        color=series_color,
    )
    ax.plot(
        plot_df["top_k"],
        plot_df["median_pct"],
        color=series_color,
        marker="o",
        linewidth=2.86,
        markersize=_scaled_marker(8.45),
    )
    ax.axhline(0.0, color="black", linestyle="-", linewidth=2.2, alpha=0.98, zorder=0)

    for _, row in plot_df.iterrows():
        label = _pvalue_to_stars(_row_significance_value(row))
        is_significant = label != "ns"
        label_color = SAMPLING_SIGNIFICANT_COLOR if is_significant else "#303030"
        ax.annotate(
            label,
            (float(row["top_k"]), float(row["median_pct"])),
            xytext=(0, 10 if is_significant else 7),
            textcoords="offset points",
            fontsize=_scaled_font(14.0) if is_significant else _scaled_font(10.0),
            fontweight="bold" if is_significant else "normal",
            ha="center",
            va="bottom",
            color=label_color,
            clip_on=False,
        )

    y_min = float(np.nanmin(plot_df[["ci_low_pct", "median_pct"]].to_numpy(dtype=float)))
    y_max = float(np.nanmax(plot_df[["ci_high_pct", "median_pct"]].to_numpy(dtype=float)))
    y_span = max(1e-6, y_max - y_min)
    y_lower = min(SENSITIVITY_Y_AXIS_MIN, y_min - 0.06 * y_span, 0.0)
    y_upper = max(y_max + 0.18 * y_span, y_lower + 0.2, 0.3)
    ax.set_ylim(y_lower, y_upper)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label if show_y_axis_label else "")
    ax.grid(axis="x", color="black", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", color="black", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    legend_item_specs = [
        (
            "patch",
            {
                "facecolor": series_color,
                "edgecolor": series_color,
                "alpha": 0.18,
                "label": "95% CI",
            },
        ),
        (
            "line2d",
            {
                "color": series_color,
                "marker": "o",
                "linewidth": 2.86,
                "label": "Dataset-average effect",
            },
        ),
    ]
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=True,
            fontsize=_scaled_font(9.5),
            borderpad=0.35,
            labelspacing=0.5,
            handletextpad=0.65,
            handlelength=2.0,
            columnspacing=1.2,
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sensitivity_semantics",
            section_title="Sensitivity Curve Semantics",
            items=legend_item_specs,
        )
    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    subplot_left = _publication_left_margin(
        show_y_tick_labels=True,
        left_margin_override=left_margin_override,
    )
    subplot_right = 0.98 if inline_legends else 0.996
    if axes_width_scale_override is not None:
        base_axes_width = max(0.05, subplot_right - subplot_left)
        scaled_axes_width = max(0.05, base_axes_width * float(axes_width_scale_override))
        subplot_right = min(0.98, subplot_left + scaled_axes_width)
    subplot_bottom = float(bottom_margin_override) if bottom_margin_override is not None else (0.12 if inline_legends else 0.08)
    subplot_top = 0.90 if show_titles else 0.965
    fig.subplots_adjust(left=subplot_left, right=subplot_right, bottom=subplot_bottom, top=subplot_top)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.2, "transparent": True}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

def _plot_sensitivity_curve_multiseries(
    sensitivity_series_by_mode: Dict[str, pd.DataFrame],
    mode_styles: Dict[str, Dict[str, Any]],
    primary_mode: Optional[str],
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    x_axis_label: str = "Top-k trials used per paired unit",
    y_axis_label: str = "Global dataset-average paired mean % improvement",
    show_y_axis_label: bool = True,
    use_tight_bbox: bool = True,
    left_margin_override: Optional[float] = None,
    bottom_margin_override: Optional[float] = None,
    figure_width_override: Optional[float] = None,
    axes_width_scale_override: Optional[float] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    active_modes = [
        mode
        for mode, frame in sensitivity_series_by_mode.items()
        if frame is not None and not frame.empty
    ]
    if not active_modes:
        return

    preferred_order = [mode for mode in HEX_SENSITIVITY_MODE_STYLES if mode in active_modes]
    ordered_modes = preferred_order + [mode for mode in active_modes if mode not in preferred_order]
    if primary_mode is None or primary_mode not in ordered_modes:
        primary_mode = ordered_modes[0]

    _set_plot_theme(style="whitegrid", context="talk")
    fig_width = float(figure_width_override) if figure_width_override is not None else 9.5
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    legend_item_specs: List[Tuple[str, Dict[str, Any]]] = []
    y_values: List[float] = []

    for mode_key in ordered_modes:
        plot_df = sensitivity_series_by_mode[mode_key].copy().sort_values("top_k")
        if plot_df.empty:
            continue

        style = dict(mode_styles.get(mode_key, {}))
        series_color = style.get("color", DEFAULT_SERIES_COLOR)
        ci_bound_color = _darken_color(series_color, factor=0.50)
        series_marker = style.get("marker", "o")
        series_label = style.get("label", str(mode_key))
        is_primary = mode_key == primary_mode

        line_alpha = 1.0 if is_primary else 0.72
        line_width = 3.38 if is_primary else 2.34
        marker_size = _scaled_marker(8.45 if is_primary else 7.28)
        bound_alpha = 1.0 if is_primary else 0.85
        bound_width = 2.47 if is_primary else 2.08

        x_values = plot_df["top_k"].to_numpy(dtype=float)
        median_values = plot_df["median_pct"].to_numpy(dtype=float)
        ci_low_values = plot_df["ci_low_pct"].to_numpy(dtype=float)
        ci_high_values = plot_df["ci_high_pct"].to_numpy(dtype=float)

        ax.plot(
            x_values,
            median_values,
            color=series_color,
            marker=series_marker,
            linewidth=line_width,
            markersize=marker_size,
            alpha=line_alpha,
            label=series_label,
        )
        ax.plot(
            x_values,
            ci_low_values,
            color=ci_bound_color,
            linestyle=":",
            linewidth=bound_width,
            alpha=bound_alpha,
        )
        ax.plot(
            x_values,
            ci_high_values,
            color=ci_bound_color,
            linestyle=":",
            linewidth=bound_width,
            alpha=bound_alpha,
        )

        finite_values = np.concatenate([median_values, ci_low_values, ci_high_values])
        y_values.extend(finite_values[np.isfinite(finite_values)].tolist())

        legend_item_specs.append(
            (
                "line2d",
                {
                    "color": series_color,
                    "marker": series_marker,
                    "linewidth": 2.86,
                    "markersize": 9.1,
                    "label": str(series_label),
                },
            )
        )

        if is_primary:
            for _, row in plot_df.iterrows():
                label = _pvalue_to_stars(_row_significance_value(row))
                is_significant = label != "ns"
                label_color = SAMPLING_SIGNIFICANT_COLOR if is_significant else "#303030"
                ax.annotate(
                    label,
                    (float(row["top_k"]), float(row["median_pct"])),
                    xytext=(0, 10 if is_significant else 7),
                    textcoords="offset points",
                    fontsize=_scaled_font(14.0) if is_significant else _scaled_font(10.0),
                    fontweight="bold" if is_significant else "normal",
                    ha="center",
                    va="bottom",
                    color=label_color,
                    clip_on=False,
                )

    if not y_values:
        plt.close(fig)
        return

    legend_item_specs.append(
        (
            "line2d",
            {
                "color": "#4f4f4f",
                "linestyle": ":",
                "linewidth": 2.0,
                "label": "Dotted bounds: 95% CI",
            },
        )
    )

    ax.axhline(0.0, color="black", linestyle="-", linewidth=2.2, alpha=0.98, zorder=0)

    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = max(1e-6, y_max - y_min)
    y_lower = min(SENSITIVITY_Y_AXIS_MIN, y_min - 0.06 * y_span, 0.0)
    y_upper = max(y_max + 0.18 * y_span, y_lower + 0.2, 0.3)
    ax.set_ylim(y_lower, y_upper)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label if show_y_axis_label else "")
    ax.grid(axis="x", color="black", linewidth=0.7, alpha=0.65)
    ax.grid(axis="y", color="black", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=True,
            fontsize=_scaled_font(9.5),
            borderpad=0.35,
            labelspacing=0.5,
            handletextpad=0.65,
            handlelength=2.0,
            columnspacing=1.2,
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sensitivity_semantics",
            section_title="Sensitivity Curve Semantics",
            items=legend_item_specs,
        )

    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

    subplot_left = _publication_left_margin(
        show_y_tick_labels=True,
        left_margin_override=left_margin_override,
    )
    subplot_right = 0.98 if inline_legends else 0.996
    if axes_width_scale_override is not None:
        base_axes_width = max(0.05, subplot_right - subplot_left)
        scaled_axes_width = max(0.05, base_axes_width * float(axes_width_scale_override))
        subplot_right = min(0.98, subplot_left + scaled_axes_width)
    subplot_bottom = float(bottom_margin_override) if bottom_margin_override is not None else (0.12 if inline_legends else 0.08)
    subplot_top = 0.90 if show_titles else 0.965
    fig.subplots_adjust(left=subplot_left, right=subplot_right, bottom=subplot_bottom, top=subplot_top)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.2, "transparent": True}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

def _plot_pvalue_panel(
    summary_df: pd.DataFrame,
    dataset_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
    stats_table_output_path: Optional[Path] = None,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df[~summary_df[dataset_col].map(_is_global_dataset_label)].copy()
    if plot_df.empty:
        return

    plot_df["dataset"] = plot_df[dataset_col].astype(str)
    plot_df["significance_value"] = plot_df.apply(_row_significance_value, axis=1)
    plot_df["minus_log10_p"] = -np.log10(
        np.clip(plot_df["significance_value"].astype(float), 1e-300, 1.0)
    )
    plot_df["is_significant"] = plot_df.apply(
        lambda row: _row_is_significant(row, alpha=alpha),
        axis=1,
    )
    plot_df["dataset_group"] = plot_df["dataset"].map(_dataset_group_key)
    plot_df["group_order"] = plot_df["dataset_group"].map(_dataset_group_rank)
    plot_df = plot_df.sort_values(
        ["group_order", "minus_log10_p", "dataset"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    plot_df["y"] = np.arange(len(plot_df))[::-1]

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(5.8, 0.55 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.4, fig_height))

    colors = np.where(plot_df["is_significant"], "#d7301f", "#7a7a7a")
    ax.hlines(y=plot_df["y"], xmin=0.0, xmax=plot_df["minus_log10_p"], color="#9e9e9e", linewidth=1.2)
    ax.scatter(plot_df["minus_log10_p"], plot_df["y"], c=colors, s=_scaled_scatter(75), zorder=3)

    threshold = -np.log10(alpha)
    uses_q_values = bool(
        "q_value" in plot_df.columns
        and pd.to_numeric(plot_df["q_value"], errors="coerce").notna().any()
    )
    significance_label = "q-value" if uses_q_values else "p-value"
    threshold_label = "q" if uses_q_values else "p"
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.9,
        label=f"{threshold_label}={alpha:.2g} threshold",
    )
    y_positions = {
        str(dataset): float(len(plot_df) - 1 - idx)
        for idx, dataset in enumerate(plot_df["dataset"].tolist())
    }
    _draw_dataset_group_separators(ax, plot_df["dataset"].tolist(), y_positions)

    x_max = float(np.nanmax(plot_df["minus_log10_p"].to_numpy(dtype=float)))
    span = max(1.0, x_max)
    ax.set_xlim(0.0, x_max + 0.08 * span)

    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["dataset"])
    ax.set_xlabel(f"-log10({significance_label})")
    ax.set_ylabel("")

    legend_item_specs = [
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": "#d7301f",
                "markeredgecolor": "#d7301f",
                "markersize": 7,
                "label": "Significant",
            },
        ),
        (
            "line2d",
            {
                "marker": "o",
                "color": "none",
                "markerfacecolor": "#7a7a7a",
                "markeredgecolor": "#7a7a7a",
                "markersize": 7,
                "label": "Non-significant",
            },
        ),
        (
            "line2d",
            {
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.1,
                "label": f"Threshold {threshold_label}={alpha:.2g}",
            },
        ),
    ]
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=True,
            fontsize=_scaled_font(9.5),
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="pvalue_semantics",
            section_title="P-value Panel Semantics",
            items=legend_item_specs,
        )

    _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)
    bottom_margin = 0.14 if inline_legends else 0.08
    fig.subplots_adjust(left=0.08, right=0.98, bottom=bottom_margin, top=0.92)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    _write_per_row_stats_table(plot_df, dataset_col="dataset", output_path=stats_table_output_path)

def _plot_dataset_forest_publication(
    *,
    summary_df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    stats_table_output_path: Optional[Path] = None,
    x_axis_label: str = "Mean % Improvement (positive means first group is better)",
    show_y_tick_labels: bool = True,
    x_axis_label_font_size: Optional[float] = 13.8,
    x_tick_label_font_size: Optional[float] = 13.8,
    y_tick_label_font_size: Optional[float] = 15.0,
    left_margin_override: Optional[float] = None,
    figure_width_override: Optional[float] = None,
) -> None:
    _plot_dataset_forest(
        summary_df=summary_df,
        dataset_col="dataset",
        title=title,
        subtitle=subtitle,
        output_path=output_path,
        dpi=dpi,
        alpha=alpha,
        inline_legends=inline_legends,
        legend_registry=legend_registry,
        stats_table_output_path=stats_table_output_path,
        x_axis_label=x_axis_label,
        show_y_tick_labels=show_y_tick_labels,
        x_axis_label_font_size=x_axis_label_font_size,
        x_tick_label_font_size=x_tick_label_font_size,
        y_tick_label_font_size=y_tick_label_font_size,
        left_margin_override=left_margin_override,
        figure_width_override=figure_width_override,
        use_tight_bbox=True,
    )

def _plot_sampling_mode_outcome_panels(
    panel_specs: Sequence[Tuple[str, str, pd.DataFrame]],
    title: str,
    output_path: Path,
    dpi: int,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if not panel_specs:
        return

    color_map = {
        "full": "#4C78A8",
        "random": "#F58518",
        "hdsssom": "#54A24B",
    }
    n_panels = len(panel_specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.2 * n_panels, 5.8), sharey=False)
    if n_panels == 1:
        axes = np.array([axes])

    for idx, (panel_letter, panel_title, panel_df) in enumerate(panel_specs):
        ax = axes[idx]
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        if panel_df.empty:
            ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes, ha="center", va="center", color="#666666")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ordered_modes = _ordered_sampling_modes(panel_df["pair_sampling"].astype(str).tolist())
            work = panel_df.copy()
            work["pair_sampling"] = work["pair_sampling"].astype(str).str.lower().str.strip()
            work["x"] = work["pair_sampling"].map({mode: idx_mode for idx_mode, mode in enumerate(ordered_modes)})
            work = work.sort_values("x")
            x = work["x"].to_numpy(dtype=float)
            y = work["median_pct"].to_numpy(dtype=float)
            yerr_low = y - work["ci_low_pct"].to_numpy(dtype=float)
            yerr_high = work["ci_high_pct"].to_numpy(dtype=float) - y
            colors = [color_map.get(mode, "#7A7A7A") for mode in work["pair_sampling"]]

            ax.errorbar(
                x=x,
                y=y,
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="none",
                ecolor="#8c8c8c",
                elinewidth=1.8,
                capsize=4,
                zorder=1,
            )
            ax.scatter(x, y, s=_scaled_scatter(90), c=colors, zorder=2)
            ax.plot(x, y, color="#666666", linewidth=1.3, alpha=0.7, zorder=1)

            for _, row in work.iterrows():
                ax.annotate(
                    f"n={int(row['n_pairs'])}",
                    (float(row["x"]), float(row["median_pct"])),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=_scaled_font(9.5),
                    color="#303030",
                )

            ax.set_xticks(range(len(ordered_modes)))
            ax.set_xticklabels([_sampling_mode_display(mode) for mode in ordered_modes], rotation=0)
            ax.set_ylabel("Global mean % improvement", labelpad=SAMPLING_AXIS_LABEL_PAD)

        ax.set_title(panel_title, fontsize=_scaled_font(11.5), loc="left", pad=8, color="#2f2f2f")
        ax.text(
            0.01,
            0.99,
            panel_letter,
            transform=ax.transAxes,
            fontsize=_scaled_font(15),
            fontweight="bold",
            ha="left",
            va="top",
            color="#1f1f1f",
        )
        ax.set_xlabel("Sampling mode", labelpad=SAMPLING_AXIS_LABEL_PAD)

    if show_titles:
        fig.suptitle(title, fontsize=_scaled_font(17), x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95], w_pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

def _plot_sampling_mode_disaggregated_grid(
    panel_data: Dict[Tuple[str, str], pd.DataFrame],
    title: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if not panel_data:
        return

    has_minibatch = any(key[1] == "minibatch" for key in panel_data.keys())
    if has_minibatch:
        panel_order = [
            ("hexagonal", "full_batch"),
            ("hexagonal", "minibatch"),
            ("hexagonal", "colors"),
            ("mst", "full_batch"),
            ("mst", "minibatch"),
            ("mst", "colors"),
        ]
        panel_titles = {
            ("hexagonal", "full_batch"): "Hexagonal / Full Batch",
            ("hexagonal", "minibatch"): "Hexagonal / Mini-batch",
            ("hexagonal", "colors"): "Hexagonal / Colors",
            ("mst", "full_batch"): "MST / Full Batch",
            ("mst", "minibatch"): "MST / Mini-batch",
            ("mst", "colors"): "MST / Colors",
        }
    else:
        panel_order = [
            ("hexagonal", "full_batch"),
            ("hexagonal", "colors"),
            ("mst", "full_batch"),
            ("mst", "colors"),
        ]
        panel_titles = {
            ("hexagonal", "full_batch"): "Hexagonal / Full Batch",
            ("hexagonal", "colors"): "Hexagonal / Colors",
            ("mst", "full_batch"): "MST / Full Batch",
            ("mst", "colors"): "MST / Colors",
        }

    comparison_order: List[str] = [
        "full_vs_random",
        "full_vs_hdsssom",
        "random_vs_hdsssom",
    ]
    comparison_display: Dict[str, str] = {
        "full_vs_random": "Full vs Random",
        "full_vs_hdsssom": "Full vs HDSSOM",
        "random_vs_hdsssom": "Random vs HDSSOM",
    }
    comparison_styles: Dict[str, Dict[str, Any]] = {
        "full_vs_random": {"marker": "o", "linestyle": "-"},
        "full_vs_hdsssom": {"marker": "s", "linestyle": "--"},
        "random_vs_hdsssom": {"marker": "^", "linestyle": ":"},
    }
    comparison_offsets = {
        "full_vs_random": -0.38,
        "full_vs_hdsssom": 0.0,
        "random_vs_hdsssom": 0.38,
    }
    line_color = "#4f4f4f"

    all_datasets: List[str] = []
    for panel_key in panel_order:
        df = panel_data.get(panel_key)
        if df is None or df.empty:
            continue
        all_datasets.extend(df["dataset"].astype(str).tolist())

    if not all_datasets:
        return

    dataset_order = _ordered_dataset_labels(all_datasets)
    lane_stride = 1.55
    y_base = {dataset: lane_stride * (len(dataset_order) - 1 - idx) for idx, dataset in enumerate(dataset_order)}
    y_tick_values = [y_base[name] for name in dataset_order]
    y_min = min(y_tick_values) - 0.85
    y_max = max(y_tick_values) + 0.85

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(14.0, 0.84 * len(dataset_order) + 9.0)
    n_rows = 2
    n_cols = 3 if has_minibatch else 2
    fig_width = 24.0 if has_minibatch else 18.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for idx, panel_key in enumerate(panel_order):
        ax = axes_flat[idx]
        panel_df = panel_data.get(panel_key, pd.DataFrame()).copy()

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
        ax.set_ylim(y_min, y_max)
        _draw_dataset_group_separators(ax, dataset_order, y_base)

        if panel_df.empty:
            ax.set_xlim(-1.0, 1.0)
            ax.text(
                0.5,
                0.5,
                "No paired data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=_scaled_font(11),
            )
        else:
            panel_df["dataset"] = panel_df["dataset"].astype(str)
            panel_df["comparison_key"] = panel_df["comparison_key"].astype(str)
            panel_df["base_y"] = panel_df["dataset"].map(y_base).astype(float)
            panel_df["offset"] = panel_df["comparison_key"].map(comparison_offsets).astype(float)
            panel_df["y"] = panel_df["base_y"] + panel_df["offset"]
            panel_df = panel_df.sort_values(["base_y", "comparison_key"]).reset_index(drop=True)

            finite_x = np.concatenate(
                [
                    panel_df["ci_low_pct"].to_numpy(dtype=float),
                    panel_df["ci_high_pct"].to_numpy(dtype=float),
                    panel_df["median_pct"].to_numpy(dtype=float),
                ]
            )
            finite_x = finite_x[np.isfinite(finite_x)]
            if finite_x.size == 0:
                x_left, x_right = -1.0, 1.0
            else:
                x_min = float(np.min(finite_x))
                x_max = float(np.max(finite_x))
                x_span = max(1.0, x_max - x_min)
                x_left = x_min - 0.10 * x_span
                x_right = x_max + 0.10 * x_span
            ax.set_xlim(x_left, x_right)

            for _, row in panel_df.iterrows():
                comp_key = str(row["comparison_key"])
                style = comparison_styles.get(comp_key, comparison_styles["full_vs_random"])
                row_color = SAMPLING_SIGNIFICANT_COLOR if _row_is_significant(row, alpha=alpha) else SAMPLING_NON_SIGNIFICANT_COLOR
                ax.hlines(
                    y=float(row["y"]),
                    xmin=float(row["ci_low_pct"]),
                    xmax=float(row["ci_high_pct"]),
                    color=row_color,
                    linewidth=1.5,
                    linestyle=style["linestyle"],
                    alpha=0.9,
                    zorder=1,
                )
                is_global = _is_global_dataset_label(row["dataset"])
                marker_size = _scaled_scatter(78) if not is_global else _scaled_scatter(105)
                marker_style = "D" if is_global else style["marker"]
                ax.scatter(
                    float(row["median_pct"]),
                    float(row["y"]),
                    color=row_color,
                    s=marker_size,
                    marker=marker_style,
                    zorder=2,
                )

        ax.set_title(panel_titles[panel_key], fontsize=_scaled_font(12.0), loc="left", pad=8, color="#2f2f2f")
        ax.set_yticks(y_tick_values)
        ax.set_yticklabels(dataset_order)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(
                "Mean % Improvement (positive means first group is better)",
                labelpad=SAMPLING_AXIS_LABEL_PAD,
            )
        else:
            ax.set_xlabel("", labelpad=SAMPLING_AXIS_LABEL_PAD)
        ax.set_ylabel("", labelpad=SAMPLING_AXIS_LABEL_PAD)

    legend_item_specs: List[Tuple[str, Dict[str, Any]]] = []
    for key in comparison_order:
        style = comparison_styles[key]
        legend_item_specs.append(
            (
                "line2d",
                {
                    "color": line_color,
                    "linestyle": style["linestyle"],
                    "marker": style["marker"],
                    "markersize": 7,
                    "linewidth": 1.8,
                    "label": comparison_display[key],
                },
            )
        )
    legend_item_specs.extend(
        [
            (
                "line2d",
                {
                    "linestyle": "none",
                    "marker": "o",
                    "color": "none",
                    "markerfacecolor": SAMPLING_SIGNIFICANT_COLOR,
                    "markeredgecolor": SAMPLING_SIGNIFICANT_COLOR,
                    "markersize": 8,
                    "label": f"Significant (q/p<{alpha:.2g})",
                },
            ),
            (
                "line2d",
                {
                    "linestyle": "none",
                    "marker": "o",
                    "color": "none",
                    "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                    "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                    "markersize": 8,
                    "label": f"Non-significant (q/p>={alpha:.2g})",
                },
            ),
        ]
    )

    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=True,
            fontsize=SAMPLING_LEGEND_FONT_SIZE,
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sampling_mode_disaggregated_semantics",
            section_title="Sampling-Mode Disaggregated Semantics",
            items=legend_item_specs,
        )
    if show_titles:
        fig.suptitle(title, fontsize=_scaled_font(17), x=0.01, ha="left", y=0.995)
    top_rect = 0.95 if show_titles else 0.98
    bottom_rect = 0.10 if inline_legends else 0.04
    fig.tight_layout(rect=[0.0, bottom_rect, 1.0, top_rect], h_pad=1.0, w_pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

def _plot_sampling_mode_disaggregated_no_stack_grid(
    panel_data: Dict[Tuple[str, str], pd.DataFrame],
    title: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if not panel_data:
        return

    has_minibatch = any(key[1] == "minibatch" for key in panel_data.keys())
    if has_minibatch:
        panel_order = [
            ("hexagonal", "full_batch"),
            ("hexagonal", "minibatch"),
            ("hexagonal", "colors"),
            ("mst", "full_batch"),
            ("mst", "minibatch"),
            ("mst", "colors"),
        ]
        panel_titles = {
            ("hexagonal", "full_batch"): "Hexagonal / Full Batch",
            ("hexagonal", "minibatch"): "Hexagonal / Mini-batch",
            ("hexagonal", "colors"): "Hexagonal / Colors",
            ("mst", "full_batch"): "MST / Full Batch",
            ("mst", "minibatch"): "MST / Mini-batch",
            ("mst", "colors"): "MST / Colors",
        }
    else:
        panel_order = [
            ("hexagonal", "full_batch"),
            ("hexagonal", "colors"),
            ("mst", "full_batch"),
            ("mst", "colors"),
        ]
        panel_titles = {
            ("hexagonal", "full_batch"): "Hexagonal / Full Batch",
            ("hexagonal", "colors"): "Hexagonal / Colors",
            ("mst", "full_batch"): "MST / Full Batch",
            ("mst", "colors"): "MST / Colors",
        }

    comparison_order: List[str] = [
        "full_vs_random",
        "full_vs_hdsssom",
        "random_vs_hdsssom",
    ]
    comparison_display: Dict[str, str] = {
        "full_vs_random": "Full vs Random",
        "full_vs_hdsssom": "Full vs HDSSOM",
        "random_vs_hdsssom": "Random vs HDSSOM",
    }
    comparison_styles: Dict[str, Dict[str, Any]] = {
        "full_vs_random": {"marker": "o", "linestyle": "-"},
        "full_vs_hdsssom": {"marker": "s", "linestyle": "--"},
        "random_vs_hdsssom": {"marker": "^", "linestyle": ":"},
    }

    all_datasets: List[str] = []
    for frame in panel_data.values():
        if frame is None or frame.empty:
            continue
        all_datasets.extend(frame["dataset"].astype(str).tolist())
    if not all_datasets:
        return

    dataset_order = _ordered_dataset_labels(all_datasets)
    lane_stride = 1.35
    y_base = {dataset: lane_stride * (len(dataset_order) - 1 - idx) for idx, dataset in enumerate(dataset_order)}
    y_tick_values = [y_base[name] for name in dataset_order]
    y_min = min(y_tick_values) - 0.7
    y_max = max(y_tick_values) + 0.7

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(18.5, 0.95 * len(dataset_order) + 12.0)
    fig, axes = plt.subplots(len(panel_order), 3, figsize=(24.0, fig_height), sharex=False, sharey=True)

    for row_idx, panel_key in enumerate(panel_order):
        panel_df = panel_data.get(panel_key, pd.DataFrame()).copy()
        if not panel_df.empty:
            panel_df["dataset"] = panel_df["dataset"].astype(str)
            panel_df["comparison_key"] = panel_df["comparison_key"].astype(str)

        for col_idx, comparison_key in enumerate(comparison_order):
            ax = axes[row_idx, col_idx]
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
            ax.set_ylim(y_min, y_max)
            _draw_dataset_group_separators(ax, dataset_order, y_base)

            subset = panel_df[panel_df["comparison_key"] == comparison_key].copy() if not panel_df.empty else pd.DataFrame()
            if subset.empty:
                ax.set_xlim(-1.0, 1.0)
                ax.text(
                    0.5,
                    0.5,
                    "No paired data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=_scaled_font(10.5),
                )
            else:
                subset["y"] = subset["dataset"].map(y_base).astype(float)
                subset = subset.sort_values("y", ascending=False).reset_index(drop=True)

                finite_x = np.concatenate(
                    [
                        subset["ci_low_pct"].to_numpy(dtype=float),
                        subset["ci_high_pct"].to_numpy(dtype=float),
                        subset["median_pct"].to_numpy(dtype=float),
                    ]
                )
                finite_x = finite_x[np.isfinite(finite_x)]
                if finite_x.size == 0:
                    x_left, x_right = -1.0, 1.0
                else:
                    x_min = float(np.min(finite_x))
                    x_max = float(np.max(finite_x))
                    x_span = max(1.0, x_max - x_min)
                    x_left = x_min - 0.12 * x_span
                    x_right = x_max + 0.12 * x_span
                ax.set_xlim(x_left, x_right)

                style = comparison_styles.get(comparison_key, comparison_styles["full_vs_random"])
                for _, summary_row in subset.iterrows():
                    row_color = (
                        SAMPLING_SIGNIFICANT_COLOR
                        if _row_is_significant(summary_row, alpha=alpha)
                        else SAMPLING_NON_SIGNIFICANT_COLOR
                    )
                    ax.hlines(
                        y=float(summary_row["y"]),
                        xmin=float(summary_row["ci_low_pct"]),
                        xmax=float(summary_row["ci_high_pct"]),
                        color=row_color,
                        linewidth=1.5,
                        linestyle=style["linestyle"],
                        alpha=0.9,
                        zorder=1,
                    )
                    is_global = _is_global_dataset_label(summary_row["dataset"])
                    marker_size = _scaled_scatter(76) if not is_global else _scaled_scatter(102)
                    marker_style = "D" if is_global else style["marker"]
                    ax.scatter(
                        float(summary_row["median_pct"]),
                        float(summary_row["y"]),
                        color=row_color,
                        s=marker_size,
                        marker=marker_style,
                        zorder=2,
                    )

            if row_idx == 0:
                ax.set_title(comparison_display[comparison_key], fontsize=_scaled_font(11.5), loc="left", pad=8, color="#2f2f2f")
            if row_idx == len(panel_order) - 1:
                ax.set_xlabel(
                    "Mean % Improvement (positive means first group is better)",
                    labelpad=SAMPLING_AXIS_LABEL_PAD,
                )
            else:
                ax.set_xlabel("", labelpad=SAMPLING_AXIS_LABEL_PAD)

            ax.set_yticks(y_tick_values)
            if col_idx == 0:
                ax.set_yticklabels(dataset_order)
                ax.set_ylabel(panel_titles[panel_key], labelpad=24.0, fontsize=_scaled_font(11.0), color="#303030")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    legend_item_specs: List[Tuple[str, Dict[str, Any]]] = [
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Significant (q/p<{alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Non-significant (q/p>={alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "D",
                "color": "#4f4f4f",
                "markerfacecolor": "#4f4f4f",
                "markeredgecolor": "#4f4f4f",
                "markersize": 7.5,
                "label": "GLOBAL group summary",
            },
        ),
    ]
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=True,
            fontsize=SAMPLING_LEGEND_FONT_SIZE,
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sampling_mode_no_stack_semantics",
            section_title="Sampling-Mode No-Stack Semantics",
            items=legend_item_specs,
        )
    if show_titles:
        fig.suptitle(title, fontsize=_scaled_font(17.5), x=0.01, ha="left", y=0.995)
    top_rect = 0.96 if show_titles else 0.99
    bottom_rect = 0.08 if inline_legends else 0.03
    fig.tight_layout(rect=[0.0, bottom_rect, 1.0, top_rect], h_pad=1.0, w_pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

def _plot_sampling_mode_main_hex_batch_panel(
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int,
    alpha: float,
    inline_legends: bool = True,
    legend_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if summary_df.empty:
        return

    comparison_order: List[str] = [
        "full_vs_random",
        "full_vs_hdsssom",
        "random_vs_hdsssom",
    ]
    comparison_display: Dict[str, str] = {
        "full_vs_random": "Full vs Random",
        "full_vs_hdsssom": "Full vs HDSSOM",
        "random_vs_hdsssom": "Random vs HDSSOM",
    }
    comparison_styles: Dict[str, Dict[str, Any]] = {
        "full_vs_random": {"marker": "o"},
        "full_vs_hdsssom": {"marker": "s"},
        "random_vs_hdsssom": {"marker": "^"},
    }

    work = summary_df.copy()
    work["dataset"] = work["dataset"].astype(str)
    work["comparison_key"] = work["comparison_key"].astype(str)
    work = work[work["comparison_key"].isin(comparison_order)].copy()
    if work.empty:
        return

    all_datasets = work["dataset"].tolist()
    dataset_order = _ordered_dataset_labels(all_datasets)
    lane_stride = 1.15
    y_base = {dataset: lane_stride * (len(dataset_order) - 1 - idx) for idx, dataset in enumerate(dataset_order)}
    work["base_y"] = work["dataset"].map(y_base).astype(float)
    work["y"] = work["base_y"]
    work = work.sort_values(["base_y", "dataset"]).reset_index(drop=True)

    y_tick_values = [y_base[name] for name in dataset_order]
    y_min = min(y_tick_values) - 0.55
    y_max = max(y_tick_values) + 0.55

    finite_x = np.concatenate(
        [
            work["ci_low_pct"].to_numpy(dtype=float),
            work["ci_high_pct"].to_numpy(dtype=float),
            work["median_pct"].to_numpy(dtype=float),
        ]
    )
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size == 0:
        x_left, x_right = -1.0, 1.0
    else:
        x_min = float(np.min(finite_x))
        x_max = float(np.max(finite_x))
        x_span = max(1.0, x_max - x_min)
        x_left = x_min - 0.10 * x_span
        x_right = x_max + 0.10 * x_span

    _set_plot_theme(style="whitegrid", context="talk")
    fig_height = max(8.2, 0.52 * len(dataset_order) + 3.8)
    fig, axes = plt.subplots(1, 3, figsize=(17.6, fig_height), sharey=True)
    axes_flat = np.array(axes).reshape(-1)

    for idx, comparison_key in enumerate(comparison_order):
        ax = axes_flat[idx]
        panel = work[work["comparison_key"] == comparison_key].copy()
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_min, y_max)
        _draw_dataset_group_separators(ax, dataset_order, y_base)

        if panel.empty:
            ax.text(
                0.5,
                0.5,
                "No paired data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=_scaled_font(11),
            )
        else:
            style = comparison_styles.get(comparison_key, comparison_styles["full_vs_random"])
            for _, row in panel.iterrows():
                row_color = (
                    SAMPLING_SIGNIFICANT_COLOR if _row_is_significant(row, alpha=alpha) else SAMPLING_NON_SIGNIFICANT_COLOR
                )
                ax.hlines(
                    y=float(row["y"]),
                    xmin=float(row["ci_low_pct"]),
                    xmax=float(row["ci_high_pct"]),
                    color=row_color,
                    linewidth=1.6,
                    linestyle="-",
                    alpha=0.9,
                    zorder=1,
                )
                is_global = _is_global_dataset_label(row["dataset"])
                marker_size = _scaled_scatter(78) if not is_global else _scaled_scatter(108)
                marker_style = "D" if is_global else style["marker"]
                ax.scatter(
                    float(row["median_pct"]),
                    float(row["y"]),
                    color=row_color,
                    s=marker_size,
                    marker=marker_style,
                    zorder=2,
                )

        ax.set_title(comparison_display[comparison_key], fontsize=_scaled_font(12.5), loc="left", pad=8, color="#2f2f2f")
        ax.set_yticks(y_tick_values)
        if idx == 0:
            ax.set_yticklabels(dataset_order)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Mean % Improvement", labelpad=SAMPLING_AXIS_LABEL_PAD)

    legend_item_specs: List[Tuple[str, Dict[str, Any]]] = [
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Significant (q/p<{alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "o",
                "color": "none",
                "markerfacecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markeredgecolor": SAMPLING_NON_SIGNIFICANT_COLOR,
                "markersize": 8,
                "label": f"Non-significant (q/p>={alpha:.2g})",
            },
        ),
        (
            "line2d",
            {
                "linestyle": "none",
                "marker": "D",
                "color": "#4f4f4f",
                "markerfacecolor": "#4f4f4f",
                "markeredgecolor": "#4f4f4f",
                "markersize": 7.5,
                "label": "GLOBAL group summary",
            },
        ),
    ]
    if inline_legends:
        fig.legend(
            handles=[_build_legend_handle_from_spec({"kind": kind, "kwargs": kwargs}) for kind, kwargs in legend_item_specs],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=True,
            fontsize=SAMPLING_LEGEND_FONT_SIZE,
        )
    elif legend_registry is not None:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id="sampling_mode_main_panel_semantics",
            section_title="Sampling-Mode Main Panel Semantics",
            items=legend_item_specs,
        )
    if show_titles:
        fig.suptitle(title, fontsize=_scaled_font(17.0), x=0.01, ha="left", y=0.995)
    top_rect = 0.95 if show_titles else 0.98
    bottom_rect = 0.08 if inline_legends else 0.03
    fig.tight_layout(rect=[0.0, bottom_rect, 1.0, top_rect], w_pad=0.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

def _plot_selected_parameter_stability_figure(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
    topology_left: str,
    topology_right: str,
    dpi: int,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    for column in [
        "stability_score_mean_left",
        "stability_score_mean_right",
        "stability_score_median_left",
        "stability_score_median_right",
        "delta_left_minus_right",
    ]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["stability_score_mean_left", "stability_score_mean_right"])
    if plot_df.empty:
        return

    plot_df["parameter_label"] = plot_df["parameter"].map(_format_stability_parameter_label)
    plot_df["is_overall"] = plot_df["parameter"].astype(str).str.lower().str.startswith("overall")
    plot_df = pd.concat(
        [plot_df[~plot_df["is_overall"]], plot_df[plot_df["is_overall"]]],
        ignore_index=True,
    )
    plot_df = plot_df.reset_index(drop=True)
    plot_df["y"] = np.arange(len(plot_df))[::-1]

    left_color = TOPOLOGY_BASE_COLORS.get(str(topology_left).strip().lower(), "#4C78A8")
    right_color = TOPOLOGY_BASE_COLORS.get(str(topology_right).strip().lower(), "#F58518")
    tie_color = "#7a7a7a"

    rc = {
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.8,
        "axes.labelpad": 8.0,
    }
    fig_height = max(3.8, 1.6 + 0.54 * len(plot_df))
    fig_width = 8.8
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for _, row in plot_df[plot_df["is_overall"]].iterrows():
            y_value = float(row["y"])
            ax.axhspan(y_value - 0.48, y_value + 0.48, color="#f6f8fb", zorder=0)

        for _, row in plot_df.iterrows():
            winner_value = str(row.get("winner", "")).strip().lower()
            line_color = tie_color
            if winner_value == str(topology_left).strip().lower():
                line_color = left_color
            elif winner_value == str(topology_right).strip().lower():
                line_color = right_color
            line_width = 2.1 if bool(row.get("is_overall")) else 1.6
            ax.hlines(
                y=float(row["y"]),
                xmin=float(row["stability_score_mean_left"]),
                xmax=float(row["stability_score_mean_right"]),
                color=line_color,
                linewidth=line_width,
                alpha=0.88,
            )
            # Filled markers are means (the quantities used for winner/delta).
            ax.scatter(
                float(row["stability_score_mean_left"]),
                float(row["y"]),
                s=44,
                color=left_color,
                zorder=3,
            )
            ax.scatter(
                float(row["stability_score_mean_right"]),
                float(row["y"]),
                s=44,
                color=right_color,
                zorder=3,
            )

        ax.set_yticks(plot_df["y"].to_numpy(dtype=float))
        ax.set_yticklabels(plot_df["parameter_label"].astype(str).tolist())
        ax.set_xlabel("Stability Score (Lower Is Better)", fontsize=10.5)
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=3.5, width=0.8, color="#5c5c5c")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_axis_tick(x)))
        ax.grid(axis="x", color="black", linewidth=0.7, alpha=1.0)
        ax.axvline(0.0, color="#8a8a8a", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_facecolor("white")
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
        ax.spines["left"].set_color("#b3b3b3")
        ax.spines["bottom"].set_color("#b3b3b3")
        _apply_optional_titles(fig, ax, title=title, subtitle=subtitle, show_titles=show_titles)

        x_min = float(
            np.nanmin(
                np.concatenate(
                    [
                        plot_df["stability_score_mean_left"].to_numpy(dtype=float),
                        plot_df["stability_score_mean_right"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        x_max = float(
            np.nanmax(
                np.concatenate(
                    [
                        plot_df["stability_score_mean_left"].to_numpy(dtype=float),
                        plot_df["stability_score_mean_right"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        span = max(1e-6, x_max - x_min)
        left_bound = min(-0.01, x_min - span * 0.12)
        right_bound = min(1.06, x_max + span * 0.12)
        ax.set_xlim(left_bound, right_bound)

        fig.tight_layout(rect=[0.0, 0.02, 0.98, 0.98])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.14)
        plt.close(fig)

def _plot_selected_parameter_stability_stratified_figure(
    sampling_tables: Dict[str, pd.DataFrame],
    *,
    output_path: Path,
    title: str,
    subtitle: str,
    dpi: int,
    show_titles: bool = SHOW_PANEL_PLOT_TITLES,
    show_mode_titles: bool = True,
    show_shared_y_label: bool = True,
    show_shared_x_label: bool = True,
    x_axis_label: str = "",
    figure_width_override: Optional[float] = None,
    figure_height_override: Optional[float] = None,
    left_margin_override: Optional[float] = None,
    bottom_margin_override: Optional[float] = None,
    use_tight_bbox: bool = True,
) -> None:
    plotting_modes = [
        mode
        for mode in DEFAULT_AWARE_STABILITY_PUBLICATION_SAMPLING_MODES
        if mode in sampling_tables and not sampling_tables[mode].empty
    ]
    if not plotting_modes:
        return

    topology_columns = {
        "hexagonal": "stability_score_mean_hexagonal",
        "mst": "stability_score_mean_mst",
        "rng": "stability_score_mean_rng",
    }
    topology_colors = {
        topology: TOPOLOGY_BASE_COLORS.get(topology, "#4C78A8")
        for topology in topology_columns
    }

    all_values: List[float] = []
    for mode in plotting_modes:
        frame = sampling_tables[mode]
        for col in topology_columns.values():
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            all_values.extend([float(v) for v in values if np.isfinite(v)])
    if not all_values:
        return

    x_min = float(np.min(all_values))
    x_max = float(np.max(all_values))
    span = max(1e-6, x_max - x_min)
    left_bound = min(-0.01, x_min - span * 0.12)
    right_bound = min(1.06, x_max + span * 0.12)

    rc = {
        "font.size": 15.0,
        "axes.titlesize": 17.0,
        "axes.labelsize": 16.4,
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 15.2,
        "axes.labelpad": 10.0,
    }
    fig_width = (
        float(figure_width_override)
        if figure_width_override is not None
        else max(8.8, 5.6 * len(plotting_modes))
    )
    max_rows = max(len(sampling_tables[mode]) for mode in plotting_modes)
    fig_height = (
        float(figure_height_override)
        if figure_height_override is not None
        else max(4.2, 1.8 + 0.56 * max_rows)
    )

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(plotting_modes), figsize=(fig_width, fig_height), squeeze=False)
        axes_list = [axes[0, idx] for idx in range(len(plotting_modes))]

        for axis_idx, mode in enumerate(plotting_modes):
            ax = axes_list[axis_idx]
            frame = sampling_tables[mode].copy()
            frame["is_overall"] = frame["parameter"].astype(str).str.lower().str.startswith("overall")
            frame = pd.concat([frame[~frame["is_overall"]], frame[frame["is_overall"]]], ignore_index=True)
            frame["y"] = np.arange(len(frame))[::-1]

            for _, row in frame[frame["is_overall"]].iterrows():
                y_value = float(row["y"])
                ax.axhspan(y_value - 0.48, y_value + 0.48, color="#f6f8fb", zorder=0)

            for _, row in frame.iterrows():
                points: List[float] = []
                for col in topology_columns.values():
                    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        points.append(float(value))
                if len(points) >= 2:
                    ax.hlines(
                        y=float(row["y"]),
                        xmin=float(np.min(points)),
                        xmax=float(np.max(points)),
                        color="#8f8f8f",
                        linewidth=1.9 if not bool(row.get("is_overall")) else 2.4,
                        alpha=0.85,
                        zorder=1,
                    )

                for topology, col in topology_columns.items():
                    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                    if pd.isna(value):
                        continue
                    ax.scatter(
                        float(value),
                        float(row["y"]),
                        s=82,
                        color=topology_colors[topology],
                        zorder=3,
                    )

            ax.set_xlim(left_bound, right_bound)
            ax.set_yticks(frame["y"].to_numpy(dtype=float))
            if axis_idx == 0:
                ax.set_yticklabels(frame["parameter_label"].astype(str).tolist())
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=7, labelsize=15.2)
            ax.tick_params(axis="x", length=4.1, width=1.0, color="#5c5c5c", labelsize=15.0)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _format_axis_tick(x)))
            ax.grid(False, axis="y")
            ax.grid(axis="x", color="black", linewidth=0.7, alpha=1.0)
            ax.axvline(0.0, color="#8a8a8a", linestyle="--", linewidth=1.0, alpha=0.78)
            ax.set_facecolor("white")
            for spine_name in ("top", "right"):
                ax.spines[spine_name].set_visible(False)
            ax.spines["left"].set_color("#b3b3b3")
            ax.spines["bottom"].set_color("#b3b3b3")
            if show_mode_titles:
                ax.set_title(_sampling_mode_display(mode), pad=9, fontsize=17.0)
            if str(x_axis_label).strip():
                ax.set_xlabel(str(x_axis_label), fontsize=16.0, labelpad=9)
            else:
                ax.set_xlabel("")

        _apply_optional_titles(fig, axes_list[0], title=title, subtitle=subtitle, show_titles=show_titles)
        if show_shared_y_label:
            fig.supylabel("Parameter", fontsize=14.0, x=0.008)
        if show_shared_x_label:
            fig.supxlabel("Stability Score (Lower Is Better)", fontsize=14.0, y=0.022)
        tight_left = (
            float(left_margin_override)
            if left_margin_override is not None
            else (0.05 if show_shared_y_label else 0.02)
        )
        tight_bottom = (
            float(bottom_margin_override)
            if bottom_margin_override is not None
            else (0.07 if show_shared_x_label or str(x_axis_label).strip() else 0.03)
        )
        fig.tight_layout(rect=[tight_left, tight_bottom, 1.0, 0.995])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: Dict[str, Any] = {"dpi": dpi, "pad_inches": 0.08}
        if use_tight_bbox:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(output_path, **save_kwargs)
        plt.close(fig)

def _render_selected_parameter_stability_legend(
    *,
    output_path: Path,
    topology_values: Sequence[str],
    dpi: int,
    font_scale_override: float = 1.0,
) -> Optional[Path]:
    topology_list = [str(value).strip().lower() for value in topology_values if str(value).strip()]
    if not topology_list:
        return None
    legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    items: List[Tuple[str, Dict[str, Any]]] = []
    for topology in topology_list:
        items.append(
            (
                "line2d",
                {
                    "marker": "o",
                    "linestyle": "None",
                    "color": TOPOLOGY_BASE_COLORS.get(topology, "#4C78A8"),
                    "markersize": 13,
                    "label": f"{_architecture_label(topology)} mean stability",
                },
            )
        )
    items.append(
        (
            "line2d",
            {
                "color": "#8f8f8f",
                "linewidth": 2.8,
                "label": "Range across topologies",
            },
        )
    )
    _legend_registry_add_items(
        legend_registry=legend_registry,
        section_id="selected_stability_semantics",
        section_title="Selected Stability Semantics",
        items=items,
    )
    if _render_legend_key(
        legend_registry=legend_registry,
        output_path=output_path,
        dpi=dpi,
        compact=False,
        font_scale_override=font_scale_override,
    ):
        return output_path
    return None

__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
