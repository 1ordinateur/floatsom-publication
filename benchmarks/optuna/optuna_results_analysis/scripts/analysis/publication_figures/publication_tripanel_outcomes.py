from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .composition import _compose_three_panel_stats_figure
from .constants import PUBLICATION_LEGEND_ROW_HEIGHT
from .helpers import _ordered_dataset_labels
from .plots import _plot_dataset_forest_publication, _tripanel_forest_panel_style

CANONICAL_PUBLICATION_QE_CHANGE_LABEL = "QE change (%)"


@dataclass(frozen=True)
class PublicationOutcomePanelInput:
    panel_letter: str
    metric_label: str
    metric_slug: str
    plot_df: pd.DataFrame
    output_path: Path
    title: str
    subtitle: str
    stats_table_output_path: Optional[Path] = None
    show_y_tick_labels: bool = False
    empty_message: str = "Summary exists but has no plottable rows."


def render_publication_outcome_panels(
    *,
    panel_inputs: Sequence[PublicationOutcomePanelInput],
    dpi: int,
    alpha: float,
    generated_files: List[str],
) -> Dict[str, object]:
    panel_specs: List[Tuple[str, str, Path]] = []
    per_metric_figures: Dict[str, str] = {}
    missing_or_empty: Dict[str, str] = {}
    top_row_dataset_labels: List[str] = []

    for panel in panel_inputs:
        plot_df = panel.plot_df
        if plot_df.empty:
            missing_or_empty[panel.metric_slug] = panel.empty_message
            continue

        if not top_row_dataset_labels:
            top_row_dataset_labels = _ordered_dataset_labels(plot_df["dataset"].astype(str).tolist())

        _plot_dataset_forest_publication(
            summary_df=plot_df,
            title=panel.title,
            subtitle=panel.subtitle,
            output_path=panel.output_path,
            dpi=int(dpi),
            alpha=float(alpha),
            inline_legends=False,
            stats_table_output_path=panel.stats_table_output_path,
            x_axis_label="",
            show_y_tick_labels=bool(panel.show_y_tick_labels),
            **_tripanel_forest_panel_style(show_dataset_labels=bool(panel.show_y_tick_labels)),
        )
        if not panel.output_path.exists():
            missing_or_empty[panel.metric_slug] = f"Figure was not generated for {panel.metric_slug}."
            continue

        generated_files.append(str(panel.output_path.resolve()))
        if panel.stats_table_output_path is not None and panel.stats_table_output_path.exists():
            generated_files.append(str(panel.stats_table_output_path.resolve()))

        per_metric_figures[panel.metric_slug] = str(panel.output_path.resolve())
        panel_specs.append((panel.panel_letter, panel.metric_label, panel.output_path))

    return {
        "panel_specs": panel_specs,
        "per_metric_figures": per_metric_figures,
        "missing_or_empty": missing_or_empty,
        "top_row_dataset_labels": top_row_dataset_labels,
    }


def compose_publication_outcome_tripanel(
    *,
    panel_specs: Sequence[Tuple[str, str, Path]],
    figure_title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path],
    direction_label: str,
    generated_files: List[str],
    external_y_labels: Optional[Sequence[str]] = None,
    shared_x_label: str = CANONICAL_PUBLICATION_QE_CHANGE_LABEL,
    legend_band_height_override: int = PUBLICATION_LEGEND_ROW_HEIGHT,
) -> bool:
    if not panel_specs:
        return False

    generated = _compose_three_panel_stats_figure(
        panel_rows=[list(panel_specs)],
        title=figure_title,
        output_path=output_path,
        dpi=int(dpi),
        legend_path=legend_path if legend_path is not None and legend_path.exists() else None,
        legend_band_height_override=legend_band_height_override,
        direction_labels_by_row=[direction_label] if str(direction_label).strip() else None,
        shared_x_labels_by_row=[str(shared_x_label).strip()],
        external_y_labels_by_row=[list(external_y_labels)] if external_y_labels else None,
    )
    if generated and output_path.exists():
        generated_files.append(str(output_path.resolve()))
    return bool(generated)
