"""
Visualization utilities for FloatSOM benchmarking
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None  # cupy not available
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
try:
    import seaborn as sns
except ImportError:
    sns = None
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple
from datetime import datetime
import matplotlib.colors as mcolors

from floatsom_benchmarks.optuna.optuna_results_analysis.scripts.analysis.publication_figures.constants import (
    MARKER_SIZE_SCALE as PUBLICATION_MARKER_SIZE_SCALE,
    TEXT_SIZE_SCALE as PUBLICATION_TEXT_SIZE_SCALE,
)

PUBLICATION_DPI = 300
PUBLICATION_FIGSIZE = (14.0, 11.5)
SCALING_X_AXIS_RIGHT_PAD_RATIO = 0.10
SCALING_MARKER_SIZE = 8.8
TOPOLOGY_MARKER_SIZE = 8.2
LEGEND_MARKER_SCALE = 1.25
PUBLICATION_FONT_DELTA = 22.0
PUBLICATION_LEGEND_FONT_DELTA = 5.0
BASE_PUBLICATION_PLOT_FONTSIZE = 15.0
BASE_PUBLICATION_LEGEND_FONTSIZE = 15.0 + PUBLICATION_LEGEND_FONT_DELTA
UNIFIED_FIGURE_FONTSIZE = BASE_PUBLICATION_PLOT_FONTSIZE + PUBLICATION_FONT_DELTA
SCALING_AXIS_LABEL_FONTSIZE = UNIFIED_FIGURE_FONTSIZE
SCALING_AXIS_TICK_FONTSIZE = UNIFIED_FIGURE_FONTSIZE
SCALING_AXIS_TITLE_FONTSIZE = 2.0 * UNIFIED_FIGURE_FONTSIZE
PLOT_ANNOTATION_FONTSIZE = UNIFIED_FIGURE_FONTSIZE
BOTTOM_LEGEND_FONTSIZE = BASE_PUBLICATION_LEGEND_FONTSIZE
# Match the standalone scaling legends to the publication-figure legend renderer.
SHARED_LEGEND_VISUAL_SCALE = 1.0
SHARED_LEGEND_TEXT_SCALE = PUBLICATION_TEXT_SIZE_SCALE * SHARED_LEGEND_VISUAL_SCALE
SHARED_LEGEND_HANDLE_LENGTH_SCALE = 1.18
SHARED_FLAT_LEGEND_MAX_COLUMNS = 5
SHARED_FLAT_LEGEND_VISUAL_SCALE = 1.18
SHARED_METHOD_LEGEND_FONTSIZE = 11.0 * SHARED_LEGEND_TEXT_SCALE
SHARED_METHOD_HEADING_FONTSIZE = 11.8 * SHARED_LEGEND_TEXT_SCALE
SHARED_TOPOLOGY_LEGEND_FONTSIZE = 11.0 * SHARED_LEGEND_TEXT_SCALE
SHARED_TOPOLOGY_HEADING_FONTSIZE = 11.8 * SHARED_LEGEND_TEXT_SCALE
BASE_X_LABEL_PAD = 14
X_AXIS_LABEL_PAD = 2 * (BASE_X_LABEL_PAD + 15)
X_AXIS_TICK_LABEL_PAD = 9
AXIS_LINE_COLOR = '#000000'
AXIS_LINE_WIDTH = 1.2
SHADED_REGION_EDGE_COLOR = '#7F7F7F'
ESTIMATED_BASELINE_SHADE_COLOR = '#B8B8B8'
ESTIMATED_BASELINE_SHADE_ALPHA = 0.30
ESTIMATED_BASELINE_LEGEND_LABEL = 'Shaded area: 1-GPU extrapolated'
MISSING_TRAILING_X_SHADE_COLOR = '#B8B8B8'
MISSING_TRAILING_X_SHADE_ALPHA = 0.40
METHOD_BASE_COLORS = {
    'batch': '#00C853',
    'colors': '#FF6F00',
    'minibatch': '#CC79A7',
}
GPU_GREEN_VARIANT_PALETTE: Tuple[str, ...] = (
    '#1B5E20',
    '#0B6E4F',
    '#2E7D32',
    '#4F8A10',
    '#00A152',
    '#7CB342',
)
TOPOLOGY_BASE_COLORS = {
    'hexagonal': '#FF4FA3',
    'mst': '#00E5FF',
    'rng': '#8A2BE2',
    'grid': '#17BECF',
}
def create_output_directory(prefix: str) -> str:
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _apply_publication_theme() -> None:
    """Apply publication-oriented seaborn style used by Optuna analysis plots."""
    if sns is not None:
        sns.set_theme(style='whitegrid', context='talk', palette='colorblind')
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.28,
            'grid.linestyle': '--',
            'axes.axisbelow': True,
            'legend.frameon': False,
            'axes.edgecolor': AXIS_LINE_COLOR,
            'axes.labelcolor': AXIS_LINE_COLOR,
            'xtick.color': AXIS_LINE_COLOR,
            'ytick.color': AXIS_LINE_COLOR,
        }
    )


def _build_method_base_colors(methods: List[str]) -> Dict[str, Any]:
    """Build deterministic algorithm base colors."""
    base_colors: Dict[str, Any] = {}
    unknown_methods = []

    for method in methods:
        method_name = str(method)
        if method_name in METHOD_BASE_COLORS:
            base_colors[method_name] = METHOD_BASE_COLORS[method_name]
        else:
            unknown_methods.append(method_name)

    if unknown_methods:
        if sns is not None:
            fallback_palette = sns.color_palette('colorblind', n_colors=len(unknown_methods) + 3)
        else:
            fallback_palette = plt.cm.tab10(np.linspace(0, 1, len(unknown_methods) + 3))

        used_colors = {mcolors.to_hex(color) for color in base_colors.values()}
        fallback_idx = 0
        for method_name in sorted(unknown_methods):
            while fallback_idx < len(fallback_palette):
                candidate = mcolors.to_hex(fallback_palette[fallback_idx])
                fallback_idx += 1
                if candidate not in used_colors:
                    base_colors[method_name] = candidate
                    used_colors.add(candidate)
                    break
            if method_name not in base_colors:
                base_colors[method_name] = mcolors.to_hex(plt.cm.tab10(0))

    return base_colors


def _shade_method_color_for_gpu(
    base_color: Any,
    gpu_count: int,
    gpu_counts: List[int],
) -> Tuple[float, float, float]:
    """Encode GPU count within an algorithm hue family."""
    ordered_gpus = sorted(set(int(gpu) for gpu in gpu_counts))
    if not ordered_gpus:
        return mcolors.to_rgb(base_color)

    try:
        rank = ordered_gpus.index(int(gpu_count))
    except ValueError:
        rank = 0

    normalized_base_hex = mcolors.to_hex(base_color).lower()
    batch_base_hex = mcolors.to_hex(METHOD_BASE_COLORS['batch']).lower()
    if normalized_base_hex == batch_base_hex:
        if len(ordered_gpus) == 1:
            palette_idx = 0
        else:
            palette_positions = np.linspace(
                0,
                len(GPU_GREEN_VARIANT_PALETTE) - 1,
                num=len(ordered_gpus),
            )
            palette_idx = int(round(float(palette_positions[rank])))
        palette_idx = max(0, min(len(GPU_GREEN_VARIANT_PALETTE) - 1, palette_idx))
        return mcolors.to_rgb(GPU_GREEN_VARIANT_PALETTE[palette_idx])

    if len(ordered_gpus) == 1:
        blend_with_white = 0.2
    else:
        # Lowest GPU count: much lighter tint; highest: saturated base hue.
        t = rank / float(len(ordered_gpus) - 1)
        blend_with_white = 0.72 * (1.0 - t)

    r, g, b = mcolors.to_rgb(base_color)
    return (
        (1.0 - blend_with_white) * r + blend_with_white,
        (1.0 - blend_with_white) * g + blend_with_white,
        (1.0 - blend_with_white) * b + blend_with_white,
    )


def _build_gpu_marker_map(gpu_counts: List[int]) -> Dict[int, str]:
    """Build deterministic marker shapes for GPU-count series."""
    ordered_gpus = sorted(set(int(gpu) for gpu in gpu_counts))
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']
    return {
        gpu: marker_cycle[idx % len(marker_cycle)]
        for idx, gpu in enumerate(ordered_gpus)
    }


def _build_gpu_linewidth_map(gpu_counts: List[int]) -> Dict[int, float]:
    """Build deterministic line widths for GPU-count series."""
    ordered_gpus = sorted(set(int(gpu) for gpu in gpu_counts))
    if not ordered_gpus:
        return {}
    if len(ordered_gpus) == 1:
        return {ordered_gpus[0]: 2.4}
    return {
        gpu: 1.8 + (1.2 * idx / float(len(ordered_gpus) - 1))
        for idx, gpu in enumerate(ordered_gpus)
    }


def _build_method_marker_map(methods: List[str]) -> Dict[str, str]:
    """Build deterministic marker shapes for method series."""
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']
    return {
        method: marker_cycle[idx % len(marker_cycle)]
        for idx, method in enumerate(sorted(set(str(value) for value in methods)))
    }


def _build_topology_linestyle_map(topologies: List[str]) -> Dict[str, Any]:
    """Build deterministic line styles for topology series."""
    dash_cycle: List[Any] = [
        '-',
        '--',
        '-.',
        ':',
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (7, 2, 1, 2)),
        (0, (2, 2)),
    ]
    return {
        topology: dash_cycle[idx % len(dash_cycle)]
        for idx, topology in enumerate(sorted(set(str(value) for value in topologies)))
    }


def _build_topology_base_colors(topologies: List[str]) -> Dict[str, Any]:
    """Build deterministic base colors for topology series."""
    base_colors: Dict[str, Any] = {}
    unknown_topologies = []

    for topology in topologies:
        topology_name = str(topology)
        if topology_name in TOPOLOGY_BASE_COLORS:
            base_colors[topology_name] = TOPOLOGY_BASE_COLORS[topology_name]
        else:
            unknown_topologies.append(topology_name)

    if unknown_topologies:
        if sns is not None:
            fallback_palette = sns.color_palette('colorblind', n_colors=len(unknown_topologies) + 3)
        else:
            fallback_palette = plt.cm.tab10(np.linspace(0, 1, len(unknown_topologies) + 3))

        used_colors = {mcolors.to_hex(color) for color in base_colors.values()}
        fallback_idx = 0
        for topology_name in sorted(unknown_topologies):
            while fallback_idx < len(fallback_palette):
                candidate = mcolors.to_hex(fallback_palette[fallback_idx])
                fallback_idx += 1
                if candidate not in used_colors:
                    base_colors[topology_name] = candidate
                    used_colors.add(candidate)
                    break
            if topology_name not in base_colors:
                base_colors[topology_name] = mcolors.to_hex(plt.cm.tab10(0))

    return base_colors


def _build_method_linestyle_map(methods: List[str]) -> Dict[str, Any]:
    """Build deterministic line styles for method series."""
    dash_cycle: List[Any] = [
        '-',
        '--',
        '-.',
        ':',
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (7, 2, 1, 2)),
        (0, (2, 2)),
    ]
    return {
        method: dash_cycle[idx % len(dash_cycle)]
        for idx, method in enumerate(sorted(set(str(value) for value in methods)))
    }


def _resolve_speed_plot_methods(
    available_methods: List[str],
    methods_to_plot: Optional[List[str]] = None,
) -> List[str]:
    """
    Select which methods to display in speed benchmark plots.

    Default behavior (methods_to_plot=None): show only batch and colors.
    """
    if not available_methods:
        return []

    normalized_available = [
        (str(method), str(method).strip().lower())
        for method in available_methods
    ]

    target_methods = methods_to_plot
    if target_methods is None:
        target_methods = ['batch', 'colors']

    normalized_targets = [
        str(method).strip().lower()
        for method in target_methods
        if str(method).strip()
    ]
    if not normalized_targets:
        return [method for method, _ in normalized_available]

    selected: List[str] = []
    for target in normalized_targets:
        for original, normalized in normalized_available:
            if normalized == target and original not in selected:
                selected.append(original)

    return selected


def _normalize_method_name(method: str) -> str:
    """Normalize processing method names for stable labels."""
    return str(method).strip().lower()


def _method_display_name(method: str) -> str:
    """Human-friendly processing method label."""
    normalized = _normalize_method_name(method)
    if normalized == 'batch':
        return 'Batch (full)'
    if normalized == 'colors':
        return 'Colors'
    if normalized == 'minibatch':
        return 'Minibatch'
    return normalized.replace('_', ' ').title()


def _topology_display_name(topology: str) -> str:
    """Human-friendly topology label."""
    normalized = str(topology).strip().lower()
    lookup = {
        'hexagonal': 'Hexagonal',
        'mst': 'MST',
        'rng': 'RNG',
        'grid': 'Grid',
    }
    return lookup.get(normalized, normalized.replace('_', ' ').title())


def _resolve_topology_legend_column_style(topologies: Sequence[str]) -> Tuple[bool, bool]:
    """Return (include_topology_swatch, color_column_headings) for topology legends."""
    normalized = {
        str(topology).strip().lower()
        for topology in topologies
        if str(topology).strip()
    }
    if not normalized:
        return True, False
    # Topology is already encoded by legend column, so keep just method rows and color headings.
    return False, True


def _legend_registry_add_items(
    legend_registry: Dict[str, Dict[str, Any]],
    section_id: str,
    section_title: str,
    items: Sequence[Tuple[str, Dict[str, Any]]],
    heading_color: Optional[Any] = None,
) -> None:
    """Add unique legend entries to a sectioned registry."""
    section = legend_registry.setdefault(
        section_id,
        {
            'title': section_title,
            'items': OrderedDict(),
            'heading_color': None,
        },
    )
    if not section.get('title'):
        section['title'] = section_title
    if heading_color is not None:
        section['heading_color'] = heading_color

    section_items = section['items']
    for kind, kwargs in items:
        label = str(kwargs.get('label', '')).strip()
        if not label or label in section_items:
            continue
        section_items[label] = {
            'kind': str(kind),
            'kwargs': dict(kwargs),
        }


def _build_legend_handle_from_spec(spec: Dict[str, Any], *, scale: float = 1.0) -> Any:
    """Build a legend handle from a declarative style spec."""
    kind = str(spec.get('kind', 'line2d')).strip().lower()
    kwargs = dict(spec.get('kwargs', {}))
    if 'markersize' in kwargs and kwargs['markersize'] is not None:
        kwargs['markersize'] = float(kwargs['markersize']) * PUBLICATION_MARKER_SIZE_SCALE * float(scale)
    if 'linewidth' in kwargs and kwargs['linewidth'] is not None:
        kwargs['linewidth'] = float(kwargs['linewidth']) * 1.2 * float(scale)
    if kind == 'patch':
        return Patch(**kwargs)
    return Line2D([0], [0], **kwargs)


def _max_display_ticks() -> int:
    """Choose a conservative tick count to avoid overlap at large publication font sizes."""
    if SCALING_AXIS_TICK_FONTSIZE >= 32:
        return 4
    if SCALING_AXIS_TICK_FONTSIZE >= 24:
        return 5
    return 6


def _select_tick_subset(values: Sequence[float], max_ticks: Optional[int] = None) -> List[float]:
    """Select an evenly spaced subset of sorted numeric ticks."""
    ordered = sorted({float(value) for value in values})
    if not ordered:
        return []
    max_ticks = int(max_ticks if max_ticks is not None else _max_display_ticks())
    if max_ticks <= 0 or len(ordered) <= max_ticks:
        return ordered
    raw_indices = np.linspace(0, len(ordered) - 1, num=max_ticks, dtype=int)
    indices = sorted(set(int(idx) for idx in raw_indices))
    if indices[-1] != len(ordered) - 1:
        indices[-1] = len(ordered) - 1
    return [ordered[idx] for idx in indices]


def _render_grouped_legend_key(
    legend_registry: Dict[str, Dict[str, Any]],
    output_path: str,
    dpi: int = PUBLICATION_DPI,
    shaded_region_label: Optional[str] = None,
    shaded_region_color: str = ESTIMATED_BASELINE_SHADE_COLOR,
    shaded_region_alpha: float = ESTIMATED_BASELINE_SHADE_ALPHA,
) -> bool:
    """
    Render a standalone legend image with one column per logical section.

    This layout supports:
    - algorithm-grouped columns (section title = algorithm; rows = GPU counts)
    - topology-grouped columns (section title = topology; rows = topology swatch + methods)
    """
    sections = [
        section
        for section in legend_registry.values()
        if section.get('items')
    ]
    if not sections:
        return False

    n_sections = len(sections)
    max_items = max(len(section['items']) for section in sections)
    rows_per_column = max_items + 1  # +1 reserved heading row

    shade_label = str(shaded_region_label or '').strip()
    fig_width = max(14.0, 5.0 * n_sections)
    fig_height = max(2.2, 0.48 * rows_per_column + 0.9)
    legend_fontsize = SHARED_TOPOLOGY_LEGEND_FONTSIZE
    heading_fontsize = SHARED_TOPOLOGY_HEADING_FONTSIZE
    borderpad = 0.85
    labelspacing = 0.55
    handletextpad = 0.68
    handlelength = 2.8
    columnspacing = 2.0

    shade_row_height = 1.0
    if shade_label:
        fig_height += shade_row_height

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.axis('off')
    artists_for_tight_bbox: List[Any] = []

    def _blank_handle() -> Line2D:
        return Line2D([], [], linestyle='None', marker='', color='none')

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    heading_indices: List[int] = []

    # Build one standard legend where each column corresponds to one section.
    # First slot in each column is a placeholder for the column heading.
    for section in sections:
        section_specs = list(section['items'].values())
        section_handles = [_blank_handle()]
        section_labels = ['']

        section_handles.extend(
            _build_legend_handle_from_spec(spec, scale=SHARED_LEGEND_VISUAL_SCALE)
            for spec in section_specs
        )
        section_labels.extend(
            str(spec.get('kwargs', {}).get('label', '')).strip()
            for spec in section_specs
        )

        heading_indices.append(len(legend_labels))

        while len(section_handles) < rows_per_column:
            section_handles.append(_blank_handle())
            section_labels.append('')

        legend_handles.extend(section_handles)
        legend_labels.extend(section_labels)

    legend = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        ncol=n_sections,
        frameon=False,
        fontsize=legend_fontsize,
        borderpad=borderpad * SHARED_LEGEND_VISUAL_SCALE,
        labelspacing=labelspacing * SHARED_LEGEND_VISUAL_SCALE,
        handletextpad=handletextpad * SHARED_LEGEND_VISUAL_SCALE,
        handlelength=handlelength * SHARED_LEGEND_VISUAL_SCALE * SHARED_LEGEND_HANDLE_LENGTH_SCALE,
        columnspacing=columnspacing * SHARED_LEGEND_VISUAL_SCALE,
    )

    if legend is not None:
        artists_for_tight_bbox.append(legend)
        texts = legend.get_texts()
        for section_idx, section in enumerate(sections):
            heading_idx = heading_indices[section_idx]
            if heading_idx >= len(texts):
                continue
            heading_color = section.get('heading_color', '#1f1f1f')
            texts[heading_idx].set_text(str(section.get('title', '')).strip())
            texts[heading_idx].set_fontweight('bold')
            texts[heading_idx].set_fontsize(heading_fontsize)
            texts[heading_idx].set_color(heading_color)

    if shade_label:
        swatch_w = 0.032
        swatch_h = 0.11
        swatch_gap = 0.014
        swatch_y = 0.05
        shade_fontsize = max(legend_fontsize, heading_fontsize * 0.94)

        # Measure text width so the swatch+label group can be centered.
        text_probe = fig.text(
            0.0,
            0.0,
            shade_label,
            ha='left',
            va='center',
            fontsize=shade_fontsize,
            color='#1f1f1f',
            alpha=0.0,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_bbox = text_probe.get_window_extent(renderer=renderer)
        text_probe.remove()

        fig_width_px = max(1.0, float(fig.get_size_inches()[0]) * float(fig.dpi))
        text_width_fig = float(text_bbox.width) / fig_width_px
        total_width = swatch_w + swatch_gap + text_width_fig
        swatch_x = max(0.02, 0.5 - (total_width / 2.0))

        shade_patch = Rectangle(
            (swatch_x, swatch_y),
            swatch_w,
            swatch_h,
            transform=fig.transFigure,
            facecolor=shaded_region_color,
            edgecolor=SHADED_REGION_EDGE_COLOR,
            linewidth=3.2,
            alpha=shaded_region_alpha,
        )
        fig.add_artist(shade_patch)
        artists_for_tight_bbox.append(shade_patch)
        shade_text = fig.text(
            swatch_x + swatch_w + swatch_gap,
            swatch_y + (swatch_h / 2.0),
            shade_label,
            ha='left',
            va='center',
            fontsize=shade_fontsize,
            color='#1f1f1f',
        )
        artists_for_tight_bbox.append(shade_text)

    fig.tight_layout(
        pad=0.34 * SHARED_LEGEND_VISUAL_SCALE
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = None
    for artist in artists_for_tight_bbox:
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        tight_bbox = bbox if tight_bbox is None else tight_bbox.union([bbox])
    if tight_bbox is not None:
        tight_bbox = tight_bbox.expanded(1.02, 1.06)
        tight_bbox_inches = tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    else:
        tight_bbox_inches = 'tight'
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=tight_bbox_inches,
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)
    return True


def _render_flat_wrapped_legend_key(
    legend_registry: Dict[str, Dict[str, Any]],
    output_path: str,
    dpi: int = PUBLICATION_DPI,
    shaded_region_label: Optional[str] = None,
    shaded_region_color: str = ESTIMATED_BASELINE_SHADE_COLOR,
    shaded_region_alpha: float = ESTIMATED_BASELINE_SHADE_ALPHA,
) -> bool:
    """Render a two-row flat legend with centered GPU and shade rows."""
    sections = [
        section
        for section in legend_registry.values()
        if section.get('items')
    ]
    if not sections:
        return False

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    for section in sections:
        section_title = str(section.get('title', '')).strip()
        for spec in section.get('items', {}).values():
            item_label = str(spec.get('kwargs', {}).get('label', '')).strip()
            if item_label and item_label != section_title:
                label_text = f"{section_title}: {item_label}"
            else:
                label_text = section_title or item_label
            if not label_text:
                continue
            scaled_spec = {
                'kind': str(spec.get('kind', 'line2d')),
                'kwargs': dict(spec.get('kwargs', {})),
            }
            legend_handles.append(
                _build_legend_handle_from_spec(
                    scaled_spec,
                    scale=SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
                )
            )
            legend_labels.append(label_text)

    if not legend_handles:
        return False

    shade_label = str(shaded_region_label or '').strip()
    shade_handle: Optional[Any] = None
    if shade_label:
        shade_handle = Patch(
            facecolor=shaded_region_color,
            edgecolor=SHADED_REGION_EDGE_COLOR,
            linewidth=3.2 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
            alpha=shaded_region_alpha,
        )

    n_items = len(legend_handles)
    ncol = min(SHARED_FLAT_LEGEND_MAX_COLUMNS, n_items)
    legend_fontsize = SHARED_METHOD_LEGEND_FONTSIZE * SHARED_FLAT_LEGEND_VISUAL_SCALE
    fig_width = max(13.0, 3.25 * ncol)
    fig_height = 2.05 if shade_handle is not None else 1.75

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.axis('off')

    artists_for_tight_bbox: List[Any] = []
    gpu_legend = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        bbox_to_anchor=(0.5, 0.60 if shade_handle is not None else 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        borderpad=0.50 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
        labelspacing=0.58 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
        handletextpad=0.60 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
        handlelength=2.25 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
        columnspacing=1.30 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
    )
    if gpu_legend is not None:
        artists_for_tight_bbox.append(gpu_legend)

    if shade_handle is not None:
        shade_legend = fig.legend(
            handles=[shade_handle],
            labels=[shade_label],
            loc='center',
            bbox_to_anchor=(0.5, 0.32),
            ncol=1,
            frameon=False,
            fontsize=legend_fontsize,
            borderpad=0.28 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
            labelspacing=0.32 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
            handletextpad=0.44 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
            handlelength=1.95 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
            columnspacing=1.0 * SHARED_LEGEND_VISUAL_SCALE * SHARED_FLAT_LEGEND_VISUAL_SCALE,
        )
        if shade_legend is not None:
            artists_for_tight_bbox.append(shade_legend)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        bbox_extra_artists=artists_for_tight_bbox if artists_for_tight_bbox else None,
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)
    return True


def _apply_scaling_axis_typography(ax: Any) -> None:
    """Increase axis label/tick typography for panel-composed publication figures."""
    ax.tick_params(axis='both', labelsize=SCALING_AXIS_TICK_FONTSIZE)
    ax.tick_params(axis='x', pad=X_AXIS_TICK_LABEL_PAD)
    ax.xaxis.label.set_size(SCALING_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(SCALING_AXIS_LABEL_FONTSIZE)
    ax.title.set_size(SCALING_AXIS_TITLE_FONTSIZE)
    ax.set_title('')


def _apply_axis_line_style(ax: Any) -> None:
    """Standardize axis/tick styling for publication figures."""
    for spine_name in ('left', 'bottom'):
        spine = ax.spines.get(spine_name)
        if spine is None:
            continue
        spine.set_color(AXIS_LINE_COLOR)
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis='both', colors=AXIS_LINE_COLOR, labelcolor=AXIS_LINE_COLOR)
    ax.xaxis.label.set_color(AXIS_LINE_COLOR)
    ax.yaxis.label.set_color(AXIS_LINE_COLOR)
    ax.title.set_color(AXIS_LINE_COLOR)


def _register_method_gpu_legend_item(
    legend_registry: Dict[str, Dict[str, Any]],
    method: str,
    gpu_count: int,
    series_color: Any,
    marker_style: str,
    series_linewidth: float,
    linestyle: Any,
) -> None:
    """Register one algorithm/GPU series in a sectioned legend registry."""
    gpu_label = f'{gpu_count} GPU{"s" if gpu_count > 1 else ""}'
    section_id = gpu_label.lower()
    section_title = gpu_label
    _legend_registry_add_items(
        legend_registry=legend_registry,
        section_id=section_id,
        section_title=section_title,
        items=[
            (
                'line2d',
                {
                    'label': _method_display_name(method),
                    'color': series_color,
                    'linestyle': linestyle,
                    'marker': marker_style,
                    'linewidth': series_linewidth,
                    'markersize': 6.8,
                },
            )
        ],
    )


def _register_topology_method_legend_item(
    legend_registry: Dict[str, Dict[str, Any]],
    topology: str,
    method: str,
    series_color: Any,
    marker_style: str,
    series_linestyle: Any,
    include_topology_swatch: bool = True,
    color_column_heading: bool = False,
) -> None:
    """Register topology-comparison entries with topology swatch + method rows."""
    topology_key = str(topology).strip().lower()
    topology_label = _topology_display_name(topology)

    # First entry per section: topology swatch at the top of the column.
    if include_topology_swatch:
        _legend_registry_add_items(
            legend_registry=legend_registry,
            section_id=topology_key,
            section_title=topology_label,
            items=[
                (
                    'line2d',
                    {
                        'label': topology_label,
                        'color': series_color,
                        'linestyle': '-',
                        'marker': '',
                        'linewidth': 2.6,
                    },
                )
            ],
        )
    _legend_registry_add_items(
        legend_registry=legend_registry,
        section_id=topology_key,
        section_title=topology_label,
        items=[
            (
                'line2d',
                {
                    'label': _method_display_name(method),
                    'color': series_color,
                    'linestyle': series_linestyle,
                    'marker': marker_style,
                    'linewidth': 2.2,
                    'markersize': 6.8,
                },
            )
        ],
        heading_color=(series_color if color_column_heading else None),
    )


def plot_topology_comparison_scaling_benchmark(
    results_by_topology: Dict[str, Dict[str, Dict[Any, Any]]],
    output_dir: str,
    title: str = "Topology Comparison",
    axis_mode: str = "dimension",
    comparison_gpu_count: int = 8,
    show_speedup: bool = True,
    show_error_bars: bool = False,
    show_legend: bool = True,
    methods_to_plot: Optional[List[str]] = None,
    emit_shared_legend: bool = False,
    legend_filename: str = "legend_key.svg",
    include_performance: bool = True,
    topology_color_overrides: Optional[Dict[str, Any]] = None,
    method_linestyle_overrides: Optional[Dict[str, Any]] = None,
    sample_right_pad_ratio: float = 0.03,
    sample_min_upper_bound: Optional[float] = None,
    dimension_min_upper_bound: Optional[float] = None,
    bottom_legend_fontsize_scale: float = 1.0,
    bottom_legend_markerscale: float = 1.0,
    highlight_trailing_missing_x_range: bool = False,
):
    """
    Plot cross-topology comparisons at a fixed GPU count.

    Visual encoding:
    - topology -> color
    - algorithm/method -> line style

    Args:
        results_by_topology: {topology -> {method -> {axis_value -> {gpu -> stats}}}}
        output_dir: Directory to save plots.
        title: Figure title prefix.
        axis_mode: One of {'dimension', 'sample', 'grid_size'}.
        comparison_gpu_count: Fixed GPU count to compare (typically 8).
        show_speedup: Whether to generate speedup and efficiency plots.
        show_error_bars: Whether to draw runtime error bars (std).
        show_legend: Whether to include legend.
        methods_to_plot: Optional method list to plot; default shows ['batch', 'colors'].
        emit_shared_legend: Whether to emit a standalone grouped legend SVG.
        legend_filename: Standalone legend filename written under output_dir.
        include_performance: Whether to emit the runtime/performance panel.
        topology_color_overrides: Optional color map keyed by topology labels.
        method_linestyle_overrides: Optional line-style overrides keyed by method label.
        sample_min_upper_bound: Optional minimum sample-axis upper bound.
        dimension_min_upper_bound: Optional minimum dimension-axis upper bound.
        bottom_legend_fontsize_scale: Optional bottom legend text scale multiplier.
        bottom_legend_markerscale: Optional bottom legend marker scale multiplier.
        highlight_trailing_missing_x_range: When true, shade the x-range beyond
            the last observed point up to the configured panel maximum.
    """
    if not results_by_topology:
        return

    topologies = [
        str(topology)
        for topology, topology_payload in sorted(results_by_topology.items())
        if isinstance(topology_payload, dict) and topology_payload
    ]
    if not topologies:
        return
    (
        legend_use_topology_swatch,
        legend_color_column_headings,
    ) = _resolve_topology_legend_column_style(topologies)

    methods = sorted(
        {
            str(method)
            for topology in topologies
            for method in results_by_topology.get(topology, {}).keys()
        }
    )
    methods = _resolve_speed_plot_methods(methods, methods_to_plot=methods_to_plot)
    if not methods:
        return

    axis_values = sorted(
        {
            axis_numeric
            for topology in topologies
            for method in methods
            for axis_key in results_by_topology.get(topology, {}).get(method, {}).keys()
            for axis_numeric in [_parse_numeric_key(axis_key)]
            if axis_numeric is not None
        }
    )
    if not axis_values:
        return

    os.makedirs(output_dir, exist_ok=True)
    _apply_publication_theme()

    topology_base_colors = _build_topology_base_colors(topologies)
    if topology_color_overrides:
        for topology_name, color in topology_color_overrides.items():
            normalized_name = str(topology_name).strip()
            if not normalized_name:
                continue
            topology_base_colors[normalized_name] = color
    method_markers = _build_method_marker_map(methods)
    method_linestyles = _build_method_linestyle_map(methods)
    if method_linestyle_overrides:
        for method_name, linestyle in method_linestyle_overrides.items():
            normalized_name = str(method_name).strip()
            if not normalized_name:
                continue
            method_linestyles[normalized_name] = linestyle
    topology_legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    estimated_baseline_used = False

    axis_key_maps: Dict[str, Dict[str, Dict[float, List[Any]]]] = {}
    for topology in topologies:
        axis_key_maps[topology] = {}
        for method in methods:
            method_results = results_by_topology.get(topology, {}).get(method, {})
            if isinstance(method_results, dict):
                axis_key_maps[topology][method] = _build_numeric_key_map(method_results)
            else:
                axis_key_maps[topology][method] = {}

    prefix = 'sample_scaling' if axis_mode == 'sample' else 'gpu_scaling'

    def _axis_is_log(values: List[float]) -> bool:
        if not values:
            return False
        if axis_mode == 'sample':
            return len(values) > 1 and min(values) > 0 and (max(values) / min(values)) > 100
        return len(values) > 1 and min(values) > 0

    def _configure_x_axis(ax, values: List[float]) -> bool:
        use_log = _axis_is_log(values)
        display_ticks = _select_tick_subset(values)
        tick_values = [int(v) if float(v).is_integer() else float(v) for v in display_ticks]
        tick_labels = [f"{int(v):,}" if float(v).is_integer() else f"{v:g}" for v in display_ticks]
        if axis_mode == 'sample':
            ax.set_xscale('log' if use_log else 'linear')
            ax.set_xticks(tick_values)
            if use_log:
                ax.set_xticklabels(
                    [_format_sample_tick_label(float(v), use_log=True) for v in display_ticks]
                )
            else:
                ax.set_xticklabels(tick_labels)
            _set_sample_x_limits(
                ax,
                values,
                right_pad_ratio=sample_right_pad_ratio,
                min_upper_bound=sample_min_upper_bound,
            )
            ax.set_xlabel(
                'Number of Samples (log scale)' if use_log else 'Number of Samples',
                labelpad=X_AXIS_LABEL_PAD,
            )
        elif axis_mode == 'dimension':
            _set_dimension_x_axis(ax, values, min_upper_bound=dimension_min_upper_bound)
            ax.set_xlabel(
                'Number of Dimensions (log scale)' if use_log else 'Number of Dimensions',
                labelpad=X_AXIS_LABEL_PAD,
            )
        else:
            use_log = False
            ax.set_xscale('linear')
            ax.set_xticks(tick_values)
            ax.set_xticklabels(tick_labels)
            if len(values) > 1:
                min_v = float(min(values))
                max_v = float(max(values))
                pad = (max_v - min_v) * 0.04
                ax.set_xlim(min_v - pad, max_v + pad)
            ax.set_xlabel('Grid Size', labelpad=X_AXIS_LABEL_PAD)

        _apply_scaling_axis_typography(ax)
        return use_log

    if include_performance:
        # Performance plot
        fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
        plotted_performance_x: List[float] = []
        for topology in topologies:
            for method in methods:
                method_results = results_by_topology.get(topology, {}).get(method, {})
                if not isinstance(method_results, dict):
                    continue

                axis_key_map = axis_key_maps[topology].get(method, {})
                times: List[float] = []
                errors: List[float] = []
                valid_axis: List[float] = []

                for axis_value in axis_values:
                    axis_key = _resolve_numeric_key(axis_value, method_results, axis_key_map)
                    if axis_key is None:
                        continue
                    axis_payload = method_results.get(axis_key, {})
                    if not isinstance(axis_payload, dict):
                        continue
                    current_key = _resolve_gpu_key(comparison_gpu_count, axis_payload)
                    if current_key is None:
                        continue

                    current = axis_payload[current_key]
                    if isinstance(current, dict):
                        mean_val = current.get('mean')
                        if mean_val is None and current.get('times'):
                            mean_val = float(np.mean(current['times']))
                        std_val = current.get('std')
                        if std_val is None and current.get('times'):
                            std_val = float(np.std(current['times']))
                        if mean_val is None:
                            continue
                        times.append(float(mean_val))
                        errors.append(float(std_val) if std_val is not None else 0.0)
                    else:
                        mean_val = _extract_mean_from_result(current)
                        if mean_val is None:
                            continue
                        times.append(float(mean_val))
                        errors.append(0.0)

                    axis_for_plot = int(axis_value) if float(axis_value).is_integer() else axis_value
                    valid_axis.append(axis_for_plot)

                if not times:
                    continue

                plotted_performance_x.extend(float(value) for value in valid_axis)
                series_color = topology_base_colors[topology]
                series_marker = method_markers.get(method, 'o')
                series_linestyle = method_linestyles.get(method, '-')
                label = f"{_topology_display_name(topology)} - {_method_display_name(method)}"
                _register_topology_method_legend_item(
                    legend_registry=topology_legend_registry,
                    topology=topology,
                    method=method,
                    series_color=series_color,
                    marker_style=series_marker,
                    series_linestyle=series_linestyle,
                    include_topology_swatch=legend_use_topology_swatch,
                    color_column_heading=legend_color_column_headings,
                )

                if show_error_bars and any(error > 0 for error in errors):
                    ax.errorbar(
                        valid_axis,
                        times,
                        yerr=errors,
                        marker=series_marker,
                        markersize=TOPOLOGY_MARKER_SIZE,
                        linewidth=2.2,
                        color=series_color,
                        capsize=5,
                        linestyle=series_linestyle,
                        label=label,
                    )
                else:
                    ax.plot(
                        valid_axis,
                        times,
                        marker=series_marker,
                        markersize=TOPOLOGY_MARKER_SIZE,
                        linewidth=2.2,
                        color=series_color,
                        linestyle=series_linestyle,
                        label=label,
                    )

        ax.set_ylabel('Time (s)', labelpad=12)
        _configure_x_axis(ax, axis_values)
        if highlight_trailing_missing_x_range:
            _highlight_trailing_missing_x_range(
                ax=ax,
                plotted_axis_values=plotted_performance_x,
            )
        ax.grid(True, alpha=0.28, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        if sns is not None:
            sns.despine(ax=ax)
        _finalize_layout_with_bottom_legend(
            fig,
            ax,
            show_legend=show_legend,
            legend_fontsize_scale=bottom_legend_fontsize_scale,
            legend_markerscale=bottom_legend_markerscale,
        )

        performance_file = os.path.join(output_dir, f'{prefix}_performance.svg')
        plt.savefig(performance_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.savefig(performance_file.replace('.svg', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Topology comparison performance plot saved to: {performance_file}")

    if not show_speedup or comparison_gpu_count <= 1:
        if emit_shared_legend:
            legend_path = os.path.join(output_dir, legend_filename)
            if _render_grouped_legend_key(
                legend_registry=topology_legend_registry,
                output_path=legend_path,
                dpi=PUBLICATION_DPI,
                shaded_region_label=None,
            ):
                print(f"Topology comparison legend saved to: {legend_path}")
        return

    # Speedup plot
    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    has_speedup_data = False
    estimated_axis_values: Set[float] = set()
    plotted_speedup_x: List[float] = []

    for topology in topologies:
        for method in methods:
            method_results = results_by_topology.get(topology, {}).get(method, {})
            if not isinstance(method_results, dict):
                continue

            axis_key_map = axis_key_maps[topology].get(method, {})
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=axis_key_map,
                baseline_gpu=1,
            )

            speedups: List[float] = []
            valid_axis: List[float] = []
            for axis_value in axis_values:
                axis_key = _resolve_numeric_key(axis_value, method_results, axis_key_map)
                if axis_key is None:
                    continue
                axis_payload = method_results.get(axis_key, {})
                if not isinstance(axis_payload, dict):
                    continue
                current_key = _resolve_gpu_key(comparison_gpu_count, axis_payload)
                if current_key is None:
                    continue

                current_mean = _extract_mean_from_result(axis_payload[current_key])
                baseline_mean, is_estimated = _resolve_baseline_mean(
                    axis_numeric=axis_value,
                    axis_payload=axis_payload,
                    baseline_predictor=baseline_predictor,
                    baseline_gpu=1,
                )
                if baseline_mean is None or current_mean is None or current_mean == 0:
                    continue

                speedup = baseline_mean / current_mean
                if is_estimated:
                    estimated_axis_values.add(float(axis_value))
                axis_for_plot = int(axis_value) if float(axis_value).is_integer() else axis_value
                speedups.append(speedup)
                valid_axis.append(axis_for_plot)

            if not speedups:
                continue

            has_speedup_data = True
            plotted_speedup_x.extend(float(value) for value in valid_axis)
            series_color = topology_base_colors[topology]
            series_marker = method_markers.get(method, 'o')
            series_linestyle = method_linestyles.get(method, '-')
            _register_topology_method_legend_item(
                legend_registry=topology_legend_registry,
                topology=topology,
                method=method,
                series_color=series_color,
                marker_style=series_marker,
                series_linestyle=series_linestyle,
                include_topology_swatch=legend_use_topology_swatch,
                color_column_heading=legend_color_column_headings,
            )
            ax.plot(
                valid_axis,
                speedups,
                marker=series_marker,
                markersize=TOPOLOGY_MARKER_SIZE,
                linewidth=2.2,
                color=series_color,
                linestyle=series_linestyle,
                label=f"{_topology_display_name(topology)} - {_method_display_name(method)}",
            )

    if has_speedup_data:
        baseline_note = (
            '* Estimated 1-GPU baseline used where missing'
            if estimated_axis_values
            else None
        )
        estimated_baseline_used = estimated_baseline_used or bool(estimated_axis_values)
        use_log_axis = _axis_is_log(plotted_speedup_x or axis_values)
        ideal_anchor_x = _left_annotation_anchor_x(
            plotted_speedup_x or axis_values,
            use_log=use_log_axis,
        )
        ax.axhline(
            y=comparison_gpu_count,
            color='#6a6a6a',
            linestyle='--',
            alpha=0.55,
            linewidth=1.2,
        )
        if ideal_anchor_x is not None:
            ax.annotate(
                f'Ideal {comparison_gpu_count}x',
                xy=(ideal_anchor_x, comparison_gpu_count),
                xytext=(4, 2),
                textcoords='offset points',
                va='bottom',
                ha='left',
                color='#4a4a4a',
                fontsize=PLOT_ANNOTATION_FONTSIZE,
                fontweight='semibold',
            )

        ax.set_ylabel('Speedup Factor', labelpad=12)
        use_log_axis_for_plot = _configure_x_axis(ax, axis_values)
        _highlight_estimated_baseline_regions(
            ax=ax,
            estimated_axis_values=estimated_axis_values,
            all_axis_values=axis_values,
            use_log=use_log_axis_for_plot,
        )
        ax.grid(True, alpha=0.28, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=1)
        if sns is not None:
            sns.despine(ax=ax)
        _finalize_layout_with_bottom_legend(
            fig,
            ax,
            show_legend=show_legend,
            note_text=None,
            shaded_region_label=(
                ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
            ),
        )

        speedup_file = os.path.join(output_dir, f'{prefix}_speedup.svg')
        plt.savefig(speedup_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.savefig(speedup_file.replace('.svg', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Topology comparison speedup plot saved to: {speedup_file}")
    else:
        plt.close(fig)

    # Efficiency plot
    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    has_efficiency_data = False
    all_efficiencies: List[float] = []
    plotted_efficiency_x: List[float] = []
    estimated_axis_values = set()

    for topology in topologies:
        for method in methods:
            method_results = results_by_topology.get(topology, {}).get(method, {})
            if not isinstance(method_results, dict):
                continue

            axis_key_map = axis_key_maps[topology].get(method, {})
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=axis_key_map,
                baseline_gpu=1,
            )

            efficiencies: List[float] = []
            valid_axis: List[float] = []
            for axis_value in axis_values:
                axis_key = _resolve_numeric_key(axis_value, method_results, axis_key_map)
                if axis_key is None:
                    continue
                axis_payload = method_results.get(axis_key, {})
                if not isinstance(axis_payload, dict):
                    continue
                current_key = _resolve_gpu_key(comparison_gpu_count, axis_payload)
                if current_key is None:
                    continue

                current_mean = _extract_mean_from_result(axis_payload[current_key])
                baseline_mean, is_estimated = _resolve_baseline_mean(
                    axis_numeric=axis_value,
                    axis_payload=axis_payload,
                    baseline_predictor=baseline_predictor,
                    baseline_gpu=1,
                )
                if baseline_mean is None or current_mean is None or current_mean == 0:
                    continue

                speedup = baseline_mean / current_mean
                efficiency = (speedup / comparison_gpu_count) * 100
                if is_estimated:
                    estimated_axis_values.add(float(axis_value))
                axis_for_plot = int(axis_value) if float(axis_value).is_integer() else axis_value
                efficiencies.append(efficiency)
                valid_axis.append(axis_for_plot)

            if not efficiencies:
                continue

            has_efficiency_data = True
            all_efficiencies.extend(efficiencies)
            plotted_efficiency_x.extend(float(value) for value in valid_axis)
            ax.plot(
                valid_axis,
                efficiencies,
                marker=method_markers.get(method, 'o'),
                markersize=TOPOLOGY_MARKER_SIZE,
                linewidth=2.2,
                color=topology_base_colors[topology],
                linestyle=method_linestyles.get(method, '-'),
                label=f"{_topology_display_name(topology)} - {_method_display_name(method)}",
            )

    if has_efficiency_data:
        baseline_note = (
            '* Estimated 1-GPU baseline used where missing'
            if estimated_axis_values
            else None
        )
        estimated_baseline_used = estimated_baseline_used or bool(estimated_axis_values)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
        use_log_axis = _axis_is_log(plotted_efficiency_x or axis_values)
        reference_x = _left_annotation_anchor_x(
            plotted_efficiency_x or axis_values,
            use_log=use_log_axis,
        )
        if reference_x is not None:
            ax.annotate(
                'Perfect Scaling',
                xy=(reference_x, 100),
                xytext=(4, 2),
                textcoords='offset points',
                va='bottom',
                ha='left',
                color='green',
                fontsize=PLOT_ANNOTATION_FONTSIZE,
            )

        ax.set_ylabel('Scaling Efficiency (%)', labelpad=12)
        use_log_axis_for_plot = _configure_x_axis(ax, axis_values)
        _highlight_estimated_baseline_regions(
            ax=ax,
            estimated_axis_values=estimated_axis_values,
            all_axis_values=axis_values,
            use_log=use_log_axis_for_plot,
        )
        ax.grid(True, alpha=0.28, linestyle='--')
        ax.set_axisbelow(True)
        y_min, y_max = _auto_efficiency_ylim(all_efficiencies)
        ax.set_ylim(y_min, y_max)
        if sns is not None:
            sns.despine(ax=ax)
        _finalize_layout_with_bottom_legend(
            fig,
            ax,
            show_legend=show_legend,
            note_text=None,
            shaded_region_label=(
                ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
            ),
        )

        efficiency_file = os.path.join(output_dir, f'{prefix}_efficiency.svg')
        plt.savefig(efficiency_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.savefig(efficiency_file.replace('.svg', '.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Topology comparison efficiency plot saved to: {efficiency_file}")
    else:
        plt.close(fig)

    if emit_shared_legend:
        legend_path = os.path.join(output_dir, legend_filename)
        if _render_grouped_legend_key(
            legend_registry=topology_legend_registry,
            output_path=legend_path,
            dpi=PUBLICATION_DPI,
            shaded_region_label=(
                ESTIMATED_BASELINE_LEGEND_LABEL if estimated_baseline_used else None
            ),
        ):
            print(f"Topology comparison legend saved to: {legend_path}")


def visualize_som_grid_overlay(data_np: np.ndarray, weights_np: np.ndarray, 
                             title: str, output_file: str, mst_edges=None):
    """
    Create grid overlay visualization showing data and SOM nodes
    
    Args:
        data_np: Data points (numpy array)
        weights_np: SOM node weights (numpy array)
        title: Title for the plot
        output_file: Path to save the visualization
        mst_edges: Optional MST edges for MST topology
    """
    if data_np.shape[1] != 2:
        print(f"Skipping grid overlay visualization - data is {data_np.shape[1]}D, need 2D")
        return
        
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(data_np[:, 0], data_np[:, 1], alpha=0.3, s=10, c='gray', label='Data')
    
    # Plot SOM nodes
    plt.scatter(weights_np[:, 0], weights_np[:, 1], c='red', s=100, marker='o', 
                edgecolors='black', linewidths=1, label='SOM nodes', zorder=5)
    
    # If MST edges provided, draw them
    if mst_edges is not None:
        for i, j in mst_edges:
            plt.plot([weights_np[i, 0], weights_np[j, 0]], 
                    [weights_np[i, 1], weights_np[j, 1]], 
                    'k-', alpha=0.5, linewidth=1)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_quantization_error(errors: List[float], title: str, output_file: str):
    """
    Plot quantization error over iterations
    
    Args:
        errors: List of quantization errors
        title: Title for the plot
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o')
    plt.title(title)
    plt.xlabel('Save Point')
    plt.ylabel('Quantization Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_combined_quantization_errors(results: Dict[str, Dict[str, Any]], 
                                    title: str, output_file: str):
    """
    Plot quantization errors for multiple configurations on the same plot
    
    Args:
        results: Dictionary of results with quantization error curves
        title: Title for the plot
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for config_name, result in results.items():
        if 'quant_errors' in result and len(result['quant_errors']) > 1:
            iterations = list(range(0, len(result['quant_errors']) * 10, 10))
            plt.plot(iterations, result['quant_errors'], 
                    marker='o', markersize=4, label=config_name, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Quantization Error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def create_cluster_visualization(bmu_indices: np.ndarray, unique_clusters: np.ndarray,
                               height: int, width: int, title: str, output_file: str,
                               grid_size: Optional[int] = None):
    """
    Create cluster visualization showing BMU assignments
    
    Args:
        bmu_indices: BMU indices for each data point
        unique_clusters: Unique cluster/BMU indices
        height: Height of the data grid (for reshaping)
        width: Width of the data grid (for reshaping)
        title: Title for the plot
        output_file: Path to save the plot
        grid_size: Optional grid size for regular topologies
    """
    # Create color map for clusters
    n_clusters = len(unique_clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    
    if n_clusters > 20:
        # Use additional color maps for more clusters
        extra_colors = plt.cm.tab20b(np.linspace(0, 1, min(n_clusters - 20, 20)))
        colors = np.vstack([colors, extra_colors])
    
    # Create cluster map
    cluster_map = np.zeros(bmu_indices.shape[0])
    for i, cluster_id in enumerate(unique_clusters):
        cluster_map[bmu_indices == cluster_id] = i
    
    plt.figure(figsize=(10, 8))
    
    # If we have a grid topology, reshape and show as image
    if grid_size is not None and len(bmu_indices) == grid_size * grid_size:
        cluster_grid = cluster_map.reshape(grid_size, grid_size)
        plt.imshow(cluster_grid, cmap='tab20', interpolation='nearest')
        plt.colorbar(label='Cluster ID')
    else:
        # Otherwise show as scatter plot
        plt.scatter(range(len(cluster_map)), cluster_map, 
                   c=cluster_map, cmap='tab20', alpha=0.6)
        plt.ylabel('Cluster ID')
        plt.xlabel('Sample Index')
    
    plt.title(f"{title} - Cluster Assignments ({n_clusters} clusters)")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def visualize_results(results: Dict[str, Dict[str, Any]], data_np: np.ndarray, 
                     topology: str = 'rectangular'):
    """
    Create comprehensive visualization of results including U-matrix
    
    Args:
        results: Dictionary of training results
        data_np: Original data as numpy array
        topology: Topology type for visualization
    """
    # This is a placeholder for U-matrix visualization
    # U-matrix visualization would require additional implementation
    print("U-matrix visualization not yet implemented for FloatSOM")


def create_comparison_visualizations(
    results: Dict[str, Dict[str, Any]],
    data: cp.ndarray,
    output_dir: str,
    title_prefix: str
) -> None:
    """
    Create visualizations comparing the different FloatSOM configurations
    
    Args:
        results: Dictionary with benchmark results
        data: Input data used for training
        output_dir: Directory to save visualizations
        title_prefix: Prefix for visualization titles
    """
    if not results:
        print("No results to visualize")
        return
    
    # Convert data to numpy if needed
    data_np = data.get() if hasattr(data, 'get') else data
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Training time comparison
    plt.figure(figsize=(12, 6))
    methods = []
    times = []
    
    for name, result in results.items():
        methods.append(name)
        times.append(result['train_time'])
    
    # Sort by training time
    idx = np.argsort(times)
    methods = [methods[i] for i in idx]
    times = [times[i] for i in idx]
    
    plt.barh(methods, times)
    plt.xlabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"{title_prefix}_training_time_comparison.png"))
    plt.close()
    
    # 2. Quantization error comparison
    plt.figure(figsize=(12, 6))
    methods = []
    errors = []
    
    for name, result in results.items():
        methods.append(name)
        errors.append(result['quant_error'])
    
    # Sort by quantization error
    idx = np.argsort(errors)
    methods = [methods[i] for i in idx]
    errors = [errors[i] for i in idx]
    
    plt.barh(methods, errors)
    plt.xlabel('Quantization Error')
    plt.title(f'{title_prefix} - Quantization Error Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"{title_prefix}_quantization_error_comparison.png"))
    plt.close()
    
    # 3. Create combined plot with all performance metrics
    plt.figure(figsize=(15, 10))
    
    # Plot training time
    plt.subplot(2, 1, 1)
    plt.barh(methods, times)
    plt.xlabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time')
    plt.tight_layout()
    
    # Plot quantization error
    plt.subplot(2, 1, 2)
    plt.barh(methods, errors)
    plt.xlabel('Quantization Error')
    plt.title(f'{title_prefix} - Quantization Error')
    plt.tight_layout()
    
    plt.savefig(os.path.join(comparison_dir, f"{title_prefix}_combined_performance.png"))
    plt.close()
    
    # 4. Visualize weights for each method
    for name, result in results.items():
        weights = result['weights']
        weights_np = weights.get() if hasattr(weights, 'get') else weights
        
        # Create grid overlay visualization
        visualize_som_grid_overlay(
            data_np,
            weights_np,
            name,
            os.path.join(comparison_dir, f"{title_prefix}_{name}_grid_overlay.png")
        )
    
    # 5. Combined quantization error curves
    plot_combined_quantization_errors(
        results, 
        f"{title_prefix} - Quantization Error Curves",
        os.path.join(comparison_dir, f"{title_prefix}_combined_error_curves.png")
    )
    
    # Write summary report
    with open(os.path.join(output_dir, f"{title_prefix}_benchmark_summary.txt"), "w") as f:
        f.write(f"FloatSOM Benchmark Summary ({title_prefix})\n")
        f.write("=" * 60 + "\n\n")
        
        # Table header
        f.write(f"{'Configuration':<30} {'Architecture':<40} {'Time (s)':<10} {'QE':<10}\n")
        f.write("-" * 90 + "\n")
        
        # Table content
        for name, result in results.items():
            architecture = f"{result['sampling']}×{result['processing']}×{result['topology']}"
            f.write(f"{name:<30} {architecture:<40} ")
            f.write(f"{result['train_time']:<10.3f} {result['quant_error']:<10.6f}\n")
        
        f.write("\n\n")
        f.write("Detailed Configuration Settings:\n")
        f.write("-" * 50 + "\n")
        
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  - Sampling Method: {result['sampling']}\n")
            f.write(f"  - Processing Method: {result['processing']}\n")
            f.write(f"  - Topology: {result['topology']}\n")
            f.write(f"  - Training Time: {result['train_time']:.3f}s\n")
            f.write(f"  - Quantization Error: {result['quant_error']:.6f}\n")
            if 'params' in result:
                f.write(f"  - Grid Size: {result['params'].topology_config.grid_size}\n")
                f.write(f"  - Iterations: {result['params'].total_iterations}\n")
                f.write(f"  - Initial LR: {result['params'].initial_learning_rate}\n")


def create_animation_from_iterations(config_dir: str, config_name: str, 
                                   output_file: str = None, fps: int = 10):
    """
    Create an animation from saved iteration visualizations
    
    Args:
        config_dir: Directory containing iteration images
        config_name: Name of the configuration
        output_file: Output file for animation (if None, uses default)
        fps: Frames per second for animation
    """
    try:
        import imageio
        
        # Get list of iteration files
        iter_dir = os.path.join(config_dir, f"{config_name}_iterations")
        if not os.path.exists(iter_dir):
            print(f"No iteration directory found at {iter_dir}")
            return
        
        # Get sorted list of iteration files
        files = sorted([f for f in os.listdir(iter_dir) if f.startswith('iteration_')])
        if not files:
            print(f"No iteration files found in {iter_dir}")
            return
        
        # Read images
        images = []
        for filename in files:
            images.append(imageio.imread(os.path.join(iter_dir, filename)))
        
        # Create output filename if not provided
        if output_file is None:
            output_file = os.path.join(config_dir, f"{config_name}_training_animation.gif")
        
        # Write animation
        imageio.mimsave(output_file, images, fps=fps, loop=0)
        print(f"Animation saved to {output_file}")
        
    except ImportError:
        print("imageio not installed. Cannot create animation.")
        print("Install with: pip install imageio")


def _parse_numeric_key(raw_key: Any) -> Optional[float]:
    """Return float key value when possible, otherwise None."""
    if isinstance(raw_key, (int, float)):
        return float(raw_key)
    if isinstance(raw_key, str):
        try:
            return float(raw_key.strip())
        except ValueError:
            return None
    return None


def _build_numeric_key_map(mapping: Dict[Any, Any]) -> Dict[float, List[Any]]:
    """Map numeric key value -> original keys (handles int/float/string mixes)."""
    key_map: Dict[float, List[Any]] = {}
    for raw_key in mapping.keys():
        numeric_key = _parse_numeric_key(raw_key)
        if numeric_key is None:
            continue
        key_map.setdefault(numeric_key, []).append(raw_key)
    return key_map


def _resolve_numeric_key(
    numeric_value: float,
    mapping: Dict[Any, Any],
    key_map: Dict[float, List[Any]],
) -> Optional[Any]:
    """
    Resolve the original dict key for a numeric axis value.

    This keeps plotting robust when persisted JSON has string keys.
    """
    candidate_keys = key_map.get(float(numeric_value), [])
    if not candidate_keys:
        return None

    preferred: List[Any] = []
    if float(numeric_value).is_integer():
        int_value = int(numeric_value)
        preferred.extend([int_value, str(int_value), f"{float(int_value)}"])
    preferred.extend(candidate_keys)

    for candidate in preferred:
        if candidate in mapping:
            return candidate
    return None


def _resolve_gpu_key(gpu_count: int, mapping: Dict[Any, Any]) -> Optional[Any]:
    """Resolve GPU key supporting int/float/string key variants."""
    candidates: List[Any] = [gpu_count, str(gpu_count), f"{float(gpu_count)}"]
    for candidate in candidates:
        if candidate in mapping:
            return candidate
    return None


def _extract_mean_from_result(value: Any) -> Optional[float]:
    """Extract mean runtime from either stats dict or legacy scalar format."""
    if isinstance(value, dict):
        mean_val = value.get('mean')
        if mean_val is None:
            stats_times = value.get('times')
            if stats_times:
                return float(np.mean(stats_times))
            return None
        return float(mean_val)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_baseline_predictor(
    method_results: Dict[Any, Any],
    axis_key_map: Dict[float, List[Any]],
    baseline_gpu: int = 1,
):
    """
    Build a local linear extrapolator for missing baseline runtimes.

    When a 1-GPU baseline is missing for an axis value, we anchor to the last
    available observed 1-GPU point on that curve and scale runtime by the axis
    ratio. For example, if the anchor is 500M samples and the target is 1B
    samples, the estimated 1-GPU runtime is doubled.
    """
    observed: List[Tuple[float, float]] = []
    for axis_numeric in sorted(axis_key_map.keys()):
        axis_key = _resolve_numeric_key(axis_numeric, method_results, axis_key_map)
        if axis_key is None:
            continue
        axis_payload = method_results.get(axis_key, {})
        if not isinstance(axis_payload, dict):
            continue
        baseline_key = _resolve_gpu_key(baseline_gpu, axis_payload)
        if baseline_key is None:
            continue

        baseline_mean = _extract_mean_from_result(axis_payload.get(baseline_key))
        if baseline_mean is None or baseline_mean <= 0:
            continue
        observed.append((float(axis_numeric), float(baseline_mean)))

    if not observed:
        return None

    observed = sorted((float(axis_value), float(mean_value)) for axis_value, mean_value in observed)

    def _predict(axis_value: float) -> Optional[float]:
        target_axis = float(axis_value)
        if not np.isfinite(target_axis):
            return None

        anchor_axis, anchor_mean = observed[0]
        for candidate_axis, candidate_mean in observed:
            if candidate_axis <= target_axis:
                anchor_axis, anchor_mean = candidate_axis, candidate_mean
            else:
                break

        if anchor_axis <= 0:
            return float(anchor_mean) if anchor_mean > 0 else None

        estimated = float(anchor_mean) * (target_axis / float(anchor_axis))
        if not np.isfinite(estimated) or estimated <= 0:
            return None
        return estimated

    return _predict


def _resolve_baseline_mean(
    axis_numeric: float,
    axis_payload: Dict[Any, Any],
    baseline_predictor,
    baseline_gpu: int = 1,
) -> Tuple[Optional[float], bool]:
    """Return (baseline_mean, is_estimated) for one axis value."""
    baseline_key = _resolve_gpu_key(baseline_gpu, axis_payload)
    if baseline_key is not None:
        baseline_mean = _extract_mean_from_result(axis_payload.get(baseline_key))
        if baseline_mean is not None and baseline_mean > 0:
            return baseline_mean, False

    if baseline_predictor is None:
        return None, False

    estimated = baseline_predictor(axis_numeric)
    if estimated is None:
        return None, False
    estimated = float(estimated)
    if not np.isfinite(estimated) or estimated <= 0:
        return None, False
    return estimated, True


def _auto_efficiency_ylim(efficiencies: List[float]) -> Tuple[float, float]:
    """Compute adaptive y-limits for efficiency plots."""
    if not efficiencies:
        return 0.0, 110.0

    eff_min = float(min(efficiencies))
    eff_max = float(max(efficiencies))
    span = eff_max - eff_min
    pad = max(2.0, span * 0.08)

    lower = eff_min - pad
    upper = eff_max + pad

    # Keep the 100% reference line visible.
    lower = min(lower, 100.0 - pad)
    upper = max(upper, 100.0 + pad)

    if eff_min >= 0:
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + 10.0

    return lower, upper


def _set_sample_x_limits(
    ax,
    sample_sizes: List[float],
    right_pad_ratio: float = SCALING_X_AXIS_RIGHT_PAD_RATIO,
    min_upper_bound: Optional[float] = None,
) -> None:
    """
    Keep sample-size x-axis focused on available data with light right-side whitespace.
    """
    if not sample_sizes:
        return

    min_sample = float(min(sample_sizes))
    max_sample = float(max(sample_sizes))
    if max_sample <= 0:
        return

    if ax.get_xscale() == 'log':
        lower = max(1.0, min_sample * 0.95)
        upper = max_sample * (1.0 + right_pad_ratio)
    else:
        lower = 0.0
        upper = max_sample * (1.0 + right_pad_ratio)

    if min_upper_bound is not None:
        upper = max(upper, float(min_upper_bound))

    if upper <= lower:
        upper = lower + 1.0
    ax.set_xlim(lower, upper)


def _extend_axis_values_to_upper_bound(
    axis_values: Sequence[float],
    min_upper_bound: Optional[float] = None,
) -> List[float]:
    """Extend axis tick candidates so an explicit upper bound can be labeled."""
    extended = sorted({float(value) for value in axis_values})
    if min_upper_bound is None:
        return extended

    upper_bound = float(min_upper_bound)
    if upper_bound <= 0:
        return extended
    if upper_bound not in extended:
        extended.append(upper_bound)
        extended.sort()
    return extended


def _format_sample_tick_label(value: float, use_log: bool) -> str:
    """Format sample-axis tick labels; use scientific notation on log scale."""
    value = float(value)
    if use_log and value > 0:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10 ** exponent)
        rounded = int(round(mantissa))
        if np.isclose(mantissa, rounded):
            mantissa_str = f"{rounded}"
        else:
            mantissa_str = f"{mantissa:.1f}".rstrip('0').rstrip('.')
        return rf'${mantissa_str}\times10^{{{exponent}}}$'

    if value.is_integer():
        return f'{int(value):,}'
    return f'{value:g}'


def _set_sample_x_axis(
    ax,
    sample_sizes: List[float],
    min_upper_bound: Optional[float] = None,
) -> bool:
    """Configure sample x-axis and return whether log scaling is used."""
    if not sample_sizes:
        return False

    displayed_sample_sizes = _extend_axis_values_to_upper_bound(sample_sizes, min_upper_bound)
    use_log = len(sample_sizes) > 1 and min(sample_sizes) > 0 and (max(sample_sizes) / min(sample_sizes)) > 100
    if use_log:
        ax.set_xscale('log')
    display_ticks = _select_tick_subset(displayed_sample_sizes)
    ax.set_xticks(display_ticks)
    ax.set_xticklabels([_format_sample_tick_label(float(s), use_log=use_log) for s in display_ticks])
    return use_log


def _set_dimension_x_axis(
    ax,
    dimensions: List[float],
    min_upper_bound: Optional[float] = None,
) -> None:
    """Configure dimension x-axis ticks and optional log scaling."""
    if not dimensions:
        return

    min_dim = float(min(dimensions))
    max_dim = float(max(dimensions))
    displayed_dimensions = _extend_axis_values_to_upper_bound(dimensions, min_upper_bound)
    use_log = len(dimensions) > 1 and min_dim > 0
    if use_log:
        ax.set_xscale('log')

    display_ticks = _select_tick_subset(displayed_dimensions)
    tick_values = [int(dim) if float(dim).is_integer() else dim for dim in display_ticks]
    tick_labels = [str(int(dim)) if float(dim).is_integer() else f"{dim:g}" for dim in display_ticks]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    if use_log:
        lower = min_dim * 0.92
        upper = max_dim * (1.0 + SCALING_X_AXIS_RIGHT_PAD_RATIO)
    else:
        lower = 0.0 if min_dim >= 0 else min_dim * 0.92
        upper = (
            max_dim * (1.0 + SCALING_X_AXIS_RIGHT_PAD_RATIO)
            if max_dim > 0
            else max_dim + 1.0
        )

    if min_upper_bound is not None:
        upper = max(upper, float(min_upper_bound))
    if upper > lower:
        ax.set_xlim(lower, upper)


def _set_dimension_or_grid_x_axis(
    ax,
    axis_values: List[float],
    axis_mode: str,
    dimension_min_upper_bound: Optional[float] = None,
) -> bool:
    """Configure x-axis for dimension or grid-size scaling and return log usage."""
    if not axis_values:
        return False

    if axis_mode == 'grid_size':
        display_ticks = _select_tick_subset(axis_values)
        tick_values = [int(v) if float(v).is_integer() else float(v) for v in display_ticks]
        tick_labels = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in display_ticks]
        ax.set_xscale('linear')
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
        ax.minorticks_off()
        if len(axis_values) > 1:
            min_v = float(min(axis_values))
            max_v = float(max(axis_values))
            left_pad = (max_v - min_v) * 0.04
            right_pad = (max_v - min_v) * SCALING_X_AXIS_RIGHT_PAD_RATIO
            ax.set_xlim(min_v - left_pad, max_v + right_pad)
        ax.set_xlabel('Grid Size', labelpad=X_AXIS_LABEL_PAD)
        return False

    _set_dimension_x_axis(ax, axis_values, min_upper_bound=dimension_min_upper_bound)
    use_log = len(axis_values) > 1 and min(axis_values) > 0
    if use_log:
        ax.set_xlabel('Number of Dimensions (log scale)', labelpad=X_AXIS_LABEL_PAD)
    else:
        ax.set_xlabel('Number of Dimensions', labelpad=X_AXIS_LABEL_PAD)
    return use_log


def _left_annotation_anchor_x(
    x_values: List[float],
    use_log: bool,
    fraction_from_left: float = 0.04,
) -> Optional[float]:
    """Choose an in-panel x anchor near the left edge for reference labels."""
    if not x_values:
        return None

    xs = [float(value) for value in x_values if np.isfinite(float(value))]
    if not xs:
        return None

    x_min = min(xs)
    x_max = max(xs)
    if x_max <= x_min:
        return x_min

    fraction = float(np.clip(fraction_from_left, 0.0, 1.0))
    if use_log and x_min > 0 and x_max > 0:
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        return float(10 ** (log_min + fraction * (log_max - log_min)))

    return float(x_min + fraction * (x_max - x_min))


def _highlight_estimated_baseline_regions(
    ax: Any,
    estimated_axis_values: Set[float],
    all_axis_values: Sequence[float],
    use_log: bool,
    color: str = ESTIMATED_BASELINE_SHADE_COLOR,
    alpha: float = ESTIMATED_BASELINE_SHADE_ALPHA,
) -> None:
    """Shade x-axis regions where the 1-GPU baseline was estimated."""
    if not estimated_axis_values or not all_axis_values:
        return

    ordered = sorted(
        {
            float(value)
            for value in all_axis_values
            if np.isfinite(float(value))
        }
    )
    if not ordered:
        return

    def _resolve_match(target: float) -> Optional[float]:
        for candidate in ordered:
            if np.isclose(candidate, target, rtol=1e-8, atol=1e-8):
                return candidate
        return None

    n_values = len(ordered)
    for raw_value in sorted(estimated_axis_values):
        value = _resolve_match(float(raw_value))
        if value is None:
            continue

        idx = ordered.index(value)
        prev_value = ordered[idx - 1] if idx > 0 else None
        next_value = ordered[idx + 1] if idx + 1 < n_values else None

        if use_log and value > 0:
            if prev_value is not None and prev_value > 0:
                left_edge = float(np.sqrt(prev_value * value))
            elif next_value is not None and next_value > 0:
                ratio = np.sqrt(next_value / value)
                left_edge = value / ratio
            else:
                left_edge = value / 1.25

            if next_value is not None and next_value > 0:
                right_edge = float(np.sqrt(value * next_value))
            elif prev_value is not None and prev_value > 0:
                ratio = np.sqrt(value / prev_value)
                right_edge = value * ratio
            else:
                right_edge = value * 1.25
        else:
            if prev_value is not None:
                left_edge = value - 0.5 * (value - prev_value)
            elif next_value is not None:
                left_edge = value - 0.5 * (next_value - value)
            else:
                left_edge = value - max(1.0, abs(value) * 0.1)

            if next_value is not None:
                right_edge = value + 0.5 * (next_value - value)
            elif prev_value is not None:
                right_edge = value + 0.5 * (value - prev_value)
            else:
                right_edge = value + max(1.0, abs(value) * 0.1)

        if right_edge <= left_edge:
            continue
        ax.axvspan(left_edge, right_edge, color=color, alpha=alpha, zorder=0.2, linewidth=0)


def _highlight_trailing_missing_x_range(
    ax: Any,
    plotted_axis_values: Sequence[float],
    color: str = MISSING_TRAILING_X_SHADE_COLOR,
    alpha: float = MISSING_TRAILING_X_SHADE_ALPHA,
) -> None:
    """Shade the trailing x-range beyond the last observed point in a panel."""
    if not plotted_axis_values:
        return

    xs = [
        float(value)
        for value in plotted_axis_values
        if np.isfinite(float(value))
    ]
    if not xs:
        return

    max_plotted = max(xs)
    x_left, x_right = ax.get_xlim()
    upper = max(float(x_left), float(x_right))
    lower = min(float(x_left), float(x_right))
    if upper <= lower:
        return

    shade_start = max(max_plotted, lower)
    if shade_start >= upper:
        return

    ax.axvspan(shade_start, upper, color=color, alpha=alpha, zorder=0.15, linewidth=0)


def _finalize_layout_with_bottom_legend(
    fig,
    ax,
    show_legend: bool,
    note_text: Optional[str] = None,
    max_cols: int = 6,
    shaded_region_label: Optional[str] = None,
    shaded_region_color: str = ESTIMATED_BASELINE_SHADE_COLOR,
    shaded_region_alpha: float = ESTIMATED_BASELINE_SHADE_ALPHA,
    legend_fontsize_scale: float = 1.0,
    legend_markerscale: float = 1.0,
) -> None:
    """Place note + legend below panel and reserve enough bottom margin."""
    _apply_axis_line_style(ax)

    def _parse_gpu_method_label(label: str) -> Optional[Tuple[int, str]]:
        match = re.match(r'^\s*(\d+)\s*GPU(?:s)?\s*\(([^)]+)\)\s*$', str(label))
        if not match:
            return None
        return int(match.group(1)), match.group(2).strip()

    def _parse_method_topology_label(label: str) -> Optional[Tuple[str, str]]:
        match = re.match(r'^\s*([^)]+?)\s*\(([^)]+)\)\s*$', str(label))
        if not match:
            return None
        method = match.group(1).strip()
        topology = match.group(2).strip()
        if not method or not topology:
            return None
        # Avoid clobbering GPU/method labels handled above.
        if re.match(r'^\d+\s*GPU(?:s)?$', method, flags=re.IGNORECASE):
            return None
        return method, topology

    def _reorder_legend_entries(
        handles_in: List[Any],
        labels_in: List[str],
    ) -> Tuple[List[Any], List[str], Optional[int]]:
        parsed = []
        for idx, label in enumerate(labels_in):
            parsed_label = _parse_gpu_method_label(label)
            if parsed_label is None:
                parsed = []
                break
            gpu_count, method = parsed_label
            parsed.append((idx, gpu_count, method))

        if parsed:
            method_order = []
            for _, _, method in parsed:
                if method not in method_order:
                    method_order.append(method)
            gpu_order = sorted({gpu_count for _, gpu_count, _ in parsed})

            index_by_key = {(gpu_count, method): idx for idx, gpu_count, method in parsed}
            ordered_indices = []
            for method in method_order:
                for gpu_count in gpu_order:
                    idx = index_by_key.get((gpu_count, method))
                    if idx is not None:
                        ordered_indices.append(idx)

            if len(ordered_indices) != len(labels_in):
                seen = set(ordered_indices)
                ordered_indices.extend(idx for idx in range(len(labels_in)) if idx not in seen)

            return (
                [handles_in[idx] for idx in ordered_indices],
                [labels_in[idx] for idx in ordered_indices],
                None,
            )

        parsed_topology = []
        for idx, label in enumerate(labels_in):
            parsed_label = _parse_method_topology_label(label)
            if parsed_label is None:
                return handles_in, labels_in, None
            method, topology = parsed_label
            parsed_topology.append((idx, method, topology))

        topology_order = []
        method_order = []
        for _, method, topology in parsed_topology:
            if topology not in topology_order:
                topology_order.append(topology)
            if method not in method_order:
                method_order.append(method)

        index_by_key = {(method, topology): idx for idx, method, topology in parsed_topology}
        ordered_indices = []
        # Matplotlib legend entries are filled column-major when ncol > 1.
        # To make each column correspond to one topology, order entries by:
        # topology (outer) -> method (inner).
        for topology in topology_order:
            for method in method_order:
                idx = index_by_key.get((method, topology))
                if idx is not None:
                    ordered_indices.append(idx)

        if len(ordered_indices) != len(labels_in):
            seen = set(ordered_indices)
            ordered_indices.extend(idx for idx in range(len(labels_in)) if idx not in seen)

        preferred_ncol = len(topology_order) if len(topology_order) > 1 else None
        return (
            [handles_in[idx] for idx in ordered_indices],
            [labels_in[idx] for idx in ordered_indices],
            preferred_ncol,
        )

    legend_rows = 0
    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    ncol = 0

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        preferred_ncol = None
        if handles:
            handles, labels, preferred_ncol = _reorder_legend_entries(handles, labels)
        shade_label = str(shaded_region_label or '').strip()
        if shade_label and shade_label not in labels:
            handles = list(handles)
            labels = list(labels)
            handles.append(
                Patch(
                    facecolor=shaded_region_color,
                    edgecolor=SHADED_REGION_EDGE_COLOR,
                    linewidth=3.2,
                    alpha=shaded_region_alpha,
                )
            )
            labels.append(shade_label)
        if handles:
            n_items = len(handles)
            if preferred_ncol is not None:
                ncol = min(max_cols, n_items, max(1, preferred_ncol))
            else:
                ncol = min(3, max_cols, n_items)
            legend_rows = int(np.ceil(n_items / float(ncol)))
            legend_handles = handles
            legend_labels = labels

    # Keep the legend block just below axis labels instead of near page bottom.
    legend_bottom = 0.07 if legend_rows > 0 else 0.0
    legend_height = (0.024 + (0.040 * legend_rows)) if legend_rows > 0 else 0.0
    note_height = 0.018 if note_text else 0.0
    note_gap = 0.012 if (note_text and legend_rows > 0) else (0.008 if note_text else 0.0)

    if legend_rows > 0 or note_text:
        bottom_margin = 0.08 + legend_height + note_gap + note_height
    else:
        bottom_margin = 0.1
    bottom_margin = min(0.4, max(0.12, bottom_margin))
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.985])

    if note_text:
        note_y = (legend_bottom + legend_height + 0.006) if legend_rows > 0 else 0.035
        note_y = min(note_y, bottom_margin - 0.01)
        fig.text(
            0.08,
            note_y,
            note_text,
            fontsize=PLOT_ANNOTATION_FONTSIZE,
            alpha=0.9,
            ha='left',
            va='top',
        )

    if legend_rows > 0 and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, legend_bottom),
            ncol=ncol,
            fontsize=BOTTOM_LEGEND_FONTSIZE * float(legend_fontsize_scale),
            frameon=False,
            markerscale=LEGEND_MARKER_SCALE * float(legend_markerscale),
            columnspacing=1.4,
            handlelength=2.2,
            handletextpad=0.6,
            labelspacing=0.55,
        )


def plot_weight_statistics(weight_stats: Dict[str, List[float]], 
                         title: str, output_file: str):
    """
    Plot weight statistics over iterations
    
    Args:
        weight_stats: Dictionary with statistics (min, max, mean, std, range)
        title: Title for the plot
        output_file: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = list(range(len(weight_stats['mean'])))
    
    # Plot mean and std
    ax = axes[0, 0]
    ax.plot(iterations, weight_stats['mean'], label='Mean', color='blue')
    ax.fill_between(iterations, 
                    np.array(weight_stats['mean']) - np.array(weight_stats['std']),
                    np.array(weight_stats['mean']) + np.array(weight_stats['std']),
                    alpha=0.3, color='blue', label='±1 std')
    ax.set_title('Weight Mean ± Std')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot min and max
    ax = axes[0, 1]
    ax.plot(iterations, weight_stats['min'], label='Min', color='red')
    ax.plot(iterations, weight_stats['max'], label='Max', color='green')
    ax.set_title('Weight Min/Max')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot range
    ax = axes[1, 0]
    ax.plot(iterations, weight_stats['range'], color='purple')
    ax.set_title('Weight Range (Max - Min)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Range')
    ax.grid(True, alpha=0.3)
    
    # Plot relative spread
    ax = axes[1, 1]
    ax.plot(iterations, weight_stats['relative_spread'], color='orange')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax.set_title('Weight Spread / Data Spread')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_gpu_scaling_benchmark(results: Dict, 
                              output_dir: str,
                              title: str = "GPU Scaling Performance",
                              show_speedup: bool = True,
                              show_error_bars: bool = False,
                              show_legend: bool = True,
                              axis_mode: str = "dimension",
                              methods_to_plot: Optional[List[str]] = None,
                              emit_shared_legend: bool = False,
                              legend_filename: str = "legend_key.svg",
                              axis_min_upper_bound: Optional[float] = None):
    """
    Create GPU scaling benchmark visualizations with optional error bars.
    Supports both single method and multi-method results.
    
    Args:
        results: Either:
            - Single method: {dimension -> gpu_count -> stats}
            - Multiple methods: {method -> {dimension -> gpu_count -> stats}}
        output_dir: Directory to save plots
        title: Title for the plots
        show_speedup: Whether to also create speedup plot
        show_error_bars: Whether to show error bars (requires stats dict in results)
        show_legend: Whether to draw an in-figure legend
        axis_mode: X-axis mode for GPU plots: 'dimension' (default) or 'grid_size'
        methods_to_plot: Optional method list to plot; default shows ['batch', 'colors'].
        emit_shared_legend: Whether to emit a standalone grouped legend SVG.
        legend_filename: Standalone legend filename written under output_dir.
        axis_min_upper_bound: Optional minimum upper bound for the dimension axis.
    """
    if axis_mode not in {'dimension', 'grid_size'}:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")

    # Check if results contain multiple methods
    # Multi-method format: {'batch': {dim: {gpu: stats}}, 'colors': {dim: {gpu: stats}}}
    # Single-method format: {dim: {gpu: stats}}
    first_key = next(iter(results))
    
    # Check if first key represents a processing method (string) instead of a dimension (int)
    # Allow for any string key since new methods may be added (e.g., 'minibatch').
    is_multi_method = isinstance(first_key, str)
    
    def _parse_numeric_keys(iterable):
        parsed = []
        for key in iterable:
            if isinstance(key, (int, float)):
                parsed.append(float(key))
                continue
            if isinstance(key, str):
                try:
                    parsed.append(float(key))
                except ValueError:
                    continue
        return sorted(set(parsed))

    if is_multi_method:
        methods = list(results.keys())
    else:
        # Single method results (backward compatibility)
        methods = ['batch']
        results = {'batch': results}
    methods = _resolve_speed_plot_methods(methods, methods_to_plot=methods_to_plot)
    if not methods:
        return

    # Union dimensions across all methods so late/missing method ordering does not drop axis values.
    dimensions = _parse_numeric_keys(
        key
        for method in methods
        for key in results[method].keys()
    )

    # Pre-build axis-key maps to resolve int/float/string key variants consistently.
    dimension_key_maps = {
        method: _build_numeric_key_map(results[method])
        for method in methods
    }

    gpu_counts = sorted(
        {
            int(gpu_numeric)
            for method in methods
            for dim_results in results[method].values()
            for gpu in dim_results.keys()
            for gpu_numeric in [_parse_numeric_key(gpu)]
            if gpu_numeric is not None
            and float(gpu_numeric).is_integer()
        }
    )
    has_multi_gpu = any(gpu_count > 1 for gpu_count in gpu_counts)

    _apply_publication_theme()
    method_base_colors = _build_method_base_colors(methods)
    gpu_markers = _build_gpu_marker_map(gpu_counts)
    gpu_linewidths = _build_gpu_linewidth_map(gpu_counts)
    method_legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    estimated_baseline_used = False
    
    # Create main performance plot
    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    
    for method in methods:
        method_results = results[method]
        method_dimension_map = dimension_key_maps[method]
        for gpu_count in gpu_counts:
            times = []
            errors = []
            valid_dims = []
            
            for dim in dimensions:
                dim_key = _resolve_numeric_key(dim, method_results, method_dimension_map)
                if dim_key is None:
                    continue
                gpu_key = _resolve_gpu_key(gpu_count, method_results[dim_key])
                if gpu_key is None:
                    continue

                result = method_results[dim_key][gpu_key]
                dim_for_plot = int(dim) if float(dim).is_integer() else dim
                if isinstance(result, dict):
                    stats_times = result.get('times')
                    mean_val = result.get('mean')
                    if mean_val is None and stats_times:
                        mean_val = float(np.mean(stats_times))
                    std_val = result.get('std')
                    if std_val is None and stats_times:
                        std_val = float(np.std(stats_times))

                    if mean_val is None:
                        # Cannot plot without a mean value
                        continue

                    times.append(mean_val)
                    errors.append(std_val if std_val is not None else 0.0)
                else:
                    # Old format with just time value
                    times.append(result)
                    errors.append(0.0)
                valid_dims.append(dim_for_plot)
            
            if times:
                linestyle = '-'
                series_color = _shade_method_color_for_gpu(
                    base_color=method_base_colors[method],
                    gpu_count=gpu_count,
                    gpu_counts=gpu_counts,
                )
                marker_style = gpu_markers.get(gpu_count, 'o')
                series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                label = f'{gpu_count} GPU{"s" if gpu_count > 1 else ""}'
                if len(methods) > 1:
                    label += f' - {_method_display_name(method)}'
                _register_method_gpu_legend_item(
                    legend_registry=method_legend_registry,
                    method=method,
                    gpu_count=gpu_count,
                    series_color=series_color,
                    marker_style=marker_style,
                    series_linewidth=series_linewidth,
                    linestyle=linestyle,
                )

                if show_error_bars and any(e > 0 for e in errors):
                    ax.errorbar(valid_dims, times, yerr=errors,
                               marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                               color=series_color, capsize=5,
                               linestyle=linestyle,
                               label=label)
                else:
                    ax.plot(valid_dims, times, 
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color, 
                           linestyle=linestyle,
                           label=label)
    
    ax.set_ylabel('Time (s)', labelpad=12)
    
    # Configure x-axis according to scale mode.
    _set_dimension_or_grid_x_axis(
        ax,
        dimensions,
        axis_mode,
        dimension_min_upper_bound=axis_min_upper_bound,
    )
    _apply_scaling_axis_typography(ax)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.28, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)
    
    if sns is not None:
        sns.despine(ax=ax)
    
    _finalize_layout_with_bottom_legend(fig, ax, show_legend=show_legend)
    
    # Save the plot
    performance_file = os.path.join(output_dir, 'gpu_scaling_performance.svg')
    plt.savefig(performance_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.savefig(performance_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to: {performance_file}")
    
    # Create speedup plot if requested
    if show_speedup and has_multi_gpu:
        fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
        has_speedup_data = False
        estimated_dims: Set[float] = set()
        
        for method in methods:
            method_results = results[method]
            method_dimension_map = dimension_key_maps[method]
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=method_dimension_map,
                baseline_gpu=1,
            )
            for gpu_count in gpu_counts:
                if gpu_count == 1:
                    continue
                    
                speedups = []
                valid_dims = []
                
                for dim in dimensions:
                    dim_key = _resolve_numeric_key(dim, method_results, method_dimension_map)
                    if dim_key is None:
                        continue

                    dim_payload = method_results[dim_key]
                    current_key = _resolve_gpu_key(gpu_count, dim_payload)
                    if current_key is None:
                        continue

                    current = dim_payload[current_key]
                    baseline_mean, is_estimated = _resolve_baseline_mean(
                        axis_numeric=dim,
                        axis_payload=dim_payload,
                        baseline_predictor=baseline_predictor,
                        baseline_gpu=1,
                    )
                    current_mean = _extract_mean_from_result(current)
                    if baseline_mean is None or current_mean is None or current_mean == 0:
                        continue

                    speedup = baseline_mean / current_mean
                    if is_estimated:
                        estimated_dims.add(float(dim))
                    dim_for_plot = int(dim) if float(dim).is_integer() else dim
                    speedups.append(speedup)
                    valid_dims.append(dim_for_plot)
                
                if speedups:
                    has_speedup_data = True
                    linestyle = '-'
                    series_color = _shade_method_color_for_gpu(
                        base_color=method_base_colors[method],
                        gpu_count=gpu_count,
                        gpu_counts=gpu_counts,
                    )
                    marker_style = gpu_markers.get(gpu_count, 'o')
                    series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                    label = f'{gpu_count} GPUs'
                    if len(methods) > 1:
                        label += f' - {_method_display_name(method)}'
                    ax.plot(valid_dims, speedups,
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color,
                           linestyle=linestyle,
                           label=label)
        
        if has_speedup_data:
            baseline_note = (
                '* Estimated 1-GPU baseline used where missing'
                if estimated_dims
                else None
            )
            estimated_baseline_used = estimated_baseline_used or bool(estimated_dims)
            use_log_dimensions = len(dimensions) > 1 and min(dimensions) > 0
            ideal_anchor_x = _left_annotation_anchor_x(
                [float(dim) for dim in dimensions],
                use_log=use_log_dimensions,
            )
            # Add ideal scaling lines
            for gpu_count in gpu_counts:
                if gpu_count > 1:
                    ax.axhline(
                        y=gpu_count,
                        color='#6a6a6a',
                        linestyle='--',
                        alpha=0.55,
                        linewidth=1.2,
                    )
                    if ideal_anchor_x is not None:
                        ax.annotate(
                            f'Ideal {gpu_count}x',
                            xy=(ideal_anchor_x, gpu_count),
                            xytext=(4, 2),
                            textcoords='offset points',
                            va='bottom',
                            ha='left',
                            color='#4a4a4a',
                            fontsize=PLOT_ANNOTATION_FONTSIZE,
                            fontweight='semibold',
                        )
            
            ax.set_ylabel('Speedup Factor', labelpad=12)
            use_log_axis_for_plot = _set_dimension_or_grid_x_axis(
                ax,
                dimensions,
                axis_mode,
                dimension_min_upper_bound=axis_min_upper_bound,
            )
            _highlight_estimated_baseline_regions(
                ax=ax,
                estimated_axis_values=estimated_dims,
                all_axis_values=dimensions,
                use_log=use_log_axis_for_plot,
            )
            _apply_scaling_axis_typography(ax)
            
            ax.grid(True, alpha=0.28, linestyle='--')
            ax.set_axisbelow(True)
            
            # Set y-axis to start from 1
            ax.set_ylim(bottom=1)
            
            if sns is not None:
                sns.despine(ax=ax)
            
            _finalize_layout_with_bottom_legend(
                fig,
                ax,
                show_legend=show_legend,
                note_text=None,
                shaded_region_label=(
                    ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
                ),
            )
            
            speedup_file = os.path.join(output_dir, 'gpu_scaling_speedup.svg')
            plt.savefig(speedup_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
            plt.savefig(speedup_file.replace('.svg', '.pdf'), bbox_inches='tight')
            plt.close()
            
            print(f"Speedup plot saved to: {speedup_file}")
        else:
            plt.close()
    
    # Create efficiency plot
    if show_speedup and has_multi_gpu:
        fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
        has_efficiency_data = False
        all_efficiencies = []
        plotted_efficiency_x: List[float] = []
        estimated_dims: Set[float] = set()
        
        # Plot efficiency for each method with overlay
        for method in methods:
            method_results = results[method]
            method_dimension_map = dimension_key_maps[method]
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=method_dimension_map,
                baseline_gpu=1,
            )
            
            for gpu_count in gpu_counts:
                if gpu_count == 1:
                    continue
                    
                efficiencies = []
                valid_dims = []
                
                for dim in dimensions:
                    dim_key = _resolve_numeric_key(dim, method_results, method_dimension_map)
                    if dim_key is None:
                        continue

                    dim_payload = method_results[dim_key]
                    current_key = _resolve_gpu_key(gpu_count, dim_payload)
                    if current_key is None:
                        continue

                    current = dim_payload[current_key]
                    baseline_mean, is_estimated = _resolve_baseline_mean(
                        axis_numeric=dim,
                        axis_payload=dim_payload,
                        baseline_predictor=baseline_predictor,
                        baseline_gpu=1,
                    )
                    current_mean = _extract_mean_from_result(current)
                    if baseline_mean is None or current_mean is None or current_mean == 0:
                        continue

                    speedup = baseline_mean / current_mean
                    efficiency = (speedup / gpu_count) * 100
                    if is_estimated:
                        estimated_dims.add(float(dim))
                    dim_for_plot = int(dim) if float(dim).is_integer() else dim
                    efficiencies.append(efficiency)
                    valid_dims.append(dim_for_plot)
                
                if efficiencies:
                    has_efficiency_data = True
                    all_efficiencies.extend(efficiencies)
                    plotted_efficiency_x.extend(float(value) for value in valid_dims)
                    linestyle = '-'
                    series_color = _shade_method_color_for_gpu(
                        base_color=method_base_colors[method],
                        gpu_count=gpu_count,
                        gpu_counts=gpu_counts,
                    )
                    marker_style = gpu_markers.get(gpu_count, 'o')
                    series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                    label = f'{gpu_count} GPUs'
                    if len(methods) > 1:
                        label += f' - {_method_display_name(method)}'
                    ax.plot(valid_dims, efficiencies,
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color,
                           linestyle=linestyle,
                           label=label)
        
        if has_efficiency_data:
            baseline_note = (
                '* Estimated 1-GPU baseline used where missing'
                if estimated_dims
                else None
            )
            estimated_baseline_used = estimated_baseline_used or bool(estimated_dims)
            # Add 100% efficiency line
            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
            use_log_dimensions = len(plotted_efficiency_x) > 1 and min(plotted_efficiency_x) > 0
            reference_x = _left_annotation_anchor_x(
                plotted_efficiency_x,
                use_log=use_log_dimensions,
            )
            if reference_x is not None:
                ax.annotate(
                    'Perfect Scaling',
                    xy=(reference_x, 100),
                    xytext=(4, 2),
                    textcoords='offset points',
                    va='bottom',
                    ha='left',
                    color='green',
                    fontsize=PLOT_ANNOTATION_FONTSIZE,
                )
            
            ax.set_ylabel('Scaling Efficiency (%)', labelpad=12)
            use_log_axis_for_plot = _set_dimension_or_grid_x_axis(
                ax,
                dimensions,
                axis_mode,
                dimension_min_upper_bound=axis_min_upper_bound,
            )
            _highlight_estimated_baseline_regions(
                ax=ax,
                estimated_axis_values=estimated_dims,
                all_axis_values=dimensions,
                use_log=use_log_axis_for_plot,
            )
            _apply_scaling_axis_typography(ax)
            
            ax.grid(True, alpha=0.28, linestyle='--')
            ax.set_axisbelow(True)
            
            # Auto-scale efficiency axis to show >100% results while keeping the reference line visible.
            y_min, y_max = _auto_efficiency_ylim(all_efficiencies)
            ax.set_ylim(y_min, y_max)
            
            if sns is not None:
                sns.despine(ax=ax)
            
            _finalize_layout_with_bottom_legend(
                fig,
                ax,
                show_legend=show_legend,
                note_text=None,
                shaded_region_label=(
                    ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
                ),
            )
            
            efficiency_file = os.path.join(output_dir, 'gpu_scaling_efficiency.svg')
            plt.savefig(efficiency_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
            plt.savefig(efficiency_file.replace('.svg', '.pdf'), bbox_inches='tight')
            plt.close()
            
            print(f"Efficiency plot saved to: {efficiency_file}")
        else:
            plt.close()

    if emit_shared_legend:
        legend_path = os.path.join(output_dir, legend_filename)
        if _render_grouped_legend_key(
            legend_registry=method_legend_registry,
            output_path=legend_path,
            dpi=PUBLICATION_DPI,
            shaded_region_label=(
                ESTIMATED_BASELINE_LEGEND_LABEL if estimated_baseline_used else None
            ),
        ):
            print(f"Shared legend saved to: {legend_path}")


def plot_sample_scaling_benchmark(results: Dict, 
                                 output_dir: str,
                                 title: str = "Sample Size Scaling Performance",
                                 show_speedup: bool = True,
                                 show_error_bars: bool = False,
                                 show_legend: bool = True,
                                 methods_to_plot: Optional[List[str]] = None,
                                 emit_shared_legend: bool = False,
                                 legend_filename: str = "legend_key.svg",
                                 sample_min_upper_bound: Optional[float] = None):
    """
    Create sample size scaling benchmark visualizations with optional error bars.
    Supports both single method and multi-method results.
    
    Args:
        results: Either:
            - Single method: {sample_size -> gpu_count -> stats}
            - Multiple methods: {method -> {sample_size -> gpu_count -> stats}}
        output_dir: Directory to save plots
        title: Title for the plots
        show_speedup: Whether to also create speedup plot
        show_error_bars: Whether to show error bars (requires stats dict in results)
        show_legend: Whether to draw an in-figure legend
        methods_to_plot: Optional method list to plot; default shows ['batch', 'colors'].
        emit_shared_legend: Whether to emit a standalone grouped legend SVG.
        legend_filename: Standalone legend filename written under output_dir.
        sample_min_upper_bound: Optional minimum upper bound for the sample axis.
    """
    # Check if results contain multiple methods
    # Multi-method format: {'batch': {dim: {gpu: stats}}, 'colors': {dim: {gpu: stats}}}
    # Single-method format: {dim: {gpu: stats}}
    first_key = next(iter(results))
    
    # Treat any string-rooted mapping as multi-method (supports minibatch and future methods).
    is_multi_method = isinstance(first_key, str)
    
    if is_multi_method:
        methods = list(results.keys())
    else:
        methods = ['batch']
        results = {'batch': results}
    methods = _resolve_speed_plot_methods(methods, methods_to_plot=methods_to_plot)
    if not methods:
        return

    raw_sample_sizes = [
        axis_key
        for method in methods
        for axis_key in results[method].keys()
    ]
    gpu_counts = sorted(
        {
            int(gpu_numeric)
            for method in methods
            for axis_payload in results[method].values()
            for gpu_key in axis_payload.keys()
            for gpu_numeric in [_parse_numeric_key(gpu_key)]
            if gpu_numeric is not None
            and float(gpu_numeric).is_integer()
        }
    )
    has_multi_gpu = any(gpu_count > 1 for gpu_count in gpu_counts)

    # Build sorted numeric sample axis values.
    numeric_sample_sizes = []
    for size in raw_sample_sizes:
        try:
            size_numeric = float(size)
        except (TypeError, ValueError):
            continue
        numeric_sample_sizes.append(size_numeric)
    sample_sizes = sorted(set(numeric_sample_sizes))
    sample_key_maps = {
        method: _build_numeric_key_map(results[method])
        for method in methods
    }

    _apply_publication_theme()
    method_base_colors = _build_method_base_colors(methods)
    gpu_markers = _build_gpu_marker_map(gpu_counts)
    gpu_linewidths = _build_gpu_linewidth_map(gpu_counts)
    method_legend_registry: Dict[str, Dict[str, Any]] = OrderedDict()
    estimated_baseline_used = False
    plotted_sample_sizes: Set[float] = set()
    
    # Create main performance plot with log scale for samples
    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
    
    for method in methods:
        method_results = results[method]
        method_sample_map = sample_key_maps[method]
        for gpu_count in gpu_counts:
            times = []
            errors = []
            valid_samples = []
            
            for samples in sample_sizes:
                selected_key = _resolve_numeric_key(samples, method_results, method_sample_map)
                if selected_key is None:
                    continue
                gpu_key = _resolve_gpu_key(gpu_count, method_results[selected_key])
                if gpu_key is None:
                    continue
                result = method_results[selected_key][gpu_key]
                if isinstance(result, dict):
                    # New format with statistics
                    times.append(result['mean'])
                    errors.append(result['std'])
                else:
                    # Old format with just time value
                    times.append(result)
                    errors.append(0)
                samples_numeric = int(samples) if samples.is_integer() else samples
                valid_samples.append(samples_numeric)
            
            if times:
                plotted_sample_sizes.update(float(sample) for sample in valid_samples)
                linestyle = '-'
                series_color = _shade_method_color_for_gpu(
                    base_color=method_base_colors[method],
                    gpu_count=gpu_count,
                    gpu_counts=gpu_counts,
                )
                marker_style = gpu_markers.get(gpu_count, 'o')
                series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                label = f'{gpu_count} GPU{"s" if gpu_count > 1 else ""}'
                if len(methods) > 1:
                    label += f' - {_method_display_name(method)}'
                _register_method_gpu_legend_item(
                    legend_registry=method_legend_registry,
                    method=method,
                    gpu_count=gpu_count,
                    series_color=series_color,
                    marker_style=marker_style,
                    series_linewidth=series_linewidth,
                    linestyle=linestyle,
                )
                
                if show_error_bars and any(e > 0 for e in errors):
                    ax.errorbar(valid_samples, times, yerr=errors,
                               marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                               color=series_color, capsize=5,
                               linestyle=linestyle,
                               label=label)
                else:
                    ax.plot(valid_samples, times, 
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color, 
                           linestyle=linestyle,
                           label=label)
    
    ax.set_ylabel('Time (s)', labelpad=12)
    
    displayed_sample_sizes = sorted(plotted_sample_sizes) if plotted_sample_sizes else sample_sizes
    use_log_samples = _set_sample_x_axis(
        ax,
        displayed_sample_sizes,
        min_upper_bound=sample_min_upper_bound,
    )
    if use_log_samples:
        ax.set_xlabel('Number of Samples (log scale)', labelpad=X_AXIS_LABEL_PAD)
    else:
        ax.set_xlabel('Number of Samples', labelpad=X_AXIS_LABEL_PAD)
    _set_sample_x_limits(
        ax,
        displayed_sample_sizes,
        min_upper_bound=sample_min_upper_bound,
    )
    _apply_scaling_axis_typography(ax)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.28, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)
    
    if sns is not None:
        sns.despine(ax=ax)
    
    _finalize_layout_with_bottom_legend(fig, ax, show_legend=show_legend)
    
    # Save the plot
    performance_file = os.path.join(output_dir, 'sample_scaling_performance.svg')
    plt.savefig(performance_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.savefig(performance_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to: {performance_file}")
    
    # Create speedup plot if requested
    if show_speedup and has_multi_gpu:
        fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
        has_speedup_data = False
        estimated_samples: Set[float] = set()
        
        for method in methods:
            method_results = results[method]
            method_sample_map = sample_key_maps[method]
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=method_sample_map,
                baseline_gpu=1,
            )
            for gpu_count in gpu_counts:
                if gpu_count == 1:
                    continue
                    
                speedups = []
                valid_samples = []
                
                for samples in sample_sizes:
                    selected_key = _resolve_numeric_key(samples, method_results, method_sample_map)
                    if selected_key is None:
                        continue

                    sample_payload = method_results[selected_key]
                    current_key = _resolve_gpu_key(gpu_count, sample_payload)
                    if current_key is None:
                        continue

                    current = sample_payload[current_key]
                    baseline_mean, is_estimated = _resolve_baseline_mean(
                        axis_numeric=samples,
                        axis_payload=sample_payload,
                        baseline_predictor=baseline_predictor,
                        baseline_gpu=1,
                    )
                    current_mean = _extract_mean_from_result(current)
                    if baseline_mean is None or current_mean is None or current_mean == 0:
                        continue

                    speedup = baseline_mean / current_mean
                    if is_estimated:
                        estimated_samples.add(float(samples))
                    speedups.append(speedup)
                    samples_numeric = int(samples) if samples.is_integer() else samples
                    valid_samples.append(samples_numeric)
                
                if speedups:
                    has_speedup_data = True
                    linestyle = '-'
                    series_color = _shade_method_color_for_gpu(
                        base_color=method_base_colors[method],
                        gpu_count=gpu_count,
                        gpu_counts=gpu_counts,
                    )
                    marker_style = gpu_markers.get(gpu_count, 'o')
                    series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                    label = f'{gpu_count} GPUs'
                    if len(methods) > 1:
                        label += f' - {_method_display_name(method)}'
                    ax.plot(valid_samples, speedups,
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color,
                           linestyle=linestyle,
                           label=label)
        
        if has_speedup_data:
            baseline_note = (
                '* Estimated 1-GPU baseline used where missing'
                if estimated_samples
                else None
            )
            estimated_baseline_used = estimated_baseline_used or bool(estimated_samples)
            use_log_samples = (
                len(sample_sizes) > 1
                and min(sample_sizes) > 0
                and (max(sample_sizes) / min(sample_sizes)) > 100
            )
            ideal_anchor_x = _left_annotation_anchor_x(
                [float(sample_size) for sample_size in sample_sizes],
                use_log=use_log_samples,
            )
            # Add ideal scaling lines
            for gpu_count in gpu_counts:
                if gpu_count > 1:
                    ax.axhline(
                        y=gpu_count,
                        color='#6a6a6a',
                        linestyle='--',
                        alpha=0.55,
                        linewidth=1.2,
                    )
                    if ideal_anchor_x is not None:
                        ax.annotate(
                            f'Ideal {gpu_count}x',
                            xy=(ideal_anchor_x, gpu_count),
                            xytext=(4, 2),
                            textcoords='offset points',
                            va='bottom',
                            ha='left',
                            color='#4a4a4a',
                            fontsize=PLOT_ANNOTATION_FONTSIZE,
                            fontweight='semibold',
                        )
            
            ax.set_ylabel('Speedup Factor', labelpad=12)
            use_log_samples = _set_sample_x_axis(
                ax,
                sample_sizes,
                min_upper_bound=sample_min_upper_bound,
            )
            if use_log_samples:
                ax.set_xlabel('Number of Samples (log scale)', labelpad=X_AXIS_LABEL_PAD)
            else:
                ax.set_xlabel('Number of Samples', labelpad=X_AXIS_LABEL_PAD)
            _set_sample_x_limits(
                ax,
                sample_sizes,
                min_upper_bound=sample_min_upper_bound,
            )
            _highlight_estimated_baseline_regions(
                ax=ax,
                estimated_axis_values=estimated_samples,
                all_axis_values=sample_sizes,
                use_log=use_log_samples,
            )
            _apply_scaling_axis_typography(ax)
            
            ax.grid(True, alpha=0.28, linestyle='--')
            ax.set_axisbelow(True)
            
            # Set y-axis to start from 1
            ax.set_ylim(bottom=1)
            
            if sns is not None:
                sns.despine(ax=ax)
            
            _finalize_layout_with_bottom_legend(
                fig,
                ax,
                show_legend=show_legend,
                note_text=None,
                shaded_region_label=(
                    ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
                ),
            )
            
            speedup_file = os.path.join(output_dir, 'sample_scaling_speedup.svg')
            plt.savefig(speedup_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
            plt.savefig(speedup_file.replace('.svg', '.pdf'), bbox_inches='tight')
            plt.close()
            
            print(f"Speedup plot saved to: {speedup_file}")
        else:
            plt.close()
    
    # Create efficiency plot
    if show_speedup and has_multi_gpu:
        fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE)
        has_efficiency_data = False
        all_efficiencies = []
        plotted_efficiency_x: List[float] = []
        estimated_samples: Set[float] = set()
        
        # Plot efficiency for each method with overlay
        for method in methods:
            method_results = results[method]
            method_sample_map = sample_key_maps[method]
            baseline_predictor = _build_baseline_predictor(
                method_results=method_results,
                axis_key_map=method_sample_map,
                baseline_gpu=1,
            )
            
            for gpu_count in gpu_counts:
                if gpu_count == 1:
                    continue
                    
                efficiencies = []
                valid_samples = []
                
                for samples in sample_sizes:
                    selected_key = _resolve_numeric_key(samples, method_results, method_sample_map)
                    if selected_key is None:
                        continue

                    sample_payload = method_results[selected_key]
                    current_key = _resolve_gpu_key(gpu_count, sample_payload)
                    if current_key is None:
                        continue

                    current = sample_payload[current_key]
                    baseline_mean, is_estimated = _resolve_baseline_mean(
                        axis_numeric=samples,
                        axis_payload=sample_payload,
                        baseline_predictor=baseline_predictor,
                        baseline_gpu=1,
                    )
                    current_mean = _extract_mean_from_result(current)
                    if baseline_mean is None or current_mean is None or current_mean == 0:
                        continue

                    speedup = baseline_mean / current_mean
                    efficiency = (speedup / gpu_count) * 100
                    if is_estimated:
                        estimated_samples.add(float(samples))
                    efficiencies.append(efficiency)
                    sample_for_plot = int(samples) if samples.is_integer() else samples
                    valid_samples.append(sample_for_plot)
                
                if efficiencies:
                    has_efficiency_data = True
                    all_efficiencies.extend(efficiencies)
                    plotted_efficiency_x.extend(float(value) for value in valid_samples)
                    linestyle = '-'
                    series_color = _shade_method_color_for_gpu(
                        base_color=method_base_colors[method],
                        gpu_count=gpu_count,
                        gpu_counts=gpu_counts,
                    )
                    marker_style = gpu_markers.get(gpu_count, 'o')
                    series_linewidth = gpu_linewidths.get(gpu_count, 2.2)
                    label = f'{gpu_count} GPUs'
                    if len(methods) > 1:
                        label += f' - {_method_display_name(method)}'
                    ax.plot(valid_samples, efficiencies,
                           marker=marker_style, markersize=SCALING_MARKER_SIZE, linewidth=series_linewidth,
                           color=series_color,
                           linestyle=linestyle,
                           label=label)
        
        if has_efficiency_data:
            baseline_note = (
                '* Estimated 1-GPU baseline used where missing'
                if estimated_samples
                else None
            )
            estimated_baseline_used = estimated_baseline_used or bool(estimated_samples)
            # Add 100% efficiency line
            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
            use_log_samples = (
                len(plotted_efficiency_x) > 1
                and min(plotted_efficiency_x) > 0
                and (max(plotted_efficiency_x) / min(plotted_efficiency_x)) > 100
            )
            reference_x = _left_annotation_anchor_x(
                plotted_efficiency_x,
                use_log=use_log_samples,
            )
            if reference_x is not None:
                ax.annotate(
                    'Perfect Scaling',
                    xy=(reference_x, 100),
                    xytext=(4, 2),
                    textcoords='offset points',
                    va='bottom',
                    ha='left',
                    color='green',
                    fontsize=PLOT_ANNOTATION_FONTSIZE,
                )
            
            ax.set_ylabel('Scaling Efficiency (%)', labelpad=12)
            use_log_samples = _set_sample_x_axis(
                ax,
                sample_sizes,
                min_upper_bound=sample_min_upper_bound,
            )
            if use_log_samples:
                ax.set_xlabel('Number of Samples (log scale)', labelpad=X_AXIS_LABEL_PAD)
            else:
                ax.set_xlabel('Number of Samples', labelpad=X_AXIS_LABEL_PAD)
            _set_sample_x_limits(
                ax,
                sample_sizes,
                min_upper_bound=sample_min_upper_bound,
            )
            _highlight_estimated_baseline_regions(
                ax=ax,
                estimated_axis_values=estimated_samples,
                all_axis_values=sample_sizes,
                use_log=use_log_samples,
            )
            _apply_scaling_axis_typography(ax)
            
            ax.grid(True, alpha=0.28, linestyle='--')
            ax.set_axisbelow(True)
            
            # Auto-scale efficiency axis to show >100% results while keeping the reference line visible.
            y_min, y_max = _auto_efficiency_ylim(all_efficiencies)
            ax.set_ylim(y_min, y_max)
            
            if sns is not None:
                sns.despine(ax=ax)
            
            _finalize_layout_with_bottom_legend(
                fig,
                ax,
                show_legend=show_legend,
                note_text=None,
                shaded_region_label=(
                    ESTIMATED_BASELINE_LEGEND_LABEL if baseline_note is not None else None
                ),
            )
            
            efficiency_file = os.path.join(output_dir, 'sample_scaling_efficiency.svg')
            plt.savefig(efficiency_file, dpi=PUBLICATION_DPI, bbox_inches='tight')
            plt.savefig(efficiency_file.replace('.svg', '.pdf'), bbox_inches='tight')
            plt.close()
            
            print(f"Efficiency plot saved to: {efficiency_file}")
        else:
            plt.close()

    if emit_shared_legend:
        legend_path = os.path.join(output_dir, legend_filename)
        if _render_grouped_legend_key(
            legend_registry=method_legend_registry,
            output_path=legend_path,
            dpi=PUBLICATION_DPI,
            shaded_region_label=(
                ESTIMATED_BASELINE_LEGEND_LABEL if estimated_baseline_used else None
            ),
        ):
            print(f"Shared legend saved to: {legend_path}")
