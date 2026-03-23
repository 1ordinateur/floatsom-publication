from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import *
from .helpers import *

PUBLICATION_FIGURE_TITLE_FONT_SIZE = 36.0
PUBLICATION_UNIFORM_DISPLAY_WIDTH = 3520.0
PUBLICATION_CANONICAL_CANVAS_WIDTH = 4391.0
_SVG_VIEWBOX_PATTERN = re.compile(
    r'viewBox=["\']\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*["\']'
)
_SVG_WIDTH_PATTERN = re.compile(r'width=["\']\s*([-+0-9.eE]+)')
_SVG_HEIGHT_PATTERN = re.compile(r'height=["\']\s*([-+0-9.eE]+)')
_SVG_OPEN_TAG_PATTERN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
_SVG_ID_PATTERN = re.compile(r'\bid=["\']([^"\']+)["\']')
_SVG_XML_DECL_PATTERN = re.compile(r"<\?xml[^>]*\?>", re.IGNORECASE)
_SVG_DOCTYPE_PATTERN = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE)

def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"

def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_image_mime_type(path)};base64,{encoded}"

def _image_intrinsic_size(path: Path) -> Tuple[float, float]:
    if path.suffix.lower() != ".svg":
        return (1.0, 1.0)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return (1.0, 1.0)
    viewbox_match = _SVG_VIEWBOX_PATTERN.search(text)
    if viewbox_match is not None:
        return (
            max(1.0, float(viewbox_match.group(3))),
            max(1.0, float(viewbox_match.group(4))),
        )
    width_match = _SVG_WIDTH_PATTERN.search(text)
    height_match = _SVG_HEIGHT_PATTERN.search(text)
    if width_match is not None and height_match is not None:
        return (
            max(1.0, float(width_match.group(1))),
            max(1.0, float(height_match.group(1))),
        )
    return (1.0, 1.0)


def _svg_viewbox(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if path.suffix.lower() != ".svg":
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    viewbox_match = _SVG_VIEWBOX_PATTERN.search(text)
    if viewbox_match is not None:
        return tuple(float(viewbox_match.group(idx)) for idx in range(1, 5))
    width_match = _SVG_WIDTH_PATTERN.search(text)
    height_match = _SVG_HEIGHT_PATTERN.search(text)
    if width_match is not None and height_match is not None:
        return (0.0, 0.0, float(width_match.group(1)), float(height_match.group(1)))
    return None


def _prefix_svg_fragment_ids(svg_fragment: str, id_prefix: str) -> str:
    ids = list(dict.fromkeys(_SVG_ID_PATTERN.findall(svg_fragment)))
    if not ids:
        return svg_fragment
    updated = svg_fragment
    for raw_id in ids:
        prefixed_id = f"{id_prefix}_{raw_id}"
        updated = re.sub(
            rf'(\bid=["\']){re.escape(raw_id)}(["\'])',
            rf'\1{prefixed_id}\2',
            updated,
        )
        updated = updated.replace(f"url(#{raw_id})", f"url(#{prefixed_id})")
        updated = updated.replace(f'href="#{raw_id}"', f'href="#{prefixed_id}"')
        updated = updated.replace(f"href='#{raw_id}'", f"href='#{prefixed_id}'")
        updated = updated.replace(f'xlink:href="#{raw_id}"', f'xlink:href="#{prefixed_id}"')
        updated = updated.replace(f"xlink:href='#{raw_id}'", f"xlink:href='#{prefixed_id}'")
    return updated


def _inline_svg_fragment_markup(
    path: Path,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    id_prefix: str,
    clip_path_attrs: str = "",
) -> Optional[str]:
    viewbox = _svg_viewbox(path)
    if viewbox is None:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    text = _SVG_XML_DECL_PATTERN.sub("", text)
    text = _SVG_DOCTYPE_PATTERN.sub("", text)
    open_tag_match = _SVG_OPEN_TAG_PATTERN.search(text)
    close_tag_idx = text.lower().rfind("</svg>")
    if open_tag_match is None or close_tag_idx < 0 or close_tag_idx <= open_tag_match.end():
        return None
    svg_body = text[open_tag_match.end():close_tag_idx]
    svg_body = _prefix_svg_fragment_ids(svg_body, id_prefix)
    min_x, min_y, src_width, src_height = viewbox
    scale = min(
        float(width) / max(1.0, float(src_width)),
        float(height) / max(1.0, float(src_height)),
    )
    rendered_width = float(src_width) * scale
    rendered_height = float(src_height) * scale
    offset_x = float(x) + ((float(width) - rendered_width) / 2.0)
    offset_y = float(y) + ((float(height) - rendered_height) / 2.0)
    return (
        f'    <g transform="translate({offset_x},{offset_y}) scale({scale},{scale}) '
        f'translate({-min_x},{-min_y})"{clip_path_attrs}>\n'
        f'{svg_body}\n'
        f'    </g>'
    )


def _embed_visual_markup(
    path: Path,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    id_prefix: str,
    clip_path_attrs: str = "",
    preserve_aspect_ratio: str = "xMidYMid meet",
) -> str:
    inline_markup = _inline_svg_fragment_markup(
        path,
        x=x,
        y=y,
        width=width,
        height=height,
        id_prefix=id_prefix,
        clip_path_attrs=clip_path_attrs,
    )
    if inline_markup is not None:
        return inline_markup
    image_uri = _image_data_uri(path)
    return (
        f'    <image href="{image_uri}" xlink:href="{image_uri}" '
        f'x="{x}" y="{y}" width="{width}" height="{height}" '
        f'preserveAspectRatio="{preserve_aspect_ratio}"{clip_path_attrs}/>'
    )


def _resolve_shared_x_label_baseline_y(
    *,
    row_bottom: float,
    obstacle_top: float,
    font_size: float,
) -> float:
    """Return a shared x-label baseline that reserves vertical space for the glyphs."""
    ascent = max(1.0, float(font_size) * 0.78)
    descent = max(1.0, float(font_size) * 0.22)
    panel_clearance = max(8.0, float(font_size) * 0.18)
    obstacle_clearance = max(8.0, float(font_size) * 0.18)

    baseline_min = float(row_bottom) + ascent + panel_clearance
    baseline_max = float(obstacle_top) - descent - obstacle_clearance
    if baseline_max >= baseline_min:
        return baseline_min + ((baseline_max - baseline_min) * 0.45)

    # Tight layouts can still occur when a legend band is very close to the panel.
    # Fall back to the center of the remaining corridor, clamped to stay below the obstacle.
    corridor = max(0.0, float(obstacle_top) - float(row_bottom))
    fallback_low = float(row_bottom) + min(corridor * 0.35, max(10.0, float(font_size) * 0.4))
    fallback_high = float(obstacle_top) - max(6.0, float(font_size) * 0.2)
    if fallback_high >= fallback_low:
        return fallback_low + ((fallback_high - fallback_low) * 0.5)
    return min(fallback_low, fallback_high)


def _external_row_center_positions(
    *,
    row_labels: Sequence[str],
    label_top_y: float,
    label_span: float,
) -> Dict[str, float]:
    if not row_labels:
        return {}
    row_height = float(label_span) / float(len(row_labels))
    return {
        label: float(label_top_y) + ((idx + 0.5) * row_height)
        for idx, label in enumerate(row_labels)
    }

def _compose_panel_matrix_svg(
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = None,
    legend_gap_override: Optional[float] = None,
    secondary_legend_path: Optional[Path] = None,
    secondary_legend_after_row: Optional[int] = None,
    secondary_legend_y_nudge: float = 0.0,
    secondary_legend_band_height_override: Optional[int] = None,
    secondary_legend_gap_override: Optional[float] = None,
    legend_scale_override: Optional[float] = None,
    secondary_legend_scale_override: Optional[float] = None,
    margin_bottom_override: Optional[float] = None,
    gap_x_override: Optional[float] = None,
    gap_y_override: Optional[float] = None,
    panel_width_override: Optional[float] = None,
    panel_height_override: Optional[float] = None,
    panel_header_height_override: Optional[float] = None,
    title_font_size_override: Optional[float] = None,
    panel_label_font_size_override: Optional[float] = None,
    panel_caption_font_size_override: Optional[float] = None,
    direction_label_font_size_override: Optional[float] = None,
    direction_labels_by_row: Optional[Sequence[str]] = None,
    direction_label_positions_by_row: Optional[Sequence[str]] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_y_label_font_size_override: Optional[float] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_label_font_size_override: Optional[float] = None,
    display_width_override: Optional[float] = None,
    canvas_width_override: Optional[float] = None,
    external_y_labels_by_row: Optional[Sequence[Optional[Sequence[str]]]] = None,
    external_y_label_font_size_override: Optional[float] = None,
    plot_box_fractions_by_row: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    left_annotation_gutter_extra_override: Optional[float] = None,
    title_band_override: Optional[float] = None,
    show_panel_annotations: bool = True,
) -> bool:
    if not panel_rows:
        return False
    n_rows = len(panel_rows)
    n_cols = len(panel_rows[0])
    if n_rows <= 0 or n_cols <= 0:
        return False
    for row in panel_rows:
        if len(row) != n_cols:
            raise ValueError("All panel rows must have the same number of columns.")

    missing = [path for row in panel_rows for _, _, path in row if not path.exists()]
    if missing:
        return False

    base_margin_left = 24.0
    margin_left = base_margin_left
    margin_right = 12
    margin_top = 42
    margin_bottom = (
        float(margin_bottom_override)
        if margin_bottom_override is not None
        else 10.0
    )
    margin_bottom = max(0.0, margin_bottom)
    panel_width = float(panel_width_override) if panel_width_override is not None else 700.0
    panel_height = float(panel_height_override) if panel_height_override is not None else 465.0
    panel_header_height = (
        float(panel_header_height_override)
        if panel_header_height_override is not None
        else 58.0
    )
    gap_x = float(gap_x_override) if gap_x_override is not None else 8.0
    gap_y = float(gap_y_override) if gap_y_override is not None else 24.0
    title_font_size = (
        float(title_font_size_override)
        if title_font_size_override is not None
        else PUBLICATION_FIGURE_TITLE_FONT_SIZE
    )
    title_lines = [line.strip() for line in str(title).splitlines() if line.strip()]
    if not title_lines:
        title_lines = [str(title).strip()]
    title_line_height = max(title_font_size + 6.0, title_font_size * 1.15)
    min_title_band = 24.0 + (len(title_lines) - 1) * title_line_height + 18.0
    title_band = (
        max(float(title_band_override), min_title_band)
        if title_band_override is not None
        else max(52.0, title_font_size + 24.0, min_title_band)
    )
    panel_label_font_size = (
        float(panel_label_font_size_override)
        if panel_label_font_size_override is not None
        else 30.0
    )
    panel_caption_font_size = (
        float(panel_caption_font_size_override)
        if panel_caption_font_size_override is not None
        else 24.0
    )
    direction_label_font_size = (
        float(direction_label_font_size_override)
        if direction_label_font_size_override is not None
        else panel_caption_font_size
    )
    shared_x_label_font_size = (
        float(shared_x_label_font_size_override)
        if shared_x_label_font_size_override is not None
        else 21.0
    )
    shared_y_label_font_size = (
        float(shared_y_label_font_size_override)
        if shared_y_label_font_size_override is not None
        else direction_label_font_size
    )
    external_y_label_font_size = (
        float(external_y_label_font_size_override)
        if external_y_label_font_size_override is not None
        else TRIPANEL_FOREST_Y_TICK_LABEL_FONT_SIZE * 1.4
    )
    row_shared_y_labels: List[str] = []
    for row_idx in range(n_rows):
        label = ""
        if shared_y_labels_by_row and row_idx < len(shared_y_labels_by_row):
            label = str(shared_y_labels_by_row[row_idx]).strip()
        row_shared_y_labels.append(label)
    row_shared_x_labels: List[str] = []
    for row_idx in range(n_rows):
        label = ""
        if shared_x_labels_by_row and row_idx < len(shared_x_labels_by_row):
            label = str(shared_x_labels_by_row[row_idx]).strip()
        row_shared_x_labels.append(label)
    row_external_y_labels: List[Optional[List[str]]] = []
    for row_idx in range(n_rows):
        row_labels: Optional[List[str]] = None
        if external_y_labels_by_row and row_idx < len(external_y_labels_by_row):
            raw_labels = external_y_labels_by_row[row_idx]
            if raw_labels:
                row_labels = [str(label).strip() for label in raw_labels if str(label).strip()]
        row_external_y_labels.append(row_labels)
    max_external_label_length = max(
        (len(label) for row_labels in row_external_y_labels if row_labels for label in row_labels),
        default=0,
    )
    external_y_label_gutter_width = 0.0
    if max_external_label_length > 0:
        external_y_label_gutter_width = max(
            104.0,
            min(420.0, external_y_label_font_size * 0.78 * max_external_label_length + 36.0),
        )
    has_left_vertical_direction_label = any(
        str(direction_labels_by_row[row_idx]).strip()
        and (
            not direction_label_positions_by_row
            or row_idx >= len(direction_label_positions_by_row)
            or str(direction_label_positions_by_row[row_idx]).strip().lower() == "left_vertical"
        )
        for row_idx in range(min(n_rows, len(direction_labels_by_row or [])))
    )
    has_shared_y_labels = any(row_shared_y_labels)
    left_annotation_gutter_extra = max(
        0.0,
        float(left_annotation_gutter_extra_override)
        if left_annotation_gutter_extra_override is not None
        else 0.0,
    )
    left_vertical_direction_lane_width = (
        max(24.0, direction_label_font_size * 0.62) if has_left_vertical_direction_label else 0.0
    )
    shared_y_label_lane_width = (
        max(24.0, shared_y_label_font_size * 0.62) if has_shared_y_labels else 0.0
    )
    left_side_lane_gap = 28.0 if has_left_vertical_direction_label and has_shared_y_labels else 0.0
    side_annotation_gutter_width = (
        left_vertical_direction_lane_width
        + left_side_lane_gap
        + shared_y_label_lane_width
    )
    if has_left_vertical_direction_label or has_shared_y_labels:
        side_annotation_gutter_width += left_annotation_gutter_extra
    annotation_band_height = (
        max(
            float(panel_header_height),
            panel_caption_font_size * 2.4,
            panel_label_font_size * 1.45,
            direction_label_font_size * 2.15 if direction_labels_by_row else 0.0,
        )
        if show_panel_annotations
        else 0.0
    )
    panel_label_x = 44.0
    panel_caption_x = max(132.0, panel_label_x + panel_label_font_size * 1.65)
    panel_annotation_baseline_y = max(
        panel_label_font_size + 8.0,
        annotation_band_height * 0.48,
    )
    direction_label_baseline_y = min(
        annotation_band_height - 4.0,
        panel_annotation_baseline_y + direction_label_font_size + 18.0,
    )
    panel_image_y_offset = annotation_band_height
    panel_image_height = max(1.0, panel_height - panel_image_y_offset)
    row_plot_box_fractions: List[Optional[Tuple[float, float, float, float]]] = []
    for row_idx in range(n_rows):
        row_plot_box: Optional[Tuple[float, float, float, float]] = None
        if plot_box_fractions_by_row and row_idx < len(plot_box_fractions_by_row):
            raw_box = plot_box_fractions_by_row[row_idx]
            if raw_box is not None and len(raw_box) == 4:
                left, right, top, bottom = [float(value) for value in raw_box]
                left = min(max(0.0, left), 1.0)
                right = min(max(left, right), 1.0)
                top = min(max(0.0, top), 1.0)
                bottom = min(max(top, bottom), 1.0)
                if right > left and bottom > top:
                    row_plot_box = (left, right, top, bottom)
        row_plot_box_fractions.append(row_plot_box)

    def _normalize_after_row(candidate: Optional[int]) -> Optional[int]:
        if candidate is None:
            return None
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            return None
        if 1 <= value < n_rows:
            return value
        return None

    legend_slots: List[Dict[str, Any]] = []

    def _append_legend_slot(
        path: Optional[Path],
        after_row: Optional[int],
        y_nudge: float,
        band_height_override: Optional[int],
        gap_override: Optional[float],
        scale_override: Optional[float],
    ) -> None:
        if path is None or not path.exists():
            return
        band_height = (
            int(band_height_override)
            if band_height_override is not None
            else PUBLICATION_LEGEND_ROW_HEIGHT
        )
        band_height = max(1, band_height)
        gap = float(gap_override) if gap_override is not None else 3.0
        gap = max(0.0, gap)
        legend_slots.append(
            {
                "path": path,
                "after_row": _normalize_after_row(after_row),
                "y_nudge": float(y_nudge),
                "band_height": band_height,
                "gap": gap,
                "scale": (max(0.1, float(scale_override)) if scale_override is not None else None),
                "order": len(legend_slots),
            }
        )

    _append_legend_slot(
        path=legend_path,
        after_row=legend_after_row,
        y_nudge=legend_y_nudge,
        band_height_override=legend_band_height_override,
        gap_override=legend_gap_override,
        scale_override=legend_scale_override,
    )
    _append_legend_slot(
        path=secondary_legend_path,
        after_row=secondary_legend_after_row,
        y_nudge=secondary_legend_y_nudge,
        band_height_override=secondary_legend_band_height_override,
        gap_override=secondary_legend_gap_override,
        scale_override=secondary_legend_scale_override,
    )

    between_slots = sorted(
        [slot for slot in legend_slots if slot["after_row"] is not None],
        key=lambda slot: (int(slot["after_row"]), int(slot["order"])),
    )
    bottom_slots = sorted(
        [slot for slot in legend_slots if slot["after_row"] is None],
        key=lambda slot: int(slot["order"]),
    )
    between_shared_x_label_band_heights: List[float] = []
    for row_idx in range(n_rows):
        if row_idx < (n_rows - 1) and row_shared_x_labels[row_idx]:
            between_shared_x_label_band_heights.append(
                max(44.0, shared_x_label_font_size * 1.25)
            )
        else:
            between_shared_x_label_band_heights.append(0.0)
    total_between_shared_x_label_space = sum(between_shared_x_label_band_heights)
    total_between_space = (
        total_between_shared_x_label_space
        + sum(float(slot["gap"]) + float(slot["band_height"]) for slot in between_slots)
    )
    total_bottom_space = sum(float(slot["gap"]) + float(slot["band_height"]) for slot in bottom_slots)
    last_row_shared_x_label = row_shared_x_labels[n_rows - 1] if n_rows > 0 else ""
    bottom_shared_x_label_band_height = (
        max(44.0, shared_x_label_font_size * 1.25) if last_row_shared_x_label else 0.0
    )
    panel_block_height = n_rows * panel_height + (n_rows - 1) * gap_y
    panel_block_width = n_cols * panel_width + (n_cols - 1) * gap_x
    natural_panel_block_left = (
        margin_left + side_annotation_gutter_width + external_y_label_gutter_width
    )
    natural_canvas_width = float(
        natural_panel_block_left + margin_right + panel_block_width
    )
    canvas_width_target = natural_canvas_width
    if canvas_width_override is not None:
        canvas_width_target = max(float(canvas_width_override), natural_canvas_width)
    left_canvas_pad = max(0.0, canvas_width_target - natural_canvas_width) / 2.0
    margin_left = base_margin_left + left_canvas_pad
    panel_block_left = margin_left + side_annotation_gutter_width + external_y_label_gutter_width
    canvas_width = int(round(canvas_width_target))
    canvas_height = (
        margin_top
        + title_band
        + panel_block_height
        + total_between_space
        + bottom_shared_x_label_band_height
        + total_bottom_space
        + margin_bottom
    )
    display_width = (
        float(display_width_override)
        if display_width_override is not None
        else PUBLICATION_UNIFORM_DISPLAY_WIDTH
    )
    display_width = max(1.0, display_width)
    display_scale = display_width / float(canvas_width)
    display_height = float(canvas_height) * display_scale

    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{display_width:.1f}" height="{display_height:.1f}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}">'
        ),
        f'  <rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" fill="white"/>',
    ]
    for line_idx, line in enumerate(title_lines):
        parts.append(
            f'  <text x="{margin_left}" y="{margin_top + 24.0 + line_idx * title_line_height}" '
            f'font-family="DejaVu Sans, Arial, sans-serif" '
            f'font-size="{title_font_size}" font-weight="700" fill="#1f1f1f">{html.escape(line)}</text>'
        )

    between_shift_by_row: List[float] = []
    for row_idx in range(n_rows):
        row_shift = 0.0
        for prior_row_idx in range(row_idx):
            row_shift += between_shared_x_label_band_heights[prior_row_idx]
        for slot in between_slots:
            after_row_idx = int(slot["after_row"])
            if row_idx >= after_row_idx:
                row_shift += float(slot["gap"]) + float(slot["band_height"])
        between_shift_by_row.append(row_shift)

    row_top_by_idx: List[float] = [
        margin_top + title_band + row_idx * (panel_height + gap_y) + between_shift_by_row[row_idx]
        for row_idx in range(n_rows)
    ]

    between_legend_layouts: List[Dict[str, Any]] = []
    between_legend_cumulative = 0.0
    for slot in between_slots:
        after_row_idx = int(slot["after_row"])
        legend_band_height = int(slot["band_height"])
        shared_x_label_band_before = sum(between_shared_x_label_band_heights[:after_row_idx])
        legend_y = (
            margin_top
            + title_band
            + after_row_idx * panel_height
            + (after_row_idx - 1) * gap_y
            + shared_x_label_band_before
            + between_legend_cumulative
            + float(slot["gap"])
            + float(slot["y_nudge"])
        )
        between_legend_layouts.append(
            {
                "slot": slot,
                "y": legend_y,
                "band_height": legend_band_height,
            }
        )
        between_legend_cumulative += float(slot["gap"]) + legend_band_height

    bottom_base_y = margin_top + title_band + panel_block_height + total_between_space + bottom_shared_x_label_band_height
    bottom_legend_layouts: List[Dict[str, Any]] = []
    bottom_cumulative = 0.0
    for slot in bottom_slots:
        legend_band_height = int(slot["band_height"])
        legend_y = (
            bottom_base_y
            + bottom_cumulative
            + float(slot["gap"])
            + float(slot["y_nudge"])
        )
        bottom_legend_layouts.append(
            {
                "slot": slot,
                "y": legend_y,
                "band_height": legend_band_height,
            }
        )
        bottom_cumulative += float(slot["gap"]) + legend_band_height

    def _row_panel_left(row_idx: int) -> float:
        return panel_block_left

    def _natural_image_render_size(image_path: Path) -> Tuple[float, float]:
        source_width, source_height = _image_intrinsic_size(image_path)
        render_scale = min(panel_width / source_width, panel_image_height / source_height)
        return (source_width * render_scale, source_height * render_scale)

    reference_plot_box_row_idx: Optional[int] = next(
        (row_idx for row_idx, row_plot_box in enumerate(row_plot_box_fractions) if row_plot_box is not None),
        None,
    )
    target_plot_left = 0.0
    target_plot_top = 0.0
    target_plot_width = panel_width
    target_plot_height = panel_image_height
    if reference_plot_box_row_idx is not None:
        reference_plot_box = row_plot_box_fractions[reference_plot_box_row_idx]
        reference_image_path = panel_rows[reference_plot_box_row_idx][0][2]
        reference_render_width, reference_render_height = _natural_image_render_size(reference_image_path)
        if reference_plot_box is not None:
            ref_left, ref_right, ref_top, ref_bottom = reference_plot_box
            target_plot_left = reference_render_width * ref_left
            target_plot_top = reference_render_height * ref_top
            target_plot_width = reference_render_width * (ref_right - ref_left)
            target_plot_height = reference_render_height * (ref_bottom - ref_top)

    panel_parts: List[str] = []
    for row_idx, row in enumerate(panel_rows):
        row_panel_left = _row_panel_left(row_idx)
        for col_idx, (panel_letter, panel_title, image_path) in enumerate(row):
            x = row_panel_left + col_idx * (panel_width + gap_x)
            y = row_top_by_idx[row_idx]
            letter = html.escape(str(panel_letter))
            caption = html.escape(str(panel_title))
            image_x = 0.0
            image_y = panel_image_y_offset
            image_width = panel_width
            image_height = panel_image_height
            clip_path_attrs = ""
            row_plot_box = row_plot_box_fractions[row_idx]
            if row_plot_box is not None:
                natural_render_width, natural_render_height = _natural_image_render_size(image_path)
                left_fraction, right_fraction, top_fraction, bottom_fraction = row_plot_box
                plot_box_width_fraction = max(1e-6, right_fraction - left_fraction)
                plot_box_height_fraction = max(1e-6, bottom_fraction - top_fraction)
                scale_x = target_plot_width / max(1e-6, natural_render_width * plot_box_width_fraction)
                scale_y = target_plot_height / max(1e-6, natural_render_height * plot_box_height_fraction)
                max_scale_x = panel_width / max(1e-6, natural_render_width)
                max_scale_y = panel_image_height / max(1e-6, natural_render_height)
                scale = min(scale_x, scale_y, max_scale_x, max_scale_y)
                image_width = natural_render_width * scale
                image_height = natural_render_height * scale
                image_x = target_plot_left - image_width * left_fraction
                image_y = panel_image_y_offset + target_plot_top - image_height * top_fraction
                max_image_x = panel_width - image_width
                max_image_y = panel_image_y_offset + panel_image_height - image_height
                image_x = min(max(image_x, 0.0), max_image_x)
                image_y = min(max(image_y, panel_image_y_offset), max_image_y)
                clip_id = f"panel_clip_r{row_idx}_c{col_idx}"
                clip_path_attrs = f' clip-path="url(#{clip_id})"'
            panel_parts.extend(
                [
                    f'  <g transform="translate({x},{y})">',
                    (
                    f'    <defs><clipPath id="{clip_id}"><rect x="0" y="{panel_image_y_offset}" '
                        f'width="{panel_width}" height="{panel_image_height}"/></clipPath></defs>'
                        if row_plot_box is not None
                        else ""
                    ),
                    _embed_visual_markup(
                        image_path,
                        x=image_x,
                        y=image_y,
                        width=image_width,
                        height=image_height,
                        id_prefix=f"panel_r{row_idx}_c{col_idx}",
                        clip_path_attrs=clip_path_attrs,
                        preserve_aspect_ratio="xMinYMin meet",
                    ),
                    "  </g>",
                ]
            )
            if show_panel_annotations:
                panel_parts[-1:-1] = [
                    (
                        f'    <rect x="0" y="0" width="{panel_width}" height="{annotation_band_height}" '
                        f'fill="white" fill-opacity="0.98"/>'
                    ),
                    f'    <text x="{panel_label_x}" y="{panel_annotation_baseline_y}" text-anchor="start" font-family="DejaVu Sans, Arial, sans-serif" font-size="{panel_label_font_size}" '
                    f'font-weight="700" fill="#1f1f1f">{letter}</text>',
                    f'    <text x="{panel_caption_x}" y="{panel_annotation_baseline_y}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{panel_caption_font_size}" '
                    f'font-weight="600" fill="#2f2f2f">{caption}</text>',
                ]

    direction_label_parts: List[str] = []
    panel_block_center_x = panel_block_left + (panel_block_width / 2.0)

    def _row_plot_span(row_idx: int) -> Tuple[float, float]:
        row_panel_left = _row_panel_left(row_idx)
        first_panel_x = row_panel_left
        last_panel_x = row_panel_left + (n_cols - 1) * (panel_width + gap_x)
        if row_external_y_labels[row_idx]:
            left_fraction = PUBLICATION_SAFE_LEFT_MARGIN_NO_Y_LABELS
            right_fraction = 0.996
        else:
            left_fraction = 0.0
            right_fraction = 1.0
        return (
            first_panel_x + panel_width * left_fraction,
            last_panel_x + panel_width * right_fraction,
        )

    def _row_vertical_annotation_x(row_idx: int, *, kind: str) -> float:
        row_external_gutter_shift = (
            external_y_label_gutter_width
            if external_y_label_gutter_width > 0.0 and not row_external_y_labels[row_idx]
            else 0.0
        )
        if kind == "direction":
            return (
                margin_left
                + (left_vertical_direction_lane_width / 2.0)
                + row_external_gutter_shift
            )
        default_shared_y_x = (
            margin_left
            + left_vertical_direction_lane_width
            + left_side_lane_gap
            + (shared_y_label_lane_width / 2.0)
        )
        return default_shared_y_x + row_external_gutter_shift

    if direction_labels_by_row:
        for row_idx in range(min(n_rows, len(direction_labels_by_row))):
            direction_label = str(direction_labels_by_row[row_idx]).strip()
            if not direction_label:
                continue
            direction_label_position = "top_center"
            if direction_label_positions_by_row and row_idx < len(direction_label_positions_by_row):
                direction_label_position = str(direction_label_positions_by_row[row_idx]).strip().lower() or "top_center"
            row_top = row_top_by_idx[row_idx]
            if direction_label_position == "left_vertical":
                direction_label_x = _row_vertical_annotation_x(row_idx, kind="direction")
                direction_label_y = row_top + (panel_height / 2.0)
                direction_label_parts.append(
                    f'  <text x="{direction_label_x}" y="{direction_label_y}" text-anchor="middle" '
                    f'transform="rotate(-90 {direction_label_x} {direction_label_y})" '
                    f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{direction_label_font_size}" '
                    f'font-weight="700" fill="#2f2f2f">{html.escape(direction_label)}</text>'
                )
            else:
                row_plot_left, row_plot_right = _row_plot_span(row_idx)
                row_plot_center_x = row_plot_left + ((row_plot_right - row_plot_left) / 2.0)
                direction_label_y = row_top + direction_label_baseline_y
                direction_label_parts.append(
                    f'  <text x="{row_plot_center_x}" y="{direction_label_y}" text-anchor="middle" '
                    f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{direction_label_font_size}" '
                    f'font-weight="700" fill="#2f2f2f">{html.escape(direction_label)}</text>'
                )

    if has_shared_y_labels:
        for row_idx, row_label in enumerate(row_shared_y_labels):
            if not row_label:
                continue
            row_top = row_top_by_idx[row_idx]
            label_x = _row_vertical_annotation_x(row_idx, kind="shared_y")
            label_y = row_top + (panel_height / 2.0)
            parts.append(
                f'  <text x="{label_x}" y="{label_y}" text-anchor="middle" '
                f'transform="rotate(-90 {label_x} {label_y})" '
                f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{shared_y_label_font_size}" '
                f'font-weight="600" fill="#2f2f2f">{html.escape(row_label)}</text>'
            )

    external_y_label_parts: List[str] = []
    if external_y_label_gutter_width > 0.0:
        plot_area_top_fraction = 0.045
        plot_area_bottom_fraction = 0.08
        plot_area_height_fraction = 1.0 - plot_area_top_fraction - plot_area_bottom_fraction
        for row_idx, row_labels in enumerate(row_external_y_labels):
            if not row_labels:
                continue
            row_panel_left = _row_panel_left(row_idx)
            label_anchor_x = row_panel_left - 8.0
            separator_x1 = max(margin_left + 10.0, row_panel_left - min(72.0, external_y_label_gutter_width * 0.58))
            separator_x2 = row_panel_left - 12.0
            row_top = row_top_by_idx[row_idx]
            label_top_y = row_top + panel_image_y_offset + panel_image_height * plot_area_top_fraction
            label_span = panel_image_height * plot_area_height_fraction
            row_height = label_span / float(len(row_labels))
            band_shift = row_height * 0.18
            bottom_compression = row_height * 0.18
            label_top_y += band_shift
            label_span -= band_shift + bottom_compression
            label_positions = _external_row_center_positions(
                row_labels=row_labels,
                label_top_y=label_top_y,
                label_span=label_span,
            )
            for label, label_y in label_positions.items():
                external_y_label_parts.append(
                    f'  <text x="{label_anchor_x}" y="{label_y}" text-anchor="end" dominant-baseline="middle" '
                    f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{external_y_label_font_size}" '
                    f'font-weight="500" fill="#2f2f2f">{html.escape(label)}</text>'
                )
            for upper_label, lower_label in _dataset_group_boundaries(row_labels):
                if upper_label not in label_positions or lower_label not in label_positions:
                    continue
                separator_y = (float(label_positions[upper_label]) + float(label_positions[lower_label])) / 2.0
                external_y_label_parts.append(
                    f'  <line x1="{separator_x1}" y1="{separator_y}" x2="{separator_x2}" y2="{separator_y}" '
                    f'stroke="{DATASET_GROUP_SEPARATOR_COLOR}" stroke-width="{DATASET_GROUP_SEPARATOR_LINEWIDTH}" '
                    f'stroke-opacity="{DATASET_GROUP_SEPARATOR_ALPHA}"/>'
                )

    shared_x_label_parts: List[str] = []
    if row_shared_x_labels:
        for row_idx, row_label in enumerate(row_shared_x_labels):
            if not row_label:
                continue
            row_top = row_top_by_idx[row_idx]
            row_bottom = row_top + panel_height
            if row_idx == (n_rows - 1) and bottom_shared_x_label_band_height > 0.0:
                label_y = row_bottom + max(
                    shared_x_label_font_size * 0.95,
                    bottom_shared_x_label_band_height * 0.62,
                )
            elif row_idx < (n_rows - 1) and between_shared_x_label_band_heights[row_idx] > 0.0:
                label_band_height = between_shared_x_label_band_heights[row_idx]
                label_y = row_bottom + max(
                    shared_x_label_font_size * 0.95,
                    label_band_height * 0.62,
                )
            else:
                obstacle_candidates: List[float] = [canvas_height - margin_bottom]
                if row_idx + 1 < len(row_top_by_idx):
                    obstacle_candidates.append(float(row_top_by_idx[row_idx + 1]))
                for layout in between_legend_layouts + bottom_legend_layouts:
                    legend_top = float(layout["y"])
                    if legend_top >= row_bottom - 1.0:
                        obstacle_candidates.append(legend_top)
                nearest_obstacle = min(obstacle_candidates) if obstacle_candidates else (row_bottom + 40.0)
                label_y = _resolve_shared_x_label_baseline_y(
                    row_bottom=row_bottom,
                    obstacle_top=nearest_obstacle,
                    font_size=shared_x_label_font_size,
                )
            row_plot_left, row_plot_right = _row_plot_span(row_idx)
            row_plot_center_x = row_plot_left + ((row_plot_right - row_plot_left) / 2.0)
            shared_x_label_parts.append(
                f'  <text x="{row_plot_center_x}" y="{label_y}" text-anchor="middle" '
                f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{shared_x_label_font_size}" '
                f'font-weight="600" fill="#2f2f2f">{html.escape(row_label)}</text>'
            )
    parts.extend(external_y_label_parts)
    parts.extend(panel_parts)
    parts.extend(shared_x_label_parts)
    parts.extend(direction_label_parts)

    legend_fit_width = float(canvas_width)
    legend_center_x = panel_block_left + (panel_block_width / 2.0)

    def _legend_embed_geometry(
        slot: Dict[str, Any],
        legend_band_height: int,
        legend_fit_width: float,
        legend_center_x: float,
    ) -> Tuple[float, float, float, float]:
        legend_scale_override = slot.get("scale", None)
        source_width, source_height = _image_intrinsic_size(Path(slot["path"]))
        fit_scale = min(
            legend_fit_width / max(1.0, source_width),
            float(legend_band_height) / max(1.0, source_height),
        )
        if legend_scale_override is not None:
            fit_scale = min(fit_scale, float(legend_scale_override))
        fit_scale = max(0.1, fit_scale)
        legend_width_scaled = source_width * fit_scale
        legend_height_scaled = source_height * fit_scale
        legend_offset_x = float(legend_center_x) - (legend_width_scaled / 2.0)
        legend_offset_y = (legend_band_height - legend_height_scaled) / 2.0
        return legend_width_scaled, legend_height_scaled, legend_offset_x, legend_offset_y

    for layout in between_legend_layouts:
        slot = layout["slot"]
        legend_band_height = int(layout["band_height"])
        legend_y = float(layout["y"])
        (
            legend_width_scaled,
            legend_height_scaled,
            legend_offset_x,
            legend_offset_y,
        ) = _legend_embed_geometry(slot, legend_band_height, legend_fit_width, legend_center_x)
        parts.extend(
            [
                f'  <g transform="translate(0,{legend_y})">',
                _embed_visual_markup(
                    Path(slot["path"]),
                    x=legend_offset_x,
                    y=legend_offset_y,
                    width=legend_width_scaled,
                    height=legend_height_scaled,
                    id_prefix=f"between_legend_{layout['slot']['order']}",
                ),
                "  </g>",
            ]
        )

    for layout in bottom_legend_layouts:
        slot = layout["slot"]
        legend_band_height = int(layout["band_height"])
        legend_y = float(layout["y"])
        (
            legend_width_scaled,
            legend_height_scaled,
            legend_offset_x,
            legend_offset_y,
        ) = _legend_embed_geometry(slot, legend_band_height, legend_fit_width, legend_center_x)
        parts.extend(
            [
                f'  <g transform="translate(0,{legend_y})">',
                _embed_visual_markup(
                    Path(slot["path"]),
                    x=legend_offset_x,
                    y=legend_offset_y,
                    width=legend_width_scaled,
                    height=legend_height_scaled,
                    id_prefix=f"bottom_legend_{layout['slot']['order']}",
                ),
                "  </g>",
            ]
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return True

def _compose_panel_matrix_from_images(
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = None,
    legend_gap_override: Optional[float] = None,
    secondary_legend_path: Optional[Path] = None,
    secondary_legend_after_row: Optional[int] = None,
    secondary_legend_y_nudge: float = 0.0,
    secondary_legend_band_height_override: Optional[int] = None,
    secondary_legend_gap_override: Optional[float] = None,
    legend_scale_override: Optional[float] = None,
    secondary_legend_scale_override: Optional[float] = None,
    margin_bottom_override: Optional[float] = None,
    gap_x_override: Optional[float] = None,
    gap_y_override: Optional[float] = None,
    panel_width_override: Optional[float] = None,
    panel_height_override: Optional[float] = None,
    panel_header_height_override: Optional[float] = None,
    title_font_size_override: Optional[float] = None,
    panel_label_font_size_override: Optional[float] = None,
    panel_caption_font_size_override: Optional[float] = None,
    direction_label_font_size_override: Optional[float] = None,
    direction_labels_by_row: Optional[Sequence[str]] = None,
    direction_label_positions_by_row: Optional[Sequence[str]] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_y_label_font_size_override: Optional[float] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_label_font_size_override: Optional[float] = None,
    display_width_override: Optional[float] = None,
    canvas_width_override: Optional[float] = None,
    external_y_labels_by_row: Optional[Sequence[Optional[Sequence[str]]]] = None,
    external_y_label_font_size_override: Optional[float] = None,
    plot_box_fractions_by_row: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    left_annotation_gutter_extra_override: Optional[float] = None,
    title_band_override: Optional[float] = None,
    show_panel_annotations: bool = True,
) -> bool:
    _ = dpi
    return _compose_panel_matrix_svg(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        secondary_legend_path=secondary_legend_path,
        secondary_legend_after_row=secondary_legend_after_row,
        secondary_legend_y_nudge=secondary_legend_y_nudge,
        secondary_legend_band_height_override=secondary_legend_band_height_override,
        secondary_legend_gap_override=secondary_legend_gap_override,
        legend_scale_override=legend_scale_override,
        secondary_legend_scale_override=secondary_legend_scale_override,
        margin_bottom_override=margin_bottom_override,
        gap_x_override=gap_x_override,
        gap_y_override=gap_y_override,
        panel_width_override=panel_width_override,
        panel_height_override=panel_height_override,
        panel_header_height_override=panel_header_height_override,
        title_font_size_override=title_font_size_override,
        panel_label_font_size_override=panel_label_font_size_override,
        panel_caption_font_size_override=panel_caption_font_size_override,
        direction_label_font_size_override=direction_label_font_size_override,
        direction_labels_by_row=direction_labels_by_row,
        direction_label_positions_by_row=direction_label_positions_by_row,
        shared_y_labels_by_row=shared_y_labels_by_row,
        shared_y_label_font_size_override=shared_y_label_font_size_override,
        shared_x_labels_by_row=shared_x_labels_by_row,
        shared_x_label_font_size_override=shared_x_label_font_size_override,
        display_width_override=display_width_override,
        canvas_width_override=canvas_width_override,
        external_y_labels_by_row=external_y_labels_by_row,
        external_y_label_font_size_override=external_y_label_font_size_override,
        plot_box_fractions_by_row=plot_box_fractions_by_row,
        left_annotation_gutter_extra_override=left_annotation_gutter_extra_override,
        title_band_override=title_band_override,
        show_panel_annotations=show_panel_annotations,
    )

def _compose_three_panel_stats_figure(
    *,
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    legend_gap_override: Optional[float] = 20.0,
    secondary_legend_path: Optional[Path] = None,
    secondary_legend_after_row: Optional[int] = None,
    secondary_legend_y_nudge: float = 0.0,
    secondary_legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    secondary_legend_gap_override: Optional[float] = 12.0,
    legend_scale_override: Optional[float] = None,
    secondary_legend_scale_override: Optional[float] = None,
    margin_bottom_override: Optional[float] = 24.0,
    direction_labels_by_row: Optional[Sequence[str]] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    gap_x_override: Optional[float] = None,
    gap_y_override: Optional[float] = None,
    panel_width_override: Optional[float] = None,
    panel_height_override: Optional[float] = None,
    panel_header_height_override: Optional[float] = None,
    title_font_size_override: Optional[float] = None,
    direction_label_font_size_override: Optional[float] = None,
    external_y_labels_by_row: Optional[Sequence[Optional[Sequence[str]]]] = None,
    external_y_label_font_size_override: Optional[float] = None,
) -> bool:
    # Shared layout profile for 3-panel publication comparison figures.
    # Keep shared/external labels and direction banners, but otherwise align
    # with the leaner scaling-figure composition profile.
    return _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        secondary_legend_path=secondary_legend_path,
        secondary_legend_after_row=secondary_legend_after_row,
        secondary_legend_y_nudge=secondary_legend_y_nudge,
        secondary_legend_band_height_override=secondary_legend_band_height_override,
        secondary_legend_gap_override=secondary_legend_gap_override,
        legend_scale_override=legend_scale_override,
        secondary_legend_scale_override=secondary_legend_scale_override,
        margin_bottom_override=margin_bottom_override,
        gap_x_override=8.0 if gap_x_override is None else gap_x_override,
        gap_y_override=24.0 if gap_y_override is None else gap_y_override,
        panel_width_override=700.0 if panel_width_override is None else panel_width_override,
        panel_height_override=620.0 if panel_height_override is None else panel_height_override,
        panel_header_height_override=(
            88.0 if panel_header_height_override is None else panel_header_height_override
        ),
        title_font_size_override=(
            36.0 if title_font_size_override is None else title_font_size_override
        ),
        panel_label_font_size_override=30.0,
        panel_caption_font_size_override=24.0,
        direction_label_font_size_override=(
            24.0 if direction_label_font_size_override is None else direction_label_font_size_override
        ),
        direction_labels_by_row=direction_labels_by_row,
        shared_y_labels_by_row=shared_y_labels_by_row,
        shared_x_labels_by_row=shared_x_labels_by_row,
        shared_x_label_font_size_override=28.35,
        shared_y_label_font_size_override=32.4,
        external_y_labels_by_row=external_y_labels_by_row,
        external_y_label_font_size_override=(
            24.0 if external_y_label_font_size_override is None else external_y_label_font_size_override
        ),
    )


def _compose_mixed_three_panel_stats_figure(
    *,
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    legend_gap_override: Optional[float] = 20.0,
    secondary_legend_path: Optional[Path] = None,
    secondary_legend_after_row: Optional[int] = None,
    secondary_legend_y_nudge: float = 0.0,
    secondary_legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    secondary_legend_gap_override: Optional[float] = 12.0,
    legend_scale_override: Optional[float] = None,
    secondary_legend_scale_override: Optional[float] = None,
    margin_bottom_override: Optional[float] = 56.0,
    direction_labels_by_row: Optional[Sequence[str]] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    gap_x_override: Optional[float] = None,
    gap_y_override: Optional[float] = None,
    panel_width_override: Optional[float] = None,
    panel_height_override: Optional[float] = None,
    external_y_labels_by_row: Optional[Sequence[Optional[Sequence[str]]]] = None,
) -> bool:
    # Mixed 3-panel publication figures keep the Optuna label/banner features,
    # but otherwise follow the same scaling-style composition defaults.
    return _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        secondary_legend_path=secondary_legend_path,
        secondary_legend_after_row=secondary_legend_after_row,
        secondary_legend_y_nudge=secondary_legend_y_nudge,
        secondary_legend_band_height_override=secondary_legend_band_height_override,
        secondary_legend_gap_override=secondary_legend_gap_override,
        legend_scale_override=legend_scale_override,
        secondary_legend_scale_override=secondary_legend_scale_override,
        margin_bottom_override=margin_bottom_override,
        gap_x_override=8.0 if gap_x_override is None else gap_x_override,
        gap_y_override=24.0 if gap_y_override is None else gap_y_override,
        panel_width_override=700.0 if panel_width_override is None else panel_width_override,
        panel_height_override=620.0 if panel_height_override is None else panel_height_override,
        panel_header_height_override=88.0,
        title_font_size_override=36.0,
        panel_label_font_size_override=30.0,
        panel_caption_font_size_override=24.0,
        direction_label_font_size_override=24.0,
        direction_labels_by_row=direction_labels_by_row,
        shared_y_labels_by_row=shared_y_labels_by_row,
        shared_x_labels_by_row=shared_x_labels_by_row,
        shared_x_label_font_size_override=32.4,
        shared_y_label_font_size_override=32.4,
        external_y_labels_by_row=external_y_labels_by_row,
        external_y_label_font_size_override=24.0,
    )


def _compose_six_panel_stats_figure(
    *,
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    legend_gap_override: Optional[float] = 82.0,
    secondary_legend_path: Optional[Path] = None,
    secondary_legend_after_row: Optional[int] = None,
    secondary_legend_y_nudge: float = 0.0,
    secondary_legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    secondary_legend_gap_override: Optional[float] = 82.0,
    legend_scale_override: Optional[float] = None,
    secondary_legend_scale_override: Optional[float] = None,
    margin_bottom_override: Optional[float] = 56.0,
    direction_labels_by_row: Optional[Sequence[str]] = None,
    direction_label_positions_by_row: Optional[Sequence[str]] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    external_y_labels_by_row: Optional[Sequence[Optional[Sequence[str]]]] = None,
    plot_box_fractions_by_row: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    left_annotation_gutter_extra_override: Optional[float] = None,
) -> bool:
    # Canonical 2x3 publication comparison profile used for topology-style figures.
    # Keep the Optuna-specific shared-axis and direction-banner features, but
    # otherwise align this profile with the leaner scaling-figure composition.
    return _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        secondary_legend_path=secondary_legend_path,
        secondary_legend_after_row=secondary_legend_after_row,
        secondary_legend_y_nudge=secondary_legend_y_nudge,
        secondary_legend_band_height_override=secondary_legend_band_height_override,
        secondary_legend_gap_override=secondary_legend_gap_override,
        legend_scale_override=legend_scale_override,
        secondary_legend_scale_override=secondary_legend_scale_override,
        margin_bottom_override=margin_bottom_override,
        gap_x_override=8.0,
        gap_y_override=24.0,
        panel_width_override=700.0,
        panel_height_override=620.0,
        panel_header_height_override=116.0,
        title_font_size_override=36.0,
        panel_label_font_size_override=30.0,
        panel_caption_font_size_override=24.0,
        direction_label_font_size_override=24.0,
        direction_labels_by_row=direction_labels_by_row,
        direction_label_positions_by_row=direction_label_positions_by_row,
        shared_y_labels_by_row=shared_y_labels_by_row,
        shared_x_labels_by_row=shared_x_labels_by_row,
        shared_x_label_font_size_override=32.4,
        shared_y_label_font_size_override=32.4,
        external_y_labels_by_row=external_y_labels_by_row,
        external_y_label_font_size_override=24.0,
        plot_box_fractions_by_row=plot_box_fractions_by_row,
        left_annotation_gutter_extra_override=left_annotation_gutter_extra_override,
    )


def _compose_stability_2x2_figure(
    *,
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    dpi: int,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = PUBLICATION_LEGEND_ROW_HEIGHT,
    legend_gap_override: Optional[float] = 0.0,
    margin_bottom_override: Optional[float] = 0.0,
    gap_x_override: Optional[float] = None,
    panel_width_override: Optional[float] = None,
    panel_height_override: Optional[float] = None,
    shared_y_labels_by_row: Optional[Sequence[str]] = None,
    shared_x_labels_by_row: Optional[Sequence[str]] = None,
    display_width_override: Optional[float] = None,
) -> bool:
    # Legacy 2x2 stability composition helper. Keep the shared-axis option
    # available, but newer Figure 7 layouts may use the standard tripanel path.
    return _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        margin_bottom_override=margin_bottom_override,
        gap_x_override=8.0 if gap_x_override is None else gap_x_override,
        gap_y_override=24.0,
        panel_width_override=760.0 if panel_width_override is None else panel_width_override,
        panel_height_override=500.0 if panel_height_override is None else panel_height_override,
        panel_header_height_override=88.0,
        title_font_size_override=36.0,
        panel_label_font_size_override=30.0,
        panel_caption_font_size_override=24.0,
        shared_y_labels_by_row=shared_y_labels_by_row,
        shared_x_labels_by_row=shared_x_labels_by_row,
        shared_x_label_font_size_override=28.35,
        shared_y_label_font_size_override=32.4,
        display_width_override=2600.0 if display_width_override is None else display_width_override,
    )

def _compose_optuna_figure_1(
    block_output_dir: Path,
    metric: str,
    metric_label: str,
    dpi: int,
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
) -> None:
    metric_slug = _slug(metric)
    fig_dir = block_output_dir / "figures"
    panel_rows: List[List[Tuple[str, str, Path]]] = [
        [
            ("A", "Full Batch: Colors vs Batch", fig_dir / f"fig_hex_colors_vs_batch_full_batch_{metric_slug}.svg"),
            ("B", "Sensitivity (Full Batch)", fig_dir / f"fig_hex_colors_vs_batch_sensitivity_full_batch_{metric_slug}.svg"),
        ],
    ]

    output_path = fig_dir / f"fig_publication_figure1_optuna_{metric_slug}.svg"
    legend_path = fig_dir / "legend_key.svg"
    ok = _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=f"Figure 1: Colors vs Batch (Hexagonal, Full Batch, {metric_label})",
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path if legend_path.exists() else None,
    )
    if ok:
        generated_files.append(str(output_path))
        diagnostics[f"{diagnostics_prefix}figure1_optuna_{metric_slug}"] = {
            "output": str(output_path),
            "panels": [{label: str(path)} for row in panel_rows for label, _, path in row],
        }

def _compose_optuna_figure_2(
    block_output_dir: Path,
    metric: str,
    metric_label: str,
    dpi: int,
    generated_files: List[str],
    diagnostics: Dict[str, object],
    diagnostics_prefix: str = "",
) -> None:
    metric_slug = _slug(metric)
    fig_dir = block_output_dir / "topology_main" / "figures"
    panel_rows: List[List[Tuple[str, str, Path]]] = [
        [
            ("A", "Hexagonal vs MST", fig_dir / f"fig_hex_vs_mst_main_{metric_slug}.svg"),
            ("B", "Hexagonal vs RNG", fig_dir / f"fig_hex_vs_rng_main_{metric_slug}.svg"),
            ("C", "MST vs RNG", fig_dir / f"fig_mst_vs_rng_main_{metric_slug}.svg"),
        ],
        [
            ("D", "Sensitivity (Hexagonal vs MST)", fig_dir / f"fig_hex_vs_mst_sensitivity_main_{metric_slug}.svg"),
            ("E", "Sensitivity (Hexagonal vs RNG)", fig_dir / f"fig_hex_vs_rng_sensitivity_main_{metric_slug}.svg"),
            ("F", "Sensitivity (MST vs RNG)", fig_dir / f"fig_mst_vs_rng_sensitivity_main_{metric_slug}.svg"),
        ],
    ]

    output_path = block_output_dir / "figures" / f"fig_publication_figure2_optuna_{metric_slug}.svg"
    legend_path = block_output_dir / "figures" / "legend_key.svg"
    ok = _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=f"Figure 2: Pairwise Topology Comparison (QE-centric, {metric_label})",
        output_path=output_path,
        dpi=dpi,
        legend_path=legend_path if legend_path.exists() else None,
    )
    if ok:
        generated_files.append(str(output_path))
        diagnostics[f"{diagnostics_prefix}figure2_optuna_{metric_slug}"] = {
            "output": str(output_path),
            "panels": [{label: str(path)} for row in panel_rows for label, _, path in row],
        }

__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
