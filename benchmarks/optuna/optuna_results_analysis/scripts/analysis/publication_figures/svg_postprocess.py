from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_SVG_VIEWBOX_PATTERN = re.compile(
    r'(\bviewBox=["\'])\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*(["\'])'
)
_SVG_WIDTH_PATTERN = re.compile(r'\bwidth=["\']\s*([-+0-9.eE]+)')
_SVG_HEIGHT_PATTERN = re.compile(r'\bheight=["\']\s*([-+0-9.eE]+)')
_SVG_OPEN_TAG_PATTERN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)

_PUBLICATION_SVG_NORMALIZATION_TARGETS: Tuple[str, ...] = (
    "fig_4.svg",
    "fig_5.svg",
    "fig_6.svg",
    "fig_7.svg",
    "supp_fig_s5.svg",
    "supp_fig_s8.svg",
    "supp_fig_s9.svg",
    "supp_fig_s10.svg",
    "supp_fig_s11.svg",
    "supp_fig_s13.svg",
    "supp_fig_s14.svg",
)
_NORMALIZATION_GROUP_ID = "publication_svg_canvas_normalized"
_SVG_TOLERANCE = 1e-6


def _format_svg_number(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _parse_svg_canvas(text: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    viewbox_match = _SVG_VIEWBOX_PATTERN.search(text)
    width_match = _SVG_WIDTH_PATTERN.search(text)
    height_match = _SVG_HEIGHT_PATTERN.search(text)
    if viewbox_match is None or width_match is None or height_match is None:
        return None

    min_x = float(viewbox_match.group(2))
    min_y = float(viewbox_match.group(3))
    viewbox_width = float(viewbox_match.group(4))
    viewbox_height = float(viewbox_match.group(5))
    width = float(width_match.group(1))
    height = float(height_match.group(1))

    if viewbox_width <= 0.0 or viewbox_height <= 0.0 or width <= 0.0 or height <= 0.0:
        return None
    return (min_x, min_y, viewbox_width, viewbox_height, width, height)


def _build_normalized_wrapper(
    *,
    min_x: float,
    min_y: float,
    viewbox_width: float,
    viewbox_height: float,
    width: float,
    height: float,
    inner_svg: str,
) -> str:
    scale_x = float(width) / float(viewbox_width)
    scale_y = float(height) / float(viewbox_height)
    translate_x = -float(min_x) * scale_x
    translate_y = -float(min_y) * scale_y
    return (
        f'<g id="{_NORMALIZATION_GROUP_ID}" '
        f'transform="matrix({_format_svg_number(scale_x)} 0 0 '
        f'{_format_svg_number(scale_y)} {_format_svg_number(translate_x)} '
        f'{_format_svg_number(translate_y)})">\n'
        f"{inner_svg}\n"
        f"</g>"
    )


def _normalize_svg_document(path: Path) -> Optional[Dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if _NORMALIZATION_GROUP_ID in text:
        return None

    canvas = _parse_svg_canvas(text)
    if canvas is None:
        return None
    min_x, min_y, viewbox_width, viewbox_height, width, height = canvas

    if (
        abs(min_x) <= _SVG_TOLERANCE
        and abs(min_y) <= _SVG_TOLERANCE
        and abs(viewbox_width - width) <= _SVG_TOLERANCE
        and abs(viewbox_height - height) <= _SVG_TOLERANCE
    ):
        return None

    open_tag_match = _SVG_OPEN_TAG_PATTERN.search(text)
    close_tag_idx = text.lower().rfind("</svg>")
    if open_tag_match is None or close_tag_idx < 0 or close_tag_idx <= open_tag_match.end():
        return None

    open_tag = open_tag_match.group(0)
    inner_svg = text[open_tag_match.end():close_tag_idx].strip("\n")
    viewbox_match = _SVG_VIEWBOX_PATTERN.search(open_tag)
    if viewbox_match is None:
        return None

    normalized_open_tag = _SVG_VIEWBOX_PATTERN.sub(
        (
            f"\\g<1>0 0 {_format_svg_number(width)} {_format_svg_number(height)}"
            f"\\g<6>"
        ),
        open_tag,
        count=1,
    )
    normalized_body = _build_normalized_wrapper(
        min_x=min_x,
        min_y=min_y,
        viewbox_width=viewbox_width,
        viewbox_height=viewbox_height,
        width=width,
        height=height,
        inner_svg=inner_svg,
    )
    normalized_text = (
        f"{text[:open_tag_match.start()]}"
        f"{normalized_open_tag}\n"
        f"{normalized_body}\n"
        f"{text[close_tag_idx:]}"
    )
    path.write_text(normalized_text, encoding="utf-8")

    return {
        "path": str(path.resolve()),
        "scale_x": float(width) / float(viewbox_width),
        "scale_y": float(height) / float(viewbox_height),
        "source_viewBox": [
            float(min_x),
            float(min_y),
            float(viewbox_width),
            float(viewbox_height),
        ],
        "normalized_viewBox": [0.0, 0.0, float(width), float(height)],
    }


def _normalize_publication_svg_assets(
    assets_figures_dir: Path,
    *,
    target_filenames: Iterable[str] = _PUBLICATION_SVG_NORMALIZATION_TARGETS,
) -> Dict[str, object]:
    assets_dir = Path(assets_figures_dir)
    results: Dict[str, object] = {
        "assets_figures_dir": str(assets_dir.resolve()),
        "targets": list(target_filenames),
        "normalized": [],
        "missing": [],
    }
    normalized_entries: List[Dict[str, object]] = []
    missing: List[str] = []

    for filename in target_filenames:
        figure_path = assets_dir / str(filename)
        if not figure_path.exists():
            missing.append(str(filename))
            continue
        normalized = _normalize_svg_document(figure_path)
        if normalized is not None:
            normalized_entries.append(normalized)

    results["normalized"] = normalized_entries
    results["missing"] = missing
    results["updated"] = bool(normalized_entries)
    return results


__all__ = [
    name
    for name in globals()
    if name.startswith("_") and not name.startswith("__")
]
