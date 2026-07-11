#!/usr/bin/env python3
"""Compose the three radius-response SVGs into an editable paper tripanel."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

CANVAS_WIDTH = 3333.0
CANVAS_HEIGHT = 1200.0
SIDE_MARGIN = 60.0
PANEL_WIDTH = 1051.0
PANEL_HEIGHT = 900.0
PANEL_GAP = 30.0
HEADER_HEIGHT = 175.0


def _parse_view_box(root: ET.Element, path: Path) -> tuple[float, float, float, float]:
    raw = root.get("viewBox")
    if not raw:
        raise ValueError(f"Input SVG has no viewBox: {path}")
    values = [float(value) for value in raw.replace(",", " ").split()]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"Invalid SVG viewBox in {path}: {raw!r}")
    return values[0], values[1], values[2], values[3]


def _prefix_ids(element: ET.Element, prefix: str) -> None:
    replacements: dict[str, str] = {}
    for node in element.iter():
        node_id = node.get("id")
        if node_id:
            replacements[node_id] = f"{prefix}_{node_id}"
            node.set("id", replacements[node_id])

    if not replacements:
        return

    url_pattern = re.compile(r"url\(#([^)]+)\)")
    for node in element.iter():
        for attribute, value in list(node.attrib.items()):
            value = url_pattern.sub(
                lambda match: f"url(#{replacements.get(match.group(1), match.group(1))})",
                value,
            )
            if value.startswith("#") and value[1:] in replacements:
                value = f"#{replacements[value[1:]]}"
            node.set(attribute, value)


def _append_panel(
    output_root: ET.Element,
    *,
    source_path: Path,
    panel_id: str,
    label: str,
    heading: str,
    x: float,
) -> None:
    source_root = ET.parse(source_path).getroot()
    view_box = _parse_view_box(source_root, source_path)
    source_copy = deepcopy(source_root)
    _prefix_ids(source_copy, f"panel_{panel_id}")

    label_text = ET.SubElement(
        output_root,
        f"{{{SVG_NS}}}text",
        {
            "id": f"panel_{panel_id}_label",
            "x": f"{x + 20:g}",
            "y": "145",
            "fill": "#1f1f1f",
            "font-family": "Arial, sans-serif",
            "font-size": "60",
            "font-weight": "700",
        },
    )
    label_text.text = label

    heading_text = ET.SubElement(
        output_root,
        f"{{{SVG_NS}}}text",
        {
            "id": f"panel_{panel_id}_heading",
            "x": f"{x + PANEL_WIDTH / 2:g}",
            "y": "145",
            "fill": "#2f2f2f",
            "font-family": "Arial, sans-serif",
            "font-size": "54.495",
            "font-weight": "700",
            "text-anchor": "middle",
        },
    )
    heading_text.text = heading

    nested = ET.SubElement(
        output_root,
        f"{{{SVG_NS}}}svg",
        {
            "id": f"panel_{panel_id}",
            "x": f"{x:g}",
            "y": f"{HEADER_HEIGHT:g}",
            "width": f"{PANEL_WIDTH:g}",
            "height": f"{PANEL_HEIGHT:g}",
            "viewBox": " ".join(f"{value:g}" for value in view_box),
            "preserveAspectRatio": "xMidYMid meet",
            "overflow": "visible",
        },
    )

    for child in source_copy:
        if child.tag == f"{{{SVG_NS}}}metadata":
            continue
        nested.append(child)


def compose(qe_path: Path, mtr_path: Path, utilisation_path: Path, output_path: Path) -> None:
    for path in (qe_path, mtr_path, utilisation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Radius-response panel does not exist: {path}")

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "id": "radius_response_tripanel",
            "version": "1.1",
            "width": f"{CANVAS_WIDTH:g}",
            "height": f"{CANVAS_HEIGHT:g}",
            "viewBox": f"0 0 {CANVAS_WIDTH:g} {CANVAS_HEIGHT:g}",
        },
    )
    title = ET.SubElement(root, f"{{{SVG_NS}}}title")
    title.text = "Initial-radius response across FloatSOM topologies"
    description = ET.SubElement(root, f"{{{SVG_NS}}}desc")
    description.text = (
        "Matched-seed, Hex-optimal-normalized balanced quantisation error and observed "
        "mean tied rank and node utilisation across the initial-radius sweep."
    )
    title_text = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "id": "tripanel_title",
            "x": f"{CANVAS_WIDTH / 2:g}",
            "y": "75",
            "fill": "#1f1f1f",
            "font-family": "Arial, sans-serif",
            "font-size": "67.5",
            "font-weight": "700",
            "text-anchor": "middle",
        },
    )
    title_text.text = "Initial-radius response across FloatSOM topologies"

    panels = (
        (qe_path, "a", "A", "Balanced QE"),
        (mtr_path, "b", "B", "Balanced MTR"),
        (utilisation_path, "c", "C", "Balanced node utilisation"),
    )
    for index, (path, panel_id, label, heading) in enumerate(panels):
        _append_panel(
            root,
            source_path=path,
            panel_id=panel_id,
            label=label,
            heading=heading,
            x=SIDE_MARGIN + index * (PANEL_WIDTH + PANEL_GAP),
        )

    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qe", type=Path, required=True)
    parser.add_argument("--mtr", type=Path, required=True)
    parser.add_argument("--utilisation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.qe, args.mtr, args.utilisation, args.output)


if __name__ == "__main__":
    main()
