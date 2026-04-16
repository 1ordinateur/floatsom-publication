#!/usr/bin/env python3
"""
Generate a representative topology figure for Hexagonal, MST, and RNG.

This script reuses benchmark defaults/training paths from
`floatsom.benchmarks.run_sklearn_benchmarks` and mirrors its edge-collection
logic for overlay rendering. It accepts any dataset supported by
`run_sklearn_benchmarks` via `--data_type`. For non-2D datasets, a shared PCA
projection is used for visualization while training stays in original space.
SVG output is written directly so no plotting package dependency is required.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
if GRANDPARENT_DIR not in sys.path:
    sys.path.append(GRANDPARENT_DIR)

from floatsom.benchmarks import run_sklearn_benchmarks as benchmark  # noqa: E402


TOPOLOGIES: Sequence[str] = ("hexagonal", "mst", "rng")
PANEL_LABELS: Sequence[str] = ("A", "B", "C")
TOPOLOGY_DISPLAY_NAMES: Dict[str, str] = {
    "hexagonal": "Hexagonal",
    "mst": "MST",
    "rng": "RNG",
}


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark on Hexagonal/MST/RNG and export a 3-panel "
            "representative overlay figure with a shared legend."
        )
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="sklearn_circles",
        help=(
            "Dataset identifier accepted by run_sklearn_benchmarks "
            "(e.g., sklearn_circles, sklearn_moons, sklearn_iris, clusters_nd, random)."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=("easy", "medium", "hard"),
        help="Difficulty level for supported synthetic/sklearn datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation and training.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Grid size (used for hexagonal; mst/rng default to grid_size**2 nodes).",
    )
    parser.add_argument(
        "--mst_nodes",
        type=int,
        default=None,
        help="Optional explicit graph node count for MST/RNG.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional override for total training iterations.",
    )
    parser.add_argument(
        "--output_svg",
        type=Path,
        default=None,
        help=(
            "Output SVG path for the combined representative figure. "
            "Default: paper/assets/figures/fig_5.svg for sklearn_circles; "
            "otherwise paper/assets/figures/fig_topology_<data_type>_representative.svg"
        ),
    )
    parser.add_argument(
        "--summary_tsv",
        type=Path,
        default=None,
        help=(
            "Output TSV path for benchmark summary statistics. "
            "Default: paper/assets/tables/table_topology_<data_type>_representative.tsv"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose benchmark logging.",
    )
    return parser.parse_args()


def _dataset_tag(data_type: str) -> str:
    value = str(data_type).strip().lower()
    if value.startswith("sklearn_"):
        value = value.replace("sklearn_", "", 1)
    safe = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
    return safe or "dataset"


def _dataset_display_name(data_type: str) -> str:
    value = str(data_type).strip()
    if value.startswith("sklearn_"):
        return f"sklearn {value.replace('sklearn_', '', 1)}"
    return value.replace("_", " ")


def _resolve_output_paths(script_args: argparse.Namespace) -> Tuple[Path, Path]:
    tag = _dataset_tag(script_args.data_type)
    figures_dir = Path(__file__).resolve().parents[1] / "paper" / "assets" / "figures"
    tables_dir = Path(__file__).resolve().parents[1] / "paper" / "assets" / "tables"
    output_svg = script_args.output_svg
    summary_tsv = script_args.summary_tsv
    if output_svg is None:
        if tag == "circles":
            output_svg = figures_dir / "fig_5.svg"
        else:
            output_svg = figures_dir / f"fig_topology_{tag}_representative.svg"
    if summary_tsv is None:
        summary_tsv = tables_dir / f"table_topology_{tag}_representative.tsv"
    return Path(output_svg), Path(summary_tsv)


def _get_benchmark_default_args() -> argparse.Namespace:
    """Build defaults from run_sklearn_benchmarks.parse_args."""
    original_argv = list(sys.argv)
    try:
        sys.argv = ["run_sklearn_benchmarks.py"]
        return benchmark.parse_args()
    finally:
        sys.argv = original_argv


def _configure_topology_args(
    base_args: argparse.Namespace,
    script_args: argparse.Namespace,
    topology_type: str,
) -> argparse.Namespace:
    """Create per-topology benchmark args from canonical defaults."""
    args = copy.deepcopy(base_args)
    args.data_type = script_args.data_type
    args.difficulty = script_args.difficulty
    args.seed = script_args.seed
    args.grid_size = script_args.grid_size
    args.topology_type = topology_type
    args.mst_nodes = script_args.mst_nodes
    args.visualize = False
    args.verbose = script_args.verbose
    args.use_gpu = True
    if topology_type != "hexagonal":
        args.initial_radius = 1
        args.radius_decay_type = "asymptotic"
        args.radius_decay_factor = 1
    if script_args.iterations is not None:
        args.iterations = 50
    return args


def _project_to_2d(
    data_np: np.ndarray,
    weights_by_topology: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """
    Project data and topology weights into a shared 2D display space.

    Returns:
        data_2d: projected data coordinates
        weights_2d: projected topology weights
        axis_mode: "dimension" (native 2D/1D) or "pca" (projected)
    """
    if data_np.ndim != 2:
        raise ValueError(f"Expected 2D matrix data, got shape {data_np.shape}.")

    if data_np.shape[1] == 2:
        return data_np[:, :2], {k: v[:, :2] for k, v in weights_by_topology.items()}, "dimension"

    if data_np.shape[1] == 1:
        data_2d = np.concatenate([data_np, np.zeros((data_np.shape[0], 1), dtype=data_np.dtype)], axis=1)
        weights_2d: Dict[str, np.ndarray] = {}
        for topology, weights in weights_by_topology.items():
            weights_2d[topology] = np.concatenate(
                [weights, np.zeros((weights.shape[0], 1), dtype=weights.dtype)],
                axis=1,
            )
        return data_2d, weights_2d, "dimension"

    mean_vec = np.mean(data_np, axis=0, keepdims=True)
    centered_data = data_np - mean_vec
    _, _, vh = np.linalg.svd(centered_data, full_matrices=False)
    basis = vh[:2].T
    data_2d = centered_data @ basis
    weights_2d = {
        topology: (weights - mean_vec) @ basis
        for topology, weights in weights_by_topology.items()
    }
    return data_2d, weights_2d, "pca"


def _collect_connection_edges(
    som: Any,
    args: argparse.Namespace,
    weights: Any,
    weights_np: np.ndarray,
) -> Tuple[List[Tuple[int, int]], str]:
    """Mirror run_sklearn_benchmarks.visualize_results connection logic."""
    edge_color = "#111111"
    if args.topology_type in ["grid", "hexagonal"]:
        grid_size = som.topology.grid_size
        total_nodes = weights_np.shape[0]
        edges: List[Tuple[int, int]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx >= total_nodes:
                    continue
                if j < grid_size - 1:
                    idx_right = i * grid_size + (j + 1)
                    if idx_right < total_nodes:
                        edges.append((idx, idx_right))
                if i < grid_size - 1:
                    idx_bottom = (i + 1) * grid_size + j
                    if idx_bottom < total_nodes:
                        edges.append((idx, idx_bottom))
        return edges, edge_color

    if args.topology_type in {"mst", "rng"}:
        if hasattr(som.topology, "is_reformed") and som.topology.is_reformed:
            adjacency = getattr(som, "adjacency_list", None)
            if adjacency is None:
                raise ValueError("Topology is reformed but adjacency_list not found.")
            edge_set = set()
            for node, neighbors in adjacency.items():
                for neighbor in neighbors:
                    if neighbor == node:
                        continue
                    edge_set.add(tuple(sorted((int(node), int(neighbor)))))
            return sorted(edge_set), edge_color

        edges = getattr(som.topology, "mst_edges", None)
        if edges is None or len(edges) == 0:
            if hasattr(som, "topology") and hasattr(som.topology, "update_topology"):
                som.topology.update_topology(weights)
            edges = getattr(som.topology, "mst_edges", None)
        if edges is None:
            return [], edge_color
        return [(int(u), int(v)) for u, v in edges], edge_color

    return [], edge_color


def _xml_escape(value: str) -> str:
    """Escape XML text/attribute content."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_panel_mapper(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
):
    """Build equal-aspect coordinate mapper from data space into a panel plot box."""
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    data_aspect = x_span / y_span
    plot_aspect = plot_w / plot_h

    if data_aspect >= plot_aspect:
        draw_w = plot_w
        draw_h = plot_w / data_aspect
        offset_x = 0.0
        offset_y = 0.5 * (plot_h - draw_h)
    else:
        draw_h = plot_h
        draw_w = plot_h * data_aspect
        offset_x = 0.5 * (plot_w - draw_w)
        offset_y = 0.0

    def map_point(x_val: float, y_val: float) -> Tuple[float, float]:
        x_norm = (x_val - x_min) / x_span
        y_norm = (y_val - y_min) / y_span
        sx = plot_x + offset_x + (x_norm * draw_w)
        sy = plot_y + offset_y + ((1.0 - y_norm) * draw_h)
        return sx, sy

    return map_point


def _write_combined_svg(
    output_svg: Path,
    data_np: np.ndarray,
    panel_records: Sequence[Dict[str, Any]],
    *,
    dataset_label: str,
    axis_label_mode: str,
) -> None:
    """Write combined representative figure as SVG."""
    canvas_w = 1560
    canvas_h = 560
    margin_x = 36
    panel_gap = 18
    panel_y = 72
    panel_h = 430
    panel_w = (canvas_w - (2 * margin_x) - (2 * panel_gap)) / 3.0

    x_values = [data_np[:, 0]]
    y_values = [data_np[:, 1]]
    for record in panel_records:
        weights_np = record["weights_plot_np"]
        x_values.append(weights_np[:, 0])
        y_values.append(weights_np[:, 1])

    x_min = float(min(np.min(arr) for arr in x_values))
    x_max = float(max(np.max(arr) for arr in x_values))
    y_min = float(min(np.min(arr) for arr in y_values))
    y_max = float(max(np.max(arr) for arr in y_values))
    x_pad = max(1e-6, 0.04 * (x_max - x_min))
    y_pad = max(1e-6, 0.04 * (y_max - y_min))
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def fmt(value: float) -> str:
        return f"{value:.2f}"

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" '
            f'height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" role="img" '
            f'aria-label="{_xml_escape(f"Representative topology overlays on {dataset_label}")}">'
        ),
        f'  <rect x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="#FFFFFF"/>',
        (
            f'  <text x="{margin_x}" y="36" text-anchor="start" font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="28" font-weight="700" fill="#1f1f1f">'
            f"{_xml_escape(f'Figure 1: Representative Node-Connection Overlays on {dataset_label}')}"
            "</text>"
        ),
    ]

    for idx, record in enumerate(panel_records):
        panel_x = margin_x + idx * (panel_w + panel_gap)
        topology = str(record["topology"])
        topology_name = TOPOLOGY_DISPLAY_NAMES[topology]
        panel_label = PANEL_LABELS[idx]
        weights_np = record["weights_plot_np"]
        edges = record["edges"]
        edge_color = record["edge_color"]
        edge_alpha = 0.72

        plot_x = panel_x + 12.0
        plot_y = panel_y + 52.0
        plot_w = panel_w - 24.0
        plot_h = panel_h - 82.0

        map_point = _build_panel_mapper(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=plot_h,
        )

        lines.append(
            f'  <text x="{fmt(panel_x + 16)}" y="{fmt(panel_y + 28)}" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="30" font-weight="700" fill="#111111">'
            f"{_xml_escape(panel_label)}"
            "</text>"
        )
        lines.append(
            f'  <text x="{fmt(panel_x + 56)}" y="{fmt(panel_y + 28)}" text-anchor="start" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="24" font-weight="600" fill="#222222">'
            f"{_xml_escape(topology_name)}"
            "</text>"
        )

        # Data points.
        for x_val, y_val in data_np[:, :2]:
            sx, sy = map_point(float(x_val), float(y_val))
            lines.append(
                f'  <circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="1.35" fill="#808080" fill-opacity="0.45" />'
            )

        # Topology connections.
        for u, v in edges:
            if u < 0 or v < 0 or u >= weights_np.shape[0] or v >= weights_np.shape[0]:
                continue
            x1, y1 = map_point(float(weights_np[u, 0]), float(weights_np[u, 1]))
            x2, y2 = map_point(float(weights_np[v, 0]), float(weights_np[v, 1]))
            lines.append(
                f'  <line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
                f'stroke="{edge_color}" stroke-opacity="{edge_alpha:.2f}" stroke-width="1.35"/>'
            )

        # SOM nodes.
        for x_val, y_val in weights_np[:, :2]:
            sx, sy = map_point(float(x_val), float(y_val))
            lines.append(
                f'  <circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="3.00" fill="#D62728" '
                'stroke="#8B0000" stroke-width="0.7"/>'
            )

    # Shared legend.
    legend_y = panel_y + panel_h + 24
    legend_entries = [
        ("point", "Data points"),
        ("node", "SOM nodes"),
        ("line", "Connections"),
    ]
    legend_gap = 340.0
    legend_start_x = (canvas_w - (legend_gap * (len(legend_entries) - 1))) / 2.0
    for idx, (kind, label) in enumerate(legend_entries):
        x0 = legend_start_x + idx * legend_gap
        if kind == "point":
            lines.append(
                f'  <circle cx="{fmt(x0)}" cy="{fmt(legend_y)}" r="4.0" fill="#808080" fill-opacity="0.65" />'
            )
        elif kind == "node":
            lines.append(
                f'  <circle cx="{fmt(x0)}" cy="{fmt(legend_y)}" r="4.4" fill="#D62728" stroke="#8B0000" stroke-width="0.8"/>'
            )
        else:
            lines.append(
                f'  <line x1="{fmt(x0 - 10)}" y1="{fmt(legend_y)}" x2="{fmt(x0 + 10)}" y2="{fmt(legend_y)}" '
                'stroke="#111111" stroke-width="2.2"/>'
            )
        lines.append(
            f'  <text x="{fmt(x0 + 16)}" y="{fmt(legend_y + 5)}" font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="22" fill="#242424">'
            f"{_xml_escape(label)}"
            "</text>"
        )

    lines.append("</svg>")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_tsv(summary_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write benchmark summary table."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "topology",
        "train_time_s",
        "iterations_completed",
        "total_samples_processed",
        "quantization_error",
        "n_nodes",
        "n_edges",
    ]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        str(row["topology"]),
                        f"{row['train_time_s']:.6f}",
                        str(int(row["iterations_completed"])),
                        str(int(row["total_samples_processed"])),
                        f"{row['quantization_error']:.6f}",
                        str(int(row["n_nodes"])),
                        str(int(row["n_edges"])),
                    ]
                )
                + "\n"
            )


def main() -> None:
    """Run representative topology benchmark and emit SVG + summary TSV."""
    script_args = parse_args()
    output_svg, summary_tsv = _resolve_output_paths(script_args)
    base_defaults = _get_benchmark_default_args()

    data_args = _configure_topology_args(
        base_args=base_defaults,
        script_args=script_args,
        topology_type="hexagonal",
    )
    data, metadata = benchmark.generate_data(data_args)
    data_np = data.get() if hasattr(data, "get") else data

    panel_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for topology in TOPOLOGIES:
        topology_args = _configure_topology_args(
            base_args=base_defaults,
            script_args=script_args,
            topology_type=topology,
        )
        som, training_stats, train_time = benchmark.train_floatsom(data, topology_args, metadata)
        weights = som.get_weights()
        weights_np = weights.get() if hasattr(weights, "get") else weights

        quant_error = benchmark.QuantizationError(
            use_optimized=bool(topology_args.use_gpu)
        ).compute(
            benchmark.FloatSOMWrapper(som),
            data,
        )

        edges, edge_color = _collect_connection_edges(
            som=som,
            args=topology_args,
            weights=weights,
            weights_np=weights_np,
        )

        panel_records.append(
            {
                "topology": topology,
                "weights_np": weights_np,
                "edges": edges,
                "edge_color": edge_color,
            }
        )
        summary_rows.append(
            {
                "topology": topology,
                "train_time_s": float(train_time),
                "iterations_completed": training_stats.get("iterations_completed", 0),
                "total_samples_processed": training_stats.get("total_samples_processed", 0),
                "quantization_error": float(quant_error),
                "n_nodes": int(weights_np.shape[0]),
                "n_edges": int(len(edges)),
            }
        )

    weights_by_topology = {
        str(record["topology"]): np.asarray(record["weights_np"])
        for record in panel_records
    }
    data_plot_np, projected_weights, axis_label_mode = _project_to_2d(
        data_np=np.asarray(data_np),
        weights_by_topology=weights_by_topology,
    )
    for record in panel_records:
        topology = str(record["topology"])
        record["weights_plot_np"] = projected_weights[topology]

    _write_combined_svg(
        output_svg=output_svg,
        data_np=data_plot_np,
        panel_records=panel_records,
        dataset_label=_dataset_display_name(script_args.data_type),
        axis_label_mode=axis_label_mode,
    )
    _write_summary_tsv(summary_tsv, summary_rows)

    print(f"Representative figure saved to: {output_svg}")
    print(f"Benchmark summary saved to: {summary_tsv}")


if __name__ == "__main__":
    main()
