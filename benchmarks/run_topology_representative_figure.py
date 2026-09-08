#!/usr/bin/env python3
"""Generate the six-panel representative topology figure used as Figure 5.

The default figure compares hexagonal, MST, and RNG SOMs on sklearn circles
(native 2D) and Covertype (trained on standardized 54D data and displayed with
one PCA basis shared by the observations and all three sets of prototypes).
Observation clouds are rasterized once per dataset row; topology connections,
nodes, labels, and the legend remain SVG vectors.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
if GRANDPARENT_DIR not in sys.path:
    sys.path.append(GRANDPARENT_DIR)


TOPOLOGIES: Sequence[str] = ("hexagonal", "mst", "rng")
DEFAULT_DATA_TYPES: Sequence[str] = ("sklearn_circles", "sklearn_covertype")
TOPOLOGY_DISPLAY_NAMES: Dict[str, str] = {
    "hexagonal": "Hexagonal",
    "mst": "MST",
    "rng": "RNG",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments while retaining the old single-dataset API."""
    parser = argparse.ArgumentParser(
        description="Train Hexagonal/MST/RNG SOMs and export representative topology overlays."
    )
    parser.add_argument(
        "--data_type",
        default=None,
        help="Backward-compatible single dataset identifier; overrides --data-types.",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=None,
        help="Dataset rows to render (default: sklearn_circles sklearn_covertype).",
    )
    parser.add_argument("--difficulty", default="hard", choices=("easy", "medium", "hard"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--mst_nodes", type=int, default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Training iterations (Figure 5 uses 50; explicit overrides are honored).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.5,
        help="Initial learning rate (default: 0.5, matching XPySOM).",
    )
    parser.add_argument(
        "--initial-radius",
        type=float,
        default=5.0,
        help="Initial neighbourhood radius (default: 5, matching the untuned XPySOM-like profile).",
    )
    parser.add_argument(
        "--radius-decay-type",
        choices=("exponential", "linear", "asymptotic"),
        default="exponential",
        help="Neighbourhood-radius decay (default: exponential, matching the untuned profile).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="Initial momentum coefficient; ignored unless --use-momentum is supplied.",
    )
    parser.add_argument(
        "--use-momentum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable momentum (default: disabled, matching the untuned XPySOM-like profile).",
    )
    parser.add_argument(
        "--display-limit",
        type=int,
        default=30_000,
        help="Maximum observations displayed per dataset row; training still uses all observations.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--raster-scale", type=float, default=2.0)
    parser.add_argument("--output_svg", type=Path, default=None)
    parser.add_argument("--summary_tsv", type=Path, default=None)
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Directory for the two source JPEG backgrounds (default: beside output SVG).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.display_limit < 1:
        parser.error("--display-limit must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.raster_scale <= 0:
        parser.error("--raster-scale must be positive")
    args.data_types = [args.data_type] if args.data_type else (args.data_types or list(DEFAULT_DATA_TYPES))
    return args


def _load_benchmark_module() -> Any:
    """Import the GPU benchmark only when training is requested."""
    from floatsom_benchmarks import run_sklearn_benchmarks as benchmark

    return benchmark


def _dataset_tag(data_type: str) -> str:
    value = str(data_type).strip().lower()
    if value.startswith("sklearn_"):
        value = value[len("sklearn_") :]
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "dataset"


def _dataset_display_name(data_type: str) -> str:
    tag = _dataset_tag(data_type)
    return {
        "circles": "Circles",
        "covertype": "Covertype",
        "kddcup99": "KDD Cup 99",
    }.get(tag, tag.replace("_", " ").title())


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    figures_dir = root / "paper" / "assets" / "figures"
    tables_dir = root / "paper" / "assets" / "tables"
    if args.output_svg is None:
        output_svg = figures_dir / "fig_5.svg" if list(args.data_types) == list(DEFAULT_DATA_TYPES) else figures_dir / f"fig_topology_{_dataset_tag(args.data_types[0])}_representative.svg"
    else:
        output_svg = Path(args.output_svg)
    if args.summary_tsv is None:
        summary_tsv = tables_dir / "table_topology_circles_representative.tsv" if list(args.data_types) == list(DEFAULT_DATA_TYPES) else tables_dir / f"table_topology_{_dataset_tag(args.data_types[0])}_representative.tsv"
    else:
        summary_tsv = Path(args.summary_tsv)
    background_dir = Path(args.background_dir) if args.background_dir else output_svg.parent
    return output_svg, summary_tsv, background_dir


def _get_benchmark_default_args(benchmark: Any) -> argparse.Namespace:
    original_argv = list(sys.argv)
    try:
        sys.argv = ["run_sklearn_benchmarks.py"]
        return benchmark.parse_args()
    finally:
        sys.argv = original_argv


def _configure_topology_args(
    base_args: argparse.Namespace,
    script_args: argparse.Namespace,
    data_type: str,
    topology_type: str,
) -> argparse.Namespace:
    """Create the matched Figure 5 configuration for one topology."""
    args = copy.deepcopy(base_args)
    args.data_type = data_type
    args.difficulty = script_args.difficulty
    args.seed = script_args.seed
    args.grid_size = script_args.grid_size
    args.topology_type = topology_type
    args.mst_nodes = script_args.mst_nodes or script_args.grid_size**2
    args.iterations = script_args.iterations  # Do not replace an explicit override.
    args.learning_rate = script_args.learning_rate
    args.initial_radius = script_args.initial_radius
    args.radius_decay_type = script_args.radius_decay_type
    args.radius_decay_factor = 1.0
    args.sampling_method = "full"
    args.initialization_method = "random"
    args.use_momentum = script_args.use_momentum
    args.momentum_init = script_args.momentum
    args.normalization = "xpysom"
    args.visualize = False
    args.verbose = script_args.verbose
    args.use_gpu = True
    return args


def _stable_pca_basis(data_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic two-component PCA basis fitted to all observations."""
    mean = np.mean(data_np, axis=0, dtype=np.float64, keepdims=True)
    centered = np.asarray(data_np, dtype=np.float64) - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1][:2]]
    # Eigenvector signs are arbitrary. Fix them for byte-stable reruns.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return mean, basis


def _project_to_2d(
    data_np: np.ndarray,
    weights_by_topology: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    if data_np.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {data_np.shape}")
    if data_np.shape[1] == 2:
        return data_np[:, :2], {key: value[:, :2] for key, value in weights_by_topology.items()}, "native"
    if data_np.shape[1] == 1:
        data_2d = np.column_stack((data_np[:, 0], np.zeros(data_np.shape[0])))
        weights_2d = {key: np.column_stack((value[:, 0], np.zeros(value.shape[0]))) for key, value in weights_by_topology.items()}
        return data_2d, weights_2d, "native"
    mean, basis = _stable_pca_basis(data_np)
    return (
        (np.asarray(data_np, dtype=np.float64) - mean) @ basis,
        {key: (np.asarray(value, dtype=np.float64) - mean) @ basis for key, value in weights_by_topology.items()},
        "pca",
    )


def _deterministic_display_sample(data_np: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Select a stable, order-preserving subset for raster display only."""
    if data_np.shape[0] <= limit:
        return np.asarray(data_np)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(data_np.shape[0], size=limit, replace=False))
    return np.asarray(data_np)[indices]


def _collect_connection_edges(som: Any, args: argparse.Namespace, weights: Any, weights_np: np.ndarray) -> List[Tuple[int, int]]:
    if args.topology_type in {"grid", "hexagonal"}:
        grid_size = som.topology.grid_size
        edges: List[Tuple[int, int]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if j < grid_size - 1 and idx + 1 < weights_np.shape[0]:
                    edges.append((idx, idx + 1))
                if i < grid_size - 1 and idx + grid_size < weights_np.shape[0]:
                    edges.append((idx, idx + grid_size))
        return edges
    if hasattr(som.topology, "is_reformed") and som.topology.is_reformed:
        adjacency = getattr(som, "adjacency_list", None)
        if adjacency is None:
            raise ValueError("Topology is reformed but adjacency_list is unavailable")
        return sorted({tuple(sorted((int(node), int(neighbor)))) for node, neighbors in adjacency.items() for neighbor in neighbors if node != neighbor})
    edges = getattr(som.topology, "mst_edges", None)
    if edges is None or len(edges) == 0:
        som.topology.update_topology(weights)
        edges = getattr(som.topology, "mst_edges", None)
    return [] if edges is None else [(int(u), int(v)) for u, v in edges]


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _limits(data: np.ndarray, records: Sequence[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs = [data[:, 0], *(record["weights_plot_np"][:, 0] for record in records)]
    ys = [data[:, 1], *(record["weights_plot_np"][:, 1] for record in records)]
    x_min, x_max = min(float(np.min(x)) for x in xs), max(float(np.max(x)) for x in xs)
    y_min, y_max = min(float(np.min(y)) for y in ys), max(float(np.max(y)) for y in ys)
    x_pad, y_pad = max(1e-6, 0.04 * (x_max - x_min)), max(1e-6, 0.04 * (y_max - y_min))
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _build_panel_mapper(limits: Tuple[float, float, float, float], plot_x: float, plot_y: float, plot_w: float, plot_h: float):
    x_min, x_max, y_min, y_max = limits
    x_span, y_span = max(1e-9, x_max - x_min), max(1e-9, y_max - y_min)
    if x_span / y_span >= plot_w / plot_h:
        draw_w, draw_h = plot_w, plot_w / (x_span / y_span)
        offset_x, offset_y = 0.0, 0.5 * (plot_h - draw_h)
    else:
        draw_h, draw_w = plot_h, plot_h * (x_span / y_span)
        offset_x, offset_y = 0.5 * (plot_w - draw_w), 0.0

    def map_point(x: float, y: float) -> Tuple[float, float]:
        return (
            plot_x + offset_x + ((x - x_min) / x_span) * draw_w,
            plot_y + offset_y + (1.0 - ((y - y_min) / y_span)) * draw_h,
        )

    return map_point


def _make_background_jpeg(
    points: np.ndarray,
    limits: Tuple[float, float, float, float],
    panel_width: float,
    panel_height: float,
    scale: float,
    quality: int,
) -> bytes:
    width, height = max(1, round(panel_width * scale)), max(1, round(panel_height * scale))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    mapper = _build_panel_mapper(limits, 0, 0, width, height)
    radius = max(1.0, 1.35 * scale)
    for x, y in points[:, :2]:
        sx, sy = mapper(float(x), float(y))
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=(128, 128, 128, 115))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return buffer.getvalue()


def _write_combined_svg(
    output_svg: Path,
    dataset_records: Sequence[Dict[str, Any]],
    *,
    display_limit: int,
    seed: int,
    jpeg_quality: int,
    raster_scale: float,
    background_dir: Path,
) -> List[Path]:
    """Write the 2x3 SVG and its two reusable row-background JPEGs."""
    if not dataset_records:
        raise ValueError("At least one dataset row is required")
    canvas_w, canvas_h = 1560, 132 + (464 * len(dataset_records))
    margin_x, panel_gap = 82.0, 18.0
    panel_w = (canvas_w - (2 * margin_x) - (2 * panel_gap)) / 3.0
    plot_inset_x, plot_w, plot_h = 14.0, panel_w - 28.0, 330.0
    row_tops = tuple(76.0 + (464.0 * index) for index in range(len(dataset_records)))
    background_dir.mkdir(parents=True, exist_ok=True)
    background_paths: List[Path] = []
    prepared: List[Dict[str, Any]] = []
    for row_index, dataset in enumerate(dataset_records):
        limits = _limits(dataset["data_plot_np"], dataset["panels"])
        sampled = _deterministic_display_sample(dataset["data_plot_np"], display_limit, seed + row_index)
        jpeg = _make_background_jpeg(sampled, limits, plot_w, plot_h, raster_scale, jpeg_quality)
        path = background_dir / f"{output_svg.stem}_{_dataset_tag(dataset['data_type'])}_background.jpg"
        path.write_bytes(jpeg)
        background_paths.append(path)
        prepared.append({**dataset, "limits": limits, "jpeg": jpeg, "display_count": sampled.shape[0]})

    fmt = lambda value: f"{value:.2f}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="Representative topology overlays across {len(prepared)} dataset rows">',
        f'  <rect width="{canvas_w}" height="{canvas_h}" fill="#FFFFFF"/>',
        "  <defs>",
    ]
    for row_index, dataset in enumerate(prepared):
        uri = "data:image/jpeg;base64," + base64.b64encode(dataset["jpeg"]).decode("ascii")
        lines.append(f'    <image id="cloud-row-{row_index}" width="{fmt(plot_w)}" height="{fmt(plot_h)}" preserveAspectRatio="none" xlink:href="{uri}"/>')
    lines.append("  </defs>")
    lines.append('  <text x="780" y="38" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="30" font-weight="700" fill="#111111">Figure 5: Representative Topologies</text>')

    for row_index, dataset in enumerate(prepared):
        row_top = row_tops[row_index]
        for column, panel in enumerate(dataset["panels"]):
            panel_x = margin_x + column * (panel_w + panel_gap)
            plot_x, plot_y = panel_x + plot_inset_x, row_top + 47.0
            label = chr(ord("A") + row_index * 3 + column)
            lines.append(f'  <text x="{fmt(panel_x + 4)}" y="{fmt(row_top + 28)}" font-family="DejaVu Sans, Arial, sans-serif" font-size="28" font-weight="700" fill="#111111">{label}</text>')
            lines.append(f'  <text x="{fmt(panel_x + 46)}" y="{fmt(row_top + 28)}" font-family="DejaVu Sans, Arial, sans-serif" font-size="23" font-weight="600" fill="#222222">{TOPOLOGY_DISPLAY_NAMES[panel["topology"]]}</text>')
            lines.append(f'  <use xlink:href="#cloud-row-{row_index}" x="{fmt(plot_x)}" y="{fmt(plot_y)}"/>')
            mapper = _build_panel_mapper(dataset["limits"], plot_x, plot_y, plot_w, plot_h)
            weights = panel["weights_plot_np"]
            for u, v in panel["edges"]:
                if min(u, v) < 0 or max(u, v) >= weights.shape[0]:
                    continue
                x1, y1 = mapper(float(weights[u, 0]), float(weights[u, 1]))
                x2, y2 = mapper(float(weights[v, 0]), float(weights[v, 1]))
                lines.append(f'  <line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" stroke="#111111" stroke-opacity="0.72" stroke-width="1.35"/>')
            for x, y in weights[:, :2]:
                sx, sy = mapper(float(x), float(y))
                lines.append(f'  <circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="3.00" fill="#D62728" stroke="#8B0000" stroke-width="0.70"/>')

    legend_y = row_tops[-1] + 455.0
    entries = (("point", "Observations"), ("node", "SOM nodes"), ("line", "Connections"))
    for idx, (kind, label) in enumerate(entries):
        x = 455.0 + idx * 300.0
        if kind == "point":
            lines.append(f'  <circle cx="{fmt(x)}" cy="{fmt(legend_y)}" r="5.5" fill="#808080" fill-opacity="0.65"/>')
        elif kind == "node":
            lines.append(f'  <circle cx="{fmt(x)}" cy="{fmt(legend_y)}" r="6" fill="#D62728" stroke="#8B0000" stroke-width="1"/>')
        else:
            lines.append(f'  <line x1="{fmt(x - 13)}" y1="{fmt(legend_y)}" x2="{fmt(x + 13)}" y2="{fmt(legend_y)}" stroke="#111111" stroke-width="3"/>')
        lines.append(f'  <text x="{fmt(x + 20)}" y="{fmt(legend_y + 8)}" font-family="DejaVu Sans, Arial, sans-serif" font-size="26" fill="#242424">{label}</text>')
    lines.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(lines), encoding="utf-8")
    return background_paths


def _write_summary_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "dataset", "training_dimensions", "display_mode", "generation_backend",
        "topology", "seed", "iterations_requested", "initial_learning_rate",
        "initial_radius", "radius_decay_type", "momentum_enabled",
        "initial_momentum", "normalization", "iterations_completed",
        "total_samples_processed", "train_time_s", "quantization_error",
        "n_nodes", "n_edges",
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.6f}" if column in {"train_time_s", "quantization_error"} else str(value))
            handle.write("\t".join(values) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_svg, summary_tsv, background_dir = _resolve_output_paths(args)
    benchmark = _load_benchmark_module()
    base_defaults = _get_benchmark_default_args(benchmark)
    dataset_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for data_type in args.data_types:
        data_args = _configure_topology_args(base_defaults, args, data_type, "hexagonal")
        data, metadata = benchmark.generate_data(data_args)
        data_np = np.asarray(data.get() if hasattr(data, "get") else data)
        panels: List[Dict[str, Any]] = []
        for topology in TOPOLOGIES:
            topology_args = _configure_topology_args(base_defaults, args, data_type, topology)
            som, stats, train_time = benchmark.train_floatsom(data, topology_args, metadata)
            weights = som.get_weights()
            weights_np = np.asarray(weights.get() if hasattr(weights, "get") else weights)
            qe = benchmark.QuantizationError(use_optimized=True).compute(benchmark.FloatSOMWrapper(som), data)
            edges = _collect_connection_edges(som, topology_args, weights, weights_np)
            panels.append({"topology": topology, "weights_np": weights_np, "edges": edges})
            summary_rows.append({
                "dataset": _dataset_tag(data_type), "training_dimensions": data_np.shape[1], "display_mode": "native" if data_np.shape[1] <= 2 else "shared_pca", "generation_backend": "FloatSOM_CuPy", "topology": topology, "seed": args.seed,
                "initial_learning_rate": topology_args.learning_rate, "initial_radius": topology_args.initial_radius,
                "radius_decay_type": topology_args.radius_decay_type, "momentum_enabled": topology_args.use_momentum,
                "initial_momentum": topology_args.momentum_init, "normalization": topology_args.normalization,
                "iterations_requested": args.iterations, "iterations_completed": int(stats.get("iterations_completed", 0)), "total_samples_processed": int(stats.get("total_samples_processed", 0)),
                "train_time_s": float(train_time), "quantization_error": float(qe), "n_nodes": weights_np.shape[0], "n_edges": len(edges),
            })
        data_plot, projected, axis_mode = _project_to_2d(data_np, {panel["topology"]: panel["weights_np"] for panel in panels})
        for panel in panels:
            panel["weights_plot_np"] = projected[panel["topology"]]
        dataset_records.append({"data_type": data_type, "training_dimensions": data_np.shape[1], "data_plot_np": data_plot, "panels": panels, "axis_mode": axis_mode})
    backgrounds = _write_combined_svg(output_svg, dataset_records, display_limit=args.display_limit, seed=args.seed, jpeg_quality=args.jpeg_quality, raster_scale=args.raster_scale, background_dir=background_dir)
    _write_summary_tsv(summary_tsv, summary_rows)
    print(f"Representative figure saved to: {output_svg}")
    print(f"Background JPEGs saved to: {', '.join(map(str, backgrounds))}")
    print(f"Benchmark summary saved to: {summary_tsv}")


if __name__ == "__main__":
    main()
