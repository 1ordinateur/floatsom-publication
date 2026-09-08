#!/usr/bin/env python3
"""
FloatSOM benchmarks with sklearn toy datasets and topology metrics.

This script provides benchmarking capabilities for FloatSOM using sklearn
toy datasets and comprehensive topology evaluation metrics.
"""

import os
import sys
import time
import argparse
import copy
import cProfile
import pstats
import io
import tempfile
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

# Optional Ray import for multi-GPU support
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

# Add the parent directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# Import FloatSOM components
from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig, RayConfig
from floatsom.processing.processing_params import calculate_auto_chunk_size_for_method

# Import evaluation framework from FloatSOM
from floatsom_benchmarks.evaluation.sklearn_datasets import (
    generate_sklearn_dataset, 
    get_available_datasets,
    get_dataset_info,
    SKLEARN_AVAILABLE
)
from floatsom_benchmarks.evaluation.metrics import (
    QuantizationError,
    TopographicError, 
    Trustworthiness, 
    NeighborhoodPreservation,
    DistortionMeasure,
    TopographicFunction
)

# Import data generation utilities
from floatsom_benchmarks.data_generation import (
    generate_2d_test_data,
    generate_3d_test_data,
    generate_test_data,
    generate_random_data
)

# Import MiniSOM adapter
try:
    from floatsom.adapters.minisom_adapter import MiniSOMAdapter, MINISOM_AVAILABLE
except ImportError:
    MINISOM_AVAILABLE = False


REPRESENTATIVE_TOPOLOGY_ORDER: Tuple[str, ...] = ("hexagonal", "mst", "rng")
REPRESENTATIVE_TOPOLOGY_DISPLAY_NAMES: Dict[str, str] = {
    "hexagonal": "Hexagonal",
    "mst": "MST",
    "rng": "RNG",
}
REPRESENTATIVE_PANEL_LABELS: Tuple[str, ...] = ("A", "B", "C")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FloatSOM benchmarks with sklearn datasets and topology metrics')
    
    # Data generation parameters
    parser.add_argument('--data_type', type=str, default='clusters',
                        choices=['2d', '3d', 'clusters', 'clusters_nd', 'random', 'sklearn_swiss_roll', 'sklearn_s_curve', 
                                'sklearn_moons', 'sklearn_circles', 'sklearn_blobs', 'sklearn_breast_cancer', 
                                'sklearn_wine', 'sklearn_iris', 'sklearn_digits', 'sklearn_diabetes', 'sklearn_olivetti_faces'],
                        help='Type of data to generate')
    
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to a pre-generated dataset (.zarr). Overrides data generation when provided')

    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for sklearn datasets')
    
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to generate (overridden by sklearn dataset configs)')
    
    parser.add_argument('--input_dim', type=int, default=30,
                        help='Input dimension for cluster data (also used for clusters_nd, ignored for 2d and 3d)')
    
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters in cluster data (ignored for 2d and 3d)')
    
    # SOM configuration parameters
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Size of SOM grid (e.g., 10 will create a 10x10 grid for 2D)')
    parser.add_argument('--mst_nodes', type=int, default=None,
                        help='Explicit node count for graph topologies (MST/RNG); overrides grid_size**2 default')
    
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Initial learning rate')
    
    parser.add_argument('--initial_radius', type=float, default=None,
                        help='Initial neighborhood radius (if None, will be calculated based on topology)')
    
    parser.add_argument('--lr_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'asymptotic'],
                        help='Decay type for learning rate')
    
    parser.add_argument('--radius_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'asymptotic'],
                        help='Decay type for radius')
    
    parser.add_argument('--lr_decay_factor', type=float, default=8.0,
                        help='Decay factor for learning rate (higher = slower decay)')
    
    parser.add_argument('--radius_decay_factor', type=float, default=1.0,
                        help='Decay factor for radius (higher = slower decay)')
    
    # Sampling configuration
    parser.add_argument('--sampling_method', type=str, default='full',
                        choices=['full', 'random', 'hdsssom'],
                        help='Sample selection method')
    
    # Processing configuration
    parser.add_argument('--processing_method', type=str, default='batch',
                        choices=['batch', 'colors', 'minisom'],
                        help='Processing method')
    
    parser.add_argument('--batch_mode', type=str, default='full_batch',
                        choices=['full_batch', 'minibatch'],
                        help='Batch processing mode')
    
    
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for batch processing (None for auto-calculate)')
    
    parser.add_argument('--use_momentum', action='store_true', default=False,
                        help='Enable momentum in weight updates')
    
    parser.add_argument('--no_momentum', action='store_false', dest='use_momentum',
                        help='Disable momentum in weight updates')
    
    parser.add_argument('--momentum_init', type=float, default=0.5,
                        help='Initial momentum coefficient')
    
    parser.add_argument('--normalization', type=str, default='xpysom',
                        choices=['count_based', 'weighted', 'hybrid', 'clamped_weighted', 'local', 'adaptive', 'none', 'minisom_weighted', 'xpysom'],
                        help='Normalization method for weight updates')
    
    parser.add_argument('--norm_alpha', type=float, default=None,
                        help='Blend ratio for hybrid normalization (0.0-1.0, required for hybrid)')
    
    parser.add_argument('--norm_clamp_factor', type=float, default=None,
                        help='Clamp factor for clamped_weighted normalization (>0, required for clamped_weighted)')
    
    parser.add_argument('--norm_percentile', type=float, default=None,
                        help='Percentile threshold for local normalization (0-100, required for local)')
    
    parser.add_argument('--norm_max_update_threshold', type=float, default=None,
                        help='Threshold for extreme value validation (optional, >0)')
    
    parser.add_argument('--virtual_ratio', type=float, default=0.5,
                        help='Virtual samples ratio for count_based/weighted normalization (0.2=fast, 0.5=balanced, 1.0=stable)')
    
    # Topology configuration
    parser.add_argument('--topology_type', type=str, default='grid',
                        choices=['grid', 'hexagonal', 'mst', 'rng'],
                        help='Type of topology')
    
    parser.add_argument('--topology_variant', type=str, default='planar',
                        choices=['planar', 'toroidal'],
                        help='Topology variant (for grid/hexagonal)')
    
    # Graph-topology parameters (MST/RNG)
    parser.add_argument('--mst_update_frequency', type=int, default=None,
                        help='How often to update graph topology during training (when not using dynamic frequency)')
    
    parser.add_argument('--dynamic_mst_frequency', action='store_true', default=True,
                        help='Use dynamic MST update frequency with decay')
    
    parser.add_argument('--no_dynamic_mst_frequency', action='store_false', dest='dynamic_mst_frequency',
                        help='Use fixed MST update frequency')
    
    parser.add_argument('--mst_decay_function', type=str, default='exponential',
                        choices=['exponential', 'linear', 'sigmoid', 'gaussian', 'asymptotic'],
                        help='Decay function for dynamic MST frequency')
    
    parser.add_argument('--initial_mst_frequency', type=int, default=1,
                        help='Initial MST update frequency (update every N iterations)')
    
    parser.add_argument('--final_mst_frequency', type=int, default=10,
                        help='Final MST update frequency (update every N iterations)')
    
    # Graph-topology grid reformation parameters
    parser.add_argument('--reform_grid', action='store_true',
                        help='Reform graph topology (MST/RNG) to grid structure')
    
    parser.add_argument('--reform_grid_type', type=str, default='regular',
                        choices=['regular', 'hexagonal'],
                        help='Type of grid for graph-topology reformation')
    
    # Evaluation parameters
    parser.add_argument('--enable_topology_metrics', action='store_true',
                        help='Enable topology preservation metrics')
    
    parser.add_argument('--topology_metrics', type=str, nargs='+',
                        default=['topographic_error', 'trustworthiness', 'neighborhood_preservation', 
                                'distortion_measure', 'topographic_function', 'quantization_error'],
                        help='Topology metrics to compute')
    
    parser.add_argument('--topology_k', type=int, default=7,
                        help='Number of neighbors for k-NN based topology metrics')
    
    # Visualization parameters
    parser.add_argument('--save_iterations', action='store_true', default=True,
                        help='Save intermediate weights during training')
    
    parser.add_argument('--no_save_iterations', action='store_false', dest='save_iterations',
                        help='Do not save intermediate weights during training')
    
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save weights every N iterations')
    
    # Other parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for computation')
    
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling to output code performance statistics')
    
    parser.add_argument('--profile_output', type=str, default='floatsom_benchmark_profile.txt',
                        help='Output file for profiling results')
    
    parser.add_argument('--tempdir', type=str, default=None,
                        help='Directory to use for Ray temporary storage (auto-created if omitted)')
    
    parser.add_argument('--output_file', type=str, default='floatsom_benchmark_results.txt',
                        help='Output file for benchmark results report')
    
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help='Convergence threshold for early stopping')
    
    parser.add_argument('--min_iterations', type=int, default=20,
                        help='Minimum iterations before checking convergence')
    
    parser.add_argument('--initialization_method', type=str, default='random',
                        choices=['random', 'pca', 'pca_sampling', 'pca_sampling_snake', 'pca_density'],
                        help='Method to initialize weights (pca=standard PCA, pca_sampling=sampled PCA with grid ordering, pca_sampling_snake=snake pattern, pca_density=density-weighted PCA)')
    
    parser.add_argument('--minisom_defaults', action='store_true',
                        help='Use MiniSom default hyperparameters instead of framework defaults when using MiniSOM')

    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the results')
    parser.add_argument(
        '--representative_topology_column',
        action='store_true',
        help='Generate a single-column representative SVG for Hexagonal, MST, and RNG.'
    )
    parser.add_argument(
        '--representative_output_svg',
        type=str,
        default='topology_representative_hex_mst_rng_column.svg',
        help='Representative SVG output path (absolute, or relative to benchmark output directory).'
    )
    
    # Ray multi-GPU parameters
    parser.add_argument('--use_ray', action='store_true',
                        help='Enable Ray multi-GPU processing for batch method')
    
    parser.add_argument('--ray_gpu_count', type=int, default=None,
                        help='Number of GPUs to use with Ray (None for auto-detect)')
    
    parser.add_argument('--ray_collective_group', type=str, default='default',
                        help='Name for NCCL collective group (default: "default")')
    
    parser.add_argument('--ray_chunk_size', type=int, default=None,
                        help='Internal chunk size for processing within each GPU (None for auto-calculate)')
    
    return parser.parse_args()


def _to_numpy(array: Any) -> np.ndarray:
    """Convert array-likes (including CuPy) to NumPy for plotting/reporting."""
    if hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


def _xml_escape(value: str) -> str:
    """Escape XML text/attribute content."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _collect_connection_edges(
    som: Union[FloatSOM, MiniSOMAdapter],
    args: argparse.Namespace,
    weights: Union[np.ndarray, cp.ndarray],
    weights_np: np.ndarray,
) -> Tuple[List[Tuple[int, int]], str]:
    """Return (edge list, matplotlib color shorthand) describing node connectivity."""
    if args.topology_type in ['grid', 'hexagonal']:
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
        return edges, 'r'

    if args.topology_type in {'mst', 'rng'}:
        if hasattr(som.topology, 'is_reformed') and som.topology.is_reformed:
            adjacency = getattr(som, 'adjacency_list', None)
            if adjacency is None:
                raise ValueError("Topology is reformed but adjacency_list not found!")
            edge_set: set = set()
            for node, neighbors in adjacency.items():
                for neighbor in neighbors:
                    if neighbor == node:
                        continue
                    edge = tuple(sorted((int(node), int(neighbor))))
                    edge_set.add(edge)
            return sorted(list(edge_set)), 'r'

        edges = getattr(som.topology, 'mst_edges', None)
        if (edges is None) or (len(edges) == 0):
            if hasattr(som, 'topology') and hasattr(som.topology, 'update_topology'):
                som.topology.update_topology(weights)
            edges = getattr(som.topology, 'mst_edges', None)
        if edges is None:
            return [], 'k'
        return [(int(u), int(v)) for u, v in edges], 'k'

    return [], 'r'


def _draw_edges_2d(ax: Any, coords: np.ndarray, edges: List[Tuple[int, int]], color: str, alpha: float) -> None:
    """Draw 2D edges between indexed coordinates."""
    for u, v in edges:
        ax.plot(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            color + '-',
            alpha=alpha,
            linewidth=1,
        )


def _draw_edges_3d(ax: Any, coords: np.ndarray, edges: List[Tuple[int, int]], color: str, alpha: float) -> None:
    """Draw 3D edges between indexed coordinates."""
    for u, v in edges:
        ax.plot(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            [coords[u, 2], coords[v, 2]],
            color + '-',
            alpha=alpha,
            linewidth=1,
        )


def _project_representative_coordinates(
    data_np: np.ndarray,
    weights_by_topology: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """
    Project data + topology weights into a shared 2D coordinate system.

    Returns:
        data_2d: Data coordinates in 2D.
        weights_2d: Mapping topology -> projected 2D weight coordinates.
        axis_label_mode: Either "dimension" or "pca".
    """
    if data_np.ndim != 2:
        raise ValueError(f"Representative figure expects 2D matrix data, got shape {data_np.shape}.")

    if data_np.shape[1] == 2:
        return data_np[:, :2], {k: v[:, :2] for k, v in weights_by_topology.items()}, "dimension"

    if data_np.shape[1] == 1:
        zeros_data = np.zeros((data_np.shape[0], 1), dtype=data_np.dtype)
        data_2d = np.concatenate([data_np, zeros_data], axis=1)
        weights_2d = {}
        for topology, weights in weights_by_topology.items():
            zeros_weights = np.zeros((weights.shape[0], 1), dtype=weights.dtype)
            weights_2d[topology] = np.concatenate([weights, zeros_weights], axis=1)
        return data_2d, weights_2d, "dimension"

    mean_vec = np.mean(data_np, axis=0, keepdims=True)
    centered_data = data_np - mean_vec
    _, _, vh = np.linalg.svd(centered_data, full_matrices=False)
    basis = vh[:2].T
    data_2d = centered_data @ basis
    weights_2d = {}
    for topology, weights in weights_by_topology.items():
        weights_2d[topology] = (weights - mean_vec) @ basis
    return data_2d, weights_2d, "pca"


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
    """Build equal-aspect mapper from data space into an SVG panel box."""
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


def _darken_hex_color(hex_color: str, factor: float = 0.58) -> str:
    """Return darker variant of a #RRGGBB color."""
    value = hex_color.strip()
    if not value.startswith('#') or len(value) != 7:
        return value
    r = int(value[1:3], 16)
    g = int(value[3:5], 16)
    b = int(value[5:7], 16)
    r = max(0, min(255, int(round(r * factor))))
    g = max(0, min(255, int(round(g * factor))))
    b = max(0, min(255, int(round(b * factor))))
    return f"#{r:02X}{g:02X}{b:02X}"


def _load_representative_topology_colors() -> Dict[str, str]:
    """Load canonical topology colors from shared visualization configuration."""
    try:
        from floatsom_benchmarks.visualization import TOPOLOGY_BASE_COLORS
    except Exception as exc:
        raise ImportError(
            "Representative topology figure requires floatsom_benchmarks.visualization "
            "to load canonical topology colors."
        ) from exc

    color_map: Dict[str, str] = {}
    for topology in REPRESENTATIVE_TOPOLOGY_ORDER:
        color_value = TOPOLOGY_BASE_COLORS.get(topology)
        if color_value is None:
            raise KeyError(f"Missing canonical color for topology '{topology}' in visualization.py")
        color_map[topology] = str(color_value)
    return color_map


def _write_representative_topology_column_svg(
    output_svg: Path,
    data_2d: np.ndarray,
    panel_records: List[Dict[str, Any]],
    dataset_name: str,
    axis_label_mode: str,
) -> None:
    """Write one-column Hex/MST/RNG representative overlay SVG with shared legend."""
    canvas_w = 1160
    canvas_h = 2220
    margin_x = 60
    panel_y_start = 72
    panel_gap = 28
    panel_h = 620
    panel_w = canvas_w - (2 * margin_x)

    x_values = [data_2d[:, 0]]
    y_values = [data_2d[:, 1]]
    for record in panel_records:
        weights_np = record["weights_2d"]
        x_values.append(weights_np[:, 0])
        y_values.append(weights_np[:, 1])

    x_min = float(min(np.min(arr) for arr in x_values))
    x_max = float(max(np.max(arr) for arr in x_values))
    y_min = float(min(np.min(arr) for arr in y_values))
    y_max = float(max(np.max(arr) for arr in y_values))
    x_pad = max(1e-6, 0.06 * (x_max - x_min))
    y_pad = max(1e-6, 0.06 * (y_max - y_min))
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    x_label = "Dimension 1" if axis_label_mode == "dimension" else "First Principal Component"
    y_label = "Dimension 2" if axis_label_mode == "dimension" else "Second Principal Component"

    def fmt(value: float) -> str:
        return f"{value:.2f}"

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" '
            f'height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" role="img" '
            'aria-label="Representative topology overlays">'
        ),
        f'  <rect x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="#FFFFFF"/>',
        (
            f'  <text x="{fmt(canvas_w / 2.0)}" y="42" text-anchor="middle" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="28" font-weight="700" fill="#1f1f1f">'
            f'{_xml_escape("Representative Node-Connection Overlays on " + str(dataset_name))}'
            '</text>'
        ),
    ]

    for idx, record in enumerate(panel_records):
        panel_x = margin_x
        panel_y = panel_y_start + idx * (panel_h + panel_gap)
        topology = str(record["topology"])
        topology_name = REPRESENTATIVE_TOPOLOGY_DISPLAY_NAMES[topology]
        panel_label = REPRESENTATIVE_PANEL_LABELS[idx]
        weights_2d = record["weights_2d"]
        edges = record["edges"]
        series_color = str(record["color"])
        border_color = _darken_hex_color(series_color)

        plot_x = panel_x + 34.0
        plot_y = panel_y + 74.0
        plot_w = panel_w - 68.0
        plot_h = panel_h - 118.0

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
            f'  <rect x="{fmt(panel_x)}" y="{fmt(panel_y)}" width="{fmt(panel_w)}" '
            f'height="{fmt(panel_h)}" fill="#FCFCFC" stroke="#D7D7D7" stroke-width="1.2"/>'
        )
        lines.append(
            f'  <text x="{fmt(panel_x + 16)}" y="{fmt(panel_y + 31)}" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="30" font-weight="700" fill="#111111">'
            f"{_xml_escape(panel_label)}"
            "</text>"
        )
        lines.append(
            f'  <text x="{fmt(panel_x + panel_w / 2.0)}" y="{fmt(panel_y + 34)}" text-anchor="middle" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="24" font-weight="600" fill="#222222">'
            f"{_xml_escape(topology_name)}"
            "</text>"
        )
        lines.append(
            f'  <rect x="{fmt(plot_x)}" y="{fmt(plot_y)}" width="{fmt(plot_w)}" height="{fmt(plot_h)}" '
            'fill="#FFFFFF" stroke="#C4C4C4" stroke-width="1"/>'
        )

        for x_val, y_val in data_2d[:, :2]:
            sx, sy = map_point(float(x_val), float(y_val))
            lines.append(
                f'  <circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="1.30" fill="#808080" fill-opacity="0.45" />'
            )

        for u, v in edges:
            if u < 0 or v < 0 or u >= weights_2d.shape[0] or v >= weights_2d.shape[0]:
                continue
            x1, y1 = map_point(float(weights_2d[u, 0]), float(weights_2d[u, 1]))
            x2, y2 = map_point(float(weights_2d[v, 0]), float(weights_2d[v, 1]))
            lines.append(
                f'  <line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
                f'stroke="{series_color}" stroke-opacity="0.70" stroke-width="1.15"/>'
            )

        for x_val, y_val in weights_2d[:, :2]:
            sx, sy = map_point(float(x_val), float(y_val))
            lines.append(
                f'  <circle cx="{fmt(sx)}" cy="{fmt(sy)}" r="3.30" fill="{series_color}" '
                f'stroke="{border_color}" stroke-width="0.85"/>'
            )

        lines.append(
            f'  <text x="{fmt(panel_x + panel_w / 2.0)}" y="{fmt(panel_y + panel_h - 15)}" text-anchor="middle" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="16" fill="#303030">'
            f'{_xml_escape(x_label)}'
            '</text>'
        )
        lines.append(
            f'  <text x="{fmt(panel_x + 10)}" y="{fmt(panel_y + panel_h / 2.0)}" text-anchor="middle" '
            'font-family="DejaVu Sans, Arial, sans-serif" font-size="16" fill="#303030" '
            f'transform="rotate(-90 {fmt(panel_x + 10)} {fmt(panel_y + panel_h / 2.0)})">'
            f'{_xml_escape(y_label)}'
            '</text>'
        )

    legend_y = panel_y_start + (3 * panel_h) + (2 * panel_gap) + 100
    legend_label_y = legend_y - 28
    legend_start_x = 120
    legend_gap = 248
    lines.append(
        f'  <text x="{fmt(canvas_w / 2.0)}" y="{fmt(legend_label_y)}" text-anchor="middle" '
        'font-family="DejaVu Sans, Arial, sans-serif" font-size="20" font-weight="600" fill="#242424">'
        'Common Legend'
        '</text>'
    )

    lines.append(
        f'  <circle cx="{fmt(legend_start_x)}" cy="{fmt(legend_y)}" r="4.0" fill="#808080" fill-opacity="0.65" />'
    )
    lines.append(
        f'  <text x="{fmt(legend_start_x + 16)}" y="{fmt(legend_y + 5)}" font-family="DejaVu Sans, Arial, sans-serif" '
        'font-size="16" fill="#242424">Data points</text>'
    )

    for idx, topology in enumerate(REPRESENTATIVE_TOPOLOGY_ORDER, start=1):
        color = str(panel_records[idx - 1]["color"])
        border_color = _darken_hex_color(color)
        x0 = legend_start_x + (idx * legend_gap)
        lines.append(
            f'  <line x1="{fmt(x0 - 13)}" y1="{fmt(legend_y)}" x2="{fmt(x0 + 9)}" y2="{fmt(legend_y)}" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
        lines.append(
            f'  <circle cx="{fmt(x0 + 16)}" cy="{fmt(legend_y)}" r="4.4" fill="{color}" stroke="{border_color}" stroke-width="0.8"/>'
        )
        lines.append(
            f'  <text x="{fmt(x0 + 28)}" y="{fmt(legend_y + 5)}" font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="16" fill="#242424">'
            f'{_xml_escape(REPRESENTATIVE_TOPOLOGY_DISPLAY_NAMES[topology] + " (nodes + connections)")}'
            '</text>'
        )

    lines.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def _resolve_representative_output_svg_path(output_dir: str, representative_output_svg: str) -> Path:
    """Resolve representative SVG path using output dir when given a relative path."""
    output_path = Path(representative_output_svg).expanduser()
    if output_path.is_absolute():
        return output_path
    return Path(output_dir) / output_path


def _cleanup_ray_resources_if_needed(som: Any, args: argparse.Namespace) -> None:
    """Release Ray processor resources for a trained SOM when applicable."""
    if args.processing_method == 'batch' and args.use_ray and hasattr(som, 'processor'):
        if hasattr(som.processor, 'cleanup'):
            som.processor.cleanup()


def generate_representative_topology_column_figure(
    data: Union[cp.ndarray, np.ndarray, np.memmap],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: str,
    pre_trained_soms: Optional[Dict[str, Union[FloatSOM, MiniSOMAdapter]]] = None,
) -> str:
    """Generate one-column representative SVG comparing Hexagonal, MST, and RNG."""
    topology_colors = _load_representative_topology_colors()
    data_np = _to_numpy(data)
    if data_np.ndim != 2:
        raise ValueError(
            f"Representative topology figure requires tabular matrix data, got shape {data_np.shape}."
        )

    trained_map = dict(pre_trained_soms or {})
    locally_trained: List[Tuple[Union[FloatSOM, MiniSOMAdapter], argparse.Namespace]] = []
    weights_by_topology: Dict[str, np.ndarray] = {}
    edges_by_topology: Dict[str, List[Tuple[int, int]]] = {}

    for topology in REPRESENTATIVE_TOPOLOGY_ORDER:
        topology_args = copy.deepcopy(args)
        topology_args.topology_type = topology

        som = trained_map.get(topology)
        if som is None:
            som, _, _ = train_floatsom(data, topology_args, metadata)
            trained_map[topology] = som
            locally_trained.append((som, topology_args))

        weights = som.get_weights()
        weights_np = _to_numpy(weights)
        edges, _ = _collect_connection_edges(som, topology_args, weights, weights_np)
        weights_by_topology[topology] = weights_np
        edges_by_topology[topology] = edges

    data_2d, weights_2d_map, axis_label_mode = _project_representative_coordinates(
        data_np=data_np,
        weights_by_topology=weights_by_topology,
    )

    panel_records: List[Dict[str, Any]] = []
    for topology in REPRESENTATIVE_TOPOLOGY_ORDER:
        panel_records.append(
            {
                "topology": topology,
                "weights_2d": weights_2d_map[topology],
                "edges": edges_by_topology[topology],
                "color": topology_colors[topology],
            }
        )

    output_path = _resolve_representative_output_svg_path(
        output_dir=output_dir,
        representative_output_svg=args.representative_output_svg,
    )
    _write_representative_topology_column_svg(
        output_svg=output_path,
        data_2d=data_2d,
        panel_records=panel_records,
        dataset_name=str(metadata.get("dataset_name", "dataset")),
        axis_label_mode=axis_label_mode,
    )

    for local_som, local_args in locally_trained:
        _cleanup_ray_resources_if_needed(local_som, local_args)

    return str(output_path)


def generate_data(args):
    """Generate data based on specified type and parameters."""
    # Create dataset cache path based on configuration
    import os
    import hashlib
    import numpy as np
    
    # Set random seed
    np.random.seed(args.seed)
    
    if args.dataset_path is not None:
        dataset_path = os.path.abspath(args.dataset_path)
        if not dataset_path.endswith('.zarr'):
            raise ValueError(f"dataset_path must point to a .zarr store: {dataset_path}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        try:
            import zarr
        except ImportError as exc:
            raise ImportError("The 'zarr' package is required to load datasets via --dataset_path."
                              " Install it with 'pip install zarr'.") from exc

        print(f"Loading dataset from {dataset_path}...")
        zarr_obj = zarr.open(dataset_path, mode='r')

        if getattr(zarr_obj, 'shape', None) is None:
            raise ValueError("dataset_path must reference a zarr array, not a group."
                             " Provide the path to the array store (e.g., dataset.zarr/array).")

        data_np = np.array(zarr_obj[:], dtype=np.float32)

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        args.samples = data_np.shape[0]
        args.input_dim = data_np.shape[1] if data_np.ndim > 1 else 1

        data = cp.array(data_np) if args.use_gpu else data_np

        dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
        if dataset_name.endswith('.zarr'):
            dataset_name = dataset_name[:-5]

        metadata = {
            'dataset_name': dataset_name,
            'shape': data_np.shape,
            'intrinsic_dim': data_np.shape[1] if data_np.ndim > 1 else 1,
            'embedding_dim': data_np.shape[1] if data_np.ndim > 1 else 1,
            'dtype': str(data_np.dtype),
            'data_path': dataset_path,
            'source': 'zarr'
        }

        return data, metadata

    if args.data_type.startswith('sklearn_'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for toy datasets")
        
        dataset_name = args.data_type.replace('sklearn_', '')
        
        print(f"Generating sklearn {dataset_name} dataset (difficulty: {args.difficulty})...")
        
        # Override n_features for blobs if needed
        n_features = args.input_dim if dataset_name == 'blobs' else None
        
        
        cache_dir = os.path.join(os.getcwd(), 'dataset_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create unique filename based on dataset config
        config_str = f"{dataset_name}_{args.difficulty}_{args.seed}_{n_features}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        data_file = os.path.join(cache_dir, f"{dataset_name}_{args.difficulty}_{config_hash}_data.npy")
        metadata_file = os.path.join(cache_dir, f"{dataset_name}_{args.difficulty}_{config_hash}_metadata.npz")
        
        # Check if cached dataset exists
        if os.path.exists(data_file) and os.path.exists(metadata_file):
            print(f"Loading cached dataset from {data_file}...")
            data = np.load(data_file)
            metadata_cache = np.load(metadata_file, allow_pickle=True)
            metadata = metadata_cache['metadata'].item()
            metadata['data_path'] = data_file
        else:
            print(f"Generating new sklearn {dataset_name} dataset (difficulty: {args.difficulty})...")
            data, metadata = generate_sklearn_dataset(
                dataset_name=dataset_name,
                difficulty=args.difficulty,
                seed=args.seed,
                n_features=n_features,
                normalize=True
            )
            
            # Save dataset and metadata to cache
            print(f"Caching dataset to {data_file}...")
            np.save(data_file, data)
            np.savez(metadata_file, metadata=metadata)
            metadata['data_path'] = data_file
        


        if 'n_clusters' in metadata:
            print(f"Number of clusters: {metadata['n_clusters']}")
        
        return data, metadata
        
    elif args.data_type == '2d':
        print(f"Generating 2D W-shaped data with {args.samples} samples...")
        data = generate_2d_test_data(n_points=args.samples, seed=args.seed)
        metadata = {'dataset_name': '2d_w_shaped', 'shape': data.shape, 'intrinsic_dim': 2, 'embedding_dim': 2}
        
    elif args.data_type == '3d':
        print(f"Generating 3D bowl-shaped data with {args.samples} samples...")
        data = generate_3d_test_data(n_points=args.samples, seed=args.seed)
        metadata = {'dataset_name': '3d_bowl_shaped', 'shape': data.shape, 'intrinsic_dim': 3, 'embedding_dim': 3}
        
    elif args.data_type == 'clusters_nd':
        # N-dimensional clustered data with user-specified dimensions
        cache_dir = os.path.join(os.getcwd(), 'dataset_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create unique filename based on configuration
        config_str = f"clusters_nd_{args.samples}_{args.input_dim}_{args.clusters}_{args.seed}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        data_file = os.path.join(cache_dir, f"clusters_nd_{args.samples}_{args.input_dim}_{args.clusters}_{config_hash}_data.npy")
        metadata_file = os.path.join(cache_dir, f"clusters_nd_{args.samples}_{args.input_dim}_{args.clusters}_{config_hash}_metadata.npz")
        
        # Check if cached dataset exists
        if os.path.exists(data_file) and os.path.exists(metadata_file):
            print(f"Loading cached clusters_nd dataset from {data_file}...")
            data_np = np.load(data_file)
            data = cp.array(data_np, dtype=cp.float32)
            metadata_cache = np.load(metadata_file, allow_pickle=True)
            metadata = metadata_cache['metadata'].item()
            metadata['data_path'] = data_file
        else:
            print(f"Generating {args.input_dim}D clustered data with {args.samples} samples, {args.clusters} clusters...")
            data = generate_test_data(
                n_samples=args.samples,
                input_dim=args.input_dim,
                n_clusters=args.clusters,
                verbose=args.verbose
            )
            metadata = {
                'dataset_name': f'clusters_{args.input_dim}d', 
                'shape': data.shape, 
                'intrinsic_dim': args.input_dim, 
                'embedding_dim': args.input_dim,
                'n_clusters': args.clusters
            }
            
            # Save dataset and metadata to cache
            print(f"Caching clusters_nd dataset to {data_file}...")
            data_np = data.get() if hasattr(data, 'get') else data
            np.save(data_file, data_np)
            np.savez(metadata_file, metadata=metadata)
            metadata['data_path'] = data_file
        
    elif args.data_type == 'random':
        # Random uniform data with memory-mapped file for large datasets
        cache_dir = os.path.join(os.getcwd(), 'dataset_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create unique filename based on configuration
        config_str = f"random_{args.samples}_{args.input_dim}_{args.seed}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        memmap_file = os.path.join(cache_dir, 'memmap_temp', f"random_{args.samples}_{args.input_dim}_{config_hash}.dat")
        metadata_file = os.path.join(cache_dir, f"random_{args.samples}_{args.input_dim}_{config_hash}_metadata.npz")
        
        # Check if cached dataset exists
        if os.path.exists(memmap_file) and os.path.exists(metadata_file):
            print(f"Loading cached random dataset from {memmap_file}...")
            # Load as memory-mapped array
            data = np.memmap(memmap_file, dtype=np.float32, mode='r', shape=(args.samples, args.input_dim))
            metadata_cache = np.load(metadata_file, allow_pickle=True)
            metadata = metadata_cache['metadata'].item()
            metadata['data_path'] = memmap_file
            metadata['is_memmap'] = True
        else:
            print(f"Generating random data with {args.samples} samples, {args.input_dim} dimensions...")
            data = generate_random_data(
                n_samples=args.samples,
                input_dim=args.input_dim,
                seed=args.seed,
                verbose=args.verbose,
                memmap_file=memmap_file
            )
            metadata = {
                'dataset_name': 'random',
                'shape': data.shape,
                'intrinsic_dim': args.input_dim,
                'embedding_dim': args.input_dim,
                'seed': args.seed,
                'data_path': memmap_file,
                'is_memmap': True
            }
            
            # Save metadata only (data is already in memmap file)
            print(f"Saving metadata to {metadata_file}...")
            np.savez(metadata_file, metadata=metadata)
        
    else:  # 'clusters' (default 30D)
        print(f"Generating clustered data with {args.samples} samples, {args.input_dim} dimensions, {args.clusters} clusters...")
        data = generate_test_data(
            n_samples=args.samples,
            input_dim=args.input_dim,
            n_clusters=args.clusters,
            verbose=args.verbose
        )
        metadata = {'dataset_name': 'clusters', 'shape': data.shape, 'intrinsic_dim': args.input_dim, 'embedding_dim': args.input_dim}
    
    return data, metadata


def compute_topology_metrics(som: FloatSOM, data: cp.ndarray, args) -> Dict[str, float]:
    """Compute topology preservation metrics for FloatSOM results."""
    
    if not args.enable_topology_metrics:
        return {}
    
    print("\nComputing topology metrics...")
    
    # Initialize metrics
    metrics = {}
    use_gpu = args.use_gpu
    available_metrics = {
        'distortion_measure': DistortionMeasure(use_gpu=use_gpu, sigma=1.0),
        'topographic_error': TopographicError(use_gpu=use_gpu),
        'trustworthiness': Trustworthiness(k=args.topology_k),
        'neighborhood_preservation': NeighborhoodPreservation(k=args.topology_k),
        'topographic_function': TopographicFunction(use_gpu=use_gpu),
        'quantization_error': QuantizationError(use_optimized=use_gpu)
    }
    
    # Select requested metrics
    selected_metrics = []
    for metric_name in args.topology_metrics:
        if metric_name in available_metrics:
            selected_metrics.append(available_metrics[metric_name])
        else:
            print(f"Warning: Unknown metric '{metric_name}', skipping")
    
    # Add quantization error if not already included
    if 'quantization_error' not in args.topology_metrics:
        selected_metrics.append(available_metrics['quantization_error'])
    
    # Create a wrapper to make FloatSOM compatible with metric interface
    som_wrapper = FloatSOMWrapper(som)
    
    # Compute each metric
    for metric in selected_metrics:
        metric_value = metric.compute(som_wrapper, data)
        metrics[metric.name] = metric_value
        print(f"  {metric.name}: {metric_value:.6f}")
    
    return metrics


class FloatSOMWrapper:
    """Wrapper to make FloatSOM/MiniSOM compatible with metric interfaces."""
    
    def __init__(self, som: Union[FloatSOM, MiniSOMAdapter]):
        self.floatsom = som
        self.weights = som.get_weights()
        self.topology = getattr(som, 'topology', None)

        topology_config = getattr(getattr(som, 'params', None), 'topology_config', None)
        self.grid_size = getattr(topology_config, 'grid_size', None)
        if self.grid_size is None and self.topology is not None:
            self.grid_size = getattr(self.topology, 'grid_size', None)

        self.topology_type = getattr(topology_config, 'topology_type', None)
        if self.topology_type is None and self.topology is not None:
            topology_name = getattr(self.topology, 'name', None)
            if isinstance(topology_name, str):
                self.topology_type = topology_name.lower()

        self.adjacency_list = getattr(som, 'adjacency_list', None)
        if self.adjacency_list is None and self.topology is not None:
            self.adjacency_list = getattr(self.topology, 'adjacency_list', None)
        self.grid_node_mapping = getattr(som, 'grid_node_mapping', None)
    
    def get_weights(self):
        return self.weights
    
    def get_bmu(self, sample):
        """Get best matching unit for a single sample."""
        # FloatSOM predict returns BMU indices
        sample_batch = sample.reshape(1, -1)
        bmu_idx = self.floatsom.predict(sample_batch)[0]
        
        # Convert linear index to grid coordinates if needed
        if self.topology is not None and hasattr(self.topology, 'grid_size'):
            grid_size = self.topology.grid_size
            bmu_coords = (bmu_idx // grid_size, bmu_idx % grid_size)
            return bmu_coords
        else:
            return bmu_idx
    
    def get_bmu_distances(self, sample):
        """Get distances to all neurons for a single sample."""
        weights = self.weights
        if isinstance(sample, cp.ndarray):
            sample_np = sample.get()
        else:
            sample_np = sample
            
        if isinstance(weights, cp.ndarray):
            weights_np = weights.get()
        else:
            weights_np = weights
        
        # Compute distances to all neurons
        distances = np.linalg.norm(weights_np - sample_np, axis=-1)
        
        # Reshape to grid if applicable
        if self.topology is not None and hasattr(self.topology, 'grid_size'):
            grid_size = self.topology.grid_size
            distances = distances.reshape(grid_size, grid_size)
        
        return distances
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get adjacency matrix for the current topology (reformed or original).
        
        Returns:
            adjacency_matrix: Boolean matrix where True indicates adjacent nodes
        """
        n_neurons = self._get_total_nodes()
        adjacency_list = self.adjacency_list
        if adjacency_list is None and self.topology is not None:
            adjacency_list = getattr(self.topology, 'adjacency_list', None)

        if adjacency_list is not None:
            return self._adjacency_from_list(adjacency_list, n_neurons)

        if self._is_graph_topology():
            raise ValueError(
                f"Graph topology '{self.topology_type}' has no adjacency list (n_neurons={n_neurons})"
            )

        grid_size = self.grid_size
        if grid_size is None and self.topology is not None:
            grid_size = getattr(self.topology, 'grid_size', None)
        if grid_size is None:
            raise ValueError("Unable to determine grid_size for grid adjacency construction")

        grid_size = int(grid_size)
        if grid_size * grid_size != n_neurons:
            raise ValueError(
                f"Grid size {grid_size}x{grid_size} does not match neuron count {n_neurons}"
            )

        return self._create_default_grid_adjacency(grid_size)

    def _get_total_nodes(self) -> int:
        """Infer total node count from trained weights or topology metadata."""
        if isinstance(self.weights, (np.ndarray, cp.ndarray)):
            if self.weights.ndim >= 3:
                return int(self.weights.shape[0] * self.weights.shape[1])
            if self.weights.ndim >= 1:
                return int(self.weights.shape[0])

        if self.topology is not None and hasattr(self.topology, 'total_nodes'):
            return int(self.topology.total_nodes)

        if self.grid_size is not None:
            return int(self.grid_size) * int(self.grid_size)

        raise ValueError("Unable to determine neuron count from weights or topology")

    def _is_graph_topology(self) -> bool:
        topology_type = (self.topology_type or '').lower()
        if topology_type in {'mst', 'rng'}:
            return True
        if self.topology is not None:
            topology_name = getattr(self.topology, 'name', '')
            if isinstance(topology_name, str):
                return topology_name.lower() in {'mst', 'rng'}
        return False
    
    def _adjacency_from_list(self, adjacency_list, n_neurons: int) -> np.ndarray:
        """Convert adjacency representations (dict/list) to a dense adjacency matrix."""
        adjacency_matrix = np.zeros((n_neurons, n_neurons), dtype=bool)
        items = adjacency_list.items() if isinstance(adjacency_list, dict) else enumerate(adjacency_list)
        for node, neighbors in items:
            node_idx = int(node)
            if not (0 <= node_idx < n_neurons) or neighbors is None:
                continue
            for neighbor in neighbors:
                neighbor_idx = int(neighbor)
                if 0 <= neighbor_idx < n_neurons and neighbor_idx != node_idx:
                    adjacency_matrix[node_idx, neighbor_idx] = True
                    adjacency_matrix[neighbor_idx, node_idx] = True
        return adjacency_matrix

    def _create_default_grid_adjacency(self, grid_size: int) -> np.ndarray:
        """Create default adjacency matrix for regular grid topology."""
        n_rows = grid_size
        n_cols = grid_size
        n_neurons = n_rows * n_cols
        
        # Create grid coordinates for all neurons
        neuron_rows = np.arange(n_neurons) // n_cols
        neuron_cols = np.arange(n_neurons) % n_cols
        
        # Compute pairwise distances
        row_diff = np.abs(neuron_rows[:, np.newaxis] - neuron_rows[np.newaxis, :])
        col_diff = np.abs(neuron_cols[:, np.newaxis] - neuron_cols[np.newaxis, :])
        
        # Adjacent if both row and col differences are <= 1, but not both 0
        adjacency = (row_diff <= 1) & (col_diff <= 1) & ((row_diff != 0) | (col_diff != 0))
        
        return adjacency


def train_floatsom(data: Union[cp.ndarray, np.ndarray, np.memmap], args, metadata=None) -> Tuple[Union[FloatSOM, MiniSOMAdapter], Dict[str, Any], float]:
    """Train a FloatSOM or MiniSOM with the specified configuration."""
    
    # Get data shape
    n_samples, input_dim = data.shape

    if args.topology_type in {'mst', 'rng'}:
        node_count = args.mst_nodes if args.mst_nodes is not None else args.grid_size ** 2
        topology_summary = f"{args.topology_type} ({node_count} nodes)"
    else:
        topology_summary = f"{args.topology_type} ({args.grid_size}x{args.grid_size})"

    
    # Check if using MiniSOM
    if args.processing_method == 'minisom':
        if not MINISOM_AVAILABLE:
            raise ImportError("MiniSOM package not available. Please install it with 'pip install minisom'")
        
        # Create topology configuration for MiniSOM
        topology_kwargs = dict(
            topology_type=args.topology_type,
            topology_variant=args.topology_variant,
            grid_size=args.grid_size,
            grid_dim=2
        )
        if args.topology_type in {'mst', 'rng'}:
            topology_kwargs['num_nodes'] = args.mst_nodes
        topology_config = TopologyConfig(**topology_kwargs)
        
        # Create default configs for MiniSOM (not actually used but required for validation)
        sampling_config = SamplingConfig(method="full")
        # Use FloatSOM defaults to set a valid chunk_size for required parameter
        default_chunk_size = FloatSOMParams(defaults_profile="publication", ).processing_config.chunk_size
        processing_config = ProcessingConfig(method="minisom", chunk_size=default_chunk_size)
        
        # Create parameters for MiniSOM adapter
        params = FloatSOMParams(
            defaults_profile="publication",
            input_dim=input_dim,
            total_iterations=args.iterations,
            initial_learning_rate=args.learning_rate,
            initial_radius=args.initial_radius,
            lr_decay_type=args.lr_decay_type,
            radius_decay_type=args.radius_decay_type,
            lr_decay_factor=args.lr_decay_factor,
            radius_decay_factor=args.radius_decay_factor,
            sampling_config=sampling_config,
            processing_config=processing_config,
            topology_config=topology_config,
            convergence_threshold=args.convergence_threshold,
            min_iterations=args.min_iterations,
            verbose=args.verbose,
            use_gpu=False,  # MiniSOM is CPU-based
            seed=args.seed,
            store_history=args.save_iterations
        )
        
        # Create MiniSOM adapter
        som = MiniSOMAdapter(params, use_default_hyperparams=args.minisom_defaults)
        
        print(f"\nTraining MiniSOM...")
        print(f"Configuration: MiniSOM CPU implementation, {topology_summary}")
        
    else:
        # Create parameter configuration for FloatSOM
        sampling_config = SamplingConfig(
            method=args.sampling_method,
            random_seed=args.seed
        )
        
        # Set default normalization parameters if not provided
        norm_alpha = args.norm_alpha
        norm_clamp_factor = args.norm_clamp_factor
        norm_percentile = args.norm_percentile
        norm_max_update_threshold = args.norm_max_update_threshold
        
        if args.normalization == 'hybrid' and norm_alpha is None:
            norm_alpha = 0.5  # Balanced blend between weighted and count-based
        elif args.normalization == 'clamped_weighted' and norm_clamp_factor is None:
            norm_clamp_factor = 2.0  # Reasonable clamp factor
        elif args.normalization == 'local' and norm_percentile is None:
            norm_percentile = 85.0  # Use 85th percentile threshold
        
        # Prepare Ray configuration if Ray is requested and processing method supports it
        ray_config = None
        if args.use_ray and args.processing_method in ['batch', 'colors']:
            if not RAY_AVAILABLE:
                raise ImportError("Ray requested but not available. Please install ray: pip install ray")
            
            if args.tempdir:
                storage_path = os.path.abspath(os.path.expanduser(args.tempdir))
                os.makedirs(storage_path, exist_ok=True)
            else:
                storage_path = tempfile.mkdtemp(prefix="floatsom_ray_")
            args.tempdir = storage_path
            print(f"Using Ray temp directory: {storage_path}")

            # Build Ray config dict, only including non-None values
            ray_config_kwargs = {
                'chunk_size': args.chunk_size,
                'num_gpus': args.ray_gpu_count,
                'collective_group_name': args.ray_collective_group,
            }
            
            # Only add chunk_size if explicitly provided (not None)
            if args.ray_chunk_size is not None:
                ray_config_kwargs['chunk_size'] = args.ray_chunk_size
            
            ray_config_kwargs['storage_path'] = storage_path
            ray_config_kwargs['local_storage_path'] = storage_path
            
            ray_config = RayConfig(**ray_config_kwargs)
        
        # Calculate optimal chunk size based on shared heuristic
        if args.chunk_size is None:
            optimal_chunk_size = calculate_auto_chunk_size_for_method(
                input_dim, args.processing_method
            )
            optimal_chunk_size = min(optimal_chunk_size, 500_000)
        else:
            optimal_chunk_size = args.chunk_size
            
        if args.verbose:
            print(f"Using chunk size: {optimal_chunk_size:,} (calculated for {input_dim} dimensions)")
        
        processing_config = ProcessingConfig(
            method=args.processing_method,
            batch_mode=args.batch_mode,
            chunk_size=optimal_chunk_size,  # Pass the calculated chunk_size
            enable_momentum=args.use_momentum,
            initial_momentum=args.momentum_init,
            normalization=args.normalization,
            norm_alpha=norm_alpha,
            norm_clamp_factor=norm_clamp_factor,
            norm_percentile=norm_percentile,
            norm_max_update_threshold=norm_max_update_threshold,
            virtual_ratio=args.virtual_ratio,
            use_gpu=args.use_gpu,
            ray_config=ray_config  # Pass ray_config directly
        )
        
        topology_kwargs = dict(
            topology_type=args.topology_type,
            topology_variant=args.topology_variant,
            grid_size=args.grid_size,
            grid_dim=2,  # Always 2D for now
            mst_update_frequency=args.mst_update_frequency,
            dynamic_mst_frequency=args.dynamic_mst_frequency,
            mst_decay_function=args.mst_decay_function,
            initial_mst_frequency=args.initial_mst_frequency,
            final_mst_frequency=args.final_mst_frequency
        )
        if args.topology_type in {'mst', 'rng'}:
            topology_kwargs['num_nodes'] = args.mst_nodes
        topology_config = TopologyConfig(**topology_kwargs)
        
        params = FloatSOMParams(

            defaults_profile="publication",
            input_dim=input_dim,
            total_iterations=args.iterations,
            initial_learning_rate=args.learning_rate,
            initial_radius=args.initial_radius,
            lr_decay_type=args.lr_decay_type,
            radius_decay_type=args.radius_decay_type,
            lr_decay_factor=args.lr_decay_factor,
            radius_decay_factor=args.radius_decay_factor,
            sampling_config=sampling_config,
            processing_config=processing_config,
            topology_config=topology_config,
            convergence_threshold=args.convergence_threshold,
            min_iterations=args.min_iterations,
            verbose=args.verbose,
            use_gpu=args.use_gpu,
            seed=args.seed,
            store_history=args.save_iterations,
            initialization_method=args.initialization_method,
            reform_grid=args.reform_grid,
            reform_grid_type=args.reform_grid_type
        )
        
        # Create FloatSOM using factory with data shape
        som = create_floatsom(params)
        
    # Display configuration info
    if args.processing_method == 'minisom':
        defaults_info = "with MiniSOM defaults" if args.minisom_defaults else "with FloatSOM parameters"
        print(f"\nTraining MiniSOM {defaults_info}...")
        print(f"Configuration: MiniSOM CPU implementation, {topology_summary}")
    else:
        print(f"\nTraining FloatSOM...")
        print(f"Configuration: {args.sampling_method} sampling, {args.processing_method} processing, {topology_summary}")
    
    # Train the SOM
    start_time = time.time()
    training_stats = som.train(data)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.3f}s")
    
    return som, training_stats, train_time


def visualize_results(som: FloatSOM, data: cp.ndarray, metadata: Dict[str, Any],
                     output_dir: str, args):
    """Create visualizations of the trained SOM."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization output. "
            "Install matplotlib or run without --visualize."
        )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert data to numpy for visualization
    data_np = _to_numpy(data)
    weights = som.get_weights()
    weights_np = _to_numpy(weights)
    
    # Get BMUs for all data points
    bmu_indices = som.predict(data)
    bmu_indices_np = _to_numpy(bmu_indices)
    edges_indices, edge_color = _collect_connection_edges(som, args, weights, weights_np)
    connection_alpha = 0.5 if edge_color == 'k' else 0.7

    # 1. Plot SOM grid overlay on data (similar to parallel_som visualization)
    if data_np.shape[1] == 2:
        def render_2d(show_connections: bool, filename: str):
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.scatter(data_np[:, 0], data_np[:, 1], s=10, alpha=0.5, c='gray', label='Data points')

            if show_connections and edges_indices:
                _draw_edges_2d(ax, weights_np[:, :2], edges_indices, edge_color, connection_alpha)

            ax.scatter(weights_np[:, 0], weights_np[:, 1], s=50, c='red', marker='o',
                       edgecolors='darkred', linewidths=0.5, label='SOM nodes', zorder=5)

            ax.set_title(f'FloatSOM - {metadata["dataset_name"]} - SOM Grid Overlay')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)

        render_2d(True, 'som_grid_overlay.png')
        render_2d(False, 'som_grid_overlay_nodes_only.png')

    elif data_np.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def render_3d(show_connections: bool, filename: str):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], s=10, alpha=0.3, c='gray', label='Data points')

            if show_connections and edges_indices:
                _draw_edges_3d(ax, weights_np[:, :3], edges_indices, edge_color, connection_alpha)

            ax.scatter(weights_np[:, 0], weights_np[:, 1], weights_np[:, 2],
                      s=50, c='red', marker='o', edgecolors='darkred', linewidths=0.5,
                      label='SOM nodes')

            ax.set_title(f'FloatSOM - {metadata["dataset_name"]} - SOM Grid Overlay')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.legend()
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)

        render_3d(True, 'som_grid_overlay.png')
        render_3d(False, 'som_grid_overlay_nodes_only.png')

    else:
        from sklearn.decomposition import PCA
        print(f"Data has {data_np.shape[1]} dimensions, using PCA for 2D visualization")

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_np)
        weights_pca = pca.transform(weights_np)

        def render_pca(show_connections: bool, filename: str):
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.scatter(data_pca[:, 0], data_pca[:, 1], s=10, alpha=0.5, c='gray', label='Data points (PCA)')

            if show_connections and edges_indices:
                _draw_edges_2d(ax, weights_pca, edges_indices, edge_color, connection_alpha)

            ax.scatter(weights_pca[:, 0], weights_pca[:, 1], s=50, c='red', marker='o',
                       edgecolors='darkred', linewidths=0.5, label='SOM nodes (PCA)', zorder=5)

            ax.set_title(f'FloatSOM - {metadata["dataset_name"]} - SOM Grid Overlay (PCA projection)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)

        render_pca(True, 'som_grid_overlay.png')
        render_pca(False, 'som_grid_overlay_nodes_only.png')

    # 2. Plot U-matrix (distance matrix)
    if args.topology_type in ['grid', 'hexagonal']:
        grid_size = som.topology.grid_size
        u_matrix = np.zeros((grid_size, grid_size))

        # Calculate U-matrix values
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                neighbors = []

                # Get neighbors based on topology type
                if args.topology_type == 'hexagonal':
                    # Hexagonal has 6 neighbors
                    # For even rows (i % 2 == 0), neighbors are offset differently than odd rows
                    if i > 0:  # top
                        neighbors.append((i-1) * grid_size + j)
                    if i < grid_size - 1:  # bottom
                        neighbors.append((i+1) * grid_size + j)
                    if j > 0:  # left
                        neighbors.append(i * grid_size + (j-1))
                    if j < grid_size - 1:  # right
                        neighbors.append(i * grid_size + (j+1))

                    # Additional diagonal neighbors for hexagonal
                    if i % 2 == 0:  # even row
                        if i > 0 and j > 0:  # top-left
                            neighbors.append((i-1) * grid_size + (j-1))
                        if i < grid_size - 1 and j > 0:  # bottom-left
                            neighbors.append((i+1) * grid_size + (j-1))
                    else:  # odd row
                        if i > 0 and j < grid_size - 1:  # top-right
                            neighbors.append((i-1) * grid_size + (j+1))
                        if i < grid_size - 1 and j < grid_size - 1:  # bottom-right
                            neighbors.append((i+1) * grid_size + (j+1))
                else:
                    # Grid topology - 4 neighbors
                    if i > 0:  # top
                        neighbors.append((i-1) * grid_size + j)
                    if i < grid_size - 1:  # bottom
                        neighbors.append((i+1) * grid_size + j)
                    if j > 0:  # left
                        neighbors.append(i * grid_size + (j-1))
                    if j < grid_size - 1:  # right
                        neighbors.append(i * grid_size + (j+1))

                # Calculate average distance to neighbors
                if neighbors:
                    distances = [np.linalg.norm(weights_np[idx] - weights_np[n]) for n in neighbors]
                    u_matrix[i, j] = np.mean(distances)

        plt.figure(figsize=(8, 6))
        plt.imshow(u_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Average distance to neighbors')
        plt.title(f'U-Matrix - {metadata["dataset_name"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'u_matrix.png'))
        plt.close()

    # 3. Plot BMU hit map
    plt.figure(figsize=(8, 6))
    unique_bmus, counts = np.unique(bmu_indices_np, return_counts=True)

    if args.topology_type in ['grid', 'hexagonal']:
        grid_size = som.topology.grid_size
        hit_map = np.zeros((grid_size, grid_size))

        for bmu, count in zip(unique_bmus, counts):
            i = bmu // grid_size
            j = bmu % grid_size
            hit_map[i, j] = count

        plt.imshow(hit_map, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Number of samples')
        plt.title(f'BMU Hit Map - {metadata["dataset_name"]}')
    else:
        plt.bar(unique_bmus, counts)
        plt.xlabel('BMU Index')
        plt.ylabel('Number of samples')
        plt.title(f'BMU Distribution - {metadata["dataset_name"]}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bmu_distribution.png'))
    plt.close()

    # 4. Plot training history if available
    if args.save_iterations and hasattr(som, 'training_history') and som.training_history:
        history = som.training_history
        iterations = [h['iteration'] for h in history]
        weight_changes = [h['weight_change'] for h in history]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, weight_changes, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Weight Change')
        plt.title('Training Progress - Weight Changes')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_progress.png'))
        plt.close()


def write_results_report(som: FloatSOM, training_stats: Dict[str, Any], 
                        train_time: float, topology_metrics: Dict[str, float],
                        metadata: Dict[str, Any], args, output_file: str):
    """Write comprehensive results report to file."""
    
    with open(output_file, 'w') as f:
        f.write(f"FloatSOM Benchmark Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset information
        f.write("Dataset Information:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {metadata['dataset_name']}\n")
        f.write(f"Shape: {metadata['shape']}\n")
        f.write(f"Intrinsic dimension: {metadata.get('intrinsic_dim', 'N/A')}\n")
        f.write(f"Embedding dimension: {metadata.get('embedding_dim', 'N/A')}\n")
        if 'n_clusters' in metadata:
            f.write(f"Number of clusters: {metadata['n_clusters']}\n")
        f.write("\n")
        
        # Complete SOM Configuration
        f.write("Complete SOM Configuration:\n")
        f.write("-" * 40 + "\n")
        if hasattr(som, 'params'):
            params = som.params
            f.write(f"Input dimension: {params.input_dim}\n")
            f.write(f"Total iterations: {params.total_iterations}\n")
            f.write(f"Initial learning rate: {params.initial_learning_rate}\n")
            f.write(f"Initial radius: {params.initial_radius}\n")
            f.write(f"LR decay type: {params.lr_decay_type}\n")
            f.write(f"LR decay factor: {params.lr_decay_factor}\n")
            f.write(f"Radius decay type: {params.radius_decay_type}\n")
            f.write(f"Radius decay factor: {params.radius_decay_factor}\n")
            f.write(f"Convergence threshold: {params.convergence_threshold}\n")
            f.write(f"Min iterations: {params.min_iterations}\n")
            f.write(f"Verbose: {params.verbose}\n")
            f.write(f"Use GPU: {params.use_gpu}\n")
            f.write(f"Seed: {params.seed}\n")
            f.write(f"Store history: {params.store_history}\n")
            f.write(f"Initialization method: {params.initialization_method}\n")
            
            if params.sampling_config:
                f.write(f"\nSampling Configuration:\n")
                f.write(f"  Method: {params.sampling_config.method}\n")
                f.write(f"  Samples per epoch: {params.sampling_config.samples_per_epoch}\n")
                f.write(f"  Random seed: {params.sampling_config.random_seed}\n")
            
            if params.processing_config:
                f.write(f"\nProcessing Configuration:\n")
                f.write(f"  Method: {params.processing_config.method}\n")
                f.write(f"  Batch mode: {params.processing_config.batch_mode}\n")
                f.write(f"  Chunk size: {params.processing_config.chunk_size}\n")
                f.write(f"  Enable momentum: {params.processing_config.enable_momentum}\n")
                f.write(f"  Initial momentum: {params.processing_config.initial_momentum}\n")
                f.write(f"  Normalization: {params.processing_config.normalization}\n")
                f.write(f"  Use GPU: {params.processing_config.use_gpu}\n")
                
                # Add Ray configuration if present
                if params.processing_config.ray_config:
                    ray_config = params.processing_config.ray_config
                    f.write(f"\nRay Multi-GPU Configuration:\n")
                    f.write(f"  Ray enabled: True\n")
                    f.write(f"  Number of GPUs: {ray_config.num_gpus or 'auto-detect'}\n")
                    f.write(f"  Collective group: {ray_config.collective_group_name}\n")
                    f.write(f"  Internal chunk size: {ray_config.chunk_size}\n")
            
            if params.topology_config:
                f.write(f"\nTopology Configuration:\n")
                f.write(f"  Topology type: {params.topology_config.topology_type}\n")
                f.write(f"  Topology variant: {params.topology_config.topology_variant}\n")
                f.write(f"  Grid size: {params.topology_config.grid_size}\n")
                f.write(f"  Grid dimension: {params.topology_config.grid_dim}\n")
        else:
            # Fallback to args-based configuration for compatibility
            if args.topology_type in {'mst', 'rng'}:
                node_count = args.mst_nodes if args.mst_nodes is not None else args.grid_size ** 2
                f.write(f"Topology: {args.topology_type} ({node_count} nodes)\n")
            else:
                f.write(f"Topology: {args.topology_type} ({args.topology_variant})\n")
                f.write(f"Grid size: {args.grid_size}\n")
            f.write(f"Sampling method: {args.sampling_method}\n")
            f.write(f"Processing method: {args.processing_method}\n")
            if args.processing_method == 'batch':
                f.write(f"Batch mode: {args.batch_mode}\n")
                f.write(f"Chunk size: {args.chunk_size}\n")
            f.write(f"Iterations: {args.iterations}\n")
            f.write(f"Initial learning rate: {args.learning_rate}\n")
            f.write(f"Initial radius: {args.initial_radius or 'auto'}\n")
            f.write(f"LR decay type: {args.lr_decay_type}\n")
            f.write(f"LR decay factor: {args.lr_decay_factor}\n")
            f.write(f"Radius decay type: {args.radius_decay_type}\n")
            f.write(f"Radius decay factor: {args.radius_decay_factor}\n")
            f.write(f"Normalization: {args.normalization}\n")
            f.write(f"Momentum: {'enabled' if args.use_momentum else 'disabled'}\n")
            if args.use_momentum:
                f.write(f"Initial momentum: {args.momentum_init}\n")
            f.write(f"GPU: {'enabled' if args.use_gpu else 'disabled'}\n")
            if args.use_ray:
                f.write(f"Ray multi-GPU: enabled\n")
                f.write(f"Ray GPUs: {args.ray_gpu_count or 'auto-detect'}\n")
                f.write(f"Ray collective group: {args.ray_collective_group}\n")
        f.write("\n")
        
        # Training results
        f.write("Training Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training time: {train_time:.3f}s\n")
        f.write(f"Iterations completed: {training_stats['iterations_completed']}\n")
        f.write(f"Total samples processed: {training_stats['total_samples_processed']}\n")
        f.write(f"Convergence detected: {training_stats.get('convergence_detected', False)}\n")
        f.write(f"Final weights norm: {training_stats['final_weights_norm']:.6f}\n")
        f.write("\n")
        
        # Topology metrics
        if topology_metrics:
            f.write("Topology Metrics:\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in sorted(topology_metrics.items()):
                f.write(f"{metric_name}: {value:.6f}\n")
            f.write("\n")
        
        # Performance summary
        f.write("Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Samples per second: {training_stats['total_samples_processed'] / train_time:.2f}\n")
        f.write(f"Iterations per second: {training_stats['iterations_completed'] / train_time:.2f}\n")
        avg_time_per_iter = train_time / training_stats['iterations_completed']
        f.write(f"Average time per iteration: {avg_time_per_iter * 1000:.2f}ms\n")


def main():
    """Main function for running FloatSOM benchmarks"""
    args = parse_args()
    
    # Show available sklearn datasets if requested
    if args.data_type.startswith('sklearn_') and args.verbose:
        available = get_available_datasets()
        print("Available sklearn datasets:")
        for name, difficulties in available.items():
            print(f"  {name}: {difficulties}")
        print()
    
    # Generate data
    data, metadata = generate_data(args)
    
    # Make sure data is not empty
    if data.size == 0:
        print("Error: Data is empty. Cannot proceed with benchmarking.")
        return
    
    # Setup profiling if requested
    profiler_instance = None
    if args.profile:
        print(f"Profiling enabled. Results will be saved to: {args.profile_output}")
        profiler_instance = cProfile.Profile()
        profiler_instance.enable()
    
    # Create output directory
    output_dir = f"floatsom_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Train FloatSOM
    som, training_stats, train_time = train_floatsom(data, args, metadata)
    
    # Cleanup Ray resources if used
    _cleanup_ray_resources_if_needed(som, args)
    
    # Compute topology metrics
    topology_metrics = compute_topology_metrics(som, data, args)
    
    # Create visualizations
    if args.verbose:
        print("\nCreating visualizations...")
    if args.visualize:
        visualize_results(som, data, metadata, output_dir, args)
    representative_svg = None
    if args.representative_topology_column:
        pre_trained_soms: Dict[str, Union[FloatSOM, MiniSOMAdapter]] = {}
        if args.topology_type in REPRESENTATIVE_TOPOLOGY_ORDER:
            pre_trained_soms[args.topology_type] = som
        representative_svg = generate_representative_topology_column_figure(
            data=data,
            metadata=metadata,
            args=args,
            output_dir=output_dir,
            pre_trained_soms=pre_trained_soms,
        )
    
    # Write results report
    output_file = os.path.join(output_dir, args.output_file)
    write_results_report(som, training_stats, train_time, topology_metrics, 
                        metadata, args, output_file)
    
    # Handle profiling output
    if args.profile and profiler_instance:
        profiler_instance.disable()
        
        # Save profile results
        profile_output = os.path.join(output_dir, args.profile_output)
        with open(profile_output, 'w') as f:
            s = io.StringIO()
            ps = pstats.Stats(profiler_instance, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            f.write(s.getvalue())
        
        print(f"\nProfiling results saved to: {profile_output}")
    
    print(f"\nBenchmark complete!")
    print(f"Results saved to: {output_file}")
    print(f"Visualizations saved to: {output_dir}/")
    print(f"  - som_grid_overlay.png: Shows SOM nodes overlaid on data points")
    print(f"  - u_matrix.png: Shows distances between neighboring nodes")
    print(f"  - bmu_distribution.png: Shows sample distribution across nodes")
    if representative_svg is not None:
        print(f"  - {representative_svg}: One-column Hex/MST/RNG representative topology SVG")
    if args.save_iterations:
        print(f"  - training_progress.png: Shows weight changes during training")
    
    # Print summary
    print("\nSummary:")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Training time: {train_time:.3f}s")
    if 'quantization_error' in topology_metrics:
        print(f"  Quantization error: {topology_metrics['quantization_error']:.6f}")
    if args.enable_topology_metrics:
        print(f"  Topographic error: {topology_metrics.get('topographic_error', 'N/A')}")
        print(f"  Trustworthiness: {topology_metrics.get('trustworthiness', 'N/A')}")


if __name__ == "__main__":
    # Set number of threads to avoid OMP conflicts
    os.environ["OMP_NUM_THREADS"] = "1"
    
    main()
