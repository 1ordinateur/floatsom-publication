#!/usr/bin/env python3
"""
Harmonize GPU scaling benchmark results produced in separate run directories.

Example:
  python -m floatsom.benchmarks.speed_benchmarks.harmonize_gpu_scaling_results \
    --inputs /g/data/eu59/SIFEAN/sfa \
    --output_dir /g/data/eu59/SIFEAN/sfa/harmonized_gpu_scaling

The script discovers result directories (e.g. ``1gpu_scaling/results`` or
``1gpu_scaling/results_20260321``),
merges per-mode JSON artifacts, rewrites harmonized CSV/JSON/analysis files,
and regenerates harmonized figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from floatsom.benchmarks.speed_benchmarks.gpu_scaling.results import ResultsHandler

NumericKey = Union[int, float]
StatsDict = Dict[str, Union[float, int, List[float]]]
AxisResults = Dict[NumericKey, Dict[int, StatsDict]]
MergedModeResults = Dict[Tuple[str, str], AxisResults]
RandomComparisonModeResults = Dict[Tuple[str, str, str], AxisResults]
RESULTS_DIR_PATTERN = re.compile(r"^results_(\d{8})(?:_([A-Za-z0-9]+))?$")


MODE_CONFIG: Dict[str, Dict[str, str]] = {
    "dimension_scaling": {
        "prefix": "dimension_scaling",
        "save_mode": "dimension",
        "plot_type": "gpu",
        "title": "Harmonized Dimension Scaling",
    },
    "sample_scaling": {
        "prefix": "sample_scaling",
        "save_mode": "sample",
        "plot_type": "sample",
        "title": "Harmonized Sample Scaling",
    },
    "grid_size_scaling": {
        "prefix": "grid_size_scaling",
        "save_mode": "grid_size",
        "plot_type": "gpu",
        "title": "Harmonized Grid Size Scaling",
    },
}

PRIMARY_METHODS: Tuple[str, ...] = ("batch",)
PUBLICATION_METHOD_ORDER: Tuple[str, ...] = PRIMARY_METHODS
PUBLICATION_MODE_ORDER: Tuple[str, ...] = ("sample_scaling", "dimension_scaling", "grid_size_scaling")
PUBLICATION_MODE_LABELS: Dict[str, str] = {
    "sample_scaling": "Sample Scaling",
    "dimension_scaling": "Dimension Scaling",
    "grid_size_scaling": "Grid-Size Scaling",
}
PUBLICATION_METHOD_COLORS: Dict[str, str] = {
    "batch": "#00C853",
}
PUBLICATION_VISIBLE_GPU_COUNTS: Tuple[int, ...] = (1, 2, 3, 4, 8)
PUBLICATION_SAMPLE_AXIS_MIN_UPPER_BOUND = 1_000_000_000
PUBLICATION_DIMENSION_AXIS_MIN_UPPER_BOUND = 5000
RANDOM_COMPARISON_METHODS: Tuple[str, ...] = ("batch",)
RANDOM_COMPARISON_RUN_TYPES: Tuple[str, ...] = ("full", "random")
RANDOM_COMPARISON_COLORS: Dict[str, str] = {
    "hexagonal": "#FF4FA3",
    "mst": "#00E5FF",
    "rng": "#8A2BE2",
}
FIGURE_8_TRIPANEL_GPU_COUNTS: Tuple[int, ...] = (1, 2, 4)
PUBLICATION_FIGURE_ORDER: Tuple[str, ...] = ("figure_8", "figure_9", "figure_10")
PUBLICATION_PANEL_MODE_ORDER: Tuple[str, ...] = (
    "dimension_scaling",
    "sample_scaling",
    "grid_size_scaling",
)
PUBLICATION_COMPOSITE_LEGEND_BAND_HEIGHT = 180
PUBLICATION_COMPOSITE_LEGEND_SCALE = 1.12
PUBLICATION_LEGEND_ROW_HEIGHT = 35
PUBLICATION_LEGEND_GAP = 15
PUBLICATION_FIGURE_9_LEGEND_GAP = 26
PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_GAP = 26
PUBLICATION_ONE_ROW_LEGEND_BAND_HEIGHT = 1 * PUBLICATION_LEGEND_ROW_HEIGHT
PUBLICATION_TWO_ROW_LEGEND_BAND_HEIGHT = 2 * PUBLICATION_LEGEND_ROW_HEIGHT
PUBLICATION_THREE_ROW_LEGEND_BAND_HEIGHT = 3 * PUBLICATION_LEGEND_ROW_HEIGHT
PUBLICATION_TWO_ROW_PLUS_SHADE_LEGEND_BAND_HEIGHT = PUBLICATION_THREE_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_COMPOSITE_PANEL_LABEL_FONT_SIZE = 30
PUBLICATION_COMPOSITE_PANEL_CAPTION_FONT_SIZE = 24
PUBLICATION_PANEL_IMAGE_Y_OFFSET = 0
PUBLICATION_COMPOSITE_TITLE_FONT_SIZE = 36
PUBLICATION_FIGURE_9_TOPOLOGY = "rng"
PUBLICATION_FIGURE_8_PANEL_WIDTH = 700
PUBLICATION_FIGURE_8_PANEL_HEIGHT = 465
PUBLICATION_FIGURE_8_TITLE_FONT_SIZE = PUBLICATION_COMPOSITE_TITLE_FONT_SIZE
PUBLICATION_FIGURE_8_TITLE_BAND = 60
PUBLICATION_FIGURE_8_LEGEND_BAND_HEIGHT = PUBLICATION_THREE_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_FIGURE_8_LEGEND_SCALE = 1.0
PUBLICATION_FIGURE_9_TITLE_FONT_SIZE = PUBLICATION_COMPOSITE_TITLE_FONT_SIZE
PUBLICATION_FIGURE_9_PANEL_LABEL_FONT_SIZE = PUBLICATION_COMPOSITE_PANEL_LABEL_FONT_SIZE
PUBLICATION_FIGURE_9_PANEL_CAPTION_FONT_SIZE = PUBLICATION_COMPOSITE_PANEL_CAPTION_FONT_SIZE
PUBLICATION_FIGURE_9_LEGEND_BAND_HEIGHT = 2 * PUBLICATION_TWO_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_FIGURE_9_LEGEND_SCALE = 1.0
PUBLICATION_FIGURE_10_LEGEND_BAND_HEIGHT = PUBLICATION_TWO_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_FIGURE_10_LEGEND_SCALE = PUBLICATION_COMPOSITE_LEGEND_SCALE
PUBLICATION_SUPPLEMENTARY_TOPOLOGY_COMPARISON_LEGEND_BAND_HEIGHT = PUBLICATION_TWO_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_SUPPLEMENTARY_TOPOLOGY_COMPARISON_LEGEND_SCALE = PUBLICATION_COMPOSITE_LEGEND_SCALE
PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_BAND_HEIGHT = PUBLICATION_TWO_ROW_LEGEND_BAND_HEIGHT
PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_SCALE = PUBLICATION_FIGURE_9_LEGEND_SCALE
PUBLICATION_PAPER_ASSET_FIGURE_MAP: Dict[str, str] = {
    "figure_8_sampling_speed_random_over_full.svg": "fig_10.svg",
    "figure_9_algorithm_first_gpu_scaling.svg": "fig_11.svg",
    "figure_10_topology_mst_hex_performance.svg": "fig_12.svg",
    "supplementary_figure_s12_mst_hex_gpu_scaling.svg": "supp_fig_s11.svg",
}
PUBLICATION_PAPER_ASSET_EXTRA_COPIES: Dict[str, Tuple[str, ...]] = {}
PUBLICATION_PAPER_ASSET_TABLE_MAP: Dict[str, str] = {
    "supp_table_figure_10_topology_runtime_summary.tsv": "supp_table_figure_12_topology_runtime_summary.tsv",
    "supp_table_figure_9_rng_scaling_diagnostics.tsv": "supp_table_figure_11_rng_scaling_diagnostics.tsv",
}
SCALING_LOG_CONFIG_PATTERN = re.compile(
    r"^(?P<topology>[A-Za-z0-9_]+)"
    r"_dim(?P<dimension>\d+)"
    r"_samples(?P<samples>\d+)"
    r"_grid(?P<grid_size>\d+)"
    r"_gpu(?P<gpu_count>\d+)"
    r"_(?P<method>[A-Za-z0-9_]+)"
    r"_seed(?P<seed>\d+)$"
)

# CLI configuration ---------------------------------------------------------


def _has_method(results: Dict[str, object], method_name: str) -> bool:
    """Return True when a method exists at the top level (case-insensitive)."""
    target = str(method_name).strip().lower()
    return any(str(method).strip().lower() == target for method in results.keys())


def _resolve_primary_methods(colors_enabled: bool) -> Tuple[str, ...]:
    """
    Resolve the set/order of methods to plot.

    Default behavior keeps scaling figures batch-only.
    Enabling colors adds the colors processor traces without changing inputs.
    """
    if not colors_enabled:
        return PRIMARY_METHODS
    return ("batch", "colors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge separate GPU scaling benchmark result folders into one harmonized result set."
    )
    parser.add_argument(
        "--colors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the colors processor in scaling/comparison figures (default: false).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help=(
            "Input paths. Each path may be a results directory, a GPU-scaling parent "
            "directory containing `results` or `results_*` children, or a root "
            "containing `*gpu_scaling*/results*`."
        ),
    )
    parser.add_argument(
        "--search_root",
        type=str,
        default=None,
        help="Optional root directory to scan for GPU-scaling directories and their `results*` children.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for harmonized outputs.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation and only emit harmonized CSV/JSON/analysis artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--topology_comparison",
        action="store_true",
        help="Generate cross-topology comparison figures at a fixed GPU count.",
    )
    parser.add_argument(
        "--comparison_gpu_count",
        type=int,
        default=8,
        help="GPU count to use for topology-comparison figures (default: 8).",
    )
    parser.add_argument(
        "--comparison_output_subdir",
        type=str,
        default="topology_comparison_8gpu",
        help="Subdirectory under --output_dir for topology-comparison figures.",
    )
    parser.add_argument(
        "--random_comparison",
        action="store_true",
        help="Generate random and full comparison figures for batch runs across topologies.",
    )
    parser.add_argument(
        "--random_comparison_gpu_count",
        type=int,
        default=1,
        help="GPU count to use for random and full comparison figures (default: 1).",
    )
    parser.add_argument(
        "--random_path_pattern",
        type=str,
        default="random",
        help=(
            "Case-insensitive substring that marks a source directory as a random run "
            "(default: 'random')."
        ),
    )
    parser.add_argument(
        "--random_comparison_output_subdir",
        type=str,
        default="random_comparison_1gpu",
        help="Subdirectory under --output_dir for random and full comparison figures.",
    )
    parser.add_argument(
        "--figure8-random-1gpu-dir",
        type=str,
        default=None,
        help="Directory or results path for the 1-GPU random/full sample-scaling panel used in Figure 8.",
    )
    parser.add_argument(
        "--figure8-random-2gpu-dir",
        type=str,
        default=None,
        help="Directory or results path for the 2-GPU random/full sample-scaling panel used in Figure 8.",
    )
    parser.add_argument(
        "--figure8-random-4gpu-dir",
        type=str,
        default=None,
        help="Directory or results path for the 4-GPU random/full sample-scaling panel used in Figure 8.",
    )
    return parser.parse_args()


# Input discovery -----------------------------------------------------------


def _is_results_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for mode_name in MODE_CONFIG:
        if (path / mode_name).is_dir():
            return True
    return False


def _normalize_input_candidate(path: Path, discovered: Set[Path]) -> None:
    if _is_results_dir(path):
        discovered.add(path.resolve())
        return

    latest_versioned_results = _find_latest_versioned_results_dir(path)
    if latest_versioned_results is not None:
        discovered.add(latest_versioned_results.resolve())
        return

    nested_results = path / "results"
    if _is_results_dir(nested_results):
        discovered.add(nested_results.resolve())
        return

    for gpu_dir in sorted(path.glob("*gpu_scaling*")):
        if not gpu_dir.is_dir():
            continue

        latest_versioned_results = _find_latest_versioned_results_dir(gpu_dir)
        if latest_versioned_results is not None:
            discovered.add(latest_versioned_results.resolve())
            continue

        nested_results = gpu_dir / "results"
        if _is_results_dir(nested_results):
            discovered.add(nested_results.resolve())


def _parse_results_dir_date(path: Path) -> Optional[datetime]:
    match = RESULTS_DIR_PATTERN.match(path.name)
    if match is None:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def _find_latest_versioned_results_dir(path: Path) -> Optional[Path]:
    if not path.is_dir():
        return None

    candidates: List[Tuple[datetime, float, str, Path]] = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_dir():
            continue
        parsed_date = _parse_results_dir_date(candidate)
        if parsed_date is None or not _is_results_dir(candidate):
            continue
        try:
            modified_at = candidate.stat().st_mtime
        except OSError:
            modified_at = 0.0
        candidates.append((parsed_date, modified_at, candidate.name, candidate))

    if not candidates:
        return None

    _, _, _, selected = max(candidates)
    return selected


def discover_results_dirs(inputs: Iterable[str], search_root: Optional[str]) -> List[Path]:
    discovered: Set[Path] = set()

    for raw_path in inputs:
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            _normalize_input_candidate(candidate, discovered)

    if search_root:
        root = Path(search_root).expanduser()
        if root.exists():
            _normalize_input_candidate(root, discovered)

    return sorted(discovered)


def _resolve_optional_input_dirs(
    raw_path: Optional[str],
    *,
    warnings: List[str],
    label: str,
) -> List[Path]:
    if raw_path is None or not str(raw_path).strip():
        return []

    candidate = Path(raw_path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} path does not exist: {candidate}")

    resolved = discover_results_dirs([str(candidate)], None)
    if not resolved:
        raise ValueError(
            f"Could not resolve {label} into any harmonizable results directories: {candidate}"
        )
    return resolved


def _resolve_figure_8_tripanel_results_dirs(
    args: argparse.Namespace,
    *,
    warnings: List[str],
) -> Dict[int, List[Path]]:
    option_map = {
        1: getattr(args, "figure8_random_1gpu_dir", None),
        2: getattr(args, "figure8_random_2gpu_dir", None),
        4: getattr(args, "figure8_random_4gpu_dir", None),
    }
    provided_gpu_counts = [
        gpu_count
        for gpu_count, raw_path in option_map.items()
        if raw_path is not None and str(raw_path).strip()
    ]
    if provided_gpu_counts and len(provided_gpu_counts) != len(FIGURE_8_TRIPANEL_GPU_COUNTS):
        missing_gpu_counts = [
            gpu_count
            for gpu_count in FIGURE_8_TRIPANEL_GPU_COUNTS
            if gpu_count not in provided_gpu_counts
        ]
        raise ValueError(
            "Figure 8 generation requires all dedicated random/full input directories. "
            f"Provided GPU counts: {sorted(provided_gpu_counts)}; missing GPU counts: {missing_gpu_counts}."
        )

    resolved: Dict[int, List[Path]] = {}
    for gpu_count, raw_path in option_map.items():
        paths = _resolve_optional_input_dirs(
            raw_path,
            warnings=warnings,
            label=f"Figure 8 {gpu_count}-GPU input",
        )
        if paths:
            resolved[gpu_count] = paths
    return resolved


# Result normalization and merging -----------------------------------------


def _infer_gpu_count_from_path(path: Path) -> Optional[int]:
    """Infer GPU count from path components like `4gpu_scaling/...`."""
    pattern = re.compile(r"(?<!\d)(\d+)\s*gpu", re.IGNORECASE)
    for part in reversed(path.parts):
        match = pattern.search(part)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _coerce_numeric_key(raw_key: Union[str, int, float]) -> Optional[NumericKey]:
    if isinstance(raw_key, bool):
        return None
    if isinstance(raw_key, int):
        return raw_key
    if isinstance(raw_key, float):
        if math.isnan(raw_key) or math.isinf(raw_key):
            return None
        return int(raw_key) if raw_key.is_integer() else raw_key
    if isinstance(raw_key, str):
        stripped = raw_key.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return int(numeric) if numeric.is_integer() else numeric
    return None


def _coerce_gpu_key(raw_key: Union[str, int, float]) -> Optional[int]:
    numeric = _coerce_numeric_key(raw_key)
    if numeric is None:
        return None
    if isinstance(numeric, float):
        if not numeric.is_integer():
            return None
        numeric = int(numeric)
    if numeric <= 0:
        return None
    return int(numeric)


def _normalize_stats(stats: MutableMapping[str, object]) -> StatsDict:
    times_raw = stats.get("times", [])
    times: List[float] = []
    if isinstance(times_raw, list):
        for value in times_raw:
            try:
                times.append(float(value))
            except (TypeError, ValueError):
                continue

    mean_raw = stats.get("mean", None)
    std_raw = stats.get("std", None)
    count_raw = stats.get("count", None)

    if mean_raw is None:
        mean_val = float(np.mean(times)) if times else 0.0
    else:
        try:
            mean_val = float(mean_raw)
        except (TypeError, ValueError):
            mean_val = float(np.mean(times)) if times else 0.0

    if std_raw is None:
        std_val = float(np.std(times)) if times else 0.0
    else:
        try:
            std_val = float(std_raw)
        except (TypeError, ValueError):
            std_val = float(np.std(times)) if times else 0.0

    if count_raw is None:
        count_val = len(times)
    else:
        try:
            count_val = int(count_raw)
        except (TypeError, ValueError):
            count_val = len(times)

    if count_val < 0:
        count_val = 0
    if count_val == 0 and times:
        count_val = len(times)

    return {
        "times": times,
        "mean": mean_val,
        "std": std_val,
        "count": count_val,
    }


def _load_mode_json(path: Path, warnings: List[str]) -> AxisResults:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        warnings.append(f"Failed to read JSON {path}: {exc}")
        return {}

    if not isinstance(raw, dict):
        warnings.append(f"Ignoring non-dict JSON payload in {path}")
        return {}

    parsed: AxisResults = {}
    for axis_key, axis_payload in raw.items():
        axis = _coerce_numeric_key(axis_key)
        if axis is None:
            warnings.append(f"Skipping non-numeric axis key `{axis_key}` in {path}")
            continue
        if not isinstance(axis_payload, dict):
            continue

        parsed.setdefault(axis, {})
        for gpu_key, stats_payload in axis_payload.items():
            gpu = _coerce_gpu_key(gpu_key)
            if gpu is None:
                warnings.append(f"Skipping invalid GPU key `{gpu_key}` in {path}")
                continue
            if not isinstance(stats_payload, dict):
                continue
            parsed[axis][gpu] = _normalize_stats(stats_payload)

    return parsed


def _combine_stats(existing: StatsDict, incoming: StatsDict) -> StatsDict:
    a_count = int(existing.get("count", 0))
    b_count = int(incoming.get("count", 0))

    a_times = [float(value) for value in existing.get("times", [])]
    b_times = [float(value) for value in incoming.get("times", [])]
    merged_times = a_times + b_times

    if merged_times and len(merged_times) == (a_count + b_count):
        return {
            "times": merged_times,
            "mean": float(np.mean(merged_times)),
            "std": float(np.std(merged_times)),
            "count": len(merged_times),
        }

    if a_count <= 0:
        return deepcopy(incoming)
    if b_count <= 0:
        merged = deepcopy(existing)
        if merged_times:
            merged["times"] = merged_times
        return merged

    a_mean = float(existing.get("mean", 0.0))
    b_mean = float(incoming.get("mean", 0.0))
    a_std = float(existing.get("std", 0.0))
    b_std = float(incoming.get("std", 0.0))

    total = a_count + b_count
    mean = ((a_mean * a_count) + (b_mean * b_count)) / total

    a_var = a_std * a_std
    b_var = b_std * b_std
    total_var = (
        a_count * (a_var + (a_mean - mean) ** 2)
        + b_count * (b_var + (b_mean - mean) ** 2)
    ) / total
    std = math.sqrt(max(total_var, 0.0))

    return {
        "times": merged_times,
        "mean": float(mean),
        "std": float(std),
        "count": int(total),
    }


def _merge_axis_results(target: AxisResults, incoming: AxisResults) -> None:
    for axis_key, axis_payload in incoming.items():
        target.setdefault(axis_key, {})
        for gpu_count, incoming_stats in axis_payload.items():
            if gpu_count in target[axis_key]:
                target[axis_key][gpu_count] = _combine_stats(
                    target[axis_key][gpu_count],
                    incoming_stats,
                )
            else:
                target[axis_key][gpu_count] = deepcopy(incoming_stats)


def _filter_axis_results_gpu_counts(
    axis_results: AxisResults,
    allowed_gpu_counts: Sequence[int],
) -> AxisResults:
    allowed = {int(value) for value in allowed_gpu_counts if int(value) > 0}
    if not allowed:
        return deepcopy(axis_results)

    filtered: AxisResults = {}
    for axis_key, axis_payload in axis_results.items():
        if not isinstance(axis_payload, dict):
            continue
        filtered_axis_payload: Dict[int, StatsDict] = {}
        for raw_gpu_key, stats_payload in axis_payload.items():
            parsed_gpu = _coerce_gpu_key(raw_gpu_key)  # type: ignore[arg-type]
            if parsed_gpu is None or parsed_gpu not in allowed:
                continue
            if not isinstance(stats_payload, dict):
                continue
            filtered_axis_payload[int(parsed_gpu)] = deepcopy(stats_payload)
        if filtered_axis_payload:
            filtered[axis_key] = filtered_axis_payload
    return filtered


def _parse_topology_and_method(path: Path, prefix: str) -> Optional[Tuple[str, str]]:
    stem = path.stem
    base = f"{prefix}_results"
    if not stem.startswith(f"{base}_"):
        return None

    suffix = stem[len(base) + 1 :]
    if "_" not in suffix:
        return None

    known_topologies = {"hex", "hexagonal", "mst", "rng", "grid"}
    known_methods = {"batch", "colors", "minibatch", "minisom"}

    parts = [chunk for chunk in suffix.split("_") if chunk]
    if len(parts) < 2:
        return None

    for split_idx in range(1, len(parts)):
        left = "_".join(parts[:split_idx])
        right = "_".join(parts[split_idx:])
        left_norm = _normalize_topology_name(left)
        right_norm = _normalize_method_name(right)
        if left_norm in known_topologies and right_norm in known_methods:
            return left_norm, right_norm

        left_method = _normalize_method_name(left)
        right_topology = _normalize_topology_name(right)
        if left_method in known_methods and right_topology in known_topologies:
            return right_topology, left_method

    topology, method = suffix.rsplit("_", 1)
    topology_norm = _normalize_topology_name(topology)
    method_norm = _normalize_method_name(method)
    if not topology_norm or not method_norm:
        return None
    return topology_norm, method_norm


def _normalize_topology_name(topology: str) -> str:
    normalized = str(topology).strip().lower()
    if normalized == "hex":
        return "hexagonal"
    return normalized


def _topology_display_name(topology: str) -> str:
    normalized = _normalize_topology_name(topology)
    labels = {
        "hexagonal": "Hexagonal",
        "mst": "MST",
        "rng": "RNG",
        "grid": "Grid",
    }
    return labels.get(normalized, normalized.replace("_", " ").title())


def _normalize_method_name(method: str) -> str:
    return str(method).strip().lower()


def _classify_run_type(source_dir: Path, random_path_pattern: str) -> str:
    pattern = str(random_path_pattern).strip().lower()
    source_path = str(source_dir).lower()
    if pattern and pattern in source_path:
        return "random"
    return "full"


# Plot and publication output generation -----------------------------------


def _write_mode_outputs(
    mode_name: str,
    mode_results: MergedModeResults,
    output_dir: Path,
    skip_plots: bool,
    warnings: List[str],
    methods_to_plot: Sequence[str],
) -> None:
    if not mode_results:
        return

    mode_cfg = MODE_CONFIG[mode_name]
    mode_output = output_dir / mode_name
    mode_output.mkdir(parents=True, exist_ok=True)

    by_topology: Dict[str, Dict[str, AxisResults]] = defaultdict(dict)

    for (topology, method), axis_results in sorted(mode_results.items()):
        method_suffix = f"_{topology}_{method}"
        ResultsHandler.save_results(
            axis_results,
            str(mode_output),
            mode=mode_cfg["save_mode"],
            method_suffix=method_suffix,
            merge_existing=False,
        )
        by_topology[topology][method] = axis_results

    if skip_plots:
        return

    try:
        from floatsom.benchmarks.visualization import (
            plot_gpu_scaling_benchmark,
            plot_sample_scaling_benchmark,
        )
    except Exception as exc:
        warnings.append(f"Could not import plotting utilities; skipping plots for {mode_name}: {exc}")
        return

    for topology, topology_results in sorted(by_topology.items()):
        topology_output = mode_output / topology
        topology_output.mkdir(parents=True, exist_ok=True)
        title = f"{mode_cfg['title']} ({topology})"
        filtered_topology_results: Dict[str, AxisResults] = {}
        for method_name, axis_results in topology_results.items():
            filtered_axis_results = _filter_axis_results_gpu_counts(
                axis_results=axis_results,
                allowed_gpu_counts=PUBLICATION_VISIBLE_GPU_COUNTS,
            )
            if filtered_axis_results:
                filtered_topology_results[method_name] = filtered_axis_results
        if not filtered_topology_results:
            warnings.append(
                f"No plotting data left after GPU filtering for {mode_name}/{topology}; skipping plots."
            )
            continue

        try:
            if mode_cfg["plot_type"] == "sample":
                plot_sample_scaling_benchmark(
                    results=filtered_topology_results,
                    output_dir=str(topology_output),
                    title=title,
                    show_speedup=True,
                    show_error_bars=True,
                    show_legend=False,
                    emit_shared_legend=True,
                    methods_to_plot=list(methods_to_plot),
                    sample_min_upper_bound=PUBLICATION_SAMPLE_AXIS_MIN_UPPER_BOUND,
                )
            else:
                plot_gpu_scaling_benchmark(
                    results=filtered_topology_results,
                    output_dir=str(topology_output),
                    title=title,
                    show_speedup=True,
                    show_error_bars=True,
                    show_legend=False,
                    emit_shared_legend=True,
                    axis_mode=mode_cfg["save_mode"],
                    methods_to_plot=list(methods_to_plot),
                    axis_min_upper_bound=(
                        PUBLICATION_DIMENSION_AXIS_MIN_UPPER_BOUND
                        if mode_cfg["save_mode"] == "dimension"
                        else None
                    ),
                )
        except Exception as exc:
            warnings.append(
                f"Failed to generate plots for {mode_name}/{topology}: {exc}"
            )


def _axis_results_contains_gpu(axis_results: AxisResults, gpu_count: int) -> bool:
    """Return True when any axis entry contains the requested GPU key."""
    candidates = {gpu_count, str(gpu_count), f"{float(gpu_count)}"}
    for axis_payload in axis_results.values():
        if not isinstance(axis_payload, dict):
            continue
        if any(candidate in axis_payload for candidate in candidates):
            return True
    return False


def _extract_mean_from_stats(stats_payload: object) -> Optional[float]:
    if not isinstance(stats_payload, dict):
        return None

    mean_raw = stats_payload.get("mean")
    if mean_raw is not None:
        try:
            mean_value = float(mean_raw)
        except (TypeError, ValueError):
            mean_value = None
        else:
            if np.isfinite(mean_value):
                return mean_value

    times_raw = stats_payload.get("times", [])
    if isinstance(times_raw, list):
        values: List[float] = []
        for item in times_raw:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            return float(np.mean(values))

    return None


def _resolve_gpu_payload(axis_payload: MutableMapping[object, object], gpu_count: int) -> Optional[object]:
    if gpu_count in axis_payload:
        return axis_payload[gpu_count]
    for raw_key, raw_value in axis_payload.items():
        parsed = _coerce_gpu_key(raw_key)  # type: ignore[arg-type]
        if parsed == gpu_count:
            return raw_value
    return None


def _build_algorithm_scaling_publication_table(
    merged_modes: Dict[str, MergedModeResults],
    topology: str = "hexagonal",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for mode_name in PUBLICATION_MODE_ORDER:
        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        for method in PUBLICATION_METHOD_ORDER:
            axis_results = mode_results.get((topology, method))
            if not axis_results:
                continue

            gpu_counts: Set[int] = set()
            for axis_payload in axis_results.values():
                if not isinstance(axis_payload, dict):
                    continue
                for raw_gpu in axis_payload.keys():
                    parsed_gpu = _coerce_gpu_key(raw_gpu)  # type: ignore[arg-type]
                    if parsed_gpu is not None:
                        gpu_counts.add(parsed_gpu)
            if 1 not in gpu_counts:
                continue

            for gpu_count in sorted(gpu_counts):
                speedups: List[float] = []
                efficiencies: List[float] = []
                for axis_payload in axis_results.values():
                    if not isinstance(axis_payload, dict):
                        continue

                    baseline_payload = _resolve_gpu_payload(axis_payload, 1)
                    current_payload = _resolve_gpu_payload(axis_payload, gpu_count)
                    if baseline_payload is None or current_payload is None:
                        continue

                    baseline_mean = _extract_mean_from_stats(baseline_payload)
                    current_mean = _extract_mean_from_stats(current_payload)
                    if baseline_mean is None or current_mean is None:
                        continue
                    if baseline_mean <= 0 or current_mean <= 0:
                        continue

                    speedup = baseline_mean / current_mean
                    efficiency = (speedup / float(gpu_count)) * 100.0
                    if np.isfinite(speedup) and np.isfinite(efficiency):
                        speedups.append(float(speedup))
                        efficiencies.append(float(efficiency))

                if not speedups:
                    continue

                rows.append(
                    {
                        "mode_name": mode_name,
                        "mode_label": PUBLICATION_MODE_LABELS.get(mode_name, mode_name),
                        "topology": topology,
                        "method": method,
                        "gpu_count": int(gpu_count),
                        "n_axes": int(len(speedups)),
                        "speedup_median": float(np.median(speedups)),
                        "speedup_mean": float(np.mean(speedups)),
                        "efficiency_median": float(np.median(efficiencies)),
                        "efficiency_mean": float(np.mean(efficiencies)),
                    }
                )

    return rows


def _build_topology_ratio_publication_table(
    merged_modes: Dict[str, MergedModeResults],
    comparison_gpu_count: int,
    method: str = "batch",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for mode_name in PUBLICATION_MODE_ORDER:
        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        axis_hex = mode_results.get(("hexagonal", method))
        axis_mst = mode_results.get(("mst", method))
        axis_rng = mode_results.get(("rng", method))
        if not axis_hex:
            continue

        axis_keys: Set[NumericKey] = set()
        for payload in [axis_hex, axis_mst, axis_rng]:
            if not payload:
                continue
            axis_keys.update(payload.keys())

        ordered_axis_keys = sorted(
            [key for key in axis_keys if _coerce_numeric_key(key) is not None],
            key=lambda value: float(_coerce_numeric_key(value)),  # type: ignore[arg-type]
        )

        for axis_key in ordered_axis_keys:
            axis_numeric = _coerce_numeric_key(axis_key)
            if axis_numeric is None:
                continue

            def _ratio_row(numerator_topology: str, numerator_axis: Optional[AxisResults], ratio_name: str) -> None:
                if not numerator_axis:
                    return
                numerator_payload_axis = numerator_axis.get(axis_key)
                denominator_payload_axis = axis_hex.get(axis_key) if axis_hex else None
                if not isinstance(numerator_payload_axis, dict) or not isinstance(denominator_payload_axis, dict):
                    return

                numerator_payload = _resolve_gpu_payload(numerator_payload_axis, comparison_gpu_count)
                denominator_payload = _resolve_gpu_payload(denominator_payload_axis, comparison_gpu_count)
                if numerator_payload is None or denominator_payload is None:
                    return

                numerator_mean = _extract_mean_from_stats(numerator_payload)
                denominator_mean = _extract_mean_from_stats(denominator_payload)
                if numerator_mean is None or denominator_mean is None:
                    return
                if numerator_mean <= 0 or denominator_mean <= 0:
                    return

                ratio = numerator_mean / denominator_mean
                if not np.isfinite(ratio):
                    return

                rows.append(
                    {
                        "mode_name": mode_name,
                        "mode_label": PUBLICATION_MODE_LABELS.get(mode_name, mode_name),
                        "gpu_count": int(comparison_gpu_count),
                        "method": method,
                        "axis_value": float(axis_numeric),
                        "ratio_type": ratio_name,
                        "numerator_topology": numerator_topology,
                        "denominator_topology": "hexagonal",
                        "runtime_ratio": float(ratio),
                    }
                )

            _ratio_row("mst", axis_mst, "mst_over_hex")
            _ratio_row("rng", axis_rng, "rng_over_hex")

    return rows


def _build_figure_10_topology_runtime_summary_table(
    merged_modes: Dict[str, MergedModeResults],
    comparison_gpu_count: int,
    method: str = "batch",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    topology_order: Tuple[str, ...] = ("hexagonal", "mst", "rng")

    for mode_name in PUBLICATION_PANEL_MODE_ORDER:
        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        axis_by_topology: Dict[str, Optional[AxisResults]] = {
            topology: mode_results.get((topology, method))
            for topology in topology_order
        }
        if any(axis_results is None for axis_results in axis_by_topology.values()):
            continue

        common_axis_keys: Optional[Set[NumericKey]] = None
        for axis_results in axis_by_topology.values():
            assert axis_results is not None
            topology_axis_keys = {
                axis_key
                for axis_key in axis_results.keys()
                if _coerce_numeric_key(axis_key) is not None
            }
            if common_axis_keys is None:
                common_axis_keys = topology_axis_keys
            else:
                common_axis_keys &= topology_axis_keys

        if not common_axis_keys:
            continue

        ordered_axis_keys = sorted(
            common_axis_keys,
            key=lambda value: float(_coerce_numeric_key(value)),  # type: ignore[arg-type]
        )

        selected_axis_key: Optional[NumericKey] = None
        runtime_means: Dict[str, float] = {}
        for axis_key in reversed(ordered_axis_keys):
            axis_runtime_means: Dict[str, float] = {}
            valid_axis = True
            for topology in topology_order:
                topology_axis_results = axis_by_topology[topology]
                assert topology_axis_results is not None
                axis_payload = topology_axis_results.get(axis_key)
                if not isinstance(axis_payload, dict):
                    valid_axis = False
                    break
                gpu_payload = _resolve_gpu_payload(axis_payload, comparison_gpu_count)
                if gpu_payload is None:
                    valid_axis = False
                    break
                runtime_mean = _extract_mean_from_stats(gpu_payload)
                if runtime_mean is None or runtime_mean <= 0 or not np.isfinite(runtime_mean):
                    valid_axis = False
                    break
                axis_runtime_means[topology] = float(runtime_mean)
            if valid_axis and len(axis_runtime_means) == len(topology_order):
                selected_axis_key = axis_key
                runtime_means = axis_runtime_means
                break

        if selected_axis_key is None or len(runtime_means) != len(topology_order):
            continue

        fastest_topology, fastest_runtime = min(runtime_means.items(), key=lambda item: item[1])
        slowest_topology, slowest_runtime = max(runtime_means.items(), key=lambda item: item[1])
        max_pairwise_runtime_spread_pct = ((slowest_runtime - fastest_runtime) / fastest_runtime) * 100.0

        rows.append(
            {
                "mode_name": mode_name,
                "mode_label": PUBLICATION_MODE_LABELS.get(mode_name, mode_name),
                "gpu_count": int(comparison_gpu_count),
                "method": method,
                "axis_value": float(_coerce_numeric_key(selected_axis_key)),
                "hexagonal_runtime_mean_s": float(runtime_means["hexagonal"]),
                "mst_runtime_mean_s": float(runtime_means["mst"]),
                "rng_runtime_mean_s": float(runtime_means["rng"]),
                "fastest_topology": fastest_topology,
                "fastest_runtime_mean_s": float(fastest_runtime),
                "slowest_topology": slowest_topology,
                "slowest_runtime_mean_s": float(slowest_runtime),
                "max_pairwise_runtime_spread_pct": float(max_pairwise_runtime_spread_pct),
            }
        )

    return rows


def _collect_gpu_counts(axis_results: AxisResults) -> Set[int]:
    gpu_counts: Set[int] = set()
    for axis_payload in axis_results.values():
        if not isinstance(axis_payload, dict):
            continue
        for raw_gpu_key in axis_payload.keys():
            parsed_gpu = _coerce_gpu_key(raw_gpu_key)  # type: ignore[arg-type]
            if parsed_gpu is not None:
                gpu_counts.add(parsed_gpu)
    return gpu_counts


def _build_axis_numeric_key_map(axis_results: AxisResults) -> Dict[float, NumericKey]:
    numeric_map: Dict[float, NumericKey] = {}
    for axis_key in axis_results.keys():
        numeric_axis = _coerce_numeric_key(axis_key)
        if numeric_axis is None:
            continue
        numeric_map[float(numeric_axis)] = axis_key
    return numeric_map


def _build_sampling_ratio_publication_table(
    random_modes: Dict[str, RandomComparisonModeResults],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for mode_name in PUBLICATION_MODE_ORDER:
        mode_results = random_modes.get(mode_name, {})
        if not mode_results:
            continue

        for topology in sorted({key[1] for key in mode_results.keys()}):
            for method in RANDOM_COMPARISON_METHODS:
                full_axis_results = mode_results.get(("full", topology, method))
                random_axis_results = mode_results.get(("random", topology, method))
                if not full_axis_results or not random_axis_results:
                    continue

                full_numeric_map = _build_axis_numeric_key_map(full_axis_results)
                random_numeric_map = _build_axis_numeric_key_map(random_axis_results)
                common_axis_values = sorted(set(full_numeric_map.keys()).intersection(random_numeric_map.keys()))
                if not common_axis_values:
                    continue

                common_gpu_counts = sorted(
                    _collect_gpu_counts(full_axis_results).intersection(_collect_gpu_counts(random_axis_results))
                )
                if not common_gpu_counts:
                    continue

                for gpu_count in common_gpu_counts:
                    axis_ratios: List[float] = []
                    for axis_numeric in common_axis_values:
                        full_axis_payload = full_axis_results.get(full_numeric_map[axis_numeric])
                        random_axis_payload = random_axis_results.get(random_numeric_map[axis_numeric])
                        if not isinstance(full_axis_payload, dict) or not isinstance(random_axis_payload, dict):
                            continue

                        full_gpu_payload = _resolve_gpu_payload(full_axis_payload, gpu_count)
                        random_gpu_payload = _resolve_gpu_payload(random_axis_payload, gpu_count)
                        if full_gpu_payload is None or random_gpu_payload is None:
                            continue

                        full_mean = _extract_mean_from_stats(full_gpu_payload)
                        random_mean = _extract_mean_from_stats(random_gpu_payload)
                        if full_mean is None or random_mean is None:
                            continue
                        if full_mean <= 0 or random_mean <= 0:
                            continue

                        runtime_ratio = random_mean / full_mean
                        if np.isfinite(runtime_ratio):
                            axis_ratios.append(float(runtime_ratio))

                    if not axis_ratios:
                        continue

                    rows.append(
                        {
                            "mode_name": mode_name,
                            "mode_label": PUBLICATION_MODE_LABELS.get(mode_name, mode_name),
                            "topology": topology,
                            "method": method,
                            "gpu_count": int(gpu_count),
                            "n_axes": int(len(axis_ratios)),
                            "runtime_ratio_median": float(np.median(axis_ratios)),
                            "runtime_ratio_mean": float(np.mean(axis_ratios)),
                        }
                    )

    return rows


def _resolve_paper_assets_figures_dir() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "floatsom" / "paper" / "assets" / "figures"
        if candidate.is_dir():
            return candidate
    return None


def _resolve_paper_assets_tables_dir() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "floatsom" / "paper" / "assets" / "tables"
        if candidate.is_dir():
            return candidate
    return None


def _resolve_paper_manual_figures_dir() -> Optional[Path]:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "floatsom" / "paper" / "assets_manual" / "figures"
        if candidate.is_dir():
            return candidate
    return None


def _mirror_publication_figure_to_manual_assets(src_path: Path, dst_name: str) -> Optional[Path]:
    if dst_name in {"fig_3.svg", "fig_11.svg"}:
        return None
    manual_figures_dir = _resolve_paper_manual_figures_dir()
    if manual_figures_dir is None:
        return None
    manual_figures_dir.mkdir(parents=True, exist_ok=True)
    manual_path = manual_figures_dir / dst_name
    shutil.copy2(src_path, manual_path)
    return manual_path


def _sync_publication_figures_to_paper_assets(
    publication_dir: Path,
    warnings: List[str],
) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    if not publication_dir.is_dir():
        return copied

    paper_figures_dir = _resolve_paper_assets_figures_dir()
    if paper_figures_dir is None:
        warnings.append(
            "Could not locate floatsom/paper/assets/figures; skipped publication figure sync."
        )
        return copied

    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in PUBLICATION_PAPER_ASSET_FIGURE_MAP.items():
        src_path = publication_dir / src_name
        if not src_path.exists():
            continue
        dst_path = paper_figures_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied[src_name] = str(dst_path.resolve())
        manual_path = _mirror_publication_figure_to_manual_assets(dst_path, dst_name)
        if manual_path is not None:
            copied[f"{src_name} -> manual/{dst_name}"] = str(manual_path.resolve())
        extra_targets = PUBLICATION_PAPER_ASSET_EXTRA_COPIES.get(src_name, ())
        for extra_name in extra_targets:
            extra_path = paper_figures_dir / extra_name
            shutil.copy2(src_path, extra_path)
            copied[f"{src_name} -> {extra_name}"] = str(extra_path.resolve())

    return copied


def _sync_publication_tables_to_paper_assets(
    publication_tables_dir: Path,
    warnings: List[str],
) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    if not publication_tables_dir.is_dir():
        return copied

    paper_tables_dir = _resolve_paper_assets_tables_dir()
    if paper_tables_dir is None:
        warnings.append(
            "Could not locate floatsom/paper/assets/tables; skipped publication table sync."
        )
        return copied

    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in PUBLICATION_PAPER_ASSET_TABLE_MAP.items():
        src_path = publication_tables_dir / src_name
        if not src_path.exists():
            continue
        dst_path = paper_tables_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied[src_name] = str(dst_path.resolve())

    return copied


def _parse_scaling_log_entry(log_path: Path, mode_name: str) -> Optional[Dict[str, object]]:
    match = SCALING_LOG_CONFIG_PATTERN.match(log_path.stem)
    if match is None:
        return None

    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return None

    staging_mode = "unknown"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("Data staging mode:"):
            continue
        parsed_mode = line.split(":", 1)[1].strip().lower()
        if parsed_mode:
            staging_mode = parsed_mode
        break

    axis_value_by_mode = {
        "dimension_scaling": int(match.group("dimension")),
        "sample_scaling": int(match.group("samples")),
        "grid_size_scaling": int(match.group("grid_size")),
    }
    if mode_name not in axis_value_by_mode:
        return None

    return {
        "mode_name": mode_name,
        "topology": _normalize_topology_name(match.group("topology")),
        "method": _normalize_method_name(match.group("method")),
        "axis_value": int(axis_value_by_mode[mode_name]),
        "gpu_count": int(match.group("gpu_count")),
        "seed": int(match.group("seed")),
        "staging_mode": staging_mode,
        "log_path": str(log_path.resolve()),
    }


def _collect_scaling_log_index(
    input_dirs: Sequence[Path],
) -> Dict[Tuple[str, str, str, int, int], List[Dict[str, object]]]:
    index: Dict[Tuple[str, str, str, int, int], List[Dict[str, object]]] = defaultdict(list)

    for source_dir in input_dirs:
        for mode_name in MODE_CONFIG:
            logs_dir = source_dir / mode_name / "logs"
            if not logs_dir.is_dir():
                continue
            for log_path in sorted(logs_dir.glob("*.log")):
                entry = _parse_scaling_log_entry(log_path, mode_name)
                if entry is None:
                    continue
                key = (
                    str(entry["mode_name"]),
                    str(entry["topology"]),
                    str(entry["method"]),
                    int(entry["axis_value"]),
                    int(entry["gpu_count"]),
                )
                index[key].append(entry)

    return index


def _build_rng_scaling_diagnostics_publication_table(
    merged_modes: Dict[str, MergedModeResults],
    input_dirs: Sequence[Path],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    log_index = _collect_scaling_log_index(input_dirs)
    target_topology = "rng"
    target_method = "batch"

    for mode_name in PUBLICATION_MODE_ORDER:
        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        axis_results = mode_results.get((target_topology, target_method))
        if not axis_results:
            continue

        ordered_axis_keys = sorted(
            [key for key in axis_results.keys() if _coerce_numeric_key(key) is not None],
            key=lambda value: float(_coerce_numeric_key(value)),  # type: ignore[arg-type]
        )
        for axis_key in ordered_axis_keys:
            axis_numeric = _coerce_numeric_key(axis_key)
            axis_payload = axis_results.get(axis_key)
            if axis_numeric is None or not isinstance(axis_payload, dict):
                continue

            for gpu_count in sorted(_collect_gpu_counts({axis_key: axis_payload})):
                gpu_payload = _resolve_gpu_payload(axis_payload, gpu_count)
                if not isinstance(gpu_payload, dict):
                    continue

                runtime_mean_s = _extract_mean_from_stats(gpu_payload)
                if runtime_mean_s is None:
                    continue

                std_raw = gpu_payload.get("std")
                try:
                    runtime_std_s = float(std_raw) if std_raw is not None else 0.0
                except (TypeError, ValueError):
                    runtime_std_s = 0.0

                count_raw = gpu_payload.get("count")
                try:
                    n_repeats = int(count_raw) if count_raw is not None else 0
                except (TypeError, ValueError):
                    n_repeats = 0

                axis_value = int(float(axis_numeric))
                log_entries = log_index.get(
                    (mode_name, target_topology, target_method, axis_value, int(gpu_count)),
                    [],
                )
                staging_modes = sorted(
                    {
                        str(entry.get("staging_mode", "")).strip().lower()
                        for entry in log_entries
                        if str(entry.get("staging_mode", "")).strip()
                    }
                )
                if len(staging_modes) == 1:
                    resolved_staging_mode = staging_modes[0]
                elif len(staging_modes) > 1:
                    resolved_staging_mode = "mixed"
                else:
                    resolved_staging_mode = "unknown"

                rows.append(
                    {
                        "mode_name": mode_name,
                        "mode_label": PUBLICATION_MODE_LABELS.get(mode_name, mode_name),
                        "topology": target_topology,
                        "method": target_method,
                        "axis_value": axis_value,
                        "gpu_count": int(gpu_count),
                        "runtime_mean_s": float(runtime_mean_s),
                        "runtime_std_s": float(runtime_std_s),
                        "n_repeats": int(n_repeats),
                        "log_count": int(len(log_entries)),
                        "staging_mode": resolved_staging_mode,
                        "staging_modes": ",".join(staging_modes) if staging_modes else "unknown",
                        "any_repeat_disk": bool("disk" in staging_modes),
                        "all_repeats_disk": bool(staging_modes and set(staging_modes) == {"disk"}),
                    }
                )

    return rows


def _write_publication_support_tables(
    merged_modes: Dict[str, MergedModeResults],
    input_dirs: Sequence[Path],
    output_dir: Path,
    comparison_gpu_count: int,
    warnings: List[str],
) -> None:
    publication_tables_dir = output_dir / "publication_figures" / "tables"
    publication_tables_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_rows = _build_rng_scaling_diagnostics_publication_table(
        merged_modes=merged_modes,
        input_dirs=input_dirs,
    )
    if diagnostics_rows:
        diagnostics_path = publication_tables_dir / "supp_table_figure_9_rng_scaling_diagnostics.tsv"
        with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "mode_name",
                    "mode_label",
                    "topology",
                    "method",
                    "axis_value",
                    "gpu_count",
                    "runtime_mean_s",
                    "runtime_std_s",
                    "n_repeats",
                    "log_count",
                    "staging_mode",
                    "staging_modes",
                    "any_repeat_disk",
                    "all_repeats_disk",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(diagnostics_rows)

    figure_10_summary_rows = _build_figure_10_topology_runtime_summary_table(
        merged_modes=merged_modes,
        comparison_gpu_count=comparison_gpu_count,
    )
    if figure_10_summary_rows:
        figure_10_summary_path = publication_tables_dir / "supp_table_figure_10_topology_runtime_summary.tsv"
        with figure_10_summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "mode_name",
                    "mode_label",
                    "gpu_count",
                    "method",
                    "axis_value",
                    "hexagonal_runtime_mean_s",
                    "mst_runtime_mean_s",
                    "rng_runtime_mean_s",
                    "fastest_topology",
                    "fastest_runtime_mean_s",
                    "slowest_topology",
                    "slowest_runtime_mean_s",
                    "max_pairwise_runtime_spread_pct",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(figure_10_summary_rows)

    _sync_publication_tables_to_paper_assets(publication_tables_dir, warnings)


# SVG composition helpers ---------------------------------------------------


def _compose_panel_matrix_svg(
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    legend_path: Optional[Path] = None,
    legend_after_row: Optional[int] = None,
    legend_y_nudge: float = 0.0,
    legend_band_height_override: Optional[int] = None,
    legend_gap_override: Optional[float] = None,
    legend_scale_override: Optional[float] = None,
    panel_width_override: Optional[int] = None,
    panel_height_override: Optional[int] = None,
    title_band_override: Optional[int] = None,
    title_font_size_override: Optional[float] = None,
    panel_label_font_size_override: Optional[float] = None,
    panel_caption_font_size_override: Optional[float] = None,
    show_panel_annotations: bool = True,
) -> bool:
    from floatsom.benchmarks.optuna.optuna_results_analysis.scripts.analysis.publication_figures.composition import (
        _compose_panel_matrix_from_images,
    )

    return _compose_panel_matrix_from_images(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        dpi=300,
        legend_path=legend_path,
        legend_after_row=legend_after_row,
        legend_y_nudge=legend_y_nudge,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        legend_scale_override=legend_scale_override,
        panel_width_override=panel_width_override,
        panel_height_override=panel_height_override,
        title_font_size_override=title_font_size_override,
        panel_label_font_size_override=panel_label_font_size_override,
        panel_caption_font_size_override=panel_caption_font_size_override,
        title_band_override=title_band_override,
        show_panel_annotations=show_panel_annotations,
    )


def _mode_plot_svg_filename(mode_name: str, plot_kind: str) -> str:
    prefix = "sample_scaling" if mode_name == "sample_scaling" else "gpu_scaling"
    return f"{prefix}_{plot_kind}.svg"


def _mode_plot_svg_path(
    root_dir: Path,
    mode_name: str,
    plot_kind: str,
    topology: Optional[str] = None,
) -> Path:
    path = root_dir / mode_name
    if topology:
        path = path / topology
    return path / _mode_plot_svg_filename(mode_name, plot_kind)


def _mode_legend_dir(
    root_dir: Path,
    mode_name: str,
    topology: Optional[str] = None,
) -> Path:
    path = root_dir / mode_name
    if topology:
        path = path / topology
    return path


def _build_random_comparison_topology_payload(
    mode_results: RandomComparisonModeResults,
) -> Dict[str, Dict[str, AxisResults]]:
    by_topology: Dict[str, Dict[str, AxisResults]] = defaultdict(dict)
    for (run_type, topology, method), axis_results in sorted(mode_results.items()):
        if method not in RANDOM_COMPARISON_METHODS:
            continue
        by_topology[topology][run_type] = axis_results
    return by_topology


def _build_figure_8_topology_payload(
    full_mode_results: MergedModeResults,
    random_mode_results: RandomComparisonModeResults,
) -> Dict[str, Dict[str, AxisResults]]:
    by_topology: Dict[str, Dict[str, AxisResults]] = defaultdict(dict)

    for (topology, method), axis_results in sorted(full_mode_results.items()):
        if method not in RANDOM_COMPARISON_METHODS:
            continue
        by_topology[topology]["full"] = axis_results

    for (run_type, topology, method), axis_results in sorted(random_mode_results.items()):
        if run_type != "random" or method not in RANDOM_COMPARISON_METHODS:
            continue
        by_topology[topology]["random"] = axis_results

    return by_topology


def _pick_first_existing_legend(legend_dirs: Sequence[Path]) -> Optional[Path]:
    for legend_dir in legend_dirs:
        candidate = legend_dir / "legend_key.svg"
        if candidate.exists():
            return candidate
    return None


def _compose_publication_matrix_from_sources(
    panel_rows: Sequence[Sequence[Tuple[str, str, Path]]],
    title: str,
    output_path: Path,
    legend_dirs: Sequence[Path],
    warnings: List[str],
    warning_label: str,
    panel_width_override: Optional[int] = None,
    panel_height_override: Optional[int] = None,
    title_band_override: Optional[int] = None,
    title_font_size_override: Optional[float] = None,
    legend_band_height_override: Optional[int] = None,
    legend_gap_override: Optional[float] = None,
    legend_scale_override: Optional[float] = None,
    panel_label_font_size_override: Optional[float] = None,
    panel_caption_font_size_override: Optional[float] = None,
    show_panel_annotations: bool = True,
) -> bool:
    missing = [path for row in panel_rows for _, _, path in row if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        warnings.append(f"Missing panel source files for {warning_label}: {preview}{suffix}")
        return False

    legend_path = _pick_first_existing_legend(legend_dirs)
    if legend_path is None:
        warnings.append(f"Missing shared legend for {warning_label}; composing without legend band.")

    return _compose_panel_matrix_svg(
        panel_rows=panel_rows,
        title=title,
        output_path=output_path,
        legend_path=legend_path,
        panel_width_override=panel_width_override,
        panel_height_override=panel_height_override,
        title_band_override=title_band_override,
        title_font_size_override=title_font_size_override,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        legend_scale_override=legend_scale_override,
        panel_label_font_size_override=panel_label_font_size_override,
        panel_caption_font_size_override=panel_caption_font_size_override,
        show_panel_annotations=show_panel_annotations,
    )


def _write_publication_panel_outputs(
    merged_modes: Dict[str, MergedModeResults],
    random_modes: Dict[str, RandomComparisonModeResults],
    output_dir: Path,
    comparison_output_subdir: str,
    random_comparison_output_subdir: str,
    comparison_gpu_count: int,
    figure_8_tripanel_results_dirs: Dict[int, List[Path]],
    random_path_pattern: str,
    warnings: List[str],
) -> Dict[str, bool]:
    generated = {"figure_8": False, "figure_9": False, "figure_10": False}

    publication_dir = output_dir / "publication_figures"
    publication_dir.mkdir(parents=True, exist_ok=True)
    letters = [chr(ord("A") + idx) for idx in range(26)]

    # Figure 8: Random-vs-full sample-scaling runtime tripanel at 1/2/4 GPUs.
    figure_8_path = publication_dir / "figure_8_sampling_speed_random_over_full.svg"
    panel_output_root = publication_dir / "figure_8_tripanel_panels"
    if all(gpu_count in figure_8_tripanel_results_dirs for gpu_count in FIGURE_8_TRIPANEL_GPU_COUNTS):
        try:
            from floatsom.benchmarks.visualization import (
                plot_topology_comparison_scaling_benchmark,
            )
        except Exception as exc:
            warnings.append(
                f"Could not import comparison scaling plotter for Figure 8 tripanel: {exc}"
            )
        else:
            figure_8_row: List[Tuple[str, str, Path]] = []
            legend_dirs_8: List[Path] = []
            for panel_idx, gpu_count in enumerate(FIGURE_8_TRIPANEL_GPU_COUNTS):
                source_dirs = figure_8_tripanel_results_dirs[gpu_count]
                _, figure_8_random_modes, _ = _ingest_harmonization_inputs(
                    input_dirs=source_dirs,
                    random_path_pattern=random_path_pattern,
                    warnings=warnings,
                )
                panel_by_topology = _build_figure_8_topology_payload(
                    full_mode_results=merged_modes.get("sample_scaling", {}),
                    random_mode_results=figure_8_random_modes.get("sample_scaling", {}),
                )
                if not panel_by_topology:
                    raise RuntimeError(
                        "Figure 8 "
                        f"{gpu_count}-GPU input produced no batch sample-scaling data after combining "
                        f"main full inputs with Figure 8 random inputs: {', '.join(str(path) for path in source_dirs)}"
                    )
                missing_pairs = [
                    topology
                    for topology, run_payload in sorted(panel_by_topology.items())
                    if not all(run_type in run_payload for run_type in RANDOM_COMPARISON_RUN_TYPES)
                ]
                if missing_pairs:
                    raise RuntimeError(
                        "Figure 8 "
                        f"{gpu_count}-GPU input is missing combined full/random sample-scaling pairs for topologies "
                        f"{missing_pairs}. Figure 8 random source dirs: {', '.join(str(path) for path in source_dirs)}"
                    )
                has_requested_gpu = any(
                    _axis_results_contains_gpu(axis_results, gpu_count)
                    for topology_payload in panel_by_topology.values()
                    for axis_results in topology_payload.values()
                )
                if not has_requested_gpu:
                    raise RuntimeError(
                        "Figure 8 "
                        f"{gpu_count}-GPU input has no {gpu_count}-GPU sample-scaling entries: "
                        f"{', '.join(str(path) for path in source_dirs)}"
                    )

                panel_output_dir = panel_output_root / f"{gpu_count}gpu"
                panel_output_dir.mkdir(parents=True, exist_ok=True)
                for stale_name in (
                    _mode_plot_svg_filename("sample_scaling", "performance"),
                    _mode_plot_svg_filename("sample_scaling", "speedup"),
                    _mode_plot_svg_filename("sample_scaling", "efficiency"),
                    "legend_key.svg",
                ):
                    stale_path = panel_output_dir / stale_name
                    if stale_path.exists():
                        stale_path.unlink()

                gpu_label = f"{gpu_count} GPU" if gpu_count == 1 else f"{gpu_count} GPUs"
                title = f"Harmonized Sample Scaling - Random and Full ({gpu_label})"
                try:
                    plot_topology_comparison_scaling_benchmark(
                        results_by_topology=panel_by_topology,
                        output_dir=str(panel_output_dir),
                        title=title,
                        axis_mode="sample",
                        comparison_gpu_count=gpu_count,
                        show_speedup=False,
                        show_error_bars=True,
                        show_legend=False,
                        methods_to_plot=list(RANDOM_COMPARISON_RUN_TYPES),
                        emit_shared_legend=True,
                        include_performance=True,
                        topology_color_overrides=RANDOM_COMPARISON_COLORS,
                        sample_min_upper_bound=PUBLICATION_SAMPLE_AXIS_MIN_UPPER_BOUND,
                        highlight_trailing_missing_x_range=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to generate Figure 8 "
                        f"{gpu_count}-GPU sample-scaling panel from "
                        f"{', '.join(str(path) for path in source_dirs)}: {exc}"
                    ) from exc

                panel_path = panel_output_dir / _mode_plot_svg_filename("sample_scaling", "performance")
                if not panel_path.exists():
                    raise FileNotFoundError(
                        f"Figure 8 {gpu_count}-GPU panel was not written at expected path: {panel_path}"
                    )

                figure_8_row.append(
                    (
                        letters[panel_idx],
                        f"{gpu_count} GPU" if gpu_count == 1 else f"{gpu_count} GPUs",
                        panel_path,
                    )
                )
                legend_dirs_8.append(panel_output_dir)

            if len(figure_8_row) == len(FIGURE_8_TRIPANEL_GPU_COUNTS):
                if _compose_publication_matrix_from_sources(
                    panel_rows=[figure_8_row],
                    title="Figure 8: Sampling Runtime",
                    output_path=figure_8_path,
                    legend_dirs=legend_dirs_8,
                    warnings=warnings,
                    warning_label="Figure 8 random-vs-full sample-scaling tripanel",
                    panel_width_override=PUBLICATION_FIGURE_8_PANEL_WIDTH,
                    panel_height_override=PUBLICATION_FIGURE_8_PANEL_HEIGHT,
                    title_band_override=PUBLICATION_FIGURE_8_TITLE_BAND,
                    title_font_size_override=PUBLICATION_FIGURE_8_TITLE_FONT_SIZE,
                    legend_band_height_override=PUBLICATION_FIGURE_8_LEGEND_BAND_HEIGHT,
                    legend_gap_override=PUBLICATION_LEGEND_GAP,
                    legend_scale_override=PUBLICATION_FIGURE_8_LEGEND_SCALE,
                    show_panel_annotations=True,
                ):
                    generated["figure_8"] = True
                else:
                    raise RuntimeError(
                        "Figure 8 composite composition failed after generating all tripanel sources."
                    )
    else:
        missing_gpu_counts = [
            gpu_count
            for gpu_count in FIGURE_8_TRIPANEL_GPU_COUNTS
            if gpu_count not in figure_8_tripanel_results_dirs
        ]
        if figure_8_tripanel_results_dirs and missing_gpu_counts:
            raise RuntimeError(
                "Figure 8 generation entered an invalid partial state. "
                f"Missing GPU counts: {missing_gpu_counts}."
            )
        if missing_gpu_counts:
            warnings.append(
                "Figure 8 tripanel skipped because explicit random/full input directories were not "
                f"resolved for GPU counts {missing_gpu_counts}."
            )

    figure_9_path = publication_dir / "figure_9_algorithm_first_gpu_scaling.svg"
    if _compose_single_topology_scaling_figure(
        output_root=output_dir,
        output_path=figure_9_path,
        topology=PUBLICATION_FIGURE_9_TOPOLOGY,
        title=(
            f"Figure 9: Multi-GPU Full-Batch Scaling "
            f"({_topology_display_name(PUBLICATION_FIGURE_9_TOPOLOGY)})"
        ),
        warnings=warnings,
        warning_label=f"Figure 9 {PUBLICATION_FIGURE_9_TOPOLOGY} performance+efficiency",
        title_font_size_override=PUBLICATION_FIGURE_9_TITLE_FONT_SIZE,
        legend_band_height_override=PUBLICATION_FIGURE_9_LEGEND_BAND_HEIGHT,
        legend_gap_override=PUBLICATION_FIGURE_9_LEGEND_GAP,
        legend_scale_override=PUBLICATION_FIGURE_9_LEGEND_SCALE,
        panel_label_font_size_override=PUBLICATION_FIGURE_9_PANEL_LABEL_FONT_SIZE,
        panel_caption_font_size_override=PUBLICATION_FIGURE_9_PANEL_CAPTION_FONT_SIZE,
    ):
        generated["figure_9"] = True

    # Figure 10: Topology scaling comparison.
    topology_root = output_dir / comparison_output_subdir
    figure_5_row: List[Tuple[str, str, Path]] = []
    legend_dirs_5: List[Path] = []
    for col_idx, mode_name in enumerate(PUBLICATION_PANEL_MODE_ORDER):
        mode_label = PUBLICATION_MODE_LABELS.get(mode_name, mode_name)
        figure_5_row.append(
            (
                letters[col_idx],
                mode_label,
                _mode_plot_svg_path(topology_root, mode_name, "performance"),
            )
        )
        legend_dirs_5.append(_mode_legend_dir(topology_root, mode_name))

    figure_10_path = publication_dir / "figure_10_topology_mst_hex_performance.svg"
    if _compose_publication_matrix_from_sources(
        panel_rows=[figure_5_row],
        title="Figure 11: Topology Runtime Comparison (8 GPUs, Full Batch)",
        output_path=figure_10_path,
        legend_dirs=legend_dirs_5,
        warnings=warnings,
        warning_label="Figure 10 topology scaling",
        legend_band_height_override=PUBLICATION_FIGURE_10_LEGEND_BAND_HEIGHT,
        legend_gap_override=PUBLICATION_LEGEND_GAP,
        legend_scale_override=PUBLICATION_FIGURE_10_LEGEND_SCALE,
    ):
        generated["figure_10"] = True

    return generated


def _compose_single_topology_scaling_figure(
    *,
    output_root: Path,
    output_path: Path,
    topology: str,
    title: str,
    warnings: List[str],
    warning_label: str,
    title_font_size_override: Optional[float] = None,
    legend_band_height_override: Optional[int] = None,
    legend_gap_override: Optional[float] = None,
    legend_scale_override: Optional[float] = None,
    panel_label_font_size_override: Optional[float] = None,
    panel_caption_font_size_override: Optional[float] = None,
) -> bool:
    normalized_topology = _normalize_topology_name(topology)
    top_row: List[Tuple[str, str, Path]] = []
    bottom_row: List[Tuple[str, str, Path]] = []
    legend_dirs: List[Path] = []
    letters = [chr(ord("A") + idx) for idx in range(26)]
    for col_idx, mode_name in enumerate(PUBLICATION_PANEL_MODE_ORDER):
        mode_label = PUBLICATION_MODE_LABELS.get(mode_name, mode_name)
        top_row.append(
            (
                letters[col_idx],
                mode_label,
                _mode_plot_svg_path(output_root, mode_name, "performance", topology=normalized_topology),
            )
        )
        bottom_row.append(
            (
                letters[col_idx + 3],
                f"{mode_label} (Efficiency)",
                _mode_plot_svg_path(output_root, mode_name, "efficiency", topology=normalized_topology),
            )
        )
        legend_dirs.append(_mode_legend_dir(output_root, mode_name, topology=normalized_topology))

    return _compose_publication_matrix_from_sources(
        panel_rows=[top_row, bottom_row],
        title=title,
        output_path=output_path,
        legend_dirs=legend_dirs,
        warnings=warnings,
        warning_label=warning_label,
        title_font_size_override=title_font_size_override,
        legend_band_height_override=legend_band_height_override,
        legend_gap_override=legend_gap_override,
        legend_scale_override=legend_scale_override,
        panel_label_font_size_override=panel_label_font_size_override,
        panel_caption_font_size_override=panel_caption_font_size_override,
    )


def _write_topology_scaling_supplementary_outputs(
    output_dir: Path,
    warnings: List[str],
) -> Dict[str, bool]:
    publication_dir = output_dir / "publication_figures"
    publication_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, bool] = {}
    letters = [chr(ord("A") + idx) for idx in range(26)]

    mst_hex_rows: List[List[Tuple[str, str, Path]]] = [[], []]
    mst_hex_legend_dirs: List[Path] = []
    for col_idx, mode_name in enumerate(PUBLICATION_PANEL_MODE_ORDER):
        mode_label = PUBLICATION_MODE_LABELS.get(mode_name, mode_name)
        mst_hex_rows[0].append(
            (
                letters[col_idx],
                f"MST: {mode_label}",
                _mode_plot_svg_path(output_dir, mode_name, "performance", topology="mst"),
            )
        )
        mst_hex_rows[1].append(
            (
                letters[col_idx + len(PUBLICATION_PANEL_MODE_ORDER)],
                f"Hexagonal: {mode_label}",
                _mode_plot_svg_path(output_dir, mode_name, "performance", topology="hexagonal"),
            )
        )
        mst_hex_legend_dirs.extend(
            [
                _mode_legend_dir(output_dir, mode_name, topology="mst"),
                _mode_legend_dir(output_dir, mode_name, topology="hexagonal"),
            ]
        )

    mst_hex_output_path = publication_dir / "supplementary_figure_s12_mst_hex_gpu_scaling.svg"
    if mst_hex_output_path.exists():
        mst_hex_output_path.unlink()
    generated[mst_hex_output_path.name] = _compose_publication_matrix_from_sources(
        panel_rows=mst_hex_rows,
        title="Supplementary Figure S11: Per-Topology GPU Scaling Performance\n(MST and Hexagonal)",
        output_path=mst_hex_output_path,
        legend_dirs=mst_hex_legend_dirs,
        warnings=warnings,
        warning_label="Supplementary Figure S11 MST/Hex GPU scaling performance",
        legend_band_height_override=PUBLICATION_SUPPLEMENTARY_TOPOLOGY_COMPARISON_LEGEND_BAND_HEIGHT,
        legend_gap_override=PUBLICATION_LEGEND_GAP,
        legend_scale_override=PUBLICATION_SUPPLEMENTARY_TOPOLOGY_COMPARISON_LEGEND_SCALE,
    )

    supplementary_specs: Sequence[Tuple[str, str, str]] = (
        (
            "hexagonal",
            "supplementary_figure_s15_hexagonal_gpu_scaling.svg",
            "Supplementary Figure S15: Hexagonal Full GPU Scaling",
        ),
        (
            "mst",
            "supplementary_figure_s16_mst_gpu_scaling.svg",
            "Supplementary Figure S16: MST Full GPU Scaling",
        ),
    )
    for topology, filename, title in supplementary_specs:
        output_path = publication_dir / filename
        if output_path.exists():
            output_path.unlink()
        generated[filename] = _compose_single_topology_scaling_figure(
            output_root=output_dir,
            output_path=output_path,
            topology=topology,
            title=title,
            warnings=warnings,
            warning_label=f"{title} performance+efficiency",
            legend_band_height_override=PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_BAND_HEIGHT,
            legend_gap_override=PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_GAP,
            legend_scale_override=PUBLICATION_SUPPLEMENTARY_SINGLE_TOPOLOGY_LEGEND_SCALE,
        )
    return generated


def _write_topology_comparison_outputs(
    merged_modes: Dict[str, MergedModeResults],
    output_dir: Path,
    comparison_output_subdir: str,
    comparison_gpu_count: int,
    skip_plots: bool,
    warnings: List[str],
    methods_to_plot: Sequence[str],
) -> Dict[str, bool]:
    generated: Dict[str, bool] = {mode_name: False for mode_name in MODE_CONFIG}

    if skip_plots:
        warnings.append(
            "Topology comparison requested but --skip_plots is enabled; no comparison figures generated."
        )
        return generated

    if comparison_gpu_count <= 0:
        warnings.append(
            f"Invalid comparison GPU count `{comparison_gpu_count}`; skipping topology comparison figures."
        )
        return generated

    try:
        from floatsom.benchmarks.visualization import (
            plot_topology_comparison_scaling_benchmark,
        )
    except Exception as exc:
        warnings.append(
            f"Could not import topology comparison scaling plotter; skipping comparison figures: {exc}"
        )
        return generated

    comparison_root = output_dir / comparison_output_subdir
    comparison_root.mkdir(parents=True, exist_ok=True)

    for mode_name, mode_cfg in MODE_CONFIG.items():
        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        by_topology: Dict[str, Dict[str, AxisResults]] = defaultdict(dict)
        for (topology, method), axis_results in sorted(mode_results.items()):
            by_topology[topology][method] = axis_results

        has_requested_gpu = any(
            _axis_results_contains_gpu(axis_results, comparison_gpu_count)
            for topology_payload in by_topology.values()
            for axis_results in topology_payload.values()
        )
        if not has_requested_gpu:
            warnings.append(
                f"No {comparison_gpu_count}-GPU entries found for {mode_name}; skipping topology comparison plot."
            )
            continue

        mode_output = comparison_root / mode_name
        mode_output.mkdir(parents=True, exist_ok=True)

        title = f"{mode_cfg['title']} - Topology Scaling"
        try:
            plot_topology_comparison_scaling_benchmark(
                results_by_topology=by_topology,
                output_dir=str(mode_output),
                title=title,
                axis_mode=mode_cfg["save_mode"],
                comparison_gpu_count=comparison_gpu_count,
                show_speedup=False,
                show_error_bars=True,
                show_legend=False,
                methods_to_plot=list(methods_to_plot),
                emit_shared_legend=True,
                sample_min_upper_bound=(
                    PUBLICATION_SAMPLE_AXIS_MIN_UPPER_BOUND
                    if mode_cfg["save_mode"] == "sample"
                    else None
                ),
                dimension_min_upper_bound=(
                    PUBLICATION_DIMENSION_AXIS_MIN_UPPER_BOUND
                    if mode_cfg["save_mode"] == "dimension"
                    else None
                ),
            )
            generated[mode_name] = True
        except Exception as exc:
            warnings.append(
                f"Failed to generate topology comparison plot for {mode_name}: {exc}"
            )

    return generated


def _write_random_comparison_outputs(
    merged_modes: Dict[str, RandomComparisonModeResults],
    output_dir: Path,
    comparison_output_subdir: str,
    comparison_gpu_count: int,
    random_path_pattern: str,
    skip_plots: bool,
    warnings: List[str],
) -> Dict[str, bool]:
    generated: Dict[str, bool] = {mode_name: False for mode_name in MODE_CONFIG}

    if skip_plots:
        warnings.append(
            "Random and full comparison requested but --skip_plots is enabled; no comparison figures generated."
        )
        return generated

    if comparison_gpu_count <= 0:
        warnings.append(
            f"Invalid random comparison GPU count `{comparison_gpu_count}`; skipping random and full figures."
        )
        return generated

    try:
        from floatsom.benchmarks.visualization import (
            plot_topology_comparison_scaling_benchmark,
        )
    except Exception as exc:
        warnings.append(
            f"Could not import comparison scaling plotter; skipping random and full figures: {exc}"
        )
        return generated

    comparison_root = output_dir / comparison_output_subdir
    comparison_root.mkdir(parents=True, exist_ok=True)

    for mode_name, mode_cfg in MODE_CONFIG.items():
        mode_output = comparison_root / mode_name
        mode_output.mkdir(parents=True, exist_ok=True)
        # Prevent stale publication panels from reusing outdated traces
        # (e.g., legacy colors-method overlays) when current generation is skipped.
        for stale_name in (
            _mode_plot_svg_filename(mode_name, "performance"),
            _mode_plot_svg_filename(mode_name, "speedup"),
            _mode_plot_svg_filename(mode_name, "efficiency"),
            "legend_key.svg",
        ):
            stale_path = mode_output / stale_name
            if stale_path.exists():
                stale_path.unlink()

        mode_results = merged_modes.get(mode_name, {})
        if not mode_results:
            continue

        by_topology = _build_random_comparison_topology_payload(mode_results)

        if not by_topology:
            warnings.append(
                f"Missing batch random/full runs for {mode_name} (pattern='{random_path_pattern}'); skipping random and full plot."
            )
            continue
        missing_pairs = [
            topology
            for topology, run_payload in sorted(by_topology.items())
            if not all(run_type in run_payload for run_type in RANDOM_COMPARISON_RUN_TYPES)
        ]
        if missing_pairs:
            warnings.append(
                "Missing random/full pairs for "
                f"{mode_name} topologies {missing_pairs}; skipping random and full plot."
            )
            continue

        has_requested_gpu = any(
            _axis_results_contains_gpu(axis_results, comparison_gpu_count)
            for topology_payload in by_topology.values()
            for axis_results in topology_payload.values()
        )
        if not has_requested_gpu:
            warnings.append(
                f"No {comparison_gpu_count}-GPU entries found for {mode_name}; skipping random and full plot."
            )
            continue

        title = f"{mode_cfg['title']} - Random and Full (batch, topology comparison)"
        emit_speedup = comparison_gpu_count > 1
        try:
            plot_topology_comparison_scaling_benchmark(
                results_by_topology=by_topology,
                output_dir=str(mode_output),
                title=title,
                axis_mode=mode_cfg["save_mode"],
                comparison_gpu_count=comparison_gpu_count,
                show_speedup=emit_speedup,
                show_error_bars=True,
                show_legend=False,
                methods_to_plot=list(RANDOM_COMPARISON_RUN_TYPES),
                emit_shared_legend=True,
                include_performance=not emit_speedup,
                topology_color_overrides=RANDOM_COMPARISON_COLORS,
                sample_min_upper_bound=(
                    PUBLICATION_SAMPLE_AXIS_MIN_UPPER_BOUND
                    if mode_cfg["save_mode"] == "sample"
                    else None
                ),
                dimension_min_upper_bound=(
                    PUBLICATION_DIMENSION_AXIS_MIN_UPPER_BOUND
                    if mode_cfg["save_mode"] == "dimension"
                    else None
                ),
            )
            generated[mode_name] = True
        except Exception as exc:
            warnings.append(
                f"Failed to generate random and full plot for {mode_name}: {exc}"
            )

    return generated


# Summary and orchestration -------------------------------------------------


def _write_summary(
    output_dir: Path,
    input_dirs: List[Path],
    ingested_files: Dict[str, int],
    merged_pairs: Dict[str, int],
    warnings: List[str],
    comparison_enabled: bool = False,
    comparison_gpu_count: Optional[int] = None,
    comparison_output_subdir: str = "",
    comparison_generated: Optional[Dict[str, bool]] = None,
    random_comparison_enabled: bool = False,
    random_comparison_gpu_count: Optional[int] = None,
    random_comparison_output_subdir: str = "",
    random_comparison_pattern: str = "",
    random_comparison_generated: Optional[Dict[str, bool]] = None,
    publication_generated: Optional[Dict[str, bool]] = None,
    publication_assets_figures: Optional[Dict[str, str]] = None,
) -> None:
    summary_path = output_dir / "harmonization_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("GPU Scaling Result Harmonization Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Output directory: {output_dir}\n\n")

        handle.write("Input result directories:\n")
        for path in input_dirs:
            handle.write(f"  - {path}\n")
        handle.write("\n")

        handle.write("Ingested result files:\n")
        for mode_name in MODE_CONFIG:
            handle.write(f"  - {mode_name}: {ingested_files.get(mode_name, 0)} JSON file(s)\n")
        handle.write("\n")

        handle.write("Merged topology/method groups:\n")
        for mode_name in MODE_CONFIG:
            handle.write(f"  - {mode_name}: {merged_pairs.get(mode_name, 0)} group(s)\n")
        handle.write("\n")

        if comparison_enabled:
            handle.write("Topology comparison suite: enabled\n")
            if comparison_gpu_count is not None:
                handle.write(f"Comparison GPU count: {comparison_gpu_count}\n")
            if comparison_output_subdir:
                handle.write(f"Comparison output subdirectory: {comparison_output_subdir}\n")
            if comparison_generated is not None:
                handle.write("Generated comparison figures:\n")
                for mode_name in MODE_CONFIG.keys():
                    generated_flag = comparison_generated.get(mode_name, False)
                    status = "yes" if generated_flag else "no"
                    handle.write(f"  - {mode_name}: {status}\n")
            handle.write("\n")
        else:
            handle.write("Topology comparison suite: disabled\n\n")

        if random_comparison_enabled:
            handle.write("Random and full comparison suite: enabled\n")
            if random_comparison_gpu_count is not None:
                handle.write(f"Random comparison GPU count: {random_comparison_gpu_count}\n")
            if random_comparison_pattern:
                handle.write(f"Random path pattern: {random_comparison_pattern}\n")
            if random_comparison_output_subdir:
                handle.write(
                    f"Random comparison output subdirectory: {random_comparison_output_subdir}\n"
                )
            if random_comparison_generated is not None:
                handle.write("Generated random and full figures:\n")
                for mode_name in MODE_CONFIG.keys():
                    generated_flag = random_comparison_generated.get(mode_name, False)
                    status = "yes" if generated_flag else "no"
                    handle.write(f"  - {mode_name}: {status}\n")
            handle.write("\n")
        else:
            handle.write("Random and full comparison suite: disabled\n\n")

        if publication_generated is not None:
            handle.write("Publication panel composites:\n")
            for figure_key in PUBLICATION_FIGURE_ORDER:
                status = "yes" if publication_generated.get(figure_key, False) else "no"
                handle.write(f"  - {figure_key}: {status}\n")
            handle.write("\n")

        if publication_assets_figures is not None:
            handle.write("Paper assets figure sync:\n")
            if publication_assets_figures:
                for src_name, dst_path in sorted(publication_assets_figures.items()):
                    handle.write(f"  - {src_name} -> {dst_path}\n")
            else:
                handle.write("  - no files copied\n")
            handle.write("\n")

        if warnings:
            handle.write("Warnings:\n")
            for warning in warnings:
                handle.write(f"  - {warning}\n")
        else:
            handle.write("Warnings: none\n")


def _resolve_input_dirs(args: argparse.Namespace) -> List[Path]:
    if not args.inputs and not args.search_root:
        raise ValueError("Provide at least one input via --inputs or --search_root.")

    input_dirs = discover_results_dirs(args.inputs, args.search_root)
    if not input_dirs:
        raise FileNotFoundError(
            "No benchmark result directories discovered. "
            "Expected directories containing mode folders such as "
            "`dimension_scaling`, `sample_scaling`, or `grid_size_scaling`."
        )
    return input_dirs


def _prepare_output_dir(output_dir_raw: str, overwrite: bool) -> Path:
    output_dir = Path(output_dir_raw).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to allow writing into this directory."
        )
    return output_dir


def _ingest_harmonization_inputs(
    input_dirs: Sequence[Path],
    random_path_pattern: str,
    warnings: List[str],
) -> Tuple[
    Dict[str, MergedModeResults],
    Dict[str, RandomComparisonModeResults],
    Dict[str, int],
]:
    merged: Dict[str, MergedModeResults] = {mode_name: {} for mode_name in MODE_CONFIG}
    random_comparison_merged: Dict[str, RandomComparisonModeResults] = {
        mode_name: {} for mode_name in MODE_CONFIG
    }
    ingested_files: Dict[str, int] = defaultdict(int)

    for source_dir in input_dirs:
        run_type = _classify_run_type(source_dir, random_path_pattern)
        for mode_name, mode_cfg in MODE_CONFIG.items():
            mode_dir = source_dir / mode_name
            if not mode_dir.is_dir():
                continue

            prefix = mode_cfg["prefix"]
            for json_path in sorted(mode_dir.glob(f"{prefix}_results_*.json")):
                parsed = _parse_topology_and_method(json_path, prefix)
                if parsed is None:
                    warnings.append(f"Could not infer topology/method from {json_path}")
                    continue

                topology, method = parsed
                loaded = _load_mode_json(json_path, warnings)
                if not loaded:
                    continue

                if run_type == "full":
                    merged_key = (topology, method)
                    merged[mode_name].setdefault(merged_key, {})
                    _merge_axis_results(merged[mode_name][merged_key], loaded)
                    ingested_files[mode_name] += 1

                normalized_topology = _normalize_topology_name(topology)
                normalized_method = _normalize_method_name(method)
                if normalized_method in RANDOM_COMPARISON_METHODS:
                    random_key = (run_type, normalized_topology, normalized_method)
                    random_comparison_merged[mode_name].setdefault(random_key, {})
                    _merge_axis_results(random_comparison_merged[mode_name][random_key], loaded)

    return merged, random_comparison_merged, ingested_files


def _write_harmonized_scaling_outputs(
    merged: Dict[str, MergedModeResults],
    output_dir: Path,
    skip_plots: bool,
    warnings: List[str],
    primary_methods: Sequence[str],
) -> None:
    for mode_name in MODE_CONFIG:
        _write_mode_outputs(
            mode_name=mode_name,
            mode_results=merged[mode_name],
            output_dir=output_dir,
            skip_plots=skip_plots,
            warnings=warnings,
            methods_to_plot=primary_methods,
        )


def _generate_optional_output_suites(
    args: argparse.Namespace,
    merged: Dict[str, MergedModeResults],
    random_comparison_merged: Dict[str, RandomComparisonModeResults],
    input_dirs: Sequence[Path],
    output_dir: Path,
    warnings: List[str],
    primary_methods: Sequence[str],
    figure_8_tripanel_results_dirs: Dict[int, List[Path]],
) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool], Dict[str, str]]:
    comparison_generated: Dict[str, bool] = {mode_name: False for mode_name in MODE_CONFIG}
    if args.topology_comparison:
        comparison_generated = _write_topology_comparison_outputs(
            merged_modes=merged,
            output_dir=output_dir,
            comparison_output_subdir=args.comparison_output_subdir,
            comparison_gpu_count=args.comparison_gpu_count,
            skip_plots=args.skip_plots,
            warnings=warnings,
            methods_to_plot=primary_methods,
        )

    random_comparison_generated: Dict[str, bool] = {mode_name: False for mode_name in MODE_CONFIG}
    if args.random_comparison:
        random_comparison_generated = _write_random_comparison_outputs(
            merged_modes=random_comparison_merged,
            output_dir=output_dir,
            comparison_output_subdir=args.random_comparison_output_subdir,
            comparison_gpu_count=args.random_comparison_gpu_count,
            random_path_pattern=args.random_path_pattern,
            skip_plots=args.skip_plots,
            warnings=warnings,
        )

    publication_generated: Dict[str, bool] = {figure_key: False for figure_key in PUBLICATION_FIGURE_ORDER}
    publication_assets_figures: Dict[str, str] = {}
    _write_publication_support_tables(
        merged_modes=merged,
        input_dirs=input_dirs,
        output_dir=output_dir,
        comparison_gpu_count=args.comparison_gpu_count,
        warnings=warnings,
    )
    if not args.skip_plots:
        publication_generated = _write_publication_panel_outputs(
            merged_modes=merged,
            random_modes=random_comparison_merged,
            output_dir=output_dir,
            comparison_output_subdir=args.comparison_output_subdir,
            random_comparison_output_subdir=args.random_comparison_output_subdir,
            comparison_gpu_count=args.comparison_gpu_count,
            figure_8_tripanel_results_dirs=figure_8_tripanel_results_dirs,
            random_path_pattern=args.random_path_pattern,
            warnings=warnings,
        )
        _write_topology_scaling_supplementary_outputs(
            output_dir=output_dir,
            warnings=warnings,
        )
        publication_assets_figures = _sync_publication_figures_to_paper_assets(
            publication_dir=output_dir / "publication_figures",
            warnings=warnings,
        )

    return (
        comparison_generated,
        random_comparison_generated,
        publication_generated,
        publication_assets_figures,
    )


def _print_completion_summary(
    *,
    output_dir: Path,
    input_dirs: Sequence[Path],
    ingested_files: Dict[str, int],
    merged_pairs: Dict[str, int],
    comparison_generated: Dict[str, bool],
    comparison_output_subdir: str,
    topology_comparison_enabled: bool,
    random_comparison_generated: Dict[str, bool],
    random_comparison_output_subdir: str,
    random_comparison_enabled: bool,
    publication_generated: Dict[str, bool],
    publication_assets_figures: Dict[str, str],
    skip_plots: bool,
    warnings: Sequence[str],
) -> None:
    print("Harmonization complete.")
    print(f"Output directory: {output_dir}")
    print(f"Discovered result directories: {len(input_dirs)}")
    for mode_name in MODE_CONFIG:
        print(
            f"  - {mode_name}: {ingested_files.get(mode_name, 0)} file(s), "
            f"{merged_pairs.get(mode_name, 0)} merged group(s)"
        )

    if topology_comparison_enabled:
        comparison_root = output_dir / comparison_output_subdir
        print(f"  - topology comparison output: {comparison_root}")
        for mode_name in MODE_CONFIG.keys():
            status = "generated" if comparison_generated.get(mode_name, False) else "skipped"
            print(f"    · {mode_name}: {status}")

    if random_comparison_enabled:
        random_comparison_root = output_dir / random_comparison_output_subdir
        print(f"  - random and full comparison output: {random_comparison_root}")
        for mode_name in MODE_CONFIG.keys():
            status = "generated" if random_comparison_generated.get(mode_name, False) else "skipped"
            print(f"    · {mode_name}: {status}")

    publication_root = output_dir / "publication_figures"
    print(f"  - publication figures output: {publication_root}")
    for figure_key in PUBLICATION_FIGURE_ORDER:
        status = "generated" if publication_generated.get(figure_key, False) else "skipped"
        print(f"    · {figure_key}: {status}")

    if publication_assets_figures:
        print("  - paper assets figures synced:")
        for src_name, dst_path in sorted(publication_assets_figures.items()):
            print(f"    · {src_name} -> {dst_path}")
    elif not skip_plots:
        print("  - paper assets figures synced: none")

    print(f"Warnings: {len(warnings)} (see harmonization_summary.txt)")


def main() -> None:
    args = parse_args()
    primary_methods = _resolve_primary_methods(bool(args.colors))
    warnings: List[str] = []
    figure_8_tripanel_results_dirs = _resolve_figure_8_tripanel_results_dirs(
        args,
        warnings=warnings,
    )
    input_dirs = _resolve_input_dirs(args)
    output_dir = _prepare_output_dir(args.output_dir, args.overwrite)
    merged, random_comparison_merged, ingested_files = _ingest_harmonization_inputs(
        input_dirs=input_dirs,
        random_path_pattern=args.random_path_pattern,
        warnings=warnings,
    )
    _write_harmonized_scaling_outputs(
        merged=merged,
        output_dir=output_dir,
        skip_plots=args.skip_plots,
        warnings=warnings,
        primary_methods=primary_methods,
    )
    (
        comparison_generated,
        random_comparison_generated,
        publication_generated,
        publication_assets_figures,
    ) = _generate_optional_output_suites(
        args=args,
        merged=merged,
        random_comparison_merged=random_comparison_merged,
        input_dirs=input_dirs,
        output_dir=output_dir,
        warnings=warnings,
        primary_methods=primary_methods,
        figure_8_tripanel_results_dirs=figure_8_tripanel_results_dirs,
    )

    merged_pairs = {mode_name: len(results) for mode_name, results in merged.items()}
    _write_summary(
        output_dir=output_dir,
        input_dirs=input_dirs,
        ingested_files=ingested_files,
        merged_pairs=merged_pairs,
        warnings=warnings,
        comparison_enabled=args.topology_comparison,
        comparison_gpu_count=args.comparison_gpu_count,
        comparison_output_subdir=args.comparison_output_subdir,
        comparison_generated=comparison_generated,
        random_comparison_enabled=args.random_comparison,
        random_comparison_gpu_count=args.random_comparison_gpu_count,
        random_comparison_output_subdir=args.random_comparison_output_subdir,
        random_comparison_pattern=args.random_path_pattern,
        random_comparison_generated=random_comparison_generated,
        publication_generated=publication_generated,
        publication_assets_figures=publication_assets_figures,
    )
    _print_completion_summary(
        output_dir=output_dir,
        input_dirs=input_dirs,
        ingested_files=ingested_files,
        merged_pairs=merged_pairs,
        comparison_generated=comparison_generated,
        comparison_output_subdir=args.comparison_output_subdir,
        topology_comparison_enabled=args.topology_comparison,
        random_comparison_generated=random_comparison_generated,
        random_comparison_output_subdir=args.random_comparison_output_subdir,
        random_comparison_enabled=args.random_comparison,
        publication_generated=publication_generated,
        publication_assets_figures=publication_assets_figures,
        skip_plots=args.skip_plots,
        warnings=warnings,
    )


if __name__ == "__main__":
    main()
