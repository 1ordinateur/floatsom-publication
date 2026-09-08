#!/usr/bin/env python3
"""
Benchmark FloatSOM vs XPySOM on the Optuna full dataset panel.

Protocol:
- Sampling mode: full
- Algorithm/mode: batch / full_batch
- FloatSOM topologies: hexagonal, MST, RNG
- XPySOM topology: hexagonal
- One run per seed (no top-k aggregation)
- Metrics: QE train, QE holdout, balanced QE, and training wall time

Outputs:
- Raw per-run log CSV
- Supplementary-style paired summary TSV with percent-improvement stats and raw p-values
- XPySOM calibration figure (copied into paper assets for the hexagonal topology)
- Final MST vs RNG comparison figure (FloatSOM topology-only)
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Keep worker progress lines distinct during Ray-backed staging/debugging unless
# the caller explicitly chose a different dedup setting.
os.environ.setdefault("RAY_DEDUP_LOGS", "0")
import cupy as cp
import numpy as np
import pandas as pd
from scipy import stats

from floatsom.base.floatsom_factories import create_floatsom
from floatsom_benchmarks.evaluation.sklearn_datasets import generate_sklearn_dataset
from floatsom_benchmarks.optuna.config.benchmark_config import Phase3BenchmarkConfig
from floatsom_benchmarks.optuna.core.objective import create_floatsom_params
from floatsom_benchmarks.optuna.run_matched_default_floatsom_batch import (
    _resolve_effective_manual_fixed_params,
    _resolve_manual_fixed_params,
    _resolve_manual_fixed_params_by_sampling_topology,
    _resolve_manual_fixed_params_by_topology,
    _validate_manual_fixed_params_for_scope,
)
from floatsom.evaluation.quantization import QuantizationError
from floatsom.floatsom_params import FloatSOMParams

METHOD_FLOATSOM = "floatsom"
METHOD_XPYSOM = "xpysom"
COMPARISON_LABEL = "FloatSOM vs XPySOM"
MST_VS_RNG_COMPARISON_LABEL = "MST vs RNG"
SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES: tuple[str, ...] = ("hexagonal", "mst", "rng")
DEFAULT_FLOATSOM_BATCH_TOPOLOGIES: tuple[str, ...] = SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES
OPTUNA_TRAIN_FRACTION = 0.7
PAPER_SUPP_TABLE_FILENAME = "supp_xpysom_calibration_qe_hexagonal.tsv"
PAPER_SUPP_FIGURE_FILENAME = "supp_fig_s3.svg"
FOREST_STATS_FILENAME = "xpysom_calibration_forest_stats.csv"
MST_VS_RNG_FIGURE_FILENAME = "fig_mst_vs_rng_final.svg"
MST_VS_RNG_FOREST_STATS_FILENAME = "mst_vs_rng_final_forest_stats.csv"
MST_VS_RNG_XLABEL = "Median % Improvement vs RNG (positive means MST is better)"
DEFAULT_SKLEARN_DATA_CANDIDATES: tuple[str, ...] = (
    "/g/data/eu59/SIFEAN/sklearn_data",
    "/g/data/eu59/SIFEAN/sfa/sklearn_data",
)
CANONICAL_DATASET_ORDER: tuple[str, ...] = (
    "blobs",
    "breast_cancer",
    "circles",
    "digits",
    "iris",
    "moons",
    "olivetti_faces",
    "s_curve",
    "swiss_roll",
    "wine",
    "diabetes",
    "california_housing",
    "covertype",
    "kddcup99",
)
DATASET_SAMPLE_SIZE_FALLBACK: dict[str, int] = {
    "swiss_roll": 30000,
    "moons": 30000,
    "circles": 30000,
    "blobs": 30000,
    "s_curve": 30000,
    "breast_cancer": 569,
    "wine": 178,
    "iris": 150,
    "digits": 1797,
    "olivetti_faces": 400,
    "diabetes": 442,
    "california_housing": 20640,
    "covertype": 581012,
    "kddcup99": 494021,
}


def _require_gpu() -> None:
    try:
        gpu_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GPU check failed via CuPy. This benchmark is GPU-only.") from exc
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU detected. This benchmark is GPU-only.")


def _extract_xpysom_initial_weights(xpysom_model: object) -> np.ndarray:
    """
    Extract XPySOM's initial weights tensor in its native layout.

    XPySOM stores weights as (x, y, input_dim). We keep that here and convert to
    FloatSOM's node ordering separately.
    """
    getter = getattr(xpysom_model, "get_weights", None)
    if callable(getter):
        weights = getter()
    elif hasattr(xpysom_model, "_weights"):
        weights = getattr(xpysom_model, "_weights")
    else:
        raise TypeError(
            "Unable to extract XPySOM initial weights. Expected xpysom_model.get_weights() "
            "or xpysom_model._weights to exist."
        )

    weights = np.asarray(weights)
    if weights.ndim != 3:
        raise ValueError(
            "Unexpected XPySOM weight tensor rank. "
            f"Expected 3D (x, y, input_dim), got shape={weights.shape}."
        )
    return weights


def _xpysom_weights_to_floatsom_init(
    xpysom_weights: np.ndarray,
    *,
    expected_grid_size: int,
    expected_input_dim: int,
) -> np.ndarray:
    """
    Convert XPySOM weights (x, y, dim) into FloatSOM weights (n_nodes, dim).

    FloatSOM flattens its meshgrid-derived coordinate grid in (y, x) order, i.e.
    node_index = y * grid_size + x.

    XPySOM stores weights in (x, y, dim) order, so we transpose (x, y) -> (y, x)
    before flattening to align neurons by the same Euclidean (x, y) coordinate.

    Note: our hexagonal coordinate grid offsets rows by +0.5, while XPySOM offsets
    the same parity rows by -0.5. Those coordinate systems are related by an x-axis
    reflection, so we also flip the x axis to keep neuron positions aligned.
    """
    grid_size = int(expected_grid_size)
    input_dim = int(expected_input_dim)
    if grid_size <= 0:
        raise ValueError(f"expected_grid_size must be positive, got {grid_size}")
    if input_dim <= 0:
        raise ValueError(f"expected_input_dim must be positive, got {input_dim}")

    weights_np = np.asarray(xpysom_weights, dtype=np.float32)
    expected_shape = (grid_size, grid_size, input_dim)
    if tuple(weights_np.shape) != expected_shape:
        raise ValueError(
            "XPySOM weight tensor shape mismatch. "
            f"Expected {expected_shape}, got {tuple(weights_np.shape)}."
        )

    weights_np = weights_np[::-1, :, :]
    weights_np = np.transpose(weights_np, (1, 0, 2))
    weights_flat = weights_np.reshape(grid_size * grid_size, input_dim)
    return np.ascontiguousarray(weights_flat, dtype=np.float32)


def _inject_floatsom_initial_weights(
    floatsom_model: object,
    initial_weights: cp.ndarray | np.ndarray,
) -> None:
    """
    Force FloatSOM to start from a specific initial weight matrix.

    FloatSOM always calls topology.initialize_weights(...) during training init,
    so we monkeypatch that method per-model. For MST/RNG we also rebuild the
    adjacency from the injected weights via topology._update_mst(...).
    """
    topology = getattr(floatsom_model, "topology", None)
    if topology is None:
        raise TypeError("FloatSOM model has no .topology; cannot inject initial weights.")

    total_nodes = int(getattr(topology, "total_nodes", 0))
    if total_nodes <= 0:
        raise ValueError(f"Invalid FloatSOM topology.total_nodes={total_nodes}.")

    weights = cp.asarray(initial_weights, dtype=cp.float32)
    if weights.ndim != 2:
        raise ValueError(f"Expected initial_weights to be 2D (n_nodes, dim), got shape={weights.shape}.")
    if int(weights.shape[0]) != total_nodes:
        raise ValueError(
            "Initial weight node-count mismatch for FloatSOM injection. "
            f"Expected n_nodes={total_nodes}, got shape={weights.shape}."
        )

    # Ensure each model gets its own mutable weight matrix.
    injected = weights.copy()

    def _initialize_weights_override(_data: object | None = None) -> cp.ndarray:
        update_mst = getattr(topology, "_update_mst", None)
        if callable(update_mst):
            update_mst(injected)
        return injected

    topology.initialize_weights = _initialize_weights_override


def _resolve_datasets(requested: Sequence[str] | None) -> List[str]:
    available = list(Phase3BenchmarkConfig().datasets)
    if not requested:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown dataset(s): {missing}. Available: {available}")
    return list(requested)


def _resolve_seeds(num_seeds: int, base_seed: int, explicit_seeds: Sequence[int] | None) -> List[int]:
    if explicit_seeds:
        return [int(seed) for seed in explicit_seeds]
    return [int(base_seed) + idx for idx in range(int(num_seeds))]


def _resolve_floatsom_topologies(requested: Sequence[str] | None) -> List[str]:
    if not requested:
        return list(DEFAULT_FLOATSOM_BATCH_TOPOLOGIES)
    normalized = [str(name).strip().lower() for name in requested]
    missing = [name for name in normalized if name not in SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES]
    if missing:
        raise ValueError(
            f"Unknown FloatSOM topology option(s): {missing}. "
            f"Supported: {list(SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES)}"
        )
    ordered_unique: List[str] = []
    for candidate in normalized:
        if candidate not in ordered_unique:
            ordered_unique.append(candidate)
    return ordered_unique


def _configure_sklearn_data_home(data_home: str | None) -> str | None:
    env_home = os.environ.get("SCIKIT_LEARN_DATA")
    if data_home is not None:
        resolved_path = Path(data_home).expanduser()
    elif env_home:
        resolved_path = Path(env_home).expanduser()
    else:
        candidate_path = None
        for candidate in DEFAULT_SKLEARN_DATA_CANDIDATES:
            path = Path(candidate)
            if path.exists() or path.parent.exists():
                candidate_path = path
                break
        resolved_path = candidate_path if candidate_path is not None else (Path.home() / "scikit_learn_data")

    resolved = str(resolved_path.resolve())
    os.environ["SCIKIT_LEARN_DATA"] = resolved
    return resolved


def _ensure_olivetti_cache_available(datasets: Sequence[str], data_home: str | None) -> None:
    if "olivetti_faces" not in datasets:
        return
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn olivetti loader is unavailable in this environment. "
            "Cannot verify required olivetti_faces cache."
        ) from exc

    try:
        fetch_olivetti_faces(
            data_home=data_home,
            shuffle=False,
            random_state=0,
            download_if_missing=False,
            return_X_y=False,
        )
    except Exception as exc:
        cache_root = data_home or os.environ.get("SCIKIT_LEARN_DATA", "<default sklearn cache>")
        raise RuntimeError(
            "olivetti_faces cache is missing or unreadable for offline job execution. "
            f"SCIKIT_LEARN_DATA={cache_root}. "
            "Pre-populate the cache on a networked node, then rerun."
        ) from exc


def _train_holdout_split(
    data: cp.ndarray,
    seed: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Match the Optuna split policy: deterministic 70/30 via seed permutation."""
    n_samples = int(data.shape[0])
    n_train = int(OPTUNA_TRAIN_FRACTION * n_samples)
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    holdout_indices = indices[n_train:]
    return data[train_indices], data[holdout_indices]


def _create_floatsom_batch_topology(
    *,
    input_dim: int,
    seed: int,
    grid_size: int,
    epochs: int,
    topology_type: str,
    manual_fixed_params: Optional[Dict[str, Any]] = None,
) -> tuple[object, FloatSOMParams]:
    resolved_topology = str(topology_type).strip().lower()
    if resolved_topology not in SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES:
        raise ValueError(
            f"Unsupported FloatSOM topology '{topology_type}'. "
            f"Supported: {list(SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES)}"
        )

    benchmark_params: Dict[str, Any] = dict(manual_fixed_params or {})
    benchmark_params.update(
        {
            "sampling_method": "full",
            "processing_method": "batch",
            "batch_mode": "full_batch",
            "topology_type": resolved_topology,
            "grid_size": int(grid_size),
            "iterations": int(epochs),
            "seed": int(seed),
        }
    )
    # Preserve strict epoch parity with XPySOM unless the caller explicitly overrides it.
    benchmark_params.setdefault("min_iterations", int(epochs) + 1)
    params = create_floatsom_params(
        cp.empty((1, int(input_dim)), dtype=cp.float32),
        benchmark_params,
    )
    return create_floatsom(params), params


def _resolve_floatsom_fixed_param_overrides(
    *,
    topologies: Sequence[str],
    fixed_param_entries: Optional[Sequence[str]],
    fixed_params_json: Optional[str],
    fixed_params_by_topology_json: Optional[str],
    fixed_params_by_sampling_topology_json: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    manual_fixed_params = _resolve_manual_fixed_params(fixed_param_entries, fixed_params_json)
    manual_fixed_params_by_topology = _resolve_manual_fixed_params_by_topology(
        topologies=topologies,
        json_path=fixed_params_by_topology_json,
    )
    manual_fixed_params_by_sampling_topology = _resolve_manual_fixed_params_by_sampling_topology(
        sampling_methods=["full"],
        topologies=topologies,
        json_path=fixed_params_by_sampling_topology_json,
        allow_unselected_keys=True,
    )

    _validate_manual_fixed_params_for_scope(
        manual_fixed_params,
        processing_method="batch",
        topologies=topologies,
    )
    for topology, topology_params in manual_fixed_params_by_topology.items():
        _validate_manual_fixed_params_for_scope(
            topology_params,
            processing_method="batch",
            topologies=[topology],
        )
    for topology_map in manual_fixed_params_by_sampling_topology.values():
        for topology, topology_params in topology_map.items():
            _validate_manual_fixed_params_for_scope(
                topology_params,
                processing_method="batch",
                topologies=[topology],
            )

    return (
        manual_fixed_params,
        manual_fixed_params_by_topology,
        manual_fixed_params_by_sampling_topology,
    )


def _floatsom_run_profile_label(
    *,
    manual_fixed_params: Dict[str, Any],
    manual_fixed_params_by_topology: Dict[str, Dict[str, Any]],
    manual_fixed_params_by_sampling_topology: Dict[str, Dict[str, Dict[str, Any]]],
) -> str:
    has_any_fixed_overrides = (
        bool(manual_fixed_params)
        or bool(manual_fixed_params_by_topology)
        or bool(manual_fixed_params_by_sampling_topology)
    )
    return "manual_fixed" if has_any_fixed_overrides else "true_default"


def _extract_floatsom_param_columns(params: FloatSOMParams) -> Dict[str, object]:
    return {
        "param_initial_radius": float(params.initial_radius),
        "param_radius_decay_type": str(params.radius_decay_type),
        "param_initialization_method": str(params.initialization_method),
        "param_use_momentum": bool(params.processing_config.enable_momentum),
        "param_momentum_init": float(params.processing_config.initial_momentum),
    }


def _create_xpysom_hex(
    *,
    input_dim: int,
    grid_size: int,
    seed: int,
    n_parallel: int | None = None,
) -> object:
    try:
        from xpysom import XPySom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xpysom is not installed. Install it before running this benchmark."
        ) from exc

    try:
        kwargs = {
            "x": int(grid_size),
            "y": int(grid_size),
            "input_len": int(input_dim),
            "topology": "hexagonal",
            "random_seed": int(seed),
        }
        if n_parallel is not None:
            kwargs["n_parallel"] = int(n_parallel)
        return XPySom(
            **kwargs,
        )
    except TypeError as exc:
        raise TypeError(
            "Failed to construct xpysom.XPySom with expected API "
            "(x, y, input_len, topology, [optional n_parallel])."
        ) from exc


def _train_floatsom(model: object, train_data: cp.ndarray) -> float:
    start = time.perf_counter()
    model.train(train_data)
    return float(time.perf_counter() - start)


def _train_xpysom(model: object, train_data: cp.ndarray, epochs: int) -> float:
    start = time.perf_counter()
    model.train(train_data, num_epochs=int(epochs), verbose=False)
    return float(time.perf_counter() - start)


def _compute_metric_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    training_epochs: int,
    train_time_s: float,
    qe_train: float,
    qe_holdout: float,
    algorithm: str = "batch",
    batch_mode: str | None = "full_batch",
    architecture: str = "hexagonal",
    comparison_topology: str | None = None,
    run_profile: str = "",
    run_label: str = "",
    extra_fields: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row = {
        "dataset": str(dataset),
        "seed": int(seed),
        "method": str(method),
        "training_epochs": int(training_epochs),
        "train_time_s": float(train_time_s),
        "quantization_error_train": float(qe_train),
        "quantization_error_holdout": float(qe_holdout),
        "balanced_qe_raw": float((qe_train + qe_holdout) / 2.0),
        "sampling_method": "full",
        "algorithm": str(algorithm),
        "batch_mode": "" if batch_mode is None else str(batch_mode),
        "architecture": str(architecture),
        "comparison_topology": str(comparison_topology) if comparison_topology is not None else str(architecture),
        "train_test_split": float(OPTUNA_TRAIN_FRACTION),
        "run_profile": str(run_profile),
        "run_label": str(run_label),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _wilcoxon_two_sided_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                nonzero,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                mode="auto",
            ).pvalue
        )
    except ValueError:
        return float("nan")


def _paired_pct_improvements(floatsom_values: np.ndarray, xpysom_values: np.ndarray) -> np.ndarray:
    baseline = np.abs(np.asarray(xpysom_values, dtype=float))
    pct = np.where(
        baseline > 0.0,
        ((np.asarray(xpysom_values, dtype=float) - np.asarray(floatsom_values, dtype=float)) / baseline) * 100.0,
        np.nan,
    )
    pct = np.asarray(pct, dtype=float)
    return pct[np.isfinite(pct)]


def _ordered_dataset_labels(dataset_values: Sequence[str]) -> List[str]:
    names = [str(value) for value in dataset_values if pd.notna(value)]
    if not names:
        return []
    unique_names = set(names)
    ordered = [name for name in CANONICAL_DATASET_ORDER if name in unique_names]
    extras = sorted(name for name in unique_names if name not in CANONICAL_DATASET_ORDER and name != "GLOBAL")
    ordered.extend(extras)
    if "GLOBAL" in unique_names:
        ordered.append("GLOBAL")
    return ordered


def _resolve_dataset_sample_sizes(dataset_values: Sequence[str]) -> np.ndarray:
    sample_sizes: List[float] = []
    for dataset in dataset_values:
        dataset_key = str(dataset).strip().lower()
        sample_size = DATASET_SAMPLE_SIZE_FALLBACK.get(dataset_key)
        sample_sizes.append(float(sample_size) if sample_size is not None else np.nan)
    return np.asarray(sample_sizes, dtype=float)


def _dataset_index_lookup(dataset_values: Sequence[str]) -> Dict[str, int]:
    dataset_names = sorted(
        {
            str(value).strip()
            for value in dataset_values
            if pd.notna(value) and str(value).strip() and str(value).strip().upper() != "GLOBAL"
        }
    )
    if not dataset_names:
        return {}

    metadata = pd.DataFrame({"dataset": dataset_names})
    metadata["sample_size"] = _resolve_dataset_sample_sizes(metadata["dataset"].tolist())
    metadata = metadata.sort_values(["sample_size", "dataset"], ascending=[True, True], na_position="last")
    metadata = metadata.reset_index(drop=True)
    metadata["dataset_index"] = np.arange(1, len(metadata) + 1, dtype=int)
    return {
        str(row["dataset"]): int(row["dataset_index"])
        for _, row in metadata.iterrows()
    }


def _pvalue_to_stars(value: float) -> str:
    if value is None or pd.isna(value):
        return "ns"
    value = float(value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def _row_is_significant(row: pd.Series, alpha: float) -> bool:
    p_raw = row.get("p_value", np.nan)
    if p_raw is None or pd.isna(p_raw):
        return False
    try:
        p_value = float(p_raw)
    except (TypeError, ValueError):
        return False
    return np.isfinite(p_value) and p_value < alpha


@lru_cache(maxsize=128)
def _wilcoxon_rank_sum_cdf(n_nonzero: int) -> np.ndarray:
    max_rank_sum = int(n_nonzero * (n_nonzero + 1) // 2)
    probs = np.zeros(max_rank_sum + 1, dtype=float)
    probs[0] = 1.0
    for rank in range(1, int(n_nonzero) + 1):
        updated = np.zeros_like(probs)
        updated += 0.5 * probs
        updated[rank:] += 0.5 * probs[:-rank]
        probs = updated
    return np.cumsum(probs)


@lru_cache(maxsize=256)
def _wilcoxon_two_sided_cutoff(n_nonzero: int, alpha: float) -> int:
    if n_nonzero <= 0 or alpha <= 0.0:
        return -1
    max_rank_sum = int(n_nonzero * (n_nonzero + 1) // 2)
    half_rank_sum = int(max_rank_sum // 2)
    tail_alpha = float(alpha) / 2.0
    strict_threshold = np.nextafter(tail_alpha, float("-inf"))
    cdf = _wilcoxon_rank_sum_cdf(int(n_nonzero))
    accepted = np.flatnonzero(cdf[: half_rank_sum + 1] <= strict_threshold)
    if accepted.size == 0:
        return -1
    return int(accepted[-1])


def _walsh_averages(nonzero: np.ndarray) -> np.ndarray:
    pairwise = (nonzero[:, None] + nonzero[None, :]) / 2.0
    return pairwise[np.triu_indices(nonzero.size)].astype(float, copy=False)


def _wilcoxon_location_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")

    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 0.0, 0.0, 0.0
    if nonzero.size == 1:
        point = float(nonzero[0])
        return point, point, point

    walsh = _walsh_averages(nonzero)
    count = int(walsh.size)
    cutoff = _wilcoxon_two_sided_cutoff(int(nonzero.size), float(alpha))
    if cutoff < 0:
        lower_idx = 0
        upper_idx = count - 1
    else:
        lower_idx = min(max(cutoff, 0), count - 1)
        upper_idx = min(max(count - cutoff - 1, lower_idx), count - 1)

    middle = count // 2
    if count % 2 == 1:
        partition_idx = sorted({lower_idx, upper_idx, middle})
        selected = np.partition(walsh, partition_idx)
        location = float(selected[middle])
    else:
        middle_low = middle - 1
        partition_idx = sorted({lower_idx, upper_idx, middle_low, middle})
        selected = np.partition(walsh, partition_idx)
        location = float((selected[middle_low] + selected[middle]) / 2.0)

    ci_low = float(selected[lower_idx])
    ci_high = float(selected[upper_idx])
    return location, ci_low, ci_high


def _wilcoxon_location_shift(values: np.ndarray) -> float:
    return float(_wilcoxon_location_ci(values)[0])


def _paired_pct_improvement_summary_generic(
    runs_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    win_label_a: str,
    win_label_b: str,
    win_rate_label_a: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pooled_pct: List[float] = []
    pooled_delta_raw: List[float] = []
    value_a_str = str(value_a)
    value_b_str = str(value_b)
    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(index="seed", columns=compare_col, values=metric_col, aggfunc="first")
        if pivot.empty:
            continue
        pivot.columns = pivot.columns.astype(str)
        required_cols = {value_a_str, value_b_str}
        if not required_cols.issubset(set(pivot.columns.tolist())):
            continue

        paired = pivot[[value_a_str, value_b_str]].dropna()
        if paired.empty:
            continue

        values_a = paired[value_a_str].to_numpy(dtype=float)
        values_b = paired[value_b_str].to_numpy(dtype=float)
        deltas = values_a - values_b
        pct = _paired_pct_improvements(values_a, values_b)
        if pct.size == 0:
            continue

        pooled_pct.extend(float(value) for value in pct.tolist())
        pooled_delta_raw.extend(float(value) for value in deltas[np.isfinite(deltas)].tolist())
        location_pct, ci_low_pct, ci_high_pct = _wilcoxon_location_ci(pct)
        location_raw = _wilcoxon_location_shift(deltas)
        wins_a = int(np.sum(pct > 0.0))
        wins_b = int(np.sum(pct < 0.0))
        ties = int(len(pct) - wins_a - wins_b)
        win_rate_a = float(wins_a) / float(len(pct)) if len(pct) > 0 else float("nan")
        denom = wins_a + wins_b
        effect_size_signed = float(wins_a - wins_b) / float(denom) if denom > 0 else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "metric": metric_label,
                "n_pairs": int(pct.size),
                win_label_a: wins_a,
                win_label_b: wins_b,
                "ties": ties,
                win_rate_label_a: win_rate_a,
                "median_delta_raw": float(location_raw),
                "median_pct": float(location_pct),
                "mean_pct": float(np.nanmean(pct)),
                "ci_low_pct": float(ci_low_pct),
                "ci_high_pct": float(ci_high_pct),
                "p_value": _wilcoxon_two_sided_pvalue(pct),
                "effect_size_signed": effect_size_signed,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "metric",
                "n_pairs",
                "wins_floatsom",
                "wins_xpysom",
                "ties",
                "win_rate_floatsom",
                "median_delta_raw",
                "median_pct",
                "mean_pct",
                "ci_low_pct",
                "ci_high_pct",
                "p_value",
                "effect_size_signed",
            ]
        )

    summary_df = pd.DataFrame(rows)
    pooled_pct_array = np.asarray(pooled_pct, dtype=float)
    pooled_pct_array = pooled_pct_array[np.isfinite(pooled_pct_array)]
    pooled_delta_array = np.asarray(pooled_delta_raw, dtype=float)
    pooled_delta_array = pooled_delta_array[np.isfinite(pooled_delta_array)]
    if pooled_pct_array.size > 0:
        location_pct, ci_low_pct, ci_high_pct = _wilcoxon_location_ci(pooled_pct_array)
        wins_a = int(np.sum(pooled_pct_array > 0.0))
        wins_b = int(np.sum(pooled_pct_array < 0.0))
        ties = int(pooled_pct_array.size - wins_a - wins_b)
        win_rate_a = float(wins_a) / float(pooled_pct_array.size)
        denom = wins_a + wins_b
        effect_size_signed = float(wins_a - wins_b) / float(denom) if denom > 0 else float("nan")
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "dataset": "GLOBAL",
                            "metric": metric_label,
                            "n_pairs": int(pooled_pct_array.size),
                            win_label_a: wins_a,
                            win_label_b: wins_b,
                            "ties": ties,
                            win_rate_label_a: win_rate_a,
                            "median_delta_raw": float(_wilcoxon_location_shift(pooled_delta_array)) if pooled_delta_array.size > 0 else np.nan,
                            "median_pct": float(location_pct),
                            "mean_pct": float(np.mean(pooled_pct_array)),
                            "ci_low_pct": float(ci_low_pct),
                            "ci_high_pct": float(ci_high_pct),
                            "p_value": _wilcoxon_two_sided_pvalue(pooled_pct_array),
                            "effect_size_signed": effect_size_signed,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return summary_df


def _paired_pct_improvement_summary(
    runs_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
) -> pd.DataFrame:
    return _paired_pct_improvement_summary_generic(
        runs_df,
        metric_col=metric_col,
        metric_label=metric_label,
        compare_col="method",
        value_a=METHOD_FLOATSOM,
        value_b=METHOD_XPYSOM,
        win_label_a="wins_floatsom",
        win_label_b="wins_xpysom",
        win_rate_label_a="win_rate_floatsom",
    )


def _plot_dataset_forest_axis(
    ax: object,
    summary_df: pd.DataFrame,
    panel_title: str,
    alpha: float = 0.05,
    x_label: str | None = None,
    show_y_tick_labels: bool = True,
    show_x_label: bool = False,
) -> None:
    if summary_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = summary_df.copy()
    order = _ordered_dataset_labels(plot_df["dataset"].astype(str).tolist())
    order_map = {name: idx for idx, name in enumerate(order)}
    plot_df["plot_order"] = plot_df["dataset"].map(order_map)
    plot_df = plot_df.sort_values("plot_order", ascending=True).reset_index(drop=True)
    plot_df["y"] = np.arange(len(plot_df))[::-1]
    plot_df["is_global"] = plot_df["dataset"].astype(str) == "GLOBAL"
    plot_df["is_significant"] = plot_df.apply(lambda row: _row_is_significant(row, alpha=alpha), axis=1)
    plot_df["sig_color"] = np.where(plot_df["is_significant"], "#d7301f", "#7a7a7a")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    for _, row in plot_df.iterrows():
        ax.hlines(
            y=float(row["y"]),
            xmin=float(row["ci_low_pct"]),
            xmax=float(row["ci_high_pct"]),
            color=str(row["sig_color"]),
            linewidth=2.0,
            alpha=0.9,
        )

    nonglobal = plot_df[~plot_df["is_global"]]
    if not nonglobal.empty:
        ax.scatter(
            nonglobal["median_pct"].to_numpy(dtype=float),
            nonglobal["y"].to_numpy(dtype=float),
            c=nonglobal["sig_color"].tolist(),
            s=85.0,
            marker="o",
            zorder=3,
        )

    global_rows = plot_df[plot_df["is_global"]]
    if not global_rows.empty:
        ax.scatter(
            global_rows["median_pct"].to_numpy(dtype=float),
            global_rows["y"].to_numpy(dtype=float),
            c=global_rows["sig_color"].tolist(),
            s=95.0,
            marker="D",
            zorder=4,
        )

    for _, row in plot_df.iterrows():
        if not bool(row["is_significant"]):
            continue
        stars = _pvalue_to_stars(float(row["p_value"]))
        if stars == "ns":
            continue
        ax.annotate(
            stars,
            (float(row["median_pct"]), float(row["y"])),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=14.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=str(row["sig_color"]),
            clip_on=False,
        )

    x_bounds = plot_df[["ci_low_pct", "ci_high_pct", "median_pct"]].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_bounds))
    x_max = float(np.nanmax(x_bounds))
    span = max(1.0, x_max - x_min)
    ax.set_xlim(min(-0.5, x_min - 0.08 * span), max(0.5, x_max + 0.08 * span))
    ax.set_yticks(plot_df["y"].to_numpy(dtype=float))
    if show_y_tick_labels:
        ax.set_yticklabels(plot_df["dataset"].astype(str).tolist())
    else:
        ax.set_yticklabels([""] * len(plot_df))
    ax.tick_params(axis="x", labelsize=16.0)
    ax.tick_params(axis="y", labelsize=15.0, length=0)
    if show_x_label:
        ax.set_xlabel(
            x_label
            if x_label is not None
            else "Median % Improvement vs XPySOM (positive means FloatSOM is better)",
            fontsize=15.0,
        )
    else:
        ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.25)


def _build_runtime_delta_dataset_summary(
    runs_df: pd.DataFrame,
    *,
    compare_col: str,
    value_a: str,
    value_b: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    value_a_str = str(value_a)
    value_b_str = str(value_b)
    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(index="seed", columns=compare_col, values="train_time_s", aggfunc="first")
        if pivot.empty:
            continue
        pivot.columns = pivot.columns.astype(str)
        if not {value_a_str, value_b_str}.issubset(set(pivot.columns.tolist())):
            continue

        paired = pivot[[value_a_str, value_b_str]].dropna()
        if paired.empty:
            continue

        deltas = paired[value_a_str].to_numpy(dtype=float) - paired[value_b_str].to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue

        rows.append(
            {
                "dataset": str(dataset),
                "n_pairs": int(deltas.size),
                "sample_size": float(_resolve_dataset_sample_sizes([dataset])[0]),
                "median_delta_s": float(np.median(deltas)),
                "mean_delta_s": float(np.mean(deltas)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["dataset", "n_pairs", "sample_size", "median_delta_s", "mean_delta_s"])

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_size", "median_delta_s"])
    if summary_df.empty:
        return summary_df
    summary_df = summary_df.sort_values("sample_size", ascending=True).reset_index(drop=True)
    dataset_index_lookup = _dataset_index_lookup(summary_df["dataset"].tolist())
    summary_df["dataset_index"] = summary_df["dataset"].map(dataset_index_lookup).astype("Int64")
    return summary_df[
        ["dataset_index", "dataset", "n_pairs", "sample_size", "median_delta_s", "mean_delta_s"]
    ].copy()


def _plot_runtime_delta_vs_dataset_size_axis(
    ax: object,
    runtime_df: pd.DataFrame,
    *,
    panel_title: str,
    label_a: str,
    label_b: str,
) -> None:
    if runtime_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = runtime_df.copy()
    plot_df["sample_size"] = pd.to_numeric(plot_df["sample_size"], errors="coerce")
    plot_df["median_delta_s"] = pd.to_numeric(plot_df["median_delta_s"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_size", "median_delta_s"])
    if plot_df.empty:
        ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = plot_df.sort_values("sample_size", ascending=True).reset_index(drop=True)
    delta_values = plot_df["median_delta_s"].to_numpy(dtype=float)
    point_colors = np.where(delta_values <= 0.0, "#1b9e77", "#d95f02")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.scatter(
        plot_df["sample_size"].to_numpy(dtype=float),
        delta_values,
        c=point_colors.tolist(),
        s=88.0,
        edgecolors="white",
        linewidths=0.9,
        zorder=2,
    )
    for _, row in plot_df.iterrows():
        dataset_index = row.get("dataset_index", pd.NA)
        if pd.isna(dataset_index):
            continue
        ax.annotate(
            str(int(dataset_index)),
            (float(row["sample_size"]), float(row["median_delta_s"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#1f1f1f",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=3,
            clip_on=False,
        )

    max_abs_delta = float(np.nanmax(np.abs(delta_values))) if len(delta_values) else 1.0
    linthresh = max(1e-3, min(0.25, max_abs_delta * 0.12))
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
    ax.set_xscale("log", base=10)

    x_values = plot_df["sample_size"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    ax.set_xlim(x_min * 0.92, x_max * 1.08)
    ax.set_xlabel("Dataset size (samples)", fontsize=15.0)
    ax.set_ylabel("Runtime delta (FloatSOM - baseline, s)", fontsize=15.0)
    ax.tick_params(axis="x", labelsize=14.0)
    ax.tick_params(axis="y", labelsize=14.0)
    ax.grid(True, axis="both", alpha=0.22)
    ax.set_title(panel_title, loc="left", fontsize=19, fontweight="bold", pad=12)


def _generate_pairwise_calibration_figure(
    runs_df: pd.DataFrame,
    *,
    output_path: Path,
    stats_output_path: Path,
    comparison_label: str,
    baseline_label: str,
    x_axis_label: str,
    compare_col: str,
    value_a: str,
    value_b: str,
    win_label_a: str,
    win_label_b: str,
    win_rate_label_a: str,
    figure_label: str,
    figure_caption: str,
) -> pd.DataFrame:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate publication-style XPySOM calibration figures."
        ) from exc

    panel_specs = [
        ("balanced_qe_raw", "A  Balanced QE"),
        ("quantization_error_holdout", "B  Holdout QE"),
        ("quantization_error_train", "C  Train QE"),
    ]

    panel_stats: List[pd.DataFrame] = []
    for metric_col, panel_title in panel_specs:
        panel_df = _paired_pct_improvement_summary_generic(
            runs_df,
            metric_col=metric_col,
            metric_label=panel_title,
            compare_col=compare_col,
            value_a=value_a,
            value_b=value_b,
            win_label_a=win_label_a,
            win_label_b=win_label_b,
            win_rate_label_a=win_rate_label_a,
        )
        panel_stats.append(panel_df)

    runtime_df = _build_runtime_delta_dataset_summary(
        runs_df,
        compare_col=compare_col,
        value_a=value_a,
        value_b=value_b,
    )

    all_stats = pd.concat(panel_stats, ignore_index=True) if panel_stats else pd.DataFrame()
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(stats_output_path, index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(22.0, 16.0),
        gridspec_kw={"hspace": 0.27, "wspace": 0.14},
    )
    flat_axes = axes.flatten()
    for idx, (metric_col, panel_title) in enumerate(panel_specs):
        panel_df = all_stats[all_stats["metric"] == panel_title].copy()
        _plot_dataset_forest_axis(
            flat_axes[idx],
            panel_df,
            panel_title=panel_title,
            alpha=0.05,
            x_label=x_axis_label,
            show_y_tick_labels=idx in (0, 2),
            show_x_label=False,
        )
    _plot_runtime_delta_vs_dataset_size_axis(
        flat_axes[3],
        runtime_df,
        panel_title="D  Runtime Delta vs Dataset Size",
        label_a=comparison_label.split(" vs ")[0] if " vs " in comparison_label else value_a,
        label_b=baseline_label,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7301f", markeredgecolor="#d7301f", markersize=12, label="Significant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7a7a7a", markeredgecolor="#7a7a7a", markersize=12, label="Non-significant"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#7a7a7a", markeredgecolor="#7a7a7a", markersize=12, label="GLOBAL"),
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="95% CI"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero effect"),
    ]
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.992))

    top_left = axes[0, 0].get_position()
    top_right = axes[0, 1].get_position()
    bottom_left = axes[1, 0].get_position()
    bottom_right = axes[1, 1].get_position()
    title_y = min(0.999, float(top_left.y1) + 0.032)
    fig.text(
        0.02,
        title_y,
        f"{figure_label}: {figure_caption}",
        ha="left",
        va="bottom",
        fontsize=24,
        fontweight="bold",
    )
    top_x_center = (top_left.x0 + top_right.x1) / 2.0
    bottom_x_center = (bottom_left.x0 + bottom_right.x1) / 2.0
    # Keep both row-level x labels at the same visual distance from their row panels.
    row_xlabel_offset = 0.030
    fig.text(top_x_center, top_left.y0 - row_xlabel_offset, x_axis_label, ha="center", va="top", fontsize=16)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.001),
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return all_stats


def _generate_xpysom_calibration_figure(
    runs_df: pd.DataFrame,
    *,
    topology: str,
    output_path: Path,
    stats_output_path: Path,
) -> pd.DataFrame:
    topology_key = str(topology).strip().lower()
    if topology_key == "mst":
        topology_label = "MST"
    elif topology_key == "rng":
        topology_label = "RNG"
    elif topology_key == "hexagonal":
        topology_label = "Hexagonal"
    else:
        topology_label = str(topology).replace("_", " ").title()

    figure_label_map = {
        "mst": "Supplementary Figure S1",
        "rng": "Supplementary Figure S2",
        "hexagonal": "Supplementary Figure S3",
    }
    figure_label = figure_label_map.get(topology_key, "Supplementary Figure")

    return _generate_pairwise_calibration_figure(
        runs_df,
        output_path=output_path,
        stats_output_path=stats_output_path,
        comparison_label=COMPARISON_LABEL,
        baseline_label="XPySOM",
        x_axis_label="Median % Improvement vs XPySOM (positive means FloatSOM is better)",
        compare_col="method",
        value_a=METHOD_FLOATSOM,
        value_b=METHOD_XPYSOM,
        win_label_a="wins_floatsom",
        win_label_b="wins_xpysom",
        win_rate_label_a="win_rate_floatsom",
        figure_label=figure_label,
        figure_caption=f"FloatSOM vs XPySOM Calibration (Default Settings, {topology_label})",
    )


def _generate_mst_vs_rng_final_figure(
    runs_df: pd.DataFrame,
    *,
    output_path: Path,
    stats_output_path: Path,
) -> pd.DataFrame:
    floatsom_only = runs_df[runs_df["method"].astype(str) == METHOD_FLOATSOM].copy()
    floatsom_only = floatsom_only[floatsom_only["architecture"].astype(str).isin({"mst", "rng"})].copy()
    return _generate_pairwise_calibration_figure(
        floatsom_only,
        output_path=output_path,
        stats_output_path=stats_output_path,
        comparison_label=MST_VS_RNG_COMPARISON_LABEL,
        baseline_label="RNG",
        x_axis_label=MST_VS_RNG_XLABEL,
        compare_col="architecture",
        value_a="mst",
        value_b="rng",
        win_label_a="wins_mst",
        win_label_b="wins_rng",
        win_rate_label_a="win_rate_mst",
        figure_label="Supplementary Figure",
        figure_caption="FloatSOM Topology Comparison (Default Settings): MST vs RNG",
    )


def _publish_to_paper_assets(
    *,
    topology: str,
    summary_tsv_path: Path,
    figure_path: Path,
) -> dict[str, Path]:
    floatsom_root = Path(__file__).resolve().parents[2]
    paper_tables_dir = floatsom_root / "paper" / "assets" / "tables"
    paper_figures_dir = floatsom_root / "paper" / "assets" / "figures"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    table_target = paper_tables_dir / _table_filename_for_topology(topology)
    figure_target = paper_figures_dir / _figure_filename_for_topology(topology)
    shutil.copy2(summary_tsv_path, table_target)
    shutil.copy2(figure_path, figure_target)

    published_paths: dict[str, Path] = {
        "table": table_target,
        "figure": figure_target,
    }
    return published_paths


def _table_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return PAPER_SUPP_TABLE_FILENAME
    return f"supp_xpysom_calibration_qe_{str(topology)}.tsv"


def _figure_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return PAPER_SUPP_FIGURE_FILENAME
    if str(topology) == "mst":
        return "supp_fig_s1.svg"
    if str(topology) == "rng":
        return "supp_fig_s2.svg"
    raise ValueError(f"Unsupported topology for paper figure filename: {topology}")


def _forest_stats_filename_for_topology(topology: str) -> str:
    if str(topology) == "hexagonal":
        return FOREST_STATS_FILENAME
    return f"xpysom_calibration_forest_stats_{str(topology)}.csv"


def _paired_metric_rows(
    runs_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_name: str,
    split_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pooled_pct: List[float] = []
    pooled_delta_raw: List[float] = []

    for dataset in sorted(runs_df["dataset"].dropna().astype(str).unique()):
        subset = runs_df[runs_df["dataset"].astype(str) == dataset].copy()
        pivot = subset.pivot_table(
            index="seed",
            columns="method",
            values=metric_col,
            aggfunc="first",
        )
        required_cols = {METHOD_FLOATSOM, METHOD_XPYSOM}
        if not required_cols.issubset(set(pivot.columns.astype(str).tolist())):
            raise ValueError(
                f"Missing paired methods for dataset '{dataset}' metric '{metric_col}'. "
                f"Expected methods: {sorted(required_cols)}. Found: {list(pivot.columns)}"
            )
        paired = pivot[[METHOD_FLOATSOM, METHOD_XPYSOM]].dropna()
        if paired.empty:
            continue

        floatsom_values = paired[METHOD_FLOATSOM].to_numpy(dtype=float)
        xpysom_values = paired[METHOD_XPYSOM].to_numpy(dtype=float)
        deltas = floatsom_values - xpysom_values
        pct = _paired_pct_improvements(floatsom_values, xpysom_values)
        if pct.size == 0:
            continue
        pooled_pct.extend(float(value) for value in pct.tolist())
        pooled_delta_raw.extend(float(value) for value in deltas[np.isfinite(deltas)].tolist())
        location_pct, ci_low_pct, ci_high_pct = _wilcoxon_location_ci(pct)
        location_raw = _wilcoxon_location_shift(deltas)

        wins_floatsom = int(np.sum(pct > 0.0))
        wins_xpysom = int(np.sum(pct < 0.0))
        ties = int(len(pct) - wins_floatsom - wins_xpysom)
        win_rate_floatsom = float(wins_floatsom) / float(len(pct))
        denom = wins_floatsom + wins_xpysom
        effect_size_signed = float(wins_floatsom - wins_xpysom) / float(denom) if denom > 0 else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "metric": metric_name,
                "split": split_name,
                "wins_floatsom": wins_floatsom,
                "wins_xpysom": wins_xpysom,
                "ties": ties,
                "win_rate_floatsom": win_rate_floatsom,
                "median_delta_raw": float(location_raw),
                "median_pct_improvement": float(location_pct),
                "mean_pct_improvement": float(np.mean(pct)),
                "ci_low_pct": float(ci_low_pct),
                "ci_high_pct": float(ci_high_pct),
                "p_value": _wilcoxon_two_sided_pvalue(pct),
                "n_pairs": int(len(pct)),
                "effect_size_signed": effect_size_signed,
                "notes": "positive means FloatSOM is better",
            }
        )

    if rows:
        pooled_values = np.asarray(pooled_pct, dtype=float)
        pooled_values = pooled_values[np.isfinite(pooled_values)]
        pooled_raw_values = np.asarray(pooled_delta_raw, dtype=float)
        pooled_raw_values = pooled_raw_values[np.isfinite(pooled_raw_values)]
        macro_location, macro_ci_low, macro_ci_high = _wilcoxon_location_ci(pooled_values)
        wins_floatsom = int(np.sum(pooled_values > 0.0))
        wins_xpysom = int(np.sum(pooled_values < 0.0))
        ties = int(len(pooled_values) - wins_floatsom - wins_xpysom)
        win_rate_floatsom = float(wins_floatsom) / float(len(pooled_values)) if len(pooled_values) > 0 else float("nan")
        denom = wins_floatsom + wins_xpysom
        effect_size_signed = float(wins_floatsom - wins_xpysom) / float(denom) if denom > 0 else float("nan")
        rows.append(
            {
                "dataset": "GLOBAL",
                "metric": metric_name,
                "split": split_name,
                "wins_floatsom": wins_floatsom,
                "wins_xpysom": wins_xpysom,
                "ties": ties,
                "win_rate_floatsom": win_rate_floatsom,
                "median_delta_raw": float(_wilcoxon_location_shift(pooled_raw_values)) if pooled_raw_values.size > 0 else np.nan,
                "median_pct_improvement": float(macro_location) if pooled_values.size > 0 else np.nan,
                "mean_pct_improvement": float(np.mean(pooled_values)) if pooled_values.size > 0 else np.nan,
                "ci_low_pct": float(macro_ci_low) if pooled_values.size > 0 else np.nan,
                "ci_high_pct": float(macro_ci_high) if pooled_values.size > 0 else np.nan,
                "p_value": _wilcoxon_two_sided_pvalue(pooled_values),
                "n_pairs": int(len(pooled_values)),
                "effect_size_signed": effect_size_signed,
                "notes": "pooled paired samples across datasets",
            }
        )

    return rows


def _build_summary_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("quantization_error_train", "quantization_error", "train"),
        ("quantization_error_holdout", "quantization_error", "holdout"),
        ("balanced_qe_raw", "balanced_qe", "both"),
        ("train_time_s", "train_time", "train"),
    ]
    rows: List[Dict[str, object]] = []
    for metric_col, metric_name, split_name in metric_specs:
        rows.extend(
            _paired_metric_rows(
                runs_df,
                metric_col=metric_col,
                metric_name=metric_name,
                split_name=split_name,
            )
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    dataset_index_lookup = _dataset_index_lookup(summary_df["dataset"].tolist())
    summary_df["dataset_index"] = summary_df["dataset"].map(dataset_index_lookup).astype("Int64")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FloatSOM batch topologies vs XPySOM (full sampling)."
    )
    parser.add_argument("--output-dir", type=str, default="xpysom_hex_batch_full_benchmark")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=None,
        help="Dataset names to benchmark (default: all datasets from Phase3BenchmarkConfig).",
    )
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument(
        "--floatsom-topologies",
        nargs="+",
        type=str,
        default=None,
        help=(
            "FloatSOM topology sweep for batch mode. "
            f"Supported: {list(SUPPORTED_FLOATSOM_BATCH_TOPOLOGIES)}"
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--scikit-learn-data-home",
        type=str,
        default=None,
        help="Path to sklearn dataset cache root (sets SCIKIT_LEARN_DATA for this process).",
    )
    parser.add_argument(
        "--xpysom-n-parallel",
        type=int,
        default=None,
        help="XPySOM n_parallel; default None lets XPySOM auto-determine.",
    )
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=None,
        help=(
            "Manual FloatSOM fixed hyperparameter override in key=value form. "
            "Can be repeated (example: --fixed-param initial_radius=3.5 --fixed-param use_momentum=false)."
        ),
    )
    parser.add_argument(
        "--fixed-params-json",
        type=str,
        default=None,
        help="Optional JSON file containing FloatSOM fixed hyperparameter overrides as a dictionary.",
    )
    parser.add_argument(
        "--fixed-params-by-topology-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing topology-specific FloatSOM fixed overrides as "
            '{"hexagonal": {...}, "mst": {...}, "rng": {...}}.'
        ),
    )
    parser.add_argument(
        "--fixed-params-by-sampling-topology-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing sampling+topology-specific FloatSOM fixed overrides as "
            '{"full": {"hexagonal": {...}, "mst": {...}, "rng": {...}}}. '
            "Only the 'full' branch is used by this benchmark."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_gpu()
    sklearn_data_home = _configure_sklearn_data_home(args.scikit_learn_data_home)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _resolve_datasets(args.datasets)
    seeds = _resolve_seeds(args.num_seeds, args.base_seed, args.seeds)
    floatsom_topologies = _resolve_floatsom_topologies(args.floatsom_topologies)
    (
        manual_fixed_params,
        manual_fixed_params_by_topology,
        manual_fixed_params_by_sampling_topology,
    ) = _resolve_floatsom_fixed_param_overrides(
        topologies=floatsom_topologies,
        fixed_param_entries=args.fixed_param,
        fixed_params_json=args.fixed_params_json,
        fixed_params_by_topology_json=args.fixed_params_by_topology_json,
        fixed_params_by_sampling_topology_json=args.fixed_params_by_sampling_topology_json,
    )
    floatsom_run_profile = _floatsom_run_profile_label(
        manual_fixed_params=manual_fixed_params,
        manual_fixed_params_by_topology=manual_fixed_params_by_topology,
        manual_fixed_params_by_sampling_topology=manual_fixed_params_by_sampling_topology,
    )
    _ensure_olivetti_cache_available(datasets, sklearn_data_home)
    default_epochs = int(FloatSOMParams(defaults_profile="publication", ).total_iterations)
    training_epochs = int(args.epochs) if args.epochs is not None else default_epochs
    if training_epochs <= 0:
        raise ValueError(f"epochs must be positive, got {training_epochs}")

    print("Running FloatSOM vs XPySOM benchmark")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"FloatSOM topologies: {floatsom_topologies}")
    print(f"FloatSOM run profile: {floatsom_run_profile}")
    if manual_fixed_params:
        print(f"FloatSOM fixed params ({len(manual_fixed_params)}): {manual_fixed_params}")
    if manual_fixed_params_by_topology:
        print(
            "FloatSOM fixed params by topology: "
            + ", ".join(
                f"{topology}={params}" for topology, params in sorted(manual_fixed_params_by_topology.items())
            )
        )
    if manual_fixed_params_by_sampling_topology:
        print(
            "FloatSOM fixed params by sampling+topology: "
            + ", ".join(
                f"{sampling}={topology_map}"
                for sampling, topology_map in sorted(manual_fixed_params_by_sampling_topology.items())
            )
        )
    print("XPySOM topology: ['hexagonal']")
    print(f"Epochs (aligned FloatSOM/XPySOM): {training_epochs}")
    if sklearn_data_home is not None:
        print(f"SCIKIT_LEARN_DATA: {sklearn_data_home}")
    print(f"Output: {output_dir}")

    qe_metric = QuantizationError(use_optimized=True)
    run_rows: List[Dict[str, object]] = []
    run_rows_by_topology: Dict[str, List[Dict[str, object]]] = {
        topology: [] for topology in floatsom_topologies
    }

    for dataset in datasets:
        for seed in seeds:
            data, _ = generate_sklearn_dataset(
                dataset_name=dataset,
                difficulty=args.difficulty,
                seed=int(seed),
                normalize=True,
            )
            if not isinstance(data, cp.ndarray):
                raise TypeError(
                    f"Expected CuPy data for GPU benchmark, got {type(data)} for dataset '{dataset}'."
                )

            train_data, holdout_data = _train_holdout_split(data, seed=int(seed))
            input_dim = int(train_data.shape[1])

            xpysom_model = _create_xpysom_hex(
                input_dim=input_dim,
                grid_size=int(args.grid_size),
                seed=int(seed),
                n_parallel=args.xpysom_n_parallel,
            )
            xpysom_init_weights = _extract_xpysom_initial_weights(xpysom_model)
            floatsom_init_weights = _xpysom_weights_to_floatsom_init(
                xpysom_init_weights,
                expected_grid_size=int(args.grid_size),
                expected_input_dim=input_dim,
            )
            xpysom_time = _train_xpysom(
                xpysom_model,
                train_data=train_data,
                epochs=training_epochs,
            )
            xpysom_qe_train = float(qe_metric.compute(xpysom_model, train_data))
            xpysom_qe_holdout = float(qe_metric.compute(xpysom_model, holdout_data))
            print(
                f"[{dataset}][seed={seed}] xpysom "
                f"time={xpysom_time:.6f}s qe_train={xpysom_qe_train:.6f} qe_holdout={xpysom_qe_holdout:.6f}"
            )

            for topology in floatsom_topologies:
                effective_manual_fixed_params = _resolve_effective_manual_fixed_params(
                    manual_fixed_params=manual_fixed_params,
                    topology_fixed_params_by_topology=manual_fixed_params_by_topology,
                    sampling_topology_fixed_params=manual_fixed_params_by_sampling_topology,
                    topology=topology,
                    sampling_method="full",
                )
                floatsom_model, floatsom_params = _create_floatsom_batch_topology(
                    input_dim=input_dim,
                    seed=int(seed),
                    grid_size=int(args.grid_size),
                    epochs=training_epochs,
                    topology_type=topology,
                    manual_fixed_params=effective_manual_fixed_params,
                )
                configured_epochs = int(getattr(floatsom_params, "total_iterations", -1))
                if configured_epochs != training_epochs:
                    raise RuntimeError(
                        "FloatSOM epoch mismatch: "
                        f"configured={configured_epochs} requested={training_epochs}"
                    )

                _inject_floatsom_initial_weights(floatsom_model, floatsom_init_weights)
                floatsom_time = _train_floatsom(floatsom_model, train_data)
                floatsom_qe_train = float(qe_metric.compute(floatsom_model, train_data))
                floatsom_qe_holdout = float(qe_metric.compute(floatsom_model, holdout_data))
                floatsom_row = _compute_metric_row(
                    dataset=dataset,
                    seed=int(seed),
                    method=METHOD_FLOATSOM,
                    training_epochs=training_epochs,
                    train_time_s=floatsom_time,
                    qe_train=floatsom_qe_train,
                    qe_holdout=floatsom_qe_holdout,
                    algorithm="batch",
                    batch_mode="full_batch",
                    architecture=topology,
                    comparison_topology=topology,
                    run_profile=floatsom_run_profile,
                    run_label=floatsom_run_profile,
                    extra_fields=_extract_floatsom_param_columns(floatsom_params),
                )
                xpysom_row = _compute_metric_row(
                    dataset=dataset,
                    seed=int(seed),
                    method=METHOD_XPYSOM,
                    training_epochs=training_epochs,
                    train_time_s=xpysom_time,
                    qe_train=xpysom_qe_train,
                    qe_holdout=xpysom_qe_holdout,
                    algorithm="batch",
                    batch_mode="full_batch",
                    architecture="hexagonal",
                    comparison_topology=topology,
                    run_profile="xpysom_default",
                    run_label="xpysom_default",
                )

                run_rows.append(floatsom_row)
                run_rows.append(xpysom_row)
                run_rows_by_topology[topology].append(floatsom_row)
                run_rows_by_topology[topology].append(xpysom_row)

                print(
                    f"[{dataset}][seed={seed}][topology={topology}] floatsom "
                    f"time={floatsom_time:.6f}s qe_train={floatsom_qe_train:.6f} qe_holdout={floatsom_qe_holdout:.6f}"
                )

    if not run_rows:
        raise RuntimeError(
            "No successful benchmark runs were produced. "
            "If this is an offline environment and olivetti_faces is selected, "
            "set --scikit-learn-data-home to a pre-populated cache."
        )

    runs_df = pd.DataFrame(run_rows)
    runs_df = runs_df.sort_values(["comparison_topology", "dataset", "seed", "method"]).reset_index(drop=True)

    runs_csv_path = output_dir / "xpysom_batch_topology_sweep_runs.csv"
    runs_df.to_csv(runs_csv_path, index=False)

    latest_hex_paths: Dict[str, Path] = {}
    mst_rng_paths: Dict[str, Path] = {}
    for topology in floatsom_topologies:
        topology_rows = run_rows_by_topology.get(topology, [])
        if not topology_rows:
            continue
        topology_output_dir = output_dir / topology
        topology_output_dir.mkdir(parents=True, exist_ok=True)

        topology_runs_df = pd.DataFrame(topology_rows).sort_values(
            ["dataset", "seed", "method"]
        ).reset_index(drop=True)
        topology_runs_csv = topology_output_dir / f"xpysom_{topology}_batch_full_runs.csv"
        topology_runs_df.to_csv(topology_runs_csv, index=False)

        summary_df = _build_summary_table(topology_runs_df)
        summary_df = summary_df[
            [
                "dataset_index",
                "dataset",
                "metric",
                "split",
                "wins_floatsom",
                "wins_xpysom",
                "ties",
                "win_rate_floatsom",
                "median_delta_raw",
                "median_pct_improvement",
                "mean_pct_improvement",
                "ci_low_pct",
                "ci_high_pct",
                "p_value",
                "n_pairs",
                "notes",
                "effect_size_signed",
            ]
        ]

        summary_tsv_path = topology_output_dir / _table_filename_for_topology(topology)
        summary_df.to_csv(summary_tsv_path, sep="\t", index=False)

        figure_path = topology_output_dir / _figure_filename_for_topology(topology)
        forest_stats_path = topology_output_dir / _forest_stats_filename_for_topology(topology)
        _generate_xpysom_calibration_figure(
            runs_df=topology_runs_df,
            topology=topology,
            output_path=figure_path,
            stats_output_path=forest_stats_path,
        )

        print(f"\n[{topology}] Raw per-run log: {topology_runs_csv}")
        print(f"[{topology}] Summary table: {summary_tsv_path}")
        print(f"[{topology}] Forest stats table: {forest_stats_path}")
        print(f"[{topology}] Calibration figure: {figure_path}")

        paper_assets = _publish_to_paper_assets(
            topology=topology,
            summary_tsv_path=summary_tsv_path,
            figure_path=figure_path,
        )
        print(f"[{topology}] Paper table asset: {paper_assets['table']}")
        print(f"[{topology}] Paper figure asset: {paper_assets['figure']}")

        if topology == "hexagonal":
            latest_hex_paths = {
                "summary_tsv": summary_tsv_path,
                "figure": figure_path,
                "forest": forest_stats_path,
                "paper_table": paper_assets["table"],
                "paper_figure": paper_assets["figure"],
            }

    if {"mst", "rng"}.issubset(set(floatsom_topologies)):
        mst_rng_figure_path = output_dir / MST_VS_RNG_FIGURE_FILENAME
        mst_rng_stats_path = output_dir / MST_VS_RNG_FOREST_STATS_FILENAME
        _generate_mst_vs_rng_final_figure(
            runs_df=runs_df,
            output_path=mst_rng_figure_path,
            stats_output_path=mst_rng_stats_path,
        )
        mst_rng_paths = {
            "figure": mst_rng_figure_path,
            "forest": mst_rng_stats_path,
        }
        print(f"\n[MST vs RNG] Final figure: {mst_rng_figure_path}")
        print(f"[MST vs RNG] Forest stats table: {mst_rng_stats_path}")
    else:
        print("\n[MST vs RNG] Final figure skipped (requires both 'mst' and 'rng' topologies).")

    print("\nBenchmark complete.")
    print(f"Raw per-run log: {runs_csv_path}")
    if latest_hex_paths:
        print(f"Hex summary table: {latest_hex_paths['summary_tsv']}")
        print(f"Hex forest stats table: {latest_hex_paths['forest']}")
        print(f"Hex calibration figure: {latest_hex_paths['figure']}")
        print(f"Paper table asset: {latest_hex_paths['paper_table']}")
        print(f"Paper figure asset: {latest_hex_paths['paper_figure']}")
    if mst_rng_paths:
        print(f"MST vs RNG final figure: {mst_rng_paths['figure']}")
        print(f"MST vs RNG forest stats table: {mst_rng_paths['forest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
