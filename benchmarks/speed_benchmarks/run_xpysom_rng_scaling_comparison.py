#!/usr/bin/env python3
"""
Benchmark XPySOM defaults against FloatSOM RNG sample scaling on 4 GPUs.

This script is intentionally narrow:
- topology is fixed to RNG for FloatSOM
- XPySOM uses its default hexagonal configuration from the existing Optuna
  calibration benchmark helper
- FloatSOM runs are driven by a fixed JSON profile for `full` and `random`
  sampling
- output includes raw timing rows, aggregated stats, and a standalone scaling
  panel rendered with the existing scaling figure generator
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import cupy as cp
import numpy as np
import pandas as pd
import zarr

from floatsom.benchmarks.optuna.benchmark_xpysom_hex_batch_full import (
    _create_xpysom_hex,
    _train_xpysom,
)
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.config import BenchmarkConfig
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.large_dataset import (
    cleanup_staged_dataset,
    generate_or_load_data,
    stage_dataset_for_execution,
    train_floatsom_with_zarr,
)
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.ray_utils import ensure_clean_ray_state
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.resume_state import ResumeManager
from floatsom.benchmarks.visualization import plot_topology_comparison_scaling_benchmark
from floatsom.processing.processing_params import get_visible_gpu_vram_mib

MP_CONTEXT = multiprocessing.get_context("spawn")


DEFAULT_SAMPLE_SIZES: List[int] = [
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
]
SCALING_PANEL_FILENAME = "xpysom_rng_scaling_panel.svg"
RAW_TIMES_CSV_FILENAME = "xpysom_rng_scaling_runs.csv"
AGGREGATED_STATS_CSV_FILENAME = "xpysom_rng_scaling_stats.csv"
RUN_METADATA_FILENAME = "xpysom_rng_scaling_metadata.json"
RESUME_STATE_FILENAME = "xpysom_rng_scaling_resume_state.json"
RUN_TIMEOUT_MINUTES = 30
RUN_TIMEOUT_SECONDS = RUN_TIMEOUT_MINUTES * 60
OVERWRITE_COMPONENT_CHOICES: tuple[str, ...] = (
    "xpysom",
    "floatsom-full",
    "floatsom-random",
    "floatsom-all",
    "all",
)
SKIP_COMPONENT_CHOICES: tuple[str, ...] = OVERWRITE_COMPONENT_CHOICES


def _build_result_row(
    *,
    sample_size: int,
    repeat_idx: int,
    seed: int,
    gpu_count: int,
    topology: str,
    method: str,
    sampling_method: str,
    train_time_s: Optional[float],
    status: str,
    error_type: str = "",
    error_message: str = "",
) -> Dict[str, Any]:
    return {
        "sample_size": int(sample_size),
        "repeat": int(repeat_idx) + 1,
        "seed": int(seed),
        "gpu_count": int(gpu_count),
        "topology": str(topology),
        "method": str(method),
        "sampling_method": str(sampling_method),
        "train_time_s": (float(train_time_s) if train_time_s is not None else float("nan")),
        "status": str(status),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XPySOM-vs-FloatSOM RNG sample-scaling benchmarks."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for CSV/SVG artifacts.")
    parser.add_argument("--cache-dir", type=str, required=True, help="Cache directory for generated Zarr arrays.")
    parser.add_argument("--temp-dir", type=str, required=False, default=None, help="Optional scratch directory for FloatSOM staging.")
    parser.add_argument(
        "--ray-local-storage-path",
        type=str,
        required=True,
        help="Local directory for per-worker FastArrayStore copies (for Ray-based FloatSOM runs).",
    )
    parser.add_argument(
        "--fixed-params-by-sampling-topology-json",
        type=str,
        required=True,
        help=(
            "JSON file containing sampling+topology-specific FloatSOM overrides. "
            "Expected shape: {'full': {'rng': {...}}, 'random': {'rng': {...}}}."
        ),
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Sample sizes to benchmark.",
    )
    parser.add_argument("--fixed-dimension", type=int, default=50, help="Fixed input dimension.")
    parser.add_argument("--grid-size", type=int, default=32, help="Fixed SOM grid size.")
    parser.add_argument("--gpu-count", type=int, default=4, help="GPU count for the scaling panel metadata.")
    parser.add_argument("--ray-gpu-count", type=int, default=None, help="Ray GPU count for FloatSOM runs.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of paired repeats per sample size.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for repeat scheduling.")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations for both models.")
    parser.add_argument("--xpysom-n-parallel", type=int, default=None, help="Optional XPySOM n_parallel.")
    parser.add_argument("--zarr-chunk-size", type=int, default=None, help="Optional Zarr chunk size override.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional FloatSOM batch chunk size override.")
    parser.add_argument(
        "--sampling-fraction-random",
        type=float,
        default=None,
        help="Optional CLI override for FloatSOM random sampling fraction.",
    )
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate cached datasets.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing raw CSV and resume manifest in the output directory.",
    )
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help=f"Path to resume manifest (defaults to <output-dir>/{RESUME_STATE_FILENAME}).",
    )
    parser.add_argument(
        "--reset-resume-state",
        action="store_true",
        help="Ignore any existing resume manifest and restart this benchmark run from scratch.",
    )
    parser.add_argument(
        "--overwrite-components",
        nargs="+",
        choices=list(OVERWRITE_COMPONENT_CHOICES),
        default=None,
        help=(
            "When resuming, rerun only the selected component(s) and reuse existing results for the rest. "
            "Choices: xpysom, floatsom-full, floatsom-random, floatsom-all, all."
        ),
    )
    parser.add_argument(
        "--skip-components",
        nargs="+",
        choices=list(SKIP_COMPONENT_CHOICES),
        default=None,
        help=(
            "Skip the selected benchmark component(s) entirely. "
            "Choices: xpysom, floatsom-full, floatsom-random, floatsom-all, all."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def _normalize_component_selection(raw_components: Optional[Iterable[str]]) -> set[str]:
    expanded: set[str] = set()
    for raw_component in raw_components or ():
        component = str(raw_component).strip().lower()
        if component == "all":
            return {"xpysom", "floatsom-full", "floatsom-random"}
        if component == "floatsom-all":
            expanded.update({"floatsom-full", "floatsom-random"})
            continue
        expanded.add(component)
    return expanded


def _component_name(*, topology: str, method: str) -> str:
    topology_key = str(topology).strip().lower()
    method_key = str(method).strip().lower()
    if topology_key == "xpysom":
        return "xpysom"
    if topology_key == "floatsom":
        return f"floatsom-{method_key}"
    raise ValueError(f"Unsupported component identity for topology={topology!r}, method={method!r}")


def _task_key(
    *,
    sample_size: int,
    repeat_idx: int,
    seed: int,
    topology: str,
    method: str,
) -> str:
    return "|".join(
        [
            f"samples={int(sample_size)}",
            f"repeat={int(repeat_idx) + 1}",
            f"seed={int(seed)}",
            f"topology={str(topology).strip().lower()}",
            f"method={str(method).strip().lower()}",
        ]
    )


def _task_metadata(
    *,
    sample_size: int,
    repeat_idx: int,
    seed: int,
    gpu_count: int,
    topology: str,
    method: str,
    sampling_method: str,
) -> Dict[str, Any]:
    component = _component_name(topology=topology, method=method)
    return {
        "sample_size": int(sample_size),
        "repeat": int(repeat_idx) + 1,
        "seed": int(seed),
        "gpu_count": int(gpu_count),
        "topology": str(topology),
        "method": str(method),
        "sampling_method": str(sampling_method),
        "component": component,
    }


def _task_satisfied_by_resume(
    *,
    task_key: str,
    component: str,
    existing_success_rows: Dict[str, Dict[str, Any]],
    overwrite_components: set[str],
) -> bool:
    return (
        task_key in existing_success_rows
        and str(component).strip().lower() not in overwrite_components
    )


def _task_explicitly_skipped(*, component: str, skip_components: set[str]) -> bool:
    return str(component).strip().lower() in skip_components


def _normalize_loaded_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    normalized["sample_size"] = int(normalized["sample_size"])
    normalized["repeat"] = int(normalized["repeat"])
    normalized["seed"] = int(normalized["seed"])
    normalized["gpu_count"] = int(normalized["gpu_count"])
    normalized["topology"] = str(normalized["topology"])
    normalized["method"] = str(normalized["method"])
    normalized["sampling_method"] = str(normalized["sampling_method"])
    normalized["status"] = str(normalized.get("status", "success"))
    train_time = normalized.get("train_time_s", float("nan"))
    normalized["train_time_s"] = float(train_time) if pd.notna(train_time) else float("nan")
    for key in ("error_type", "error_message"):
        value = normalized.get(key, "")
        normalized[key] = "" if pd.isna(value) else str(value)
    return normalized


def _load_existing_success_rows(raw_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not raw_csv_path.exists():
        return {}

    existing_df = pd.read_csv(raw_csv_path)
    if existing_df.empty:
        return {}

    existing_rows: Dict[str, Dict[str, Any]] = {}
    for raw_row in existing_df.to_dict(orient="records"):
        row = _normalize_loaded_row(raw_row)
        if str(row.get("status", "")).strip().lower() != "success":
            continue
        if not np.isfinite(float(row.get("train_time_s", float("nan")))):
            continue
        task_key = _task_key(
            sample_size=int(row["sample_size"]),
            repeat_idx=int(row["repeat"]) - 1,
            seed=int(row["seed"]),
            topology=str(row["topology"]),
            method=str(row["method"]),
        )
        existing_rows[task_key] = row
    return existing_rows


def _build_raw_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(
        ["sample_size", "topology", "method", "repeat"], ascending=[True, True, True, True]
    ).reset_index(drop=True)


def _persist_raw_rows(task_rows: Dict[str, Dict[str, Any]], raw_csv_path: Path) -> None:
    raw_df = _build_raw_frame(task_rows.values())
    raw_df.to_csv(raw_csv_path, index=False)


def _load_sampling_overrides(path: str | Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixed-params JSON must be an object.")

    out: Dict[str, Dict[str, Any]] = {}
    for sampling_method in ("full", "random"):
        raw_sampling_payload = payload.get(sampling_method)
        if raw_sampling_payload is None:
            raise ValueError(
                f"Fixed-params JSON must contain a top-level '{sampling_method}' entry."
            )
        if not isinstance(raw_sampling_payload, dict):
            raise ValueError(
                f"Fixed-params JSON entry '{sampling_method}' must map topology names to parameter objects."
            )
        raw_rng_payload = raw_sampling_payload.get("rng")
        if raw_rng_payload is None:
            raise ValueError(
                f"Fixed-params JSON entry '{sampling_method}' must contain an 'rng' topology payload."
            )
        if not isinstance(raw_rng_payload, dict):
            raise ValueError(
                f"Fixed-params JSON entry '{sampling_method}.rng' must be an object of parameter overrides."
            )
        out[sampling_method] = dict(raw_rng_payload)
    return out


def _create_data_args(
    *,
    cache_dir: str,
    sample_size: int,
    input_dim: int,
    chunk_size: Optional[int],
    zarr_chunk_size: Optional[int],
    seed: int,
    force_regenerate: bool,
    verbose: bool,
) -> SimpleNamespace:
    _, resolved_zarr_chunk_size, _, _ = _resolve_floatsom_batch_chunking(
        input_dim=int(input_dim),
        chunk_size=chunk_size,
        zarr_chunk_size=zarr_chunk_size,
    )
    return SimpleNamespace(
        cache_dir=str(cache_dir),
        samples=int(sample_size),
        input_dim=int(input_dim),
        zarr_chunk_size=int(resolved_zarr_chunk_size),
        seed=int(seed),
        force_regenerate=bool(force_regenerate),
        verbose=bool(verbose),
    )


def _resolve_floatsom_batch_chunking(
    *,
    input_dim: int,
    chunk_size: Optional[int],
    zarr_chunk_size: Optional[int],
) -> tuple[int, int, str, Optional[int]]:
    chunk_size_visible_vram_mib = None
    if chunk_size is not None:
        resolved_chunk_size = int(chunk_size)
        if resolved_chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer when provided")
        chunk_size_source = "manual_cli_chunk_size"
    else:
        resolved_chunk_size = BenchmarkConfig.calculate_chunk_size(int(input_dim), "batch")
        chunk_size_source = "auto_dimension_vram"
        chunk_size_visible_vram_mib = get_visible_gpu_vram_mib()

    if zarr_chunk_size is None:
        resolved_zarr_chunk_size = int(resolved_chunk_size)
    else:
        resolved_zarr_chunk_size = int(zarr_chunk_size)
        if resolved_zarr_chunk_size <= 0:
            resolved_zarr_chunk_size = int(resolved_chunk_size)

    return (
        int(resolved_chunk_size),
        int(resolved_zarr_chunk_size),
        chunk_size_source,
        chunk_size_visible_vram_mib,
    )


def _apply_floatsom_fixed_params(args: SimpleNamespace, fixed_params: Dict[str, Any]) -> None:
    if not fixed_params:
        return

    for param_name, raw_value in fixed_params.items():
        if param_name == "initial_radius":
            args.initial_radius = float(raw_value)
        elif param_name == "initialization_method":
            args.initialization_method = str(raw_value)
        elif param_name == "normalization":
            args.normalization = str(raw_value)
        elif param_name == "radius_decay_type":
            args.radius_decay_type = str(raw_value)
        elif param_name == "lr_decay_type":
            args.lr_decay_type = str(raw_value)
        elif param_name == "lr_decay_factor":
            args.lr_decay_factor = float(raw_value)
        elif param_name == "radius_decay_factor":
            args.radius_decay_factor = float(raw_value)
        elif param_name == "learning_rate":
            args.initial_learning_rate = float(raw_value)
        elif param_name == "use_momentum":
            args.use_momentum = bool(raw_value)
        elif param_name == "momentum_init":
            args.initial_momentum = float(raw_value)
        elif param_name == "topology_variant":
            args.topology_variant = str(raw_value)
        elif param_name in {"target_proportion", "sampling_fraction"}:
            args.sampling_fraction = float(raw_value)
            args.target_proportion = float(raw_value)
        elif param_name == "virtual_ratio":
            args.virtual_ratio = float(raw_value)
        elif param_name == "convergence_threshold":
            args.convergence_threshold = float(raw_value)
        elif param_name == "min_iterations":
            args.min_iterations = int(raw_value)
        elif param_name in {"sampling_method", "processing_method", "batch_mode", "topology_type"}:
            raise ValueError(
                f"'{param_name}' is structural for this benchmark and must not be set via fixed params."
            )
        else:
            raise ValueError(f"Unsupported FloatSOM fixed parameter for RNG scaling benchmark: {param_name}")


def _build_floatsom_args(
    *,
    args: argparse.Namespace,
    sample_size: int,
    seed: int,
    sampling_method: str,
    fixed_params: Dict[str, Any],
) -> SimpleNamespace:
    sampling_fraction = None
    if sampling_method == "random":
        sampling_fraction = args.sampling_fraction_random
    (
        resolved_chunk_size,
        resolved_zarr_chunk_size,
        chunk_size_source,
        chunk_size_visible_vram_mib,
    ) = _resolve_floatsom_batch_chunking(
        input_dim=int(args.fixed_dimension),
        chunk_size=args.chunk_size,
        zarr_chunk_size=args.zarr_chunk_size,
    )
    if args.temp_dir:
        run_temp_dir_path = (
            Path(args.temp_dir).expanduser().resolve()
            / f"{sampling_method}_samples{int(sample_size)}_seed{int(seed)}"
        )
        run_temp_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        prefix = (
            f"xpysom_rng_{sampling_method}_samples{int(sample_size)}_seed{int(seed)}_"
        )
        run_temp_dir_path = Path(tempfile.mkdtemp(prefix=prefix))
    run_temp_dir = str(run_temp_dir_path)

    out = SimpleNamespace(
        samples=int(sample_size),
        input_dim=int(args.fixed_dimension),
        cache_dir=str(args.cache_dir),
        zarr_chunk_size=int(resolved_zarr_chunk_size),
        force_regenerate=False,
        chunk_size=int(resolved_chunk_size),
        temp_dir=(str(args.temp_dir) if args.temp_dir is not None else None),
        run_temp_dir=str(run_temp_dir),
        ray_local_storage_path=str(args.ray_local_storage_path),
        grid_size=int(args.grid_size),
        total_iterations=int(args.iterations),
        initial_learning_rate=2.0,
        initial_radius=None,
        lr_decay_type="asymptotic",
        radius_decay_type="asymptotic",
        lr_decay_factor=8.0,
        radius_decay_factor=3.0,
        processing_method="batch",
        batch_mode="full_batch",
        minibatch_size=32,
        use_momentum=True,
        initial_momentum=0.5,
        normalization="count_based",
        norm_alpha=None,
        norm_clamp_factor=None,
        norm_percentile=None,
        norm_max_update_threshold=None,
        virtual_ratio=0.5,
        sampling_method=str(sampling_method),
        sampling_fraction=sampling_fraction,
        target_proportion=sampling_fraction,
        topology_type="rng",
        topology_variant="planar",
        initialization_method="random",
        mst_update_frequency=None,
        dynamic_mst_frequency=True,
        mst_decay_function="exponential",
        initial_mst_frequency=1,
        final_mst_frequency=10,
        convergence_threshold=1e-6,
        min_iterations=100,
        save_iterations=False,
        save_every=1,
        verbose=True,
        profile=False,
        profile_output="profile_results.txt",
        safe_cleanup=False,
        profile_workers=False,
        profile_workers_output_dir=None,
        profile_workers_max_stats=50,
        seed=int(seed),
        use_gpu=True,
        output_file=(
            f"xpysom_rng_scaling_{sampling_method}_samples{int(sample_size)}"
            f"_grid{int(args.grid_size)}_gpu{int(args.gpu_count)}_seed{int(seed)}.txt"
        ),
        use_ray=True,
        ray_gpu_count=int(args.ray_gpu_count) if args.ray_gpu_count is not None else int(args.gpu_count),
        ray_collective_group="default",
        ray_collective_barriers=False,
        force_disk_mode=False,
        ray_chunk_size=None,
        minibatch_chunk_size=5_000,
        chunk_size_source=chunk_size_source,
        chunk_size_visible_vram_mib=chunk_size_visible_vram_mib,
    )
    _apply_floatsom_fixed_params(out, fixed_params)
    if sampling_method == "random" and out.sampling_fraction is None:
        raise ValueError(
            "Random FloatSOM scaling requires target_proportion/sampling_fraction in the fixed-params JSON "
            "or via --sampling-fraction-random."
        )
    return out


def _cleanup_run_temp_dir(path: Optional[str]) -> None:
    if not path:
        return
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _load_dense_gpu_array(zarr_path: str) -> cp.ndarray:
    z = zarr.open_array(zarr_path, mode="r")
    return cp.asarray(z[:], dtype=cp.float32)


def _run_xpysom_once(
    *,
    zarr_path: str,
    input_dim: int,
    grid_size: int,
    seed: int,
    epochs: int,
    xpysom_n_parallel: Optional[int],
) -> float:
    model = _create_xpysom_hex(
        input_dim=int(input_dim),
        grid_size=int(grid_size),
        seed=int(seed),
        n_parallel=xpysom_n_parallel,
    )
    start = time.perf_counter()
    data = _load_dense_gpu_array(zarr_path)
    try:
        _train_xpysom(model, train_data=data, epochs=int(epochs))
        return float(time.perf_counter() - start)
    finally:
        del data
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def _run_floatsom_once(
    *,
    floatsom_args: SimpleNamespace,
    zarr_path: str,
    metadata: Dict[str, Any],
) -> float:
    _, _, train_time = train_floatsom_with_zarr(zarr_path, floatsom_args, metadata)
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return float(train_time)


def _enable_floatsom_live_iteration_prints() -> None:
    """Emit iteration-completion progress with plain prints inside this benchmark worker."""
    from floatsom.base.floatsom import FloatSOM

    if getattr(FloatSOM, "_xpysom_rng_live_iteration_prints_enabled", False):
        return

    original_process_training_iteration = FloatSOM._process_training_iteration

    def _wrapped_process_training_iteration(
        self,
        iteration: int,
        data_source,
        data_reference,
        schedules: Dict[str, Any],
        weight_change: float,
        training_stats: Dict[str, Any],
    ) -> float:
        new_weight_change = original_process_training_iteration(
            self,
            iteration,
            data_source,
            data_reference,
            schedules,
            weight_change,
            training_stats,
        )
        total_iterations = int(
            schedules.get(
                "total_iterations",
                getattr(self.params, "total_iterations", int(iteration) + 1),
            )
        )
        sampling_method = str(
            getattr(getattr(self.params, "sampling_config", None), "method", "unknown")
        )
        topology_name = str(getattr(self.topology, "name", "unknown"))
        print(
            f"[floatsom][{sampling_method}][{topology_name}] completed iteration "
            f"{int(iteration) + 1}/{total_iterations}",
            flush=True,
        )
        return new_weight_change

    FloatSOM._process_training_iteration = _wrapped_process_training_iteration
    FloatSOM._xpysom_rng_live_iteration_prints_enabled = True


class BenchmarkRunTimeoutError(Exception):
    """Raised when an individual comparison benchmark run exceeds the hard timeout."""


def _worker_run_xpysom(
    result_queue: multiprocessing.queues.Queue,
    *,
    zarr_path: str,
    input_dim: int,
    grid_size: int,
    seed: int,
    epochs: int,
    xpysom_n_parallel: Optional[int],
) -> None:
    try:
        train_time = _run_xpysom_once(
            zarr_path=zarr_path,
            input_dim=input_dim,
            grid_size=grid_size,
            seed=seed,
            epochs=epochs,
            xpysom_n_parallel=xpysom_n_parallel,
        )
    except Exception as exc:
        result_queue.put(
            (
                "error",
                {
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        )
        return

    result_queue.put(("success", {"train_time": float(train_time)}))


def _worker_run_floatsom(
    result_queue: multiprocessing.queues.Queue,
    *,
    floatsom_args: SimpleNamespace,
    zarr_path: str,
    metadata: Dict[str, Any],
) -> None:
    try:
        ensure_clean_ray_state(free_gpu_memory=False)
        _enable_floatsom_live_iteration_prints()
        train_time = _run_floatsom_once(
            floatsom_args=floatsom_args,
            zarr_path=zarr_path,
            metadata=metadata,
        )
    except Exception as exc:
        result_queue.put(
            (
                "error",
                {
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        )
        return
    finally:
        ensure_clean_ray_state(free_gpu_memory=False)

    result_queue.put(("success", {"train_time": float(train_time)}))


def _run_with_timeout(
    *,
    run_label: str,
    target: Any,
    kwargs: Dict[str, Any],
    timeout_seconds: int = RUN_TIMEOUT_SECONDS,
) -> float:
    ensure_clean_ray_state(free_gpu_memory=False)
    result_queue = MP_CONTEXT.Queue()
    process = MP_CONTEXT.Process(
        target=target,
        kwargs={"result_queue": result_queue, **kwargs},
    )

    try:
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            print(
                f"[timeout] {run_label} exceeded {RUN_TIMEOUT_MINUTES} minutes; terminating run."
            )
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            raise BenchmarkRunTimeoutError(
                f"{run_label} exceeded {RUN_TIMEOUT_MINUTES} minutes"
            )

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(f"{run_label} finished without returning a result") from exc

        if status != "success":
            error_message = str(payload.get("message", "Unknown error"))
            error_type = str(payload.get("error_type", "RuntimeError"))
            trace = payload.get("traceback")
            if trace:
                print(trace)
            raise RuntimeError(f"{error_type}: {error_message}")

        train_time = float(payload["train_time"])
        if not np.isfinite(train_time):
            raise RuntimeError(f"{run_label} returned a non-finite train time")
        return train_time
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
        result_queue.close()
        result_queue.join_thread()
        ensure_clean_ray_state(free_gpu_memory=False)


def _aggregate_scaling_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[Any, Any]]]:
    successful_rows = [
        row
        for row in rows
        if str(row.get("status", "success")).strip().lower() == "success"
        and np.isfinite(float(row.get("train_time_s", float("nan"))))
    ]
    grouped: Dict[str, Dict[str, Dict[int, Dict[int, List[float]]]]] = {}
    for row in successful_rows:
        topology = str(row["topology"]).strip().lower()
        method = str(row["method"]).strip().lower()
        sample_size = int(row["sample_size"])
        gpu_count = int(row["gpu_count"])
        train_time_s = float(row["train_time_s"])
        grouped.setdefault(topology, {}).setdefault(method, {}).setdefault(sample_size, {}).setdefault(gpu_count, [])
        grouped[topology][method][sample_size][gpu_count].append(train_time_s)

    results_by_topology: Dict[str, Dict[str, Dict[Any, Any]]] = {}
    for topology, method_payload in grouped.items():
        results_by_topology[topology] = {}
        for method, sample_payload in method_payload.items():
            results_by_topology[topology][method] = {}
            for sample_size, gpu_payload in sample_payload.items():
                results_by_topology[topology][method][sample_size] = {}
                for gpu_count, times in gpu_payload.items():
                    times_array = np.asarray(times, dtype=float)
                    results_by_topology[topology][method][sample_size][gpu_count] = {
                        "times": [float(value) for value in times_array.tolist()],
                        "mean": float(np.mean(times_array)),
                        "std": float(np.std(times_array)),
                        "count": int(times_array.size),
                    }
    return results_by_topology


def _build_aggregated_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    successful_rows = [
        row
        for row in rows
        if str(row.get("status", "success")).strip().lower() == "success"
        and np.isfinite(float(row.get("train_time_s", float("nan"))))
    ]
    if not successful_rows:
        return pd.DataFrame(
            columns=[
                "topology",
                "method",
                "sample_size",
                "gpu_count",
                "train_time_mean_s",
                "train_time_std_s",
                "n_runs",
            ]
        )
    grouped = (
        pd.DataFrame(successful_rows)
        .groupby(["topology", "method", "sample_size", "gpu_count"], dropna=False)["train_time_s"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "train_time_mean_s", "std": "train_time_std_s", "count": "n_runs"})
        .sort_values(["topology", "method", "sample_size", "gpu_count"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )
    return grouped


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / RAW_TIMES_CSV_FILENAME
    benchmark_staging_root: Optional[Path] = None
    if args.temp_dir:
        temp_dir_root = Path(args.temp_dir).expanduser().resolve()
        temp_dir_root.mkdir(parents=True, exist_ok=True)
        benchmark_staging_root = Path(
            tempfile.mkdtemp(prefix="xpysom_rng_staging_", dir=str(temp_dir_root))
        )

    if args.reset_resume_state and not args.resume:
        raise ValueError("--reset-resume-state requires --resume.")
    if args.overwrite_components and not args.resume:
        raise ValueError("--overwrite-components requires --resume.")

    overwrite_components = _normalize_component_selection(args.overwrite_components)
    skip_components = _normalize_component_selection(args.skip_components)
    resume_manager: Optional[ResumeManager] = None
    existing_success_rows: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        resume_state_path = (
            Path(args.resume_state).expanduser().resolve()
            if args.resume_state is not None
            else output_dir / RESUME_STATE_FILENAME
        )
        resume_manager = ResumeManager(
            state_path=str(resume_state_path),
            enabled=True,
            reset=bool(args.reset_resume_state),
        )
        if not args.reset_resume_state:
            existing_success_rows = _load_existing_success_rows(raw_csv_path=raw_csv_path)
        print(f"Resume manifest: {resume_state_path}")
        if args.reset_resume_state:
            print("Existing resume manifest reset; restarting from scratch.")
        elif existing_success_rows:
            print(
                "Bootstrapping resume state from existing raw timing rows in output directory: "
                f"{raw_csv_path}"
            )
        else:
            print("No existing successful raw timing rows found; resume will start from scratch.")
        if overwrite_components:
            print(
                "Overwrite components on resume: "
                + ", ".join(sorted(overwrite_components))
            )
    if skip_components:
        print(
            "Skip components: "
            + ", ".join(sorted(skip_components))
        )

    sampling_overrides = _load_sampling_overrides(args.fixed_params_by_sampling_topology_json)
    task_rows: Dict[str, Dict[str, Any]] = {}

    try:
        for sample_size in [int(value) for value in args.sample_sizes]:
            for repeat_idx in range(int(args.repeats)):
                seed = int(args.base_seed) + int(repeat_idx)
                xpysom_task_key = _task_key(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    topology="xpysom",
                    method="default",
                )
                xpysom_task_metadata = _task_metadata(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    gpu_count=int(args.gpu_count),
                    topology="xpysom",
                    method="default",
                    sampling_method="full",
                )
                floatsom_full_task_key = _task_key(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    topology="floatsom",
                    method="full",
                )
                floatsom_full_task_metadata = _task_metadata(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    gpu_count=int(args.gpu_count),
                    topology="floatsom",
                    method="full",
                    sampling_method="full",
                )
                floatsom_random_task_key = _task_key(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    topology="floatsom",
                    method="random",
                )
                floatsom_random_task_metadata = _task_metadata(
                    sample_size=int(sample_size),
                    repeat_idx=int(repeat_idx),
                    seed=int(seed),
                    gpu_count=int(args.gpu_count),
                    topology="floatsom",
                    method="random",
                    sampling_method="random",
                )

                print(
                    f"[sample_size={sample_size:,}] repeat={repeat_idx + 1}/{int(args.repeats)} seed={seed}"
                )

                tuple_execution_required = any(
                    (
                        not _task_explicitly_skipped(
                            component=xpysom_task_metadata["component"],
                            skip_components=skip_components,
                        )
                        and not _task_satisfied_by_resume(
                            task_key=xpysom_task_key,
                            component=xpysom_task_metadata["component"],
                            existing_success_rows=existing_success_rows,
                            overwrite_components=overwrite_components,
                        ),
                        not _task_explicitly_skipped(
                            component=floatsom_full_task_metadata["component"],
                            skip_components=skip_components,
                        )
                        and not _task_satisfied_by_resume(
                            task_key=floatsom_full_task_key,
                            component=floatsom_full_task_metadata["component"],
                            existing_success_rows=existing_success_rows,
                            overwrite_components=overwrite_components,
                        ),
                        not _task_explicitly_skipped(
                            component=floatsom_random_task_metadata["component"],
                            skip_components=skip_components,
                        )
                        and not _task_satisfied_by_resume(
                            task_key=floatsom_random_task_key,
                            component=floatsom_random_task_metadata["component"],
                            existing_success_rows=existing_success_rows,
                            overwrite_components=overwrite_components,
                        ),
                    )
                )
                if not tuple_execution_required:
                    for task_key in (
                        xpysom_task_key,
                        floatsom_full_task_key,
                        floatsom_random_task_key,
                    ):
                        if task_key in existing_success_rows:
                            task_rows[task_key] = dict(existing_success_rows[task_key])
                    print(
                        "[resume-skip-or-skip][all-components] "
                        f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                    )
                    if resume_manager and resume_manager.enabled:
                        if xpysom_task_key in task_rows:
                            resume_manager.mark_done(
                                xpysom_task_key,
                                train_time=float(task_rows[xpysom_task_key]["train_time_s"]),
                                metadata=xpysom_task_metadata,
                            )
                        if floatsom_full_task_key in task_rows:
                            resume_manager.mark_done(
                                floatsom_full_task_key,
                                train_time=float(task_rows[floatsom_full_task_key]["train_time_s"]),
                                metadata=floatsom_full_task_metadata,
                            )
                        if floatsom_random_task_key in task_rows:
                            resume_manager.mark_done(
                                floatsom_random_task_key,
                                train_time=float(task_rows[floatsom_random_task_key]["train_time_s"]),
                                metadata=floatsom_random_task_metadata,
                            )
                    continue

                data_args = _create_data_args(
                    cache_dir=str(args.cache_dir),
                    sample_size=int(sample_size),
                    input_dim=int(args.fixed_dimension),
                    chunk_size=args.chunk_size,
                    zarr_chunk_size=args.zarr_chunk_size,
                    seed=int(seed),
                    force_regenerate=bool(args.force_regenerate),
                    verbose=bool(args.verbose),
                )
                stage_info: Optional[Dict[str, Any]] = None
                try:
                    zarr_path, metadata = generate_or_load_data(data_args)
                    stage_info = stage_dataset_for_execution(
                        zarr_path,
                        temp_root=args.temp_dir,
                        staging_root=str(benchmark_staging_root) if benchmark_staging_root else None,
                        verbose=bool(args.verbose),
                    )
                    execution_zarr_path = str(stage_info["execution_path"])
                    staged_metadata = dict(metadata)
                    staged_metadata["source_zarr_path"] = str(stage_info["source_path"])
                    staged_metadata["execution_zarr_path"] = execution_zarr_path
                    staged_metadata["staging_applied"] = bool(stage_info["staged"])
                    staged_metadata["staging_reason"] = str(stage_info["staging_reason"])
                    staged_metadata["staging_elapsed_s"] = float(stage_info["staging_elapsed_s"])
                    staged_metadata["staging_copy_method"] = str(stage_info["staging_copy_method"])
                    staged_metadata["zarr_path"] = execution_zarr_path

                    if _task_explicitly_skipped(
                        component=xpysom_task_metadata["component"],
                        skip_components=skip_components,
                    ):
                        print(
                            "[xpysom][skip] "
                            f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                        )
                        if xpysom_task_key in existing_success_rows:
                            task_rows[xpysom_task_key] = dict(existing_success_rows[xpysom_task_key])
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_done(
                                    xpysom_task_key,
                                    train_time=float(task_rows[xpysom_task_key]["train_time_s"]),
                                    metadata=xpysom_task_metadata,
                                )
                    elif (
                        _task_satisfied_by_resume(
                            task_key=xpysom_task_key,
                            component=xpysom_task_metadata["component"],
                            existing_success_rows=existing_success_rows,
                            overwrite_components=overwrite_components,
                        )
                    ):
                        task_rows[xpysom_task_key] = dict(existing_success_rows[xpysom_task_key])
                        print(
                            "[xpysom][resume-skip] "
                            f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                        )
                        if resume_manager and resume_manager.enabled:
                            resume_manager.mark_done(
                                xpysom_task_key,
                                train_time=float(task_rows[xpysom_task_key]["train_time_s"]),
                                metadata=xpysom_task_metadata,
                            )
                    else:
                        if resume_manager and resume_manager.enabled:
                            resume_manager.register_task(xpysom_task_key, metadata=xpysom_task_metadata)
                            resume_manager.mark_running(xpysom_task_key)
                        try:
                            xpysom_time = _run_with_timeout(
                                run_label=(
                                    f"xpysom samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                                ),
                                target=_worker_run_xpysom,
                                kwargs={
                                    "zarr_path": execution_zarr_path,
                                    "input_dim": int(args.fixed_dimension),
                                    "grid_size": int(args.grid_size),
                                    "seed": int(seed),
                                    "epochs": int(args.iterations),
                                    "xpysom_n_parallel": args.xpysom_n_parallel,
                                },
                            )
                        except Exception as exc:
                            print(
                                "[xpysom][failed] "
                                f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed} "
                                f"{type(exc).__name__}: {exc}"
                            )
                            task_rows[xpysom_task_key] = _build_result_row(
                                sample_size=int(sample_size),
                                repeat_idx=int(repeat_idx),
                                seed=int(seed),
                                gpu_count=int(args.gpu_count),
                                topology="xpysom",
                                method="default",
                                sampling_method="full",
                                train_time_s=None,
                                status="failed",
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_failed(xpysom_task_key, error=str(exc))
                        else:
                            task_rows[xpysom_task_key] = _build_result_row(
                                sample_size=int(sample_size),
                                repeat_idx=int(repeat_idx),
                                seed=int(seed),
                                gpu_count=int(args.gpu_count),
                                topology="xpysom",
                                method="default",
                                sampling_method="full",
                                train_time_s=float(xpysom_time),
                                status="success",
                            )
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_done(
                                    xpysom_task_key,
                                    train_time=float(xpysom_time),
                                    metadata=xpysom_task_metadata,
                                )
                        _persist_raw_rows(task_rows, raw_csv_path)

                    for sampling_method in ("full", "random"):
                        if sampling_method == "full":
                            floatsom_task_key = floatsom_full_task_key
                            floatsom_task_metadata = floatsom_full_task_metadata
                        else:
                            floatsom_task_key = floatsom_random_task_key
                            floatsom_task_metadata = floatsom_random_task_metadata
                        if _task_explicitly_skipped(
                            component=floatsom_task_metadata["component"],
                            skip_components=skip_components,
                        ):
                            print(
                                f"[floatsom:{sampling_method}][skip] "
                                f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                            )
                            if floatsom_task_key in existing_success_rows:
                                task_rows[floatsom_task_key] = dict(existing_success_rows[floatsom_task_key])
                                if resume_manager and resume_manager.enabled:
                                    resume_manager.mark_done(
                                        floatsom_task_key,
                                        train_time=float(task_rows[floatsom_task_key]["train_time_s"]),
                                        metadata=floatsom_task_metadata,
                                    )
                            continue
                        if (
                            _task_satisfied_by_resume(
                                task_key=floatsom_task_key,
                                component=floatsom_task_metadata["component"],
                                existing_success_rows=existing_success_rows,
                                overwrite_components=overwrite_components,
                            )
                        ):
                            task_rows[floatsom_task_key] = dict(existing_success_rows[floatsom_task_key])
                            print(
                                f"[floatsom:{sampling_method}][resume-skip] "
                                f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed}"
                            )
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_done(
                                    floatsom_task_key,
                                    train_time=float(task_rows[floatsom_task_key]["train_time_s"]),
                                    metadata=floatsom_task_metadata,
                                )
                            continue

                        floatsom_args = _build_floatsom_args(
                            args=args,
                            sample_size=int(sample_size),
                            seed=int(seed),
                            sampling_method=sampling_method,
                            fixed_params=sampling_overrides[sampling_method],
                        )
                        if resume_manager and resume_manager.enabled:
                            resume_manager.register_task(floatsom_task_key, metadata=floatsom_task_metadata)
                            resume_manager.mark_running(floatsom_task_key)
                        try:
                            train_time = _run_with_timeout(
                                run_label=(
                                    f"floatsom:{sampling_method} samples={sample_size:,} "
                                    f"repeat={repeat_idx + 1} seed={seed}"
                                ),
                                target=_worker_run_floatsom,
                                kwargs={
                                    "floatsom_args": floatsom_args,
                                    "zarr_path": execution_zarr_path,
                                    "metadata": dict(staged_metadata),
                                },
                            )
                        except Exception as exc:
                            print(
                                f"[floatsom:{sampling_method}][failed] "
                                f"samples={sample_size:,} repeat={repeat_idx + 1} seed={seed} "
                                f"{type(exc).__name__}: {exc}"
                            )
                            task_rows[floatsom_task_key] = _build_result_row(
                                sample_size=int(sample_size),
                                repeat_idx=int(repeat_idx),
                                seed=int(seed),
                                gpu_count=int(args.gpu_count),
                                topology="floatsom",
                                method=sampling_method,
                                sampling_method=sampling_method,
                                train_time_s=None,
                                status="failed",
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_failed(floatsom_task_key, error=str(exc))
                            _persist_raw_rows(task_rows, raw_csv_path)
                        else:
                            task_rows[floatsom_task_key] = _build_result_row(
                                sample_size=int(sample_size),
                                repeat_idx=int(repeat_idx),
                                seed=int(seed),
                                gpu_count=int(args.gpu_count),
                                topology="floatsom",
                                method=sampling_method,
                                sampling_method=sampling_method,
                                train_time_s=float(train_time),
                                status="success",
                            )
                            if resume_manager and resume_manager.enabled:
                                resume_manager.mark_done(
                                    floatsom_task_key,
                                    train_time=float(train_time),
                                    metadata=floatsom_task_metadata,
                                )
                            _persist_raw_rows(task_rows, raw_csv_path)
                        finally:
                            _cleanup_run_temp_dir(getattr(floatsom_args, "run_temp_dir", None))
                finally:
                    cleanup_staged_dataset(stage_info, verbose=bool(args.verbose))
    finally:
        if benchmark_staging_root is not None:
            shutil.rmtree(benchmark_staging_root, ignore_errors=True)

    raw_df = _build_raw_frame(task_rows.values())
    raw_df.to_csv(raw_csv_path, index=False)

    raw_rows = list(task_rows.values())
    aggregated_df = _build_aggregated_frame(raw_rows)
    aggregated_csv_path = output_dir / AGGREGATED_STATS_CSV_FILENAME
    aggregated_df.to_csv(aggregated_csv_path, index=False)

    scaling_results = _aggregate_scaling_rows(raw_rows)
    panel_dir = output_dir / "scaling_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    plot_topology_comparison_scaling_benchmark(
        results_by_topology=scaling_results,
        output_dir=str(panel_dir),
        title="XPySOM Defaults vs FloatSOM RNG Scaling",
        axis_mode="sample",
        comparison_gpu_count=int(args.gpu_count),
        show_speedup=False,
        show_error_bars=True,
        show_legend=True,
        methods_to_plot=["default", "full", "random"],
        emit_shared_legend=False,
        include_performance=True,
        topology_color_overrides={
            "xpysom": "#1f77b4",
            "floatsom": "#d62728",
        },
        method_linestyle_overrides={
            "default": "-",
            "full": "-",
            "random": ":",
        },
    )
    scaling_svg_path = panel_dir / "sample_scaling_performance.svg"
    if not scaling_svg_path.exists():
        raise FileNotFoundError(
            f"Expected scaling panel SVG was not generated: {scaling_svg_path}"
        )

    final_panel_path = output_dir / SCALING_PANEL_FILENAME
    final_panel_path.write_text(scaling_svg_path.read_text(encoding="utf-8"), encoding="utf-8")

    metadata = {
        "output_dir": str(output_dir),
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "temp_dir": (
            str(Path(args.temp_dir).expanduser().resolve())
            if args.temp_dir is not None
            else None
        ),
        "ray_local_storage_path": str(Path(args.ray_local_storage_path).expanduser().resolve()),
        "fixed_params_by_sampling_topology_json": str(
            Path(args.fixed_params_by_sampling_topology_json).expanduser().resolve()
        ),
        "sample_sizes": [int(value) for value in args.sample_sizes],
        "fixed_dimension": int(args.fixed_dimension),
        "grid_size": int(args.grid_size),
        "gpu_count": int(args.gpu_count),
        "ray_gpu_count": int(args.ray_gpu_count) if args.ray_gpu_count is not None else int(args.gpu_count),
        "repeats": int(args.repeats),
        "base_seed": int(args.base_seed),
        "iterations": int(args.iterations),
        "run_timeout_minutes": RUN_TIMEOUT_MINUTES,
        "xpysom_n_parallel": int(args.xpysom_n_parallel) if args.xpysom_n_parallel is not None else None,
        "resume": bool(args.resume),
        "resume_state": (
            str(Path(args.resume_state).expanduser().resolve())
            if args.resume_state is not None
            else (str(output_dir / RESUME_STATE_FILENAME) if args.resume else None)
        ),
        "reset_resume_state": bool(args.reset_resume_state),
        "overwrite_components": sorted(overwrite_components),
        "skip_components": sorted(skip_components),
        "raw_runs_csv": str(raw_csv_path),
        "aggregated_stats_csv": str(aggregated_csv_path),
        "scaling_panel_svg": str(final_panel_path),
    }
    metadata_path = output_dir / RUN_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote raw timing rows to: {raw_csv_path}")
    print(f"Wrote aggregated timing stats to: {aggregated_csv_path}")
    print(f"Wrote scaling panel to: {final_panel_path}")
    print(f"Wrote metadata to: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
