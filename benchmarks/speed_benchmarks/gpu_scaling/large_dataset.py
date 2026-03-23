"""Utility functions for large dataset FloatSOM benchmarks."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zarr

from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader
from floatsom.data.sources.file import FileDataSource
from floatsom.floatsom_params import (
    FloatSOMParams,
    ProcessingConfig,
    SamplingConfig,
    TopologyConfig,
)
from floatsom.processing.processing_params import (
    CleanupConfig,
    WorkerProfileConfig,
    calculate_auto_chunk_size_for_method,
    get_visible_gpu_vram_mib,
)

# Ray is optional; record availability so callers can surface a clear error.
try:
    import ray  # type: ignore  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


VERY_LARGE_CACHE_SKIP_OPEN_RAW_GIB = 64.0


def generate_random_zarr_data(
    n_samples: int,
    input_dim: int,
    zarr_path: str,
    zarr_chunk_size: int = 10_000,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> zarr.Array:
    """Generate random data and persist it to a Zarr array."""

    if seed is not None:
        np.random.seed(seed)

    if verbose:
        size_gb = (n_samples * input_dim * 4) / (1024 ** 3)
        print(
            f"Generating random data: {n_samples:,} samples × {input_dim} dimensions",
            f"Estimated size: {size_gb:.2f} GB (uncompressed)",
            f"Zarr path: {zarr_path}",
            f"Zarr chunk size: {zarr_chunk_size:,} samples",
            sep="\n",
        )

    z = zarr.open_array(
        zarr_path,
        mode="w",
        shape=(n_samples, input_dim),
        chunks=(min(zarr_chunk_size, n_samples), input_dim),
        dtype="float32",
    )

    for start_idx in range(0, n_samples, zarr_chunk_size):
        end_idx = min(start_idx + zarr_chunk_size, n_samples)
        chunk_data = np.random.rand(end_idx - start_idx, input_dim).astype(np.float32)
        z[start_idx:end_idx] = chunk_data

        if verbose and (start_idx % (zarr_chunk_size * 10) == 0 or end_idx == n_samples):
            progress = 100 * end_idx / n_samples
            print(f"  Generated {end_idx:,}/{n_samples:,} samples ({progress:.1f}%)")

    if verbose:
        actual_size_mb = 0.0
        if os.path.isdir(zarr_path):
            actual_size_mb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(zarr_path)
                for filename in filenames
            ) / (1024 ** 2)
        elif os.path.isfile(zarr_path):
            actual_size_mb = os.path.getsize(zarr_path) / (1024 ** 2)

        print(f"Data generation complete. Actual size: {actual_size_mb:.1f} MB (compressed)")

    return z


def generate_or_load_data(args) -> Tuple[str, Dict[str, Any]]:
    """Generate a random dataset or load it from cache."""

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_hash = hashlib.md5(
        f"random_{args.samples}_{args.input_dim}_{args.seed}_{args.zarr_chunk_size}".encode()
    ).hexdigest()[:8]
    zarr_path = str(cache_dir / f"random_{args.samples}_{args.input_dim}_{config_hash}.zarr")

    def _regenerate_dataset() -> zarr.Array:
        print("Generating new random dataset...")
        return generate_random_zarr_data(
            n_samples=args.samples,
            input_dim=args.input_dim,
            zarr_path=zarr_path,
            zarr_chunk_size=args.zarr_chunk_size,
            seed=args.seed,
            verbose=getattr(args, "verbose", False),
        )

    def _remove_cached_path(path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _is_recoverable_cache_error(exc: Exception) -> bool:
        if isinstance(exc, (FileNotFoundError, KeyError, UnicodeDecodeError, JSONDecodeError)):
            return True
        if isinstance(exc, ValueError):
            message = str(exc).lower()
            if "expecting value" in message or "json" in message:
                return True
        return False

    estimated_raw_gib = (
        int(args.samples) * int(args.input_dim) * np.dtype(np.float32).itemsize
    ) / (1024 ** 3)

    if os.path.exists(zarr_path) and not getattr(args, "force_regenerate", False):
        if estimated_raw_gib >= VERY_LARGE_CACHE_SKIP_OPEN_RAW_GIB:
            print(
                "Reusing large cached data without eager open: "
                f"{zarr_path} (estimated raw size: {estimated_raw_gib:.2f} GiB)"
            )
            metadata = {
                "dataset_name": "random",
                "shape": (int(args.samples), int(args.input_dim)),
                "dtype": str(np.dtype(np.float32)),
                "chunks": (min(int(args.zarr_chunk_size), int(args.samples)), int(args.input_dim)),
                "zarr_path": zarr_path,
            }
            return zarr_path, metadata

        print(f"Loading cached data from {zarr_path}")
        try:
            z = zarr.open_array(zarr_path, mode="r")
        except Exception as exc:
            if not _is_recoverable_cache_error(exc):
                raise
            print(
                f"Cached dataset appears invalid ({exc}); removing and regenerating: {zarr_path}"
            )
            _remove_cached_path(zarr_path)
            z = _regenerate_dataset()
    else:
        z = _regenerate_dataset()

    metadata = {
        "dataset_name": "random",
        "shape": z.shape,
        "dtype": str(z.dtype),
        "chunks": z.chunks,
        "zarr_path": zarr_path,
    }

    return zarr_path, metadata


def _path_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _copy_path_best_effort(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    cp_path = shutil.which("cp")
    if cp_path:
        commands = (
            ([cp_path, "-a", "--reflink=auto", "--sparse=auto", str(source), str(destination)], "cp_archive_reflink"),
            ([cp_path, "-a", str(source), str(destination)], "cp_archive"),
        )
        for command, label in commands:
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return label
            except Exception:
                _remove_path(destination)

    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        return "shutil_copytree"

    shutil.copy2(source, destination)
    return "shutil_copy2"


def stage_dataset_for_execution(
    source_path: str | Path,
    *,
    temp_root: str | Path | None,
    staging_root: str | Path | None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Stage a file-backed dataset into run-local scratch when an explicit temp root is
    provided and the source does not already live there.
    """
    source = Path(source_path).expanduser().resolve()
    info: Dict[str, Any] = {
        "source_path": str(source),
        "execution_path": str(source),
        "staged": False,
        "staging_reason": "staging disabled",
        "staging_elapsed_s": 0.0,
        "staging_copy_method": "none",
        "temp_root": None,
        "staging_root": None,
        "cleanup_path": None,
    }

    if temp_root is None or staging_root is None:
        info["staging_reason"] = "no temp_root/staging_root provided"
        return info

    temp_root_path = Path(temp_root).expanduser().resolve()
    staging_root_path = Path(staging_root).expanduser().resolve()
    info["temp_root"] = str(temp_root_path)
    info["staging_root"] = str(staging_root_path)

    if _path_within(temp_root_path, source) or _path_within(source, temp_root_path):
        info["staging_reason"] = (
            "source already overlaps temp_root; using source path directly"
        )
        return info

    staged_parent = staging_root_path / "staged_inputs"
    dataset_tag = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:8]
    staged_path = staged_parent / f"{source.stem}_{dataset_tag}{source.suffix}"

    _remove_path(staged_path)

    if verbose:
        print(f"Staging dataset to scratch: {source} -> {staged_path}")

    start = time.perf_counter()
    copy_method = _copy_path_best_effort(source, staged_path)
    elapsed_s = float(time.perf_counter() - start)

    info.update(
        {
            "execution_path": str(staged_path),
            "staged": True,
            "staging_reason": "source outside temp_root; staged before execution",
            "staging_elapsed_s": elapsed_s,
            "staging_copy_method": copy_method,
            "cleanup_path": str(staged_path),
        }
    )

    if verbose:
        print(
            "Dataset staging complete: "
            f"elapsed_s={elapsed_s:.3f} method={copy_method} path={staged_path}"
        )

    return info


def cleanup_staged_dataset(
    stage_info: Optional[Dict[str, Any]],
    *,
    verbose: bool = False,
) -> None:
    """Remove a staged dataset copy when this run no longer needs it."""
    if not isinstance(stage_info, dict) or not bool(stage_info.get("staged", False)):
        return

    cleanup_path_raw = stage_info.get("cleanup_path")
    if not cleanup_path_raw:
        return

    cleanup_path = Path(str(cleanup_path_raw)).expanduser().resolve()
    _remove_path(cleanup_path)
    if verbose:
        print(f"Removed staged dataset: {cleanup_path}")


def train_floatsom_with_zarr(
    zarr_path: str,
    args,
    metadata: Dict[str, Any],
) -> Tuple[FloatSOM, Dict[str, Any], float]:
    """Train FloatSOM using a Zarr-backed dataset."""

    n_samples, input_dim = metadata["shape"]

    source_zarr_path = str(metadata.get("source_zarr_path", metadata.get("zarr_path", zarr_path)))
    execution_zarr_path = str(metadata.get("execution_zarr_path", zarr_path))

    print("\nConfiguring FloatSOM for large dataset processing...")
    print(f"Execution data source: Zarr array at {execution_zarr_path}")
    if source_zarr_path != execution_zarr_path:
        print(f"Original data source: Zarr array at {source_zarr_path}")
        staging_elapsed_s = metadata.get("staging_elapsed_s")
        reused_prepared_stage = bool(
            metadata.get("staging_reused_from_prepared_dataset", False)
        )
        if reused_prepared_stage:
            print("Shared staged dataset: reusing execution copy prepared earlier")
            if isinstance(staging_elapsed_s, (int, float)):
                print(f"Original staging elapsed: {float(staging_elapsed_s):.3f}s")
        elif isinstance(staging_elapsed_s, (int, float)):
            print(f"Dataset staging elapsed: {float(staging_elapsed_s):.3f}s")
    print(f"Shape: {n_samples:,} × {input_dim}")

    sampling_fraction = getattr(args, "sampling_fraction", None)
    if sampling_fraction is None:
        sampling_fraction = getattr(args, "target_proportion", None)

    dataset_samples = metadata["shape"][0]
    samples_per_epoch = None

    if args.sampling_method != "full":
        if sampling_fraction is None:
            raise ValueError(
                "sampling_fraction must be provided when using non-full sampling methods."
            )
        samples_per_epoch = max(1, int(dataset_samples * sampling_fraction))

    sampling_config = SamplingConfig(
        method=args.sampling_method,
        samples_per_epoch=samples_per_epoch,
        target_proportion=sampling_fraction,
        random_seed=args.seed,
    )

    setattr(args, "computed_samples_per_epoch", samples_per_epoch)

    norm_alpha = getattr(args, "norm_alpha", None)
    norm_clamp_factor = getattr(args, "norm_clamp_factor", None)
    norm_percentile = getattr(args, "norm_percentile", None)
    norm_max_update_threshold = getattr(args, "norm_max_update_threshold", None)

    if args.normalization == "hybrid" and norm_alpha is None:
        norm_alpha = 0.5
    elif args.normalization == "clamped_weighted" and norm_clamp_factor is None:
        norm_clamp_factor = 2.0
    elif args.normalization == "local" and norm_percentile is None:
        norm_percentile = 85.0

    optimal_chunk_size = args.chunk_size
    chunk_size_source = getattr(args, "chunk_size_source", None)
    if optimal_chunk_size is None:
        optimal_chunk_size = calculate_auto_chunk_size_for_method(
            input_dim, args.processing_method
        )
        chunk_size_source = "auto_dimension_vram"

    ray_config = None
    temp_run_dir = args.run_temp_dir
    if args.use_ray and args.processing_method in ["batch", "colors"]:
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray requested but not available. Install ray or disable --use_ray."
            )

        from floatsom.floatsom_params import RayConfig

        ray_config = RayConfig(
            num_gpus=args.ray_gpu_count,
            collective_group_name=args.ray_collective_group,
            chunk_size=optimal_chunk_size,
            cache_path=temp_run_dir,
            storage_path=temp_run_dir,
            local_storage_path=args.ray_local_storage_path,
            enable_collective_barriers=bool(getattr(args, "ray_collective_barriers", True)),
            force_disk_mode=bool(getattr(args, "force_disk_mode", False)),
        )

    if getattr(args, "verbose", False):
        if chunk_size_source == "manual_cli_chunk_size":
            source_note = "manual override from --chunk_size"
        elif chunk_size_source == "manual_minibatch_chunk_size":
            source_note = "manual override from --minibatch_chunk_size"
        else:
            visible_vram_mib = getattr(args, "chunk_size_visible_vram_mib", None)
            if visible_vram_mib is None:
                visible_vram_mib = get_visible_gpu_vram_mib()
            source_note = (
                f"auto-resolved from dimension={input_dim} and "
                f"visible_vram={int(visible_vram_mib):,} MiB"
            )
        print(f"Using chunk size: {optimal_chunk_size:,} ({source_note})")

    worker_profile_config = None
    if getattr(args, "profile_workers", False):
        worker_profile_config = WorkerProfileConfig(
            enabled=True,
            output_dir=getattr(args, "profile_workers_output_dir", None),
            max_stats=getattr(args, "profile_workers_max_stats", 50),
        )

    processing_config = ProcessingConfig(
        method=args.processing_method,
        batch_mode=args.batch_mode,
        chunk_size=optimal_chunk_size,
        enable_momentum=args.use_momentum,
        initial_momentum=args.initial_momentum,
        normalization=args.normalization,
        norm_alpha=norm_alpha,
        norm_clamp_factor=norm_clamp_factor,
        norm_percentile=norm_percentile,
        norm_max_update_threshold=norm_max_update_threshold,
        virtual_ratio=args.virtual_ratio,
        use_gpu=args.use_gpu,
        ray_config=ray_config,
        cleanup_config=CleanupConfig(safe_cleanup=getattr(args, "safe_cleanup", False)),
        worker_profile_config=worker_profile_config,
    )

    topology_config = TopologyConfig(
        topology_type=args.topology_type,
        topology_variant=args.topology_variant,
        grid_size=args.grid_size,
        grid_dim=2,
        mst_update_frequency=args.mst_update_frequency,
        dynamic_mst_frequency=args.dynamic_mst_frequency,
        mst_decay_function=args.mst_decay_function,
        initial_mst_frequency=args.initial_mst_frequency,
        final_mst_frequency=args.final_mst_frequency,
    )

    params = FloatSOMParams(
        input_dim=input_dim,
        total_iterations=args.total_iterations,
        initial_learning_rate=args.initial_learning_rate,
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
    )

    som = create_floatsom(params)

    use_ray_streaming = bool(args.use_ray and args.processing_method in ["batch", "colors"])
    if use_ray_streaming:
        training_input = FileDataSource(zarr_path)
    else:
        # Local (non-Ray) processors require in-memory samples, not a file path reference.
        dataset_size_gib = (n_samples * input_dim * np.dtype("float32").itemsize) / (1024 ** 3)
        print(
            "Ray disabled for this run; loading dataset into memory for local processing "
            f"(~{dataset_size_gib:.2f} GiB)."
        )
        zarr_array = zarr.open_array(zarr_path, mode="r")
        training_input = np.asarray(zarr_array, dtype=np.float32)
        print(
            f"In-memory dataset ready: {training_input.shape[0]:,} samples × "
            f"{training_input.shape[1]:,} dimensions"
        )

    print("\nTraining FloatSOM...")
    print(f"Configuration: {args.sampling_method} sampling, {args.processing_method} processing")
    print(f"Grid: {args.grid_size}×{args.grid_size} {args.topology_type}")
    print(f"Iterations: {args.total_iterations}")
    if args.sampling_method != "full" and sampling_fraction is not None:
        print(f"Sampling fraction: {sampling_fraction:.2%}")
        print(f"Samples per iteration: {samples_per_epoch:,}")

    # Enable verbose logging if requested
    if getattr(args, "verbose", False):
        print(f"\n{'='*60}")
        print("VERBOSE MODE ENABLED - Full logging active")
        print(f"Processing: {args.processing_method} with {args.ray_gpu_count if args.use_ray else 1} GPU(s)")
        print(f"Chunk size: {optimal_chunk_size:,}")
        print(f"Ray config: {ray_config}")
        print(f"{'='*60}\n")

    start_time = time.time()
    training_stats = som.train(training_input)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f}s")

    return som, training_stats, train_time


def calculate_quantization_error(
    zarr_path: str,
    som: FloatSOM,
    args,
    output_dir: str,
    *,
    report_dataset_path: Optional[str] = None,
) -> float:
    """Calculate the quantization error for a trained SOM."""

    import cupy as cp  # type: ignore

    from floatsom.data.convert_zarr_to_fast import convert_zarr_to_fast
    from floatsom.processing.utils import find_bmus

    print("\nCalculating quantization error...")

    data_path = zarr_path
    report_path = str(report_dataset_path or zarr_path)
    fast_path = None
    is_fast_format = os.path.exists(os.path.join(data_path, "array_metadata.json"))

    if not is_fast_format:
        print("  Detected Zarr format - converting to FastArrayStore for optimized processing")
        fast_path = os.path.join(output_dir, "temp_fast_array.fast")
        grid_nodes = args.grid_size * args.grid_size
        convert_chunk_size = min(50_000, max(10_000, int(2 * 1024 ** 3 / (grid_nodes * 4))))
        convert_zarr_to_fast(data_path, fast_path, chunk_size=convert_chunk_size)
        data_path = fast_path
        print(f"  Conversion complete - using FastArrayStore at {fast_path}")
    else:
        print("  Detected FastArrayStore format - using optimized loader")

    grid_nodes = args.grid_size * args.grid_size
    optimal_chunk_size = min(50_000, max(10_000, int(2 * 1024 ** 3 / (grid_nodes * 4))))

    print(f"  Using optimized chunk size: {optimal_chunk_size:,} samples")
    print(f"  Grid nodes: {grid_nodes} ({args.grid_size}×{args.grid_size})")

    loader = CPUGPUFastLoader(
        data_path,
        chunk_size=optimal_chunk_size,
        randomize_chunks=False,
        prefetch_buffer_size=1,
    )
    n_samples = loader.n_samples

    weights = som.get_weights()
    if isinstance(weights, np.ndarray):
        weights = cp.array(weights)

    total_error = 0.0
    for chunk_idx in range(loader.total_chunks):
        gpu_data, _ = loader.get_chunk(chunk_idx)
        _, min_distances = find_bmus(gpu_data, weights, return_distances=True, chunk_size=None)
        total_error += cp.sum(min_distances).item()

        if chunk_idx % max(1, loader.total_chunks // 20) == 0 or chunk_idx == loader.total_chunks - 1:
            samples_processed = min((chunk_idx + 1) * optimal_chunk_size, n_samples)
            progress = 100 * samples_processed / n_samples
            print(f"  Processed {samples_processed:,}/{n_samples:,} samples ({progress:.1f}%)")

    del loader

    avg_error = total_error / n_samples
    original_format = (
        "Zarr (converted to FastArrayStore)" if not is_fast_format else "FastArrayStore"
    )
    qe_file = os.path.join(output_dir, "quantization_error.txt")

    with open(qe_file, "w") as f:
        f.write(
            "\n".join(
                [
                    f"Quantization Error Analysis - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 60,
                    "",
                    f"Dataset: {report_path}",
                    f"Format: {original_format}",
                    f"Total samples: {n_samples:,}",
                    f"SOM grid size: {args.grid_size}×{args.grid_size}",
                    "",
                    f"Total quantization error: {total_error:.6f}",
                    f"Average quantization error: {avg_error:.6f}",
                    f"Average error per dimension: {avg_error / args.input_dim:.6f}",
                ]
            )
        )

    print(f"Quantization error: {avg_error:.6f}")
    print(f"Results saved to: {qe_file}")

    if fast_path and os.path.exists(fast_path):
        try:
            import shutil

            shutil.rmtree(fast_path)
            print("  Cleaned up temporary FastArrayStore")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"  Warning: Could not clean up temporary FastArrayStore: {exc}")

    return avg_error


def write_results_report(
    som: FloatSOM,
    training_stats: Dict[str, Any],
    train_time: float,
    metadata: Dict[str, Any],
    args,
    output_file: str,
) -> None:
    """Write a detailed results report for a benchmark run."""

    sampling_fraction = getattr(args, "sampling_fraction", None)
    if sampling_fraction is None:
        sampling_fraction = getattr(args, "target_proportion", None)
    computed_samples = getattr(args, "computed_samples_per_epoch", None)

    with open(output_file, "w") as f:
        f.write(
            f"FloatSOM Large Dataset Benchmark Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write("Dataset Information:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset type: Random uniform [0, 1]\n")
        f.write(f"Shape: {metadata['shape'][0]:,} × {metadata['shape'][1]}\n")
        f.write(f"Data type: {metadata['dtype']}\n")
        f.write(f"Zarr chunks: {metadata['chunks']}\n")
        source_path = str(metadata.get("source_zarr_path", metadata["zarr_path"]))
        execution_path = str(metadata.get("execution_zarr_path", metadata["zarr_path"]))
        f.write(f"Source path: {source_path}\n")
        f.write(f"Execution path: {execution_path}\n")
        if bool(metadata.get("staging_applied", False)):
            f.write("Dataset staging: enabled\n")
            staging_elapsed_s = metadata.get("staging_elapsed_s")
            if isinstance(staging_elapsed_s, (int, float)):
                f.write(f"Dataset staging elapsed: {float(staging_elapsed_s):.3f}s\n")
            staging_copy_method = metadata.get("staging_copy_method")
            if staging_copy_method:
                f.write(f"Dataset staging copy method: {staging_copy_method}\n")
        else:
            f.write("Dataset staging: disabled\n")
        f.write("\n")

        f.write("SOM Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Grid size: {args.grid_size}×{args.grid_size}\n")
        f.write(f"Topology: {args.topology_type} ({args.topology_variant})\n")
        if args.topology_type == "mst":
            if args.dynamic_mst_frequency:
                f.write(
                    "MST update: Dynamic ("
                    f"initial: every {args.initial_mst_frequency} iter, "
                    f"final: every {args.final_mst_frequency} iter, "
                    f"decay: {args.mst_decay_function})\n"
                )
            else:
                freq = args.mst_update_frequency if args.mst_update_frequency is not None else "dynamic"
                f.write(f"MST update frequency: Every {freq} iterations\n")
        f.write(f"Total iterations: {args.total_iterations}\n")
        f.write(f"Initial learning rate: {args.initial_learning_rate}\n")
        f.write(f"Initial radius: {getattr(args, 'initial_radius', 'auto') or 'auto'}\n")
        f.write(f"LR decay type: {args.lr_decay_type}\n")
        f.write(f"LR decay factor: {args.lr_decay_factor}\n")
        f.write(f"Radius decay type: {args.radius_decay_type}\n")
        f.write(f"Radius decay factor: {args.radius_decay_factor}\n")
        f.write(f"Convergence threshold: {args.convergence_threshold}\n")
        f.write(f"Min iterations: {args.min_iterations}\n")
        f.write(f"Initialization: {args.initialization_method}\n")
        f.write(f"Store history: {args.save_iterations}\n\n")

        f.write("Processing Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Sampling method: {args.sampling_method}\n")
        if args.sampling_method != "full" and sampling_fraction is not None:
            f.write(f"Sampling fraction: {sampling_fraction:.6f}\n")
            if computed_samples is not None:
                f.write(f"Samples per iteration: {computed_samples:,}\n")
        f.write(f"Processing method: {args.processing_method}\n")
        f.write(f"Batch mode: {args.batch_mode}\n")
        if getattr(args, "zarr_chunk_size", None):
            f.write(f"Zarr chunk size: {args.zarr_chunk_size}\n")
        f.write(f"Normalization: {args.normalization}\n")
        if args.normalization == "hybrid" and getattr(args, "norm_alpha", None):
            f.write(f"  Alpha: {args.norm_alpha}\n")
        if args.normalization == "clamped_weighted" and getattr(args, "norm_clamp_factor", None):
            f.write(f"  Clamp factor: {args.norm_clamp_factor}\n")
        if args.normalization == "local" and getattr(args, "norm_percentile", None):
            f.write(f"  Percentile: {args.norm_percentile}\n")
        if getattr(args, "norm_max_update_threshold", None):
            f.write(f"  Max update threshold: {args.norm_max_update_threshold}\n")
        f.write(f"Virtual ratio: {args.virtual_ratio}\n")
        f.write(f"Momentum: {'enabled' if args.use_momentum else 'disabled'}\n")
        if args.use_momentum:
            f.write(f"  Initial momentum: {args.initial_momentum}\n")
        f.write(f"GPU enabled: {args.use_gpu}\n\n")

        if getattr(args, "use_ray", False) and args.processing_method in ["batch", "colors"]:
            f.write("Ray Multi-GPU Configuration:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Ray enabled: True\n")
            f.write(f"  Number of GPUs: {args.ray_gpu_count or 'auto-detect'}\n")
            f.write(f"  Collective group: {args.ray_collective_group}\n\n")
            f.write(
                f"  Force disk mode: {bool(getattr(args, 'force_disk_mode', False))}\n\n"
            )

        f.write("Training Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training time: {train_time:.2f}s\n")
        f.write(f"Iterations completed: {training_stats['iterations_completed']}\n")
        f.write(f"Total samples processed: {training_stats['total_samples_processed']:,}\n")
        f.write(f"Final weights norm: {training_stats['final_weights_norm']:.6f}\n")
        if "quantization_error" in training_stats:
            f.write(f"Quantization error: {training_stats['quantization_error']:.6f}\n")
        f.write("\n")

        samples_per_sec = training_stats['total_samples_processed'] / train_time
        f.write("Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Samples per second: {samples_per_sec:,.0f}\n")
        f.write(
            f"Iterations per second: {training_stats['iterations_completed'] / train_time:.2f}\n"
        )
        avg_time_per_iter = train_time / training_stats['iterations_completed']
        f.write(f"Average time per iteration: {avg_time_per_iter:.3f}s\n")

        dataset_size_gb = metadata['shape'][0] * metadata['shape'][1] * 4 / (1024 ** 3)
        f.write(f"\nDataset size (uncompressed): {dataset_size_gb:.2f} GB\n")
        if args.sampling_method != "full" and sampling_fraction is not None and computed_samples is not None:
            batch_memory_mb = (
                computed_samples * metadata['shape'][1] * 4 / (1024 ** 2)
            )
            f.write(f"Estimated memory per iteration (fractional): {batch_memory_mb:.1f} MB\n")



def write_mst_details(
    som: FloatSOM,
    args,
    results_path: str,
) -> None:
    """Append MST topology details to the standard results report."""

    topology = getattr(som, "topology", None)
    if topology is None or getattr(topology, "name", "").lower() != "mst":
        return

    mst_edges = list(getattr(topology, "mst_edges", []) or [])
    adjacency_list = getattr(topology, "adjacency_list", {}) or {}

    def _as_int(value):
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return None

    def _node_sort_key(node):
        coerced = _as_int(node)
        if coerced is not None:
            return (0, coerced)
        return (1, str(node))

    edge_sample_limit = 25
    serialised_edges = []
    for edge in mst_edges:
        if hasattr(edge, "tolist"):
            edge = edge.tolist()
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src = _as_int(edge[0])
            dst = _as_int(edge[1])
            if src is not None and dst is not None:
                serialised_edges.append((src, dst))
        if len(serialised_edges) >= edge_sample_limit:
            break

    adjacency_items = []
    sorted_nodes = sorted(adjacency_list.keys(), key=_node_sort_key)
    adjacency_sample_limit = 10
    neighbour_sample_limit = 8
    for node_key in sorted_nodes:
        neighbours = adjacency_list[node_key]
        if hasattr(neighbours, "tolist"):
            neighbours = neighbours.tolist()
        neighbours = list(neighbours)
        original_count = len(neighbours)
        cleaned = [_as_int(neighbour) for neighbour in neighbours[:neighbour_sample_limit]]
        adjacency_items.append((_as_int(node_key), cleaned, original_count))
        if len(adjacency_items) >= adjacency_sample_limit:
            break

    lines = [
        "",
        "MST Topology Summary:",
        "-" * 40,
    ]

    mode = "dynamic" if getattr(args, "dynamic_mst_frequency", False) else "fixed"
    lines.append(f"Update frequency mode: {mode}")
    if mode == "dynamic":
        lines.append(f"Initial update frequency: {getattr(args, 'initial_mst_frequency', 'n/a')}")
        lines.append(f"Final update frequency: {getattr(args, 'final_mst_frequency', 'n/a')}")
        lines.append(f"Decay function: {getattr(args, 'mst_decay_function', 'n/a')}")
    else:
        lines.append(f"Update every N iterations: {getattr(args, 'mst_update_frequency', 'n/a')}")

    total_nodes = getattr(topology, "total_nodes", None)
    lines.append(f"Total nodes: {total_nodes if total_nodes is not None else len(adjacency_list)}")
    lines.append(f"Total edges captured: {len(mst_edges)}")

    if serialised_edges:
        lines.append(f"Sample edges (first {len(serialised_edges)}):")
        for src, dst in serialised_edges:
            lines.append(f"  {src} -> {dst}")
        if len(mst_edges) > edge_sample_limit:
            lines.append("  ...")

    if adjacency_items:
        lines.append(f"Adjacency sample (first {len(adjacency_items)} nodes):")
        for node, neighbours, original_count in adjacency_items:
            if node is None:
                continue
            neighbour_str = ", ".join(str(n) for n in neighbours if n is not None)
            if original_count > neighbour_sample_limit:
                neighbour_str += ", ..."
            lines.append(f"  {node}: {neighbour_str}")

    try:
        with open(results_path, "a") as report_file:
            report_file.write("\n".join(lines))
            report_file.write("\n")
    except Exception as exc:  # pragma: no cover - best effort append
        print(
            f"Warning: Could not append MST details to {results_path}: {exc}"
        )
