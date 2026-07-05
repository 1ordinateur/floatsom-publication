#!/usr/bin/env python3
"""
Run a targeted aweSOM baseline at the FloatSOM sample-scaling reference point.

Default protocol:
- N = 100,000,000 samples
- F = 50 features
- 32 x 32 SOM nodes, matching the normal 1024-node benchmark size
- requested comparator topology = hexagonal
- actual aweSOM topology = aweSOM's native regular x-y lattice

The script intentionally records timeout/import/failure states to CSV/JSON so a
non-finishing run is still usable evidence in a reviewer response.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import sys
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_SAMPLE_SIZE = 100_000_000
DEFAULT_INPUT_DIM = 50
DEFAULT_GRID_SIZE = 32
DEFAULT_TIMEOUT_MINUTES = 30
RAW_RESULT_FILENAME = "awesom_100m_hex_baseline_result.csv"
METADATA_FILENAME = "awesom_100m_hex_baseline_metadata.json"
AUDITED_MODULES = (
    "aweSOM",
    "numpy",
    "numba",
    "scipy",
    "sklearn",
    "matplotlib",
    "jax",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark aweSOM at the 100M x 50-feature sample-scaling point "
            "used by the FloatSOM systems benchmarks."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory for result CSV/JSON files.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Deprecated; aweSOM input data are generated in memory for this benchmark.",
    )
    parser.add_argument(
        "--awesom-source-root",
        default=None,
        help=(
            "Optional path to an aweSOM repository checkout. If supplied, "
            "<root>/src is added to sys.path before importing aweSOM."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--xdim", type=int, default=None, help="Override aweSOM x lattice dimension.")
    parser.add_argument("--ydim", type=int, default=None, help="Override aweSOM y lattice dimension.")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help=(
            "aweSOM online update steps. Defaults to sample-size, matching "
            "the aweSOM SOM scaling protocol. Use a small value for smoke tests."
        ),
    )
    parser.add_argument("--alpha-0", type=float, default=0.5)
    parser.add_argument(
        "--alpha-type",
        choices=("static", "decay"),
        default="static",
        help="aweSOM learning-rate mode; static matches aweSOM's performance benchmark script.",
    )
    parser.add_argument(
        "--sampling-type",
        choices=("uniform", "sampling"),
        default="uniform",
        help="aweSOM lattice initialization; uniform matches aweSOM's performance benchmark script.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-chunk-rows",
        type=int,
        default=250_000,
        help="Rows per chunk when generating the in-memory float32 NumPy array.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Deprecated no-op; data are generated in memory for each run.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=DEFAULT_TIMEOUT_MINUTES,
        help="Training watchdog timeout. Set <=0 to disable.",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Optional NUMBA_NUM_THREADS value to set before importing aweSOM.",
    )
    parser.add_argument(
        "--qe-sample-size",
        type=int,
        default=0,
        help="Optional post-training QE sample size. Default 0 disables QE computation.",
    )
    parser.add_argument(
        "--qe-chunk-rows",
        type=int,
        default=512,
        help="Rows per chunk for optional sampled QE computation.",
    )
    parser.add_argument(
        "--requested-topology",
        default="hexagonal",
        help="Comparator topology requested by the FloatSOM protocol; recorded for provenance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write metadata and validate data path/import configuration without training.",
    )
    return parser.parse_args()


def _result_row_base(
    args: argparse.Namespace,
    generation_seconds: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    xdim = int(args.xdim if args.xdim is not None else args.grid_size)
    ydim = int(args.ydim if args.ydim is not None else args.grid_size)
    train_steps = int(args.train_steps if args.train_steps is not None else args.sample_size)
    timeout_seconds = None if args.timeout_minutes <= 0 else float(args.timeout_minutes) * 60.0
    metadata = metadata or {}
    chunks = metadata.get("chunks", "")
    if isinstance(chunks, (tuple, list)):
        chunks = "x".join(str(value) for value in chunks)
    return {
        "benchmark": "awesom_100m_hex_baseline",
        "implementation": "aweSOM",
        "requested_topology": str(args.requested_topology),
        "actual_topology": "aweSOM native regular x-y lattice",
        "sample_size": int(args.sample_size),
        "input_dim": int(args.input_dim),
        "xdim": xdim,
        "ydim": ydim,
        "n_nodes": xdim * ydim,
        "train_steps": train_steps,
        "alpha_0": float(args.alpha_0),
        "alpha_type": str(args.alpha_type),
        "sampling_type": str(args.sampling_type),
        "seed": int(args.seed),
        "data_path": "in_memory",
        "data_storage_format": "numpy.ndarray",
        "data_dtype": str(metadata.get("dtype", "float32")),
        "data_chunks": chunks,
        "data_distribution": "uniform[0,1)",
        "data_generation_seconds": float(generation_seconds),
        "timeout_seconds": timeout_seconds if timeout_seconds is not None else "",
        "numba_threads": int(args.numba_threads) if args.numba_threads is not None else "",
        "qe_sample_size": int(args.qe_sample_size),
    }


def _write_single_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _add_awesom_source_root(source_root: Optional[str]) -> None:
    if not source_root:
        return
    root = Path(source_root).expanduser().resolve()
    src = root / "src"
    sys.path.insert(0, str(src if src.exists() else root))


def _validate_awesom_import(source_root: Optional[str]) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    _add_awesom_source_root(source_root)
    module = importlib.import_module("aweSOM")
    return {
        "awesom_version": getattr(module, "__version__", ""),
        "awesom_file": getattr(module, "__file__", ""),
    }


def _module_origin(module_name: str) -> str:
    if module_name == "numpy":
        return getattr(np, "__file__", "")
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return "missing"
    return str(spec.origin or "namespace")


def _collect_import_provenance() -> Dict[str, str]:
    return {
        f"module_origin_{module_name.lower().replace('.', '_')}": _module_origin(module_name)
        for module_name in AUDITED_MODULES
    }


def _resolve_generation_chunk_rows(args: argparse.Namespace) -> int:
    return int(args.data_chunk_rows)


def _generate_in_memory_data(
    *,
    sample_size: int,
    input_dim: int,
    seed: int,
    chunk_rows: int,
) -> tuple[np.ndarray, Dict[str, Any], float]:
    if chunk_rows <= 0:
        raise ValueError(f"data generation chunk_rows must be positive, got {chunk_rows}")

    # Match the existing scaling benchmark's random-data convention while
    # avoiding a full float64 temporary for the 100M x 50 case.
    start_time = time.perf_counter()
    np.random.seed(seed)
    data = np.empty((sample_size, input_dim), dtype=np.float32)
    for start_idx in range(0, sample_size, chunk_rows):
        end_idx = min(start_idx + chunk_rows, sample_size)
        data[start_idx:end_idx, :] = np.random.rand(end_idx - start_idx, input_dim).astype(
            np.float32
        )
        if start_idx == 0 or end_idx == sample_size or end_idx % (chunk_rows * 10) == 0:
            progress = 100.0 * end_idx / sample_size
            print(f"Generated {end_idx:,}/{sample_size:,} rows ({progress:.1f}%).", flush=True)

    metadata = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "chunks": (min(chunk_rows, sample_size), input_dim),
        "generation_backend": "numpy.random.rand",
    }
    return data, metadata, time.perf_counter() - start_time


def _sampled_quantization_error(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    sample_size: int,
    chunk_rows: int,
    seed: int,
) -> float:
    if sample_size <= 0:
        return float("nan")
    n_samples = int(data.shape[0])
    actual_sample_size = min(int(sample_size), n_samples)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_samples, size=actual_sample_size, replace=False)
    sample_indices.sort()

    weights_np = np.asarray(weights, dtype=np.float32)
    if weights_np.ndim > 2:
        weights_np = weights_np.reshape(-1, weights_np.shape[-1])

    total = 0.0
    count = 0
    rows_per_chunk = max(1, int(chunk_rows))
    for start_idx in range(0, actual_sample_size, rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk, actual_sample_size)
        batch = np.asarray(data[sample_indices[start_idx:end_idx], :], dtype=np.float32)
        diff = batch[:, None, :] - weights_np[None, :, :]
        min_dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        total += float(np.sum(min_dist))
        count += int(min_dist.shape[0])
    return total / max(count, 1)


def _child_train_awesom(args_dict: Dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        if args_dict.get("numba_threads") is not None:
            os.environ["NUMBA_NUM_THREADS"] = str(args_dict["numba_threads"])

        _add_awesom_source_root(args_dict.get("awesom_source_root"))
        from aweSOM import Lattice  # type: ignore
        import aweSOM  # type: ignore

        data, data_metadata, generation_seconds = _generate_in_memory_data(
            sample_size=int(args_dict["sample_size"]),
            input_dim=int(args_dict["input_dim"]),
            seed=int(args_dict["seed"]),
            chunk_rows=int(args_dict["data_chunk_rows"]),
        )
        feature_names = [f"feature{i}" for i in range(1, int(args_dict["input_dim"]) + 1)]

        lattice = Lattice(
            int(args_dict["xdim"]),
            int(args_dict["ydim"]),
            float(args_dict["alpha_0"]),
            int(args_dict["train_steps"]),
            alpha_type=str(args_dict["alpha_type"]),
            sampling_type=str(args_dict["sampling_type"]),
        )

        train_start = time.perf_counter()
        lattice.train_lattice(data, feature_names, labels=None)
        train_time_s = time.perf_counter() - train_start

        qe_value = float("nan")
        qe_time_s = float("nan")
        if int(args_dict["qe_sample_size"]) > 0:
            qe_start = time.perf_counter()
            qe_value = _sampled_quantization_error(
                data,
                lattice.lattice,
                sample_size=int(args_dict["qe_sample_size"]),
                chunk_rows=int(args_dict["qe_chunk_rows"]),
                seed=int(args_dict["seed"]) + 1,
            )
            qe_time_s = time.perf_counter() - qe_start

        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        result_queue.put(
            {
                "status": "ok",
                "data_generation_seconds": float(generation_seconds),
                "data_shape": "x".join(str(value) for value in data_metadata["shape"]),
                "data_generation_backend": str(data_metadata["generation_backend"]),
                "train_time_s": float(train_time_s),
                "qe_sampled": qe_value,
                "qe_time_s": qe_time_s,
                "peak_rss_mb": float(peak_rss_mb),
                "lattice_shape": list(np.asarray(lattice.lattice).shape),
                "awesom_version": getattr(aweSOM, "__version__", ""),
                "awesom_file": getattr(aweSOM, "__file__", ""),
                "error_type": "",
                "error_message": "",
                "traceback": "",
            }
        )
    except BaseException as exc:  # noqa: BLE001 - benchmark should record failures
        try:
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            peak_rss_mb = float("nan")
        result_queue.put(
            {
                "status": "error",
                "data_generation_seconds": float("nan"),
                "data_shape": "",
                "data_generation_backend": "",
                "train_time_s": float("nan"),
                "qe_sampled": float("nan"),
                "qe_time_s": float("nan"),
                "peak_rss_mb": float(peak_rss_mb),
                "lattice_shape": "",
                "awesom_version": "",
                "awesom_file": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_training_with_watchdog(args: argparse.Namespace) -> Dict[str, Any]:
    xdim = int(args.xdim if args.xdim is not None else args.grid_size)
    ydim = int(args.ydim if args.ydim is not None else args.grid_size)
    train_steps = int(args.train_steps if args.train_steps is not None else args.sample_size)
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    child_args = {
        "awesom_source_root": args.awesom_source_root,
        "sample_size": int(args.sample_size),
        "input_dim": int(args.input_dim),
        "xdim": xdim,
        "ydim": ydim,
        "train_steps": train_steps,
        "alpha_0": float(args.alpha_0),
        "alpha_type": str(args.alpha_type),
        "sampling_type": str(args.sampling_type),
        "seed": int(args.seed),
        "data_chunk_rows": _resolve_generation_chunk_rows(args),
        "numba_threads": int(args.numba_threads) if args.numba_threads is not None else None,
        "qe_sample_size": int(args.qe_sample_size),
        "qe_chunk_rows": int(args.qe_chunk_rows),
    }

    process = ctx.Process(target=_child_train_awesom, args=(child_args, result_queue))
    wall_start = time.perf_counter()
    process.start()
    timeout_seconds = None if args.timeout_minutes <= 0 else float(args.timeout_minutes) * 60.0
    process.join(timeout_seconds)
    wall_time_s = time.perf_counter() - wall_start

    if process.is_alive():
        process.terminate()
        process.join(30)
        if process.is_alive():
            process.kill()
            process.join(30)
        return {
            "status": "timeout",
            "train_time_s": float("nan"),
            "total_wall_time_s": float(wall_time_s),
            "qe_sampled": float("nan"),
            "qe_time_s": float("nan"),
            "peak_rss_mb": float("nan"),
            "lattice_shape": "",
            "awesom_version": "",
            "awesom_file": "",
            "error_type": "TimeoutExpired",
            "error_message": f"aweSOM training exceeded timeout_seconds={timeout_seconds}",
            "traceback": "",
            "child_exitcode": process.exitcode,
        }

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "status": "error",
            "train_time_s": float("nan"),
            "qe_sampled": float("nan"),
            "qe_time_s": float("nan"),
            "peak_rss_mb": float("nan"),
            "lattice_shape": "",
            "awesom_version": "",
            "awesom_file": "",
            "error_type": "NoResult",
            "error_message": "Child process exited without returning benchmark data.",
            "traceback": "",
        }
    result["total_wall_time_s"] = float(wall_time_s)
    result["child_exitcode"] = process.exitcode
    return result


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_chunk_rows = _resolve_generation_chunk_rows(args)
    placeholder_metadata = {
        "dtype": str(np.dtype(np.float32)),
        "chunks": (min(data_chunk_rows, int(args.sample_size)), int(args.input_dim)),
    }

    run_start = time.perf_counter()
    import_probe: Dict[str, Any]
    try:
        import_probe = _validate_awesom_import(args.awesom_source_root)
    except BaseException as exc:  # noqa: BLE001 - result should record import failures
        row = _result_row_base(args, generation_seconds=0.0, metadata=placeholder_metadata)
        row.update(_collect_import_provenance())
        row.update(
            {
                "status": "import_error",
                "train_time_s": float("nan"),
                "total_wall_time_s": time.perf_counter() - run_start,
                "qe_sampled": float("nan"),
                "qe_time_s": float("nan"),
                "peak_rss_mb": float("nan"),
                "lattice_shape": "",
                "awesom_version": "",
                "awesom_file": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "child_exitcode": "",
            }
        )
        _write_single_row_csv(output_dir / RAW_RESULT_FILENAME, row)
        _write_json(output_dir / METADATA_FILENAME, row)
        print(f"aweSOM import failed; wrote {output_dir / RAW_RESULT_FILENAME}", flush=True)
        return 1

    row = _result_row_base(
        args,
        generation_seconds=float("nan"),
        metadata=placeholder_metadata,
    )
    row.update(_collect_import_provenance())
    row.update(import_probe)

    if args.dry_run:
        row.update(
            {
                "status": "dry_run",
                "train_time_s": float("nan"),
                "total_wall_time_s": time.perf_counter() - run_start,
                "qe_sampled": float("nan"),
                "qe_time_s": float("nan"),
                "peak_rss_mb": float("nan"),
                "lattice_shape": "",
                "error_type": "",
                "error_message": "",
                "traceback": "",
                "child_exitcode": "",
            }
        )
    else:
        result = _run_training_with_watchdog(args)
        row.update(result)

    row["script"] = str(Path(__file__).resolve())
    row["published_awesom_som_context"] = (
        "aweSOM JOSS Figure 1 SOM scaling reports F=6 and F=10 on one CPU node; "
        "the JAX/GPU panel is for SCE rather than SOM training."
    )
    row["jax_note"] = (
        "This benchmark exercises aweSOM's SOM Lattice path, which uses NumPy/Numba "
        "rather than JAX; JAX is audited for environment provenance but not invoked."
    )
    row["topology_note"] = (
        "aweSOM Lattice does not expose a hexagonal topology flag; this run records "
        "the requested FloatSOM comparator topology and uses aweSOM's native regular lattice."
    )

    _write_single_row_csv(output_dir / RAW_RESULT_FILENAME, row)
    _write_json(output_dir / METADATA_FILENAME, row)
    print(f"Wrote result CSV: {output_dir / RAW_RESULT_FILENAME}", flush=True)
    print(f"Wrote metadata JSON: {output_dir / METADATA_FILENAME}", flush=True)
    return 0 if row.get("status") in {"ok", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
