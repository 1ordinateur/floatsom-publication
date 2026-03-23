#!/usr/bin/env python3
"""
Pre-generate GPU-scaling benchmark cache datasets for non-minibatch runs.

This script creates the same Zarr cache entries that
`run_gpu_scaling_benchmark.py` would request for non-minibatch processing, so
benchmark runtime can load from cache instead of generating data on-demand.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Set, Tuple

DEFAULT_DIMENSIONS = [50, 100, 200, 500, 1000, 2000]
DEFAULT_SAMPLE_SIZES = [
    1_000_000_000,
]
DEFAULT_NON_MINIBATCH_METHODS = ["batch"]
LOCKED_MODE = "sample_scaling"


@dataclass(frozen=True, order=True)
class DatasetSpec:
    samples: int
    input_dim: int
    seed: int
    zarr_chunk_size: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-generate random Zarr cache datasets used by non-minibatch runs "
            "in the GPU-scaling benchmark."
        )
    )
    parser.add_argument(
        "--from_pbs_script",
        type=str,
        default=None,
        help=(
            "Path to a PBS script containing a run_gpu_scaling_benchmark command. "
            "When provided, dataset-related parameters are sourced from that command."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["dimension_scaling", "sample_scaling", "grid_size_scaling", "both", "all"],
        help="Match dataset coverage for the selected benchmark mode(s).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_DIMENSIONS,
        help="Dimensions for dimension-scaling mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000_000,
        help="Fixed sample count used by dimension/grid modes.",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes for sample-scaling mode.",
    )
    parser.add_argument(
        "--fixed_dimension",
        type=int,
        default=50,
        help="Fixed dimension for sample/grid modes.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat count; seeds are expanded as seed + repeat_index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (same convention as benchmark runner).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./gpu_scaling_cache",
        help="Cache directory where Zarr datasets are stored.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Global processing chunk-size override for batch/colors. "
            "When omitted, method-specific auto chunk sizing is used."
        ),
    )
    parser.add_argument(
        "--zarr_chunk_size",
        type=int,
        default=None,
        help=(
            "Explicit Zarr chunk size. If omitted, defaults to the resolved "
            "processing chunk size for each method/dimension combination."
        ),
    )
    parser.add_argument(
        "--processing_methods",
        type=str,
        nargs="+",
        default=DEFAULT_NON_MINIBATCH_METHODS,
        choices=["batch", "colors"],
        help=(
            "Non-minibatch processing methods to pre-generate cache entries for."
        ),
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Regenerate datasets even when cache entries already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show planned datasets without creating/loading cache entries.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose dataset generation output.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help=(
            "Dataset-level parallel workers. "
            "Defaults to an auto value (up to 4)."
        ),
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on first failed dataset generation.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")
    if args.samples <= 0:
        raise ValueError("--samples must be a positive integer")
    if args.fixed_dimension <= 0:
        raise ValueError("--fixed_dimension must be a positive integer")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer when provided")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max_workers must be a positive integer when provided")
    if not args.processing_methods:
        raise ValueError("--processing_methods must include at least one method")

    for value in args.dimensions:
        if value <= 0:
            raise ValueError("--dimensions values must be positive integers")
    for value in args.sample_sizes:
        if value <= 0:
            raise ValueError("--sample_sizes values must be positive integers")


def _fallback_auto_chunk_size(input_dim: int) -> int:
    base_samples = 500_000
    base_dimension = 50
    min_samples = 10_000
    dim_scaled = int(round(base_samples * base_dimension / int(input_dim)))
    return max(min_samples, min(base_samples, dim_scaled))


def _resolve_method_zarr_chunk_size(
    args: argparse.Namespace,
    processing_method: str,
    input_dim: int,
) -> Tuple[int, str]:
    if args.chunk_size is not None:
        processing_chunk_size = int(args.chunk_size)
        source = "manual_cli_chunk_size"
    else:
        try:
            from floatsom.processing.processing_params import calculate_auto_chunk_size_for_method

            processing_chunk_size = int(
                calculate_auto_chunk_size_for_method(
                    int(input_dim), str(processing_method)
                )
            )
            source = "auto_dimension_vram"
        except Exception:
            processing_chunk_size = _fallback_auto_chunk_size(int(input_dim))
            source = "auto_dimension_fallback"

    if args.zarr_chunk_size is None:
        return int(processing_chunk_size), f"default_from_{source}"

    requested = int(args.zarr_chunk_size)
    if requested <= 0:
        return int(processing_chunk_size), "invalid_zarr_chunk_size_fallback"
    return requested, "manual_zarr_chunk_size"


def _extract_benchmark_cli_tokens_from_pbs_script(pbs_script_path: str) -> List[str]:
    pbs_path = Path(pbs_script_path).expanduser().resolve()
    if not pbs_path.exists():
        raise FileNotFoundError(f"PBS script not found: {pbs_path}")
    if not pbs_path.is_file():
        raise ValueError(f"PBS script path is not a file: {pbs_path}")

    lines = pbs_path.read_text(encoding="utf-8").splitlines()
    commands: List[str] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if "run_gpu_scaling_benchmark" not in stripped:
            i += 1
            continue

        command = stripped
        while command.rstrip().endswith("\\") and i + 1 < len(lines):
            command = command.rstrip()
            command = command[:-1] + " " + lines[i + 1].strip()
            i += 1
        commands.append(command)
        i += 1

    if not commands:
        raise ValueError(
            f"No run_gpu_scaling_benchmark command found in PBS script: {pbs_path}"
        )

    # Use the last matching invocation in the file.
    command = commands[-1]
    tokens = shlex.split(command)

    module_name = "floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark"
    if "-m" in tokens:
        try:
            module_idx = tokens.index(module_name)
            return tokens[module_idx + 1 :]
        except ValueError:
            pass

    # Fallback: if command uses script path invocation, return tokens after script token.
    for idx, token in enumerate(tokens):
        if token.endswith("run_gpu_scaling_benchmark.py"):
            return tokens[idx + 1 :]

    raise ValueError(
        "Could not parse benchmark CLI arguments from PBS script command: "
        f"{pbs_path}"
    )


def _load_benchmark_params_from_pbs_script(pbs_script_path: str) -> argparse.Namespace:
    cli_tokens = _extract_benchmark_cli_tokens_from_pbs_script(pbs_script_path)
    from floatsom.benchmarks.speed_benchmarks.gpu_scaling.cli import parse_args as parse_benchmark_args

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["run_gpu_scaling_benchmark.py", *cli_tokens]
        benchmark_args = parse_benchmark_args()
    finally:
        sys.argv = previous_argv
    return benchmark_args


def _apply_pbs_dataset_params(
    pregen_args: argparse.Namespace,
    benchmark_args: argparse.Namespace,
) -> argparse.Namespace:
    pregen_args.dimensions = list(benchmark_args.dimensions)
    pregen_args.samples = int(benchmark_args.samples)
    pregen_args.fixed_dimension = int(benchmark_args.fixed_dimension)
    pregen_args.repeats = int(benchmark_args.repeats)
    pregen_args.seed = int(benchmark_args.seed)
    pregen_args.chunk_size = benchmark_args.chunk_size
    pregen_args.zarr_chunk_size = benchmark_args.zarr_chunk_size
    pregen_args.cache_dir = str(benchmark_args.cache_dir)
    return pregen_args


def _lock_to_1b_batch_target(args: argparse.Namespace) -> argparse.Namespace:
    """
    Restrict pre-generation to the single non-minibatch target we still need:
    sample-scaling batch caches for the 1B dataset.
    """
    args.mode = LOCKED_MODE
    args.sample_sizes = list(DEFAULT_SAMPLE_SIZES)
    args.processing_methods = list(DEFAULT_NON_MINIBATCH_METHODS)
    return args


def _expand_modes(mode: str) -> List[str]:
    if mode == "both":
        return ["dimension_scaling", "sample_scaling"]
    if mode == "all":
        return ["dimension_scaling", "sample_scaling", "grid_size_scaling"]
    return [mode]


def _repeat_seeds(base_seed: int, repeats: int) -> List[int]:
    return [int(base_seed) + i for i in range(int(repeats))]


def _collect_dataset_specs(
    args: argparse.Namespace,
) -> Dict[DatasetSpec, Set[str]]:
    modes = _expand_modes(args.mode)
    repeat_seeds = _repeat_seeds(args.seed, args.repeats)
    specs_to_modes: Dict[DatasetSpec, Set[str]] = {}

    def _add_spec(
        samples: int,
        input_dim: int,
        seed: int,
        mode_name: str,
        processing_method: str,
    ) -> None:
        zarr_chunk_size, _ = _resolve_method_zarr_chunk_size(
            args=args,
            processing_method=processing_method,
            input_dim=int(input_dim),
        )
        spec = DatasetSpec(
            samples=int(samples),
            input_dim=int(input_dim),
            seed=int(seed),
            zarr_chunk_size=int(zarr_chunk_size),
        )
        specs_to_modes.setdefault(spec, set()).add(f"{mode_name}:{processing_method}")

    for processing_method in args.processing_methods:
        if "dimension_scaling" in modes:
            for dim in args.dimensions:
                for seed in repeat_seeds:
                    _add_spec(args.samples, dim, seed, "dimension_scaling", processing_method)

        if "sample_scaling" in modes:
            for sample_size in args.sample_sizes:
                for seed in repeat_seeds:
                    _add_spec(
                        sample_size,
                        args.fixed_dimension,
                        seed,
                        "sample_scaling",
                        processing_method,
                    )

        if "grid_size_scaling" in modes:
            for seed in repeat_seeds:
                _add_spec(args.samples, args.fixed_dimension, seed, "grid_size_scaling", processing_method)

    return specs_to_modes


def _cache_path(cache_dir: Path, spec: DatasetSpec) -> Path:
    config_hash = hashlib.md5(
        f"random_{spec.samples}_{spec.input_dim}_{spec.seed}_{spec.zarr_chunk_size}".encode()
    ).hexdigest()[:8]
    return cache_dir / f"random_{spec.samples}_{spec.input_dim}_{config_hash}.zarr"


def _estimate_raw_size_gib(spec: DatasetSpec) -> float:
    return (spec.samples * spec.input_dim * 4) / (1024 ** 3)


def _format_modes(modes: Iterable[str]) -> str:
    return ",".join(sorted(set(modes)))


def _resolve_max_workers(args: argparse.Namespace, dataset_count: int) -> Tuple[int, str]:
    dataset_count = max(1, int(dataset_count))
    if args.max_workers is not None:
        requested = int(args.max_workers)
        return max(1, min(requested, dataset_count)), "manual_cli_max_workers"

    cpu_count = os.cpu_count() or 1
    auto_workers = max(1, min(4, cpu_count, dataset_count))
    return auto_workers, "auto_conservative"


def _generate_or_load_dataset_for_spec(
    spec: DatasetSpec,
    cache_dir: str,
    force_regenerate: bool,
    verbose: bool,
) -> str:
    from floatsom.benchmarks.speed_benchmarks.gpu_scaling.large_dataset import generate_or_load_data

    dataset_args = SimpleNamespace(
        samples=spec.samples,
        input_dim=spec.input_dim,
        seed=spec.seed,
        zarr_chunk_size=spec.zarr_chunk_size,
        cache_dir=str(cache_dir),
        force_regenerate=bool(force_regenerate),
        verbose=bool(verbose),
    )
    zarr_path, metadata = generate_or_load_data(dataset_args)
    _ = metadata
    return zarr_path


def main() -> int:
    args = _parse_args()
    if args.from_pbs_script:
        benchmark_args = _load_benchmark_params_from_pbs_script(args.from_pbs_script)
        args = _apply_pbs_dataset_params(args, benchmark_args)
    args = _lock_to_1b_batch_target(args)
    _validate_args(args)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if not args.dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)

    specs_to_modes = _collect_dataset_specs(args)
    specs = sorted(specs_to_modes.keys())
    total_raw_gib = sum(_estimate_raw_size_gib(spec) for spec in specs)
    chunk_sizes = sorted({spec.zarr_chunk_size for spec in specs})

    print("=" * 72)
    print("Pre-generating GPU-scaling non-minibatch cache datasets")
    print("=" * 72)
    print(
        "Locked target: sample_scaling only, sample_sizes=[1,000,000,000], "
        "processing_methods=[batch]"
    )
    print(f"Mode selection: {_format_modes(_expand_modes(args.mode))}")
    print(f"Processing methods: {', '.join(args.processing_methods)}")
    print(f"Cache directory: {cache_dir}")
    if len(chunk_sizes) == 1:
        print(f"Resolved zarr_chunk_size: {chunk_sizes[0]:,}")
    else:
        print(
            "Resolved zarr_chunk_size values: "
            + ", ".join(f"{value:,}" for value in chunk_sizes)
        )
    print(f"Unique datasets to ensure: {len(specs)}")
    print(f"Estimated aggregate raw size (uncompressed): {total_raw_gib:,.2f} GiB")
    print(
        "Note: topology/grid/GPU count do not affect dataset cache keys; "
        "only samples, dimension, seed, and zarr_chunk_size do."
    )
    print("=" * 72)

    if args.dry_run:
        for index, spec in enumerate(specs, start=1):
            path = _cache_path(cache_dir, spec)
            exists = path.exists()
            modes = _format_modes(specs_to_modes[spec])
            status = "exists" if exists else "missing"
            print(
                f"[{index}/{len(specs)}] {status:7s} "
                f"samples={spec.samples:,} dim={spec.input_dim} seed={spec.seed} "
                f"zarr_chunk_size={spec.zarr_chunk_size:,} modes={modes} path={path}"
            )
        return 0

    generated = 0
    loaded_existing = 0
    failures = 0
    max_workers, worker_source = _resolve_max_workers(args, len(specs))
    print(f"Dataset worker pool size: {max_workers} ({worker_source})")

    def _run_serial() -> bool:
        nonlocal generated, loaded_existing, failures
        stopped_early = False
        for index, spec in enumerate(specs, start=1):
            path = _cache_path(cache_dir, spec)
            existed_before = path.exists()
            modes = _format_modes(specs_to_modes[spec])
            print(
                f"\n[{index}/{len(specs)}] "
                f"samples={spec.samples:,} dim={spec.input_dim} seed={spec.seed} "
                f"zarr_chunk_size={spec.zarr_chunk_size:,} modes={modes}"
            )
            print(f"Target cache path: {path}")
            try:
                zarr_path = _generate_or_load_dataset_for_spec(
                    spec=spec,
                    cache_dir=str(cache_dir),
                    force_regenerate=bool(args.force_regenerate),
                    verbose=bool(args.verbose),
                )
            except Exception as exc:
                failures += 1
                print(f"FAILED: {exc}")
                if args.fail_fast:
                    stopped_early = True
                    break
                continue

            if existed_before and not args.force_regenerate:
                loaded_existing += 1
                print(f"Ready from existing cache: {zarr_path}")
            else:
                generated += 1
                print(f"Generated cache entry: {zarr_path}")
        return stopped_early

    if max_workers == 1:
        failed_fast = _run_serial()
        if failed_fast:
            print("Stopped early due to --fail_fast after the first failure.")
    else:
        print("Running datasets in parallel.")
        future_map = {}
        failed_fast = False
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for index, spec in enumerate(specs, start=1):
                    path = _cache_path(cache_dir, spec)
                    existed_before = path.exists()
                    modes = _format_modes(specs_to_modes[spec])
                    print(
                        f"\n[{index}/{len(specs)}] queued "
                        f"samples={spec.samples:,} dim={spec.input_dim} seed={spec.seed} "
                        f"zarr_chunk_size={spec.zarr_chunk_size:,} modes={modes}"
                    )
                    print(f"Target cache path: {path}")
                    future = executor.submit(
                        _generate_or_load_dataset_for_spec,
                        spec=spec,
                        cache_dir=str(cache_dir),
                        force_regenerate=bool(args.force_regenerate),
                        verbose=bool(args.verbose),
                    )
                    future_map[future] = (index, spec, path, existed_before)

                for future in as_completed(future_map):
                    index, spec, path, existed_before = future_map[future]
                    try:
                        zarr_path = future.result()
                    except Exception as exc:
                        failures += 1
                        print(
                            f"\n[{index}/{len(specs)}] FAILED "
                            f"samples={spec.samples:,} dim={spec.input_dim} seed={spec.seed}"
                        )
                        print(f"Target cache path: {path}")
                        print(f"FAILED: {exc}")
                        if args.fail_fast:
                            failed_fast = True
                            for pending in future_map:
                                pending.cancel()
                            break
                        continue

                    print(
                        f"\n[{index}/{len(specs)}] completed "
                        f"samples={spec.samples:,} dim={spec.input_dim} seed={spec.seed}"
                    )
                    print(f"Target cache path: {path}")
                    if existed_before and not args.force_regenerate:
                        loaded_existing += 1
                        print(f"Ready from existing cache: {zarr_path}")
                    else:
                        generated += 1
                        print(f"Generated cache entry: {zarr_path}")
        except (OSError, PermissionError) as exc:
            print(f"Parallel worker pool unavailable ({exc}); falling back to sequential mode.")
            failed_fast = _run_serial()

        if failed_fast:
            print("Stopped early due to --fail_fast after the first failure.")

    print("\n" + "=" * 72)
    print("Non-minibatch cache pre-generation summary")
    print("=" * 72)
    print(f"Total dataset specs: {len(specs)}")
    print(f"Generated now: {generated}")
    print(f"Already cached: {loaded_existing}")
    print(f"Failures: {failures}")
    print("=" * 72)

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
