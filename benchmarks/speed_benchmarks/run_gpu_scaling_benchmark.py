#!/usr/bin/env python3
"""
GPU Scaling Benchmark for FloatSOM

This script orchestrates GPU scaling benchmarks using modular components.
It coordinates different benchmark types through a clean, maintainable interface.

Refactored following SOLID principles:
- Single Responsibility: Each module has one clear purpose
- Open/Closed: Easy to add new benchmark types
- DRY: Common logic extracted to reusable modules
"""

import logging
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Optional

from floatsom.benchmarks.speed_benchmarks.gpu_scaling.cli import parse_args
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.benchmark_types import (
    DimensionScalingBenchmark,
    SampleScalingBenchmark,
    GridSizeScalingBenchmark,
)
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.reporting import ReportGenerator
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.runners import BenchmarkTimeoutError
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.ray_utils import ensure_clean_ray_state
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.resume_state import ResumeManager
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.results import ResultsHandler

SUPPLEMENTARY_MINIBATCH_SUBDIR = "supplementary_minibatch"


def _has_method(results: Dict, method_name: str) -> bool:
    """Return True when a method exists at the top level (case-insensitive)."""
    if not isinstance(results, dict):
        return False
    target = str(method_name).strip().lower()
    return any(str(method).strip().lower() == target for method in results.keys())


def _validate_temp_dir(path: str, *, allow_tmp: bool = False) -> str:
    """Ensure the base temporary directory exists (creating it if necessary)."""
    if path is None:
        raise ValueError("Temporary directory path was not provided")

    abs_path = os.path.abspath(path)
    real_path = os.path.realpath(abs_path)
    if not allow_tmp and real_path.startswith("/tmp"):
        raise ValueError(
            "Temporary directory cannot be under /tmp; specify a scratch directory such as your VAST area"
        )

    if os.path.exists(abs_path) and not os.path.isdir(abs_path):
        raise NotADirectoryError(f"Temporary path exists but is not a directory: {abs_path}")

    if not os.path.isdir(abs_path):
        try:
            os.makedirs(abs_path, exist_ok=True)
        except Exception as exc:
            raise FileNotFoundError(f"Could not create temporary directory {abs_path}: {exc}") from exc

    if not os.access(abs_path, os.W_OK):
        raise PermissionError(f"Temporary directory is not writable: {abs_path}")

    return abs_path


def _create_run_temp_dir(base_temp_dir: Optional[str]) -> str:
    """Create a unique run-specific temporary directory."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    if not base_temp_dir:
        return tempfile.mkdtemp(prefix=f"gpu_scaling_tmp_{run_id}_{unique}_")
    run_dir = os.path.join(base_temp_dir, f"gpu_scaling_tmp_{run_id}_{unique}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _cleanup_run_temp_dir(run_temp_dir: str) -> None:
    """Best-effort removal of the run-specific temporary directory."""
    if not run_temp_dir:
        return

    try:
        shutil.rmtree(run_temp_dir)
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - cleanup best effort
        logging.warning("Could not remove temporary directory %s: %s", run_temp_dir, exc)

def _configure_logging():
    """Force verbose logging so benchmark runs surface all diagnostics."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Suppress extremely chatty third-party debug logs (e.g., Matplotlib font search)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def _set_default_omp_num_threads() -> None:
    """
    Set a safe default for OpenMP/BLAS thread pools without overriding user intent.

    Historically this benchmark forced `OMP_NUM_THREADS=os.cpu_count()`, which can
    oversubscribe CPU cores on HPC schedulers (cpusets/cgroups) and in multi-node
    Ray runs. We now:
      - Respect an explicitly-set OMP_NUM_THREADS
      - Otherwise pick a value consistent with CPU affinity / scheduler allocation
    """

    if os.environ.get("OMP_NUM_THREADS"):
        return

    available = None
    try:
        available = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        available = None

    if not available:
        for key in ("PBS_NCPUS", "NCPUS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            raw = os.environ.get(key)
            if not raw:
                continue
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                available = parsed
                break

    if not available:
        available = os.cpu_count() or 1

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(available)))


def _filter_numeric_hierarchy(raw_results: Dict) -> Dict:
    """Return a copy of results containing only numeric axis keys."""
    if not isinstance(raw_results, dict):
        return {}

    filtered = {}
    for method, method_results in raw_results.items():
        if not isinstance(method_results, dict):
            continue

        cleaned_method = {}
        for axis_key, axis_results in method_results.items():
            try:
                float(axis_key)
            except (TypeError, ValueError):
                continue
            cleaned_method[axis_key] = axis_results

        if cleaned_method:
            filtered[method] = cleaned_method

    return filtered


def _rewrite_all_outputs(
    args,
    dim_results: Optional[Dict],
    sample_results: Optional[Dict],
    grid_results: Optional[Dict],
) -> None:
    """Rewrite persisted artifacts to reflect the final aggregate state."""
    try:
        from floatsom.benchmarks.visualization import (
            plot_gpu_scaling_benchmark,
            plot_sample_scaling_benchmark,
        )
    except Exception:
        plot_gpu_scaling_benchmark = None
        plot_sample_scaling_benchmark = None

    if dim_results:
        dim_output_dir = os.path.join(args.output_dir, "dimension_scaling")
        os.makedirs(dim_output_dir, exist_ok=True)
        for topology, topology_results in dim_results.items():
            for processing_method, method_results in topology_results.items():
                ResultsHandler.save_results(
                    method_results,
                    dim_output_dir,
                    mode="dimension",
                    method_suffix=f"_{topology}_{processing_method}",
                    merge_existing=getattr(args, "merge_existing", False),
                )
            if plot_gpu_scaling_benchmark:
                topology_dir = os.path.join(dim_output_dir, topology)
                os.makedirs(topology_dir, exist_ok=True)
                filtered_results = _filter_numeric_hierarchy(topology_results)
                if filtered_results:
                    plot_gpu_scaling_benchmark(
                        results=filtered_results,
                        output_dir=topology_dir,
                        title=(
                            f"FloatSOM Dimension Scaling - {args.samples:,} samples"
                            f" ({topology} topology)"
                        ),
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                        axis_mode="dimension",
                    )
                    if _has_method(filtered_results, "minibatch"):
                        supplementary_dir = os.path.join(
                            topology_dir,
                            SUPPLEMENTARY_MINIBATCH_SUBDIR,
                        )
                        os.makedirs(supplementary_dir, exist_ok=True)
                        plot_gpu_scaling_benchmark(
                            results=filtered_results,
                            output_dir=supplementary_dir,
                            title=(
                                f"FloatSOM Dimension Scaling - {args.samples:,} samples"
                                f" ({topology} topology, with minibatch)"
                            ),
                            show_speedup=True,
                            show_error_bars=True,
                            show_legend=False,
                            emit_shared_legend=True,
                            axis_mode="dimension",
                            methods_to_plot=["batch", "colors", "minibatch"],
                        )

    if sample_results:
        sample_output_dir = os.path.join(args.output_dir, "sample_scaling")
        os.makedirs(sample_output_dir, exist_ok=True)
        for topology, topology_results in sample_results.items():
            for processing_method, method_results in topology_results.items():
                ResultsHandler.save_results(
                    method_results,
                    sample_output_dir,
                    mode="sample",
                    method_suffix=f"_{topology}_{processing_method}",
                    merge_existing=getattr(args, "merge_existing", False),
                )
            if plot_sample_scaling_benchmark:
                topology_dir = os.path.join(sample_output_dir, topology)
                os.makedirs(topology_dir, exist_ok=True)
                filtered_results = _filter_numeric_hierarchy(topology_results)
                if filtered_results:
                    plot_sample_scaling_benchmark(
                        results=filtered_results,
                        output_dir=topology_dir,
                        title=(
                            f"FloatSOM Sample Size Scaling - {args.fixed_dimension}D"
                            f" ({topology} topology)"
                        ),
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                    )
                    if _has_method(filtered_results, "minibatch"):
                        supplementary_dir = os.path.join(
                            topology_dir,
                            SUPPLEMENTARY_MINIBATCH_SUBDIR,
                        )
                        os.makedirs(supplementary_dir, exist_ok=True)
                        plot_sample_scaling_benchmark(
                            results=filtered_results,
                            output_dir=supplementary_dir,
                            title=(
                                f"FloatSOM Sample Size Scaling - {args.fixed_dimension}D"
                                f" ({topology} topology, with minibatch)"
                            ),
                            show_speedup=True,
                            show_error_bars=True,
                            show_legend=False,
                            emit_shared_legend=True,
                            methods_to_plot=["batch", "colors", "minibatch"],
                        )

    if grid_results:
        grid_output_dir = os.path.join(args.output_dir, "grid_size_scaling")
        os.makedirs(grid_output_dir, exist_ok=True)
        for topology, topology_results in grid_results.items():
            for processing_method, method_results in topology_results.items():
                ResultsHandler.save_results(
                    method_results,
                    grid_output_dir,
                    mode="grid_size",
                    method_suffix=f"_{topology}_{processing_method}",
                    merge_existing=getattr(args, "merge_existing", False),
                )
            if plot_gpu_scaling_benchmark:
                topology_dir = os.path.join(grid_output_dir, topology)
                os.makedirs(topology_dir, exist_ok=True)
                filtered_results = _filter_numeric_hierarchy(topology_results)
                if not filtered_results:
                    continue
                plot_gpu_scaling_benchmark(
                    results=filtered_results,
                    output_dir=topology_dir,
                    title=(
                        f"FloatSOM Grid Size Scaling - {topology} topology"
                    ),
                    show_speedup=True,
                    show_error_bars=True,
                    show_legend=False,
                    emit_shared_legend=True,
                    axis_mode="grid_size",
                )
                if _has_method(filtered_results, "minibatch"):
                    supplementary_dir = os.path.join(
                        topology_dir,
                        SUPPLEMENTARY_MINIBATCH_SUBDIR,
                    )
                    os.makedirs(supplementary_dir, exist_ok=True)
                    plot_gpu_scaling_benchmark(
                        results=filtered_results,
                        output_dir=supplementary_dir,
                        title=(
                            f"FloatSOM Grid Size Scaling - {topology} topology"
                            " (with minibatch)"
                        ),
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                        axis_mode="grid_size",
                        methods_to_plot=["batch", "colors", "minibatch"],
                    )

def main():
    """Main function for running GPU scaling benchmarks."""
    _configure_logging()
    # Ensure clean state before any work begins
    ensure_clean_ray_state(initial=True, free_gpu_memory=False)
    args = parse_args()

    if args.temp_dir:
        args.temp_dir = _validate_temp_dir(args.temp_dir, allow_tmp=not args.use_ray)

    args.run_temp_dir = _create_run_temp_dir(args.temp_dir)
    if args.use_ray:
        if not getattr(args, "ray_local_storage_path", None):
            raise ValueError("--ray_local_storage_path is required when --use_ray is set")
        args.ray_local_storage_path = _validate_temp_dir(args.ray_local_storage_path)

    if args.minibatch_chunk_size <= 0:
        raise ValueError("minibatch_chunk_size must be a positive integer")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer when provided")
    if args.run_timeout_minutes <= 0:
        raise ValueError("run_timeout_minutes must be a positive integer")
    
    resume_manager: Optional[ResumeManager] = None

    try:
        # Create output directory
        if args.output_dir is None:
            args.output_dir = f"gpu_scaling_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        args.output_dir = os.path.abspath(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.resume_state is None:
            args.resume_state = os.path.join(args.output_dir, "resume_state.json")
        else:
            args.resume_state = os.path.abspath(args.resume_state)

        resume_manager = ResumeManager(
            state_path=args.resume_state,
            enabled=args.resume,
            reset=args.reset_resume_state,
        )
        args.resume_manager = resume_manager

        if args.resume:
            print(f"Resume manifest: {args.resume_state}")
            if args.reset_resume_state:
                print("Existing resume manifest reset; starting fresh.")

        print("GPU Scaling Benchmark for FloatSOM")
        print(f"{'='*60}")
        print(f"Output directory: {args.output_dir}")
        print(f"Mode: {args.mode}")
        print(f"Topologies: {', '.join(args.topologies)}")
        print(f"Processing methods: {', '.join(args.processing_methods)}")
        print(f"Repeats per configuration: {args.repeats}")

        dim_results = None
        sample_results = None
        grid_results = None

        if args.mode in ['dimension_scaling', 'both', 'all']:
            print(f"\n{'='*60}")
            print("RUNNING DIMENSION SCALING BENCHMARKS")
            print(f"{'='*60}")
            print(
                f"Configurations: {len(args.topologies)} topologies × {len(args.processing_methods)} methods "
                f"× {len(args.dimensions)} dimensions × {len(args.gpu_counts)} GPU counts × {args.repeats} repeats"
            )
            print(
                f"Total runs: {len(args.topologies) * len(args.processing_methods) * len(args.dimensions) * len(args.gpu_counts) * args.repeats}"
            )

            dim_results = DimensionScalingBenchmark.run(args, args.output_dir)

            try:
                from floatsom.benchmarks.visualization import plot_gpu_scaling_benchmark

                dim_output_dir = os.path.join(args.output_dir, "dimension_scaling")
                for topology, topology_results in (dim_results or {}).items():
                    topology_dir = os.path.join(dim_output_dir, topology)
                    os.makedirs(topology_dir, exist_ok=True)
                    filtered_results = _filter_numeric_hierarchy(topology_results)
                    if not filtered_results:
                        print(f"Skipping visualization for {topology}: no numeric dimension keys found.")
                        continue
                    plot_gpu_scaling_benchmark(
                        results=filtered_results,
                        output_dir=topology_dir,
                        title=(
                            f"FloatSOM Dimension Scaling - {args.samples:,} samples"
                            f" ({topology} topology)"
                        ),
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                        axis_mode="dimension",
                    )
                    if _has_method(filtered_results, "minibatch"):
                        supplementary_dir = os.path.join(
                            topology_dir,
                            SUPPLEMENTARY_MINIBATCH_SUBDIR,
                        )
                        os.makedirs(supplementary_dir, exist_ok=True)
                        plot_gpu_scaling_benchmark(
                            results=filtered_results,
                            output_dir=supplementary_dir,
                            title=(
                                f"FloatSOM Dimension Scaling - {args.samples:,} samples"
                                f" ({topology} topology, with minibatch)"
                            ),
                            show_speedup=True,
                            show_error_bars=True,
                            show_legend=False,
                            emit_shared_legend=True,
                            axis_mode="dimension",
                            methods_to_plot=["batch", "colors", "minibatch"],
                        )

                if dim_results:
                    print(f"Dimension scaling visualizations saved under: {dim_output_dir}")
            except Exception as e:
                print(f"Error creating dimension scaling visualizations: {e}")
            finally:
                ensure_clean_ray_state(free_gpu_memory=False)

        if args.mode in ['sample_scaling', 'both', 'all']:
            print(f"\n{'='*60}")
            print("RUNNING SAMPLE SIZE SCALING BENCHMARKS")
            print(f"{'='*60}")
            print(f"Fixed dimension: {args.fixed_dimension}")
            print(
                f"Configurations: {len(args.topologies)} topologies × {len(args.processing_methods)} methods × {len(args.sample_sizes)} sample sizes "
                f"× {len(args.gpu_counts)} GPU counts × {args.repeats} repeats"
            )
            print(
                f"Total runs: {len(args.topologies) * len(args.processing_methods) * len(args.sample_sizes) * len(args.gpu_counts) * args.repeats}"
            )

            sample_results = SampleScalingBenchmark.run(args, args.output_dir)

            try:
                from floatsom.benchmarks.visualization import plot_sample_scaling_benchmark

                sample_output_dir = os.path.join(args.output_dir, "sample_scaling")
                for topology, topology_results in (sample_results or {}).items():
                    topology_dir = os.path.join(sample_output_dir, topology)
                    os.makedirs(topology_dir, exist_ok=True)
                    filtered_results = _filter_numeric_hierarchy(topology_results)
                    if not filtered_results:
                        print(f"Skipping visualization for {topology}: no numeric sample keys found.")
                        continue
                    plot_sample_scaling_benchmark(
                        results=filtered_results,
                        output_dir=topology_dir,
                        title=(
                            f"FloatSOM Sample Size Scaling - {args.fixed_dimension}D"
                            f" ({topology} topology)"
                        ),
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                    )
                    if _has_method(filtered_results, "minibatch"):
                        supplementary_dir = os.path.join(
                            topology_dir,
                            SUPPLEMENTARY_MINIBATCH_SUBDIR,
                        )
                        os.makedirs(supplementary_dir, exist_ok=True)
                        plot_sample_scaling_benchmark(
                            results=filtered_results,
                            output_dir=supplementary_dir,
                            title=(
                                f"FloatSOM Sample Size Scaling - {args.fixed_dimension}D"
                                f" ({topology} topology, with minibatch)"
                            ),
                            show_speedup=True,
                            show_error_bars=True,
                            show_legend=False,
                            emit_shared_legend=True,
                            methods_to_plot=["batch", "colors", "minibatch"],
                        )

                if sample_results:
                    print(f"Sample scaling visualizations saved under: {sample_output_dir}")
            except Exception as e:
                print(f"Error creating sample scaling visualizations: {e}")
            finally:
                ensure_clean_ray_state(free_gpu_memory=False)

        if args.mode in ['grid_size_scaling', 'all']:
            print(f"\n{'='*60}")
            print("RUNNING GRID SIZE SCALING BENCHMARKS")
            print(f"{'='*60}")
            print(f"Fixed dimension: {args.fixed_dimension}")
            print(f"Fixed samples: {args.samples:,}")
            print(
                f"Configurations: {len(args.topologies)} topologies × {len(args.processing_methods)} methods × {len(args.grid_sizes)} grid sizes "
                f"× {len(args.gpu_counts)} GPU counts × {args.repeats} repeats"
            )
            print(
                f"Total runs: {len(args.topologies) * len(args.processing_methods) * len(args.grid_sizes) * len(args.gpu_counts) * args.repeats}"
            )

            grid_results = GridSizeScalingBenchmark.run(args, args.output_dir)

            try:
                from floatsom.benchmarks.visualization import plot_gpu_scaling_benchmark

                grid_output_dir = os.path.join(args.output_dir, "grid_size_scaling")
                for topology, topology_results in (grid_results or {}).items():
                    topology_dir = os.path.join(grid_output_dir, topology)
                    os.makedirs(topology_dir, exist_ok=True)
                    filtered_results = _filter_numeric_hierarchy(topology_results)
                    if not filtered_results:
                        print(f"Skipping visualization for {topology}: no numeric grid-size keys found.")
                        continue
                    plot_gpu_scaling_benchmark(
                        results=filtered_results,
                        output_dir=topology_dir,
                        title=f"FloatSOM Grid Size Scaling - {topology} topology",
                        show_speedup=True,
                        show_error_bars=True,
                        show_legend=False,
                        emit_shared_legend=True,
                        axis_mode="grid_size",
                    )
                    if _has_method(filtered_results, "minibatch"):
                        supplementary_dir = os.path.join(
                            topology_dir,
                            SUPPLEMENTARY_MINIBATCH_SUBDIR,
                        )
                        os.makedirs(supplementary_dir, exist_ok=True)
                        plot_gpu_scaling_benchmark(
                            results=filtered_results,
                            output_dir=supplementary_dir,
                            title=(
                                f"FloatSOM Grid Size Scaling - {topology} topology"
                                " (with minibatch)"
                            ),
                            show_speedup=True,
                            show_error_bars=True,
                            show_legend=False,
                            emit_shared_legend=True,
                            axis_mode="grid_size",
                            methods_to_plot=["batch", "colors", "minibatch"],
                        )
                print(f"Grid size scaling visualizations saved to: {grid_output_dir}")
            except Exception as e:
                print(f"Error creating grid size scaling visualizations: {e}")
            finally:
                ensure_clean_ray_state(free_gpu_memory=False)

        ReportGenerator.create_summary_report(
            dim_results,
            sample_results,
            grid_results,
            args,
            args.output_dir,
        )

        ReportGenerator.print_final_summary(args, args.output_dir)

        _rewrite_all_outputs(args, dim_results, sample_results, grid_results)

    except BenchmarkTimeoutError as exc:
        # Individual benchmark runners now handle timeouts per algorithm,
        # but we keep this handler as a final safeguard.
        print(f"\nBenchmark run reported an unhandled timeout: {exc}")
        ensure_clean_ray_state(free_gpu_memory=False)

    finally:
        if resume_manager and resume_manager.enabled:
            try:
                resume_manager.flush()
            except Exception as exc:
                logging.warning("Could not flush resume manifest: %s", exc)

        ensure_clean_ray_state(free_gpu_memory=False)

        keep_temp_dir = bool(
            resume_manager
            and resume_manager.enabled
            and not resume_manager.all_done()
        )

        if keep_temp_dir:
            logging.info(
                "Leaving temporary directory in place for resume: %s",
                args.run_temp_dir,
            )
        else:
            _cleanup_run_temp_dir(args.run_temp_dir)

if __name__ == "__main__":
    _set_default_omp_num_threads()
    main()
