#!/usr/bin/env python3
"""
Individual benchmark runners for GPU scaling benchmarks.

This module handles execution of individual benchmark runs.
Single Responsibility: Execute and manage individual benchmark runs.
"""

import cProfile
import io
import multiprocessing
import os
import pstats
import queue
import shutil
import tempfile
import traceback
from typing import Any, Dict, Optional

MP_CONTEXT = multiprocessing.get_context("spawn")
from multiprocessing.queues import Queue as MPQueue

from floatsom.processing.ray_ops.workers.numexpr_config import configure_numexpr_threads

configure_numexpr_threads()

from .large_dataset import (
    calculate_quantization_error,
    cleanup_staged_dataset,
    generate_or_load_data,
    stage_dataset_for_execution,
    train_floatsom_with_zarr,
    write_results_report,
    write_mst_details,
)
from .config import BenchmarkConfig
from .ray_utils import ensure_clean_ray_state


class BenchmarkTimeoutError(Exception):
    """Raised when a benchmark run exceeds the configured timeout."""



class BenchmarkRunner:
    """Executes individual benchmark runs."""

    @staticmethod
    def prepare_dataset_for_execution(
        *,
        base_args,
        dimension: int,
        processing_method: str,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate/load a dataset once and optionally stage it into scratch so
        multiple benchmark configs can reuse the same execution path.
        """
        samples_for_run = num_samples if num_samples is not None else base_args.samples
        seed_for_run = seed if seed is not None else base_args.seed
        data_args = BenchmarkConfig.create_benchmark_args(
            base_args,
            dimension,
            num_gpus=1,
            num_samples=samples_for_run,
            seed=seed_for_run,
            processing_method=processing_method,
            topology_type="grid",
            grid_size=base_args.grid_size,
            total_iterations=base_args.total_iterations,
        )

        zarr_path, metadata = generate_or_load_data(data_args)
        dataset_stage_dir = BenchmarkRunner._create_dataset_stage_dir(
            base_args.run_temp_dir,
            samples=samples_for_run,
            dimension=dimension,
            seed=seed_for_run,
            processing_method=processing_method,
            zarr_chunk_size=int(data_args.zarr_chunk_size),
        )

        try:
            stage_info = stage_dataset_for_execution(
                zarr_path,
                temp_root=getattr(base_args, "temp_dir", None),
                staging_root=dataset_stage_dir,
                verbose=bool(getattr(base_args, "verbose", False)),
            )
        except Exception:
            BenchmarkRunner._cleanup_run_temp_dir(dataset_stage_dir)
            raise

        execution_zarr_path = str(stage_info["execution_path"])
        staged_metadata = dict(metadata)
        staged_metadata["source_zarr_path"] = str(stage_info["source_path"])
        staged_metadata["execution_zarr_path"] = execution_zarr_path
        staged_metadata["staging_applied"] = bool(stage_info["staged"])
        staged_metadata["staging_reason"] = str(stage_info["staging_reason"])
        staged_metadata["staging_elapsed_s"] = float(stage_info["staging_elapsed_s"])
        staged_metadata["staging_copy_method"] = str(stage_info["staging_copy_method"])
        staged_metadata["staging_prepared_for_shared_group"] = True
        staged_metadata["zarr_path"] = execution_zarr_path

        return {
            "execution_zarr_path": execution_zarr_path,
            "metadata": staged_metadata,
            "stage_info": stage_info,
            "dataset_stage_dir": dataset_stage_dir,
            "samples": int(samples_for_run),
            "dimension": int(dimension),
            "seed": int(seed_for_run),
            "processing_method": str(processing_method),
            "zarr_chunk_size": int(data_args.zarr_chunk_size),
        }

    @staticmethod
    def cleanup_prepared_dataset(prepared_dataset: Optional[Dict[str, Any]], *, verbose: bool = False) -> None:
        """Release a dataset prepared by prepare_dataset_for_execution()."""
        if not isinstance(prepared_dataset, dict):
            return
        cleanup_staged_dataset(prepared_dataset.get("stage_info"), verbose=bool(verbose))
        BenchmarkRunner._cleanup_run_temp_dir(prepared_dataset.get("dataset_stage_dir"))
    
    @staticmethod
    def run_benchmark(
        dimension: int,
        num_gpus: int,
        base_args,
        output_base_dir: str,
        num_samples: int = None,
        seed: int = None,
        config_name: str = None,
        processing_method: str = 'batch',
        topology_type: str = 'grid',
        grid_size: int = None,
        total_iterations: int = None,
        compute_qe: bool = False,
        prepared_dataset: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single benchmark configuration with timeout protection."""

        actual_grid_size = grid_size if grid_size is not None else base_args.grid_size
        actual_total_iterations = (
            total_iterations if total_iterations is not None else base_args.total_iterations
        )
        if actual_total_iterations is None:
            raise ValueError("total_iterations must be provided for benchmark runs")
        samples_for_run = num_samples if num_samples is not None else base_args.samples
        seed_for_run = seed if seed is not None else base_args.seed

        if config_name is None:
            config_name = BenchmarkConfig.generate_config_name(
                dimension,
                num_gpus,
                num_samples,
                topology_type,
                processing_method,
                actual_grid_size,
                seed_for_run,
            )

        output_dir = os.path.join(output_base_dir, "runs", config_name)
        os.makedirs(output_dir, exist_ok=True)

        args = BenchmarkConfig.create_benchmark_args(
            base_args,
            dimension,
            num_gpus,
            num_samples,
            seed,
            processing_method,
            topology_type,
            grid_size,
            actual_total_iterations,
        )

        if getattr(args, "profile_workers", False):
            if not getattr(args, "profile_workers_output_dir", None):
                args.profile_workers_output_dir = os.path.join(
                    output_dir, "worker_profiles"
                )

        BenchmarkRunner._print_benchmark_header(
            config_name,
            topology_type,
            dimension,
            samples_for_run,
            actual_grid_size,
            actual_total_iterations,
            num_gpus,
            processing_method,
            seed_for_run,
            args.chunk_size,
            getattr(args, "chunk_size_source", "resolved"),
        )

        if base_args.dry_run:
            print("DRY RUN - Skipping actual execution")
            return None

        run_temp_dir = None
        try:
            run_temp_dir = BenchmarkRunner._create_run_temp_dir(
                base_args.run_temp_dir,
                config_name,
            )
            args.run_temp_dir = run_temp_dir

            run_timeout_seconds = max(1, base_args.run_timeout_minutes * 60)
            ensure_clean_ray_state(free_gpu_memory=False)
            result_queue: MPQueue = MP_CONTEXT.Queue()

            process = MP_CONTEXT.Process(
                target=BenchmarkRunner._worker_entry,
                args=(
                    result_queue,
                args,
                output_dir,
                base_args.verbose,
                compute_qe,
                prepared_dataset,
            ),
        )

            process.start()
            process.join(run_timeout_seconds)

            if process.is_alive():
                print(
                    f"Benchmark {config_name} exceeded {base_args.run_timeout_minutes} minutes; terminating run."
                )
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
                result_queue.close()
                result_queue.join_thread()
                BenchmarkRunner._ensure_ray_shutdown()
                raise BenchmarkTimeoutError(config_name)

            try:
                status, payload = result_queue.get_nowait()
            except queue.Empty:
                status, payload = 'error', {'message': 'No result returned from worker.', 'traceback': None}
            finally:
                process.join()
                process.close()
                result_queue.close()
                result_queue.join_thread()

            if status != 'success':
                message = payload.get('message', 'Unknown error')
                print(f"ERROR: Benchmark failed with exception: {message}")
                if payload.get('traceback'):
                    print(payload['traceback'])
                BenchmarkRunner._ensure_ray_shutdown()
                return None

            train_time = payload['train_time']
            training_stats = payload['training_stats']

            print(f"✓ Training completed in {train_time:.2f}s")

            BenchmarkRunner._save_log_file(
                output_base_dir,
                config_name,
                topology_type,
                dimension,
                samples_for_run,
                actual_grid_size,
                actual_total_iterations,
                num_gpus,
                seed_for_run,
                train_time,
                training_stats,
            )

            BenchmarkRunner._ensure_ray_shutdown()

            payload['config_name'] = config_name
            payload['output_dir'] = output_dir
            return payload
        finally:
            BenchmarkRunner._cleanup_run_temp_dir(run_temp_dir)

    @staticmethod
    def _worker_entry(
        result_queue: MPQueue,
        args,
        output_dir: str,
        verbose: bool,
        compute_qe: bool,
        prepared_dataset: Optional[Dict[str, Any]],
    ) -> None:
        """Worker process entry point for executing a single benchmark run."""

        profiler = None
        profile_output_path = None
        profiling_active = False
        profile_written = False
        stage_info: Optional[Dict[str, Any]] = None

        try:
            ensure_clean_ray_state()
            if prepared_dataset is None:
                zarr_path, metadata = generate_or_load_data(args)
                stage_info = stage_dataset_for_execution(
                    zarr_path,
                    temp_root=getattr(args, "temp_dir", None),
                    staging_root=getattr(args, "run_temp_dir", None),
                    verbose=bool(verbose),
                )
                execution_zarr_path = str(stage_info["execution_path"])
                metadata = dict(metadata)
                metadata["source_zarr_path"] = str(stage_info["source_path"])
                metadata["execution_zarr_path"] = execution_zarr_path
                metadata["staging_applied"] = bool(stage_info["staged"])
                metadata["staging_reason"] = str(stage_info["staging_reason"])
                metadata["staging_elapsed_s"] = float(stage_info["staging_elapsed_s"])
                metadata["staging_copy_method"] = str(stage_info["staging_copy_method"])
                metadata["zarr_path"] = execution_zarr_path
            else:
                execution_zarr_path = str(prepared_dataset["execution_zarr_path"])
                metadata = dict(prepared_dataset["metadata"])
                metadata["staging_reused_from_prepared_dataset"] = True

            if getattr(args, "profile", False):
                profile_output_name = getattr(args, "profile_output", "profile_results.txt")
                if not profile_output_name:
                    profile_output_name = "profile_results.txt"
                profile_output_path = os.path.join(output_dir, profile_output_name)
                print(f"Profiling enabled. Results will be saved to: {profile_output_path}")
                profiler = cProfile.Profile()
                profiler.enable()
                profiling_active = True

            som, training_stats, train_time = train_floatsom_with_zarr(execution_zarr_path, args, metadata)

            try:
                BenchmarkRunner._cleanup_ray_resources(som, args)
            except Exception:
                pass

            if profiling_active:
                profiler.disable()
                profiling_active = False

            quantization_error = None
            if compute_qe or getattr(args, 'calculate_qe', False):
                quantization_error = calculate_quantization_error(
                    execution_zarr_path,
                    som,
                    args,
                    output_dir,
                    report_dataset_path=str(metadata["source_zarr_path"]),
                )
                training_stats['quantization_error'] = quantization_error

            output_file = os.path.join(output_dir, args.output_file)
            write_results_report(som, training_stats, train_time, metadata, args, output_file)

            write_mst_details(som, args, output_file)

            if profiler and profile_output_path:
                BenchmarkRunner._write_profile_stats(profiler, profile_output_path)
                profile_written = True

            result_queue.put(
                (
                    'success',
                    {
                        'train_time': train_time,
                        'training_stats': BenchmarkRunner._serialize_training_stats(training_stats),
                        'quantization_error': quantization_error,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive path
            result_queue.put(
                (
                    'error',
                    {
                        'message': str(exc),
                        'traceback': traceback.format_exc(),
                    },
                )
            )
        finally:
            if profiler and profile_output_path:
                if profiling_active:
                    profiler.disable()
                if not profile_written:
                    BenchmarkRunner._write_profile_stats(profiler, profile_output_path)
            cleanup_staged_dataset(stage_info, verbose=bool(verbose))
            BenchmarkRunner._ensure_ray_shutdown(free_gpu_memory=True)

    @staticmethod
    def _print_benchmark_header(config_name: str, topology_type: str, dimension: int,
                               samples: int, grid_size: int, total_iterations: int,
                               num_gpus: int, processing_method: str, seed: int,
                               chunk_size: int, chunk_size_source: str):
        """Print benchmark configuration header."""
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config_name}")
        print(f"  Topology: {topology_type}")
        print(f"  Dimension: {dimension}")
        print(f"  Samples: {samples:,}")
        print(f"  Grid Size: {grid_size}×{grid_size}")
        print(f"  total_iterations: {total_iterations}")
        print(f"  GPUs: {num_gpus}")
        print(f"  Processing Method: {processing_method}")
        print(f"  Seed: {seed}")
        if chunk_size_source == "manual_cli_chunk_size":
            chunk_note = "manual override (--chunk_size)"
        elif chunk_size_source == "manual_minibatch_chunk_size":
            chunk_note = "manual override (--minibatch_chunk_size)"
        elif chunk_size_source == "auto_dimension_vram":
            chunk_note = "auto-resolved from dimension + visible VRAM"
        else:
            chunk_note = "resolved"
        print(f"  Chunk Size: {chunk_size:,} ({chunk_note})")
        print(f"{'='*60}")

    @staticmethod
    def _create_run_temp_dir(base_run_temp_dir: Optional[str], config_name: str) -> str:
        """Create a per-run temp directory under the benchmark temp root."""
        safe_name = config_name.replace(os.sep, "_")
        if os.altsep:
            safe_name = safe_name.replace(os.altsep, "_")
        prefix = f"run_{safe_name}_"
        if not base_run_temp_dir:
            return tempfile.mkdtemp(prefix=prefix)
        return tempfile.mkdtemp(prefix=prefix, dir=base_run_temp_dir)

    @staticmethod
    def _create_dataset_stage_dir(
        base_run_temp_dir: Optional[str],
        *,
        samples: int,
        dimension: int,
        seed: int,
        processing_method: str,
        zarr_chunk_size: int,
    ) -> str:
        """Create a temp directory for a staged dataset shared across compatible configs."""
        prefix = (
            "dataset_"
            f"s{int(samples)}_"
            f"d{int(dimension)}_"
            f"seed{int(seed)}_"
            f"{str(processing_method)}_"
            f"zc{int(zarr_chunk_size)}_"
        )
        if not base_run_temp_dir:
            return tempfile.mkdtemp(prefix=prefix)
        return tempfile.mkdtemp(prefix=prefix, dir=base_run_temp_dir)

    @staticmethod
    def _cleanup_run_temp_dir(run_temp_dir: Optional[str]) -> None:
        """Best-effort cleanup of per-run temp directories."""
        if not run_temp_dir:
            return
        try:
            shutil.rmtree(run_temp_dir)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"Warning: Could not remove temporary directory {run_temp_dir}: {exc}")
    
    @staticmethod
    def _cleanup_ray_resources(som, args):
        """Cleanup Ray resources after training."""
        if args.use_ray and hasattr(som, 'processor'):
            processor = som.processor
            cleanup_fn = getattr(processor, "cleanup", None)
            if callable(cleanup_fn):
                worker_manager = getattr(processor, "worker_manager", None)
                has_live_workers = False
                if worker_manager is not None:
                    workers = getattr(worker_manager, "workers", None)
                    try:
                        has_live_workers = len(workers or []) > 0
                    except Exception:
                        has_live_workers = bool(workers)

                # FloatSOM finalization already cleans Ray worker managers. Only invoke
                # processor.cleanup() here if workers are still live (or no manager exists).
                if worker_manager is None or has_live_workers:
                    cleanup_fn()

        ensure_clean_ray_state(free_gpu_memory=False)

    @staticmethod
    def _save_log_file(output_base_dir: str, config_name: str, topology_type: str,
                      dimension: int, samples: int, grid_size: int, total_iterations: int,
                      num_gpus: int, seed: int, train_time: float, training_stats: dict):
        """Save detailed log file for the benchmark run."""
        log_file = os.path.join(output_base_dir, "logs", f"{config_name}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Configuration: {config_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Topology: {topology_type}\n")
            f.write(f"Dimension: {dimension}\n")
            f.write(f"Samples: {samples:,}\n")
            f.write(f"Grid Size: {grid_size}×{grid_size}\n")
            f.write(f"total_iterations: {total_iterations}\n")
            f.write(f"GPUs: {num_gpus}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Training time: {train_time:.2f}s\n")

            iterations_completed = training_stats.get("iterations_completed")
            total_samples_processed = training_stats.get("total_samples_processed")

            if iterations_completed is not None:
                f.write(f"Iterations completed: {int(iterations_completed)}\n")
            if total_samples_processed is not None:
                f.write(f"Total samples processed: {int(total_samples_processed):,}\n")

            # Timing breakdown (if available)
            setup_s = training_stats.get("time_setup_s")
            iter_s = training_stats.get("time_iteration_s")
            finalize_s = training_stats.get("time_finalize_s")
            if setup_s is not None:
                f.write(f"Setup time: {float(setup_s):.2f}s\n")
            if iter_s is not None:
                f.write(f"Iteration time: {float(iter_s):.2f}s\n")
                if iterations_completed:
                    try:
                        per_iter = float(iter_s) / max(1, int(iterations_completed))
                        f.write(f"Iteration time per iter: {per_iter:.4f}s\n")
                    except Exception:
                        pass
            if finalize_s is not None:
                f.write(f"Finalize time: {float(finalize_s):.2f}s\n")

            data_staging_s = training_stats.get("data_staging_s")
            if data_staging_s is not None:
                f.write(f"Data staging time: {float(data_staging_s):.2f}s\n")

            staging_mode = training_stats.get("data_staging_mode")
            staging_strategy = training_stats.get("data_staging_strategy")
            if staging_mode is not None:
                f.write(f"Data staging mode: {staging_mode}\n")
            if staging_strategy is not None:
                f.write(f"Data staging strategy: {staging_strategy}\n")

            staging_workers = training_stats.get("data_staging_workers")
            if staging_workers is not None:
                f.write(f"Data staging workers: {int(staging_workers)}\n")

            # Worker staging summaries (Zarr direct-to-local and similar)
            for label, key in (
                ("Worker staging elapsed min", "data_staging_worker_elapsed_s_min"),
                ("Worker staging elapsed max", "data_staging_worker_elapsed_s_max"),
                ("Worker staging elapsed mean", "data_staging_worker_elapsed_s_mean"),
                ("Worker staging throughput min", "data_staging_worker_throughput_gb_s_min"),
                ("Worker staging throughput max", "data_staging_worker_throughput_gb_s_max"),
                ("Worker staging throughput mean", "data_staging_worker_throughput_gb_s_mean"),
            ):
                value = training_stats.get(key)
                if value is None:
                    continue
                suffix = " GB/s" if "throughput" in key else "s"
                f.write(f"{label}: {float(value):.2f}{suffix}\n")

            # Ray per-iteration timing breakdown (print all collected entries; see RayConfig.timing_log_every_n_iterations).
            ray_iter_timing = training_stats.get("ray_iteration_timing")
            if isinstance(ray_iter_timing, list) and ray_iter_timing:
                entries = [entry for entry in ray_iter_timing if isinstance(entry, dict)]
                if entries:
                    f.write("\nRay iteration timing breakdown:\n")
                    f.write("-" * 40 + "\n")

                    for idx, entry in enumerate(entries):
                        iter_idx = entry.get("iteration")
                        workers = entry.get("workers")
                        driver_submit_s = entry.get("driver_submit_s")
                        driver_get_s = entry.get("driver_ray_get_s")
                        submit_repr = "n/a"
                        if isinstance(driver_submit_s, (int, float)):
                            submit_repr = f"{float(driver_submit_s):.3f}"
                        get_repr = "n/a"
                        if isinstance(driver_get_s, (int, float)):
                            get_repr = f"{float(driver_get_s):.3f}"
                        f.write(
                            f"Iteration {iter_idx} (workers={workers}): "
                            f"driver_submit_s={submit_repr} "
                            f"ray_get_s={get_repr}\n"
                        )

                        chunking = entry.get("chunking")
                        if isinstance(chunking, dict):
                            num_chunks = chunking.get("num_chunks")
                            update_freq = chunking.get("weight_update_frequency")
                            f.write(f"  chunking: num_chunks={num_chunks} weight_update_frequency={update_freq}\n")

                        slowest_worker = entry.get("slowest_worker_id")
                        slowest_total = entry.get("slowest_worker_total_s")
                        if slowest_worker is not None and isinstance(slowest_total, (int, float)):
                            f.write(
                                f"  slowest_worker: id={slowest_worker} iteration_total_s={float(slowest_total):.3f}\n"
                            )

                        breakdown = entry.get("worker_breakdown_s")
                        if isinstance(breakdown, dict) and breakdown:
                            for key in sorted(breakdown.keys()):
                                stats = breakdown.get(key)
                                if not isinstance(stats, dict):
                                    continue
                                mean_s = stats.get("mean_s")
                                max_s = stats.get("max_s")
                                if mean_s is None or max_s is None:
                                    continue
                                f.write(f"  {key}: mean={float(mean_s):.3f}s max={float(max_s):.3f}s\n")

                        source_counts = entry.get("chunk_source_counts")
                        if isinstance(source_counts, dict) and source_counts:
                            for key in sorted(source_counts.keys()):
                                stats = source_counts.get(key)
                                if not isinstance(stats, dict):
                                    continue
                                mean = stats.get("mean")
                                max_v = stats.get("max")
                                if mean is None or max_v is None:
                                    continue
                                f.write(f"  {key}: mean={float(mean):.2f} max={float(max_v):.2f}\n")

                        if idx != len(entries) - 1:
                            f.write("\n")

            if 'quantization_error' in training_stats:
                f.write(f"Quantization error: {training_stats['quantization_error']:.6f}\n")

    @staticmethod
    def _write_profile_stats(profiler: cProfile.Profile, output_path: str) -> None:
        """Persist profiling results to a text file."""
        if not output_path:
            return

        try:
            with open(output_path, "w") as f:
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
                stats.print_stats(50)
                f.write(stream.getvalue())
            print(f"Profiling results saved to: {output_path}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Warning: Could not write profiling results to {output_path}: {exc}")

    @staticmethod
    def _serialize_training_stats(training_stats: dict) -> dict:
        """Convert training stats to types safe for cross-process communication."""

        serializable = {}
        for key, value in training_stats.items():
            if hasattr(value, 'item'):
                try:
                    serializable[key] = value.item()
                    continue
                except Exception:
                    pass
            serializable[key] = value
        return serializable

    @staticmethod
    def _ensure_ray_shutdown(*, free_gpu_memory: bool = False) -> None:
        """Best-effort Ray shutdown to avoid stale workers."""

        ensure_clean_ray_state(free_gpu_memory=free_gpu_memory)
