#!/usr/bin/env python3
"""
Specific benchmark type implementations for GPU scaling.

This module contains implementations for different benchmark strategies.
Single Responsibility: Implement different benchmark scaling strategies.
"""

import numpy as np
import traceback
from typing import Any, Callable, Dict, Optional
import os 

from .config import BenchmarkConfig
from .runners import BenchmarkRunner, BenchmarkTimeoutError
from .ray_utils import ensure_clean_ray_state
from .results import ResultsHandler


def _run_shared_dataset_group(
    *,
    base_args,
    dataset_dimension: int,
    dataset_processing_method: str,
    dataset_num_samples: Optional[int],
    dataset_seed: Optional[int],
    pending_runs: list[Dict[str, Any]],
    on_success: Callable[[Dict[str, Any], Dict[str, Any]], None],
    on_failure: Callable[[Dict[str, Any], str], None],
) -> None:
    """Prepare one staged dataset and reuse it across all compatible runs."""
    if not pending_runs:
        return

    resume_manager = getattr(base_args, "resume_manager", None)
    prepared_dataset = None

    try:
        prepared_dataset = BenchmarkRunner.prepare_dataset_for_execution(
            base_args=base_args,
            dimension=dataset_dimension,
            processing_method=dataset_processing_method,
            num_samples=dataset_num_samples,
            seed=dataset_seed,
        )

        for pending in pending_runs:
            task_key = pending.get("task_key")
            if resume_manager and resume_manager.enabled and task_key:
                resume_manager.mark_running(task_key)

            result = None
            try:
                run_kwargs = {
                    "dimension": dataset_dimension,
                    "base_args": base_args,
                    "seed": dataset_seed,
                    "processing_method": dataset_processing_method,
                    "prepared_dataset": prepared_dataset,
                }
                run_kwargs.update(pending["run_kwargs"])
                result = BenchmarkRunner.run_benchmark(**run_kwargs)
            except BenchmarkTimeoutError as timeout_exc:
                print(
                    f"Timeout for {pending['config_label']}: {timeout_exc}. "
                    "Continuing with remaining configurations in this shared dataset group."
                )
                if resume_manager and resume_manager.enabled and task_key:
                    resume_manager.mark_failed(task_key, error=str(timeout_exc))
                on_failure(pending, f"timeout: {timeout_exc}")
                continue
            except Exception as exc:
                print(
                    f"Error while running {pending['config_label']}: {exc}. "
                    "Continuing with remaining configurations in this shared dataset group."
                )
                traceback.print_exc()
                if resume_manager and resume_manager.enabled and task_key:
                    resume_manager.mark_failed(task_key, error=str(exc))
                on_failure(pending, str(exc))
                continue
            finally:
                ensure_clean_ray_state()

            if result is None:
                print(
                    f"Benchmark failed for {pending['config_label']}. "
                    "Continuing with remaining configurations in this shared dataset group."
                )
                if resume_manager and resume_manager.enabled and task_key:
                    resume_manager.mark_failed(task_key, error="run returned no result")
                on_failure(pending, "run returned no result")
                continue

            if resume_manager and resume_manager.enabled and task_key:
                done_metadata = {}
                log_file = pending.get("log_file")
                if log_file:
                    done_metadata["log_file"] = log_file
                done_metadata_builder = pending.get("done_metadata_builder")
                if callable(done_metadata_builder):
                    done_metadata.update(done_metadata_builder(result))
                resume_manager.mark_done(
                    task_key,
                    train_time=result["train_time"],
                    metadata=done_metadata,
                )

            on_success(pending, result)
    finally:
        BenchmarkRunner.cleanup_prepared_dataset(
            prepared_dataset,
            verbose=bool(getattr(base_args, "verbose", False)),
        )


class DimensionScalingBenchmark:
    """Run dimension scaling benchmarks."""

    @staticmethod
    def run(base_args, output_dir: str) -> Dict:
        """
        Run dimension scaling benchmarks with multiple repeats for each processing method and topology.

        Returns:
            Dictionary with results: {topology: {method: {dimension: {gpu_count: {'times': [...], 'mean': ..., 'std': ...}}}}}
        """
        results = {}

        dim_output_dir = os.path.join(output_dir, "dimension_scaling")
        os.makedirs(os.path.join(dim_output_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(dim_output_dir, "logs"), exist_ok=True)

        total_configs = (
            len(base_args.topologies)
            * len(base_args.processing_methods)
            * len(base_args.dimensions)
            * len(base_args.gpu_counts)
            * base_args.repeats
        )
        current_config = 0

        for topology_type in base_args.topologies:
            results.setdefault(topology_type, {})
            for processing_method in base_args.processing_methods:
                results[topology_type].setdefault(processing_method, {})

        times_by_config: Dict[tuple[str, str, int, int], list[float]] = {}
        failed_configs: set[tuple[str, str, int, int]] = set()

        def _refresh_results_snapshot() -> None:
            for topology_name in list(results.keys()):
                for method_name in list(results[topology_name].keys()):
                    results[topology_name][method_name] = {}

            for (topology_name, method_name, dimension_value, gpu_value), times in times_by_config.items():
                if not times or (topology_name, method_name, dimension_value, gpu_value) in failed_configs:
                    continue
                results.setdefault(topology_name, {}).setdefault(method_name, {})
                results[topology_name][method_name].setdefault(dimension_value, {})
                results[topology_name][method_name][dimension_value][gpu_value] = {
                    "times": list(times),
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "count": len(times),
                }

            for topology_name, topology_results in results.items():
                for method_name, method_results in topology_results.items():
                    if not method_results:
                        continue
                    ResultsHandler.save_results(
                        method_results,
                        dim_output_dir,
                        mode="dimension",
                        method_suffix=f"_{topology_name}_{method_name}",
                        merge_existing=getattr(base_args, "merge_existing", False),
                    )

        def _record_success(pending: Dict[str, Any], result: Dict[str, Any]) -> None:
            times_by_config[pending["config_key"]].append(result["train_time"])
            _refresh_results_snapshot()

        def _record_failure(pending: Dict[str, Any], _error: str) -> None:
            failed_configs.add(pending["config_key"])
            _refresh_results_snapshot()

        for dimension in base_args.dimensions:
            for repeat in range(base_args.repeats):
                seed = base_args.seed + repeat

                for processing_method in base_args.processing_methods:
                    pending_runs = []
                    resume_manager = getattr(base_args, "resume_manager", None)

                    for num_gpus in base_args.gpu_counts:
                        for topology_type in base_args.topologies:
                            config_key = (
                                str(topology_type),
                                str(processing_method),
                                int(dimension),
                                int(num_gpus),
                            )
                            if config_key in failed_configs:
                                continue

                            times = times_by_config.setdefault(config_key, [])
                            current_config += 1

                            print(
                                f"\n[{current_config}/{total_configs}] Topology={topology_type}, "
                                f"Method={processing_method}, Dimension={dimension}, "
                                f"GPUs={num_gpus}, Repeat={repeat+1}/{base_args.repeats}"
                            )

                            config_label = (
                                f"topology={topology_type}, method={processing_method}, "
                                f"dimension={dimension}, gpus={num_gpus}, repeat={repeat+1}"
                            )
                            config_name = (
                                f"{topology_type}_dim{dimension}_grid{base_args.grid_size}_gpu{num_gpus}_"
                                f"{processing_method}_seed{seed}"
                            )
                            log_file = os.path.join(dim_output_dir, "logs", f"{config_name}.log")

                            task_metadata = {
                                "mode": "dimension_scaling",
                                "topology": topology_type,
                                "processing_method": processing_method,
                                "dimension": dimension,
                                "gpu_count": num_gpus,
                                "repeat": repeat + 1,
                                "config_name": config_name,
                            }

                            skip_run, task_key, _, _ = _maybe_skip_completed_run(
                                base_args=base_args,
                                resume_manager=resume_manager,
                                times=times,
                                mode="dimension_scaling",
                                topology_type=topology_type,
                                processing_method=processing_method,
                                axis_name="dimension",
                                axis_value=dimension,
                                gpu_count=num_gpus,
                                repeat_index=repeat + 1,
                                log_file=log_file,
                                metadata=task_metadata,
                                config_label=config_label,
                                config_name=config_name,
                            )

                            if skip_run:
                                _refresh_results_snapshot()
                                continue

                            if base_args.dry_run:
                                continue

                            pending_runs.append(
                                {
                                    "config_key": config_key,
                                    "config_label": config_label,
                                    "task_key": task_key,
                                    "log_file": log_file,
                                    "run_kwargs": {
                                        "num_gpus": num_gpus,
                                        "output_base_dir": dim_output_dir,
                                        "topology_type": topology_type,
                                        "compute_qe": False,
                                    },
                                }
                            )

                    _run_shared_dataset_group(
                        base_args=base_args,
                        dataset_dimension=dimension,
                        dataset_processing_method=processing_method,
                        dataset_num_samples=base_args.samples,
                        dataset_seed=seed,
                        pending_runs=pending_runs,
                        on_success=_record_success,
                        on_failure=_record_failure,
                    )

        return results


class SampleScalingBenchmark:
    """Run sample size scaling benchmarks."""

    @staticmethod
    def run(base_args, output_dir: str) -> Dict:
        """
        Run sample size scaling benchmarks with multiple repeats.

        Returns:
            Dictionary with results
        """
        results = {}

        sample_output_dir = os.path.join(output_dir, "sample_scaling")
        os.makedirs(os.path.join(sample_output_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(sample_output_dir, "logs"), exist_ok=True)

        total_configs = (
            len(base_args.topologies)
            * len(base_args.processing_methods)
            * len(base_args.sample_sizes)
            * len(base_args.gpu_counts)
            * base_args.repeats
        )
        current_config = 0

        for topology_type in base_args.topologies:
            results.setdefault(topology_type, {})
            for processing_method in base_args.processing_methods:
                results[topology_type].setdefault(processing_method, {})

        times_by_config: Dict[tuple[str, str, int, int], list[float]] = {}
        failed_configs: set[tuple[str, str, int, int]] = set()

        def _refresh_results_snapshot() -> None:
            for topology_name in list(results.keys()):
                for method_name in list(results[topology_name].keys()):
                    results[topology_name][method_name] = {}

            for (topology_name, method_name, samples_value, gpu_value), times in times_by_config.items():
                if not times or (topology_name, method_name, samples_value, gpu_value) in failed_configs:
                    continue
                results.setdefault(topology_name, {}).setdefault(method_name, {})
                results[topology_name][method_name].setdefault(samples_value, {})
                results[topology_name][method_name][samples_value][gpu_value] = {
                    "times": list(times),
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "count": len(times),
                }

            for topology_name, topology_results in results.items():
                for method_name, method_results in topology_results.items():
                    if not method_results:
                        continue
                    ResultsHandler.save_results(
                        method_results,
                        sample_output_dir,
                        mode="sample",
                        method_suffix=f"_{topology_name}_{method_name}",
                        merge_existing=getattr(base_args, "merge_existing", False),
                    )

        def _record_success(pending: Dict[str, Any], result: Dict[str, Any]) -> None:
            times_by_config[pending["config_key"]].append(result["train_time"])
            _refresh_results_snapshot()

        def _record_failure(pending: Dict[str, Any], _error: str) -> None:
            failed_configs.add(pending["config_key"])
            _refresh_results_snapshot()

        for sample_size in base_args.sample_sizes:
            for repeat in range(base_args.repeats):
                seed = base_args.seed + repeat

                for processing_method in base_args.processing_methods:
                    pending_runs = []
                    resume_manager = getattr(base_args, "resume_manager", None)

                    for num_gpus in base_args.gpu_counts:
                        for topology_type in base_args.topologies:
                            config_key = (
                                str(topology_type),
                                str(processing_method),
                                int(sample_size),
                                int(num_gpus),
                            )
                            if config_key in failed_configs:
                                continue

                            times = times_by_config.setdefault(config_key, [])
                            current_config += 1

                            print(
                                f"\n[{current_config}/{total_configs}] Topology={topology_type}, "
                                f"Method={processing_method}, Samples={sample_size:,}, "
                                f"GPUs={num_gpus}, Repeat={repeat+1}/{base_args.repeats}"
                            )

                            config_label = (
                                f"topology={topology_type}, method={processing_method}, "
                                f"samples={sample_size:,}, gpus={num_gpus}, repeat={repeat+1}"
                            )
                            config_name = BenchmarkConfig.generate_config_name(
                                base_args.fixed_dimension,
                                num_gpus,
                                sample_size,
                                topology_type,
                                processing_method,
                                base_args.grid_size,
                                seed,
                            )
                            log_file = os.path.join(sample_output_dir, "logs", f"{config_name}.log")

                            task_metadata = {
                                "mode": "sample_scaling",
                                "topology": topology_type,
                                "processing_method": processing_method,
                                "samples": sample_size,
                                "gpu_count": num_gpus,
                                "repeat": repeat + 1,
                                "config_name": config_name,
                            }

                            skip_run, task_key, _, _ = _maybe_skip_completed_run(
                                base_args=base_args,
                                resume_manager=resume_manager,
                                times=times,
                                mode="sample_scaling",
                                topology_type=topology_type,
                                processing_method=processing_method,
                                axis_name="samples",
                                axis_value=sample_size,
                                gpu_count=num_gpus,
                                repeat_index=repeat + 1,
                                log_file=log_file,
                                metadata=task_metadata,
                                config_label=config_label,
                                config_name=config_name,
                            )

                            if skip_run:
                                _refresh_results_snapshot()
                                continue

                            if base_args.dry_run:
                                continue

                            pending_runs.append(
                                {
                                    "config_key": config_key,
                                    "config_label": config_label,
                                    "task_key": task_key,
                                    "log_file": log_file,
                                    "run_kwargs": {
                                        "num_gpus": num_gpus,
                                        "output_base_dir": sample_output_dir,
                                        "num_samples": sample_size,
                                        "topology_type": topology_type,
                                        "compute_qe": False,
                                    },
                                }
                            )

                    _run_shared_dataset_group(
                        base_args=base_args,
                        dataset_dimension=base_args.fixed_dimension,
                        dataset_processing_method=processing_method,
                        dataset_num_samples=sample_size,
                        dataset_seed=seed,
                        pending_runs=pending_runs,
                        on_success=_record_success,
                        on_failure=_record_failure,
                    )

        return results


class GridSizeScalingBenchmark:
    """Run grid size scaling benchmarks."""

    @staticmethod
    def run(base_args, output_dir: str) -> Dict:
        """
        Run grid size scaling benchmarks with multiple repeats.

        Returns:
            Dictionary with results
        """
        results = {}

        grid_output_dir = os.path.join(output_dir, "grid_size_scaling")
        os.makedirs(os.path.join(grid_output_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(grid_output_dir, "logs"), exist_ok=True)

        total_configs = (
            len(base_args.topologies)
            * len(base_args.processing_methods)
            * len(base_args.grid_sizes)
            * len(base_args.gpu_counts)
            * base_args.repeats
        )
        current_config = 0

        for topology_type in base_args.topologies:
            results.setdefault(topology_type, {})
            for processing_method in base_args.processing_methods:
                results[topology_type].setdefault(processing_method, {})

        times_by_config: Dict[tuple[str, str, int, int], list[float]] = {}
        failed_configs: set[tuple[str, str, int, int]] = set()

        def _refresh_results_snapshot() -> None:
            for topology_name in list(results.keys()):
                for method_name in list(results[topology_name].keys()):
                    results[topology_name][method_name] = {}

            for (topology_name, method_name, grid_value, gpu_value), times in times_by_config.items():
                if not times or (topology_name, method_name, grid_value, gpu_value) in failed_configs:
                    continue
                results.setdefault(topology_name, {}).setdefault(method_name, {})
                results[topology_name][method_name].setdefault(grid_value, {})
                results[topology_name][method_name][grid_value][gpu_value] = {
                    "times": list(times),
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "count": len(times),
                }

            for topology_name, topology_results in results.items():
                for method_name, method_results in topology_results.items():
                    if not method_results:
                        continue
                    ResultsHandler.save_results(
                        method_results,
                        grid_output_dir,
                        mode="grid_size",
                        method_suffix=f"_{topology_name}_{method_name}",
                        merge_existing=getattr(base_args, "merge_existing", False),
                    )

        def _record_success(pending: Dict[str, Any], result: Dict[str, Any]) -> None:
            times_by_config[pending["config_key"]].append(result["train_time"])
            _refresh_results_snapshot()

        def _record_failure(pending: Dict[str, Any], _error: str) -> None:
            failed_configs.add(pending["config_key"])
            _refresh_results_snapshot()

        for repeat in range(base_args.repeats):
            seed = base_args.seed + repeat

            for processing_method in base_args.processing_methods:
                pending_runs = []
                resume_manager = getattr(base_args, "resume_manager", None)

                for grid_size in base_args.grid_sizes:
                    for num_gpus in base_args.gpu_counts:
                        for topology_type in base_args.topologies:
                            config_key = (
                                str(topology_type),
                                str(processing_method),
                                int(grid_size),
                                int(num_gpus),
                            )
                            if config_key in failed_configs:
                                continue

                            times = times_by_config.setdefault(config_key, [])
                            current_config += 1

                            print(
                                f"\n[{current_config}/{total_configs}] Topology={topology_type}, "
                                f"Method={processing_method}, Grid={grid_size}×{grid_size}, "
                                f"GPUs={num_gpus}, Repeat={repeat+1}/{base_args.repeats}"
                            )

                            config_label = (
                                f"topology={topology_type}, method={processing_method}, "
                                f"grid={grid_size}×{grid_size}, gpus={num_gpus}, repeat={repeat+1}"
                            )
                            config_name = (
                                f"{topology_type}_dim{base_args.fixed_dimension}_grid{grid_size}_"
                                f"gpu{num_gpus}_{processing_method}_seed{seed}"
                            )
                            log_file = os.path.join(grid_output_dir, "logs", f"{config_name}.log")

                            task_metadata = {
                                "mode": "grid_size_scaling",
                                "topology": topology_type,
                                "processing_method": processing_method,
                                "grid_size": grid_size,
                                "gpu_count": num_gpus,
                                "repeat": repeat + 1,
                                "config_name": config_name,
                            }

                            skip_run, task_key, _, _ = _maybe_skip_completed_run(
                                base_args=base_args,
                                resume_manager=resume_manager,
                                times=times,
                                mode="grid_size_scaling",
                                topology_type=topology_type,
                                processing_method=processing_method,
                                axis_name="grid_size",
                                axis_value=grid_size,
                                gpu_count=num_gpus,
                                repeat_index=repeat + 1,
                                log_file=log_file,
                                metadata=task_metadata,
                                config_label=config_label,
                                config_name=config_name,
                            )

                            if skip_run:
                                _refresh_results_snapshot()
                                continue

                            if base_args.dry_run:
                                continue

                            pending_runs.append(
                                {
                                    "config_key": config_key,
                                    "config_label": config_label,
                                    "task_key": task_key,
                                    "log_file": log_file,
                                    "run_kwargs": {
                                        "num_gpus": num_gpus,
                                        "output_base_dir": grid_output_dir,
                                        "topology_type": topology_type,
                                        "grid_size": grid_size,
                                        "compute_qe": False,
                                    },
                                }
                            )

                _run_shared_dataset_group(
                    base_args=base_args,
                    dataset_dimension=base_args.fixed_dimension,
                    dataset_processing_method=processing_method,
                    dataset_num_samples=base_args.samples,
                    dataset_seed=seed,
                    pending_runs=pending_runs,
                    on_success=_record_success,
                    on_failure=_record_failure,
                )

        return results


# Helper functions
def _load_existing_time(log_file: str) -> Optional[float]:
    """Return the training time recorded in the log file, if present."""
    if not log_file or not os.path.exists(log_file):
        return None
    times = _parse_existing_time(log_file, [])
    if times:
        return times[-1]
    return None


def _maybe_skip_completed_run(
    *,
    base_args,
    resume_manager,
    times,
    mode: str,
    topology_type: str,
    processing_method: str,
    axis_name: str,
    axis_value,
    gpu_count,
    repeat_index: int,
    log_file: Optional[str],
    metadata: Dict[str, Any],
    config_label: str,
    config_name: Optional[str] = None,
):
    """
    Determine whether a benchmark configuration has already completed.

    Returns:
        (skip_run, task_key, recorded_time, record)
    """
    resume_enabled = bool(resume_manager and resume_manager.enabled)
    task_key = None
    record = None
    recorded_time = None

    if resume_enabled:
        task_key = resume_manager.build_task_key(
            mode=mode,
            topology=topology_type,
            processing_method=processing_method,
            gpu_count=gpu_count,
            repeat_index=repeat_index,
            axis_name=axis_name,
            axis_value=axis_value,
        )
        record = resume_manager.register_task(task_key, metadata=metadata)

    existing_time = _load_existing_time(log_file) if log_file else None
    if resume_enabled and existing_time is not None and task_key:
        resume_manager.update_train_time_if_missing(task_key, existing_time)

    if resume_enabled and record and record.get("status") == "done":
        recorded_time = record.get("train_time") or existing_time
        if recorded_time is not None:
            if times is not None:
                times.append(recorded_time)
            print(f"Skipping {config_label} (resume: already completed)")
            return True, task_key, recorded_time, record

    if (
        resume_enabled
        and existing_time is not None
        and record
        and record.get("status") in {"pending", "failed", "running"}
    ):
        resume_manager.mark_done(task_key, train_time=existing_time, metadata=metadata)
        record = resume_manager.get_record(task_key)
        if times is not None:
            times.append(existing_time)
        print(f"Skipping {config_label} (resume: imported existing log)")
        return True, task_key, existing_time, record

    if base_args.skip_existing and existing_time is not None:
        if times is not None:
            times.append(existing_time)
        if resume_enabled and task_key:
            already_done = record and record.get("status") == "done" and record.get("train_time") == existing_time
            if not already_done:
                resume_manager.mark_done(task_key, train_time=existing_time, metadata=metadata)
        message_target = config_name or config_label
        print(f"Skipping {message_target} (already exists)")
        return True, task_key, existing_time, record

    return False, task_key, None, record

def _parse_existing_time(log_file: str, times: list) -> list:
    """Parse training time from existing log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('Training time:'):
                    time_str = line.split(':')[1].strip().rstrip('s')
                    times.append(float(time_str))
                    break
    except:
        pass
    return times


