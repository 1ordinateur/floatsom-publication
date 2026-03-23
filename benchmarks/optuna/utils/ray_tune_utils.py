"""
Ray Tune utilities for distributed hyperparameter optimization.

This module provides Ray Tune integration for the Optuna benchmarking system,
enabling distributed execution with GPU resource management.
"""

import os
import time
import pickle
import shutil
import numpy as np
from typing import Dict, Any, Optional, List, Union, Set

# Ray Tune imports
try:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter
    import optuna
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..config.parameters import get_optimizable_parameters, get_discrete_parameters


TUNABLE_PARAM_NAMES: Set[str] = set(get_optimizable_parameters()) | set(get_discrete_parameters())
NON_OBJECTIVE_ATTR_KEYS: Set[str] = {
    'train_time',
    'iterations_completed',
    'n_train_samples',
    'n_test_samples',
    'train_test_split',
    'timestamp',
    'done',
    'training_iteration',
    'time_this_iter_s',
    'time_total_s',
    'time_since_restore',
    'iterations_since_restore',
    'pid',
    'experiment_id',
    'trial_id',
    'hostname',
}


def get_penalty_value(objective_name: str) -> float:
    """
    Get appropriate penalty value for a failed objective.
    
    Args:
        objective_name: Name of the objective metric
        
    Returns:
        Large finite penalty value appropriate for the objective
    """
    # Define which objectives should be maximized (higher is better)
    maximize_objectives = {'trustworthiness', 'neighborhood_preservation'}
    
    # For minimization objectives: use large positive value
    # For maximization objectives: use large negative value
    if objective_name in maximize_objectives:
        return -1e10  # Large negative penalty for maximization
    else:
        return 1e10   # Large positive penalty for minimization


def _create_ray_tune_search_space(algo_type: str, forced_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parameter configs to a Ray Tune search space."""
    from ..config.parameters import (
        get_optimizable_parameters,
        PARAMETER_CONFIGS,
    )

    search_space: Dict[str, Any] = {}

    # Use all optimizable params for batch/colors algorithms
    optimizable_params = get_optimizable_parameters()

    for name in optimizable_params:
        if name in forced_params:
            continue
        if name not in PARAMETER_CONFIGS:
            continue
        cfg = PARAMETER_CONFIGS[name]
        if cfg.get('forced', False):
            continue

        if cfg['type'] == 'float':
            low, high = cfg['range']
            search_space[name] = tune.loguniform(low, high) if cfg.get('log', False) else tune.uniform(low, high)
        elif cfg['type'] == 'int':
            low, high = cfg['range']
            search_space[name] = tune.randint(low, high + 1)
        elif cfg['type'] == 'categorical':
            search_space[name] = tune.choice(cfg['choices'])

    return search_space


class MockOptunaTrialForRayTune:
    """Mock Optuna trial interface for Ray Tune compatibility."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.user_attrs = {}
        self.number = 0
    
    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Suggest float value - returns pre-sampled value from Ray Tune."""
        return self.params.get(name, (low + high) / 2)
    
    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Suggest int value - returns pre-sampled value from Ray Tune."""
        return self.params.get(name, (low + high) // 2)
    
    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        """Suggest categorical value - returns pre-sampled value from Ray Tune."""
        return self.params.get(name, choices[0])
    
    def set_user_attr(self, key: str, value: Any) -> None:
        """Set user attribute."""
        self.user_attrs[key] = value
    
    def report(self, value: float, step: int) -> None:
        """Report intermediate value - not used in our single-shot approach."""
        pass
    
    def should_prune(self) -> bool:
        """Check if trial should be pruned - always False for our use case."""
        return False


def _ray_tune_objective(config: Dict[str, Any]) -> None:
    """Ray Tune trainable function that wraps the Optuna objective."""
    # Extract factory parameters from config
    algo_type = config['_algo_type']
    dataset_name = config['_dataset_name']
    dataset_config = config['_dataset_config']
    forced_params_base = config['_forced_params']
    metrics_config = config['_metrics_config']
    seed = config['_seed']
    objectives = config['_objectives']
    
    # Merge tuned parameters with forced params
    tuned_params = {k: v for k, v in config.items() if not k.startswith('_')}
    all_params = {**forced_params_base, **tuned_params}
    
    # Create objective function
    from ..core.objective import create_objective
    objective = create_objective(
        algo_type=algo_type,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        forced_params=all_params,
        metrics_config=metrics_config,
        seed=seed
    )
    
    # Create mock trial
    mock_trial = MockOptunaTrialForRayTune(all_params)
    
    # Run objective and get results
    try:
        result = objective(mock_trial)
        
        # Collect additional numeric metrics surfaced via user attributes (e.g., train metrics)
        extra_metrics = {}
        for key, value in mock_trial.user_attrs.items():
            if key in objectives or key in ('metrics_holdout', 'metrics_train'):
                continue
            if key in all_params or key in TUNABLE_PARAM_NAMES:
                continue
            if key in NON_OBJECTIVE_ATTR_KEYS:
                continue
            if key in metrics_config.get('metrics', []):
                continue
            if isinstance(value, (int, float)):
                extra_metrics[key] = float(value)
            elif hasattr(value, 'item'):
                try:
                    extra_metrics[key] = float(value.item())
                except Exception:
                    continue

        # Report results to Ray Tune
        if isinstance(result, (list, tuple)):
            # Multi-objective - sanitize values to ensure they're finite
            sanitized_metrics = {}
            for obj, val in zip(objectives, result):
                if np.isnan(val) or np.isinf(val):
                    print(f"Warning: Objective '{obj}' returned inf/NaN, replacing with penalty")
                    sanitized_metrics[obj] = get_penalty_value(obj)
                else:
                    sanitized_metrics[obj] = val
            metrics = sanitized_metrics
        else:
            # Single objective - sanitize value
            if np.isnan(result) or np.isinf(result):
                print(f"Warning: Objective '{objectives[0]}' returned inf/NaN, replacing with penalty")
                metrics = {objectives[0]: get_penalty_value(objectives[0])}
            else:
                metrics = {objectives[0]: result}

        # Merge sanitized objective metrics with any additional metrics we captured
        combined_metrics = {**metrics, **extra_metrics}
        ray.tune.report(metrics=combined_metrics)
    except Exception as e:
        # Report failure - use large finite penalty values for failed trials
        print(f"Trial failed with error: {str(e)}")
        failure_metrics = {obj: get_penalty_value(obj) for obj in objectives}
        ray.tune.report(metrics=failure_metrics)


def run_ray_tune_optimization(
    algo_type: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    forced_params: Dict[str, Any],
    objectives: List[str],
    metrics_config: Dict[str, Any],
    seed: int,
    n_trials: int,
    study_name: str,
    gpu_fraction: float = 0.3,
    max_concurrent: Optional[int] = None,
    output_dir: Optional[str] = None
) -> 'ray.tune.ResultGrid':
    """
    Run Ray Tune optimization with the specified configuration.
    
    Args:
        algo_type: Algorithm variant ('colors', 'batch')
        dataset_name: Name of dataset
        dataset_config: Dataset configuration parameters
        forced_params: Dict of parameters that are fixed for this run
        objectives: List of objective metrics to optimize
        metrics_config: Metrics configuration
        seed: Random seed for reproducibility
        n_trials: Number of optimization trials to run
        study_name: Unique study name for this benchmark
        gpu_fraction: GPU fraction per trial (default: 0.3)
        max_concurrent: Maximum concurrent trials (None for auto-detection based on available resources)
        output_dir: Directory to save results (None for no saving)
        
    Returns:
        Ray Tune ResultGrid object
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not available. Cannot use Ray Tune optimization.")
    
    # Initialize Ray if not already initialized
    ray_was_initialized = ray.is_initialized()
    ray_initialized_here = False
    if not ray_was_initialized:
        try:
            # Try to connect to existing cluster first
            ray.init(address='auto', ignore_reinit_error=True)
            print("Connected to existing Ray cluster")
            ray_initialized_here = True
        except Exception:
            # Fall back to local Ray
            ray.init(ignore_reinit_error=True)
            print("Started local Ray instance")
            ray_initialized_here = True
    
    # Determine if multi-objective
    multi_objective = len(objectives) > 1
    
    # Auto-detect max concurrent if not specified
    if max_concurrent is None:
        resources = ray.available_resources()
        available_gpus = resources.get('GPU', 0)
        available_cpus = resources.get('CPU', 0)
        
        if available_gpus > 0 and gpu_fraction > 0:
            # Calculate based on GPU resources
            trials_per_gpu = max(1, int(1.0 / gpu_fraction))
            max_concurrent = int(available_gpus * trials_per_gpu)
            print(f"  Auto-detected max_concurrent: {max_concurrent} (based on {available_gpus} GPUs with {gpu_fraction} fraction per trial)")
        else:
            # CPU resource mode - use available CPUs
            max_concurrent = max(1, available_cpus)
            print(f"  Auto-detected max_concurrent: {max_concurrent} (CPU resource mode with {available_cpus} CPUs)")
    
    print(f"  Using Ray Tune with {gpu_fraction} GPU per trial, max {max_concurrent} concurrent")
    
    # Create search space for hyperparameters only
    hyperparam_search_space = _create_ray_tune_search_space(algo_type, forced_params)
    
    # Create factory parameters dict
    factory_params = {
        '_algo_type': algo_type,
        '_dataset_name': dataset_name,
        '_dataset_config': dataset_config,
        '_forced_params': forced_params,
        '_metrics_config': metrics_config,
        '_seed': seed,
        '_objectives': objectives,
        '_base_objectives': metrics_config.get('base_objectives', objectives)
    }
    
    # Merge factory params with search space for param_space
    combined_param_space = {**factory_params, **hyperparam_search_space}
    
    # Get default parameters for the first trial
    from ..core.single_benchmark import get_default_trial_params
    default_params = get_default_trial_params(forced_params)
    
    # Prepare points_to_evaluate with default parameters
    points_to_evaluate = []
    if default_params:
        # Only include parameters that are in the search space
        default_point = {k: v for k, v in default_params.items() if k in hyperparam_search_space}
        # Only warm-start if dimensions match exactly to avoid Ray/Optuna mismatch
        if set(default_point.keys()) == set(hyperparam_search_space.keys()) and len(default_point) > 0:
            points_to_evaluate = [default_point]
            print(f"  Starting with default parameters from FloatSOMParams")
            print(f"  Default params: {default_point}")
        else:
            # Skip warm-start to prevent validate_warmstart dimension errors
            if len(hyperparam_search_space) > 0:
                print("  Skipping warm-start: default params do not cover entire search space")
                print(f"  Space keys: {sorted(list(hyperparam_search_space.keys()))}")
                print(f"  Default keys: {sorted(list(default_point.keys()))}")
    
    # Configure OptunaSearch WITHOUT the search space (will use param_space instead)
    def _base_metric_name(metric_name: str) -> str:
        if metric_name.endswith('_holdout'):
            return metric_name[:-8]
        if metric_name.endswith('_train'):
            return metric_name[:-6]
        return metric_name

    if multi_objective:
        # Multi-objective optimization
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
        modes = ['max' if _base_metric_name(obj) in maximize_metrics else 'min' 
                for obj in objectives]

        searcher = OptunaSearch(
            # Don't pass space here - Ray Tune will handle it via param_space
            metric=objectives,
            mode=modes,
            seed=seed,
            points_to_evaluate=points_to_evaluate  # Start with default parameters
        )
    else:
        # Single objective optimization
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
        mode = 'max' if _base_metric_name(objectives[0]) in maximize_metrics else 'min'

        searcher = OptunaSearch(
            # Don't pass space here - Ray Tune will handle it via param_space
            metric=objectives[0],
            mode=mode,
            seed=seed,
            points_to_evaluate=points_to_evaluate  # Start with default parameters
        )
    
    # Auto-calculate max concurrent trials and CPU allocation based on GPU resources
    cpus_per_trial = 1  # Default CPU allocation
    
    if gpu_fraction > 0:
        # Check if Ray is initialized and get available resources
        if ray.is_initialized():
            resources = ray.available_resources()
            available_gpus = resources.get('GPU', 0)
            available_cpus = resources.get('CPU', 0)
            
            if available_gpus > 0:
                # Calculate total concurrent trials across all GPUs
                # Each GPU can handle floor(1/gpu_fraction) trials
                trials_per_gpu = max(1, int(1.0 / gpu_fraction))
                calculated_concurrent = int(available_gpus * trials_per_gpu)
                # Use the calculated concurrent value (max_concurrent already set above)
                actual_concurrent = max_concurrent
                
                # Calculate CPUs per trial based on total available CPUs and concurrent trials
                if available_cpus > 0 and actual_concurrent > 0:
                    # Divide available CPUs evenly among concurrent trials
                    # Reserve some CPUs for Ray system overhead (10% or at least 1 CPU)
                    overhead_cpus = max(1, int(available_cpus * 0.1))
                    usable_cpus = available_cpus - overhead_cpus
                    cpus_per_trial = max(1, int(usable_cpus / actual_concurrent))
                    
                    print(f"  Auto-calculated concurrency: {trials_per_gpu} trials/GPU × {available_gpus} GPUs = {calculated_concurrent} total")
                    print(f"  Using max_concurrent: {actual_concurrent}")
                    print(f"  CPU allocation: {available_cpus} total CPUs - {overhead_cpus} overhead = {usable_cpus} usable")
                    print(f"  CPUs per trial: {cpus_per_trial} ({usable_cpus} CPUs ÷ {actual_concurrent} concurrent trials)")
                else:
                    print(f"  Auto-calculated concurrency: {trials_per_gpu} trials/GPU × {available_gpus} GPUs = {calculated_concurrent} total")
                    print(f"  Using max_concurrent: {actual_concurrent}")
            else:
                actual_concurrent = max_concurrent
                # CPU resource mode - use available CPUs
                if available_cpus > 0 and actual_concurrent > 0:
                    overhead_cpus = max(1, int(available_cpus * 0.1))
                    usable_cpus = available_cpus - overhead_cpus
                    cpus_per_trial = max(1, int(usable_cpus / actual_concurrent))
                    print(f"  CPU resource mode: {cpus_per_trial} CPUs per trial")
        else:
            actual_concurrent = max_concurrent
    else:
        actual_concurrent = max_concurrent
        # CPU resource mode without Ray initialized
        if ray.is_initialized():
            resources = ray.available_resources()
            available_cpus = resources.get('CPU', 0)
            if available_cpus > 0 and actual_concurrent > 0:
                overhead_cpus = max(1, int(available_cpus * 0.1))
                usable_cpus = available_cpus - overhead_cpus
                cpus_per_trial = max(1, int(usable_cpus / actual_concurrent))
                print(f"  CPU resource mode: {cpus_per_trial} CPUs per trial")
    
    # Apply concurrency limiter
    algo = ConcurrencyLimiter(searcher, max_concurrent=actual_concurrent)
    
    # Configure resources with dynamic CPU allocation
    resources = {"cpu": cpus_per_trial, "gpu": gpu_fraction}
    
    # Create and run tuner with minimal logging
    import tempfile
    storage_path = tempfile.mkdtemp(prefix="ray_tune_")
    storage_path = os.path.abspath(storage_path)
    
    try:
        tuner = tune.Tuner(
            tune.with_resources(_ray_tune_objective, resources=resources),
            tune_config=tune.TuneConfig(
                search_alg=algo,
                num_samples=n_trials,
                metric=objectives[0] if not multi_objective else None,
                mode=mode if not multi_objective else None,
            ),
            run_config=ray.tune.RunConfig(
                storage_path=storage_path,
                name=study_name,
                checkpoint_config=None,  # Disable checkpointing entirely
                log_to_file=False,  # Disable trial log files
                verbose=0,  # Minimize verbosity
            ),
            param_space=combined_param_space,  # Pass both factory params and search distributions
        )
    
        # Run optimization
        start_time = time.time()
        results = tuner.fit()
        total_time = time.time() - start_time
        
        # Process and print results
        print(f"\nBenchmark completed in {total_time:.2f}s")
        
        try:
            if multi_objective:
                # Get best results for multi-objective
                best_results = []
                for result in results:
                    if all(obj in result.metrics for obj in objectives):
                        # Check if any metrics are inf/NaN
                        has_invalid = any(
                            np.isnan(result.metrics[obj]) or np.isinf(result.metrics[obj]) 
                            for obj in objectives
                        )
                        if not has_invalid:
                            best_results.append(result)
                        else:
                            print(f"Warning: Skipping result with inf/NaN values")
                
                print(f"  Completed trials: {len(best_results)}")
                print(f"  Failed trials: {len(results) - len(best_results)}")
                
                if best_results:
                    print("  Best solutions found:")
                    # Sort by first objective for display
                    best_results.sort(key=lambda r: r.metrics[objectives[0]])
                    
                    # Display best solutions
                    for i, result in enumerate(best_results[:5]):
                        obj_values = ", ".join(f"{obj}={result.metrics[obj]:.6f}" 
                                             for obj in objectives)
                        print(f"    Solution {i+1}: {obj_values}")
                        
                        # Show key parameters
                        config_items = [(k, v) for k, v in result.config.items() 
                                      if not k.startswith('_')][:3]
                        param_str = ", ".join(f"{k}={v}" for k, v in config_items)
                        if len([k for k in result.config.keys() if not k.startswith('_')]) > 3:
                            remaining = len([k for k in result.config.keys() if not k.startswith('_')]) - 3
                            param_str += f" (+{remaining} more)"
                        print(f"      Key params: {param_str}")
                else:
                    print("  No successful trials found")
            else:
                # Single objective
                try:
                    maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
                    mode = 'max' if _base_metric_name(objectives[0]) in maximize_metrics else 'min'

                    best_result = results.get_best_result(objectives[0], mode)
                    completed_trials = len([r for r in results if objectives[0] in r.metrics])
                    failed_trials = len(results) - completed_trials

                    print(f"  Completed trials: {completed_trials}")
                    print(f"  Failed trials: {failed_trials}")

                    if best_result and objectives[0] in best_result.metrics:
                        print(f"  Best {objectives[0]}: {best_result.metrics[objectives[0]]:.6f}")
                        config_items = [(k, v) for k, v in best_result.config.items() 
                                      if not k.startswith('_')]
                        print(f"  Best params: {dict(config_items)}")
                    else:
                        print("  No best trial found")
                except Exception as e:
                    print(f"  Error getting best result: {str(e)}")
                    print(f"  Total trials run: {len(results)}")
        except Exception as e:
            print(f"  Error processing results: {str(e)}")
            print(f"  Total trials attempted: {len(results) if results else 0}")
        
        # Save results if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, f"{study_name}_raytune.pkl")
            try:
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"  Results saved to: {results_path}")
            except Exception as e:
                print(f"  Warning: Could not save results: {str(e)}")
        
        return results
    finally:
        # Remove Ray Tune storage artefacts
        if storage_path and os.path.isdir(storage_path):
            shutil.rmtree(storage_path, ignore_errors=True)
        
        # Shut down Ray if we started it in this helper
        if ray_initialized_here and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                pass


def is_ray_available() -> bool:
    """Check if Ray Tune is available."""
    return RAY_AVAILABLE
