"""
Single benchmark execution pipeline for Optuna optimization with Ray Tune integration.

This module provides the complete pipeline for running a single benchmark
with specific algorithm, dataset, and forced parameter configuration.
Can run either with traditional Optuna or with Ray Tune for distributed execution.
"""

import os
import time
import pickle
from typing import Dict, Any, Optional, List, Union
import optuna
import numpy as np

from .objective import create_objective, get_metrics_config
from .study_manager import StudyManager, create_benchmark_study_name, create_default_sampler_config
from ..config.parameters import validate_forced_params, PARAMETER_CONFIGS, get_conditional_parameters
from ..utils.utils import set_global_seed, save_study_results
from ..utils.ray_tune_utils import run_ray_tune_optimization, is_ray_available
from ..harmonization.pareto_utils import find_best_trial_by_distance
from ....floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig


def _build_default_floatsom_for_context(forced_params: Dict[str, Any]) -> FloatSOMParams:
    """Build a default FloatSOMParams object for the requested optimization context."""
    sampling_method = str(forced_params.get('sampling_method', 'full')).strip().lower()
    processing_method = str(forced_params.get('processing_method', 'batch')).strip().lower()
    topology_type = str(forced_params.get('topology_type', 'grid')).strip().lower()

    return FloatSOMParams(
        sampling_config=SamplingConfig(method=sampling_method),
        processing_config=ProcessingConfig(
            chunk_size=None,
            method=processing_method,
            batch_mode=str(forced_params.get('batch_mode', 'full_batch')).strip().lower(),
            normalization='xpysom',
            use_sparse_influence=False,
        ),
        topology_config=TopologyConfig(topology_type=topology_type),
    )


def get_default_trial_params(forced_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get default parameter values for the first trial based on FloatSOMParams defaults.
    
    Args:
        forced_params: Dictionary of forced parameters for this benchmark
        
    Returns:
        Dictionary of default parameter values to use for the first trial
    """
    # Start with FloatSOM default parameters
    default_floatsom = _build_default_floatsom_for_context(forced_params)
    
    # Create default parameter dictionary
    default_params = {}
    
    # Map FloatSOM defaults to Optuna parameter names
    # Get ALL values from FloatSOMParams defaults
        
    param_mapping = {
        # Core SOM parameters
        'initial_radius': default_floatsom.initial_radius,
        'radius_decay_type': default_floatsom.radius_decay_type,
        'learning_rate': default_floatsom.initial_learning_rate,
        'lr_decay_type': default_floatsom.lr_decay_type,
        'lr_decay_factor': default_floatsom.lr_decay_factor,
        
        # Processing parameters
        'batch_mode': default_floatsom.processing_config.batch_mode,
        'chunk_size': default_floatsom.processing_config.chunk_size,
        'max_rounds': default_floatsom.processing_config.max_rounds,
        'use_momentum': default_floatsom.processing_config.enable_momentum,
        'momentum_init': default_floatsom.processing_config.initial_momentum,
        
        # Normalization parameters (xpysom)
        'normalization': default_floatsom.processing_config.normalization,
        
        # Topology parameters
        'topology_variant': default_floatsom.topology_config.topology_variant,
        'grid_size': default_floatsom.topology_config.grid_size,
        
        # Other parameters
        'initialization_method': default_floatsom.initialization_method,
        'iterations': default_floatsom.total_iterations,
        'convergence_threshold': default_floatsom.convergence_threshold,
        'min_iterations': default_floatsom.min_iterations
    }
    
    
    # Get conditional parameters
    conditional_params = get_conditional_parameters()
    processing_method = forced_params.get('processing_method')
    
    # Add ALL parameters that are not forced (Ray Tune needs all of them)
    for param_name, default_value in param_mapping.items():
        # Skip if parameter is forced
        if param_name in forced_params:
            continue
            
        # Skip if parameter doesn't exist in PARAMETER_CONFIGS
        if param_name not in PARAMETER_CONFIGS:
            continue
            
        config = PARAMETER_CONFIGS[param_name]

        # Skip if parameter is marked as forced in config
        if config.get('forced', False):
            continue

        applicable_methods = config.get('applicable_processing_methods')
        if applicable_methods:
            if processing_method not in set(applicable_methods):
                continue

        # Guard against optional defaults that are unset in FloatSOMParams
        if default_value is None:
            continue

        # For Ray Tune, we need to include ALL parameters in the search space
        # even if they're conditional, so don't skip conditional params

        # Validate that default value is within allowed range/choices
        if config['type'] == 'float':
            min_val, max_val = config['range']
            if min_val <= default_value <= max_val:
                default_params[param_name] = float(default_value)
            else:
                # Use middle of range if default is out of bounds
                default_params[param_name] = (min_val + max_val) / 2
        elif config['type'] == 'categorical':
            if default_value in config['choices']:
                default_params[param_name] = default_value
            else:
                # Use first choice if default is not valid
                default_params[param_name] = config['choices'][0]
        elif config['type'] == 'int':
            min_val, max_val = config['range']
            if min_val <= default_value <= max_val:
                default_params[param_name] = int(default_value)
            else:
                # Use middle of range if default is out of bounds
                default_params[param_name] = (min_val + max_val) // 2
    
    return default_params


def _run_distributed_optimization(
    study_name: str,
    storage_url: str, 
    objective,
    n_trials: int,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run distributed optimization on a GPU worker.
    This function runs on the dask-cuda worker nodes.
    """
    import optuna
    
    # Load the study on this worker
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Run optimization with allocated trials
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),
        gc_after_trial=True,
        show_progress_bar=False  # Disable progress bar for workers
    )
    
    return {
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'worker_id': id(study)  # Simple worker identification
    }

def _run_optuna_optimization(
    algo_type: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    forced_params: Dict[str, Any],
    objectives: List[str],
    metrics_config: Dict[str, Any],
    seed: int,
    n_trials: int,
    timeout: Optional[float],
    study_name: str,
    output_dir: Optional[str]
) -> optuna.Study:
    """
    Run traditional Optuna optimization.
    
    This function contains the original Optuna execution logic.
    """
    # Determine if this is multi-objective optimization
    multi_objective = len(objectives) > 1

    # Create study manager
    study_manager = StudyManager()
    
    # Configure sampler and directions based on optimization type
    def _base_metric_name(metric_name: str) -> str:
        if metric_name.endswith('_holdout'):
            return metric_name[:-8]
        if metric_name.endswith('_train'):
            return metric_name[:-6]
        return metric_name

    if multi_objective:
        # Multi-objective optimization using NSGA-II
        num_objectives = len(objectives)

        # Map objectives to their optimization directions
        # Metrics that should be maximized (higher is better)
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}

        directions = []
        for obj in objectives:
            base_obj = _base_metric_name(obj)
            if base_obj in maximize_metrics:
                directions.append('maximize')
            else:
                directions.append('minimize')
        
        sampler_config = {'seed': seed}  
        print(f"  Multi-objective optimization with {num_objectives} objectives")
        print(f"  Directions: {list(zip(objectives, directions))}")
    else:
        # Single objective optimization using TPE
        base_obj = _base_metric_name(objectives[0])
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
        directions = ['maximize'] if base_obj in maximize_metrics else ['minimize']
        sampler_config = create_default_sampler_config(seed)
    
    # Create study
    study = study_manager.create_study(
        study_name=study_name,
        directions=directions,
        sampler_config=sampler_config,
        load_if_exists=False
    )

    try:
        study.set_user_attr('objectives', objectives)
        study.set_user_attr('base_objectives', metrics_config.get('base_objectives', objectives))
    except Exception:
        pass
    
    # Enqueue the first trial with default parameters
    default_params = get_default_trial_params(forced_params)
    if default_params:
        study.enqueue_trial(default_params)
        print(f"  Enqueued first trial with default parameters from FloatSOMParams")
        print(f"  Default params: {default_params}")
    
    # Create objective function
    objective = create_objective(
        algo_type=algo_type,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        forced_params=forced_params,
        metrics_config=metrics_config,
        seed=seed
    )
    
    # Run optimization
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),  # Continue on individual trial failures
        gc_after_trial=True,  # Garbage collect after each trial
        show_progress_bar=True
    )
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\nBenchmark completed in {total_time:.2f}s")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if multi_objective:
        # Multi-objective results (Pareto front)
        if hasattr(study, 'best_trials') and study.best_trials:
            print(f"  Pareto front solutions: {len(study.best_trials)}")
            
            # Find best trial by distance to origin
            best_trial, min_distance = find_best_trial_by_distance(study.best_trials, study.directions)
            if min_distance is not None and not np.isnan(min_distance):
                print(f"  Best trial by distance to origin: {min_distance:.6f}")
                # Store distance information in study user attributes
                study.set_user_attr('best_trial_distance', min_distance)
                study.set_user_attr('best_trial_number', best_trial.number)
            else:
                print(f"  Best trial by distance to origin: None")
                study.set_user_attr('best_trial_distance', None)
                study.set_user_attr('best_trial_number', None)
            
            print("  Best solutions on Pareto front:")
            
            # Show first 5 solutions with all objective values
            for i, trial in enumerate(study.best_trials[:5]):
                if len(trial.values) == len(objectives):
                    obj_values = ", ".join(f"{name}={val:.6f}" for name, val in zip(objectives, trial.values))
                    distance_marker = " (BEST)" if trial.number == best_trial.number else ""
                    print(f"    Solution {i+1}: {obj_values}{distance_marker}")
                    # Show top 3 most important parameters
                    param_items = list(trial.params.items())[:3]
                    param_str = ", ".join(f"{k}={v}" for k, v in param_items)
                    if len(trial.params) > 3:
                        param_str += f" (+{len(trial.params)-3} more)"
                    print(f"      Key params: {param_str}")
                else:
                    print(f"    Solution {i+1}: {trial.values} (mismatch in objective count)")
        else:
            print("  No Pareto front solutions found")
    else:
        # Single objective results
        if study.best_trial:
            print(f"  Best {objectives[0]}: {study.best_value:.6f}")
            print(f"  Best params: {study.best_params}")
        else:
            print("  No best trial found")
    
    # Save results
    if output_dir:
        save_study_results(study, study_name, output_dir)
        print(f"  Results saved to: {output_dir}")
    
    return study


def run_single_benchmark(
    algo_type: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    forced_params: Dict[str, Any],
    seed: int = 42,
    n_trials: int = 100,
    timeout: Optional[float] = None,
    objectives: Optional[List[str]] = None,
    evaluation_split: str = 'both',
    output_dir: Optional[str] = None,
    use_ray_tune: bool = False,
    gpu_fraction: float = 0.3,
    max_concurrent: Optional[int] = None,
) -> Union[optuna.Study, Any]:
    """
    Run a single benchmark with specific configuration.
    
    Args:
        algo_type: Algorithm variant ('colors', 'batch')
        dataset_name: Name of dataset ('swiss_roll', 'moons', etc.)
        dataset_config: Dataset configuration parameters
        forced_params: Dict of parameters that are fixed for this run
        seed: Random seed for reproducibility
        n_trials: Number of optimization trials to run
        timeout: Maximum time in seconds for optimization (None for no limit)
        objectives: List of objective metrics to optimize. If None or single objective,
                   uses single-objective TPE. If multiple objectives, uses NSGA-II.
        evaluation_split: Which dataset split(s) to optimise ('both', 'holdout', 'train')
        output_dir: Directory to save results (None for default)
        use_ray_tune: Whether to use Ray Tune for distributed execution (default: False)
        gpu_fraction: GPU fraction per trial when using Ray Tune (default: 0.3)
        max_concurrent: Maximum concurrent trials when using Ray Tune (None for auto-detection)
        
    Returns:
        Completed study (Optuna Study or Ray Tune ResultGrid)
    """
    
    # Validate inputs
    validate_forced_params(forced_params)
    
    # Set default objectives if none provided
    if objectives is None:
        objectives = ['quantization_error']

    base_objectives = list(objectives)

    # Create metrics configuration according to requested evaluation split
    include_train_objectives = evaluation_split in ('both', 'train')
    metrics_config = get_metrics_config(
        objectives=base_objectives,
        include_train_objectives=include_train_objectives,
        evaluation_split=evaluation_split
    )
    expanded_objectives = metrics_config['objectives']

    # Set global random seed
    set_global_seed(seed)

    # Create unique study name
    study_name = create_benchmark_study_name(algo_type, dataset_name, forced_params, seed)

    # Determine if this is multi-objective optimization
    multi_objective = len(expanded_objectives) > 1

    print(f"Starting benchmark: {study_name}")
    print(f"  Algorithm: {algo_type}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Objectives (base): {base_objectives}")
    print(f"  Optimizing metrics: {expanded_objectives} ({'multi-objective' if multi_objective else 'single-objective'})")
    print(f"  Evaluation split: {evaluation_split}")
    print(f"  Forced params: {forced_params}")
    print(f"  Seed: {seed}")
    print(f"  Trials: {n_trials}")

    # Choose execution method
    if use_ray_tune and is_ray_available():
        # Use Ray Tune for distributed execution
        return run_ray_tune_optimization(
            algo_type=algo_type,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            forced_params=forced_params,
            objectives=expanded_objectives,
            metrics_config=metrics_config,
            seed=seed,
            n_trials=n_trials,
            study_name=study_name,
            gpu_fraction=gpu_fraction,
            max_concurrent=max_concurrent,
            output_dir=output_dir
        )
    else:
        # Use traditional Optuna execution
        if use_ray_tune and not is_ray_available():
            print("  Warning: Ray Tune requested but not available. Falling back to Optuna.")
        
        return _run_optuna_optimization(
            algo_type=algo_type,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            forced_params=forced_params,
            objectives=expanded_objectives,
            metrics_config=metrics_config,
            seed=seed,
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name,
            output_dir=output_dir
        )


