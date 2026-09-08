#!/usr/bin/env python3
"""
End-to-end FloatSOM Optuna benchmarking pipeline.

This script provides two modes:
- Single: Run a single benchmark with specified parameters
- Full: Run all scenarios sequentially for a given seed

Usage:
    python run_optuna.py --mode single --dataset swiss_roll --algo batch --trials 50
    python run_optuna.py --mode full --config development --seed 42 --output-dir results
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import json
import optuna

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from .core.single_benchmark import run_single_benchmark
from .core.objective import get_metrics_config
from .config.benchmark_config import create_phase3_development_config, create_phase3_full_config
from .config.parameters import (
    get_default_forced_combinations,
    get_discrete_parameters,
    get_optimizable_parameters,
)
from .harmonization.pareto_utils import calculate_pareto_front_from_trials, calculate_pareto_front_from_ray_results
from floatsom.floatsom_params import FloatSOMParams


TUNABLE_HYPERPARAMETER_NAMES: Set[str] = set(get_optimizable_parameters()) | set(get_discrete_parameters())
NON_OBJECTIVE_ATTRIBUTE_KEYS: Set[str] = {
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


def get_default_params() -> FloatSOMParams:
    """Get default FloatSOM parameters from floatsom_params.py"""
    return FloatSOMParams(defaults_profile="publication", )


def resolve_dataset_config() -> Dict[str, Any]:
    """Get dataset configuration from default params"""
    default_params = get_default_params()
    return {
        'difficulty': 'hard',  # Use hard datasets for challenging benchmarks
        'normalize': True
    }


def resolve_default_objectives() -> List[str]:
    """Get default objectives - could be moved to floatsom_params in future"""
    return ['quantization_error']


def resolve_default_seed() -> int:
    """Get default seed from floatsom_params"""
    default_params = get_default_params()
    return default_params.seed if default_params.seed is not None else 42


def resolve_default_trials() -> int:
    """Get default number of Optuna trials."""
    return 200


def resolve_default_random_target_proportion() -> Optional[float]:
    """Optional random-sampling target proportion override."""
    return None


def resolve_default_evaluation_split() -> str:
    """Default evaluation split for optimization."""
    return 'both'


def resolve_default_gpu() -> int:
    """Get default GPU setting"""
    default_params = get_default_params()
    return 0 if default_params.use_gpu else -1  # -1 could indicate CPU mode




def resolve_default_dataset() -> str:
    """Get default dataset name"""
    return 'swiss_roll'  # This could be moved to floatsom_params in future


def resolve_default_algo() -> str:
    """Get default algorithm from floatsom_params"""
    default_params = get_default_params()
    return default_params.processing_config.method


def resolve_default_config_mode() -> str:
    """Get default configuration mode for full runs"""
    return 'development'  # This could be moved to floatsom_params in future




def expand_objectives_for_split(base_objectives: List[str], evaluation_split: str) -> List[str]:
    """
    Expand objective names according to the evaluation split, mirroring the optimisation logic.
    """
    metrics_config = get_metrics_config(
        objectives=base_objectives,
        evaluation_split=evaluation_split
    )
    return metrics_config['objectives']




def get_large_run_warning_threshold() -> int:
    """Get threshold for warning about large runs"""
    # This could be moved to floatsom_params in future
    return 10000


def get_scenario_display_limit() -> int:
    """Get limit for displaying sample scenarios"""
    # This could be moved to floatsom_params in future
    return 5


def create_timestamped_output_dir(base_dir: str, run_name: str) -> str:
    """
    Create a timestamped subdirectory for benchmark outputs.
    
    Args:
        base_dir: Base output directory
        run_name: Name identifier for this run
        
    Returns:
        Path to the timestamped subdirectory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"{run_name}_{timestamp}"
    output_path = Path(base_dir) / timestamped_name
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def run_single_mode(args) -> None:
    """Run a single benchmark execution."""
    print("=" * 60)
    print("SINGLE BENCHMARK EXECUTION")
    print("=" * 60)

    base_objectives = list(args.objectives)
    expanded_objectives = expand_objectives_for_split(base_objectives, args.evaluation_split)

    # Create timestamped output directory
    if args.output_dir:
        run_name = f"single_{args.dataset}_{args.algo}"
        output_dir = create_timestamped_output_dir(args.output_dir, run_name)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Get forced parameter combination
    forced_combinations = get_default_forced_combinations()
    forced_params = None
    
    # Find matching combination or use first
    for combo in forced_combinations:
        if combo['processing_method'] == args.algo:
            forced_params = combo
            break
    
    if forced_params is None:
        forced_params = forced_combinations[0]
        print(f"Warning: Algorithm {args.algo} not found, using {forced_params['processing_method']}")
    
    # Dataset configuration from defaults
    dataset_config = resolve_dataset_config()
    
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset} ({dataset_config['difficulty']})")
    print(f"  Algorithm: {forced_params['processing_method']}")
    print(f"  Sampling: {forced_params['sampling_method']}")
    print(f"  Topology: {forced_params['topology_type']}")
    print(f"  Objectives (base): {base_objectives}")
    print(f"  Optimizing metrics: {expanded_objectives}")
    print(f"  Trials: {args.trials}")
    print(f"  Evaluation split: {args.evaluation_split}")
    default_params = get_default_params()
    gpu_status = "GPU" if default_params.use_gpu else "CPU"
    print(f"  Compute: {gpu_status}")
    print(f"  Seed: {args.seed}")
    print(f"  Categorical Parameters:")
    print(f"    - processing_method: {forced_params['processing_method']}")
    print(f"    - sampling_method: {forced_params['sampling_method']}")
    print(f"    - topology_type: {forced_params['topology_type']}")
    # Add any other discrete parameters that might be present
    discrete_params = get_discrete_parameters()
    for param_name in discrete_params:
        if param_name in forced_params and param_name not in ['processing_method', 'sampling_method', 'topology_type']:
            print(f"    - {param_name}: {forced_params[param_name]}")
    print(f"  Note: First trial will use default parameters from FloatSOMParams")
    print()
    
    # Run single benchmark directly
    study = run_single_benchmark(
        algo_type=forced_params['processing_method'],
        dataset_name=args.dataset,
        dataset_config=dataset_config,
        forced_params=forced_params,
        seed=args.seed,
        n_trials=args.trials,
        objectives=base_objectives,
        evaluation_split=args.evaluation_split,
        output_dir=output_dir,
        use_ray_tune=False,
        max_concurrent=args.max_concurrent
    )

    # Save results if output directory specified
    if output_dir and study:
        save_study_json(
            study,
            output_dir,
            args.dataset,
            forced_params,
            objectives=expanded_objectives,
            expected_trials=args.trials,
        )
    
    # Results summary
    print("RESULTS:")
    print("-" * 30)
    if study:
        is_ray_results = hasattr(study, '__class__') and ('ResultGrid' in str(study.__class__) or 'ray.tune' in str(study.__class__))
        if is_ray_results:
            # Ray Tune results
            valid_count = len([r for r in study if all(obj in r.metrics for obj in expanded_objectives)])
            print(f"Valid trials found: {valid_count}")
        elif hasattr(study, 'best_trials') and study.best_trials:
            print(f"Best trials found: {len(study.best_trials)}")
            if getattr(study, "n_objectives", 1) == 1:
                if hasattr(study, 'best_trial') and study.best_trial:
                    print(f"Best value: {study.best_value}")


def save_study_json(
    study,
    output_dir: str,
    dataset_name: str,
    forced_params: Dict[str, Any],
    objectives: List[str] = None,
    expected_trials: Optional[int] = None,
) -> None:
    """Save Optuna study or Ray Tune ResultGrid as JSON for later harmonization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create scenario identifier for matching across seeds
    scenario_id = create_scenario_id(dataset_name, forced_params)

    # Check if this is Ray Tune ResultGrid
    is_ray_results = hasattr(study, '__class__') and ('ResultGrid' in str(study.__class__) or 'ray.tune' in str(study.__class__))

    def _coerce_json_value(value: Any) -> Any:
        """Convert values to JSON-serializable primitives."""
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return [_coerce_json_value(v) for v in value]
        if isinstance(value, dict):
            return {k: _coerce_json_value(v) for k, v in value.items()}
        return str(value)

    def _serialize_user_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        return {key: _coerce_json_value(val) for key, val in attrs.items()}

    def _extract_numeric(attrs: Dict[str, Any]) -> Dict[str, float]:
        numeric_attrs: Dict[str, float] = {}
        for key, val in attrs.items():
            coerced = _coerce_json_value(val)
            if isinstance(coerced, (int, float)):
                numeric_attrs[key] = coerced
        return numeric_attrs

    def _filter_numeric_metrics(
        numeric: Dict[str, float],
        param_keys: Set[str]
    ) -> Dict[str, float]:
        filtered: Dict[str, float] = {}
        for key, value in numeric.items():
            if key in param_keys:
                continue
            if key in TUNABLE_HYPERPARAMETER_NAMES:
                continue
            if key in NON_OBJECTIVE_ATTRIBUTE_KEYS:
                continue
            filtered[key] = value
        return filtered

    def _base_metric_name(metric_name: str) -> str:
        if metric_name.endswith('_holdout'):
            return metric_name[:-8]
        if metric_name.endswith('_train'):
            return metric_name[:-6]
        return metric_name

    base_objectives = []
    if objectives:
        seen = set()
        for obj in objectives:
            base = _base_metric_name(obj)
            if base not in seen:
                base_objectives.append(base)
                seen.add(base)

    # Initialize study data
    study_data = {
        'scenario_id': scenario_id,
        'dataset': dataset_name,
        'forced_params': forced_params,
        'objectives': objectives if objectives else ['objective_1'],
        'base_objectives': base_objectives if base_objectives else (objectives if objectives else ['objective_1']),
        'best_trials': [],
        'all_trials': [],
        'directions': ['MINIMIZE'] * len(objectives) if objectives else ['MINIMIZE']
    }

    # Set proper directions based on objectives
    if objectives:
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
        study_data['directions'] = ['MAXIMIZE' if _base_metric_name(obj) in maximize_metrics else 'MINIMIZE' for obj in objectives]
    
    if is_ray_results:
        # Handle Ray Tune ResultGrid
        try:
            # Use the utility function to calculate Pareto front from Ray results
            pareto_trials = calculate_pareto_front_from_ray_results(
                study, 
                objectives, 
                study_data['directions']
            )
            study_data['best_trials'] = pareto_trials
            
            # Also extract all valid trials for completeness
            all_trials = []
            for i, result in enumerate(study):
                if objectives and all(obj in result.metrics for obj in objectives):
                    trial_data = {
                        'number': i,
                        'values': [result.metrics[obj] for obj in objectives],
                        'params': dict((k, v) for k, v in result.config.items() if not k.startswith('_')),
                        'state': 'COMPLETE'
                    }
                    metrics_dict = getattr(result, 'metrics', None)
                    if isinstance(metrics_dict, dict):
                        serialized_metrics = _serialize_user_attrs(metrics_dict)
                        numeric_metrics = _extract_numeric(serialized_metrics)
                        if numeric_metrics:
                            param_keys = set(trial_data['params'].keys()) | set(forced_params.keys())
                            cleaned_metrics = _filter_numeric_metrics(numeric_metrics, param_keys)
                            objective_names = objectives or study_data.get('objectives', [])
                            objective_metrics = {name: cleaned_metrics[name] for name in objective_names if name in cleaned_metrics}
                            if objective_metrics:
                                trial_data['metrics'] = objective_metrics
                            if base_objectives:
                                holdout_metrics = {base: cleaned_metrics.get(f"{base}_holdout") for base in base_objectives if f"{base}_holdout" in cleaned_metrics}
                                train_metrics = {base: cleaned_metrics.get(f"{base}_train") for base in base_objectives if f"{base}_train" in cleaned_metrics}
                                if holdout_metrics:
                                    trial_data['metrics_holdout'] = holdout_metrics
                                if train_metrics:
                                    trial_data['metrics_train'] = train_metrics
                    all_trials.append(trial_data)
            study_data['all_trials'] = all_trials
                
        except Exception as e:
            print(f"Warning: Error processing Ray Tune results: {e}")
    else:
        # Handle traditional Optuna study
        if objectives is None and hasattr(study, 'user_attrs') and 'objectives' in study.user_attrs:
            objectives = study.user_attrs['objectives']
            study_data['objectives'] = objectives
        
        if hasattr(study, 'directions'):
            study_data['directions'] = [str(d) for d in study.directions]
        
        # Extract best trials
        if hasattr(study, 'best_trials') and study.best_trials:
            for trial in study.best_trials:
                user_attrs = _serialize_user_attrs(dict(trial.user_attrs)) if hasattr(trial, 'user_attrs') else {}
                numeric_attrs = _extract_numeric(user_attrs)
                param_keys = set(dict(trial.params).keys()) | set(forced_params.keys())
                cleaned_metrics = _filter_numeric_metrics(numeric_attrs, param_keys)
                objective_names = objectives or study_data.get('objectives', [])
                objective_metrics = {name: cleaned_metrics[name] for name in objective_names if name in cleaned_metrics}
                trial_entry = {
                    'number': trial.number,
                    'values': list(trial.values) if trial.values is not None else [],
                    'params': dict(trial.params),
                    'state': str(trial.state),
                    'user_attrs': user_attrs
                }
                if objective_metrics:
                    trial_entry['metrics'] = objective_metrics
                study_data['best_trials'].append(trial_entry)
                if user_attrs.get('metrics_holdout'):
                    study_data['best_trials'][-1]['metrics_holdout'] = user_attrs['metrics_holdout']
                if user_attrs.get('metrics_train'):
                    study_data['best_trials'][-1]['metrics_train'] = user_attrs['metrics_train']

        # Extract all completed trials
        if hasattr(study, 'trials'):
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    user_attrs = _serialize_user_attrs(dict(trial.user_attrs)) if hasattr(trial, 'user_attrs') else {}
                    numeric_attrs = _extract_numeric(user_attrs)
                    param_keys = set(dict(trial.params).keys()) | set(forced_params.keys())
                    cleaned_metrics = _filter_numeric_metrics(numeric_attrs, param_keys)
                    objective_names = objectives or study_data.get('objectives', [])
                    objective_metrics = {name: cleaned_metrics[name] for name in objective_names if name in cleaned_metrics}
                    trial_entry = {
                        'number': trial.number,
                        'values': list(trial.values) if hasattr(trial, 'values') and trial.values is not None else ([trial.value] if hasattr(trial, 'value') else []),
                        'params': dict(trial.params),
                        'state': str(trial.state),
                        'user_attrs': user_attrs
                    }
                    if objective_metrics:
                        trial_entry['metrics'] = objective_metrics
                    study_data['all_trials'].append(trial_entry)
                    if user_attrs.get('metrics_holdout'):
                        study_data['all_trials'][-1]['metrics_holdout'] = user_attrs['metrics_holdout']
                    if user_attrs.get('metrics_train'):
                        study_data['all_trials'][-1]['metrics_train'] = user_attrs['metrics_train']

    if not study_data['base_objectives']:
        seen = set()
        base_from_objectives: List[str] = []
        for obj in study_data.get('objectives', []):
            base = _base_metric_name(obj)
            if base not in seen:
                base_from_objectives.append(base)
                seen.add(base)
        if base_from_objectives:
            study_data['base_objectives'] = base_from_objectives

    completed_trials = len(study_data.get('all_trials', []))
    study_data['completed_trials'] = completed_trials
    if expected_trials is not None:
        study_data['expected_trials'] = int(expected_trials)

    # Save to JSON file with scenario ID in filename
    json_file = output_path / f"study_{scenario_id}.json"
    with open(json_file, 'w') as f:
        json.dump(study_data, f, indent=2)
    
    print(f"Study saved to: {json_file}")


def create_scenario_id(dataset_name: str, forced_params: Dict[str, Any]) -> str:
    """Create a unique identifier for a scenario that can be matched across seeds."""
    # Use dataset and key forced params to create a consistent ID
    parts = [
        dataset_name,
        forced_params.get('processing_method', 'unknown'),
        forced_params.get('sampling_method', 'unknown'),
        forced_params.get('batch_mode', 'unknown'),
        forced_params.get('topology_type', 'unknown')
    ]
    if (
        str(forced_params.get('sampling_method', '')).lower() == 'random'
        and 'target_proportion' in forced_params
    ):
        target = float(forced_params['target_proportion'])
        target_tag = f"{target:.6g}".replace('.', 'p').replace('-', 'm')
        parts.append(f"tp{target_tag}")
    return "_".join(parts)

def get_completed_scenarios_for_seed(
    output_dir: str,
    seed: int,
    expected_trials: Optional[int] = None,
) -> set:
    """Get set of completed scenario IDs for a given seed.
    
    Args:
        output_dir: Base output directory
        seed: Seed number
        
    Returns:
        Set of completed scenario IDs
    """
    from pathlib import Path
    import json
    
    seed_dir = Path(output_dir) / f"seed_{seed}"
    completed = set()
    
    if not seed_dir.exists():
        return completed
    
    # Look for study_*.json files
    for json_file in seed_dir.glob("study_*.json"):
        if json_file.name.startswith("study_") and json_file.name.endswith(".json"):
            # Extract scenario ID from filename
            scenario_id = json_file.name[6:-5]  # Remove 'study_' and '.json'
            
            # Verify the file has actual results
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    completed_trials = data.get('completed_trials')
                    if completed_trials is None:
                        completed_trials = len(data.get('all_trials', []))
                        if completed_trials == 0:
                            completed_trials = len(data.get('best_trials', []))

                    recorded_expected = data.get('expected_trials')
                    resolved_expected = (
                        int(recorded_expected)
                        if recorded_expected is not None
                        else (int(expected_trials) if expected_trials is not None else None)
                    )

                    if resolved_expected is not None:
                        if completed_trials >= resolved_expected:
                            completed.add(scenario_id)
                    else:
                        if completed_trials > 0:
                            completed.add(scenario_id)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return completed


def normalize_list_args(values: Optional[List[str]]) -> List[str]:
    """Normalize whitespace/comma separated CLI list arguments while preserving order."""
    if not values:
        return []

    normalized: List[str] = []
    seen = set()
    for raw in values:
        for item in raw.split(','):
            name = item.strip()
            if not name or name in seen:
                continue
            normalized.append(name)
            seen.add(name)
    return normalized


def run_full_mode(args) -> None:
    """Run Phase 3: Full orchestration across scenarios."""
    print("=" * 60)
    print("PHASE 3: FULL ORCHESTRATION")
    print("=" * 60)

    # Create configuration
    if args.config == 'development':
        config = create_phase3_development_config(
            output_dir=args.output_dir,
            n_trials_per_benchmark=args.trials,
            n_seeds=1  # Use single seed for now
        )
        print("Using DEVELOPMENT configuration (smaller scope)")
    elif args.config == 'full':
        config = create_phase3_full_config(
            output_dir=args.output_dir,
            n_trials_per_benchmark=args.trials,
            n_seeds=1  # Use single seed for now
        )
        print("Using FULL configuration (complete benchmark suite)")
    else:
        raise ValueError(f"Unknown config: {args.config}. Use 'development' or 'full'")

    dataset_filter = normalize_list_args(getattr(args, 'datasets', None))
    if dataset_filter:
        available_datasets = list(config.datasets)
        invalid_datasets = [dataset for dataset in dataset_filter if dataset not in available_datasets]
        if invalid_datasets:
            raise ValueError(
                f"Unknown dataset(s) for config '{args.config}': {invalid_datasets}. "
                f"Available datasets: {available_datasets}"
            )
        config.datasets = dataset_filter
        print(f"Using dataset filter: {config.datasets}")

    # Override topology types if specified
    if hasattr(args, 'topology') and args.topology != ['hexagonal']:
        print(f"Using custom topology types: {args.topology}")
        # Modify config to use custom topologies by updating the forced combinations
        from .config.parameters import get_default_forced_combinations
        original_get_combinations = config.generate_all_scenarios
        
        def generate_scenarios_with_custom_topologies(*scenario_args, **scenario_kwargs):
            # Get forced combinations with custom topology types
            forced_combinations = get_default_forced_combinations(topology_types=args.topology)
            
            # Temporarily replace get_default_forced_combinations in benchmark_config
            import floatsom_benchmarks.optuna.config.benchmark_config as bc_module
            original_get_forced = bc_module.get_default_forced_combinations
            bc_module.get_default_forced_combinations = lambda: forced_combinations
            
            try:
                return original_get_combinations(*scenario_args, **scenario_kwargs)
            finally:
                bc_module.get_default_forced_combinations = original_get_forced
                
        config.generate_all_scenarios = generate_scenarios_with_custom_topologies

    sampling_filter = normalize_list_args(getattr(args, 'sampling_methods', None))
    if sampling_filter:
        allowed = {'full', 'random', 'hdsssom'}
        invalid = [value for value in sampling_filter if value not in allowed]
        if invalid:
            raise ValueError(
                f"Unknown sampling method(s): {invalid}. "
                f"Supported: {sorted(allowed)}"
            )
        print(f"Using sampling-method filter: {sampling_filter}")

    processing_filter = normalize_list_args(getattr(args, 'processing_methods', None))
    if processing_filter:
        allowed = {'batch', 'colors'}
        invalid = [value for value in processing_filter if value not in allowed]
        if invalid:
            raise ValueError(
                f"Unknown processing method(s): {invalid}. "
                f"Supported: {sorted(allowed)}"
            )
        print(f"Using processing-method filter: {processing_filter}")

    batch_mode_filter = normalize_list_args(getattr(args, 'batch_modes', None))
    if batch_mode_filter:
        allowed = {'full_batch', 'minibatch'}
        invalid = [value for value in batch_mode_filter if value not in allowed]
        if invalid:
            raise ValueError(
                f"Unknown batch mode(s): {invalid}. "
                f"Supported: {sorted(allowed)}"
            )
        print(f"Using batch-mode filter: {batch_mode_filter}")

    random_target_proportion = getattr(args, 'random_target_proportion', None)
    if random_target_proportion is not None:
        if not (0.0 < float(random_target_proportion) <= 1.0):
            raise ValueError(
                f"Invalid --random-target-proportion={random_target_proportion}. "
                "Expected a float in (0, 1]."
            )
        print(f"Using random target proportion override: {float(random_target_proportion)}")
    
    # Create seed-based output directory (no timestamp since bash script will handle multiple seeds)
    if args.output_dir:
        seed_dir = f"seed_{args.seed}"
        output_dir = Path(args.output_dir) / seed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir = str(output_dir)
    elif config.output_dir:
        seed_dir = f"seed_{args.seed}"
        output_dir = Path(config.output_dir) / seed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir = str(output_dir)
    
    print(f"\nConfiguration:")
    print(f"  Algorithms: {config.algorithms}")
    print(f"  Datasets: {config.datasets}")
    print(f"  Seed: {args.seed}")
    print(f"  Trials per benchmark: {config.n_trials_per_benchmark}")
    print(f"  Output directory: {config.output_dir}")
    default_params = get_default_params()
    print(f"  Evaluation split: {args.evaluation_split}")
    gpu_status = "GPU" if default_params.use_gpu else "CPU"
    print(f"  Compute: {gpu_status}")

    base_objectives = list(args.objectives)
    expanded_objectives = expand_objectives_for_split(base_objectives, args.evaluation_split)

    # Calculate execution plan
    scenarios = config.generate_all_scenarios()
    if sampling_filter:
        sampling_set = {value.lower() for value in sampling_filter}
        scenarios = [
            scenario
            for scenario in scenarios
            if str(scenario.forced_params.get('sampling_method', '')).lower() in sampling_set
        ]
    if processing_filter:
        processing_set = {value.lower() for value in processing_filter}
        scenarios = [
            scenario
            for scenario in scenarios
            if str(scenario.forced_params.get('processing_method', '')).lower() in processing_set
        ]
    if batch_mode_filter:
        batch_mode_set = {value.lower() for value in batch_mode_filter}
        scenarios = [
            scenario
            for scenario in scenarios
            if str(scenario.forced_params.get('batch_mode', '')).lower() in batch_mode_set
        ]
    if random_target_proportion is not None:
        random_target_proportion = float(random_target_proportion)
        for scenario in scenarios:
            if str(scenario.forced_params.get('sampling_method', '')).lower() == 'random':
                scenario.forced_params['target_proportion'] = random_target_proportion
    
    # Check for completed scenarios if resuming
    completed_scenarios = set()
    if args.resume:
        completed_scenarios = get_completed_scenarios_for_seed(
            args.output_dir,
            args.seed,
            expected_trials=config.n_trials_per_benchmark,
        )
        if completed_scenarios:
            print(f"\nResuming from previous run - found {len(completed_scenarios)} completed scenarios")
            
            # Filter out completed scenarios
            original_count = len(scenarios)
            scenarios = [s for s in scenarios if create_scenario_id(s.dataset.name, s.forced_params) not in completed_scenarios]
            
            print(f"  Skipping {original_count - len(scenarios)} completed scenarios")
            print(f"  Remaining scenarios to run: {len(scenarios)}")
    
    total_trials = sum(scenario.n_trials for scenario in scenarios)
    
    print(f"\nExecution Plan for Seed {args.seed}:")
    print(f"  Total scenarios to run: {len(scenarios)}")
    print(f"  Total Optuna trials: {total_trials:,}")
    if scenarios:
        print(f"  Estimated runtime: {config.get_estimated_runtime_hours() * len(scenarios) / len(config.generate_all_scenarios()) / config.n_seeds:.1f} hours")
    
    # Show sample scenarios
    scenario_limit = get_scenario_display_limit()
    print(f"\nSample scenarios:")
    for i, scenario in enumerate(scenarios[:scenario_limit]):
        print(f"  {i+1}. {scenario.name}")
        print(f"     Algorithm: {scenario.algorithm}")
        print(f"     Dataset: {scenario.dataset.name}")
        print(f"     Forced params: {scenario.forced_params}")
    
    if len(scenarios) > scenario_limit:
        print(f"  ... and {len(scenarios) - scenario_limit} more scenarios")
    
    # Confirmation for large runs
    warning_threshold = get_large_run_warning_threshold()
    if total_trials > warning_threshold:
        print(f"\nRunning {total_trials:,} trials (automatic mode for PBS job)...")
    
    # Check if there are any scenarios to run
    if not scenarios:
        print(f"\nAll scenarios have been completed for seed {args.seed}!")
        print(f"Results are in: {config.output_dir}")
        return
    
    print(f"\nStarting sequential execution of {len(scenarios)} scenarios...")
    
    # Run each scenario sequentially
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*50}")
        print(f"Scenario {i+1}/{len(scenarios)}: {scenario.name}")
        print(f"{'='*50}")
        
        # Run the benchmark for this scenario
        study = run_single_benchmark(
            algo_type=scenario.algorithm,
            dataset_name=scenario.dataset.name,
            dataset_config=scenario.dataset.to_dict(),
            forced_params=scenario.forced_params,
            seed=args.seed,
            n_trials=scenario.n_trials,
            objectives=base_objectives,
            evaluation_split=args.evaluation_split,
            output_dir=config.output_dir,
            use_ray_tune=False,
            max_concurrent=args.max_concurrent
        )

        # Save the study as JSON with scenario ID
        if study and config.output_dir:
            save_study_json(
                study,
                config.output_dir,
                scenario.dataset.name,
                scenario.forced_params,
                objectives=expanded_objectives,
                expected_trials=scenario.n_trials,
            )
            
        # Print scenario summary
        if study:
            is_ray_results = hasattr(study, '__class__') and ('ResultGrid' in str(study.__class__) or 'ray.tune' in str(study.__class__))
            if is_ray_results:
                valid_count = len([r for r in study if all(obj in r.metrics for obj in expanded_objectives)])
                print(f"Scenario completed. Valid trials: {valid_count}")
            elif hasattr(study, 'best_trials'):
                print(f"Scenario completed. Best trials: {len(study.best_trials)}")
            
    print(f"\n{'='*50}")
    print(f"All scenarios completed for seed {args.seed}")
    print(f"Results saved to: {config.output_dir}")
    print(f"{'='*50}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FloatSOM Optuna benchmarking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single benchmark
  python run_optuna.py --mode single --dataset swiss_roll --algo colors --trials 50
  
  # Full scenario execution for seed 42 (development config)
  python run_optuna.py --mode full --config development --seed 42 --output-dir results
  # Creates: results/seed_42_development_hexagonal/
  
  # Full scenario execution for seed 59 (full config)
  python run_optuna.py --mode full --config full --seed 59 --output-dir results
  # Creates: results/seed_59_full_hexagonal/

Note: The first trial in each optimization always uses default parameters from FloatSOMParams
to establish a baseline. Subsequent trials use Optuna's optimization algorithms (TPE/NSGA-II).
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'full'], required=True,
                        help='Execution mode: single benchmark or full scenario execution')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str,
                        help='Base output directory for results')
    
    parser.add_argument('--seed', type=int, default=resolve_default_seed(),
                        help='Random seed (default from floatsom_params)')
    
    # Single mode arguments
    parser.add_argument('--dataset', type=str, default=resolve_default_dataset(),
                        help='Dataset name for single mode (default from floatsom_params)')
    parser.add_argument('--algo', type=str, default=resolve_default_algo(),
                        choices=['colors', 'batch'],
                        help='Algorithm type for single mode (default from floatsom_params)')
    
    # Common optimization arguments
    parser.add_argument('--topology', type=str, nargs='+', 
                        default=['hexagonal'],
                        choices=['hexagonal', 'grid', 'mst', 'rng'],
                        help='Topology types to use for full runs (default: hexagonal only)')
    parser.add_argument(
        '--sampling-methods',
        type=str,
        nargs='+',
        default=None,
        choices=['full', 'random', 'hdsssom'],
        help=(
            'Optional sampling-method filter for full runs. '
            'Accepts whitespace and/or comma-separated names.'
        ),
    )
    parser.add_argument(
        '--random-target-proportion',
        type=float,
        default=resolve_default_random_target_proportion(),
        help=(
            "Optional override for SamplingConfig.target_proportion when sampling_method=random. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        '--processing-methods',
        type=str,
        nargs='+',
        default=None,
        choices=['batch', 'colors'],
        help=(
            'Optional processing-method filter for full runs. '
            'Accepts whitespace and/or comma-separated names.'
        ),
    )
    parser.add_argument(
        '--batch-modes',
        type=str,
        nargs='+',
        default=None,
        choices=['full_batch', 'minibatch'],
        help=(
            "Optional batch-mode filter for full runs. "
            "Accepts whitespace and/or comma-separated names."
        ),
    )
    parser.add_argument('--trials', type=int, default=resolve_default_trials(),
                        help='Number of trials per scenario (default from floatsom_params)')
    parser.add_argument('--objectives', type=str, nargs='+', 
                        default=resolve_default_objectives(),
                        choices=['quantization_error', 'topographic_error', 'trustworthiness', 
                                'neighborhood_preservation', 'topographic_function'],
                        help='Objective metrics to optimize (default from floatsom_params). ' +
                             'Multiple objectives enable multi-objective optimization with NSGA-II.')
    
    parser.add_argument('--evaluation-split', type=str, choices=['both', 'holdout', 'train'],
                        default=resolve_default_evaluation_split(),
                        help='Which dataset split to optimise on (default: both).')

# Full mode only
    parser.add_argument('--config', type=str, choices=['development', 'full'], default=resolve_default_config_mode(),
                        help='Configuration preset for full mode (default from floatsom_params)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Optional dataset filter for full mode. Accepts whitespace and/or comma-separated names.')
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run by skipping completed scenarios')
    
    # Ray Tune configuration
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Maximum concurrent trials for Ray Tune (default: auto-detect based on available resources)')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"optuna_results_{args.mode}"
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"FloatSOM Optuna Benchmarking Pipeline")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print(f"Evaluation split: {args.evaluation_split}")
    print()
        
    # Route to appropriate mode
    if args.mode == 'single':
        run_single_mode(args)
    elif args.mode == 'full':
        run_full_mode(args)
    
    print(f"\nExecution completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
