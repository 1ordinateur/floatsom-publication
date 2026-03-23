"""
Utility functions for Optuna benchmarking.

This module provides helper functions for seeding, file I/O, and other
common operations needed during benchmarking.
"""

import os
import json
import pickle
import hashlib
import random
import numpy as np
import cupy as cp
from typing import Dict, Any, Optional, Union
import optuna


def set_global_seed(seed: int):
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    
    # Set environment variable for CuPy
    os.environ['PYTHONHASHSEED'] = str(seed)


def hash_params(params: Dict[str, Any]) -> str:
    """
    Create deterministic hash of parameter dictionary.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        8-character hash string
    """
    # Sort parameters for consistent hashing
    sorted_params = sorted(params.items())
    param_str = "_".join(f"{k}={v}" for k, v in sorted_params)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


def save_study_results(
    study: optuna.Study, 
    study_name: str, 
    output_dir: str,
    include_trials_df: bool = True,
    include_params_importance: bool = True
):
    """
    Save Optuna study results to files.
    
    Args:
        study: Completed Optuna study
        study_name: Name of the study
        output_dir: Directory to save results
        include_trials_df: Whether to save trials dataframe as CSV
        include_params_importance: Whether to save parameter importance
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save study summary
    summary_file = os.path.join(output_dir, f"{study_name}_summary.json")
    summary = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'direction': [d.name for d in study.directions],
        'sampler': str(type(study.sampler).__name__),
        'pruner': str(type(study.pruner).__name__) if study.pruner else None,
        'state_counts': {}
    }
    
    # Count trials by state
    for trial in study.trials:
        state = trial.state.name
        summary['state_counts'][state] = summary['state_counts'].get(state, 0) + 1
    
    # Add best trial info if available
    if len(study.directions) == 1:
        try:
            best_trial = study.best_trial
            best_trial_data = {
                'number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params,
                'user_attrs': dict(best_trial.user_attrs)
            }
            
            # Extract discrete parameters from user_attrs if they exist
            from ..config.parameters import get_discrete_parameters
            discrete_keys = get_discrete_parameters()
            
            discrete_params = {}
            for key in discrete_keys:
                if key in best_trial.user_attrs:
                    discrete_params[key] = best_trial.user_attrs[key]
            
            # Add discrete parameters as a separate section if any exist
            if discrete_params:
                best_trial_data['discrete_params'] = discrete_params
            
            summary['best_trial'] = best_trial_data
        except ValueError:
            summary['best_trial'] = None
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save trials dataframe
    if include_trials_df and len(study.trials) > 0:
        csv_file = os.path.join(output_dir, f"{study_name}_trials.csv")
        df = study.trials_dataframe()
        df.to_csv(csv_file, index=False)
    
    # Save parameter importance
    if include_params_importance and len(study.trials) > 10:  # Need sufficient trials
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_file = os.path.join(output_dir, f"{study_name}_param_importance.json")
            with open(importance_file, 'w') as f:
                json.dump(importance, f, indent=2)
        except Exception:
            # Parameter importance calculation might fail for some configurations
            pass
    
    # Save study object using pickle for complete restoration
    study_file = os.path.join(output_dir, f"{study_name}_study.pkl")
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)


def load_study_results(study_file: str) -> optuna.Study:
    """
    Load Optuna study from pickle file.
    
    Args:
        study_file: Path to pickled study file
        
    Returns:
        Loaded Optuna study
    """
    with open(study_file, 'rb') as f:
        return pickle.load(f)


def create_output_directory(base_name: str, timestamp: bool = True) -> str:
    """
    Create timestamped output directory.
    
    Args:
        base_name: Base name for directory
        timestamp: Whether to append timestamp
        
    Returns:
        Path to created directory
    """
    if timestamp:
        import time
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        dir_name = f"{base_name}_{timestamp_str}"
    else:
        dir_name = base_name
    
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def validate_dataset_config(dataset_name: str, dataset_config: Dict[str, Any]) -> bool:
    """
    Validate dataset configuration parameters.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    
    required_keys = []
    
    # Define required keys per dataset type
    if dataset_name in ['swiss_roll', 's_curve', 'moons', 'circles', 'breast_cancer', 'wine', 'iris', 'digits', 'olivetti_faces']:
        required_keys = ['difficulty']
    elif dataset_name == 'blobs':
        required_keys = ['difficulty']
    
    # Check required keys
    for key in required_keys:
        if key not in dataset_config:
            return False
    
    # Validate difficulty levels
    if 'difficulty' in dataset_config:
        valid_difficulties = ['easy', 'medium', 'hard']
        if dataset_config['difficulty'] not in valid_difficulties:
            return False
    
    return True


def extract_best_configs(
    results: Dict[str, optuna.Study], 
    top_k: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Extract best configurations from multiple study results.
    
    Args:
        results: Dictionary mapping study names to Optuna studies
        top_k: Number of top configurations to extract
        
    Returns:
        Dictionary of best configurations
    """
    
    best_configs = []
    
    for study_name, study in results.items():
        if study.best_trial:
            config = {
                'study_name': study_name,
                'value': study.best_value,
                'params': study.best_params,
                'user_attrs': dict(study.best_trial.user_attrs),
                'trial_number': study.best_trial.number
            }
            best_configs.append(config)
    
    # Sort by value (assuming minimization)
    best_configs.sort(key=lambda x: x['value'])
    
    # Return top k configurations
    return {f"rank_{i+1}": config for i, config in enumerate(best_configs[:top_k])}


def merge_parameter_dicts(base_params: Dict[str, Any], override_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge parameter dictionaries with override priority.
    
    Args:
        base_params: Base parameter dictionary
        override_params: Override parameter dictionary
        
    Returns:
        Merged parameter dictionary
    """
    merged = base_params.copy()
    merged.update(override_params)
    return merged


def format_study_summary(study: optuna.Study) -> str:
    """
    Format study summary as readable string.
    
    Args:
        study: Optuna study
        
    Returns:
        Formatted summary string
    """
    
    lines = []
    lines.append(f"Study: {study.study_name}")
    lines.append(f"Trials: {len(study.trials)}")
    
    # State counts
    state_counts = {}
    for trial in study.trials:
        state = trial.state.name
        state_counts[state] = state_counts.get(state, 0) + 1
    
    for state, count in state_counts.items():
        lines.append(f"  {state}: {count}")
    
    # Best result
    if study.best_trial:
        lines.append(f"Best value: {study.best_value:.6f}")
        lines.append("Best params:")
        for param, value in study.best_params.items():
            lines.append(f"  {param}: {value}")
    
    return "\n".join(lines)


def calculate_improvement_over_baseline(
    study: optuna.Study, 
    baseline_value: float
) -> Dict[str, float]:
    """
    Calculate improvement metrics over a baseline value.
    
    Args:
        study: Optuna study
        baseline_value: Baseline value to compare against
        
    Returns:
        Dictionary with improvement metrics
    """
    
    if not study.best_trial:
        return {'improvement': 0.0, 'improvement_percent': 0.0}
    
    best_value = study.best_value
    improvement = baseline_value - best_value  # Assuming minimization
    improvement_percent = (improvement / baseline_value) * 100 if baseline_value != 0 else 0.0
    
    return {
        'baseline_value': baseline_value,
        'best_value': best_value,
        'improvement': improvement,
        'improvement_percent': improvement_percent
    }


def get_trial_statistics(study: optuna.Study) -> Dict[str, Any]:
    """
    Get detailed statistics about trials in a study.
    
    Args:
        study: Optuna study
        
    Returns:
        Dictionary with trial statistics
    """
    
    if not study.trials:
        return {}
    
    # Get all complete trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not complete_trials:
        return {'complete_trials': 0}
    
    values = [t.value for t in complete_trials]
    
    stats = {
        'total_trials': len(study.trials),
        'complete_trials': len(complete_trials),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'best_value': min(values),
        'worst_value': max(values),
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'median_value': np.median(values)
    }
    
    return stats


def export_results_to_csv(
    results: Dict[str, optuna.Study], 
    output_file: str,
    include_all_trials: bool = False
):
    """
    Export results from multiple studies to a single CSV file.
    
    Args:
        results: Dictionary mapping study names to studies
        output_file: Path to output CSV file
        include_all_trials: Whether to include all trials or just best ones
    """
    
    import pandas as pd
    
    all_data = []
    
    for study_name, study in results.items():
        if include_all_trials:
            # Include all completed trials
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    row = {
                        'study_name': study_name,
                        'trial_number': trial.number,
                        'value': trial.value
                    }
                    row.update(trial.params)
                    row.update(trial.user_attrs)
                    all_data.append(row)
        else:
            # Include only best trial
            if study.best_trial:
                row = {
                    'study_name': study_name,
                    'trial_number': study.best_trial.number,
                    'value': study.best_trial.value
                }
                row.update(study.best_trial.params)
                row.update(study.best_trial.user_attrs)
                all_data.append(row)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
    else:
        # Create empty CSV with headers
        pd.DataFrame().to_csv(output_file, index=False)