"""
Pareto front extraction utilities for Optuna studies.

This module provides functionality to extract Pareto-optimal solutions from
Optuna studies, leveraging Optuna's built-in multi-objective optimization support.
"""

from typing import List, Dict, Optional, Any, Tuple, Union
import optuna
import numpy as np
from optuna.trial import FrozenTrial, TrialState


def extract_pareto_front(study: optuna.Study) -> List[FrozenTrial]:
    """
    Extract Pareto-optimal trials from study using Optuna's built-in functionality.
    
    For single-objective studies, returns the best trial.
    For multi-objective studies, uses study.best_trials to get Pareto-optimal solutions.
    
    Args:
        study: Optuna study object containing completed trials
        
    Returns:
        List of Pareto-optimal FrozenTrial objects
        
    Raises:
        ValueError: If study is None or invalid
        TypeError: If study is not an Optuna Study object
    """
    # Input validation
    if study is None:
        raise ValueError("Study cannot be None")
    
    if not isinstance(study, optuna.Study):
        raise TypeError("Input must be an Optuna Study object")
    
    # Get completed trials only
    completed_trials = [
        trial for trial in study.trials 
        if trial.state == TrialState.COMPLETE
    ]
    
    # Handle empty studies
    if not completed_trials:
        return []
    
    # Single-objective case: return best trial
    if len(study.directions) == 1:
        best_trial = study.best_trial
        return [best_trial] if best_trial is not None else []
    
    # Multi-objective case: use Optuna's built-in Pareto-optimal extraction
    try:
        # study.best_trials returns Pareto-optimal trials for multi-objective studies
        pareto_trials = study.best_trials
        return pareto_trials if pareto_trials else []
    except ValueError:
        # Fallback: if study.best_trials fails, return empty list
        return []


def extract_metrics_from_trial(trial: FrozenTrial) -> Dict[str, float]:
    """
    Extract all metrics from trial user attributes and objective values.
    
    Args:
        trial: FrozenTrial object containing user attributes
        
    Returns:
        Dictionary mapping metric names to values
        
    Raises:
        ValueError: If trial is None
        TypeError: If trial is not a FrozenTrial object
    """
    if trial is None:
        raise ValueError("Trial cannot be None")
    
    if not isinstance(trial, FrozenTrial):
        raise TypeError("Input must be a FrozenTrial object")
    
    metrics = {}
    
    # Extract numeric user attributes as metrics
    for key, value in trial.user_attrs.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    
    # Also include objective values
    if trial.values is not None:
        if len(trial.values) == 1:
            metrics['objective'] = trial.values[0]
        else:
            for i, obj_value in enumerate(trial.values):
                metrics[f'objective_{i}'] = obj_value
    
    return metrics


def validate_study_for_harmonization(studies: List[optuna.Study]) -> bool:
    """
    Validate that all studies are compatible for harmonization.
    
    Checks that all studies have the same number of objectives and directions.
    
    Args:
        studies: List of Optuna studies to validate
        
    Returns:
        True if all studies are compatible, False otherwise
        
    Raises:
        ValueError: If studies list is empty or contains None values
    """
    if not studies:
        raise ValueError("Studies list cannot be empty")
    
    if any(study is None for study in studies):
        raise ValueError("Studies list cannot contain None values")
    
    # Get reference study for comparison
    reference_study = studies[0]
    ref_directions = reference_study.directions
    ref_n_objectives = len(ref_directions)
    
    # Check all studies have same structure
    for i, study in enumerate(studies[1:], 1):
        if len(study.directions) != ref_n_objectives:
            return False
        
        if study.directions != ref_directions:
            return False
    
    return True


def get_study_statistics(study: optuna.Study) -> Dict[str, Any]:
    """
    Get basic statistics about an Optuna study.
    
    Args:
        study: Optuna study to analyze
        
    Returns:
        Dictionary containing study statistics
    """
    if study is None:
        raise ValueError("Study cannot be None")
    
    completed_trials = [
        trial for trial in study.trials 
        if trial.state == TrialState.COMPLETE
    ]
    
    failed_trials = [
        trial for trial in study.trials 
        if trial.state == TrialState.FAIL
    ]
    
    pruned_trials = [
        trial for trial in study.trials 
        if trial.state == TrialState.PRUNED
    ]
    
    stats = {
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'failed_trials': len(failed_trials),
        'pruned_trials': len(pruned_trials),
        'n_objectives': len(study.directions),
        'directions': study.directions,
        'study_name': study.study_name,
    }
    
    # Add best values if available
    if completed_trials:
        if len(study.directions) == 1:
            if study.best_trial:
                stats['best_value'] = study.best_trial.value
                stats['best_params'] = study.best_trial.params
        else:
            pareto_trials = extract_pareto_front(study)
            stats['pareto_front_size'] = len(pareto_trials)
    
    return stats


def calculate_distance_to_origin(trial: FrozenTrial) -> float:
    """
    Calculate Euclidean distance from trial values to origin (0,0,...,0).
    
    For multi-objective optimization, finds the distance to the ideal point
    where all objectives are minimized to 0.
    
    Args:
        trial: FrozenTrial object with objective values
        
    Returns:
        Euclidean distance to origin
        
    Raises:
        ValueError: If trial is None or has no values
    """
    if trial is None:
        raise ValueError("Trial cannot be None")
    
    if trial.values is None or len(trial.values) == 0:
        raise ValueError("Trial must have objective values")
    
    # Calculate Euclidean distance to origin using numpy
    values = np.array(trial.values)
    return np.linalg.norm(values)


def find_best_trial_by_distance(trials: List[FrozenTrial], 
                                directions: Optional[List[Union[str, optuna.study.StudyDirection]]] = None) -> Tuple[Optional[FrozenTrial], Optional[float]]:
    """
    Find the trial with minimum distance to Pareto optimal point (origin).
    
    Args:
        trials: List of FrozenTrial objects
        directions: List of optimization directions (MINIMIZE/MAXIMIZE)
        
    Returns:
        Tuple of (best_trial, min_distance) or (None, None) if no valid trials
    """
    if not trials:
        return None, None
    
    # Filter out trials with NaN, Inf, or extreme penalty values
    # Note: We use 1e10/-1e10 as penalty values for failed objectives
    # These should still be filtered out for Pareto front calculation
    valid_trials = []
    for trial in trials:
        if trial.values is not None:
            has_invalid = False
            for v in trial.values:
                # Check for NaN, Inf, or penalty values (≥1e10 or ≤-1e10)
                if np.isnan(v) or np.isinf(v) or abs(v) >= 1e10:
                    has_invalid = True
                    break
            if not has_invalid:
                valid_trials.append(trial)
    
    if not valid_trials:
        return None, None
    
    # Transform values based on directions
    values_matrix = []
    for trial in valid_trials:
        transformed_values = []
        for i, val in enumerate(trial.values):
            if directions and i < len(directions):
                # For MAXIMIZE objectives, negate the value
                direction = directions[i]
                if (isinstance(direction, str) and direction == 'MAXIMIZE') or \
                   (hasattr(optuna.study.StudyDirection, 'MAXIMIZE') and direction == optuna.study.StudyDirection.MAXIMIZE):
                    transformed_values.append(-val)
                else:
                    transformed_values.append(val)
            else:
                transformed_values.append(val)
        values_matrix.append(transformed_values)
    
    values_matrix = np.array(values_matrix)
    distances = np.linalg.norm(values_matrix, axis=1)
    
    # Check for NaN in distances
    if np.all(np.isnan(distances)):
        return None, None
    
    # Find minimum distance, ignoring NaN values
    min_idx = np.nanargmin(distances)
    return valid_trials[min_idx], distances[min_idx]


def calculate_pareto_front_from_trials(
    trials: List[Dict[str, Any]], 
    objectives: List[str],
    directions: List[str]
) -> List[Dict[str, Any]]:
    """
    Calculate Pareto front from a list of trial dictionaries.
    
    This is useful for processing Ray Tune ResultGrid or other non-Optuna trial formats.
    
    Args:
        trials: List of trial dictionaries with 'values' key containing objective values
        objectives: List of objective names
        directions: List of optimization directions ('MINIMIZE' or 'MAXIMIZE')
        
    Returns:
        List of non-dominated trial dictionaries forming the Pareto front
        
    Raises:
        ValueError: If inputs are invalid or inconsistent
    """
    if not trials:
        return []
    
    if not objectives or not directions:
        raise ValueError("Objectives and directions must be provided")
    
    if len(objectives) != len(directions):
        raise ValueError("Number of objectives must match number of directions")
    
    # Filter out trials with invalid or penalty values
    valid_trials = []
    for trial in trials:
        if 'values' in trial and trial['values'] is not None:
            has_invalid = False
            for v in trial['values']:
                # Check for NaN, Inf, or penalty values (≥1e10 or ≤-1e10)
                if np.isnan(v) or np.isinf(v) or abs(v) >= 1e10:
                    has_invalid = True
                    break
            if not has_invalid:
                valid_trials.append(trial)
    
    if not valid_trials:
        return []
    
    # Single objective case
    if len(objectives) == 1:
        if directions[0] == 'MINIMIZE':
            best_trial = min(valid_trials, key=lambda t: t['values'][0])
        else:
            best_trial = max(valid_trials, key=lambda t: t['values'][0])
        return [best_trial]
    
    # Multi-objective case - find Pareto front
    pareto_trials = []
    
    for trial in valid_trials:
        is_dominated = False
        
        for other in valid_trials:
            if trial == other:
                continue
            
            # Check if 'other' dominates 'trial'
            all_equal_or_worse = True
            at_least_one_worse = False
            
            for i, direction in enumerate(directions):
                trial_val = trial['values'][i]
                other_val = other['values'][i]
                
                if direction == 'MINIMIZE':
                    if other_val > trial_val:
                        # Other is worse on this objective
                        all_equal_or_worse = False
                    elif other_val < trial_val:
                        # Trial is worse on this objective
                        at_least_one_worse = True
                else:  # MAXIMIZE
                    if other_val < trial_val:
                        # Other is worse on this objective
                        all_equal_or_worse = False
                    elif other_val > trial_val:
                        # Trial is worse on this objective
                        at_least_one_worse = True
            
            # If other dominates trial (equal or better on all, strictly better on at least one)
            if all_equal_or_worse and at_least_one_worse:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_trials.append(trial)
    
    return pareto_trials


def calculate_pareto_front_from_ray_results(
    results,  # Ray Tune ResultGrid
    objectives: List[str],
    directions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Calculate Pareto front from Ray Tune ResultGrid.
    
    Args:
        results: Ray Tune ResultGrid object
        objectives: List of objective names to extract from results
        directions: List of optimization directions. If None, infers from objective names.
        
    Returns:
        List of trial dictionaries forming the Pareto front
    """
    # Infer directions if not provided
    def _base_metric_name(metric_name: str) -> str:
        if metric_name.endswith('_holdout'):
            return metric_name[:-8]
        if metric_name.endswith('_train'):
            return metric_name[:-6]
        return metric_name

    if directions is None:
        maximize_metrics = {'trustworthiness', 'neighborhood_preservation'}
        directions = ['MAXIMIZE' if _base_metric_name(obj) in maximize_metrics else 'MINIMIZE' for obj in objectives]
    
    # Extract valid trials from ResultGrid
    valid_trials = []
    for i, result in enumerate(results):
        if all(obj in result.metrics for obj in objectives):
            # Check if any metrics are invalid or penalty values
            values = [result.metrics[obj] for obj in objectives]
            has_invalid = any(np.isnan(v) or np.isinf(v) or abs(v) >= 1e10 for v in values)
            
            if not has_invalid:
                trial_data = {
                    'number': i,
                    'values': values,
                    'params': dict((k, v) for k, v in result.config.items() if not k.startswith('_')),
                    'state': 'COMPLETE'
                }
                valid_trials.append(trial_data)
    
    # Calculate Pareto front
    return calculate_pareto_front_from_trials(valid_trials, objectives, directions)
