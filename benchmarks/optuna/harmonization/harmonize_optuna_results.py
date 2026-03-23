#!/usr/bin/env python3
"""
Post-hoc harmonization of Optuna results from multiple seed runs.

This script reads results from multiple seed directories, groups them by scenario,
and applies the existing Pareto harmonization logic.

Usage:
    python harmonize_optuna_results.py --results-dir results --output-dir harmonized
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import optuna
from optuna.trial import FrozenTrial
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from harmonization.harmonization import ParetoHarmonizer
from harmonization.pareto_utils import find_best_trial_by_distance


@dataclass
class MockStudy:
    """Mock Study object that mimics optuna.Study for harmonization."""
    best_trials: List[FrozenTrial]
    directions: List[optuna.study.StudyDirection]
    study_name: str = "mock_study"


DATASET_UNIFIED_LABELS: Dict[str, str] = {
    "swiss_roll": "Swiss Roll",
    "s_curve": "S Curve",
    "moons": "Moons",
    "circles": "Circles",
    "blobs": "Blobs",
    "breast_cancer": "Breast Cancer",
    "wine": "Wine",
    "iris": "Iris",
    "digits": "Digits",
    "olivetti_faces": "Olivetti Faces",
}


def _slug_to_title(text: str) -> str:
    """Convert a dataset slug into a consistent human-readable label."""
    token_map = {
        "mnist": "MNIST",
        "som": "SOM",
        "mst": "MST",
        "qe": "QE",
        "2d": "2D",
        "3d": "3D",
    }
    pieces = [piece for piece in re.split(r"[_\-\s]+", text.strip()) if piece]
    return " ".join(token_map.get(piece.lower(), piece.capitalize()) for piece in pieces)


def _resolve_dataset_label(dataset_name: Optional[str], dataset_labeling: str) -> Optional[str]:
    """Resolve dataset label for downstream display/axis usage."""
    if dataset_name is None:
        return None
    if dataset_labeling == "raw":
        return str(dataset_name)

    key = str(dataset_name).strip().lower()
    if not key:
        return str(dataset_name)
    return DATASET_UNIFIED_LABELS.get(key, _slug_to_title(str(dataset_name)))


def load_study_json(json_path: Path) -> Dict[str, Any]:
    """Load a study from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def _parse_seed_from_seed_name(seed_name: Optional[str]) -> Optional[int]:
    """Parse integer seed from a directory-like name such as 'seed_42'."""
    if not seed_name:
        return None
    match = re.search(r"seed_(\d+)", str(seed_name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_seed_from_study_name(study_name: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Extract seed_name/seed from study_name suffix when available."""
    if not study_name:
        return None, None
    match = re.search(r"(seed_(\d+))$", str(study_name))
    if not match:
        return None, None
    seed_name = match.group(1)
    try:
        seed_value = int(match.group(2))
    except ValueError:
        seed_value = None
    return seed_name, seed_value


def json_to_study(
    study_data: Dict[str, Any],
    study_name: str = "mock_study",
    seed_name: Optional[str] = None,
    seed_value: Optional[int] = None
) -> MockStudy:
    """
    Convert JSON study data to a MockStudy object that ParetoHarmonizer can use.
    """
    # Parse directions
    directions_str = study_data.get('directions', ['MINIMIZE'])
    directions = []
    for d in directions_str:
        if 'MINIMIZE' in str(d):
            directions.append(optuna.study.StudyDirection.MINIMIZE)
        else:
            directions.append(optuna.study.StudyDirection.MAXIMIZE)
    
    # Convert best_trials from JSON to FrozenTrial objects
    best_trials = []
    for trial_dict in study_data.get('best_trials', []):
        # Create a FrozenTrial with the essential data
        # Note: We're creating a minimal FrozenTrial - some attributes are empty
        # FrozenTrial expects 'value' and 'trial_id' as first positional arguments
        # For multi-objective, 'value' should be None and 'values' should be used
        values = trial_dict.get('values', []) or []
        if not isinstance(values, (list, tuple)):
            values = [values]
        value = None if len(values) > 1 else (values[0] if values else None)
        stored_user_attrs = trial_dict.get('user_attrs', {})
        stored_metrics = trial_dict.get('metrics', {})
        user_attrs = {**stored_user_attrs, **stored_metrics}
        user_attrs.setdefault('study_name', study_name)
        user_attrs.setdefault('scenario_id', study_data.get('scenario_id', 'unknown'))
        if seed_name is not None:
            user_attrs.setdefault('seed_name', seed_name)
        if seed_value is not None:
            user_attrs.setdefault('seed', seed_value)
        trial = FrozenTrial(
            value=value,
            trial_id=trial_dict['number'],  # Use trial number as trial_id
            number=trial_dict['number'],
            values=tuple(values),
            params=trial_dict.get('params', {}),
            distributions={},  # Not saved in JSON
            user_attrs=user_attrs,
            system_attrs={},
            intermediate_values={},
            datetime_start=None,
            datetime_complete=None,
            state=optuna.trial.TrialState.COMPLETE
        )
        best_trials.append(trial)
    
    # Create mock study
    return MockStudy(
        best_trials=best_trials,
        directions=directions,
        study_name=study_name
    )


def get_objectives_from_args() -> List[str]:
    """
    Try to determine what objectives were used from command line args or use defaults.
    This could be improved by saving objectives in the study JSON.
    """
    # Default objectives that might have been used
    # This should match what was used in run_optuna.py
    return ['quantization_error_holdout', 'quantization_error_train']


def collect_studies_by_scenario(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all studies grouped by scenario ID.
    
    Returns:
        Dictionary mapping scenario_id to list of (seed_name, study_data, file_path) dicts
    """
    studies_by_scenario = defaultdict(list)
    
    # Find all seed directories
    seed_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    
    if not seed_dirs:
        print(f"No seed directories found in {results_dir}")
        return {}
    
    print(f"Found {len(seed_dirs)} seed directories")
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"\nProcessing {seed_name}...")
        
        # Find all study JSON files in this seed directory
        study_files = list(seed_dir.glob('study_*.json'))
        print(f"  Found {len(study_files)} study files")
        
        for study_file in study_files:
            try:
                study_data = load_study_json(study_file)
                scenario_id = study_data.get('scenario_id')
                
                if scenario_id:
                    studies_by_scenario[scenario_id].append({
                        'seed_name': seed_name,
                        'data': study_data,
                        'file_path': str(study_file)
                    })
                    print(f"    Loaded scenario: {scenario_id}")
                else:
                    print(f"    Warning: No scenario_id in {study_file}")
                    
            except Exception as e:
                print(f"    Error loading {study_file}: {e}")
    
    return dict(studies_by_scenario)


def infer_objectives(num_values: int, directions: List[str]) -> List[str]:
    """
    Infer objective names based on number of values and directions.
    This is a heuristic - ideally objectives should be saved in the study JSON.
    """
    if num_values == 1:
        return ['quantization_error_holdout']
    elif num_values == 2:
        return ['quantization_error_holdout', 'quantization_error_train']
    elif num_values == 3:
        return ['quantization_error', 'topographic_error', 'trustworthiness']
    elif num_values == 4:
        return ['quantization_error', 'topographic_error', 'trustworthiness', 'neighborhood_preservation']
    else:
        # Generic names if we can't infer
        return [f'objective_{i+1}' for i in range(num_values)]


def harmonize_scenario(
    scenario_id: str,
    seed_studies_data: List[Dict[str, Any]],
    objectives: Optional[List[str]] = None,
    dataset_labeling: str = "raw",
) -> Optional[Dict[str, Any]]:
    """
    Harmonize results for a single scenario across multiple seeds using existing harmonization logic.
    """
    print(f"\nHarmonizing scenario: {scenario_id}")
    print(f"  Number of seeds: {len(seed_studies_data)}")
    
    # Convert JSON data to MockStudy objects
    mock_studies = []
    seed_info = []
    
    # Get objectives from the study data if not provided via command line
    if objectives is None and seed_studies_data:
        first_study = seed_studies_data[0]['data']
        # Check if objectives are saved in the study JSON
        if 'objectives' in first_study:
            objectives = first_study['objectives']
            print(f"  Objectives from study: {objectives}")
        else:
            raise ValueError(f"No objectives found in study JSON for scenario {scenario_id}. Please re-run benchmarks with updated code.")
    
    for seed_study in seed_studies_data:
        seed_name = seed_study['seed_name']
        seed_value = _parse_seed_from_seed_name(seed_name)
        study_data = seed_study['data']
        
        # Create mock study from JSON
        mock_study = json_to_study(
            study_data,
            study_name=f"{scenario_id}_{seed_name}",
            seed_name=seed_name,
            seed_value=seed_value
        )
        
        if mock_study.best_trials:
            mock_studies.append(mock_study)
            seed_info.append({
                'seed_name': seed_name,
                'seed': seed_value,
                'num_best_trials': len(mock_study.best_trials),
                'file_path': seed_study['file_path']
            })
            print(f"  {seed_name}: {len(mock_study.best_trials)} best trials")
        else:
            print(f"  {seed_name}: No best trials found, skipping")
    
    if not mock_studies:
        print(f"  No valid studies found for {scenario_id}")
        return None
    
    # Use existing ParetoHarmonizer
    harmonizer = ParetoHarmonizer()
    try:
        harmonized_front = harmonizer.harmonize_fronts(mock_studies)
    except Exception as e:
        print(f"  Error harmonizing: {e}")
        return None
    
    print(f"  Harmonized front size: {len(harmonized_front)}")
    
    # Find overall best trial (if multi-objective)
    overall_best_trial = None
    overall_best_distance = None
    
    is_multi_objective = len(mock_studies[0].directions) > 1
    if is_multi_objective and harmonized_front:
        overall_best_trial, overall_best_distance = find_best_trial_by_distance(
            harmonized_front, 
            mock_studies[0].directions
        )
        if overall_best_distance is not None and not np.isnan(overall_best_distance):
            print(f"  Overall best distance to origin: {overall_best_distance:.6f}")
        else:
            print(f"  Overall best distance to origin: None")
            overall_best_distance = None
    elif not is_multi_objective and harmonized_front:
        # For single-objective, find best value
        if mock_studies[0].directions[0] == optuna.study.StudyDirection.MINIMIZE:
            overall_best_trial = min(harmonized_front, key=lambda t: t.values[0] if t.values else float('inf'))
        else:
            overall_best_trial = max(harmonized_front, key=lambda t: t.values[0] if t.values else float('-inf'))
        if overall_best_trial and overall_best_trial.values:
            overall_best_distance = overall_best_trial.values[0]
            print(f"  Overall best value: {overall_best_distance:.6f}")
    
    # Create result dictionary
    first_study_data = seed_studies_data[0]['data']

    def _coerce_json_value(value: Any) -> Any:
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

    dataset_raw = first_study_data.get('dataset')
    result = {
        'scenario_id': scenario_id,
        'dataset': dataset_raw,
        'dataset_label': _resolve_dataset_label(dataset_raw, dataset_labeling),
        'dataset_labeling': dataset_labeling,
        'forced_params': first_study_data.get('forced_params'),
        'directions': first_study_data.get('directions'),
        'objectives': objectives,
        'num_seeds': len(seed_studies_data),
        'num_seeds_with_results': len(mock_studies),
        'seed_info': seed_info,
        'harmonized_front_size': len(harmonized_front),
        'overall_best_distance': float(overall_best_distance) if overall_best_distance is not None else None,
        'harmonized_trials': []
    }
    
    # Convert harmonized trials to JSON-serializable format
    for trial in harmonized_front:
        trial_data = {
            'values': list(trial.values) if trial.values else [],
            'params': dict(trial.params),
            'number': trial.number
        }

        user_attrs = _serialize_user_attrs(dict(trial.user_attrs)) if hasattr(trial, 'user_attrs') else {}
        if user_attrs:
            trial_data['user_attrs'] = user_attrs
        numeric_metrics = _extract_numeric(user_attrs)
        if numeric_metrics:
            trial_data['metrics'] = numeric_metrics

        trial_seed_name = user_attrs.get('seed_name')
        trial_seed = user_attrs.get('seed')
        if trial_seed_name is None or trial_seed is None:
            derived_seed_name, derived_seed = _extract_seed_from_study_name(user_attrs.get('study_name'))
            if trial_seed_name is None:
                trial_seed_name = derived_seed_name
            if trial_seed is None:
                trial_seed = derived_seed
        if trial_seed_name is not None:
            trial_data['seed_name'] = trial_seed_name
        if trial_seed is not None:
            try:
                trial_data['seed'] = int(trial_seed)
            except (TypeError, ValueError):
                trial_data['seed'] = trial_seed

        # Add metric scores as a dictionary
        if trial.values and objectives:
            metric_scores = {}
            for i, (obj, val) in enumerate(zip(objectives, trial.values)):
                metric_scores[obj] = float(val)
            trial_data['metric_scores'] = metric_scores
        
        # Mark the overall best trial
        if overall_best_trial and trial == overall_best_trial:
            trial_data['is_overall_best'] = True
            
        # Add distance for multi-objective
        if is_multi_objective and trial.values:
            trial_data['distance_to_origin'] = float(np.linalg.norm(trial.values))
            
        result['harmonized_trials'].append(trial_data)

    # Also store the overall best trial's metrics separately for easy access
    if overall_best_trial and overall_best_trial.values and objectives:
        result['overall_best_metrics'] = {}
        for i, (obj, val) in enumerate(zip(objectives, overall_best_trial.values)):
            result['overall_best_metrics'][obj] = float(val)
        best_numeric_attrs = _extract_numeric(_serialize_user_attrs(dict(overall_best_trial.user_attrs)))
        extra_metrics = {k: v for k, v in best_numeric_attrs.items() if k not in objectives}
        if extra_metrics:
            result['overall_best_additional_metrics'] = extra_metrics
    
    return result


def save_harmonized_results(
    harmonized_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dataset_labeling: str = "raw",
):
    """Save harmonized results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual scenario results
    for scenario_id, result in harmonized_results.items():
        if result:
            output_file = output_dir / f"harmonized_{scenario_id}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {output_file}")
    
    # Create and save summary
    summary = {
        'num_scenarios': len(harmonized_results),
        'scenarios': [],
        'total_harmonized_solutions': 0,
        'dataset_labeling': dataset_labeling,
    }
    
    # Sort scenarios alphabetically by scenario_id
    sorted_scenario_ids = sorted(harmonized_results.keys())
    
    for scenario_id in sorted_scenario_ids:
        result = harmonized_results[scenario_id]
        if result:
            scenario_summary = {
                'scenario_id': scenario_id,
                'dataset': result['dataset'],
                'dataset_label': result.get('dataset_label', result['dataset']),
                'forced_params': result['forced_params'],
                'objectives': result.get('objectives', []),
                'num_seeds': result['num_seeds'],
                'num_seeds_with_results': result['num_seeds_with_results'],
                'harmonized_front_size': result['harmonized_front_size'],
                'overall_best_distance': result['overall_best_distance']
            }
            
            # Add overall best metrics if available
            if 'overall_best_metrics' in result:
                scenario_summary['overall_best_metrics'] = result['overall_best_metrics']
            
            summary['scenarios'].append(scenario_summary)
            summary['total_harmonized_solutions'] += result['harmonized_front_size']
    
    summary_file = output_dir / "harmonization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_file}")
    
    # Create markdown report
    report_lines = [
        "# Harmonization Results",
        "",
        f"**Total scenarios**: {summary['num_scenarios']}",
        f"**Total Pareto solutions**: {summary['total_harmonized_solutions']}",
        "",
        "## Scenarios",
        ""
    ]
    
    # Sort scenarios alphabetically by scenario_id
    sorted_scenarios = sorted(summary['scenarios'], key=lambda x: x['scenario_id'])
    
    for scenario in sorted_scenarios:
        report_lines.append(f"### {scenario['scenario_id']}")
        dataset_label = scenario.get('dataset_label', scenario['dataset'])
        dataset_raw = scenario['dataset']
        if dataset_label != dataset_raw:
            report_lines.append(f"- Dataset: {dataset_label} (raw: {dataset_raw})")
        else:
            report_lines.append(f"- Dataset: {dataset_label}")
        report_lines.append(f"- Seeds harmonized: {scenario['num_seeds_with_results']}/{scenario['num_seeds']}")
        report_lines.append(f"- Pareto front size: {scenario['harmonized_front_size']}")
        
        # Show objectives
        if scenario.get('objectives'):
            report_lines.append(f"- Objectives: {', '.join(scenario['objectives'])}")
        
        # Show overall best metrics
        if 'overall_best_metrics' in scenario:
            report_lines.append("- Overall best metrics:")
            for metric, value in scenario['overall_best_metrics'].items():
                report_lines.append(f"  - {metric}: {value:.6f}")
        elif scenario['overall_best_distance'] is not None:
            report_lines.append(f"- Best distance/value: {scenario['overall_best_distance']:.6f}")
        
        # Add forced params info
        if scenario['forced_params']:
            report_lines.append("- Configuration:")
            for key, value in scenario['forced_params'].items():
                report_lines.append(f"  - {key}: {value}")
        report_lines.append("")
    
    report_file = output_dir / "harmonization_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Harmonize Optuna results from multiple seed runs"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing seed_* subdirectories with results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='harmonized',
        help='Output directory for harmonized results (default: harmonized)'
    )
    parser.add_argument(
        '--objectives',
        type=str,
        nargs='+',
        help='List of objective names used in the optimization (e.g., quantization_error topographic_error)'
    )
    parser.add_argument(
        '--dataset-labeling',
        type=str,
        choices=['raw', 'unified'],
        default='raw',
        help='Dataset label style in outputs: raw keeps source values, unified writes canonical display labels'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return 1
    
    print(f"Harmonizing results from: {results_dir}")
    print(f"Output directory: {output_dir}")
    if args.objectives:
        print(f"Objectives: {args.objectives}")
    print(f"Dataset labeling: {args.dataset_labeling}")
    
    # Collect all studies by scenario
    studies_by_scenario = collect_studies_by_scenario(results_dir)
    
    if not studies_by_scenario:
        print("No studies found to harmonize")
        return 1
    
    print(f"\nFound {len(studies_by_scenario)} unique scenarios")
    
    # Harmonize each scenario
    harmonized_results = {}
    for scenario_id, seed_studies in studies_by_scenario.items():
        result = harmonize_scenario(
            scenario_id,
            seed_studies,
            objectives=args.objectives,
            dataset_labeling=args.dataset_labeling,
        )
        if result:
            harmonized_results[scenario_id] = result
    
    if not harmonized_results:
        print("\nNo scenarios could be harmonized")
        return 1
    
    # Save results
    print("\nSaving harmonized results...")
    save_harmonized_results(
        harmonized_results,
        output_dir,
        dataset_labeling=args.dataset_labeling,
    )
    
    print("\nHarmonization complete!")
    print(f"Results saved to: {output_dir}")
    
    # Print summary statistics
    total_scenarios = len(harmonized_results)
    total_solutions = sum(r['harmonized_front_size'] for r in harmonized_results.values() if r)
    print(f"\nSummary:")
    print(f"  Scenarios harmonized: {total_scenarios}")
    print(f"  Total Pareto solutions: {total_solutions}")
    
    # Show a sample of metrics for the first scenario
    first_scenario = next(iter(harmonized_results.values()))
    if first_scenario and 'overall_best_metrics' in first_scenario:
        print(f"\n  Sample best metrics ({first_scenario['scenario_id']}):")
        for metric, value in first_scenario['overall_best_metrics'].items():
            print(f"    {metric}: {value:.6f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
