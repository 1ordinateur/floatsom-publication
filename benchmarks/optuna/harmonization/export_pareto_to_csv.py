#!/usr/bin/env python3
"""
Export harmonized Pareto front data to CSV format.

This script reads harmonized JSON files and exports the Pareto front solutions
to a CSV file with columns for dataset, processing type, sampling method, and
raw metric values (without normalization).

Usage:
    python export_pareto_to_csv.py --input-dir harmonized --output pareto_front_results.csv
"""

import argparse
import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_harmonized_json(json_path: Path) -> Dict[str, Any]:
    """Load a harmonized result from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_scenario_info(scenario_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract dataset, processing type, and sampling method from scenario data.
    
    Args:
        scenario_data: Dictionary containing scenario information
        
    Returns:
        Dictionary with extracted scenario info
    """
    scenario_id = scenario_data.get('scenario_id', 'unknown')
    scenario_info = {
        'scenario_id': scenario_id,
        'dataset': scenario_data.get('dataset', 'unknown'),
        'processing_type': 'unknown',
        'sampling_method': 'unknown'
    }
    
    # Parse from scenario_id if possible
    # Example formats:
    # - moons_colors_hdsssom_hexagonal
    # - blobs_batch_full_hexagonal
    # - iris_batch_random_grid
    
    parts = scenario_id.split('_')
    
    # Known processing types
    processing_types = ['batch', 'colors']
    # Known sampling methods
    sampling_methods = ['random', 'hdsssom', 'full']
    # Known topology types (to exclude from parsing)
    topology_types = ['hexagonal', 'grid', 'mst']
    
    # Find processing type in parts
    for part in parts:
        if part in processing_types:
            scenario_info['processing_type'] = part
            break
    
    # Find sampling method in parts
    for part in parts:
        if part in sampling_methods:
            scenario_info['sampling_method'] = part
            break
    
    # If not found in scenario_id, try forced_params
    forced_params = scenario_data.get('forced_params', {})
    
    # Look for processing type indicators in forced_params if not found
    if scenario_info['processing_type'] == 'unknown':
        if 'processor_type' in forced_params:
            scenario_info['processing_type'] = forced_params['processor_type']
        elif 'processing' in forced_params:
            scenario_info['processing_type'] = forced_params['processing']
    
    # Look for sampling method indicators in forced_params if not found
    if scenario_info['sampling_method'] == 'unknown':
        if 'selector_type' in forced_params:
            scenario_info['sampling_method'] = forced_params['selector_type']
        elif 'sampling' in forced_params:
            scenario_info['sampling_method'] = forced_params['sampling']
        elif 'selector' in forced_params:
            scenario_info['sampling_method'] = forced_params['selector']
    
    return scenario_info


def _parse_seed_from_seed_name(seed_name: Any) -> Any:
    """Parse integer seed from strings like 'seed_42'."""
    if seed_name is None:
        return None
    match = re.search(r"seed_(\d+)", str(seed_name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_seed_fields(trial: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Extract (seed, seed_name) from explicit trial fields or user_attrs fallback.
    """
    seed = trial.get('seed')
    seed_name = trial.get('seed_name')

    user_attrs = trial.get('user_attrs') or {}
    if seed_name is None:
        seed_name = user_attrs.get('seed_name')
    if seed is None:
        seed = user_attrs.get('seed')

    if seed_name is None:
        study_name = user_attrs.get('study_name')
        if study_name:
            match = re.search(r"(seed_(\d+))$", str(study_name))
            if match:
                seed_name = match.group(1)
                if seed is None:
                    try:
                        seed = int(match.group(2))
                    except ValueError:
                        pass

    if seed is None and seed_name is not None:
        seed = _parse_seed_from_seed_name(seed_name)

    return seed, seed_name


def extract_pareto_solutions(scenario_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all Pareto front solutions from scenario data.
    
    Args:
        scenario_data: Dictionary containing scenario information
        
    Returns:
        List of solution dictionaries with metric values
    """
    solutions = []
    scenario_info = parse_scenario_info(scenario_data)
    objectives = scenario_data.get('objectives', [])
    
    # Extract each trial from the harmonized front
    forced_params = scenario_data.get('forced_params', {}) or {}

    for trial in scenario_data.get('harmonized_trials', []):
        seed, seed_name = _extract_seed_fields(trial)
        solution = {
            'scenario_id': scenario_info['scenario_id'],
            'dataset': scenario_info['dataset'],
            'processing_type': scenario_info['processing_type'],
            'sampling_method': scenario_info['sampling_method'],
            'seed': seed,
            'seed_name': seed_name,
            'trial_number': trial.get('number', -1),
            'is_overall_best': trial.get('is_overall_best', False)
        }

        # Add raw metric values
        metric_scores = trial.get('metric_scores', {})
        for metric_name, value in metric_scores.items():
            solution[metric_name] = value

        # Add distance to origin if available (raw, not normalized)
        if 'distance_to_origin' in trial:
            solution['distance_to_origin'] = trial['distance_to_origin']

        # Add raw values if metric_scores not available
        if not metric_scores and 'values' in trial and objectives:
            for i, (obj, val) in enumerate(zip(objectives, trial['values'])):
                solution[obj] = val

        # Add trial parameters under param_ prefix so they are easy to identify
        for param_name, value in (trial.get('params') or {}).items():
            solution[f'param_{param_name}'] = value

        # Include forced scenario parameters once per row for completeness
        for param_name, value in forced_params.items():
            solution.setdefault(f'config_{param_name}', value)

        solutions.append(solution)

    return solutions


def collect_all_solutions(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Collect all Pareto solutions from all harmonized JSON files.
    
    Args:
        input_dir: Directory containing harmonized_*.json files
        
    Returns:
        List of all solution dictionaries
    """
    all_solutions = []
    
    # Find all harmonized JSON files
    harmonized_files = list(input_dir.glob('harmonized_*.json'))
    
    if not harmonized_files:
        print(f"No harmonized_*.json files found in {input_dir}")
        return []
    
    print(f"Found {len(harmonized_files)} harmonized result files")
    
    for file_path in harmonized_files:
        try:
            scenario_data = load_harmonized_json(file_path)
            solutions = extract_pareto_solutions(scenario_data)
            all_solutions.extend(solutions)
            print(f"  {file_path.name}: {len(solutions)} solutions")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    return all_solutions


def export_to_csv(solutions: List[Dict[str, Any]], output_path: Path):
    """
    Export solutions to CSV format.
    
    Args:
        solutions: List of solution dictionaries
        output_path: Path for output CSV file
    """
    if not solutions:
        print("No solutions to export")
        return
    
    # Create DataFrame
    df = pd.DataFrame(solutions)
    
    # Sort by scenario_id/seed/trial_number for organization
    sort_columns = [col for col in ['scenario_id', 'seed', 'seed_name', 'trial_number'] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    
    # Reorder columns for better readability
    base_columns = ['scenario_id', 'dataset', 'processing_type', 'sampling_method', 'seed', 'seed_name', 'trial_number']
    param_columns = sorted(col for col in df.columns if col.startswith('param_'))
    config_columns = sorted(col for col in df.columns if col.startswith('config_'))
    excluded_columns = set(base_columns + ['is_overall_best', 'distance_to_origin'])
    excluded_columns.update(param_columns)
    excluded_columns.update(config_columns)
    metric_columns = [col for col in df.columns if col not in excluded_columns]
    metric_columns = sorted(metric_columns)

    final_columns = base_columns + metric_columns + param_columns + config_columns + ['distance_to_origin', 'is_overall_best']

    # Only include columns that exist in the data
    available_columns = [col for col in final_columns if col in df.columns]
    df = df[available_columns]
    
    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"\nExported {len(df)} solutions to {output_path}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Total solutions: {len(df)}")
    print(f"  Unique scenarios: {df['scenario_id'].nunique()}")
    print(f"  Unique datasets: {df['dataset'].nunique()}")
    print(f"  Processing types: {sorted(df['processing_type'].unique())}")
    print(f"  Sampling methods: {sorted(df['sampling_method'].unique())}")
    if 'seed' in df.columns:
        print(f"  Unique seeds: {df['seed'].nunique(dropna=True)}")
    
    if 'distance_to_origin' in df.columns:
        print(f"  Distance range: {df['distance_to_origin'].min():.4f} - {df['distance_to_origin'].max():.4f}")
    
    # Show metric columns
    print(f"  Metric columns: {metric_columns}")
    print(f"  Parameter columns: {param_columns}")
    if config_columns:
        print(f"  Scenario config columns: {config_columns}")


def main():
    parser = argparse.ArgumentParser(
        description="Export harmonized Pareto front data to CSV"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='harmonized',
        help='Directory containing harmonized_*.json files (default: harmonized)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pareto_front_results.csv',
        help='Output CSV file path (default: pareto_front_results.csv)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    print(f"Reading harmonized data from: {input_dir}")
    print(f"Output CSV file: {output_path}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Collect all solutions from harmonized files
    solutions = collect_all_solutions(input_dir)
    
    if not solutions:
        print("No solutions to export")
        return 1
    
    # Export to CSV
    export_to_csv(solutions, output_path)
    
    print("\nCSV export complete!")
    return 0


if __name__ == "__main__":
    exit(main())
