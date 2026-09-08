"""
Statistical evaluation mode for MST comparison benchmarks.

This module handles statistical evaluation across multiple datasets.
"""

import os
import time
import logging
import numpy as np
import cupy as cp
from typing import Dict, List, Any
from scipy import stats

from .utils import generate_data
from .training import (
    train_direct_mst_som,
    train_hexagonal_som, 
    convert_hexagonal_to_mst,
    get_mst_edge_weights
)

# Import MST clustering metrics
from floatsom.evaluation.mst_clustering_metrics import MSTClusteringMetrics

logger = logging.getLogger(__name__)

# Dataset categories
GENERATIVE_DATASETS = ['blobs', 'moons', 'circles', 'swiss_roll', 's_curve']
FIXED_DATASETS = ['iris', 'diabetes', 'breast_cancer', 'wine', 'olivetti_faces', 'digits']

# Check sklearn availability
try:
    from floatsom_benchmarks.evaluation.sklearn_datasets import SKLEARN_AVAILABLE
except ImportError:
    SKLEARN_AVAILABLE = False


def run_single_dataset_comparison(dataset_name: str, args, output_dir: str = None, 
                                 repeats: int = None, base_seed: int = None) -> Dict[str, Any]:
    """
    Run both MST approaches on a single dataset and return metrics.
    
    For generative datasets: runs multiple repeats with different seeds
    For fixed datasets: runs once (ignores repeats)
    
    Args:
        dataset_name: Name of the dataset
        args: Command line arguments
        output_dir: Directory for outputs
        repeats: Number of repeats (for generative datasets)
        base_seed: Base seed for reproducibility
    
    Returns:
        Dictionary with metrics for both approaches (aggregated if multiple repeats)
    """
    logger.info(f"\nProcessing dataset: {dataset_name}")
    
    # Clean dataset name
    clean_name = dataset_name.replace('sklearn_', '')
    
    # Determine if this is a generative or fixed dataset
    is_generative = clean_name in GENERATIVE_DATASETS
    
    # Set number of repeats
    if repeats is None:
        repeats = args.repeats if hasattr(args, 'repeats') else 1
    
    # Fixed datasets always run once
    if not is_generative:
        repeats = 1
        logger.info(f"  Fixed dataset - running once")
    else:
        logger.info(f"  Generative dataset - running {repeats} repeats")
    
    # Override args for this specific dataset
    args.data_type = f'sklearn_{clean_name}' if clean_name in (GENERATIVE_DATASETS + FIXED_DATASETS) else clean_name
    args.iterations = args.stats_iterations if hasattr(args, 'stats_iterations') else args.iterations
    
    # Store results from all repeats
    all_direct_metrics = {metric: [] for metric in ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']}
    all_hex_metrics = {metric: [] for metric in ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']}
    all_direct_times = []
    all_hex_times = []
    
    # Use base_seed if provided; else use args.seed; if still None, use entropy RNG
    if base_seed is None:
        base_seed = args.seed
    rng = None if base_seed is not None else np.random.default_rng()
    
    for repeat_idx in range(repeats):
        # Set unique seed for each repeat (or random if base not provided)
        if base_seed is not None:
            current_seed = int(base_seed + repeat_idx)
        else:
            current_seed = int(rng.integers(0, 2**31 - 1))
        args.seed = current_seed
        
        if repeats > 1:
            logger.info(f"  Repeat {repeat_idx + 1}/{repeats} (seed={current_seed})")
        
        try:
            # Generate data
            data, metadata = generate_data(args)
            
            # Train Direct MST-SOM (use distinct seed if no base seed)
            if repeats == 1:
                logger.info(f"  Training Direct MST-SOM on {clean_name}...")
            if base_seed is None:
                prev_seed = args.seed
                args.seed = int(rng.integers(0, 2**31 - 1))
            direct_mst_som, direct_stats, direct_time = train_direct_mst_som(data, args)
            if base_seed is None:
                args.seed = prev_seed
            all_direct_times.append(direct_time)
            
            # Train Hexagonal SOM (use distinct seed if no base seed)
            if repeats == 1:
                logger.info(f"  Training Hexagonal SOM on {clean_name}...")
            if base_seed is None:
                prev_seed = args.seed
                args.seed = int(rng.integers(0, 2**31 - 1))
            hexagonal_som, hex_stats, hex_time = train_hexagonal_som(data, args)
            if base_seed is None:
                args.seed = prev_seed
            all_hex_times.append(hex_time)
            
            # Convert hexagonal to MST
            hex_mst_edges, hex_mst_weights = convert_hexagonal_to_mst(hexagonal_som)
            
            # Get edge weights from direct MST
            direct_mst_weights = get_mst_edge_weights(direct_mst_som)
            
            # Per-run logging and visualization
            per_run_dir = os.path.join(
                args.output_dir, 'runs', clean_name, f'seed_{current_seed}'
            )
            os.makedirs(per_run_dir, exist_ok=True)
            
            # Calculate metrics
            metrics_calculator = MSTClusteringMetrics()
            
            # Gap statistics
            direct_gap = metrics_calculator.gap_statistic(direct_mst_weights)
            hex_gap = metrics_calculator.gap_statistic(hex_mst_weights)
            all_direct_metrics['gap_ratio'].append(direct_gap['gap_ratio'])
            all_hex_metrics['gap_ratio'].append(hex_gap['gap_ratio'])
            
            # Edge heterogeneity
            direct_hetero = metrics_calculator.edge_heterogeneity(direct_mst_weights)
            hex_hetero = metrics_calculator.edge_heterogeneity(hex_mst_weights)
            all_direct_metrics['gini'].append(direct_hetero['gini'])
            all_hex_metrics['gini'].append(hex_hetero['gini'])
            
            # Edge length ratio
            ratio_direct = metrics_calculator.edge_length_ratio(direct_mst_weights)
            ratio_hex = metrics_calculator.edge_length_ratio(hex_mst_weights)
            all_direct_metrics['edge_ratio'].append(ratio_direct['top_10_percent_ratio'])
            all_hex_metrics['edge_ratio'].append(ratio_hex['top_10_percent_ratio'])
            
            # Bimodality
            all_direct_metrics['bimodality'].append(metrics_calculator.bimodality_coefficient(direct_mst_weights))
            all_hex_metrics['bimodality'].append(metrics_calculator.bimodality_coefficient(hex_mst_weights))
            
            # Cluster separation
            sep_direct = metrics_calculator.clustering_score_range(direct_mst_weights)
            sep_hex = metrics_calculator.clustering_score_range(hex_mst_weights)
            all_direct_metrics['separation'].append(sep_direct['mean_separation'])
            all_hex_metrics['separation'].append(sep_hex['mean_separation'])
            
            # Cut stability
            stab_direct = metrics_calculator.cut_stability_proxy(direct_mst_weights)
            stab_hex = metrics_calculator.cut_stability_proxy(hex_mst_weights)
            all_direct_metrics['stability'].append(stab_direct['gap_uniformity'])
            all_hex_metrics['stability'].append(stab_hex['gap_uniformity'])

            # Save per-run metrics
            try:
                import json
                run_record = {
                    'mode': 'mst_stats',
                    'dataset': clean_name,
                    'seed': int(current_seed),
                    'samples': int(metadata['shape'][0]),
                    'features': int(metadata['shape'][1]),
                    'grid_size': int(getattr(args, 'grid_size', 0)),
                    'iterations': int(getattr(args, 'stats_iterations', args.iterations if hasattr(args, 'iterations') else 0)),
                    'training_times': {
                        'direct_mst_som': float(direct_time),
                        'hexagonal_som': float(hex_time)
                    },
                    'metrics': {
                        'direct_mst': {
                            'gap_ratio': direct_gap['gap_ratio'],
                            'gini': direct_hetero['gini'],
                            'edge_ratio': ratio_direct['top_10_percent_ratio'],
                            'bimodality': metrics_calculator.bimodality_coefficient(direct_mst_weights),
                            'separation': sep_direct['mean_separation'],
                            'stability': stab_direct['gap_uniformity']
                        },
                        'hex_to_mst': {
                            'gap_ratio': hex_gap['gap_ratio'],
                            'gini': hex_hetero['gini'],
                            'edge_ratio': ratio_hex['top_10_percent_ratio'],
                            'bimodality': metrics_calculator.bimodality_coefficient(hex_mst_weights),
                            'separation': sep_hex['mean_separation'],
                            'stability': stab_hex['gap_uniformity']
                        }
                    }
                }
                with open(os.path.join(per_run_dir, 'metrics.json'), 'w') as f:
                    json.dump(run_record, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save per-run MST metrics: {e}")

            # Save per-run visualization if requested
            try:
                from .visualization import visualize_msts
                # Always generate per-run visualization when --visualize is set
                if getattr(args, 'visualize', False):
                    visualize_msts(direct_mst_som, hexagonal_som, hex_mst_edges,
                                   data, per_run_dir, clean_name)
            except Exception as e:
                logger.warning(f"Failed to save per-run MST visualization: {e}")
            
        except Exception as e:
            logger.error(f"  Error in repeat {repeat_idx + 1}: {str(e)}")
            continue
    
    # Check if we got any successful runs
    if len(all_direct_times) == 0:
        return {
            'dataset': clean_name,
            'success': False,
            'error': 'No successful runs'
        }
    
    # Return aggregated results
    return {
        'dataset': clean_name,
        'is_generative': is_generative,
        'n_repeats': len(all_direct_times),
        'direct_metrics_all': all_direct_metrics,  # All values for statistical tests
        'hex_metrics_all': all_hex_metrics,        # All values for statistical tests
        'direct_metrics': {  # Mean values for reporting
            metric: np.mean(values) for metric, values in all_direct_metrics.items()
        },
        'hex_metrics': {  # Mean values for reporting
            metric: np.mean(values) for metric, values in all_hex_metrics.items()
        },
        'direct_metrics_std': {  # Std values for reporting
            metric: np.std(values) if len(values) > 1 else 0.0 
            for metric, values in all_direct_metrics.items()
        },
        'hex_metrics_std': {  # Std values for reporting
            metric: np.std(values) if len(values) > 1 else 0.0 
            for metric, values in all_hex_metrics.items()
        },
        'direct_time': np.mean(all_direct_times),
        'hex_time': np.mean(all_hex_times),
        'n_samples': metadata['shape'][0],
        'n_features': metadata['shape'][1],
        'success': True
    }


def run_statistical_evaluation(args) -> Dict[str, Any]:
    """
    Run statistical evaluation across all datasets.
    
    Handles generative datasets with multiple repeats and fixed datasets with single runs.
    
    Returns:
        Dictionary with statistical test results and recommendations
    """
    logger.info("="*80)
    logger.info("STARTING STATISTICAL EVALUATION MODE")
    logger.info("="*80)
    
    # Get all available datasets
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for statistical evaluation mode")
        return None
    
    # Separate datasets by type
    generative_datasets = [d for d in GENERATIVE_DATASETS]
    fixed_datasets = [d for d in FIXED_DATASETS]
    
    logger.info(f"Will evaluate on {len(generative_datasets)} generative datasets with {args.repeats} repeats each")
    logger.info(f"Will evaluate on {len(fixed_datasets)} fixed datasets (single run each)")
    logger.info(f"Difficulty: {args.stats_difficulty}")
    logger.info(f"Iterations per dataset: {args.stats_iterations}")
    
    # Create output directory for visualizations if needed
    output_dir = None
    if args.visualize:
        output_dir = os.path.join(args.output_dir, 'stats_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {output_dir}")
    
    # Store results separately for generative and fixed datasets
    generative_results = []
    fixed_results = []
    
    # Process generative datasets with repeats
    logger.info("\n" + "="*60)
    logger.info("PROCESSING GENERATIVE DATASETS")
    logger.info("="*60)
    
    for dataset_name in generative_datasets:
        try:
            # Override difficulty for this run
            args.difficulty = args.stats_difficulty
            
            # Run with repeats (handled inside run_single_dataset_comparison)
            result = run_single_dataset_comparison(
                dataset_name, args, output_dir, 
                repeats=args.repeats, base_seed=args.seed
            )
            
            if result['success']:
                generative_results.append(result)
                logger.info(f"  ✓ Completed {dataset_name}: {result['n_repeats']} repeats")
                
        except Exception as e:
            logger.warning(f"Skipping dataset {dataset_name} due to error: {str(e)}")
            continue
    
    # Process fixed datasets (single run each)
    logger.info("\n" + "="*60)
    logger.info("PROCESSING FIXED DATASETS")
    logger.info("="*60)
    
    for dataset_name in fixed_datasets:
        try:
            # Override difficulty for this run
            args.difficulty = args.stats_difficulty
            
            # Run once (repeats=1 will be enforced for fixed datasets)
            result = run_single_dataset_comparison(
                dataset_name, args, output_dir,
                repeats=1, base_seed=args.seed
            )
            
            if result['success']:
                fixed_results.append(result)
                logger.info(f"  ✓ Completed {dataset_name}")
                
        except Exception as e:
            logger.warning(f"Skipping dataset {dataset_name} due to error: {str(e)}")
            continue
    
    # Check if we have any results
    if len(generative_results) == 0 and len(fixed_results) == 0:
        logger.error("No successful results to analyze")
        return None
    
    # Perform statistical analysis with separate sections
    statistical_results = perform_statistical_analysis_separated(
        generative_results, fixed_results
    )
    
    # Generate visualization if requested
    if args.graphs:
        logger.info("\nGenerating metric comparison dot plots...")
        from .visualization import create_metric_dot_plots
        create_metric_dot_plots(generative_results, fixed_results, args.output_dir)
    
    # Generate and save report
    save_statistical_report_separated(statistical_results, args)
    
    # Log visualization output directory if applicable
    if args.visualize and output_dir:
        total_viz = len([r for r in generative_results + fixed_results if r['success']])
        logger.info(f"\nVisualizations saved to: {output_dir}")
        logger.info(f"Generated {total_viz} visualization(s)")
    
    if args.graphs:
        logger.info(f"Metric comparison violin plot saved to: {os.path.join(args.output_dir, 'mst_metrics_comparison_violinplot.png')}")
    
    return statistical_results


def perform_statistical_analysis_separated(generative_results: List[Dict[str, Any]], 
                                          fixed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform statistical analysis with separate sections for generative and fixed datasets.
    
    Args:
        generative_results: Results from generative datasets (with repeats)
        fixed_results: Results from fixed datasets (single run each)
    
    Returns:
        Dictionary with statistical test results for both dataset types
    """
    logger.info("\n" + "="*80)
    logger.info("PERFORMING STATISTICAL ANALYSIS (paired t-tests)")
    logger.info("="*80)
    
    # Metrics to analyze
    metrics = ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']
    
    statistical_results = {
        'generative': {
            'n_datasets': len(generative_results),
            'datasets': [r['dataset'] for r in generative_results],
            'n_repeats_per_dataset': generative_results[0]['n_repeats'] if generative_results else 0,
            'metric_tests': {},
            'winner': None,
            'significant_metrics': []
        },
        'fixed': {
            'n_datasets': len(fixed_results),
            'datasets': [r['dataset'] for r in fixed_results],
            'metric_tests': {},
            'winner': None,
            'significant_metrics': []
        },
        'overall': {
            'winner': None,
            'recommendation': ''
        }
    }

    def _paired_t_and_ci(direct_list: List[float], hex_list: List[float]) -> Dict[str, Any]:
        import numpy as _np
        from scipy import stats as _stats
        diffs = _np.asarray(direct_list) - _np.asarray(hex_list)
        n = diffs.size
        if n < 2:
            return {
                't_statistic': _np.nan,
                'df': n - 1,
                'p_value': _np.nan,
                'mean_diff': _np.nan,
                'std_diff': _np.nan,
                'ci_low': _np.nan,
                'ci_high': _np.nan
            }
        t_stat, p_val = _stats.ttest_rel(direct_list, hex_list, alternative='two-sided')
        mean_diff = float(_np.mean(diffs))
        std_diff = float(_np.std(diffs, ddof=1)) if n > 1 else 0.0
        se = std_diff / _np.sqrt(n) if n > 0 else _np.nan
        t_crit = _stats.t.ppf(0.975, df=n-1) if n > 1 else _np.nan
        ci_low = mean_diff - t_crit * se if _np.isfinite(t_crit) else _np.nan
        ci_high = mean_diff + t_crit * se if _np.isfinite(t_crit) else _np.nan
        return {
            't_statistic': float(t_stat),
            'df': int(n - 1),
            'p_value': float(p_val),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'ci_low': float(ci_low) if _np.isfinite(ci_low) else _np.nan,
            'ci_high': float(ci_high) if _np.isfinite(ci_high) else _np.nan
        }

    # No multiple-comparisons adjustment per user guidance; use raw p-values (alpha=0.05)
    
    # Analyze generative datasets using replicate-level pairing (no averaging across repeats)
    if generative_results:
        logger.info("\n" + "-"*60)
        logger.info("GENERATIVE DATASETS ANALYSIS (paired t-tests)")
        logger.info("-"*60)

        for metric in metrics:
            # Use all replicate-level pairs across all generative datasets
            direct_values = []
            hex_values = []

            for result in generative_results:
                # Collect replicate-level metric values; pair by repeat index within dataset
                if 'direct_metrics_all' in result and 'hex_metrics_all' in result:
                    dvals = result['direct_metrics_all'].get(metric, [])
                    hvals = result['hex_metrics_all'].get(metric, [])
                    # Ensure equal pairing length per dataset
                    min_len = min(len(dvals), len(hvals))
                    if min_len > 0:
                        direct_values.extend(dvals[:min_len])
                        hex_values.extend(hvals[:min_len])

            if len(direct_values) >= 2:
                ttest = _paired_t_and_ci(direct_values, hex_values)
                direct_mean = float(np.mean(direct_values))
                hex_mean = float(np.mean(hex_values))
                if direct_mean > hex_mean:
                    winner = 'Direct MST-SOM'
                    win_margin = (direct_mean - hex_mean) / (hex_mean if hex_mean != 0 else 1.0) * 100
                else:
                    winner = 'Hexagonal-to-MST'
                    win_margin = (hex_mean - direct_mean) / (direct_mean if direct_mean != 0 else 1.0) * 100

                # Mark significance by raw p-value
                statistical_results['generative']['metric_tests'][metric] = {
                    'direct_mean': direct_mean,
                    'direct_std': float(np.std(direct_values, ddof=1)) if len(direct_values) > 1 else 0.0,
                    'hex_mean': hex_mean,
                    'hex_std': float(np.std(hex_values, ddof=1)) if len(hex_values) > 1 else 0.0,
                    't_statistic': ttest['t_statistic'],
                    'df': ttest['df'],
                    'p_value_raw': ttest['p_value'],
                    'significant': bool(ttest['p_value'] < 0.05),
                    'ci_low': ttest['ci_low'],
                    'ci_high': ttest['ci_high'],
                    'mean_diff': ttest['mean_diff'],
                    'std_diff': ttest['std_diff'],
                    'winner': winner,
                    'win_margin_percent': win_margin,
                    'n_pairs': len(direct_values)
                }
                if ttest['p_value'] < 0.05:
                    statistical_results['generative']['significant_metrics'].append(metric)
    
    # Analyze fixed datasets (paired across datasets)
    if fixed_results:
        logger.info("\n" + "-"*60)
        logger.info("FIXED DATASETS ANALYSIS (paired t-tests)")
        logger.info("-"*60)
        
        for metric in metrics:
            direct_values = []
            hex_values = []
            for result in fixed_results:
                if 'direct_metrics' in result and 'hex_metrics' in result:
                    direct_values.append(result['direct_metrics'][metric])
                    hex_values.append(result['hex_metrics'][metric])

            if len(direct_values) >= 2:
                ttest = _paired_t_and_ci(direct_values, hex_values)
                direct_mean = float(np.mean(direct_values))
                hex_mean = float(np.mean(hex_values))
                if direct_mean > hex_mean:
                    winner = 'Direct MST-SOM'
                    win_margin = (direct_mean - hex_mean) / (hex_mean if hex_mean != 0 else 1.0) * 100
                else:
                    winner = 'Hexagonal-to-MST'
                    win_margin = (hex_mean - direct_mean) / (direct_mean if direct_mean != 0 else 1.0) * 100

                statistical_results['fixed']['metric_tests'][metric] = {
                    'direct_mean': direct_mean,
                    'direct_std': float(np.std(direct_values, ddof=1)) if len(direct_values) > 1 else 0.0,
                    'hex_mean': hex_mean,
                    'hex_std': float(np.std(hex_values, ddof=1)) if len(hex_values) > 1 else 0.0,
                    't_statistic': ttest['t_statistic'],
                    'df': ttest['df'],
                    'p_value_raw': ttest['p_value'],
                    'significant': bool(ttest['p_value'] < 0.05),
                    'ci_low': ttest['ci_low'],
                    'ci_high': ttest['ci_high'],
                    'mean_diff': ttest['mean_diff'],
                    'std_diff': ttest['std_diff'],
                    'winner': winner,
                    'win_margin_percent': win_margin,
                    'n_pairs': len(direct_values)
                }
                if ttest['p_value'] < 0.05:
                    statistical_results['fixed']['significant_metrics'].append(metric)
    
    # Determine winners for each section
    for section in ['generative', 'fixed']:
        if statistical_results[section]['metric_tests']:
            direct_wins = sum(1 for m in statistical_results[section]['metric_tests'].values() 
                            if m['winner'] == 'Direct MST-SOM')
            hex_wins = sum(1 for m in statistical_results[section]['metric_tests'].values() 
                          if m['winner'] == 'Hexagonal-to-MST')
            
            sig_direct_wins = sum(1 for m in statistical_results[section]['metric_tests'].values() 
                                 if m['winner'] == 'Direct MST-SOM' and m.get('significant', False))
            sig_hex_wins = sum(1 for m in statistical_results[section]['metric_tests'].values() 
                              if m['winner'] == 'Hexagonal-to-MST' and m.get('significant', False))
            
            if sig_direct_wins > sig_hex_wins:
                statistical_results[section]['winner'] = 'Direct MST-SOM'
            elif sig_hex_wins > sig_direct_wins:
                statistical_results[section]['winner'] = 'Hexagonal-to-MST'
            elif direct_wins > hex_wins:
                statistical_results[section]['winner'] = 'Direct MST-SOM (non-significant)'
            elif hex_wins > direct_wins:
                statistical_results[section]['winner'] = 'Hexagonal-to-MST (non-significant)'
            else:
                statistical_results[section]['winner'] = 'Tie'
            
            statistical_results[section]['direct_wins'] = direct_wins
            statistical_results[section]['hex_wins'] = hex_wins
            statistical_results[section]['significant_direct_wins'] = sig_direct_wins
            statistical_results[section]['significant_hex_wins'] = sig_hex_wins
    
    # Determine overall winner
    gen_winner = statistical_results['generative'].get('winner', 'None')
    fixed_winner = statistical_results['fixed'].get('winner', 'None')
    
    if 'Direct MST-SOM' in gen_winner and 'Direct MST-SOM' in fixed_winner:
        statistical_results['overall']['winner'] = 'Direct MST-SOM'
        statistical_results['overall']['recommendation'] = (
            'Direct MST-SOM is consistently superior across both generative and fixed datasets. '
            'The dynamic MST updates during training lead to better clustering structure.'
        )
    elif 'Hexagonal-to-MST' in gen_winner and 'Hexagonal-to-MST' in fixed_winner:
        statistical_results['overall']['winner'] = 'Hexagonal-to-MST'
        statistical_results['overall']['recommendation'] = (
            'Hexagonal-to-MST is consistently superior across both generative and fixed datasets. '
            'The stable hexagonal training produces weights that form better MSTs.'
        )
    else:
        statistical_results['overall']['winner'] = 'Mixed results'
        statistical_results['overall']['recommendation'] = (
            f'Results vary by dataset type. Generative datasets favor {gen_winner}, '
            f'while fixed datasets favor {fixed_winner}. Consider your specific use case.'
        )
    
    return statistical_results


def save_statistical_report_separated(results: Dict[str, Any], args):
    """Save statistical report with separate sections for generative and fixed datasets."""
    
    output_file = os.path.join(args.output_dir, args.stats_output)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MST STATISTICAL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Difficulty: {args.stats_difficulty}\n")
        f.write(f"Iterations per dataset: {args.stats_iterations}\n")
        f.write(f"Grid size: {args.grid_size}x{args.grid_size}\n\n")
        
        # Generative datasets section
        if results['generative']['n_datasets'] > 0:
            f.write("="*60 + "\n")
            f.write("GENERATIVE DATASETS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Datasets tested: {', '.join(results['generative']['datasets'])}\n")
            f.write(f"Repeats per dataset: {results['generative']['n_repeats_per_dataset']}\n")
            f.write(f"Total paired samples: {sum(m['n_pairs'] for m in results['generative']['metric_tests'].values())//len(results['generative']['metric_tests']) if results['generative']['metric_tests'] else 0}\n\n")
            
            f.write("Metric-by-Metric Comparison:\n")
            f.write("-"*40 + "\n")
            
            for metric, test in results['generative']['metric_tests'].items():
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Direct MST-SOM: {test['direct_mean']:.4f} ± {test['direct_std']:.4f}\n")
                f.write(f"  Hexagonal-to-MST: {test['hex_mean']:.4f} ± {test['hex_std']:.4f}\n")
                f.write(f"  Winner: {test['winner']} ({test['win_margin_percent']:.1f}% better)\n")
                f.write(f"  Statistical test:\n")
                f.write(f"    t-statistic: {test['t_statistic']:.3f}\n")
                f.write(f"    p-value: {test['p_value_raw']:.4f}\n")
                f.write(f"    95% CI of difference: [{test['ci_low']:.4f}, {test['ci_high']:.4f}]\n")
                f.write(f"    Significant: {'Yes' if test['significant'] else 'No'}\n")
            
            f.write("\n" + "="*40 + "\n")
            f.write("Generative Summary:\n")
            f.write(f"  Direct MST-SOM wins: {results['generative']['direct_wins']} metrics "
                   f"({results['generative']['significant_direct_wins']} significant)\n")
            f.write(f"  Hexagonal-to-MST wins: {results['generative']['hex_wins']} metrics "
                   f"({results['generative']['significant_hex_wins']} significant)\n")
            f.write(f"  Significant differences in: {', '.join(results['generative']['significant_metrics']) if results['generative']['significant_metrics'] else 'None'}\n")
            f.write(f"  Overall winner: {results['generative']['winner']}\n")
        
        # Fixed datasets section
        if results['fixed']['n_datasets'] > 0:
            f.write("\n" + "="*60 + "\n")
            f.write("FIXED DATASETS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Datasets tested: {', '.join(results['fixed']['datasets'])}\n")
            f.write(f"Total paired samples: {results['fixed']['n_datasets']}\n\n")
            
            f.write("Metric-by-Metric Comparison:\n")
            f.write("-"*40 + "\n")
            
            for metric, test in results['fixed']['metric_tests'].items():
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Direct MST-SOM: {test['direct_mean']:.4f} ± {test['direct_std']:.4f}\n")
                f.write(f"  Hexagonal-to-MST: {test['hex_mean']:.4f} ± {test['hex_std']:.4f}\n")
                f.write(f"  Winner: {test['winner']} ({test['win_margin_percent']:.1f}% better)\n")
                f.write(f"  Statistical test:\n")
                f.write(f"    t-statistic: {test['t_statistic']:.3f}\n")
                f.write(f"    p-value: {test['p_value_raw']:.4f}\n")
                f.write(f"    95% CI of difference: [{test['ci_low']:.4f}, {test['ci_high']:.4f}]\n")
                f.write(f"    Significant: {'Yes' if test['significant'] else 'No'}\n")
            
            f.write("\n" + "="*40 + "\n")
            f.write("Fixed Summary:\n")
            f.write(f"  Direct MST-SOM wins: {results['fixed']['direct_wins']} metrics "
                   f"({results['fixed']['significant_direct_wins']} significant)\n")
            f.write(f"  Hexagonal-to-MST wins: {results['fixed']['hex_wins']} metrics "
                   f"({results['fixed']['significant_hex_wins']} significant)\n")
            f.write(f"  Significant differences in: {', '.join(results['fixed']['significant_metrics']) if results['fixed']['significant_metrics'] else 'None'}\n")
            f.write(f"  Overall winner: {results['fixed']['winner']}\n")
        
        # Overall conclusion
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Winner: {results['overall']['winner']}\n\n")
        f.write("Recommendation:\n")
        f.write(results['overall']['recommendation'] + "\n")
        
        # Statistical interpretation
        f.write("\n" + "-"*60 + "\n")
        f.write("Statistical Notes:\n")
        f.write("- Paired t-tests used for within-dataset/repeat comparisons\n")
        f.write("- Significance threshold: α = 0.05 (no multiple comparison adjustment)\n")
        f.write("- Generative datasets use replicate-level pairing\n")
        f.write("- Fixed datasets use dataset-level pairing\n")
    
    logger.info(f"\nStatistical report saved to {output_file}")
