#!/usr/bin/env python3
"""
Analyze individual topology metrics using Mann-Whitney U tests
Compares batch vs colors for all 4 topology measures separately
"""

import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import required modules
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.clean import clean_data, remove_zero_topographic_error
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.filter_best import filter_top_performers
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.normalization import normalize_all_metrics
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.statistical_analysis import compare_groups
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.visualization_helpers import add_analysis_columns

def analyze_individual_topology_metrics(percentile=10):
    """
    Run Mann-Whitney U tests on all 4 topology metrics individually
    
    Parameters:
    -----------
    percentile : int
        Percentile cutoff for filtering (default: 10, meaning keep top 10% performers)
        Use None to skip filtering
    """
    
    # Step 1: Load and clean data
    print("="*70)
    print("INDIVIDUAL TOPOLOGY METRICS ANALYSIS")
    print("="*70)
    
    print("\n1. Loading and cleaning data...")
    print("-"*40)
    df = pd.read_csv('Data/pareto_front_results.csv')
    print(f"Initial shape: {df.shape}")
    
    # Clean the data
    df_clean = clean_data(df)
    print(f"Shape after cleaning: {df_clean.shape}")
    
    # Remove zero topographic error
    df_clean = remove_zero_topographic_error(df_clean)
    print(f"Shape after removing zero TE: {df_clean.shape}")
    
    # Step 2: Filter if requested
    if percentile is not None:
        print(f"\n2. Filtering to top {percentile}% performers per dataset...")
        print("-"*40)
        df_filtered = filter_top_performers(df_clean, percentile=percentile)
        print(f"Shape after filtering: {df_filtered.shape}")
    else:
        df_filtered = df_clean
        print("\n2. No filtering applied - using all data")
    
    # Step 3: Normalize
    print("\n3. Normalizing metrics...")
    print("-"*40)
    df_normalized = normalize_all_metrics(df_filtered)
    print(f"Shape after normalization: {df_normalized.shape}")
    
    # Step 4: Add analysis columns (including algorithm from processing_type)
    print("\n4. Adding analysis columns...")
    print("-"*40)
    df_final = add_analysis_columns(df_normalized)
    print(f"Shape after adding analysis columns: {df_final.shape}")
    
    # Define the 4 topology metrics (using normalized versions)
    topology_metrics = [
        'topographic_error_normalized',
        'neighborhood_preservation_normalized', 
        'distortion_measure_normalized',
        'topographic_function_normalized'
    ]
    
    # Get unique sampling methods
    sampling_col = 'sampling_method_parsed'
    if sampling_col not in df_final.columns:
        if 'sampling_method' in df_final.columns:
            sampling_col = 'sampling_method'
        elif 'sampling_method_final' in df_final.columns:
            sampling_col = 'sampling_method_final'
    
    sampling_methods = df_final[sampling_col].dropna().unique()
    
    print("\n" + "="*70)
    print("MANN-WHITNEY U TESTS: INDIVIDUAL TOPOLOGY METRICS")
    print("="*70)
    
    # Store all results
    all_results = {}
    
    # For each sampling method
    for sampling in sampling_methods:
        print(f"\n{'='*60}")
        print(f"SAMPLING METHOD: {sampling.upper()}")
        print(f"{'='*60}")
        
        # Filter data for this sampling method
        df_sampling = df_final[df_final[sampling_col] == sampling]
        
        # Get available algorithms for this sampling method
        algorithms = df_sampling['algorithm'].unique()
        if len(algorithms) < 2:
            print(f"  Skipping {sampling}: Only {len(algorithms)} algorithm(s) found, need at least 2 for comparison")
            continue
        
        # Use the first two algorithms for comparison (or you could compare all pairs)
        algo1, algo2 = algorithms[0], algorithms[1]
        
        sampling_results = {}
        
        # Test each topology metric
        for metric in topology_metrics:
            print(f"\n{'-'*50}")
            # Clean metric name for display
            metric_display = metric.replace('_normalized', '').replace('_', ' ').upper()
            print(f"METRIC: {metric_display}")
            print(f"{'-'*50}")
            
            # Extract data for each algorithm
            data1 = df_sampling[df_sampling['algorithm'] == algo1][metric].values
            data2 = df_sampling[df_sampling['algorithm'] == algo2][metric].values
            
            # Print descriptive statistics
            print(f"\nDescriptive Statistics:")
            print(f"  {algo1} (n={len(data1)}):")
            print(f"    Mean:   {data1.mean():.4f}")
            print(f"    Median: {pd.Series(data1).median():.4f}")
            print(f"    Std:    {data1.std():.4f}")
            
            print(f"\n  {algo2} (n={len(data2)}):")
            print(f"    Mean:   {data2.mean():.4f}")
            print(f"    Median: {pd.Series(data2).median():.4f}")
            print(f"    Std:    {data2.std():.4f}")
            
            # Perform Mann-Whitney U test
            from scipy import stats
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Calculate effect size
            n1, n2 = len(data1), len(data2)
            effect_size = 1 - (2*statistic)/(n1*n2)
            
            # Determine significance
            is_significant = p_value < 0.05
            
            # Determine which is better (lower is better for topology)
            mean1 = data1.mean()
            mean2 = data2.mean()
            
            if mean1 < mean2:
                better = algo1.upper()
                difference = ((mean2 - mean1) / mean2) * 100
            else:
                better = algo2.upper()
                difference = ((mean1 - mean2) / mean1) * 100
            
            # Print test results
            print(f"\nMann-Whitney U Test Results:")
            print(f"  U Statistic:   {statistic:.2f}")
            print(f"  P-value:       {p_value:.6f}")
            print(f"  Effect size:   {abs(effect_size):.3f}")
            print(f"  Significant:   {'YES' if is_significant else 'NO'} (α = 0.05)")
            
            if is_significant:
                print(f"\n  → {better} performs better (lower is better)")
                print(f"  → Difference: {difference:.1f}% lower mean value")
            else:
                print(f"\n  → No significant difference between algorithms")
            
            # Store results
            sampling_results[metric] = {
                f'{algo1}_mean': mean1,
                f'{algo1}_median': pd.Series(data1).median(),
                f'{algo2}_mean': mean2,
                f'{algo2}_median': pd.Series(data2).median(),
                'p_value': p_value,
                'significant': is_significant,
                'effect_size': effect_size,
                'better_algorithm': better.lower() if is_significant else 'none',
                'algorithms_compared': f"{algo1} vs {algo2}"
            }
        
        all_results[sampling] = sampling_results
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: SIGNIFICANT DIFFERENCES BY METRIC")
    print("="*70)
    
    for sampling in all_results:
        print(f"\n{sampling.upper()} Sampling:")
        for metric in topology_metrics:
            if metric in all_results[sampling]:
                result = all_results[sampling][metric]
                metric_display = metric.replace('_normalized', '').replace('_', ' ')
                if result['significant']:
                    print(f"  {metric_display:30} → {result['better_algorithm'].upper()} better (p={result['p_value']:.4f})")
                else:
                    print(f"  {metric_display:30} → No significant difference (p={result['p_value']:.4f})")
    
    return all_results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze individual topology metrics')
    parser.add_argument('--percentile', '-p', type=int, default=10,
                        help='Percentile cutoff for filtering (default: 10)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip filtering entirely (use all data)')
    args = parser.parse_args()
    
    # Determine filter percentage
    if args.no_filter:
        filter_percent = None
        print(f"Running analysis with NO FILTERING (using all data)")
    else:
        filter_percent = args.percentile
        print(f"Running analysis with {filter_percent}% percentile cutoff")
    
    # Run the analysis
    results = analyze_individual_topology_metrics(percentile=filter_percent)
    
    print("\n" + "="*70)
    print("Analysis complete!")