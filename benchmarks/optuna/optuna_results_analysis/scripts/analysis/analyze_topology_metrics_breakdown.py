#!/usr/bin/env python3
"""
Analyze the individual topology metrics for best performers.
Compares topographic_error, neighborhood_preservation, distortion_measure, 
and topographic_function between batch and colors processing within each dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import argparse
warnings.filterwarnings('ignore')

def load_best_performer_details(architecture=''):
    """Load the complete records of best performers."""
    suffix = f"_{architecture}" if architecture else ""
    # Try Results folder first, then Data folder for backward compatibility
    results_file = Path(f"Results/tables/best_performers_complete_records{suffix}.csv")
    data_file = Path(f"Data/best_performers_complete_records{suffix}.csv")
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        print(f"Loaded complete records from {results_file}")
    elif data_file.exists():
        df = pd.read_csv(data_file)
        print(f"Loaded complete records from {data_file}")
    else:
        # Fall back to loading original data and filtering
        print("Complete records file not found. Loading from original data...")
        suffix = f"_{architecture}" if architecture else ""
        best_performers = pd.read_csv(f"Data/best_performers_by_sampling{suffix}.csv")
        original_data = pd.read_csv(f"Data/processed_data_with_overall_score{suffix}.csv")
        
        # Filter original data for best scenarios
        best_scenario_ids = best_performers['Best_Scenario'].tolist()
        df = original_data[original_data['scenario_id'].isin(best_scenario_ids)]
        
        # Get the best record for each scenario
        best_records = []
        for scenario_id in best_scenario_ids:
            scenario_records = df[df['scenario_id'] == scenario_id]
            if not scenario_records.empty:
                best_record = scenario_records.loc[scenario_records['Normalized_Overall_Score'].idxmax()]
                best_records.append(best_record)
        
        df = pd.DataFrame(best_records)
    
    return df

def analyze_metric_differences(df):
    """
    Analyze differences in individual topology metrics between batch and colors.
    """
    print("\n" + "="*80)
    print("TOPOLOGY METRICS COMPARISON: BATCH vs COLORS")
    print("="*80)
    
    # Define the topology metrics to analyze
    topology_metrics = [
        'topographic_error',
        'neighborhood_preservation', 
        'distortion_measure',
        'topographic_function'
    ]
    
    # Also check for normalized versions if they exist
    normalized_metrics = []
    for metric in topology_metrics:
        norm_col = f"{metric}_normalized"
        if norm_col in df.columns:
            normalized_metrics.append(norm_col)
    
    # Overall comparison across all datasets
    print("\n" + "-"*50)
    print("OVERALL COMPARISON (All Datasets)")
    print("-"*50)
    
    results_summary = []
    
    for metric in topology_metrics:
        if metric in df.columns:
            batch_values = df[df['processing_type'] == 'batch'][metric].dropna()
            colors_values = df[df['processing_type'] == 'colors'][metric].dropna()
            
            if len(batch_values) > 0 and len(colors_values) > 0:
                # Calculate statistics
                batch_mean = batch_values.mean()
                batch_std = batch_values.std()
                colors_mean = colors_values.mean()
                colors_std = colors_values.std()
                
                # Perform Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(batch_values, colors_values, alternative='two-sided')
                
                # Calculate effect size (rank biserial correlation)
                n1, n2 = len(batch_values), len(colors_values)
                effect_size = 1 - (2*u_stat)/(n1*n2)
                
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Batch:  {batch_mean:.6f} ± {batch_std:.6f}")
                print(f"  Colors: {colors_mean:.6f} ± {colors_std:.6f}")
                print(f"  Difference: {colors_mean - batch_mean:+.6f} ({(colors_mean - batch_mean)/batch_mean*100:+.1f}%)")
                print(f"  P-value: {p_value:.6f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
                print(f"  Effect size: {effect_size:.3f}")
                
                # Interpret which is better (depends on metric)
                if metric in ['topographic_error', 'distortion_measure']:
                    better = "Batch" if batch_mean < colors_mean else "Colors"
                    print(f"  Better: {better} (lower is better)")
                elif metric == 'neighborhood_preservation':
                    better = "Batch" if batch_mean > colors_mean else "Colors"  
                    print(f"  Better: {better} (higher is better)")
                else:  # topographic_function
                    print(f"  Interpretation: Metric-specific")
                
                results_summary.append({
                    'metric': metric,
                    'batch_mean': batch_mean,
                    'colors_mean': colors_mean,
                    'difference': colors_mean - batch_mean,
                    'percent_diff': (colors_mean - batch_mean)/batch_mean*100,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': effect_size
                })
    
    return pd.DataFrame(results_summary)

def analyze_within_dataset_differences(df):
    """
    Analyze metric differences between batch and colors within each dataset.
    """
    print("\n" + "="*80)
    print("WITHIN-DATASET METRIC DIFFERENCES")
    print("="*80)
    
    topology_metrics = [
        'topographic_error',
        'neighborhood_preservation',
        'distortion_measure', 
        'topographic_function'
    ]
    
    datasets = df['dataset'].unique()
    
    # Store results for each dataset
    dataset_results = {}
    
    for dataset in sorted(datasets):
        print(f"\n" + "-"*50)
        print(f"Dataset: {dataset.upper()}")
        print("-"*50)
        
        dataset_df = df[df['dataset'] == dataset]
        batch_df = dataset_df[dataset_df['processing_type'] == 'batch']
        colors_df = dataset_df[dataset_df['processing_type'] == 'colors']
        
        metric_diffs = {}
        
        for metric in topology_metrics:
            if metric in df.columns:
                # Get mean values for each processing type
                batch_vals = batch_df[metric].dropna()
                colors_vals = colors_df[metric].dropna()
                
                if len(batch_vals) > 0 and len(colors_vals) > 0:
                    batch_mean = batch_vals.mean()
                    colors_mean = colors_vals.mean()
                    diff = colors_mean - batch_mean
                    percent_diff = (diff / batch_mean * 100) if batch_mean != 0 else 0
                    
                    metric_diffs[metric] = {
                        'batch': batch_mean,
                        'colors': colors_mean,
                        'difference': diff,
                        'percent_diff': percent_diff
                    }
                    
                    # Determine which is better
                    if metric in ['topographic_error', 'distortion_measure']:
                        better = "batch" if batch_mean < colors_mean else "colors"
                    elif metric == 'neighborhood_preservation':
                        better = "batch" if batch_mean > colors_mean else "colors"
                    else:
                        better = "unclear"
                    
                    metric_diffs[metric]['better'] = better
                    
                    print(f"\n  {metric.replace('_', ' ').title()}:")
                    print(f"    Batch:  {batch_mean:.6f}")
                    print(f"    Colors: {colors_mean:.6f}")
                    print(f"    Diff:   {diff:+.6f} ({percent_diff:+.1f}%)")
                    print(f"    Better: {better}")
        
        dataset_results[dataset] = metric_diffs
        
        # Summarize which processing type is better overall for this dataset
        batch_wins = sum(1 for m in metric_diffs.values() if m.get('better') == 'batch')
        colors_wins = sum(1 for m in metric_diffs.values() if m.get('better') == 'colors')
        
        print(f"\n  Summary for {dataset}:")
        print(f"    Batch better in: {batch_wins} metrics")
        print(f"    Colors better in: {colors_wins} metrics")
        
        if batch_wins > colors_wins:
            print(f"    Overall: BATCH performs better")
        elif colors_wins > batch_wins:
            print(f"    Overall: COLORS performs better")
        else:
            print(f"    Overall: MIXED performance")
    
    return dataset_results

def analyze_metric_correlations(df):
    """
    Analyze correlations between topology metrics.
    """
    print("\n" + "="*80)
    print("METRIC CORRELATIONS")
    print("="*80)
    
    topology_metrics = [
        'topographic_error',
        'neighborhood_preservation',
        'distortion_measure',
        'topographic_function'
    ]
    
    # Filter to only include these columns
    available_metrics = [m for m in topology_metrics if m in df.columns]
    
    if len(available_metrics) > 1:
        # Calculate correlation matrix
        corr_matrix = df[available_metrics].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Find strong correlations
        print("\n" + "-"*50)
        print("Strong Correlations (|r| > 0.5):")
        print("-"*50)
        
        for i in range(len(available_metrics)):
            for j in range(i+1, len(available_metrics)):
                metric1 = available_metrics[i]
                metric2 = available_metrics[j]
                corr_val = corr_matrix.loc[metric1, metric2]
                
                if abs(corr_val) > 0.5:
                    print(f"  {metric1} <-> {metric2}: r = {corr_val:.3f}")
                    if corr_val > 0:
                        print(f"    -> Positive correlation: metrics tend to move together")
                    else:
                        print(f"    -> Negative correlation: trade-off between metrics")
        
        return corr_matrix
    else:
        print("Not enough metrics available for correlation analysis")
        return None

def create_detailed_visualizations(df, dataset_results, architecture=''):
    """
    Create comprehensive visualizations for metric analysis.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    topology_metrics = [
        'topographic_error',
        'neighborhood_preservation',
        'distortion_measure',
        'topographic_function'
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in topology_metrics if m in df.columns]
    
    # Create figure with subplots
    n_metrics = len(available_metrics)
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Box plots for each metric (top row)
    for i, metric in enumerate(available_metrics):
        ax = plt.subplot(3, n_metrics, i+1)
        
        # Prepare data for box plot
        batch_data = df[df['processing_type'] == 'batch'][metric].dropna()
        colors_data = df[df['processing_type'] == 'colors'][metric].dropna()
        
        bp = ax.boxplot([batch_data, colors_data], labels=['Batch', 'Colors'], patch_artist=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # 2. Violin plots by dataset (middle row)
    for i, metric in enumerate(available_metrics):
        ax = plt.subplot(3, n_metrics, n_metrics+i+1)
        
        # Create violin plot
        sns.violinplot(data=df, x='processing_type', y=metric, ax=ax, palette=['lightblue', 'lightcoral'])
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=10)
        ax.set_xlabel('Processing Type')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Difference plots (bottom row)
    for i, metric in enumerate(available_metrics):
        ax = plt.subplot(3, n_metrics, 2*n_metrics+i+1)
        
        # Calculate differences for each dataset
        datasets = []
        differences = []
        
        for dataset in sorted(df['dataset'].unique()):
            if dataset in dataset_results and metric in dataset_results[dataset]:
                datasets.append(dataset)
                differences.append(dataset_results[dataset][metric]['percent_diff'])
        
        # Create bar plot
        colors = ['green' if d < 0 else 'red' for d in differences]
        bars = ax.bar(range(len(datasets)), differences, color=colors, alpha=0.6)
        
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{metric.replace("_", " ").title()}\n% Diff (Colors - Batch)', fontsize=10)
        ax.set_ylabel('% Difference')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, differences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=7)
    
    plt.suptitle('Topology Metrics Analysis: Batch vs Colors Processing', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path("Results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{architecture}" if architecture else ""
    output_file = figures_dir / f"topology_metrics_breakdown{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Create correlation heatmap
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    if len(available_metrics) > 1:
        corr_matrix = df[available_metrics].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Topology Metrics')
        
        suffix = f"_{architecture}" if architecture else ""
        output_file2 = figures_dir / f"topology_metrics_correlation{suffix}.png"
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {output_file2}")
    
    plt.show()
    
    return fig

def generate_detailed_report(df, results_summary, dataset_results, corr_matrix, architecture=''):
    """
    Generate a comprehensive report of the metric analysis.
    """
    reports_dir = Path("Results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{architecture}" if architecture else ""
    output_file = reports_dir / f"topology_metrics_breakdown_report{suffix}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TOPOLOGY METRICS BREAKDOWN ANALYSIS\n")
        f.write("Comparing Batch vs Colors Processing\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*50 + "\n\n")
        
        # Count which processing type is better for each metric
        if not results_summary.empty:
            for _, row in results_summary.iterrows():
                metric = row['metric']
                if row['batch_mean'] < row['colors_mean']:
                    if metric in ['topographic_error', 'distortion_measure']:
                        better = "Batch"
                    else:
                        better = "Colors"
                else:
                    if metric in ['topographic_error', 'distortion_measure']:
                        better = "Colors"
                    else:
                        better = "Batch"
                
                f.write(f"* {metric.replace('_', ' ').title()}: {better} performs better ")
                f.write(f"({abs(row['percent_diff']):.1f}% difference, ")
                f.write(f"{'significant' if row['significant'] else 'not significant'} p={row['p_value']:.4f})\n")
        
        f.write("\n")
        
        # Overall Statistics
        f.write("\nOVERALL STATISTICS (All Datasets Combined)\n")
        f.write("-"*50 + "\n\n")
        
        if not results_summary.empty:
            f.write(results_summary.to_string(index=False))
        
        f.write("\n\n")
        
        # Dataset-specific patterns
        f.write("DATASET-SPECIFIC PATTERNS\n")
        f.write("-"*50 + "\n\n")
        
        dataset_summary = []
        for dataset, metrics in dataset_results.items():
            batch_better = 0
            colors_better = 0
            
            for metric_name, metric_data in metrics.items():
                if metric_data.get('better') == 'batch':
                    batch_better += 1
                elif metric_data.get('better') == 'colors':
                    colors_better += 1
            
            overall = "Batch" if batch_better > colors_better else "Colors" if colors_better > batch_better else "Mixed"
            dataset_summary.append({
                'Dataset': dataset,
                'Batch_Better_Count': batch_better,
                'Colors_Better_Count': colors_better,
                'Overall_Better': overall
            })
        
        summary_df = pd.DataFrame(dataset_summary)
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        # Detailed dataset breakdown
        f.write("DETAILED DATASET BREAKDOWN\n")
        f.write("-"*50 + "\n\n")
        
        for dataset, metrics in sorted(dataset_results.items()):
            f.write(f"{dataset.upper()}:\n")
            for metric_name, metric_data in metrics.items():
                f.write(f"  {metric_name}:\n")
                f.write(f"    Batch:  {metric_data['batch']:.6f}\n")
                f.write(f"    Colors: {metric_data['colors']:.6f}\n")
                f.write(f"    Diff:   {metric_data['difference']:+.6f} ({metric_data['percent_diff']:+.1f}%)\n")
                f.write(f"    Better: {metric_data['better']}\n")
            f.write("\n")
        
        # Correlation analysis
        if corr_matrix is not None:
            f.write("\nMETRIC CORRELATIONS\n")
            f.write("-"*50 + "\n\n")
            f.write("Correlation Matrix:\n")
            f.write(corr_matrix.round(3).to_string())
            f.write("\n\n")
            
            f.write("Interpretation:\n")
            # Find strongest correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    metric1 = corr_matrix.columns[i]
                    metric2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if abs(corr_val) > 0.5:
                        f.write(f"* {metric1} <-> {metric2}: r={corr_val:.3f} - ")
                        if corr_val > 0:
                            f.write("Strong positive correlation\n")
                        else:
                            f.write("Strong negative correlation\n")
        
        # Key insights
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-"*50 + "\n\n")
        
        # Count overall patterns
        batch_dominant_datasets = sum(1 for d in dataset_summary if d['Overall_Better'] == 'Batch')
        colors_dominant_datasets = sum(1 for d in dataset_summary if d['Overall_Better'] == 'Colors')
        
        f.write(f"1. Dataset Preferences:\n")
        f.write(f"   - {batch_dominant_datasets} datasets perform better with Batch processing\n")
        f.write(f"   - {colors_dominant_datasets} datasets perform better with Colors processing\n\n")
        
        f.write(f"2. Metric-Specific Patterns:\n")
        if not results_summary.empty:
            for _, row in results_summary.iterrows():
                if row['significant']:
                    f.write(f"   - {row['metric']}: Significant difference (p={row['p_value']:.4f})\n")
        
        f.write("\n")
    
    print(f"\nDetailed report saved to {output_file}")
    return output_file

def compare_within_sampling_methods(df):
    """
    Compare batch vs colors within each sampling method across all datasets.
    Shows how similar or different the algorithms are when using the same sampling approach.
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON WITHIN EACH SAMPLING METHOD")
    print("="*80)
    
    metrics = [
        'quantization_error',
        'topographic_error',
        'neighborhood_preservation',
        'distortion_measure',
        'topographic_function'
    ]
    
    sampling_methods = ['full', 'random', 'hdsssom']
    
    all_comparisons = []
    
    for sampling in sampling_methods:
        print(f"\n{'='*70}")
        print(f"SAMPLING METHOD: {sampling.upper()}")
        print(f"{'='*70}")
        
        # Filter for this sampling method
        sampling_df = df[df['sampling_method'] == sampling]
        
        if sampling_df.empty:
            print(f"No data for {sampling}")
            continue
        
        datasets = sorted(sampling_df['dataset'].unique())
        
        for dataset in datasets:
            print(f"\n{dataset.upper()}:")
            print("-" * 50)
            
            # Get batch and colors for this dataset and sampling
            batch_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                    (sampling_df['processing_type'] == 'batch')]
            colors_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                     (sampling_df['processing_type'] == 'colors')]
            
            if not batch_data.empty and not colors_data.empty:
                comparison = {
                    'dataset': dataset,
                    'sampling_method': sampling
                }
                
                for metric in metrics:
                    if metric in batch_data.columns and metric in colors_data.columns:
                        batch_val = batch_data[metric].iloc[0]
                        colors_val = colors_data[metric].iloc[0]
                        
                        if not pd.isna(batch_val) and not pd.isna(colors_val):
                            diff = colors_val - batch_val
                            percent_diff = (diff / batch_val * 100) if batch_val != 0 else 0
                            
                            # Determine similarity
                            if abs(percent_diff) < 5:
                                similarity = "VERY SIMILAR"
                                symbol = "≈≈"
                            elif abs(percent_diff) < 15:
                                similarity = "SIMILAR"
                                symbol = "≈"
                            elif abs(percent_diff) < 30:
                                similarity = "DIFFERENT"
                                symbol = "≠"
                            else:
                                similarity = "VERY DIFFERENT"
                                symbol = "≠≠"
                            
                            # Determine which is better
                            if metric in ['quantization_error', 'topographic_error', 'distortion_measure']:
                                better = "Batch" if batch_val < colors_val else "Colors"
                            elif metric == 'neighborhood_preservation':
                                better = "Batch" if batch_val > colors_val else "Colors"
                            else:
                                better = "N/A"
                            
                            print(f"\n  {metric.replace('_', ' ').title()}:")
                            print(f"    Batch:  {batch_val:10.6f}")
                            print(f"    Colors: {colors_val:10.6f}")
                            print(f"    Diff:   {diff:+10.6f} ({percent_diff:+6.1f}%)")
                            print(f"    Status: {symbol} {similarity} - {better} better")
                            
                            comparison[f'{metric}_batch'] = batch_val
                            comparison[f'{metric}_colors'] = colors_val
                            comparison[f'{metric}_diff'] = diff
                            comparison[f'{metric}_percent'] = percent_diff
                            comparison[f'{metric}_similarity'] = similarity
                            comparison[f'{metric}_better'] = better
                
                all_comparisons.append(comparison)
    
    # Create summary
    print("\n" + "="*80)
    print("SIMILARITY SUMMARY")
    print("="*80)
    
    comparisons_df = pd.DataFrame(all_comparisons)
    
    for metric in metrics:
        if f'{metric}_similarity' in comparisons_df.columns:
            print(f"\n{metric.replace('_', ' ').title()}:")
            
            for sampling in sampling_methods:
                sampling_data = comparisons_df[comparisons_df['sampling_method'] == sampling]
                if not sampling_data.empty and f'{metric}_similarity' in sampling_data.columns:
                    similarity_counts = sampling_data[f'{metric}_similarity'].value_counts()
                    total = len(sampling_data)
                    
                    print(f"\n  {sampling.upper()}:")
                    for category in ['VERY SIMILAR', 'SIMILAR', 'DIFFERENT', 'VERY DIFFERENT']:
                        count = similarity_counts.get(category, 0)
                        print(f"    {category:15}: {count}/{total} ({count/total*100:.0f}%)")
                    
                    # Which algorithm tends to be better?
                    if f'{metric}_better' in sampling_data.columns:
                        better_counts = sampling_data[f'{metric}_better'].value_counts()
                        batch_wins = better_counts.get('Batch', 0)
                        colors_wins = better_counts.get('Colors', 0)
                        print(f"    Batch better: {batch_wins}, Colors better: {colors_wins}")
    
    return comparisons_df

def main(architecture=''):
    """
    Main analysis pipeline for topology metrics breakdown.
    """
    arch_name = f" ({architecture.upper()})" if architecture else ""
    print("="*80)
    print(f"TOPOLOGY METRICS BREAKDOWN ANALYSIS{arch_name}")
    print("="*80)
    
    # Load data
    print("\nLoading best performer complete records...")
    df = load_best_performer_details(architecture)
    print(f"Loaded {len(df)} records")
    print(f"Columns available: {', '.join(df.columns[:10])}...")
    
    # Perform analyses
    results_summary = analyze_metric_differences(df)
    dataset_results = analyze_within_dataset_differences(df)
    corr_matrix = analyze_metric_correlations(df)
    
    # NEW: Compare within sampling methods
    print("\n" + "="*80)
    print("NEW ANALYSIS: Comparing algorithms within each sampling method")
    print("="*80)
    sampling_comparisons = compare_within_sampling_methods(df)
    
    # Create visualizations
    fig = create_detailed_visualizations(df, dataset_results, architecture)
    
    # Generate report
    report_file = generate_detailed_report(df, results_summary, dataset_results, corr_matrix, architecture)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    suffix = f"_{architecture}" if architecture else ""
    print("\nGenerated files:")
    print(f"  - Results/figures/topology_metrics_breakdown{suffix}.png (metric comparisons)")
    print(f"  - Results/figures/topology_metrics_correlation{suffix}.png (correlation heatmap)")
    print(f"  - Results/reports/topology_metrics_breakdown_report{suffix}.txt (detailed report)")
    
    return df, results_summary, dataset_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze topology metrics breakdown')
    parser.add_argument('--architecture', '-a', type=str, default='',
                        help='Architecture to analyze (mst or hexagonal)')
    args = parser.parse_args()
    
    df, results_summary, dataset_results = main(args.architecture)