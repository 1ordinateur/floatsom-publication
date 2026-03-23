import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def mann_whitney_u_test(group1, group2, group1_name, group2_name):
    """
    Perform a Mann-Whitney U test to compare the distribution of a measure between two groups. (non-parametric, two-tailed)

    Parameters:
    group1: numpy array of the first group
    group2: numpy array of the second group
    group1_name: string, name of the first group
    group2_name: string, name of the second group

    Returns:
    dict with test results
    """

    # Perform the test
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Calculate effect size (rank biserial correlation)
    n1 = len(group1)
    n2 = len(group2)
    effect_size = 1-(2*statistic)/(n1*n2)

    # Prepare Results Dictionary
    results = {
        'test_statistic': 'Mann-Whitney U',
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_size': n1,
        'group2_size': n2,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_median': np.median(group1),
        'group2_median': np.median(group2),
        'group1_std': np.std(group1),
        'group2_std': np.std(group2),
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'significance': 'significant' if p_value < 0.05 else 'not significant'
    }

    return results


def write_results(results, filename='results.txt', metric_type=None):
    """
    Write the results to a text file in Results/Explanations folder

    Parameters:
    results: dict with test results
    filename: string, name of the file to save (will be saved in Results/Explanations/)
    metric_type: string, 'topology' or 'quantization_error' to add specific interpretation
    """

    # Build the full filepath
    filepath = os.path.join('Results', 'Explanations', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"STATISTICAL ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")

        f.write(f"Test: {results['test_statistic']}\n")
        f.write(f"Comparison: {results['group1_name']} vs {results['group2_name']}\n\n")

        f.write("NULL HYPOTHESIS\n")
        f.write("-"*30 + "\n")
        f.write(f"H0: There is no difference in distributions between {results['group1_name']} and {results['group2_name']}\n")
        f.write(f"H1: There is a significant difference in distributions between {results['group1_name']} and {results['group2_name']}\n")
        f.write(f"(Two-tailed test at significance level alpha = 0.05)\n\n")

        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"{results['group1_name']}:\n")
        f.write(f"  N = {results['group1_size']}\n")
        f.write(f"  Mean = {results['group1_mean']:.4f}\n")
        f.write(f"  Median = {results['group1_median']:.4f}\n")
        f.write(f"  Std Dev = {results['group1_std']:.4f}\n\n")

        f.write(f"{results['group2_name']}:\n")
        f.write(f"  N = {results['group2_size']}\n")
        f.write(f"  Mean = {results['group2_mean']:.4f}\n")
        f.write(f"  Median = {results['group2_median']:.4f}\n")
        f.write(f"  Std Dev = {results['group2_std']:.4f}\n\n")

        f.write("TEST RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"U Statistic: {results['statistic']:.2f}\n")
        f.write(f"P-value: {results['p_value']:.6f}\n")
        f.write(f"Effect size (rank biserial): {results['effect_size']:.3f}\n\n")

        f.write("INTERPRETATION\n")
        f.write("-"*30 + "\n")
        if results['significance'] == 'significant':
            f.write(f"The difference is STATISTICALLY SIGNIFICANT (p < 0.05)\n")
            f.write(f"We reject the null hypothesis.\n")
        else:
            f.write(f"The difference is NOT statistically significant (p >= 0.05)\n")
            f.write(f"We fail to reject the null hypothesis.\n")
        
        # Add metric-specific interpretation
        if metric_type:
            f.write(f"\nMETRIC-SPECIFIC INTERPRETATION\n")
            f.write("-"*30 + "\n")
            
            # Normalize metric type for comparison
            metric_lower = metric_type.lower().replace('_', '').replace(' ', '')
            
            if 'topology' in metric_lower or 'pc1' in metric_lower:
                f.write("For TOPOLOGY (lower is better):\n")
                if results['group1_mean'] < results['group2_mean']:
                    f.write(f"  → {results['group1_name']} has LOWER mean topology ({results['group1_mean']:.4f}) than {results['group2_name']} ({results['group2_mean']:.4f})\n")
                    f.write(f"  → {results['group1_name']} has LOWER median topology ({results['group1_median']:.4f}) than {results['group2_name']} ({results['group2_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group1_name']} performs BETTER for topology preservation\n")
                else:
                    f.write(f"  → {results['group2_name']} has LOWER mean topology ({results['group2_mean']:.4f}) than {results['group1_name']} ({results['group1_mean']:.4f})\n")
                    f.write(f"  → {results['group2_name']} has LOWER median topology ({results['group2_median']:.4f}) than {results['group1_name']} ({results['group1_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group2_name']} performs BETTER for topology preservation\n")
                        
            elif 'quantization' in metric_lower or 'qe' in metric_lower:
                f.write("For QUANTIZATION ERROR (lower is better):\n")
                if results['group1_mean'] < results['group2_mean']:
                    f.write(f"  → {results['group1_name']} has LOWER mean QE ({results['group1_mean']:.4f}) than {results['group2_name']} ({results['group2_mean']:.4f})\n")
                    f.write(f"  → {results['group1_name']} has LOWER median QE ({results['group1_median']:.4f}) than {results['group2_name']} ({results['group2_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group1_name']} performs BETTER for quantization error\n")
                else:
                    f.write(f"  → {results['group2_name']} has LOWER mean QE ({results['group2_mean']:.4f}) than {results['group1_name']} ({results['group1_mean']:.4f})\n")
                    f.write(f"  → {results['group2_name']} has LOWER median QE ({results['group2_median']:.4f}) than {results['group1_name']} ({results['group1_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group2_name']} performs BETTER for quantization error\n")
            else:
                # Generic metric (assuming lower is better)
                f.write(f"For {metric_type.upper()} (assuming lower is better):\n")
                if results['group1_mean'] < results['group2_mean']:
                    f.write(f"  → {results['group1_name']} has LOWER mean ({results['group1_mean']:.4f}) than {results['group2_name']} ({results['group2_mean']:.4f})\n")
                    f.write(f"  → {results['group1_name']} has LOWER median ({results['group1_median']:.4f}) than {results['group2_name']} ({results['group2_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group1_name']} performs BETTER\n")
                else:
                    f.write(f"  → {results['group2_name']} has LOWER mean ({results['group2_mean']:.4f}) than {results['group1_name']} ({results['group1_mean']:.4f})\n")
                    f.write(f"  → {results['group2_name']} has LOWER median ({results['group2_median']:.4f}) than {results['group1_name']} ({results['group1_median']:.4f})\n")
                    if results['significance'] == 'significant':
                        f.write(f"  → CONCLUSION: {results['group2_name']} performs BETTER\n")
        
        f.write("\n")
        # Effect size interpretation
        abs_effect = abs(results['effect_size'])
        if abs_effect < 0.1:
            effect_interpretation = "negligible"
        elif abs_effect < 0.3:
            effect_interpretation = "small"
        elif abs_effect < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        f.write(f"\nEffect size is {effect_interpretation} (r = {results['effect_size']:.3f})\n")
    
    print(f"Results saved to {filepath}")


def create_boxplot(group1, group2, group1_name='Group 1', group2_name='Group 2', 
                   ylabel='Value', title=None, filename='boxplot.png'):
    """
    Create and save a boxplot comparing two groups in Results/Graphics folder
    
    Parameters:
    group1, group2: array-like data for each group
    group1_name, group2_name: labels for groups
    ylabel: label for y-axis
    title: plot title
    filename: name of the file to save (will be saved in Results/plots/)
    """
    
    # Build the full filepath
    save_path = os.path.join('Results', 'Graphics', filename)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplot
    bp = ax.boxplot([group1, group2], labels=[group1_name, group2_name], 
                    patch_artist=True, notch=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add labels and title
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Comparison: {group1_name} vs {group2_name}')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    ax.scatter([1, 2], [np.mean(group1), np.mean(group2)], 
              color='red', marker='D', s=50, zorder=3, label='Mean')
    ax.legend()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure without displaying
    
    print(f"Boxplot saved to {save_path}")


def compare_groups(df, value_column, group_column, group1_value, group2_value, 
                   output_name='analysis', create_individual_plot=True):
    """
    Complete statistical comparison of two groups using Mann-Whitney U test
    
    Parameters:
    df: pandas DataFrame
    value_column: column with values to compare
    group_column: column that defines the groups
    group1_value: value in group_column for first group
    group2_value: value in group_column for second group
    output_name: base name for output files (e.g., 'sampling_comparison')
    create_individual_plot: whether to create individual boxplot (default True)
    
    Returns:
    dict with test results
    """
    
    # Extract groups
    group1 = df[df[group_column] == group1_value][value_column].values
    group2 = df[df[group_column] == group2_value][value_column].values
    
    print(f"\nComparing '{value_column}' between groups:")
    print(f"  {group1_value} (n={len(group1)})")
    print(f"  {group2_value} (n={len(group2)})")
    
    # Perform test
    results = mann_whitney_u_test(group1, group2, 
                                  group1_name=str(group1_value), 
                                  group2_name=str(group2_value))
    
    # Determine metric type from value_column or output_name
    metric_type = None
    if 'topology' in value_column.lower() or 'topology' in output_name.lower():
        metric_type = 'topology'
    elif 'quantization' in value_column.lower() or 'quantization' in output_name.lower():
        metric_type = 'quantization_error'
    
    # Write results with custom filename
    results_filename = f'{output_name}_results.txt'
    write_results(results, filename=results_filename, metric_type=metric_type)
    
    # Create individual boxplot if requested
    if create_individual_plot:
        plot_filename = f'{output_name}_boxplot.png'
        create_boxplot(group1, group2, 
                       group1_name=str(group1_value),
                       group2_name=str(group2_value),
                       ylabel=value_column.replace('_', ' ').title(),
                       filename=plot_filename)
    
    # Only print quick summary if we're not in batch mode (i.e., creating individual plot)
    if create_individual_plot:
        print(f"\nQuick Summary:")
        print(f"  P-value: {results['p_value']:.6f}")
        print(f"  Significant: {results['significance']}")
        print(f"  Effect size: {abs(results['effect_size']):.3f}")
    
    return results


def perform_algorithm_comparisons(df, group_column='algorithm', group1_value='batch', 
                                 group2_value='colors', metrics_to_test=None,
                                 create_individual_plots=False, save_path='Results/plots/', 
                                 filename_suffix=''):
    """
    Perform Mann-Whitney U tests comparing two groups for each sampling method
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with group and sampling method columns
    group_column : str
        Column name that defines the groups (default: 'algorithm')
    group1_value : str
        First group value to compare (default: 'batch')
    group2_value : str
        Second group value to compare (default: 'colors')
    metrics_to_test : list
        List of metrics to test (default: ['quantization_error', 'Topology_PC1'])
    create_individual_plots : bool
        Whether to create individual plots for each sampling method (default: False)
        
    Returns:
    --------
    dict : Dictionary of test results for each sampling method and metric
    """
    print("\n" + "="*60)
    print(f"MANN-WHITNEY U TESTS: {group1_value} vs {group2_value} by Sampling Method")
    print("="*60)
    
    # Default metrics if not specified
    if metrics_to_test is None:
        candidate_metrics = [
            'quantization_error_holdout_normalized',
            'quantization_error_train_normalized',
            'balanced_qe_normalized'
        ]
        metrics_to_test = [metric for metric in candidate_metrics if metric in df.columns]
    
    if not metrics_to_test:
        print("Warning: No normalized train/holdout metrics available for Mann-Whitney comparison.")
        return {}
    
    # Determine which sampling method column to use
    sampling_col = None
    for col in ['sampling_method', 'sampling_method_final', 'sampling_method_parsed']:
        if col in df.columns:
            sampling_col = col
            break
    
    if sampling_col is None:
        print("Warning: No sampling method column found")
        return {}
    
    # Get unique sampling methods
    sampling_methods = df[sampling_col].dropna().unique()
    
    # Store all results
    all_results = {}
    
    for sampling in sorted(sampling_methods):
        print(f"\n--- Sampling Method: {sampling} ---")
        
        # Filter data for this sampling method
        df_sampling = df[df[sampling_col] == sampling]
        
        # Check if both groups exist for this sampling method
        groups_present = df_sampling[group_column].dropna().unique()
        
        if group1_value in groups_present and group2_value in groups_present:
            # Store results for this sampling method
            sampling_results = {}
            
            # Test each metric
            for metric in metrics_to_test:
                if metric not in df_sampling.columns:
                    print(f"\n  Skipping {metric}: Column not found")
                    continue
                    
                print(f"\n  Testing {metric}:")
                
                # Perform Mann-Whitney U test (without individual files)
                # Extract groups directly to avoid writing individual files
                group1 = df_sampling[df_sampling[group_column] == group1_value][metric].values
                group2 = df_sampling[df_sampling[group_column] == group2_value][metric].values
                
                # Perform test
                results = mann_whitney_u_test(group1, group2, 
                                             group1_name=str(group1_value), 
                                             group2_name=str(group2_value))
                
                sampling_results[metric] = results
            
            # Only create individual combined boxplot if requested
            if create_individual_plots:
                valid_metrics = [m for m in metrics_to_test if m in df_sampling.columns]
                if len(valid_metrics) > 0:
                    print(f"\n  Creating combined boxplot for {sampling}...")
                    create_combined_boxplot(
                        df_sampling,
                        group_column=group_column,
                        group1_value=group1_value,
                        group2_value=group2_value,
                        sampling_method=sampling,
                        metrics=valid_metrics
                    )
            
            all_results[sampling] = sampling_results
        else:
            print(f"  Skipping: Not both groups present")
            print(f"  Available: {list(groups_present)}")
    
    # Create the single figure with all boxplots
    print("\n" + "="*60)
    print("Creating combined figure with all boxplots...")
    create_all_boxplots_single_figure(df, group_column, group1_value, group2_value, metrics_to_test, 
                                     save_path=save_path, filename_suffix=filename_suffix)
    
    return all_results


def determine_better_performer(results, metric):
    """
    Determine which group performs better based on metric type
    For both QE and Topology, lower is better
    """
    if results['significance'] != 'significant':
        return 'No significant difference'
    
    # Both metrics: lower is better
    if results['group1_mean'] < results['group2_mean']:
        return results['group1_name']
    else:
        return results['group2_name']


def create_summary_csv(mann_whitney_results, output_file='Results/Summary_Outputs/mann_whitney_summary.csv'):
    """
    Create a CSV file with all Mann-Whitney U test results
    
    Parameters:
    -----------
    mann_whitney_results : dict
        Dictionary of test results from perform_algorithm_comparisons
    output_file : str
        Path for output CSV file
    """
    
    # Create list to store all rows
    rows = []
    
    for sampling_method, sampling_results in mann_whitney_results.items():
        for metric, results in sampling_results.items():
            row = {
                'Sampling Method': sampling_method,
                'Metric': metric.replace('_', ' ').title(),
                'Group 1': results['group1_name'],
                'Group 2': results['group2_name'],
                'Group 1 Mean': round(results['group1_mean'], 4),
                'Group 2 Mean': round(results['group2_mean'], 4),
                'Group 1 Median': round(results['group1_median'], 4),
                'Group 2 Median': round(results['group2_median'], 4),
                'Group 1 Std': round(results['group1_std'], 4),
                'Group 2 Std': round(results['group2_std'], 4),
                'U Statistic': round(results['statistic'], 2),
                'P-value': round(results['p_value'], 6),
                'Effect Size': round(results['effect_size'], 3),
                'Significant': results['significance'].upper(),
                'Better Performer': determine_better_performer(results, metric)
            }
            rows.append(row)
    
    # Create DataFrame
    df_summary = pd.DataFrame(rows)
    
    # Sort by sampling method and metric
    df_summary = df_summary.sort_values(['Sampling Method', 'Metric'])
    
    # Save to CSV
    df_summary.to_csv(output_file, index=False)
    print(f"Summary CSV saved to {output_file}")
    
    return df_summary


def create_summary_markdown(mann_whitney_results, df_summary, output_file='Results/Summary_Outputs/mann_whitney_summary.md'):
    """
    Create a markdown document with comprehensive summary
    
    Parameters:
    -----------
    mann_whitney_results : dict
        Dictionary of test results
    df_summary : DataFrame
        Summary DataFrame created by create_summary_csv
    output_file : str
        Path for output markdown file
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Mann-Whitney U Test Results Summary\n\n")
        f.write("## Overview\n\n")
        f.write("This document summarizes all Mann-Whitney U tests comparing **batch** vs **colors** algorithms ")
        f.write("across different sampling methods and metrics.\n\n")
        
        # Statistical interpretation guide
        f.write("## Statistical Interpretation Guide\n\n")
        f.write("- **P-value < 0.05**: Statistically significant difference between groups\n")
        f.write("- **Effect Size Interpretation**:\n")
        f.write("  - |r| < 0.1: Negligible\n")
        f.write("  - 0.1 ≤ |r| < 0.3: Small\n")
        f.write("  - 0.3 ≤ |r| < 0.5: Medium\n")
        f.write("  - |r| ≥ 0.5: Large\n\n")
        f.write("- **For both metrics (Quantization Error and Topology PC1)**: Lower values indicate better performance\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        total_tests = len(df_summary)
        significant_tests = len(df_summary[df_summary['Significant'] == 'SIGNIFICANT'])
        f.write(f"- Total tests performed: {total_tests}\n")
        f.write(f"- Significant results: {significant_tests} ({significant_tests/total_tests*100:.1f}%)\n")
        f.write(f"- Non-significant results: {total_tests - significant_tests} ({(total_tests-significant_tests)/total_tests*100:.1f}%)\n\n")
        
        # Winner summary
        f.write("## Performance Summary by Metric\n\n")
        
        for metric in df_summary['Metric'].unique():
            f.write(f"### {metric}\n\n")
            metric_df = df_summary[df_summary['Metric'] == metric]
            
            # Count wins for each algorithm
            batch_wins = len(metric_df[metric_df['Better Performer'] == 'batch'])
            colors_wins = len(metric_df[metric_df['Better Performer'] == 'colors'])
            no_diff = len(metric_df[metric_df['Better Performer'] == 'No significant difference'])
            
            f.write(f"- **Batch performs better**: {batch_wins} sampling methods\n")
            f.write(f"- **Colors performs better**: {colors_wins} sampling methods\n")
            f.write(f"- **No significant difference**: {no_diff} sampling methods\n\n")
            
            # List sampling methods where each performs better
            if batch_wins > 0:
                batch_sampling = metric_df[metric_df['Better Performer'] == 'batch']['Sampling Method'].tolist()
                f.write(f"Batch performs better in: {', '.join(batch_sampling)}\n\n")
            
            if colors_wins > 0:
                colors_sampling = metric_df[metric_df['Better Performer'] == 'colors']['Sampling Method'].tolist()
                f.write(f"Colors performs better in: {', '.join(colors_sampling)}\n\n")
        
        # Detailed results by sampling method
        f.write("## Detailed Results by Sampling Method\n\n")
        
        for sampling in sorted(df_summary['Sampling Method'].unique()):
            f.write(f"### {sampling.upper()} Sampling\n\n")
            sampling_df = df_summary[df_summary['Sampling Method'] == sampling]
            
            for _, row in sampling_df.iterrows():
                f.write(f"#### {row['Metric']}\n\n")
                f.write(f"- **Means**: Batch = {row['Group 1 Mean']:.4f}, Colors = {row['Group 2 Mean']:.4f}\n")
                f.write(f"- **Medians**: Batch = {row['Group 1 Median']:.4f}, Colors = {row['Group 2 Median']:.4f}\n")
                f.write(f"- **P-value**: {row['P-value']:.6f} ")
                
                if row['Significant'] == 'SIGNIFICANT':
                    f.write("✓ **SIGNIFICANT**\n")
                else:
                    f.write("(not significant)\n")
                
                f.write(f"- **Effect size**: {row['Effect Size']:.3f} ")
                
                # Interpret effect size
                abs_effect = abs(row['Effect Size'])
                if abs_effect < 0.1:
                    f.write("(negligible)\n")
                elif abs_effect < 0.3:
                    f.write("(small)\n")
                elif abs_effect < 0.5:
                    f.write("(medium)\n")
                else:
                    f.write("(large)\n")
                
                if row['Better Performer'] != 'No significant difference':
                    f.write(f"- **Better performer**: **{row['Better Performer'].upper()}**\n")
                
                f.write("\n")
        
        # Create a compact table
        f.write("## Compact Results Table\n\n")
        f.write("| Sampling | Metric | Batch Mean | Colors Mean | P-value | Significant | Better |\n")
        f.write("|----------|--------|------------|-------------|---------|-------------|--------|\n")
        
        for _, row in df_summary.iterrows():
            sig_mark = "✓" if row['Significant'] == 'SIGNIFICANT' else ""
            better = row['Better Performer'] if row['Better Performer'] != 'No significant difference' else "-"
            f.write(f"| {row['Sampling Method']} | {row['Metric']} | ")
            f.write(f"{row['Group 1 Mean']:.4f} | {row['Group 2 Mean']:.4f} | ")
            f.write(f"{row['P-value']:.6f} | {sig_mark} | {better} |\n")
    
    print(f"Summary markdown saved to {output_file}")


def create_combined_boxplot(df, group_column, group1_value, group2_value,
                           sampling_method, metrics=None, save_path='Results/plots/'):
    """
    Create combined boxplot showing comparisons for specified metrics
    
    Parameters:
    df: pandas DataFrame with the data
    group_column: column that defines the groups (e.g., 'algorithm')
    group1_value: first group value (e.g., 'batch')
    group2_value: second group value (e.g., 'colors')
    sampling_method: name of the sampling method for title
    metrics: list of metric columns to plot (default: ['quantization_error', 'Topology_PC1'])
    save_path: directory to save the plot
    """
    
    # Default metrics if not specified
    if metrics is None:
        metrics = [
            'quantization_error_holdout_normalized',
            'quantization_error_train_normalized',
            'balanced_qe_normalized'
        ]
    
    # Filter to only metrics that exist in the dataframe
    metrics = [m for m in metrics if m in df.columns]
    
    if len(metrics) == 0:
        print("Warning: No valid metrics found for combined boxplot")
        return
    
    # Create subplots based on number of metrics
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    # Make axes always iterable
    if n_metrics == 1:
        axes = [axes]
    
    # Define colors for the boxes
    colors = ['lightblue', 'lightcoral']
    
    # Create boxplot for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get data for both groups
        group1_data = df[df[group_column] == group1_value][metric].values
        group2_data = df[df[group_column] == group2_value][metric].values
        
        # Create boxplot
        bp = ax.boxplot([group1_data, group2_data], 
                       labels=[group1_value, group2_value],
                       patch_artist=True, notch=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add mean markers
        ax.scatter([1, 2], [np.mean(group1_data), np.mean(group2_data)], 
                  color='red', marker='D', s=50, zorder=3, label='Mean')
        
        # Format metric name for display
        metric_display = metric.replace('_', ' ').title()
        ax.set_ylabel(metric_display, fontsize=11)
        ax.set_title(f'{metric_display} Comparison\n{sampling_method} sampling', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    # Overall title
    plt.suptitle(f'Mann-Whitney U Test: {group1_value} vs {group2_value}\nSampling Method: {sampling_method}', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{save_path}mann_whitney_combined_{sampling_method}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Combined boxplot saved to {filename}")
    plt.close()  # Close the figure instead of showing it
    
    return fig  # Return the figure for later use


def create_all_boxplots_single_figure(df, group_column='algorithm', group1_value='batch', 
                                      group2_value='colors', metrics_to_test=None,
                                      save_path='Results/plots/', filename_suffix=''):
    """
    Create ALL Mann-Whitney boxplots in a single figure
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with all data
    group_column : str
        Column name that defines the groups
    group1_value : str
        First group value to compare
    group2_value : str
        Second group value to compare
    metrics_to_test : list
        List of metrics to test
    save_path : str
        Directory to save the plot
    """
    
    # Default metrics if not specified
    if metrics_to_test is None:
        metrics_to_test = [
            'quantization_error_holdout_normalized',
            'quantization_error_train_normalized',
            'balanced_qe_normalized'
        ]
    
    # Get sampling methods
    sampling_col = None
    for col in ['sampling_method', 'sampling_method_final', 'sampling_method_parsed']:
        if col in df.columns:
            sampling_col = col
            break
    
    if sampling_col is None:
        print("Warning: No sampling method column found")
        return
    
    # Get unique sampling methods
    sampling_methods = sorted(df[sampling_col].dropna().unique())
    n_sampling = len(sampling_methods)
    n_metrics = len(metrics_to_test)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(n_sampling, n_metrics, figsize=(6*n_metrics, 4*n_sampling))
    
    # Make axes always a 2D array
    if n_sampling == 1:
        axes = axes.reshape(1, -1)
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors for the boxes
    colors = ['lightblue', 'lightcoral']
    
    # Create plots for each sampling method and metric
    for i, sampling in enumerate(sampling_methods):
        df_sampling = df[df[sampling_col] == sampling]
        
        for j, metric in enumerate(metrics_to_test):
            ax = axes[i, j]
            
            # Check if both groups exist for this sampling method
            groups_present = df_sampling[group_column].dropna().unique()
            
            if group1_value in groups_present and group2_value in groups_present and metric in df_sampling.columns:
                # Get data for both groups
                group1_data = df_sampling[df_sampling[group_column] == group1_value][metric].values
                group2_data = df_sampling[df_sampling[group_column] == group2_value][metric].values
                
                # Create boxplot
                bp = ax.boxplot([group1_data, group2_data], 
                               labels=[group1_value, group2_value],
                               patch_artist=True, notch=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                # Add mean markers
                ax.scatter([1, 2], [np.mean(group1_data), np.mean(group2_data)], 
                          color='red', marker='D', s=30, zorder=3)
                
                # Perform Mann-Whitney U test for p-value
                from scipy import stats
                statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                
                # Add p-value to plot
                y_max = ax.get_ylim()[1]
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                ax.text(1.5, y_max * 0.95, f'p={p_value:.4f}\n{significance}', 
                       ha='center', va='top', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Set titles and labels
            metric_display = metric.replace('_', ' ').title()
            if i == 0:  # Top row
                ax.set_title(f'{metric_display}', fontsize=12, fontweight='bold')
            if j == 0:  # Left column
                ax.set_ylabel(f'{sampling.upper()}\n\nValue', fontsize=11)
            else:
                ax.set_ylabel('Value', fontsize=10)
            
            ax.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    plt.suptitle(f'Mann-Whitney U Tests: {group1_value} vs {group2_value}\nAll Sampling Methods and Metrics', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{save_path}mann_whitney_all_comparisons{filename_suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nAll boxplots saved to {filename}")
    plt.close()  # Close figure without displaying
