"""
Mann-Whitney U Test Analysis for Normalized Overall Score
Normalized Overall Score is derived from normalized train QE and holdout QE.
Comparing color vs batch algorithms across sampling methods using filtered best performers data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

# Add the modules directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the existing Mann-Whitney functions
from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis.statistical_analysis import (
    mann_whitney_u_test,
    write_results,
    create_summary_csv,
    create_summary_markdown,
    create_all_boxplots_single_figure
)

# Import the normalized score calculation
from floatsom_benchmarks.optuna.optuna_results_analysis.modules.overall_score_analysis.calculate_normalized_overall_score import (
    calculate_normalized_overall_score
)


def load_and_prepare_filtered_data():
    """Load the filtered best performers data and calculate normalized scores"""
    
    # Load best performers info
    best_performers_info = pd.read_csv('Data/best_performers_by_sampling.csv')
    
    # Load all data
    all_data = pd.read_csv('Data/processed_data_with_overall_score.csv')
    
    # Get list of best scenarios
    best_scenarios = best_performers_info['Best_Scenario'].unique()
    
    # Filter the main data to only include these best scenarios
    filtered_data = all_data[all_data['scenario_id'].isin(best_scenarios)]
    
    # Calculate normalized overall score if not present
    if 'Normalized_Overall_Score' not in filtered_data.columns:
        print("Calculating normalized overall scores...")
        filtered_data = calculate_normalized_overall_score(filtered_data)
    
    return filtered_data


def perform_normalized_comparisons(df, metric='Normalized_Overall_Score'):
    """
    Perform Mann-Whitney U tests comparing Normalized Overall Score between colors and batch
    for each sampling method
    """
    print("\n" + "="*60)
    print(f"MANN-WHITNEY U TESTS: batch vs colors")
    print(f"Metric: Normalized Overall Score")
    print("Formula: sqrt((QE_train_norm² + QE_holdout_norm²) / 2)")
    print("Using Filtered Best Performers Data")
    print("="*60)
    
    # Determine which sampling method column to use
    sampling_col = None
    for col in ['sampling_method_final', 'sampling_method', 'sampling_method_parsed']:
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
        
        # Check if both groups exist
        groups_present = df_sampling['algorithm'].dropna().unique()
        
        if 'batch' in groups_present and 'colors' in groups_present:
            # Get data for both groups
            batch_data = df_sampling[df_sampling['algorithm'] == 'batch'][metric].values
            colors_data = df_sampling[df_sampling['algorithm'] == 'colors'][metric].values
            
            print(f"\n  Testing {metric}:")
            print(f"    Batch samples: n={len(batch_data)}")
            print(f"    Colors samples: n={len(colors_data)}")
            
            # Perform Mann-Whitney U test
            results = mann_whitney_u_test(
                batch_data, colors_data,
                group1_name='batch',
                group2_name='colors'
            )
            
            # Individual results files are no longer saved - included in unified summary
            
            # Store results
            all_results[sampling] = {metric: results}
            
            # Print quick summary
            print(f"    P-value: {results['p_value']:.6f}")
            print(f"    Significant: {results['significance']}")
            print(f"    Effect size: {abs(results['effect_size']):.3f}")
            
            # Determine which is better (lower score is better)
            if results['significance'] == 'significant':
                if results['group1_mean'] < results['group2_mean']:
                    print(f"    → Batch performs BETTER (lower normalized score)")
                else:
                    print(f"    → Colors performs BETTER (lower normalized score)")
        else:
            print(f"  Skipping: Not both groups present")
            print(f"  Available: {list(groups_present)}")
    
    return all_results


def create_normalized_visualizations(df, mann_whitney_results, metric='Normalized_Overall_Score'):
    """Create visualizations for normalized score analysis"""
    
    # Use the existing function to create all boxplots in a single figure
    create_all_boxplots_single_figure(
        df,
        group_column='algorithm',
        group1_value='batch',
        group2_value='colors',
        metrics_to_test=[metric],
        save_path='Results/plots/'
    )
    
    # Create a summary bar plot showing effect sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sampling_methods = []
    effect_sizes = []
    p_values = []
    
    for sampling, results in mann_whitney_results.items():
        if metric in results:
            sampling_methods.append(sampling)
            effect_sizes.append(results[metric]['effect_size'])
            p_values.append(results[metric]['p_value'])
    
    # Color bars based on significance
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    
    bars = ax.barh(sampling_methods, effect_sizes, color=colors, alpha=0.7)
    
    # Add vertical lines for effect size interpretation
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.3, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel("Effect Size (Rank Biserial Correlation)")
    ax.set_ylabel("Sampling Method")
    ax.set_title(f"Mann-Whitney U Test Effect Sizes: Batch vs Colors\nNormalized Overall Score (QE_norm + PC1_norm)\n(Negative = Batch Better, Positive = Colors Better)")
    
    # Add text annotations
    y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.text(0.1, y_pos, 'Small', ha='center', fontsize=8, alpha=0.5)
    ax.text(0.3, y_pos, 'Medium', ha='center', fontsize=8, alpha=0.5)
    ax.text(0.5, y_pos, 'Large', ha='center', fontsize=8, alpha=0.5)
    ax.text(-0.1, y_pos, 'Small', ha='center', fontsize=8, alpha=0.5)
    ax.text(-0.3, y_pos, 'Medium', ha='center', fontsize=8, alpha=0.5)
    ax.text(-0.5, y_pos, 'Large', ha='center', fontsize=8, alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('Results/plots/mann_whitney_normalized_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Optional pairplot to visualise the normalized QE metrics
    normalized_pair = {
        'quantization_error_train_normalized',
        'quantization_error_holdout_normalized'
    }
    if normalized_pair.issubset(df.columns):
        import seaborn as sns
        trio_df = df[list(normalized_pair) + ['algorithm']].dropna()
        pairplot = sns.pairplot(
            trio_df,
            hue='algorithm',
            corner=True,
            plot_kws={'alpha': 0.45, 's': 25}
        )
        pairplot.fig.suptitle(
            'Normalized Metric Relationships (Train QE, Holdout QE)',
            y=1.02,
            fontsize=14
        )
        pairplot.savefig('Results/plots/normalized_metric_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close(pairplot.fig)
    
    print(f"\nVisualizations saved to Results/plots/")


def main():
    """Main function to run the normalized Mann-Whitney analysis"""
    
    print("\nLoading and preparing filtered best performers data...")
    df = load_and_prepare_filtered_data()
    
    print(f"\nLoaded {len(df)} samples from filtered best performers")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Batch samples: {len(df[df['algorithm'] == 'batch'])}")
    print(f"Colors samples: {len(df[df['algorithm'] == 'colors'])}")
    
    # Check normalized score statistics
    if 'Normalized_Overall_Score' in df.columns:
        print(f"\nNormalized Overall Score range: [{df['Normalized_Overall_Score'].min():.4f}, {df['Normalized_Overall_Score'].max():.4f}]")
        print(f"Mean: {df['Normalized_Overall_Score'].mean():.4f}")
    
    # Perform Mann-Whitney U tests
    mann_whitney_results = perform_normalized_comparisons(df)
    
    if mann_whitney_results:
        print("\n" + "="*60)
        print("Creating summary documents...")
        print("="*60)
        
        # Create CSV summary
        df_summary = create_summary_csv(mann_whitney_results, 
                                       output_file='Results/Summary_Outputs/mann_whitney_normalized_summary.csv')
        
        # Create markdown summary
        create_summary_markdown(mann_whitney_results, df_summary,
                              output_file='Results/Summary_Outputs/mann_whitney_normalized_summary.md')
        
        print("\nSummary files created:")
        print("  - Results/Summary_Outputs/mann_whitney_normalized_summary.csv")
        print("  - Results/Summary_Outputs/mann_whitney_normalized_summary.md")
        
        # Create visualizations
        print("\n" + "="*60)
        print("Creating visualizations...")
        print("="*60)
        
        create_normalized_visualizations(df, mann_whitney_results)
        
        # Print final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*60)
        
        for sampling, sampling_results in mann_whitney_results.items():
            if 'Normalized_Overall_Score' in sampling_results:
                results = sampling_results['Normalized_Overall_Score']
                print(f"\n{sampling.upper()} Sampling:")
                print(f"  Batch mean: {results['group1_mean']:.6f}")
                print(f"  Colors mean: {results['group2_mean']:.6f}")
                print(f"  P-value: {results['p_value']:.6f}")
                print(f"  Effect size: {results['effect_size']:.3f}")
                
                if results['significance'] == 'significant':
                    if results['group1_mean'] < results['group2_mean']:
                        print(f"  → BATCH performs significantly BETTER ✓")
                    else:
                        print(f"  → COLORS performs significantly BETTER ✓")
                else:
                    print(f"  → No significant difference")
    else:
        print("\nNo results to summarize.")


# This module is now called from main.py
