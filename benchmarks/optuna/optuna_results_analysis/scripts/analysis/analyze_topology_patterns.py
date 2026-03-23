#!/usr/bin/env python3
"""
Analyze patterns in topology metrics for best performers.
Examines how sampling methods and processing types affect performance
in quantization error vs topological preservation (PC1).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse

def load_best_performers(architecture=''):
    """Load the best performers data."""
    suffix = f"_{architecture}" if architecture else ""
    data_file = Path(f"Data/best_performers_by_sampling{suffix}.csv")
    df = pd.read_csv(data_file)
    return df

def analyze_sampling_method_dominance(df):
    """
    Analyze which sampling methods excel at QE vs PC1.
    """
    print("\n" + "="*80)
    print("SAMPLING METHOD ANALYSIS")
    print("="*80)
    
    # Group by sampling method
    sampling_stats = df.groupby('Sampling_Method').agg({
        'QE_Normalized': ['mean', 'std', 'count'],
        'PC1_Normalized': ['mean', 'std', 'count'],
        'Best_Normalized_Score': ['mean', 'std']
    }).round(4)
    
    print("\nSampling Method Performance Summary:")
    print(sampling_stats)
    
    # Identify which sampling method is best for each metric
    print("\n" + "-"*50)
    print("Best Sampling Method by Metric:")
    print("-"*50)
    
    best_qe_sampling = df.groupby('Sampling_Method')['QE_Normalized'].mean().idxmin()
    best_pc1_sampling = df.groupby('Sampling_Method')['PC1_Normalized'].mean().idxmin()
    best_overall_sampling = df.groupby('Sampling_Method')['Best_Normalized_Score'].mean().idxmax()
    
    print(f"Best for QE (lowest):           {best_qe_sampling}")
    print(f"Best for PC1 (lowest):          {best_pc1_sampling}")
    print(f"Best Overall Score (highest):   {best_overall_sampling}")
    
    # Analyze trade-offs
    print("\n" + "-"*50)
    print("Trade-off Analysis (QE vs PC1):")
    print("-"*50)
    
    for method in df['Sampling_Method'].unique():
        method_data = df[df['Sampling_Method'] == method]
        qe_mean = method_data['QE_Normalized'].mean()
        pc1_mean = method_data['PC1_Normalized'].mean()
        
        # Determine strength
        if qe_mean < 0.3:
            qe_strength = "Excellent"
        elif qe_mean < 0.5:
            qe_strength = "Good"
        else:
            qe_strength = "Poor"
            
        if pc1_mean < 0.15:
            pc1_strength = "Excellent"
        elif pc1_mean < 0.25:
            pc1_strength = "Good"
        else:
            pc1_strength = "Poor"
        
        print(f"\n{method}:")
        print(f"  QE Performance:  {qe_strength} (mean={qe_mean:.4f})")
        print(f"  PC1 Performance: {pc1_strength} (mean={pc1_mean:.4f})")
        
        # Identify pattern
        if qe_mean < 0.3 and pc1_mean < 0.15:
            pattern = "Balanced Excellence"
        elif qe_mean < 0.5 and pc1_mean > 0.3:
            pattern = "QE-focused"
        elif qe_mean > 0.5 and pc1_mean < 0.2:
            pattern = "Topology-focused"
        else:
            pattern = "Mixed Performance"
        
        print(f"  Pattern: {pattern}")
    
    return sampling_stats

def analyze_processing_type_dominance(df):
    """
    Analyze which processing types excel at QE vs PC1.
    """
    print("\n" + "="*80)
    print("PROCESSING TYPE ANALYSIS")
    print("="*80)
    
    # Group by processing type
    processing_stats = df.groupby('Processing_Type').agg({
        'QE_Normalized': ['mean', 'std', 'count'],
        'PC1_Normalized': ['mean', 'std', 'count'],
        'Best_Normalized_Score': ['mean', 'std']
    }).round(4)
    
    print("\nProcessing Type Performance Summary:")
    print(processing_stats)
    
    # Identify which processing type is best for each metric
    print("\n" + "-"*50)
    print("Best Processing Type by Metric:")
    print("-"*50)
    
    best_qe_proc = df.groupby('Processing_Type')['QE_Normalized'].mean().idxmin()
    best_pc1_proc = df.groupby('Processing_Type')['PC1_Normalized'].mean().idxmin()
    best_overall_proc = df.groupby('Processing_Type')['Best_Normalized_Score'].mean().idxmax()
    
    print(f"Best for QE (lowest):           {best_qe_proc}")
    print(f"Best for PC1 (lowest):          {best_pc1_proc}")
    print(f"Best Overall Score (highest):   {best_overall_proc}")
    
    # Statistical test between batch and colors
    print("\n" + "-"*50)
    print("Statistical Comparison (Batch vs Colors):")
    print("-"*50)
    
    batch_qe = df[df['Processing_Type'] == 'batch']['QE_Normalized']
    colors_qe = df[df['Processing_Type'] == 'colors']['QE_Normalized']
    
    batch_pc1 = df[df['Processing_Type'] == 'batch']['PC1_Normalized']
    colors_pc1 = df[df['Processing_Type'] == 'colors']['PC1_Normalized']
    
    # Mann-Whitney U test
    u_stat_qe, p_val_qe = stats.mannwhitneyu(batch_qe, colors_qe, alternative='two-sided')
    u_stat_pc1, p_val_pc1 = stats.mannwhitneyu(batch_pc1, colors_pc1, alternative='two-sided')
    
    print(f"\nQE Comparison:")
    print(f"  Batch mean:  {batch_qe.mean():.4f} ± {batch_qe.std():.4f}")
    print(f"  Colors mean: {colors_qe.mean():.4f} ± {colors_qe.std():.4f}")
    print(f"  P-value: {p_val_qe:.6f} ({'Significant' if p_val_qe < 0.05 else 'Not significant'})")
    
    print(f"\nPC1 Comparison:")
    print(f"  Batch mean:  {batch_pc1.mean():.4f} ± {batch_pc1.std():.4f}")
    print(f"  Colors mean: {colors_pc1.mean():.4f} ± {colors_pc1.std():.4f}")
    print(f"  P-value: {p_val_pc1:.6f} ({'Significant' if p_val_pc1 < 0.05 else 'Not significant'})")
    
    return processing_stats

def analyze_interaction_effects(df):
    """
    Analyze interaction between sampling method and processing type.
    """
    print("\n" + "="*80)
    print("INTERACTION ANALYSIS (Sampling × Processing)")
    print("="*80)
    
    # Create pivot tables
    qe_pivot = df.pivot_table(values='QE_Normalized', 
                               index='Processing_Type', 
                               columns='Sampling_Method', 
                               aggfunc='mean')
    
    pc1_pivot = df.pivot_table(values='PC1_Normalized', 
                                index='Processing_Type', 
                                columns='Sampling_Method', 
                                aggfunc='mean')
    
    overall_pivot = df.pivot_table(values='Best_Normalized_Score', 
                                    index='Processing_Type', 
                                    columns='Sampling_Method', 
                                    aggfunc='mean')
    
    print("\nMean QE by Processing Type and Sampling Method:")
    print(qe_pivot.round(4))
    
    print("\nMean PC1 by Processing Type and Sampling Method:")
    print(pc1_pivot.round(4))
    
    print("\nMean Overall Score by Processing Type and Sampling Method:")
    print(overall_pivot.round(4))
    
    # Find best combinations
    print("\n" + "-"*50)
    print("Best Combinations:")
    print("-"*50)
    
    # Best for each metric
    best_combos = []
    for proc in df['Processing_Type'].unique():
        for samp in df['Sampling_Method'].unique():
            combo_data = df[(df['Processing_Type'] == proc) & (df['Sampling_Method'] == samp)]
            if not combo_data.empty:
                best_combos.append({
                    'Combination': f"{proc}+{samp}",
                    'QE_mean': combo_data['QE_Normalized'].mean(),
                    'PC1_mean': combo_data['PC1_Normalized'].mean(),
                    'Overall_mean': combo_data['Best_Normalized_Score'].mean(),
                    'Count': len(combo_data)
                })
    
    combo_df = pd.DataFrame(best_combos)
    combo_df = combo_df.sort_values('Overall_mean', ascending=False)
    
    print("\nTop 3 Combinations by Overall Score:")
    for i, row in combo_df.head(3).iterrows():
        print(f"{i+1}. {row['Combination']}: Score={row['Overall_mean']:.4f}, QE={row['QE_mean']:.4f}, PC1={row['PC1_mean']:.4f}")
    
    return qe_pivot, pc1_pivot, overall_pivot

def analyze_dataset_patterns(df):
    """
    Analyze which datasets favor which metrics.
    """
    print("\n" + "="*80)
    print("DATASET-SPECIFIC PATTERNS")
    print("="*80)
    
    # For each dataset, determine if it's QE-dominant or PC1-dominant
    dataset_patterns = []
    
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        
        # Get the best performer for this dataset
        best_row = dataset_data.loc[dataset_data['Best_Normalized_Score'].idxmax()]
        
        qe_norm = best_row['QE_Normalized']
        pc1_norm = best_row['PC1_Normalized']
        
        # Determine dominance (lower is better for both metrics)
        if qe_norm < pc1_norm * 0.5:
            pattern = "QE-dominant"
        elif pc1_norm < qe_norm * 0.5:
            pattern = "Topology-dominant"
        else:
            pattern = "Balanced"
        
        dataset_patterns.append({
            'Dataset': dataset,
            'Best_Method': best_row['Sampling_Method'],
            'Best_Processing': best_row['Processing_Type'],
            'QE_Normalized': qe_norm,
            'PC1_Normalized': pc1_norm,
            'Pattern': pattern
        })
    
    pattern_df = pd.DataFrame(dataset_patterns)
    pattern_df = pattern_df.sort_values('Dataset')
    
    print("\nDataset Patterns:")
    print(pattern_df.to_string(index=False))
    
    # Count patterns
    print("\n" + "-"*50)
    print("Pattern Distribution:")
    print("-"*50)
    pattern_counts = pattern_df['Pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} datasets ({count/len(pattern_df)*100:.1f}%)")
    
    return pattern_df

def create_visualizations(df, architecture=''):
    """
    Create visualizations for pattern analysis.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. QE vs PC1 scatter by sampling method
    ax1 = axes[0, 0]
    for method in df['Sampling_Method'].unique():
        method_data = df[df['Sampling_Method'] == method]
        ax1.scatter(method_data['QE_Normalized'], method_data['PC1_Normalized'], 
                   label=method, alpha=0.7, s=100)
    ax1.set_xlabel('QE Normalized')
    ax1.set_ylabel('PC1 Normalized')
    ax1.set_title('QE vs PC1 by Sampling Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. QE vs PC1 scatter by processing type
    ax2 = axes[0, 1]
    colors_map = {'batch': 'blue', 'colors': 'orange'}
    for proc in df['Processing_Type'].unique():
        proc_data = df[df['Processing_Type'] == proc]
        ax2.scatter(proc_data['QE_Normalized'], proc_data['PC1_Normalized'], 
                   label=proc, alpha=0.7, s=100, color=colors_map.get(proc, 'gray'))
    ax2.set_xlabel('QE Normalized')
    ax2.set_ylabel('PC1 Normalized')
    ax2.set_title('QE vs PC1 by Processing Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of QE by sampling method
    ax3 = axes[0, 2]
    sampling_order = ['full', 'random', 'hdsssom']
    df_sorted = df.copy()
    df_sorted['Sampling_Method'] = pd.Categorical(df_sorted['Sampling_Method'], 
                                                  categories=sampling_order, 
                                                  ordered=True)
    df_sorted.boxplot(column='QE_Normalized', by='Sampling_Method', ax=ax3)
    ax3.set_title('QE Distribution by Sampling Method')
    ax3.set_xlabel('Sampling Method')
    ax3.set_ylabel('QE Normalized')
    plt.sca(ax3)
    plt.xticks(rotation=0)
    
    # 4. Box plot of PC1 by sampling method
    ax4 = axes[1, 0]
    df_sorted.boxplot(column='PC1_Normalized', by='Sampling_Method', ax=ax4)
    ax4.set_title('PC1 Distribution by Sampling Method')
    ax4.set_xlabel('Sampling Method')
    ax4.set_ylabel('PC1 Normalized')
    plt.sca(ax4)
    plt.xticks(rotation=0)
    
    # 5. Heatmap of mean overall scores
    ax5 = axes[1, 1]
    pivot_data = df.pivot_table(values='Best_Normalized_Score', 
                                index='Processing_Type', 
                                columns='Sampling_Method', 
                                aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5)
    ax5.set_title('Mean Overall Score Heatmap')
    
    # 6. Correlation between QE and PC1
    ax6 = axes[1, 2]
    ax6.scatter(df['QE_Normalized'], df['PC1_Normalized'], alpha=0.5)
    
    # Add correlation line
    z = np.polyfit(df['QE_Normalized'], df['PC1_Normalized'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['QE_Normalized'].min(), df['QE_Normalized'].max(), 100)
    ax6.plot(x_line, p(x_line), "r-", alpha=0.8, label=f'Correlation: {df["QE_Normalized"].corr(df["PC1_Normalized"]):.3f}')
    
    ax6.set_xlabel('QE Normalized')
    ax6.set_ylabel('PC1 Normalized')
    ax6.set_title('QE vs PC1 Correlation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    figures_dir = Path("Results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{architecture}" if architecture else ""
    output_file = figures_dir / f"topology_patterns_analysis{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    
    plt.show()
    
    return fig

def generate_summary_report(df, sampling_stats, processing_stats, pattern_df, architecture=''):
    """
    Generate a comprehensive summary report.
    """
    reports_dir = Path("Results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{architecture}" if architecture else ""
    output_file = reports_dir / f"topology_patterns_summary{suffix}.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TOPOLOGY METRICS PATTERN ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-"*50 + "\n\n")
        
        # Best sampling method for each metric
        best_qe_samp = df.groupby('Sampling_Method')['QE_Normalized'].mean().idxmin()
        best_pc1_samp = df.groupby('Sampling_Method')['PC1_Normalized'].mean().idxmin()
        best_overall_samp = df.groupby('Sampling_Method')['Best_Normalized_Score'].mean().idxmax()
        
        f.write(f"1. Best Sampling Methods:\n")
        f.write(f"   - For Quantization Error: {best_qe_samp}\n")
        f.write(f"   - For Topology (PC1): {best_pc1_samp}\n")
        f.write(f"   - Overall Performance: {best_overall_samp}\n\n")
        
        # Best processing type for each metric
        best_qe_proc = df.groupby('Processing_Type')['QE_Normalized'].mean().idxmin()
        best_pc1_proc = df.groupby('Processing_Type')['PC1_Normalized'].mean().idxmin()
        best_overall_proc = df.groupby('Processing_Type')['Best_Normalized_Score'].mean().idxmax()
        
        f.write(f"2. Best Processing Types:\n")
        f.write(f"   - For Quantization Error: {best_qe_proc}\n")
        f.write(f"   - For Topology (PC1): {best_pc1_proc}\n")
        f.write(f"   - Overall Performance: {best_overall_proc}\n\n")
        
        # Pattern distribution
        pattern_counts = pattern_df['Pattern'].value_counts()
        f.write(f"3. Dataset Pattern Distribution:\n")
        for pattern, count in pattern_counts.items():
            f.write(f"   - {pattern}: {count} datasets ({count/len(pattern_df)*100:.1f}%)\n")
        f.write("\n")
        
        # Correlation
        correlation = df['QE_Normalized'].corr(df['PC1_Normalized'])
        f.write(f"4. QE-PC1 Correlation: {correlation:.3f}\n")
        if correlation > 0.5:
            f.write(f"   Interpretation: Strong positive correlation - improvements in one metric tend to accompany improvements in the other\n")
        elif correlation > 0:
            f.write(f"   Interpretation: Weak positive correlation - some tendency for metrics to move together\n")
        else:
            f.write(f"   Interpretation: Negative correlation - trade-off between metrics\n")
        f.write("\n")
        
        # Detailed statistics
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Sampling Method Statistics:\n")
        f.write(str(sampling_stats))
        f.write("\n\n")
        
        f.write("Processing Type Statistics:\n")
        f.write(str(processing_stats))
        f.write("\n\n")
        
        f.write("Dataset-Specific Patterns:\n")
        f.write(pattern_df.to_string(index=False))
        f.write("\n")
    
    print(f"\nSummary report saved to {output_file}")
    
    return output_file

def main(architecture=''):
    """
    Main analysis pipeline for topology pattern analysis.
    """
    arch_name = f" ({architecture.upper()})" if architecture else ""
    print("="*80)
    print(f"TOPOLOGY PATTERNS ANALYSIS{arch_name}")
    print("="*80)
    
    # Load data
    print("\nLoading best performers data...")
    df = load_best_performers(architecture)
    print(f"Loaded {len(df)} records")
    
    # Perform analyses
    sampling_stats = analyze_sampling_method_dominance(df)
    processing_stats = analyze_processing_type_dominance(df)
    qe_pivot, pc1_pivot, overall_pivot = analyze_interaction_effects(df)
    pattern_df = analyze_dataset_patterns(df)
    
    # Create visualizations
    fig = create_visualizations(df, architecture)
    
    # Generate summary report
    summary_file = generate_summary_report(df, sampling_stats, processing_stats, pattern_df, architecture)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    suffix = f"_{architecture}" if architecture else ""
    print("\nGenerated files:")
    print(f"  - Results/figures/topology_patterns_analysis{suffix}.png (visualizations)")
    print(f"  - Results/reports/topology_patterns_summary{suffix}.txt (detailed report)")
    
    return df, pattern_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze topology patterns')
    parser.add_argument('--architecture', '-a', type=str, default='',
                        help='Architecture to analyze (mst or hexagonal)')
    args = parser.parse_args()
    
    df, pattern_df = main(args.architecture)