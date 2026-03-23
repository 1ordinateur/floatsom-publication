import pandas as pd
import numpy as np

print("="*80)
print("GENERATING BEST PERFORMERS REPORT (Using Normalized Scores)")
print("="*80)

# Load the processed data with overall scores
print("\nLoading data...")
df = pd.read_csv('Data/processed_data_with_overall_score.csv')
print(f"Loaded {len(df)} rows")

# Check if Normalized_Overall_Score exists, if not calculate it
if 'Normalized_Overall_Score' not in df.columns:
    print("\nNormalized_Overall_Score not found, calculating...")

    new_required = [
        'quantization_error_train_normalized',
        'quantization_error_holdout_normalized'
    ]
    default_weights = {
        'quantization_error_train_normalized': 1.0,
        'quantization_error_holdout_normalized': 1.0,
    }

    if all(col in df.columns for col in new_required):
        numerator = np.zeros(len(df))
        weight_sum = 0.0
        for metric, weight in default_weights.items():
            numerator += weight * (df[metric] ** 2)
            weight_sum += weight
        df['Normalized_Overall_Score'] = np.sqrt(numerator / weight_sum)
        print("Calculated Normalized_Overall_Score using weighted QE metrics")
    elif 'quantization_error_normalized' in df.columns and 'Topology_PC1_normalized' in df.columns:
        df['Normalized_Overall_Score'] = np.sqrt(
            df['quantization_error_normalized']**2 +
            df['Topology_PC1_normalized']**2
        )
        print("Calculated Normalized_Overall_Score using QE + Topology PC1 fallback")
    else:
        print("ERROR: Required normalized columns not found!")
        print("Please ensure normalized metrics are available.")
        exit(1)

# Get unique values
datasets = df['dataset'].unique()
processing_types = df['processing_type'].unique() if 'processing_type' in df.columns else df['algorithm'].unique()
sampling_col = None
for col in ['sampling_method_final', 'sampling_method', 'sampling_method_parsed']:
    if col in df.columns:
        sampling_col = col
        break

if sampling_col:
    sampling_methods = df[sampling_col].unique()
else:
    print("Warning: No sampling method column found")
    sampling_methods = []

print("\nDatasets found:", len(datasets))
print("Processing types found:", len(processing_types))
print("Sampling methods found:", len(sampling_methods))

# Create results dictionary
results = {}

print("\n" + "="*80)
print("FINDING BEST PERFORMERS BY CATEGORY")
print("="*80)

# Iterate through each dataset
for dataset in sorted(datasets):
    print(f"\nProcessing dataset: {dataset}")
    
    df_dataset = df[df['dataset'] == dataset]
    results[dataset] = {}
    
    # Use algorithm column if processing_type doesn't exist
    proc_col = 'processing_type' if 'processing_type' in df.columns else 'algorithm'
    
    for proc_type in sorted(df_dataset[proc_col].unique()):
        df_proc = df_dataset[df_dataset[proc_col] == proc_type]
        results[dataset][proc_type] = {}
        
        if sampling_col:
            for sampling in sorted(df_proc[sampling_col].unique()):
                df_sampling = df_proc[df_proc[sampling_col] == sampling]
                
                if len(df_sampling) > 0:
                    # Find the best performer (lowest normalized score)
                    best_idx = df_sampling['Normalized_Overall_Score'].idxmin()
                    best_row = df_sampling.loc[best_idx]
                    
                    results[dataset][proc_type][sampling] = {
                        'scenario_id': best_row['scenario_id'],
                        'normalized_score': best_row['Normalized_Overall_Score'],
                        'qe_normalized': best_row['quantization_error_normalized'],
                        'pc1_normalized': best_row['Topology_PC1_normalized'] if 'Topology_PC1_normalized' in best_row else best_row.get('Topology_PC1', 0),
                        'qe_raw': best_row['quantization_error'],
                        'pc1_raw': best_row.get('Topology_PC1', 0),
                        'map_type': best_row.get('map_type', 'N/A')
                    }

# Create summary DataFrame for best performers
summary_data = []
for dataset in sorted(results.keys()):
    for proc_type in sorted(results[dataset].keys()):
        for sampling in sorted(results[dataset][proc_type].keys()):
            data = results[dataset][proc_type][sampling]
            summary_data.append({
                'Dataset': dataset,
                'Processing': proc_type,
                'Sampling': sampling,
                'Best_Score': data['normalized_score'],
                'QE_Norm': data['qe_normalized'],
                'PC1_Norm': data['pc1_normalized'],
                'Scenario': data['scenario_id']
            })

summary_df = pd.DataFrame(summary_data)

# Create pivot table for better visualization
pivot_score = summary_df.pivot_table(
    index='Dataset',
    columns=['Processing', 'Sampling'],
    values='Best_Score',
    aggfunc='first'
)

# Save results to CSV
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

summary_df.to_csv('Results/Summary_Outputs/best_performers_normalized.csv', index=False)
print("✓ Saved summary to 'Results/Summary_Outputs/best_performers_normalized.csv'")

# Save the pivot table as markdown
with open('Results/Summary_Outputs/best_performers_summary.md', 'w', encoding='utf-8') as f:
    f.write("# Best Performers Summary Report (Normalized Scores)\n\n")
    f.write("## Overview\n\n")
    f.write("This report shows the best performing SOM configurations for each combination of:\n")
    f.write("- **Dataset** (9 datasets)\n")
    f.write("- **Processing Type** (batch vs colors)\n")
    f.write("- **Sampling Method** (full, hdsssom, random)\n\n")
    f.write("### Scoring Method\n")
    f.write("- **Normalized Overall Score** = sqrt(QE_normalized^2 + PC1_normalized^2)\n")
    f.write("- Both metrics are normalized to [0,1] range within each dataset\n")
    f.write("- Lower scores indicate better performance\n")
    f.write("- Theoretical range: [0, 1.414]\n\n")
    
    # Add overall statistics
    f.write("## Overall Statistics\n\n")
    f.write(f"- Total configurations tested: {len(df)}\n")
    f.write(f"- Best overall score: {summary_df['Best_Score'].min():.4f}\n")
    f.write(f"- Worst best score: {summary_df['Best_Score'].max():.4f}\n")
    f.write(f"- Mean best score: {summary_df['Best_Score'].mean():.4f}\n\n")
    
    # Find global best performers
    f.write("## Global Best Performers (Top 5)\n\n")
    top5 = summary_df.nsmallest(5, 'Best_Score')
    f.write("| Rank | Dataset | Processing | Sampling | Score | QE (norm) | PC1 (norm) |\n")
    f.write("|------|---------|------------|----------|-------|-----------|------------|\n")
    for i, row in enumerate(top5.iterrows(), 1):
        r = row[1]
        f.write(f"| {i} | {r['Dataset']} | {r['Processing']} | {r['Sampling']} | ")
        f.write(f"{r['Best_Score']:.4f} | {r['QE_Norm']:.4f} | {r['PC1_Norm']:.4f} |\n")
    
    # Best by processing type
    f.write("\n## Best Performers by Processing Type\n\n")
    f.write("### Batch Processing\n\n")
    batch_df = summary_df[summary_df['Processing'] == 'batch']
    f.write(f"- Best batch score: {batch_df['Best_Score'].min():.4f}\n")
    f.write(f"- Mean batch score: {batch_df['Best_Score'].mean():.4f}\n")
    f.write(f"- Worst batch score: {batch_df['Best_Score'].max():.4f}\n\n")
    
    f.write("### Colors Processing\n\n")
    colors_df = summary_df[summary_df['Processing'] == 'colors']
    f.write(f"- Best colors score: {colors_df['Best_Score'].min():.4f}\n")
    f.write(f"- Mean colors score: {colors_df['Best_Score'].mean():.4f}\n")
    f.write(f"- Worst colors score: {colors_df['Best_Score'].max():.4f}\n\n")
    
    # Best by sampling method
    f.write("## Best Performers by Sampling Method\n\n")
    for sampling in summary_df['Sampling'].unique():
        sampling_df = summary_df[summary_df['Sampling'] == sampling]
        f.write(f"### {sampling.upper()} Sampling\n\n")
        f.write(f"- Best score: {sampling_df['Best_Score'].min():.4f}\n")
        f.write(f"- Mean score: {sampling_df['Best_Score'].mean():.4f}\n")
        f.write(f"- Worst score: {sampling_df['Best_Score'].max():.4f}\n\n")
    
    # Detailed results by dataset
    f.write("## Detailed Results by Dataset\n\n")
    
    for dataset in sorted(summary_df['Dataset'].unique()):
        f.write(f"### {dataset.upper()}\n\n")
        dataset_df = summary_df[summary_df['Dataset'] == dataset].sort_values('Best_Score')
        
        f.write("| Processing | Sampling | Score | QE (norm) | PC1 (norm) | Scenario |\n")
        f.write("|------------|----------|-------|-----------|------------|----------|\n")
        
        for _, row in dataset_df.iterrows():
            f.write(f"| {row['Processing']:10} | {row['Sampling']:8} | ")
            f.write(f"{row['Best_Score']:.4f} | {row['QE_Norm']:.4f} | ")
            f.write(f"{row['PC1_Norm']:.4f} | {row['Scenario']} |\n")
        
        # Add dataset-specific insights
        best_row = dataset_df.iloc[0]
        f.write(f"\n**Best for {dataset}**: {best_row['Processing']} with {best_row['Sampling']} ")
        f.write(f"sampling (score: {best_row['Best_Score']:.4f})\n\n")
    
    # Comparative analysis
    f.write("## Comparative Analysis\n\n")
    
    # Count wins by processing type
    f.write("### Processing Type Performance\n\n")
    wins = {'batch': 0, 'colors': 0}
    for dataset in summary_df['Dataset'].unique():
        dataset_df = summary_df[summary_df['Dataset'] == dataset]
        best = dataset_df.loc[dataset_df['Best_Score'].idxmin()]
        wins[best['Processing']] += 1
    
    f.write(f"- **Batch** wins in {wins['batch']}/9 datasets\n")
    f.write(f"- **Colors** wins in {wins['colors']}/9 datasets\n\n")
    
    # Count wins by sampling method
    f.write("### Sampling Method Performance\n\n")
    sampling_wins = {}
    for sampling in summary_df['Sampling'].unique():
        sampling_wins[sampling] = 0
    
    for dataset in summary_df['Dataset'].unique():
        dataset_df = summary_df[summary_df['Dataset'] == dataset]
        best = dataset_df.loc[dataset_df['Best_Score'].idxmin()]
        sampling_wins[best['Sampling']] += 1
    
    for sampling, count in sorted(sampling_wins.items()):
        f.write(f"- **{sampling}** wins in {count}/9 datasets\n")
    
    # Create summary table
    f.write("\n## Summary Table (All Best Scores)\n\n")
    f.write("```\n")
    f.write(pivot_score.round(4).to_string())
    f.write("\n```\n\n")
    
    # Add recommendations
    f.write("## Recommendations\n\n")
    
    # Find most consistent performer
    mean_scores = summary_df.groupby(['Processing', 'Sampling'])['Best_Score'].mean().sort_values()
    best_combo = mean_scores.index[0]
    
    f.write(f"1. **Most consistent configuration**: {best_combo[0]} processing with {best_combo[1]} sampling\n")
    f.write(f"   - Average score: {mean_scores.iloc[0]:.4f}\n\n")
    
    f.write("2. **Dataset-specific recommendations**:\n")
    for dataset in sorted(summary_df['Dataset'].unique()):
        dataset_df = summary_df[summary_df['Dataset'] == dataset]
        best = dataset_df.loc[dataset_df['Best_Score'].idxmin()]
        f.write(f"   - {dataset}: Use {best['Processing']} with {best['Sampling']}\n")
    
    f.write("\n---\n")
    f.write("*Report generated using normalized metrics (QE and PC1 both in [0,1] range)*\n")

print("✓ Saved markdown report to 'Results/Summary_Outputs/best_performers_summary.md'")

# Print summary to console
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nTop 5 Best Performers:")
print("-"*40)
top5 = summary_df.nsmallest(5, 'Best_Score')
for i, row in enumerate(top5.iterrows(), 1):
    r = row[1]
    print(f"{i}. {r['Dataset']:15} {r['Processing']:8} {r['Sampling']:8} Score: {r['Best_Score']:.4f}")

print("\nProcessing Type Wins:")
print("-"*40)
wins = {'batch': 0, 'colors': 0}
for dataset in summary_df['Dataset'].unique():
    dataset_df = summary_df[summary_df['Dataset'] == dataset]
    best = dataset_df.loc[dataset_df['Best_Score'].idxmin()]
    wins[best['Processing']] += 1
print(f"Batch:  {wins['batch']}/9 datasets")
print(f"Colors: {wins['colors']}/9 datasets")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  - Results/Summary_Outputs/best_performers_summary.md (detailed report)")
print("  - Results/Summary_Outputs/best_performers_normalized.csv (data)")
