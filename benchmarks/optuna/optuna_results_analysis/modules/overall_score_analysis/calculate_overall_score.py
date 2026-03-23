import pandas as pd
import numpy as np

# Load the processed data with PCA (created by main.py)
print("Loading data from processed_data_with_pca.csv...")
df = pd.read_csv('Data/processed_data_with_pca.csv')
print(f"Loaded {len(df)} rows")

# Check if normalized columns exist
if 'Topology_PC1_normalized' not in df.columns:
    print("\nNormalizing Topology_PC1 within each dataset...")
    
    def minmax_normalize_group(group):
        min_val = group.min()
        max_val = group.max()
        range_val = max_val - min_val
        
        if range_val == 0:
            return pd.Series([0.5] * len(group), index=group.index)
        
        return (group - min_val) / range_val
    
    df['Topology_PC1_normalized'] = df.groupby('dataset')['Topology_PC1'].transform(minmax_normalize_group)
    print("Added Topology_PC1_normalized column")

# Calculate RAW Overall Score (using raw QE and shifted PC1)
print("\n" + "="*60)
print("CALCULATING RAW OVERALL SCORE")
print("="*60)

# Calculate the minimum of Topology_PC1
min_pc1 = df['Topology_PC1'].min()

# Add the absolute minimum to all PC1 values to make them positive
df['PC1_positive'] = df['Topology_PC1'] - min_pc1

# Calculate the raw overall score: sqrt(quantization_error^2 + PC1_positive^2)
df['Overall_Score'] = np.sqrt(df['quantization_error']**2 + df['PC1_positive']**2)

print(f"Raw Overall Score Statistics:")
print(f"  Minimum: {df['Overall_Score'].min():.4f}")
print(f"  Maximum: {df['Overall_Score'].max():.4f}")
print(f"  Mean: {df['Overall_Score'].mean():.4f}")
print(f"  Median: {df['Overall_Score'].median():.4f}")

# Calculate NORMALIZED Overall Score (using normalized QE and PC1)
print("\n" + "="*60)
print("CALCULATING NORMALIZED OVERALL SCORE")
print("="*60)

# Check if quantization_error_normalized exists
if 'quantization_error_normalized' not in df.columns:
    print("ERROR: quantization_error_normalized column not found!")
    print("Please ensure main.py has been run with normalization enabled.")
else:
    # Calculate normalized overall score
    df['Normalized_Overall_Score'] = np.sqrt(
        df['quantization_error_normalized']**2 + 
        df['Topology_PC1_normalized']**2
    )
    
    print(f"Normalized Overall Score Statistics:")
    print(f"  Minimum: {df['Normalized_Overall_Score'].min():.4f}")
    print(f"  Maximum: {df['Normalized_Overall_Score'].max():.4f}")
    print(f"  Mean: {df['Normalized_Overall_Score'].mean():.4f}")
    print(f"  Median: {df['Normalized_Overall_Score'].median():.4f}")
    print(f"  Theoretical Maximum: {np.sqrt(2):.4f}")

# Display the best performer for each processing type (using normalized score)
print("\n" + "="*60)
print("BEST PERFORMERS BY PROCESSING TYPE (Normalized Score)")
print("="*60)

for proc_type in sorted(df['processing_type'].unique()):
    df_type = df[df['processing_type'] == proc_type].sort_values('Normalized_Overall_Score')
    best = df_type.iloc[0]
    print(f"\nProcessing Type: {proc_type}")
    print("-" * 50)
    print(f"  Scenario ID: {best['scenario_id']}")
    print(f"  Dataset: {best['dataset']}")
    print(f"  Sampling Method: {best['sampling_method']}")
    print(f"  Map Type: {best['map_type'] if 'map_type' in best else 'N/A'}")
    print(f"  QE (normalized): {best['quantization_error_normalized']:.4f}")
    print(f"  PC1 (normalized): {best['Topology_PC1_normalized']:.4f}")
    print(f"  Normalized Overall Score: {best['Normalized_Overall_Score']:.4f}")

# Find best performers for each dataset/algorithm/sampling combination
print("\n" + "="*60)
print("FINDING BEST PERFORMERS PER CATEGORY")
print("="*60)

best_performers = []

# Determine which columns exist for grouping
group_cols = ['dataset']
if 'algorithm' in df.columns:
    group_cols.append('algorithm')
elif 'processing_type' in df.columns:
    # Map processing_type to algorithm if needed
    df['algorithm'] = df['processing_type']
    group_cols.append('algorithm')

# Find the sampling method column
sampling_col = None
for col in ['sampling_method_final', 'sampling_method', 'sampling_method_parsed']:
    if col in df.columns:
        sampling_col = col
        break

if sampling_col:
    group_cols.append(sampling_col)
    # Ensure we have a standard column name
    if 'sampling_method_final' not in df.columns:
        df['sampling_method_final'] = df[sampling_col]

print(f"Grouping by: {group_cols}")

for group_values, group_df in df.groupby(group_cols):
    if len(group_df) > 0:
        # Find the row with minimum normalized overall score
        best_idx = group_df['Normalized_Overall_Score'].idxmin()
        best_row = group_df.loc[best_idx]
        
        result = {
            'Dataset': group_values[0],
            'Processing_Type': group_values[1] if len(group_values) > 1 else 'N/A',
            'Sampling_Method': group_values[2] if len(group_values) > 2 else 'N/A',
            'Best_Score': best_row['Normalized_Overall_Score'],
            'Best_Scenario': best_row['scenario_id'],
            'QE_Normalized': best_row['quantization_error_normalized'],
            'PC1_Normalized': best_row['Topology_PC1_normalized']
        }
        best_performers.append(result)

best_performers_df = pd.DataFrame(best_performers)

# Save the updated dataframe with all scores
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

df.to_csv('Data/processed_data_with_overall_score.csv', index=False)
print("✓ Saved full data to 'Data/processed_data_with_overall_score.csv'")

# Save best performers summary
best_performers_df.to_csv('Data/best_performers_by_sampling.csv', index=False)
print("✓ Saved best performers to 'Data/best_performers_by_sampling.csv'")

# Display summary
print("\n" + "="*60)
print("SUMMARY OF BEST PERFORMERS")
print("="*60)

for dataset in sorted(best_performers_df['Dataset'].unique()):
    print(f"\n{dataset}:")
    dataset_df = best_performers_df[best_performers_df['Dataset'] == dataset]
    for _, row in dataset_df.iterrows():
        print(f"  {row['Processing_Type']:8} / {row['Sampling_Method']:8} : Score = {row['Best_Score']:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  1. Data/processed_data_with_overall_score.csv - Full data with both raw and normalized scores")
print("  2. Data/best_performers_by_sampling.csv - Best performer for each category")
print("\nYou can now run Mann-Whitney analysis with:")
print("  python mann_whitney_normalized_analysis.py")
print("    OR")
print("  python mann_whitney_distance_analysis.py")