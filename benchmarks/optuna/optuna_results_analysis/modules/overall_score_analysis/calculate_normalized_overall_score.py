"""
Calculate Overall Score using NORMALIZED metrics
Default formula combines train QE and holdout QE (all normalized)
"""

import pandas as pd
import numpy as np
import os

def calculate_normalized_overall_score(df, metric_weights=None):
    """
    Calculate Overall Score using normalized metrics.

    Parameters:
    -----------
    df : pandas DataFrame
        Must contain the normalized metrics listed below.
    metric_weights : dict or None
        Optional dictionary mapping metric names to weights (default: equal weights).

    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added Normalized_Overall_Score column
    """

    df = df.copy()

    required_metrics = [
        'quantization_error_train_normalized',
        'quantization_error_holdout_normalized',
    ]

    missing = [col for col in required_metrics if col not in df.columns]
    if missing:
        raise ValueError(f"Missing normalized metrics needed for overall score: {missing}")

    default_weights = {
        'quantization_error_train_normalized': 1.0,
        'quantization_error_holdout_normalized': 1.0,
    }
    active_metrics = required_metrics
    weights = {
        metric: (
            metric_weights.get(metric)
            if metric_weights and metric in metric_weights
            else default_weights.get(metric, 1.0)
        )
        for metric in active_metrics
    }

    print("\nCalculating Normalized Overall Score...")
    print("Formula: sqrt(Σ w_i * metric_i^2) / sqrt(Σ w_i)")
    print("Weights:")
    for metric, weight in weights.items():
        print(f"  - {metric}: {weight}")

    numerator = np.zeros(len(df), dtype=float)
    weight_sum = np.zeros(len(df), dtype=float)

    for metric, weight in weights.items():
        series = pd.to_numeric(df[metric], errors='coerce')
        valid_mask = series.notna().to_numpy()
        values = series.to_numpy(dtype=float)
        numerator[valid_mask] += weight * (values[valid_mask] ** 2)
        weight_sum[valid_mask] += weight

    if np.all(weight_sum == 0):
        raise ValueError("No valid metric values available to compute normalized overall score.")

    df['Normalized_Overall_Score'] = np.where(
        weight_sum > 0,
        np.sqrt(numerator / weight_sum),
        np.nan,
    )
    
    # Print summary statistics
    print("\nNormalized Overall Score Statistics:")
    print(f"  Mean: {df['Normalized_Overall_Score'].mean(skipna=True):.4f}")
    print(f"  Std:  {df['Normalized_Overall_Score'].std(skipna=True):.4f}")
    print(f"  Min:  {df['Normalized_Overall_Score'].min(skipna=True):.4f}")
    print(f"  Max:  {df['Normalized_Overall_Score'].max(skipna=True):.4f}")
    
    # Print per dataset
    print("\nPer Dataset Statistics:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        print(f"  {dataset}:")
        print(f"    Mean: {dataset_df['Normalized_Overall_Score'].mean(skipna=True):.4f}")
        print(
            f"    Range: [{dataset_df['Normalized_Overall_Score'].min(skipna=True):.4f}, "
            f"{dataset_df['Normalized_Overall_Score'].max(skipna=True):.4f}]"
        )
    
    return df

def update_best_performers_with_normalized_score(df):
    """
    Find best performers using the normalized overall score
    
    Parameters:
    -----------
    df : pandas DataFrame with Normalized_Overall_Score
    
    Returns:
    --------
    best_performers : pandas DataFrame
    """
    
    # Get best performer for each combination
    best_performers = []
    
    for dataset in df['dataset'].unique():
        for processing in df['algorithm'].unique():
            # Check which sampling method column exists
            if 'sampling_method_parsed' in df.columns:
                sampling_col = 'sampling_method_parsed'
            elif 'sampling_method_final' in df.columns:
                sampling_col = 'sampling_method_final'
            else:
                sampling_col = 'sampling_method'
            
            for sampling in df[sampling_col].unique():
                # Get subset
                subset = df[
                    (df['dataset'] == dataset) & 
                    (df['algorithm'] == processing) & 
                    (df[sampling_col] == sampling)
                ]
                
                subset = subset.dropna(subset=['Normalized_Overall_Score'])
                if len(subset) > 0:
                    # Find the one with minimum normalized overall score
                    best_idx = subset['Normalized_Overall_Score'].idxmin()
                    best_row = subset.loc[best_idx]
                    
                    best_performers.append({
                        'Dataset': dataset,
                        'Architecture': best_row.get('architecture', best_row.get('map_type', 'N/A')),
                        'Processing_Type': processing,
                        'Sampling_Method': sampling,
                        'Best_Normalized_Score': best_row['Normalized_Overall_Score'],
                        'QE_Holdout_Normalized': best_row.get('quantization_error_holdout_normalized', np.nan),
                        'QE_Train_Normalized': best_row.get('quantization_error_train_normalized', np.nan),
                        'Best_Scenario': best_row['scenario_id']
                    })
    
    return pd.DataFrame(best_performers)

def save_normalized_results(df, best_performers):
    """Save the results with normalized overall scores"""
    
    # Save updated dataframe
    df.to_csv('Data/processed_data_with_normalized_overall_score.csv', index=False)
    print("\nSaved updated data to 'Data/processed_data_with_normalized_overall_score.csv'")
    
    # Save best performers
    best_performers.to_csv('Data/best_performers_normalized_score.csv', index=False)
    print("Saved best performers to 'Data/best_performers_normalized_score.csv'")
    
    # Create summary report
    with open('Results/Summary_Outputs/normalized_score_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Normalized Overall Score Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis uses **normalized metrics** to calculate the overall score:\n")
        f.write("- Train QE and holdout QE are normalized to [0,1] range within each dataset\n")
        f.write("- Overall Score = sqrt(Σ w_i * metric_i²) / sqrt(Σ w_i)\n")
        f.write("- Default weights give equal influence to train QE and holdout QE\n\n")
        
        f.write("## Key Statistics\n\n")
        f.write(f"- Mean Normalized Overall Score: {df['Normalized_Overall_Score'].mean():.4f}\n")
        f.write(f"- Score Range: [{df['Normalized_Overall_Score'].min():.4f}, {df['Normalized_Overall_Score'].max():.4f}]\n")
        f.write(f"- Theoretical Range: [0, 1]\n\n")
        
        f.write("## Best Performers by Category\n\n")
        
        for dataset in sorted(best_performers['Dataset'].unique()):
            f.write(f"\n### {dataset}\n\n")
            dataset_best = best_performers[best_performers['Dataset'] == dataset]
            f.write("| Processing | Sampling | Normalized Score | QE Holdout (norm) | QE Train (norm) |\n")
            f.write("|------------|----------|------------------|-------------------|-----------------|\n")
            for _, row in dataset_best.iterrows():
                f.write(f"| {row['Processing_Type']} | {row['Sampling_Method']} | ")
                f.write(f"{row['Best_Normalized_Score']:.4f} | ")
                f.write(f"{row['QE_Holdout_Normalized']:.4f} | ")
                f.write(f"{row['QE_Train_Normalized']:.4f} |\n")
    
    print("Created summary report at 'Results/Summary_Outputs/normalized_score_summary.md'")

def main():
    """Main function to run the normalized overall score calculation"""
    
    print("="*60)
    print("CALCULATING NORMALIZED OVERALL SCORE")
    print("="*60)
    
    # Load the data - try multiple possible file names
    print("\nLoading data...")
    
    # Try to load the file with the best name
    import os
    possible_files = [
        'Data/processed_data_with_overall_score.csv',
        'Data/processed_data_with_pca.csv',
        'Data/processed_data.csv'
    ]
    
    df = None
    for filename in possible_files:
        if os.path.exists(filename):
            print(f"Found file: {filename}")
            df = pd.read_csv(filename)
            break
    
    if df is None:
        print("ERROR: No processed data file found!")
        print("Please run 'python main.py' first to generate the processed data.")
        return None, None
    
    print(f"Loaded {len(df)} rows")
    
    # Calculate normalized overall score
    df = calculate_normalized_overall_score(df)
    
    # Find best performers
    print("\nFinding best performers with normalized score...")
    best_performers = update_best_performers_with_normalized_score(df)
    
    # Save results
    save_normalized_results(df, best_performers)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    return df, best_performers

if __name__ == "__main__":
    df, best_performers = main()
