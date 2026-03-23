"""
Module for filtering the best performing rows based on a target metric (e.g., QE holdout)
"""

import pandas as pd
import numpy as np


def filter_top_performers(df, percentile=20, metric='quantization_error_holdout'):
    """
    Keep only the best performing rows (lowest metric value) for each dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned dataframe with a 'dataset' column
    percentile : float
        Percentage of best rows to keep per dataset (default: 20 for top 20%)
    metric : str
        Column name to use for ranking (default: 'quantization_error_holdout')
        
    Returns:
    --------
    df_filtered : pandas DataFrame
        DataFrame with only the top performers for each dataset
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns: {df.columns.tolist()}")

    if percentile >= 100:
        print(f"\nFiltering requested for {percentile}%, metric '{metric}' – retaining all rows.")
        return df.copy()

    print(f"\nFiltering to top {percentile}% performers (lowest QE) per dataset...")
    print("="*60)
    
    # List to store filtered data from each dataset
    filtered_dfs = []
    
    # Process each dataset separately
    for dataset in sorted(df['dataset'].unique()):
        # Get data for this dataset
        df_dataset = df[df['dataset'] == dataset]
        n_total = len(df_dataset)
        
        # Calculate the threshold (percentile of QE values)
        threshold = df_dataset[metric].quantile(percentile / 100)
        
        # Keep only rows with QE <= threshold
        df_best = df_dataset[df_dataset[metric] <= threshold]
        n_kept = len(df_best)
        
        # Add to results
        filtered_dfs.append(df_best)
        
        # Print info
        print(f"{dataset:20} Total: {n_total:5} → Kept: {n_kept:5} ({100*n_kept/n_total:.1f}%)")
        print(f"  QE threshold: {threshold:.6f}")
        print(f"  QE range kept: [{df_best[metric].min():.6f}, {df_best[metric].max():.6f}]")
    
    # Combine all filtered datasets
    df_filtered = pd.concat(filtered_dfs, ignore_index=True)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(df_filtered)} ({100*len(df_filtered)/len(df):.1f}%)")
    print(f"Mean QE before: {df[metric].mean():.6f}")
    print(f"Mean QE after:  {df_filtered[metric].mean():.6f}")
    print("="*60)
    
    return df_filtered
