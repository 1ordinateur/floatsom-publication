import pandas as pd
import numpy as np

def normalize_within_dataset(df, columns_to_normalize):
    """
    Normalize columns within each dataset using min-max scaling (0-1 range)
    Each dataset gets normalized separately
    
    Parameters:
    df: pandas DataFrame with a 'dataset' column
    columns_to_normalize: list of column names to normalize
    
    Returns:
    DataFrame with new normalized columns (adds _normalized suffix)
    """
    
    df_normalized = df.copy()
    
    for col in columns_to_normalize:
        print(f"\nNormalizing {col}:")
        
        # For each dataset, scale to 0-1 range
        def minmax_normalize_group(group):
            min_val = group.min()
            max_val = group.max()
            range_val = max_val - min_val
            
            # Handle case where all values are the same
            if range_val == 0:
                print(f"  Dataset '{group.name}': all values are {min_val:.4f} (no variation)")
                return pd.Series([0.5] * len(group), index=group.index)
            
            print(f"  Dataset '{group.name}': range [{min_val:.4f}, {max_val:.4f}]")
            return (group - min_val) / range_val
        
        # Apply normalization within each dataset
        new_col_name = f"{col}_normalized"
        df_normalized[new_col_name] = df.groupby('dataset')[col].transform(minmax_normalize_group)
        
    print("\nNormalization complete!")
    return df_normalized


def check_normalization(df, original_col, normalized_col):
    """
    Verify that normalization worked correctly
    
    Parameters:
    df: DataFrame with both original and normalized columns
    original_col: name of original column
    normalized_col: name of normalized column
    """
    
    print(f"\nChecking normalization for {original_col}:")
    print("="*50)
    
    print("\nOriginal vs Normalized ranges by dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        
        orig_min = subset[original_col].min()
        orig_max = subset[original_col].max()
        norm_min = subset[normalized_col].min()
        norm_max = subset[normalized_col].max()
        
        print(f"\n{dataset}:")
        print(f"  Original range: [{orig_min:.4f}, {orig_max:.4f}]")
        print(f"  Normalized range: [{norm_min:.4f}, {norm_max:.4f}]")
        
        # Check if properly normalized to 0-1
        if abs(norm_min) < 0.001 and abs(norm_max - 1.0) < 0.001:
            print(f"  ✓ Correctly normalized to [0, 1]")
        else:
            print(f"  ⚠ Warning: Not exactly [0, 1]")


def compare_before_after(df, column, save_path=None):
    """
    Visualize the effect of normalization
    
    Parameters:
    df: DataFrame with both original and normalized columns
    column: original column name (without _normalized suffix)
    save_path: optional path to save the figure
    """
    
    import matplotlib.pyplot as plt
    
    normalized_col = f"{column}_normalized"
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    
    # Create subplots
    fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 4*n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for i, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset]
        
        # Original data
        axes[i, 0].hist(subset[column], bins=20, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'{dataset} - Original')
        axes[i, 0].set_xlabel(column)
        axes[i, 0].set_ylabel('Frequency')
        
        # Normalized data
        axes[i, 1].hist(subset[normalized_col], bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[i, 1].set_title(f'{dataset} - Normalized [0,1]')
        axes[i, 1].set_xlabel(f'{column}_normalized')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].set_xlim(0, 1)
    
    plt.suptitle(f'Normalization Effect: {column}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def normalize_all_metrics(df):
    """
    Convenience function to normalize all standard metrics
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with normalized columns added (with _normalized suffix)
    """
    
    # Candidate columns to normalize (ordered for readability)
    columns_to_normalize = [
        'quantization_error_train',
        'quantization_error_holdout',
        'quantization_error',
        'topographic_error',
        'neighborhood_preservation',
        'topographic_function',
        'distance_to_origin'
        # Note: Topology_PC1 is normalized separately after PCA computation
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    ordered_candidates = []
    for col in columns_to_normalize:
        if col not in seen:
            ordered_candidates.append(col)
            seen.add(col)
    
    # Check which columns exist in the dataframe
    existing_columns = [col for col in ordered_candidates if col in df.columns]
    
    if len(existing_columns) == 0:
        print("No standard columns found to normalize!")
        return df
    
    print(f"Normalizing {len(existing_columns)} columns: {existing_columns}")
    
    df_normalized = normalize_within_dataset(df, existing_columns)

    # Derived QE-only helper metrics (useful for cross-architecture comparisons).
    if (
        'quantization_error_train' in df_normalized.columns
        and 'quantization_error_holdout' in df_normalized.columns
        and 'balanced_qe_raw' not in df_normalized.columns
    ):
        df_normalized['balanced_qe_raw'] = df_normalized[
            ['quantization_error_train', 'quantization_error_holdout']
        ].mean(axis=1)

    if (
        'quantization_error_train_normalized' in df_normalized.columns
        and 'quantization_error_holdout_normalized' in df_normalized.columns
        and 'balanced_qe_normalized' not in df_normalized.columns
    ):
        df_normalized['balanced_qe_normalized'] = df_normalized[
            ['quantization_error_train_normalized', 'quantization_error_holdout_normalized']
        ].mean(axis=1)

    return df_normalized
