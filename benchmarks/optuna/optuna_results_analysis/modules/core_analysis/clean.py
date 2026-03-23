"""
Data cleaning module - removes NaN and Inf values only
"""

import pandas as pd
import numpy as np


def clean_data(df, ignore_na_columns=None):
    """
    Clean dataframe by removing rows with NaN or infinite values

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to clean
    ignore_na_columns : list or None
        Column names to exclude from NaN-based row dropping

    Returns:
    --------
    df_clean : pandas DataFrame
        Cleaned DataFrame with NaN and Inf rows removed
    """
    print(f"Loaded {len(df)} rows")

    ignored = [col for col in (ignore_na_columns or []) if col in df.columns]
    if ignored:
        print(f"Ignoring NaN checks for columns: {ignored}")

    # Find rows with NaN values
    if ignored:
        na_check_columns = [col for col in df.columns if col not in ignored]
        rows_with_nan = df[na_check_columns].isna().any(axis=1)
    else:
        rows_with_nan = df.isna().any(axis=1)
    print(f"Rows with NaN: {rows_with_nan.sum()}")

    # Find rows with Inf values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    rows_with_inf = np.isinf(df[numeric_columns]).any(axis=1)
    print(f"Rows with Inf: {rows_with_inf.sum()}")

    # Combine both conditions
    bad_rows = rows_with_nan | rows_with_inf
    print(f"Total bad rows: {bad_rows.sum()}")

    # Create clean dataframe
    df_clean = df[~bad_rows]
    print(f"Rows after cleaning: {len(df_clean)}")

    return df_clean


def remove_zero_topographic_error(df):
    """
    Remove rows where topographic_error is 0
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with topographic_error column
        
    Returns:
    --------
    df_filtered : pandas DataFrame
        DataFrame with zero topographic_error rows removed
    """
    if 'topographic_error' not in df.columns:
        print("\nSkipping topographic_error filter – column not present.")
        return df

    print(f"\nRemoving rows with topographic_error = 0...")
    print(f"Rows before: {len(df)}")
    
    # Find rows where topographic_error is 0
    zero_te = df['topographic_error'] == 0
    print(f"Rows with topographic_error = 0: {zero_te.sum()}")
    
    if zero_te.sum() == 0:
        print("No rows removed.")
        return df
    
    # Remove these rows
    df_filtered = df[~zero_te].copy()
    print(f"Rows after removal: {len(df_filtered)}")
    print(f"Removed {zero_te.sum()} rows ({100*zero_te.sum()/len(df):.2f}%)")
    
    return df_filtered
