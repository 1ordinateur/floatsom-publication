#!/usr/bin/env python3
"""
Extract the best performing scenarios from the original dataset based on the best performers analysis.
This script finds the records with the best overall scores for each combination of 
processing type and sampling method, then saves their complete records to a new CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_best_performer_records(output_dir="results", architecture_suffix=""):
    """
    Extract complete records for the best performing scenarios from the original dataset.
    
    Parameters:
    -----------
    output_dir : str
        Output directory path (default: "results")
    architecture_suffix : str
        Suffix to append to filenames (e.g., "_mst" or "_hexagonal")
    """
    # Define file paths
    output_path = Path(output_dir)
    results_dir = output_path / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    best_performers_file = output_path / f"best_performers_by_sampling{architecture_suffix}.csv"
    original_data_file = output_path / f"processed_data_with_overall_score{architecture_suffix}.csv"
    output_file = results_dir / f"best_performers_complete_records{architecture_suffix}.csv"
    
    # Load the best performers summary
    print("Loading best performers summary...")
    best_performers = pd.read_csv(best_performers_file)
    
    # Load the original complete dataset
    print("Loading original dataset...")
    original_data = pd.read_csv(original_data_file)
    
    # Extract the scenario IDs from the Best_Scenario column
    best_scenario_ids = best_performers['Best_Scenario'].tolist()
    print(f"Found {len(best_scenario_ids)} best performing scenarios")
    
    # Filter the original data to include only the best performers
    print("Filtering original data for best performers...")
    best_records = original_data[original_data['scenario_id'].isin(best_scenario_ids)]
    
    # For each best scenario, we want to get the record with the highest overall score
    print("Selecting records with highest overall scores for each scenario...")
    final_records = []
    
    for scenario_id in best_scenario_ids:
        scenario_records = best_records[best_records['scenario_id'] == scenario_id]
        if not scenario_records.empty:
            # Get the record with the maximum Normalized_Overall_Score
            best_record = scenario_records.loc[scenario_records['Normalized_Overall_Score'].idxmax()]
            final_records.append(best_record)
            print(f"  {scenario_id}: Score = {best_record['Normalized_Overall_Score']:.6f}")
    
    # Create DataFrame from the selected records
    best_performers_df = pd.DataFrame(final_records)
    
    # Sort by dataset, processing_type, and sampling_method for better organization
    best_performers_df = best_performers_df.sort_values(
        by=['dataset', 'processing_type', 'sampling_method']
    )
    
    # Save to CSV
    print(f"\nSaving {len(best_performers_df)} best performer records to {output_file}")
    best_performers_df.to_csv(output_file, index=False)
    
    # Display summary statistics
    print("\nSummary of extracted best performers:")
    print(f"Total records: {len(best_performers_df)}")
    print(f"Unique datasets: {best_performers_df['dataset'].nunique()}")
    print(f"Processing types: {best_performers_df['processing_type'].unique().tolist()}")
    print(f"Sampling methods: {best_performers_df['sampling_method'].unique().tolist()}")
    
    # Show distribution by dataset
    print("\nDistribution by dataset:")
    dataset_counts = best_performers_df['dataset'].value_counts().sort_index()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} records")
    
    # Show average scores by processing type and sampling method
    print("\nAverage normalized overall scores:")
    avg_scores = best_performers_df.groupby(['processing_type', 'sampling_method'])['Normalized_Overall_Score'].mean()
    for (proc_type, samp_method), score in avg_scores.items():
        print(f"  {proc_type} + {samp_method}: {score:.6f}")
    
    return best_performers_df

def create_compact_summary(df, output_dir="results", architecture_suffix=""):
    """
    Create a compact summary CSV with only the most important columns.
    
    Parameters:
    -----------
    output_dir : str
        Output directory path (default: "results")
    architecture_suffix : str
        Suffix to append to filenames (e.g., "_mst" or "_hexagonal")
    """
    output_path = Path(output_dir)
    results_dir = output_path / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"best_performers_compact{architecture_suffix}.csv"
    
    # Select the most important columns
    important_columns = [
        'scenario_id',
        'dataset',
        'processing_type',
        'sampling_method',
        'trial_number',
        'quantization_error_train',
        'quantization_error_holdout',
        'Normalized_Overall_Score',
        'quantization_error_train_normalized',
        'quantization_error_holdout_normalized'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in important_columns if col in df.columns]
    compact_df = df[available_columns].copy()
    
    # Round numerical values for readability
    numeric_columns = compact_df.select_dtypes(include=[np.number]).columns
    compact_df[numeric_columns] = compact_df[numeric_columns].round(6)
    
    # Save the compact version
    print(f"\nSaving compact summary to {output_file}")
    compact_df.to_csv(output_file, index=False)
    
    print(f"Compact summary contains {len(available_columns)} columns")
    
    return compact_df

if __name__ == "__main__":
    print("=" * 80)
    print("Extracting Best Performer Records")
    print("=" * 80)
    
    # Extract the best performers
    best_performers_df = extract_best_performer_records()
    
    # Create a compact summary as well
    print("\n" + "=" * 80)
    print("Creating Compact Summary")
    print("=" * 80)
    compact_df = create_compact_summary(best_performers_df)
    
    print("\n" + "=" * 80)
    print("Extraction complete!")
    print("Generated files:")
    print("  - Results/tables/best_performers_complete_records.csv (full records)")
    print("  - Results/tables/best_performers_compact.csv (compact summary)")
    print("=" * 80)
