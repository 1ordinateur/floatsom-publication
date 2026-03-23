#!/usr/bin/env python3
"""
Main analysis pipeline for Pareto front data
Orchestrates: Clean -> Normalize -> PCA -> Statistical Analysis
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our modules
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.clean import clean_data, remove_zero_topographic_error
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.filter_best import filter_top_performers
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.normalization import normalize_all_metrics
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.visualization_helpers import add_analysis_columns, plot_algorithm_comparison, save_statistics, plot_pareto_fronts
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.statistical_analysis import perform_algorithm_comparisons
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.paired_analysis import (
    generate_algorithm_paired_analysis_report,
    generate_topology_paired_analysis_report,
)
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.overall_score_analysis.mann_whitney_normalized import perform_normalized_comparisons
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.unified_summary import create_unified_summary
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.core_analysis.extract_best_performers import extract_best_performer_records, create_compact_summary
from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.parameter_summary import build_parameter_summary, MetricConfig


def _optional_columns_to_ignore_na(df):
    """Optional metrics/params should not drive row-level NaN filtering."""
    optional_candidates = [
        'distortion_measure_holdout',
        'distortion_measure_train',
        'distortion_measure',
        'distortion_measure_holdout_normalized',
        'distortion_measure_train_normalized',
    ]
    optional_columns = [col for col in optional_candidates if col in df.columns]
    optional_columns.extend(col for col in df.columns if col.startswith('param_'))
    optional_columns.extend(col for col in df.columns if col.startswith('config_'))
    return optional_columns


def _coalesce_text_column(df, target, candidates):
    if target not in df.columns:
        df[target] = np.nan
    for candidate in candidates:
        if candidate in df.columns:
            df[target] = df[target].fillna(df[candidate])
    df[target] = df[target].astype(str).str.lower().str.strip()
    df.loc[df[target].isin(['nan', 'none', '']), target] = np.nan


def _coalesce_processing_columns(df):
    _coalesce_text_column(
        df,
        target='processing_type',
        candidates=[
            'processing_type',
            'config_processing_method',
            'config_processing',
            'processing_method',
            'processing',
            'processor_type',
            'config_processor_type',
        ],
    )
    _coalesce_text_column(
        df,
        target='sampling_method',
        candidates=[
            'sampling_method',
            'config_sampling_method',
            'config_sampling',
            'sampling',
            'selector_type',
            'config_selector_type',
            'selector',
        ],
    )


def _coalesce_distortion_columns(df):
    """Populate holdout distortion aliases from train distortion when needed."""
    if (
        'distortion_measure_holdout' in df.columns
        and pd.to_numeric(df['distortion_measure_holdout'], errors='coerce').notna().any()
    ):
        return
    if 'distortion_measure_train' in df.columns:
        train_values = pd.to_numeric(df['distortion_measure_train'], errors='coerce')
        if train_values.notna().any():
            df['distortion_measure_holdout'] = train_values


def _coalesce_architecture_column(df):
    """Ensure a canonical architecture column exists for downstream tooling."""
    df['architecture'] = df['map_type']
    if 'config_topology_type' in df.columns:
        df['architecture'] = df['architecture'].fillna(df['config_topology_type'])
    if 'param_topology_type' in df.columns:
        df['architecture'] = df['architecture'].fillna(df['param_topology_type'])
    df['architecture'] = df['architecture'].astype(str).str.lower().str.strip()
    df.loc[df['architecture'].isin(['nan', 'none', '']), 'architecture'] = np.nan


def run_analysis(data_file='results/pareto_front_results.csv', 
                 output_dir='results',
                 save_results=True,
                 perform_statistical_tests=True,
                 datasets_to_exclude=None,
                 filter_top_percent=100,
                 calculate_best_performers=False,
                 architecture_filter=None,
                 colors=False):
    """
    Run complete analysis pipeline
    
    Parameters:
    -----------
    data_file : str
        Path to input CSV file
    output_dir : str
        Output directory for results (default: 'results')
    save_results : bool
        Whether to save processed data and plots
    perform_statistical_tests : bool
        Whether to perform Mann-Whitney U tests comparing algorithms
    datasets_to_exclude : list or None
        List of dataset names to exclude from plots (e.g., ['breast_cancer', 'swiss_roll'])
    filter_top_percent : float or None
        If specified, keep only the top X% performers (lowest QE) per dataset (default: 20)
        Set to None to keep all data
    calculate_best_performers : bool
        Whether to calculate overall scores and identify best performers
    architecture_filter : str or None
        If specified, filter to only this architecture type (e.g., 'hexagonal', 'mst')
        
    Returns:
    --------
    df_final : pandas DataFrame
        Processed dataframe with all analysis columns
    pca_info : dict
        PCA information dictionary
    mann_whitney_results : dict or None
        Results from Mann-Whitney U tests for each sampling method
    """
    
    print("="*60)
    print("PARETO FRONT ANALYSIS PIPELINE")
    print("="*60)
    
    # Create output directories - use subdirectory for architecture-specific analysis
    if architecture_filter:
        output_path = Path(output_dir) / architecture_filter.lower()
        print(f"Output directory: {output_path}")
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'tables').mkdir(exist_ok=True)
    (output_path / 'summary_outputs').mkdir(exist_ok=True)
    
    # Step 1: Load and clean data
    print("\n1. Loading and cleaning data...")
    print("-"*40)
    df = pd.read_csv(data_file)
    _coalesce_processing_columns(df)
    print(f"Original shape: {df.shape}")
    
    # Remove NaN and Inf
    df_clean = clean_data(df, ignore_na_columns=_optional_columns_to_ignore_na(df))
    _coalesce_distortion_columns(df_clean)
    print(f"Shape after cleaning NaN/Inf: {df_clean.shape}")

    required_new_metrics = [
        'quantization_error_train',
        'quantization_error_holdout',
    ]
    missing_new_metrics = [col for col in required_new_metrics if col not in df_clean.columns]
    if missing_new_metrics:
        raise ValueError(f"Missing required metric columns: {missing_new_metrics}")

    # Remove rows with topographic_error = 0
    df_clean = remove_zero_topographic_error(df_clean)
    print(f"Shape after removing zero TE: {df_clean.shape}")
    
    # Step 2: Filter to top performers if requested
    if filter_top_percent is not None and filter_top_percent < 100:
        print(f"\n2. Filtering to top {filter_top_percent}% performers per dataset...")
        print("-"*40)
        df_filtered = filter_top_performers(
            df_clean,
            percentile=filter_top_percent,
            metric='quantization_error_holdout'
        )
        print(f"Final shape after filtering: {df_filtered.shape}")
    else:
        df_filtered = df_clean
        print(f"\n2. No filtering applied - keeping all data")
    
    # Step 3: Normalize
    print("\n3. Normalizing metrics...")
    print("-"*40)
    df_normalized = normalize_all_metrics(df_filtered)
    print(f"Shape after normalization: {df_normalized.shape}")
    
    # Step 4: No PCA (legacy topology bundle not present)
    print("\n4. Skipping PCA - topology bundle not available in new schema")
    df_with_pca = df_normalized.copy()
    pca_info = None
    
    # Step 5: Add analysis columns
    print("\n5. Parsing scenario information...")
    print("-"*40)
    df_final = add_analysis_columns(df_with_pca)
    if not colors and 'algorithm' in df_final.columns:
        mask_colors = df_final['algorithm'].astype(str).str.lower().str.strip() == 'colors'
        df_final = df_final[~mask_colors].copy()
    
    print(f"Algorithms found: {list(df_final['algorithm'].dropna().unique())}")
    
    # Check which sampling column exists and use it
    if 'sampling_method' in df_final.columns:
        print(f"Sampling methods: {list(df_final['sampling_method'].dropna().unique())}")
    elif 'sampling_method_final' in df_final.columns:
        print(f"Sampling methods: {list(df_final['sampling_method_final'].dropna().unique())}")
    else:
        print(f"Sampling methods: {list(df_final['sampling_method_parsed'].dropna().unique())}")
    
    print(f"Datasets: {list(df_final['dataset'].unique())}")
    _coalesce_architecture_column(df_final)
    print(f"Architectures found: {list(df_final['architecture'].dropna().unique())}")
    
    # Filter by architecture if specified
    if architecture_filter:
        print(f"\nFiltering to architecture: {architecture_filter}")
        df_final = df_final[df_final['architecture'] == architecture_filter]
        print(f"Shape after architecture filtering: {df_final.shape}")
    
    # Step 6: Create visualizations and compute statistics
    print("\n6. Creating visualizations and computing statistics...")
    print("-"*40)
    
    # Create plots
    fig = plot_algorithm_comparison(df_final, save_path=str(output_path / 'plots') + '/',
                                   datasets_to_exclude=datasets_to_exclude, 
                                   filename_suffix='')
    
    # Create Pareto front plots (both normalized and raw versions)
    # If architecture_filter is specified, it's already filtered in df_final
    # Otherwise, create separate plots for each architecture if they exist
    if architecture_filter:
        # Single architecture specified
        # Create normalized version
        pareto_fig_norm = plot_pareto_fronts(df_final, save_path=str(output_path / 'plots') + '/',
                                            architecture=None,  # Already filtered
                                            datasets_to_exclude=datasets_to_exclude,
                                            filename_suffix='',
                                            use_raw=False)
        # Create raw version
        pareto_fig_raw = plot_pareto_fronts(df_final, save_path=str(output_path / 'plots') + '/',
                                           architecture=None,  # Already filtered
                                           datasets_to_exclude=datasets_to_exclude,
                                           filename_suffix='',
                                           use_raw=True)
    else:
        # Check what architectures exist in the data
        if 'architecture' in df_final.columns:
            architectures = df_final['architecture'].dropna().unique()
            for arch in architectures:
                # Create normalized version
                pareto_fig_norm = plot_pareto_fronts(df_final, save_path=str(output_path / 'plots') + '/',
                                                    architecture=arch,
                                                    datasets_to_exclude=datasets_to_exclude,
                                                    filename_suffix='',
                                                    use_raw=False)
                # Create raw version
                pareto_fig_raw = plot_pareto_fronts(df_final, save_path=str(output_path / 'plots') + '/',
                                                   architecture=arch,
                                                   datasets_to_exclude=datasets_to_exclude,
                                                   filename_suffix='',
                                                   use_raw=True)
    
    # Save statistics
    stats_file = str(output_path / 'summary_outputs' / 'statistics.txt')
    save_statistics(df_final, output_file=stats_file)
    
    # Step 7: Perform Mann-Whitney U tests if requested
    mann_whitney_results = None
    if perform_statistical_tests:
        print("\n7. Performing Mann-Whitney U Tests...")
        print("-"*40)
        
        # Run the comparisons
        mann_whitney_results = perform_algorithm_comparisons(df_final, 
                                                            save_path=str(output_path / 'plots') + '/',
                                                            filename_suffix='')
        
        # Results are now included in the unified summary at the end
    
    print("\n8. Calculating Overall Scores and Best Performers...")
    print("-"*40)
    
    # Import the functions we need
    from floatsom.benchmarks.optuna.optuna_results_analysis.modules.overall_score_analysis.calculate_normalized_overall_score import (
        calculate_normalized_overall_score,
        update_best_performers_with_normalized_score
    )
    
    # Calculate normalized overall score
    df_final = calculate_normalized_overall_score(df_final)
    
    # Find best performers
    best_performers_df = update_best_performers_with_normalized_score(df_final)
    
    # Display summary
    print("\n" + "="*60)
    print("BEST PERFORMERS BY ALGORITHM (Normalized Score)")
    print("="*60)
    
    for algorithm in sorted(df_final['algorithm'].unique()):
        df_alg = df_final[df_final['algorithm'] == algorithm].sort_values('Normalized_Overall_Score')
        best = df_alg.iloc[0]
        print(f"\nAlgorithm: {algorithm}")
        print("-" * 50)
        print(f"  Scenario ID: {best['scenario_id']}")
        print(f"  Dataset: {best['dataset']}")
        print(f"  Sampling Method: {best['sampling_method_parsed']}")
        if 'quantization_error_holdout_normalized' in best:
            print(f"  QE Holdout (normalized): {best['quantization_error_holdout_normalized']:.4f}")
        if 'quantization_error_train_normalized' in best:
            print(f"  QE Train (normalized):   {best['quantization_error_train_normalized']:.4f}")
        if 'distortion_measure_holdout_normalized' in best:
            print(f"  Distortion (normalized): {best['distortion_measure_holdout_normalized']:.4f}")
        if 'Topology_PC1_normalized' in best:
            print(f"  Topology PC1 (normalized): {best['Topology_PC1_normalized']:.4f}")
        print(f"  Normalized Overall Score: {best['Normalized_Overall_Score']:.4f}")

    # Step 8a: Generate paired analysis markdown reports
    print("\n8a. Generating paired analysis reports...")
    print("-"*40)
    algo_paired_analysis_path = generate_algorithm_paired_analysis_report(
        df=df_final,
        output_file=output_path / 'algo_comparison_paired.md',
    )
    print(f"Saved algorithm paired analysis report to {algo_paired_analysis_path}")

    topo_paired_analysis_path = generate_topology_paired_analysis_report(
        df=df_final,
        output_file=output_path / 'topo_comparison_paired.md',
    )
    print(f"Saved topology paired analysis report to {topo_paired_analysis_path}")

    # Step 8b: Perform Mann-Whitney tests on normalized overall score
    print("\n8b. Performing Mann-Whitney U Tests on Normalized Overall Score...")
    print("-"*40)
    
    mann_whitney_normalized_results = perform_normalized_comparisons(df_final)
    

    print("\n9. Saving processed data...")
    print("-"*40)
    
    # Save processed data (no suffix needed since we're using subdirectories)
    output_csv = output_path / 'processed_data_with_pca.csv'
    df_final.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")
    
    output_csv_scores = output_path / 'processed_data_with_overall_score.csv'
    df_final.to_csv(output_csv_scores, index=False)
    print(f"Saved data with overall scores to {output_csv_scores}")
    
    # Save best performers
    best_performers_csv = output_path / 'best_performers_by_sampling.csv'
    best_performers_df.to_csv(best_performers_csv, index=False)
    print(f"Saved best performers to {best_performers_csv}")
    
    # Extract complete records for best performers
    print("\n9a. Extracting complete records for best performers...")
    print("-"*40)
    extracted_df = extract_best_performer_records(output_dir=str(output_path), architecture_suffix='')
    compact_df = create_compact_summary(extracted_df, output_dir=str(output_path), architecture_suffix='')
    print(f"Extracted {len(extracted_df)} best performer records")
    
    # Step 10: Create unified summary report
    print("\n10. Creating Unified Summary Report...")
    print("-"*40)
    
    # Create the comprehensive summary
    create_unified_summary(
        df_final=df_final,
        mann_whitney_results=mann_whitney_results,
        mann_whitney_normalized_results=mann_whitney_normalized_results if calculate_best_performers else None,
        best_performers_df=best_performers_df if calculate_best_performers else None,
        pca_info=pca_info,
        output_file=str(output_path / 'ANALYSIS_SUMMARY.md')
    )

    print("\n11. Building Parameter Summary Tables...")
    print("-"*40)
    metric_configs = [
        MetricConfig(
            column='Normalized_Overall_Score',
            label='Normalized Overall Score',
            slug='normalized_overall'
        ),
        MetricConfig(
            column='balanced_qe_normalized',
            label='Balanced QE (Train+Holdout, Normalized)',
            slug='balanced_qe_normalized'
        ),
        MetricConfig(
            column='quantization_error_holdout_normalized',
            label='Quantization Error (Holdout, Normalized)',
            slug='qe_holdout_normalized'
        ),
        MetricConfig(
            column='quantization_error_train_normalized',
            label='Quantization Error (Train, Normalized)',
            slug='qe_train_normalized'
        ),
    ]

    parameter_summary_path = build_parameter_summary(
        df=df_final,
        metrics=metric_configs,
        output_dir=output_path,
        architecture_label=architecture_filter,
        per_dataset_top_k=10
    )
    print(f"Saved parameter summary markdown to {parameter_summary_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Total rows processed: {len(df_final)}")
    print(f"  Number of datasets: {df_final['dataset'].nunique()}")
    print(f"  Number of algorithms: {df_final['algorithm'].nunique()}")
    print(f"  Number of sampling methods: {df_final['sampling_method_parsed'].nunique()}")
    
    if mann_whitney_results:
        print(f"  Mann-Whitney U tests performed: {len(mann_whitney_results)}")
        
        # Print quick summary
        print("\n" + "="*60)
        print("MANN-WHITNEY U TEST SUMMARY")
        print("="*60)
        
        metric_names = {
            'quantization_error_holdout_normalized': 'Quantization Error (Holdout, normalized)',
            'quantization_error_train_normalized': 'Quantization Error (Train, normalized)',
            'balanced_qe_normalized': 'Balanced QE (Train+Holdout, normalized)',
            'distortion_measure_holdout_normalized': 'Distortion (Holdout, normalized)'
        }

        for sampling, sampling_results in mann_whitney_results.items():
            print(f"\n{sampling} sampling:")

            for metric_key, display_name in metric_names.items():
                if metric_key in sampling_results:
                    result = sampling_results[metric_key]
                    print(f"\n  {display_name}:")
                    print(f"    Batch mean:  {result['group1_mean']:.4f}")
                    print(f"    Colors mean: {result['group2_mean']:.4f}")
                    print(f"    P-value:     {result['p_value']:.6f}")
                    print(f"    Significant: {result['significance']}")
                    print(f"    Effect size: {result['effect_size']:.3f}")
    
    return df_final, pca_info, mann_whitney_results


def run_analysis_by_architecture(data_file='results/pareto_front_results.csv',
                                output_dir='results',
                                save_results=True,
                                perform_statistical_tests=True,
                                datasets_to_exclude=None,
                                filter_top_percent=100,
                                calculate_best_performers=False,
                                colors=False):
    """
    Detect all architectures in the data and run analysis for each separately
    
    Parameters are the same as run_analysis()
    
    Returns:
    --------
    dict : Dictionary with architecture as key and (df_final, pca_info, mann_whitney_results) as values
    """
    
    print("\n" + "="*70)
    print("DETECTING ARCHITECTURES IN DATA")
    print("="*70)
    
    # First, load and parse the data to detect architectures
    df_temp = pd.read_csv(data_file)
    _coalesce_processing_columns(df_temp)
    df_temp = clean_data(df_temp, ignore_na_columns=_optional_columns_to_ignore_na(df_temp))
    _coalesce_distortion_columns(df_temp)
    df_temp = remove_zero_topographic_error(df_temp)
    df_temp = add_analysis_columns(df_temp)
    if not colors and 'algorithm' in df_temp.columns:
        mask_colors = df_temp['algorithm'].astype(str).str.lower().str.strip() == 'colors'
        df_temp = df_temp[~mask_colors].copy()
    _coalesce_architecture_column(df_temp)
    
    # Get unique architectures
    architectures = df_temp['architecture'].dropna().unique()
    print(f"\nFound {len(architectures)} architecture(s): {list(architectures)}")
    
    # Store results for each architecture
    results = {}
    
    # Analyze each architecture separately
    for i, architecture in enumerate(architectures, 1):
        print("\n" + "="*70)
        print(f"ANALYZING ARCHITECTURE {i}/{len(architectures)}: {architecture.upper()}")
        print("="*70)
        
        df_result, pca_info, mann_whitney_results = run_analysis(
            data_file=data_file,
            output_dir=output_dir,
            save_results=save_results,
            perform_statistical_tests=perform_statistical_tests,
            datasets_to_exclude=datasets_to_exclude,
            filter_top_percent=filter_top_percent,
            calculate_best_performers=calculate_best_performers,
            architecture_filter=architecture,
            colors=colors,
        )
        
        results[architecture] = (df_result, pca_info, mann_whitney_results)
    
    print("\n" + "="*70)
    print("ALL ARCHITECTURES ANALYZED SUCCESSFULLY")
    print("="*70)
    
    # Summary across all architectures
    print("\nSummary by Architecture:")
    for arch, (df, _, _) in results.items():
        print(f"\n{arch.upper()}:")
        print(f"  Total rows: {len(df)}")
        print(f"  Datasets: {df['dataset'].nunique()}")
        print(f"  Algorithms: {df['algorithm'].nunique()}")
        print(f"  Sampling methods: {df['sampling_method_parsed'].nunique()}")
    
    # Create unified analysis if multiple architectures exist
    if len(results) > 1 and save_results:
        print("\n" + "="*70)
        print("CREATING UNIFIED ANALYSIS")
        print("="*70)
        
        # Combine all dataframes with architecture column
        all_dfs = []
        
        for arch, (df, pca_info, mann_whitney) in results.items():
            df_copy = df.copy()
            df_copy['architecture'] = arch
            all_dfs.append(df_copy)
        
        # Create unified dataframe
        df_unified = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined data shape: {df_unified.shape}")
        
        # Create unified output directory
        unified_path = Path(output_dir) / 'unified'
        unified_path.mkdir(parents=True, exist_ok=True)
        (unified_path / 'plots').mkdir(exist_ok=True)
        (unified_path / 'tables').mkdir(exist_ok=True)
        (unified_path / 'summary_outputs').mkdir(exist_ok=True)
        print(f"Unified output directory: {unified_path}")
        
        # Normalize metrics (ensure consistency across merged architectures)
        df_unified_with_metrics = normalize_all_metrics(df_unified)
        df_unified_with_pca = df_unified_with_metrics.copy()
        unified_pca_info = None

        # Calculate overall scores (needed for paired analysis) and optional best performers
        unified_best_performers = None
        unified_mann_whitney_norm = None
        print("\nCalculating overall scores for unified data...")
        from floatsom.benchmarks.optuna.optuna_results_analysis.modules.overall_score_analysis.calculate_normalized_overall_score import (
            calculate_normalized_overall_score,
            update_best_performers_with_normalized_score
        )
        df_unified_with_pca = calculate_normalized_overall_score(df_unified_with_pca)

        if calculate_best_performers:
            unified_best_performers = update_best_performers_with_normalized_score(df_unified_with_pca)
            
            # Perform Mann-Whitney on normalized score
            from floatsom.benchmarks.optuna.optuna_results_analysis.modules.overall_score_analysis.mann_whitney_normalized import perform_normalized_comparisons
            unified_mann_whitney_norm = perform_normalized_comparisons(df_unified_with_pca)
        
        # Perform Mann-Whitney tests on unified data if requested
        unified_mann_whitney = None
        if perform_statistical_tests:
            print("\nPerforming Mann-Whitney tests on unified data...")
            unified_mann_whitney = perform_algorithm_comparisons(
                df_unified_with_pca,
                save_path=str(unified_path / 'plots') + '/',
                filename_suffix=''
            )
        
        # Save unified CSV files
        print("\nSaving unified data files...")
        df_unified_with_pca.to_csv(unified_path / 'processed_data_with_pca.csv', index=False)
        print(f"Saved: {unified_path / 'processed_data_with_pca.csv'}")
        
        if calculate_best_performers:
            df_unified_with_pca.to_csv(unified_path / 'processed_data_with_overall_score.csv', index=False)
            unified_best_performers.to_csv(unified_path / 'best_performers_by_sampling.csv', index=False)
            print(f"Saved: {unified_path / 'processed_data_with_overall_score.csv'}")
            print(f"Saved: {unified_path / 'best_performers_by_sampling.csv'}")
        
        # Create unified plots
        print("\nCreating unified visualizations...")
        
        fig = plot_algorithm_comparison(
            df_unified_with_pca,
            save_path=str(unified_path / 'plots') + '/',
            datasets_to_exclude=datasets_to_exclude,
            filename_suffix=''
        )
        
        # Create Pareto front plots for each architecture (both normalized and raw)
        for arch in sorted(df_unified_with_pca['architecture'].dropna().astype(str).unique()):
            arch_data = df_unified_with_pca[df_unified_with_pca['architecture'] == arch]
            if len(arch_data) > 0:
                # Create normalized version
                pareto_fig_norm = plot_pareto_fronts(
                    arch_data,
                    save_path=str(unified_path / 'plots') + '/',
                    architecture=None,  # Already filtered
                    datasets_to_exclude=datasets_to_exclude,
                    filename_suffix=f'_{arch}',
                    use_raw=False
                )
                # Create raw version
                pareto_fig_raw = plot_pareto_fronts(
                    arch_data,
                    save_path=str(unified_path / 'plots') + '/',
                    architecture=None,  # Already filtered
                    datasets_to_exclude=datasets_to_exclude,
                    filename_suffix=f'_{arch}',
                    use_raw=True
                )
        
        # Save unified statistics
        stats_file = str(unified_path / 'summary_outputs' / 'statistics.txt')
        save_statistics(df_unified_with_pca, output_file=stats_file)

        # Create paired analysis reports for unified data
        print("\nGenerating paired analysis reports for unified data...")
        unified_algo_paired_path = generate_algorithm_paired_analysis_report(
            df=df_unified_with_pca,
            output_file=unified_path / 'algo_comparison_paired.md'
        )
        print(f"Saved unified algorithm paired analysis report to {unified_algo_paired_path}")

        unified_topo_paired_path = generate_topology_paired_analysis_report(
            df=df_unified_with_pca,
            output_file=unified_path / 'topo_comparison_paired.md'
        )
        print(f"Saved unified topology paired analysis report to {unified_topo_paired_path}")
        
        # Create unified summary report
        print("\nCreating unified summary report...")
        create_unified_summary(
            df_final=df_unified_with_pca,
            mann_whitney_results=unified_mann_whitney,
            mann_whitney_normalized_results=unified_mann_whitney_norm if calculate_best_performers else None,
            best_performers_df=unified_best_performers if calculate_best_performers else None,
            pca_info=unified_pca_info,
            output_file=str(unified_path / 'ANALYSIS_SUMMARY.md')
        )
        
        print(f"\n✅ Unified analysis complete! Results saved to: {unified_path}")
    
    return results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Pareto front analysis with configurable filtering')
    parser.add_argument('--data-file', '-d', type=str, default='results/pareto_front_results.csv',
                        help='Path to input CSV file (default: results/pareto_front_results.csv)')
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--percentile', '-p', type=int, default=100,
                        help='Percentile cutoff for filtering (default: 100, meaning keep all performers)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip filtering entirely (use all data)')
    parser.add_argument('--best-performers', '-b', action='store_true',
                        help='Calculate overall scores and identify best performers (includes Mann-Whitney on normalized score)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip Mann-Whitney U statistical tests on individual metrics')
    parser.add_argument('--by-architecture', action='store_true',
                        help='Analyze each architecture separately')
    parser.add_argument(
        '--colors',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Include colors runs in plots and statistical comparisons (default: false).',
    )
    args = parser.parse_args()
    
    # Determine filter percentage
    if args.no_filter:
        filter_percent = 100 # Type compatibility change: none to int
        print(f"Running analysis with NO FILTERING (using all data)")
    else:
        filter_percent = args.percentile
        print(f"Running analysis with {filter_percent}% percentile cutoff")
    
    # Show what will be calculated
    print(f"Statistical tests: {'DISABLED' if args.no_stats else 'ENABLED'}")
    print(f"Best performers analysis: {'ENABLED' if args.best_performers else 'DISABLED'}")
    print(f"Architecture separation: {'ENABLED' if args.by_architecture else 'DISABLED'}")
    print()
    
    if args.by_architecture:
        # Run analysis for each architecture separately
        results = run_analysis_by_architecture(
            data_file=args.data_file,
            output_dir=args.output_dir,
            save_results=True,
            perform_statistical_tests=not args.no_stats,
            datasets_to_exclude=None,
            filter_top_percent=filter_percent,
            calculate_best_performers=args.best_performers,
            colors=args.colors,
        )
    else:
        # Run standard analysis (all architectures together)
        df_result, pca_info, mann_whitney_results = run_analysis(
            data_file=args.data_file,
            output_dir=args.output_dir,
            save_results=True,
            perform_statistical_tests=not args.no_stats,
            datasets_to_exclude=None,
            filter_top_percent=filter_percent,
            calculate_best_performers=args.best_performers,
            colors=args.colors,
        )
        
        print(f"\nFinal dataset shape: {df_result.shape}")
        print(f"\nColumns available:")
        for col in df_result.columns:
            print(f"  - {col}")
