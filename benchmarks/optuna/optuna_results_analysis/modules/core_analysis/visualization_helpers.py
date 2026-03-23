"""
Analysis module - statistical analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import plotly.express as px
except ImportError:
    px = None

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_scenario(scenario_id):
    """
    Parse scenario_id to extract algorithm and sampling method
    
    Parameters:
    -----------
    scenario_id : str
        Scenario identifier string
        
    Returns:
    --------
    tuple : (dataset, algorithm, sampling, maptype)
    """
    # Handle multi-word datasets with underscores
    # Known multi-word datasets
    multi_word_datasets = ['swiss_roll', 's_curve', 'breast_cancer', 'olivetti_faces']
    
    dataset = None
    remaining_parts = None
    
    # Check if scenario starts with a multi-word dataset
    for mwd in multi_word_datasets:
        if scenario_id.startswith(mwd + '_'):
            dataset = mwd
            # Remove dataset name and the underscore after it
            remaining = scenario_id[len(mwd)+1:]
            remaining_parts = remaining.split('_')
            break
    
    # If not a multi-word dataset, parse normally
    if dataset is None:
        parts = scenario_id.split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            remaining_parts = parts[1:]
        else:
            return None, None, None, None
    
    if len(remaining_parts) >= 3:
        algorithm = remaining_parts[0]
        sampling = remaining_parts[1]
        maptype = remaining_parts[-1]
        return dataset, algorithm, sampling, maptype
    
    return None, None, None, None


def add_analysis_columns(df):
    """
    Add parsed columns for analysis
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with scenario_id column
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with added columns for algorithm, sampling method, etc.
    """
    # Parse scenario_id for additional info
    df[['dataset_parsed', 'algorithm_parsed', 'sampling_method_parsed', 'map_type']] = df['scenario_id'].apply(
        lambda x: pd.Series(parse_scenario(x))
    )
    
    # Use the existing processing_type column as algorithm (it has 'batch' or 'colors')
    df['algorithm'] = df['processing_type']
    
    # If sampling_method column exists, use it (otherwise use parsed version)
    if 'sampling_method' in df.columns:
        df['sampling_method_final'] = df['sampling_method']
    else:
        df['sampling_method_final'] = df['sampling_method_parsed']
    
    return df


def compute_statistics(df, group_by_cols, metrics=None):
    """
    Compute statistics for grouped data
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with data to analyze
    group_by_cols : str or list
        Column(s) to group by
        
    Returns:
    --------
    pandas DataFrame
        Statistics for each group
    """
    if metrics is None:
        preferred_metrics = [
            'quantization_error_holdout',
            'quantization_error_train',
            'quantization_error',
            'Topology_PC1'
        ]
        metrics = [m for m in preferred_metrics if m in df.columns]
    
    if not metrics:
        raise ValueError("No available metrics to compute statistics on.")
    
    agg_map = {metric: ['mean', 'median', 'std', 'count'] for metric in metrics}
    stats = df.groupby(group_by_cols).agg(agg_map).round(4)
    return stats


def plot_algorithm_comparison(df, save_path='Results/plots/', datasets_to_include=None, datasets_to_exclude=None, filename_suffix=''):
    """
    Create comparison plots for algorithms and sampling methods using available metrics.
    Preference order: holdout QE, train QE, legacy QE, Topology_PC1.
    """
    
    df_plot = df.copy()
    
    # Filter datasets if specified
    if datasets_to_include is not None:
        df_plot = df_plot[df_plot['dataset'].isin(datasets_to_include)]
        print(f"Including only datasets: {datasets_to_include}")
    elif datasets_to_exclude is not None:
        df_plot = df_plot[~df_plot['dataset'].isin(datasets_to_exclude)]
        print(f"Excluding datasets: {datasets_to_exclude}")
    
    metric_candidates = [
        ('quantization_error_holdout', 'Quantization Error (Holdout)'),
        ('quantization_error_train', 'Quantization Error (Train)'),
        ('quantization_error', 'Quantization Error'),
        ('Topology_PC1', 'Topology PC1')
    ]
    metrics_to_plot = [(col, label) for col, label in metric_candidates if col in df_plot.columns]
    
    if not metrics_to_plot:
        raise ValueError("No metrics available for plotting in plot_algorithm_comparison.")
    
    # Determine which sampling method column to use
    sampling_col = 'sampling_method' if 'sampling_method' in df_plot.columns else 'sampling_method_parsed'
    if 'sampling_method_final' in df_plot.columns:
        sampling_col = 'sampling_method_final'
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 6 * n_metrics))
    axes = np.atleast_2d(axes)
    
    palette = sns.color_palette("husl", len(df_plot['algorithm'].dropna().unique()))
    
    for idx, (metric, display_name) in enumerate(metrics_to_plot):
        current_df = df_plot.dropna(subset=[metric])
        if current_df.empty:
            continue
        
        # Left subplot: distribution by sampling & algorithm
        ax_left = axes[idx, 0]
        sns.boxplot(
            data=current_df,
            x=sampling_col,
            y=metric,
            hue='algorithm',
            palette=palette,
            ax=ax_left
        )
        sns.stripplot(
            data=current_df,
            x=sampling_col,
            y=metric,
            hue='algorithm',
            dodge=True,
            palette=palette,
            ax=ax_left,
            alpha=0.4,
            legend=False
        )
        ax_left.set_title(f"{display_name} by Sampling Method", fontsize=12, fontweight='bold')
        ax_left.set_ylabel(display_name, fontsize=11)
        ax_left.set_xlabel('Sampling Method', fontsize=11)
        ax_left.grid(True, axis='y', alpha=0.3)
        plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=20, ha='right')
        handles, labels = ax_left.get_legend_handles_labels()
        legend_map = {}
        for handle, label in zip(handles, labels):
            if label not in legend_map:
                legend_map[label] = handle
        if legend_map:
            ax_left.legend(legend_map.values(), legend_map.keys(), title='Algorithm', frameon=True)
        
        # Right subplot: dataset perspective
        ax_right = axes[idx, 1]
        sns.stripplot(
            data=current_df,
            x='dataset',
            y=metric,
            hue='algorithm',
            dodge=True,
            palette=palette,
            ax=ax_right,
            alpha=0.6
        )
        ax_right.set_title(f"{display_name} by Dataset", fontsize=12, fontweight='bold')
        ax_right.set_ylabel(display_name, fontsize=11)
        ax_right.set_xlabel('Dataset', fontsize=11)
        ax_right.grid(True, axis='y', alpha=0.3)
        plt.setp(ax_right.xaxis.get_majorticklabels(), rotation=30, ha='right')
        if ax_right.legend_:
            ax_right.legend(title='Algorithm', frameon=True, loc='best')
    
    plt.suptitle('Algorithm and Sampling Method Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_file = f'{save_path}algorithm_comparison{filename_suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    return fig


def save_statistics(df, output_file=None):
    """
    Compute and save statistical analysis to file
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with analysis columns
    output_file : str
        Path to save statistics file
    """
    
    # Determine which sampling method column to use
    sampling_col = 'sampling_method' if 'sampling_method' in df.columns else 'sampling_method_parsed'
    if 'sampling_method_final' in df.columns:
        sampling_col = 'sampling_method_final'
    
    metrics = [m for m in [
        'quantization_error_holdout',
        'quantization_error_train',
        'quantization_error',
        'Topology_PC1'
    ] if m in df.columns]
    
    # By algorithm (processing_type)
    stats_algo = compute_statistics(df, 'algorithm', metrics=metrics)
    
    # By sampling method
    stats_sampling = compute_statistics(df, sampling_col, metrics=metrics)
    
    # By combination
    stats_combo = compute_statistics(df, ['algorithm', sampling_col], metrics=metrics)
    
    # Only save to file if output_file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("ANALYSIS STATISTICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("By Algorithm:\n")
            f.write("="*40 + "\n")
            f.write(str(stats_algo) + "\n\n")
            
            f.write("By Sampling Method:\n")
            f.write("="*40 + "\n")
            f.write(str(stats_sampling) + "\n\n")
            
            f.write("By Algorithm-Sampling Combination:\n")
            f.write("="*40 + "\n")
            f.write(str(stats_combo))
        
        print(f"Saved statistics to {output_file}")
    
    # Also print to console
    print("\nStatistics by Algorithm:")
    print(stats_algo)
    print("\nStatistics by Sampling Method:")
    print(stats_sampling)
    print("\nStatistics by Combination:")
    print(stats_combo)


def compute_pareto_front(points):
    """
    Compute the Pareto front from a set of points.
    A point is on the Pareto front if no other point dominates it
    (i.e., no point has both lower x AND lower y values).
    
    Parameters:
    -----------
    points : array-like of shape (n_points, 2)
        Points with (x, y) coordinates
        
    Returns:
    --------
    pareto_points : array of shape (n_pareto, 2)
        Points on the Pareto front, sorted by x coordinate
    """
    points = np.array(points)
    if len(points) == 0:
        return np.array([])
    
    # Sort points by x coordinate (QE)
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    # Find Pareto front
    pareto_points = []
    current_min_y = float('inf')
    
    for point in sorted_points:
        if point[1] < current_min_y:
            pareto_points.append(point)
            current_min_y = point[1]
    
    return np.array(pareto_points)


def plot_pareto_fronts(df, save_path='Results/plots/', architecture=None,
                       datasets_to_include=None, datasets_to_exclude=None, 
                       filename_suffix='', use_raw=False):
    """
    Create Pareto front plots for batch vs colors within a single architecture.
    Shows one subplot per dataset with all sampling methods and algorithms.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with analysis columns including QE and PC1 metrics
    save_path : str
        Path to save the figure
    architecture : str or None
        'mst' or 'hexagonal' - if None, uses all data
    datasets_to_include : list or None
        List of datasets to include (if None, includes all)
    datasets_to_exclude : list or None
        List of datasets to exclude
    filename_suffix : str
        Suffix to add to filename
    use_raw : bool
        If True, use raw metrics (quantization_error, Topology_PC1)
        If False, use normalized metrics (default)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter data
    df_plot = df.copy()
    
    # Filter by architecture if specified
    if architecture:
        if 'architecture' in df_plot.columns:
            df_plot = df_plot[df_plot['architecture'] == architecture]
        elif 'map_type' in df_plot.columns:
            df_plot = df_plot[df_plot['map_type'] == architecture]
        print(f"Plotting Pareto fronts for {architecture} architecture")
    
    # Filter datasets
    if datasets_to_include is not None:
        df_plot = df_plot[df_plot['dataset'].isin(datasets_to_include)]
        print(f"Including only datasets: {datasets_to_include}")
    elif datasets_to_exclude is not None:
        df_plot = df_plot[~df_plot['dataset'].isin(datasets_to_exclude)]
        print(f"Excluding datasets: {datasets_to_exclude}")
    
    # Determine metric availability
    if use_raw:
        qe_holdout_col = 'quantization_error_holdout' if 'quantization_error_holdout' in df_plot.columns else None
        qe_train_col = 'quantization_error_train' if 'quantization_error_train' in df_plot.columns else None
        fallback_qe_col = 'quantization_error' if 'quantization_error' in df_plot.columns else None
        fallback_pc1_col = 'Topology_PC1' if 'Topology_PC1' in df_plot.columns else None
        title_suffix = ' (Raw Metrics)'
    else:
        qe_holdout_col = 'quantization_error_holdout_normalized' if 'quantization_error_holdout_normalized' in df_plot.columns else None
        qe_train_col = 'quantization_error_train_normalized' if 'quantization_error_train_normalized' in df_plot.columns else None
        fallback_qe_col = 'quantization_error_normalized' if 'quantization_error_normalized' in df_plot.columns else None
        fallback_pc1_col = 'Topology_PC1_normalized' if 'Topology_PC1_normalized' in df_plot.columns else None
        title_suffix = ' (Normalized)'
    
    pareto_pair = None
    axis_labels = None
    
    if qe_train_col and qe_holdout_col:
        pareto_pair = (qe_train_col, qe_holdout_col)
        axis_labels = ('Train QE', 'Holdout QE')
        title_suffix += ' | Train vs Holdout QE'
    elif fallback_qe_col and fallback_pc1_col:
        pareto_pair = (fallback_qe_col, fallback_pc1_col)
        axis_labels = ('Quantization Error', 'Topology PC1')
        title_suffix += ' | Legacy Topology View'
    else:
        print("Warning: No suitable metric pair found for Pareto plotting.")
        return None
    
    x_col, y_col = pareto_pair
    
    # Define visual encoding
    sampling_colors = {
        'full': '#1f77b4',      # blue
        'random': '#ff7f0e',    # orange  
        'hdsssom': '#2ca02c'    # green
    }
    
    # Colors for algorithm points (non-Pareto scatter)
    algo_colors = {
        'batch': '#2E86AB',     # blue for batch
        'colors': '#A23B72'     # red/pink for colors
    }
    
    algo_styles = {
        'batch': '-',     # solid line
        'colors': '--'    # dashed line
    }
    
    algo_markers = {
        'batch': 'x',     # x marker
        'colors': 'o'     # circle marker
    }
    
    # Determine which sampling method column to use
    sampling_col = 'sampling_method' if 'sampling_method' in df_plot.columns else 'sampling_method_parsed'
    if 'sampling_method_final' in df_plot.columns:
        sampling_col = 'sampling_method_final'
    
    # Get unique datasets
    datasets = sorted(df_plot['dataset'].unique())
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        print("No datasets found in the data")
        return None
    
    # Calculate grid dimensions
    n_cols = 3  # 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Ensure axes is always a list
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get unique sampling methods
    sampling_methods = sorted(df_plot[sampling_col].dropna().unique())
    
    # Plot each dataset
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = df_plot[df_plot['dataset'] == dataset]
        
        # Track if we've added any plots to this subplot
        has_data = False
        
        # For each sampling method and algorithm combination
        for sampling in sampling_methods:
            if sampling in sampling_colors:
                for algo in ['batch', 'colors']:
                    # Get data for this combination
                    mask = (dataset_data[sampling_col] == sampling) & (dataset_data['algorithm'] == algo)
                    data = dataset_data[mask]
                    
                    if len(data) > 0 and x_col in data.columns and y_col in data.columns:
                        filtered_data = data.dropna(subset=[x_col, y_col])
                        if filtered_data.empty:
                            continue
                        # Extract points using selected metric columns
                        points = filtered_data[[x_col, y_col]].values
                        
                        # Compute Pareto front for this specific dataset/sampling/algo combination
                        pareto_points = compute_pareto_front(points)
                        
                        if len(pareto_points) > 0:
                            has_data = True
                            # Create label
                            label = f'{sampling[:3]}-{algo[:3]}'  # Shortened labels
                            
                            # Plot Pareto front
                            ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                                   color=sampling_colors[sampling],
                                   linestyle=algo_styles[algo],
                                   marker=algo_markers[algo],
                                   markersize=4,
                                   label=label,
                                   linewidth=1.5,
                                   alpha=0.8)
                            
                            # Also plot all points as scatter (semi-transparent, colored by algorithm)
                            ax.scatter(points[:, 0], points[:, 1],
                                      color=algo_colors[algo],  # Color by algorithm for better distinction
                                      marker=algo_markers[algo],
                                      alpha=0.2,
                                      s=10)
        
        # Format subplot
        ax.set_title(f'{dataset}', fontsize=10, fontweight='bold')
        ax.set_xlabel(axis_labels[0], fontsize=8)
        ax.set_ylabel(axis_labels[1], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)
        
        # Only add legend if there's data
        if has_data:
            ax.legend(fontsize=6, ncol=2, loc='best', frameon=True, fancybox=True)
        
        # Plotly export disabled in QE-only mode.
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    arch_str = f" - {architecture.upper()}" if architecture else ""
    plt.suptitle(f'Pareto Fronts: Batch vs Colors{arch_str}{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    metric_type = "_raw" if use_raw else "_normalized"
    arch_suffix = f"_{architecture}" if architecture else ""
    output_file = f'{save_path}pareto_fronts{arch_suffix}{metric_type}{filename_suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure without displaying
    print(f"Saved Pareto front plot to {output_file}")
    
    return fig


def export_plotly_pareto_3d(df_subset, save_path, architecture, dataset, use_raw, metric_cols, sampling_col):
    """
    Export a 3D Pareto visualization (train QE × holdout QE × third metric) using Plotly.
    """
    if px is None:
        return
    
    required_cols = [metric_cols['train'], metric_cols['holdout'], metric_cols['third']]
    if any(col not in df_subset.columns for col in required_cols):
        return
    
    plot_data = df_subset.dropna(subset=required_cols)
    if plot_data.empty:
        return
    
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    arch_suffix = f"_{architecture}" if architecture else ""
    metric_suffix = "_raw" if use_raw else "_normalized"
    filename = f"pareto_front_3d_{dataset}{arch_suffix}{metric_suffix}.html"
    output_file = output_dir / filename
    
    hover_columns = ['scenario_id', 'trial_number'] if 'trial_number' in plot_data.columns else ['scenario_id']
    
    title_parts = [f"3D Pareto Front — {dataset}"]
    if architecture:
        title_parts.append(architecture.upper())
    title_parts.append("Raw" if use_raw else "Normalized")
    
    fig = px.scatter_3d(
        plot_data,
        x=metric_cols['train'],
        y=metric_cols['holdout'],
        z=metric_cols['third'],
        color='algorithm' if 'algorithm' in plot_data.columns else None,
        symbol=sampling_col if sampling_col in plot_data.columns else None,
        hover_data=[col for col in hover_columns if col in plot_data.columns],
        title=" | ".join(title_parts)
    )
    
    fig.update_layout(
        legend=dict(title="Legend"),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    fig.write_html(str(output_file), include_plotlyjs='cdn')
    print(f"Saved 3D Pareto plot to {output_file}")
