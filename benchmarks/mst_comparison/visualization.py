"""
Visualization functions for MST comparison benchmarks.

This module handles visualization of MST structures and metric comparisons.
"""

import os
import logging
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Import FloatSOM for type hints
from floatsom.base.floatsom import FloatSOM

# Dataset categories (kept locally to avoid cross-module dependencies)
GENERATIVE_DATASETS = ['blobs', 'moons', 'circles', 'swiss_roll', 's_curve']
FIXED_DATASETS = ['iris', 'diabetes', 'breast_cancer', 'wine', 'olivetti_faces', 'digits']


def visualize_msts(direct_mst_som: FloatSOM, hexagonal_som: FloatSOM, 
                   hex_mst_edges: List[Tuple[int, int]], 
                   data: np.ndarray, output_dir: str, dataset_name: str = None):
    """Create visualizations comparing the two MST approaches."""
    
    # Get weights
    direct_weights = direct_mst_som.get_weights()
    hex_weights = hexagonal_som.get_weights()
    
    # Convert to numpy
    if isinstance(direct_weights, cp.ndarray):
        direct_weights = direct_weights.get()
    if isinstance(hex_weights, cp.ndarray):
        hex_weights = hex_weights.get()
    
    # Convert data to numpy
    if isinstance(data, cp.ndarray):
        data_np = data.get()
    else:
        data_np = data
    
    # Use PCA for visualization if dimensions > 3
    if data_np.shape[1] > 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_np)
        direct_weights_2d = pca.transform(direct_weights)
        hex_weights_2d = pca.transform(hex_weights)
    elif data_np.shape[1] == 3:
        # Use first 2 dimensions for 3D data
        data_2d = data_np[:, :2]
        direct_weights_2d = direct_weights[:, :2]
        hex_weights_2d = hex_weights[:, :2]
    else:
        data_2d = data_np
        direct_weights_2d = direct_weights
        hex_weights_2d = hex_weights
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Direct MST-SOM
    ax1.scatter(data_2d[:, 0], data_2d[:, 1], s=10, alpha=0.3, c='gray', label='Data points')
    
    # Draw MST edges
    direct_mst_edges = direct_mst_som.topology.mst_edges
    for u, v in direct_mst_edges:
        ax1.plot([direct_weights_2d[u, 0], direct_weights_2d[v, 0]],
                [direct_weights_2d[u, 1], direct_weights_2d[v, 1]],
                'b-', alpha=0.6, linewidth=1.5)
    
    # Plot nodes
    ax1.scatter(direct_weights_2d[:, 0], direct_weights_2d[:, 1], 
               s=80, c='blue', marker='o', edgecolors='darkblue', 
               linewidths=1.5, label='MST nodes', zorder=5)
    
    ax1.set_title('Direct MST-SOM\n(Trained with MST topology)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Component 1' if data_np.shape[1] > 3 else 'X')
    ax1.set_ylabel('Component 2' if data_np.shape[1] > 3 else 'Y')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hexagonal-to-MST
    ax2.scatter(data_2d[:, 0], data_2d[:, 1], s=10, alpha=0.3, c='gray', label='Data points')
    
    # Draw original hexagonal connections in light gray
    grid_size = hexagonal_som.topology.grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < hex_weights_2d.shape[0]:
                # Connect to neighbors (hexagonal pattern)
                neighbors = []
                if j < grid_size - 1:  # Right
                    neighbors.append(i * grid_size + (j + 1))
                if i < grid_size - 1:  # Bottom
                    neighbors.append((i + 1) * grid_size + j)
                if i % 2 == 0:  # Even row
                    if i > 0 and j > 0:  # Top-left
                        neighbors.append((i - 1) * grid_size + (j - 1))
                    if i < grid_size - 1 and j > 0:  # Bottom-left
                        neighbors.append((i + 1) * grid_size + (j - 1))
                else:  # Odd row
                    if i > 0 and j < grid_size - 1:  # Top-right
                        neighbors.append((i - 1) * grid_size + (j + 1))
                    if i < grid_size - 1 and j < grid_size - 1:  # Bottom-right
                        neighbors.append((i + 1) * grid_size + (j + 1))
                
                for neighbor in neighbors:
                    if neighbor < hex_weights_2d.shape[0]:
                        ax2.plot([hex_weights_2d[idx, 0], hex_weights_2d[neighbor, 0]],
                                [hex_weights_2d[idx, 1], hex_weights_2d[neighbor, 1]],
                                'lightgray', alpha=0.3, linewidth=0.5)
    
    # Draw MST edges on top
    for u, v in hex_mst_edges:
        ax2.plot([hex_weights_2d[u, 0], hex_weights_2d[v, 0]],
                [hex_weights_2d[u, 1], hex_weights_2d[v, 1]],
                'r-', alpha=0.8, linewidth=2)
    
    # Plot nodes
    ax2.scatter(hex_weights_2d[:, 0], hex_weights_2d[:, 1], 
               s=80, c='red', marker='o', edgecolors='darkred', 
               linewidths=1.5, label='Hexagonal nodes', zorder=5)
    
    # Create custom legend
    gray_line = mpatches.Patch(color='lightgray', label='Original hexagonal edges')
    red_line = mpatches.Patch(color='red', label='MST edges (computed post-training)')
    ax2.legend(handles=[gray_line, red_line], loc='best')
    
    ax2.set_title('Hexagonal-to-MST\n(Trained as hexagonal, converted to MST)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Component 1' if data_np.shape[1] > 3 else 'X')
    ax2.set_ylabel('Component 2' if data_np.shape[1] > 3 else 'Y')
    ax2.grid(True, alpha=0.3)
    
    # Add dataset name to title if provided
    if dataset_name:
        plt.suptitle(f'MST Topology Comparison - {dataset_name}', fontsize=16, fontweight='bold', y=1.02)
    else:
        plt.suptitle('MST Topology Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Generate filename based on dataset name if provided
    if dataset_name:
        output_filename = f'mst_comparison_{dataset_name}.png'
    else:
        output_filename = 'mst_comparison_visualization.png'
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")


def create_metric_dot_plots(generative_results, fixed_results, output_dir: str) -> None:
    """
    Create separate visualizations for generative and fixed datasets.

    - Violin plots for generative datasets (with distributions)
    - Dot plots for fixed datasets (single values)

    Each subplot shows one metric with:
    - X-axis: Method (Direct MST-SOM vs Hexagonal-to-MST)
    - Y-axis: Metric value
    """
    import numpy as np
    from matplotlib.lines import Line2D

    # Metrics to plot
    metrics = ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']
    metric_labels = {
        'gap_ratio': 'Gap Ratio\n(Higher = better separation)',
        'gini': 'Gini Coefficient\n(Higher = heterogeneous)',
        'edge_ratio': 'Edge Ratio\n(Higher = better clusters)',
        'bimodality': 'Bimodality\n(>0.555 = bimodal)',
        'separation': 'Separation\n(Higher = better)',
        'stability': 'Stability\n(Higher = more stable)'
    }

    # Color palettes
    gen_colors = plt.cm.Set1(np.linspace(0, 0.5, len(GENERATIVE_DATASETS)))
    fixed_colors = plt.cm.Set2(np.linspace(0, 0.7, len(FIXED_DATASETS)))

    # Create dataset name to color mapping
    color_map = {}
    for i, dataset in enumerate(GENERATIVE_DATASETS):
        color_map[dataset] = gen_colors[i]
    for i, dataset in enumerate(FIXED_DATASETS):
        color_map[dataset] = fixed_colors[i]

    # ========== PART 1: VIOLIN PLOTS FOR GENERATIVE DATASETS ==========
    if generative_results:
        fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
        axes1 = axes1.flatten()

        create_violin_plots_for_generative(axes1, generative_results, metrics, metric_labels, color_map)

        # Add overall title
        fig1.suptitle('MST Method Comparison: Generative Datasets (Violin Plots)',
                      fontsize=16, fontweight='bold', y=0.98)

        # Create legend for generative datasets
        legend_elements = []
        for dataset in GENERATIVE_DATASETS:
            if dataset in color_map:
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                              label=dataset,
                                              markerfacecolor=color_map[dataset],
                                              markersize=8, markeredgecolor='black', markeredgewidth=0.5))

        fig1.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02),
                    ncol=5, fontsize=10, frameon=True, fancybox=True, shadow=True)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        output_path1 = os.path.join(output_dir, 'mst_metrics_comparison_violinplot_generative.png')
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Generative datasets violin plot saved to {output_path1}")

    # ========== PART 2: DOT PLOTS FOR FIXED DATASETS ==========
    if fixed_results:
        fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
        axes2 = axes2.flatten()

        create_dot_plots_for_fixed(axes2, fixed_results, metrics, metric_labels, color_map)

        # Add overall title
        fig2.suptitle('MST Method Comparison: Fixed Datasets (Dot Plots)',
                      fontsize=16, fontweight='bold', y=0.98)

        # Create legend for fixed datasets
        legend_elements = []
        for dataset in FIXED_DATASETS:
            if dataset in color_map:
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                              label=dataset,
                                              markerfacecolor=color_map[dataset],
                                              markersize=8, markeredgecolor='black', markeredgewidth=0.5))

        fig2.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02),
                    ncol=6, fontsize=10, frameon=True, fancybox=True, shadow=True)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        output_path2 = os.path.join(output_dir, 'mst_metrics_comparison_dotplot_fixed.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Fixed datasets dot plot saved to {output_path2}")


def create_violin_plots_for_generative(axes, generative_results, metrics, metric_labels, color_map):
    """Create violin plots for generative datasets."""
    import numpy as np
    from scipy import stats

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Collect data by dataset and method
        direct_data_by_dataset = {}
        hex_data_by_dataset = {}

        # Process generative results (using all replicates)
        for result in generative_results:
            dataset_name = result['dataset']

            if 'direct_metrics_all' in result and 'hex_metrics_all' in result:
                if dataset_name not in direct_data_by_dataset:
                    direct_data_by_dataset[dataset_name] = []
                    hex_data_by_dataset[dataset_name] = []

                direct_data_by_dataset[dataset_name].extend(result['direct_metrics_all'][metric])
                hex_data_by_dataset[dataset_name].extend(result['hex_metrics_all'][metric])

        # Calculate positions for violin plots
        n_datasets = len(direct_data_by_dataset)
        if n_datasets > 0:
            # Width of each violin
            violin_width = 0.08
            # Total width needed for all violins
            total_width = violin_width * n_datasets * 1.2  # Add some spacing
            # Starting positions (centered around 1.0 for direct, 2.0 for hex)
            direct_positions = np.linspace(1.0 - total_width/2, 1.0 + total_width/2, n_datasets)
            hex_positions = np.linspace(2.0 - total_width/2, 2.0 + total_width/2, n_datasets)

            # Create violin plots for each dataset
            for i, dataset_name in enumerate(sorted(direct_data_by_dataset.keys())):
                color = color_map.get(dataset_name, 'gray')

                # Direct MST-SOM violins
                if dataset_name in direct_data_by_dataset and len(direct_data_by_dataset[dataset_name]) > 0:
                    parts = ax.violinplot([direct_data_by_dataset[dataset_name]],
                                          positions=[direct_positions[i]],
                                          widths=violin_width,
                                          showmeans=False,
                                          showmedians=False,
                                          showextrema=False)

                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.5)

                # Hexagonal-to-MST violins
                if dataset_name in hex_data_by_dataset and len(hex_data_by_dataset[dataset_name]) > 0:
                    parts = ax.violinplot([hex_data_by_dataset[dataset_name]],
                                          positions=[hex_positions[i]],
                                          widths=violin_width,
                                          showmeans=False,
                                          showmedians=False,
                                          showextrema=False)

                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.5)

            # Add overall means and confidence intervals
            all_direct = []
            all_hex = []
            for dataset_name in direct_data_by_dataset:
                all_direct.extend(direct_data_by_dataset[dataset_name])
                all_hex.extend(hex_data_by_dataset[dataset_name])

            if all_direct:
                direct_mean = np.mean(all_direct)
                direct_std = np.std(all_direct)
                ax.errorbar(1.0, direct_mean, yerr=direct_std, fmt='D',
                            color='darkblue', markersize=8, linewidth=2, capsize=10, capthick=2,
                            label='Overall Mean ± SD', zorder=10)
                ax.axhline(y=direct_mean, xmin=0.2, xmax=0.48, color='darkblue',
                           linestyle='--', alpha=0.3, linewidth=1)

            if all_hex:
                hex_mean = np.mean(all_hex)
                hex_std = np.std(all_hex)
                ax.errorbar(2.0, hex_mean, yerr=hex_std, fmt='D',
                            color='darkred', markersize=8, linewidth=2, capsize=10, capthick=2,
                            zorder=10)
                ax.axhline(y=hex_mean, xmin=0.52, xmax=0.8, color='darkred',
                           linestyle='--', alpha=0.3, linewidth=1)

            # Add significance marker if we have enough data
            if len(all_direct) >= 2 and len(all_hex) >= 2:
                from scipy import stats
                _, p_value = stats.ttest_ind(all_direct, all_hex)
                if p_value < 0.05:
                    y_max = max(max(all_direct, default=0), max(all_hex, default=0))
                    ax.plot([1.0, 2.0], [y_max * 1.08, y_max * 1.08], 'k-', linewidth=1)
                    ax.text(1.5, y_max * 1.10, f'p={p_value:.3f}*', ha='center', fontsize=9)

        # Customize axes
        ax.set_xlim(0.3, 2.7)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(['Direct\nMST-SOM', 'Hexagonal\nto MST'], fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')


def create_dot_plots_for_fixed(axes, fixed_results, metrics, metric_labels, color_map):
    """Create dot plots for fixed datasets (single value per dataset)."""
    import numpy as np
    from scipy import stats

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        direct_points = []
        hex_points = []
        colors = []

        # Collect one mean value per dataset
        for result in fixed_results:
            dataset_name = result['dataset']
            if 'direct_metrics' in result and 'hex_metrics' in result:
                direct_points.append(result['direct_metrics'][metric])
                hex_points.append(result['hex_metrics'][metric])
                colors.append(color_map.get(dataset_name, 'gray'))

        # X positions for the two methods
        x_direct = np.full(len(direct_points), 1.0)
        x_hex = np.full(len(hex_points), 2.0)

        # Jitter to avoid overlap
        jitter = 0.04
        x_direct += (np.random.rand(len(x_direct)) - 0.5) * jitter
        x_hex += (np.random.rand(len(x_hex)) - 0.5) * jitter

        # Scatter points
        ax.scatter(x_direct, direct_points, c=colors, edgecolors='black', linewidths=0.5, s=60, alpha=0.8, label='Direct')
        ax.scatter(x_hex, hex_points, c=colors, edgecolors='black', linewidths=0.5, s=60, alpha=0.8, label='Hex')

        # Add means and standard deviations
        if direct_points:
            direct_mean = np.mean(direct_points)
            direct_std = np.std(direct_points)
            ax.errorbar(1.0, direct_mean, yerr=direct_std, fmt='D',
                        color='darkblue', markersize=8, linewidth=2, capsize=10, capthick=2,
                        label='Overall Mean ± SD', zorder=10)
            ax.axhline(y=direct_mean, xmin=0.2, xmax=0.48, color='darkblue', linestyle='--', alpha=0.3, linewidth=1)

        if hex_points:
            hex_mean = np.mean(hex_points)
            hex_std = np.std(hex_points)
            ax.errorbar(2.0, hex_mean, yerr=hex_std, fmt='D',
                        color='darkred', markersize=8, linewidth=2, capsize=10, capthick=2,
                        zorder=10)
            ax.axhline(y=hex_mean, xmin=0.52, xmax=0.8, color='darkred', linestyle='--', alpha=0.3, linewidth=1)

        # Add significance marker
        if len(direct_points) >= 2 and len(hex_points) >= 2:
            _, p_value = stats.ttest_ind(direct_points, hex_points)
            if p_value < 0.05:
                y_max = max(max(direct_points), max(hex_points))
                ax.plot([1.0, 2.0], [y_max * 1.08, y_max * 1.08], 'k-', linewidth=1)
                ax.text(1.5, y_max * 1.10, f'p={p_value:.3f}*', ha='center', fontsize=9)

        # Customize axes
        ax.set_xlim(0.3, 2.7)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(['Direct\nMST-SOM', 'Hexagonal\nto MST'], fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')


__all__ = [
    'visualize_msts',
    'create_metric_dot_plots',
    'create_violin_plots_for_generative',
    'create_dot_plots_for_fixed',
    'create_clustering_offset_line_plots',
    'create_qe_violin_plot',
    'create_qe_bar_plot',
]


def create_clustering_metrics_violin_plot(mst_values: dict, grid_values: dict, output_dir: str, *, grid_method: str = 'agglomerative', mst_method: str = 'agglomerative') -> None:
    """Create violin plots for clustering metrics across repeats on blobs.

    Shows three subplots: Accuracy, ARI, NMI. For each, draws two violins: Direct MST-SOM vs Grid SOM.
    """
    import numpy as np
    from scipy import stats

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('accuracy', 'Accuracy (Majority Vote)'),
        ('adjusted_rand_score', 'Adjusted Rand Index (ARI)'),
        ('normalized_mutual_info', 'Normalized Mutual Information (NMI)')
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        direct_vals = np.asarray(mst_values.get(key, []))
        grid_vals = np.asarray(grid_values.get(key, []))

        # Draw violins
        parts1 = ax.violinplot([direct_vals], positions=[1.0], widths=0.3,
                               showmeans=False, showmedians=False, showextrema=False)
        parts2 = ax.violinplot([grid_vals], positions=[2.0], widths=0.3,
                               showmeans=False, showmedians=False, showextrema=False)

        for pc in parts1['bodies']:
            pc.set_facecolor('tab:blue')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        for pc in parts2['bodies']:
            pc.set_facecolor('tab:red')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Means ± SD
        if direct_vals.size > 0:
            ax.errorbar(1.0, np.mean(direct_vals), yerr=np.std(direct_vals, ddof=1) if direct_vals.size > 1 else 0.0,
                        fmt='D', color='darkblue', markersize=8, linewidth=2, capsize=10, capthick=2, zorder=10)
        if grid_vals.size > 0:
            ax.errorbar(2.0, np.mean(grid_vals), yerr=np.std(grid_vals, ddof=1) if grid_vals.size > 1 else 0.0,
                        fmt='D', color='darkred', markersize=8, linewidth=2, capsize=10, capthick=2, zorder=10)

        # Significance bracket
        if direct_vals.size >= 2 and grid_vals.size >= 2:
            _, p_value = stats.ttest_rel(direct_vals, grid_vals)
            if p_value < 0.05:
                y_max = max(np.max(direct_vals), np.max(grid_vals))
                ax.plot([1.0, 2.0], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
                ax.text(1.5, y_max * 1.07, f'p={p_value:.3f}*', ha='center', fontsize=9)

        # Axes formatting
        ax.set_xlim(0.3, 2.7)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(['Direct\nMST-SOM', 'Grid\nSOM'], fontsize=11)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"grid-{grid_method}_mst-{mst_method}"
    out_path = os.path.join(output_dir, f'clustering_metrics_violinplot_blobs__{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_qe_violin_plot(mst_values: dict, grid_values: dict, output_dir: str, *, grid_method: str = 'agglomerative', mst_method: str = 'agglomerative') -> None:
    """Create a violin plot for Quantization Error across repeats on blobs.

    - Two violins: Direct MST-SOM vs Grid SOM
    - Lower is better
    """
    import numpy as np
    from scipy import stats

    direct_vals = np.asarray(mst_values.get('quantization_error', []))
    grid_vals = np.asarray(grid_values.get('quantization_error', []))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if direct_vals.size > 0:
        parts1 = ax.violinplot([direct_vals], positions=[1.0], widths=0.3,
                               showmeans=False, showmedians=False, showextrema=False)
        for pc in parts1['bodies']:
            pc.set_facecolor('tab:blue')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        ax.errorbar(1.0, np.mean(direct_vals), yerr=np.std(direct_vals, ddof=1) if direct_vals.size > 1 else 0.0,
                    fmt='D', color='darkblue', markersize=8, linewidth=2, capsize=10, capthick=2, zorder=10)

    if grid_vals.size > 0:
        parts2 = ax.violinplot([grid_vals], positions=[2.0], widths=0.3,
                               showmeans=False, showmedians=False, showextrema=False)
        for pc in parts2['bodies']:
            pc.set_facecolor('tab:red')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        ax.errorbar(2.0, np.mean(grid_vals), yerr=np.std(grid_vals, ddof=1) if grid_vals.size > 1 else 0.0,
                    fmt='D', color='darkred', markersize=8, linewidth=2, capsize=10, capthick=2, zorder=10)

    # Significance bracket (paired t-test when lengths match >=2)
    if direct_vals.size >= 2 and grid_vals.size >= 2 and direct_vals.size == grid_vals.size:
        _, p_value = stats.ttest_rel(direct_vals, grid_vals)
        if np.isfinite(p_value):
            y_max = max(np.max(direct_vals), np.max(grid_vals))
            ax.plot([1.0, 2.0], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
            star = '*' if p_value < 0.05 else ''
            ax.text(1.5, y_max * 1.07, f'p={p_value:.3f}{star}', ha='center', fontsize=9)

    ax.set_xlim(0.3, 2.7)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(['Direct\nMST-SOM', 'Grid\nSOM'], fontsize=11)
    ax.set_ylabel('Quantization Error (lower is better)', fontsize=10)
    ax.set_title('Quantization Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"grid-{grid_method}_mst-{mst_method}"
    out_path = os.path.join(output_dir, f'clustering_qe_violinplot_blobs__{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_qe_bar_plot(qe_direct: float, qe_grid: float, output_dir: str) -> None:
    """Create a simple bar chart comparing QE for a single run (clustering mode)."""
    try:
        vals = [float(qe_direct), float(qe_grid)]
    except Exception:
        # Missing values; skip plotting
        logger.warning("QE values missing; skipping QE bar plot")
        return

    labels = ['Direct MST-SOM', 'Grid SOM']
    colors = ['tab:blue', 'tab:red']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.6f}", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Quantization Error (lower is better)')
    ax.set_title('Quantization Error Comparison')
    ax.grid(True, alpha=0.2, axis='y')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'clustering_qe_barplot.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_clustering_results(grid_som: FloatSOM, mst_som: FloatSOM,
                                 data: np.ndarray,
                                 true_labels: np.ndarray,
                                 grid_predictions: np.ndarray,
                                 mst_predictions: np.ndarray,
                                 grid_metaclusters: np.ndarray,
                                 mst_metaclusters: np.ndarray,
                                 metadata: dict,
                                 output_dir: str,
                                 grid_method_name: str = 'agglomerative',
                                 mst_method_name: str = 'agglomerative'):
    """
    Visualize clustering results with color-coded correct/incorrect assignments.

    Handles cases where agglomerative clustering returns more clusters than true labels
    by using many-to-one mapping (predicted cluster -> best matching true cluster) for coloring.
    """
    import matplotlib.cm as cm
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    os.makedirs(output_dir, exist_ok=True)

    # Convert data to numpy if needed
    data_np = data.get() if hasattr(data, 'get') else data

    # Get SOM weights
    grid_weights = grid_som.get_weights()
    grid_weights_np = grid_weights.get() if hasattr(grid_weights, 'get') else grid_weights

    mst_weights = mst_som.get_weights()
    mst_weights_np = mst_weights.get() if hasattr(mst_weights, 'get') else mst_weights

    # Many-to-one mapping for predicted -> true labels for visualization
    def many_to_one_map(true_labels_arr, predicted_labels_arr):
        unique_true = np.unique(true_labels_arr)
        unique_pred = np.unique(predicted_labels_arr)
        cmatrix = confusion_matrix(true_labels_arr, predicted_labels_arr)
        mapping = {}
        for pred_idx, pred_label in enumerate(unique_pred):
            if pred_idx < cmatrix.shape[1]:
                best_true_idx = np.argmax(cmatrix[:, pred_idx])
                if best_true_idx < len(unique_true):
                    mapping[pred_label] = unique_true[best_true_idx]
        mapped = np.array([mapping.get(p, -1) for p in predicted_labels_arr])
        correct = (true_labels_arr == mapped)
        return correct, mapped

    grid_correct, grid_mapped = many_to_one_map(true_labels, grid_predictions)
    mst_correct, mst_mapped = many_to_one_map(true_labels, mst_predictions)

    # Accuracies (percentage)
    grid_accuracy = float(np.mean(grid_correct) * 100)
    mst_accuracy = float(np.mean(mst_correct) * 100)

    # Determine number of clusters for coloring (use node-level metaclusters)
    n_true_clusters = metadata.get('n_clusters', len(np.unique(true_labels)))
    unique_grid_clusters = np.unique(grid_metaclusters)
    unique_mst_clusters = np.unique(mst_metaclusters)
    n_agglom_clusters = int(max(len(unique_grid_clusters), len(unique_mst_clusters)))

    # Colors
    cmap = cm.get_cmap('tab20' if n_agglom_clusters > 10 else 'tab10')
    cluster_colors = [cmap(i / max(n_agglom_clusters - 1, 1)) for i in range(n_agglom_clusters)]

    # Dimensionality reduction for plotting
    if data_np.shape[1] == 2:
        x_data, y_data = data_np[:, 0], data_np[:, 1]
        grid_x, grid_y = grid_weights_np[:, 0], grid_weights_np[:, 1]
        mst_x, mst_y = mst_weights_np[:, 0], mst_weights_np[:, 1]
        xlabel, ylabel = 'Dimension 1', 'Dimension 2'
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_np)
        grid_weights_pca = pca.transform(grid_weights_np)
        mst_weights_pca = pca.transform(mst_weights_np)
        x_data, y_data = data_pca[:, 0], data_pca[:, 1]
        grid_x, grid_y = grid_weights_pca[:, 0], grid_weights_pca[:, 1]
        mst_x, mst_y = mst_weights_pca[:, 0], mst_weights_pca[:, 1]
        xlabel, ylabel = 'First Principal Component', 'Second Principal Component'

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left: Grid SOM
    ax1.set_title(f'Grid/Hexagonal SOM - Accuracy: {grid_accuracy:.2f}%', fontsize=13, fontweight='bold')
    # Plot data points colored by correctness
    for correct_val, color, label in [(True, 'green', 'Correctly clustered'),
                                      (False, 'red', 'Incorrectly clustered')]:
        mask = (grid_correct == correct_val)
        if np.any(mask):
            ax1.scatter(x_data[mask], y_data[mask], s=30, c=color, alpha=0.6,
                        edgecolors='black', linewidths=0.5, label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)

    # Draw grid connections if grid-shaped
    grid_size = getattr(grid_som.topology, 'grid_size', None)
    if grid_size is not None and grid_weights_np.shape[0] == grid_size * grid_size:
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if j < grid_size - 1:
                    idx_right = i * grid_size + (j + 1)
                    ax1.plot([grid_x[idx], grid_x[idx_right]], [grid_y[idx], grid_y[idx_right]], 'gray', alpha=0.3, linewidth=1)
                if i < grid_size - 1:
                    idx_bottom = (i + 1) * grid_size + j
                    ax1.plot([grid_x[idx], grid_x[idx_bottom]], [grid_y[idx], grid_y[idx_bottom]], 'gray', alpha=0.3, linewidth=1)

    for cluster_id in unique_grid_clusters:
        mask = (grid_metaclusters == cluster_id)
        if np.any(mask):
            color_idx = int(cluster_id) % len(cluster_colors)
            ax1.scatter(grid_x[mask], grid_y[mask], s=100,
                        c=[cluster_colors[color_idx]], marker='s',
                        edgecolors='black', linewidths=2,
                        label=f'Node Cluster {cluster_id}', zorder=5)

    # Legend removed for cleaner single-run visualization

    # Right: MST-SOM
    ax2.set_title(f'Direct MST-SOM - Accuracy: {mst_accuracy:.2f}%', fontsize=13, fontweight='bold')
    # Plot data points colored by correctness
    for correct_val, color, label in [(True, 'green', 'Correctly clustered'),
                                      (False, 'red', 'Incorrectly clustered')]:
        mask = (mst_correct == correct_val)
        if np.any(mask):
            ax2.scatter(x_data[mask], y_data[mask], s=30, c=color, alpha=0.6,
                        edgecolors='black', linewidths=0.5, label=label)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.grid(True, alpha=0.3)

    # Draw MST edges
    if hasattr(mst_som.topology, 'mst_edges') and mst_som.topology.mst_edges:
        for u, v in mst_som.topology.mst_edges:
            ax2.plot([mst_x[u], mst_x[v]], [mst_y[u], mst_y[v]], 'gray', alpha=0.4, linewidth=1.5)

    for cluster_id in unique_mst_clusters:
        mask = (mst_metaclusters == cluster_id)
        if np.any(mask):
            color_idx = int(cluster_id) % len(cluster_colors)
            ax2.scatter(mst_x[mask], mst_y[mask], s=100,
                        c=[cluster_colors[color_idx]], marker='D',
                        edgecolors='black', linewidths=2,
                        label=f'Node Cluster {cluster_id}', zorder=5)

    # Legend removed for cleaner single-run visualization

    # Super title summarizing counts
    fig.suptitle(
        f'Clustering Evaluation - {metadata.get("dataset_name", "blobs").title()}\n'
        f'{metadata.get("shape", [0])[0]} samples | '
        f'{n_true_clusters} true clusters | '
        f'{len(unique_grid_clusters)} Grid {grid_method_name} clusters, '
        f'{len(unique_mst_clusters)} MST {mst_method_name} clusters',
        fontsize=15, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = f"grid-{grid_method_name}_mst-{mst_method_name}"
    out_path = os.path.join(output_dir, f'clustering_visualization__{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Clustering visualization saved to {out_path}")


def create_clustering_offset_line_plots(per_run_records: list, output_dir: str, *, grid_method: str = 'agglomerative', mst_method: str = 'agglomerative') -> None:
    """Create line plots for clustering stats across agglomerative K with offset-colored trends.

    - One figure per metric: Accuracy, ARI, NMI
    - X-axis: agglomerative cluster count (K)
    - Y-axis: metric value
    - Dots: per-repeat values, colored by ΔK (offset), semi-transparent
    - Lines: mean per K for each (method, offset) pair
        * Color by ΔK offset (consistent across plot)
        * Style by method: solid = Direct MST-SOM, dotted = Grid SOM
    - Legends: two legends only — method styles and offset colors
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Group per-run values by (method, offset, K) for means, and collect raw points for scatter
    buckets = defaultdict(list)  # key: (metric, method, offset, K) -> list of values
    points = defaultdict(list)   # key: (metric, method, offset) -> list of (K, value)

    for rec in per_run_records:
        centers = rec['centers']
        K = rec['agglomerative_clusters']
        offset = int(K) - int(centers)
        for method in ['direct_mst', 'grid']:
            acc = rec[method]['accuracy']
            ari = rec[method]['adjusted_rand_score']
            nmi = rec[method]['normalized_mutual_info']
            buckets[('accuracy', method, offset, K)].append(acc)
            buckets[('adjusted_rand_score', method, offset, K)].append(ari)
            buckets[('normalized_mutual_info', method, offset, K)].append(nmi)
            points[('accuracy', method, offset)].append((K, acc))
            points[('adjusted_rand_score', method, offset)].append((K, ari))
            points[('normalized_mutual_info', method, offset)].append((K, nmi))

    # Define metrics and titles
    metric_list = [
        ('accuracy', 'Accuracy (Majority Vote)'),
        ('adjusted_rand_score', 'Adjusted Rand Index (ARI)'),
        ('normalized_mutual_info', 'Normalized Mutual Information (NMI)')
    ]

    # Colors for offsets (ΔK)
    import itertools
    base_colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    color_cycle_off = itertools.cycle(base_colors)
    offsets_present = sorted({(k[2]) for k in buckets.keys()})
    offset_color = {off: next(color_cycle_off) for off in offsets_present}

    out_dir = os.path.join(output_dir, 'clustering_stats_lineplots')
    os.makedirs(out_dir, exist_ok=True)

    # Prepare per-metric plots
    for metric, title in metric_list:
        # Dynamically scale figure size with number of offsets to improve readability
        n_off = max(1, len(offsets_present))
        width = min(14 + 0.8 * max(0, n_off - 3), 26)  # base 14", grow with offsets, cap at 26"
        height = 8 if n_off <= 6 else 9
        fig, ax = plt.subplots(figsize=(width, height))

        # Scatter per-repeat dots, colored by offset, low opacity
        for method in ['direct_mst', 'grid']:
            for off in offsets_present:
                pts = points.get((metric, method, off), [])
                if not pts:
                    continue
                Ks = np.array([p[0] for p in pts], dtype=float)
                vals = np.array([p[1] for p in pts], dtype=float)
                color = offset_color.get(off, 'gray')
                jitter = (np.random.rand(len(Ks)) - 0.5) * 0.2
                marker = 'o' if method == 'direct_mst' else 'x'
                ax.scatter(Ks + jitter, vals, c=color, alpha=0.25, s=16, marker=marker, edgecolors='none')

        # Plot mean trends per (method, offset): color by offset, style by method
        for off in offsets_present:
            for method in ['direct_mst', 'grid']:
                Ks_sorted = sorted({key[3] for key in buckets.keys() if key[0] == metric and key[1] == method and key[2] == off})
                series = []
                for K in Ks_sorted:
                    vals = buckets.get((metric, method, off, K), [])
                    if vals:
                        series.append((K, float(np.mean(vals))))
                if series:
                    xs, ys = zip(*series)
                    linestyle = '-' if method == 'direct_mst' else ':'  # solid=Direct MST, dotted=Grid
                    ax.plot(xs, ys, linestyle=linestyle, color=offset_color.get(off, 'gray'), linewidth=2.0,
                            label=None)

        ax.set_title(f'{title} vs Agglomerative Clusters K', fontsize=13, fontweight='bold')
        ax.set_xlabel('Agglomerative clusters K')
        ax.set_ylabel(title.split(' ')[0] if metric == 'accuracy' else title)
        ax.grid(True, alpha=0.3)

        # Create two legends: one for methods (line styles), one for offset colors
        from matplotlib.lines import Line2D
        line_handles = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=1.8, label='Direct MST-SOM'),
            Line2D([0], [0], color='black', linestyle=':', linewidth=1.8, label='Grid SOM')
        ]
        legend1 = ax.legend(
            line_handles,
            [h.get_label() for h in line_handles],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            title='Method',
            frameon=False
        )
        ax.add_artist(legend1)
        off_handles = [
            Line2D([0], [0], color=offset_color[off], linestyle='-', linewidth=2.0, label=f'ΔK={off}')
            for off in offsets_present
        ]
        ax.legend(
            handles=off_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.24),
            ncol=min(len(off_handles), 6),
            title='Offset (ΔK)',
            frameon=False
        )

        # Compute summary for subtitle and filename
        centers_sorted = sorted({rec['centers'] for rec in per_run_records})
        offsets_sorted = sorted({rec['delta_k'] for rec in per_run_records})
        try:
            repeats_est = max(
                len({r['seed'] for r in per_run_records if r['centers'] == c and r['agglomerative_clusters'] == (c + o)})
                for c in centers_sorted for o in offsets_sorted if (c + o) >= 2
            )
        except ValueError:
            repeats_est = len({r['seed'] for r in per_run_records})

        subtitle = f"Color by offset (ΔK={offsets_present}). Methods: solid=Direct MST, dotted=Grid. Repeats≈{repeats_est}"
        fig.suptitle(subtitle, fontsize=10, y=0.97)

        # Filename with centers and offsets
        centers_str = '_'.join(str(c) for c in centers_sorted)
        offsets_str = '_'.join(f"{o:+d}" for o in sorted(set(offsets_sorted)))
        method_suffix = f"grid-{grid_method}_mst-{mst_method}"
        out_path = os.path.join(out_dir, f"{metric}_vs_K_centers_[{centers_str}]_dK_{offsets_str}__{method_suffix}.png")

        # Make room at the bottom for legends placed outside the axes
        plt.tight_layout(rect=[0, 0.22, 1, 0.95])
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
