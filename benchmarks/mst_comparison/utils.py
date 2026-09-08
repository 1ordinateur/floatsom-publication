"""
Utility functions for MST comparison benchmarks.

This module contains common utility functions used across different modes:
- Argument parsing
- Data generation
- Result saving
- Dataset utilities
"""

import os
import time
import argparse
import numpy as np
import cupy as cp
import json
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Import data generation utilities
from floatsom_benchmarks.data_generation import (
    generate_2d_test_data,
    generate_3d_test_data,
    generate_test_data,
)

# Import sklearn dataset utilities
try:
    from floatsom_benchmarks.evaluation.sklearn_datasets import (
        generate_sklearn_dataset,
        SKLEARN_AVAILABLE
    )
except ImportError:
    SKLEARN_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare MST quality from direct MST-SOM vs hexagonal-to-MST conversion'
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'stats_eval', 'clustering', 'clustering_stats'],
                        help='Mode of operation: comparison (default), stats_eval, or clustering')
    
    # Data generation parameters
    parser.add_argument('--data_type', type=str, default='clusters',
                        choices=['2d', '3d', 'clusters', 'sklearn_blobs', 'sklearn_moons', 
                                'sklearn_circles', 'sklearn_swiss_roll'],
                        help='Type of data to generate')
    
    parser.add_argument('--difficulty', type=str, default='hard',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for sklearn datasets')
    
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate')
    
    parser.add_argument('--input_dim', type=int, default=15,
                        help='Input dimension for cluster data')
    
    parser.add_argument('--clusters', type=int, default=None,
                        help='Number of clusters (None = dataset default). For sklearn_blobs, None keeps configured centers unless overridden.')
    
    # SOM configuration
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Size of SOM grid (creates grid_size x grid_size nodes)')
    
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Initial learning rate')
    
    parser.add_argument('--initial_radius', type=float, default=None,
                        help='Initial neighborhood radius (None for auto-calculate)')
    
    parser.add_argument('--mst_update_frequency', type=int, default=50,
                        help='How often to update MST during training (when not using dynamic frequency)')
    
    parser.add_argument('--dynamic_mst_frequency', action='store_true', default=True,
                        help='Use dynamic MST update frequency with decay')
    
    parser.add_argument('--no_dynamic_mst_frequency', action='store_false', dest='dynamic_mst_frequency',
                        help='Use fixed MST update frequency')
    
    parser.add_argument('--mst_decay_function', type=str, default='exponential',
                        choices=['exponential', 'linear', 'sigmoid', 'gaussian', 'asymptotic'],
                        help='Decay function for dynamic MST frequency')
    
    parser.add_argument('--initial_mst_frequency', type=int, default=1,
                        help='Initial MST update frequency (update every N iterations)')
    
    parser.add_argument('--final_mst_frequency', type=int, default=10,
                        help='Final MST update frequency (update every N iterations)')
    
    # Processing configuration
    parser.add_argument('--sampling_method', type=str, default='full',
                        choices=['full', 'random', 'hdsssom'],
                        help='Sample selection method')

    parser.add_argument('--processing_method', type=str, default='batch',
                        choices=['batch', 'colors'],
                        help='Processing method')

    parser.add_argument('--batch_mode', type=str, default='full_batch',
                        choices=['full_batch', 'minibatch'],
                        help='Batch processing mode')

    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for batch processing')

    # Training dynamics: expose momentum, normalization, and decay types
    parser.add_argument('--use_momentum', action='store_true', default=False,
                        help='Enable momentum in weight updates')

    parser.add_argument('--no_momentum', action='store_false', dest='use_momentum',
                        help='Disable momentum in weight updates')

    parser.add_argument('--momentum_init', type=float, default=0.5,
                        help='Initial momentum coefficient')

    parser.add_argument('--normalization', type=str, default='xpysom',
                        choices=['count_based', 'weighted', 'hybrid', 'clamped_weighted', 'local', 'adaptive', 'none', 'minisom_weighted', 'xpysom'],
                        help='Normalization method for weight updates')

    parser.add_argument('--lr_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'sigmoid', 'gaussian', 'asymptotic', 'fixed'],
                        help='Decay type for learning rate')

    parser.add_argument('--radius_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'sigmoid', 'gaussian', 'asymptotic', 'fixed'],
                        help='Decay type for radius')
    
    parser.add_argument('--lr_decay_factor', type=float, default=8.0,
                        help='Decay factor for learning rate (higher = slower decay)')
    
    parser.add_argument('--radius_decay_factor', type=float, default=None,
                        help='Decay factor for radius (higher = slower decay)')

    # Normalization parameters for parity
    parser.add_argument('--norm_alpha', type=float, default=None,
                        help='Blend ratio for hybrid normalization (0.0-1.0)')

    parser.add_argument('--norm_clamp_factor', type=float, default=None,
                        help='Clamp factor for clamped_weighted normalization (>0)')

    parser.add_argument('--norm_percentile', type=float, default=None,
                        help='Percentile threshold for local normalization (0-100)')

    parser.add_argument('--norm_max_update_threshold', type=float, default=None,
                        help='Threshold for extreme value validation (>0)')

    parser.add_argument('--virtual_ratio', type=float, default=0.5,
                        help='Virtual samples ratio for count_based/weighted normalization (0.2=fast, 0.5=balanced, 1.0=stable)')

    parser.add_argument('--initialization_method', type=str, default='random',
                        choices=['random', 'pca', 'pca_sampling', 'pca_sampling_snake', 'pca_density'],
                        help='Weight initialization method')

    # Convergence / early stopping parity
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help='Convergence threshold for early stopping')
    
    parser.add_argument('--min_iterations', type=int, default=20,
                        help='Minimum iterations before checking convergence')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='mst_comparison_results',
                        help='Directory for output files')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of MSTs')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (None = random per run/method)')
    
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for computation')
    
    # Statistical evaluation mode
    parser.add_argument('--stats_eval', action='store_true',
                        help='Run statistical evaluation across all datasets')
    
    parser.add_argument('--stats_difficulty', type=str, default='hard',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for sklearn datasets in stats mode')
    
    parser.add_argument('--stats_iterations', type=int, default=100,
                        help='Number of iterations per dataset in stats mode')
    
    parser.add_argument('--stats_runs', type=int, default=1,
                        help='Number of runs per dataset for robustness (deprecated, use --repeats)')
    
    parser.add_argument('--repeats', type=int, default=10,
                        help='Number of repeats for generative datasets (fixed datasets run once)')
    
    parser.add_argument('--stats_output', type=str, default='mst_statistical_evaluation.txt',
                        help='Output file for statistical evaluation results')
    
    parser.add_argument('--graphs', action='store_true',
                        help='Generate metric comparison dot plots when running statistical evaluation')

    # Clustering stats mode (blobs) options
    parser.add_argument('--clusters_list', type=int, nargs='+', default=None,
                        help='Space-separated list of cluster counts (e.g., --clusters_list 3 4 5). Comma strings also supported for back-compat.')
    
    # Agglomerative clustering hyperparameter (decoupled from true centers)
    parser.add_argument('--agglomerative_clusters', type=int, default=None,
                        help='Number of clusters to use for agglomerative clustering (if None, defaults to true centers)')

    # Offsets for relative agglomerative clusters in clustering_stats mode
    parser.add_argument('--agglomerative_offsets', type=int, nargs='+', default=[-3, 0, 3],
                        help='Space-separated integer offsets relative to centers (e.g., --agglomerative_offsets -3 0 3).')

    # Metacluster method selection
    parser.add_argument('--metacluster_method', type=str, default='agglomerative',
                        choices=['agglomerative', 'consensus'],
                        help='Global method to create metaclusters on SOM weights (applies to both methods unless overridden).')

    # Optional per-method overrides
    parser.add_argument('--grid_metacluster_method', type=str, default=None,
                        choices=['agglomerative', 'consensus'],
                        help='Override metacluster method for Grid SOM only.')
    parser.add_argument('--mst_metacluster_method', type=str, default=None,
                        choices=['agglomerative', 'consensus'],
                        help='Override metacluster method for Direct MST-SOM only.')

    # Consensus clustering options (used when --metacluster_method consensus)
    parser.add_argument('--consensus_runs', type=int, default=20,
                        help='Number of bootstrap runs for consensus clustering.')
    parser.add_argument('--consensus_subsample', type=float, default=0.8,
                        help='Proportion of features to subsample per consensus run (0 < p ≤ 1).')
    parser.add_argument('--consensus_linkage', type=str, default='average',
                        choices=['average', 'complete', 'single', 'ward'],
                        help='Linkage for final clustering of the consensus matrix.')
    
    args = parser.parse_args()

    # Back-compat for deprecated flag.
    stats_runs = getattr(args, "stats_runs", None)
    if stats_runs is not None and stats_runs != 1:
        import warnings

        if getattr(args, "repeats", 10) != 10:
            warnings.warn(
                "--stats_runs is deprecated and ignored because --repeats is set.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "--stats_runs is deprecated; mapping to --repeats.",
                DeprecationWarning,
                stacklevel=2,
            )
            args.repeats = stats_runs

    return args


def generate_data(args) -> Tuple[cp.ndarray, Dict[str, Any]]:
    """Generate dataset based on configuration."""
    if getattr(args, 'seed', None) is not None:
        np.random.seed(args.seed)
    
    if args.data_type.startswith('sklearn_'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for sklearn datasets")
        
        dataset_name = args.data_type.replace('sklearn_', '')
        # Optional override for number of centers in blobs (supports --clusters and internal blobs_centers)
        centers_override = None
        if dataset_name == 'blobs':
            if hasattr(args, 'blobs_centers') and args.blobs_centers is not None:
                centers_override = args.blobs_centers
            elif hasattr(args, 'clusters') and args.clusters is not None:
                centers_override = args.clusters

        data, metadata = generate_sklearn_dataset(
            dataset_name=dataset_name,
            difficulty=args.difficulty if hasattr(args, 'difficulty') else 'hard',
            seed=args.seed,
            n_features=args.input_dim if dataset_name == 'blobs' else None,
            normalize=True,
            centers=centers_override
        )
        
    elif args.data_type == '2d':
        data = generate_2d_test_data(n_points=args.samples, seed=args.seed)
        metadata = {'dataset_name': '2d_w_shaped', 'shape': data.shape}
        
    elif args.data_type == '3d':
        data = generate_3d_test_data(n_points=args.samples, seed=args.seed)
        metadata = {'dataset_name': '3d_bowl_shaped', 'shape': data.shape}
        
    else:  # 'clusters' (synthetic)
        # Fallback to 5 clusters if not provided
        n_clusters_val = args.clusters if args.clusters is not None else 5
        data = generate_test_data(
            n_samples=args.samples,
            input_dim=args.input_dim,
            n_clusters=n_clusters_val,
            verbose=args.verbose
        )
        metadata = {
            'dataset_name': f'clusters_{args.input_dim}d',
            'shape': data.shape,
            'n_clusters': n_clusters_val
        }
    
    # Convert to CuPy array if using GPU
    if args.use_gpu and not isinstance(data, cp.ndarray):
        data = cp.asarray(data, dtype=cp.float32)
    
    return data, metadata


def save_results(comparison_results: Dict[str, Any], 
                direct_train_time: float, hex_train_time: float,
                args, output_dir: str):
    """Save comprehensive results to file."""
    
    output_file = os.path.join(output_dir, 'mst_comparison_results.txt')
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MST TOPOLOGY COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {args.data_type}\n")
        f.write(f"Samples: {args.samples}\n")
        f.write(f"Input dimension: {args.input_dim}\n")
        f.write(f"Clusters: {args.clusters}\n")
        f.write(f"Grid size: {args.grid_size} x {args.grid_size} = {args.grid_size**2} nodes\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        if args.dynamic_mst_frequency:
            f.write(f"MST update: Dynamic (initial: every {args.initial_mst_frequency} iter, "
                   f"final: every {args.final_mst_frequency} iter, decay: {args.mst_decay_function})\n")
        else:
            f.write(f"MST update frequency: Every {args.mst_update_frequency or 50} iterations\n")
        f.write(f"Processing method: {args.processing_method}\n")
        f.write(f"Initialization: {args.initialization_method}\n")
        f.write(f"GPU: {'enabled' if args.use_gpu else 'disabled'}\n")
        f.write("\n")
        
        # Training times
        f.write("TRAINING PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Direct MST-SOM training time: {direct_train_time:.3f}s\n")
        f.write(f"Hexagonal SOM training time: {hex_train_time:.3f}s\n")
        f.write(f"Speed difference: {abs(direct_train_time - hex_train_time):.3f}s ")
        f.write(f"({'Direct' if direct_train_time < hex_train_time else 'Hexagonal'} faster)\n\n")
        
        # Detailed metrics comparison
        f.write("DETAILED METRIC COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        for metric, values in comparison_results['comparison'].items():
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Direct MST-SOM: {values['MST1']:.6f}\n")
            f.write(f"  Hexagonal-to-MST: {values['MST2']:.6f}\n")
            f.write(f"  Better approach: {values['better'].replace('MST1', 'Direct MST-SOM').replace('MST2', 'Hexagonal-to-MST')}\n")
            
            # Add interpretation
            if metric == 'gap_ratio':
                f.write("  (Higher = clearer cluster separation)\n")
            elif metric == 'gini':
                f.write("  (Higher = more heterogeneous edge weights, better for clustering)\n")
            elif metric == 'edge_ratio':
                f.write("  (Higher = better separation between clusters)\n")
            elif metric == 'bimodality':
                f.write("  (>0.555 suggests bimodal distribution, good for clustering)\n")
            elif metric == 'separation':
                f.write("  (Higher = better cluster separability)\n")
            elif metric == 'stability':
                f.write("  (Higher = more stable cluster cuts)\n")
        
        f.write("\n")
        f.write("OVERALL SCORES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Direct MST-SOM score: {comparison_results['overall_scores']['MST1']:.4f}\n")
        f.write(f"Hexagonal-to-MST score: {comparison_results['overall_scores']['MST2']:.4f}\n")
        f.write("\n")
        
        f.write("FINAL RECOMMENDATION:\n")
        f.write("-" * 40 + "\n")
        recommendation = comparison_results['recommendation']
        recommendation = recommendation.replace('MST1', 'Direct MST-SOM')
        recommendation = recommendation.replace('MST2', 'Hexagonal-to-MST')
        f.write(f"{recommendation}\n")
        
        # Add interpretation (match original behavior)
        f.write("\nINTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        
        score1 = comparison_results['overall_scores']['MST1']
        score2 = comparison_results['overall_scores']['MST2']
        
        if abs(score1 - score2) < 0.05:
            f.write("Both approaches produce similar MST quality.\n")
            f.write("Consider using hexagonal-to-MST if training speed is important.\n")
            f.write("Consider using direct MST-SOM if MST structure updates during training are valuable.\n")
        elif score1 > score2:
            improvement = ((score1 - score2) / score2) * 100
            f.write(f"Direct MST-SOM produces {improvement:.1f}% better MST quality.\n")
            f.write("The dynamic MST updates during training lead to better final structure.\n")
            f.write("This approach is recommended when MST quality is critical.\n")
        else:
            improvement = ((score2 - score1) / score1) * 100
            f.write(f"Hexagonal-to-MST produces {improvement:.1f}% better MST quality.\n")
            f.write("The stable hexagonal structure during training leads to better final weights.\n")
            f.write("This approach is recommended for faster training with good MST quality.\n")
    
    logger.info(f"Results saved to {output_file}")


def get_all_sklearn_datasets() -> List[str]:
    """Get list of all available sklearn datasets."""
    generative = ['blobs', 'moons', 'circles', 'swiss_roll', 's_curve']
    fixed = ['iris', 'diabetes', 'breast_cancer', 'wine', 'olivetti_faces', 'digits']
    return generative + fixed
