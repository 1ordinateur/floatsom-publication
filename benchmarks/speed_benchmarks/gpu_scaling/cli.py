#!/usr/bin/env python3
"""
Command-line interface for GPU scaling benchmarks.

This module handles argument parsing for the benchmark suite.
Single Responsibility: Parse and validate command-line arguments.
"""

import argparse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run GPU scaling benchmarks for FloatSOM')
    
    # Benchmark mode
    parser.add_argument('--mode', type=str, default='all',
                       choices=['dimension_scaling', 'sample_scaling', 'grid_size_scaling', 'both', 'all'],
                       help='Benchmark mode: dimension_scaling, sample_scaling, grid_size_scaling, both (dim+sample), or all (all 3 modes) (default: all)')
    
    # Repeats configuration
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of repeats for each configuration')
    
    # Dimension scaling parameters
    parser.add_argument('--dimensions', type=int, nargs='+', 
                       default=[50, 100, 200, 500, 1000, 2000],
                       help='List of dimensions to test (for dimension_scaling mode)')
    
    parser.add_argument('--samples', type=int, default=10000000,
                       help='Number of samples (fixed for dimension_scaling mode)')
    
    # Sample scaling parameters
    parser.add_argument('--sample_sizes', type=int, nargs='+',
                       default=[1000000, 5000000, 10000000, 
                       50000000, 100000000, 500000000],
                       help='List of sample sizes to test (for sample_scaling mode)')

    parser.add_argument('--fixed_dimension', type=int, default=50,
                       help='Fixed dimension for sample_scaling mode')

    parser.add_argument('--sampling_method', type=str, default='full',
                       choices=['full', 'random', 'hdsssom'],
                       help='Sampling method to use when generating training batches')

    parser.add_argument('--sampling_fraction', type=float, default=None,
                       help='Fraction of the dataset to sample per iteration when using stochastic sampling methods')
    
    # GPU configuration
    parser.add_argument('--gpu_counts', type=int, nargs='+',
                       default=[1],
                       help='List of GPU counts to test')
    
    # Topology configuration
    parser.add_argument('--topologies', type=str, nargs='+',
                       default=['hexagonal', 'mst', 'rng'],
                       choices=['grid', 'hexagonal', 'mst', 'rng'],
                       help='List of topology types to test (default: grid)')
    
    # SOM parameters
    parser.add_argument('--grid_size', type=int, default=32,
                       help='SOM grid size')
    
    # Grid size scaling parameters
    parser.add_argument('--grid_sizes', type=int, nargs='+',
                       default=[8, 16, 24, 32, 48, 64],
                       help='List of grid sizes to test (for grid_size_scaling mode)')
    
    parser.add_argument('--total_iterations', '--iterations', dest='total_iterations',
                       type=int, default=10,
                       help='Number of training iterations')
    
    parser.add_argument('--initial_learning_rate', '--learning_rate',
                       dest='initial_learning_rate', type=float, default=2.0,
                       help='Initial learning rate')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')

    parser.add_argument('--temp_dir', type=str, default=None,
                       help='Optional shared base directory for run scratch and pre-execution dataset staging.')

    parser.add_argument('--ray_local_storage_path', type=str, default=None,
                       help='Local directory for per-worker FastArrayStore copies (e.g. $PBS_JOBFS)')

    parser.add_argument('--resume', action='store_true',
                       help='Resume incomplete benchmark runs using a manifest file')

    parser.add_argument('--resume_state', type=str, default=None,
                       help='Path to the resume state manifest (defaults to <output_dir>/resume_state.json)')

    parser.add_argument('--reset_resume_state', action='store_true',
                       help='Ignore any existing resume manifest and start a fresh run')

    parser.add_argument('--cache_dir', type=str, default='./gpu_scaling_cache',
                       help='Cache directory for Zarr arrays')
    
    # Processing method configuration
    parser.add_argument('--processing_methods', type=str, nargs='+',
                       default=['colors', 'batch', 'minibatch'],
                       choices=['batch', 'colors', 'minibatch'],
                       help='Processing methods to benchmark (options: batch, minibatch, colors)')

    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help=(
            'Global processing chunk size override for all processing methods. '
            'If omitted, chunk size is resolved automatically.'
        ),
    )

    parser.add_argument('--minibatch_chunk_size', type=int, default=5000,
                       help='Chunk size to use when running minibatch processing')

    parser.add_argument('--run_timeout_minutes', type=int, default=30,
                       help='Maximum minutes to allow each benchmark run before timing out')

    parser.add_argument('--safe-cleanup', action='store_true',
                       help='Use safe Ray cleanup (slower) instead of fast cleanup')
    
    # Execution options
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration without executing')
    
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip configurations that already have results')

    parser.add_argument('--merge_existing', action='store_true',
                       help='Merge new results into existing result files (replace collisions)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    
    # Additional parameters required for large dataset benchmarking
    parser.add_argument(
        '--zarr_chunk_size',
        type=int,
        default=None,
        help='Chunk size for generated Zarr array storage (rows per chunk). '
             'Defaults to the FloatSOM processing chunk size.',
    )
    
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Force regeneration of cached data')
    
    # Ray multi-GPU parameters
    parser.add_argument(
        '--use_ray',
        dest='use_ray',
        action='store_true',
        help='Enable Ray for multi-GPU support',
    )
    parser.add_argument(
        '--no_use_ray',
        dest='use_ray',
        action='store_false',
        help='Disable Ray and run local single-GPU processors',
    )
    parser.set_defaults(use_ray=True)

    parser.add_argument('--ray_gpu_count', type=int, default=None,
                       help='Number of GPUs to use with Ray (None for auto-detect)')

    parser.add_argument(
        '--force_disk_mode',
        action='store_true',
        default=False,
        help=(
            'Force disk staging for Ray data distribution (disable RAM-mode loading '
            'heuristics and always use worker-local FastArrayStore shards).'
        ),
    )

    parser.add_argument(
        '--ray_collective_barriers',
        action='store_true',
        default=False,
        help=(
            'Enable extra ray.util.collective.barrier synchronizations at iteration boundaries. '
            'Useful for debugging/stability, but typically slows multi-node scaling.'
        ),
    )

    # Profiling parameters
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling to output code performance statistics')

    parser.add_argument('--profile_output', type=str, default='profile_results.txt',
                       help='Output file for profiling results')

    parser.add_argument('--profile-workers', action='store_true',
                       help='Enable worker-side CPU profiling for Ray workers')

    parser.add_argument('--profile-workers-output-dir', type=str, default=None,
                       help='Directory for worker profiling outputs (default: <run>/worker_profiles)')

    parser.add_argument('--profile-workers-max-stats', type=int, default=50,
                       help='Number of worker profile entries to emit')
    
    return parser.parse_args()
