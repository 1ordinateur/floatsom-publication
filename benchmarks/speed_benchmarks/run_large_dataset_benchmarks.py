#!/usr/bin/env python3
"""
FloatSOM benchmarks for large datasets using Zarr for out-of-core processing.

This script handles datasets that are too large to fit in memory by:
1. Generating random data and caching it to Zarr arrays
2. Passing the Zarr path to FloatSOM for processing
3. Using memory-efficient processing strategies
"""

import os
import sys
import time
import argparse
import numpy as np
import zarr
import hashlib
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

os.environ['RAY_DEDUP_LOGS'] = '0'

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Optional Ray import for multi-GPU support
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

# Add the parent directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# Import FloatSOM components
from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig, RayConfig
from floatsom.processing.processing_params import calculate_auto_chunk_size_for_method
from floatsom.data.sources.file import FileDataSource
from floatsom.data.fast_array_store import FastArrayStore
from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FloatSOM benchmarks on large random datasets using Zarr')
    
    # Data generation parameters
    parser.add_argument('--samples', type=int, default=10000000,
                        help='Number of samples to generate')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to an existing dataset (Zarr store or FastArrayStore). Skips generation when provided')
    
    parser.add_argument('--input_dim', type=int, default=50,
                        help='Input dimension for data')
    
    # Zarr storage parameters
    parser.add_argument('--cache_dir', type=str, default='./large_dataset_cache',
                        help='Directory for caching Zarr arrays')
    
    parser.add_argument('--zarr_chunk_size', type=int, default=1000000,
                        help='Chunk size for Zarr array storage')
    
    # Removed --processing_chunk_size argument as we always auto-calculate based on dimensions
    
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Force regeneration of cached data')
    
    # SOM configuration parameters
    parser.add_argument('--grid_size', type=int, default=20,
                        help='Size of SOM grid')
    
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Initial learning rate')
    
    parser.add_argument('--initial_radius', type=float, default=None,
                        help='Initial neighborhood radius')
    
    parser.add_argument('--lr_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'asymptotic'],
                        help='Decay type for learning rate')
    
    parser.add_argument('--radius_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'asymptotic'],
                        help='Decay type for radius')
    
    parser.add_argument('--lr_decay_factor', type=float, default=8.0,
                        help='Decay factor for learning rate (higher = slower decay)')
    
    parser.add_argument('--radius_decay_factor', type=float, default=1.0,
                        help='Decay factor for radius (higher = slower decay)')
    
    # Processing configuration
    parser.add_argument('--processing_method', type=str, default='batch',
                        choices=['batch', 'colors'],
                        help='Processing method')
        
    parser.add_argument('--batch_mode', type=str, default='full_batch',
                        choices=['full_batch', 'minibatch'],
                        help='Batch processing mode')
    
        
    parser.add_argument('--use_momentum', action='store_true', default=False,
                        help='Enable momentum in weight updates')
    
    parser.add_argument('--no_momentum', action='store_false', dest='use_momentum',
                        help='Disable momentum in weight updates')
    
    parser.add_argument('--momentum_init', type=float, default=0.5,
                        help='Initial momentum coefficient')
    
    parser.add_argument('--normalization', type=str, default='xpysom',
                        choices=['count_based', 'weighted', 'hybrid', 'clamped_weighted', 'local', 'adaptive', 'none', 'minisom_weighted', 'xpysom'],
                        help='Normalization method for weight updates')
    
    parser.add_argument('--norm_alpha', type=float, default=None,
                        help='Blend ratio for hybrid normalization (0.0-1.0, required for hybrid)')
    
    parser.add_argument('--norm_clamp_factor', type=float, default=None,
                        help='Clamp factor for clamped_weighted normalization (>0, required for clamped_weighted)')
    
    parser.add_argument('--norm_percentile', type=float, default=None,
                        help='Percentile threshold for local normalization (0-100, required for local)')
    
    parser.add_argument('--norm_max_update_threshold', type=float, default=None,
                        help='Threshold for extreme value validation (optional, >0)')
    
    parser.add_argument('--virtual_ratio', type=float, default=0.5,
                        help='Virtual samples ratio for count_based/weighted normalization (0.2=fast, 0.5=balanced, 1.0=stable)')
    
    # Sampling configuration
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for memory-efficient processing')
    
    parser.add_argument('--sampling_method', type=str, default='full',
                        choices=['full', 'random', 'hdsssom'],
                        help='Sample selection method')
    
    # Topology configuration
    parser.add_argument('--topology_type', type=str, default='hexagonal',
                        choices=['grid', 'hexagonal', 'mst'],
                        help='Type of topology')
    
    parser.add_argument('--topology_variant', type=str, default='planar',
                        choices=['planar', 'toroidal'],
                        help='Topology variant (for grid/hexagonal)')
    
    parser.add_argument('--initialization_method', type=str, default='random',
                        choices=['random', 'pca', 'pca_sampling', 'pca_sampling_snake', 'pca_density'],
                        help='Method to initialize weights (pca=standard PCA, pca_sampling=sampled PCA with grid ordering, pca_sampling_snake=snake pattern, pca_density=density-weighted PCA)')
    
    # Convergence parameters
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help='Convergence threshold for early stopping')
    
    parser.add_argument('--min_iterations', type=int, default=20,
                        help='Minimum iterations before checking convergence')
    
    # Visualization parameters
    parser.add_argument('--save_iterations', action='store_true', default=False,
                        help='Save intermediate weights during training')
    
    parser.add_argument('--no_save_iterations', action='store_false', dest='save_iterations',
                        help='Do not save intermediate weights during training')
    
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save weights every N iterations')
    
    # Other parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for computation')
    
    parser.add_argument('--output_file', type=str, default='large_dataset_benchmark_results.txt',
                        help='Output file for benchmark results')
    
    # Ray multi-GPU parameters
    parser.add_argument('--use_ray', action='store_true',
                        help='Enable Ray multi-GPU processing for batch method')
    
    parser.add_argument('--ray_num_gpus', type=int, default=None,
                        help='Number of GPUs to use with Ray (None for auto-detect)')
    
    parser.add_argument('--ray_collective_group', type=str, default='default',
                        help='Name for NCCL collective group (default: "default")')

    parser.add_argument('--ray_storage_path', type=str, default=None,
                        help='Shared directory for FastArrayStore staging')

    parser.add_argument('--ray_local_storage_path', type=str, default=None,
                        help='Local directory for per-worker FastArrayStore copies')
        
    # Profiling parameters
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling to output code performance statistics')
                        
    parser.add_argument('--calculate_qe', action='store_true', default=False,
                        help='Calculate quantization error')
    
    parser.add_argument('--profile_output', type=str, default='profile_results.txt',
                        help='Output file for profiling results')
    
    return parser.parse_args()


def _load_existing_dataset(dataset_path: str, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Load metadata for an existing dataset without rewriting it."""

    resolved_path = os.path.abspath(dataset_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")

    dataset_name = Path(resolved_path).stem

    metadata: Dict[str, Any]

    # FastArrayStore detection (directory containing array_metadata.json)
    metadata_path = os.path.join(resolved_path, 'array_metadata.json')
    if os.path.isdir(resolved_path) and os.path.exists(metadata_path):
        store = FastArrayStore(resolved_path, mode='r')
        shape = store.shape if len(store.shape) > 1 else (store.shape[0], 1)
        metadata = {
            'dataset_name': dataset_name,
            'shape': shape,
            'dtype': 'float32',
            'chunks': store.chunks,
            'format': 'fast_array_store'
        }
        store.close()
    else:
        # Assume Zarr-compatible path
        try:
            z = zarr.open(resolved_path, mode='r')
        except Exception as exc:
            raise ValueError(f"Unable to open dataset at {resolved_path}: {exc}") from exc

        if not hasattr(z, 'shape'):
            raise ValueError(f"Dataset at {resolved_path} is not an array store")

        shape = z.shape if len(z.shape) > 1 else (z.shape[0], 1)
        metadata = {
            'dataset_name': dataset_name,
            'shape': shape,
            'dtype': str(z.dtype),
            'chunks': getattr(z, 'chunks', None),
            'format': 'zarr'
        }

    metadata['zarr_path'] = resolved_path
    metadata['source'] = 'provided'

    if verbose:
        logger.info(f"Using existing dataset '{metadata['dataset_name']}' ({metadata['shape'][0]:,} samples × {metadata['shape'][1]})")
        logger.info(f"  Format: {metadata['format']}")

    return resolved_path, metadata


def generate_random_zarr_data(n_samples: int, input_dim: int, zarr_path: str, 
                              zarr_chunk_size: int = 10000,
                              seed: Optional[int] = None, verbose: bool = True) -> zarr.Array:
    """Generate random data and save to Zarr array.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Dimensionality of each sample
        zarr_path: Path to save Zarr array
        zarr_chunk_size: Chunk size for Zarr array storage
        seed: Random seed for reproducibility
        verbose: Whether to print progress messages
    
    Returns:
        Zarr array with random data
    """
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        size_gb = (n_samples * input_dim * 4) / (1024**3)
        print(f"Generating random data: {n_samples:,} samples × {input_dim} dimensions")
        print(f"Estimated size: {size_gb:.2f} GB (uncompressed)")
        print(f"Zarr path: {zarr_path}")
        print(f"Zarr chunk size: {zarr_chunk_size:,} samples")
    
    # Create Zarr array
    z = zarr.open_array(
        zarr_path,
        mode='w',
        shape=(n_samples, input_dim),
        chunks=(min(zarr_chunk_size, n_samples), input_dim),
        dtype='float32'
    )
    
    # Generate data in chunks
    for i in range(0, n_samples, zarr_chunk_size):
        end_idx = min(i + zarr_chunk_size, n_samples)
        chunk_data = np.random.rand(end_idx - i, input_dim).astype(np.float32)
        z[i:end_idx] = chunk_data
        
        if verbose and (i % (zarr_chunk_size * 10) == 0 or end_idx == n_samples):
            progress = 100 * end_idx / n_samples
            print(f"  Generated {end_idx:,}/{n_samples:,} samples ({progress:.1f}%)")
    
    if verbose:
        actual_size_mb = os.path.getsize(zarr_path) / (1024**2) if os.path.isfile(zarr_path) else 0
        if os.path.isdir(zarr_path):
            # Calculate directory size for Zarr arrays stored as directories
            actual_size_mb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(zarr_path)
                for filename in filenames
            ) / (1024**2)
        print(f"Data generation complete. Actual size: {actual_size_mb:.1f} MB (compressed)")
    
    return z


def generate_or_load_data(args) -> Tuple[str, Dict[str, Any]]:
    """Generate or load cached data.
    
    Returns:
        Tuple of (zarr_path, metadata)
    """
    if args.dataset_path:
        if args.force_regenerate:
            logger.warning("--force_regenerate is ignored when --dataset_path is provided")
        dataset_path, metadata = _load_existing_dataset(args.dataset_path, verbose=args.verbose)
        print(f"Using provided dataset at {dataset_path}")
        args.samples = metadata['shape'][0]
        args.input_dim = metadata['shape'][1] if len(metadata['shape']) > 1 else 1
        return dataset_path, metadata

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique identifier for dataset
    config_str = f"random_{args.samples}_{args.input_dim}_{args.seed}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Zarr path
    zarr_filename = f"random_{args.samples}_{args.input_dim}_{config_hash}.zarr"
    zarr_path = str(cache_dir / zarr_filename)
    
    # Check if data exists and not forcing regeneration
    if os.path.exists(zarr_path) and not args.force_regenerate:
        print(f"Loading cached data from {zarr_path}")
        z = zarr.open_array(zarr_path, mode='r')
        metadata = {
            'dataset_name': 'random',
            'shape': z.shape,
            'dtype': str(z.dtype),
            'chunks': z.chunks,
            'zarr_path': zarr_path,
            'format': 'zarr',
            'source': 'generated_random'
        }
    else:
        # Generate new data
        print(f"Generating new random dataset...")
        
        z = generate_random_zarr_data(
            n_samples=args.samples,
            input_dim=args.input_dim,
            zarr_path=zarr_path,
            zarr_chunk_size=args.zarr_chunk_size,
            seed=args.seed,
            verbose=args.verbose
        )
        
        metadata = {
            'dataset_name': 'random',
            'shape': z.shape,
            'dtype': str(z.dtype),
            'chunks': z.chunks,
            'zarr_path': zarr_path,
            'format': 'zarr',
            'source': 'generated_random'
        }
    
    return zarr_path, metadata


def train_floatsom_with_zarr(zarr_path: str, args, metadata: Dict[str, Any]) -> Tuple[FloatSOM, Dict[str, Any], float]:
    """Train FloatSOM using Zarr data source.
    
    Args:
        zarr_path: Path to Zarr array
        args: Command line arguments
        metadata: Dataset metadata
    
    Returns:
        Tuple of (trained SOM, training stats, training time)
    """
    n_samples, input_dim = metadata['shape']
    
    print(f"\nConfiguring FloatSOM for large dataset processing...")
    print(f"Data source: {metadata.get('format', 'dataset')} at {zarr_path}")
    print(f"Shape: {n_samples:,} × {input_dim}")
    
    # Create sampling configuration
    sampling_config = SamplingConfig(
        method=args.sampling_method,
        random_seed=args.seed
    )
    
    # Set default normalization parameters if not provided
    norm_alpha = args.norm_alpha
    norm_clamp_factor = args.norm_clamp_factor
    norm_percentile = args.norm_percentile
    norm_max_update_threshold = args.norm_max_update_threshold
    
    if args.normalization == 'hybrid' and norm_alpha is None:
        norm_alpha = 0.5  # Balanced blend between weighted and count-based
    elif args.normalization == 'clamped_weighted' and norm_clamp_factor is None:
        norm_clamp_factor = 2.0  # Reasonable clamp factor
    elif args.normalization == 'local' and norm_percentile is None:
        norm_percentile = 85.0  # Use 85th percentile threshold
    
    # Calculate optimal chunk size based on shared heuristic
    if args.chunk_size is None:
        optimal_chunk_size = calculate_auto_chunk_size_for_method(
            input_dim, args.processing_method
        )
        optimal_chunk_size = min(optimal_chunk_size, 500_000)
    else:
        optimal_chunk_size = args.chunk_size

    # Prepare Ray configuration if Ray is requested and processing method supports it
    ray_config = None
    if args.use_ray and args.processing_method in ['batch', 'colors']:
        if not RAY_AVAILABLE:
            raise ImportError("Ray requested but not available. Please install ray: pip install ray")
        
        # Import RayConfig
        from floatsom.floatsom_params import RayConfig
        
        # Build Ray config dict, only including non-None values
        ray_config_kwargs = {
            'num_gpus': args.ray_num_gpus,
            'collective_group_name': args.ray_collective_group,
        }
        
        ray_config_kwargs['chunk_size'] = optimal_chunk_size

        if not args.ray_storage_path or not args.ray_local_storage_path:
            raise ValueError(
                "Ray requires --ray_storage_path (shared) and --ray_local_storage_path (local) to be set."
            )

        shared_path = os.path.abspath(os.path.expanduser(args.ray_storage_path))
        local_path = os.path.abspath(os.path.expanduser(args.ray_local_storage_path))
        os.makedirs(shared_path, exist_ok=True)
        os.makedirs(local_path, exist_ok=True)

        ray_config_kwargs['storage_path'] = shared_path
        ray_config_kwargs['local_storage_path'] = local_path

        ray_config = RayConfig(**ray_config_kwargs)

    if args.verbose:
        print(f"Using chunk size: {optimal_chunk_size:,} (calculated for {input_dim} dimensions)")
    
    # Create processing configuration
    processing_config = ProcessingConfig(
        method=args.processing_method,
        batch_mode=args.batch_mode,
        chunk_size=optimal_chunk_size,  # Explicitly pass calculated chunk size
        enable_momentum=args.use_momentum,
        initial_momentum=args.momentum_init,
        normalization=args.normalization,
        norm_alpha=norm_alpha,
        norm_clamp_factor=norm_clamp_factor,
        norm_percentile=norm_percentile,
        norm_max_update_threshold=norm_max_update_threshold,
        virtual_ratio=args.virtual_ratio,
        use_gpu=args.use_gpu,
        ray_config=ray_config  # Pass ray_config directly
    )
    
    # Create topology configuration
    topology_config = TopologyConfig(
        topology_type=args.topology_type,
        topology_variant=args.topology_variant,
        grid_size=args.grid_size,
        grid_dim=2
    )
    
    # Create FloatSOM parameters
    params = FloatSOMParams(
        defaults_profile="publication",
        input_dim=input_dim,
        total_iterations=args.iterations,
        initial_learning_rate=args.learning_rate,
        initial_radius=args.initial_radius,
        lr_decay_type=args.lr_decay_type,
        radius_decay_type=args.radius_decay_type,
        lr_decay_factor=args.lr_decay_factor,
        radius_decay_factor=args.radius_decay_factor,
        sampling_config=sampling_config,
        processing_config=processing_config,
        topology_config=topology_config,
        convergence_threshold=args.convergence_threshold,
        min_iterations=args.min_iterations,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        seed=args.seed,
        store_history=args.save_iterations,
        initialization_method=args.initialization_method
    )
    
    # Create FloatSOM with file data source and data shape
    som = create_floatsom(params)
    
    # Create file data source
    data_source = FileDataSource(zarr_path)
    
    print(f"\nTraining FloatSOM...")
    print(f"Configuration: {args.sampling_method} sampling, {args.processing_method} processing")
    print(f"Grid: {args.grid_size}×{args.grid_size} {args.topology_type}")
    print(f"Iterations: {args.iterations}")
    if args.sampling_method != 'full':
        print(f"Sampling target proportion: {sampling_config.target_proportion}")
    
    # Train the SOM
    start_time = time.time()
    training_stats = som.train(data_source)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f}s")
    
    return som, training_stats, train_time


def calculate_quantization_error(zarr_path: str, som: FloatSOM, args, output_dir: str) -> float:
    """Calculate quantization error using BMU finder from utils.
    Converts Zarr format to FastArrayStore if needed, then uses CPUGPUFastLoader.
    
    Args:
        zarr_path: Path to data array (FastArrayStore or Zarr)
        som: Trained SOM
        args: Command line arguments
        output_dir: Directory to save results
        
    Returns:
        Average quantization error
    """
    import cupy as cp
    from floatsom.processing.utils import find_bmus
    from floatsom.data.convert_zarr_to_fast import convert_zarr_to_fast
    
    print("\nCalculating quantization error...")
    
    # Detect format
    data_path = zarr_path
    is_fast_format = os.path.exists(os.path.join(data_path, 'array_metadata.json'))
    
    # Convert Zarr to FastArrayStore if needed
    if not is_fast_format:
        print("  Detected Zarr format - converting to FastArrayStore for optimized processing")
        
        # Create temporary FastArrayStore path
        fast_path = os.path.join(output_dir, 'temp_fast_array.fast')
        
        # Convert Zarr to FastArrayStore using a reasonable chunk size
        # Use the same calculation as below, or a simple default
        convert_chunk_size = min(50000, max(10000, int(2 * 1024**3 / (args.grid_size * args.grid_size * 4))))
        convert_zarr_to_fast(data_path, fast_path, chunk_size=convert_chunk_size)
        
        # Update data_path to use the converted FastArrayStore
        data_path = fast_path
        print(f"  Conversion complete - using FastArrayStore at {fast_path}")
    else:
        print("  Detected FastArrayStore format - using optimized loader")
    
    # Calculate optimal chunk size based on GPU memory and grid size
    # Larger chunks = better GPU utilization
    grid_nodes = args.grid_size * args.grid_size
    
    # Use larger chunks - up to 50k samples at once for better GPU efficiency
    # Memory needed: chunk_size * grid_nodes * 4 bytes (float32) for distance matrix
    optimal_chunk_size = min(
        50000,  # Process up to 50k samples at once
        max(10000, int(2 * 1024**3 / (grid_nodes * 4)))  # Use up to 2GB for distance matrix
    )
    
    print(f"  Using optimized chunk size: {optimal_chunk_size:,} samples")
    print(f"  Grid nodes: {grid_nodes} ({args.grid_size}×{args.grid_size})")
    
    # Now always use CPUGPUFastLoader since we've ensured FastArrayStore format
    loader = CPUGPUFastLoader(
        data_path, 
        chunk_size=optimal_chunk_size,  # Use larger optimal chunk size
        randomize_chunks=False,
        prefetch_buffer_size=1  # Reduce prefetch for sequential QE calculation
    )
    n_samples = loader.n_samples
    
    # Get SOM weights
    weights = som.get_weights()
    
    # Process in chunks
    total_error = 0.0
    
    for chunk_idx in range(loader.total_chunks):
        gpu_data, chunk_info = loader.get_chunk(
            chunk_idx,
            advance_cursor=True,
        )
        
        # Find BMUs without additional chunking for better efficiency
        bmus, min_distances = find_bmus(
            gpu_data, 
            weights, 
            return_distances=True,
            chunk_size=None  # No secondary chunking - process entire chunk at once
        )
        
        # Accumulate error
        chunk_error = cp.sum(min_distances).item()
        total_error += chunk_error
        
        # Update progress less frequently for large datasets
        if chunk_idx % max(1, loader.total_chunks // 20) == 0 or chunk_idx == loader.total_chunks - 1:
            samples_processed = min((chunk_idx + 1) * optimal_chunk_size, n_samples)
            progress = 100 * samples_processed / n_samples
            print(f"  Processed {samples_processed:,}/{n_samples:,} samples ({progress:.1f}%)")
    
    del loader
    
    # Calculate average
    avg_error = total_error / n_samples
    
    # Save to file
    original_format = 'Zarr (converted to FastArrayStore)' if not is_fast_format else 'FastArrayStore'
    qe_file = os.path.join(output_dir, 'quantization_error.txt')
    with open(qe_file, 'w') as f:
        f.write(f"Quantization Error Analysis - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {zarr_path}\n")
        f.write(f"Format: {original_format}\n")
        f.write(f"Total samples: {n_samples:,}\n")
        f.write(f"SOM grid size: {args.grid_size}×{args.grid_size}\n")
        f.write("\n")
        f.write(f"Total quantization error: {total_error:.6f}\n")
        f.write(f"Average quantization error: {avg_error:.6f}\n")
        f.write(f"Average error per dimension: {avg_error / args.input_dim:.6f}\n")
    
    print(f"Quantization error: {avg_error:.6f}")
    print(f"Results saved to: {qe_file}")
    
    # Clean up temporary FastArrayStore if we created one
    if not is_fast_format and os.path.exists(fast_path):
        try:
            import shutil
            shutil.rmtree(fast_path)
            print(f"  Cleaned up temporary FastArrayStore")
        except Exception as e:
            print(f"  Warning: Could not clean up temporary FastArrayStore: {e}")
    
    return avg_error


def write_results_report(som: FloatSOM, training_stats: Dict[str, Any],
                        train_time: float, metadata: Dict[str, Any],
                        args, output_file: str):
    """Write comprehensive results report to file."""
    
    with open(output_file, 'w') as f:
        f.write(f"FloatSOM Large Dataset Benchmark Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset information
        f.write("Dataset Information:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset name: {metadata.get('dataset_name', 'unknown')}\n")
        f.write(f"Source: {metadata.get('source', 'generated')}\n")
        f.write(f"Shape: {metadata['shape'][0]:,} × {metadata['shape'][1]}\n")
        f.write(f"Data type: {metadata['dtype']}\n")
        if metadata.get('chunks') is not None:
            f.write(f"Chunks: {metadata['chunks']}\n")
        f.write(f"Storage format: {metadata.get('format', 'zarr')}\n")
        f.write(f"Storage path: {metadata.get('zarr_path', 'N/A')}\n")
        f.write("\n")
        
        # SOM Configuration
        f.write("SOM Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Grid size: {args.grid_size}×{args.grid_size}\n")
        f.write(f"Topology: {args.topology_type} ({args.topology_variant})\n")
        f.write(f"Total iterations: {args.total_iterations}\n")
        f.write(f"Initial learning rate: {args.learning_rate}\n")
        f.write(f"Initial radius: {args.initial_radius or 'auto'}\n")
        f.write(f"LR decay type: {args.lr_decay_type}\n")
        f.write(f"LR decay factor: {args.lr_decay_factor}\n")
        f.write(f"Radius decay type: {args.radius_decay_type}\n")
        f.write(f"Radius decay factor: {args.radius_decay_factor}\n")
        f.write(f"Convergence threshold: {args.convergence_threshold}\n")
        f.write(f"Min iterations: {args.min_iterations}\n")
        f.write(f"Initialization: {args.initialization_method}\n")
        f.write(f"Store history: {args.save_iterations}\n")
        f.write("\n")
        
        # Processing Configuration
        f.write("Processing Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Sampling method: {args.sampling_method}\n")
        if args.sampling_method != 'full':
            f.write("Samples per epoch: auto (derived from SamplingConfig target proportion)\n")
        f.write(f"Processing method: {args.processing_method}\n")
        f.write(f"Batch mode: {args.batch_mode}\n")
        # Processing chunk size is now auto-calculated
        if args.zarr_chunk_size:
            f.write(f"Zarr chunk size: {args.zarr_chunk_size}\n")
        f.write(f"Normalization: {args.normalization}\n")
        if args.normalization == 'hybrid' and args.norm_alpha:
            f.write(f"  Alpha: {args.norm_alpha}\n")
        if args.normalization == 'clamped_weighted' and args.norm_clamp_factor:
            f.write(f"  Clamp factor: {args.norm_clamp_factor}\n")
        if args.normalization == 'local' and args.norm_percentile:
            f.write(f"  Percentile: {args.norm_percentile}\n")
        if args.norm_max_update_threshold:
            f.write(f"  Max update threshold: {args.norm_max_update_threshold}\n")
        f.write(f"Virtual ratio: {args.virtual_ratio}\n")
        f.write(f"Momentum: {'enabled' if args.use_momentum else 'disabled'}\n")
        if args.use_momentum:
            f.write(f"  Initial momentum: {args.momentum_init}\n")
        f.write(f"GPU enabled: {args.use_gpu}\n")
        
        # Add Ray configuration if present
        if args.use_ray and args.processing_method in ['batch', 'colors']:
            f.write(f"\nRay Multi-GPU Configuration:\n")
            f.write(f"  Ray enabled: True\n")
            f.write(f"  Number of GPUs: {args.ray_num_gpus or 'auto-detect'}\n")
            f.write(f"  Collective group: {args.ray_collective_group}\n")
        
        f.write("\n")
        
        # Training Results
        f.write("Training Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training time: {train_time:.2f}s\n")
        f.write(f"Iterations completed: {training_stats['iterations_completed']}\n")
        f.write(f"Total samples processed: {training_stats['total_samples_processed']:,}\n")
        f.write(f"Final weights norm: {training_stats['final_weights_norm']:.6f}\n")
        if 'quantization_error' in training_stats:
            f.write(f"Quantization error: {training_stats['quantization_error']:.6f}\n")
        f.write("\n")
        
        # Performance Summary
        f.write("Performance Summary:\n")
        f.write("-" * 40 + "\n")
        samples_per_sec = training_stats['total_samples_processed'] / train_time
        f.write(f"Samples per second: {samples_per_sec:,.0f}\n")
        f.write(f"Iterations per second: {training_stats['iterations_completed'] / train_time:.2f}\n")
        avg_time_per_iter = train_time / training_stats['iterations_completed']
        f.write(f"Average time per iteration: {avg_time_per_iter:.3f}s\n")
        
        # Memory usage estimate
        dataset_size_gb = metadata['shape'][0] * metadata['shape'][1] * 4 / (1024**3)
        f.write(f"\nDataset size (uncompressed): {dataset_size_gb:.2f} GB\n")


def main():
    """Main function for running large dataset benchmarks"""
    args = parse_args()
    
    print("=" * 80)
    print("FloatSOM Large Dataset Benchmark")
    print("=" * 80)
    
    # Create output directory with timestamp
    output_dir = f"large_dataset_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate or load data
    zarr_path, metadata = generate_or_load_data(args)
    
    # Setup profiling if requested (after data loading)
    profiler_instance = None
    if args.profile:
        print(f"\nProfiling enabled. Results will be saved to: {output_dir}/{args.profile_output}")
        profiler_instance = cProfile.Profile()
        profiler_instance.enable()
    
    # Train FloatSOM
    som, training_stats, train_time = train_floatsom_with_zarr(zarr_path, args, metadata)
    
    # Cleanup Ray resources if used
    if args.processing_method == 'batch' and args.use_ray and hasattr(som, 'processor'):
        if hasattr(som.processor, 'cleanup'):
            som.processor.cleanup()
    
    # Stop profiling before quantization error calculation
    if args.profile and profiler_instance:
        profiler_instance.disable()
    
    # Calculate quantization error (not profiled)
    if args.calculate_qe:
        qe = calculate_quantization_error(zarr_path, som, args, output_dir)
        training_stats['quantization_error'] = qe
    
    # Save profiling results if enabled
    if args.profile and profiler_instance:
        profile_output = os.path.join(output_dir, args.profile_output)
        with open(profile_output, 'w') as f:
            s = io.StringIO()
            ps = pstats.Stats(profiler_instance, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            f.write(s.getvalue())
        print(f"\nProfiling results saved to: {profile_output}")
    
    # Write results report to output directory
    output_file = os.path.join(output_dir, args.output_file)
    write_results_report(som, training_stats, train_time, metadata, args, output_file)
    
    print(f"\nBenchmark complete!")
    print(f"Results saved to directory: {output_dir}/")
    print(f"  - {args.output_file}: Main benchmark results")
    print(f"  - quantization_error.txt: Detailed QE analysis")
    if args.profile:
        print(f"  - {args.profile_output}: Profiling statistics")
    print(f"\nSummary:")
    print(f"  Dataset: {metadata['shape'][0]:,} × {metadata['shape'][1]}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Samples/sec: {training_stats['total_samples_processed'] / train_time:,.0f}")


if __name__ == "__main__":
    # Set number of threads to avoid OMP conflicts
    os.environ["OMP_NUM_THREADS"] = "1"
    
    main()
