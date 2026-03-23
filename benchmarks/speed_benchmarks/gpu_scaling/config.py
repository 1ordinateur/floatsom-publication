#!/usr/bin/env python3
"""
Configuration management for GPU scaling benchmarks.

This module handles benchmark configuration and parameter management.
Single Responsibility: Create and manage benchmark configurations.
"""

from types import SimpleNamespace
from typing import Optional

from floatsom.processing.processing_params import (
    calculate_auto_chunk_size_for_method,
    get_visible_gpu_vram_mib,
)


class BenchmarkConfig:
    """Encapsulate benchmark configuration parameters."""
    
    @staticmethod
    def calculate_chunk_size(dimension: int, processing_method: str) -> int:
        """
        Calculate optimal chunk size based on dimension and processing method.
        
        Args:
            dimension: Input dimension
            processing_method: Processing method name
            
        Returns:
            Calculated chunk size
        """
        return calculate_auto_chunk_size_for_method(dimension, processing_method)
    
    @staticmethod
    def create_benchmark_args(base_args, dimension: int, num_gpus: int, 
                             num_samples: int = None, seed: int = None,
                             processing_method: str = 'batch', topology_type: str = 'grid',
                             grid_size: int = None, total_iterations: int = None) -> SimpleNamespace:
        """
        Create an args object for a specific benchmark configuration.
        
        Args:
            base_args: Base arguments from command line
            dimension: Input dimension for this run
            num_gpus: Number of GPUs for this run
            num_samples: Number of samples (overrides base_args.samples if provided)
            seed: Random seed for this run (overrides base_args.seed if provided)
            processing_method: Processing method to use ('batch' or 'colors')
            topology_type: Type of topology to use ('grid', 'hexagonal', or 'mst')
            grid_size: Grid size for this run (overrides base_args.grid_size if provided)
            total_iterations: Number of iterations (overrides base_args.total_iterations if provided)
        
        Returns:
            SimpleNamespace object with all necessary arguments
        """
        # Use provided values or defaults
        samples = num_samples if num_samples is not None else base_args.samples
        seed_val = seed if seed is not None else base_args.seed
        actual_grid_size = grid_size if grid_size is not None else base_args.grid_size
        actual_total_iterations = (
            total_iterations if total_iterations is not None else base_args.total_iterations
        )
        if actual_total_iterations is None:
            raise ValueError("total_iterations must be specified for GPU scaling benchmarks")
        sampling_method = base_args.sampling_method
        sampling_fraction = base_args.sampling_fraction

        if sampling_fraction is not None:
            if not (0 < sampling_fraction <= 1.0):
                raise ValueError("sampling_fraction must be between 0 and 1 (exclusive of 0).")
        
        # Determine processing mode specifics
        is_minibatch = processing_method == 'minibatch'
        processing_method_arg = 'batch' if is_minibatch else processing_method
        batch_mode = 'minibatch' if is_minibatch else 'full_batch'

        cli_chunk_size = getattr(base_args, "chunk_size", None)
        chunk_size_visible_vram_mib = None
        chunk_size_source = "auto"

        # Resolve processing chunk size with explicit CLI override precedence.
        if cli_chunk_size is not None:
            chunk_size = int(cli_chunk_size)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer when provided")
            chunk_size_source = "manual_cli_chunk_size"
        elif is_minibatch:
            chunk_size = base_args.minibatch_chunk_size
            if chunk_size <= 0:
                raise ValueError("minibatch_chunk_size must be a positive integer")
            chunk_size_source = "manual_minibatch_chunk_size"
        else:
            chunk_size = BenchmarkConfig.calculate_chunk_size(dimension, processing_method)
            chunk_size_source = "auto_dimension_vram"
            chunk_size_visible_vram_mib = get_visible_gpu_vram_mib()

        zarr_chunk_size = getattr(base_args, "zarr_chunk_size", None)
        if zarr_chunk_size is None:
            zarr_chunk_size = int(chunk_size)
        else:
            zarr_chunk_size = int(zarr_chunk_size)
            if zarr_chunk_size <= 0:
                zarr_chunk_size = int(chunk_size)

        # Create a new namespace with all the parameters needed by train_floatsom_with_zarr
        args = SimpleNamespace(
            # Data parameters
            samples=samples,
            input_dim=dimension,
            cache_dir=base_args.cache_dir,
            zarr_chunk_size=zarr_chunk_size,
            force_regenerate=base_args.force_regenerate,
            chunk_size=chunk_size,
            temp_dir=base_args.temp_dir,
            run_temp_dir=base_args.run_temp_dir,
            ray_local_storage_path=getattr(base_args, "ray_local_storage_path", None),

            # SOM parameters
            grid_size=actual_grid_size,
            total_iterations=actual_total_iterations,
            initial_learning_rate=base_args.initial_learning_rate,
            initial_radius=None,  # Auto-calculate
            lr_decay_type='asymptotic',
            radius_decay_type='asymptotic',
            lr_decay_factor=8.0,
            radius_decay_factor=3.0,

            # Processing configuration
            processing_method=processing_method_arg,
            batch_mode=batch_mode,
            minibatch_size=32,
            use_momentum=True,
            initial_momentum=0.5,
            normalization='count_based',
            norm_alpha=None,
            norm_clamp_factor=None,
            norm_percentile=None,
            norm_max_update_threshold=None,
            virtual_ratio=0.5,
            
            # Sampling configuration
            sampling_method=sampling_method,
            sampling_fraction=sampling_fraction,
            target_proportion=sampling_fraction,
            
            # Topology configuration
            topology_type=topology_type,
            topology_variant='planar',
            initialization_method='random',
            mst_update_frequency=None,
            dynamic_mst_frequency=True,
            mst_decay_function='exponential',
            initial_mst_frequency=1,
            final_mst_frequency=10,
            
            # Convergence parameters
            convergence_threshold=1e-6,
            min_iterations=100,
            
            # Visualization parameters
            save_iterations=False,
            save_every=1,
            
            # Other parameters
            verbose=base_args.verbose,
            profile=getattr(base_args, "profile", False),
            profile_output=getattr(base_args, "profile_output", "profile_results.txt"),
            safe_cleanup=getattr(base_args, "safe_cleanup", False),
            profile_workers=getattr(base_args, "profile_workers", False),
            profile_workers_output_dir=getattr(base_args, "profile_workers_output_dir", None),
            profile_workers_max_stats=getattr(base_args, "profile_workers_max_stats", 50),
            seed=seed_val,
            use_gpu=True,
            output_file=f'results_{topology_type}_dim{dimension}_samples{samples}_grid{actual_grid_size}_gpu{num_gpus}_{processing_method}_seed{seed_val}.txt',

            # Ray multi-GPU parameters
            use_ray=bool(getattr(base_args, "use_ray", True)),
            ray_gpu_count=num_gpus,
            ray_collective_group='default',
            ray_collective_barriers=getattr(base_args, "ray_collective_barriers", False),
            force_disk_mode=getattr(base_args, "force_disk_mode", False),
            ray_chunk_size=None,  # Auto-calculate
            minibatch_chunk_size=base_args.minibatch_chunk_size,
            chunk_size_source=chunk_size_source,
            chunk_size_visible_vram_mib=chunk_size_visible_vram_mib,
        )

        return args
    
    @staticmethod
    def generate_config_name(dimension: int, num_gpus: int, num_samples: Optional[int],
                            topology_type: str, processing_method: str, 
                            grid_size: int, seed: int) -> str:
        """
        Generate a configuration name for logging and identification.
        
        Args:
            dimension: Input dimension
            num_gpus: Number of GPUs
            num_samples: Number of samples (optional)
            topology_type: Type of topology
            processing_method: Processing method
            grid_size: Grid size
            seed: Random seed
            
        Returns:
            Configuration name string
        """
        if num_samples is not None:
            return f"{topology_type}_dim{dimension}_samples{num_samples}_grid{grid_size}_gpu{num_gpus}_{processing_method}_seed{seed}"
        else:
            return f"{topology_type}_dim{dimension}_grid{grid_size}_gpu{num_gpus}_{processing_method}_seed{seed}"
