#!/usr/bin/env python3
"""
Benchmarking utilities for the FloatSOM framework.

This module provides comprehensive benchmarking capabilities for the FloatSOM framework,
allowing for comparison of different sampling, processing, and topology combinations.
"""

import os
import sys
import argparse
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import FloatSOM components
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig
from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.topology.topology_factory import create_topology
from floatsom.sampling.base_selector import SampleSelector
from floatsom.sampling.full_selector import FullSelector
from floatsom.sampling.random_selector import RandomSelector
from floatsom.sampling.hdsssom_selector import HDSSSOMSelector
from floatsom.processing.base_processor import ProcessingMethod
from floatsom.processing.colors_processor import ColorsProcessor
from floatsom.processing.batch_processor import BatchProcessor

# Import evaluation metrics from sklearn_datasets
from floatsom_benchmarks.evaluation.sklearn_datasets import (
    generate_sklearn_dataset,
    get_available_datasets,
    get_dataset_info,
    SKLEARN_AVAILABLE
)

# Import standardized quantization error calculation
from floatsom_benchmarks.evaluation.metrics import QuantizationError

# Import visualization utilities
from floatsom_benchmarks.visualization import (
    create_output_directory,
    visualize_som_grid_overlay,
    plot_quantization_error,
    plot_combined_quantization_errors,
    create_cluster_visualization,
    visualize_results,
    create_comparison_visualizations,
    create_animation_from_iterations,
    plot_weight_statistics
)

# Data generation functions (preserved from original)
def generate_2d_test_data(n_points: int = 5000, seed: Optional[int] = None) -> cp.ndarray:
    """Generate 2D test data with a W-shaped distribution"""
    if seed is not None:
        np.random.seed(seed)
    
    # W-shaped distribution
    # Create five segments to form a W shape
    n_points_per_segment = n_points // 5
    
    # First segment (left side of W)
    angles1 = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4  # 0-45 degrees
    radius1 = np.random.normal(10, 1, n_points_per_segment)  # Add some noise to radius
    x1 = np.cos(angles1) * radius1
    y1 = np.sin(angles1) * radius1
    
    # Second segment (first middle of W)
    angles2 = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4 + np.pi / 4  # 45-90 degrees
    radius2 = np.random.normal(8, 1, n_points_per_segment)  # Shorter radius for middle
    x2 = np.cos(angles2) * radius2
    y2 = np.sin(angles2) * radius2
    
    # Third segment (bottom of W)
    angles3 = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4 + np.pi / 2  # 90-135 degrees
    radius3 = np.random.normal(6, 1, n_points_per_segment)  # Shortest radius for bottom
    x3 = np.cos(angles3) * radius3
    y3 = np.sin(angles3) * radius3
    
    # Fourth segment (second middle of W)
    angles4 = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4 + 3 * np.pi / 4  # 135-180 degrees
    radius4 = np.random.normal(8, 1, n_points_per_segment)  # Shorter radius for middle
    x4 = np.cos(angles4) * radius4
    y4 = np.sin(angles4) * radius4
    
    # Fifth segment (right side of W)
    angles5 = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4 + np.pi  # 180-225 degrees
    radius5 = np.random.normal(10, 1, n_points_per_segment)  # Add some noise to radius
    x5 = np.cos(angles5) * radius5
    y5 = np.sin(angles5) * radius5
    
    # Combine all segments
    x = np.concatenate([x1, x2, x3, x4, x5])
    y = np.concatenate([y1, y2, y3, y4, y5])
    
    # Add some noise clusters
    noise_x = np.random.normal(0, 5, n_points // 10)
    noise_y = np.random.normal(0, 5, n_points // 10)
    
    x = np.concatenate([x, noise_x])
    y = np.concatenate([y, noise_y])
    
    # Combine into 2D array and normalize
    data = np.vstack([x, y]).T
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return cp.array(data, dtype=cp.float32)

def generate_3d_test_data(n_points: int = 5000, seed: Optional[int] = None) -> cp.ndarray:
    """Generate 3D test data similar to a bowl-shaped distribution"""
    if seed is not None:
        np.random.seed(seed)
    
    # Bowl-shaped distribution 
    theta = np.random.uniform(0, 1, n_points) * 2 * np.pi  # Full circle
    phi = np.random.uniform(0, 1, n_points) * np.pi / 3  # Limited angle for bowl shape
    radius = np.random.normal(10, 1, n_points)  # Add some noise to radius
    
    # Convert spherical to cartesian
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    
    # Add some noise clusters
    noise_x = np.random.normal(0, 5, n_points // 10)
    noise_y = np.random.normal(0, 5, n_points // 10)
    noise_z = np.random.normal(0, 5, n_points // 10)
    
    x = np.concatenate([x, noise_x])
    y = np.concatenate([y, noise_y])
    z = np.concatenate([z, noise_z])
    
    # Combine into 3D array and normalize
    data = np.vstack([x, y, z]).T
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return cp.array(data, dtype=cp.float32)

def generate_test_data(n_samples: int, input_dim: int, n_clusters: int = 5, verbose: bool = True) -> cp.ndarray:
    """
    Generate test data with clusters for SOM training
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Dimensionality of the data
        n_clusters: Number of clusters in the data
        verbose: Whether to show detailed output
        
    Returns:
        data: Generated test data
    """
    if verbose:
        print(f"Generating {n_samples} samples with dimension {input_dim} in {n_clusters} clusters...")
    else:
        print(f"Generating test data...")
    
    # Generate centroids
    centroids = np.random.randn(n_clusters, input_dim)
    
    # Assign each sample to a random cluster
    cluster_assignments = np.random.randint(0, n_clusters, size=n_samples)
    
    # Generate data points around centroids
    data = np.zeros((n_samples, input_dim))
    for i in range(n_samples):
        centroid_idx = cluster_assignments[i]
        data[i] = centroids[centroid_idx] + np.random.randn(input_dim) * 0.5
    
    return cp.array(data, dtype=cp.float32)

class IterationCallback:
    """
    Callback class for tracking SOM weights during training iterations
    
    This class stores intermediate weights and can visualize how the SOM
    nodes move throughout training.
    """
    
    def __init__(self, data: cp.ndarray, output_dir: str, config_name: str, 
                 save_every: int = 10):
        """
        Initialize the callback
        
        Args:
            data: Training data
            output_dir: Directory to save visualizations
            config_name: Name of this configuration
            save_every: Save weights every N iterations
        """
        self.data = data
        self.output_dir = output_dir
        self.config_name = config_name
        self.save_every = save_every
        
        # Storage for metrics
        self.quant_errors = []
        self.iteration_times = []
        self.weight_stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'range': [],
            'relative_spread': []
        }
        
        # Initialize quantization error calculator
        self.qe_calculator = QuantizationError(use_optimized=True)
        
        # Create iterations directory
        self.iter_dir = os.path.join(output_dir, f"{config_name}_iterations")
        os.makedirs(self.iter_dir, exist_ok=True)
        
        # Track elapsed time
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def on_iteration_end(self, iteration: int, weights: cp.ndarray, som: Any, last_iteration: bool = False):
        """
        Called at the end of each iteration
        
        Args:
            iteration: Current iteration number
            weights: Current SOM weights
            som: The SOM being trained
            last_iteration: Whether this is the last iteration
        """
        # Track elapsed time for this iteration
        current_time = time.time()
        iter_time = current_time - self.last_time
        self.last_time = current_time
        self.iteration_times.append(iter_time)
        
        # Save weights and metrics periodically
        if (iteration % self.save_every == 0) or last_iteration:
            # Convert weights to NumPy if needed
            weights_np = weights.get() if hasattr(weights, 'get') else weights
            
            # Calculate quantization error using standardized implementation
            quant_error = self.qe_calculator.compute(som, self.data)
            self.quant_errors.append(quant_error)
            
            # Calculate BMU distribution
            bmu_counts = self._calculate_bmu_counts(som)
            
            # Store weight statistics
            self._store_weight_statistics(weights_np)
            
            # Create visualization
            self._save_iteration_visualization(iteration, weights_np, bmu_counts, quant_error)
    
    def _calculate_bmu_counts(self, som: Any) -> np.ndarray:
        """
        Calculate the distribution of samples across BMUs
        
        Args:
            som: The SOM being trained
            
        Returns:
            bmu_counts: Array with counts of samples per node
        """
        # Map all data points to their BMUs
        all_bmus, _ = som.map_vectors(self.data)
        all_bmus_np = all_bmus.get() if hasattr(all_bmus, 'get') else all_bmus
        
        # Count occurrences of each BMU
        total_nodes = som.get_weights().shape[0]
        bmu_counts = np.zeros(total_nodes)
        
        # Use numpy's histogram function to count occurrences efficiently
        counts, _ = np.histogram(all_bmus_np, bins=np.arange(total_nodes+1))
        bmu_counts[:len(counts)] = counts
        
        return bmu_counts
    
    def _store_weight_statistics(self, weights: np.ndarray):
        """Store weight statistics for later visualization"""
        self.weight_stats['mean'].append(np.mean(weights))
        self.weight_stats['std'].append(np.std(weights))
        self.weight_stats['min'].append(np.min(weights))
        self.weight_stats['max'].append(np.max(weights))
        weight_range = np.max(weights) - np.min(weights)
        self.weight_stats['range'].append(weight_range)
        
        # Calculate relative spread vs data
        data_np = self.data.get() if hasattr(self.data, 'get') else self.data
        data_range = np.max(data_np) - np.min(data_np)
        relative_spread = weight_range / data_range if data_range > 0 else 0
        self.weight_stats['relative_spread'].append(relative_spread)
    
    def _save_iteration_visualization(self, iteration: int, weights: np.ndarray, 
                                    bmu_counts: np.ndarray, quant_error: float):
        """
        Save visualization for the current iteration
        
        Args:
            iteration: Current iteration number
            weights: Current SOM weights
            bmu_counts: Counts of samples per node
            quant_error: Current quantization error
        """
        # Calculate and print weight statistics
        weight_min = np.min(weights)
        weight_max = np.max(weights)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        weight_range = weight_max - weight_min
        
        # Calculate data statistics for comparison
        data_np = self.data.get() if hasattr(self.data, 'get') else self.data
        data_min = np.min(data_np)
        data_max = np.max(data_np)
        data_range = data_max - data_min
        
        # Calculate weight spread relative to data spread
        relative_spread = weight_range / data_range if data_range > 0 else 0
        
        print(f"=== ITERATION {iteration:04d} WEIGHT STATISTICS ===")
        print(f"Weight Range: [{weight_min:.6f}, {weight_max:.6f}] (span: {weight_range:.6f})")
        print(f"Weight Mean±Std: {weight_mean:.6f} ± {weight_std:.6f}")
        print(f"Data Range: [{data_min:.6f}, {data_max:.6f}] (span: {data_range:.6f})")
        print(f"Weight span / Data span: {relative_spread:.4f} ({relative_spread*100:.1f}%)")
        if relative_spread < 0.1:
            print(f"WARNING: Weights span only {relative_spread*100:.1f}% of data range - VERY COMPRESSED!")
        print(f"Quantization Error: {quant_error:.6f}")
        print(f"=" * 50)
        
        # Create visualization with BMU counts for node sizing
        output_file = os.path.join(self.iter_dir, f"iteration_{iteration:04d}.png")
        
        # Create grid overlay visualization with enhanced title showing weight stats
        title = (f"{self.config_name} - Iteration {iteration}\n"
                f"Weight Range: [{weight_min:.3f}, {weight_max:.3f}] "
                f"({relative_spread*100:.1f}% of data range) | QE: {quant_error:.6f}")
        
        visualize_som_grid_overlay(
            data_np,
            weights,
            title,
            output_file
        )
    
    def finalize(self):
        """Finalize visualizations and metrics"""
        # Plot quantization error over time
        plot_quantization_error(
            self.quant_errors,
            f"{self.config_name} - Quantization Error",
            os.path.join(self.output_dir, f"{self.config_name}_quant_error_curve.png")
        )
        
        # Plot iteration times
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_times, marker='o')
        plt.title(f"{self.config_name} - Iteration Times")
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(self.output_dir, f"{self.config_name}_iteration_times.png"))
        plt.close()
        
        # Plot weight statistics
        plot_weight_statistics(
            self.weight_stats,
            f"{self.config_name} - Weight Statistics",
            os.path.join(self.output_dir, f"{self.config_name}_weight_stats.png")
        )
        
        # Create animation if possible
        create_animation_from_iterations(self.output_dir, self.config_name)

def train_floatsom(
    config_name: str,
    sampling_method: str,
    processing_method: str,
    topology_type: str,
    data: cp.ndarray,
    params: FloatSOMParams,
    output_dir: str = None,
    save_iterations: bool = True,
    save_every: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train a FloatSOM with the specified parameters
    
    Args:
        config_name: Name of this config for logging and output
        sampling_method: Sampling method ('full', 'random', 'hdsssom')
        processing_method: Processing method ('colors', 'batch')
        topology_type: Type of topology ('grid', 'hexagonal', 'mst')
        data: Input data for training
        params: FloatSOMParams configuration
        output_dir: Directory for output files
        save_iterations: Whether to save intermediate weights during training
        save_every: Save weights every N iterations
        verbose: Whether to print verbose output
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_output_directory(f"{sampling_method}_{processing_method}_{topology_type}")
    
    # Create config directory within output dir
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Update params with the specified methods
    params.sampling_config.method = sampling_method
    params.processing_config.method = processing_method
    params.topology_config.topology_type = topology_type
    
    # Set input dimension from data
    params.input_dim = data.shape[1]
    
    # Create topology
    topology = create_topology(params.topology_config, data)
    
    # Create sample selector
    if sampling_method == 'full':
        selector = FullSelector(params)
    elif sampling_method == 'random':
        selector = RandomSelector(params)
    elif sampling_method == 'hdsssom':
        selector = HDSSSOMSelector(params, data)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Create processor
    if processing_method == 'colors':
        processor = ColorsProcessor(params, topology)
    elif processing_method == 'batch':
        # Extract batch config from processing config
        batch_config = {
            'batch_mode': params.processing_config.batch_mode,
            'minibatch_size': params.processing_config.minibatch_size,
            'chunk_size': params.processing_config.chunk_size
        }
        processor = BatchProcessor(batch_config)
    else:
        raise ValueError(f"Unknown processing method: {processing_method}")
    
    # Create FloatSOM
    som = FloatSOM(selector, processor, topology, params)
    
    if verbose:
        print(f"\nTraining {config_name}")
        print(f"Architecture: {params.get_architecture_summary()}")
        print(f"Grid size: {params.topology_config.grid_size}, Input dimensions: {params.input_dim}")
        print(f"Learning rate: {params.initial_learning_rate}, Iterations: {params.total_iterations}")
    
    # Train the SOM with timing
    start_time = time.time()
    
    # Enable history tracking if save_iterations is enabled
    if save_iterations:
        params.store_history = True
    
    # Train the SOM
    training_stats = som.train(data)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get final weights
    final_weights = som.get_weights()
    
    # Calculate quantization error using standardized implementation
    qe_calculator = QuantizationError(use_optimized=True)
    quant_error = qe_calculator.compute(som, data)
    
    # Log final results
    if verbose:
        print(f"Training completed in {total_time:.3f}s")
        print(f"Final quantization error: {quant_error:.6f}")
    
    # Save final visualization
    data_np = data.get() if hasattr(data, 'get') else data
    weights_np = final_weights.get() if hasattr(final_weights, 'get') else final_weights
    
    # Create grid overlay visualization
    mst_edges = None
    if topology_type == 'mst' and hasattr(topology, 'mst_edges'):
        mst_edges = topology.mst_edges
    
    visualize_som_grid_overlay(
        data_np,
        weights_np,
        f"{config_name} - Final SOM Weights",
        os.path.join(config_dir, f"{config_name}_final_weights.png"),
        mst_edges=mst_edges
    )
    
    # If we have training history, create visualizations from it
    if save_iterations and hasattr(som, 'training_history') and som.training_history:
        # Create iteration plots from training history
        iterations = [h['iteration'] for h in som.training_history]
        weight_changes = [h['weight_change'] for h in som.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, weight_changes, marker='o')
        plt.title(f"{config_name} - Weight Change Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Weight Change")
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(config_dir, f"{config_name}_weight_changes.png"))
        plt.close()
    
    # Return results dictionary
    quant_errors = [quant_error]  # Just the final error since we don't have iteration callbacks
    
    return {
        'config_name': config_name,
        'sampling': sampling_method,
        'processing': processing_method,
        'topology': topology_type,
        'som': som,
        'weights': weights_np,
        'train_time': total_time,
        'quant_error': quant_error,
        'quant_errors': quant_errors,
        'params': params
    }

def run_benchmark(
    data: cp.ndarray,
    params: FloatSOMParams,
    sampling_methods: List[str] = None,
    processing_methods: List[str] = None,
    topology_types: List[str] = None,
    combinations: List[str] = None,
    verbose: bool = False,
    save_iterations: bool = True,
    save_every: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks for FloatSOM using different method combinations
    
    Args:
        data: Input data for training
        params: Base FloatSOM parameters (will be modified for each config)
        sampling_methods: List of sampling methods to benchmark
        processing_methods: List of processing methods to benchmark
        topology_types: List of topology types to benchmark
        combinations: List of specific combinations (e.g., "full:colors:grid")
        verbose: Whether to print verbose output
        save_iterations: Whether to save intermediate weights during training
        save_every: Save weights every N iterations
    """
    # Create output directory
    output_dir = create_output_directory("floatsom_benchmark")
    
    # Set defaults if not provided
    if sampling_methods is None:
        sampling_methods = ["full", "random", "hdsssom"]
    
    if processing_methods is None:
        processing_methods = ["colors", "batch", "minisom"]
    
    if topology_types is None:
        topology_types = ["grid", "hexagonal", "mst"]
    
    # Define configurations to test
    configs = []
    
    # If combinations are provided, parse them
    if combinations:
        for combo in combinations:
            try:
                sampling, processing, topology = combo.split(":")
                config_name = f"{sampling}_{processing}_{topology}"
                configs.append((config_name, sampling, processing, topology))
            except ValueError:
                print(f"Invalid combination format: {combo}. Expected format: 'sampling:processing:topology'")
    else:
        # Generate all combinations
        for sampling in sampling_methods:
            for processing in processing_methods:
                for topology in topology_types:
                    config_name = f"{sampling}_{processing}_{topology}"
                    configs.append((config_name, sampling, processing, topology))
    
    if verbose:
        print(f"Running {len(configs)} configurations for benchmark:")
        for config_name, sampling, processing, topology in configs:
            print(f"  - {config_name} ({sampling}×{processing}×{topology})")
    
    # Train each configuration
    results = {}
    
    for config_name, sampling, processing, topology in configs:
        # Create a copy of params for this configuration
        config_params = FloatSOMParams(
            defaults_profile="publication",
            input_dim=params.input_dim,
            total_iterations=params.total_iterations,
            initial_learning_rate=params.initial_learning_rate,
            initial_radius=params.initial_radius,
            final_radius=params.final_radius,
            decay_type=params.decay_type,
            lr_decay_factor=params.lr_decay_factor,
            radius_decay_factor=params.radius_decay_factor,
            radius_warmup_iters=params.radius_warmup_iters,
            sampling_config=SamplingConfig(method=sampling),
            processing_config=ProcessingConfig(
                method=processing,
                batch_mode=params.processing_config.batch_mode,
                minibatch_size=params.processing_config.minibatch_size,
                chunk_size=params.processing_config.chunk_size
            ),
            topology_config=TopologyConfig(
                topology_type=topology,
                grid_size=params.topology_config.grid_size,
                grid_dim=params.topology_config.grid_dim
            ),
            initialization_method=params.initialization_method,
            seed=params.seed,
            verbose=params.verbose,
            use_gpu=params.use_gpu,
        )
        
        result = train_floatsom(
            config_name=config_name,
            sampling_method=sampling,
            processing_method=processing,
            topology_type=topology,
            data=data,
            params=config_params,
            output_dir=output_dir,
            save_iterations=save_iterations,
            save_every=save_every,
            verbose=verbose
        )
        
        results[config_name] = result
    
    # Create comparison visualizations
    create_comparison_visualizations(results, data, output_dir, "floatsom")
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark FloatSOM with different configurations')
    
    # Data generation parameters
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Size of the SOM grid (grid_size × grid_size for grid topologies)')
    
    parser.add_argument('--grid_dim', type=int, default=2, choices=[1, 2],
                        help='Dimensionality of the SOM grid (1 for 1D chain, 2 for 2D grid)')
    
    parser.add_argument('--input_dim', type=int, default=None,
                        help='Input dimension for generated data (if not loading from a file)')
    
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to generate for test data')
    
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters in generated test data')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Initial learning rate')
    
    parser.add_argument('--initial_radius', type=float, default=None,
                        help='Initial neighborhood radius (if None, will be calculated based on topology)')
    
    parser.add_argument('--decay_type', type=str, default='asymptotic',
                        choices=['asymptotic', 'exponential', 'linear', 'sigmoid', 'gaussian', 'fixed'],
                        help='Type of decay for learning rate and neighborhood radius')
    
    parser.add_argument('--radius_decay_factor', type=float, default=0.5,
                        help='Factor to slow down radius decay (0-1, default 0.5 = 2x slower)')
    
    parser.add_argument('--radius_warmup_iters', type=int, default=0,
                        help='Number of iterations to keep radius constant before decay')
    
    parser.add_argument('--lr_decay_factor', type=float, default=8.0,
                        help='Learning rate decay factor (higher = slower decay)')
    
    # Batch processing parameters
    parser.add_argument('--batch_mode', type=str, default='full_batch',
                        choices=['full_batch', 'minibatch'],
                        help='Batch processing mode')
    
    parser.add_argument('--minibatch_size', type=int, default=32,
                        help='Size of minibatches for minibatch mode')
    
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk size for memory-efficient processing (None for auto-calculate based on GPU memory)')
    
    # Configuration parameters
    parser.add_argument('--sampling', type=str, nargs='+',
                        help='List of sampling methods to benchmark (e.g., "full random hdsssom")')
    
    parser.add_argument('--processing', type=str, nargs='+',
                        help='List of processing methods to benchmark (e.g., "colors batch")')
    
    parser.add_argument('--topology', type=str, nargs='+',
                        help='List of topology types to benchmark (e.g., "grid hexagonal mst")')
    
    parser.add_argument('--combinations', type=str, nargs='+',
                        help='Specific combinations to benchmark (e.g., "full:colors:grid random:batch:hexagonal")')
    
    # Visualization parameters
    parser.add_argument('--save_iterations', action='store_true', default=True,
                        help='Save intermediate weights during training')
    
    parser.add_argument('--no_save_iterations', action='store_false', dest='save_iterations',
                        help='Do not save intermediate weights during training')
    
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save weights every N iterations')
    
    # Other parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['swiss_roll', 'moons', 'circles', 'blobs', 's_curve'],
                        help='Use sklearn toy dataset instead of generated data')
    
    parser.add_argument('--dataset_difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for sklearn dataset')
    
    return parser.parse_args()

def main():
    """Main function for running FloatSOM benchmarks"""
    # Parse arguments
    args = parse_args()
    
    # Generate or load test data
    if args.dataset and SKLEARN_AVAILABLE:
        # Use sklearn toy dataset
        print(f"Loading {args.dataset} dataset (difficulty: {args.dataset_difficulty})...")
        data, metadata = generate_sklearn_dataset(
            args.dataset, 
            difficulty=args.dataset_difficulty,
            seed=args.seed,
            normalize=True
        )
        input_dim = data.shape[1]
        if args.verbose:
            print(f"Dataset info: {metadata}")
    elif args.input_dim is None:
        # Use 2D data if input_dim not specified
        print("Generating 2D test data...")
        data = generate_2d_test_data(n_points=args.samples, seed=args.seed)
        input_dim = 2
    else:
        # Generate test data with specified dimensions
        data = generate_test_data(
            n_samples=args.samples,
            input_dim=args.input_dim,
            n_clusters=args.clusters,
            verbose=args.verbose
        )
        input_dim = args.input_dim
    
    # Create base FloatSOM parameters
    params = FloatSOMParams(
        defaults_profile="publication",
        input_dim=input_dim,
        total_iterations=args.iterations,
        initial_learning_rate=args.learning_rate,
        initial_radius=args.initial_radius,
        decay_type=args.decay_type,
        lr_decay_factor=args.lr_decay_factor,
        radius_decay_factor=args.radius_decay_factor,
        radius_warmup_iters=args.radius_warmup_iters,
        processing_config=ProcessingConfig(
            batch_mode=args.batch_mode,
            minibatch_size=args.minibatch_size,
            chunk_size=args.chunk_size
        ),
        topology_config=TopologyConfig(
            grid_size=args.grid_size,
            grid_dim=args.grid_dim
        ),
        initialization_method='pca',
        seed=args.seed,
        verbose=args.verbose,
    )
    
    # Process method options
    sampling_methods = None
    processing_methods = None
    topology_types = None
    combinations = None
    
    # Handle 'all' in methods
    if args.sampling and 'all' in args.sampling:
        sampling_methods = ['full', 'random', 'hdsssom']
    elif args.sampling:
        sampling_methods = args.sampling
    
    if args.processing and 'all' in args.processing:
        processing_methods = ['colors', 'batch', 'minisom']
    elif args.processing:
        processing_methods = args.processing
    
    if args.topology and 'all' in args.topology:
        topology_types = ['grid', 'hexagonal', 'mst']
    elif args.topology:
        topology_types = args.topology
    
    # Process combinations if provided
    if args.combinations:
        combinations = args.combinations
        if args.verbose:
            print(f"Using specific combinations: {combinations}")
    
    # Run benchmarks
    results = run_benchmark(
        data=data,
        params=params,
        sampling_methods=sampling_methods,
        processing_methods=processing_methods,
        topology_types=topology_types,
        combinations=combinations,
        verbose=args.verbose,
        save_iterations=args.save_iterations,
        save_every=args.save_every
    )
    
    print("Benchmarking complete")

if __name__ == "__main__":
    # Set number of threads to avoid OMP conflicts
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Run benchmarks
    main()
