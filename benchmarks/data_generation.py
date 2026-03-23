#!/usr/bin/env python3
"""
Simple data generation utilities for FloatSOM benchmarks.
"""

import numpy as np
import cupy as cp
from typing import Optional


def generate_2d_test_data(n_points: int = 5000, seed: Optional[int] = None) -> cp.ndarray:
    """Generate 2D test data with a W-shaped distribution"""
    if seed is not None:
        np.random.seed(seed)
    
    # Simple W-shaped distribution
    n_points_per_segment = n_points // 5
    
    segments = []
    for i in range(5):
        # Create segments that form a W shape
        angle_start = i * np.pi / 4
        angles = np.random.uniform(0, 1, n_points_per_segment) * np.pi / 4 + angle_start
        radius = np.random.normal(8 + (i % 2) * 2, 1, n_points_per_segment)
        
        x = np.cos(angles) * radius
        y = np.sin(angles) * radius
        segments.append(np.vstack([x, y]).T)
    
    # Combine segments
    data = np.concatenate(segments, axis=0)
    
    # Add noise
    noise = np.random.normal(0, 1, (n_points // 10, 2))
    data = np.concatenate([data, noise], axis=0)
    
    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return cp.array(data, dtype=cp.float32)


def generate_3d_test_data(n_points: int = 5000, seed: Optional[int] = None) -> cp.ndarray:
    """Generate 3D test data with bowl shape"""
    if seed is not None:
        np.random.seed(seed)
    
    # Bowl-shaped distribution 
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi / 3, n_points)
    radius = np.random.normal(10, 1, n_points)
    
    # Convert to cartesian
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    
    data = np.vstack([x, y, z]).T
    
    # Add noise
    noise = np.random.normal(0, 1, (n_points // 10, 3))
    data = np.concatenate([data, noise], axis=0)
    
    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return cp.array(data, dtype=cp.float32)


def generate_test_data(n_samples: int, input_dim: int, n_clusters: int = 5, verbose: bool = True) -> cp.ndarray:
    """Generate clustered test data"""
    if verbose:
        print(f"Generating {n_samples} samples with dimension {input_dim} in {n_clusters} clusters...")
    
    # Generate centroids
    centroids = np.random.randn(n_clusters, input_dim)
    
    # Assign samples to clusters
    cluster_assignments = np.random.randint(0, n_clusters, size=n_samples)
    
    # Generate data points
    data = np.zeros((n_samples, input_dim))
    for i in range(n_samples):
        centroid_idx = cluster_assignments[i]
        data[i] = centroids[centroid_idx] + np.random.randn(input_dim) * 0.5
    
    return cp.array(data, dtype=cp.float32)


def generate_random_data(n_samples: int, input_dim: int, seed: Optional[int] = None, verbose: bool = True,
                        memmap_file: Optional[str] = None) -> np.memmap:
    """Generate random data with uniform distribution [0, 1] using memory-mapped files
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Dimensionality of each sample
        seed: Random seed for reproducibility
        verbose: Whether to print progress messages
        memmap_file: Path to memory-mapped file (if None, returns regular array for small datasets)
    
    Returns:
        Memory-mapped numpy array or regular numpy array (not moved to GPU to save memory)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        print(f"Generating {n_samples} random samples with dimension {input_dim}...")
    
    # Calculate size in bytes (float32 = 4 bytes)
    size_bytes = n_samples * input_dim * 4
    
    # Use memmap for datasets > 100MB or if explicitly requested
    if memmap_file or size_bytes > 100 * 1024 * 1024:
        import tempfile
        import os
        
        if memmap_file is None:
            # Create temporary file for memmap
            temp_dir = os.path.join(os.getcwd(), 'dataset_cache', 'memmap_temp')
            os.makedirs(temp_dir, exist_ok=True)
            memmap_file = os.path.join(temp_dir, f'random_data_{n_samples}_{input_dim}_{seed}.dat')
            
        if verbose:
            print(f"Using memory-mapped file: {memmap_file}")
            print(f"Dataset size: {size_bytes / (1024**3):.2f} GB")
            
        # Create memory-mapped array
        data = np.memmap(memmap_file, dtype=np.float32, mode='w+', shape=(n_samples, input_dim))
        
        # Generate data in chunks to avoid memory issues
        chunk_size = min(10000, n_samples)  # Process 10k samples at a time
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            data[i:end_idx] = np.random.rand(end_idx - i, input_dim).astype(np.float32)
            if verbose and (i % (chunk_size * 10) == 0 or end_idx == n_samples):
                print(f"  Generated {end_idx}/{n_samples} samples ({100*end_idx/n_samples:.1f}%)")
        
        # Flush to disk
        data.flush()
        
        if verbose:
            print(f"Random data generation complete. File: {memmap_file}")
        
        return data
    else:
        # For small datasets, use regular numpy array but still return numpy (not cupy) to save GPU memory
        if verbose:
            print(f"Small dataset ({size_bytes / (1024**2):.1f} MB), using in-memory generation")
        data = np.random.rand(n_samples, input_dim).astype(np.float32)
        return data