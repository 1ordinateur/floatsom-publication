"""
Training functions for MST comparison benchmarks.

This module contains functions for training different SOM topologies:
- Direct MST-SOM training
- Hexagonal grid SOM training
- Hexagonal to MST conversion
"""

import time
import numpy as np
import cupy as cp
import logging
from typing import Dict, List, Tuple, Any

from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig
from floatsom.processing.processing_params import calculate_auto_chunk_size_for_method
from floatsom.topology.mst_topology import MSTTopology, calculate_mst_cpu, build_adjacency_list

logger = logging.getLogger(__name__)


def train_direct_mst_som(data: cp.ndarray, args) -> Tuple[FloatSOM, Dict[str, Any], float]:
    """Train SOM with direct MST topology."""
    n_samples, input_dim = data.shape
    
    # Calculate optimal chunk size if not provided
    if args.chunk_size is None:
        optimal_chunk_size = calculate_auto_chunk_size_for_method(
            input_dim, args.processing_method
        )
        optimal_chunk_size = min(optimal_chunk_size, 500_000)
    else:
        optimal_chunk_size = args.chunk_size
    
    # Create parameters
    params = FloatSOMParams(
        input_dim=input_dim,
        total_iterations=args.iterations,
        initial_learning_rate=args.learning_rate,
        lr_decay_type=args.lr_decay_type,
        radius_decay_type=args.radius_decay_type,
        lr_decay_factor=args.lr_decay_factor,
        radius_decay_factor=args.radius_decay_factor,
        initial_radius=args.initial_radius,
        convergence_threshold=args.convergence_threshold,
        min_iterations=args.min_iterations,
        sampling_config=SamplingConfig(
            method=args.sampling_method,
            random_seed=args.seed
        ),
        processing_config=ProcessingConfig(
            method=args.processing_method,
            batch_mode=args.batch_mode,
            chunk_size=optimal_chunk_size,
            enable_momentum=args.use_momentum,
            initial_momentum=args.momentum_init,
            normalization=args.normalization,
            norm_alpha=args.norm_alpha,
            norm_clamp_factor=args.norm_clamp_factor,
            norm_percentile=args.norm_percentile,
            norm_max_update_threshold=args.norm_max_update_threshold,
            virtual_ratio=args.virtual_ratio,
            use_gpu=args.use_gpu
        ),
        topology_config=TopologyConfig(
            topology_type='mst',
            grid_size=args.grid_size,
            mst_update_frequency=args.mst_update_frequency,
            dynamic_mst_frequency=args.dynamic_mst_frequency,
            mst_decay_function=args.mst_decay_function,
            initial_mst_frequency=args.initial_mst_frequency,
            final_mst_frequency=args.final_mst_frequency
        ),
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        seed=args.seed,
        initialization_method=args.initialization_method
    )
    
    # Create FloatSOM with MST topology
    som = create_floatsom(params)
    
    if args.verbose:
        logger.info("Training Direct MST-SOM...")
    
    start_time = time.time()
    training_stats = som.train(data)
    train_time = time.time() - start_time
    
    return som, training_stats, train_time


def train_hexagonal_som(data: cp.ndarray, args) -> Tuple[FloatSOM, Dict[str, Any], float]:
    """Train SOM with hexagonal grid topology."""
    n_samples, input_dim = data.shape
    
    # Calculate optimal chunk size if not provided
    if args.chunk_size is None:
        optimal_chunk_size = calculate_auto_chunk_size_for_method(
            input_dim, args.processing_method
        )
        optimal_chunk_size = min(optimal_chunk_size, 500_000)
    else:
        optimal_chunk_size = args.chunk_size
    
    # Create parameters for hexagonal topology
    params = FloatSOMParams(
        input_dim=input_dim,
        total_iterations=args.iterations,
        initial_learning_rate=args.learning_rate,
        lr_decay_type=args.lr_decay_type,
        radius_decay_type=args.radius_decay_type,
        lr_decay_factor=args.lr_decay_factor,
        radius_decay_factor=args.radius_decay_factor,
        initial_radius=args.initial_radius,
        convergence_threshold=args.convergence_threshold,
        min_iterations=args.min_iterations,
        sampling_config=SamplingConfig(
            method=args.sampling_method,
            random_seed=args.seed
        ),
        processing_config=ProcessingConfig(
            method=args.processing_method,
            batch_mode=args.batch_mode,
            chunk_size=optimal_chunk_size,
            enable_momentum=args.use_momentum,
            initial_momentum=args.momentum_init,
            normalization=args.normalization,
            norm_alpha=args.norm_alpha,
            norm_clamp_factor=args.norm_clamp_factor,
            norm_percentile=args.norm_percentile,
            norm_max_update_threshold=args.norm_max_update_threshold,
            virtual_ratio=args.virtual_ratio,
            use_gpu=args.use_gpu
        ),
        topology_config=TopologyConfig(
            topology_type='hexagonal',
            grid_size=args.grid_size
        ),
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        seed=args.seed,
        initialization_method=args.initialization_method
    )
    
    # Create FloatSOM with hexagonal topology
    som = create_floatsom(params)
    
    if args.verbose:
        logger.info("Training Hexagonal SOM...")
    
    start_time = time.time()
    training_stats = som.train(data)
    train_time = time.time() - start_time
    
    return som, training_stats, train_time


def convert_hexagonal_to_mst(hexagonal_som: FloatSOM) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Convert trained hexagonal SOM to MST by computing MST on final weights.

    Returns:
        mst_edges: List of MST edges
        edge_weights: Array of edge weights in MST
    """
    # Get the weights from the trained hexagonal SOM
    weights = hexagonal_som.get_weights()

    # Convert to numpy if needed
    if hasattr(weights, 'get'):
        weights_np = weights.get()
    else:
        weights_np = weights

    # Calculate pairwise distance matrix (NumPy)
    n_nodes = weights_np.shape[0]
    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.linalg.norm(weights_np[i] - weights_np[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    # Calculate MST edges using CPU algorithm
    mst_edges = calculate_mst_cpu(distance_matrix)

    # Extract edge weights from the distance matrix
    edge_weights = np.array([distance_matrix[u, v] for u, v in mst_edges])

    return mst_edges, edge_weights


def get_mst_edge_weights(mst_som: FloatSOM) -> np.ndarray:
    """Extract edge weights from trained MST-SOM."""
    # Get the MST topology
    topology = mst_som.topology

    if not isinstance(topology, MSTTopology):
        raise ValueError("SOM does not have MST topology")

    # Get the weights and convert to NumPy if needed
    weights = mst_som.get_weights()
    weights_np = weights.get() if hasattr(weights, 'get') else weights

    # Get MST edges from topology; recalculate if missing
    mst_edges = getattr(topology, 'mst_edges', None)
    if not mst_edges:
        topology._update_mst(weights)
        mst_edges = topology.mst_edges

    # Calculate edge weights along the MST
    edge_weights = [np.linalg.norm(weights_np[u] - weights_np[v]) for u, v in mst_edges]
    return np.array(edge_weights)
