#!/usr/bin/env python3
"""
MiniSOM adapter for FloatSOM framework.

This module provides an adapter for the popular CPU-based MiniSOM implementation 
to work within the FloatSOM framework, allowing for consistent benchmarking
and comparison against the other implementations.
"""

import numpy as np
import cupy as cp
import logging
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

from floatsom.floatsom_params import FloatSOMParams


class SimpleTopology:
    """Simple topology class for compatibility with visualization code."""
    
    def __init__(self, topology_config):
        self.grid_size = topology_config.grid_size
        self.topology_type = topology_config.topology_type
        self.topology_variant = topology_config.topology_variant


class MiniSOMAdapter:
    """
    Adapter class for MiniSOM to work with FloatSOM framework.
    
    This adapter wraps the MiniSOM implementation to provide a consistent interface
    with other SOM implementations in the framework, allowing for fair benchmarking.
    """
    
    def __init__(self, params: FloatSOMParams, use_default_hyperparams: bool = False):
        """
        Initialize the MiniSOM adapter.
        
        Args:
            params: FloatSOM parameters
            use_default_hyperparams: If True, uses MiniSom's default hyperparameters 
                                    (only sets grid dimensions and topology)
        """
        if not MINISOM_AVAILABLE:
            raise ImportError("MiniSOM package not available. Please install it with 'pip install minisom'")
            
        self.params = params
        
        # Handle grid dimensions based on topology
        if params.topology_config.topology_type in ['grid', 'hexagonal']:
            x_dim = params.topology_config.grid_size
            y_dim = params.topology_config.grid_size
            minisom_topology = 'rectangular' if params.topology_config.topology_type == 'grid' else 'hexagonal'
        else:
            # For other topologies, use rectangular grid
            x_dim = params.topology_config.grid_size
            y_dim = params.topology_config.grid_size
            minisom_topology = 'rectangular'
        
        # Create MiniSOM instance
        if use_default_hyperparams:
            # Stock MiniSom with only grid dimensions and topology specified
            self.minisom = MiniSom(
                x=x_dim,
                y=y_dim,
                input_len=params.input_dim,
                topology=minisom_topology,
                random_seed=params.seed
            )
        else:
            # Enhanced parameter handling - use more FloatSOM parameters
            # Calculate sigma (neighborhood radius) from initial_radius if provided
            sigma = params.initial_radius if params.initial_radius else max(x_dim, y_dim) / 2.0
            
            # Map decay type to MiniSOM decay function
            # MiniSOM supports: inverse_decay_to_zero, linear_decay_to_zero, asymptotic_decay
            decay_function = 'asymptotic_decay'  # Default
            if params.lr_decay_type == 'exponential':
                decay_function = 'inverse_decay_to_zero'  # Closest to exponential
            elif params.lr_decay_type == 'linear':
                decay_function = 'linear_decay_to_zero'
            
            self.minisom = MiniSom(
                x=x_dim,
                y=y_dim,
                input_len=params.input_dim,
                sigma=sigma,  # Use initial_radius as sigma
                learning_rate=params.initial_learning_rate,
                decay_function=decay_function,  # Use mapped decay function
                neighborhood_function='gaussian',
                topology=minisom_topology,
                random_seed=params.seed
            )
        
        # Initialize weights based on initialization_method
        # MiniSOM will initialize weights internally, but we can control the method
        self.initialization_method = params.initialization_method
        
        # Initialize weights and coordinates
        self.weights = None
        self.training_history = []
        
        # Create a simple topology object for compatibility with visualization
        self.topology = SimpleTopology(params.topology_config)
    
    def train(self, dataset: Union[np.ndarray, cp.ndarray]) -> Dict[str, Any]:
        """
        Train the SOM using MiniSOM.
        
        Args:
            dataset: Training data
            
        Returns:
            Dictionary with training statistics
        """
        # Convert data to NumPy if it's a CuPy array
        data_np = dataset.get() if hasattr(dataset, 'get') else dataset
        
        # Initialize weights based on initialization_method
        if self.initialization_method == 'pca':
            # MiniSOM's PCA initialization
            self.minisom.pca_weights_init(data_np)
        elif self.initialization_method == 'random':
            # MiniSOM's random initialization (already done in constructor)
            self.minisom.random_weights_init(data_np)
        else:
            # Default to random if unknown method
            self.minisom.random_weights_init(data_np)
        
        # Train MiniSOM
        logger.info(f"Training MiniSOM for {self.params.total_iterations} iterations...")
        
        # Use MiniSOM's built-in training
        # Note: MiniSOM internally handles decay based on the decay_function we set
        self.minisom.train(data_np, self.params.total_iterations, verbose=self.params.verbose, use_epochs=True)
        
        # Update weights
        minisom_weights = np.array(self.minisom.get_weights())
        self.weights = minisom_weights.reshape(-1, self.params.input_dim)
        
        # Calculate final quantization error
        final_qe = self.minisom.quantization_error(data_np)
        
        if self.params.verbose:
            logger.info(f"MiniSOM training completed. Final QE: {final_qe:.6f}")
        
        # Return training statistics
        return {
            'iterations_completed': self.params.total_iterations,
            'total_samples_processed': len(data_np) * self.params.total_iterations,
            'final_weights_norm': final_qe,
            'convergence_detected': False
        }
    
    def predict(self, data: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """
        Map input vectors to their best matching units (BMUs).
        
        Args:
            data: Input data vectors
            
        Returns:
            Array of BMU indices for each input vector
        """
        # Convert data to NumPy if it's a CuPy array
        data_np = data.get() if hasattr(data, 'get') else data
        
        # Get BMUs using MiniSOM's winner function
        winners = np.array([self.minisom.winner(x) for x in data_np])
        
        # Convert 2D coordinates to 1D indices
        grid_size = self.params.topology_config.grid_size
        bmu_indices = winners[:, 0] * grid_size + winners[:, 1]
        
        # Return in same format as input (CuPy or NumPy)
        if hasattr(data, 'get'):
            return cp.array(bmu_indices)
        else:
            return bmu_indices
    
    def get_weights(self) -> Union[np.ndarray, cp.ndarray]:
        """
        Get the current weights of the SOM.
        
        Returns:
            SOM weights as array
        """
        if self.weights is None:
            # Get weights from MiniSOM if not cached
            minisom_weights = np.array(self.minisom.get_weights())
            self.weights = minisom_weights.reshape(-1, self.params.input_dim)
        
        # Return as NumPy array (MiniSOM is CPU-based)
        return self.weights
    
    def quantization_error(self, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Calculate the quantization error for the given data.
        
        Args:
            data: Input data vectors
            
        Returns:
            Quantization error value
        """
        # Convert data to NumPy if it's a CuPy array
        data_np = data.get() if hasattr(data, 'get') else data
        
        # Use MiniSOM's quantization error function
        return self.minisom.quantization_error(data_np)