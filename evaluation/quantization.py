"""
Quantization error metrics for SOM evaluation (self-contained).

This module provides a CPU-friendly QuantizationError without depending on
a base Metric protocol or requiring CuPy at import time.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp  # Optional; only used if available at runtime
except Exception:
    cp = None


def calculate_quantization_error(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the average quantization error for a SOM (optimized vectorized implementation)
    
    Args:
        data: Input data (samples x features)
        weights: SOM weights (neurons x features)
        
    Returns:
        error: Average quantization error
    """
    # Reshape weights to match format (total_neurons, features)
    if len(weights.shape) > 2:
        # For 1D SOM, shape is (grid_size, 1, features)
        # For 2D SOM, shape is (grid_size, grid_size, features)
        total_neurons = weights.shape[0] * weights.shape[1]
        weights = weights.reshape(total_neurons, -1)
    
    # Vectorized calculation of distances between each sample and all neurons
    # Expand dimensions for broadcasting
    data_expanded = data[:, np.newaxis, :]  # Shape: (n_samples, 1, input_dim)
    weights_expanded = weights[np.newaxis, :, :]  # Shape: (1, n_neurons, input_dim)
    
    # Calculate squared differences
    sq_diff = np.sum((data_expanded - weights_expanded) ** 2, axis=2)  # Shape: (n_samples, n_neurons)
    
    # Find minimum distance for each sample and take the square root
    min_distances = np.sqrt(np.min(sq_diff, axis=1))  # Shape: (n_samples,)
    
    # Return average error
    return float(np.mean(min_distances))


class QuantizationError:
    """
    Quantization Error metric for SOM evaluation.
    
    Measures the average distance between each data point and its
    best matching unit (BMU) in the SOM.
    
    Range: [0, ∞), lower is better
    """
    
    name = "quantization_error"
    
    def __init__(self, use_optimized: bool = True):
        """
        Initialize QuantizationError metric.
        
        Args:
            use_optimized: Whether to use the optimized vectorized implementation
        """
        self.use_optimized = use_optimized
    
    @property
    def requires_gpu(self) -> bool:
        return False  # CPU implementation is optimized
    
    def compute(self, som, data: Union[np.ndarray, "cp.ndarray"], **kwargs) -> float:
        """
        Compute quantization error.
        
        Args:
            som: Trained SOM instance
            data: Input data
            
        Returns:
            Quantization error value
        """
        # Convert to numpy for optimized computation (handle optional CuPy)
        if cp is not None and hasattr(data, 'get'):
            data_np = data.get()
        else:
            data_np = data
        
        # Get SOM weights
        weights = som.weights if hasattr(som, 'weights') else som.get_weights()
        if cp is not None and hasattr(weights, 'get'):
            weights_np = weights.get()
        else:
            weights_np = weights
        
        if self.use_optimized:
            # Use existing optimized implementation
            return calculate_quantization_error(data_np, weights_np)
        else:
            # Fallback simple implementation
            return self._compute_simple(data_np, weights_np)
    
    def _compute_simple(self, data: np.ndarray, weights: np.ndarray) -> float:
        """Simple fallback implementation."""
        # Reshape weights if needed
        if len(weights.shape) > 2:
            total_neurons = weights.shape[0] * weights.shape[1]
            weights = weights.reshape(total_neurons, -1)
        
        total_error = 0.0
        n_samples = len(data)
        
        for sample in data:
            distances = np.linalg.norm(weights - sample, axis=1)
            min_distance = np.min(distances)
            total_error += min_distance
        
        return float(total_error / n_samples)

