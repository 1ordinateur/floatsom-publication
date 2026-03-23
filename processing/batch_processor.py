"""
Batch processor - simple batch processing without color set complexity
Supports both single-GPU and multi-GPU (Ray) processing
"""

import cupy as cp
import numpy as np
from typing import Union, Tuple, Optional
import logging

from .base_processor import ProcessingMethod
from .utils import (
    find_bmus, update_weights_vectorized,
    calculate_updates, apply_weight_updates_with_momentum, apply_normalization
)
from .processing_params import BatchConfig, TrainingStepParams

logger = logging.getLogger(__name__)


class BatchProcessor(ProcessingMethod):
    """
    Simple batch processing without color set restrictions
    Supports full batch mode or minibatch mode
    """
    
    def __init__(self, batch_config: BatchConfig):
        """
        Initialize batch processor for single-GPU training.
        
        Args:
            batch_config: BatchConfig with batch processing parameters
        """
        super().__init__()
        self.selector_callback = None  # Will be set by FloatSOM if HDSSSOM is used
        
        # Store batch configuration
        self.batch_config = batch_config
        self.mode = batch_config.batch_mode
        self.chunk_size = batch_config.chunk_size  # Always set in config, also used as minibatch size
        self.verbose = False  # Will be set from params in initialize
        
        logger.info(f"BatchProcessor initialized for single-GPU training (mode: {self.mode}, chunk_size: {self.chunk_size})")
    
    def set_selector_callback(self, callback):
        """
        Set callback function for HDSSSOM metadata updates
        """
        self.selector_callback = callback
    
    def initialize(self, som_weights, topology, params, data_source):
        """
        Initialize batch processor

        Args:
            som_weights: Initial SOM weights
            topology: SOM topology
            params: Training parameters
            data_source: Data source
        """
        # Store params for later use
        self.params = params
        self.verbose = params.verbose
    
    def process_samples(self, samples, som_weights, topology, params):
        """
        Process samples based on configured mode
        - full_batch: Process entire batch, accumulate updates, apply once
        - minibatch: Process in minibatches, update weights after each
        - Ray multi-GPU: Distribute processing across multiple GPUs with NCCL
        - GDS streaming: Stream data directly from storage to GPU
        
        Args:
            samples: Input batch (n_samples, n_features) or path to data file (str)
            som_weights: Current SOM weights (n_nodes, n_features)
            topology: SOMTopology instance
            params: Training parameters dict with radius, learning_rate, etc.
            
        Returns:
            Updated weights or (updated_weights, delta_weights) if momentum enabled
        """
        # Check if samples is a path string (not supported in single-GPU mode)
        if isinstance(samples, str):
            raise ValueError(
                "File path provided but BatchProcessor only supports in-memory data. "
                "Use ProcessorFactory with ray_config for GDS file streaming support."
            )
        
        # chunk_size is always set in config, no need for fallback
        
        # Use standard single-GPU processing
        if self.mode == 'minibatch':
            return self._process_minibatch_mode(samples, som_weights, topology, params)
        else:
            return self._process_full_batch_mode(samples, som_weights, topology, params)
    
    def _process_full_batch_mode(self, samples, som_weights, topology, params):
        """
        Process entire batch using memory-efficient chunking that avoids creating full influence matrix
        """
        # Ensure inputs are CuPy arrays
        if not isinstance(samples, cp.ndarray):
            samples = cp.asarray(samples)
        
        n_samples = samples.shape[0]
        chunk_size = self.chunk_size  # Use configured chunk size
        
        # Create TrainingStepParams from the provided params
        training_params = TrainingStepParams(
            radius=params.current_radius,
            learning_rate=params.current_learning_rate,
            momentum=params.current_momentum,
            delta_weights=params.delta_weights,
            normalization=params.processing_config.normalization,
            total_samples=n_samples,
            norm_alpha=params.processing_config.norm_alpha,
            norm_clamp_factor=params.processing_config.norm_clamp_factor,
            norm_percentile=params.processing_config.norm_percentile,
            norm_max_update_threshold=params.processing_config.norm_max_update_threshold,
            training_progress=params.processing_config.training_progress,
            current_epoch=params.processing_config.current_epoch,
            total_epochs=params.processing_config.total_epochs,
            virtual_ratio=getattr(params.processing_config, 'virtual_ratio', 0.5),
            use_sparse_influence=getattr(params.processing_config, "use_sparse_influence", False),
            distance_metric=params.processing_config.distance_metric,
        )

        metric = params.processing_config.distance_metric
        metric_kwargs = params.processing_config.distance_metric_params or {}

        # For BMU selection we only need argmin indices; skip distances and sqrt for speed
        bmus = find_bmus(samples, som_weights, return_distances=False,
                         chunk_size=self.chunk_size,
                         metric=metric,
                         **metric_kwargs)
        
        # Get full influence matrix - no need to extract BMU rows anymore
        full_influence_matrix = topology.get_precomputed_influence_matrix(params.current_radius, 'gaussian')
        
        updated_weights, new_delta_weights = update_weights_vectorized(
            data=samples,
            bmus=bmus,
            influence_matrix=full_influence_matrix,  # Pass the full matrix directly
            weights=som_weights,
            training_params=training_params,
            chunk_size=chunk_size,
            verbose=False
        )

        # Return format depends on momentum usage
        if training_params.momentum > 0 or training_params.delta_weights is not None:
            return updated_weights, new_delta_weights
        else:
            return updated_weights
    
    def _process_minibatch_mode(self, samples, som_weights, topology, params):
        """
        Process samples in minibatches (using chunk_size), updating weights after each minibatch
        """
        # Extract parameters
        radius = params.current_radius
        learning_rate = params.current_learning_rate
        momentum_coefficient = params.current_momentum
        delta_weights = params.delta_weights
        
        # Ensure inputs are CuPy arrays
        if not isinstance(samples, cp.ndarray):
            samples = cp.asarray(samples)
        
        n_samples = samples.shape[0]
        current_weights = som_weights.copy()
        current_delta_weights = delta_weights
        
        # Process in minibatches (using chunk_size as minibatch size)
        for i in range(0, n_samples, self.chunk_size):
            # Get minibatch
            end_idx = min(i + self.chunk_size, n_samples)
            minibatch = samples[i:end_idx]
            
            # Find BMUs for minibatch
            bmus, distances = self._find_bmus_batch(minibatch, current_weights, params)
            
            # Update selector metadata if HDSSSOM is being used
            if self.selector_callback is not None:
                self.selector_callback(minibatch, bmus, distances)
            
            # Use the precomputed full influence matrix; chunked BMU indexing happens downstream.
            influence_matrix = topology.get_precomputed_influence_matrix(radius, 'gaussian')
            
            # Update weights with minibatch
            result = self._update_weights_vectorized(
                minibatch, bmus, influence_matrix, current_weights, learning_rate,
                momentum_coefficient=momentum_coefficient,
                delta_weights=current_delta_weights,
                params=params
            )
            
            # Extract updated weights and delta weights
            if isinstance(result, tuple):
                current_weights, current_delta_weights = result
            else:
                current_weights = result
        
        # Return final weights (and delta weights if using momentum)
        if momentum_coefficient > 0 or delta_weights is not None:
            return current_weights, current_delta_weights
        else:
            return current_weights
    
    def _find_bmus_batch(self, samples, som_weights, params):
        """
        Find BMUs for entire batch efficiently using optimized utils function with chunking
        
        Args:
            samples: Input samples (n_samples, n_features)
            som_weights: SOM weights (n_nodes, n_features)
            
        Returns:
            BMU indices (n_samples,), distances (n_samples,)
        """
        # Ensure inputs are CuPy arrays
        if not isinstance(samples, cp.ndarray):
            samples = cp.asarray(samples)
        if not isinstance(som_weights, cp.ndarray):
            som_weights = cp.asarray(som_weights)

        metric = params.processing_config.distance_metric
        metric_kwargs = params.processing_config.distance_metric_params or {}

        # Use optimized BMU finding from utils with optimal chunk size
        bmus, distances = find_bmus(samples, som_weights, return_distances=True, 
                                   chunk_size=self.chunk_size,
                                   metric=metric,
                                   **metric_kwargs)
        
        return bmus, distances
    
    def _update_weights_vectorized(self, samples, bmus, influence, som_weights, learning_rate, 
                                 momentum_coefficient=0.0, delta_weights=None, params=None):
        """
        Vectorized weight update for entire batch using optimized utils function
        
        Args:
            samples: Input samples (n_samples, n_features)
            bmus: BMU indices (n_samples,)
            influence: Full influence matrix (n_nodes, n_nodes)
            som_weights: Current weights (n_nodes, n_features)
            learning_rate: Learning rate
            momentum_coefficient: Momentum coefficient
            delta_weights: Previous weight changes for momentum
            params: FloatSOMParams object with normalization parameters
            
        Returns:
            (updated_weights, new_delta_weights) if momentum enabled, else updated_weights
        """
        from .processing_params import TrainingStepParams
        
        # Create TrainingStepParams from current parameters
        config = params.processing_config
        training_params = TrainingStepParams(
            radius=params.current_radius,
            learning_rate=learning_rate,
            momentum=momentum_coefficient,
            delta_weights=delta_weights,
            normalization=config.normalization,
            total_samples=len(samples),
            norm_alpha=config.norm_alpha,
            norm_clamp_factor=config.norm_clamp_factor,
            norm_percentile=config.norm_percentile,
            norm_max_update_threshold=config.norm_max_update_threshold,
            training_progress=config.training_progress,
            current_epoch=config.current_epoch,
            total_epochs=config.total_epochs,
            virtual_ratio=config.virtual_ratio,
            use_sparse_influence=getattr(config, "use_sparse_influence", False),
        )
        
        # Pass TrainingStepParams object directly - let update_weights_vectorized extract what it needs
        updated_weights, new_delta_weights = update_weights_vectorized(
            data=samples,
            bmus=bmus,
            influence_matrix=influence,  # Now expecting full influence matrix
            weights=som_weights,
            training_params=training_params
        )
        
        # Return format depends on momentum usage
        if momentum_coefficient > 0 or delta_weights is not None:
            return updated_weights, new_delta_weights
        else:
            return updated_weights
    
    def cleanup(self):
        """Clean up resources"""
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
