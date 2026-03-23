"""
Colors processor - color set based processing (formerly "smart" processing)
Maintains existing color set algorithms with GPU optimizations
Supports multi-GPU processing via Ray when available
"""

import numpy as np
import cupy as cp
import time
from typing import List, Dict, Optional, Union, Tuple
import logging

from .base_processor import ProcessingMethod
from .utils import update_weights_vectorized
from .colour_operations.bmu_scheduler import BMUScheduler
from .colour_operations.colour_set_creation import (
    calculate_equal_sized_color_sets, 
    calculate_batch_all_color_sets,
)
from .processing_params import ColorsConfig

logger = logging.getLogger(__name__)


class ColorsProcessor(ProcessingMethod):
    """
    Color set based processing with parallel independent sets
    Preserves existing GPU optimizations and influence bitmaps
    """
    
    def __init__(self, colors_config: ColorsConfig):
        """
        Initialize colors processor for single-GPU training ONLY.
        No Ray logic here!
        
        Args:
            colors_config: ColorsConfig with color processing parameters
        """
        super().__init__()
        self.colors_config = colors_config
        
        # Extract parameters from config
        self.processing_mode = colors_config.processing_mode
        self.max_rounds = colors_config.max_rounds
        self.sample_order = colors_config.sample_order
        self.color_set_algorithm = colors_config.color_set_algorithm
        self.num_color_sets = colors_config.num_color_sets
        self.enable_adaptive_bmu = colors_config.enable_adaptive_bmu
        self.bmu_recalc_initial = colors_config.bmu_recalc_initial
        self.bmu_recalc_decay_type = colors_config.bmu_recalc_decay_type
        self.bmu_recalc_min_samples = colors_config.bmu_recalc_min_samples
        self.recompute_bmus_per_color_set = bool(
            getattr(colors_config, "recompute_bmus_per_color_set", False)
        )
        
        # Initialize state variables
        self.bmu_scheduler = None
        self.color_sets = None
        self.verbose = False
        
        # Momentum support
        self.delta_weights = None
        self.use_momentum = False
        
        # HDSSSOM callback support
        self.selector_callback = None
        
        # Color set caching
        self._cached_color_sets = None
        self._cached_radius = None
        
        # Memory optimization
        self.chunk_size = colors_config.chunk_size  # Always set in config
        
        # Track processed samples for selective BMU recalculation
        self.processed_mask = None
        self.all_bmus = None

        # RNG control for deterministic behavior across runs
        self._rng_seed = None  # Populated from training params during initialize

        # Chunk/round mapping for localized BMU recalculation
        self._chunk_boundaries = None  # cp.ndarray of shape (num_chunks, 2) int32
        self._round_to_chunks = None   # List of lists: rounds -> [chunk_indices]
        self._iteration_total_samples = 0
        
        logger.info(f"ColorsProcessor initialized for single-GPU training (chunk_size: {self.chunk_size})")
    
    def initialize(self, som_weights, topology, params, data_source):
        """
        Initialize color set processing data structures

        Args:
            som_weights: Initial SOM weights
            topology: SOM topology
            params: Training parameters
            data_source: Data source
        """
        self.params = params  # Store params for later use
        self.verbose = params.verbose
        # Access processing config attributes correctly
        processing_config = params.processing_config
        self.processing_mode = processing_config.processing_mode
        self.use_momentum = processing_config.enable_momentum
        
        # Initialize BMU scheduler based on processing mode
        self.bmu_scheduler = BMUScheduler(
            enabled=self.enable_adaptive_bmu,
            initial_frequency=self.bmu_recalc_initial,
            decay_type=self.bmu_recalc_decay_type,
            min_samples=self.bmu_recalc_min_samples,
            verbose=self.verbose
        )

        # Record seed from training parameters (if provided) so that
        # chunk ordering and color-set processing can be made
        # deterministic across runs in parity tests.
        self._rng_seed = getattr(params, "seed", None)
            
        if self.verbose:
            if self.processing_mode == "equal_sized":
                logger.info(f"Equal-sized strategy: {self.max_rounds} rounds, {self.sample_order} sampling, adaptive_bmu={self.enable_adaptive_bmu}")
            else:
                logger.info(f"Batch-all strategy: adaptive_bmu={self.enable_adaptive_bmu}")
        
        # Initialize delta weights for momentum
        if som_weights is not None and self.use_momentum:
            self.delta_weights = cp.zeros_like(som_weights)
            if self.verbose:
                logger.info("Momentum enabled for color set processing")
        
    
    def set_selector_callback(self, callback):
        """
        Set callback function for HDSSSOM metadata updates
        """
        self.selector_callback = callback
    
    def process_samples(self, samples, som_weights, topology, params):
        """
        Process samples using color set algorithm for single-GPU
        Delegates to process_full_iteration for optimized GPU processing
        
        Args:
            samples: Input batch (n_samples, n_features)
            som_weights: Current SOM weights (n_nodes, n_features)
            topology: SOMTopology instance
            params: Training parameters
            
        Returns:
            Updated weights
        """
        # Check if samples is a path string (not supported in single-GPU mode)
        if isinstance(samples, str):
            raise ValueError(
                "File path provided but ColorsProcessor only supports in-memory data. "
                "Use ProcessorFactory with ray_config for GDS file streaming support."
            )
        
        # Delegate to optimized full iteration processing
        return self.process_full_iteration(samples, som_weights, topology, params)
    
    def process_full_iteration(self, samples, som_weights, topology, params):
        """
        Process a complete iteration keeping all data on GPU.
        Reduces CPU-GPU synchronization overhead.
        
        Args:
            samples: Input batch (n_samples, n_features) 
            som_weights: Initial SOM weights for this iteration
            topology: SOMTopology instance
            params: Training parameters
            
        Returns:
            Updated weights (and delta_weights if momentum enabled)
        """
        if self.verbose:
            start_time = time.time()
            
        # Ensure everything is on GPU from the start
        if isinstance(samples, np.ndarray):
            samples = cp.asarray(samples)
        if isinstance(som_weights, np.ndarray):
            som_weights = cp.asarray(som_weights)

        self._iteration_total_samples = int(samples.shape[0])
        
        # Update max_rounds from params each iteration so color set planning stays in sync
        processing_config = params.processing_config
        self.max_rounds = processing_config.max_rounds

        metric = processing_config.distance_metric
        metric_kwargs = processing_config.distance_metric_params or {}

        # Setup for processing
        # Re-seed RNGs for deterministic behavior when a seed is provided.
        if getattr(self, "_rng_seed", None) is not None:
            np.random.seed(self._rng_seed)
            cp.random.seed(self._rng_seed)

        color_sets = self._prepare_color_sets(samples, som_weights, topology, params)
        sample_indices = self._create_sample_ordering(samples.shape[0], params)

        # Initialize tracking (BMUs will be calculated per round)
        from .utils import find_bmus
        self.all_bmus = cp.zeros(samples.shape[0], dtype=cp.int32)  # Placeholder, will be updated per round
        self.processed_mask = cp.zeros(samples.shape[0], dtype=bool)

        # Initialize chunk-round mapping for localized BMU recalculation
        self._chunk_boundaries, self._round_to_chunks = self._init_chunk_round_mapping(
            total_samples=samples.shape[0],
            chunk_size=self.chunk_size,
            max_rounds=processing_config.max_rounds,
            sample_order=processing_config.sample_order,
            seed=getattr(self, "_rng_seed", None)
        )
        
        # Initialize weights and momentum
        current_weights = som_weights  # Use reference, not copy
        current_delta_weights = self._initialize_momentum(params, som_weights)
        
        # Store samples and influence matrix for BMU recalculation
        params.samples = samples
        params.influence_matrix = self.influence_matrix
        
        # Setup BMU scheduler if needed
        bmu_scheduler = self.bmu_scheduler if params.processing_config.enable_adaptive_bmu else None
        
        # Process all rounds using existing methods
        global_color_set_idx = 0
        for round_idx, chunk_ids in enumerate(self._round_to_chunks):
            if not chunk_ids:
                continue

            round_index_slices = []
            for chunk_idx in chunk_ids:
                chunk_idx_int = int(chunk_idx)
                chunk_start = int(self._chunk_boundaries[chunk_idx_int, 0].item())
                chunk_end = int(self._chunk_boundaries[chunk_idx_int, 1].item())
                if chunk_end <= chunk_start:
                    continue
                round_index_slices.append(sample_indices[chunk_start:chunk_end])

            if not round_index_slices:
                continue

            if len(round_index_slices) == 1:
                round_sample_indices = round_index_slices[0]
            else:
                round_sample_indices = cp.concatenate(round_index_slices)

            if round_sample_indices.size == 0:
                continue

            round_data = samples[round_sample_indices]

            # Calculate BMUs for this round using CURRENT weights
            # This matches Ray behavior where each chunk gets BMUs calculated with current weights
            round_bmus = find_bmus(
                round_data,
                current_weights,
                verbose=False,
                chunk_size=self.chunk_size,
                metric=metric,
                **metric_kwargs,
            )
            
            # Store in the global array for tracking
            self.all_bmus[round_sample_indices] = round_bmus
            
            # Use existing _process_single_round method with GPU-optimized color sets
            current_weights, current_delta_weights, global_color_set_idx = self._process_single_round_gpu(
                round_idx, round_sample_indices, round_data, round_bmus,
                color_sets, params, topology, current_weights, current_delta_weights,
                global_color_set_idx, bmu_scheduler, metric, metric_kwargs
            )
        
        # Store delta_weights for next iteration if using momentum
        if self.use_momentum and current_delta_weights is not None:
            self.delta_weights = current_delta_weights
        
        if self.verbose:
            end_time = time.time()
            logger.info(f"Color processing (GPU-optimized) completed in {end_time - start_time:.2f}s")
        
        # Return final weights
        if params.current_momentum > 0 or current_delta_weights is not None:
            return current_weights, current_delta_weights
        else:
            return current_weights

    def _init_chunk_round_mapping(
        self,
        total_samples: int,
        chunk_size: int,
        max_rounds: int,
        sample_order: str,
        seed: Optional[int] = None,
    ):
        """Create chunk boundaries and a round->chunks map without altering max_rounds."""
        if max_rounds <= 0:
            max_rounds = 1

        num_chunks = (total_samples + chunk_size - 1) // chunk_size if chunk_size > 0 else 0

        cb = cp.zeros((num_chunks, 2), dtype=cp.int32)
        for i in range(num_chunks):
            cb[i, 0] = i * chunk_size
            cb[i, 1] = min((i + 1) * chunk_size, total_samples)

        if num_chunks == 0:
            return cb, [[] for _ in range(max_rounds)]

        chunk_indices = np.arange(num_chunks, dtype=np.int32)
        if sample_order == "random" and num_chunks > 1:
            rng = np.random.default_rng(seed)
            rng.shuffle(chunk_indices)
        partitions = np.array_split(chunk_indices, max_rounds)
        r2c_list = [[int(idx) for idx in partition.tolist()] for partition in partitions]

        return cb, r2c_list

    # BMU recalculation method removed - now calculating BMUs per round with current weights
    
    def _prepare_color_sets(self, samples, som_weights, topology, params):
        """Prepare color sets with GPU optimization"""
        radius = params.current_radius
        
        # Get influence matrix
        self.influence_matrix, cached_radius = topology.get_precomputed_influence_matrix(
            radius, 'gaussian', return_radius=True
        )
        
        # Calculate or retrieve cached color sets
        if self._cached_color_sets is not None and self._cached_radius == cached_radius:
            if self.verbose:
                logger.debug(f"Using cached color sets (radius={cached_radius:.3f})")
            return self._cached_color_sets
        
        # Calculate new color sets
        color_params = type('obj', (object,), {
            'color_set_algorithm': self.color_set_algorithm,
            'num_color_sets': self.num_color_sets,
            'max_rounds': self.max_rounds
        })()
        
        if self.processing_mode == "equal_sized":
            color_sets = calculate_equal_sized_color_sets(
                samples, som_weights, topology, color_params, self.influence_matrix
            )
        else:
            color_sets = calculate_batch_all_color_sets(
                samples, som_weights, topology, color_params, self.influence_matrix
            )
        
        # Convert to GPU arrays once to avoid repeated conversions
        color_sets = [cp.asarray(cs, dtype=cp.int32) for cs in color_sets]
        
        # Cache for future use
        self._cached_color_sets = color_sets
        self._cached_radius = cached_radius
        
        if self.verbose:
            logger.debug(f"Calculated new color sets and cached (radius={cached_radius:.3f})")
        
        return color_sets
    
    def _create_sample_ordering(self, total_samples, params):
        """Create sample ordering on GPU"""
        sample_order = params.processing_config.sample_order
        max_rounds = params.processing_config.max_rounds
        
        if sample_order == 'random':
            return cp.random.permutation(total_samples)
        else:
            from .utils import create_strided_indices
            return create_strided_indices(total_samples, max_rounds)
    
    def _initialize_momentum(self, params, som_weights):
        """Initialize momentum weights if needed"""
        if params.current_momentum > 0:
            if hasattr(params, 'delta_weights') and params.delta_weights is not None:
                return cp.asarray(params.delta_weights)
            else:
                return cp.zeros_like(som_weights)
        return None
    
    def _process_single_round_gpu(self, round_idx, round_sample_indices, round_data, round_bmus,
                                  color_sets, params, topology, current_weights, current_delta_weights,
                                  global_color_set_idx, bmu_scheduler, metric, metric_kwargs):
        """GPU-optimized version of _process_single_round that uses pre-converted color sets"""
        # Randomize color set processing order
        color_set_order = cp.random.permutation(len(color_sets))
        
        # Process each color set
        for cs_idx in color_set_order:
            cs_idx_int = int(cs_idx)
            color_set = color_sets[cs_idx_int]  # Already a GPU array
            global_color_set_idx += 1

            if self.recompute_bmus_per_color_set:
                from .utils import find_bmus
                round_bmus = find_bmus(
                    round_data,
                    current_weights,
                    verbose=False,
                    chunk_size=self.chunk_size,
                    metric=metric,
                    **metric_kwargs,
                )
                self.all_bmus[round_sample_indices] = round_bmus
            
            # Process using GPU-optimized method
            current_weights, current_delta_weights = self._process_single_color_set_gpu(
                color_set, round_data, round_bmus, current_weights,
                topology, params, current_delta_weights, bmu_scheduler, round_sample_indices
            )
        
        return current_weights, current_delta_weights, global_color_set_idx
    
    def _process_single_color_set_gpu(self, color_set, round_data, round_bmus, current_weights,
                                      topology, params, current_delta_weights, bmu_scheduler, round_sample_indices):
        """GPU-optimized version using _apply_gpu_updates"""
        # Filter samples for this color set (color_set is already a CuPy array)
        mask = cp.isin(round_bmus, color_set)
        if not cp.any(mask):
            return current_weights, current_delta_weights
        
        samples_for_color_set = round_data[mask]
        bmus_for_color_set = round_bmus[mask]
        samples_indices_for_color_set = round_sample_indices[mask]
        
        # Apply GPU-optimized updates
        current_weights, current_delta_weights = self._apply_gpu_updates(
            samples_for_color_set, bmus_for_color_set,
            current_weights, current_delta_weights,
            params, params.influence_matrix
        )
        
        # Mark samples as processed
        self.processed_mask[samples_indices_for_color_set] = True
        if bmu_scheduler:
            bmu_scheduler.mark_samples_processed(samples_indices_for_color_set)
        
        return current_weights, current_delta_weights
    
    def _apply_gpu_updates(self, samples, bmus, weights, delta_weights, params, influence_matrix):
        """
        Apply weight updates staying entirely on GPU.
        Uses direct function calls to avoid wrapper overhead.
        
        Args:
            samples: Samples for this color set
            bmus: BMUs for these samples
            weights: Current weights (modified in-place)
            delta_weights: Current delta weights for momentum
            params: Training parameters
            influence_matrix: Full influence matrix
            
        Returns:
            tuple: (updated_weights, new_delta_weights)
        """
        from .utils import compute_weight_updates, apply_normalization, apply_momentum
        from .processing_params import TrainingStepParams
        
        config = params.processing_config
        
        # Create training params
        training_params = TrainingStepParams(
            radius=params.current_radius,
            learning_rate=params.current_learning_rate,
            momentum=params.current_momentum,
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
            virtual_ratio=getattr(config, 'virtual_ratio', 0.5)
        )
        
        # Direct calls to avoid wrapper overhead
        # Optimization #1: Use adaptive chunk size for smaller color sets
        accumulated_updates, accumulated_influence = compute_weight_updates(
            batch=samples,
            bmus=bmus,
            weights=weights,
            influence_matrix=influence_matrix,
            learning_rate=training_params.learning_rate,
            apply_lr_in_update=(config.normalization != "xpysom"),
            chunk_size=min(len(samples), config.chunk_size),  # Adaptive chunk size
            verbose=False
        )
        
        # Apply normalization
        normalized_updates = apply_normalization(
            accumulated_summed_updates=accumulated_updates,
            accumulated_influence_sum=accumulated_influence,
            original_batch_size=len(samples),
            normalization=config.normalization,
            learning_rate=training_params.learning_rate,
            current_weights=weights,
            full_bmu_counts=None,
            norm_alpha=config.norm_alpha,
            norm_clamp_factor=config.norm_clamp_factor,
            norm_percentile=config.norm_percentile,
            norm_max_update_threshold=config.norm_max_update_threshold,
            training_progress=config.training_progress,
            current_epoch=config.current_epoch,
            total_epochs=config.total_epochs,
            virtual_ratio=getattr(config, 'virtual_ratio', 0.5),
            verbose=False
        )

        if bool(getattr(config, "scale_color_updates_by_fraction", True)):
            # Match minibatch-equivalent semantics: each color-set update contributes
            # proportionally to its sample share within this iteration.
            color_set_samples = int(len(samples))
            iteration_total_samples = int(
                getattr(self, "_iteration_total_samples", color_set_samples) or color_set_samples
            )
            if color_set_samples > 0 and iteration_total_samples > 0:
                normalized_updates *= float(color_set_samples) / float(iteration_total_samples)
        
        # Apply momentum
        weight_changes, new_delta = apply_momentum(
            normalized_updates=normalized_updates,
            momentum_coefficient=params.current_momentum,
            delta_weights=delta_weights
        )
        
        # Update weights in-place
        weights += weight_changes
        
        return weights, new_delta
    
    
    def cleanup(self):
        """Clean up resources"""
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
    
