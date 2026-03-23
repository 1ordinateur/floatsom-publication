"""
Shared utility functions for sample processing methods
Uses optimized implementations from parallel_som update_strategy_utils
"""

import cupy as cp
import numpy as np
import logging
from typing import Union, Tuple, Optional, Callable, Any, Dict
from abc import ABC, abstractmethod

# Import JIT kernels
from .jit_kernels import (
    launch_euclidean_distance_jit,
    launch_manhattan_distance_jit,
    launch_cosine_distance_jit,
    launch_partial_updates_jit
)

logger = logging.getLogger(__name__)


# ==================== DISTANCE METRICS ====================

def calculate_distances(batch: cp.ndarray, weights: cp.ndarray, 
                        metric: str = "euclidean", **kwargs) -> cp.ndarray:
    """
    Calculate distances using JIT-compiled kernels.
    
    Args:
        batch: Input batch (batch_size, input_dim)
        weights: Weight vectors (num_nodes, input_dim)
        metric: Distance metric ("euclidean", "cosine", "manhattan")
        **kwargs: Additional parameters for specific metrics
        
    Returns:
        Distance matrix (batch_size, num_nodes)
    """

    eps = kwargs.get('epsilon', 1e-8)  # Keep epsilon as float
    apply_sqrt = kwargs.get('apply_sqrt', True)

    # Defensive alignment of batch and weights to uint32 due to obscure error 
    batch_size = cp.uint32(batch.shape[0])
    num_nodes = cp.uint32(weights.shape[0])
    input_dim = cp.uint32(batch.shape[1])

    if metric == "euclidean":
        return launch_euclidean_distance_jit(
            batch, weights, apply_sqrt=apply_sqrt, batch_size=batch_size, 
            num_nodes=num_nodes, input_dim=input_dim)
    elif metric == "cosine":
        # Pass through precomputed norms if provided to avoid redundant reductions
        return launch_cosine_distance_jit(
            batch,
            weights,
            epsilon=eps,
            batch_size=batch_size,
            num_nodes=num_nodes,
            input_dim=input_dim,
            batch_norms=kwargs.get('batch_norms'),
            weight_norms=kwargs.get('weight_norms'),
        )
    elif metric == "manhattan":
        return launch_manhattan_distance_jit(
            batch, weights, batch_size=batch_size, 
            num_nodes=num_nodes, input_dim=input_dim)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


# Keep NormPDistance for now as it doesn't have JIT implementation yet
class NormPDistance:
    """Generalized Lp norm distance metric with configurable p."""
    
    def __init__(self, p: float = 3.0):
        """Initialize with p value for Lp norm.
        
        Args:
            p: The p value for Lp norm (must be >= 1)
        """
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        self.p = p
    
    @property
    def can_cache(self) -> bool:
        return False
    
    def calculate_distances(self, batch: cp.ndarray, weights: cp.ndarray) -> cp.ndarray:
        """Calculate Lp norm distances with in-place operations."""
        # Expand dimensions for broadcasting
        batch_expanded = batch[:, cp.newaxis, :]  # (batch_size, 1, input_dim)
        weights_expanded = weights[cp.newaxis, :, :]  # (1, total_nodes, input_dim)
        
        # Calculate Lp distance with in-place operations where possible
        diff = cp.abs(batch_expanded - weights_expanded)
        diff **= self.p  # In-place power
        distances = cp.sum(diff, axis=2)
        distances **= (1 / self.p)  # In-place root
        
        return distances


def process_in_chunks(data: cp.ndarray, chunk_size: int, process_func: Callable, 
                     return_tuple: bool = False, **kwargs) -> Union[cp.ndarray, Tuple[cp.ndarray, ...]]:
    """
    Generic utility to process data in chunks and concatenate results
    
    Args:
        data: Input data to be chunked (first dimension is chunked)
        chunk_size: Size of each chunk
        process_func: Function to apply to each chunk
        return_tuple: Whether process_func returns a tuple of arrays
        **kwargs: Additional arguments passed to process_func
        
    Returns:
        Concatenated results from all chunks
    """
    data_size = data.shape[0]
    
    if data_size <= chunk_size:
        return process_func(data, **kwargs)
    
    results = []
    
    for i in range(0, data_size, chunk_size):
        end_idx = min(i + chunk_size, data_size)
        chunk = data[i:end_idx]
        chunk_result = process_func(chunk, **kwargs)
        results.append(chunk_result)
    
    if return_tuple:
        # Concatenate each element of the tuple separately
        num_outputs = len(results[0])
        concatenated = tuple(
            cp.concatenate([r[j] for r in results], axis=0) 
            for j in range(num_outputs)
        )
        return concatenated
    else:
        # Single array output
        return cp.concatenate(results, axis=0)


def find_bmus(batch: cp.ndarray, weights: cp.ndarray, verbose: bool = False, 
              return_distances: bool = False, chunk_size: Optional[int] = None,
              metric: str = "euclidean", **metric_kwargs) -> Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]:
    """
    Find Best Matching Units for batch of samples using memory-efficient operations
    
    Args:
        batch: Input batch of shape (batch_size, input_dim)
        weights: SOM weights of shape (total_nodes, input_dim)
        verbose: Print progress information
        return_distances: Also return distances for HDSSSOM difficulty tracking
        chunk_size: Optional chunk size for memory management (None = no chunking)
        metric: Distance metric name ("euclidean", "cosine", "manhattan")
        **metric_kwargs: Additional parameters for the distance metric
        
    Returns:
        BMU indices or (BMU indices, distances) if return_distances=True
    """
    # Use chunking if chunk_size is provided and data is large enough
    if chunk_size is not None and batch.shape[0] > chunk_size:
        return process_in_chunks(
            data=batch,
            chunk_size=chunk_size,
            process_func=_find_bmus_core,
            return_tuple=return_distances,
            weights=weights,
            verbose=verbose,
            return_distances=return_distances,
            metric=metric,
            **metric_kwargs
        )
    else:
        # Process directly without chunking
        return _find_bmus_core(batch, weights, verbose, return_distances, metric, **metric_kwargs)


def _find_bmus_core(batch: cp.ndarray, weights: cp.ndarray, verbose: bool = False, 
                   return_distances: bool = False,
                   metric: str = "euclidean",
                   node_chunk_size: int = 900, **metric_kwargs) -> Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]:
    """
    Core BMU finding logic for a single chunk
    
    Args:
        batch: Input batch of shape (batch_size, input_dim)
        weights: SOM weights of shape (total_nodes, input_dim)
        verbose: Print progress information
        return_distances: Also return minimum distances
        metric: Distance metric name
        node_chunk_size: Chunk size for node processing (default 900 = 30x30)
        **metric_kwargs: Additional parameters for the distance metric
    """
    total_nodes = weights.shape[0]
    batch_size = batch.shape[0]
    
    # Precompute batch norms once for cosine to avoid per-chunk recomputation
    batch_norms = None
    if metric == "cosine":
        eps = metric_kwargs.get('epsilon', 1e-8)
        batch_norms = cp.maximum(cp.linalg.norm(batch, axis=1), eps)

    # Use chunked calculation for large node counts to save memory
    if total_nodes > node_chunk_size:
        if verbose:
            logger.debug(f"Using chunked BMU calculation for {total_nodes} nodes (chunks of {node_chunk_size})")
        
        # Initialize arrays to track minimum distances and BMUs
        min_distances = cp.full(batch_size, cp.inf, dtype=cp.float32)
        bmus = cp.zeros(batch_size, dtype=cp.int32)
        row_indices = cp.arange(batch_size, dtype=cp.int32)
        apply_sqrt = metric_kwargs.get('apply_sqrt', return_distances)
        
        # Process nodes in chunks
        for chunk_start in range(0, total_nodes, node_chunk_size):
            chunk_end = min(chunk_start + node_chunk_size, total_nodes)
            weight_chunk = weights[chunk_start:chunk_end]
            
            # Calculate distances for this chunk; keep squared values for Euclidean.
            metric_kwargs_chunk = dict(metric_kwargs)
            if metric == "cosine":
                metric_kwargs_chunk['batch_norms'] = batch_norms
                weight_norms = metric_kwargs_chunk.get('weight_norms')
                if weight_norms is not None:
                    metric_kwargs_chunk['weight_norms'] = weight_norms[chunk_start:chunk_end]
            chunk_apply_sqrt = apply_sqrt
            if metric == "euclidean":
                chunk_apply_sqrt = False
            chunk_distances = calculate_distances(
                batch,
                weight_chunk,
                metric,
                apply_sqrt=chunk_apply_sqrt,
                **metric_kwargs_chunk,
            )
            # Ensure contiguous float32 before reduction to avoid CUB instability at scale.
            chunk_distances = cp.ascontiguousarray(chunk_distances, dtype=cp.float32)
            
            # Find minimum distances and indices within this chunk
            chunk_min_indices = cp.argmin(chunk_distances, axis=1).astype(cp.int32, copy=False)
            chunk_min_distances = chunk_distances[row_indices, chunk_min_indices]
            
            # Update global minimums where this chunk has better values
            better_mask = chunk_min_distances < min_distances
            min_distances[better_mask] = chunk_min_distances[better_mask]
            # Adjust indices to global node numbering
            bmus[better_mask] = chunk_min_indices[better_mask] + chunk_start
            
            # Free memory from this chunk
            del chunk_distances
        
        if return_distances:
            if metric == "euclidean" and apply_sqrt:
                min_distances = cp.sqrt(min_distances)
            return bmus, min_distances
        return bmus
    
    # Small enough to calculate all at once
    apply_sqrt = metric_kwargs.get('apply_sqrt', return_distances)
    distance_apply_sqrt = apply_sqrt
    if metric == "euclidean":
        distance_apply_sqrt = False
    distances = calculate_distances(batch, weights, metric, apply_sqrt=distance_apply_sqrt)
    
    # Find indices of minimum distances for each sample
    bmus = cp.argmin(distances, axis=1)
    
    if return_distances:
        # Get the minimum distances for HDSSSOM difficulty tracking
        row_indices = cp.arange(batch_size)
        min_distances = distances[row_indices, bmus]
        if metric == "euclidean" and apply_sqrt:
            min_distances = cp.sqrt(min_distances)
        return bmus, min_distances
    
    return bmus


def find_bmu_single(sample: cp.ndarray, weights: cp.ndarray, 
                    metric: str = "euclidean", **metric_kwargs) -> int:
    """
    Find Best Matching Unit for single sample
    
    Args:
        sample: Input sample (n_features,)
        weights: SOM weights (total_nodes, n_features)
        metric: Distance metric name
        **metric_kwargs: Additional parameters for the distance metric
        
    Returns:
        BMU index (single integer for flattened weights)
    """
    # Expand sample to batch of size 1 and use optimized batch function
    batch = sample.reshape(1, -1)
    bmu = find_bmus(batch, weights, verbose=False, metric=metric, **metric_kwargs)
    return bmu[0]


def calculate_influence_matrix(bmu_positions, som_shape, radius, influence_type="gaussian"):
    """
    Calculate influence matrix for weight updates
    
    Args:
        bmu_positions: BMU positions [(row, col), ...]
        som_shape: Shape of SOM (height, width)
        radius: Neighborhood radius
        influence_type: "gaussian" or "bubble"
        
    Returns:
        Influence matrix
    """
    pass


@cp.fuse
def gaussian_influence(distance: cp.ndarray, radius: float) -> cp.ndarray:
    """
    Calculate Gaussian neighborhood influence
    
    Args:
        distance: Distance from BMU (can be array)
        radius: Neighborhood radius
        
    Returns:
        Influence value (0-1) using Gaussian decay
    """
    # Standard deviation is radius/3 so that radius covers ~3 sigma
    # Enforce minimum sigma to prevent explosion when radius approaches 0
    sigma = cp.maximum(radius / 3.0, 0.01)
    return cp.exp(-distance**2 / (2 * sigma**2))


@cp.fuse
def bubble_influence(distance: cp.ndarray, radius: float) -> cp.ndarray:
    """
    Calculate bubble (binary) neighborhood influence
    
    Args:
        distance: Distance from BMU (can be array)
        radius: Neighborhood radius
        
    Returns:
        Influence value (0 or 1) - binary within radius
    """
    return (distance <= radius).astype(cp.float32)


def _calculate_partial_updates_and_influence_gpu(
    batch_chunk: cp.ndarray,
    bmus_chunk: cp.ndarray, 
    influence_chunk: cp.ndarray,
    weights: cp.ndarray,
    lr: float,
    apply_lr_in_update: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Calculate summed updates and summed influence for a chunk of the batch on GPU.
    Uses JIT-compiled fused kernel for optimal performance.
    """
    # Validate learning rate to prevent weight explosions when applied in-kernel.
    if apply_lr_in_update and not (0 < lr <= 10.0):
        raise ValueError(f"Learning rate {lr} is outside safe range (0, 10.0]")

    # Use JIT-compiled fused kernel
    effective_lr = float(lr) if apply_lr_in_update else 1.0
    return launch_partial_updates_jit(batch_chunk, influence_chunk, weights, effective_lr)


def apply_normalization(accumulated_summed_updates: cp.ndarray, 
                       accumulated_influence_sum: cp.ndarray,
                       original_batch_size: int,
                       normalization: str,
                       learning_rate: Optional[float] = None,
                       current_weights: Optional[cp.ndarray] = None,
                       full_bmu_counts: Optional[cp.ndarray] = None,
                       norm_alpha: Optional[float] = None,
                       norm_clamp_factor: Optional[float] = None, 
                       norm_percentile: Optional[float] = None,
                       norm_max_update_threshold: Optional[float] = None,
                       training_progress: Optional[float] = None,
                       current_epoch: Optional[int] = None,
                       total_epochs: Optional[int] = None,
                       virtual_ratio: float = 0.5,
                       verbose: bool = False) -> cp.ndarray:
    """
    Apply normalization to accumulated weight updates
    
    Args:
        accumulated_summed_updates: Accumulated weight updates (total_nodes, input_dim)
        accumulated_influence_sum: Accumulated influence values (total_nodes,)
        original_batch_size: Original batch size for normalization
        normalization: Normalization method
        learning_rate: Current learning rate (optional, for mode-specific semantics)
        current_weights: Current weights (required for xpysom mode to form delta updates)
        full_bmu_counts: BMU counts for minisom_weighted (optional)
        norm_alpha: Blend ratio for hybrid normalization
        norm_clamp_factor: Clamp factor for clamped_weighted normalization
        norm_percentile: Percentile threshold for local normalization
        norm_max_update_threshold: Threshold for extreme value validation
        training_progress: Training progress 0-1 for adaptive normalization
        current_epoch: Current epoch for adaptive normalization
        total_epochs: Total epochs for adaptive normalization
        virtual_ratio: Virtual samples ratio for count_based normalization (0.2=fast, 0.5=balanced, 1.0=stable)
        verbose: Print warnings for extreme values
        
    Returns:
        normalized_updates: Normalized weight updates (total_nodes, input_dim)
    """
    # Calculate training progress for adaptive normalization if needed
    if normalization == "adaptive":
        if training_progress is None:
            if current_epoch is not None and total_epochs is not None:
                training_progress = min(current_epoch / total_epochs, 1.0)
            else:
                raise ValueError("Adaptive normalization requires either training_progress or (current_epoch and total_epochs)")
    
    # Apply normalization based on method
    if normalization == "count_based" or normalization == "weighted":
        # Calculate expected distribution
        total_nodes = accumulated_influence_sum.shape[0]
        expected_samples_per_node = original_batch_size / total_nodes
        
        # Add virtual samples (this prevents explosions!)
        virtual_samples = max(virtual_ratio * expected_samples_per_node, 1.0)
        
        # Safe normalization
        effective_influence = accumulated_influence_sum + virtual_samples
        normalized_updates = accumulated_summed_updates / effective_influence.reshape(-1, 1)
        
    elif normalization == "sqrt_weighted":
        # Use square root to dampen the normalization effect
        sqrt_denominator = cp.sqrt(cp.maximum(accumulated_influence_sum.reshape(-1, 1), 1e-10))
        normalized_updates = accumulated_summed_updates / sqrt_denominator
        
    elif normalization == "hybrid":
        if norm_alpha is None:
            raise ValueError("norm_alpha is required for hybrid normalization")
        # Blend between weighted and fixed normalization with tunable parameter
        weighted_denominator = cp.maximum(accumulated_influence_sum.reshape(-1, 1), 1e-10)
        weighted_norm = accumulated_summed_updates / weighted_denominator
        fixed_norm = accumulated_summed_updates / original_batch_size
        # Blend them
        normalized_updates = norm_alpha * weighted_norm + (1 - norm_alpha) * fixed_norm
        
    elif normalization == "clamped_weighted":
        if norm_clamp_factor is None:
            raise ValueError("norm_clamp_factor is required for clamped_weighted normalization")
        # Clamp the influence sum to prevent over-normalization
        max_influence = norm_clamp_factor * original_batch_size
        clamped_influence = cp.minimum(accumulated_influence_sum, max_influence)
        denominator = cp.maximum(clamped_influence.reshape(-1, 1), 1e-10)
        normalized_updates = accumulated_summed_updates / denominator
        
    elif normalization == "local":
        if norm_percentile is None:
            raise ValueError("norm_percentile is required for local normalization")
        # Normalize based on local statistics rather than global
        local_threshold = cp.percentile(accumulated_influence_sum, norm_percentile)
        local_denominator = cp.maximum(local_threshold, 1e-10)
        normalized_updates = accumulated_summed_updates / local_denominator
        
    elif normalization == "adaptive":
        # Changes normalization strategy based on training progress
        # Early training: more like "none", later training: more like weighted
        adaptive_denominator = ((1 - training_progress) * original_batch_size + 
                               training_progress * cp.maximum(accumulated_influence_sum.reshape(-1, 1), 1e-10))
        normalized_updates = accumulated_summed_updates / adaptive_denominator
        
    elif normalization == "log_weighted":
        # Log scale the influence sum to reduce extreme values
        log_denominator = cp.log1p(accumulated_influence_sum)  # log(1 + x)
        log_denominator = cp.maximum(log_denominator.reshape(-1, 1), 1e-10)
        normalized_updates = accumulated_summed_updates / log_denominator
        
    elif normalization == "minisom_weighted":
        if full_bmu_counts is None:
            raise ValueError("full_bmu_counts is required for minisom_weighted normalization")
        # MiniSom-style weighted normalization that zeros out non-BMU nodes
        denominator = cp.maximum(accumulated_influence_sum.reshape(-1, 1), 1e-10)
        provisional_normalized_updates = accumulated_summed_updates / denominator
        
        # Apply MiniSom's characteristic: zero out updates for nodes that were never BMUs
        is_bmu_node_in_full_batch = (full_bmu_counts.reshape(-1, 1) > 0)
        normalized_updates = provisional_normalized_updates * is_bmu_node_in_full_batch
        
    elif normalization == "none":
        # In 'none' mode, we don't normalize by neighborhood influence.
        # However, we still average the updates over the entire batch to keep the
        # learning rate independent of the batch size.
        normalized_updates = accumulated_summed_updates / original_batch_size
    elif normalization == "xpysom":
        # XPySOM-style merge:
        # weights_new = numerator / denominator
        # and delta = weights_new - weights_old = (sum(g*x) / sum(g)) - w_old.
        # Upstream update accumulation must therefore pass raw (unscaled) numerators.
        if current_weights is None:
            raise ValueError("xpysom normalization requires current_weights to form delta updates.")
        denominator = accumulated_influence_sum.reshape(-1, 1)
        nonzero_mask = denominator != 0
        safe_denominator = cp.where(nonzero_mask, denominator, 1)
        target_weights = accumulated_summed_updates / safe_denominator
        target_weights = cp.where(nonzero_mask, target_weights, current_weights)
        normalized_updates = target_weights - current_weights
        
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Validate updates for extreme values if threshold is provided
    if norm_max_update_threshold is not None:
        max_update = cp.max(cp.abs(normalized_updates))
        if max_update > norm_max_update_threshold:
            if verbose:
                logger.warning(f"Large update detected with {normalization}: {max_update:.6f}")
            # Optional: clamp extreme updates
            normalized_updates = cp.clip(normalized_updates, -norm_max_update_threshold, norm_max_update_threshold)
    
    return normalized_updates


def create_strided_indices(total_samples: int, num_rounds: int) -> cp.ndarray:
    """
    Create strided indices for deterministic load distribution
    
    Args:
        total_samples: Total number of samples
        num_rounds: Number of rounds to distribute over
        
    Returns:
        Strided indices array
    """
    # Vectorized creation: create all indices at once
    all_indices = cp.arange(total_samples)
    round_assignments = all_indices % num_rounds
    
    # Sort indices by their round assignment to get strided pattern
    sorted_order = cp.argsort(round_assignments)
    return all_indices[sorted_order]


def get_chunked_influence(influence_map: cp.ndarray, batch_bmus: cp.ndarray, 
                                 chunk_size: Optional[int] = None, verbose: bool = False) -> cp.ndarray:
    """
    Get influence values for batch BMUs in chunks to avoid memory issues
    
    Args:
        influence_map: Full influence matrix
        batch_bmus: BMU indices for the batch
        chunk_size: Optional chunk size for memory management
        verbose: Print progress information
        
    Returns:
        Influence values for the batch BMUs
    """
    batch_size = batch_bmus.shape[0]
    total_nodes = influence_map.shape[0]
    
    # Auto-calculate chunk size for GPU if not provided
    if chunk_size is None:
        # Conservative GPU chunk size calculation
        available_memory = cp.cuda.Device().mem_info[0]  # Free memory in bytes
        memory_per_sample = total_nodes * 4  # float32
        max_samples = int(available_memory * 0.5 / memory_per_sample)  # Use 50% of free memory
        chunk_size = min(max(max_samples, 128), batch_size)  # At least 128, at most batch_size
            
    # Check if we need chunking
    if batch_size <= chunk_size:
        return influence_map[batch_bmus]
    
    # Large batch - chunked processing
    if verbose:
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        logger.debug(f"Processing influence lookup in {num_chunks} chunks of size {chunk_size}")
    
    # Pre-allocate result array
    result = cp.zeros((batch_size, total_nodes), dtype=influence_map.dtype)
    
    # Process in chunks
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_bmus = batch_bmus[i:end_idx]
        
        # Memory management
        if i > 0:  # Not first chunk
            cp.cuda.get_current_stream().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
        
        # Lookup influence for this chunk
        chunk_influence = influence_map[chunk_bmus]
        result[i:end_idx] = chunk_influence
        
        # Clear intermediate memory
        del chunk_influence
    
    return result


def check_weights_validity(weights: cp.ndarray, verbose: bool = True) -> bool:
    """
    Check if weights are valid (no NaN or Inf values)
    Copied from update_strategy_utils.py
    """
    # Check for NaN values
    has_nan = cp.isnan(weights).any()
    
    # Check for Inf values
    has_inf = cp.isinf(weights).any()
    
    # Report results
    if has_nan or has_inf:
        if verbose:
            if has_nan:
                nan_count = cp.isnan(weights).sum().item()
                logger.warning(f"Found {nan_count} NaN values in weights")
            if has_inf:
                inf_count = cp.isinf(weights).sum().item()
                logger.warning(f"Found {inf_count} Inf values in weights")
        return False
    
    return True


def _maybe_free_memory_pool(threshold_fraction: float = 0.8, min_free_bytes: int = 512 * 1024**2) -> None:
    """Release cached GPU memory when the pool grows too large."""
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        if total_bytes <= 0:
            return
        if free_bytes < min_free_bytes or used_bytes > total_bytes * threshold_fraction:
            cp.cuda.get_current_stream().synchronize()
            mempool.free_all_blocks()
    except Exception:
        return


def _normalize_sparse_influence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a CSR influence payload to CuPy arrays."""
    if not isinstance(payload, dict) or payload.get("format") != "csr":
        raise ValueError("Sparse influence payload must be a dict with format='csr'.")
    required = ("indptr", "indices", "data", "shape")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Sparse influence payload missing keys: {missing}")
    indptr = cp.asarray(payload["indptr"], dtype=cp.int32)
    indices = cp.asarray(payload["indices"], dtype=cp.int32)
    data = cp.asarray(payload["data"], dtype=cp.float32)
    shape = tuple(payload["shape"])
    if len(shape) != 2:
        raise ValueError(f"Sparse influence payload shape must be 2D, got {shape}")
    return {
        "format": "csr",
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "shape": shape,
    }


def _ensure_sparse_influence_csr(payload: Any) -> Any:
    """Return a CuPy CSR matrix from payload or pass through if already CSR."""
    if hasattr(payload, "format") and getattr(payload, "format", None) == "csr":
        return payload
    normalized = _normalize_sparse_influence_payload(payload)
    from cupyx.scipy import sparse as cusparse
    return cusparse.csr_matrix(
        (normalized["data"], normalized["indices"], normalized["indptr"]),
        shape=normalized["shape"],
    )


def compute_weight_updates(
    batch: cp.ndarray,
    bmus: cp.ndarray,
    weights: cp.ndarray,
    influence_matrix: cp.ndarray,
    learning_rate: float,
    apply_lr_in_update: bool = True,
    chunk_size: int = 1000,
    verbose: bool = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute raw accumulated updates and influence.
    Memory-efficient processing that chunks the influence matrix access.
    
    Args:
        batch: Input samples (batch_size, input_dim)
        bmus: Best matching units (batch_size,)
        weights: Current SOM weights (total_nodes, input_dim)
        influence_matrix: Precomputed influence matrix (total_nodes, total_nodes)
        learning_rate: Learning rate
        chunk_size: Chunk size for processing (smaller = less memory)
        verbose: Print progress information
        
    Returns:
        tuple: (accumulated_updates, accumulated_influence)
    """
    # Assert that inputs are already CuPy arrays
    assert isinstance(batch, cp.ndarray), "batch must be a CuPy array"
    assert isinstance(bmus, cp.ndarray), "bmus must be a CuPy array"
    assert isinstance(weights, cp.ndarray), "weights must be a CuPy array"
    assert isinstance(influence_matrix, cp.ndarray), "influence_matrix must be a CuPy array"
    
    total_nodes = weights.shape[0]
    original_batch_size = batch.shape[0]
    
    # Initialize accumulators using weights dtype for speed (default float32)
    compute_dtype = weights.dtype
    accumulated_summed_updates = cp.zeros_like(weights, dtype=compute_dtype)
    accumulated_influence_sum = cp.zeros(total_nodes, dtype=compute_dtype)
    
    # Process updates in chunks for memory efficiency
    num_chunks = (original_batch_size + chunk_size - 1) // chunk_size
    
    if verbose and num_chunks > 1:
        logger.debug(f"Processing {original_batch_size} samples in {num_chunks} chunks of {chunk_size}")
        
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, original_batch_size)
        
        batch_chunk = batch[start_idx:end_idx]
        bmus_chunk = bmus[start_idx:end_idx]
        
        # Get influence ONLY for this chunk's BMUs
        chunk_influence = influence_matrix[bmus_chunk]  # (chunk_size, n_nodes)
        
        if apply_lr_in_update:
            # Calculate partial delta updates for this chunk
            summed_updates_for_chunk, summed_influence_for_chunk = _calculate_partial_updates_and_influence_gpu(
                batch_chunk,
                bmus_chunk,
                chunk_influence,
                weights,
                learning_rate,
                apply_lr_in_update=True,
            )
        else:
            # XPySOM semantics: accumulate raw numerators/denominators and
            # merge later as weights_new = numerator / denominator.
            summed_updates_for_chunk = chunk_influence.T @ batch_chunk
            summed_influence_for_chunk = cp.sum(chunk_influence, axis=0)
        
        # Accumulate
        accumulated_summed_updates += summed_updates_for_chunk
        accumulated_influence_sum += summed_influence_for_chunk
        
        # Free chunk influence immediately
        del chunk_influence
        # Reduce synchronization overhead by freeing memory periodically
        if (i + 1) % 10 == 0:
            _maybe_free_memory_pool()
    
    # Final cleanup after all chunks
    _maybe_free_memory_pool()
    return accumulated_summed_updates, accumulated_influence_sum


def compute_weight_updates_sparse(
    batch: cp.ndarray,
    bmus: cp.ndarray,
    weights: cp.ndarray,
    influence_matrix: Any,
    learning_rate: float,
    apply_lr_in_update: bool = True,
    verbose: bool = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute raw accumulated updates and influence using sparse influence maps.

    Args:
        batch: Input samples (batch_size, input_dim)
        bmus: Best matching units (batch_size,)
        weights: Current SOM weights (total_nodes, input_dim)
        influence_matrix: Sparse influence CSR payload or CSR matrix
        learning_rate: Learning rate
        apply_lr_in_update: Apply learning rate inside update
        verbose: Print debug info

    Returns:
        tuple: (accumulated_updates, accumulated_influence)
    """
    assert isinstance(batch, cp.ndarray), "batch must be a CuPy array"
    assert isinstance(bmus, cp.ndarray), "bmus must be a CuPy array"
    assert isinstance(weights, cp.ndarray), "weights must be a CuPy array"

    total_nodes = weights.shape[0]
    compute_dtype = weights.dtype

    # Aggregate per-BMU counts and sums for the batch chunk.
    counts = cp.bincount(bmus, minlength=total_nodes).astype(compute_dtype, copy=False)
    summed_samples = cp.zeros((total_nodes, batch.shape[1]), dtype=compute_dtype)
    cp.add.at(summed_samples, bmus, batch)

    influence_csr = _ensure_sparse_influence_csr(influence_matrix)

    # Influence propagation (G.T @ counts/sums).
    influence_sum = influence_csr.T @ counts
    weighted_sum = influence_csr.T @ summed_samples

    if apply_lr_in_update:
        updates = float(learning_rate) * (weighted_sum - weights * influence_sum[:, None])
    else:
        updates = weighted_sum

    return updates, influence_sum


def apply_momentum(
    normalized_updates: cp.ndarray,
    momentum_coefficient: float,
    delta_weights: Optional[cp.ndarray] = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply momentum to normalized updates.
    
    Args:
        normalized_updates: Normalized weight updates
        momentum_coefficient: Momentum coefficient (0.0 = no momentum)
        delta_weights: Previous weight changes for momentum
        
    Returns:
        tuple: (weight_changes, new_delta_weights)
    """
    if momentum_coefficient > 0 and delta_weights is not None:
        # Add momentum term: m(t) * ΔW(t-1)
        weight_changes = normalized_updates + momentum_coefficient * delta_weights
        update_mask = cp.abs(weight_changes) > 1e-10
        new_delta_weights = delta_weights.copy()
        new_delta_weights[update_mask] = weight_changes[update_mask]
    else:
        weight_changes = normalized_updates
        new_delta_weights = normalized_updates.copy()
    return weight_changes, new_delta_weights

from .processing_params import TrainingStepParams
def calculate_updates(
    samples: cp.ndarray,
    som_weights: cp.ndarray,
    topology,
    chunk_size: int ,
    params: TrainingStepParams,
    verbose: bool = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Shared algorithm logic for calculating normalized weight updates.
    This encapsulates the core SOM update algorithm used by both single-GPU and multi-GPU implementations.
    
    Args:
        samples: Input samples (n_samples, n_features)
        som_weights: Current SOM weights (n_nodes, n_features)
        topology: SOMTopology instance
        params: TrainingStepParams with all training parameters
        chunk_size: Chunk size for memory-efficient processing
        verbose: Print progress information
        
    Returns:
        Tuple of (accumulated_updates, accumulated_influence) before normalization
    """
    n_samples = samples.shape[0]
    
    # Step 1: Find all BMUs
    all_bmus = cp.empty(n_samples, dtype=cp.int32)
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunk_samples = samples[i:end_idx]
        chunk_bmus, _ = find_bmus(chunk_samples, som_weights, return_distances=True)
        all_bmus[i:end_idx] = chunk_bmus
    
    # Step 2: Get influence matrix from topology
    influence_matrix = topology.get_precomputed_influence_matrix(params.radius, 'gaussian')
    
    # Step 3: Calculate raw accumulated updates and influence
    if params.use_sparse_influence:
        accumulated_updates, accumulated_influence = compute_weight_updates_sparse(
            batch=samples,
            bmus=all_bmus,
            weights=som_weights,
            influence_matrix=influence_matrix,
            learning_rate=params.learning_rate,
            apply_lr_in_update=(params.normalization != "xpysom"),
            verbose=verbose
        )
    else:
        accumulated_updates, accumulated_influence = compute_weight_updates(
            batch=samples,
            bmus=all_bmus,
            weights=som_weights,
            influence_matrix=influence_matrix,
            learning_rate=params.learning_rate,
            apply_lr_in_update=(params.normalization != "xpysom"),
            chunk_size=min(chunk_size, 1000),
            verbose=verbose
        )
    
    return accumulated_updates, accumulated_influence

def apply_weight_updates_with_momentum(
    som_weights: cp.ndarray,
    accumulated_updates: cp.ndarray,
    accumulated_influence: cp.ndarray,
    params: 'TrainingStepParams',
    verbose: bool = False
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply normalization and momentum to weight updates.
    
    Args:
        som_weights: Current SOM weights (n_nodes, n_features)
        accumulated_updates: Raw accumulated updates
        accumulated_influence: Accumulated influence values
        params: TrainingStepParams with all parameters
        verbose: Print warnings for extreme values
        
    Returns:
        Tuple of (updated_weights, new_delta_weights)
    """
    # Apply normalization
    normalized_updates = apply_normalization(
        accumulated_summed_updates=accumulated_updates,
        accumulated_influence_sum=accumulated_influence,
        original_batch_size=params.total_samples,
        normalization=params.normalization,
        learning_rate=params.learning_rate,
        current_weights=som_weights,
        full_bmu_counts=None,
        norm_alpha=params.norm_alpha,
        norm_clamp_factor=params.norm_clamp_factor,
        norm_percentile=params.norm_percentile,
        norm_max_update_threshold=params.norm_max_update_threshold,
        training_progress=params.training_progress,
        current_epoch=params.current_epoch,
        total_epochs=params.total_epochs,
        virtual_ratio=params.virtual_ratio,
        verbose=verbose
    )
    
    if params.momentum > 0:
        weight_changes, new_delta_weights = apply_momentum(
            normalized_updates=normalized_updates,
            momentum_coefficient=params.momentum,
            delta_weights=params.delta_weights
        )
    else:
        weight_changes = normalized_updates
        new_delta_weights = normalized_updates.copy()
    
    # Apply weight changes
    updated_weights = som_weights + weight_changes
    
    return updated_weights, new_delta_weights


def update_weights_vectorized(data: cp.ndarray, bmus: cp.ndarray, influence_matrix: cp.ndarray, 
                             weights: cp.ndarray, 
                             training_params: 'TrainingStepParams',
                             chunk_size: int = 1000,
                             verbose: bool = False, 
                             selector_callback=None) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Vectorized weight update for batch of samples using optimized parallel processing.
    Uses the full influence matrix and lets compute_weight_updates handle efficient chunked lookups.
    
    Args:
        data: Input samples (batch_size, input_dim)
        bmus: Best matching units (batch_size,)
        influence_matrix: Full influence matrix (total_nodes, total_nodes)
        weights: Current SOM weights (total_nodes, input_dim)
        training_params: TrainingStepParams object containing all training parameters
        chunk_size: Chunk size for processing
        verbose: Print progress information
        selector_callback: Optional callback function for selector metadata updates
        
    Returns:
        tuple: (updated_weights, new_delta_weights)
    """
    # Extract all parameters from TrainingStepParams
    learning_rate = training_params.learning_rate
    normalization = training_params.normalization
    momentum_coefficient = training_params.momentum
    delta_weights = training_params.delta_weights
    norm_alpha = training_params.norm_alpha
    norm_clamp_factor = training_params.norm_clamp_factor
    norm_percentile = training_params.norm_percentile
    norm_max_update_threshold = training_params.norm_max_update_threshold
    training_progress = training_params.training_progress
    current_epoch = training_params.current_epoch
    total_epochs = training_params.total_epochs
    virtual_ratio = training_params.virtual_ratio
    total_color_set_samples = training_params.total_samples
    
    # Set defaults for parameters not provided in training_params
    if momentum_coefficient is None:
        momentum_coefficient = 0.0
    if virtual_ratio is None:
        virtual_ratio = 0.5
        
    # Assert that inputs are already CuPy arrays to catch issues early
    assert isinstance(data, cp.ndarray), "data must be a CuPy array"
    assert isinstance(bmus, cp.ndarray), "bmus must be a CuPy array"
    assert isinstance(weights, cp.ndarray), "weights must be a CuPy array"

    if training_params.use_sparse_influence:
        if isinstance(influence_matrix, cp.ndarray):
            raise ValueError("Sparse influence requires CSR payload, not dense CuPy array.")
    else:
        if isinstance(influence_matrix, dict):
            raise ValueError("Dense influence expected; disable use_sparse_influence for CSR payloads.")
        assert isinstance(influence_matrix, cp.ndarray), "influence_matrix must be a CuPy array"
    
    # Validate influence matrix is the full (nodes, nodes) matrix
    total_nodes = weights.shape[0]
    if influence_matrix.shape != (total_nodes, total_nodes):
        raise ValueError(f"influence_matrix shape {influence_matrix.shape} should be ({total_nodes}, {total_nodes})")
    
    # Step 1: Use compute_weight_updates which efficiently handles the full influence matrix
    apply_lr_in_update = normalization != "xpysom"
    if training_params.use_sparse_influence:
        accumulated_summed_updates, accumulated_influence_sum = compute_weight_updates_sparse(
            batch=data,
            bmus=bmus,
            weights=weights,
            influence_matrix=influence_matrix,
            learning_rate=learning_rate,
            apply_lr_in_update=apply_lr_in_update,
            verbose=verbose
        )
    else:
        accumulated_summed_updates, accumulated_influence_sum = compute_weight_updates(
            batch=data,
            bmus=bmus,
            weights=weights,
            influence_matrix=influence_matrix,  # Pass the full matrix directly
            learning_rate=learning_rate,
            apply_lr_in_update=apply_lr_in_update,
            chunk_size=chunk_size,
            verbose=verbose
        )
    
    # Pre-calculate full_bmu_counts for "minisom_weighted" normalization
    full_bmu_counts = None
    if normalization == "minisom_weighted":
        full_bmu_counts = cp.zeros(total_nodes, dtype=cp.int32)
        unique_bmus_total, counts_total = cp.unique(bmus, return_counts=True)
        full_bmu_counts[unique_bmus_total] = counts_total
    
    # Step 2: Apply normalization
    # Use total color set samples for normalization (not just the chunk size)
    normalized_updates = apply_normalization(
        accumulated_summed_updates=accumulated_summed_updates,
        accumulated_influence_sum=accumulated_influence_sum,
        original_batch_size=total_color_set_samples,
        normalization=normalization,
        learning_rate=learning_rate,
        current_weights=weights,
        full_bmu_counts=full_bmu_counts,
        norm_alpha=norm_alpha,
        norm_clamp_factor=norm_clamp_factor,
        norm_percentile=norm_percentile,
        norm_max_update_threshold=norm_max_update_threshold,
        training_progress=training_progress,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        virtual_ratio=virtual_ratio,
        verbose=verbose
    )
    
    # Step 3: Apply momentum
    weight_changes, new_delta_weights = apply_momentum(
        normalized_updates=normalized_updates,
        momentum_coefficient=momentum_coefficient,
        delta_weights=delta_weights
    )
    
    # Step 4: Apply weight changes
    updated_weights = weights + weight_changes
    
    # Validate weights
    if not check_weights_validity(updated_weights, verbose):
        if verbose:
            logger.error("Weight update produced invalid values!")
        # Restore original weights on error
        updated_weights = weights.copy()
        # Reset momentum on error
        new_delta_weights = cp.zeros_like(weights)
    
    # Force memory cleanup
    _maybe_free_memory_pool()
    
    return updated_weights, new_delta_weights
