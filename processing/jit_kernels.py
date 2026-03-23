"""
JIT-compiled kernels for FloatSOM using CuPy's JIT compilation.
Provides optimized GPU kernels for performance-critical operations.
"""

import cupy as cp
from cupyx import jit
import logging
import math

logger = logging.getLogger(__name__)


# ==================== DISTANCE METRIC KERNELS ====================

@jit.rawkernel()
def euclidean_distance_kernel(batch, weights, distances, batch_size, num_nodes, input_dim):
        """
        JIT-compiled Euclidean distance calculation kernel.
        Computes squared Euclidean distances between batch samples and weight vectors.
        
        Grid: (batch_size, num_nodes)
        Block: Configured based on hardware
        """
        sample_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        node_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
        
        if sample_idx < batch_size and node_idx < num_nodes:
            dist = 0.0
            for d in range(input_dim):
                diff = batch[sample_idx, d] - weights[node_idx, d]
                dist += diff * diff
            
            # Store squared distance (sqrt applied later if needed)
            distances[sample_idx, node_idx] = dist


@jit.rawkernel()
def euclidean_distance_sqrt_kernel(distances, size):
    """
    Apply square root to squared distances in-place.
    Separate kernel to allow optional sqrt application.
    """
    idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if idx < size:
        distances[idx] = math.sqrt(distances[idx])


@jit.rawkernel()
def manhattan_distance_kernel(batch, weights, distances, batch_size, num_nodes, input_dim):
    """
    JIT-compiled Manhattan (L1) distance calculation kernel.
    """
    sample_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    node_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    
    if sample_idx < batch_size and node_idx < num_nodes:
        dist = 0.0
        for d in range(input_dim):
            dist += abs(batch[sample_idx, d] - weights[node_idx, d])
        
        distances[sample_idx, node_idx] = dist


@jit.rawkernel()
def cosine_distance_kernel(batch, weights, batch_norms, weight_norms, distances, 
                          batch_size, num_nodes, input_dim):
    """
    JIT-compiled Cosine distance calculation kernel.
    Assumes batch_norms and weight_norms are pre-computed.
    """
    sample_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    node_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    
    if sample_idx < batch_size and node_idx < num_nodes:
        dot_product = 0.0
        for d in range(input_dim):
            dot_product += batch[sample_idx, d] * weights[node_idx, d]
        
        # Normalize and compute cosine distance
        batch_norm = batch_norms[sample_idx]
        weight_norm = weight_norms[node_idx]
        
        # Avoid division by zero
        norm_product = batch_norm * weight_norm
        if norm_product > 1e-8:
            cosine_sim = dot_product / norm_product
            # Cosine distance = 1 - cosine_similarity
            # Clamp to [0, 2] range
            dist = 1.0 - cosine_sim
            if dist < 0.0:
                dist = 0.0
            elif dist > 2.0:
                dist = 2.0
            distances[sample_idx, node_idx] = dist
        else:
            distances[sample_idx, node_idx] = 1.0


@jit.rawkernel()
def norm_p_distance_kernel(batch, weights, distances, batch_size, num_nodes, input_dim, p):
    """
    JIT-compiled Lp norm distance calculation kernel.
    """
    sample_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    node_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    
    if sample_idx < batch_size and node_idx < num_nodes:
        dist = 0.0
        for d in range(input_dim):
            diff = abs(batch[sample_idx, d] - weights[node_idx, d])
            dist += pow(diff, p)
        
        distances[sample_idx, node_idx] = pow(dist, 1.0 / p)


# ==================== BMU FINDING KERNELS ====================

@jit.rawkernel()
def find_bmus_kernel(batch, weights, bmus, min_distances, 
                    batch_size, num_nodes, input_dim):
    """
    Find BMUs with fused distance calculation and argmin.
    Each thread handles one sample and iterates through all nodes.
    """
    sample_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    
    if sample_idx < batch_size:
        min_dist = 1e10  # Large initial value
        min_idx = 0
        
        # Iterate through all nodes for this sample
        for node_idx in range(num_nodes):
            # Calculate Euclidean distance
            dist = 0.0
            for d in range(input_dim):
                diff = batch[sample_idx, d] - weights[node_idx, d]
                dist += diff * diff
            
            # Track minimum
            if dist < min_dist:
                min_dist = dist
                min_idx = node_idx
        
        # Store results
        bmus[sample_idx] = min_idx
        min_distances[sample_idx] = math.sqrt(min_dist)


# ==================== UPDATE CALCULATION KERNELS ====================

@jit.rawkernel()
def partial_updates_kernel(batch, influence, weights, updates, influence_sum,
                          batch_size, num_nodes, input_dim, learning_rate):
    """
    Fused kernel for calculating partial updates and influence sum.
    Replaces multiple einsum operations with a single kernel.
    Each thread handles one node.
    """
    node_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    
    if node_idx < num_nodes:
        # Calculate summed influence for this node
        local_influence_sum = 0.0
        for s in range(batch_size):
            local_influence_sum += influence[s, node_idx]
        
        influence_sum[node_idx] = local_influence_sum
        
        # Calculate updates for each feature dimension
        for d in range(input_dim):
            weighted_sum = 0.0
            
            # Sum of influence-weighted batch samples
            for s in range(batch_size):
                inf_val = influence[s, node_idx]
                batch_val = batch[s, d]
                weighted_sum += inf_val * batch_val
            
            # Current weight scaled by influence sum
            weighted_weight = weights[node_idx, d] * local_influence_sum
            
            # Calculate and store update
            updates[node_idx, d] = learning_rate * (weighted_sum - weighted_weight)


# ==================== HELPER FUNCTIONS ====================

def launch_euclidean_distance_jit(batch: cp.ndarray, weights: cp.ndarray, 
                                 apply_sqrt: bool = True, batch_size: int = None, 
                                 num_nodes: int = None, input_dim: int = None) -> cp.ndarray:
    """
    Launch JIT-compiled Euclidean distance kernel.
    
    Args:
        batch: Input batch (batch_size, input_dim)
        weights: Weight vectors (num_nodes, input_dim)
        apply_sqrt: Whether to apply square root to distances
        batch_size: Batch size
        num_nodes: Number of nodes
        input_dim: Input dimension
        
    Returns:
        Distance matrix (batch_size, num_nodes)
    """
    
    # Allocate output using input dtype
    distances = cp.empty((batch_size, num_nodes), dtype=batch.dtype)
    
    # Configure kernel launch parameters
    block_x = min(16, batch_size)
    block_y = min(16, num_nodes)
    grid_x = (batch_size + block_x - 1) // block_x
    grid_y = (num_nodes + block_y - 1) // block_y
    
    # Launch kernel
    euclidean_distance_kernel[(grid_x, grid_y), (block_x, block_y)](
        batch, weights, distances, batch_size, num_nodes, input_dim
    )
    
    # Apply sqrt if needed (use CuPy ufunc on device)
    if apply_sqrt:
        cp.sqrt(distances, out=distances)
    
    return distances


def launch_manhattan_distance_jit(batch: cp.ndarray, weights: cp.ndarray, batch_size: int = None, 
                                 num_nodes: int = None, input_dim: int = None) -> cp.ndarray:
    """
    Launch JIT-compiled Manhattan distance kernel.
    """
    
    distances = cp.empty((batch_size, num_nodes), dtype=batch.dtype)
    
    block_x = min(16, batch_size)
    block_y = min(16, num_nodes)
    grid_x = (batch_size + block_x - 1) // block_x
    grid_y = (num_nodes + block_y - 1) // block_y
    
    manhattan_distance_kernel[(grid_x, grid_y), (block_x, block_y)](
        batch, weights, distances, batch_size, num_nodes, input_dim
    )
    
    return distances


def launch_cosine_distance_jit(
    batch: cp.ndarray,
    weights: cp.ndarray,
    epsilon: float = 1e-8,
    batch_size: int = None,
    num_nodes: int = None,
    input_dim: int = None,
    batch_norms: cp.ndarray | None = None,
    weight_norms: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Launch JIT-compiled Cosine distance kernel.
    Allows passing precomputed norms to avoid redundant reductions.
    """

    # Pre-compute norms only if not supplied by caller
    if batch_norms is None:
        batch_norms = cp.linalg.norm(batch, axis=1)
    if weight_norms is None:
        weight_norms = cp.linalg.norm(weights, axis=1)

    # Ensure minimum norm values
    batch_norms = cp.maximum(batch_norms, epsilon)
    weight_norms = cp.maximum(weight_norms, epsilon)

    distances = cp.empty((batch_size, num_nodes), dtype=batch.dtype)

    block_x = min(16, batch_size)
    block_y = min(16, num_nodes)
    grid_x = (batch_size + block_x - 1) // block_x
    grid_y = (num_nodes + block_y - 1) // block_y

    cosine_distance_kernel[(grid_x, grid_y), (block_x, block_y)](
        batch, weights, batch_norms, weight_norms, distances,
        batch_size, num_nodes, input_dim
    )

    return distances


def launch_partial_updates_jit(batch: cp.ndarray, influence: cp.ndarray, 
                               weights: cp.ndarray, learning_rate: float) -> tuple:
    """
    Compute partial updates and influence sums on GPU.
    Uses efficient GEMM-based computation for stability and speed.
    
    Returns:
        Tuple of (updates, influence_sum)
    """
    # Use neutral compute dtype (default float32) for speed; callers control input dtypes
    compute_dtype = cp.float32
    batch_f = batch.astype(compute_dtype, copy=False)
    weights_f = weights.astype(compute_dtype, copy=False)
    influence_f = influence.astype(compute_dtype, copy=False)

    # Sum of influences per node: shape (num_nodes,)
    influence_sum = cp.sum(influence_f, axis=0)

    # Influence-weighted sum of batch per node and feature: (num_nodes, input_dim)
    # Equivalent to sum_s(inf[s,n] * batch[s,i])
    weighted_sum = influence_f.T @ batch_f

    # Updates: lr * (weighted_sum - weights * influence_sum[:, None])
    updates = float(learning_rate) * (weighted_sum - weights_f * influence_sum[:, None])

    return updates, influence_sum
