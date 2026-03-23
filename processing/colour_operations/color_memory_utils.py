"""
Memory utilities specific to color set processing
Provides accurate memory estimation and optimal chunk size calculation
"""

import cupy as cp
from typing import Tuple, Optional


def get_available_gpu_memory() -> int:
    """
    Get available GPU memory in bytes
    
    Returns:
        Available GPU memory in bytes
    """
    mempool = cp.get_default_memory_pool()
    free_mem = cp.cuda.Device().mem_info[0]
    used_mem = mempool.used_bytes()
    available_mem = free_mem - used_mem
    return available_mem


def estimate_color_set_memory_usage(color_set_samples: cp.ndarray, 
                                  som_weights: cp.ndarray,
                                  processing_mode: str = "equal_sized") -> float:
    """
    Estimate GPU memory needed for color set processing
    
    Memory components:
    1. Influence matrix: n_samples × n_nodes × 4 bytes (float32)
    2. BMU calculations: temporary storage during distance computation
    3. Weight update accumulation: n_nodes × n_features × 8 bytes (float64)
    4. Color set specific overhead (scheduling, metadata)
    """
    n_samples, n_features = color_set_samples.shape
    n_nodes = som_weights.shape[0]
    
    # Core memory components
    influence_memory = n_samples * n_nodes * 4  # float32 influence matrix
    update_memory = n_nodes * n_features * 8  # float64 for accumulation
    bmu_temp_memory = max(influence_memory * 0.5, update_memory * 2)
    
    # Color set specific overhead
    overhead_memory = calculate_color_processing_overhead(
        processing_mode, n_samples, n_nodes, n_features
    )
    
    total_memory = influence_memory + update_memory + bmu_temp_memory + overhead_memory
    return total_memory


def calculate_color_processing_overhead(processing_mode: str, 
                                       n_samples: int, 
                                       n_nodes: int,
                                       n_features: int) -> float:
    """
    Calculate memory overhead specific to color set processing
    
    Args:
        processing_mode: 'equal_sized' or 'batch_all'
        n_samples: Number of samples in color set
        n_nodes: Number of SOM nodes
        n_features: Number of features
        
    Returns:
        Overhead memory in bytes
    """
    # Base overhead for masks and indices
    base_overhead = n_samples * 5  # boolean mask + index storage
    
    # Processing mode specific overhead
    if processing_mode == "equal_sized":
        # Round allocation overhead
        round_overhead = n_samples * 4  # round indices
        # Color set membership tracking
        color_membership = n_samples * 4  # which color each sample belongs to
        overhead = base_overhead + round_overhead + color_membership
    else:
        # Batch-all has simpler overhead
        overhead = base_overhead
    
    # Add scheduler metadata if adaptive BMU is enabled
    scheduler_overhead = n_samples * 8  # tracking processed samples
    
    # GPU memory pool overhead (typically 10-20% extra)
    total_overhead = (overhead + scheduler_overhead) * 1.2
    
    return total_overhead


# NOTE: determine_optimal_color_chunk_size has been moved to ChunkingConfig in processing_params.py
# Use ChunkingConfig(processing_type="colors", ...).calculate_optimal_chunk_size() instead


def validate_memory_configuration(n_samples: int, 
                                 n_features: int,
                                 n_nodes: int,
                                 chunk_size: int,
                                 available_memory: Optional[float] = None) -> dict:
    """
    Validate memory configuration and provide recommendations
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features per sample
        n_nodes: Number of SOM nodes
        chunk_size: Proposed chunk size
        available_memory: Available GPU memory (will query if None)
    
    Returns:
        Dictionary with validation results and recommendations
    """
    if available_memory is None:
        free_mem, total_mem = cp.cuda.Device().mem_info
        available_memory = free_mem
    
    # Calculate memory requirements
    chunk_memory = estimate_chunk_memory_usage(chunk_size, n_features, n_nodes)
    total_memory = estimate_total_memory_usage(n_samples, n_features, n_nodes)
    
    # Validation results
    results = {
        'valid': True,
        'chunk_memory': chunk_memory,
        'total_memory': total_memory,
        'available_memory': available_memory,
        'recommendations': []
    }
    
    # Check if chunk size is appropriate
    if chunk_memory > available_memory * 0.8:
        results['valid'] = False
        results['recommendations'].append(
            f"Chunk size too large. Reduce to {int(available_memory * 0.6 / (n_nodes * 4 + n_features * 8))}"
        )
    
    if chunk_size < 50:
        results['recommendations'].append(
            "Chunk size very small. May impact performance."
        )
    
    if chunk_size > n_samples:
        results['recommendations'].append(
            f"Chunk size larger than sample count. Consider reducing to {n_samples}"
        )
    
    # Check if multiple chunks will be needed
    num_chunks = (n_samples + chunk_size - 1) // chunk_size
    if num_chunks > 100:
        results['recommendations'].append(
            f"Will require {num_chunks} chunks. Consider increasing chunk size for efficiency."
        )
    
    results['num_chunks'] = num_chunks
    results['efficiency'] = min(1.0, chunk_size / 1000)  # Efficiency score
    
    return results


def estimate_chunk_memory_usage(chunk_size: int, n_features: int, n_nodes: int) -> float:
    """
    Estimate memory usage for a single chunk
    
    Args:
        chunk_size: Number of samples in chunk
        n_features: Number of features
        n_nodes: Number of SOM nodes
    
    Returns:
        Memory usage in bytes
    """
    # Data storage
    data_memory = chunk_size * n_features * 4  # float32
    
    # BMU storage
    bmu_memory = chunk_size * 4  # int32 indices
    
    # Influence matrix for chunk
    influence_memory = chunk_size * n_nodes * 4  # float32
    
    # Weight updates
    update_memory = n_nodes * n_features * 4  # float32
    
    # Temporary buffers
    temp_memory = max(data_memory, influence_memory) * 0.5
    
    return data_memory + bmu_memory + influence_memory + update_memory + temp_memory


def estimate_total_memory_usage(n_samples: int, n_features: int, n_nodes: int) -> float:
    """
    Estimate total memory usage for processing all samples
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_nodes: Number of SOM nodes
    
    Returns:
        Total memory usage in bytes
    """
    # All samples data
    data_memory = n_samples * n_features * 4
    
    # All BMUs
    bmu_memory = n_samples * 4
    
    # Full influence matrix (if not chunked)
    influence_memory = n_samples * n_nodes * 4
    
    # Weight matrix
    weight_memory = n_nodes * n_features * 4
    
    # Overhead
    overhead = (data_memory + bmu_memory) * 0.2
    
    return data_memory + bmu_memory + influence_memory + weight_memory + overhead


def calculate_distributed_memory_requirements(n_samples_total: int,
                                            n_features: int,
                                            n_nodes: int,
                                            n_gpus: int,
                                            max_rounds: int = 4) -> dict:
    """
    Calculate memory requirements for distributed multi-GPU processing
    
    Args:
        n_samples_total: Total samples across all GPUs
        n_features: Number of features
        n_nodes: Number of SOM nodes  
        n_gpus: Number of GPUs
        max_rounds: Number of processing rounds
    
    Returns:
        Dictionary with per-GPU and total memory requirements
    """
    # Samples per GPU
    samples_per_gpu = n_samples_total // n_gpus
    
    # Samples per round per GPU
    samples_per_round = samples_per_gpu // max_rounds
    
    # Per-GPU memory requirements
    per_gpu_data = samples_per_gpu * n_features * 4  # Local data partition
    per_gpu_bmus = samples_per_gpu * 4  # Local BMUs
    per_gpu_weights = n_nodes * n_features * 4  # Weight matrix (replicated)
    per_gpu_influence = n_nodes * n_nodes * 4  # Influence matrix (replicated)
    
    # Peak memory during color set processing (worst case)
    # Assuming largest color set could have up to 50% of round samples
    max_color_samples = samples_per_round * 0.5
    color_set_memory = estimate_chunk_memory_usage(
        int(max_color_samples), n_features, n_nodes
    )
    
    # Communication buffers for NCCL
    comm_buffer = n_nodes * n_features * 4 * 2  # Double buffer for async
    
    per_gpu_total = (
        per_gpu_data + 
        per_gpu_bmus + 
        per_gpu_weights + 
        per_gpu_influence +
        color_set_memory +
        comm_buffer
    )
    
    return {
        'per_gpu': {
            'data': per_gpu_data,
            'bmus': per_gpu_bmus,
            'weights': per_gpu_weights,
            'influence': per_gpu_influence,
            'color_processing': color_set_memory,
            'communication': comm_buffer,
            'total': per_gpu_total
        },
        'total_system': per_gpu_total * n_gpus,
        'samples_per_gpu': samples_per_gpu,
        'samples_per_round': samples_per_round,
        'max_color_samples': int(max_color_samples)
    }
