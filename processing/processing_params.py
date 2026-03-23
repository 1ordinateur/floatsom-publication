"""
Processing-specific parameter dataclasses
Centralizes all processor configurations in the processing module
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List, Tuple
import warnings
import cupy as cp


_AUTO_CHUNK_BATCH_BASE_SAMPLES = 500_000
_AUTO_CHUNK_COLORS_BASE_SAMPLES = 500_000
_AUTO_CHUNK_BASE_DIMENSION = 50
_AUTO_CHUNK_MIN_SAMPLES = 10_000
_AUTO_CHUNK_REFERENCE_VRAM_MIB = 32 * 1024

# Default number of GPU transfer buffers to allocate for overlapped CPU->GPU
# chunk transfers. Values <= 1 disable buffering.
DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS = 3


def get_visible_gpu_vram_mib() -> int:
    """Return minimum visible GPU VRAM (MiB) across visible CUDA devices."""
    device_count = int(cp.cuda.runtime.getDeviceCount())

    if device_count <= 0:
        raise RuntimeError("No visible CUDA devices detected while resolving VRAM")

    previous_device = None
    try:
        previous_device = int(cp.cuda.runtime.getDevice())
    except Exception:
        previous_device = None

    min_total_mib = None
    for device_idx in range(device_count):
        try:
            cp.cuda.runtime.setDevice(device_idx)
            _, total_bytes = cp.cuda.runtime.memGetInfo()
            total_mib = int(total_bytes // (1024 ** 2))
        except Exception:
            continue
        if total_mib <= 0:
            continue
        if min_total_mib is None or total_mib < min_total_mib:
            min_total_mib = total_mib

    if previous_device is not None:
        try:
            cp.cuda.runtime.setDevice(previous_device)
        except Exception:
            pass

    if min_total_mib is None:
        raise RuntimeError("Failed to query VRAM for all visible CUDA devices")

    return max(1, int(min_total_mib))

 
def validate_multi_buffering_buffers(num_buffers: int) -> int:
    """Validate multi-buffering configuration (number of buffers)."""
    if isinstance(num_buffers, bool):
        mapped = DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS if num_buffers else 0
        warnings.warn(
            "enable_multi_buffering expects an integer buffer count; bool values are deprecated. "
            f"Mapping {num_buffers!r} -> {mapped}.",
            UserWarning,
        )
        return mapped
    if not isinstance(num_buffers, int):
        raise ValueError("enable_multi_buffering must be an integer")
    if num_buffers < 0:
        raise ValueError("enable_multi_buffering must be >= 0")
    return int(num_buffers)


def _validate_auto_chunk_input(input_dim: int) -> None:
    if not isinstance(input_dim, int):
        raise ValueError("input_dim must be an integer")
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")


def _calculate_auto_chunk_size(
    input_dim: int,
    base_samples: int,
    visible_vram_mib: Optional[int] = None,
) -> int:
    _validate_auto_chunk_input(input_dim)

    resolved_vram_mib = (
        int(visible_vram_mib)
        if visible_vram_mib is not None
        else get_visible_gpu_vram_mib()
    )
    if resolved_vram_mib <= 0:
        raise ValueError(f"visible_vram_mib must be > 0, got {resolved_vram_mib}")

    # Scale the baseline chunk cap linearly with available VRAM.
    vram_scaled_base = int(
        round(base_samples * resolved_vram_mib / _AUTO_CHUNK_REFERENCE_VRAM_MIB)
    )
    vram_scaled_base = max(_AUTO_CHUNK_MIN_SAMPLES, vram_scaled_base)

    # Keep the dimension-based scaling, then apply the VRAM-scaled cap.
    dim_scaled = int(round(vram_scaled_base * _AUTO_CHUNK_BASE_DIMENSION / input_dim))
    return max(_AUTO_CHUNK_MIN_SAMPLES, min(vram_scaled_base, dim_scaled))


def calculate_auto_chunk_size_for_method(input_dim: int, processing_method: Optional[str]) -> int:
    """Return the auto-scaled chunk size based on processing method."""
    method = (processing_method or "").lower()
    base_samples = (
        _AUTO_CHUNK_COLORS_BASE_SAMPLES
        if method == "colors"
        else _AUTO_CHUNK_BATCH_BASE_SAMPLES
    )
    return _calculate_auto_chunk_size(input_dim, base_samples)


@dataclass
class TrainingStepParams:
    """Typed parameters for a single training step/iteration"""
    radius: float
    learning_rate: float
    momentum: float = 0.0
    delta_weights: Optional[Any] = None  # cp.ndarray or None
    normalization: str = "xpysom"
    total_samples: int = 0
    norm_alpha: Optional[float] = None
    norm_clamp_factor: Optional[float] = None
    norm_percentile: Optional[float] = None
    norm_max_update_threshold: Optional[float] = None
    training_progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    virtual_ratio: float = 0.5
    use_sparse_influence: bool = False
    distance_metric: Optional[Any] = None  # DistanceMetric instance
    neighborhood_function: Optional[Any] = None  # NeighborhoodFunction instance

# ==================== CHUNKING CONFIGURATION ====================

@dataclass
class ChunkingConfig:
    """Configuration for chunk processing with fixed chunk size"""
    chunk_size: int  # Must be explicitly provided
    dtype: type = cp.float32  # Data type for calculations

# ==================== BATCH PROCESSING ====================

@dataclass
class BatchConfig:
    """Configuration specifically for batch processing"""
    batch_mode: str = "full_batch"  # Options: "full_batch", "minibatch"
    chunk_size: Optional[int] = None  # Must be provided explicitly (also used as minibatch size in minibatch mode)
    weight_update_frequency: int = 10  # Accumulate updates every N chunks to reduce memory overhead

# ==================== COLORS PROCESSING ====================

@dataclass
class ColorsConfig:
    """Configuration specifically for colors processing
    
    IMPORTANT CONSTRAINT: For performance, ensure chunk_size >= total_samples/max_rounds
    This ensures num_chunks <= max_rounds, preventing excessive chunk reloading.
    """
    processing_mode: str = "equal_sized"  # "equal_sized" or "batch_all"
    max_rounds: int = 4
    sample_order: str = "random"  # "strided" or "random"
    color_set_algorithm: str = "greedy_balanced"
    num_color_sets: Optional[int] = None
    chunk_size: Optional[int] = None  # Must be provided explicitly
                                      # Minimum: total_samples/max_rounds (enforced at runtime)
    
    # Adaptive BMU parameters
    enable_adaptive_bmu: bool = True
    bmu_recalc_initial: Optional[int] = None
    bmu_recalc_decay_type: str = 'exponential'
    bmu_recalc_min_samples: int = 1000
    recompute_bmus_per_color_set: bool = False

@dataclass
class ColorStepParams:
    """Parameters for a single color processing step"""
    radius: float
    learning_rate: float
    momentum: float = 0.0
    delta_weights: Optional[Any] = None
    color_sets: List[Any] = field(default_factory=list)
    round_number: int = 0
    total_rounds: int = 1
    normalization: str = "xpysom"
    total_samples: int = 0
    # Include normalization parameters
    norm_alpha: Optional[float] = None
    norm_clamp_factor: Optional[float] = None
    virtual_ratio: float = 0.5
    norm_percentile: Optional[float] = None
    norm_max_update_threshold: Optional[float] = None
    training_progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    distance_metric: Optional[Any] = None  # DistanceMetric instance
    neighborhood_function: Optional[Any] = None  # NeighborhoodFunction instance

# ==================== WORKER PROFILING CONFIGURATION ====================

@dataclass
class WorkerProfileConfig:
    """Configuration for Ray worker-side CPU profiling."""
    enabled: bool = False
    output_dir: Optional[str] = None  # Directory to write profile outputs.
    max_stats: int = 50  # Number of profile entries to emit.

# ==================== CLEANUP CONFIGURATION ====================

@dataclass
class CleanupConfig:
    """Configuration for Ray cleanup behavior."""
    safe_cleanup: bool = True  # If True, wait for workers to finish cleanup.

# ==================== ASYNC LOADING CONFIGURATION ====================

@dataclass
class AsyncLoadingConfig:
    """Configuration for asynchronous data loading with safety guarantees.
    
    Async loading allows processing to begin while data is still being
    transferred to worker memory, significantly reducing startup latency
    for large datasets.
    """
    # Core settings - async enabled by default for performance
    enable_async: bool = True  # Default ON for better performance
    
    # Chunk loading parameters
    initial_chunks_per_worker: int = 2  # Start processing after this many chunks ready
    max_chunks_in_memory: int = 5  # Maximum chunks to keep in RAM per worker
    prefetch_ahead: int = 5  # How many chunks to prefetch ahead of current
    
    # Safety and validation
    chunk_validation: bool = True  # Verify chunk completeness before use
    require_contiguous: bool = True  # Ensure chunks are contiguous in memory
    validate_checksum: bool = False  # Optional checksum validation (slower)
    
    # Timing and retry behavior
    chunk_ready_timeout: float = 3000.0  # Timeout waiting for chunk (seconds)
    retry_failed_chunks: int = 3  # Number of retries for failed chunk loads
    
    # Progress monitoring
    enable_progress_reporting: bool = False  # Disable progress monitor by default to avoid overhead
    progress_interval: float = 5.0  # Progress reporting interval (seconds)
    
    # Performance tuning
    loading_thread_priority: int = 0  # Thread priority (-20 to 19, 0 is normal)
    use_pinned_memory: bool = True  # Use CUDA pinned memory for transfers
    parallel_chunk_loads: int = 2  # Number of chunks to load in parallel
    
    # Adaptive behavior
    auto_disable_threshold_gb: float = 0.5  # Auto-disable for datasets < this size
    auto_adjust_chunk_count: bool = True  # Adjust initial chunks based on data size
    
    def should_use_async(self, data_size_gb: float) -> bool:
        """Determine if async loading should be used based on data size."""
        if not self.enable_async:
            return False
        # Auto-disable for small datasets where overhead isn't worth it
        if data_size_gb < self.auto_disable_threshold_gb:
            return False
        return True
    
    def get_initial_chunks(self, total_chunks: int) -> int:
        """Get number of initial chunks to load based on total chunks."""
        if not self.auto_adjust_chunk_count:
            return self.initial_chunks_per_worker
        
        # Adaptive: load more initial chunks for smaller datasets
        if total_chunks <= 5:
            return min(2, total_chunks)  # Load 2 or all
        elif total_chunks <= 10:
            return 3
        else:
            return self.initial_chunks_per_worker
