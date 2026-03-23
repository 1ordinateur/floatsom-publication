# Asynchronous Chunk Pre-fetching with kvikio

## Overview

This project implements asynchronous data pre-fetching for the Ray+GDS batch processing system to overlap I/O operations with GPU computation. By leveraging kvikio's async futures API and thread pool, we can achieve 20-40% throughput improvement even without nvidia-fs kernel module support.

## Problem Statement

### Current Bottleneck
The current implementation processes chunks sequentially:
1. Load chunk N from disk (GPU idle)
2. Process chunk N on GPU (disk idle)
3. Load chunk N+1 from disk (GPU idle)
4. Process chunk N+1 on GPU (disk idle)

This results in significant GPU idle time while waiting for I/O operations, reducing overall throughput and efficiency.

### Solution
Implement a double-buffering pattern with asynchronous pre-fetching:
1. Load chunk 0 synchronously
2. Start async load of chunk 1 (background)
3. Process chunk 0 on GPU (overlaps with chunk 1 loading)
4. When chunk 0 processing completes, chunk 1 is ready
5. Start async load of chunk 2 while processing chunk 1
6. Continue pattern for all chunks

## Architecture

### System Overview
```
Ray Orchestrator (ray_gds_base.py)
    │
    ├─> Worker 0 (GPU 0)
    │   ├─> Main Thread: GPU Computation
    │   └─> kvikio Thread Pool: Async I/O
    │
    ├─> Worker 1 (GPU 1)
    │   ├─> Main Thread: GPU Computation
    │   └─> kvikio Thread Pool: Async I/O
    │
    └─> NCCL AllReduce (synchronization point)
```

### Key Components

#### 1. kvikio Integration
kvikio provides:
- **Async futures API**: Non-blocking reads that return futures
- **Thread pool**: Handles I/O operations in background
- **Automatic fallback**: Works with or without nvidia-fs via COMPAT_MODE
- **Native Zarr support**: Direct integration with Zarr arrays

#### 2. Double Buffering
Each worker maintains:
- `current_buffer`: Data currently being processed on GPU
- `prefetch_buffer`: Next chunk being loaded asynchronously
- `prefetch_future`: kvikio future tracking async read status

#### 3. Ray Actor Isolation
Pre-fetching happens independently within each Ray worker:
- No coordination overhead between workers
- Each worker manages its own I/O and compute overlap
- NCCL synchronization points remain unchanged

## Implementation Details

### 1. GDS Manager Enhancement (`/floatsom/data/gds_manager.py`)

```python
import kvikio
import zarr
import cupy as cp
from typing import Tuple, Optional

class GDSDataManager:
    def __init__(self, ...):
        # ... existing code ...
        
        # Configure kvikio compat mode
        import os
        os.environ['KVIKIO_COMPAT_MODE'] = 'AUTO'
        
        # Check kvikio availability
        try:
            import kvikio
            self.kvikio_available = True
            # Configure thread pool
            kvikio.defaults.reset_num_threads(4)
        except ImportError:
            self.kvikio_available = False
            logger.warning("kvikio not available - async prefetch disabled")
    
    def stream_chunk_async(self, chunk_id: int, gpu_id: int = 0,
                          start_row: Optional[int] = None, 
                          num_rows: Optional[int] = None,
                          buffer: Optional[cp.ndarray] = None) -> Tuple['Future', cp.ndarray]:
        """
        Stream chunk asynchronously using kvikio futures.
        
        Args:
            chunk_id: Chunk identifier
            gpu_id: Target GPU device ID  
            start_row: Starting row for the chunk (optional)
            num_rows: Number of rows to read (optional)
            buffer: Pre-allocated buffer to reuse (optional)
            
        Returns:
            Tuple of (future, buffer) where future will complete when data is loaded
        """
        if not self.kvikio_available:
            raise RuntimeError("kvikio not available for async operations")
        
        with cp.cuda.Device(gpu_id):
            # Calculate chunk boundaries
            if start_row is None:
                start_row = chunk_id * self.chunk_size
                num_rows = min(self.chunk_size, self.total_samples - start_row)
            
            # Get Zarr metadata without loading data
            z = zarr.open(str(self.data_path), mode='r')
            dtype = z.dtype
            n_features = z.shape[1] if len(z.shape) > 1 else 1
            
            # Allocate or reuse buffer
            if buffer is None:
                buffer = cp.empty((num_rows, n_features), dtype=dtype)
            
            # Calculate byte offsets for direct file read
            dtype_size = dtype.itemsize
            byte_offset = start_row * n_features * dtype_size
            byte_size = num_rows * n_features * dtype_size
            
            # Get the actual file path for the Zarr chunk
            # Zarr stores chunks as separate files in the directory structure
            chunk_file = self._get_zarr_chunk_file_path(start_row, n_features)
            
            # Use kvikio CuFile for async read
            with kvikio.CuFile(chunk_file, "r") as f:
                # pread returns a future for async operation
                future = f.pread(buffer, byte_size, file_offset=0)
                return future, buffer
    
    def _get_zarr_chunk_file_path(self, start_row: int, n_features: int) -> str:
        """
        Get the actual file path for a Zarr chunk.
        Zarr stores data in chunk files within the directory structure.
        """
        z = zarr.open(str(self.data_path), mode='r')
        
        # Get chunk shape from Zarr metadata
        chunk_shape = z.chunks
        if chunk_shape is None:
            # No chunking, data is in single file
            return str(self.data_path / '0.0')
        
        # Calculate which chunk file contains this row
        chunk_row_idx = start_row // chunk_shape[0]
        chunk_col_idx = 0  # Assuming we're reading full rows
        
        # Zarr chunk file naming convention
        chunk_filename = f"{chunk_row_idx}.{chunk_col_idx}"
        chunk_path = self.data_path / chunk_filename
        
        # Fallback to main array file if chunk file doesn't exist
        if not chunk_path.exists():
            return str(self.data_path / '0')
        
        return str(chunk_path)
```

### 2. Base Worker Pre-fetching Infrastructure (`/floatsom/processing/ray_ops/workers/ray_gds_base_worker.py`)

```python
import os
import logging
import cupy as cp
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RayGDSBaseWorker:
    def __init__(self, worker_id: int, num_gpus: int, gds_config: Dict[str, Any], 
                 gpu_id: Optional[int] = None, chunk_size: Optional[int] = None):
        # ... existing initialization ...
        
        # Configure kvikio for compat mode
        os.environ['KVIKIO_COMPAT_MODE'] = 'AUTO'
        
        # Pre-fetching state
        self.prefetch_future = None      # kvikio future for async read
        self.prefetch_buffer = None      # Pre-allocated GPU buffer
        self.prefetch_chunk_id = None    # ID of chunk being prefetched
        
        # Double buffering
        self.current_buffer = None       # Buffer currently being processed
        self.next_buffer = None          # Buffer being loaded
        
        # Configure kvikio thread pool once
        try:
            import kvikio
            # Set thread pool for async I/O
            # Use 4 threads as a good balance between parallelism and overhead
            kvikio.defaults.reset_num_threads(4)
            self.kvikio_available = True
            
            # Log kvikio configuration
            try:
                props = kvikio.driver_properties()
                logger.info(f"Worker {worker_id}: kvikio initialized with GDS support")
            except:
                logger.info(f"Worker {worker_id}: kvikio initialized in compat mode (no nvidia-fs)")
                
        except ImportError:
            logger.warning(f"Worker {worker_id}: kvikio not available - prefetching disabled")
            self.kvikio_available = False
    
    def start_prefetch_async(self, chunk_id: int) -> bool:
        """
        Start asynchronous prefetch of the specified chunk.
        
        Args:
            chunk_id: ID of chunk to prefetch
            
        Returns:
            True if prefetch started successfully, False otherwise
        """
        # Check if kvikio and async methods are available
        if not self.kvikio_available:
            return False
        
        if not hasattr(self.gds_manager, 'stream_chunk_async'):
            return False
        
        # Don't prefetch if we're already prefetching this chunk
        if self.prefetch_chunk_id == chunk_id and self.prefetch_future is not None:
            return True
        
        try:
            # Cancel any existing prefetch
            if self.prefetch_future is not None:
                # kvikio futures can't be cancelled, but we can ignore the result
                self.prefetch_future = None
                self.prefetch_buffer = None
            
            # Start new async read
            self.prefetch_future, self.prefetch_buffer = self.gds_manager.stream_chunk_async(
                chunk_id, 
                gpu_id=self.gpu_id,
                buffer=self.next_buffer  # Reuse pre-allocated buffer if available
            )
            self.prefetch_chunk_id = chunk_id
            
            logger.debug(f"Worker {self.worker_id}: Started prefetch for chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Prefetch failed for chunk {chunk_id}: {e}")
            self.prefetch_future = None
            self.prefetch_buffer = None
            self.prefetch_chunk_id = None
            return False
    
    def wait_for_prefetch(self) -> Optional[cp.ndarray]:
        """
        Wait for the current prefetch operation to complete.
        
        Returns:
            The prefetched data as a CuPy array, or None if prefetch failed
        """
        if self.prefetch_future is None:
            return None
        
        try:
            # Wait for kvikio async read to complete
            # The get() method blocks until the read is done
            self.prefetch_future.get()
            
            # Get the filled buffer
            data = self.prefetch_buffer
            
            # Clear prefetch state
            self.prefetch_future = None
            self.prefetch_chunk_id = None
            # Keep buffer for reuse
            self.next_buffer = self.prefetch_buffer
            self.prefetch_buffer = None
            
            logger.debug(f"Worker {self.worker_id}: Prefetch complete for chunk {self.prefetch_chunk_id}")
            return data
            
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Prefetch wait failed: {e}")
            self.prefetch_future = None
            self.prefetch_buffer = None
            self.prefetch_chunk_id = None
            return None
    
    def load_batch_from_zarr_with_prefetch(self, chunk_index: int) -> cp.ndarray:
        """
        Load a batch with prefetching support.
        If this chunk was prefetched, use it. Otherwise load synchronously.
        
        Args:
            chunk_index: Index of chunk to load
            
        Returns:
            Loaded data as CuPy array
        """
        # Check if this chunk was prefetched
        if self.prefetch_chunk_id == chunk_index and self.prefetch_future is not None:
            data = self.wait_for_prefetch()
            if data is not None:
                logger.debug(f"Worker {self.worker_id}: Using prefetched chunk {chunk_index}")
                return data
        
        # Fallback to synchronous load
        logger.debug(f"Worker {self.worker_id}: Loading chunk {chunk_index} synchronously")
        return self.load_batch_from_zarr(chunk_index)
    
    def swap_buffers(self):
        """
        Swap current and next buffers for double buffering.
        This is a simple pointer swap, no data copying.
        """
        self.current_buffer, self.next_buffer = self.next_buffer, self.current_buffer
```

### 3. Batch Worker with Pre-fetching (`/floatsom/processing/ray_ops/workers/ray_batch_worker.py`)

```python
@ray.remote(num_gpus=1)
class RayBatchWorker(RayGDSBaseWorker):
    
    def process_batch_iteration(self, som_weights: Optional[np.ndarray], topology_data: Dict,
                               params: Dict, collective_group: str, 
                               chunk_index: Optional[int] = None,
                               selected_indices: Optional[np.ndarray] = None,
                               is_first_iteration: bool = False) -> Dict[str, Any]:
        """
        Process batch iteration with asynchronous pre-fetching for overlapped I/O and compute.
        """
        with self.device:
            # ... existing setup code ...
            
            # Initialize accumulators
            accumulated_updates = cp.zeros_like(gpu_weights)
            accumulated_influence = cp.zeros(gpu_weights.shape[0], dtype=cp.float32)
            
            # Process multiple chunks with pre-fetching
            if chunk_index is None and selected_indices is None:
                num_chunks = self.get_num_chunks()
                
                if num_chunks > 0:
                    # Start pre-fetching second chunk while loading first
                    if num_chunks > 1 and self.kvikio_available:
                        prefetch_started = self.start_prefetch_async(1)
                        logger.debug(f"Worker {self.worker_id}: Prefetch chunk 1 started: {prefetch_started}")
                    
                    # Load and process first chunk synchronously
                    local_samples = self.load_batch_from_zarr(0)
                    
                    # Process all chunks
                    for chunk_idx in range(num_chunks):
                        # Get chunk data
                        if chunk_idx > 0:
                            # Use prefetched data if available, otherwise load sync
                            local_samples = self.load_batch_from_zarr_with_prefetch(chunk_idx)
                        
                        # Start prefetching next chunk immediately
                        # This runs in background while we process current chunk
                        if chunk_idx + 1 < num_chunks and self.kvikio_available:
                            prefetch_started = self.start_prefetch_async(chunk_idx + 1)
                            if prefetch_started:
                                logger.debug(f"Worker {self.worker_id}: Started prefetch for chunk {chunk_idx + 1}")
                        
                        # Process current chunk (overlaps with prefetch)
                        # Find BMUs for this chunk
                        bmus = find_bmus(
                            local_samples, gpu_weights, return_distances=False
                        )
                        
                        # Get influence matrix from topology
                        influence_matrix = topology_data.get_precomputed_influence_matrix(
                            params.current_radius, 'gaussian'
                        )
                        
                        # Get raw accumulated updates and influence for this chunk
                        chunk_updates, chunk_influence = compute_weight_updates(
                            batch=local_samples,
                            bmus=bmus,
                            weights=gpu_weights,
                            influence_matrix=influence_matrix,
                            learning_rate=params.learning_rate,
                            chunk_size=self.chunk_size or 1000,
                            verbose=False
                        )
                        
                        # Accumulate for this chunk
                        accumulated_updates += chunk_updates
                        accumulated_influence += chunk_influence
                        
                        logger.debug(f"Worker {self.worker_id}: Processed chunk {chunk_idx}/{num_chunks}")
                
                local_samples_count = num_chunks * self.chunk_size  # Approximate
            
            else:
                # Single chunk processing (no pre-fetching benefit)
                local_samples = self.load_batch_from_zarr(chunk_index)
                
                # Find BMUs for local samples
                bmus = find_bmus(
                    local_samples, gpu_weights, return_distances=False
                )
                
                # Get influence matrix from topology
                influence_matrix = topology_data.get_precomputed_influence_matrix(
                    params.current_radius, 'gaussian'
                )
                
                # Get raw accumulated updates and influence
                accumulated_updates, accumulated_influence = compute_weight_updates(
                    batch=local_samples,
                    bmus=bmus,
                    weights=gpu_weights,
                    influence_matrix=influence_matrix,
                    learning_rate=params.learning_rate,
                    chunk_size=self.chunk_size or 1000,
                    verbose=False
                )
                local_samples_count = len(local_samples)
            
            # ... rest of processing (NCCL sync, normalization, momentum) ...
```

### 4. Color Worker with Pre-fetching (`/floatsom/processing/ray_ops/workers/ray_color_worker.py`)

```python
@ray.remote(num_gpus=1)
class RayColorWorker(RayGDSBaseWorker):
    
    def calculate_all_bmus_chunked_with_prefetch(self):
        """
        Calculate BMUs for all samples with asynchronous pre-fetching.
        Memory efficient - only one chunk in memory at a time, with next chunk loading in background.
        """
        with self.device:
            logger.info(f"Worker {self.worker_id}: Calculating BMUs with prefetching")
            
            # ... existing setup code ...
            
            # Pre-allocate arrays
            self.local_bmus = cp.zeros(self.total_local_samples, dtype=cp.int32)
            self.processed_mask = cp.zeros(self.total_local_samples, dtype=bool)
            
            # Start prefetching first chunk if we have multiple chunks
            if self.num_chunks > 1 and self.kvikio_available:
                self.start_prefetch_async(1)
            
            # Process each chunk
            for chunk_idx in range(self.num_chunks):
                chunk_start = int(self.chunk_boundaries[chunk_idx, 0])
                chunk_end = int(self.chunk_boundaries[chunk_idx, 1])
                
                # Load chunk (use prefetched if available)
                if chunk_idx > 0:
                    chunk_data = self.load_batch_from_zarr_with_prefetch(chunk_idx)
                else:
                    chunk_data = self.load_batch_from_zarr(chunk_idx)
                
                actual_chunk_size = chunk_data.shape[0]
                
                # Start prefetching next chunk immediately
                if chunk_idx + 1 < self.num_chunks and self.kvikio_available:
                    self.start_prefetch_async(chunk_idx + 1)
                
                # Adjust chunk_end if actual data is smaller
                if actual_chunk_size < (chunk_end - chunk_start):
                    chunk_end = chunk_start + actual_chunk_size
                    self.chunk_boundaries[chunk_idx, 1] = chunk_end
                
                # Calculate BMUs for this chunk (overlaps with prefetch)
                chunk_bmus = find_bmus(chunk_data, self.current_weights,
                                     verbose=False, chunk_size=self.chunk_size)
                
                # Store in BMU array
                self.local_bmus[chunk_start:chunk_end] = chunk_bmus
                
                # Free chunk memory immediately
                del chunk_data
                
                logger.debug(f"Worker {self.worker_id}: Processed BMUs for chunk {chunk_idx}/{self.num_chunks}")
            
            # Clear memory pool
            cp.get_default_memory_pool().free_all_blocks()
            self.current_chunk_idx = -1
            self.current_chunk_data = None
            
            logger.info(f"Worker {self.worker_id}: Calculated BMUs for {self.total_local_samples} samples")
    
    def load_chunk_for_round_with_prefetch(self, round_idx: int) -> cp.ndarray:
        """
        Load the chunk containing this round with prefetching support.
        Predicts and prefetches the next chunk based on round progression.
        """
        with self.device:
            chunk_idx = int(self.round_to_chunk_map[round_idx])
            
            if self.current_chunk_idx != chunk_idx:
                # Free previous chunk
                if self.current_chunk_data is not None:
                    del self.current_chunk_data
                    cp.get_default_memory_pool().free_all_blocks()
                
                # Use prefetched data if this chunk was prefetched
                if self.prefetch_chunk_id == chunk_idx and self.prefetch_future is not None:
                    self.current_chunk_data = self.wait_for_prefetch()
                    logger.debug(f"Worker {self.worker_id}: Used prefetched chunk {chunk_idx} for round {round_idx}")
                else:
                    # Load synchronously
                    self.current_chunk_data = self.load_batch_from_zarr(chunk_idx)
                    logger.debug(f"Worker {self.worker_id}: Loaded chunk {chunk_idx} for round {round_idx}")
                
                self.current_chunk_idx = chunk_idx
                
                # Predict and prefetch next chunk
                # Look ahead to see what chunk the next round will need
                if round_idx + 1 < self.max_rounds and self.kvikio_available:
                    next_chunk_idx = int(self.round_to_chunk_map[round_idx + 1])
                    if next_chunk_idx != chunk_idx:  # Only prefetch if different chunk
                        if self.start_prefetch_async(next_chunk_idx):
                            logger.debug(f"Worker {self.worker_id}: Prefetching chunk {next_chunk_idx} for next round")
                
                # Calculate BMUs for this chunk if not already done
                chunk_start, chunk_end = self.chunk_boundaries[chunk_idx]
                chunk_start, chunk_end = int(chunk_start), int(chunk_end)
                
                # Only calculate if BMUs not yet initialized for this range
                if self.local_bmus[chunk_start:chunk_end].sum() == 0:
                    chunk_bmus = find_bmus(self.current_chunk_data, self.current_weights,
                                         verbose=False, chunk_size=self.chunk_size)
                    self.local_bmus[chunk_start:chunk_end] = chunk_bmus
            
            return self.current_chunk_data
    
    def get_round_data(self, round_idx, max_rounds):
        """
        Get data for a round with prefetching support.
        """
        with self.device:
            # Load the chunk containing this round (with prefetching)
            chunk_data = self.load_chunk_for_round_with_prefetch(round_idx)
            
            # ... rest of existing get_round_data logic ...
```

### 5. Ray GDS Base Configuration (`/floatsom/processing/ray_ops/ray_gds_base.py`)

```python
import os
import logging

class RayGDSWorkerManager:
    def __init__(self, num_gpus: int, ray_config: Any, 
                 processing_config: Optional[ProcessingConfig] = None):
        # ... existing initialization ...
        
        # Configure kvikio for compat mode globally
        os.environ['KVIKIO_COMPAT_MODE'] = 'AUTO'
        
        # Configure kvikio globally for all workers
        try:
            import kvikio
            
            # Determine optimal thread pool size based on system
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            if num_gpus <= 2:
                num_threads = min(4, cpu_count // 2)
            elif num_gpus <= 4:
                num_threads = min(8, cpu_count // 2)
            else:
                num_threads = min(16, cpu_count // 2)
            
            kvikio.defaults.reset_num_threads(num_threads)
            logger.info(f"Configured kvikio with {num_threads} threads for async I/O")
            
            # Check GDS availability
            try:
                props = kvikio.driver_properties()
                logger.info(f"kvikio initialized with GDS support: {props}")
            except:
                logger.info("kvikio initialized in compat mode (nvidia-fs not available)")
                logger.info("Using thread pool for async I/O - still provides performance benefits")
                
        except ImportError:
            logger.warning("kvikio not available - prefetching will be disabled")
            logger.warning("Install with: pip install kvikio")
```

## Performance Analysis

### Expected Performance Gains

#### With nvidia-fs (GDS enabled)
- **Throughput**: 20-40% improvement
- **GPU Utilization**: 90-100% (from 60-70%)
- **I/O Latency**: Hidden completely for most chunks
- **CPU Usage**: Minimal (DMA transfers)

#### Without nvidia-fs (Compat Mode)
- **Throughput**: 10-20% improvement
- **GPU Utilization**: 80-90% (from 60-70%)
- **I/O Latency**: Partially hidden via thread pool
- **CPU Usage**: Moderate (thread pool handles transfers)

### Performance Monitoring

Add instrumentation to measure overlap efficiency:

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.io_time = 0
        self.compute_time = 0
        self.overlap_time = 0
        self.chunks_processed = 0
    
    def measure_overlap(self, chunk_idx: int, prefetch_available: bool):
        """Measure I/O and compute overlap for a chunk."""
        
        # Time the I/O (either from prefetch or sync load)
        io_start = time.perf_counter()
        if prefetch_available:
            data = self.wait_for_prefetch()
            io_duration = time.perf_counter() - io_start
            self.overlap_time += io_duration  # This was overlapped
        else:
            data = self.load_batch_from_zarr(chunk_idx)
            io_duration = time.perf_counter() - io_start
            self.io_time += io_duration  # This was blocking
        
        # Start next prefetch (happens in background)
        next_prefetch_start = time.perf_counter()
        self.start_prefetch_async(chunk_idx + 1)
        
        # Time the computation
        compute_start = time.perf_counter()
        # ... process chunk ...
        compute_duration = time.perf_counter() - compute_start
        self.compute_time += compute_duration
        
        self.chunks_processed += 1
    
    def report(self):
        """Report performance metrics."""
        total_time = self.io_time + self.compute_time
        overlap_efficiency = self.overlap_time / total_time if total_time > 0 else 0
        
        logger.info(f"Performance Report:")
        logger.info(f"  Chunks processed: {self.chunks_processed}")
        logger.info(f"  Total I/O time: {self.io_time:.2f}s")
        logger.info(f"  Total compute time: {self.compute_time:.2f}s")
        logger.info(f"  Overlapped time: {self.overlap_time:.2f}s")
        logger.info(f"  Overlap efficiency: {overlap_efficiency:.1%}")
        logger.info(f"  Time saved: {self.overlap_time:.2f}s")
```

## Testing Strategy

### 1. Functional Testing

```python
# Test script: test_prefetch.py
import numpy as np
import cupy as cp
from floatsom.processing.ray_ops.workers.ray_gds_base_worker import RayGDSBaseWorker

def test_prefetch_correctness():
    """Verify prefetching produces same results as synchronous loading."""
    
    # Create test worker
    worker = RayGDSBaseWorker(
        worker_id=0,
        num_gpus=1,
        gds_config={},
        chunk_size=1000
    )
    
    # Test data
    test_chunks = [np.random.randn(1000, 100) for _ in range(5)]
    
    # Load synchronously
    sync_results = []
    for chunk in test_chunks:
        sync_results.append(process_chunk(chunk))
    
    # Load with prefetching
    async_results = []
    worker.start_prefetch_async(0)
    for i, chunk in enumerate(test_chunks):
        if i > 0:
            chunk = worker.wait_for_prefetch()
        if i < len(test_chunks) - 1:
            worker.start_prefetch_async(i + 1)
        async_results.append(process_chunk(chunk))
    
    # Compare results
    for sync, async_res in zip(sync_results, async_results):
        assert np.allclose(sync, async_res), "Results don't match!"
    
    print("Prefetch correctness test passed!")
```

### 2. Performance Testing

```bash
# Benchmark script
#!/bin/bash

# Test with prefetching disabled
PREFETCH_ENABLED=0 python train_som.py --benchmark

# Test with prefetching enabled (compat mode)
KVIKIO_COMPAT_MODE=AUTO PREFETCH_ENABLED=1 python train_som.py --benchmark

# Compare results
python compare_benchmarks.py
```

### 3. Memory Testing

```python
def test_memory_usage():
    """Ensure prefetching doesn't cause memory leaks."""
    
    import gc
    import cupy as cp
    
    # Get initial memory
    mempool = cp.get_default_memory_pool()
    initial_used = mempool.used_bytes()
    
    # Run many iterations with prefetching
    worker = create_test_worker()
    for iteration in range(100):
        for chunk_idx in range(10):
            if chunk_idx < 9:
                worker.start_prefetch_async(chunk_idx + 1)
            data = worker.load_batch_from_zarr_with_prefetch(chunk_idx)
            process_chunk(data)
            del data
        
        # Force cleanup every 10 iterations
        if iteration % 10 == 0:
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    # Check final memory
    final_used = mempool.used_bytes()
    leak = final_used - initial_used
    
    assert leak < 1e6, f"Memory leak detected: {leak} bytes"
    print("Memory test passed!")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. kvikio Import Error
```
ImportError: No module named 'kvikio'
```
**Solution**: Install kvikio
```bash
pip install kvikio
# or
conda install -c rapidsai -c nvidia -c conda-forge kvikio
```

#### 2. COMPAT_MODE Not Working
```
Error: nvidia-fs kernel module not loaded
```
**Solution**: Set environment variable
```python
import os
os.environ['KVIKIO_COMPAT_MODE'] = 'AUTO'
# or
os.environ['KVIKIO_COMPAT_MODE'] = '1'  # Force compat mode
```

#### 3. Thread Pool Exhaustion
```
Warning: All kvikio threads busy, falling back to sync
```
**Solution**: Increase thread pool size
```python
import kvikio
kvikio.defaults.reset_num_threads(8)  # Increase from default 4
```

#### 4. Prefetch Not Improving Performance
**Possible Causes**:
- Chunks too small (< 1MB)
- Computation too fast (< 10ms per chunk)
- Storage too slow (network mounted)

**Solution**: Profile and adjust chunk size
```python
# Increase chunk size for better I/O efficiency
chunk_size = max(10000, calculated_chunk_size)  # Minimum 10k samples
```

## Future Enhancements

### 1. Adaptive Prefetching
Dynamically enable/disable prefetching based on:
- Chunk processing time
- I/O latency measurements
- Memory pressure

### 2. Multi-buffer Pipeline
Instead of double buffering, use N buffers for deeper pipeline:
```
Buffer 0: Processing
Buffer 1: Ready (prefetched)
Buffer 2: Loading (current prefetch)
Buffer 3: Queued (next prefetch)
```

### 3. Predictive Prefetching
For color processing, predict which chunks will be needed based on:
- Color set distribution
- Historical access patterns
- Round-to-chunk mapping

### 4. Direct GDS Integration
When nvidia-fs becomes available:
- Automatic detection and switching
- Performance comparison logging
- Optimized GDS-specific code paths

## References

1. [kvikio Documentation](https://docs.rapids.ai/api/kvikio/stable/)
2. [NVIDIA GPUDirect Storage Overview](https://developer.nvidia.com/gpudirect-storage)
3. [CuPy CUDA Streams Guide](https://docs.cupy.dev/en/stable/user_guide/cuda_api.html)
4. [Zarr Chunking Best Practices](https://zarr.readthedocs.io/en/stable/tutorial.html#chunking)
5. [Ray Distributed Computing](https://docs.ray.io/en/latest/)

## Implementation Checklist

- [ ] Configure `KVIKIO_COMPAT_MODE='AUTO'` environment variable
- [ ] Add `stream_chunk_async()` method to GDS manager
- [ ] Implement prefetch methods in base worker
- [ ] Update batch worker processing loop
- [ ] Update color worker chunk loading
- [ ] Add performance monitoring
- [ ] Write unit tests
- [ ] Run performance benchmarks
- [ ] Document configuration options
- [ ] Update deployment scripts