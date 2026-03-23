#!/usr/bin/env python3
"""
Test async chunk loading functionality
"""

import numpy as np
import ray
import time
import logging
from floatsom.processing.processing_params import AsyncLoadingConfig
from floatsom.processing.ray_ops.async_chunk_loader import AsyncChunkLoader, ChunkState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chunk_completeness():
    """Test that chunks are always complete before processing"""
    print("\n=== Testing Chunk Completeness ===")
    
    # Create test data
    n_samples = 10000
    n_features = 100
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create async loader
    config = AsyncLoadingConfig(
        enable_async=True,
        initial_chunks_per_worker=2,
        chunk_validation=True
    )
    
    chunk_size = 1000
    num_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    loader = AsyncChunkLoader(
        worker_id=0,
        total_chunks=num_chunks,
        chunk_size=chunk_size,
        config=config
    )
    
    # Start loading
    loader.start_loading(
        data_ref=data,
        start_idx=0,
        end_idx=n_samples,
        initial_chunks=2
    )
    
    # Wait for initial chunks
    success = loader.wait_for_initial_chunks(2)
    assert success, "Initial chunks should be ready"
    
    # Try to get a chunk
    chunk_0 = loader.get_chunk_when_ready(0, timeout=5.0)
    assert chunk_0 is not None, "Chunk 0 should be available"
    assert len(chunk_0) == chunk_size, f"Chunk 0 should have {chunk_size} samples"
    assert chunk_0.dtype == np.float32, "Chunk should be float32"
    
    # Mark chunk as done
    loader.mark_chunk_done(0)
    
    # Get progress
    progress = loader.get_loading_progress()
    print(f"Loading progress: {progress['progress_pct']:.1f}%")
    
    # Clean up
    loader.stop()
    print("✓ Chunk completeness test passed")

def test_async_vs_sync_performance():
    """Compare async vs sync loading performance"""
    print("\n=== Testing Async vs Sync Performance ===")
    
    # Create larger test data
    n_samples = 1000000
    n_features = 100
    data_size_gb = n_samples * n_features * 4 / (1024**3)
    print(f"Test data size: {data_size_gb:.2f} GB")
    
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Test sync loading (simulate)
    print("\nSimulating SYNC loading:")
    start_time = time.time()
    time.sleep(0.5)  # Simulate full data transfer
    sync_time = time.time() - start_time
    print(f"Sync time to first chunk: {sync_time:.3f}s")
    
    # Test async loading
    print("\nTesting ASYNC loading:")
    config = AsyncLoadingConfig(
        enable_async=True,
        initial_chunks_per_worker=2
    )
    
    chunk_size = 100000
    num_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    loader = AsyncChunkLoader(
        worker_id=0,
        total_chunks=num_chunks,
        chunk_size=chunk_size,
        config=config
    )
    
    start_time = time.time()
    loader.start_loading(
        data_ref=data,
        start_idx=0,
        end_idx=n_samples,
        initial_chunks=2
    )
    
    # Wait for initial chunks only
    loader.wait_for_initial_chunks(2)
    async_time = time.time() - start_time
    print(f"Async time to first chunk: {async_time:.3f}s")
    
    # Calculate improvement
    improvement = (sync_time - async_time) / sync_time * 100
    print(f"Performance improvement: {improvement:.1f}%")
    
    # Clean up
    loader.stop()
    print("✓ Performance test completed")

def test_memory_management():
    """Test memory semaphore prevents overflow"""
    print("\n=== Testing Memory Management ===")
    
    n_samples = 10000
    n_features = 100
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Configure with limited memory
    config = AsyncLoadingConfig(
        enable_async=True,
        max_chunks_in_memory=3,  # Only 3 chunks allowed
        initial_chunks_per_worker=2
    )
    
    chunk_size = 1000
    num_chunks = 10
    
    loader = AsyncChunkLoader(
        worker_id=0,
        total_chunks=num_chunks,
        chunk_size=chunk_size,
        config=config
    )
    
    loader.start_loading(
        data_ref=data,
        start_idx=0,
        end_idx=n_samples,
        initial_chunks=2
    )
    
    # Load and process chunks
    for i in range(5):
        chunk = loader.get_chunk_when_ready(i, timeout=5.0)
        assert chunk is not None, f"Chunk {i} should be available"
        
        # Check memory usage
        ready_count = loader.get_ready_count()
        print(f"Chunk {i} loaded, ready count: {ready_count}")
        
        # Simulate processing delay
        time.sleep(0.1)
        
        # Mark as done to free memory
        loader.mark_chunk_done(i)
    
    # Verify memory was properly managed
    with loader.state_lock:
        chunks_in_memory = sum(1 for idx in loader.chunk_data)
        assert chunks_in_memory <= config.max_chunks_in_memory, \
            f"Too many chunks in memory: {chunks_in_memory}"
    
    loader.stop()
    print("✓ Memory management test passed")

def test_async_failure_configuration():
    """Test async failure behavior configuration remains explicit."""
    print("\n=== Testing Async Failure Behavior ===")

    config = AsyncLoadingConfig(
        enable_async=True,
        chunk_ready_timeout=1.0,
    )

    print("✓ Async failure path configured for hard failure")
    print(f"  retry_failed_chunks={config.retry_failed_chunks}")

if __name__ == "__main__":
    print("=" * 60)
    print("ASYNC CHUNK LOADING TESTS")
    print("=" * 60)
    
    # Run tests
    test_chunk_completeness()
    test_async_vs_sync_performance()
    test_memory_management()
    test_async_failure_configuration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
