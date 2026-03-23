#!/usr/bin/env python3
"""
Fast Array IO using memory-mapped files
A high-performance alternative to Zarr for large array reading on HPC systems
"""

import numpy as np
import cupy as cp
import time
import os
import json
from typing import Tuple, Optional, Dict, Any
import mmap


class FastArrayStore:
    """
    High-performance array storage using memory-mapped files.
    Designed to minimize overhead for GPU data loading.
    """
    
    def __init__(self, path: str, mode: str = 'r'):
        """
        Initialize the array store.
        
        Args:
            path: Directory path for the array store
            mode: 'r' for reading, 'w' for writing
        """
        self.path = path
        self.mode = mode
        self.metadata_file = os.path.join(path, 'array_metadata.json')
        self.data_file = os.path.join(path, 'array_data.bin')
        
        if mode == 'r':
            self._load_metadata()
            self._open_for_reading()
        elif mode == 'w':
            os.makedirs(path, exist_ok=True)
        else:
            raise ValueError("Mode must be 'r' or 'w'")
    
    def _load_metadata(self):
        """Load array metadata from JSON file"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.shape = tuple(metadata['shape'])
        self.dtype = np.dtype(np.float32)
        self.chunks = tuple(metadata['chunks'])
        self.n_samples = self.shape[0]
        self.n_features = self.shape[1] if len(self.shape) > 1 else 1
        self.chunk_samples = self.chunks[0]
        self.itemsize = np.dtype(np.float32).itemsize
    
    def _open_for_reading(self):
        """Open the data file for memory-mapped reading"""
        # Standard memory-mapped array
        self.mmap_array = np.memmap(
            self.data_file,
            dtype=np.float32,
            mode='r',
            shape=self.shape
        )
        
        # Also open with mmap for advanced options
        self.file_handle = open(self.data_file, 'rb')
        self.mmap_handle = mmap.mmap(
            self.file_handle.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
        
        # Pre-advise the kernel about our access pattern
        if hasattr(self.mmap_handle, 'madvise'):
            # MADV_SEQUENTIAL = 2 on Linux
            try:
                self.mmap_handle.madvise(2)  # Sequential access
            except:
                pass  # Not critical if it fails
    
    def create(self, shape: Tuple[int, ...], dtype: np.dtype, chunks: Tuple[int, ...]):
        """
        Create a new array store.
        
        Args:
            shape: Shape of the array
            dtype: Data type
            chunks: Chunk shape (for compatibility, not strictly enforced)
        """
        if self.mode != 'w':
            raise ValueError("Cannot create array in read mode")
        
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.chunks = chunks
        
        # Save metadata
        metadata = {
            'shape': list(shape),
            'dtype': str(np.float32),
            'chunks': list(chunks)
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create memory-mapped array for writing
        self.mmap_array = np.memmap(
            self.data_file,
            dtype=np.float32,
            mode='w+',
            shape=self.shape
        )
        
        return self.mmap_array
    
    def read_chunk(self, chunk_idx: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Read a chunk by index.
        
        Args:
            chunk_idx: Index of the chunk to read
            out: Optional output array to read into
        
        Returns:
            The chunk data
        """
        start_idx = chunk_idx * self.chunk_samples
        end_idx = min(start_idx + self.chunk_samples, self.n_samples)
        
        if out is None:
            return self.mmap_array[start_idx:end_idx].copy()
        else:
            actual_samples = end_idx - start_idx
            out[:actual_samples] = self.mmap_array[start_idx:end_idx]
            return out[:actual_samples]
    
    def read_chunk_to_pinned(self, chunk_idx: int, pinned_buffer: np.ndarray) -> np.ndarray:
        """
        Read a chunk directly into pinned memory.
        
        Args:
            chunk_idx: Index of the chunk to read
            pinned_buffer: Pre-allocated pinned memory buffer
        
        Returns:
            View of the filled portion of the buffer
        """
        start_idx = chunk_idx * self.chunk_samples
        end_idx = min(start_idx + self.chunk_samples, self.n_samples)
        actual_samples = end_idx - start_idx
        
        # Direct copy into pinned memory
        pinned_buffer[:actual_samples] = self.mmap_array[start_idx:end_idx]
        return pinned_buffer[:actual_samples]
    
    def read_range(self, start: int, end: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Read a range of samples.
        
        Args:
            start: Start index
            end: End index
            out: Optional output array
        
        Returns:
            The data
        """
        if out is None:
            return self.mmap_array[start:end].copy()
        else:
            out[:end-start] = self.mmap_array[start:end]
            return out[:end-start]
    
    def __getitem__(self, key):
        """Support array-style indexing"""
        return self.mmap_array[key]
    
    def __setitem__(self, key, value):
        """Support array-style assignment"""
        if self.mode == 'r':
            raise ValueError("Cannot write to array opened in read mode")
        self.mmap_array[key] = value
    
    def close(self):
        """Close the memory-mapped file"""
        if hasattr(self, 'mmap_array'):
            del self.mmap_array
        if hasattr(self, 'mmap_handle'):
            self.mmap_handle.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class FastGPUDataLoader:
    """
    Optimized data loader for GPU processing.
    Minimizes overhead and maximizes throughput.
    """
    
    def __init__(self, 
                 array_store: FastArrayStore,
                 chunk_size: int = 1_000_000,
                 n_prefetch: int = 2,
                 use_pinned: bool = True):
        """
        Initialize the GPU data loader.
        
        Args:
            array_store: FastArrayStore instance
            chunk_size: Number of samples per chunk
            n_prefetch: Number of batches to prefetch
            use_pinned: Whether to use pinned memory
        """
        self.store = array_store
        self.chunk_size = chunk_size
        self.n_prefetch = n_prefetch
        self.use_pinned = use_pinned
        
        self.n_samples = self.store.n_samples
        self.n_features = self.store.n_features
        self.n_batches = int(np.ceil(self.n_samples / chunk_size))
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Pre-allocate all buffers for maximum performance"""
        buffer_shape = (self.chunk_size, self.n_features)
        
        if self.use_pinned:
            # Allocate pinned memory buffers for prefetching
            self.pinned_buffers = []
            for _ in range(self.n_prefetch):
                pinned_mem = cp.cuda.alloc_pinned_memory(
                    self.chunk_size * self.n_features * self.store.itemsize
                )
                pinned_buffer = np.frombuffer(
                    pinned_mem,
                    dtype=np.float32,
                    count=self.chunk_size * self.n_features
                ).reshape(buffer_shape)
                self.pinned_buffers.append(pinned_buffer)
        else:
            # Regular CPU buffers
            self.cpu_buffers = [
                np.empty(buffer_shape, dtype=np.float32)
                for _ in range(self.n_prefetch)
            ]
        
        # GPU buffer
        self.gpu_buffer = cp.empty(buffer_shape, dtype=np.float32)
    
    def iterate_batches(self):
        """
        Generator that yields GPU batches with overlapped IO.
        
        Yields:
            Tuple of (batch_idx, gpu_array)
        """
        # Start prefetching
        prefetch_queue = []
        
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.n_samples)
            actual_samples = end_idx - start_idx
            
            # Select buffer for this batch
            buffer_idx = batch_idx % self.n_prefetch
            if self.use_pinned:
                buffer = self.pinned_buffers[buffer_idx]
            else:
                buffer = self.cpu_buffers[buffer_idx]
            
            # Read data into buffer
            data_view = self.store.read_range(start_idx, end_idx, out=buffer)
            
            # Transfer to GPU
            self.gpu_buffer[:actual_samples] = cp.asarray(data_view)
            
            # Yield the GPU data
            yield batch_idx, self.gpu_buffer[:actual_samples]
    
    def benchmark_throughput(self, n_iterations: int = None):
        """
        Benchmark the data loading throughput.
        
        Args:
            n_iterations: Number of iterations (None for all batches)
        
        Returns:
            Dict with timing statistics
        """
        if n_iterations is None:
            n_iterations = self.n_batches
        
        times = []
        bytes_per_batch = self.chunk_size * self.n_features * self.store.itemsize
        
        print(f"Benchmarking throughput ({n_iterations} iterations)...")
        
        for i, (batch_idx, gpu_data) in enumerate(self.iterate_batches()):
            if i >= n_iterations:
                break
            
            start_time = time.perf_counter()
            
            # Simulate some GPU work to ensure transfer completes
            _ = cp.sum(gpu_data)
            cp.cuda.Device().synchronize()
            
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                throughput = bytes_per_batch / (elapsed * 1024**3)
                print(f"  Batch {i+1}/{n_iterations}: {elapsed*1000:.2f}ms ({throughput:.2f} GB/s)", end='\r')
        
        print()
        
        # Calculate statistics
        mean_time = np.mean(times)
        throughput = bytes_per_batch / (mean_time * 1024**3)
        
        return {
            'mean_time_ms': mean_time * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'throughput_gb_s': throughput,
            'times': times
        }


def create_test_data(path: str, n_samples: int = 10_000_000, n_features: int = 500):
    """Create test data using FastArrayStore"""
    print(f"Creating test data at {path}...")
    
    store = FastArrayStore(path, mode='w')
    arr = store.create(
        shape=(n_samples, n_features),
        dtype=np.float32,
        chunks=(1_000_000, n_features)
    )
    
    # Fill with random data
    chunk_size = 1_000_000
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        arr[i:end] = np.random.randn(end - i, n_features).astype(np.float32)
        print(f"  Filled {end}/{n_samples} samples", end='\r')
    
    print(f"\nData created: {n_samples:,} x {n_features}")
    store.close()


def benchmark_fast_vs_zarr():
    """Compare FastArrayStore with Zarr performance"""
    import tempfile
    import shutil
    
    # Setup paths
    temp_dir = tempfile.mkdtemp(prefix='fast_array_bench_')
    fast_path = os.path.join(temp_dir, 'fast_array')
    
    try:
        # Create test data
        n_samples = 2_000_000
        n_features = 500
        chunk_size = 50_000
        
        create_test_data(fast_path, n_samples, n_features)
        
        # Benchmark FastArrayStore
        print("\n" + "="*60)
        print("BENCHMARKING FastArrayStore")
        print("="*60)
        
        store = FastArrayStore(fast_path, mode='r')
        loader = FastGPUDataLoader(store, chunk_size=chunk_size)
        
        # Warm up
        for i, (_, data) in enumerate(loader.iterate_batches()):
            if i >= 2:
                break
        
        # Benchmark
        stats = loader.benchmark_throughput(n_iterations=20)
        
        print(f"\nResults:")
        print(f"  Mean time: {stats['mean_time_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_gb_s']:.2f} GB/s")
        print(f"  Min time: {stats['min_time_ms']:.2f} ms")
        print(f"  Max time: {stats['max_time_ms']:.2f} ms")
        
        # Compare with raw disk speed
        print("\n" + "="*60)
        print("BENCHMARKING Raw Disk Speed")
        print("="*60)
        
        test_file = os.path.join(temp_dir, 'raw_test.bin')
        test_data = np.random.randn(chunk_size, n_features).astype(np.float32)
        
        with open(test_file, 'wb') as f:
            f.write(test_data.tobytes())
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                _ = np.frombuffer(f.read(), dtype=np.float32)
            times.append(time.perf_counter() - start)
        
        raw_mean = np.mean(times) * 1000
        raw_throughput = (chunk_size * n_features * 4) / (np.mean(times) * 1024**3)
        
        print(f"  Mean time: {raw_mean:.2f} ms")
        print(f"  Throughput: {raw_throughput:.2f} GB/s")
        
        # Calculate overhead
        overhead = ((stats['mean_time_ms'] - raw_mean) / raw_mean) * 100
        print(f"\nFastArrayStore overhead: {overhead:.1f}%")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    # Check CUDA
    if not cp.cuda.runtime.getDeviceCount():
        print("No CUDA devices found!")
        exit(1)
    
    print(f"Using GPU: {cp.cuda.Device()}")
    
    # Run benchmark
    benchmark_fast_vs_zarr()