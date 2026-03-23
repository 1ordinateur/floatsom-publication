#!/usr/bin/env python3
"""
Zarr IO Cost Benchmarking Script - HPC Optimized

This script isolates and measures the IO costs at different stages:
1. Reading zarr chunks into CPU memory
2. Reading zarr chunks into pinned memory  
3. Transferring from pinned memory to GPU

Automatically detects and uses local NVMe/SSD storage on various HPC systems.
Compatible with Zarr v3 API.
"""

import numpy as np
import cupy as cp
import zarr
import time
import os
import sys
import psutil
import tempfile
import subprocess
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics
import warnings

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from processing.processing_params import UNIVERSAL_CHUNK_SIZE
except ImportError:
    UNIVERSAL_CHUNK_SIZE = None
    warnings.warn("Could not import UNIVERSAL_CHUNK_SIZE, using defaults")


@dataclass
class TimingResults:
    """Container for timing statistics"""
    times: List[float]
    mean: float
    std: float
    min: float
    max: float
    total: float
    
    @classmethod
    def from_times(cls, times: List[float]) -> 'TimingResults':
        """Create TimingResults from a list of times"""
        return cls(
            times=times,
            mean=statistics.mean(times) if times else 0,
            std=statistics.stdev(times) if len(times) > 1 else 0,
            min=min(times) if times else 0,
            max=max(times) if times else 0,
            total=sum(times)
        )


class StorageDetector:
    """Detect and select optimal local storage on HPC systems"""
    
    @staticmethod
    def get_local_ssd_path() -> Tuple[str, str]:
        """
        Detect local SSD/NVMe storage on various HPC systems
        
        Returns:
            Tuple of (path, description)
        """
        candidates = []
        
        # PBS/Gadi (NCI) - local NVMe
        if 'PBS_JOBFS' in os.environ:
            path = os.environ['PBS_JOBFS']
            if os.path.exists(path) and os.access(path, os.W_OK):
                candidates.append((path, "PBS_JOBFS (Local NVMe on Gadi/NCI)"))
        
        # SLURM systems - local scratch
        if 'SLURM_TMPDIR' in os.environ:
            path = os.environ['SLURM_TMPDIR']
            if os.path.exists(path) and os.access(path, os.W_OK):
                candidates.append((path, "SLURM_TMPDIR (SLURM local scratch)"))
        
        # SLURM alternative - construct from job ID
        if 'SLURM_JOB_ID' in os.environ:
            for base in ['/tmp', '/scratch/local', '/local/scratch', '/nvme']:
                path = os.path.join(base, f"slurm_{os.environ['SLURM_JOB_ID']}")
                if os.path.exists(base) and os.access(base, os.W_OK):
                    candidates.append((path, f"SLURM local ({base})"))
                    break
        
        # LSF systems
        if 'LSB_JOBID' in os.environ:
            for base in ['/tmp', '/scratch/local', '/local/scratch']:
                path = os.path.join(base, f"lsf_{os.environ['LSB_JOBID']}")
                if os.path.exists(base) and os.access(base, os.W_OK):
                    candidates.append((path, f"LSF local ({base})"))
                    break
        
        # Generic local NVMe/SSD paths
        nvme_paths = [
            '/nvme',
            '/local/nvme',
            '/scratch/local',
            '/local/scratch',
            '/tmp/scratch',
            '/dev/shm'  # RAM disk, but very fast
        ]
        
        for path in nvme_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                # Check if it's actually a different filesystem
                if StorageDetector._is_different_filesystem(path, os.getcwd()):
                    candidates.append((path, f"Local storage ({path})"))
        
        # Node-specific paths (common patterns)
        hostname = os.uname().nodename
        node_paths = [
            f'/tmp/{hostname}',
            f'/scratch/{hostname}',
            f'/local/{hostname}'
        ]
        
        for path in node_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                candidates.append((path, f"Node-specific storage ({path})"))
        
        # Select the best candidate
        if candidates:
            # Prefer explicitly set environment variables first
            for path, desc in candidates:
                if 'PBS_JOBFS' in desc or 'SLURM_TMPDIR' in desc:
                    return path, desc
            # Otherwise return the first valid candidate
            return candidates[0]
        
        # Fallback to system temp
        return tempfile.gettempdir(), "System temp (no local SSD detected)"
    
    @staticmethod
    def _is_different_filesystem(path1: str, path2: str) -> bool:
        """Check if two paths are on different filesystems"""
        try:
            stat1 = os.stat(path1)
            stat2 = os.stat(path2)
            return stat1.st_dev != stat2.st_dev
        except:
            return False
    
    @staticmethod
    def detect_filesystem_type(path: str) -> str:
        """Detect the filesystem type of a given path"""
        try:
            # Try using df command
            result = subprocess.run(
                f"df -T {path} | tail -1",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.split()
                if len(parts) > 1:
                    fs_type = parts[1]
                    return fs_type
        except:
            pass
        
        # Try stat command
        try:
            result = subprocess.run(
                f"stat -f -c %T {path}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return "unknown"
    
    @staticmethod
    def check_lustre_available(path: str) -> bool:
        """Check if path is on Lustre filesystem"""
        try:
            result = subprocess.run(
                f"lfs df {path}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False


class ZarrIOBenchmark:
    """Benchmark class for measuring zarr IO costs on HPC systems"""
    
    def __init__(self, 
                 n_samples: int = 10_000_000,  # 10M samples
                 n_features: int = 500,         # 500 dimensions
                 chunk_samples: int = 1_000_000,  # 1M samples per chunk
                 dtype: np.dtype = np.float32,
                 use_compression: bool = False,
                 use_local_ssd: bool = True,     # Try to use local SSD
                 stripe_count: int = 8,          # Only used if Lustre detected
                 stripe_size: int = 16):         # MB - Only used if Lustre detected
        """
        Initialize the benchmark
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features/dimensions
            chunk_samples: Samples per chunk
            dtype: Data type
            use_compression: Whether to use compression
            use_local_ssd: Try to use local SSD/NVMe if available
            stripe_count: Number of OSTs for Lustre striping (if applicable)
            stripe_size: Stripe size in MB for Lustre (if applicable)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.chunk_samples = chunk_samples
        self.dtype = dtype
        self.use_compression = use_compression
        self.stripe_count = stripe_count
        self.stripe_size = stripe_size
        
        # Calculate chunk shape and total chunks
        self.chunk_shape = (chunk_samples, n_features)
        self.total_chunks = int(np.ceil(n_samples / chunk_samples))
        
        # Calculate sizes
        self.element_bytes = np.dtype(dtype).itemsize
        self.chunk_bytes = chunk_samples * n_features * self.element_bytes
        self.total_bytes = n_samples * n_features * self.element_bytes
        
        # Detect and setup storage
        if use_local_ssd:
            base_path, storage_desc = StorageDetector.get_local_ssd_path()
            self.storage_type = storage_desc
            # Create a subdirectory for our benchmark
            self.temp_dir = os.path.join(base_path, f'zarr_benchmark_{os.getpid()}')
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp(prefix='zarr_benchmark_')
            self.storage_type = "Default temp directory"
        
        self.zarr_path = os.path.join(self.temp_dir, 'benchmark_zarr_io.zarr')
        
        # Detect filesystem type
        self.fs_type = StorageDetector.detect_filesystem_type(self.temp_dir)
        self.is_lustre = StorageDetector.check_lustre_available(self.temp_dir)
        
        # Print configuration
        print(f"Benchmark Configuration:")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Features: {n_features}")
        print(f"  Chunk size: {chunk_samples:,} samples")
        print(f"  Total chunks: {self.total_chunks}")
        print(f"  Chunk memory: {self.chunk_bytes / (1024**2):.2f} MB")
        print(f"  Total size: {self.total_bytes / (1024**3):.2f} GB")
        print(f"  Compression: {'Enabled (LZ4)' if use_compression else 'Disabled'}")
        print(f"\nStorage Configuration:")
        print(f"  Storage type: {self.storage_type}")
        print(f"  Filesystem: {self.fs_type}")
        print(f"  Path: {self.temp_dir}")
        print(f"  Lustre detected: {self.is_lustre}")
        
        # Check available space
        try:
            stat = os.statvfs(self.temp_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            print(f"  Available space: {available_gb:.2f} GB")
            if available_gb < (self.total_bytes / (1024**3)) * 1.5:
                warnings.warn(f"Low disk space! Need ~{self.total_bytes/(1024**3):.2f} GB, have {available_gb:.2f} GB")
        except:
            pass
        
        print()
    
    def setup_lustre_striping(self, path: str):
        """Set Lustre striping if filesystem supports it"""
        if not self.is_lustre:
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Try to set Lustre striping
            cmd = f"lfs setstripe -c {self.stripe_count} -S {self.stripe_size}M {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Check striping was applied
                check_cmd = f"lfs getstripe {path}"
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if check_result.returncode == 0:
                    print(f"  Lustre striping applied:")
                    for line in check_result.stdout.split('\n')[:5]:
                        if 'stripe' in line.lower():
                            print(f"    {line.strip()}")
        except Exception as e:
            print(f"  Could not set Lustre striping: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not fully clean up {self.temp_dir}: {e}")
    
    def create_test_zarr(self):
        """Create a test zarr array with specified dimensions"""
        print(f"Creating test zarr array...")
        
        # Remove existing if present
        if os.path.exists(self.zarr_path):
            shutil.rmtree(self.zarr_path)
        
        # Setup Lustre striping only if on Lustre
        if self.is_lustre:
            self.setup_lustre_striping(self.zarr_path)
        
        # Create store - compatible with Zarr v3
        store = zarr.storage.LocalStore(self.zarr_path)
        
        # Configure compression if requested
        if self.use_compression:
            # Zarr v3 uses codecs instead of compressors
            from zarr.codecs import BloscCodec, ShuffleCodec
            codecs = [
                ShuffleCodec(),
                BloscCodec(cname='lz4', clevel=1)
            ]
        else:
            codecs = None
        
        # Create array with v3 API - use zeros() which creates and returns the array
        z = zarr.zeros(
            store=store,
            shape=(self.n_samples, self.n_features),
            chunks=self.chunk_shape,
            dtype=self.dtype,
            codecs=codecs
        )
        
        # Fill with random data chunk by chunk
        print(f"  Filling with random data...")
        for i in range(0, self.n_samples, self.chunk_samples):
            chunk_end = min(i + self.chunk_samples, self.n_samples)
            actual_samples = chunk_end - i
            z[i:chunk_end, :] = np.random.randn(actual_samples, self.n_features).astype(self.dtype)
            print(f"    Filled chunk {i // self.chunk_samples + 1}/{self.total_chunks}", end='\r')
        
        print(f"\n  Array created: shape={z.shape}, chunks={z.chunks}")
        return z
    
    def benchmark_cpu_reading(self, n_iterations: int = None) -> TimingResults:
        """Benchmark reading zarr chunks into regular CPU memory"""
        if n_iterations is None:
            n_iterations = min(self.total_chunks, 20)  # Limit iterations
            
        print(f"\nBenchmarking CPU Reading (regular memory)...")
        
        # Open zarr array with v3 API
        store = zarr.storage.LocalStore(self.zarr_path)
        z = zarr.open_array(store=store, mode='r')  # v3 uses mode='r' for read-only
        
        # Pre-allocate CPU buffer
        cpu_buffer = np.empty(self.chunk_shape, dtype=self.dtype)
        
        times = []
        
        for i in range(n_iterations):
            start_idx = (i % self.total_chunks) * self.chunk_samples
            end_idx = min(start_idx + self.chunk_samples, self.n_samples)
            
            # Clear any caches if possible (Linux only)
            if i == 0 and sys.platform == 'linux':
                os.system("sync")
            
            # Time the read operation
            start_time = time.perf_counter()
            
            # Read into pre-allocated buffer
            actual_samples = end_idx - start_idx
            cpu_buffer[:actual_samples, :] = z[start_idx:end_idx, :]
            
            read_time = time.perf_counter() - start_time
            times.append(read_time)
            
            if (i + 1) % 5 == 0 or i == n_iterations - 1:
                print(f"  Chunk {i+1}/{n_iterations}: {read_time*1000:.2f} ms", end='\r')
        
        print()
        return TimingResults.from_times(times)
    
    def benchmark_pinned_memory_reading(self, n_iterations: int = None) -> TimingResults:
        """Benchmark reading zarr chunks into pinned memory"""
        if n_iterations is None:
            n_iterations = min(self.total_chunks, 20)
            
        print(f"\nBenchmarking Pinned Memory Reading...")
        
        # Open zarr array with v3 API
        store = zarr.storage.LocalStore(self.zarr_path)
        z = zarr.open_array(store=store, mode='r')
        
        # Allocate pinned memory buffer
        pinned_memory = cp.cuda.alloc_pinned_memory(self.chunk_bytes)
        pinned_buffer = np.frombuffer(
            pinned_memory,
            dtype=self.dtype,
            count=self.chunk_samples * self.n_features
        ).reshape(self.chunk_shape)
        
        times = []
        
        for i in range(n_iterations):
            start_idx = (i % self.total_chunks) * self.chunk_samples
            end_idx = min(start_idx + self.chunk_samples, self.n_samples)
            
            # Time the read operation into pinned memory
            start_time = time.perf_counter()
            
            # Read directly into pinned memory buffer
            actual_samples = end_idx - start_idx
            pinned_buffer[:actual_samples, :] = z[start_idx:end_idx, :]
            
            read_time = time.perf_counter() - start_time
            times.append(read_time)
            
            if (i + 1) % 5 == 0 or i == n_iterations - 1:
                print(f"  Chunk {i+1}/{n_iterations}: {read_time*1000:.2f} ms", end='\r')
        
        print()
        return TimingResults.from_times(times)
    
    def benchmark_gpu_transfer(self, n_iterations: int = None) -> Tuple[TimingResults, TimingResults]:
        """
        Benchmark the complete pipeline: reading into pinned memory + GPU transfer
        Returns both read times and transfer times separately
        """
        if n_iterations is None:
            n_iterations = min(self.total_chunks, 20)
            
        print(f"\nBenchmarking GPU Transfer Pipeline...")
        
        # Open zarr array with v3 API
        store = zarr.storage.LocalStore(self.zarr_path)
        z = zarr.open_array(store=store, mode='r')
        
        # Allocate pinned memory buffer
        pinned_memory = cp.cuda.alloc_pinned_memory(self.chunk_bytes)
        pinned_buffer = np.frombuffer(
            pinned_memory,
            dtype=self.dtype,
            count=self.chunk_samples * self.n_features
        ).reshape(self.chunk_shape)
        
        # Pre-allocate GPU buffer
        gpu_buffer = cp.empty(self.chunk_shape, dtype=self.dtype)
        
        read_times = []
        transfer_times = []
        
        for i in range(n_iterations):
            start_idx = (i % self.total_chunks) * self.chunk_samples
            end_idx = min(start_idx + self.chunk_samples, self.n_samples)
            actual_samples = end_idx - start_idx
            
            # Time the read into pinned memory
            read_start = time.perf_counter()
            pinned_view = pinned_buffer[:actual_samples, :]
            pinned_view[:] = z[start_idx:end_idx, :]
            read_time = time.perf_counter() - read_start
            
            # Time the GPU transfer
            transfer_start = time.perf_counter()
            gpu_buffer[:actual_samples, :] = cp.asarray(pinned_view, dtype=self.dtype)
            cp.cuda.Device().synchronize()  # Ensure transfer completes
            transfer_time = time.perf_counter() - transfer_start
            
            read_times.append(read_time)
            transfer_times.append(transfer_time)
            
            if (i + 1) % 5 == 0 or i == n_iterations - 1:
                total_time = read_time + transfer_time
                print(f"  Chunk {i+1}/{n_iterations}: read={read_time*1000:.2f}ms, "
                      f"transfer={transfer_time*1000:.2f}ms, total={total_time*1000:.2f}ms", end='\r')
        
        print()
        return TimingResults.from_times(read_times), TimingResults.from_times(transfer_times)
    
    def benchmark_direct_gpu_transfer(self, n_iterations: int = None) -> TimingResults:
        """
        Benchmark direct CPU->GPU transfer without zarr reading
        This isolates the pure transfer cost
        """
        if n_iterations is None:
            n_iterations = min(self.total_chunks, 20)
            
        print(f"\nBenchmarking Direct CPU->GPU Transfer (no zarr IO)...")
        
        # Create random CPU data
        cpu_data = np.random.randn(*self.chunk_shape).astype(self.dtype)
        
        # Pre-allocate GPU buffer
        gpu_buffer = cp.empty(self.chunk_shape, dtype=self.dtype)
        
        times = []
        
        for i in range(n_iterations):
            # Time just the transfer
            start_time = time.perf_counter()
            gpu_buffer[:] = cp.asarray(cpu_data, dtype=self.dtype)
            cp.cuda.Device().synchronize()
            transfer_time = time.perf_counter() - start_time
            
            times.append(transfer_time)
            
            if (i + 1) % 5 == 0 or i == n_iterations - 1:
                print(f"  Iteration {i+1}/{n_iterations}: {transfer_time*1000:.2f} ms", end='\r')
        
        print()
        return TimingResults.from_times(times)
    
    def benchmark_raw_disk_speed(self) -> TimingResults:
        """Test raw disk read speed without Zarr overhead"""
        print(f"\nBenchmarking Raw Disk Speed...")
        
        test_file = os.path.join(self.temp_dir, 'raw_test.bin')
        
        # Write test data
        data = np.random.randn(self.chunk_samples, self.n_features).astype(self.dtype)
        with open(test_file, 'wb') as f:
            f.write(data.tobytes())
        
        # Sync to ensure data is written
        os.system('sync')
        
        times = []
        for i in range(10):
            # Clear cache if possible
            if i == 0 and sys.platform == 'linux':
                os.system("sync")
                # Try to drop caches if we have permission
                os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
            
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                _ = np.frombuffer(f.read(), dtype=self.dtype)
            times.append(time.perf_counter() - start)
            
            print(f"  Iteration {i+1}/10: {times[-1]*1000:.2f} ms", end='\r')
        
        print()
        os.remove(test_file)
        return TimingResults.from_times(times)
    
    def print_results(self, results: Dict[str, TimingResults]):
        """Print formatted results"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\nStorage: {self.storage_type}")
        print(f"Filesystem: {self.fs_type}")
        print(f"Path: {self.temp_dir}")
        
        # Calculate throughput
        chunk_gb = self.chunk_bytes / (1024**3)
        
        print(f"\nTiming Statistics (per {self.chunk_bytes/(1024**2):.2f} MB chunk):")
        print(f"{'Operation':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'GB/s':>10}")
        print("-"*80)
        
        for name, timing in results.items():
            if timing.mean > 0:
                throughput = chunk_gb / timing.mean
            else:
                throughput = 0
            
            print(f"{name:<30} {timing.mean*1000:>9.2f}ms {timing.std*1000:>9.2f}ms "
                  f"{timing.min*1000:>9.2f}ms {timing.max*1000:>9.2f}ms {throughput:>9.2f}")
        
        # Calculate percentages for pipeline
        if 'Pipeline Read' in results and 'Pipeline Transfer' in results:
            total_pipeline = results['Pipeline Read'].mean + results['Pipeline Transfer'].mean
            read_pct = (results['Pipeline Read'].mean / total_pipeline) * 100
            transfer_pct = (results['Pipeline Transfer'].mean / total_pipeline) * 100
            
            print(f"\nPipeline Breakdown:")
            print(f"  Read into pinned memory: {read_pct:.1f}%")
            print(f"  Transfer to GPU: {transfer_pct:.1f}%")
            
        # Compare with raw disk speed
        if 'Raw Disk Read' in results and 'CPU Read (Regular)' in results:
            overhead = ((results['CPU Read (Regular)'].mean - results['Raw Disk Read'].mean) 
                       / results['Raw Disk Read'].mean) * 100
            print(f"\nZarr Overhead:")
            print(f"  Raw disk speed: {chunk_gb / results['Raw Disk Read'].mean:.2f} GB/s")
            print(f"  Zarr read speed: {chunk_gb / results['CPU Read (Regular)'].mean:.2f} GB/s")
            print(f"  Overhead: {overhead:.1f}%")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        try:
            # Create test zarr
            self.create_test_zarr()
            
            results = {}
            
            # Warm-up GPU
            print("\nWarming up GPU...")
            _ = cp.ones((1000, 1000), dtype=self.dtype)
            cp.cuda.Device().synchronize()
            
            # Test raw disk speed first
            results['Raw Disk Read'] = self.benchmark_raw_disk_speed()
            
            print("\n" + "="*60)
            print("RUNNING BENCHMARKS")
            print("="*60)
            
            # Benchmark regular CPU reading
            results['CPU Read (Regular)'] = self.benchmark_cpu_reading()
            
            # Benchmark pinned memory reading
            results['CPU Read (Pinned)'] = self.benchmark_pinned_memory_reading()
            
            # Benchmark full pipeline
            read_times, transfer_times = self.benchmark_gpu_transfer()
            results['Pipeline Read'] = read_times
            results['Pipeline Transfer'] = transfer_times
            
            # Benchmark pure transfer cost
            results['Pure GPU Transfer'] = self.benchmark_direct_gpu_transfer()
            
            # Print results
            self.print_results(results)
            
            # Key findings
            print("\n" + "="*60)
            print("KEY FINDINGS")
            print("="*60)
            
            # Calculate where the costs lie
            total_pipeline = results['Pipeline Read'].mean + results['Pipeline Transfer'].mean
            read_cost = (results['Pipeline Read'].mean / total_pipeline) * 100
            transfer_cost = (results['Pipeline Transfer'].mean / total_pipeline) * 100
            
            print(f"\nOperational Cost Breakdown:")
            print(f"  Storage → Pinned Memory: {read_cost:.1f}% ({results['Pipeline Read'].mean*1000:.2f} ms)")
            print(f"  Pinned Memory → GPU:     {transfer_cost:.1f}% ({results['Pipeline Transfer'].mean*1000:.2f} ms)")
            
            # Memory bandwidth utilization
            chunk_gb = self.chunk_bytes / (1024**3)
            theoretical_pcie_bandwidth = 16.0  # GB/s for PCIe 3.0 x16
            actual_transfer_bandwidth = chunk_gb / results['Pipeline Transfer'].mean
            bandwidth_utilization = (actual_transfer_bandwidth / theoretical_pcie_bandwidth) * 100
            
            print(f"\nTransfer Efficiency:")
            print(f"  Actual transfer bandwidth: {actual_transfer_bandwidth:.2f} GB/s")
            print(f"  PCIe bandwidth utilization: {bandwidth_utilization:.1f}%")
            
            # Storage performance
            raw_disk_bandwidth = chunk_gb / results['Raw Disk Read'].mean
            zarr_bandwidth = chunk_gb / results['CPU Read (Regular)'].mean
            
            print(f"\nStorage Performance:")
            print(f"  Raw disk bandwidth: {raw_disk_bandwidth:.2f} GB/s")
            print(f"  Zarr effective bandwidth: {zarr_bandwidth:.2f} GB/s")
            print(f"  Storage type: {self.storage_type}")
            
            return results
        
        except Exception as e:
            print(f"\nError during benchmark: {e}")
            raise
        finally:
            # Ensure GPU memory is cleaned up
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


def main():
    """Main entry point"""
    
    # Check CUDA availability
    if not cp.cuda.runtime.getDeviceCount():
        print("No CUDA devices found!")
        sys.exit(1)
    
    # Get device info
    device = cp.cuda.Device()
    print(f"Using GPU: {device}")
    mem_info = device.mem_info
    print(f"GPU Memory: {mem_info[1] / (1024**3):.2f} GB total, "
          f"{mem_info[0] / (1024**3):.2f} GB free")
    
    # Get system info
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / (1024**3):.2f} GB total, "
          f"{ram.available / (1024**3):.2f} GB available")
    print()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Zarr IO costs on HPC systems')
    parser.add_argument('--samples', type=int, default=2_000_000,
                        help='Number of samples (default: 2M)')
    parser.add_argument('--features', type=int, default=500,
                        help='Number of features (default: 500)')
    parser.add_argument('--chunk-samples', type=int, default=50_000,
                        help='Samples per chunk (default: 1M)')
    parser.add_argument('--compression', action='store_true',
                        help='Test with compression')
    parser.add_argument('--no-local-ssd', action='store_true',
                        help='Do not attempt to use local SSD')
    parser.add_argument('--compare', action='store_true',
                        help='Compare compressed vs uncompressed')
    
    args = parser.parse_args()
    
    if args.compare:
        # Test both compressed and uncompressed
        for use_compression in [False, True]:
            print("\n" + "="*80)
            print(f"BENCHMARK {'WITH' if use_compression else 'WITHOUT'} COMPRESSION")
            print("="*80)
            
            benchmark = ZarrIOBenchmark(
                n_samples=args.samples,
                n_features=args.features,
                chunk_samples=args.chunk_samples,
                dtype=np.float32,
                use_compression=use_compression,
                use_local_ssd=not args.no_local_ssd
            )
            
            try:
                results = benchmark.run_full_benchmark()
            finally:
                benchmark.cleanup()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        # Single benchmark run
        benchmark = ZarrIOBenchmark(
            n_samples=args.samples,
            n_features=args.features,
            chunk_samples=args.chunk_samples,
            dtype=np.float32,
            use_compression=args.compression,
            use_local_ssd=not args.no_local_ssd
        )
        
        try:
            results = benchmark.run_full_benchmark()
        finally:
            benchmark.cleanup()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    
    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()