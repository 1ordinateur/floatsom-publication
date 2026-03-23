# Multi-GPU Color Set Processing for Large-Scale FloatSOM

## Overview

This document describes the distributed architecture for processing color sets across multiple GPUs when dealing with datasets too large for single-GPU memory. The key innovation is that **data never moves between GPUs** - only weight updates are synchronized.

## Core Architecture Principles

1. **Static Data Partitioning**: Each GPU permanently owns a chunk of the dataset
2. **Local Color Processing**: GPUs process only their local points belonging to each color
3. **Weight Synchronization Only**: Only weight matrices (small) are communicated between GPUs
4. **Sequential Color Processing**: Colors are processed sequentially with synchronization barriers

## Data Distribution

```
Dataset: 1 Billion points, 1024 features
Distribution across 4 GPUs:

GPU 0: data[0:250M]        → Permanent residence
GPU 1: data[250M:500M]     → Permanent residence  
GPU 2: data[500M:750M]     → Permanent residence
GPU 3: data[750M:1B]       → Permanent residence

Key: Data NEVER moves between GPUs during training
```

## Algorithm Flow

### Phase 1: One-Time BMU Registration (Per Iteration)

```python
# On each GPU (parallel, no communication)
def register_bmus():
    # Each GPU computes BMUs for its local data only
    local_bmus = find_bmus(my_data_chunk, global_weights)  # Expensive, but done once
    local_colors = assign_colors(local_bmus, topology)      # Cheap assignment
    
    # Build index for fast color lookup
    color_indices = {}
    for color_id in range(n_colors):
        # Which of MY points belong to this color?
        color_indices[color_id] = np.where(local_colors == color_id)[0]
    
    return local_bmus, color_indices
```

### Phase 2: Sequential Color Processing with Parallel Execution

```python
# Process each color set sequentially
for color_id in range(n_colors):
    
    # PARALLEL: Each GPU processes its points for this color
    # GPU 0 processing:
    my_color_points = my_data_chunk[color_indices[color_id]]  # e.g., 50K points
    my_color_bmus = local_bmus[color_indices[color_id]]
    weight_updates_gpu0 = calculate_updates(my_color_points, my_color_bmus, weights)
    
    # GPU 1 processing (simultaneously):
    my_color_points = my_data_chunk[color_indices[color_id]]  # e.g., 30K points
    my_color_bmus = local_bmus[color_indices[color_id]]
    weight_updates_gpu1 = calculate_updates(my_color_points, my_color_bmus, weights)
    
    # GPUs 2 & 3 doing the same...
    
    # SYNCHRONIZATION POINT: All-reduce weight updates
    global_weight_updates = nccl_allreduce(
        weight_updates_gpu0 + 
        weight_updates_gpu1 + 
        weight_updates_gpu2 + 
        weight_updates_gpu3
    )
    
    # All GPUs apply identical update
    weights += global_weight_updates
    
    # All GPUs now have identical weights for next color
```

## Communication Pattern

```
Timeline for processing colors:

Color 0 (Red):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│ 50K points  │  │ 30K points  │  │ 45K points  │  │ 25K points  │
│ → updates_0 │  │ → updates_1 │  │ → updates_2 │  │ → updates_3 │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                              │
                        NCCL AllReduce
                              │
                     Combined Weight Updates
                              │
       ┌────────────────┬────────────────┬────────────────┐
       ▼                ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  weights'   │  │  weights'   │  │  weights'   │  │  weights'   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘

Color 1 (Blue):
[Same pattern with different point distributions]
```

## Implementation

### Complete Multi-GPU Color Processor

```python
import cupy as cp
from mpi4py import MPI
import numpy as np

class MultiGPUColorProcessor:
    def __init__(self, data_path, n_samples, n_features, som_shape):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize NCCL for efficient GPU communication
        self.setup_nccl()
        
        # Load this GPU's data chunk (memory-mapped for large datasets)
        self.load_data_chunk(data_path, n_samples, n_features)
        
        # Initialize weights (same on all GPUs)
        self.weights = self.initialize_weights(som_shape, n_features)
        
    def load_data_chunk(self, data_path, n_samples, n_features):
        """Load only this GPU's portion of data"""
        # Memory-mapped file for out-of-core processing
        full_data = np.memmap(data_path, dtype='float32', mode='r',
                              shape=(n_samples, n_features))
        
        # Calculate this GPU's range
        chunk_size = n_samples // self.size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.size - 1 else n_samples
        
        # Load to GPU memory (this stays resident)
        self.data_chunk = cp.asarray(full_data[start_idx:end_idx])
        self.n_local_samples = end_idx - start_idx
        
        print(f"GPU {self.rank}: Loaded {self.n_local_samples} samples")
        
    def setup_nccl(self):
        """Initialize NCCL communicator for GPU-to-GPU communication"""
        from cupy.cuda import nccl
        
        # Get unique ID from rank 0
        if self.rank == 0:
            nccl_id = nccl.get_unique_id()
        else:
            nccl_id = None
        
        # Broadcast ID to all ranks
        nccl_id = self.comm.bcast(nccl_id, root=0)
        
        # Create NCCL communicator
        self.nccl_comm = nccl.NcclCommunicator(
            self.size, nccl_id, self.rank
        )
        
    def train_iteration(self, iteration):
        """One complete iteration of color set processing"""
        
        # Phase 1: BMU Registration (once per iteration)
        local_bmus, color_indices = self.register_bmus()
        
        # Phase 2: Process each color sequentially
        for color_id in range(self.n_colors):
            # Get indices of local points in this color
            local_color_idx = color_indices[color_id]
            
            if len(local_color_idx) > 0:
                # Extract data for this color
                color_data = self.data_chunk[local_color_idx]
                color_bmus = local_bmus[local_color_idx]
                
                # Calculate weight updates
                weight_updates = self.calculate_weight_updates(
                    color_data, color_bmus, self.weights
                )
            else:
                # No points in this color on this GPU
                weight_updates = cp.zeros_like(self.weights)
            
            # Synchronize weight updates across all GPUs
            global_updates = self.nccl_allreduce(weight_updates)
            
            # Apply updates (all GPUs have identical weights after this)
            self.weights += global_updates
            
            # Optional: Recompute BMUs periodically for accuracy
            if color_id > 0 and color_id % 30 == 0:
                local_bmus = self.find_bmus(self.data_chunk, self.weights)
                
    def register_bmus(self):
        """Pre-compute BMUs and color assignments for all local data"""
        # Find BMUs for all local points
        local_bmus = self.find_bmus(self.data_chunk, self.weights)
        
        # Assign colors based on BMUs and topology
        local_colors = self.assign_colors(local_bmus)
        
        # Build index for O(1) color lookup
        color_indices = {}
        for color_id in range(self.n_colors):
            color_indices[color_id] = cp.where(local_colors == color_id)[0]
            
        return local_bmus, color_indices
    
    def nccl_allreduce(self, local_tensor):
        """Sum tensors across all GPUs using NCCL"""
        # Ensure contiguous memory layout
        if not local_tensor.flags['C_CONTIGUOUS']:
            local_tensor = cp.ascontiguousarray(local_tensor)
        
        # Create output buffer
        global_tensor = cp.empty_like(local_tensor)
        
        # NCCL all-reduce (sum)
        self.nccl_comm.allReduce(
            local_tensor.data.ptr,
            global_tensor.data.ptr,
            local_tensor.size,
            nccl.NCCL_FLOAT32,
            nccl.NCCL_SUM,
            cp.cuda.Stream.null.ptr
        )
        
        # Wait for completion
        cp.cuda.Stream.null.synchronize()
        
        return global_tensor
    
    def calculate_weight_updates(self, data, bmus, weights):
        """Calculate weight updates for given data and BMUs"""
        updates = cp.zeros_like(weights)
        
        for i, bmu in enumerate(bmus):
            # Get neighborhood influence
            distances = self.topology.get_distances(bmu)
            influence = cp.exp(-distances**2 / (2 * self.current_radius**2))
            
            # Calculate update
            diff = data[i] - weights
            updates += influence[:, None] * diff[None, :] * self.learning_rate
            
        return updates / len(data)  # Normalize by number of points
```

## Memory and Communication Analysis

### Memory Requirements per GPU
```
Data chunk: 250M points × 1024 features × 4 bytes = 1TB / 4 = 250GB
Weights: 100×100 nodes × 1024 features × 4 bytes = 40MB
BMU indices: 250M × 4 bytes = 1GB  
Color indices: 250M × 1 byte = 250MB
Weight updates: 40MB

Total per GPU: ~251GB (mostly data)
```

### Communication Volume per Iteration
```
Per color: 40MB (weight updates only)
100 colors: 40MB × 100 = 4GB total
Compared to moving data: 1TB (250× more!)
```

## Optimizations

### 1. Overlapped Computation and Communication
```python
# Use CUDA streams for overlap
compute_stream = cp.cuda.Stream()
comm_stream = cp.cuda.Stream()

with compute_stream:
    weight_updates = calculate_updates(...)
    
with comm_stream:
    # Start communication while next color computes
    future = nccl_allreduce_async(weight_updates)
```

### 2. BMU Refresh Strategy
```python
# Refresh BMUs periodically to maintain accuracy
if color_id % refresh_interval == 0:
    local_bmus = find_bmus(data_chunk, weights)
    # Recompute color indices with fresh BMUs
```

### 3. Mixed Precision Training
```python
# Use FP16 for computation, FP32 for weight accumulation
data_fp16 = data.astype(cp.float16)
weight_updates_fp16 = calculate_updates(data_fp16, ...)
weight_updates_fp32 = weight_updates_fp16.astype(cp.float32)
```

## Launch Configuration

```bash
# For 4 GPUs with NCCL
mpirun -n 4 \
    --bind-to none \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    python train_multigpu_colors.py \
        --data-path /path/to/large_dataset.dat \
        --n-samples 1000000000 \
        --n-features 1024 \
        --som-shape 100,100 \
        --n-colors 100
```

## Key Advantages

1. **Zero Data Movement**: Data stays on its assigned GPU throughout training
2. **Perfect Scaling**: Each GPU processes independently within colors
3. **Minimal Communication**: Only 40MB weight updates vs 1TB data movement
4. **Memory Efficient**: Each GPU only holds 1/Nth of the dataset
5. **Maintains Algorithm Correctness**: Sequential color processing is preserved

## Comparison with Alternatives

| Approach | Data Movement | Communication/Color | Scalability |
|----------|--------------|-------------------|-------------|
| Naive (redistribute) | O(N) per color | 1TB | Poor |
| This approach | 0 | 40MB | Excellent |
| Single GPU | N/A | N/A | Limited by memory |

## Conclusion

This architecture achieves near-linear scaling for color set processing by:
- Keeping data stationary on each GPU
- Processing colors in parallel across GPUs
- Synchronizing only weight updates (small)
- Maintaining algorithmic correctness with sequential color processing

The key insight: **move computation to data, not data to computation**.