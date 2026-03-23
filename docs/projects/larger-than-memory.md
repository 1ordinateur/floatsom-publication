# Larger-Than-Memory Data Processing for FloatSOM

## Overview
This document outlines the architecture for processing datasets that exceed available memory by distributing data across multiple nodes and using memory-mapped arrays with coordinated weight updates via NCCL.

## Architecture

### Data Distribution Strategy
```
Node 0: [Memory Address 0x0000 - 0x1000] → Chunk 0
Node 1: [Memory Address 0x1000 - 0x2000] → Chunk 1
Node 2: [Memory Address 0x2000 - 0x3000] → Chunk 2
...
Node N: [Memory Address 0xN000 - 0xN+1000] → Chunk N
```

Each node is assigned a specific memory range from the memory-mapped file, ensuring no overlap and complete coverage of the dataset.

## Processing Flow

### Phase 1: Initialization
```python
# On each node
import numpy as np
import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Memory-mapped dataset (same file, different ranges per node)
dataset = np.memmap('large_dataset.dat', dtype='float32', mode='r', 
                     shape=(total_samples, n_features))

# Calculate this node's data range
samples_per_node = total_samples // size
start_idx = rank * samples_per_node
end_idx = start_idx + samples_per_node if rank < size - 1 else total_samples

# This node's data slice (virtual - not loaded yet)
node_data = dataset[start_idx:end_idx]
```

### Phase 2: Training Loop
```python
for iteration in range(total_iterations):
    # Step 1: Each node processes its chunk
    local_chunk = node_data[:]  # Load this node's data to CPU
    gpu_chunk = cp.asarray(local_chunk)  # Transfer to GPU
    
    # Step 2: Calculate local weight updates
    local_updates = calculate_weight_updates(
        gpu_chunk, 
        current_weights,
        topology,
        learning_rate,
        radius
    )
    
    # Step 3: Accumulate updates locally
    # Weight the updates by this node's sample proportion
    weight_factor = (end_idx - start_idx) / total_samples
    weighted_updates = local_updates * weight_factor
    
    # Step 4: All-reduce weight updates across nodes via NCCL
    global_updates = nccl_allreduce(weighted_updates)
    
    # Step 5: Update weights (same on all nodes after all-reduce)
    current_weights += global_updates
    
    # Step 6: Clear GPU memory for next iteration
    del gpu_chunk
    cp.cuda.MemoryPool().free_all_blocks()
```

## NCCL Communication Pattern

### Weight Update Synchronization
```
Iteration N:
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Node 0  │  │ Node 1  │  │ Node 2  │  │ Node 3  │
│ Update0 │  │ Update1 │  │ Update2 │  │ Update3 │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                        │
                  NCCL AllReduce
                        │
                   Global Update
                        │
     ┌────────────┬────────────┬────────────┐
     ▼            ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Weights0'│  │Weights1'│  │Weights2'│  │Weights3'│
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

All nodes end up with identical updated weights after NCCL all-reduce operation.

## Implementation Details

### Modified FloatSOM Training
```python
class DistributedFloatSOM(FloatSOM):
    def __init__(self, selector, processor, topology, params, comm):
        super().__init__(selector, processor, topology, params)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
    def train(self, dataset_path, total_samples, n_features):
        # Open memory-mapped file
        dataset = np.memmap(dataset_path, dtype='float32', mode='r',
                           shape=(total_samples, n_features))
        
        # Calculate this node's range
        samples_per_node = total_samples // self.size
        start_idx = self.rank * samples_per_node
        end_idx = start_idx + samples_per_node if self.rank < self.size - 1 else total_samples
        
        # Initialize weights (same on all nodes)
        self._initialize_training_distributed(dataset[start_idx:end_idx])
        
        for iteration in range(self.params.total_iterations):
            # Load and process this node's chunk
            local_data = dataset[start_idx:end_idx]
            gpu_data = cp.asarray(local_data)
            
            # Calculate local updates
            local_updates = self._calculate_local_updates(
                gpu_data, iteration
            )
            
            # Weight by sample proportion
            weight_factor = (end_idx - start_idx) / total_samples
            weighted_updates = local_updates * weight_factor
            
            # NCCL all-reduce to get global updates
            global_updates = self._nccl_allreduce(weighted_updates)
            
            # Apply updates (synchronized across all nodes)
            self.weights += global_updates
            
            # Clean up GPU memory
            del gpu_data
            cp.cuda.MemoryPool().free_all_blocks()
            
    def _nccl_allreduce(self, local_tensor):
        """NCCL all-reduce operation for weight synchronization"""
        # Convert to contiguous array if needed
        if not local_tensor.flags['C_CONTIGUOUS']:
            local_tensor = cp.ascontiguousarray(local_tensor)
            
        # Create NCCL communicator if not exists
        if not hasattr(self, 'nccl_comm'):
            from cupy.cuda import nccl
            nccl_id = nccl.get_unique_id() if self.rank == 0 else None
            nccl_id = self.comm.bcast(nccl_id, root=0)
            self.nccl_comm = nccl.NcclCommunicator(
                self.size, nccl_id, self.rank
            )
        
        # Perform all-reduce
        self.nccl_comm.allReduce(
            local_tensor.data.ptr,
            local_tensor.data.ptr,
            local_tensor.size,
            nccl.NCCL_FLOAT32,
            nccl.NCCL_SUM,
            cp.cuda.Stream.null.ptr
        )
        
        return local_tensor
```

### Memory-Mapped File Creation
```python
def create_memmap_dataset(input_files, output_path, dtype='float32'):
    """Convert multiple files to single memory-mapped array"""
    # First pass: calculate total size
    total_samples = 0
    n_features = None
    
    for file in input_files:
        data = np.load(file)
        total_samples += data.shape[0]
        if n_features is None:
            n_features = data.shape[1]
    
    # Create memory-mapped file
    mmap = np.memmap(output_path, dtype=dtype, mode='w+',
                      shape=(total_samples, n_features))
    
    # Second pass: copy data
    current_idx = 0
    for file in input_files:
        data = np.load(file)
        n_samples = data.shape[0]
        mmap[current_idx:current_idx + n_samples] = data
        current_idx += n_samples
        
    mmap.flush()
    return total_samples, n_features
```

## Advantages

1. **Memory Efficiency**: Each node only loads its assigned chunk
2. **Scalability**: Can handle datasets of any size by adding more nodes
3. **Synchronization**: NCCL ensures efficient weight synchronization
4. **Load Balancing**: Equal data distribution across nodes
5. **Fault Tolerance**: Can checkpoint weights periodically

## Configuration Example

```python
# Launch with mpirun
# mpirun -n 4 python train_distributed.py

config = {
    'dataset_path': '/path/to/large_dataset.dat',
    'total_samples': 100_000_000,
    'n_features': 1024,
    'batch_size': 50000,  # Per-node batch size
    'use_nccl': True,
    'checkpoint_every': 100  # iterations
}

# Each node automatically handles its portion
distributed_som = DistributedFloatSOM(
    selector=FullSelector(),
    processor=BatchProcessor(),
    topology=GridTopology(100, 100),
    params=params,
    comm=MPI.COMM_WORLD
)

distributed_som.train(
    config['dataset_path'],
    config['total_samples'],
    config['n_features']
)
```

## Performance Considerations

### Optimal Chunk Size
- Balance between GPU memory and I/O efficiency
- Typically 50K-100K samples per chunk
- Adjust based on feature dimensions

### NCCL Configuration
- Use InfiniBand for inter-node communication if available
- Enable GPUDirect for direct GPU-to-GPU transfers
- Set appropriate NCCL environment variables:
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_IB_DISABLE=0
  ```

### Memory Mapping Best Practices
- Use SSD storage for memory-mapped files
- Pre-allocate file to avoid fragmentation
- Consider using HDF5 for structured access patterns

## Future Enhancements

1. **Asynchronous I/O**: Overlap data loading with computation
2. **Dynamic Load Balancing**: Redistribute work based on node performance
3. **Hierarchical Reduction**: Use tree-based reduction for many nodes
4. **Compression**: Store memory-mapped data in compressed format
5. **Checkpointing**: Periodic weight snapshots for fault recovery