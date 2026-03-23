# Memory-Efficient Color Set Processing Implementation

## Project Overview

This document outlines the comprehensive implementation plan for adding memory-efficient chunking to FloatSOM's color set processing. The implementation provides automatic memory optimization, shared utilities across all processors, and unified core processing logic that works seamlessly with both chunked and non-chunked data.

**Status:** Planning Complete - Ready for Implementation  
**Priority:** High  
**Estimated Impact:** 50%+ memory reduction for large color sets, elimination of OOM errors

## Background & Motivation

### Current Limitations

1. **Memory Constraints**: Color sets can exceed GPU memory limits causing OOM errors
2. **Manual Configuration**: No automatic chunk size optimization
3. **Code Duplication**: Memory utilities scattered across processor implementations
4. **Processing Inflexibility**: Core logic tightly coupled to specific input sizes

### Requirements Based on User Feedback

1. **Memory methods in base_processor.py** for shared use across all processors
2. **Color-specific memory calculations** in dedicated `color_memory_utils.py`
3. **Auto-calculation of optimal chunk size** with caching on initialization
4. **Abstract core functionality** to work uniformly with chunks or whole datasets

## Architecture Overview

```
floatsom/processing/
├── base_processor.py
│   ├── _get_available_gpu_memory()          # Shared GPU memory query
│   ├── _calculate_optimal_chunk_size()      # Auto-calculate based on memory/data
│   ├── _should_use_chunking()               # Intelligent chunking decision
│   └── cached_chunk_size                    # Cached optimal size
│
├── colour_operations/
│   └── color_memory_utils.py (NEW)
│       ├── estimate_color_set_memory_usage()    # Color-specific memory estimation
│       ├── calculate_color_processing_overhead() # Color algorithm overhead
│       └── determine_optimal_color_chunk_size()  # Optimized for color processing
│
└── colors_processor.py
    ├── initialize() → auto-calculates chunk_size
    ├── _process_color_set_core()             # Unified processing (any size)
    ├── _process_color_set_chunked()          # Chunked wrapper using core
    └── process_samples() → intelligent routing
```

## Implementation Phases

### Phase 1: Base Memory Infrastructure

**Objective:** Add shared memory utilities to base_processor.py

**Target File:** `floatsom/processing/base_processor.py`

**Implementation Details:**

```python
# Add after existing methods (around line 48)
from typing import Optional
import cupy as cp

def _get_available_gpu_memory(self) -> float:
    """Query available GPU memory in bytes (shared across all processors)"""
    try:
        free_mem, total_mem = cp.cuda.Device().mem_info
        return free_mem
    except Exception:
        return float('inf')  # Fallback for CPU or GPU errors

def _calculate_optimal_chunk_size(self, sample_shape: tuple, som_shape: tuple, 
                                processing_type: str = "batch") -> int:
    """Auto-calculate optimal chunk size based on available memory"""
    available_memory = self._get_available_gpu_memory()
    
    if processing_type == "color_set":
        from .colour_operations.color_memory_utils import determine_optimal_color_chunk_size
        return determine_optimal_color_chunk_size(available_memory, sample_shape, som_shape)
    else:
        # Standard batch processing calculation
        return self._calculate_batch_chunk_size(available_memory, sample_shape, som_shape)

def _should_use_chunking(self, data_size: int, optimal_chunk_size: int) -> bool:
    """Determine if chunking should be applied"""
    return data_size > optimal_chunk_size

def _calculate_batch_chunk_size(self, available_memory: float, 
                               sample_shape: tuple, som_shape: tuple) -> int:
    """Calculate chunk size for standard batch processing"""
    n_features = sample_shape[1]
    n_nodes = som_shape[0]
    
    # Target 60% of available memory
    target_memory = available_memory * 0.6
    
    # Memory per sample: BMU finding + influence matrix
    memory_per_sample = (n_nodes * 4) + (n_features * 4) + 32
    
    optimal_size = int(target_memory / memory_per_sample)
    return max(100, min(optimal_size, 10000))

# Add class-level caching
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cached_chunk_size: Optional[int] = None
    self.cached_processing_type: Optional[str] = None
```

**Testing Strategy:**
- Test base_processor changes with existing batch_processor first
- Validate memory query functions across different GPU configurations
- Ensure inheritance works correctly for all processor types

### Phase 2: Color Memory Utilities Module

**Objective:** Create dedicated color-specific memory calculation utilities

**New File:** `floatsom/processing/colour_operations/color_memory_utils.py`

**Key Functions:**

1. **estimate_color_set_memory_usage()** - Accurate memory estimation for color set processing
2. **calculate_color_processing_overhead()** - Color algorithm-specific overhead calculation
3. **determine_optimal_color_chunk_size()** - Optimized chunk size for color processing
4. **validate_memory_configuration()** - Comprehensive memory validation and recommendations

**Implementation:**

```python
"""
Memory utilities specific to color set processing
Provides accurate memory estimation and optimal chunk size calculation
"""

import cupy as cp
from typing import Tuple

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
    
    influence_memory = n_samples * n_nodes * 4
    update_memory = n_nodes * n_features * 8
    bmu_temp_memory = max(influence_memory * 0.5, update_memory * 2)
    overhead_memory = calculate_color_processing_overhead(1, n_samples)
    
    total_memory = influence_memory + update_memory + bmu_temp_memory + overhead_memory
    return total_memory

def determine_optimal_color_chunk_size(available_memory: float,
                                     sample_dimensions: Tuple[int, int],
                                     som_dimensions: Tuple[int, int]) -> int:
    """
    Calculate optimal chunk size for color set processing
    
    Considers:
    - Available GPU memory
    - Color set processing complexity
    - Safety factors for stable operation
    """
    n_features = sample_dimensions[1]
    
    # Handle both 1D and 2D SOM dimensions
    if len(som_dimensions) == 2 and som_dimensions[1] != n_features:
        n_nodes = som_dimensions[0] * som_dimensions[1]
    else:
        n_nodes = som_dimensions[0]
    
    # Target: use 60% of available memory for safety
    target_memory = available_memory * 0.6
    
    # Memory per sample in color set processing
    memory_per_sample = (n_nodes * 4) + (n_features * 8) + 64
    
    optimal_chunk_size = int(target_memory / memory_per_sample)
    
    # Clamp to reasonable bounds for color set processing
    return max(100, min(optimal_chunk_size, 10000))
```

**Testing Strategy:**
- Validate memory estimates against actual measurements
- Test with various SOM sizes and data dimensions
- Verify chunk size calculations produce optimal performance

### Phase 3: Core Abstraction Implementation

**Objective:** Create unified core processing function that works with any input size

**Target File:** `floatsom/processing/colors_processor.py`

**Key Methods:**

1. **_process_color_set_core()** - Unified processing logic (chunked or whole)
2. **_process_color_set_chunked()** - Chunked wrapper using core abstraction
3. **_apply_color_set_updates()** - Weight update application

**Implementation:**

```python
def _process_color_set_core(self, color_set_samples: cp.ndarray, 
                           color_set_indices: cp.ndarray,
                           som_weights: cp.ndarray, 
                           topology, params) -> cp.ndarray:
    """
    Core color set processing logic - works with any sample size
    
    Encapsulates essential steps:
    1. BMU finding
    2. HDSSSOM callback updates  
    3. Influence matrix calculation
    4. Weight updates
    """
    # Find BMUs using memory-efficient implementation
    bmus, distances = find_bmus(color_set_samples, som_weights, return_distances=True)
    
    # Update HDSSSOM callback if present (works incrementally)
    if self.selector_callback is not None:
        self.selector_callback(color_set_samples, bmus, distances)
    
    # Calculate influence matrix using precomputed topology data
    influence_matrix = self.influence_matrix[bmus]
    
    # Apply weight updates using existing color set logic
    return self._apply_color_set_updates(
        color_set_samples, bmus, influence_matrix, som_weights, params
    )

def _process_color_set_chunked(self, color_set_samples: cp.ndarray,
                              color_set_indices: cp.ndarray, 
                              som_weights: cp.ndarray, topology, params) -> cp.ndarray:
    """
    Process large color set using chunking with core abstraction
    Maintains color set semantics through sequential processing
    """
    chunk_size = self.cached_chunk_size
    n_samples = len(color_set_samples)
    
    if self.verbose:
        num_chunks = (n_samples + chunk_size - 1) // chunk_size
        print(f"Processing large color set ({n_samples} samples) in {num_chunks} chunks")
    
    current_weights = som_weights.copy()
    
    # Process color set in chunks using core abstraction
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        
        chunk_samples = color_set_samples[start_idx:end_idx]
        chunk_indices = color_set_indices[start_idx:end_idx] if color_set_indices is not None else None
        
        # Use same core logic - just with chunk instead of full set
        current_weights = self._process_color_set_core(
            chunk_samples, chunk_indices, current_weights, topology, params
        )
    
    return current_weights
```

**Design Principles:**
- **Consistency**: Same core logic for chunked and non-chunked processing
- **Preservation**: Color set semantics maintained across chunk boundaries
- **Flexibility**: Works with any color set size
- **Integration**: Seamless HDSSSOM callback support

### Phase 4: Enhanced Initialization & Integration

**Objective:** Add auto-calculation with caching and intelligent routing

**Target Methods in `colors_processor.py`:**

1. **Enhanced initialize()** - Auto-calculate and cache optimal chunk size
2. **Enhanced process_samples()** - Intelligent chunking decisions

**Implementation:**

```python
def initialize(self, som_weights, topology, params):
    """Enhanced initialization with auto-calculated chunk size caching"""
    # ... existing initialization code ...
    
    # Auto-calculate optimal chunk size and cache it
    if som_weights is not None:
        # Estimate typical sample dimensions
        typical_samples = 1000
        sample_shape = (typical_samples, som_weights.shape[1])
        som_shape = som_weights.shape
        
        # Calculate and cache optimal chunk size
        self.cached_chunk_size = self._calculate_optimal_chunk_size(
            sample_shape, som_shape, processing_type="color_set"
        )
        
        if self.verbose:
            print(f"Auto-calculated optimal color set chunk size: {self.cached_chunk_size}")

def process_samples(self, samples, som_weights, topology, params):
    """Enhanced sample processing with intelligent chunking routing"""
    # ... existing setup code ...
    
    updated_weights = som_weights
    
    # Process each color set with intelligent chunking decision
    for color_set_indices in self.color_sets:
        color_set_samples = samples[color_set_indices]
        
        # Intelligent routing: chunked vs direct processing
        if self._should_use_chunking(len(color_set_samples), self.cached_chunk_size):
            if self.verbose:
                print(f"Using chunked processing for color set with {len(color_set_samples)} samples")
            
            updated_weights = self._process_color_set_chunked(
                color_set_samples, color_set_indices, updated_weights, topology, params
            )
        else:
            # Use core abstraction for consistency
            updated_weights = self._process_color_set_core(
                color_set_samples, color_set_indices, updated_weights, topology, params
            )
    
    return updated_weights
```

### Phase 5: Optional Parameter Enhancements

**Objective:** Add configuration options for fine-tuning (optional)

**Target File:** `floatsom/floatsom_params.py`

**Optional additions to ProcessingConfig:**

```python
# Auto-chunking configuration (optional - existing chunk_size can remain)
enable_auto_chunk_optimization: bool = True        # Enable automatic calculation
chunk_size_override: Optional[int] = None          # Manual override if needed
memory_safety_factor: float = 0.6                  # Use 60% of available memory
```

## Implementation Sequence & Dependencies

```
Phase 1: Base Memory Methods     →  Phase 2: Color Memory Utils    →  Phase 3: Core Abstraction     →  Phase 4: Integration
    |                               |                               |                              |
    v                               v                               v                              v
base_processor.py               color_memory_utils.py          colors_processor.py          colors_processor.py
(shared utilities)              (color calculations)           (unified core method)        (intelligent routing)
    |                               |                               |                              |
[Test with batch_processor]    [Validate calculations]        [Test abstraction]           [Full integration test]
```

**Dependencies:**
- Phase 1 must complete before Phase 3 (base methods needed)
- Phase 2 can develop in parallel with Phase 1 (separate concerns)
- Phase 3 depends on Phase 2 (needs color memory estimates)
- Phase 4 depends on Phase 3 (needs core abstraction)

## Testing Strategy

### Phase-by-Phase Testing

1. **Base Infrastructure Testing**
   - Test base_processor additions with batch_processor first
   - Validate memory query functions across GPU configurations
   - Ensure inheritance works for all processor types

2. **Memory Utilities Validation**
   - Compare memory estimates against actual measurements
   - Test with various SOM sizes and data dimensions
   - Verify chunk size calculations produce optimal results

3. **Core Abstraction Testing**
   - Test with small color sets before large ones
   - Validate color set semantics preservation
   - Ensure HDSSSOM callbacks work correctly

4. **Integration Testing**
   - Full color set workflows with chunking decisions
   - Performance comparison: auto vs manual chunk sizes
   - Memory usage monitoring during processing

### Performance Benchmarks

**Memory Efficiency Targets:**
- Auto-calculated chunk sizes more effective than manual
- Large color sets (>5000 samples) process without OOM
- 50%+ memory reduction for oversized color sets

**Performance Targets:**
- <10% overhead for normal-sized color sets
- Auto-optimization faster than manual tuning
- Graceful scaling for any color set size

**Functional Correctness:**
- All existing color set processing semantics preserved
- Identical results for chunked vs non-chunked processing
- HDSSSOM callbacks work correctly with chunking

## Risk Mitigation

### Technical Risks

1. **Base Class Changes**
   - **Risk**: Breaking existing processor functionality
   - **Mitigation**: Test with batch_processor first, comprehensive regression testing

2. **Memory Estimation Accuracy**
   - **Risk**: Poor chunk size calculations
   - **Mitigation**: Validate against real measurements, implement fallbacks

3. **Core Abstraction Complexity**
   - **Risk**: Breaking color set processing semantics
   - **Mitigation**: Preserve existing interfaces, incremental testing

### Performance Risks

1. **Abstraction Overhead**
   - **Risk**: Performance degradation for normal cases
   - **Mitigation**: Benchmark and optimize, minimal abstraction layers

2. **Auto-Calculation Costs**
   - **Risk**: Slow initialization
   - **Mitigation**: Cache results, optimize calculation algorithms

### Integration Risks

1. **Configuration Compatibility**
   - **Risk**: Breaking existing configurations
   - **Mitigation**: Maintain backward compatibility, graceful fallbacks

2. **Memory Management Edge Cases**
   - **Risk**: Unexpected OOM errors
   - **Mitigation**: Comprehensive error handling, memory monitoring

## Fallback Mechanisms

1. **Manual Override**: chunk_size_override parameter for failed auto-calculation
2. **Graceful Degradation**: Fall back to existing non-chunked processing on errors
3. **Error Recovery**: Comprehensive logging and recovery for memory estimation failures
4. **Safety Limits**: Conservative memory usage with configurable safety factors

## Success Criteria

### Architecture Benefits
- ✅ **Shared Infrastructure**: Memory utilities work across all processor types
- ✅ **Auto-Optimization**: Chunk sizes calculated automatically and cached effectively
- ✅ **Unified Logic**: Single core function handles both chunked and non-chunked cases
- ✅ **Memory Efficiency**: Large color sets process without OOM errors

### Performance Metrics
- ✅ **Memory Reduction**: 50%+ reduction for large color sets
- ✅ **Performance**: <10% overhead for normal cases
- ✅ **Scalability**: Graceful handling of any color set size
- ✅ **Optimization**: Auto-calculation outperforms manual tuning

### Functional Requirements
- ✅ **Backward Compatibility**: Existing configurations work unchanged
- ✅ **Semantic Preservation**: Color set processing semantics maintained
- ✅ **Integration**: HDSSSOM callbacks work correctly with chunking
- ✅ **Reliability**: Comprehensive error handling and recovery

## Implementation Checklist

### Phase 1: Base Memory Infrastructure
- [ ] Add memory query methods to base_processor.py
- [ ] Implement chunk size calculation logic
- [ ] Add class-level caching for optimization
- [ ] Test with existing batch_processor

### Phase 2: Color Memory Utilities
- [ ] Create color_memory_utils.py module
- [ ] Implement memory estimation functions
- [ ] Add chunk size optimization for color processing
- [ ] Validate calculations with real data

### Phase 3: Core Abstraction
- [ ] Implement _process_color_set_core() method
- [ ] Create chunked processing wrapper
- [ ] Add weight update application logic
- [ ] Test abstraction with small color sets

### Phase 4: Integration & Auto-Optimization
- [ ] Enhance initialize() with auto-calculation
- [ ] Add intelligent routing in process_samples()
- [ ] Implement memory analysis and logging
- [ ] Full integration testing

### Phase 5: Documentation & Validation
- [ ] Update parameter documentation
- [ ] Create comprehensive test suite
- [ ] Performance benchmarking
- [ ] Final validation and cleanup

## Related Work & Dependencies

### Existing Memory Optimizations
- Memory-efficient BMU finding in utils.py (completed)
- In-place operations for find_bmus function
- Chunking infrastructure in batch_processor.py

### Future Enhancements
- In-place normalization functions (pending)
- Memory-efficient influence calculations (pending)
- Extended memory profiling and monitoring

### Integration Points
- FloatSOM parameter system (floatsom_params.py)
- Color set processing algorithms (colour_operations/)
- Topology and influence matrix systems

## Conclusion

This implementation plan provides a comprehensive approach to memory-efficient color set processing that addresses the specific feedback requirements:

1. **Shared Infrastructure**: Memory utilities in base_processor.py benefit all processors
2. **Specialized Calculations**: Color-specific memory analysis in dedicated module
3. **Auto-Optimization**: Intelligent chunk size calculation with caching
4. **Unified Processing**: Core abstraction works with any input size

The phased approach ensures systematic development with comprehensive testing and risk mitigation. The result will be a robust, memory-efficient system that eliminates OOM errors while maintaining all existing functionality and performance characteristics.

**Next Steps**: Begin implementation with Phase 1 (base memory infrastructure) for lowest risk and foundational benefit to all processor types.