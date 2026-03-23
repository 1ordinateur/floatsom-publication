# Processor Factory Refactoring Plan

## Overview

This document outlines the comprehensive refactoring plan to apply the successful batch processor architectural improvements to the colors processor and establish a centralized processor factory pattern for all processing methods in FloatSOM.

## Current Architecture Problems

### 1. Circular Dependencies
Both `BatchProcessor` and `ColorsProcessor` suffer from the same circular instantiation problem:

```python
# Current problematic pattern in colors_processor.py (lines 80-91)
if ray_config and RAY_AVAILABLE:
    # ColorsProcessor creates RayColorsProcessor in __init__
    from .ray_ops.ray_colors_processor import RayColorsProcessor
    self._ray_processor = RayColorsProcessor(...)
    
# RayColorsProcessor then inherits from ColorsProcessor!
class RayColorsProcessor(ColorsProcessor, RayGDSWorkerManager):
    # Child inherits from parent that instantiated it = circular!
```

### 2. Multiple Inheritance Complexity
- `RayBatchProcessor` inherits from both `BatchProcessor` AND `RayGDSWorkerManager`
- `RayColorsProcessor` inherits from both `ColorsProcessor` AND `RayGDSWorkerManager`
- This leads to awkward workarounds like:
  ```python
  # In ray_colors_processor.py line 78
  RayGDSWorkerManager.initialize(self, som_weights, topology, params)
  # Explicit call to avoid circular super() references
  ```

### 3. Parameter Marshaling Without Type Safety
- Processing parameters are scattered across multiple configuration classes
- Manual dictionary marshaling is error-prone
- No type checking for processor-specific parameters

### 4. Code Duplication
- Core algorithms duplicated between single-GPU and multi-GPU implementations
- No shared abstraction for common processing patterns

## Successful Batch Processor Refactoring

We've already successfully refactored `BatchProcessor` with these improvements:

### What We Did:
1. **Created Typed Dataclasses**: `BatchConfig` and `TrainingStepParams` in `floatsom_params.py`
2. **Implemented ProcessorFactory**: Clean factory pattern in `processor_factory.py`
3. **Removed Ray Logic from BatchProcessor**: Single-GPU only, no circular dependencies
4. **Converted RayBatchProcessor to Composition**: Uses `worker_manager` as member, not parent
5. **Extracted Shared Algorithms**: `calculate_normalized_updates()` in `utils.py`

### Results:
- ✅ No circular dependencies
- ✅ Clean separation of concerns
- ✅ Type safety with dataclasses
- ✅ ~30% code reduction
- ✅ Algorithmic equivalence maintained

## Proposed Refactoring Plan for Colors Processing

### Phase 1: Create Processing-Specific Parameters Module

**NEW FILE: `floatsom/processing/processing_params.py`**

```python
"""
Processing-specific parameter dataclasses
Centralizes all processor configurations in the processing module
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List

# ==================== BATCH PROCESSING ====================
# (Move existing BatchConfig and TrainingStepParams here)

@dataclass
class BatchConfig:
    """Configuration specifically for batch processing"""
    batch_mode: str = "full_batch"
    minibatch_size: int = 32
    chunk_size: Optional[int] = None
    memory_safety_factor: float = 0.6

@dataclass
class TrainingStepParams:
    """Typed parameters for a single training step"""
    radius: float
    learning_rate: float
    momentum: float = 0.0
    delta_weights: Optional[Any] = None
    # ... (other params)

# ==================== COLORS PROCESSING ====================

@dataclass
class ColorsConfig:
    """Configuration specifically for colors processing"""
    processing_mode: str = "equal_sized"  # "equal_sized" or "batch_all"
    max_rounds: int = 4
    sample_order: str = "strided"  # "strided" or "random"
    color_set_algorithm: str = "greedy_balanced"
    num_color_sets: Optional[int] = None
    memory_safety_factor: float = 0.6
    
    # Adaptive BMU parameters
    enable_adaptive_bmu: bool = False
    bmu_recalc_initial: Optional[int] = None
    bmu_recalc_decay_type: str = 'exponential'
    bmu_recalc_min_samples: int = 1000

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
    normalization: str = "count_based"
    total_samples: int = 0
    # Include normalization parameters
    norm_alpha: Optional[float] = None
    norm_clamp_factor: Optional[float] = None
    virtual_ratio: float = 0.5
```

### Phase 2: Extend ProcessorFactory

**Update `floatsom/processing/processor_factory.py`:**

```python
from .processing_params import BatchConfig, ColorsConfig, TrainingStepParams

class ProcessorFactory:
    @staticmethod
    def create(processing_config: ProcessingConfig) -> ProcessingMethod:
        """Create appropriate processor based on configuration"""
        method = processing_config.method
        
        if method == "batch":
            batch_config = ProcessorFactory._extract_batch_config(processing_config)
            return ProcessorFactory._create_batch_processor(
                batch_config, processing_config.ray_config
            )
        elif method == "colors":
            colors_config = ProcessorFactory._extract_colors_config(processing_config)
            return ProcessorFactory._create_colors_processor(
                colors_config, processing_config.ray_config
            )
    
    @staticmethod
    def _create_colors_processor(colors_config: ColorsConfig, 
                                ray_config: Optional[RayConfig]) -> ProcessingMethod:
        """Create colors processor - single or multi-GPU"""
        if ray_config is not None:
            try:
                from .ray_ops.ray_colors_processor import RayColorsProcessor
                logger.info("Creating RayColorsProcessor for multi-GPU")
                return RayColorsProcessor(colors_config, ray_config)
            except ImportError:
                logger.warning("Falling back to single-GPU ColorsProcessor")
        
        from .colors_processor import ColorsProcessor
        return ColorsProcessor(colors_config)
    
    @staticmethod
    def _extract_colors_config(config: ProcessingConfig) -> ColorsConfig:
        """Extract colors parameters from general config"""
        return ColorsConfig(
            processing_mode=getattr(config, 'processing_mode', 'equal_sized'),
            max_rounds=getattr(config, 'max_rounds', 4),
            sample_order=getattr(config, 'sample_order', 'strided'),
            color_set_algorithm=getattr(config, 'color_set_algorithm', 'greedy_balanced'),
            # ... other mappings
        )
```

### Phase 3: Clean Up ColorsProcessor

**Refactor `colors_processor.py`:**

```python
from .processing_params import ColorsConfig, ColorStepParams

class ColorsProcessor(ProcessingMethod):
    def __init__(self, colors_config: ColorsConfig):
        """
        Initialize colors processor for single-GPU training ONLY.
        No Ray logic here!
        """
        super().__init__()
        self.colors_config = colors_config
        
        # Extract parameters
        self.processing_mode = colors_config.processing_mode
        self.max_rounds = colors_config.max_rounds
        self.sample_order = colors_config.sample_order
        
        # Remove ALL Ray-related code (lines 79-101)
        # Remove self._ray_processor
        # Remove RAY_AVAILABLE checks
        
        logger.info(f"ColorsProcessor initialized for single-GPU")
```

**Key Changes:**
- Remove lines 21-30 (Ray imports)
- Remove lines 79-101 (Ray initialization)
- Remove lines 177-179 (Ray processor delegation)
- Remove lines 202-214 (GDS path handling)
- Remove lines 216-218 (Ray multi-GPU check)
- Remove lines 415-429 (_process_ray_multi_gpu method)
- Remove lines 484-487 (Ray cleanup)

### Phase 4: Convert RayColorsProcessor to Composition

**Refactor `ray_colors_processor.py`:**

```python
from ..processing_params import ColorsConfig, ColorStepParams
from ..base_processor import ProcessingMethod  # NOT from ColorsProcessor!

class RayColorsProcessor(ProcessingMethod):  # Single inheritance only!
    def __init__(self, colors_config: ColorsConfig, ray_config: RayConfig):
        """
        Initialize Ray colors processor using COMPOSITION.
        """
        super().__init__()
        
        # Store configurations
        self.colors_config = colors_config
        self.ray_config = ray_config
        
        # Use COMPOSITION instead of inheritance
        self.worker_manager = RayGDSWorkerManager(
            num_gpus=ray_config.num_gpus,
            ray_config=ray_config
        )
        
        # Color-specific utilities (could be extracted to a helper)
        self.color_sets = None
        self._cached_color_sets = None
        self._cached_radius = None
        
        # Extract parameters
        self.processing_mode = colors_config.processing_mode
        self.max_rounds = colors_config.max_rounds
        
        logger.info(f"RayColorsProcessor initialized with composition")
```

**Key Changes:**
- Change line 29 from multiple inheritance to single
- Remove `ColorsProcessor.__init__()` call (line 44)
- Remove `RayGDSWorkerManager.__init__()` call (lines 50-54)
- Add composition pattern with `self.worker_manager`
- Change line 78 to use `self.worker_manager.initialize()`

### Phase 5: Extract Shared Color Algorithms

**NEW FILE: `floatsom/processing/colour_operations/color_processing_utils.py`**

```python
"""
Shared color processing algorithms for single and multi-GPU implementations
"""
import cupy as cp
from typing import Tuple, List
from ..processing_params import ColorStepParams

def calculate_color_updates(
    samples: cp.ndarray,
    som_weights: cp.ndarray,
    topology,
    params: ColorStepParams,
    color_sets: List,
    chunk_size: int = 1000
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Core color processing algorithm shared between implementations.
    Encapsulates the color set update logic.
    """
    # Extract from current _process_equal_sized_strategy
    # Make reusable for both single and multi-GPU
    
def apply_color_normalization(
    accumulated_updates: cp.ndarray,
    accumulated_influence: cp.ndarray,
    params: ColorStepParams
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply color-specific normalization and momentum.
    """
    # Reuse existing apply_weight_updates_with_momentum
    # but with ColorStepParams
```

### Phase 6: Update Main Parameters File

**Simplify `floatsom/floatsom_params.py`:**

```python
@dataclass
class ProcessingConfig:
    """Simplified configuration for processing methods"""
    # General parameters only
    method: str = "colors"
    use_gpu: bool = True
    memory_safety_factor: float = 0.6
    
    # Ray configuration (common to all)
    ray_config: Optional[RayConfig] = None
    
    # Common normalization parameters
    normalization: str = "count_based"
    virtual_ratio: float = 0.5
    
    # REMOVE all processor-specific parameters
    # They now live in processing_params.py
```

## Implementation Strategy

### Order of Operations:

1. **Create `processing_params.py`** with all dataclasses
2. **Update imports** in existing batch processor files
3. **Extend ProcessorFactory** for colors support
4. **Clean up ColorsProcessor** - remove all Ray logic
5. **Convert RayColorsProcessor** to composition
6. **Extract shared algorithms** to utils
7. **Update RayColorWorker** to use shared functions
8. **Test** algorithmic equivalence

### Testing Plan:

```python
# Test algorithmic equivalence
def test_colors_equivalence():
    # Create same data and params
    data = create_test_data()
    params = create_test_params()
    
    # Single GPU
    single_proc = ColorsProcessor(colors_config)
    single_result = single_proc.process_samples(data, weights, topology, params)
    
    # Multi GPU
    ray_proc = RayColorsProcessor(colors_config, ray_config)
    ray_result = ray_proc.process_samples(data, weights, topology, params)
    
    # Should be mathematically identical
    assert np.allclose(single_result, ray_result, rtol=1e-6)
```

## Benefits

### Architectural Improvements:
- **No Circular Dependencies**: Factory pattern prevents parent-child circular references
- **Clean Separation**: Single vs multi-GPU logic completely separated
- **Type Safety**: Typed dataclasses reduce runtime errors
- **Composition Over Inheritance**: More flexible and maintainable

### Code Quality:
- **~40% Code Reduction**: Through shared algorithms and removal of duplication
- **Better Testability**: Components can be tested in isolation
- **Easier Maintenance**: Clear responsibilities and interfaces
- **Consistent Patterns**: Same architecture across all processors

### Performance:
- **No Algorithmic Changes**: Mathematical equivalence maintained
- **Efficient Memory Use**: Shared chunking strategies
- **Optimized GPU Usage**: Better memory management

## Migration Path

### For Existing Code:
```python
# OLD way (problematic)
processor = ColorsProcessor(ray_config=ray_config)

# NEW way (clean)
processor = ProcessorFactory.create(processing_config)
```

### For New Processors:
1. Add config dataclass to `processing_params.py`
2. Add creation method to `ProcessorFactory`
3. Implement processor with single responsibility
4. For multi-GPU, create separate class with composition

## Conclusion

This refactoring applies the proven patterns from the successful batch processor refactoring to the colors processor, establishing a consistent, maintainable architecture across all processing methods. The factory pattern eliminates circular dependencies while typed parameters provide safety and clarity.

The key insight is that **processing method selection and configuration should be separated from implementation**, allowing each processor to focus on its core algorithm without worrying about instantiation logic or multi-GPU delegation.