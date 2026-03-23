# CPU-GPU Pipeline Parameter Refactoring

## Problem Statement

The current CPU-GPU pipeline parameter handling is convoluted and difficult to use:

### Current Issues
1. **Generic Dictionary Configs**: Workers use `Dict[str, Any]` with no type safety
2. **Too Many Parameters**: CPUGPUPipeline requires 9 parameters, MultiThreadedZarrLoader needs 8
3. **Inconsistent Units**: Mix of bytes, GB, fractions (0.5) and percentages
4. **Scattered Auto-detection**: Logic buried in helper methods like `_get_max_pinned_memory_gb()`
5. **Poor Developer Experience**: No IDE support, easy to make mistakes

### Test Script's Elegant Approach
The `test_zarr_gpu_pipeline.py` demonstrates a much cleaner approach:
```python
def __init__(self, zarr_path, chunk_size_mb=100, max_ram_usage_percent=0.8, no_copy=True)
```
- Just 4 intuitive parameters
- Human-friendly units (MB, percentages)
- Smart defaults that work out-of-the-box

## Proposed Solution

### 1. Configuration Classes

Create typed configuration dataclasses for better parameter management:

```python
from dataclasses import dataclass
from typing import Optional
import psutil

@dataclass
class PipelineConfig:
    """Simple, intuitive pipeline configuration"""
    # Memory settings (human-friendly units)
    chunk_size_mb: float = 100  # MB is more intuitive than bytes
    max_ram_percent: float = 0.7  # Percentage is clearer than absolute GB
    pinned_ram_percent: float = 0.5  # What % of total RAM for pinned memory
    
    # Threading settings
    n_cpu_threads: int = 4  # Sensible default
    n_buffers: int = 8  # Good for double/triple buffering
    
    # Optimization flags
    no_copy: bool = True  # Zero-copy by default
    random_seed: Optional[int] = None
    
    def to_bytes(self, mb: float) -> int:
        """Convert MB to bytes"""
        return int(mb * 1024 * 1024)
    
    def get_ram_gb(self) -> float:
        """Auto-detect available RAM in GB"""
        total_ram = psutil.virtual_memory().total
        return (total_ram * self.max_ram_percent) / (1024**3)
    
    def get_pinned_ram_gb(self) -> float:
        """Calculate pinned memory allocation"""
        return self.get_ram_gb() * self.pinned_ram_percent
```

### 2. Simplified CPUGPUPipeline

Reduce constructor to just 3 parameters:

```python
class CPUGPUPipeline:
    def __init__(self, 
                 zarr_path: str,
                 config: Optional[PipelineConfig] = None,
                 device: Optional[cp.cuda.Device] = None):
        """
        Simple 3-parameter constructor with smart defaults
        
        Args:
            zarr_path: Path to zarr array
            config: Pipeline configuration (uses smart defaults if None)
            device: CUDA device (auto-selects if None)
        """
        self.config = config or PipelineConfig()
        self.device = device or cp.cuda.Device()
        self.zarr_path = zarr_path
        
        # Convert user-friendly units to internal representation
        chunk_size = self.config.to_bytes(self.config.chunk_size_mb)
        max_ram_gb = self.config.get_ram_gb()
        pinned_ram_gb = self.config.get_pinned_ram_gb()
        
        # Create loader with converted values
        self.loader = MultiThreadedZarrLoader(
            zarr_path=zarr_path,
            chunk_size=chunk_size,
            n_cpu_threads=self.config.n_cpu_threads,
            n_buffers=self.config.n_buffers,
            max_memory_gb=pinned_ram_gb,
            max_total_ram_gb=max_ram_gb,
            random_seed=self.config.random_seed,
            no_copy=self.config.no_copy
        )
```

### 3. Worker Configuration

Replace generic dictionaries with typed configs:

```python
@dataclass
class WorkerConfig:
    """Typed worker configuration"""
    pipeline: PipelineConfig = None
    num_buffers: int = 3  # Worker-specific buffer count
    force_copy: bool = False  # Worker-specific overrides
    cache_path: Optional[str] = None
    
    def __post_init__(self):
        if self.pipeline is None:
            self.pipeline = PipelineConfig()

class RayPipelineBaseWorker:
    def __init__(self,
                 worker_id: int,
                 num_gpus: int,
                 config: Optional[WorkerConfig] = None):
        """
        Clean, typed configuration
        
        Args:
            worker_id: Worker identification
            num_gpus: Total GPUs in cluster
            config: Typed worker configuration
        """
        self.worker_id = worker_id
        self.num_gpus = num_gpus
        self.config = config or WorkerConfig()
        
        # Extract pipeline config
        self.chunk_size = self.config.pipeline.to_bytes(
            self.config.pipeline.chunk_size_mb
        )
        # ... rest of initialization
```

## Usage Examples

### Simple Usage (All Defaults)
```python
# Just works out of the box!
pipeline = CPUGPUPipeline(zarr_path="data.zarr")
```

### Custom Configuration
```python
# Intuitive parameter names and units
config = PipelineConfig(
    chunk_size_mb=50,  # 50 MB chunks
    max_ram_percent=0.8,  # Use 80% of available RAM
    n_cpu_threads=8  # 8 CPU threads
)
pipeline = CPUGPUPipeline(zarr_path="data.zarr", config=config)
```

### Worker Usage
```python
# Type-safe worker configuration
worker_config = WorkerConfig(
    pipeline=PipelineConfig(chunk_size_mb=100),
    num_buffers=5,  # Worker-specific override
    cache_path="/tmp/worker_cache"
)
worker = RayBatchWorker(
    worker_id=0,
    num_gpus=4,
    config=worker_config
)
```

## Benefits

1. **Type Safety**: Full IDE support with autocompletion and type checking
2. **Intuitive Units**: MB and percentages instead of bytes and fractions
3. **Smart Defaults**: Works immediately without configuration
4. **Single Source of Truth**: All configuration in one place
5. **Self-Documenting**: Clear parameter names and units
6. **Testable**: Easy to create test configurations
7. **Extensible**: Easy to add new parameters without breaking existing code

## Implementation Strategy

### Phase 1: Add New Classes (Non-breaking)
1. Create `PipelineConfig` and `WorkerConfig` dataclasses
2. Add overloaded constructors that accept both old and new styles
3. Internal conversion from new config to old parameters

### Phase 2: Migration
1. Update all internal code to use new configs
2. Add deprecation warnings for old-style parameters
3. Update documentation and examples

### Phase 3: Cleanup
1. Remove old parameter passing after deprecation period
2. Simplify internal code to only use config objects
3. Remove conversion logic

## Backwards Compatibility

During migration, support both styles:

```python
class CPUGPUPipeline:
    def __init__(self, 
                 zarr_path: str,
                 # New style (preferred)
                 config: Optional[PipelineConfig] = None,
                 # Old style (deprecated)
                 chunk_size: Optional[int] = None,
                 n_cpu_threads: Optional[int] = None,
                 **old_kwargs):
        
        if config is None and chunk_size is not None:
            # Convert old style to new
            warnings.warn(
                "Using old-style parameters is deprecated. "
                "Please use PipelineConfig instead.",
                DeprecationWarning
            )
            config = PipelineConfig(
                chunk_size_mb=chunk_size / (1024 * 1024),
                n_cpu_threads=n_cpu_threads or 4,
                # ... convert other parameters
            )
        
        self.config = config or PipelineConfig()
        # ... rest of initialization
```

## Testing

The new configuration system makes testing much easier:

```python
def test_pipeline_with_small_memory():
    """Test pipeline with limited memory"""
    config = PipelineConfig(
        chunk_size_mb=10,  # Small chunks
        max_ram_percent=0.1,  # Only use 10% of RAM
        n_buffers=3  # Minimum buffers
    )
    pipeline = CPUGPUPipeline("test.zarr", config)
    assert pipeline.loader.n_buffers == 3

def test_pipeline_defaults():
    """Test that defaults work correctly"""
    pipeline = CPUGPUPipeline("test.zarr")
    assert pipeline.config.chunk_size_mb == 100
    assert pipeline.config.max_ram_percent == 0.7
```

## Conclusion

This refactoring will dramatically improve the developer experience while maintaining all existing functionality. The parameter handling becomes intuitive, type-safe, and self-documenting, following the elegant approach demonstrated in the test script.