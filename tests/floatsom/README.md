# FloatSOM Mode Validation Test Suite

This test suite validates the three orthogonal mode dimensions of FloatSOM and their parameter dependencies.

## Test Structure

The suite consists of 88 tests across 4 modules, testing all aspects of mode validation and parameter coupling.

## Three Orthogonal Mode Dimensions

FloatSOM supports 3x4x3 = 36 potential combinations (27 valid core combinations):

1. **Sampling Methods** (3 options):
   - `full` - Process all samples every epoch
   - `random` - Random subsampling with target_proportion
   - `hdsssom` - Difficulty-based subsampling with block structure

2. **Processing Methods** (4 options):
   - `batch` - Batch gradient descent (full_batch or minibatch)
   - `colors` - Independent color set parallel processing
   - `serial` - Sequential sample-by-sample processing
   - `minisom` - MiniSOM adapter (grid/hexagonal only)

3. **Topology Types** (3 options):
   - `grid` - Rectangular grid (planar or toroidal)
   - `hexagonal` - Hexagonal grid (planar or toroidal)
   - `mst` - Minimum spanning tree (planar only)

## Test Modules

### 1. `test_floatsom_config_validation.py` (24 tests)

Tests that invalid configurations are properly rejected:

**Error Cases:**
- `test_mst_rejects_toroidal_variant` - MST + toroidal raises ValueError
- `test_minisom_rejects_mst_topology` - MiniSOM + MST raises ValueError
- `test_batch_requires_valid_batch_mode` - Invalid batch_mode raises ValueError
- `test_colors_requires_processing_mode` - Invalid processing_mode raises ValueError
- `test_colors_requires_max_rounds` - max_rounds < 1 raises ValueError
- `test_colors_requires_sample_order` - Invalid sample_order raises ValueError
- `test_colors_requires_color_set_algorithm` - Invalid color_set_algorithm raises ValueError
- `test_hdsssom_requires_alpha_in_bounds` - alpha not in (0,1] raises ValueError
- `test_hybrid_norm_requires_alpha` - Hybrid normalization without norm_alpha raises ValueError
- `test_norm_p_requires_p_param` - norm_p distance without p param defaults/validates
- `test_momentum_requires_initial_final` - Invalid momentum values raise ValueError

**Warning Cases:**
- `test_hexagonal_systematic_warns` - Hexagonal + systematic algorithm warns
- `test_mst_colors_warns` - MST + colors warns (experimental)
- `test_normalization_none_warns` - normalization=none warns about LR scaling

**Validation Cases:**
- Valid decay types (exponential, linear, sigmoid, gaussian, asymptotic, fixed)
- Valid sampling methods (full, random, hdsssom)
- Valid processing methods (batch, colors, serial, minisom)
- Valid topology types (grid, hexagonal, mst)
- Valid normalization methods (count_based, weighted, hybrid, etc.)

### 2. `test_mode_combinations.py` (11 tests)

Tests that valid mode combinations work correctly:

**Combination Tests:**
- `test_all_valid_sampling_processing_topology_combinations` - Parameterized test of 3x3x3=27 core combinations
- `test_minisom_only_with_grid_hexagonal` - MiniSOM compatibility validation
- `test_each_combination_creates_valid_floatsom` - Comprehensive valid combination test

**Variant Tests:**
- `test_grid_hexagonal_support_both_variants` - Planar/toroidal for grid/hexagonal
- `test_mst_only_supports_planar` - MST only supports planar
- `test_batch_modes_with_all_topologies` - full_batch/minibatch with all topologies
- `test_colors_processing_modes` - equal_sized/batch_all with random/strided
- `test_color_set_algorithms_with_topologies` - systematic/greedy_balanced with all topologies
- `test_distance_metrics_with_processing_methods` - All distance metrics with all processors
- `test_influence_functions_with_topologies` - All influence functions with all topologies

### 3. `test_parameter_coupling.py` (28 tests)

Tests parameter dependencies and coupling:

**Sampling Parameter Coupling:**
- `test_full_sampling_ignores_proportion` - Full sampling ignores target_proportion
- `test_random_sampling_uses_proportion` - Random sampling respects target_proportion
- `test_hdsssom_uses_block_size` - HDSSSOM uses block_size parameter
- `test_hdsssom_difficulty_parameters` - HDSSSOM difficulty parameters

**Processing Parameter Coupling:**
- `test_batch_mode_only_for_batch_processing` - batch_mode only affects batch processor
- `test_colors_specific_parameters_only_for_colors` - Colors params only for colors
- `test_chunk_size_required_for_all_methods` - chunk_size required/defaults
- `test_colors_caps_chunk_size_at_100k` - Colors caps chunk_size at 100,000

**Topology Parameter Coupling:**
- `test_mst_num_nodes_defaults_from_grid_size` - num_nodes defaults from grid_size^2
- `test_mst_num_nodes_explicit_override` - Explicit num_nodes overrides default
- `test_grid_total_nodes_from_grid_size` - Grid total_nodes = grid_size^2
- `test_hexagonal_total_nodes_from_grid_size` - Hexagonal total_nodes = grid_size^2
- `test_1d_grid_total_nodes` - 1D grid total_nodes = grid_size
- `test_mst_update_frequency_parameters` - MST update frequency parameters
- `test_grid_hexagonal_ignore_mst_parameters` - Grid/hex ignore MST params

**Decay Parameter Coupling:**
- `test_radius_decay_type_defaults_to_decay_type` - radius_decay_type defaults
- `test_lr_decay_type_defaults_to_decay_type` - lr_decay_type defaults
- `test_independent_decay_types` - Independent decay type setting
- `test_radius_decay_factor_defaults_by_topology` - Different defaults for MST (1.0) vs grid/hex (3.0)

**Other Parameter Coupling:**
- Initial radius defaults (grid_size//2 for grid/hex, sqrt(num_nodes) for MST)
- Momentum parameter coupling when enabled/disabled
- Normalization parameter dependencies (alpha, clamp_factor, percentile, virtual_ratio)

### 4. `test_factory_routing.py` (25 tests)

Tests that the factory pattern correctly instantiates classes:

**Sampler Factory Routing:**
- `test_full_sampling_creates_full_selector` - FullSelector instantiation
- `test_random_sampling_creates_random_selector` - RandomSelector instantiation
- `test_hdsssom_sampling_creates_hdsssom_selector` - HDSSSOMSelector instantiation

**Processor Factory Routing:**
- `test_batch_processing_creates_batch_processor` - BatchProcessor instantiation
- `test_batch_minibatch_creates_batch_processor` - Minibatch mode BatchProcessor
- `test_colors_processing_creates_colors_processor` - ColorsProcessor instantiation
- `test_serial_processing_creates_serial_processor` - SerialProcessor instantiation
- `test_minisom_processing_creates_minisom_adapter` - MiniSOMAdapter instantiation

**Topology Factory Routing:**
- `test_grid_topology_creates_grid_topology` - GridTopology instantiation
- `test_grid_planar_variant` - Planar GridTopology
- `test_grid_toroidal_variant` - Toroidal GridTopology
- `test_hexagonal_topology_creates_hexagonal_topology` - HexagonalTopology instantiation
- `test_hexagonal_planar_variant` - Planar HexagonalTopology
- `test_hexagonal_toroidal_variant` - Toroidal HexagonalTopology
- `test_mst_topology_creates_mst_topology` - MSTTopology instantiation

**Combined Factory Routing:**
- `test_combined_routing` - Parameterized test of combined selector+processor+topology
- Configuration propagation tests for all component types

**Factory Error Handling:**
- Invalid method/type error handling
- MiniSOM + MST incompatibility
- Architecture summary generation

## Running the Tests

### Run all mode validation tests:
```bash
pytest tests/test_mode_validation/ -v
```

### Run specific test module:
```bash
pytest tests/test_mode_validation/test_floatsom_config_validation.py -v
pytest tests/test_mode_validation/test_mode_combinations.py -v
pytest tests/test_mode_validation/test_parameter_coupling.py -v
pytest tests/test_mode_validation/test_factory_routing.py -v
```

### Run specific test class:
```bash
pytest tests/test_mode_validation/test_floatsom_config_validation.py::TestInvalidConfigurationRejection -v
```

### Run specific test:
```bash
pytest tests/test_mode_validation/test_floatsom_config_validation.py::TestInvalidConfigurationRejection::test_mst_rejects_toroidal_variant -v
```

## Key Validation Rules

### Topology Constraints:
- MST only supports planar variant (not toroidal)
- MiniSOM only supports grid and hexagonal topologies (not MST)

### Processing Constraints:
- Batch processing requires valid batch_mode (full_batch or minibatch)
- Colors processing requires: processing_mode, sample_order, max_rounds >= 1, color_set_algorithm
- Colors processing caps chunk_size at 100,000

### Sampling Constraints:
- HDSSSOM requires alpha in (0, 1]
- HDSSSOM probability parameters must be in [0, 1]

### Normalization Constraints:
- Hybrid normalization requires norm_alpha in [0, 1]
- Clamped weighted normalization requires norm_clamp_factor > 0
- Local normalization requires norm_percentile in (0, 100]
- Count-based normalization uses virtual_ratio in (0.0, 2.0]

### Distance Metric Constraints:
- norm_p distance requires p >= 1 (defaults to 3.0)

### Momentum Constraints:
- When enabled, initial_momentum and final_momentum must be in [0, 1]
- Valid momentum_decay_types: exponential, linear, fixed, inverse, sigmoid

### Decay Type Constraints:
- Valid decay types: exponential, linear, sigmoid, gaussian, asymptotic, fixed
- radius_decay_type and lr_decay_type default to decay_type if not specified
- radius_decay_factor defaults: 1.0 for MST, 3.0 for grid/hexagonal

## Coverage Summary

The test suite provides comprehensive coverage of:
- All 27 valid core mode combinations (3 sampling x 3 processing x 3 topology)
- MiniSOM compatibility (4th processing method with 2 compatible topologies)
- Invalid configuration rejection (11 error tests)
- Suboptimal configuration warnings (3 warning tests)
- Parameter coupling and dependencies (28 coupling tests)
- Factory routing to correct classes (25 routing tests)
- Configuration propagation through factory pattern
- Topology variant support (planar/toroidal)
- Batch mode variants (full_batch/minibatch)
- Colors processing variants (equal_sized/batch_all, random/strided)
- Color set algorithms (systematic/greedy_balanced)
- Distance metrics (euclidean/cosine/manhattan/norm_p)
- Influence functions (gaussian/bubble/mexican_hat/triangle)
- Normalization methods (8 variants)
- Decay types (6 variants)
- Momentum configuration
- Default value computation

## Architecture Summary

Each FloatSOMParams instance can generate an architecture summary:
```python
params = FloatSOMParams(...)
print(params.get_architecture_summary())
# Output: "FloatSOM: RandomSampling x BatchProcessing x HexagonalTopology (toroidal) | Distance: cosine, Influence: gaussian"
```

This helps verify the configured architecture is correct.
