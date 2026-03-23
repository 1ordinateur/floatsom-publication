"""
Tests for processing methods in FloatSOM
Tests BatchProcessor and ColorsProcessor
"""

import numpy as np
import cupy as cp
import pytest

from floatsom.processing.batch_processor import BatchProcessor
from floatsom.processing.colors_processor import ColorsProcessor
from floatsom.processing.processing_params import BatchConfig, ColorsConfig
from floatsom.topology.grid_topology import GridTopology
from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig, TopologyConfig


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return cp.random.randn(500, 10).astype(cp.float32)


@pytest.fixture
def som_weights():
    """Create sample SOM weights"""
    return cp.random.randn(100, 10).astype(cp.float32)


@pytest.fixture
def grid_topology():
    """Create a simple grid topology"""
    topology = GridTopology(
        grid_size=10,
        input_dim=10,
        topology_type="planar",
        initialization_method="random",
        seed=42,
        verbose=False
    )
    return topology


@pytest.fixture
def floatsom_params(grid_topology):
    """Create FloatSOM parameters"""
    params = FloatSOMParams(
        input_dim=10,
        total_iterations=100,
        initial_learning_rate=0.5,
        initial_radius=5.0,
        use_gpu=True,
        verbose=False,
        topology_config=TopologyConfig(grid_size=10),
        processing_config=ProcessingConfig(
            chunk_size=100,
            method="batch",
            batch_mode="full_batch",
            normalization="none",
            enable_momentum=False,
            distance_metric="euclidean",
        )
    )
    params.current_radius = 3.0
    params.current_learning_rate = 0.1
    params.current_momentum = 0.0
    params.delta_weights = None
    params.current_iteration = 0

    return params


class TestBatchProcessor:
    """Tests for BatchProcessor"""

    def test_batch_processor_updates_weights(self, sample_data, som_weights, grid_topology, floatsom_params):
        """BatchProcessor should update weights after processing"""
        batch_config = BatchConfig(
            batch_mode="full_batch",
            chunk_size=100
        )
        processor = BatchProcessor(batch_config)

        # Initialize topology precomputation
        radii_list = [floatsom_params.current_radius]
        grid_topology.precompute_topology_data(floatsom_params, radii_list)

        # Initialize processor
        from floatsom.data.sources.array import ArrayDataSource
        data_source = ArrayDataSource(sample_data)
        processor.initialize(som_weights, grid_topology, floatsom_params, data_source)

        initial_weights = som_weights.copy()

        # Process samples
        result = processor.process_samples(
            sample_data[:100],
            som_weights,
            grid_topology,
            floatsom_params
        )

        # Extract updated weights
        if isinstance(result, tuple):
            updated_weights, _ = result
        else:
            updated_weights = result

        # Weights should have changed
        assert not cp.array_equal(initial_weights, updated_weights)

    def test_batch_processor_learning_rate_applied(self, sample_data, som_weights, grid_topology, floatsom_params):
        """Learning rate should affect update magnitude"""
        batch_config = BatchConfig(
            batch_mode="full_batch",
            chunk_size=100
        )
        processor = BatchProcessor(batch_config)

        # Initialize topology precomputation
        radii_list = [floatsom_params.current_radius]
        grid_topology.precompute_topology_data(floatsom_params, radii_list)

        # Initialize processor
        from floatsom.data.sources.array import ArrayDataSource
        data_source = ArrayDataSource(sample_data)
        processor.initialize(som_weights, grid_topology, floatsom_params, data_source)

        # Process with high learning rate
        floatsom_params.current_learning_rate = 0.5
        result_high = processor.process_samples(
            sample_data[:100],
            som_weights.copy(),
            grid_topology,
            floatsom_params
        )

        if isinstance(result_high, tuple):
            weights_high, _ = result_high
        else:
            weights_high = result_high

        # Process with low learning rate
        floatsom_params.current_learning_rate = 0.01
        result_low = processor.process_samples(
            sample_data[:100],
            som_weights.copy(),
            grid_topology,
            floatsom_params
        )

        if isinstance(result_low, tuple):
            weights_low, _ = result_low
        else:
            weights_low = result_low

        # High learning rate should produce larger changes
        change_high = cp.linalg.norm(weights_high - som_weights)
        change_low = cp.linalg.norm(weights_low - som_weights)
        assert change_high > change_low


class TestColorsProcessor:
    """Tests for ColorsProcessor"""

    def test_colors_processor_creates_color_sets(self, sample_data, som_weights, grid_topology, floatsom_params):
        """ColorsProcessor should create color sets"""
        colors_config = ColorsConfig(
            processing_mode="equal_sized",
            max_rounds=5,
            sample_order="sequential",
            color_set_algorithm="greedy_balanced",
            num_color_sets=None,
            chunk_size=100,
            enable_adaptive_bmu=False,
            bmu_recalc_initial=10,
            bmu_recalc_decay_type="exponential",
            bmu_recalc_min_samples=5
        )
        processor = ColorsProcessor(colors_config)

        # Initialize topology precomputation
        radii_list = [floatsom_params.current_radius]
        grid_topology.precompute_topology_data(floatsom_params, radii_list)

        # Initialize processor
        from floatsom.data.sources.array import ArrayDataSource
        data_source = ArrayDataSource(sample_data)
        processor.initialize(som_weights, grid_topology, floatsom_params, data_source)

        # Prepare color sets
        color_sets = processor._prepare_color_sets(
            sample_data[:200],
            som_weights,
            grid_topology,
            floatsom_params
        )

        # Color sets should be created
        assert color_sets is not None
        assert len(color_sets) > 0
        # Each color set should be a CuPy array
        for cs in color_sets:
            assert isinstance(cs, cp.ndarray)

    def test_colors_processor_processes_all_samples(self, sample_data, som_weights, grid_topology, floatsom_params):
        """ColorsProcessor should process all samples"""
        colors_config = ColorsConfig(
            processing_mode="equal_sized",
            max_rounds=5,
            sample_order="sequential",
            color_set_algorithm="greedy_balanced",
            num_color_sets=None,
            chunk_size=100,
            enable_adaptive_bmu=False,
            bmu_recalc_initial=10,
            bmu_recalc_decay_type="exponential",
            bmu_recalc_min_samples=5
        )
        processor = ColorsProcessor(colors_config)

        # Initialize topology precomputation
        radii_list = [floatsom_params.current_radius]
        grid_topology.precompute_topology_data(floatsom_params, radii_list)

        # Initialize processor
        from floatsom.data.sources.array import ArrayDataSource
        data_source = ArrayDataSource(sample_data)
        processor.initialize(som_weights, grid_topology, floatsom_params, data_source)

        initial_weights = som_weights.copy()

        # Process samples
        result = processor.process_samples(
            sample_data[:200],
            som_weights,
            grid_topology,
            floatsom_params
        )

        # Extract updated weights
        if isinstance(result, tuple):
            updated_weights, _ = result
        else:
            updated_weights = result

        # Weights should have changed
        assert not cp.array_equal(initial_weights, updated_weights)

    def test_colors_processor_sequential_updates(self, sample_data, som_weights, grid_topology, floatsom_params):
        """ColorsProcessor should apply updates sequentially per color set"""
        colors_config = ColorsConfig(
            processing_mode="equal_sized",
            max_rounds=3,
            sample_order="sequential",
            color_set_algorithm="greedy_balanced",
            num_color_sets=None,
            chunk_size=100,
            enable_adaptive_bmu=False,
            bmu_recalc_initial=10,
            bmu_recalc_decay_type="exponential",
            bmu_recalc_min_samples=5
        )
        processor = ColorsProcessor(colors_config)

        # Initialize topology precomputation
        radii_list = [floatsom_params.current_radius]
        grid_topology.precompute_topology_data(floatsom_params, radii_list)

        # Initialize processor
        from floatsom.data.sources.array import ArrayDataSource
        data_source = ArrayDataSource(sample_data)
        processor.initialize(som_weights, grid_topology, floatsom_params, data_source)

        # Process samples
        result = processor.process_full_iteration(
            sample_data[:200],
            som_weights.copy(),
            grid_topology,
            floatsom_params
        )

        # Extract updated weights
        if isinstance(result, tuple):
            updated_weights, _ = result
        else:
            updated_weights = result

        # Should successfully complete processing
        assert updated_weights is not None
        assert updated_weights.shape == som_weights.shape
