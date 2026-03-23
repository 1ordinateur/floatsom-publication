"""
Tests for factory routing - verifying correct class instantiation.

Tests that the factory pattern correctly routes to the appropriate selector,
processor, and topology classes based on configuration.
"""

import pytest
import warnings
from floatsom.floatsom_params import (
    FloatSOMParams,
    SamplingConfig,
    ProcessingConfig,
    TopologyConfig,
)
from floatsom.base.floatsom_factories import create_floatsom

# Import selector classes
from floatsom.sampling.full_selector import FullSelector
from floatsom.sampling.random_selector import RandomSelector
from floatsom.sampling.hdsssom_selector import HDSSSOMSelector

# Import processor classes
from floatsom.processing.batch_processor import BatchProcessor
from floatsom.processing.colors_processor import ColorsProcessor
from floatsom.processing.serial_processor import SerialProcessor

# Import topology classes
from floatsom.topology.grid_topology import GridTopology
from floatsom.topology.hexagonal_topology import HexagonalTopology
from floatsom.topology.mst_topology import MSTTopology
from floatsom.topology.rng_topology import RNGTopology
from floatsom.topology.topology_factory import TopologyFactory


class TestSamplerFactoryRouting:
    """Test that sampling methods route to correct selector classes."""

    def test_full_sampling_creates_full_selector(self):
        """Full sampling method should create FullSelector."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(method="full"),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.selector, FullSelector)

    def test_random_sampling_creates_random_selector(self):
        """Random sampling method should create RandomSelector."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="random",
                target_proportion=0.2
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.selector, RandomSelector)
        assert som.selector.target_proportion == 0.2

    def test_hdsssom_sampling_creates_hdsssom_selector(self):
        """HDSSSOM sampling method should create HDSSSOMSelector."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="hdsssom",
                alpha=0.9,
                block_size=1500
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.selector, HDSSSOMSelector)
        assert som.selector.alpha == 0.9
        assert som.selector.block_size == 1500


class TestProcessorFactoryRouting:
    """Test that processing methods route to correct processor classes."""

    def test_batch_processing_creates_batch_processor(self):
        """Batch processing method should create BatchProcessor."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="full_batch",
                chunk_size=1000
            )
        )

        som = create_floatsom(params)
        assert isinstance(som.processor, BatchProcessor)

    def test_batch_minibatch_creates_batch_processor(self):
        """Minibatch mode should also create BatchProcessor."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="minibatch",
                chunk_size=1000
            )
        )

        som = create_floatsom(params)
        assert isinstance(som.processor, BatchProcessor)
        assert som.processor.mode == "minibatch"

    def test_colors_processing_creates_colors_processor(self):
        """Colors processing method should create ColorsProcessor."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                )
            )

            som = create_floatsom(params)
            assert isinstance(som.processor, ColorsProcessor)

    def test_serial_processing_creates_serial_processor(self):
        """Serial processing method should create SerialProcessor."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="serial",
                chunk_size=1000
            )
        )

        som = create_floatsom(params)
        assert isinstance(som.processor, SerialProcessor)

    def test_minisom_processing_creates_minisom_adapter(self):
        """MiniSOM processing method should create MiniSOMAdapter."""
        try:
            import minisom
            from floatsom.adapters.minisom_adapter import MiniSOMAdapter
            minisom_available = True
        except ImportError:
            minisom_available = False

        if not minisom_available:
            pytest.skip("MiniSOM package not available")

        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="grid"),
            processing_config=ProcessingConfig(
                method="minisom",
                chunk_size=1000
            )
        )

        som = create_floatsom(params)
        assert isinstance(som, MiniSOMAdapter)


class TestTopologyFactoryRouting:
    """Test that topology types route to correct topology classes."""

    def test_grid_topology_creates_grid_topology(self):
        """Grid topology type should create GridTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=20
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, GridTopology)

    def test_grid_planar_variant(self):
        """Grid with planar variant should create planar GridTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                topology_variant="planar",
                grid_size=20
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, GridTopology)
        assert som.topology.topology_type == "planar"

    def test_grid_toroidal_variant(self):
        """Grid with toroidal variant should create toroidal GridTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                topology_variant="toroidal",
                grid_size=20
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, GridTopology)
        assert som.topology.topology_type == "toroidal"

    def test_hexagonal_topology_creates_hexagonal_topology(self):
        """Hexagonal topology type should create HexagonalTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="hexagonal",
                grid_size=15
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, HexagonalTopology)

    def test_hexagonal_planar_variant(self):
        """Hexagonal with planar variant should create planar HexagonalTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="hexagonal",
                topology_variant="planar",
                grid_size=15
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, HexagonalTopology)
        assert som.topology.topology_type == "planar"

    def test_hexagonal_toroidal_variant(self):
        """Hexagonal with toroidal variant should create toroidal HexagonalTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="hexagonal",
                topology_variant="toroidal",
                grid_size=15
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, HexagonalTopology)
        assert som.topology.topology_type == "toroidal"

    def test_mst_topology_creates_mst_topology(self):
        """MST topology type should create MSTTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                num_nodes=100
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, MSTTopology)
        assert som.topology.num_nodes == 100

    def test_rng_topology_creates_rng_topology(self):
        """RNG topology type should create RNGTopology."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="rng",
                num_nodes=96
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, RNGTopology)
        assert som.topology.num_nodes == 96

    def test_validate_rng_topology_allows_none_mst_update_frequency(self):
        """RNG validation should accept dynamic update mode with None frequency."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="rng",
                num_nodes=96,
                mst_update_frequency=None,
            ),
            processing_config=ProcessingConfig(chunk_size=1000),
        )

        assert TopologyFactory.validate_topology_params("rng", params) is True


class TestCombinedFactoryRouting:
    """Test factory routing for combined selector + processor + topology."""

    @pytest.mark.parametrize("sampling,processor,topology", [
        ("full", "batch", "grid"),
        ("random", "colors", "hexagonal"),
        ("hdsssom", "serial", "mst"),
        ("hdsssom", "serial", "rng"),
        ("full", "serial", "hexagonal"),
        ("random", "batch", "mst"),
        ("random", "batch", "rng"),
    ])
    def test_combined_routing(self, sampling, processor, topology):
        """Test that combined configurations route to correct classes."""
        # Build configs
        sampling_kwargs = {"method": sampling}
        if sampling == "hdsssom":
            sampling_kwargs["alpha"] = 0.9

        processor_kwargs = {"method": processor, "chunk_size": 1000}
        if processor == "batch":
            processor_kwargs["batch_mode"] = "full_batch"
        elif processor == "colors":
            processor_kwargs.update({
                "processing_mode": "equal_sized",
                "sample_order": "random",
                "max_rounds": 1,
                "color_set_algorithm": "greedy_balanced",
            })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(**sampling_kwargs),
                processing_config=ProcessingConfig(**processor_kwargs),
                topology_config=TopologyConfig(topology_type=topology)
            )

            som = create_floatsom(params)

            # Verify correct selector
            if sampling == "full":
                assert isinstance(som.selector, FullSelector)
            elif sampling == "random":
                assert isinstance(som.selector, RandomSelector)
            elif sampling == "hdsssom":
                assert isinstance(som.selector, HDSSSOMSelector)

            # Verify correct processor
            if processor == "batch":
                assert isinstance(som.processor, BatchProcessor)
            elif processor == "colors":
                assert isinstance(som.processor, ColorsProcessor)
            elif processor == "serial":
                assert isinstance(som.processor, SerialProcessor)

            # Verify correct topology
            if topology == "grid":
                assert isinstance(som.topology, GridTopology)
            elif topology == "hexagonal":
                assert isinstance(som.topology, HexagonalTopology)
            elif topology == "mst":
                assert isinstance(som.topology, MSTTopology)
            elif topology == "rng":
                assert isinstance(som.topology, RNGTopology)


class TestProcessorConfigurationPropagation:
    """Test that processor configurations are properly propagated."""

    def test_batch_processor_receives_config(self):
        """BatchProcessor should receive complete configuration."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="minibatch",
                chunk_size=2000,
                enable_momentum=True,
                initial_momentum=0.8,
                final_momentum=0.2
            )
        )

        som = create_floatsom(params)
        assert isinstance(som.processor, BatchProcessor)
        assert som.processor.chunk_size == 2000
        assert som.processor.mode == "minibatch"

    def test_colors_processor_receives_config(self):
        """ColorsProcessor should receive complete configuration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="batch_all",
                    sample_order="strided",
                    max_rounds=3,
                    color_set_algorithm="systematic",
                    chunk_size=5000
                ),
                topology_config=TopologyConfig(topology_type="grid")
            )

            som = create_floatsom(params)
            assert isinstance(som.processor, ColorsProcessor)
            assert som.processor.processing_mode == "batch_all"
            assert som.processor.sample_order == "strided"
            assert som.processor.max_rounds == 3
            assert som.processor.color_set_algorithm == "systematic"


class TestTopologyConfigurationPropagation:
    """Test that topology configurations are properly propagated."""

    def test_grid_topology_receives_config(self):
        """GridTopology should receive complete configuration."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=25,
                topology_variant="toroidal"
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, GridTopology)
        assert som.topology.grid_size == 25
        assert som.topology.topology_type == "toroidal"

    def test_mst_topology_receives_config(self):
        """MSTTopology should receive complete configuration."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                num_nodes=150,
                dynamic_mst_frequency=True,
                initial_mst_frequency=1,
                final_mst_frequency=25
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        som = create_floatsom(params)
        assert isinstance(som.topology, MSTTopology)
        assert som.topology.num_nodes == 150
        assert som.topology.dynamic_mst_frequency is True


class TestFactoryErrorHandling:
    """Test that factory properly handles errors."""

    def test_invalid_sampling_method_raises_error(self):
        """Invalid sampling method should raise ValueError in factory."""
        with pytest.raises(ValueError, match="Invalid sampling method"):
            params = FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(method="invalid_method"),
                processing_config=ProcessingConfig(chunk_size=1000)
            )

    def test_invalid_processing_method_raises_error(self):
        """Invalid processing method should raise ValueError in factory."""
        with pytest.raises(ValueError, match="Invalid processing method"):
            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="invalid_method",
                    chunk_size=1000
                )
            )

    def test_invalid_topology_type_raises_error(self):
        """Invalid topology type should raise ValueError in factory."""
        with pytest.raises(ValueError, match="Invalid topology type"):
            params = FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(topology_type="invalid_type"),
                processing_config=ProcessingConfig(chunk_size=1000)
            )

    def test_minisom_with_mst_raises_error(self):
        """MiniSOM with MST topology should raise appropriate error."""
        try:
            import minisom
        except ImportError:
            pytest.skip("MiniSOM package not available")

        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="mst"),
            processing_config=ProcessingConfig(
                method="minisom",
                chunk_size=1000
            )
        )

        with pytest.raises(ValueError, match="MiniSOM only supports"):
            create_floatsom(params)


class TestFactoryArchitectureSummary:
    """Test that architecture summary is correctly generated."""

    def test_get_architecture_summary(self):
        """Verify architecture summary includes all mode information."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(method="random"),
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="full_batch",
                distance_metric="cosine",
                chunk_size=1000
            ),
            topology_config=TopologyConfig(
                topology_type="hexagonal",
                topology_variant="toroidal"
            )
        )

        summary = params.get_architecture_summary()

        # Verify summary contains key information
        assert "Random" in summary  # Sampling method
        assert "Batch" in summary  # Processing method
        assert "Hexagonal" in summary  # Topology type
        assert "toroidal" in summary  # Topology variant
        assert "cosine" in summary  # Distance metric
