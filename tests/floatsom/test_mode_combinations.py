"""
Tests for valid mode combinations across sampling, processing, and topology dimensions.

Tests that all valid combinations of the three orthogonal mode dimensions work correctly.
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


class TestValidModeCombinations:
    """Test that valid mode combinations work correctly."""

    @pytest.mark.parametrize("sampling_method", ["full", "random", "hdsssom"])
    @pytest.mark.parametrize("processing_method", ["batch", "colors", "serial"])
    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal", "mst", "rng"])
    def test_all_valid_sampling_processing_topology_combinations(
        self, sampling_method, processing_method, topology_type
    ):
        """Test all valid combinations of sampling x processing x topology (3x3x3 = 27)."""
        # Build processing config with method-specific requirements
        processing_kwargs = {
            "method": processing_method,
            "chunk_size": 1000,
        }

        if processing_method == "batch":
            processing_kwargs["batch_mode"] = "full_batch"
        elif processing_method == "colors":
            processing_kwargs.update({
                "processing_mode": "equal_sized",
                "sample_order": "random",
                "max_rounds": 1,
                "color_set_algorithm": "greedy_balanced",
            })

        # Build sampling config
        sampling_kwargs = {"method": sampling_method}
        if sampling_method == "hdsssom":
            sampling_kwargs["alpha"] = 0.9

        # Suppress warnings for experimental combinations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create params
            params = FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(**sampling_kwargs),
                processing_config=ProcessingConfig(**processing_kwargs),
                topology_config=TopologyConfig(topology_type=topology_type),
            )

            # Verify params were created successfully
            assert params.sampling_config.method == sampling_method
            assert params.processing_config.method == processing_method
            assert params.topology_config.topology_type == topology_type

            # Verify FloatSOM can be instantiated
            som = create_floatsom(params)
            assert som is not None

    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal"])
    def test_minisom_only_with_grid_hexagonal(self, topology_type):
        """MiniSOM processing works only with grid and hexagonal topologies."""
        try:
            import minisom
        except ImportError:
            pytest.skip("MiniSOM package not available")

        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type=topology_type),
            processing_config=ProcessingConfig(
                method="minisom",
                chunk_size=1000
            ),
        )

        # Should successfully create FloatSOM/MiniSOM adapter
        som = create_floatsom(params)
        assert som is not None

    def test_minisom_rejects_mst(self):
        """MiniSOM processing should reject MST topology."""
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
            ),
        )

        with pytest.raises(ValueError, match="MiniSOM only supports 'grid' and 'hexagonal' topologies"):
            create_floatsom(params)

    def test_minisom_rejects_rng(self):
        """MiniSOM processing should reject RNG topology."""
        try:
            import minisom
        except ImportError:
            pytest.skip("MiniSOM package not available")

        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="rng"),
            processing_config=ProcessingConfig(
                method="minisom",
                chunk_size=1000
            ),
        )

        with pytest.raises(ValueError, match="MiniSOM only supports 'grid' and 'hexagonal' topologies"):
            create_floatsom(params)

    def test_each_combination_creates_valid_floatsom(self):
        """Test a comprehensive set of valid combinations."""
        valid_combinations = [
            # Full sampling with all processing/topology
            ("full", "batch", "grid"),
            ("full", "batch", "hexagonal"),
            ("full", "batch", "mst"),
            ("full", "batch", "rng"),
            ("full", "colors", "grid"),
            ("full", "colors", "hexagonal"),
            ("full", "serial", "grid"),

            # Random sampling with all processing/topology
            ("random", "batch", "grid"),
            ("random", "colors", "hexagonal"),
            ("random", "serial", "mst"),
            ("random", "serial", "rng"),

            # HDSSSOM sampling with all processing/topology
            ("hdsssom", "batch", "grid"),
            ("hdsssom", "colors", "hexagonal"),
            ("hdsssom", "serial", "mst"),
            ("hdsssom", "serial", "rng"),
        ]

        for sampling, processing, topology in valid_combinations:
            # Build configs
            sampling_kwargs = {"method": sampling}
            if sampling == "hdsssom":
                sampling_kwargs["alpha"] = 0.9

            processing_kwargs = {"method": processing, "chunk_size": 1000}
            if processing == "batch":
                processing_kwargs["batch_mode"] = "full_batch"
            elif processing == "colors":
                processing_kwargs.update({
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
                    processing_config=ProcessingConfig(**processing_kwargs),
                    topology_config=TopologyConfig(topology_type=topology),
                )

                som = create_floatsom(params)
                assert som is not None, f"Failed to create FloatSOM for {sampling}×{processing}×{topology}"


class TestTopologyVariantCombinations:
    """Test topology variant (planar/toroidal) combinations."""

    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal"])
    @pytest.mark.parametrize("topology_variant", ["planar", "toroidal"])
    def test_grid_hexagonal_support_both_variants(self, topology_type, topology_variant):
        """Grid and hexagonal topologies support both planar and toroidal variants."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type=topology_type,
                topology_variant=topology_variant
            ),
            processing_config=ProcessingConfig(chunk_size=1000),
        )

        assert params.topology_config.topology_type == topology_type
        assert params.topology_config.topology_variant == topology_variant

        # Should successfully create FloatSOM
        som = create_floatsom(params)
        assert som is not None

    @pytest.mark.parametrize("topology_type", ["mst", "rng"])
    def test_graph_topology_only_supports_planar(self, topology_type):
        """Graph topologies only support planar variants."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type=topology_type,
                topology_variant="planar"
            ),
            processing_config=ProcessingConfig(chunk_size=1000),
        )
        assert params.topology_config.topology_variant == "planar"

        with pytest.raises(ValueError, match="Toroidal topology is not supported"):
            FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(
                    topology_type=topology_type,
                    topology_variant="toroidal"
                ),
                processing_config=ProcessingConfig(chunk_size=1000),
            )


class TestBatchModeVariants:
    """Test batch processing mode variants."""

    @pytest.mark.parametrize("batch_mode", ["full_batch", "minibatch"])
    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal", "mst", "rng"])
    def test_batch_modes_with_all_topologies(self, batch_mode, topology_type):
        """Batch processing modes work with all topology types."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode=batch_mode,
                chunk_size=1000
            ),
            topology_config=TopologyConfig(topology_type=topology_type),
        )

        assert params.processing_config.batch_mode == batch_mode
        som = create_floatsom(params)
        assert som is not None


class TestColorsProcessingModes:
    """Test colors processing mode variants."""

    @pytest.mark.parametrize("processing_mode", ["equal_sized", "batch_all"])
    @pytest.mark.parametrize("sample_order", ["random", "strided"])
    def test_colors_processing_modes(self, processing_mode, sample_order):
        """Colors processing supports different modes and sample orders."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode=processing_mode,
                    sample_order=sample_order,
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                ),
                topology_config=TopologyConfig(topology_type="grid"),
            )

            assert params.processing_config.processing_mode == processing_mode
            assert params.processing_config.sample_order == sample_order

            som = create_floatsom(params)
            assert som is not None


class TestColorSetAlgorithms:
    """Test color set algorithm variants."""

    @pytest.mark.parametrize("algorithm", ["systematic", "greedy_balanced"])
    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal", "mst", "rng"])
    def test_color_set_algorithms_with_topologies(self, algorithm, topology_type):
        """Color set algorithms work with all topologies (with warnings for suboptimal combos)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm=algorithm,
                    chunk_size=1000
                ),
                topology_config=TopologyConfig(topology_type=topology_type),
            )

            # Verify warnings for suboptimal combinations
            if topology_type == "hexagonal" and algorithm == "systematic":
                assert any("Systematic color set algorithm" in str(warning.message) for warning in w)
            elif topology_type == "mst":
                assert any("MST topology is experimental" in str(warning.message) for warning in w)
            elif topology_type == "rng":
                assert any("RNG topology is experimental" in str(warning.message) for warning in w)

            som = create_floatsom(params)
            assert som is not None


class TestDistanceMetrics:
    """Test distance metric variants."""

    @pytest.mark.parametrize("distance_metric", ["euclidean", "cosine", "manhattan", "norm_p"])
    @pytest.mark.parametrize("processing_method", ["batch", "colors", "serial"])
    def test_distance_metrics_with_processing_methods(self, distance_metric, processing_method):
        """All distance metrics work with all processing methods."""
        processing_kwargs = {
            "method": processing_method,
            "distance_metric": distance_metric,
            "chunk_size": 1000,
        }

        if processing_method == "batch":
            processing_kwargs["batch_mode"] = "full_batch"
        elif processing_method == "colors":
            processing_kwargs.update({
                "processing_mode": "equal_sized",
                "sample_order": "random",
                "max_rounds": 1,
                "color_set_algorithm": "greedy_balanced",
            })

        if distance_metric == "norm_p":
            processing_kwargs["distance_metric_params"] = {"p": 3.0}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(**processing_kwargs),
            )

            assert params.processing_config.distance_metric == distance_metric
            som = create_floatsom(params)
            assert som is not None


class TestInfluenceFunctions:
    """Test influence function variants."""

    @pytest.mark.parametrize("influence_type", ["gaussian", "bubble", "mexican_hat", "triangle"])
    @pytest.mark.parametrize("topology_type", ["grid", "hexagonal", "mst", "rng"])
    def test_influence_functions_with_topologies(self, influence_type, topology_type):
        """All influence functions work with all topologies."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                influence_type=influence_type,
                chunk_size=1000
            ),
            topology_config=TopologyConfig(topology_type=topology_type),
        )

        assert params.processing_config.influence_type == influence_type
        som = create_floatsom(params)
        assert som is not None
