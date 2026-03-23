"""
Tests for parameter coupling and dependencies between different configuration options.

Tests that parameters are correctly used or ignored based on the selected modes.
"""

import pytest
import warnings
import numpy as np
from floatsom.floatsom_params import (
    FloatSOMParams,
    SamplingConfig,
    ProcessingConfig,
    TopologyConfig,
)
from floatsom.base.floatsom_factories import create_floatsom


class TestSamplingParameterCoupling:
    """Test parameter dependencies for different sampling methods."""

    def test_full_sampling_ignores_proportion(self):
        """Full sampling should ignore target_proportion parameter."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="full",
                target_proportion=0.5  # Should be ignored
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # Full sampling should be selected regardless of proportion
        assert params.sampling_config.method == "full"
        assert params.sampling_config.target_proportion == 0.5  # Stored but not used

        som = create_floatsom(params)
        # Verify that full selector was created (ignores proportion)
        from floatsom.sampling.full_selector import FullSelector
        assert isinstance(som.selector, FullSelector)

    def test_random_sampling_uses_proportion(self):
        """Random sampling should respect target_proportion parameter."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="random",
                target_proportion=0.3
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.sampling_config.method == "random"
        assert params.sampling_config.target_proportion == 0.3

        som = create_floatsom(params)
        from floatsom.sampling.random_selector import RandomSelector
        assert isinstance(som.selector, RandomSelector)
        assert som.selector.target_proportion == 0.3

    def test_hdsssom_uses_block_size(self):
        """HDSSSOM sampling should use block_size parameter."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="hdsssom",
                block_size=2000,
                alpha=0.9
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.sampling_config.method == "hdsssom"
        assert params.sampling_config.block_size == 2000

        som = create_floatsom(params)
        from floatsom.sampling.hdsssom_selector import HDSSSOMSelector
        assert isinstance(som.selector, HDSSSOMSelector)
        assert som.selector.block_size == 2000

    def test_hdsssom_difficulty_parameters(self):
        """HDSSSOM should use difficulty-based selection parameters."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="hdsssom",
                p_block_difficulty=0.8,
                p_exemplar_difficulty=0.6,
                alpha=0.95
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.sampling_config.p_block_difficulty == 0.8
        assert params.sampling_config.p_exemplar_difficulty == 0.6
        assert params.sampling_config.alpha == 0.95

        som = create_floatsom(params)
        from floatsom.sampling.hdsssom_selector import HDSSSOMSelector
        assert isinstance(som.selector, HDSSSOMSelector)


class TestProcessingParameterCoupling:
    """Test parameter dependencies for different processing methods."""

    def test_batch_mode_only_for_batch_processing(self):
        """batch_mode parameter should only affect batch processing."""
        # Test batch processing uses batch_mode
        params_batch = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="minibatch",
                chunk_size=1000
            )
        )
        assert params_batch.processing_config.batch_mode == "minibatch"

        # Test colors processing ignores batch_mode (no validation error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params_colors = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    batch_mode="minibatch",  # Ignored for colors
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                )
            )
            # batch_mode is stored but not used for colors
            assert params_colors.processing_config.method == "colors"

    def test_colors_specific_parameters_only_for_colors(self):
        """Colors-specific parameters should only be validated for colors processing."""
        # Processing mode, sample_order, etc. only matter for colors
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                batch_mode="full_batch",
                # These are ignored for batch processing
                processing_mode="equal_sized",
                sample_order="random",
                chunk_size=1000
            )
        )

        # Should succeed - colors params ignored for batch
        assert params.processing_config.method == "batch"

    def test_chunk_size_required_for_all_methods(self):
        """chunk_size is required for all processing methods."""
        # Test with explicit chunk_size
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                chunk_size=5000
            )
        )
        assert params.processing_config.chunk_size == 5000

        # Test with None chunk_size (should default to 50000)
        params_default = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                method="batch",
                chunk_size=None
            )
        )
        assert params_default.processing_config.chunk_size == 50000

    def test_colors_caps_chunk_size_at_1m(self):
        """Colors processing should cap chunk_size at 1,000,000."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    chunk_size=2_000_000,  # Should be capped
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced"
                )
            )

            # Should be capped at 1M
            assert params.processing_config.chunk_size == 1_000_000


class TestTopologyParameterCoupling:
    """Test parameter dependencies for different topology types."""

    def test_mst_num_nodes_defaults_from_grid_size(self):
        """MST num_nodes should default from grid_size if not specified."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                grid_size=20,
                grid_dim=2,
                num_nodes=None  # Should default to grid_size^2
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # num_nodes should be set to grid_size^2 = 400
        assert params.topology_config.num_nodes == 400
        assert params.total_nodes == 400

    def test_mst_num_nodes_explicit_override(self):
        """Explicit num_nodes should override grid_size calculation."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                grid_size=20,
                num_nodes=500  # Explicit override
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.topology_config.num_nodes == 500
        assert params.total_nodes == 500

    def test_grid_total_nodes_from_grid_size(self):
        """Grid topology total_nodes calculated from grid_size."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=15,
                grid_dim=2
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # For 2D grid: total_nodes = grid_size^2
        assert params.total_nodes == 225

    def test_hexagonal_total_nodes_from_grid_size(self):
        """Hexagonal topology total_nodes calculated from grid_size."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="hexagonal",
                grid_size=12,
                grid_dim=2
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # For 2D hexagonal: total_nodes = grid_size^2
        assert params.total_nodes == 144

    def test_1d_grid_total_nodes(self):
        """1D grid should have total_nodes = grid_size."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=50,
                grid_dim=1
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # For 1D: total_nodes = grid_size
        assert params.total_nodes == 50

    def test_mst_update_frequency_parameters(self):
        """MST-specific update frequency parameters should be used."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                dynamic_mst_frequency=True,
                initial_mst_frequency=1,
                final_mst_frequency=20,
                mst_decay_function="exponential"
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.topology_config.dynamic_mst_frequency is True
        assert params.topology_config.initial_mst_frequency == 1
        assert params.topology_config.final_mst_frequency == 20
        assert params.topology_config.mst_decay_function == "exponential"

    def test_grid_hexagonal_ignore_mst_parameters(self):
        """Grid/hexagonal topologies should ignore MST-specific parameters."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                # These MST params are stored but not used
                mst_update_frequency=10,
                dynamic_mst_frequency=True
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # Should successfully create grid topology
        som = create_floatsom(params)
        from floatsom.topology.grid_topology import GridTopology
        assert isinstance(som.topology, GridTopology)


class TestDecayParameterCoupling:
    """Test decay parameter relationships."""

    def test_radius_decay_type_defaults_to_decay_type(self):
        """radius_decay_type should default to decay_type if not specified."""
        params = FloatSOMParams(
            input_dim=10,
            decay_type="exponential",
            radius_decay_type=None,  # Should default to decay_type
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.radius_decay_type == "exponential"

    def test_lr_decay_type_defaults_to_decay_type(self):
        """lr_decay_type should default to decay_type if not specified."""
        params = FloatSOMParams(
            input_dim=10,
            decay_type="sigmoid",
            lr_decay_type=None,  # Should default to decay_type
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.lr_decay_type == "sigmoid"

    def test_independent_decay_types(self):
        """radius_decay_type and lr_decay_type can be set independently."""
        params = FloatSOMParams(
            input_dim=10,
            decay_type="linear",
            radius_decay_type="exponential",
            lr_decay_type="asymptotic",
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.decay_type == "linear"
        assert params.radius_decay_type == "exponential"
        assert params.lr_decay_type == "asymptotic"

    def test_radius_decay_factor_defaults_by_topology(self):
        """radius_decay_factor should default differently for MST vs grid/hexagonal."""
        # MST should default to 1.0
        params_mst = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="mst"),
            radius_decay_factor=None,
            processing_config=ProcessingConfig(chunk_size=1000)
        )
        assert params_mst.radius_decay_factor == 1.0

        # Grid/hexagonal should default to 3.0
        params_grid = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="grid"),
            radius_decay_factor=None,
            processing_config=ProcessingConfig(chunk_size=1000)
        )
        assert params_grid.radius_decay_factor == 3.0


class TestInitialRadiusDefaults:
    """Test initial_radius default calculation."""

    def test_contextual_defaults_for_random_mst(self):
        """random+mst should use the tuned JSON-backed defaults."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(method="random"),
            topology_config=TopologyConfig(topology_type="mst", num_nodes=100),
            processing_config=ProcessingConfig(chunk_size=1000),
        )

        assert params.initial_radius == pytest.approx(1.8177068157674101)
        assert params.radius_decay_type == "asymptotic"
        assert params.initialization_method == "pca"
        assert params.processing_config.enable_momentum is True
        assert params.processing_config.initial_momentum == pytest.approx(0.5932963157239808)

    def test_initial_radius_defaults_for_grid(self):
        """Initial radius for grid should default to grid_size // 2."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=30
            ),
            initial_radius=None,
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # Should be grid_size // 2 = 15
        assert params.initial_radius == 15
        assert params.radius_decay_type == "exponential"
        assert params.initialization_method == "random"
        assert params.processing_config.enable_momentum is False
        assert params.processing_config.initial_momentum == pytest.approx(0.5)

    def test_initial_radius_defaults_for_mst(self):
        """Initial radius for MST should default to sqrt(num_nodes)."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="mst",
                num_nodes=100
            ),
            initial_radius=None,
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        # Should be sqrt(100) = 10
        assert params.initial_radius == 10

    def test_hdsssom_hexagonal_does_not_borrow_random_or_full_contextual_defaults(self):
        """Unsupported sampling/topology pairs should keep legacy defaults."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(method="hdsssom", alpha=0.9),
            topology_config=TopologyConfig(topology_type="hexagonal", grid_size=12),
            processing_config=ProcessingConfig(chunk_size=1000),
        )

        assert params.initial_radius == pytest.approx(6.0)
        assert params.radius_decay_type == "exponential"
        assert params.initialization_method == "random"
        assert params.processing_config.enable_momentum is False
        assert params.processing_config.initial_momentum == pytest.approx(0.5)

    def test_initial_radius_explicit_override(self):
        """Explicit initial_radius should override defaults."""
        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(
                topology_type="grid",
                grid_size=30
            ),
            initial_radius=20,  # Explicit override
            processing_config=ProcessingConfig(chunk_size=1000)
        )

        assert params.initial_radius == 20

    def test_contextual_default_fields_respect_explicit_overrides(self):
        """Explicit values should override JSON-backed contextual defaults."""
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(method="random"),
            topology_config=TopologyConfig(topology_type="rng"),
            initial_radius=9.0,
            radius_decay_type="linear",
            initialization_method="pca_density",
            processing_config=ProcessingConfig(
                chunk_size=1000,
                enable_momentum=False,
                initial_momentum=0.25,
            ),
        )

        assert params.initial_radius == pytest.approx(9.0)
        assert params.radius_decay_type == "linear"
        assert params.initialization_method == "pca_density"
        assert params.processing_config.enable_momentum is False
        assert params.processing_config.initial_momentum == pytest.approx(0.25)


class TestMomentumParameterCoupling:
    """Test momentum parameter dependencies."""

    def test_momentum_disabled_ignores_parameters(self):
        """When momentum is disabled, momentum parameters are stored but not used."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                enable_momentum=False,
                initial_momentum=0.9,  # Stored but not validated
                final_momentum=0.1,
                chunk_size=1000
            )
        )

        assert params.processing_config.enable_momentum is False
        # Parameters are stored but won't be used

    def test_momentum_enabled_uses_parameters(self):
        """When momentum is enabled, parameters are validated and used."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                enable_momentum=True,
                initial_momentum=0.7,
                final_momentum=0.1,
                momentum_decay_type="linear",
                chunk_size=1000
            )
        )

        assert params.processing_config.enable_momentum is True
        assert params.processing_config.initial_momentum == 0.7
        assert params.processing_config.final_momentum == 0.1
        assert params.processing_config.momentum_decay_type == "linear"


class TestNormalizationParameterCoupling:
    """Test normalization parameter dependencies."""

    def test_hybrid_normalization_requires_alpha(self):
        """Hybrid normalization requires and uses norm_alpha."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                normalization="hybrid",
                norm_alpha=0.6,
                chunk_size=1000
            )
        )

        assert params.processing_config.normalization == "hybrid"
        assert params.processing_config.norm_alpha == 0.6

    def test_clamped_weighted_requires_clamp_factor(self):
        """Clamped weighted normalization requires norm_clamp_factor."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                normalization="clamped_weighted",
                norm_clamp_factor=2.5,
                chunk_size=1000
            )
        )

        assert params.processing_config.normalization == "clamped_weighted"
        assert params.processing_config.norm_clamp_factor == 2.5

    def test_local_normalization_requires_percentile(self):
        """Local normalization requires norm_percentile."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                normalization="local",
                norm_percentile=90.0,
                chunk_size=1000
            )
        )

        assert params.processing_config.normalization == "local"
        assert params.processing_config.norm_percentile == 90.0

    def test_count_based_uses_virtual_ratio(self):
        """Count-based normalization uses virtual_ratio parameter."""
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                normalization="count_based",
                virtual_ratio=0.8,
                chunk_size=1000
            )
        )

        assert params.processing_config.normalization == "count_based"
        assert params.processing_config.virtual_ratio == 0.8
