"""
Tests for FloatSOM configuration validation - invalid configuration rejection.

Tests that invalid parameter combinations are properly rejected with appropriate errors.
"""

import pytest
import warnings
import floatsom.floatsom_params as floatsom_params_module
from floatsom.floatsom_params import (
    FloatSOMParams,
    SamplingConfig,
    ProcessingConfig,
    TopologyConfig,
    load_contextual_floatsom_defaults,
)
from floatsom.processing.processing_params import DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS


class TestInvalidConfigurationRejection:
    """Test that invalid configurations raise appropriate errors."""

    def test_whole_chunk_random_defaults_off(self):
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(chunk_size=1000),
        )

        assert params.sampling_config.whole_chunk_random is False

    def test_whole_chunk_random_requires_random_sampling(self):
        with pytest.raises(ValueError, match="whole_chunk_random requires sampling_config.method='random'"):
            FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(
                    method="full",
                    whole_chunk_random=True,
                ),
                processing_config=ProcessingConfig(chunk_size=1000),
            )

    def test_whole_chunk_random_requires_batch_processing(self):
        with pytest.raises(ValueError, match="whole_chunk_random requires processing_config.method='batch'"):
            FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(
                    method="random",
                    whole_chunk_random=True,
                ),
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000,
                ),
            )

    def test_enable_multi_buffering_bool_is_not_silent(self):
        with pytest.warns(UserWarning, match="enable_multi_buffering expects an integer"):
            cfg = ProcessingConfig(chunk_size=1000, enable_multi_buffering=True)
        assert cfg.enable_multi_buffering == DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS

    def test_mst_rejects_toroidal_variant(self):
        """MST topology should reject toroidal variant."""
        with pytest.raises(ValueError, match="Toroidal topology is not supported for MST"):
            FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(
                    topology_type="mst",
                    topology_variant="toroidal"
                ),
                processing_config=ProcessingConfig(chunk_size=1000)
            )

    def test_minisom_rejects_mst_topology(self):
        """MiniSOM processing should reject MST topology."""
        try:
            import minisom
        except ImportError:
            pytest.skip("MiniSOM package not available")

        from floatsom.base.floatsom_factories import create_floatsom

        params = FloatSOMParams(
            input_dim=10,
            topology_config=TopologyConfig(topology_type="mst"),
            processing_config=ProcessingConfig(
                method="minisom",
                chunk_size=1000
            )
        )

        with pytest.raises(ValueError, match="MiniSOM only supports 'grid' and 'hexagonal' topologies"):
            create_floatsom(params)

    def test_batch_requires_valid_batch_mode(self):
        """Batch processing with invalid batch_mode should fail."""
        with pytest.raises(ValueError, match="Invalid batch_mode"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="batch",
                    batch_mode="invalid_mode",
                    chunk_size=1000
                )
            )

    def test_colors_requires_processing_mode(self):
        """Colors processing with invalid processing_mode should fail."""
        with pytest.raises(ValueError, match="Invalid processing_mode"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="invalid_mode",
                    chunk_size=1000,
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced"
                )
            )

    def test_colors_requires_max_rounds(self):
        """Colors processing requires max_rounds >= 1."""
        with pytest.raises(ValueError, match="max_rounds must be >= 1"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=0,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                )
            )

    def test_colors_requires_sample_order(self):
        """Colors processing requires valid sample_order."""
        with pytest.raises(ValueError, match="Invalid sample_order"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="invalid_order",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                )
            )

    def test_colors_requires_color_set_algorithm(self):
        """Colors processing requires valid color_set_algorithm."""
        with pytest.raises(ValueError, match="Invalid color_set_algorithm"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="invalid_algorithm",
                    chunk_size=1000
                )
            )

    def test_hdsssom_requires_alpha_in_bounds(self):
        """HDSSSOM sampling requires alpha in (0, 1]."""
        # Test alpha = 0 (invalid)
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(
                    method="hdsssom",
                    alpha=0.0
                ),
                processing_config=ProcessingConfig(chunk_size=1000)
            )

        # Test alpha > 1 (invalid)
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(
                    method="hdsssom",
                    alpha=1.5
                ),
                processing_config=ProcessingConfig(chunk_size=1000)
            )

        # Test alpha = 1.0 (valid - boundary)
        params = FloatSOMParams(
            input_dim=10,
            sampling_config=SamplingConfig(
                method="hdsssom",
                alpha=1.0
            ),
            processing_config=ProcessingConfig(chunk_size=1000)
        )
        assert params.sampling_config.alpha == 1.0

    def test_hybrid_norm_requires_alpha(self):
        """Hybrid normalization requires norm_alpha parameter."""
        with pytest.raises(ValueError, match="norm_alpha must be specified for hybrid normalization"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    normalization="hybrid",
                    chunk_size=1000
                )
            )

    def test_norm_p_requires_p_param(self):
        """norm_p distance metric requires p parameter."""
        # Should auto-default to p=3.0 if not provided
        params = FloatSOMParams(
            input_dim=10,
            processing_config=ProcessingConfig(
                distance_metric="norm_p",
                chunk_size=1000
            )
        )
        assert params.processing_config.distance_metric_params["p"] == 3.0

        # Should reject p < 1
        with pytest.raises(ValueError, match="p value for norm_p distance must be >= 1"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    distance_metric="norm_p",
                    distance_metric_params={"p": 0.5},
                    chunk_size=1000
                )
            )

    def test_momentum_requires_initial_final(self):
        """Momentum requires valid initial and final values."""
        # Test invalid initial_momentum
        with pytest.raises(ValueError, match="initial_momentum must be between 0 and 1"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    enable_momentum=True,
                    initial_momentum=1.5,
                    final_momentum=0.0,
                    chunk_size=1000
                )
            )

        # Test invalid final_momentum
        with pytest.raises(ValueError, match="final_momentum must be between 0 and 1"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    enable_momentum=True,
                    initial_momentum=0.5,
                    final_momentum=-0.1,
                    chunk_size=1000
                )
            )


class TestSuboptimalConfigurationWarnings:
    """Test that suboptimal configurations generate appropriate warnings."""

    def test_hexagonal_systematic_warns(self):
        """Hexagonal topology with systematic algorithm should warn."""
        with pytest.warns(UserWarning, match="Systematic color set algorithm may not be optimal"):
            FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(topology_type="hexagonal"),
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="systematic",
                    chunk_size=1000
                )
            )

    def test_mst_colors_warns(self):
        """MST topology with colors processing should warn (experimental)."""
        with pytest.warns(UserWarning, match="Color set processing with MST topology is experimental"):
            FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(topology_type="mst"),
                processing_config=ProcessingConfig(
                    method="colors",
                    processing_mode="equal_sized",
                    sample_order="random",
                    max_rounds=1,
                    color_set_algorithm="greedy_balanced",
                    chunk_size=1000
                )
            )


class TestContextualDefaultsValidation:
    """Test validation for JSON-backed contextual defaults."""

    def test_contextual_defaults_reject_unknown_parameter_name(self, monkeypatch, tmp_path):
        json_path = tmp_path / "invalid_contextual_defaults.json"
        json_path.write_text(
            '{"full": {"hexagonal": {"unexpected": 1.0}}}',
            encoding="utf-8",
        )

        monkeypatch.setattr(floatsom_params_module, "_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH", json_path)
        load_contextual_floatsom_defaults.cache_clear()
        try:
            with pytest.raises(ValueError, match="Unsupported contextual FloatSOM default key 'unexpected'"):
                load_contextual_floatsom_defaults()
        finally:
            load_contextual_floatsom_defaults.cache_clear()

    def test_contextual_defaults_reject_invalid_use_momentum_type(self, monkeypatch, tmp_path):
        json_path = tmp_path / "invalid_contextual_defaults.json"
        json_path.write_text(
            '{"random": {"mst": {"use_momentum": "true"}}}',
            encoding="utf-8",
        )

        monkeypatch.setattr(floatsom_params_module, "_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH", json_path)
        load_contextual_floatsom_defaults.cache_clear()
        try:
            with pytest.raises(ValueError, match="use_momentum' must be boolean"):
                load_contextual_floatsom_defaults()
        finally:
            load_contextual_floatsom_defaults.cache_clear()

    def test_normalization_none_warns(self):
        """Normalization=none should warn about scaling LR."""
        with pytest.warns(UserWarning, match="Normalization is set to none"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    normalization="none",
                    chunk_size=1000
                )
            )


class TestValidDecayTypes:
    """Test that valid decay types are accepted."""

    def test_valid_decay_types(self):
        """All valid decay types should be accepted."""
        valid_types = ["exponential", "linear", "sigmoid", "gaussian", "asymptotic", "fixed"]

        for decay_type in valid_types:
            params = FloatSOMParams(
                input_dim=10,
                decay_type=decay_type,
                processing_config=ProcessingConfig(chunk_size=1000)
            )
            assert params.decay_type == decay_type

    def test_invalid_decay_type(self):
        """Invalid decay type should be rejected."""
        with pytest.raises(ValueError, match="Invalid decay_type"):
            FloatSOMParams(
                input_dim=10,
                decay_type="invalid_decay",
                processing_config=ProcessingConfig(chunk_size=1000)
            )


class TestSamplingMethodValidation:
    """Test sampling method validation."""

    def test_valid_sampling_methods(self):
        """All valid sampling methods should be accepted."""
        valid_methods = ["full", "random", "hdsssom"]

        for method in valid_methods:
            params = FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(method=method),
                processing_config=ProcessingConfig(chunk_size=1000)
            )
            assert params.sampling_config.method == method

    def test_invalid_sampling_method(self):
        """Invalid sampling method should be rejected."""
        with pytest.raises(ValueError, match="Invalid sampling method"):
            FloatSOMParams(
                input_dim=10,
                sampling_config=SamplingConfig(method="invalid_sampling"),
                processing_config=ProcessingConfig(chunk_size=1000)
            )


class TestProcessingMethodValidation:
    """Test processing method validation."""

    def test_valid_processing_methods(self):
        """All valid processing methods should be accepted."""
        valid_methods = ["batch", "colors", "serial", "minisom"]

        for method in valid_methods:
            config_kwargs = {"method": method, "chunk_size": 1000}

            # Add required parameters for colors
            if method == "colors":
                config_kwargs.update({
                    "processing_mode": "equal_sized",
                    "sample_order": "random",
                    "max_rounds": 1,
                    "color_set_algorithm": "greedy_balanced"
                })

            params = FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(**config_kwargs)
            )
            assert params.processing_config.method == method

    def test_invalid_processing_method(self):
        """Invalid processing method should be rejected."""
        with pytest.raises(ValueError, match="Invalid processing method"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    method="invalid_processing",
                    chunk_size=1000
                )
            )


class TestTopologyTypeValidation:
    """Test topology type validation."""

    def test_valid_topology_types(self):
        """All valid topology types should be accepted."""
        valid_types = ["grid", "hexagonal", "mst", "rng"]

        for topology_type in valid_types:
            params = FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(topology_type=topology_type),
                processing_config=ProcessingConfig(chunk_size=1000)
            )
            assert params.topology_config.topology_type == topology_type

    def test_invalid_topology_type(self):
        """Invalid topology type should be rejected."""
        with pytest.raises(ValueError, match="Invalid topology type"):
            FloatSOMParams(
                input_dim=10,
                topology_config=TopologyConfig(topology_type="invalid_topology"),
                processing_config=ProcessingConfig(chunk_size=1000)
            )


class TestNormalizationValidation:
    """Test normalization method validation."""

    def test_valid_normalization_methods(self):
        """All valid normalization methods should be accepted."""
        valid_methods = [
            "count_based", "weighted", "hybrid", "clamped_weighted",
            "local", "adaptive", "none", "minisom_weighted"
        ]

        for method in valid_methods:
            config_kwargs = {"normalization": method, "chunk_size": 1000}

            # Add required parameters for specific normalizations
            if method == "hybrid":
                config_kwargs["norm_alpha"] = 0.5
            elif method == "clamped_weighted":
                config_kwargs["norm_clamp_factor"] = 2.0
            elif method == "local":
                config_kwargs["norm_percentile"] = 95.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress normalization=none warning
                params = FloatSOMParams(
                    input_dim=10,
                    processing_config=ProcessingConfig(**config_kwargs)
                )
                assert params.processing_config.normalization == method

    def test_invalid_normalization_method(self):
        """Invalid normalization method should be rejected."""
        with pytest.raises(ValueError, match="Invalid normalization"):
            FloatSOMParams(
                input_dim=10,
                processing_config=ProcessingConfig(
                    normalization="invalid_norm",
                    chunk_size=1000
                )
            )
