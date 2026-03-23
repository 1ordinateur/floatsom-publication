"""
Test MST clustering quality metrics.

Tests the MSTClusteringMetrics class including:
- Gap statistic computation
- Edge heterogeneity metrics
- Edge length ratios
- Bimodality coefficient
- Clustering score range
- Cut stability proxy
- Comprehensive comparison
"""

import pytest
import numpy as np

from floatsom.evaluation.mst_clustering_metrics import (
    MSTClusteringMetrics,
    print_comparison_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def uniform_weights():
    """Uniform edge weights (poor for clustering)."""
    rng = np.random.RandomState(42)
    return rng.uniform(1.0, 2.0, 50)


@pytest.fixture
def bimodal_weights():
    """Bimodal edge weights (good for clustering)."""
    rng = np.random.RandomState(42)
    # Intra-cluster edges (short)
    intra = rng.normal(1.0, 0.2, 40)
    # Inter-cluster edges (long)
    inter = rng.normal(5.0, 0.3, 10)
    return np.concatenate([intra, inter])


@pytest.fixture
def single_weight():
    """Single edge weight."""
    return np.array([1.0])


@pytest.fixture
def empty_weights():
    """Empty edge weights array."""
    return np.array([])


@pytest.fixture
def constant_weights():
    """Constant edge weights."""
    return np.ones(50)


# =============================================================================
# Gap Statistic Tests
# =============================================================================

class TestGapStatistic:
    """Test gap statistic computation."""

    def test_returns_dict(self, uniform_weights):
        """Should return dictionary with expected keys."""
        result = MSTClusteringMetrics.gap_statistic(uniform_weights)

        assert isinstance(result, dict)
        assert 'max_gap' in result
        assert 'gap_cv' in result
        assert 'gap_ratio' in result

    def test_bimodal_has_larger_gap_ratio(self, uniform_weights, bimodal_weights):
        """Bimodal weights should have higher gap ratio."""
        uniform_result = MSTClusteringMetrics.gap_statistic(uniform_weights)
        bimodal_result = MSTClusteringMetrics.gap_statistic(bimodal_weights)

        assert bimodal_result['gap_ratio'] > uniform_result['gap_ratio'], \
            "Bimodal should have larger gap ratio"

    def test_max_gap_non_negative(self, uniform_weights):
        """Max gap should be non-negative."""
        result = MSTClusteringMetrics.gap_statistic(uniform_weights)
        assert result['max_gap'] >= 0

    def test_empty_weights(self, empty_weights):
        """Should handle empty weights gracefully."""
        result = MSTClusteringMetrics.gap_statistic(empty_weights)

        assert result['max_gap'] == 0
        assert result['gap_cv'] == 0
        assert result['gap_ratio'] == 0

    def test_single_weight(self, single_weight):
        """Should handle single weight gracefully."""
        result = MSTClusteringMetrics.gap_statistic(single_weight)

        assert result['max_gap'] == 0
        assert result['gap_ratio'] == 0


# =============================================================================
# Edge Heterogeneity Tests
# =============================================================================

class TestEdgeHeterogeneity:
    """Test edge heterogeneity metrics."""

    def test_returns_dict(self, uniform_weights):
        """Should return dictionary with expected keys."""
        result = MSTClusteringMetrics.edge_heterogeneity(uniform_weights)

        assert isinstance(result, dict)
        assert 'gini' in result
        assert 'entropy' in result
        assert 'cv' in result

    def test_gini_bounds(self, uniform_weights):
        """Gini coefficient should be in valid range."""
        result = MSTClusteringMetrics.edge_heterogeneity(uniform_weights)

        # Gini is typically in [0, 1] but can go slightly beyond
        assert result['gini'] >= -0.1
        assert result['gini'] <= 1.1

    def test_bimodal_higher_gini(self, uniform_weights, bimodal_weights):
        """Bimodal weights should have higher Gini (more inequality)."""
        uniform_result = MSTClusteringMetrics.edge_heterogeneity(uniform_weights)
        bimodal_result = MSTClusteringMetrics.edge_heterogeneity(bimodal_weights)

        assert bimodal_result['gini'] > uniform_result['gini'], \
            "Bimodal should have higher Gini coefficient"

    def test_entropy_non_negative(self, uniform_weights):
        """Entropy should be non-negative."""
        result = MSTClusteringMetrics.edge_heterogeneity(uniform_weights)
        assert result['entropy'] >= 0

    def test_cv_non_negative(self, uniform_weights):
        """Coefficient of variation should be non-negative."""
        result = MSTClusteringMetrics.edge_heterogeneity(uniform_weights)
        assert result['cv'] >= 0

    def test_constant_weights_low_cv(self, constant_weights):
        """Constant weights should have zero CV."""
        result = MSTClusteringMetrics.edge_heterogeneity(constant_weights)
        assert result['cv'] == 0

    def test_empty_weights(self, empty_weights):
        """Should handle empty weights gracefully."""
        result = MSTClusteringMetrics.edge_heterogeneity(empty_weights)

        assert result['gini'] == 0
        assert result['entropy'] == 0
        assert result['cv'] == 0


# =============================================================================
# Edge Length Ratio Tests
# =============================================================================

class TestEdgeLengthRatio:
    """Test edge length ratio metrics."""

    def test_returns_dict(self, uniform_weights):
        """Should return dictionary with expected keys."""
        result = MSTClusteringMetrics.edge_length_ratio(uniform_weights)

        assert isinstance(result, dict)
        assert 'max_min_ratio' in result
        assert 'top_10_percent_ratio' in result
        assert 'percentile_90_10' in result

    def test_ratios_at_least_one(self, uniform_weights):
        """Ratios should be at least 1 (max >= min)."""
        result = MSTClusteringMetrics.edge_length_ratio(uniform_weights)

        assert result['max_min_ratio'] >= 1
        assert result['top_10_percent_ratio'] >= 1
        assert result['percentile_90_10'] >= 1

    def test_bimodal_higher_ratios(self, uniform_weights, bimodal_weights):
        """Bimodal weights should have higher ratios."""
        uniform_result = MSTClusteringMetrics.edge_length_ratio(uniform_weights)
        bimodal_result = MSTClusteringMetrics.edge_length_ratio(bimodal_weights)

        assert bimodal_result['top_10_percent_ratio'] > uniform_result['top_10_percent_ratio'], \
            "Bimodal should have higher top/bottom ratio"

    def test_empty_weights(self, empty_weights):
        """Should handle empty weights gracefully."""
        result = MSTClusteringMetrics.edge_length_ratio(empty_weights)

        assert result['max_min_ratio'] == 1
        assert result['top_10_percent_ratio'] == 1
        assert result['percentile_90_10'] == 1

    def test_constant_weights_ratio_one(self, constant_weights):
        """Constant weights should have ratio of 1."""
        result = MSTClusteringMetrics.edge_length_ratio(constant_weights)

        np.testing.assert_allclose(result['max_min_ratio'], 1.0, rtol=1e-10)


# =============================================================================
# Bimodality Coefficient Tests
# =============================================================================

class TestBimodalityCoefficient:
    """Test bimodality coefficient."""

    def test_returns_float(self, uniform_weights):
        """Should return a float."""
        result = MSTClusteringMetrics.bimodality_coefficient(uniform_weights)
        assert isinstance(result, (float, np.floating))

    def test_bimodal_higher_coefficient(self, uniform_weights, bimodal_weights):
        """Bimodal weights should have higher bimodality coefficient."""
        uniform_bc = MSTClusteringMetrics.bimodality_coefficient(uniform_weights)
        bimodal_bc = MSTClusteringMetrics.bimodality_coefficient(bimodal_weights)

        assert bimodal_bc > uniform_bc, \
            "Bimodal should have higher bimodality coefficient"

    def test_short_array_returns_zero(self):
        """Arrays with < 4 elements should return 0."""
        short = np.array([1.0, 2.0, 3.0])
        result = MSTClusteringMetrics.bimodality_coefficient(short)
        assert result == 0

    def test_constant_weights_zero(self, constant_weights):
        """Constant weights should have zero bimodality."""
        result = MSTClusteringMetrics.bimodality_coefficient(constant_weights)
        assert result == 0


# =============================================================================
# Clustering Score Range Tests
# =============================================================================

class TestClusteringScoreRange:
    """Test clustering score range computation."""

    def test_returns_dict(self, uniform_weights):
        """Should return dictionary with expected keys."""
        result = MSTClusteringMetrics.clustering_score_range(uniform_weights)

        assert isinstance(result, dict)
        assert 'mean_separation' in result
        assert 'max_separation' in result

    def test_separation_at_least_one(self, uniform_weights):
        """Separation scores should be at least 1."""
        result = MSTClusteringMetrics.clustering_score_range(uniform_weights)

        assert result['mean_separation'] >= 1
        assert result['max_separation'] >= 1

    def test_bimodal_higher_separation(self, uniform_weights, bimodal_weights):
        """Bimodal weights should have higher separation scores."""
        uniform_result = MSTClusteringMetrics.clustering_score_range(uniform_weights)
        bimodal_result = MSTClusteringMetrics.clustering_score_range(bimodal_weights)

        assert bimodal_result['mean_separation'] > uniform_result['mean_separation'], \
            "Bimodal should have higher mean separation"

    def test_custom_cluster_range(self, uniform_weights):
        """Should accept custom cluster range."""
        result = MSTClusteringMetrics.clustering_score_range(
            uniform_weights,
            n_clusters_range=[2, 3]
        )

        assert 'mean_separation' in result


# =============================================================================
# Cut Stability Proxy Tests
# =============================================================================

class TestCutStabilityProxy:
    """Test cut stability proxy metrics."""

    def test_returns_dict(self, uniform_weights):
        """Should return dictionary with expected keys."""
        result = MSTClusteringMetrics.cut_stability_proxy(uniform_weights)

        assert isinstance(result, dict)
        assert 'gap_uniformity' in result
        assert 'relative_gap_variance' in result

    def test_uniformity_bounded(self, uniform_weights):
        """Gap uniformity should be in [0, 1]."""
        result = MSTClusteringMetrics.cut_stability_proxy(uniform_weights)

        assert 0 <= result['gap_uniformity'] <= 1

    def test_variance_non_negative(self, uniform_weights):
        """Relative gap variance should be non-negative."""
        result = MSTClusteringMetrics.cut_stability_proxy(uniform_weights)

        assert result['relative_gap_variance'] >= 0

    def test_short_array(self):
        """Short arrays should return default values."""
        short = np.array([1.0])
        result = MSTClusteringMetrics.cut_stability_proxy(short)

        assert result['gap_uniformity'] == 1
        assert result['relative_gap_variance'] == 0


# =============================================================================
# Comprehensive Comparison Tests
# =============================================================================

class TestComprehensiveComparison:
    """Test comprehensive MST comparison."""

    def test_returns_dict(self, uniform_weights, bimodal_weights):
        """Should return comprehensive results dictionary."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights
        )

        assert isinstance(result, dict)
        assert 'MST1' in result
        assert 'MST2' in result
        assert 'comparison' in result
        assert 'overall_scores' in result
        assert 'recommendation' in result

    def test_bimodal_preferred(self, uniform_weights, bimodal_weights):
        """Bimodal MST should be preferred for clustering."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights
        )

        assert result['overall_scores']['MST2'] > result['overall_scores']['MST1'], \
            "Bimodal (MST2) should have higher overall score"

    def test_custom_weights(self, uniform_weights, bimodal_weights):
        """Should accept custom metric weights."""
        custom_weights = {
            'gap_ratio': 2.0,
            'gini': 0.5,
            'edge_ratio': 1.0,
            'bimodality': 0.0,
            'separation': 1.0,
            'stability': 0.0
        }

        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights,
            weight_importance=custom_weights
        )

        assert 'overall_scores' in result

    def test_comparison_contains_metrics(self, uniform_weights, bimodal_weights):
        """Comparison should contain all expected metrics."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights
        )

        expected_metrics = ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']

        for metric in expected_metrics:
            assert metric in result['comparison'], f"Missing metric: {metric}"

    def test_each_comparison_has_better_field(self, uniform_weights, bimodal_weights):
        """Each metric comparison should indicate which MST is better."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights
        )

        for metric, values in result['comparison'].items():
            assert 'better' in values
            assert values['better'] in ['MST1', 'MST2']


# =============================================================================
# Edge Cases
# =============================================================================

class TestMSTEdgeCases:
    """Test edge cases for MST metrics."""

    def test_identical_msts(self, uniform_weights):
        """Comparing identical MSTs should give similar scores."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, uniform_weights
        )

        np.testing.assert_allclose(
            result['overall_scores']['MST1'],
            result['overall_scores']['MST2'],
            rtol=1e-10,
            err_msg="Identical MSTs should have identical scores"
        )

    def test_different_sized_msts(self):
        """Should handle MSTs of different sizes."""
        rng = np.random.RandomState(42)
        mst1 = rng.uniform(1.0, 2.0, 30)
        mst2 = rng.uniform(1.0, 2.0, 50)

        result = MSTClusteringMetrics.comprehensive_comparison(mst1, mst2)

        assert 'recommendation' in result

    def test_very_small_weights(self):
        """Should handle very small weight values."""
        rng = np.random.RandomState(42)
        weights = rng.uniform(1e-10, 1e-9, 50)

        result = MSTClusteringMetrics.gap_statistic(weights)

        assert np.isfinite(result['max_gap'])
        assert np.isfinite(result['gap_ratio'])

    def test_very_large_weights(self):
        """Should handle very large weight values."""
        rng = np.random.RandomState(42)
        weights = rng.uniform(1e8, 1e9, 50)

        result = MSTClusteringMetrics.gap_statistic(weights)

        assert np.isfinite(result['max_gap'])
        assert np.isfinite(result['gap_ratio'])

    def test_zero_weights_present(self):
        """Should handle weights containing zeros."""
        rng = np.random.RandomState(42)
        weights = rng.uniform(0.0, 1.0, 50)
        weights[:5] = 0  # Add some zeros

        result = MSTClusteringMetrics.edge_heterogeneity(weights)

        assert np.isfinite(result['gini'])
        assert np.isfinite(result['entropy'])


# =============================================================================
# Print Report Tests
# =============================================================================

class TestPrintComparisonReport:
    """Test comparison report printing."""

    def test_prints_without_error(self, uniform_weights, bimodal_weights, capsys):
        """print_comparison_report should execute without errors."""
        result = MSTClusteringMetrics.comprehensive_comparison(
            uniform_weights, bimodal_weights
        )

        # Should not raise
        print_comparison_report(result)

        captured = capsys.readouterr()
        assert "MST CLUSTERING QUALITY COMPARISON" in captured.out
        assert "RECOMMENDATION" in captured.out
