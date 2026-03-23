"""
Test weight update contracts.

Tests constraints on weight update operations:
- Updates bounded by threshold
- Momentum coefficient in [0, 1]
- Delta weights shape preserved
- Weight norms remain finite
"""

import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")


# =============================================================================
# Weight Update Bound Contracts
# =============================================================================

class TestWeightUpdateBounds:
    """Test that weight updates are properly bounded."""

    def test_clipped_updates_within_threshold(self, weights_for_updates, batch_for_weight_updates):
        """Clipped updates should be within threshold."""
        weights = cp.asarray(weights_for_updates)
        batch = cp.asarray(batch_for_weight_updates)
        threshold = 0.1

        # Simulate computing raw updates
        # Simple approximation: difference between batch mean and weights
        batch_mean = cp.mean(batch, axis=0)
        raw_updates = batch_mean - weights

        # Apply clipping
        clipped_updates = cp.clip(raw_updates, -threshold, threshold)

        # Verify bounds
        assert cp.all(clipped_updates >= -threshold - 1e-10), \
            f"Update below -threshold: min={float(cp.min(clipped_updates))}"
        assert cp.all(clipped_updates <= threshold + 1e-10), \
            f"Update above threshold: max={float(cp.max(clipped_updates))}"

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.5])
    def test_various_thresholds(self, threshold):
        """Test clipping with various threshold values."""
        rng = np.random.RandomState(42)
        updates = cp.asarray(rng.uniform(-1, 1, (64, 50)).astype(np.float32))

        clipped = cp.clip(updates, -threshold, threshold)

        assert cp.all(clipped >= -threshold), f"Below threshold {threshold}"
        assert cp.all(clipped <= threshold), f"Above threshold {threshold}"


# =============================================================================
# Momentum Update Contracts
# =============================================================================

class TestMomentumUpdateContracts:
    """Test momentum-based weight update properties."""

    def test_momentum_weighted_update(self):
        """Test momentum update formula: new = current + momentum * delta."""
        num_nodes, input_dim = 64, 50
        rng = np.random.RandomState(42)

        current_updates = cp.asarray(rng.uniform(-0.1, 0.1, (num_nodes, input_dim)).astype(np.float32))
        delta_weights = cp.asarray(rng.uniform(-0.1, 0.1, (num_nodes, input_dim)).astype(np.float32))
        momentum_coefficient = 0.9

        # Apply momentum formula
        weight_changes = current_updates + momentum_coefficient * delta_weights

        # Shape should be preserved
        assert weight_changes.shape == current_updates.shape, "Shape changed after momentum update"

        # Values should be finite
        assert not cp.any(cp.isnan(weight_changes)), "NaN in momentum update"
        assert not cp.any(cp.isinf(weight_changes)), "Inf in momentum update"

    @pytest.mark.parametrize("momentum", [0.0, 0.5, 0.9, 1.0])
    def test_momentum_coefficient_effect(self, momentum):
        """Test that momentum coefficient has expected effect."""
        num_nodes, input_dim = 64, 50
        rng = np.random.RandomState(42)

        current_updates = cp.asarray(rng.uniform(-0.1, 0.1, (num_nodes, input_dim)).astype(np.float32))
        delta_weights = cp.asarray(rng.uniform(-0.1, 0.1, (num_nodes, input_dim)).astype(np.float32))

        weight_changes = current_updates + momentum * delta_weights

        if momentum == 0.0:
            # No momentum contribution
            np.testing.assert_allclose(
                cp.asnumpy(weight_changes),
                cp.asnumpy(current_updates),
                rtol=1e-6,
                err_msg="Zero momentum should give only current updates"
            )
        elif momentum == 1.0:
            # Full momentum contribution
            expected = current_updates + delta_weights
            np.testing.assert_allclose(
                cp.asnumpy(weight_changes),
                cp.asnumpy(expected),
                rtol=1e-6,
                err_msg="Full momentum should add delta_weights"
            )


# =============================================================================
# Delta Weight Shape Contracts
# =============================================================================

class TestDeltaWeightShapeContracts:
    """Test that delta weights maintain shape through updates."""

    def test_delta_weights_shape_preserved(self, weights_for_updates):
        """Delta weights should have same shape as weights."""
        weights = cp.asarray(weights_for_updates)
        num_nodes, input_dim = weights.shape

        # Initialize delta weights
        delta_weights = cp.zeros_like(weights)

        assert delta_weights.shape == weights.shape, \
            f"Shape mismatch: delta={delta_weights.shape}, weights={weights.shape}"

    def test_updates_shape_matches_weights(self, weights_for_updates, batch_for_weight_updates):
        """Update array should match weight shape."""
        weights = cp.asarray(weights_for_updates)
        batch = cp.asarray(batch_for_weight_updates)

        # Simulate update computation
        batch_mean = cp.mean(batch, axis=0)  # (input_dim,)
        updates = batch_mean - weights  # Broadcasting: (num_nodes, input_dim)

        assert updates.shape == weights.shape, \
            f"Shape mismatch: updates={updates.shape}, weights={weights.shape}"

    def test_accumulated_updates_shape(self):
        """Accumulated updates should maintain shape through iterations."""
        num_nodes, input_dim = 64, 50
        rng = np.random.RandomState(42)

        accumulated = cp.zeros((num_nodes, input_dim), dtype=cp.float32)
        expected_shape = (num_nodes, input_dim)

        # Simulate multiple update iterations
        for _ in range(10):
            update = cp.asarray(rng.uniform(-0.1, 0.1, (num_nodes, input_dim)).astype(np.float32))
            accumulated += update

            assert accumulated.shape == expected_shape, \
                f"Shape changed during accumulation: {accumulated.shape}"


# =============================================================================
# Weight Norm Contracts
# =============================================================================

class TestWeightNormContracts:
    """Test weight normalization and norm properties."""

    def test_weights_remain_finite(self, weights_for_updates, batch_for_weight_updates):
        """Weights should remain finite after updates."""
        weights = cp.asarray(weights_for_updates)
        batch = cp.asarray(batch_for_weight_updates)
        learning_rate = 0.1

        # Simulate simple update
        batch_mean = cp.mean(batch, axis=0)
        update = learning_rate * (batch_mean - cp.mean(weights, axis=0))
        new_weights = weights + update

        assert not cp.any(cp.isnan(new_weights)), "NaN in updated weights"
        assert not cp.any(cp.isinf(new_weights)), "Inf in updated weights"

    def test_weight_norm_calculable(self, weights_for_updates):
        """Weight norms should be calculable and finite."""
        weights = cp.asarray(weights_for_updates)

        # L2 norm per node
        norms = cp.linalg.norm(weights, axis=1)

        assert not cp.any(cp.isnan(norms)), "NaN in weight norms"
        assert not cp.any(cp.isinf(norms)), "Inf in weight norms"
        assert cp.all(norms >= 0), "Negative weight norms"

    def test_total_weight_change_norm(self, weights_for_updates, batch_for_weight_updates):
        """Total weight change norm should be calculable."""
        weights = cp.asarray(weights_for_updates)
        batch = cp.asarray(batch_for_weight_updates)
        learning_rate = 0.1

        # Compute update
        batch_mean = cp.mean(batch, axis=0)
        update = learning_rate * (batch_mean - cp.mean(weights, axis=0))

        # Total change norm
        change_norm = float(cp.linalg.norm(update))

        assert np.isfinite(change_norm), "Change norm is not finite"
        assert change_norm >= 0, "Negative change norm"


# =============================================================================
# Learning Rate Application Contracts
# =============================================================================

class TestLearningRateContracts:
    """Test learning rate application properties."""

    @pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1, 0.5, 1.0])
    def test_learning_rate_scales_update(self, learning_rate):
        """Learning rate should scale updates proportionally."""
        num_nodes, input_dim = 64, 50
        rng = np.random.RandomState(42)

        raw_update = cp.asarray(rng.uniform(-1, 1, (num_nodes, input_dim)).astype(np.float32))
        scaled_update = learning_rate * raw_update

        # Scaled update should be proportional
        ratio = scaled_update / (raw_update + 1e-10)  # Avoid division by zero
        expected_ratio = learning_rate

        # Check ratio where raw_update is significant
        significant = cp.abs(raw_update) > 0.01
        if cp.any(significant):
            actual_ratios = cp.asnumpy(ratio[significant])
            np.testing.assert_allclose(
                actual_ratios,
                np.full_like(actual_ratios, expected_ratio),
                rtol=1e-5,
                err_msg=f"Learning rate {learning_rate} not applied correctly"
            )

    def test_zero_learning_rate_no_update(self):
        """Zero learning rate should result in no update."""
        num_nodes, input_dim = 64, 50
        rng = np.random.RandomState(42)

        raw_update = cp.asarray(rng.uniform(-1, 1, (num_nodes, input_dim)).astype(np.float32))
        scaled_update = 0.0 * raw_update

        np.testing.assert_allclose(
            cp.asnumpy(scaled_update),
            np.zeros((num_nodes, input_dim)),
            atol=1e-10,
            err_msg="Zero learning rate should give zero updates"
        )


# =============================================================================
# Neighborhood Weighting Contracts
# =============================================================================

class TestNeighborhoodWeightingContracts:
    """Test neighborhood function weighting properties."""

    def test_gaussian_neighborhood_non_negative(self):
        """Gaussian neighborhood weights should be non-negative."""
        num_nodes = 64
        rng = np.random.RandomState(42)

        # Random distances
        distances = cp.asarray(rng.uniform(0, 10, num_nodes).astype(np.float32))
        sigma = 2.0

        # Gaussian neighborhood
        weights = cp.exp(-distances**2 / (2 * sigma**2))

        assert cp.all(weights >= 0), "Negative Gaussian weights"

    def test_gaussian_neighborhood_bounded(self):
        """Gaussian neighborhood weights should be in [0, 1]."""
        num_nodes = 64
        rng = np.random.RandomState(42)

        distances = cp.asarray(rng.uniform(0, 10, num_nodes).astype(np.float32))
        sigma = 2.0

        weights = cp.exp(-distances**2 / (2 * sigma**2))

        assert cp.all(weights >= 0), "Weights below 0"
        assert cp.all(weights <= 1.0 + 1e-10), "Weights above 1"

    def test_gaussian_max_at_zero_distance(self):
        """Gaussian weight should be 1 at distance 0."""
        sigma = 2.0
        weight_at_zero = float(cp.exp(-0.0 / (2 * sigma**2)))

        np.testing.assert_allclose(
            weight_at_zero, 1.0, rtol=1e-10,
            err_msg="Gaussian weight at distance 0 should be 1"
        )

    def test_gaussian_decays_with_distance(self):
        """Gaussian weight should decrease with distance."""
        sigma = 2.0
        distances = cp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        weights = cp.exp(-distances**2 / (2 * sigma**2))

        weights_np = cp.asnumpy(weights)

        for i in range(1, len(weights_np)):
            assert weights_np[i] < weights_np[i-1], \
                f"Weight not decreasing: w[{i-1}]={weights_np[i-1]}, w[{i}]={weights_np[i]}"


# =============================================================================
# Batch Processing Contracts
# =============================================================================

class TestBatchProcessingContracts:
    """Test batch processing properties."""

    def test_batch_mean_shape(self, batch_for_weight_updates):
        """Batch mean should reduce to (input_dim,)."""
        batch = cp.asarray(batch_for_weight_updates)
        batch_mean = cp.mean(batch, axis=0)

        assert batch_mean.ndim == 1, f"Expected 1D, got {batch_mean.ndim}D"
        assert batch_mean.shape[0] == batch.shape[1], \
            f"Expected {batch.shape[1]}, got {batch_mean.shape[0]}"

    def test_batch_processing_order_independent(self, batch_for_weight_updates):
        """Mean-based updates should be order-independent."""
        batch = cp.asarray(batch_for_weight_updates)

        # Original order mean
        mean1 = cp.mean(batch, axis=0)

        # Shuffled order mean
        rng = np.random.RandomState(42)
        shuffled_indices = rng.permutation(len(batch))
        mean2 = cp.mean(batch[shuffled_indices], axis=0)

        np.testing.assert_allclose(
            cp.asnumpy(mean1),
            cp.asnumpy(mean2),
            rtol=1e-5,
            err_msg="Batch mean not order-independent"
        )
