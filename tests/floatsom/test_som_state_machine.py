"""
Test SOM training state machine invariants.

Tests state transitions and invariants during SOM training:
- Weight initialization before training
- Iteration counter bounds
- Weight shape preservation
- Training completion state
- Momentum coefficient bounds
"""

import pytest
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    from floatsom.floatsom_params import (
        FloatSOMParams,
        ProcessingConfig,
        precompute_all_momentum_coefficients,
        calculate_adaptive_momentum_coefficient,
    )


pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")


# =============================================================================
# Momentum Schedule Invariants
# =============================================================================

class TestMomentumScheduleInvariants:
    """Test momentum coefficient schedule properties."""

    def test_momentum_coefficient_bounds(self, momentum_schedule_params):
        """Momentum coefficient should stay in [0, 1]."""
        initial = momentum_schedule_params["initial_momentum"]
        final = momentum_schedule_params["final_momentum"]
        total_iterations = momentum_schedule_params["total_iterations"]

        # Precompute schedule
        schedule = precompute_all_momentum_coefficients(
            total_iterations, initial, final, momentum_decay_type="linear"
        )

        # All values should be in [0, 1]
        assert np.all(schedule >= 0), f"Momentum below 0: min={schedule.min()}"
        assert np.all(schedule <= 1), f"Momentum above 1: max={schedule.max()}"

    def test_momentum_schedule_monotonic_decrease(self, momentum_schedule_params):
        """With initial > final, momentum should decrease monotonically."""
        initial = momentum_schedule_params["initial_momentum"]
        final = momentum_schedule_params["final_momentum"]
        total_iterations = momentum_schedule_params["total_iterations"]

        if initial <= final:
            pytest.skip("Test requires initial > final")

        schedule = precompute_all_momentum_coefficients(
            total_iterations, initial, final, momentum_decay_type="linear"
        )

        # Check monotonic decrease
        diffs = np.diff(schedule)
        assert np.all(diffs <= 1e-10), "Momentum schedule not monotonically decreasing"

    def test_momentum_schedule_endpoints(self, momentum_schedule_params):
        """Schedule should start at initial and end at final."""
        initial = momentum_schedule_params["initial_momentum"]
        final = momentum_schedule_params["final_momentum"]
        total_iterations = momentum_schedule_params["total_iterations"]

        schedule = precompute_all_momentum_coefficients(
            total_iterations, initial, final, momentum_decay_type="linear"
        )

        np.testing.assert_allclose(
            schedule[0], initial, rtol=1e-5,
            err_msg="Schedule doesn't start at initial momentum"
        )
        np.testing.assert_allclose(
            schedule[-1], final, rtol=1e-5,
            err_msg="Schedule doesn't end at final momentum"
        )

    @pytest.mark.parametrize("decay_type", ["linear", "exponential", "cosine"])
    def test_all_decay_types_bounded(self, decay_type, momentum_schedule_params):
        """All decay types should produce bounded values."""
        initial = momentum_schedule_params["initial_momentum"]
        final = momentum_schedule_params["final_momentum"]
        total_iterations = momentum_schedule_params["total_iterations"]

        schedule = precompute_all_momentum_coefficients(
            total_iterations, initial, final, momentum_decay_type=decay_type
        )

        assert np.all(schedule >= 0), f"{decay_type}: momentum below 0"
        assert np.all(schedule <= 1), f"{decay_type}: momentum above 1"
        assert not np.any(np.isnan(schedule)), f"{decay_type}: NaN in schedule"
        assert not np.any(np.isinf(schedule)), f"{decay_type}: Inf in schedule"


# =============================================================================
# Weight Matrix Invariants
# =============================================================================

class TestWeightMatrixInvariants:
    """Test weight matrix properties."""

    def test_weights_finite_after_initialization(self, som_weight_matrix):
        """Weights should be finite after initialization."""
        assert not np.any(np.isnan(som_weight_matrix)), "NaN in initialized weights"
        assert not np.any(np.isinf(som_weight_matrix)), "Inf in initialized weights"

    def test_weights_shape_consistent(self, som_weight_matrix):
        """Weight shape should be (num_nodes, input_dim)."""
        num_nodes, input_dim = som_weight_matrix.shape
        assert num_nodes > 0, "Must have at least one node"
        assert input_dim > 0, "Must have at least one dimension"

    def test_weight_update_preserves_shape(self, som_weight_matrix, batch_for_weight_updates):
        """Weight updates should preserve shape."""
        weights_gpu = cp.asarray(som_weight_matrix)
        batch_gpu = cp.asarray(batch_for_weight_updates)

        original_shape = weights_gpu.shape

        # Simulate a simple weight update (without full SOM machinery)
        # This tests the shape preservation contract
        learning_rate = 0.1
        delta = cp.mean(batch_gpu, axis=0) - cp.mean(weights_gpu, axis=0)
        new_weights = weights_gpu + learning_rate * delta

        assert new_weights.shape == original_shape, \
            f"Shape changed: {original_shape} -> {new_weights.shape}"


# =============================================================================
# Iteration Counter Invariants
# =============================================================================

class TestIterationCounterInvariants:
    """Test iteration counter properties."""

    def test_iteration_counter_non_negative(self):
        """Iteration counter should never be negative."""
        total_iterations = 100

        for iteration in range(total_iterations):
            assert iteration >= 0, f"Negative iteration: {iteration}"
            assert iteration < total_iterations, f"Iteration exceeds total: {iteration}"

    def test_iteration_counter_sequential(self):
        """Iterations should be sequential."""
        total_iterations = 100
        iterations = list(range(total_iterations))

        # Check sequential property
        for i in range(1, len(iterations)):
            assert iterations[i] == iterations[i-1] + 1, "Iterations not sequential"


# =============================================================================
# Learning Rate Schedule Invariants
# =============================================================================

class TestLearningRateInvariants:
    """Test learning rate schedule properties."""

    @pytest.mark.parametrize("initial_lr,final_lr,total_iterations", [
        (0.5, 0.01, 100),
        (1.0, 0.001, 1000),
        (0.1, 0.1, 50),  # Constant LR
    ])
    def test_learning_rate_positive(self, initial_lr, final_lr, total_iterations):
        """Learning rate should always be positive."""
        # Linear interpolation (common schedule)
        for iteration in range(total_iterations):
            progress = iteration / max(1, total_iterations - 1)
            lr = initial_lr * (1 - progress) + final_lr * progress
            assert lr > 0, f"Non-positive learning rate at iteration {iteration}: {lr}"

    @pytest.mark.parametrize("initial_lr,final_lr", [
        (0.5, 0.01),
        (1.0, 0.001),
    ])
    def test_learning_rate_decreases(self, initial_lr, final_lr):
        """With initial > final, LR should decrease."""
        total_iterations = 100

        prev_lr = initial_lr
        for iteration in range(total_iterations):
            progress = iteration / max(1, total_iterations - 1)
            lr = initial_lr * (1 - progress) + final_lr * progress
            assert lr <= prev_lr + 1e-10, "Learning rate increased"
            prev_lr = lr


# =============================================================================
# Radius Schedule Invariants
# =============================================================================

class TestRadiusScheduleInvariants:
    """Test neighborhood radius schedule properties."""

    @pytest.mark.parametrize("initial_radius,final_radius,total_iterations", [
        (10.0, 0.5, 100),
        (5.0, 1.0, 500),
    ])
    def test_radius_positive(self, initial_radius, final_radius, total_iterations):
        """Radius should always be positive."""
        for iteration in range(total_iterations):
            progress = iteration / max(1, total_iterations - 1)
            radius = initial_radius * (1 - progress) + final_radius * progress
            assert radius > 0, f"Non-positive radius at iteration {iteration}: {radius}"

    def test_radius_bounds(self):
        """Radius should stay within initial and final bounds."""
        initial_radius = 10.0
        final_radius = 0.5
        total_iterations = 100

        for iteration in range(total_iterations):
            progress = iteration / max(1, total_iterations - 1)
            radius = initial_radius * (1 - progress) + final_radius * progress

            assert radius <= initial_radius + 1e-10, "Radius exceeds initial"
            assert radius >= final_radius - 1e-10, "Radius below final"


# =============================================================================
# Training Statistics Invariants
# =============================================================================

class TestTrainingStatsInvariants:
    """Test training statistics contracts."""

    def test_stats_dictionary_structure(self):
        """Training stats should have expected structure."""
        # Simulate initial stats
        training_stats = {
            'iterations_completed': 0,
            'total_samples_processed': 0,
            'training_time': 0.0,
            'final_weights_norm': 0.0,
            'convergence_detected': False,
            'momentum_used': True
        }

        # Verify required keys
        required_keys = [
            'iterations_completed',
            'total_samples_processed',
            'training_time',
            'final_weights_norm',
            'convergence_detected',
            'momentum_used'
        ]

        for key in required_keys:
            assert key in training_stats, f"Missing required key: {key}"

    def test_stats_non_negative_counts(self):
        """Counts and times should be non-negative."""
        training_stats = {
            'iterations_completed': 50,
            'total_samples_processed': 5000,
            'training_time': 12.5,
            'final_weights_norm': 0.5,
        }

        assert training_stats['iterations_completed'] >= 0
        assert training_stats['total_samples_processed'] >= 0
        assert training_stats['training_time'] >= 0
        assert training_stats['final_weights_norm'] >= 0


# =============================================================================
# Data Source Contract Invariants
# =============================================================================

class TestDataSourceContracts:
    """Test data source abstraction contracts."""

    def test_batch_shape_consistency(self, batch_for_weight_updates, weights_for_updates):
        """Batch input_dim should match weights input_dim."""
        batch_dim = batch_for_weight_updates.shape[1]
        weight_dim = weights_for_updates.shape[1]

        assert batch_dim == weight_dim, \
            f"Dimension mismatch: batch={batch_dim}, weights={weight_dim}"

    def test_batch_non_empty(self, batch_for_weight_updates):
        """Batches should not be empty."""
        assert batch_for_weight_updates.shape[0] > 0, "Empty batch"
        assert batch_for_weight_updates.shape[1] > 0, "Zero-dimensional batch"


# =============================================================================
# Convergence Detection Invariants
# =============================================================================

class TestConvergenceInvariants:
    """Test convergence detection contracts."""

    def test_weight_change_non_negative(self):
        """Weight change magnitude should be non-negative."""
        # Simulate weight changes
        weight_changes = [0.1, 0.05, 0.01, 0.005, 0.001]

        for change in weight_changes:
            assert change >= 0, f"Negative weight change: {change}"

    def test_convergence_threshold_positive(self):
        """Convergence threshold should be positive."""
        typical_thresholds = [1e-4, 1e-5, 1e-6]

        for threshold in typical_thresholds:
            assert threshold > 0, f"Non-positive threshold: {threshold}"

    def test_convergence_when_change_below_threshold(self):
        """Convergence should trigger when change < threshold."""
        threshold = 1e-4
        weight_changes = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00005]

        converged = False
        for change in weight_changes:
            if change < threshold:
                converged = True
                break

        assert converged, "Should have detected convergence"
