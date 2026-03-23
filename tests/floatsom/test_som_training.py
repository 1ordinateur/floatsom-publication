"""
Integration tests for full SOM training
Tests complete training workflow and convergence
"""

import numpy as np
import cupy as cp
import pytest

from floatsom.base.floatsom import FloatSOM
from floatsom.sampling.full_selector import FullSelector
from floatsom.sampling.random_selector import RandomSelector
from floatsom.processing.batch_processor import BatchProcessor
from floatsom.processing.processing_params import BatchConfig
from floatsom.topology.grid_topology import GridTopology
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig


@pytest.fixture
def training_data():
    """Create synthetic training data"""
    np.random.seed(42)
    cp.random.seed(42)
    # Create clustered data
    n_samples = 1000
    n_features = 10

    # Generate 4 clusters
    cluster_size = n_samples // 4
    data = []

    centers = [
        np.array([1.0] * n_features),
        np.array([-1.0] * n_features),
        np.array([0.5, -0.5] * (n_features // 2)),
        np.array([-0.5, 0.5] * (n_features // 2))
    ]

    for center in centers:
        cluster = np.random.randn(cluster_size, n_features) * 0.3 + center
        data.append(cluster)

    data = np.vstack(data).astype(np.float32)
    return cp.asarray(data)


@pytest.fixture
def basic_som_params():
    """Create basic SOM parameters"""
    params = FloatSOMParams(
        input_dim=10,
        total_iterations=50,
        initial_learning_rate=0.5,
        initial_radius=5.0,
        use_gpu=True,
        verbose=False,
        seed=42,
        topology_config=TopologyConfig(grid_size=10),
        processing_config=ProcessingConfig(chunk_size=100, method="batch")
    )
    return params


class TestSOMTraining:
    """Integration tests for SOM training"""

    def test_som_trains_without_error(self, training_data, basic_som_params):
        """Training should complete without errors"""
        # Create components
        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Create SOM
        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=basic_som_params
        )

        # Train should complete without raising exceptions
        stats = som.train(training_data)

        assert stats is not None
        assert stats['iterations_completed'] > 0

    def test_som_weights_shape(self, training_data, basic_som_params):
        """Weights shape should be (num_nodes, input_dim)"""
        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=basic_som_params
        )

        som.train(training_data)

        weights = som.get_weights()
        expected_nodes = 10 * 10  # grid_size * grid_size
        expected_features = 10

        assert weights.shape == (expected_nodes, expected_features)

    def test_som_assignments_range(self, training_data, basic_som_params):
        """Assignments should be in valid range [0, num_nodes-1]"""
        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=basic_som_params
        )

        som.train(training_data)

        # Get assignments
        assignments = som.predict(training_data)

        # All assignments should be valid node indices
        num_nodes = 10 * 10
        assert cp.all(assignments >= 0)
        assert cp.all(assignments < num_nodes)

    def test_som_quantization_error_decreases(self, training_data, basic_som_params):
        """Quantization error should decrease over training"""
        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Increase iterations for better convergence
        params = basic_som_params
        params.total_iterations = 100
        params.store_history = True

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=params
        )

        # Calculate initial quantization error
        initial_weights = topology.initialize_weights(training_data)
        bmus, distances = self._find_bmus_for_data(som, training_data, initial_weights)
        initial_qe = float(cp.mean(distances))

        # Train
        som.train(training_data)

        # Calculate final quantization error
        final_bmus, final_distances = som.map_vectors(training_data)
        final_qe = float(cp.mean(final_distances))

        # Final QE should be lower than initial
        assert final_qe < initial_qe

    def _find_bmus_for_data(self, som, data, weights):
        """Helper to find BMUs and distances"""
        data_expanded = data[:, None, :]
        weights_expanded = weights[None, :, :]
        distances = cp.sum((data_expanded - weights_expanded) ** 2, axis=2)
        bmus = cp.argmin(distances, axis=1)
        bmu_distances = cp.sqrt(distances[cp.arange(len(data)), bmus])
        return bmus, bmu_distances

    def test_som_convergence(self, training_data, basic_som_params):
        """Weights should stabilize during training"""
        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        params = basic_som_params
        params.total_iterations = 100
        params.store_history = True

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=params
        )

        som.train(training_data)

        # Check training history for decreasing weight changes
        if som.training_history and len(som.training_history) > 10:
            early_changes = [h['weight_change'] for h in som.training_history[:10]]
            late_changes = [h['weight_change'] for h in som.training_history[-10:]]

            avg_early = np.mean(early_changes)
            avg_late = np.mean(late_changes)

            # Weight changes should decrease over time
            assert avg_late < avg_early

    def test_som_reproducibility_with_seed(self, training_data):
        """Same seed should produce same results"""
        seed = 42

        def train_som_with_seed(seed_value):
            params = FloatSOMParams(
                input_dim=10,
                total_iterations=50,
                initial_learning_rate=0.5,
                initial_radius=5.0,
                use_gpu=True,
                verbose=False,
                seed=seed_value,
                topology_config=TopologyConfig(grid_size=10),
                processing_config=ProcessingConfig(chunk_size=100, method="batch")
            )

            selector = FullSelector()
            batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
            processor = BatchProcessor(batch_config)
            topology = GridTopology(
                grid_size=10,
                input_dim=10,
                initialization_method="random",
                seed=seed_value,
                verbose=False
            )

            som = FloatSOM(
                selector=selector,
                processor=processor,
                topology=topology,
                params=params
            )

            som.train(training_data)
            return som.get_weights()

        # Train two SOMs with same seed
        weights1 = train_som_with_seed(seed)
        weights2 = train_som_with_seed(seed)

        # Results should be identical
        assert np.allclose(weights1, weights2, rtol=1e-5, atol=1e-6)

    def test_som_with_random_sampling(self, training_data, basic_som_params):
        """SOM should train successfully with random sampling"""
        sampling_config = SamplingConfig(
            method="random",
            target_proportion=0.5,
            samples_per_epoch=None,
            random_seed=42
        )

        selector = RandomSelector(sampling_config)
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=10,
            input_dim=10,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=basic_som_params
        )

        # Training should complete
        stats = som.train(training_data)

        assert stats is not None
        assert stats['iterations_completed'] > 0

        # Should produce valid weights
        weights = som.get_weights()
        assert weights.shape == (100, 10)
