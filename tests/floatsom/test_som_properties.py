"""
Property-based tests for SOM behavior
Tests invariants and expected properties of trained SOMs
"""

import numpy as np
import cupy as cp
import pytest

from floatsom.base.floatsom import FloatSOM
from floatsom.sampling.full_selector import FullSelector
from floatsom.processing.batch_processor import BatchProcessor
from floatsom.processing.processing_params import BatchConfig
from floatsom.topology.grid_topology import GridTopology
from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig, TopologyConfig


@pytest.fixture
def training_data():
    """Create synthetic training data"""
    np.random.seed(42)
    cp.random.seed(42)
    n_samples = 500
    n_features = 8
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    return cp.asarray(data)


@pytest.fixture
def trained_som(training_data):
    """Create a trained SOM for property testing"""
    params = FloatSOMParams(
        input_dim=8,
        total_iterations=50,
        initial_learning_rate=0.5,
        initial_radius=4.0,
        use_gpu=True,
        verbose=False,
        seed=42,
        topology_config=TopologyConfig(grid_size=8),
        processing_config=ProcessingConfig(chunk_size=100, method="batch")
    )

    selector = FullSelector()
    batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
    processor = BatchProcessor(batch_config)
    topology = GridTopology(
        grid_size=8,
        input_dim=8,
        initialization_method="random",
        seed=42,
        verbose=False
    )

    som = FloatSOM(
        selector=selector,
        processor=processor,
        topology=topology,
        params=params
    )

    som.train(training_data)
    return som


class TestSOMProperties:
    """Property-based tests for SOM behavior"""

    def test_bmu_assignment_deterministic(self, trained_som, training_data):
        """Same input should always return same BMU"""
        # Select a single sample
        sample = training_data[0:1]

        # Get BMU multiple times
        bmu1 = trained_som.predict(sample)
        bmu2 = trained_som.predict(sample)
        bmu3 = trained_som.predict(sample)

        # All should be identical
        assert cp.array_equal(bmu1, bmu2)
        assert cp.array_equal(bmu2, bmu3)

    def test_weights_bounded_by_data(self, trained_som, training_data):
        """Weights should be within reasonable range of data"""
        weights = trained_som.get_weights()
        weights_cp = cp.asarray(weights)

        # Get data bounds
        data_min = cp.min(training_data, axis=0)
        data_max = cp.max(training_data, axis=0)
        data_range = data_max - data_min

        # Weights should be within data range (with some margin)
        margin = 2.0  # Allow weights to extend beyond data by 2x range
        weight_min = cp.min(weights_cp, axis=0)
        weight_max = cp.max(weights_cp, axis=0)

        # Check that weights are not wildly outside data range
        assert cp.all(weight_min >= data_min - margin * data_range)
        assert cp.all(weight_max <= data_max + margin * data_range)

    def test_all_nodes_used(self, trained_som, training_data):
        """All nodes should be used for sufficient data (no dead nodes)"""
        # With 500 samples and 64 nodes, all should be used
        assignments = trained_som.predict(training_data)

        # Count unique assignments
        unique_nodes = cp.unique(assignments)

        # Most nodes should be used (allow some dead nodes for small grids)
        num_nodes = trained_som.topology.total_nodes
        usage_ratio = len(unique_nodes) / num_nodes

        # At least 70% of nodes should be used
        assert usage_ratio >= 0.7

    def test_learning_rate_positive(self, training_data):
        """Learning rate should always be positive during training"""
        params = FloatSOMParams(
            input_dim=8,
            total_iterations=50,
            initial_learning_rate=0.5,
            initial_radius=4.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            store_history=True,
            topology_config=TopologyConfig(grid_size=8),
            processing_config=ProcessingConfig(chunk_size=100, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=8,
            input_dim=8,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=params
        )

        som.train(training_data)

        # Check all learning rates in history
        if som.training_history:
            for entry in som.training_history:
                assert entry['learning_rate'] > 0

    def test_neighborhood_radius_positive(self, training_data):
        """Neighborhood radius should always be positive"""
        params = FloatSOMParams(
            input_dim=8,
            total_iterations=50,
            initial_learning_rate=0.5,
            initial_radius=4.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            store_history=True,
            topology_config=TopologyConfig(grid_size=8),
            processing_config=ProcessingConfig(chunk_size=100, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=100)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=8,
            input_dim=8,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        som = FloatSOM(
            selector=selector,
            processor=processor,
            topology=topology,
            params=params
        )

        som.train(training_data)

        # Check all radii in history
        if som.training_history:
            for entry in som.training_history:
                assert entry['radius'] > 0

    def test_bmu_closest_to_input(self, trained_som, training_data):
        """BMU should be the closest node to input"""
        # Test on a few samples
        test_samples = training_data[:10]

        for i in range(len(test_samples)):
            sample = test_samples[i:i+1]
            bmu = trained_som.predict(sample)
            bmu_idx = int(bmu[0])

            weights = cp.asarray(trained_som.get_weights())

            # Calculate distances to all nodes
            sample_expanded = sample[0]
            distances = cp.sqrt(cp.sum((weights - sample_expanded) ** 2, axis=1))

            # BMU should have minimum distance
            min_distance_idx = int(cp.argmin(distances))
            assert bmu_idx == min_distance_idx

    def test_weight_vectors_normalized(self, trained_som):
        """Weight vectors should have reasonable magnitude"""
        weights = cp.asarray(trained_som.get_weights())

        # Calculate norms of weight vectors
        norms = cp.linalg.norm(weights, axis=1)

        # All norms should be finite and positive
        assert cp.all(cp.isfinite(norms))
        assert cp.all(norms > 0)

        # Norms should be in reasonable range (not too large or small)
        assert cp.all(norms < 100)  # Not exploding
        assert cp.all(norms > 0.001)  # Not vanishing

    def test_adjacent_nodes_similar(self, trained_som):
        """Adjacent nodes in grid should have similar weight vectors"""
        weights = cp.asarray(trained_som.get_weights())
        grid_size = trained_som.topology.grid_size

        # Calculate average distance between adjacent nodes
        total_distance = 0
        count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                node_idx = i * grid_size + j

                # Check right neighbor
                if j < grid_size - 1:
                    neighbor_idx = i * grid_size + (j + 1)
                    distance = cp.linalg.norm(weights[node_idx] - weights[neighbor_idx])
                    total_distance += float(distance)
                    count += 1

                # Check bottom neighbor
                if i < grid_size - 1:
                    neighbor_idx = (i + 1) * grid_size + j
                    distance = cp.linalg.norm(weights[node_idx] - weights[neighbor_idx])
                    total_distance += float(distance)
                    count += 1

        avg_neighbor_distance = total_distance / count

        # Calculate average distance between random nodes
        n_random_pairs = 100
        random_indices_1 = cp.random.randint(0, len(weights), size=n_random_pairs)
        random_indices_2 = cp.random.randint(0, len(weights), size=n_random_pairs)
        random_distances = cp.linalg.norm(
            weights[random_indices_1] - weights[random_indices_2],
            axis=1
        )
        avg_random_distance = float(cp.mean(random_distances))

        # Adjacent nodes should be more similar than random nodes
        assert avg_neighbor_distance < avg_random_distance
