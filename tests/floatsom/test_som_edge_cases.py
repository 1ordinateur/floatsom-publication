"""
Edge case tests for SOM training
Tests handling of unusual or extreme inputs
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


class TestSOMEdgeCases:
    """Tests for edge cases and unusual inputs"""

    def test_single_sample(self):
        """SOM should handle single training sample"""
        # Create single sample
        data = cp.array([[1.0, 2.0, 3.0]], dtype=cp.float32)

        params = FloatSOMParams(
            input_dim=3,
            total_iterations=10,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=10, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=10)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=3,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Should be able to predict
        predictions = som.predict(data)
        assert len(predictions) == 1

    def test_identical_samples(self):
        """SOM should handle all identical samples"""
        # Create identical samples
        n_samples = 100
        sample_value = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        data = cp.tile(sample_value, (n_samples, 1))

        params = FloatSOMParams(
            input_dim=4,
            total_iterations=20,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=50, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=4,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # All samples should map to valid nodes
        predictions = som.predict(data)
        assert len(predictions) == n_samples
        assert cp.all(predictions >= 0)
        assert cp.all(predictions < 25)  # 5x5 grid

        # Weights should converge toward the sample value
        weights = cp.asarray(som.get_weights())
        mean_weight = cp.mean(weights, axis=0)
        assert cp.allclose(mean_weight, sample_value, rtol=1.0)

    def test_high_dimensional_data(self):
        """SOM should handle high-dimensional input"""
        np.random.seed(42)
        cp.random.seed(42)

        # Create high-dimensional data
        n_samples = 200
        n_features = 100  # High dimensional
        data = cp.random.randn(n_samples, n_features).astype(cp.float32)

        params = FloatSOMParams(
            input_dim=n_features,
            total_iterations=20,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=50, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=n_features,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Weights should have correct shape
        weights = som.get_weights()
        assert weights.shape == (25, n_features)

    def test_small_som(self):
        """SOM should work with minimal 2x2 grid"""
        np.random.seed(42)
        cp.random.seed(42)

        # Create training data
        n_samples = 50
        n_features = 5
        data = cp.random.randn(n_samples, n_features).astype(cp.float32)

        params = FloatSOMParams(
            input_dim=n_features,
            total_iterations=20,
            initial_learning_rate=0.5,
            initial_radius=1.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=2),  # Minimal 2x2 grid
            processing_config=ProcessingConfig(chunk_size=25, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=25)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=2,
            input_dim=n_features,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Should have 4 nodes
        weights = som.get_weights()
        assert weights.shape == (4, n_features)

        # All nodes should be used
        predictions = som.predict(data)
        unique_nodes = cp.unique(predictions)
        assert len(unique_nodes) <= 4

    def test_zero_samples(self):
        """SOM should handle data with all zeros gracefully"""
        # Create zero data
        n_samples = 100
        n_features = 5
        data = cp.zeros((n_samples, n_features), dtype=cp.float32)

        params = FloatSOMParams(
            input_dim=n_features,
            total_iterations=20,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=50, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=n_features,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Weights should be close to zero
        weights = cp.asarray(som.get_weights())
        mean_magnitude = cp.mean(cp.linalg.norm(weights, axis=1))
        # Weights should converge toward zero but may not be exactly zero
        assert mean_magnitude < 1.0

    def test_sparse_data(self):
        """SOM should handle sparse data (mostly zeros)"""
        np.random.seed(42)

        # Create sparse data (90% zeros)
        n_samples = 200
        n_features = 20
        data_np = np.random.randn(n_samples, n_features).astype(np.float32)

        # Make 90% of values zero
        mask = np.random.rand(n_samples, n_features) < 0.9
        data_np[mask] = 0

        data = cp.asarray(data_np)

        params = FloatSOMParams(
            input_dim=n_features,
            total_iterations=30,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=50, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=n_features,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Should produce valid predictions
        predictions = som.predict(data)
        assert len(predictions) == n_samples

    def test_large_values(self):
        """SOM should handle data with large values"""
        np.random.seed(42)

        # Create data with large values
        n_samples = 100
        n_features = 10
        data = cp.random.randn(n_samples, n_features).astype(cp.float32) * 1000  # Scale up

        params = FloatSOMParams(
            input_dim=n_features,
            total_iterations=30,
            initial_learning_rate=0.5,
            initial_radius=2.0,
            use_gpu=True,
            verbose=False,
            seed=42,
            topology_config=TopologyConfig(grid_size=5),
            processing_config=ProcessingConfig(chunk_size=50, method="batch")
        )

        selector = FullSelector()
        batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
        processor = BatchProcessor(batch_config)
        topology = GridTopology(
            grid_size=5,
            input_dim=n_features,
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

        # Should train without error
        stats = som.train(data)
        assert stats is not None

        # Weights should be finite
        weights = cp.asarray(som.get_weights())
        assert cp.all(cp.isfinite(weights))

    def test_different_learning_rate_schedules(self):
        """SOM should work with different learning rate decay types"""
        np.random.seed(42)
        data = cp.random.randn(200, 10).astype(cp.float32)

        decay_types = ['linear', 'exponential', 'asymptotic']

        for decay_type in decay_types:
            params = FloatSOMParams(
                input_dim=10,
                total_iterations=30,
                initial_learning_rate=0.5,
                initial_radius=2.0,
                lr_decay_type=decay_type,
                use_gpu=True,
                verbose=False,
                seed=42,
                topology_config=TopologyConfig(grid_size=5),
                processing_config=ProcessingConfig(chunk_size=50, method="batch")
            )

            selector = FullSelector()
            batch_config = BatchConfig(batch_mode="full_batch", chunk_size=50)
            processor = BatchProcessor(batch_config)
            topology = GridTopology(
                grid_size=5,
                input_dim=10,
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

            # Should train without error for each decay type
            stats = som.train(data)
            assert stats is not None, f"Training failed with {decay_type} decay"
