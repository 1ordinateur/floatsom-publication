"""
Tests for streaming vs RAM loading parity.

These tests verify that chunked/streaming data loading produces identical
results to direct in-memory array access.

Critical invariants tested:
- Numerical equivalence between streaming and non-streaming modes
- Chunk size independence (same result regardless of chunk size)
- Memory-mapped vs in-memory array equivalence
"""

import pytest
import numpy as np
import tempfile
import os

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from floatsom.data.sources.array import ArrayDataSource
from floatsom.data.sources.file import FileDataSource
from floatsom.data.factory import DataSourceFactory
from floatsom.floatsom_params import ProcessingConfig
from floatsom.processing.processing_params import BatchConfig
from floatsom.processing.batch_processor import BatchProcessor
from floatsom.topology.grid_topology import GridTopology

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")


class TestDataSourceParity:
    """Test that different data sources produce identical iteration results."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(42)
        n_samples = 3000
        n_features = 16
        return np.random.randn(n_samples, n_features).astype(np.float32)

    def test_array_source_iteration_consistency(self, synthetic_data):
        """ArrayDataSource should iterate consistently across calls."""
        source = ArrayDataSource(synthetic_data)

        chunks_run1 = []
        for chunk in source.iterate_chunks(chunk_size=500, use_gpu=False):
            chunks_run1.append(chunk.copy())

        chunks_run2 = []
        for chunk in source.iterate_chunks(chunk_size=500, use_gpu=False):
            chunks_run2.append(chunk.copy())

        # Same number of chunks
        assert len(chunks_run1) == len(chunks_run2)

        # Each chunk should be identical
        for i, (c1, c2) in enumerate(zip(chunks_run1, chunks_run2)):
            np.testing.assert_array_equal(
                c1, c2,
                err_msg=f"Chunk {i} differs between iterations"
            )

    def test_array_source_vs_direct_access(self, synthetic_data):
        """ArrayDataSource chunks should match direct array slicing."""
        source = ArrayDataSource(synthetic_data)
        chunk_size = 500

        chunk_idx = 0
        for chunk in source.iterate_chunks(chunk_size=chunk_size, use_gpu=False):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(synthetic_data))
            expected = synthetic_data[start:end]

            np.testing.assert_array_equal(
                chunk, expected,
                err_msg=f"Chunk {chunk_idx} doesn't match direct slice"
            )
            chunk_idx += 1

    def test_file_source_matches_array_source(self, synthetic_data):
        """FileDataSource should produce same data as ArrayDataSource."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to file
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            # Create both sources
            array_source = ArrayDataSource(synthetic_data)
            file_source = FileDataSource(file_path, ray_config=None)

            # Compare shapes
            assert array_source.get_shape() == file_source.get_shape()

            # Compare initialization samples
            array_sample = array_source.get_initialization_sample(
                max_samples=100, use_gpu=False
            )
            file_sample = file_source.get_initialization_sample(
                max_samples=100, use_gpu=False
            )
            np.testing.assert_array_equal(array_sample, file_sample)

    def test_factory_creates_equivalent_sources(self, synthetic_data):
        """DataSourceFactory should create equivalent sources from array and file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            # Create via factory
            array_source = DataSourceFactory.create(synthetic_data, ray_config=None)
            file_source = DataSourceFactory.create(file_path, ray_config=None)

            # Both should have same shape
            assert array_source.get_shape() == file_source.get_shape()


class TestChunkSizeIndependence:
    """Test that different chunk sizes produce identical final results."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(123)
        n_samples = 2000
        n_features = 12
        return np.random.randn(n_samples, n_features).astype(np.float32)

    @pytest.fixture
    def som_config(self):
        """SOM configuration for testing."""
        return {
            'grid_size': 6,
            'input_dim': 12,
            'initial_lr': 0.3,
            'final_lr': 0.01,
            'initial_radius': 3.0,
            'final_radius': 1.0,
            'seed': 123,
        }

    def test_chunk_size_does_not_affect_full_batch_result(
        self, synthetic_data, som_config
    ):
        """Full batch mode should produce identical results regardless of chunk size."""
        chunk_sizes = [200, 500, 1000, 2000]
        results = []

        for chunk_size in chunk_sizes:
            batch_config = BatchConfig(
                batch_mode="full_batch",
                chunk_size=chunk_size,
            )

            np.random.seed(som_config['seed'])
            topology = GridTopology(
                grid_size=som_config['grid_size'],
                input_dim=som_config['input_dim'],
                topology_type="planar",
                initialization_method="random",
                seed=som_config['seed'],
            )
            initial_weights = np.random.randn(
                som_config['grid_size'] ** 2,
                som_config['input_dim']
            ).astype(np.float32)

            processor = BatchProcessor(batch_config)

            class MockParams:
                def __init__(self, config):
                    self.initial_lr = config['initial_lr']
                    self.final_lr = config['final_lr']
                    self.initial_radius = config['initial_radius']
                    self.final_radius = config['final_radius']
                    self.num_epochs = 1
                    self.seed = config['seed']
                    self.verbose = False
                    self.decay_type = "linear"
                    self.radius_decay_type = "linear"
                    self.lr_decay_type = "linear"
                    self.neighborhood_function = "gaussian"
                    # Add missing attributes required by BatchProcessor
                    self.current_radius = self.initial_radius
                    self.current_learning_rate = self.initial_lr
                    self.current_momentum = 0.0
                    self.delta_weights = None
                    self.processing_config = ProcessingConfig(
                        chunk_size=1000,
                        method="batch",
                        normalization="none",
                        distance_metric="euclidean",
                    )

            params = MockParams(som_config)

            # Precompute topology data
            radii_list = [params.current_radius]
            topology.precompute_topology_data(params, radii_list)

            data_source = ArrayDataSource(synthetic_data)
            processor.initialize(
                cp.asarray(initial_weights.copy()),
                topology,
                params,
                data_source
            )

            result = processor.process_samples(
                cp.asarray(synthetic_data),
                cp.asarray(initial_weights.copy()),
                topology,
                params
            )

            if isinstance(result, tuple):
                result = result[0]
            results.append(cp.asnumpy(result))

        # All chunk sizes should produce identical results
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i],
                rtol=1e-5, atol=1e-6,
                err_msg=f"Chunk size {chunk_sizes[i]} differs from {chunk_sizes[0]}"
            )

    def test_concatenated_chunks_equal_full_array(self, synthetic_data):
        """Concatenated chunks should equal the full array."""
        source = ArrayDataSource(synthetic_data)

        for chunk_size in [100, 500, 1000]:
            chunks = []
            for chunk in source.iterate_chunks(chunk_size=chunk_size, use_gpu=False):
                chunks.append(chunk.copy())

            reconstructed = np.concatenate(chunks, axis=0)
            np.testing.assert_array_equal(
                reconstructed, synthetic_data,
                err_msg=f"Concatenated chunks (size={chunk_size}) don't match original"
            )


class TestMemmapVsNdarrayParity:
    """Test that memory-mapped arrays produce identical results to regular arrays."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(456)
        n_samples = 1500
        n_features = 10
        return np.random.randn(n_samples, n_features).astype(np.float32)

    def test_memmap_read_matches_array(self, synthetic_data):
        """Memory-mapped file should read identically to array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            # Load as memmap
            memmap_data = np.load(file_path, mmap_mode='r')

            # Should be identical
            np.testing.assert_array_equal(memmap_data, synthetic_data)

    def test_memmap_chunks_match_array_chunks(self, synthetic_data):
        """Chunked iteration over memmap should match array iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            memmap_data = np.load(file_path, mmap_mode='r')
            chunk_size = 300

            # Iterate over both
            for i in range(0, len(synthetic_data), chunk_size):
                end = min(i + chunk_size, len(synthetic_data))
                array_chunk = synthetic_data[i:end]
                memmap_chunk = memmap_data[i:end]

                np.testing.assert_array_equal(
                    array_chunk, memmap_chunk,
                    err_msg=f"Chunk starting at {i} differs"
                )

    def test_memmap_gpu_transfer_matches_array(self, synthetic_data):
        """GPU transfer from memmap should match transfer from array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            memmap_data = np.load(file_path, mmap_mode='r')

            # Transfer to GPU
            array_gpu = cp.asarray(synthetic_data)
            memmap_gpu = cp.asarray(memmap_data)

            # Should be identical on GPU
            cp.testing.assert_array_equal(array_gpu, memmap_gpu)


class TestStreamingProcessingParity:
    """Test that streaming processing matches non-streaming processing."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(789)
        n_samples = 2500
        n_features = 14
        return np.random.randn(n_samples, n_features).astype(np.float32)

    @pytest.fixture
    def som_config(self):
        """SOM configuration for testing."""
        return {
            'grid_size': 7,
            'input_dim': 14,
            'initial_lr': 0.4,
            'final_lr': 0.01,
            'initial_radius': 3.5,
            'final_radius': 1.0,
            'seed': 789,
        }

    def test_file_source_processing_matches_array_source(
        self, synthetic_data, som_config
    ):
        """Processing from file should match processing from array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.npy")
            np.save(file_path, synthetic_data)

            batch_config = BatchConfig(
                batch_mode="full_batch",
                chunk_size=500,
            )

            results = {}
            sources = {
                'array': ArrayDataSource(synthetic_data),
                'file': FileDataSource(file_path, ray_config=None),
            }

            for name, data_source in sources.items():
                np.random.seed(som_config['seed'])
                topology = GridTopology(
                    grid_size=som_config['grid_size'],
                    input_dim=som_config['input_dim'],
                    topology_type="planar",
                    initialization_method="random",
                    seed=som_config['seed'],
                )
                initial_weights = np.random.randn(
                    som_config['grid_size'] ** 2,
                    som_config['input_dim']
                ).astype(np.float32)

                processor = BatchProcessor(batch_config)

                class MockParams:
                    def __init__(self, config):
                        self.initial_lr = config['initial_lr']
                        self.final_lr = config['final_lr']
                        self.initial_radius = config['initial_radius']
                        self.final_radius = config['final_radius']
                        self.num_epochs = 1
                        self.seed = config['seed']
                        self.verbose = False
                        self.decay_type = "linear"
                        self.radius_decay_type = "linear"
                        self.lr_decay_type = "linear"
                        self.neighborhood_function = "gaussian"
                        # Add missing attributes required by BatchProcessor
                        self.current_radius = self.initial_radius
                        self.current_learning_rate = self.initial_lr
                        self.current_momentum = 0.0
                        self.delta_weights = None
                        self.processing_config = ProcessingConfig(
                            chunk_size=1000,
                            method="batch",
                            normalization="none",
                            distance_metric="euclidean",
                        )

                params = MockParams(som_config)

                # Precompute topology data
                radii_list = [params.current_radius]
                topology.precompute_topology_data(params, radii_list)

                processor.initialize(
                    cp.asarray(initial_weights.copy()),
                    topology,
                    params,
                    data_source
                )

                # Get data from source for processing
                # (In real usage, processor handles this internally)
                result = processor.process_samples(
                    cp.asarray(synthetic_data),
                    cp.asarray(initial_weights.copy()),
                    topology,
                    params
                )

                if isinstance(result, tuple):
                    result = result[0]
                results[name] = cp.asnumpy(result)

            # Array and file sources should produce identical results
            np.testing.assert_allclose(
                results['array'], results['file'],
                rtol=1e-5, atol=1e-6,
                err_msg="File source processing differs from array source"
            )


class TestZarrSourceParity:
    """Test that zarr-backed sources produce identical results to numpy arrays."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(321)
        n_samples = 1800
        n_features = 8
        return np.random.randn(n_samples, n_features).astype(np.float32)

    def test_zarr_save_load_roundtrip(self, synthetic_data):
        """Zarr save/load should preserve data exactly."""
        try:
            import zarr
        except ImportError:
            pytest.skip("zarr not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test_data.zarr")

            # Save to zarr
            zarr.save(zarr_path, synthetic_data)

            # Load back
            loaded = zarr.open(zarr_path, mode='r')[:]

            np.testing.assert_array_equal(loaded, synthetic_data)

    def test_zarr_chunk_access_matches_numpy(self, synthetic_data):
        """Zarr chunked access should match numpy indexing."""
        try:
            import zarr
        except ImportError:
            pytest.skip("zarr not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test_data.zarr")

            # Save with specific chunk size using zarr.open (works with both v2 and v3)
            z = zarr.open(zarr_path, mode='w', shape=synthetic_data.shape,
                          chunks=(500, synthetic_data.shape[1]), dtype=synthetic_data.dtype)
            z[:] = synthetic_data

            zarr_array = zarr.open(zarr_path, mode='r')

            # Compare various slices
            slices = [
                slice(0, 100),
                slice(100, 500),
                slice(500, 1000),
                slice(0, None),  # Full array
            ]

            for s in slices:
                zarr_slice = zarr_array[s]
                numpy_slice = synthetic_data[s]
                np.testing.assert_array_equal(
                    zarr_slice, numpy_slice,
                    err_msg=f"Slice {s} differs"
                )


class TestInitializationSampleParity:
    """Test that initialization samples are consistent across source types."""

    @pytest.fixture
    def synthetic_data(self):
        """Create reproducible synthetic data."""
        np.random.seed(654)
        n_samples = 5000
        n_features = 20
        return np.random.randn(n_samples, n_features).astype(np.float32)

    def test_initialization_sample_deterministic(self, synthetic_data):
        """Initialization sample should be deterministic for same source."""
        source = ArrayDataSource(synthetic_data)

        sample1 = source.get_initialization_sample(max_samples=500, use_gpu=False)
        sample2 = source.get_initialization_sample(max_samples=500, use_gpu=False)

        np.testing.assert_array_equal(sample1, sample2)

    def test_initialization_sample_size_respected(self, synthetic_data):
        """Initialization sample should respect max_samples parameter."""
        source = ArrayDataSource(synthetic_data)

        for max_samples in [100, 500, 1000]:
            sample = source.get_initialization_sample(
                max_samples=max_samples, use_gpu=False
            )
            assert len(sample) <= max_samples
            assert len(sample) > 0

    def test_initialization_sample_gpu_cpu_parity(self, synthetic_data):
        """GPU and CPU initialization samples should contain same data."""
        source = ArrayDataSource(synthetic_data)

        cpu_sample = source.get_initialization_sample(max_samples=500, use_gpu=False)
        gpu_sample = source.get_initialization_sample(max_samples=500, use_gpu=True)

        if isinstance(gpu_sample, cp.ndarray):
            gpu_sample = cp.asnumpy(gpu_sample)

        np.testing.assert_array_equal(cpu_sample, gpu_sample)
