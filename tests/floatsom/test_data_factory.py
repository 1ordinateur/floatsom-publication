"""
Test data source factory and routing.

Tests the DataSourceFactory and various DataSource implementations:
- Factory routing (array vs file inputs)
- ArrayDataSource functionality
- FileDataSource functionality
- Custom creator registration
- Shape detection and iteration
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from floatsom.data.factory import DataSourceFactory
from floatsom.data.sources.base import DataSource
from floatsom.data.sources.array import ArrayDataSource


# =============================================================================
# Factory Routing Tests
# =============================================================================

class TestFactoryRouting:
    """Test factory correctly routes to appropriate data source."""

    def test_numpy_array_creates_array_source(self):
        """Numpy array input should create ArrayDataSource."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = DataSourceFactory.create(data)

        assert isinstance(source, ArrayDataSource)
        assert source.get_shape() == (100, 50)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cupy_array_creates_array_source(self):
        """CuPy array input should create ArrayDataSource."""
        data = cp.random.rand(100, 50).astype(cp.float32)
        source = DataSourceFactory.create(data)

        assert isinstance(source, ArrayDataSource)
        assert source.get_shape() == (100, 50)

    def test_file_path_creates_file_source(self, tmp_path):
        """String file path should create FileDataSource."""
        # Create a test file
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))

        assert source.supports_streaming() == True
        assert source.get_shape() == (100, 50)

    def test_unsupported_type_raises_error(self):
        """Unsupported input type should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported data input type"):
            DataSourceFactory.create([1, 2, 3])  # List is not supported

    def test_nonexistent_file_raises_error(self):
        """Non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataSourceFactory.create("/nonexistent/path/data.npy")


# =============================================================================
# Type Identification Tests
# =============================================================================

class TestTypeIdentification:
    """Test type identification for various inputs."""

    def test_identify_numpy_array(self):
        """Should identify numpy arrays."""
        data = np.zeros((10, 5))
        result = DataSourceFactory._identify_type(data)
        assert result == "numpy_array"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_identify_cupy_array(self):
        """Should identify cupy arrays."""
        data = cp.zeros((10, 5))
        result = DataSourceFactory._identify_type(data)
        assert result == "cupy_array"

    def test_identify_npy_file(self):
        """Should identify .npy files."""
        result = DataSourceFactory._identify_type("/path/to/data.npy")
        assert result == "numpy_file"

    def test_identify_parquet_file(self):
        """Should identify .parquet files."""
        result = DataSourceFactory._identify_type("/path/to/data.parquet")
        assert result == "parquet_file"

    def test_identify_csv_file(self):
        """Should identify .csv files."""
        result = DataSourceFactory._identify_type("/path/to/data.csv")
        assert result == "csv_file"

    def test_identify_zarr_file(self):
        """Should identify .zarr stores by suffix."""
        result = DataSourceFactory._identify_type("/path/to/data.zarr")
        assert result == "zarr_file"

    def test_identify_fast_array_store(self):
        """Should identify .fast FastArrayStore directories by suffix."""
        result = DataSourceFactory._identify_type("/path/to/data.fast")
        assert result == "fast_array_store"

    def test_identify_unknown_type(self):
        """Should return 'unknown' for unrecognized types."""
        result = DataSourceFactory._identify_type(42)
        assert result == "unknown"


# =============================================================================
# ArrayDataSource Tests
# =============================================================================

class TestArrayDataSource:
    """Test ArrayDataSource functionality."""

    def test_get_shape(self):
        """Should return correct shape."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        assert source.get_shape() == (100, 50)

    def test_get_reference_returns_array(self):
        """get_reference should return the underlying array."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        ref = source.get_reference()
        assert ref is data

    def test_supports_streaming_false(self):
        """Arrays don't support streaming."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        assert source.supports_streaming() == False

    def test_get_memory_requirement(self):
        """Should return correct memory estimate."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        # 100 * 50 * 4 bytes = 20000 bytes
        assert source.get_memory_requirement() == data.nbytes

    def test_get_initialization_sample_full(self):
        """Should return full array when max_samples is None or >= n."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        sample = source.get_initialization_sample(max_samples=None, use_gpu=False)
        assert sample.shape == (100, 50)

    def test_get_initialization_sample_limited(self):
        """Should return limited samples when max_samples < n."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        sample = source.get_initialization_sample(max_samples=30, use_gpu=False)
        assert sample.shape == (30, 50)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_get_initialization_sample_gpu_conversion(self):
        """Should convert to GPU when requested."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        sample = source.get_initialization_sample(max_samples=30, use_gpu=True)
        assert isinstance(sample, cp.ndarray)
        assert sample.shape == (30, 50)

    def test_iterate_chunks_correct_sizes(self):
        """Chunks should have correct sizes."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        chunks = list(source.iterate_chunks(chunk_size=30, use_gpu=False))

        assert len(chunks) == 4  # 30 + 30 + 30 + 10 = 100
        assert chunks[0].shape == (30, 50)
        assert chunks[1].shape == (30, 50)
        assert chunks[2].shape == (30, 50)
        assert chunks[3].shape == (10, 50)

    def test_iterate_chunks_total_samples(self):
        """Total samples from chunks should equal original."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        chunks = list(source.iterate_chunks(chunk_size=30, use_gpu=False))
        total_samples = sum(chunk.shape[0] for chunk in chunks)

        assert total_samples == 100

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_iterate_chunks_gpu_conversion(self):
        """Chunks should be converted to GPU when requested."""
        data = np.random.rand(100, 50).astype(np.float32)
        source = ArrayDataSource(data)

        chunks = list(source.iterate_chunks(chunk_size=30, use_gpu=True))

        for chunk in chunks:
            assert isinstance(chunk, cp.ndarray)


# =============================================================================
# FileDataSource Tests
# =============================================================================

class TestFileDataSource:
    """Test FileDataSource functionality."""

    def test_npy_file_loading(self, tmp_path):
        """Should load .npy files correctly."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))

        assert source.get_shape() == (100, 50)

    def test_npz_file_loading(self, tmp_path):
        """Should load .npz files correctly."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npz"
        np.savez(file_path, data=data)

        source = DataSourceFactory.create(str(file_path))

        assert source.get_shape() == (100, 50)

    def test_csv_file_loading(self, tmp_path):
        """Should load .csv files correctly."""
        data = np.random.rand(50, 10).astype(np.float32)
        file_path = tmp_path / "test_data.csv"
        np.savetxt(file_path, data, delimiter=',')

        source = DataSourceFactory.create(str(file_path))

        assert source.get_shape() == (50, 10)

    def test_zarr_store_loading(self, tmp_path):
        """Should load directory-backed Zarr stores correctly."""
        zarr = pytest.importorskip("zarr")
        store_path = tmp_path / "test_data.zarr"
        z = zarr.open_array(
            str(store_path),
            mode="w",
            shape=(64, 3),
            chunks=(16, 3),
            dtype="float32",
        )
        z[:] = np.random.rand(64, 3).astype(np.float32)

        source = DataSourceFactory.create(str(store_path))
        assert source.get_shape() == (64, 3)

    def test_fast_array_store_loading(self, tmp_path):
        """Should load FastArrayStore directories correctly."""
        from floatsom.data.fast_array_store import FastArrayStore

        store_path = tmp_path / "test_data.fast"
        store = FastArrayStore(str(store_path), mode="w")
        data = np.random.rand(32, 5).astype(np.float32)
        arr = store.create(shape=data.shape, dtype=np.float32, chunks=(16, 5))
        arr[:] = data
        store.close()

        source = DataSourceFactory.create(str(store_path))
        assert source.get_shape() == (32, 5)

    def test_get_reference_returns_path(self, tmp_path):
        """get_reference should return file path string."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))
        ref = source.get_reference()

        assert isinstance(ref, str)
        assert "test_data.npy" in ref

    def test_supports_streaming_true(self, tmp_path):
        """File sources should support streaming."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))

        assert source.supports_streaming() == True

    def test_iterate_chunks_file(self, tmp_path):
        """Should iterate chunks from file correctly."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))
        chunks = list(source.iterate_chunks(chunk_size=30, use_gpu=False))

        assert len(chunks) == 4
        total_samples = sum(chunk.shape[0] for chunk in chunks)
        assert total_samples == 100

    def test_cleanup(self, tmp_path):
        """cleanup should release resources."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))
        source.get_shape()  # Trigger loading
        source.cleanup()

        # After cleanup, internal data should be cleared
        assert source._data is None


# =============================================================================
# Custom Creator Registration Tests
# =============================================================================

class TestCustomCreatorRegistration:
    """Test custom creator registration functionality."""

    def test_register_and_use_custom_creator(self):
        """Should be able to register and use custom creators."""
        # Define a custom creator
        class CustomSource(DataSource):
            def __init__(self, data, ray_config):
                self.data = data
                self._shape = (10, 5)

            def get_shape(self):
                return self._shape

            def get_initialization_sample(self, max_samples=None, use_gpu=True):
                return np.zeros((10, 5))

            def iterate_chunks(self, chunk_size, use_gpu=True):
                yield np.zeros((chunk_size, 5))

            def get_reference(self):
                return self.data

            def supports_streaming(self):
                return False

            def get_memory_requirement(self):
                return 200

        def custom_creator(data, ray_config):
            return CustomSource(data, ray_config)

        # Register custom creator
        DataSourceFactory.register_creator("test_custom", custom_creator)

        # Verify it's registered
        assert "test_custom" in DataSourceFactory._creators


# =============================================================================
# Config-based Creation Tests
# =============================================================================

class TestConfigBasedCreation:
    """Test creation from configuration dictionaries."""

    def test_create_from_config_array(self, tmp_path):
        """Should create array source from config."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        config = {
            'type': 'array',
            'path': str(file_path)
        }

        source = DataSourceFactory.create_from_config(config)
        assert source.get_shape() == (100, 50)

    def test_create_from_config_file(self, tmp_path):
        """Should create file source from config."""
        data = np.random.rand(100, 50).astype(np.float32)
        file_path = tmp_path / "test_data.npy"
        np.save(file_path, data)

        config = {
            'type': 'file',
            'path': str(file_path)
        }

        source = DataSourceFactory.create_from_config(config)
        assert source.get_shape() == (100, 50)

    def test_create_from_config_missing_path(self):
        """Should raise error when path is missing."""
        config = {
            'type': 'array'
        }

        with pytest.raises(ValueError, match="requires 'path'"):
            DataSourceFactory.create_from_config(config)

    def test_create_from_config_unknown_type(self):
        """Should raise error for unknown type."""
        config = {
            'type': 'unknown_type',
            'path': '/some/path'
        }

        with pytest.raises(ValueError, match="Unknown data source type"):
            DataSourceFactory.create_from_config(config)


# =============================================================================
# Edge Cases
# =============================================================================

class TestDataSourceEdgeCases:
    """Test edge cases for data sources."""

    def test_single_sample_array(self):
        """Should handle single sample arrays."""
        data = np.random.rand(1, 50).astype(np.float32)
        source = ArrayDataSource(data)

        assert source.get_shape() == (1, 50)

        chunks = list(source.iterate_chunks(chunk_size=10, use_gpu=False))
        assert len(chunks) == 1
        assert chunks[0].shape == (1, 50)

    def test_single_feature_array(self):
        """Should handle single feature arrays."""
        data = np.random.rand(100, 1).astype(np.float32)
        source = ArrayDataSource(data)

        assert source.get_shape() == (100, 1)

    def test_large_chunk_size(self):
        """Chunk size larger than data should return single chunk."""
        data = np.random.rand(50, 20).astype(np.float32)
        source = ArrayDataSource(data)

        chunks = list(source.iterate_chunks(chunk_size=100, use_gpu=False))
        assert len(chunks) == 1
        assert chunks[0].shape == (50, 20)

    def test_exact_chunk_division(self):
        """Should handle data that divides exactly into chunks."""
        data = np.random.rand(100, 20).astype(np.float32)
        source = ArrayDataSource(data)

        chunks = list(source.iterate_chunks(chunk_size=25, use_gpu=False))
        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk.shape == (25, 20)

    def test_1d_file_reshaped(self, tmp_path):
        """1D files should be reshaped to (n, 1)."""
        data = np.random.rand(100).astype(np.float32)
        file_path = tmp_path / "test_1d.npy"
        np.save(file_path, data)

        source = DataSourceFactory.create(str(file_path))
        shape = source.get_shape()

        assert len(shape) == 2
        assert shape[0] == 100
        assert shape[1] == 1
