"""
Tests for sampling selectors in FloatSOM
Tests FullSelector, RandomSelector, and HDSSSOMSelector
"""

import numpy as np
import cupy as cp
import pytest

from floatsom.sampling.full_selector import FullSelector
from floatsom.sampling.random_selector import RandomSelector
from floatsom.sampling.hdsssom_selector import HDSSSOMSelector
from floatsom.sampling.base_selector import SampleSelector
from floatsom.floatsom_params import SamplingConfig


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return cp.random.randn(1000, 10).astype(cp.float32)


@pytest.fixture
def sample_data_cpu():
    """Create sample data on CPU for testing"""
    return np.random.randn(1000, 10).astype(np.float32)


class TestFullSelector:
    """Tests for FullSelector"""

    def test_full_selector_returns_all_indices(self, sample_data):
        """FullSelector should return every index"""
        selector = FullSelector()
        selector.initialize(sample_data)

        result = selector.select_samples(sample_data)

        # Result should be the entire dataset
        assert result is sample_data or cp.array_equal(result, sample_data)

    def test_full_selector_consistent_across_calls(self, sample_data):
        """FullSelector should return same indices each call"""
        selector = FullSelector()
        selector.initialize(sample_data)

        result1 = selector.select_samples(sample_data)
        result2 = selector.select_samples(sample_data)

        # Both calls should return the same reference or equal data
        assert result1 is result2 or cp.array_equal(result1, result2)

    def test_full_selector_file_path_mode(self):
        """FullSelector should handle file path strings"""
        selector = FullSelector()
        file_path = "/path/to/data.npy"
        selector.initialize(file_path)

        result = selector.select_samples(file_path)

        # Should return the file path unchanged
        assert result == file_path


class TestRandomSelector:
    """Tests for RandomSelector"""

    def test_random_selector_respects_proportion(self, sample_data):
        """RandomSelector should sample approximately target_proportion"""
        target_proportion = 0.3
        config = SamplingConfig(
            method="random",
            target_proportion=target_proportion,
            samples_per_epoch=None,
            random_seed=None
        )

        selector = RandomSelector(config)
        selector.total_samples = len(sample_data)
        selector.samples_per_epoch = int(len(sample_data) * target_proportion)
        selector.initialize(sample_data)

        result = selector.select_samples(sample_data)

        # Check that approximately target_proportion samples are selected
        expected_count = int(len(sample_data) * target_proportion)
        assert len(result) == expected_count

    def test_random_selector_different_each_epoch(self, sample_data):
        """RandomSelector should return different samples per epoch without seed"""
        config = SamplingConfig(
            method="random",
            target_proportion=0.5,
            samples_per_epoch=None,
            random_seed=None
        )

        selector = RandomSelector(config)
        selector.total_samples = len(sample_data)
        selector.samples_per_epoch = len(sample_data) // 2
        selector.initialize(sample_data)

        result1 = selector.select_samples(sample_data)
        result2 = selector.select_samples(sample_data)

        # Results should be different (with very high probability)
        # Check that at least some samples differ
        if isinstance(result1, cp.ndarray) and isinstance(result2, cp.ndarray):
            # Compare a subset to avoid full array comparison
            assert not cp.array_equal(result1[:10], result2[:10])

    def test_random_selector_seed_reproducibility(self, sample_data):
        """RandomSelector with same seed should produce same samples"""
        seed = 42
        config1 = SamplingConfig(
            method="random",
            target_proportion=0.5,
            samples_per_epoch=None,
            random_seed=seed
        )
        config2 = SamplingConfig(
            method="random",
            target_proportion=0.5,
            samples_per_epoch=None,
            random_seed=seed
        )

        selector1 = RandomSelector(config1)
        selector1.total_samples = len(sample_data)
        selector1.samples_per_epoch = len(sample_data) // 2
        selector1.initialize(sample_data)

        selector2 = RandomSelector(config2)
        selector2.total_samples = len(sample_data)
        selector2.samples_per_epoch = len(sample_data) // 2
        selector2.initialize(sample_data)

        # Reset seed before each selection to ensure same RNG state
        cp.random.seed(seed)
        result1 = selector1.select_samples(sample_data)
        cp.random.seed(seed)
        result2 = selector2.select_samples(sample_data)

        # With same seed reset before each call, results should be identical
        assert cp.array_equal(result1, result2)

    def test_random_selector_select_samples_keeps_gpu_indexing_on_device(self, sample_data, monkeypatch):
        """Sample-only selection should not force GPU->CPU index conversion."""
        config = SamplingConfig(
            method="random",
            target_proportion=0.5,
            samples_per_epoch=100,
            random_seed=123
        )
        selector = RandomSelector(config)
        selector.total_samples = len(sample_data)
        selector.initialize(sample_data)

        def _fail_asnumpy(*_args, **_kwargs):
            raise AssertionError("cp.asnumpy must not be called by select_samples()")

        monkeypatch.setattr(cp, "asnumpy", _fail_asnumpy)
        result = selector.select_samples(sample_data)

        assert isinstance(result, cp.ndarray)
        assert len(result) == 100


class _IndexOnlySelector(SampleSelector):
    def select_samples(self, dataset):
        del dataset
        return np.asarray([1, 3, 5], dtype=np.int64)


def test_default_selector_contract_rejects_index_payload_without_indices():
    selector = _IndexOnlySelector()

    with pytest.raises(TypeError, match="get_selected_indices"):
        selector.select(np.zeros((10, 2), dtype=np.float32))


class TestHDSSSOMSelector:
    """Tests for HDSSSOMSelector"""

    def test_hdsssom_initializes_metadata(self, sample_data):
        """HDSSSOM should initialize difficulty and age arrays"""
        config = SamplingConfig(
            method="hdsssom",
            target_proportion=0.5,
            samples_per_epoch=None,
            block_size=100,
            p_block_difficulty=0.9,
            p_exemplar_difficulty=0.7,
            alpha=0.5,
            min_blocks_to_select=2
        )

        selector = HDSSSOMSelector(config)
        selector.initialize(sample_data)

        # Check that metadata is initialized
        assert selector.metadata is not None
        assert selector.metadata.difficulties is not None
        assert selector.metadata.ages is not None
        assert len(selector.metadata.difficulties) == len(sample_data)
        assert len(selector.metadata.ages) == len(sample_data)

    def test_hdsssom_updates_difficulty(self, sample_data):
        """HDSSSOM should update difficulty after processing samples"""
        config = SamplingConfig(
            method="hdsssom",
            target_proportion=0.5,
            samples_per_epoch=500,
            block_size=100,
            p_block_difficulty=0.9,
            p_exemplar_difficulty=0.7,
            alpha=0.5,
            min_blocks_to_select=2
        )

        selector = HDSSSOMSelector(config)
        selector.initialize(sample_data)

        # Get initial difficulties
        initial_difficulties = selector.metadata.difficulties.copy()

        # Select samples
        selected_samples = selector.select_samples(sample_data)

        # Create mock BMUs and distances
        bmus = cp.random.randint(0, 100, size=len(selected_samples))
        distances = cp.random.rand(len(selected_samples)).astype(cp.float32)

        # Update metadata
        selector.update_metadata(selected_samples, bmus, distances)

        # Difficulties should have changed for processed samples
        updated_difficulties = selector.metadata.difficulties
        assert not cp.array_equal(initial_difficulties, updated_difficulties)

    def test_hdsssom_ages_increment(self, sample_data):
        """HDSSSOM should increment ages for non-selected samples"""
        config = SamplingConfig(
            method="hdsssom",
            target_proportion=0.3,
            samples_per_epoch=300,
            block_size=100,
            p_block_difficulty=0.9,
            p_exemplar_difficulty=0.7,
            alpha=0.5,
            min_blocks_to_select=2
        )

        selector = HDSSSOMSelector(config)
        selector.initialize(sample_data)

        # Get initial ages
        initial_ages = selector.metadata.ages.copy()

        # Select samples and update metadata
        selected_samples = selector.select_samples(sample_data)
        bmus = cp.random.randint(0, 100, size=len(selected_samples))
        distances = cp.random.rand(len(selected_samples)).astype(cp.float32)
        selector.update_metadata(selected_samples, bmus, distances)

        # Ages should have changed
        updated_ages = selector.metadata.ages
        assert not cp.array_equal(initial_ages, updated_ages)

        # Some ages should be greater than initial (non-selected samples)
        assert cp.any(updated_ages > initial_ages)
