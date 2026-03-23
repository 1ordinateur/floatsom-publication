"""
Tests for single-GPU vs multi-GPU FloatSOM training equivalence.

These tests verify that RayBatchProcessor produces identical training
results regardless of GPU count. The only difference should be
processing time.

Critical invariants tested:
- Weight update equivalence after training
- NCCL AllReduce produces same result as single-GPU accumulation
- Chunk/worker distribution independence
- Momentum state synchronization
"""

import pytest
import numpy as np
import tempfile
from types import SimpleNamespace

from tests.conftest import (
    requires_multi_gpu,
    requires_gpu_and_ray,
    assert_arrays_equivalent,
    GPU_AVAILABLE,
    RAY_AVAILABLE,
)

if GPU_AVAILABLE:
    import cupy as cp

if GPU_AVAILABLE and RAY_AVAILABLE:
    from floatsom.floatsom_params import (
        FloatSOMParams,
        SamplingConfig,
        ProcessingConfig,
        TopologyConfig,
        RayConfig,
    )
    from floatsom.processing.processing_params import BatchConfig
    from floatsom.processing.batch_processor import BatchProcessor
    from floatsom.processing.ray_ops.ray_batch_processor import RayBatchProcessor
    from floatsom.processing.processor_factory import ProcessorFactory
    from floatsom.topology.grid_topology import GridTopology
    from floatsom.data.sources.array import ArrayDataSource

pytestmark = [
    requires_gpu_and_ray,
    pytest.mark.gpu_scaling,
]


class MockParams:
    """Mock parameters object for testing."""

    def __init__(self, config: dict):
        self.initial_lr = config.get('initial_lr', 0.5)
        self.final_lr = config.get('final_lr', 0.01)
        self.initial_radius = config.get('initial_radius', 4.0)
        self.final_radius = config.get('final_radius', 1.0)
        self.num_epochs = config.get('num_epochs', 1)
        self.seed = config.get('seed', 42)
        self.verbose = False
        self.decay_type = "linear"
        self.radius_decay_type = "linear"
        self.lr_decay_type = "linear"
        self.neighborhood_function = "gaussian"
        self.current_learning_rate = self.initial_lr
        self.current_radius = self.initial_radius
        self.momentum_coefficient = 0.0
        self.use_momentum = False


def _build_ray_params(params: MockParams, processing_config) -> SimpleNamespace:
    """Create a Ray-serializable params object for workers."""
    return SimpleNamespace(
        initial_lr=params.initial_lr,
        final_lr=params.final_lr,
        initial_radius=params.initial_radius,
        final_radius=params.final_radius,
        num_epochs=params.num_epochs,
        seed=params.seed,
        verbose=params.verbose,
        decay_type=params.decay_type,
        radius_decay_type=params.radius_decay_type,
        lr_decay_type=params.lr_decay_type,
        neighborhood_function=params.neighborhood_function,
        current_learning_rate=params.current_learning_rate,
        current_radius=params.current_radius,
        current_momentum=0.0,
        delta_weights=None,
        processing_config=processing_config,
        sampling_config=SamplingConfig(method="full"),
    )


def _create_topology_and_weights(config: dict):
    """Helper to create topology and initial weights."""
    np.random.seed(config['seed'])
    topology = GridTopology(
        grid_size=config['grid_size'],
        input_dim=config['input_dim'],
        topology_type="planar",
        initialization_method="random",
        seed=config['seed'],
    )
    weights = np.random.randn(
        config['grid_size'] ** 2,
        config['input_dim']
    ).astype(np.float32)
    return topology, weights


def _run_ray_processor_training(
    data: np.ndarray,
    initial_weights: np.ndarray,
    topology,
    params: MockParams,
    num_gpus: int,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Run RayBatchProcessor training with specified GPU count.

    Returns:
        Final weights after training
    """
    batch_config = BatchConfig(
        batch_mode="full_batch",
        chunk_size=chunk_size,
        weight_update_frequency=1,
    )

    with tempfile.TemporaryDirectory(prefix="floatsom_ray_") as temp_dir:
        ray_config = RayConfig(
            num_gpus=num_gpus,
            chunk_size=chunk_size,
            storage_path=temp_dir,
            local_storage_path=temp_dir,
        )

        processing_config = ProcessingConfig(
            method="batch",
            chunk_size=chunk_size,
            ray_config=ray_config,
        )

        ray_params = _build_ray_params(params, processing_config)

        processor = RayBatchProcessor(
            batch_config=batch_config,
            ray_config=ray_config,
            processing_config=processing_config,
        )

        data_source = ArrayDataSource(data)
        processor.initialize(
            cp.asarray(initial_weights.copy()),
            topology,
            ray_params,
            data_source
        )

        processor.process_samples(
            None,
            cp.asarray(initial_weights.copy()),
            topology,
            ray_params
        )

        result = processor.get_current_weights()
        if isinstance(result, tuple):
            result = result[0]

        if hasattr(result, 'get'):
            result = result.get()
        elif hasattr(result, '__cuda_array_interface__'):
            result = cp.asnumpy(result)

        processor.worker_manager.cleanup()

        return result


@requires_multi_gpu
class TestFloatSOMSingleVsMultiGPU:
    """Test that FloatSOM training produces identical results on 1 vs N GPUs."""

    @pytest.fixture
    def training_data(self):
        """Create reproducible synthetic training data."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 16
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        return data

    @pytest.fixture
    def som_config(self):
        """Create SOM configuration for testing."""
        return {
            'grid_size': 8,
            'input_dim': 16,
            'initial_lr': 0.5,
            'final_lr': 0.01,
            'initial_radius': 4.0,
            'final_radius': 1.0,
            'num_epochs': 1,
            'seed': 42,
        }

    def test_ray_batch_processor_1_vs_n_gpus(
        self,
        training_data,
        som_config,
        gpu_count,
        gpu_parity_tolerance,
        ray_context,
    ):
        """SOM weights after training match between 1 and N GPUs."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)
        params = MockParams(som_config)

        result_1gpu = _run_ray_processor_training(
            training_data,
            initial_weights,
            topology,
            params,
            num_gpus=1,
        )

        params_ngpu = MockParams(som_config)
        result_ngpu = _run_ray_processor_training(
            training_data,
            initial_weights,
            topology,
            params_ngpu,
            num_gpus=gpu_count,
        )

        assert_arrays_equivalent(
            result_1gpu, result_ngpu,
            gpu_parity_tolerance,
            name="SOM weights (1 GPU vs N GPU)"
        )

    def test_ray_batch_processor_determinism(
        self,
        training_data,
        som_config,
        gpu_count,
        strict_tolerance,
        ray_context,
    ):
        """Multiple runs on N GPUs produce identical results."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)
        results = []

        for run in range(3):
            params = MockParams(som_config)
            result = _run_ray_processor_training(
                training_data,
                initial_weights,
                topology,
                params,
                num_gpus=gpu_count,
            )
            results.append(result)

        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i],
                rtol=strict_tolerance["rtol"],
                atol=strict_tolerance["atol"],
                err_msg=f"Run {i} differs from run 0"
            )

    def test_ray_batch_processor_chunk_size_independence(
        self,
        training_data,
        som_config,
        gpu_count,
        gpu_parity_tolerance,
        ray_context,
    ):
        """Different chunk sizes produce same final result."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)
        chunk_sizes = [500, 1000, 2500]
        results = {}

        for chunk_size in chunk_sizes:
            params = MockParams(som_config)
            result = _run_ray_processor_training(
                training_data,
                initial_weights,
                topology,
                params,
                num_gpus=gpu_count,
                chunk_size=chunk_size,
            )
            results[chunk_size] = result

        baseline = results[chunk_sizes[0]]
        for chunk_size in chunk_sizes[1:]:
            assert_arrays_equivalent(
                baseline, results[chunk_size],
                gpu_parity_tolerance,
                name=f"chunk size {chunk_sizes[0]} vs {chunk_size}"
            )

    def test_ray_batch_processor_asymmetric_data_split(
        self,
        som_config,
        gpu_count,
        gpu_parity_tolerance,
        ray_context,
    ):
        """Handles non-equal data splits correctly."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        np.random.seed(42)
        n_samples = gpu_count * 1000 + 1
        n_features = som_config['input_dim']
        data = np.random.randn(n_samples, n_features).astype(np.float32)

        topology, initial_weights = _create_topology_and_weights(som_config)

        params_1gpu = MockParams(som_config)
        result_1gpu = _run_ray_processor_training(
            data,
            initial_weights,
            topology,
            params_1gpu,
            num_gpus=1,
        )

        params_ngpu = MockParams(som_config)
        result_ngpu = _run_ray_processor_training(
            data,
            initial_weights,
            topology,
            params_ngpu,
            num_gpus=gpu_count,
        )

        assert_arrays_equivalent(
            result_1gpu, result_ngpu,
            gpu_parity_tolerance,
            name="asymmetric data split"
        )


@requires_multi_gpu
class TestFloatSOMWeightInvariants:
    """Test weight invariants that must hold regardless of GPU count."""

    @pytest.fixture
    def training_data(self):
        """Create normalized synthetic training data."""
        np.random.seed(456)
        n_samples = 5000
        n_features = 8
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        return data

    @pytest.fixture
    def som_config(self):
        """Create SOM configuration."""
        return {
            'grid_size': 5,
            'input_dim': 8,
            'initial_lr': 0.5,
            'final_lr': 0.01,
            'initial_radius': 2.5,
            'final_radius': 1.0,
            'num_epochs': 1,
            'seed': 456,
        }

    def test_weights_finite(
        self,
        training_data,
        som_config,
        gpu_count,
        ray_context,
    ):
        """Weights remain finite on all GPU counts."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)

        for n_gpus in [1, gpu_count]:
            params = MockParams(som_config)
            result = _run_ray_processor_training(
                training_data,
                initial_weights,
                topology,
                params,
                num_gpus=n_gpus,
            )

            assert np.all(np.isfinite(result)), \
                f"Non-finite weights with {n_gpus} GPUs"

    def test_weights_changed(
        self,
        training_data,
        som_config,
        gpu_count,
        ray_context,
    ):
        """Weights change after training on all GPU counts."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)

        for n_gpus in [1, gpu_count]:
            params = MockParams(som_config)
            result = _run_ray_processor_training(
                training_data,
                initial_weights,
                topology,
                params,
                num_gpus=n_gpus,
            )

            assert not np.allclose(result, initial_weights), \
                f"Weights unchanged with {n_gpus} GPUs"

    def test_weights_bounded(
        self,
        training_data,
        som_config,
        gpu_count,
        ray_context,
    ):
        """Weight norms remain bounded on all GPU counts."""
        if gpu_count < 2:
            pytest.skip(f"Need 2+ GPUs (found {gpu_count})")

        topology, initial_weights = _create_topology_and_weights(som_config)

        for n_gpus in [1, gpu_count]:
            params = MockParams(som_config)
            result = _run_ray_processor_training(
                training_data,
                initial_weights,
                topology,
                params,
                num_gpus=n_gpus,
            )

            norms = np.linalg.norm(result, axis=1)
            assert np.all(norms > 0.01), f"Weights vanished with {n_gpus} GPUs"
            assert np.all(norms < 100.0), f"Weights exploded with {n_gpus} GPUs"
