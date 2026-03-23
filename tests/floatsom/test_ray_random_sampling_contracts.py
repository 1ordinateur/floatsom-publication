"""
Regression tests for Ray random-sampling contracts and chunk-range mapping.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.floatsom_params import RayConfig, SamplingConfig
from floatsom.base.floatsom import FloatSOM
from floatsom.sampling.base_selector import SelectionResult
from floatsom.sampling.random_selector import RandomSelector
from floatsom.processing.processing_params import BatchConfig, ColorsConfig
from floatsom.processing.ray_ops.ray_batch_processor import RayBatchProcessor
from floatsom.processing.ray_ops.ray_colors_processor import RayColorsProcessor
from floatsom.processing.ray_ops.ray_pipeline_base import RayWorkerManager
from floatsom.processing.ray_ops.selection_utils import extract_selected_indices
from floatsom.processing.ray_ops.workers import ray_pipeline_base_worker as worker_module


def _build_ray_config(tmp_path) -> RayConfig:
    return RayConfig(
        chunk_size=64,
        storage_path=str(tmp_path / "shared"),
        local_storage_path=str(tmp_path / "local"),
        num_gpus=1,
    )


def test_random_sampling_keeps_loader_chunk_equal_target_chunk(tmp_path):
    manager = RayWorkerManager(num_gpus=1, ray_config=_build_ray_config(tmp_path))
    params = SimpleNamespace(
        sampling_config=SamplingConfig(method="random", samples_per_epoch=25, target_proportion=0.25)
    )

    manager.target_chunk_size = 64
    manager._configure_sampling_parameters(params, dataset_samples=100)

    assert manager.sampling_fraction == pytest.approx(0.25)
    assert manager.loader_chunk_size == 64
    assert manager.loader_chunk_size == manager.target_chunk_size


def test_base_worker_whole_chunk_random_disables_row_subsampling():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 0
    worker.chunk_size = 64

    worker.configure_chunk_sampling(
        loader_chunk_size=64,
        sampling_fraction=0.25,
        sampling_method="random",
        whole_chunk_random=True,
    )

    assert worker.whole_chunk_random is True
    assert worker.uses_shard_local_random_sampling() is False


def test_load_chunk_sync_to_gpu_keeps_full_chunk_when_whole_chunk_random_enabled():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 21
    worker.chunk_size = 4
    worker.last_chunk_info = None

    chunk = np.arange(8, dtype=np.float32).reshape(4, 2)
    worker.data_loader = SimpleNamespace(
        get_chunk=lambda chunk_index, *, advance_cursor: (
            chunk,
            {"chunk_index": int(chunk_index), "local_size": 4, "source": "stub"},
        )
    )
    worker.configure_chunk_sampling(
        loader_chunk_size=4,
        sampling_fraction=0.25,
        sampling_method="random",
        whole_chunk_random=True,
    )

    loaded_chunk, chunk_info = worker._load_chunk_sync_to_gpu(0)

    np.testing.assert_array_equal(loaded_chunk, chunk)
    assert chunk_info["local_size"] == 4
    assert "shard_random" not in chunk_info["source"]


def test_random_selector_target_proportion_aligns_with_ray_fraction(tmp_path):
    cfg = SamplingConfig(
        method="random",
        samples_per_epoch=None,
        target_proportion=0.25,
        random_seed=123,
    )
    selector = RandomSelector(cfg)
    selector.initialize_with_data_source(SimpleNamespace(get_shape=lambda: (100, 4)))
    selection = selector.select("/tmp/mock-data.npy")

    manager = RayWorkerManager(num_gpus=1, ray_config=_build_ray_config(tmp_path))
    manager.target_chunk_size = 64
    manager._configure_sampling_parameters(
        SimpleNamespace(sampling_config=cfg),
        dataset_samples=100,
    )

    expected_count = int(100 * manager.sampling_fraction)
    assert selector.samples_per_epoch == expected_count
    assert selection.indices is not None
    assert len(selection.indices) == expected_count


def test_align_indices_to_chunks_returns_only_touched_ranges_for_unaligned_shard():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 0
    worker.loader_chunk_size = 16
    worker.data_start_idx = 105
    worker.data_end_idx = 205
    worker.data_loader = None
    worker.n_samples = 100
    worker._chunk_sampling_counter = 0

    selected_indices = np.asarray([106, 120, 203, 9999], dtype=np.int64)
    ranges = worker.align_indices_to_chunks(selected_indices)

    assert ranges == [(105, 121), (201, 205)]
    assert worker._chunk_sampling_counter == 1


class _DummyRemoteMethod:
    def remote(self, *args, **kwargs):
        return object()


class _BatchWorkerManagerStub:
    def __init__(self) -> None:
        self.data_distributed = True
        self.workers = [SimpleNamespace(process_batch_iteration=_DummyRemoteMethod())]
        self.collective_group_name = "test_group"
        self.total_samples = 100
        self.sampling_fraction = 0.2

    def ensure_distribution_ready(self):
        return True

    def wait_for_worker_futures(self, futures, *, phase, timeout_s=None, poll_interval_s=5.0):
        return [{} for _ in futures]


class _BatchWorkerManagerSuccessStub:
    def __init__(self) -> None:
        self.data_distributed = True
        self.workers = [SimpleNamespace(process_batch_iteration=_DummyRemoteMethod())]
        self.collective_group_name = "test_group"
        self.total_samples = 100
        self.sampling_fraction = 0.2

    def ensure_distribution_ready(self):
        return True

    def wait_for_worker_futures(self, futures, *, phase, timeout_s=None, poll_interval_s=5.0):
        del futures, phase, timeout_s, poll_interval_s
        return [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 3,
                "weight_change_norm": 0.0,
            }
        ]


class _ColorsWorkerManagerStub:
    def __init__(self) -> None:
        self.data_distributed = True
        self.workers = [
            SimpleNamespace(
                setup_processing=_DummyRemoteMethod(),
                process_full_iteration=_DummyRemoteMethod(),
            )
        ]
        self.collective_group_name = "test_group"
        self.sampling_fraction = 0.2

    def ensure_distribution_ready(self):
        return True

    def wait_for_worker_futures(self, futures, *, phase, timeout_s=None, poll_interval_s=5.0):
        return [{} for _ in futures]


def test_batch_processor_random_subsampling_allows_missing_selected_indices(tmp_path):
    processor = RayBatchProcessor(
        batch_config=BatchConfig(batch_mode="full_batch", chunk_size=64, weight_update_frequency=1),
        ray_config=_build_ray_config(tmp_path),
    )
    processor.worker_manager = _BatchWorkerManagerSuccessStub()

    params = SimpleNamespace(
        sampling_config=SamplingConfig(method="random"),
        processing_config=SimpleNamespace(),
    )
    result = processor.process_samples(
        samples=None,
        som_weights=np.zeros((1, 1), dtype=np.float32),
        topology=object(),
        params=params,
    )

    assert result["samples_processed"] == 3


def test_batch_processor_prefers_local_sampling_for_file_backed_random(tmp_path):
    processor = RayBatchProcessor(
        batch_config=BatchConfig(batch_mode="full_batch", chunk_size=64, weight_update_frequency=1),
        ray_config=_build_ray_config(tmp_path),
    )
    params = SimpleNamespace(sampling_config=SamplingConfig(method="random"))

    assert processor.prefers_processor_local_sampling("/tmp/mock-data.zarr", params) is True
    assert processor.prefers_processor_local_sampling(np.zeros((2, 2), dtype=np.float32), params) is False


def test_floatsom_iteration_skips_selector_for_file_backed_local_sampling():
    som = object.__new__(FloatSOM)
    seen: dict[str, Any] = {}

    class _FailSelector:
        def select(self, _dataset):
            raise AssertionError("selector.select() should not be called for file-backed Ray random sampling")

        def select_samples(self, _dataset):
            raise AssertionError("selector.select_samples() should not be called")

    class _Processor:
        def supports_selection_result(self) -> bool:
            return True

        def prefers_processor_local_sampling(self, data_reference, params) -> bool:
            del params
            return isinstance(data_reference, str)

        def process_samples(self, samples, som_weights, topology, params):
            del som_weights, topology, params
            seen["selection"] = samples
            return {
                "update_type": "inplace",
                "weight_change_norm": 0.0,
                "samples_processed": 7,
            }

    som.selector = _FailSelector()
    som.processor = _Processor()
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(samples_per_epoch=25),
        processing_config=SimpleNamespace(initial_momentum=0.0),
        store_history=False,
    )
    som.delta_weights = None
    som.weights = np.zeros((1, 1), dtype=np.float32)
    som.topology = SimpleNamespace(requires_topology_updates=lambda: False)
    som.verbose = False
    som.previous_weight_change = 0.0

    data_source = SimpleNamespace(
        get_reference=lambda: "/tmp/mock-data.zarr",
        get_shape=lambda: (100, 4),
    )
    schedules = {
        "total_iterations": 1,
        "radii": [1.0],
        "learning_rates": [0.5],
        "momentum": 0.0,
    }
    training_stats = {"iterations_completed": 0, "total_samples_processed": 0}

    FloatSOM._process_training_iteration(
        som,
        0,
        data_source,
        data_source.get_reference(),
        schedules,
        0.0,
        training_stats,
    )

    selection = seen["selection"]
    assert isinstance(selection, SelectionResult)
    assert selection.samples == "/tmp/mock-data.zarr"
    assert selection.indices is None
    assert training_stats["total_samples_processed"] == 7


def test_floatsom_whole_chunk_random_requires_file_backed_local_sampling():
    som = object.__new__(FloatSOM)
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True),
    )

    with pytest.raises(ValueError, match="whole_chunk_random requires file-backed random batch training"):
        FloatSOM._validate_whole_chunk_random_runtime(som, np.zeros((2, 2), dtype=np.float32))


def test_floatsom_whole_chunk_random_allows_file_backed_runtime_path():
    som = object.__new__(FloatSOM)
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True),
    )

    FloatSOM._validate_whole_chunk_random_runtime(som, "/tmp/mock-data.zarr")


def test_floatsom_whole_chunk_random_warns_when_chunk_is_under_one_percent():
    som = object.__new__(FloatSOM)
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True),
    )
    som.processor = SimpleNamespace(
        worker_manager=SimpleNamespace(loader_chunk_size=50),
    )

    with pytest.warns(UserWarning, match="less than 1% of the dataset"):
        FloatSOM._warn_on_whole_chunk_random_chunk_size(som, total_samples=10_000)


def test_floatsom_whole_chunk_random_skips_warning_when_chunk_is_one_percent_or_more():
    som = object.__new__(FloatSOM)
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True),
    )
    som.processor = SimpleNamespace(
        worker_manager=SimpleNamespace(loader_chunk_size=100),
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        FloatSOM._warn_on_whole_chunk_random_chunk_size(som, total_samples=10_000)

    assert not record


def test_floatsom_initialize_with_data_source_rejects_in_memory_whole_chunk_random():
    som = object.__new__(FloatSOM)
    som.verbose = False
    som.selector = SimpleNamespace()
    som.processor = SimpleNamespace(prefers_processor_local_sampling=lambda data_reference, params: False)
    som.params = SimpleNamespace(sampling_config=SimpleNamespace(whole_chunk_random=True))

    data_source = SimpleNamespace(
        get_shape=lambda: (100, 4),
        get_reference=lambda: np.zeros((100, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="whole_chunk_random requires file-backed random batch training"):
        FloatSOM._initialize_with_data_source(som, data_source)


def test_floatsom_initialize_with_data_source_emits_whole_chunk_warning_at_real_callsite():
    som = object.__new__(FloatSOM)
    som.verbose = False
    som.selector = SimpleNamespace(initialize_with_data_source=lambda _data_source: None)
    som.processor = SimpleNamespace(
        prefers_processor_local_sampling=lambda data_reference, params: isinstance(data_reference, str),
        initialize=lambda weights, topology, params, data_source: None,
        worker_manager=SimpleNamespace(loader_chunk_size=50, total_samples=10_000),
    )
    som.topology = SimpleNamespace(
        initialization_method="random",
        initialize_weights=lambda dummy_data: np.zeros((2, dummy_data.shape[1]), dtype=np.float32),
        precompute_topology_data=lambda params, radii: None,
    )
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True),
        total_iterations=1,
        initial_radius=1.0,
    )
    som.use_momentum = False
    som._ensure_backend_array = lambda array: array
    som._compute_radius_schedule = lambda total_iterations, initial_radius: [initial_radius] * total_iterations

    data_source = SimpleNamespace(
        get_shape=lambda: (10_000, 4),
        get_reference=lambda: "/tmp/mock-data.zarr",
    )

    with pytest.warns(UserWarning, match="less than 1% of the dataset"):
        FloatSOM._initialize_with_data_source(som, data_source)


def test_floatsom_initialize_with_data_source_allows_supported_whole_chunk_random_path():
    som = object.__new__(FloatSOM)
    som.verbose = False
    som.selector = SimpleNamespace(initialize_with_data_source=lambda _data_source: None)
    seen = {}

    def _capture_initialize(weights, topology, params, data_source):
        seen["weights_shape"] = tuple(weights.shape)
        seen["reference"] = data_source.get_reference()

    som.processor = SimpleNamespace(
        initialize=_capture_initialize,
        worker_manager=SimpleNamespace(loader_chunk_size=200, total_samples=10_000),
    )
    som.topology = SimpleNamespace(
        initialization_method="random",
        initialize_weights=lambda dummy_data: np.zeros((2, dummy_data.shape[1]), dtype=np.float32),
        precompute_topology_data=lambda params, radii: None,
        total_nodes=2,
    )
    som.params = SimpleNamespace(
        sampling_config=SimpleNamespace(whole_chunk_random=True, method="random"),
        total_iterations=1,
        initial_radius=1.0,
    )
    som.use_momentum = False
    som._ensure_backend_array = lambda array: array
    som._compute_radius_schedule = lambda total_iterations, initial_radius: [initial_radius] * total_iterations

    data_source = SimpleNamespace(
        get_shape=lambda: (10_000, 4),
        get_reference=lambda: "/tmp/mock-data.zarr",
        supports_streaming=lambda: True,
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        FloatSOM._initialize_with_data_source(som, data_source)

    assert seen["weights_shape"] == (2, 4)
    assert seen["reference"] == "/tmp/mock-data.zarr"
    assert not record


def test_random_selector_select_returns_selection_result_with_indices_for_arrays():
    config = SamplingConfig(method="random", samples_per_epoch=4, random_seed=123)
    selector = RandomSelector(config)
    selector.total_samples = 10
    data = np.arange(20, dtype=np.float32).reshape(10, 2)

    result = selector.select(data)

    assert isinstance(result, SelectionResult)
    assert result.indices is not None
    assert len(result.indices) == 4
    np.testing.assert_array_equal(result.samples, data[result.indices])
    extracted = extract_selected_indices(result)
    assert extracted is not None
    np.testing.assert_array_equal(extracted, result.indices)


def test_batch_processor_accepts_selection_result_for_random_subsampling(tmp_path):
    processor = RayBatchProcessor(
        batch_config=BatchConfig(batch_mode="full_batch", chunk_size=64, weight_update_frequency=1),
        ray_config=_build_ray_config(tmp_path),
    )
    processor.worker_manager = _BatchWorkerManagerSuccessStub()

    params = SimpleNamespace(
        sampling_config=SamplingConfig(method="random"),
        processing_config=SimpleNamespace(),
    )
    selection = SelectionResult(
        samples=np.zeros((3, 2), dtype=np.float32),
        indices=np.asarray([1, 5, 9], dtype=np.int64),
    )

    result = processor.process_samples(
        samples=selection,
        som_weights=np.zeros((1, 1), dtype=np.float32),
        topology=object(),
        params=params,
    )

    assert result["samples_processed"] == 3


def test_colors_processor_random_subsampling_allows_missing_selected_indices(tmp_path, monkeypatch):
    processor = RayColorsProcessor(
        colors_config=ColorsConfig(chunk_size=64, max_rounds=1, sample_order="strided"),
        ray_config=_build_ray_config(tmp_path),
    )
    processor.worker_manager = _ColorsWorkerManagerStub()
    processor.color_sets = [np.asarray([0], dtype=np.int32)]
    processor.verbose = False
    monkeypatch.setattr(processor, "_prepare_color_sets", lambda *args, **kwargs: None)
    monkeypatch.setattr(processor, "_ensure_progress_tracker", lambda: None)
    monkeypatch.setattr(
        processor,
        "_await_iteration_with_watchdog",
        lambda *args, **kwargs: [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 3,
                "weight_change_norm": 0.0,
            }
        ],
    )
    monkeypatch.setattr(processor.worker_manager, "wait_for_worker_futures", lambda *args, **kwargs: [{}])

    topology = SimpleNamespace(
        get_precomputed_influence_matrix=lambda *_args, **_kwargs: (np.zeros((1, 1), dtype=np.float32), 1.0)
    )
    params = SimpleNamespace(
        current_radius=1.0,
        sampling_config=SamplingConfig(method="random"),
        processing_config=SimpleNamespace(max_rounds=1, sample_order="strided", random_seed=None),
        delta_weights=None,
        current_momentum=0.0,
    )

    result = processor.process_samples(
        samples=None,
        som_weights=np.zeros((1, 1), dtype=np.float32),
        topology=topology,
        params=params,
    )

    assert result["samples_processed"] == 3


def test_colors_processor_ignores_selection_result_indices_for_random_subsampling(tmp_path, monkeypatch):
    processor = RayColorsProcessor(
        colors_config=ColorsConfig(chunk_size=64, max_rounds=1, sample_order="strided"),
        ray_config=_build_ray_config(tmp_path),
    )
    processor.worker_manager = _ColorsWorkerManagerStub()
    processor.color_sets = [np.asarray([0], dtype=np.int32)]
    processor.influence_matrix = np.ones((1, 1), dtype=np.float32)
    processor.verbose = False

    monkeypatch.setattr(processor, "_ensure_progress_tracker", lambda: None)
    monkeypatch.setattr(
        processor,
        "_await_iteration_with_watchdog",
        lambda *args, **kwargs: [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 3,
                "weight_change_norm": 0.0,
            }
        ],
    )
    monkeypatch.setattr(processor.worker_manager, "wait_for_worker_futures", lambda *args, **kwargs: [{}])
    setup_calls = []

    def _capture_setup(*args, **kwargs):
        setup_calls.append((args, kwargs))
        return object()

    processor.worker_manager.workers[0].setup_processing.remote = _capture_setup

    params = SimpleNamespace(
        sampling_config=SamplingConfig(method="random"),
        processing_config=SimpleNamespace(max_rounds=1, sample_order="strided", random_seed=None),
        current_momentum=0.0,
        delta_weights=None,
    )
    selection = SelectionResult(
        samples=np.zeros((3, 2), dtype=np.float32),
        indices=np.asarray([2, 4, 8], dtype=np.int64),
    )

    result = processor._process_samples(
        samples=selection,
        som_weights=np.zeros((1, 1), dtype=np.float32),
        topology=object(),
        params=params,
    )

    assert result["samples_processed"] == 3
    assert len(setup_calls) == 1
    assert setup_calls[0][1]["selected_indices"] is None


def test_colors_processor_propagates_color_order_seed_to_workers(tmp_path, monkeypatch):
    class _CaptureSetupRemote:
        def __init__(self):
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return object()

    class _CaptureIterationRemote:
        def remote(self):
            return object()

    setup_remote = _CaptureSetupRemote()
    worker = SimpleNamespace(
        setup_processing=setup_remote,
        process_full_iteration=_CaptureIterationRemote(),
    )
    worker_manager = _ColorsWorkerManagerStub()
    worker_manager.workers = [worker]
    worker_manager.sampling_fraction = 1.0

    processor = RayColorsProcessor(
        colors_config=ColorsConfig(chunk_size=64, max_rounds=1, sample_order="strided"),
        ray_config=_build_ray_config(tmp_path),
    )
    processor.worker_manager = worker_manager
    processor.color_sets = [np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32)]
    processor.verbose = False
    monkeypatch.setattr(processor, "_prepare_color_sets", lambda *args, **kwargs: None)
    monkeypatch.setattr(processor, "_ensure_progress_tracker", lambda: None)
    monkeypatch.setattr(
        processor,
        "_await_iteration_with_watchdog",
        lambda _futures, stall_timeout_s, progress_tracker: [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 0,
                "weight_change_norm": 0.0,
            }
        ],
    )

    topology = SimpleNamespace(
        get_precomputed_influence_matrix=lambda *_args, **_kwargs: (np.zeros((1, 1), dtype=np.float32), 1.0)
    )
    params = SimpleNamespace(
        current_radius=1.0,
        sampling_config=SamplingConfig(method="full"),
        processing_config=SimpleNamespace(max_rounds=1, sample_order="strided", random_seed=None),
        delta_weights=None,
        current_momentum=0.0,
        seed=777,
    )

    processor.process_samples(
        samples=np.zeros((4, 2), dtype=np.float32),
        som_weights=np.zeros((2, 2), dtype=np.float32),
        topology=topology,
        params=params,
    )

    assert len(setup_remote.calls) == 1
    args, _kwargs = setup_remote.calls[0]
    plan = dict(args[5])
    assert plan["shuffle_seed"] is None
    assert plan["color_order_seed"] == 777


class _DeviceContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any):
        return False


class _FakeCP:
    ndarray = np.ndarray
    float32 = np.float32

    class _FakeCurrentStream:
        @staticmethod
        def synchronize():
            return None

    class _FakeCUDA:
        @staticmethod
        def get_current_stream():
            return _FakeCP._FakeCurrentStream()

    cuda = _FakeCUDA()

    @staticmethod
    def asarray(array, dtype=None):
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)


class _AsyncLoaderStub:
    def __init__(self, chunks):
        self._chunks = chunks
        self.done_chunk_ids = []
        self.priority_requests = []

    def get_chunk_when_ready(self, chunk_idx: int):
        return self._chunks.get(chunk_idx)

    def mark_chunk_done(self, chunk_idx: int):
        self.done_chunk_ids.append(chunk_idx)

    def request_priority_chunks(self, chunk_indices):
        self.priority_requests.append(tuple(int(idx) for idx in chunk_indices))


def test_load_selective_async_returns_exact_requested_slice(monkeypatch):
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 4
    worker.device = _DeviceContext()
    worker.async_mode = True
    worker.loader_chunk_size = 8
    worker.data_start_idx = 100
    worker.data_end_idx = 124

    chunks = {
        0: np.arange(100, 108, dtype=np.float32)[:, None],
        1: np.arange(108, 116, dtype=np.float32)[:, None],
        2: np.arange(116, 124, dtype=np.float32)[:, None],
    }
    worker.async_loader = _AsyncLoaderStub(chunks)

    monkeypatch.setattr(worker_module, "cp", _FakeCP)

    loaded = worker._load_selective_async(103, 118)

    np.testing.assert_array_equal(loaded[:, 0], np.arange(103, 118, dtype=np.float32))
    assert loaded.shape == (15, 1)
    assert worker.async_loader.done_chunk_ids == [0, 1, 2]


def test_load_chunk_multi_buffered_async_uses_main_chunk_contract(monkeypatch):
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 9
    worker.device = _DeviceContext()
    worker.multi_buffering_enabled = True
    worker.async_mode = True
    worker.data_loader = None

    chunks = {
        1: np.arange(201, 205, dtype=np.float32)[:, None],
    }
    worker.async_loader = _AsyncLoaderStub(chunks)

    monkeypatch.setattr(worker_module, "cp", _FakeCP)

    loaded_chunk, chunk_info = worker.load_chunk_multi_buffered(
        1,
        prefetch_indices=[2, 3, 3],
    )

    np.testing.assert_array_equal(loaded_chunk[:, 0], np.arange(201, 205, dtype=np.float32))
    assert chunk_info["chunk_index"] == 1
    assert chunk_info["local_size"] == 4
    assert chunk_info["source"] == "async_loader"
    assert chunk_info["padded"] is False
    assert worker.async_loader.done_chunk_ids == [1]
    assert worker.async_loader.priority_requests == [(1, 2, 3)]


def test_load_chunk_multi_buffered_sync_caps_prefetch_to_loader_window():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 11
    worker.device = _DeviceContext()
    worker.multi_buffering_enabled = True
    worker.multi_buffering_num_buffers = 8
    worker._multi_buffer_step = 0
    worker.async_mode = False
    worker.async_loader = None
    worker.data_loader = SimpleNamespace(prefetch_buffer_size=3)
    worker.gpu_buffers = [f"buffer-{idx}" for idx in range(worker.multi_buffering_num_buffers)]
    worker.get_num_chunks = lambda: 100

    scheduled = []

    def _schedule(chunk_index, buffer_id, *, advance_cursor=True):
        scheduled.append((int(chunk_index), int(buffer_id), bool(advance_cursor)))
        return {"chunk_index": int(chunk_index), "buffer_id": int(buffer_id)}

    worker._schedule_chunk_to_buffer = _schedule

    data, info = worker.load_chunk_multi_buffered(
        10,
        prefetch_indices=[11, 12, 13, 14, 15, 16, 17],
    )

    assert data == "buffer-0"
    assert info["chunk_index"] == 10
    assert info["buffer_id"] == 0
    # Current chunk + at most (loader_prefetch_size - 1) lookahead.
    assert scheduled == [(10, 0, True), (11, 1, False), (12, 2, False)]


def test_load_chunk_sync_to_gpu_passes_explicit_consume_intent():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 14
    worker.last_chunk_info = None

    call_log = []
    expected_chunk = np.arange(8, dtype=np.float32).reshape(4, 2)

    def _get_chunk(chunk_index: int, *, advance_cursor: bool):
        call_log.append((int(chunk_index), bool(advance_cursor)))
        return expected_chunk, {"source": "stub"}

    worker.data_loader = SimpleNamespace(get_chunk=_get_chunk)

    loaded_chunk, chunk_info = worker._load_chunk_sync_to_gpu(3)

    assert call_log == [(3, True)]
    np.testing.assert_array_equal(loaded_chunk, expected_chunk)
    assert chunk_info["chunk_index"] == 3
    assert chunk_info["source"] == "stub"


def test_schedule_chunk_to_buffer_reused_prefetch_advances_cursor_once():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 12
    worker.multi_buffering_num_buffers = 3
    worker.transfer_stream = object()
    worker.gpu_buffers = [object(), object(), object()]
    worker._buffer_chunk_indices = [5, -1, -1]
    worker._buffer_chunk_info = [{"chunk_index": 5, "cursor_advanced": False}, None, None]

    marks = []
    worker.data_loader = SimpleNamespace(mark_chunk_consumed=lambda idx: marks.append(int(idx)))

    info_first = worker._schedule_chunk_to_buffer(5, 0, advance_cursor=True)
    info_second = worker._schedule_chunk_to_buffer(5, 0, advance_cursor=True)

    assert marks == [5]
    assert info_first["cursor_advanced"] is True
    assert info_second["cursor_advanced"] is True
