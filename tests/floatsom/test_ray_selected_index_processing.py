"""Regression tests for selected-index handling in Ray batch/colors workers."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

import cupy as cp

from floatsom.processing.ray_ops.workers import ray_batch_worker as batch_module
from floatsom.processing.ray_ops.workers import ray_color_worker as color_module
from floatsom.processing.ray_ops.workers import ray_pipeline_base_worker as worker_module


def _resolve_worker_impl_class(ray_worker_cls):
    """Return undecorated worker class when Ray wraps it as an actor class."""
    cls = ray_worker_cls
    ray_meta = getattr(cls, "__ray_metadata__", None)
    modified_class = getattr(ray_meta, "modified_class", None)
    if isinstance(modified_class, type):
        return modified_class

    for attr in ("__ray_actor_class__", "_modified_class", "_ray_actor_class"):
        candidate = getattr(cls, attr, None)
        if isinstance(candidate, type):
            return candidate

    if isinstance(cls, type):
        return cls

    raise TypeError("Unable to resolve underlying worker implementation class")


def _make_batch_worker(worker_id: int = 0):
    worker_cls = _resolve_worker_impl_class(batch_module.RayBatchWorker)
    worker = object.__new__(worker_cls)
    worker.worker_id = int(worker_id)
    worker.assigned_gpu = 0
    worker.device = nullcontext()
    worker.num_gpus = 2
    worker.loader_chunk_size = 4
    worker.chunk_size = 4
    worker.data_start_idx = 0
    worker.data_end_idx = 8
    worker.data_loader = None
    worker.n_samples = 8
    worker.gpu_weights = cp.zeros((3, 2), dtype=cp.float32)
    worker.influence_matrix = cp.ones((3,), dtype=cp.float32)
    worker.delta_weights = cp.zeros_like(worker.gpu_weights)
    worker.weights_shape = worker.gpu_weights.shape
    worker.multi_buffering_enabled = False
    worker.async_mode = False
    worker.async_loader = None
    worker._async_loader_context = None
    worker._current_async_order = None
    worker._async_loader_sampling_fraction_override = None
    worker._async_loader_target_rows_override = None
    worker.sampling_fraction = 0.5
    worker.iteration_count = 0
    worker._cached_topology_data = None
    worker._start_worker_profile = lambda *_args, **_kwargs: None
    worker.collective_barriers_enabled = lambda: False
    worker.reset_for_iteration = lambda: None
    return worker


def _make_color_worker(worker_id: int = 0):
    worker_cls = _resolve_worker_impl_class(color_module.RayColorWorker)
    worker = object.__new__(worker_cls)
    worker.worker_id = int(worker_id)
    worker.multi_buffering_enabled = False
    worker.sample_order = "strided"
    worker.chunk_size = 16
    worker.gpu_weights = cp.zeros((5, 3), dtype=cp.float32)
    worker.neuron_to_color_cpu = None
    worker._selected_chunk_local_positions = None
    worker._record_timing = lambda *_args, **_kwargs: None
    return worker


def test_batch_selected_rows_only_feed_bmu_updates_and_count_matches(monkeypatch):
    worker = _make_batch_worker()

    chunk_payloads = {
        (0, 4): cp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
        (4, 8): cp.asarray(np.arange(12, 24, dtype=np.float32).reshape(4, 3)),
    }
    worker.load_selective = lambda start, end: chunk_payloads[(int(start), int(end))]

    seen_batch_rows = []

    def fake_find_bmus(batch, *_args, **_kwargs):
        seen_batch_rows.append(int(batch.shape[0]))
        return cp.zeros(int(batch.shape[0]), dtype=cp.int32)

    def fake_compute_weight_updates(
        *,
        batch,
        bmus,
        weights,
        influence_matrix,
        learning_rate,
        chunk_size,
        verbose,
    ):
        del influence_matrix, learning_rate, chunk_size, verbose
        assert int(batch.shape[0]) == int(bmus.shape[0])
        rows = float(batch.shape[0])
        return (
            cp.full_like(weights, rows),
            cp.full((weights.shape[0],), rows, dtype=weights.dtype),
        )

    monkeypatch.setattr(batch_module, "find_bmus", fake_find_bmus)
    monkeypatch.setattr(batch_module, "compute_weight_updates", fake_compute_weight_updates)

    params = SimpleNamespace(
        current_learning_rate=0.1,
        processing_config=SimpleNamespace(distance_metric="euclidean", distance_metric_params={}),
    )
    selected_indices = np.array([1, 3, 4, 7], dtype=np.int64)

    updates, influence, samples_processed = worker._process_selected_indices(
        selected_indices,
        params,
        topology_data=SimpleNamespace(),
    )

    assert seen_batch_rows == [2, 2]
    assert samples_processed == 4
    np.testing.assert_allclose(cp.asnumpy(updates), np.full((3, 2), 4.0, dtype=np.float32))
    np.testing.assert_allclose(cp.asnumpy(influence), np.full((3,), 4.0, dtype=np.float32))


def test_batch_selected_mode_uses_selected_denominator_and_async_overrides():
    worker = _make_batch_worker()
    worker.async_mode = True
    worker._async_loader_context = {"num_chunks": 5}
    worker.async_loader = SimpleNamespace(sampling_fraction=0.25, target_rows=2)

    restart = {}

    def fake_ensure_async_loader_order(order, force_restart=False):
        restart["order"] = np.asarray(order, dtype=np.int32)
        restart["force_restart"] = bool(force_restart)
        worker.async_loader.sampling_fraction = 1.0
        worker.async_loader.target_rows = None

    worker._ensure_async_loader_order = fake_ensure_async_loader_order

    worker._process_selected_indices = lambda _sel, _params, _topo: (
        cp.zeros_like(worker.gpu_weights),
        cp.zeros((worker.gpu_weights.shape[0],), dtype=worker.gpu_weights.dtype),
        3,
    )
    worker._apply_nccl_reduction = lambda _updates, _influence: None

    captured = {}

    def fake_finalize(_updates, _influence, _params, total_samples):
        captured["total_samples"] = int(total_samples)
        return cp.zeros_like(worker.gpu_weights)

    worker._finalize_weight_updates = fake_finalize

    class _Topology:
        @staticmethod
        def get_precomputed_influence_matrix(_radius, _kernel):
            return np.ones((3, 3), dtype=np.float32)

    params = SimpleNamespace(
        current_radius=1.0,
        current_learning_rate=0.1,
        current_momentum=0.0,
        delta_weights=None,
        total_samples=100,
        processing_config=SimpleNamespace(
            normalization="none",
            norm_alpha=1.0,
            norm_clamp_factor=1.0,
            norm_percentile=95.0,
            norm_max_update_threshold=None,
            training_progress=0.0,
            current_epoch=0,
            total_epochs=1,
            virtual_ratio=0.5,
            enable_multi_buffering=0,
        ),
    )
    selected_indices = np.array([10, 11, 12], dtype=np.int64)

    result = worker.process_batch_iteration(
        som_weights=None,
        topology_data=_Topology(),
        params=params,
        collective_group="test-group",
        selected_indices=selected_indices,
        is_first_iteration=False,
    )

    assert result["samples_processed"] == 3
    assert captured["total_samples"] == 3
    assert worker._async_loader_sampling_fraction_override == 1.0
    assert worker._async_loader_target_rows_override == 0
    assert restart["force_restart"] is True
    np.testing.assert_array_equal(restart["order"], np.arange(5, dtype=np.int32))


def test_colors_selected_mode_filters_before_bmu_and_keeps_local_indices(monkeypatch):
    worker = _make_color_worker()
    worker.neuron_to_color_cpu = np.array([-1, 1, -1, 0, 1], dtype=np.int32)
    worker._selected_chunk_local_positions = {0: np.array([1, 4], dtype=np.int32)}
    worker.load_chunk = lambda _chunk_idx: cp.asarray(
        np.arange(18, dtype=np.float32).reshape(6, 3)
    )

    seen_bmu_rows = []

    def fake_find_bmus(batch, *_args, **_kwargs):
        seen_bmu_rows.append(int(batch.shape[0]))
        return cp.asarray([3, 1], dtype=cp.int32)

    monkeypatch.setattr(color_module, "find_bmus", fake_find_bmus)

    descriptors = [[], []]
    used_colors = worker._collect_descriptors_for_chunk(
        0,
        descriptors,
        rng=None,
        metric="euclidean",
        metric_kwargs={},
    )

    assert used_colors == 2
    assert seen_bmu_rows == [2]

    color0 = descriptors[0][0]
    color1 = descriptors[1][0]
    np.testing.assert_array_equal(color0["local_indices"], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(color1["local_indices"], np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(color0["bmus"], np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(color1["bmus"], np.array([1], dtype=np.int32))


def test_colors_grouping_preserves_within_color_order_without_sort_unique(monkeypatch):
    worker = _make_color_worker()
    worker.neuron_to_color_cpu = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    worker._selected_chunk_local_positions = None
    worker.load_chunk = lambda _chunk_idx: cp.asarray(
        np.arange(18, dtype=np.float32).reshape(6, 3)
    )

    def fake_find_bmus(batch, *_args, **_kwargs):
        assert int(batch.shape[0]) == 6
        return cp.asarray([4, 2, 5, 1, 3, 0], dtype=cp.int32)

    monkeypatch.setattr(color_module, "find_bmus", fake_find_bmus)

    descriptors = [[], [], []]
    used_colors = worker._collect_descriptors_for_chunk(
        0,
        descriptors,
        rng=None,
        metric="euclidean",
        metric_kwargs={},
    )

    assert used_colors == 3
    np.testing.assert_array_equal(descriptors[0][0]["local_indices"], np.array([3, 5], dtype=np.int32))
    np.testing.assert_array_equal(descriptors[0][0]["bmus"], np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(descriptors[1][0]["local_indices"], np.array([1, 4], dtype=np.int32))
    np.testing.assert_array_equal(descriptors[1][0]["bmus"], np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(descriptors[2][0]["local_indices"], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(descriptors[2][0]["bmus"], np.array([4, 5], dtype=np.int32))


def test_colors_configure_chunk_sampling_sets_random_loader_overrides():
    worker = _make_color_worker()

    worker.configure_chunk_sampling(loader_chunk_size=32, sampling_fraction=0.25)

    assert worker.loader_chunk_size == 32
    assert worker.sampling_fraction == pytest.approx(0.25)
    assert worker._async_loader_sampling_fraction_override == 1.0
    assert worker._async_loader_target_rows_override == 0

    worker.configure_chunk_sampling(loader_chunk_size=32, sampling_fraction=1.0)
    assert worker._async_loader_sampling_fraction_override is None
    assert worker._async_loader_target_rows_override is None


def test_batch_worker_subsample_loaded_chunk_uses_local_random_fraction():
    worker = _make_batch_worker()
    worker.sampling_method = "random"
    worker.sampling_fraction = 0.5
    worker._current_chunk_iteration = 3

    chunk = cp.asarray(np.arange(24, dtype=np.float32).reshape(8, 3))
    sampled, info = worker._subsample_loaded_chunk(
        chunk,
        {"chunk_index": 2, "local_size": 8, "source": "mmap"},
    )

    assert int(sampled.shape[0]) == 4
    assert info["local_size_before_sampling"] == 8
    assert info["local_size"] == 4
    assert info["source"] == "mmap_shard_random"


def test_batch_worker_multi_buffered_path_applies_local_random_subsampling_after_ready_wait(monkeypatch):
    worker = _make_batch_worker()
    worker.sampling_method = "random"
    worker.sampling_fraction = 0.5
    worker.multi_buffering_enabled = True
    
    class _DummyStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def synchronize(self):
            return None

    worker.compute_stream = _DummyStream()
    worker.get_num_chunks = lambda: 1
    worker.load_chunk_multi_buffered = lambda _chunk_idx: (
        cp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
        {"buffer_id": 0, "local_size": 3, "source": "mmap", "padded": True},
    )
    wait_state = {"ready": False}
    worker._wait_for_buffer_ready = lambda _buffer_id: wait_state.__setitem__("ready", True)
    worker._mark_buffer_done = lambda _buffer_id: None

    def _fake_subsample(chunk_data, chunk_info):
        assert wait_state["ready"] is True
        assert int(chunk_data.shape[0]) == 3
        updated = dict(chunk_info)
        updated["local_size"] = 2
        updated["source"] = "mmap_shard_random"
        return chunk_data[:2], updated

    worker._subsample_loaded_chunk = _fake_subsample

    monkeypatch.setattr(
        batch_module,
        "find_bmus",
        lambda batch, *_args, **_kwargs: cp.zeros(int(batch.shape[0]), dtype=cp.int32),
    )
    monkeypatch.setattr(
        batch_module,
        "compute_weight_updates",
        lambda **kwargs: (
            cp.zeros_like(kwargs["weights"]),
            cp.zeros((kwargs["weights"].shape[0],), dtype=kwargs["weights"].dtype),
        ),
    )

    params = SimpleNamespace(
        weight_update_frequency=1,
        current_learning_rate=0.1,
        processing_config=SimpleNamespace(
            use_sparse_influence=False,
            distance_metric="euclidean",
            distance_metric_params={},
            normalization=None,
        ),
    )

    _, _, samples_processed = worker._process_all_chunks(params, topology_data={})

    assert samples_processed == 2
    assert wait_state["ready"] is True


def test_batch_worker_async_loader_uses_shard_local_random_fraction(monkeypatch):
    worker = _make_batch_worker()
    worker.async_mode = True
    worker.async_loader = None
    worker.sampling_method = "random"
    worker.sampling_fraction = 0.25
    worker._async_loader_sampling_fraction_override = None
    worker._async_loader_target_rows_override = None
    worker.async_loader_config = None
    worker._async_loader_context = {
        "config": SimpleNamespace(get_initial_chunks=lambda _num_chunks: 1),
        "data_ref": object(),
        "start_idx": 0,
        "end_idx": 64,
        "num_chunks": 2,
        "initial_chunks": 1,
    }
    worker.wait_for_initial_chunks = lambda _num_chunks: True

    captured = {}

    class _FakeAsyncLoader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start_loading(self, **kwargs):
            captured["start_loading"] = kwargs

    monkeypatch.setattr(worker_module, "AsyncChunkLoader", _FakeAsyncLoader, raising=True)

    worker._start_async_loader_with_order(np.asarray([0, 1], dtype=np.int32))

    assert captured["sampling_fraction"] == pytest.approx(0.25)
    assert captured["target_rows"] is None


def test_batch_worker_async_failure_raises_without_sync_fallback():
    worker = _make_batch_worker()
    worker.async_mode = True
    worker.sampling_method = "random"
    worker.sampling_fraction = 0.5
    worker.async_loader = SimpleNamespace(
        request_priority_chunks=lambda _chunks: None,
        get_chunk_when_ready=lambda _chunk_idx: None,
    )

    with pytest.raises(RuntimeError, match="Async loader failed to provide chunk 0"):
        worker._load_chunk_async_to_gpu(0)


def test_batch_worker_process_iteration_normalizes_by_allreduced_sample_count(monkeypatch):
    worker = _make_batch_worker()
    worker.collective_group = "test_group"
    worker._configure_async_loader_for_selected_mode = lambda _selected_mode: None
    worker._ensure_async_loader_started = lambda: None
    worker._process_all_chunks = (
        lambda _params, _topology_data, timings=None: (
            cp.zeros_like(worker.gpu_weights),
            cp.zeros(worker.gpu_weights.shape[0], dtype=worker.gpu_weights.dtype),
            3,
        )
    )
    worker._apply_nccl_reduction = lambda _updates, _influence: None
    worker._start_worker_profile = lambda *_args, **_kwargs: None
    worker.reset_for_iteration = lambda: None
    worker.iteration_count = 1

    captured: dict[str, int] = {}

    def _fake_finalize(_updates, _influence, _params, total_samples):
        captured["total_samples"] = int(total_samples)
        return cp.zeros_like(worker.gpu_weights)

    monkeypatch.setattr(
        batch_module.collective,
        "allreduce",
        lambda tensor, **_kwargs: tensor.__setitem__(slice(None), tensor * 2),
    )
    worker._finalize_weight_updates = _fake_finalize

    topology = SimpleNamespace(
        get_precomputed_influence_matrix=lambda *_args, **_kwargs: np.ones((3,), dtype=np.float32)
    )
    params = SimpleNamespace(
        processing_config=SimpleNamespace(use_sparse_influence=False),
        current_radius=1.0,
        current_learning_rate=0.1,
        current_momentum=0.0,
        delta_weights=None,
        total_samples=100,
    )

    result = worker.process_batch_iteration(
        som_weights=None,
        topology_data=topology,
        params=params,
        collective_group="test_group",
        selected_indices=None,
        is_first_iteration=False,
    )

    assert captured["total_samples"] == 6
    assert result["samples_processed"] == 3


def test_batch_worker_whole_chunk_random_routes_to_sampled_chunk_order(monkeypatch):
    worker = _make_batch_worker()
    worker.whole_chunk_random = True
    worker.collective_group = "test_group"
    worker._configure_async_loader_for_selected_mode = lambda _selected_mode: None
    worker._ensure_async_loader_started = lambda: None
    worker._build_whole_chunk_order = lambda _params: np.asarray([3, 1], dtype=np.int32)
    worker._apply_nccl_reduction = lambda _updates, _influence: None
    worker._start_worker_profile = lambda *_args, **_kwargs: None
    worker.reset_for_iteration = lambda: None
    worker.iteration_count = 1

    captured: dict[str, np.ndarray] = {}

    def _fake_process_all_chunks(_params, _topology_data, *, chunk_indices=None, timings=None):
        del timings
        captured["chunk_indices"] = np.asarray(chunk_indices, dtype=np.int32)
        return (
            cp.zeros_like(worker.gpu_weights),
            cp.zeros(worker.gpu_weights.shape[0], dtype=worker.gpu_weights.dtype),
            8,
        )

    worker._process_all_chunks = _fake_process_all_chunks

    monkeypatch.setattr(
        batch_module.collective,
        "allreduce",
        lambda tensor, **_kwargs: tensor,
    )
    worker._finalize_weight_updates = lambda _updates, _influence, _params, _total: cp.zeros_like(worker.gpu_weights)

    topology = SimpleNamespace(
        get_precomputed_influence_matrix=lambda *_args, **_kwargs: np.ones((3,), dtype=np.float32)
    )
    params = SimpleNamespace(
        sampling_config=SimpleNamespace(method="random", whole_chunk_random=True),
        processing_config=SimpleNamespace(use_sparse_influence=False, enable_multi_buffering=0),
        current_radius=1.0,
        current_learning_rate=0.1,
        current_momentum=0.0,
        delta_weights=None,
        total_samples=100,
    )

    result = worker.process_batch_iteration(
        som_weights=None,
        topology_data=topology,
        params=params,
        collective_group="test_group",
        selected_indices=None,
        is_first_iteration=False,
    )

    np.testing.assert_array_equal(captured["chunk_indices"], np.asarray([3, 1], dtype=np.int32))
    assert result["samples_processed"] == 8


def test_batch_worker_build_whole_chunk_order_rounds_up_without_replacement():
    worker = _make_batch_worker()
    worker.whole_chunk_random = True
    worker.loader_chunk_size = 4
    worker.data_start_idx = 0
    worker.data_end_idx = 20
    worker.n_samples = 20
    worker.sampling_fraction = 0.3
    worker.iteration_count = 7
    worker.get_num_chunks = lambda: 5

    params = SimpleNamespace(
        sampling_config=SimpleNamespace(method="random", whole_chunk_random=True, random_seed=13),
    )

    order = worker._build_whole_chunk_order(params)

    expected_rng = np.random.default_rng((int(worker.worker_id) << 48) ^ (7 << 16) ^ 13)
    expected = expected_rng.choice(5, size=2, replace=False).astype(np.int32, copy=False)

    assert order.dtype == np.int32
    assert len(order) == 2
    assert len(np.unique(order)) == 2
    np.testing.assert_array_equal(order, expected)


def test_batch_worker_build_whole_chunk_order_changes_with_iteration_seed():
    worker = _make_batch_worker()
    worker.whole_chunk_random = True
    worker.loader_chunk_size = 4
    worker.data_start_idx = 0
    worker.data_end_idx = 20
    worker.n_samples = 20
    worker.sampling_fraction = 0.3
    worker.get_num_chunks = lambda: 5

    params = SimpleNamespace(
        sampling_config=SimpleNamespace(method="random", whole_chunk_random=True, random_seed=13),
    )

    worker.iteration_count = 7
    order_iter7 = worker._build_whole_chunk_order(params)
    worker.iteration_count = 8
    order_iter8 = worker._build_whole_chunk_order(params)
    expected_iter7 = np.random.default_rng((int(worker.worker_id) << 48) ^ (7 << 16) ^ 13).choice(
        5,
        size=2,
        replace=False,
    ).astype(np.int32, copy=False)
    expected_iter8 = np.random.default_rng((int(worker.worker_id) << 48) ^ (8 << 16) ^ 13).choice(
        5,
        size=2,
        replace=False,
    ).astype(np.int32, copy=False)

    assert len(order_iter7) == len(order_iter8) == 2
    np.testing.assert_array_equal(order_iter7, expected_iter7)
    np.testing.assert_array_equal(order_iter8, expected_iter8)
    assert not np.array_equal(expected_iter7, expected_iter8)


def test_batch_worker_whole_chunk_random_async_path_only_restarts_for_subset(monkeypatch):
    worker = _make_batch_worker()
    worker.async_mode = True
    worker._async_loader_context = {"num_chunks": 5}
    worker.async_loader = SimpleNamespace()
    worker._current_async_order = tuple(np.arange(5, dtype=np.int32))
    worker._start_worker_profile = lambda *_args, **_kwargs: None
    worker._configure_async_loader_for_selected_mode = lambda _selected_mode: None
    worker._ensure_async_loader_started = lambda: None
    worker._apply_nccl_reduction = lambda _updates, _influence: None
    worker._finalize_weight_updates = lambda _updates, _influence, _params, _total: cp.zeros_like(worker.gpu_weights)
    worker.reset_for_iteration = lambda: None
    worker.iteration_count = 1

    restart_calls = []

    def _fake_ensure_async_loader_order(order, force_restart=False):
        restart_calls.append((np.asarray(order, dtype=np.int32), bool(force_restart)))

    worker._ensure_async_loader_order = _fake_ensure_async_loader_order

    def _fake_process_all_chunks(_params, _topology_data, *, chunk_indices=None, timings=None):
        del timings
        if chunk_indices is not None and worker._using_async_loader():
            worker._ensure_async_loader_order(chunk_indices, force_restart=True)
        return (
            cp.zeros_like(worker.gpu_weights),
            cp.zeros(worker.gpu_weights.shape[0], dtype=worker.gpu_weights.dtype),
            4,
        )

    worker._process_all_chunks = _fake_process_all_chunks

    monkeypatch.setattr(
        batch_module.collective,
        "allreduce",
        lambda tensor, **_kwargs: tensor,
    )

    topology = SimpleNamespace(
        get_precomputed_influence_matrix=lambda *_args, **_kwargs: np.ones((3,), dtype=np.float32)
    )
    base_params = SimpleNamespace(
        processing_config=SimpleNamespace(use_sparse_influence=False, enable_multi_buffering=0),
        current_radius=1.0,
        current_learning_rate=0.1,
        current_momentum=0.0,
        delta_weights=None,
        total_samples=100,
    )

    params_full = SimpleNamespace(
        **vars(base_params),
        sampling_config=SimpleNamespace(method="random", whole_chunk_random=False),
    )
    worker.process_batch_iteration(
        som_weights=None,
        topology_data=topology,
        params=params_full,
        collective_group="test_group",
        selected_indices=None,
        is_first_iteration=False,
    )
    assert restart_calls == []

    params_subset = SimpleNamespace(
        **vars(base_params),
        sampling_config=SimpleNamespace(method="random", whole_chunk_random=True),
    )
    worker.whole_chunk_random = True
    worker._build_whole_chunk_order = lambda _params: np.asarray([4, 2], dtype=np.int32)
    worker.process_batch_iteration(
        som_weights=None,
        topology_data=topology,
        params=params_subset,
        collective_group="test_group",
        selected_indices=None,
        is_first_iteration=False,
    )

    assert len(restart_calls) == 1
    np.testing.assert_array_equal(restart_calls[0][0], np.asarray([4, 2], dtype=np.int32))
    assert restart_calls[0][1] is True


def test_colors_build_color_order_is_seeded_per_partition():
    worker = _make_color_worker()
    worker.color_order_seed = 1234

    order_partition0 = worker._build_color_order(num_color_sets=6, partition_idx=0)
    order_partition0_repeat = worker._build_color_order(num_color_sets=6, partition_idx=0)
    order_partition1 = worker._build_color_order(num_color_sets=6, partition_idx=1)

    expected_partition0 = np.arange(6, dtype=np.int32)
    rng0 = np.random.default_rng(1234)
    rng0.shuffle(expected_partition0)

    expected_partition1 = np.arange(6, dtype=np.int32)
    rng1 = np.random.default_rng(1235)
    rng1.shuffle(expected_partition1)

    np.testing.assert_array_equal(order_partition0, expected_partition0)
    np.testing.assert_array_equal(order_partition0_repeat, expected_partition0)
    np.testing.assert_array_equal(order_partition1, expected_partition1)


def test_colors_build_color_order_handles_empty_and_single_color_set():
    worker = _make_color_worker()
    worker.color_order_seed = 42

    empty = worker._build_color_order(num_color_sets=0, partition_idx=0)
    single = worker._build_color_order(num_color_sets=1, partition_idx=0)

    assert empty.size == 0
    np.testing.assert_array_equal(single, np.array([0], dtype=np.int32))


def test_colors_generate_partitions_prefers_custom_order_over_loader():
    worker = _make_color_worker()
    worker.no_local_samples = False
    worker._selected_chunk_ids = None
    worker._custom_chunk_order = np.array([4, 1, 3, 0, 2], dtype=np.int32)

    class _Loader:
        def __init__(self):
            self.called = False

        def partition_chunks(self, _n_partitions, _seed):
            self.called = True
            return [np.array([99], dtype=np.int32)]

    loader = _Loader()
    worker.data_loader = loader

    partitions = worker._generate_partitions(n_partitions=2, seed=123)

    assert loader.called is False
    expected = np.array_split(worker._custom_chunk_order, 2)
    np.testing.assert_array_equal(partitions[0], expected[0].astype(np.int32, copy=False))
    np.testing.assert_array_equal(partitions[1], expected[1].astype(np.int32, copy=False))


def test_colors_setup_processing_applies_selected_first_loader_order():
    worker = _make_color_worker(worker_id=2)
    worker.device = nullcontext()
    worker.data_loader = None
    worker.async_mode = False
    worker._async_loader_context = None
    worker.async_loader = None
    worker.collective_group = None
    worker.color_sets = []
    worker._progress_tracker = None
    worker._progress_interval_s = 0.0
    worker._progress_last_report_t = 0.0
    worker._custom_chunk_order = None
    worker._selected_chunk_local_positions = {}
    worker.get_cpu_pool = lambda: None
    worker.initialize_gpu_weights = lambda *_args, **_kwargs: None

    def _init_local_metadata():
        worker.no_local_samples = False
        worker.total_local_samples = 20
        worker.num_chunks = 5

    worker._initialize_local_metadata = _init_local_metadata
    worker._configure_selected_indices = (
        lambda _selected: setattr(worker, "_selected_chunk_ids", np.array([1, 3], dtype=np.int32))
    )

    class _Loader:
        def __init__(self):
            self.orders = []

        def set_chunk_order(self, order):
            self.orders.append(np.asarray(order, dtype=np.int32).copy())

    loader = _Loader()
    worker.data_loader = loader

    params = SimpleNamespace(
        processing_config=SimpleNamespace(
            sample_order="strided",
            max_rounds=1,
        )
    )

    worker.setup_processing(
        som_weights=None,
        influence_matrix=np.ones((3, 3), dtype=np.float32),
        color_sets=[np.array([0, 1], dtype=np.int32)],
        collective_group_name="test-group",
        params=params,
        color_round_plan={"num_partitions": 1},
        selected_indices=np.array([0], dtype=np.int64),
    )

    assert len(loader.orders) == 1
    np.testing.assert_array_equal(loader.orders[0], np.array([1, 3, 0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(worker._custom_chunk_order, np.array([1, 3], dtype=np.int32))
    assert len(worker.partitioned_chunks) == 1
    np.testing.assert_array_equal(worker.partitioned_chunks[0], np.array([1, 3], dtype=np.int32))
