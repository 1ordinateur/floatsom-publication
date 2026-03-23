"""
Unit tests for Ray worker CPU reservations and staging parallelism.

These tests avoid launching a Ray cluster by monkeypatching Ray APIs used by
RayWorkerManager and by stubbing actor creation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.data.fast_array_store import FastArrayStore
from floatsom.floatsom_params import RayConfig
from floatsom.processing.ray_ops import ray_pipeline_base as manager_module
from floatsom.processing.ray_ops.workers import ray_pipeline_base_worker as worker_module


def _build_ray_config(shared_path: Path, local_path: Path) -> RayConfig:
    return RayConfig(
        chunk_size=1,
        storage_path=str(shared_path),
        local_storage_path=str(local_path),
        num_gpus=1,
    )


class _DummyWorkerActor:
    created = []
    _options = {}

    @classmethod
    def options(cls, **options):
        cls._options = dict(options)
        return cls

    @classmethod
    def remote(cls, **kwargs):
        cls.created.append((dict(cls._options), dict(kwargs)))
        return object()


def test_initialize_workers_reserves_cpus_per_node(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=4, ray_config=ray_config)

    monkeypatch.setattr(manager_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        manager_module.ray,
        "cluster_resources",
        lambda: {"CPU": 192, "GPU": 4},
    )
    monkeypatch.setattr(
        manager_module.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeManagerHostname": "node-a",
                "Resources": {"CPU": 96, "GPU": 2, "node:node-a": 1.0},
            },
            {
                "Alive": True,
                "NodeManagerHostname": "node-b",
                "Resources": {"CPU": 96, "GPU": 2, "node:node-b": 1.0},
            },
        ],
    )
    monkeypatch.setattr(
        manager,
        "_query_node_cpu_affinity",
        lambda node_key: {"node:node-a": 9, "node:node-b": 5}.get(node_key),
    )
    monkeypatch.setattr(manager, "_initialize_collective_group", lambda: None)

    _DummyWorkerActor.created = []
    workers = manager.initialize_workers(_DummyWorkerActor, chunk_size=1)
    assert len(workers) == 4
    assert len(_DummyWorkerActor.created) == 4

    node_a = [
        options
        for options, _
        in _DummyWorkerActor.created
        if "resources" in options and "node:node-a" in options["resources"]
    ]
    node_b = [
        options
        for options, _
        in _DummyWorkerActor.created
        if "resources" in options and "node:node-b" in options["resources"]
    ]

    assert len(node_a) == 2
    assert len(node_b) == 2
    assert {opt["num_cpus"] for opt in node_a} == {4}
    assert {opt["num_cpus"] for opt in node_b} == {2}
    ordered_nodes = [
        "node-a" if "node:node-a" in options.get("resources", {}) else "node-b"
        for options, _ in _DummyWorkerActor.created
    ]
    assert ordered_nodes == ["node-a", "node-a", "node-b", "node-b"]


def test_create_local_fast_array_uses_thread_pool_when_multiple_cpus(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")

    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.ray_config = ray_config
    worker.worker_id = 0
    worker.loader_chunk_size = 16

    monkeypatch.setattr(worker_module.socket, "gethostname", lambda: "node-a")
    monkeypatch.setattr(worker, "_initialize_data_loader", lambda: None)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)

    source_path = tmp_path / "source.fast"
    source_store = FastArrayStore(str(source_path), mode="w")
    source_data = np.arange(120, dtype=np.float32).reshape(40, 3)
    source_arr = source_store.create(shape=source_data.shape, dtype=np.float32, chunks=(16, 3))
    source_arr[:] = source_data
    source_store.close()

    created_executors = []

    class _RecordingExecutor:
        def __init__(self, max_workers=None, thread_name_prefix=None):
            self.max_workers = max_workers
            self.thread_name_prefix = thread_name_prefix
            created_executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            for item in iterable:
                fn(item)
                yield None

    monkeypatch.setattr(worker_module.concurrent.futures, "ThreadPoolExecutor", _RecordingExecutor)

    meta = worker.create_local_fast_array(
        source_path=str(source_path),
        start_idx=5,
        end_idx=29,
        shard_id=0,
    )

    assert created_executors, "Expected ThreadPoolExecutor to be used for staging copy"
    assert created_executors[0].max_workers == 4

    local_store = FastArrayStore(meta["path"], mode="r")
    np.testing.assert_array_equal(local_store.mmap_array, source_data[5:29])
    local_store.close()


class _RemoteCallable:
    def __init__(self, value):
        self._value = value

    def remote(self):
        return self._value


class _ProgressWorker:
    def __init__(self):
        self.get_loading_progress = _RemoteCallable(object())


class _RecordingRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}


class _RecordingRamWorker:
    def __init__(self):
        self.use_ram_data = _RecordingRemoteMethod()
        self.use_ram_data_async = _RecordingRemoteMethod()
        self.use_node_staged_ram_shard = _RecordingRemoteMethod()
        self.use_node_staged_ram_shard_async = _RecordingRemoteMethod()
        self.use_local_node_staged_ram_shard = _RecordingRemoteMethod()
        self.use_local_node_staged_ram_shard_async = _RecordingRemoteMethod()
        self.stage_zarr_node_span = _RecordingRemoteMethod()
        self.clear_node_staged_shards = _RecordingRemoteMethod()


def test_wait_for_all_data_succeeds_before_async_loaders_start(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=2, ray_config=ray_config)
    manager.async_loading_active = True
    manager.distribution_futures = [object()]
    manager.workers = [_ProgressWorker(), _ProgressWorker()]

    ensure_calls = {"count": 0}

    def _fake_ensure_distribution_ready(timeout=None):
        ensure_calls["count"] += 1
        manager.distribution_futures = None
        return True

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        assert phase == "async_loading_progress"
        return [
            {"status": "async_configured_not_started", "ready_count": 0, "total_chunks": 4},
            {"status": "async_configured_not_started", "ready_count": 0, "total_chunks": 5},
        ]

    monkeypatch.setattr(manager, "ensure_distribution_ready", _fake_ensure_distribution_ready)
    monkeypatch.setattr(manager, "wait_for_worker_futures", _fake_wait_for_worker_futures)
    monkeypatch.setattr(manager_module.time, "sleep", lambda *_args, **_kwargs: None)

    assert manager.wait_for_all_data(timeout=2.0) is True
    assert ensure_calls["count"] == 1
    assert manager.async_loading_active is True


def test_wait_for_all_data_marks_complete_when_chunks_ready(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=1, ray_config=ray_config)
    manager.async_loading_active = True
    manager.distribution_futures = None
    manager.workers = [_ProgressWorker()]

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        assert phase == "async_loading_progress"
        return [{"status": "loading", "ready_count": 3, "total_chunks": 3}]

    monkeypatch.setattr(manager, "wait_for_worker_futures", _fake_wait_for_worker_futures)

    assert manager.wait_for_all_data(timeout=2.0) is True
    assert manager.async_loading_active is False


def test_group_worker_spans_by_node_uses_worker_placement_plan(tmp_path):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=4, ray_config=ray_config)
    manager.workers = [object(), object(), object(), object()]
    manager.worker_placement_plan = [
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
    ]

    groups = manager._group_worker_spans_by_node(20)

    assert len(groups) == 2
    assert groups[0]["hostname"] == "node-a"
    assert groups[0]["node_start"] == 0
    assert groups[0]["node_end"] == 10
    assert groups[0]["workers"] == [
        {"worker_idx": 0, "start": 0, "end": 5, "local_start": 0, "local_end": 5},
        {"worker_idx": 1, "start": 5, "end": 10, "local_start": 5, "local_end": 10},
    ]
    assert groups[1]["hostname"] == "node-b"
    assert groups[1]["node_start"] == 10
    assert groups[1]["node_end"] == 20
    assert groups[0]["leader_worker_idx"] == 0
    assert groups[1]["leader_worker_idx"] == 2


def test_group_worker_spans_by_node_rejects_interleaved_node_placement(tmp_path):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=4, ray_config=ray_config)
    manager.workers = [object(), object(), object(), object()]
    manager.worker_placement_plan = [
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
    ]

    with pytest.raises(RuntimeError, match="contiguous worker placement per node"):
        manager._group_worker_spans_by_node(20)


def test_distribute_zarr_to_node_ram_shards_sync_groups_by_node(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    async_config = SimpleNamespace(
        should_use_async=lambda _gb: False,
        enable_progress_reporting=False,
        progress_interval=0.1,
        auto_disable_threshold_gb=0,
    )
    manager = manager_module.RayWorkerManager(
        num_gpus=4,
        ray_config=ray_config,
        processing_config=SimpleNamespace(async_loading_config=async_config),
    )
    workers = [_RecordingRamWorker() for _ in range(4)]
    manager.workers = workers
    manager.n_features = 3
    manager.worker_placement_plan = [
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
    ]

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        if phase == "zarr_node_ram_stage":
            return [
                {"hostname": "node-a", "workers_staged": 2, "copy_workers_used": 2},
                {"hostname": "node-b", "workers_staged": 2, "copy_workers_used": 2},
            ]
        if phase == "zarr_node_ram_distribution":
            return [{"worker_id": idx} for idx, _future in enumerate(futures)]
        if phase == "node_ram_stage_cleanup":
            return [True for _future in futures]
        raise AssertionError(f"Unexpected phase {phase}")

    monkeypatch.setattr(manager, "wait_for_worker_futures", _fake_wait_for_worker_futures)

    details = manager._distribute_zarr_to_node_ram_shards(str(tmp_path / "fake.zarr"), n_samples=20)

    assert details["data_staging_strategy"] == "zarr_to_node_local_ram_shards"
    assert details["data_staging_nodes"] == 2
    assert details["data_staging_workers"] == 4
    assert workers[0].stage_zarr_node_span.calls[0][0][0] == str(tmp_path / "fake.zarr")
    assert workers[2].stage_zarr_node_span.calls[0][0][0] == str(tmp_path / "fake.zarr")
    assert workers[0].use_local_node_staged_ram_shard.calls[0][0] == (0, 0, 5)
    assert workers[1].use_node_staged_ram_shard.calls[0][0][1:] == (1, 5, 10)
    assert workers[2].use_local_node_staged_ram_shard.calls[0][0] == (2, 10, 15)
    assert workers[3].use_node_staged_ram_shard.calls[0][0][1:] == (3, 15, 20)
    assert workers[0].clear_node_staged_shards.calls
    assert workers[2].clear_node_staged_shards.calls


def test_distribute_zarr_to_node_ram_shards_async_cleans_leaders_on_ready(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    async_config = SimpleNamespace(
        should_use_async=lambda _gb: True,
        enable_progress_reporting=False,
        progress_interval=0.1,
        auto_disable_threshold_gb=0,
    )
    manager = manager_module.RayWorkerManager(
        num_gpus=4,
        ray_config=ray_config,
        processing_config=SimpleNamespace(async_loading_config=async_config),
    )
    workers = [_RecordingRamWorker() for _ in range(4)]
    manager.workers = workers
    manager.n_features = 3
    manager.worker_placement_plan = [
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-a", "hostname": "node-a", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
        {"node_resource_key": "node:node-b", "hostname": "node-b", "num_cpus": 4},
    ]

    call_counts = {"async_distribution_setup": 0}

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        if phase == "zarr_node_ram_stage":
            return [
                {"hostname": "node-a", "workers_staged": 2, "copy_workers_used": 2},
                {"hostname": "node-b", "workers_staged": 2, "copy_workers_used": 2},
            ]
        if phase == "async_distribution_setup":
            call_counts["async_distribution_setup"] += 1
            return [{"worker_id": idx} for idx, _future in enumerate(futures)]
        if phase == "node_ram_stage_cleanup":
            return [True for _future in futures]
        raise AssertionError(f"Unexpected phase {phase}")

    monkeypatch.setattr(manager, "wait_for_worker_futures", _fake_wait_for_worker_futures)

    details = manager._distribute_zarr_to_node_ram_shards(str(tmp_path / "fake.zarr"), n_samples=20)

    assert details["data_staging_async"] is True
    assert manager.data_distributed is False
    assert manager._pending_node_ram_stage_leaders == [workers[0], workers[2]]

    assert manager.ensure_distribution_ready(timeout=2.0) is True
    assert call_counts["async_distribution_setup"] == 1
    assert workers[0].use_local_node_staged_ram_shard_async.calls[0][0] == (0, 0, 5)
    assert workers[2].use_local_node_staged_ram_shard_async.calls[0][0] == (2, 10, 15)
    assert workers[0].clear_node_staged_shards.calls
    assert workers[2].clear_node_staged_shards.calls
    assert manager._pending_node_ram_stage_leaders == []


def test_ensure_distribution_ready_async_failure_cleans_leaders(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=2, ray_config=ray_config)
    leaders = [_RecordingRamWorker(), _RecordingRamWorker()]
    manager._pending_node_ram_stage_leaders = leaders
    manager.distribution_futures = [object(), object()]
    manager.total_samples = 10

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        if phase == "async_distribution_setup":
            raise RuntimeError("synthetic async setup failure")
        if phase == "node_ram_stage_cleanup":
            return [True for _future in futures]
        raise AssertionError(f"Unexpected phase {phase}")

    monkeypatch.setattr(manager, "wait_for_worker_futures", _fake_wait_for_worker_futures)

    with pytest.raises(RuntimeError, match="synthetic async setup failure"):
        manager.ensure_distribution_ready(timeout=1.0)

    assert leaders[0].clear_node_staged_shards.calls
    assert leaders[1].clear_node_staged_shards.calls
    assert manager._pending_node_ram_stage_leaders == []


def test_cleanup_pending_node_ram_stage_leaders_suppresses_partial_dispatch_failures(tmp_path):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=1, ray_config=ray_config)

    class _ExplodingRemote:
        def remote(self):
            raise RuntimeError("synthetic cleanup dispatch failure")

    successful_calls = []

    class _SuccessfulRemote:
        def remote(self):
            successful_calls.append("dispatched")
            return {"cleared": True}

    failing_leader = SimpleNamespace(clear_node_staged_shards=_ExplodingRemote())
    successful_leader = SimpleNamespace(clear_node_staged_shards=_SuccessfulRemote())
    manager._pending_node_ram_stage_leaders = [failing_leader, successful_leader]

    wait_phases = []

    def _fake_wait_for_worker_futures(futures, *, phase, timeout_s=None, poll_interval_s=1.0):
        wait_phases.append(phase)
        return [True for _future in futures]

    manager.wait_for_worker_futures = _fake_wait_for_worker_futures

    manager._cleanup_pending_node_ram_stage_leaders(suppress_exceptions=True)

    assert successful_calls == ["dispatched"]
    assert wait_phases == ["node_ram_stage_cleanup"]
    assert manager._pending_node_ram_stage_leaders == [failing_leader]


def test_cleanup_clears_pending_node_stage_leaders_before_worker_teardown(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=1, ray_config=ray_config)

    class _RemoteCall:
        def __init__(self, owner, name):
            self._owner = owner
            self._name = name

        def remote(self, *args, **kwargs):
            self._owner.append(self._name)
            return self._name

    call_order = []
    leader = SimpleNamespace(clear_node_staged_shards=_RemoteCall(call_order, "leader_cleanup"))
    worker = SimpleNamespace(cleanup=_RemoteCall(call_order, "worker_cleanup"))
    manager._pending_node_ram_stage_leaders = [leader]
    manager.workers = [worker]

    monkeypatch.setattr(manager_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(manager_module.collective, "destroy_collective_group", lambda group_name: None)
    monkeypatch.setattr(manager_module.ray, "get", lambda futures, timeout=None: futures)
    monkeypatch.setattr(manager_module.ray, "kill", lambda worker_handle: call_order.append("worker_kill"))
    monkeypatch.setattr(
        manager,
        "wait_for_worker_futures",
        lambda futures, *, phase, timeout_s=None, poll_interval_s=1.0: [True for _future in futures],
    )

    manager.cleanup(shutdown_ray=False)

    assert call_order == ["leader_cleanup", "worker_cleanup", "worker_kill"]
    assert manager._pending_node_ram_stage_leaders == []


def test_cleanup_drops_stale_pending_leaders_after_suppressed_failure(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=1, ray_config=ray_config)

    class _ExplodingRemote:
        def remote(self, *args, **kwargs):
            raise RuntimeError("synthetic cleanup dispatch failure")

    class _WorkerRemote:
        def remote(self, *args, **kwargs):
            return True

    leader = SimpleNamespace(clear_node_staged_shards=_ExplodingRemote())
    worker = SimpleNamespace(cleanup=_WorkerRemote())
    manager._pending_node_ram_stage_leaders = [leader]
    manager.workers = [worker]

    monkeypatch.setattr(manager_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(manager_module.ray, "get", lambda futures, timeout=None: futures)
    monkeypatch.setattr(manager_module.ray, "kill", lambda worker_handle: None)

    manager.cleanup(shutdown_ray=False)

    assert manager._pending_node_ram_stage_leaders == []


def test_distribute_fast_array_once_zarr_ram_branches_by_node_count(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=2, ray_config=ray_config)
    manager.workers = [object(), object()]

    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    fake_zarr = _FakeZarrArray(data, chunks=(2, 3))
    monkeypatch.setattr(
        manager_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=fake_zarr),
    )
    monkeypatch.setattr(manager, "_should_use_ram_mode", lambda shape: (True, "fits"))

    branch_calls = []

    monkeypatch.setattr(
        manager,
        "_distribute_ram_shards",
        lambda array: branch_calls.append(("single", tuple(array.shape))),
    )
    monkeypatch.setattr(
        manager,
        "_distribute_zarr_to_node_ram_shards",
        lambda path, *, n_samples: branch_calls.append(("multi", path, n_samples)) or {
            "data_staging_strategy": "zarr_to_node_local_ram_shards"
        },
    )

    source_info = {
        "type": "file",
        "path": str(tmp_path / "fake.zarr"),
        "format": "zarr",
        "n_samples": 12,
        "n_features": 3,
    }

    monkeypatch.setattr(manager, "_estimate_workers_on_node", lambda _n_workers: 2)
    single_details = manager._distribute_fast_array_once(dict(source_info))
    assert branch_calls[0] == ("single", (12, 3))
    assert single_details["data_staging_strategy"] == "zarr_to_parallel_worker_ram_shards"

    monkeypatch.setattr(manager, "_estimate_workers_on_node", lambda _n_workers: 1)
    multi_details = manager._distribute_fast_array_once(dict(source_info))
    assert branch_calls[1] == ("multi", str(tmp_path / "fake.zarr"), 12)
    assert multi_details["data_staging_strategy"] == "zarr_to_node_local_ram_shards"


def test_create_parallel_ram_shard_refs_splits_data_by_worker(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=3, ray_config=ray_config)
    manager.workers = [object(), object(), object()]

    put_payloads = []

    def _fake_put(value):
        arr = np.array(value, copy=True)
        put_payloads.append(arr)
        return arr

    monkeypatch.setattr(manager_module.ray, "put", _fake_put)

    data = np.arange(60, dtype=np.float32).reshape(20, 3)
    specs = manager._create_parallel_ram_shard_refs(data)

    assert [(spec["start"], spec["end"]) for spec in specs] == [(0, 6), (6, 12), (12, 20)]
    assert len(put_payloads) == 3
    np.testing.assert_array_equal(specs[0]["ref"], data[0:6])
    np.testing.assert_array_equal(specs[1]["ref"], data[6:12])
    np.testing.assert_array_equal(specs[2]["ref"], data[12:20])


def test_distribute_ram_shards_sync_passes_worker_local_shards(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    manager = manager_module.RayWorkerManager(num_gpus=2, ray_config=ray_config)
    workers = [_RecordingRamWorker(), _RecordingRamWorker()]
    manager.workers = workers

    shard_specs = [
        {"worker_idx": 0, "start": 0, "end": 4, "ref": "shard-0"},
        {"worker_idx": 1, "start": 4, "end": 8, "ref": "shard-1"},
    ]
    monkeypatch.setattr(manager, "_create_parallel_ram_shard_refs", lambda data: shard_specs)
    monkeypatch.setattr(
        manager,
        "wait_for_worker_futures",
        lambda futures, **kwargs: [{"worker_id": 0}, {"worker_id": 1}],
    )

    manager._distribute_ram_shards(np.zeros((8, 2), dtype=np.float32))

    first_args, first_kwargs = workers[0].use_ram_data.calls[0]
    second_args, second_kwargs = workers[1].use_ram_data.calls[0]
    assert first_args == ("shard-0", 0, 4)
    assert second_args == ("shard-1", 4, 8)
    assert first_kwargs == {"worker_shard": True}
    assert second_kwargs == {"worker_shard": True}
    assert manager.data_distributed is True


def test_distribute_ram_shards_async_passes_worker_local_shards(tmp_path, monkeypatch):
    ray_config = _build_ray_config(tmp_path / "shared", tmp_path / "local")
    async_config = SimpleNamespace(
        should_use_async=lambda _gb: True,
        enable_progress_reporting=False,
        progress_interval=0.1,
        auto_disable_threshold_gb=0,
    )
    manager = manager_module.RayWorkerManager(
        num_gpus=2,
        ray_config=ray_config,
        processing_config=SimpleNamespace(async_loading_config=async_config),
    )
    workers = [_RecordingRamWorker(), _RecordingRamWorker()]
    manager.workers = workers

    shard_specs = [
        {"worker_idx": 0, "start": 0, "end": 4, "ref": "shard-0"},
        {"worker_idx": 1, "start": 4, "end": 8, "ref": "shard-1"},
    ]
    monkeypatch.setattr(manager, "_create_parallel_ram_shard_refs", lambda data: shard_specs)

    manager._distribute_ram_shards(np.zeros((8, 2), dtype=np.float32))

    first_args, first_kwargs = workers[0].use_ram_data_async.calls[0]
    second_args, second_kwargs = workers[1].use_ram_data_async.calls[0]
    assert first_args == ("shard-0", 0, 4)
    assert second_args == ("shard-1", 4, 8)
    assert first_kwargs["worker_shard"] is True
    assert second_kwargs["worker_shard"] is True
    assert manager.async_loading_active is True
    assert manager.data_distributed is False


def test_worker_progress_reports_async_not_started_status():
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 7
    worker.async_mode = True
    worker._async_loader_context = {"num_chunks": 11}
    worker.async_loader = None

    progress = worker.get_loading_progress()

    assert progress["status"] == "async_configured_not_started"
    assert progress["started"] is False
    assert progress["total_chunks"] == 11


class _DummyRamLoader:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def close(self):
        return None


class _FakeZarrArray:
    def __init__(self, data: np.ndarray, *, chunks=None, fail_from_row: int | None = None):
        self._data = np.asarray(data, dtype=np.float32)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks if chunks is not None else self.shape
        self.fail_from_row = fail_from_row

    def _resolve_rows(self, selection):
        if isinstance(selection, tuple):
            row_sel = selection[0]
        else:
            row_sel = selection
        start = 0 if row_sel.start is None else int(row_sel.start)
        stop = self.shape[0] if row_sel.stop is None else int(row_sel.stop)
        return start, stop

    def get_basic_selection(self, selection, out=None):
        start, stop = self._resolve_rows(selection)
        if self.fail_from_row is not None and start >= int(self.fail_from_row):
            raise RuntimeError("synthetic zarr read failure")
        data = self._data[selection]
        if out is None:
            return np.array(data, copy=True)
        out[...] = data
        return out

    def __getitem__(self, selection):
        start, stop = self._resolve_rows(selection)
        if self.fail_from_row is not None and start >= int(self.fail_from_row):
            raise RuntimeError("synthetic zarr read failure")
        return np.array(self._data[selection], copy=True)


def _build_ram_worker(monkeypatch):
    worker = worker_module.RayPipelineBaseWorker.__new__(worker_module.RayPipelineBaseWorker)
    worker.worker_id = 0
    worker.loader_chunk_size = 8
    worker.chunk_size = 8
    worker.randomize_chunk_order = True
    worker.async_loader = None
    worker._async_loader_context = None
    worker._current_async_order = None
    worker.async_mode = False
    worker.data_loader = None
    worker.ram_data = None
    worker.data_path = None
    worker.data_start_idx = 0
    worker.data_end_idx = 0
    worker.ram_mode = False
    worker._loader_initialized = False
    monkeypatch.setattr(worker_module, "CPUGPUFastLoader", _DummyRamLoader)
    monkeypatch.setattr(worker_module.socket, "gethostname", lambda: "node-a")
    return worker


def test_use_zarr_ram_slice_loads_expected_data(tmp_path, monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)
    monkeypatch.setattr(worker, "_detect_job_memory_limit_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(worker, "_detect_cgroup_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(worker, "_detect_gpu_workers_on_node", lambda: 1)
    monkeypatch.setattr(worker, "_resolve_zarr_ram_block_target_bytes", lambda: 24)

    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    fake_zarr = _FakeZarrArray(data, chunks=(2, 3))
    monkeypatch.setattr(
        worker_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=fake_zarr),
    )

    meta = worker.use_zarr_ram_slice(str(tmp_path / "fake.zarr"), 0, 12)

    np.testing.assert_array_equal(worker.ram_data, data)
    assert meta["copy_workers_used"] == 1
    assert worker._loader_initialized is True
    assert isinstance(worker.data_loader, _DummyRamLoader)


def test_use_ram_data_worker_shard_preserves_global_span(monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    shard = np.arange(24, dtype=np.float32).reshape(8, 3)

    meta = worker.use_ram_data(shard, 12, 20, worker_shard=True)

    np.testing.assert_array_equal(worker.ram_data, shard)
    assert worker.data_start_idx == 12
    assert worker.data_end_idx == 20
    assert worker._ram_data_is_worker_shard is True
    assert meta["start_idx"] == 12
    assert meta["end_idx"] == 20


def test_use_ram_data_async_worker_shard_uses_local_loader_span(monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    worker.device = SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)
    shard = np.arange(24, dtype=np.float32).reshape(8, 3)
    async_config = SimpleNamespace(get_initial_chunks=lambda num_chunks: min(2, num_chunks))
    monkeypatch.setattr(worker, "_tune_async_loader_config", lambda config, num_chunks: None)

    meta = worker.use_ram_data_async(shard, 12, 20, async_config=async_config, worker_shard=True)

    assert worker.data_start_idx == 12
    assert worker.data_end_idx == 20
    assert worker._ram_data_is_worker_shard is True
    assert worker._async_loader_context["start_idx"] == 0
    assert worker._async_loader_context["end_idx"] == 8
    assert meta["n_samples"] == 8


def test_stage_zarr_node_span_raises_when_headroom_too_small(tmp_path, monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)
    monkeypatch.setattr(worker, "_detect_job_memory_limit_bytes", lambda: 1024)
    monkeypatch.setattr(worker, "_detect_cgroup_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(worker, "_detect_gpu_workers_on_node", lambda: 1)

    data = np.arange(60, dtype=np.float32).reshape(20, 3)
    fake_zarr = _FakeZarrArray(data, chunks=(2, 3))
    monkeypatch.setattr(
        worker_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=fake_zarr),
    )

    with pytest.raises(RuntimeError, match="Insufficient memory headroom for node-local RAM staging"):
        worker.stage_zarr_node_span(
            str(tmp_path / "fake.zarr"),
            0,
            20,
            [
                {"worker_idx": 0, "start": 0, "end": 10, "local_start": 0, "local_end": 10},
                {"worker_idx": 1, "start": 10, "end": 20, "local_start": 10, "local_end": 20},
            ],
        )


def test_use_zarr_ram_slice_stays_single_threaded_under_headroom_cap(tmp_path, monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)
    monkeypatch.setattr(worker, "_detect_job_memory_limit_bytes", lambda: 600 * 1024**2)
    monkeypatch.setattr(worker, "_detect_cgroup_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(worker, "_detect_gpu_workers_on_node", lambda: 1)
    monkeypatch.setattr(worker, "_resolve_zarr_ram_block_target_bytes", lambda: 24)

    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    fake_zarr = _FakeZarrArray(data, chunks=(2, 3))
    monkeypatch.setattr(
        worker_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=fake_zarr),
    )

    meta = worker.use_zarr_ram_slice(str(tmp_path / "fake.zarr"), 0, 12)

    np.testing.assert_array_equal(worker.ram_data, data)
    assert meta["copy_workers_used"] == 1


def test_use_zarr_ram_slice_raises_when_headroom_is_zero(tmp_path, monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)
    monkeypatch.setattr(worker, "_detect_job_memory_limit_bytes", lambda: 1024)
    monkeypatch.setattr(worker, "_detect_cgroup_memory_usage_bytes", lambda: 1024)
    monkeypatch.setattr(worker, "_detect_gpu_workers_on_node", lambda: 1)
    monkeypatch.setattr(worker, "_resolve_zarr_ram_block_target_bytes", lambda: 24)

    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    fake_zarr = _FakeZarrArray(data, chunks=(2, 3))
    monkeypatch.setattr(
        worker_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=fake_zarr),
    )

    with pytest.raises(RuntimeError, match="per_worker_headroom=0 bytes"):
        worker.use_zarr_ram_slice(str(tmp_path / "fake.zarr"), 0, 12)

    assert worker.data_loader is None
    assert worker._loader_initialized is False
    assert worker.ram_data is None


def test_use_zarr_ram_slice_propagates_read_failures(tmp_path, monkeypatch):
    worker = _build_ram_worker(monkeypatch)
    monkeypatch.setattr(worker, "_detect_assigned_cpu_count", lambda: 4)
    monkeypatch.setattr(worker, "_detect_job_memory_limit_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(worker, "_detect_cgroup_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(worker, "_detect_gpu_workers_on_node", lambda: 1)
    monkeypatch.setattr(worker, "_resolve_zarr_ram_block_target_bytes", lambda: 24)

    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    failing_zarr = _FakeZarrArray(data, chunks=(2, 3), fail_from_row=4)
    monkeypatch.setattr(
        worker_module,
        "open_array_read",
        lambda path: SimpleNamespace(array=failing_zarr),
    )

    with pytest.raises(RuntimeError, match="synthetic zarr read failure"):
        worker.use_zarr_ram_slice(str(tmp_path / "fake.zarr"), 0, 12)

    assert worker.data_loader is None
    assert worker._loader_initialized is False
    assert worker.ram_data is None
