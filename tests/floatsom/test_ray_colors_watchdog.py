"""
Regression tests for RayColorsProcessor iteration watchdog retry behaviour.

These tests avoid launching a Ray cluster by monkeypatching Ray APIs used by
RayColorsProcessor and by stubbing worker actors.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.floatsom_params import ProcessingConfig, RayConfig
from floatsom.processing.processing_params import ColorsConfig
from floatsom.processing.ray_ops.ray_colors_processor import RayColorsProcessor


class _DummyRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _DummyWorker:
    def __init__(self) -> None:
        self.setup_calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self.setup_processing = _DummyRemoteMethod(self._setup_processing)
        self.process_full_iteration = _DummyRemoteMethod(self._process_full_iteration)

    def _setup_processing(self, *args, **kwargs):
        self.setup_calls.append((args, dict(kwargs)))
        return object()

    def _process_full_iteration(self):
        return object()


class _DummyProgressTracker:
    def __init__(self) -> None:
        self.reset = _DummyRemoteMethod(lambda: object())


class _SnapshotProgressTracker:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.snapshot = _DummyRemoteMethod(self._snapshot)

    def _snapshot(self):
        return dict(self._payload)


@dataclass
class _DummyWorkerManager:
    workers: List[_DummyWorker]

    def get_final_weights(self):
        return np.zeros((1, 1), dtype=np.float32), None


@dataclass
class _RecoveryWorkerManager:
    cleanup_should_raise: bool = False
    cleanup_calls: List[bool] = None
    initialize_calls: List[Tuple[Any, Any, Any, Any]] = None

    def __post_init__(self):
        self.cleanup_calls = []
        self.initialize_calls = []

    def cleanup(self, shutdown_ray: bool = True):
        self.cleanup_calls.append(bool(shutdown_ray))
        if self.cleanup_should_raise:
            raise RuntimeError("simulated cleanup failure")

    def initialize(self, som_weights, topology, params, data_source):
        self.initialize_calls.append((som_weights, topology, params, data_source))

    def get_final_weights(self):
        return np.zeros((1, 1), dtype=np.float32), None


def _build_processor(tmp_path, *, iteration_timeout_s: float, iteration_timeout_max_retries: int):
    ray_config = RayConfig(
        chunk_size=1,
        storage_path=str(tmp_path),
        local_storage_path=str(tmp_path),
        num_gpus=1,
        iteration_timeout_s=iteration_timeout_s,
        iteration_timeout_max_retries=iteration_timeout_max_retries,
    )
    processing_config = ProcessingConfig(method="colors", chunk_size=1, ray_config=ray_config)
    colors_config = ColorsConfig(chunk_size=1, max_rounds=1, sample_order="strided")
    return RayColorsProcessor(colors_config, ray_config, processing_config)


def _build_params():
    return SimpleNamespace(
        processing_config=SimpleNamespace(max_rounds=1, sample_order="strided", random_seed=None),
        delta_weights=None,
        current_momentum=0.0,
    )


def test_colors_watchdog_retries_on_timeout(tmp_path, monkeypatch):
    # Import the module under test so we can monkeypatch its `ray` handle.
    from floatsom.processing.ray_ops import ray_colors_processor as module

    def fake_ray_get(refs, timeout: Optional[float] = None):
        if isinstance(refs, list):
            return [None for _ in refs]
        return None

    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    calls = {"recoveries": 0, "awaits": 0}

    def fake_recover(self, topology, params) -> None:
        calls["recoveries"] += 1

    def fake_await(self, iteration_futures, stall_timeout_s: float, progress_tracker):
        calls["awaits"] += 1
        if calls["awaits"] == 1:
            raise TimeoutError("simulated stall")
        return [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 1,
                "weight_change_norm": 0.0,
            }
        ]

    monkeypatch.setattr(RayColorsProcessor, "_recover_from_iteration_timeout", fake_recover)
    monkeypatch.setattr(RayColorsProcessor, "_await_iteration_with_watchdog", fake_await)
    monkeypatch.setattr(RayColorsProcessor, "_ensure_progress_tracker", lambda self: _DummyProgressTracker())

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=60.0,
        iteration_timeout_max_retries=1,
    )

    dummy_workers = [_DummyWorker()]
    processor.worker_manager = _DummyWorkerManager(workers=dummy_workers)

    processor.verbose = False
    processor.color_sets = [np.array([0], dtype=np.int32)]
    processor.influence_matrix = np.zeros((1, 1), dtype=np.float32)
    processor._last_good_weights_cpu = np.zeros((1, 1), dtype=np.float32)

    params = _build_params()

    result = processor._process_samples(None, np.zeros((1, 1), dtype=np.float32), object(), params)

    assert calls["recoveries"] == 1
    assert result["update_type"] == "inplace"
    assert result["samples_processed"] == 1
    assert dummy_workers[0].setup_calls
    assert dummy_workers[0].setup_calls[0][1].get("progress_tracker") is not None


def test_colors_watchdog_retries_on_worker_failure(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    def fake_ray_get(refs, timeout: Optional[float] = None):
        if isinstance(refs, list):
            return [None for _ in refs]
        return None

    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    calls = {"recoveries": 0, "awaits": 0}

    def fake_recover(self, topology, params) -> None:
        calls["recoveries"] += 1

    def fake_await(self, iteration_futures, stall_timeout_s: float, progress_tracker):
        calls["awaits"] += 1
        if calls["awaits"] == 1:
            raise RuntimeError("simulated worker crash")
        return [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 1,
                "weight_change_norm": 0.0,
            }
        ]

    monkeypatch.setattr(RayColorsProcessor, "_recover_from_iteration_timeout", fake_recover)
    monkeypatch.setattr(RayColorsProcessor, "_await_iteration_with_watchdog", fake_await)
    monkeypatch.setattr(RayColorsProcessor, "_ensure_progress_tracker", lambda self: _DummyProgressTracker())

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=60.0,
        iteration_timeout_max_retries=1,
    )
    processor.worker_manager = _DummyWorkerManager(workers=[_DummyWorker()])
    processor.verbose = False
    processor.color_sets = [np.array([0], dtype=np.int32)]
    processor.influence_matrix = np.zeros((1, 1), dtype=np.float32)
    processor._last_good_weights_cpu = np.zeros((1, 1), dtype=np.float32)

    result = processor._process_samples(None, np.zeros((1, 1), dtype=np.float32), object(), _build_params())

    assert calls["recoveries"] == 1
    assert calls["awaits"] == 2
    assert result["samples_processed"] == 1


def test_colors_watchdog_uses_default_timeout_when_non_positive(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    def fake_ray_get(refs, timeout: Optional[float] = None):
        if isinstance(refs, list):
            return [None for _ in refs]
        return None

    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    captured = {"timeout_s": None}

    def fake_await(self, iteration_futures, stall_timeout_s: float, progress_tracker):
        captured["timeout_s"] = stall_timeout_s
        return [
            {
                "status": "success",
                "worker_id": 0,
                "samples_processed": 1,
                "weight_change_norm": 0.0,
            }
        ]

    monkeypatch.setattr(RayColorsProcessor, "_await_iteration_with_watchdog", fake_await)
    monkeypatch.setattr(RayColorsProcessor, "_ensure_progress_tracker", lambda self: _DummyProgressTracker())

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=0.0,
        iteration_timeout_max_retries=0,
    )
    worker = _DummyWorker()
    processor.worker_manager = _DummyWorkerManager(workers=[worker])
    processor.verbose = False
    processor.color_sets = [np.array([0], dtype=np.int32)]
    processor.influence_matrix = np.zeros((1, 1), dtype=np.float32)

    result = processor._process_samples(None, np.zeros((1, 1), dtype=np.float32), object(), _build_params())

    assert result["samples_processed"] == 1
    assert captured["timeout_s"] == pytest.approx(processor._DEFAULT_ITERATION_STALL_TIMEOUT_S)
    assert worker.setup_calls[0][1].get("progress_tracker") is not None


def test_watchdog_surfaces_failed_future_immediately(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    def fake_ray_wait(pending, num_returns: int, timeout: Optional[float] = None):
        assert num_returns == 1
        assert timeout is not None
        if "failed_future" in pending:
            return ["failed_future"], [entry for entry in pending if entry != "failed_future"]
        return [], pending

    def fake_ray_get(ref, timeout: Optional[float] = None):
        if ref == "failed_future":
            raise RuntimeError("simulated actor error")
        raise AssertionError(f"Unexpected ray.get call for ref={ref!r}")

    monkeypatch.setattr(module.ray, "wait", fake_ray_wait)
    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=0,
    )

    with pytest.raises(RuntimeError, match="future 1 failed"):
        processor._await_iteration_with_watchdog(
            ["ok_future", "failed_future"],
            stall_timeout_s=10.0,
            progress_tracker=None,
        )


def test_watchdog_fallback_resets_stall_timer_on_completions(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    wait_calls = {"count": 0}

    def fake_ray_wait(pending, num_returns: int, timeout: Optional[float] = None):
        assert num_returns == 1
        assert timeout is not None
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            return ["future_a"], ["future_b"]
        if wait_calls["count"] == 2:
            return [], pending
        if wait_calls["count"] == 3:
            return ["future_b"], []
        raise AssertionError("Unexpected extra ray.wait call")

    def fake_ray_get(ref, timeout: Optional[float] = None):
        if ref == "future_a":
            return {"status": "success", "worker_id": 0}
        if ref == "future_b":
            return {"status": "success", "worker_id": 1}
        raise AssertionError(f"Unexpected ray.get call for ref={ref!r}")

    monotonic_values = iter([0.0, 9.0, 18.0, 19.0])

    def fake_monotonic():
        return next(monotonic_values)

    monkeypatch.setattr(module.ray, "wait", fake_ray_wait)
    monkeypatch.setattr(module.ray, "get", fake_ray_get)
    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=0,
    )

    results = processor._await_iteration_with_watchdog(
        ["future_a", "future_b"],
        stall_timeout_s=10.0,
        progress_tracker=None,
    )

    assert results == [
        {"status": "success", "worker_id": 0},
        {"status": "success", "worker_id": 1},
    ]


def test_watchdog_triggers_on_4x_partition_straggler(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    def fake_ray_wait(pending, num_returns: int, timeout: Optional[float] = None):
        assert num_returns == 1
        assert timeout is not None
        return [], pending

    def fake_ray_get(ref, timeout: Optional[float] = None):
        # `snapshot.remote()` returns dict payload directly in this unit test.
        if isinstance(ref, dict):
            return ref
        raise AssertionError(f"Unexpected ray.get call for ref={ref!r}")

    monkeypatch.setattr(module.ray, "wait", fake_ray_wait)
    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=120.0,
        iteration_timeout_max_retries=0,
    )
    progress_tracker = _SnapshotProgressTracker(
        {
            "now": 101.0,
            "last_progress": 100.0,
            "revision": 7,
            "last_worker_id": 0,
            "last_stage": "partition_3_done",
            "active_partitions": {
                1: {"partition_idx": 3, "elapsed_s": 50.0},
            },
            "partition_median_s": {3: 10.0},
            "partition_peer_counts": {3: 2},
        }
    )

    with pytest.raises(TimeoutError, match="4.0x peer median"):
        processor._await_iteration_with_watchdog(
            ["future_a", "future_b", "future_c"],
            stall_timeout_s=120.0,
            progress_tracker=progress_tracker,
        )


def test_watchdog_does_not_trigger_when_under_4x_partition_limit(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    wait_calls = {"count": 0}

    def fake_ray_wait(pending, num_returns: int, timeout: Optional[float] = None):
        assert num_returns == 1
        assert timeout is not None
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            return [], pending
        if wait_calls["count"] == 2:
            return ["future_a"], ["future_b", "future_c"]
        if wait_calls["count"] == 3:
            return ["future_b"], ["future_c"]
        if wait_calls["count"] == 4:
            return ["future_c"], []
        raise AssertionError("Unexpected extra ray.wait call")

    def fake_ray_get(ref, timeout: Optional[float] = None):
        if isinstance(ref, dict):
            return ref
        if ref == "future_a":
            return {"status": "success", "worker_id": 0}
        if ref == "future_b":
            return {"status": "success", "worker_id": 1}
        if ref == "future_c":
            return {"status": "success", "worker_id": 2}
        raise AssertionError(f"Unexpected ray.get call for ref={ref!r}")

    monkeypatch.setattr(module.ray, "wait", fake_ray_wait)
    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=120.0,
        iteration_timeout_max_retries=0,
    )
    progress_tracker = _SnapshotProgressTracker(
        {
            "now": 201.0,
            "last_progress": 200.5,
            "revision": 11,
            "last_worker_id": 0,
            "last_stage": "partition_4_done",
            "active_partitions": {
                2: {"partition_idx": 4, "elapsed_s": 39.0},
            },
            "partition_median_s": {4: 10.0},
            "partition_peer_counts": {4: 2},
        }
    )

    results = processor._await_iteration_with_watchdog(
        ["future_a", "future_b", "future_c"],
        stall_timeout_s=120.0,
        progress_tracker=progress_tracker,
    )

    assert results == [
        {"status": "success", "worker_id": 0},
        {"status": "success", "worker_id": 1},
        {"status": "success", "worker_id": 2},
    ]


def test_watchdog_ignores_stale_active_partition_for_completed_worker(tmp_path, monkeypatch):
    from floatsom.processing.ray_ops import ray_colors_processor as module

    wait_calls = {"count": 0}

    def fake_ray_wait(pending, num_returns: int, timeout: Optional[float] = None):
        assert num_returns == 1
        assert timeout is not None
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            return ["future_a"], ["future_b", "future_c", "future_d"]
        if wait_calls["count"] == 2:
            return [], pending
        if wait_calls["count"] == 3:
            return ["future_b"], ["future_c", "future_d"]
        if wait_calls["count"] == 4:
            return ["future_c"], ["future_d"]
        if wait_calls["count"] == 5:
            return ["future_d"], []
        raise AssertionError("Unexpected extra ray.wait call")

    def fake_ray_get(ref, timeout: Optional[float] = None):
        if isinstance(ref, dict):
            return ref
        if ref == "future_a":
            return {"status": "success", "worker_id": 0}
        if ref == "future_b":
            return {"status": "success", "worker_id": 1}
        if ref == "future_c":
            return {"status": "success", "worker_id": 2}
        if ref == "future_d":
            return {"status": "success", "worker_id": 3}
        raise AssertionError(f"Unexpected ray.get call for ref={ref!r}")

    monkeypatch.setattr(module.ray, "wait", fake_ray_wait)
    monkeypatch.setattr(module.ray, "get", fake_ray_get)

    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=120.0,
        iteration_timeout_max_retries=0,
    )
    progress_tracker = _SnapshotProgressTracker(
        {
            "now": 501.0,
            "last_progress": 500.5,
            "revision": 20,
            "last_worker_id": 2,
            "last_stage": "partition_6_start",
            # Worker 0 already completed, but its stale timer exceeds 4x threshold.
            "active_partitions": {
                0: {"partition_idx": 6, "elapsed_s": 80.0},
                2: {"partition_idx": 6, "elapsed_s": 20.0},
            },
            "partition_median_s": {6: 10.0},
            "partition_peer_counts": {6: 3},
        }
    )

    results = processor._await_iteration_with_watchdog(
        ["future_a", "future_b", "future_c", "future_d"],
        stall_timeout_s=120.0,
        progress_tracker=progress_tracker,
    )

    assert results == [
        {"status": "success", "worker_id": 0},
        {"status": "success", "worker_id": 1},
        {"status": "success", "worker_id": 2},
        {"status": "success", "worker_id": 3},
    ]


def test_recover_from_iteration_timeout_requires_data_source(tmp_path):
    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=1,
    )
    processor._last_good_weights_cpu = np.zeros((1, 1), dtype=np.float32)

    with pytest.raises(RuntimeError, match="no data_source cached"):
        processor._recover_from_iteration_timeout(topology=object(), params=_build_params())


def test_recover_from_iteration_timeout_requires_checkpoint_weights(tmp_path):
    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=1,
    )
    processor._data_source = object()
    processor.worker_manager = _RecoveryWorkerManager()
    processor._last_good_weights_cpu = None

    with pytest.raises(RuntimeError, match="no CPU checkpoint weights"):
        processor._recover_from_iteration_timeout(topology=object(), params=_build_params())


def test_recover_from_iteration_timeout_reinitializes_workers(tmp_path):
    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=1,
    )
    manager = _RecoveryWorkerManager()
    topology = object()
    params = _build_params()
    data_source = object()
    checkpoint = np.array([[1.5]], dtype=np.float32)

    processor.worker_manager = manager
    processor._data_source = data_source
    processor._last_good_weights_cpu = checkpoint

    processor._recover_from_iteration_timeout(topology=topology, params=params)

    assert manager.cleanup_calls == [False]
    assert len(manager.initialize_calls) == 1
    init_weights, init_topology, init_params, init_data_source = manager.initialize_calls[0]
    np.testing.assert_array_equal(init_weights, checkpoint)
    assert init_topology is topology
    assert init_params is params
    assert init_data_source is data_source


def test_recover_from_iteration_timeout_continues_after_cleanup_failure(tmp_path):
    processor = _build_processor(
        tmp_path,
        iteration_timeout_s=30.0,
        iteration_timeout_max_retries=1,
    )
    manager = _RecoveryWorkerManager(cleanup_should_raise=True)
    topology = object()
    params = _build_params()
    data_source = object()
    checkpoint = np.array([[2.0]], dtype=np.float32)

    processor.worker_manager = manager
    processor._data_source = data_source
    processor._last_good_weights_cpu = checkpoint

    processor._recover_from_iteration_timeout(topology=topology, params=params)

    assert manager.cleanup_calls == [False]
    assert len(manager.initialize_calls) == 1
