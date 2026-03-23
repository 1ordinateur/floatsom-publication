"""Unit tests for RayColorWorker local timeout/gather helpers."""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom.processing.ray_ops.workers import ray_color_worker as worker_module

RayColorWorker = worker_module.RayColorWorker


def _resolve_worker_impl_class():
    """Return the undecorated worker class when Ray wraps it as an actor class."""
    cls = RayColorWorker

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

    raise TypeError("Unable to resolve underlying RayColorWorker implementation class")


def _make_worker(worker_id: int = 7):
    worker_cls = _resolve_worker_impl_class()
    worker = object.__new__(worker_cls)
    worker.worker_id = int(worker_id)
    worker._cached_color_stage_timeout_s = None
    worker._pending_color_jobs = []
    return worker


def _make_future_with_result(value):
    future = concurrent.futures.Future()
    future.set_result(value)
    return future


def _make_future_pending():
    return concurrent.futures.Future()


def _make_future_with_exception(exc: Exception):
    future = concurrent.futures.Future()
    future.set_exception(exc)
    return future


def test_resolve_color_stage_timeout_prefers_color_stage_then_cache():
    worker = _make_worker()
    worker.ray_config = {
        "color_stage_timeout_s": "2.5",
        "iteration_timeout_s": 99,
    }

    first = worker._resolve_color_stage_timeout_s()
    assert first == pytest.approx(2.5)

    # Cached value should be reused even if config changes afterwards.
    worker.ray_config["color_stage_timeout_s"] = 8.0
    assert worker._resolve_color_stage_timeout_s() == pytest.approx(2.5)


def test_resolve_color_stage_timeout_falls_back_to_iteration_and_default():
    worker = _make_worker()
    worker.ray_config = SimpleNamespace(color_stage_timeout_s=0.0, iteration_timeout_s=12.0)
    assert worker._resolve_color_stage_timeout_s() == pytest.approx(12.0)

    worker = _make_worker()
    worker.ray_config = {"color_stage_timeout_s": "invalid", "iteration_timeout_s": None}
    assert worker._resolve_color_stage_timeout_s() == pytest.approx(300.0)


def test_wait_cuda_event_with_timeout_succeeds_when_event_done_property_flips(monkeypatch):
    worker = _make_worker()

    class _Event:
        def __init__(self):
            self.calls = 0

        @property
        def done(self):
            self.calls += 1
            return self.calls >= 2

    event = _Event()
    monotonic_values = iter([0.0, 0.01])
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    worker._wait_cuda_event_with_timeout(event, 0.5, context="test_complete")


def test_wait_cuda_event_with_timeout_succeeds_when_event_done_callable_flips(monkeypatch):
    worker = _make_worker()

    class _Event:
        def __init__(self):
            self.calls = 0

        def done(self):
            self.calls += 1
            return self.calls >= 2

    event = _Event()
    monotonic_values = iter([0.0, 0.01])
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    worker._wait_cuda_event_with_timeout(event, 0.5, context="test_done_callable")


def test_wait_cuda_event_with_timeout_raises_after_deadline_when_done_never_set(monkeypatch):
    worker = _make_worker(worker_id=13)

    class _Event:
        @property
        def done(self):
            return False

    monotonic_values = iter([0.0, 0.6])
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="Worker 13: Timed out after 0.5s"):
        worker._wait_cuda_event_with_timeout(_Event(), 0.5, context="test_timeout")


def test_wait_cuda_event_with_timeout_uses_query_fallback_when_done_missing(monkeypatch):
    worker = _make_worker()

    class _Event:
        def __init__(self):
            self.calls = 0

        def query(self):
            self.calls += 1
            return self.calls >= 2

    event = _Event()
    monotonic_values = iter([0.0, 0.01])
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    worker._wait_cuda_event_with_timeout(event, 0.5, context="test_query_fallback")


def test_wait_cuda_event_with_timeout_raises_for_unknown_event_interface(monkeypatch):
    worker = _make_worker()

    class _Event:
        pass

    monkeypatch.setattr(worker_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(worker_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(AttributeError, match="does not expose `done` or `query\\(\\)`"):
        worker._wait_cuda_event_with_timeout(_Event(), 0.5, context="test_unknown_event")


def test_sync_stream_with_timeout_records_event_and_uses_wait(monkeypatch):
    worker = _make_worker()
    captured = {}

    class _FakeEvent:
        def __init__(self):
            self.recorded = False

        def record(self):
            self.recorded = True

    fake_cp = SimpleNamespace(cuda=SimpleNamespace(Event=_FakeEvent))
    monkeypatch.setattr(worker_module, "cp", fake_cp)

    def fake_wait(event, timeout_s, *, context):
        captured["event"] = event
        captured["timeout_s"] = timeout_s
        captured["context"] = context

    worker._wait_cuda_event_with_timeout = fake_wait

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    worker._sync_stream_with_timeout(_Stream(), 4.0, context="flush_sync")

    assert captured["timeout_s"] == pytest.approx(4.0)
    assert captured["context"] == "flush_sync"
    assert captured["event"].recorded is True


def test_iter_gathered_color_slices_yields_results_in_submission_order():
    worker = _make_worker()
    worker._cpu_pool_workers = 2

    def fake_gather(chunk_id, local_indices, bmus):
        return (
            np.array([chunk_id], dtype=np.float32),
            np.asarray(bmus, dtype=np.int32),
            np.asarray(local_indices, dtype=np.int32),
        )

    worker._gather_color_slice = fake_gather

    class _Pool:
        def submit(self, fn, *args):
            return _make_future_with_result(fn(*args))

    worker.get_cpu_pool = lambda min_workers=2: _Pool()

    entries = [
        {"chunk": 3, "local_indices": np.array([1], dtype=np.int32), "bmus": np.array([2], dtype=np.int32)},
        {"chunk": 4, "local_indices": np.array([5], dtype=np.int32), "bmus": np.array([6], dtype=np.int32)},
    ]

    got = list(
        worker._iter_gathered_color_slices(
            entries,
            partition_idx=1,
            color_idx=0,
            timeout_s=1.0,
        )
    )

    assert len(got) == 2
    assert got[0][0][0] == pytest.approx(3.0)
    assert got[1][0][0] == pytest.approx(4.0)


def test_iter_gathered_color_slices_raises_timeout_and_cancels_pending():
    worker = _make_worker(worker_id=17)
    worker._cpu_pool_workers = 1

    future = _make_future_pending()

    class _Pool:
        def submit(self, fn, *args):
            del fn, args
            return future

    worker._gather_color_slice = lambda *args, **kwargs: None
    worker.get_cpu_pool = lambda min_workers=2: _Pool()

    entries = [
        {"chunk": 9, "local_indices": np.array([1], dtype=np.int32), "bmus": np.array([2], dtype=np.int32)}
    ]

    with pytest.raises(TimeoutError, match="Worker 17: Timed out after 0.3s gathering colour slice"):
        list(
            worker._iter_gathered_color_slices(
                entries,
                partition_idx=2,
                color_idx=1,
                timeout_s=0.3,
            )
        )

    assert future.cancelled() is True


def test_iter_gathered_color_slices_wraps_worker_exception_and_cancels_pending():
    worker = _make_worker(worker_id=19)
    worker._cpu_pool_workers = 1

    future = _make_future_with_exception(ValueError("bad slice"))

    class _Pool:
        def submit(self, fn, *args):
            del fn, args
            return future

    worker._gather_color_slice = lambda *args, **kwargs: None
    worker.get_cpu_pool = lambda min_workers=2: _Pool()

    entries = [
        {"chunk": 10, "local_indices": np.array([1], dtype=np.int32), "bmus": np.array([2], dtype=np.int32)}
    ]

    with pytest.raises(RuntimeError, match="Worker 19: Failed gathering colour slice"):
        list(
            worker._iter_gathered_color_slices(
                entries,
                partition_idx=3,
                color_idx=0,
                timeout_s=1.0,
            )
        )

    assert future.done() is True


def test_wait_for_color_jobs_uses_stage_timeout_and_clears_payload():
    worker = _make_worker()
    timing_calls = []
    event_calls = []

    job = {
        "event": object(),
        "samples": object(),
        "bmus": object(),
        "indices": object(),
    }
    worker._pending_color_jobs = [job]
    worker._resolve_color_stage_timeout_s = lambda: 6.5

    def fake_wait(event, timeout_s, *, context):
        event_calls.append({"event": event, "timeout_s": timeout_s, "context": context})

    worker._wait_cuda_event_with_timeout = fake_wait
    worker._record_timing = lambda *args, **kwargs: timing_calls.append((args, kwargs))

    worker._wait_for_color_jobs()

    assert len(event_calls) == 1
    assert event_calls[0]["timeout_s"] == pytest.approx(6.5)
    assert "pending_color_job" in event_calls[0]["context"]
    assert worker._pending_color_jobs == []
    assert "samples" not in job
    assert "bmus" not in job
    assert "indices" not in job
    assert timing_calls


@pytest.mark.parametrize(
    ("topology_type", "expected"),
    [
        ("rng", True),
        ("mst", True),
        ("grid", False),
        (None, False),
    ],
)
def test_stream_color_batches_enabled_for_dynamic_graph_topologies(topology_type, expected):
    worker = _make_worker()
    worker.params = SimpleNamespace(
        topology_config=SimpleNamespace(topology_type=topology_type),
    )
    assert worker._stream_color_batches_enabled() is expected
