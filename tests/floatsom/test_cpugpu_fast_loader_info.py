import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def test_get_info_reports_total_pinned_buffer_mb():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = True
    loader.chunk_size = 10
    loader.n_samples = 100
    loader.n_features = 5
    loader.total_chunks = 10
    loader.pinned_buffer_bytes = 1024**2  # 1 MiB per pinned buffer
    loader.pinned_memories = [object(), object(), object()]
    loader.randomize_chunks = False
    loader.current_chunk_position = 0

    info = loader.get_info()

    assert info["pinned_buffer_count"] == 3
    assert info["pinned_buffer_mb"] == pytest.approx(3.0)


class _FakeFuture:
    def __init__(self, *, done: bool = False, result_value=None):
        self._done = bool(done)
        self._result_value = result_value
        self.cancel_calls = 0

    def done(self):
        return self._done

    def cancel(self):
        self.cancel_calls += 1
        return False

    def cancelled(self):
        return False

    def result(self):
        return self._result_value


def _build_prefetch_loader():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 1
    loader.total_chunks = 1
    loader.current_chunk_position = 0
    loader.chunk_order = [0]
    loader.ram_cache = {}
    loader.worker_id = 0
    loader.prefetch_future_timeout_s = 1.0
    loader._prefetch_inflight_futures = {}
    return loader


def test_prefetch_nonblocking_does_not_resubmit_inflight_chunk(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module

    pending_future = _FakeFuture(done=False)

    class _FakeExecutor:
        def __init__(self):
            self.submit_calls = 0

        def submit(self, fn, chunk_idx):
            self.submit_calls += 1
            return pending_future

    loader = _build_prefetch_loader()
    executor = _FakeExecutor()
    loader._prefetch_executor = executor

    monkeypatch.setattr(loader_module.concurrent.futures, "wait", lambda *args, **kwargs: pytest.fail("unexpected wait"))

    loader._prefetch_upcoming_chunks()
    loader._prefetch_upcoming_chunks()

    assert executor.submit_calls == 1
    assert 0 in loader._prefetch_inflight_futures


def test_try_resolve_prefetched_chunk_requires_done_future():
    loader = _build_prefetch_loader()
    pending_future = _FakeFuture(done=False)
    loader._prefetch_inflight_futures[0] = pending_future

    resolved = loader._try_resolve_prefetched_chunk(0, wait_timeout_s=0.0)

    assert resolved is False
    assert 0 in loader._prefetch_inflight_futures
    assert loader.ram_cache == {}


def test_try_resolve_prefetched_chunk_moves_done_future_to_cache():
    loader = _build_prefetch_loader()
    done_future = _FakeFuture(done=True, result_value=(0, "ready"))
    loader._prefetch_inflight_futures[0] = done_future

    resolved = loader._try_resolve_prefetched_chunk(0, wait_timeout_s=0.0)

    assert resolved is True
    assert 0 not in loader._prefetch_inflight_futures
    assert loader.ram_cache[0] == "ready"


def test_try_resolve_prefetched_chunk_waits_on_requested_future(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module

    loader = _build_prefetch_loader()
    pending_future = _FakeFuture(done=False)
    loader._prefetch_inflight_futures[0] = pending_future

    wait_calls = []

    def _fake_wait(futures, timeout, return_when):
        wait_calls.append((futures, timeout, return_when))
        return set(), set(futures)

    monkeypatch.setattr(loader_module.concurrent.futures, "wait", _fake_wait)

    resolved = loader._try_resolve_prefetched_chunk(0, wait_timeout_s=0.02)

    assert resolved is False
    assert len(wait_calls) == 1
    assert wait_calls[0][1] == pytest.approx(0.02)


def test_try_get_chunk_from_cache_or_inflight_harvests_recovered_cache():
    loader = _build_prefetch_loader()
    loader.ram_cache = {}

    def _collect(indices_to_keep=None):
        del indices_to_keep
        loader.ram_cache[0] = "recovered"

    loader._collect_completed_prefetch_futures = _collect
    loader._try_resolve_prefetched_chunk = lambda _idx, wait_timeout_s=0.0: False

    chunk = loader._try_get_chunk_from_cache_or_inflight(0, wait_timeout_s=0.01)

    assert chunk == "recovered"


def test_prefetch_collects_completed_inflight_future():
    completed_future = _FakeFuture(done=True, result_value=(0, "chunk-data"))

    class _FakeExecutor:
        def __init__(self):
            self.submit_calls = 0

        def submit(self, fn, chunk_idx):
            self.submit_calls += 1
            return _FakeFuture(done=False)

    loader = _build_prefetch_loader()
    loader._prefetch_executor = _FakeExecutor()
    loader._prefetch_inflight_futures[0] = completed_future

    loader._prefetch_upcoming_chunks()

    assert loader.ram_cache[0] == "chunk-data"
    assert 0 not in loader._prefetch_inflight_futures
    assert loader._prefetch_executor.submit_calls == 0


def test_collect_completed_prefetch_futures_handles_inflight_mutation_during_done_check():
    """Regression: avoid dict-size-change crashes when inflight map changes mid-harvest."""
    loader = _build_prefetch_loader()

    class _MutatingFuture(_FakeFuture):
        def __init__(self, owner):
            super().__init__(done=True, result_value=(0, "chunk-data"))
            self._owner = owner
            self._mutated = False

        def done(self):
            if not self._mutated:
                self._owner._prefetch_inflight_futures[99] = _FakeFuture(done=False)
                self._mutated = True
            return True

    loader._prefetch_inflight_futures = {0: _MutatingFuture(loader)}

    loader._collect_completed_prefetch_futures()

    assert loader.ram_cache[0] == "chunk-data"
    assert 0 not in loader._prefetch_inflight_futures
    assert 99 in loader._prefetch_inflight_futures


def test_prefetch_upcoming_chunks_uses_exact_upcoming_membership_window():
    submissions = []

    class _Executor:
        def submit(self, fn, chunk_idx):
            del fn
            submissions.append(int(chunk_idx))
            return _FakeFuture(done=False)

    loader = _build_prefetch_loader()
    loader.total_chunks = 5
    loader.prefetch_buffer_size = 3
    loader.current_chunk_position = 2
    loader.chunk_order = np.array([4, 1, 3, 0, 2], dtype=np.int32)
    loader.ram_cache = {3: "keep", 4: "evict"}
    loader._prefetch_inflight_futures = {0: _FakeFuture(done=False)}
    loader._prefetch_executor = _Executor()

    seen_keep_sets = []

    def _collect(indices_to_keep=None):
        seen_keep_sets.append(set(indices_to_keep or set()))

    loader._collect_completed_prefetch_futures = _collect

    loader._prefetch_upcoming_chunks()

    assert seen_keep_sets == [{3, 0, 2}]
    assert set(loader.ram_cache) == {3}
    assert submissions == [2]
    assert set(loader._prefetch_inflight_futures) == {0, 2}


def test_set_chunk_order_resets_cursor_updates_map_and_prefetches():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 2
    loader.total_chunks = 4
    loader.current_chunk_position = 3
    loader.chunk_order = np.array([0, 1, 2, 3], dtype=np.int32)
    loader.ram_cache = {}
    loader.worker_id = 0
    loader._prefetch_inflight_futures = {}
    prefetch_calls = []
    loader._prefetch_upcoming_chunks = lambda: prefetch_calls.append(tuple(loader.chunk_order.tolist()))

    loader.set_chunk_order(np.array([2, 0, 3, 1], dtype=np.int32))

    assert loader.current_chunk_position == 0
    assert tuple(loader.chunk_order.tolist()) == (2, 0, 3, 1)
    assert loader.chunk_position_map == {2: 0, 0: 1, 3: 2, 1: 3}
    assert prefetch_calls == [(2, 0, 3, 1)]


def test_set_chunk_order_validates_full_permutation():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 1
    loader.total_chunks = 3
    loader.current_chunk_position = 0
    loader.chunk_order = np.array([0, 1, 2], dtype=np.int32)
    loader.ram_cache = {}
    loader.worker_id = 0
    loader._prefetch_inflight_futures = {}
    loader._prefetch_upcoming_chunks = lambda: None

    with pytest.raises(ValueError, match="length"):
        loader.set_chunk_order(np.array([0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="within \\[0, total_chunks\\)"):
        loader.set_chunk_order(np.array([-1, 0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="within \\[0, total_chunks\\)"):
        loader.set_chunk_order(np.array([0, 1, 3], dtype=np.int32))

    with pytest.raises(ValueError, match="permutation"):
        loader.set_chunk_order(np.array([0, 0, 1], dtype=np.int32))


def test_mark_chunk_consumed_delegates_to_advance_position():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    seen = []
    loader._advance_position_after_fetch = lambda idx: seen.append(int(idx))

    loader.mark_chunk_consumed(7)

    assert seen == [7]


def test_set_prefetch_buffer_size_applies_resolved_size_and_prefetches():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 1
    loader.total_chunks = 10
    loader.worker_id = 0
    loader._prefetch_worker_count = 1
    loader._prefetch_executor = object()
    loader._prefetch_inflight_futures = {}
    loader.ram_cache = {}
    guard_calls = []
    prefetch_calls = []

    def _guard(desired):
        guard_calls.append(int(desired))
        return min(5, int(desired))

    loader._calculate_prefetch_buffer_size = _guard
    loader._prefetch_upcoming_chunks = lambda: prefetch_calls.append(True)

    applied = loader.set_prefetch_buffer_size(8)

    assert applied == 5
    assert loader.prefetch_buffer_size == 5
    assert guard_calls == [8]
    assert prefetch_calls == [True]


def test_calculate_prefetch_buffer_size_is_hard_requested_size(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    class _Vm:
        available = 1  # intentionally tiny to force max_chunks=0

    monkeypatch.setattr(loader_module.psutil, "virtual_memory", lambda: _Vm())

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.worker_id = 0
    loader.n_workers_per_node = 1
    loader.chunk_size = 100
    loader.n_features = 100

    resolved = loader._calculate_prefetch_buffer_size(6)
    assert resolved == 6


def test_set_prefetch_buffer_size_zero_clears_cache_and_inflight():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    class _Executor:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    inflight = _FakeFuture(done=False)
    executor = _Executor()

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 4
    loader.total_chunks = 10
    loader.worker_id = 0
    loader._prefetch_worker_count = 2
    loader._prefetch_executor = executor
    loader._prefetch_inflight_futures = {3: inflight}
    loader.ram_cache = {3: "chunk"}
    loader._calculate_prefetch_buffer_size = lambda _desired: 0
    loader._prefetch_upcoming_chunks = lambda: pytest.fail("should not prefetch when size becomes zero")

    applied = loader.set_prefetch_buffer_size(8)

    assert applied == 0
    assert loader.prefetch_buffer_size == 0
    assert loader._prefetch_inflight_futures == {}
    assert loader.ram_cache == {}
    assert inflight.cancel_calls == 1
    assert executor.shutdown_calls == [(False, True)]
    assert loader._prefetch_executor is None


def test_set_prefetch_buffer_size_reenable_creates_executor(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    executor_calls = {}

    class _Executor:
        def __init__(self, *, max_workers, thread_name_prefix):
            executor_calls["max_workers"] = int(max_workers)
            executor_calls["thread_name_prefix"] = str(thread_name_prefix)

    monkeypatch.setattr(
        loader_module.concurrent.futures,
        "ThreadPoolExecutor",
        lambda max_workers, thread_name_prefix: _Executor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        ),
    )

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.prefetch_buffer_size = 0
    loader.total_chunks = 10
    loader.worker_id = 7
    loader._prefetch_worker_count = 4
    loader._prefetch_executor = None
    loader._prefetch_inflight_futures = {}
    loader.ram_cache = {}
    loader._calculate_prefetch_buffer_size = lambda _desired: 3
    prefetch_calls = []
    loader._prefetch_upcoming_chunks = lambda: prefetch_calls.append(True)

    applied = loader.set_prefetch_buffer_size(3)

    assert applied == 3
    assert loader.prefetch_buffer_size == 3
    assert prefetch_calls == [True]
    assert executor_calls["max_workers"] == 4
    assert executor_calls["thread_name_prefix"] == "cpugpu-prefetch-7"
    assert loader._prefetch_executor is not None


def test_print_prepositioned_cache_miss_suppresses_inflight_by_default(capsys):
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.ram_mode = False
    loader.worker_id = 0
    loader.current_chunk_position = 0
    loader.total_chunks = 3
    loader.chunk_order = np.array([0, 1, 2], dtype=np.int32)
    loader.prefetch_buffer_size = 3
    loader.ram_cache = {}
    loader._prefetch_inflight_futures = {1: _FakeFuture(done=False)}
    loader.log_pending_prefetch_miss = False

    loader._print_prepositioned_cache_miss(
        caller="transfer_to_gpu_buffer",
        requested_chunk_index=1,
        actual_chunk_index=1,
    )
    captured = capsys.readouterr()
    assert captured.out == ""

    loader.log_pending_prefetch_miss = True
    loader._print_prepositioned_cache_miss(
        caller="transfer_to_gpu_buffer",
        requested_chunk_index=1,
        actual_chunk_index=1,
    )
    captured = capsys.readouterr()
    assert "prepositioned-cache miss" in captured.out


def _build_ram_loader_for_get_chunk_contract_tests():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.randomize_chunks = False
    loader.ram_mode = True
    loader.chunk_size = 3
    loader.n_samples = 6
    loader.n_features = 2
    loader.ram_data = np.arange(12, dtype=np.float32).reshape(6, 2)
    loader.pinned_buffer_flats = [
        np.zeros(loader.chunk_size * loader.n_features, dtype=np.float32)
    ]
    return loader


def test_get_chunk_requires_explicit_keyword_only_advance_cursor(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    signature = inspect.signature(CPUGPUFastLoader.get_chunk)
    advance_param = signature.parameters["advance_cursor"]
    assert advance_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert advance_param.default is inspect.Parameter.empty

    monkeypatch.setattr(
        loader_module,
        "cp",
        SimpleNamespace(
            asarray=lambda array, dtype=None: np.asarray(array, dtype=dtype),
            float32=np.float32,
        ),
    )
    loader = _build_ram_loader_for_get_chunk_contract_tests()

    with pytest.raises(TypeError, match="advance_cursor"):
        loader.get_chunk(0)

    with pytest.raises(TypeError, match="positional arguments"):
        loader.get_chunk(0, False, True)


def test_transfer_to_gpu_buffer_requires_explicit_keyword_only_advance_cursor():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    signature = inspect.signature(CPUGPUFastLoader.transfer_to_gpu_buffer)
    advance_param = signature.parameters["advance_cursor"]
    assert advance_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert advance_param.default is inspect.Parameter.empty

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    with pytest.raises(TypeError, match="advance_cursor"):
        CPUGPUFastLoader.transfer_to_gpu_buffer(loader, 0, object())


def test_transfer_to_gpu_buffer_advances_cursor_only_for_consume_intent():
    from floatsom.data.cpugpu_fast_loader import CPUGPUFastLoader

    class _TargetBuffer:
        def __init__(self, shape):
            self.array = np.zeros(shape, dtype=np.float32)
            self.shape = self.array.shape

        def set(self, data, stream=None):
            del stream
            self.array[...] = np.asarray(data, dtype=np.float32)

    loader = CPUGPUFastLoader.__new__(CPUGPUFastLoader)
    loader.randomize_chunks = False
    loader.ram_mode = True
    loader.chunk_size = 3
    loader.n_samples = 6
    loader.n_features = 2
    loader.ram_data = np.arange(12, dtype=np.float32).reshape(6, 2)
    loader.pinned_buffer_bytes = loader.chunk_size * loader.n_features * 4
    loader.pinned_memories = [object()]
    loader.pinned_buffer_flats = [np.zeros(loader.chunk_size * loader.n_features, dtype=np.float32)]
    loader.ensure_pinned_buffers = lambda _count: None

    advanced_chunk_ids = []
    loader._advance_position_after_fetch = lambda chunk_idx: advanced_chunk_ids.append(int(chunk_idx))

    prefetch_target = _TargetBuffer((loader.chunk_size, loader.n_features))
    consume_target = _TargetBuffer((loader.chunk_size, loader.n_features))
    prefetch_info = loader.transfer_to_gpu_buffer(0, prefetch_target, advance_cursor=False)
    consume_info = loader.transfer_to_gpu_buffer(1, consume_target, advance_cursor=True)

    np.testing.assert_array_equal(prefetch_target.array, loader.ram_data[:3])
    np.testing.assert_array_equal(consume_target.array, loader.ram_data[3:6])
    assert prefetch_info["chunk_index"] == 0
    assert consume_info["chunk_index"] == 1
    assert advanced_chunk_ids == [1]


def test_get_chunk_advances_cursor_only_for_consume_intent(monkeypatch):
    from floatsom.data import cpugpu_fast_loader as loader_module

    monkeypatch.setattr(
        loader_module,
        "cp",
        SimpleNamespace(
            asarray=lambda array, dtype=None: np.asarray(array, dtype=dtype),
            float32=np.float32,
        ),
    )

    loader = _build_ram_loader_for_get_chunk_contract_tests()
    advanced_chunk_ids = []
    loader._advance_position_after_fetch = lambda chunk_idx: advanced_chunk_ids.append(int(chunk_idx))

    prefetch_chunk, prefetch_info = loader.get_chunk(0, advance_cursor=False)
    consume_chunk, consume_info = loader.get_chunk(1, advance_cursor=True)

    assert prefetch_chunk.shape == (3, 2)
    assert consume_chunk.shape == (3, 2)
    assert prefetch_info["chunk_index"] == 0
    assert consume_info["chunk_index"] == 1
    assert advanced_chunk_ids == [1]
