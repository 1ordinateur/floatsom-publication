"""
Unit tests for GPU-scaling benchmark cleanup guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("cupy")

from floatsom_benchmarks.speed_benchmarks.gpu_scaling import runners as gpu_runners


class _ProcessorWithManager:
    def __init__(self, workers):
        self.worker_manager = SimpleNamespace(workers=workers)
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


class _ProcessorNoManager:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def _capture_clean_ray_state(monkeypatch):
    calls = []

    def _fake_clean_ray_state(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(gpu_runners, "ensure_clean_ray_state", _fake_clean_ray_state)
    return calls


def test_cleanup_ray_resources_skips_processor_cleanup_when_workers_already_gone(monkeypatch):
    calls = _capture_clean_ray_state(monkeypatch)
    processor = _ProcessorWithManager(workers=[])
    som = SimpleNamespace(processor=processor)
    args = SimpleNamespace(use_ray=True)

    gpu_runners.BenchmarkRunner._cleanup_ray_resources(som, args)

    assert processor.cleanup_calls == 0
    assert calls == [{"free_gpu_memory": False}]


def test_cleanup_ray_resources_calls_processor_cleanup_when_workers_are_live(monkeypatch):
    calls = _capture_clean_ray_state(monkeypatch)
    processor = _ProcessorWithManager(workers=["worker-handle"])
    som = SimpleNamespace(processor=processor)
    args = SimpleNamespace(use_ray=True)

    gpu_runners.BenchmarkRunner._cleanup_ray_resources(som, args)

    assert processor.cleanup_calls == 1
    assert calls == [{"free_gpu_memory": False}]


def test_cleanup_ray_resources_calls_cleanup_when_no_worker_manager(monkeypatch):
    calls = _capture_clean_ray_state(monkeypatch)
    processor = _ProcessorNoManager()
    som = SimpleNamespace(processor=processor)
    args = SimpleNamespace(use_ray=True)

    gpu_runners.BenchmarkRunner._cleanup_ray_resources(som, args)

    assert processor.cleanup_calls == 1
    assert calls == [{"free_gpu_memory": False}]
