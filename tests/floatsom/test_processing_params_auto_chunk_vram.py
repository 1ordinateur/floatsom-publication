"""Unit tests for VRAM-aware auto chunk sizing."""

from __future__ import annotations

import pytest

pytest.importorskip("cupy")

from floatsom.processing import processing_params as pp


def test_auto_chunk_matches_legacy_baseline_at_32gb(monkeypatch):
    monkeypatch.setattr(pp, "get_visible_gpu_vram_mib", lambda: 32 * 1024)
    assert pp.calculate_auto_chunk_size_for_method(50, "batch") == 500_000


def test_auto_chunk_scales_down_with_lower_vram(monkeypatch):
    monkeypatch.setattr(pp, "get_visible_gpu_vram_mib", lambda: 16 * 1024)
    assert pp.calculate_auto_chunk_size_for_method(50, "batch") == 250_000


def test_auto_chunk_cap_scales_with_vram(monkeypatch):
    monkeypatch.setattr(pp, "get_visible_gpu_vram_mib", lambda: 80 * 1024)
    # At 10 dimensions this is cap-limited. Cap should scale to 1,250,000 at 80GB.
    assert pp.calculate_auto_chunk_size_for_method(10, "batch") == 1_250_000


def test_auto_chunk_floor_is_preserved_for_tiny_vram(monkeypatch):
    monkeypatch.setattr(pp, "get_visible_gpu_vram_mib", lambda: 1)
    assert pp.calculate_auto_chunk_size_for_method(2000, "batch") == 10_000


def test_visible_vram_fallback_uses_32gb_when_runtime_probe_fails(monkeypatch):
    def _raise_runtime_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(pp.cp.cuda.runtime, "getDeviceCount", _raise_runtime_error)
    assert pp.get_visible_gpu_vram_mib() == 32 * 1024

