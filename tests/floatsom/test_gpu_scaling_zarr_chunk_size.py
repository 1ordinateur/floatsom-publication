"""
Regression tests for GPU scaling benchmark zarr chunk sizing.

These tests lock in:
- CLI/config defaulting: zarr_chunk_size defaults to FloatSOM chunk_size when unset/invalid.
- Cache key: zarr_chunk_size participates in the cached dataset hash.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path

import pytest

pytest.importorskip("cupy")

from floatsom.benchmarks.speed_benchmarks.gpu_scaling.config import BenchmarkConfig
from floatsom.benchmarks.speed_benchmarks.gpu_scaling import large_dataset as gpu_large_dataset
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.large_dataset import generate_or_load_data


def _base_args(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        samples=16,
        seed=123,
        grid_size=2,
        total_iterations=1,
        sampling_method="full",
        sampling_fraction=None,
        chunk_size=None,
        minibatch_chunk_size=32,
        cache_dir=str(tmp_path / "cache"),
        temp_dir=str(tmp_path / "temp"),
        run_temp_dir=str(tmp_path / "run"),
        ray_local_storage_path=str(tmp_path / "local"),
        initial_learning_rate=0.5,
        force_regenerate=True,
        verbose=False,
        zarr_chunk_size=None,
    )


def test_benchmark_config_defaults_zarr_chunk_size_to_processing_chunk(tmp_path):
    base = _base_args(tmp_path)
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="batch",
        topology_type="grid",
    )
    assert args.zarr_chunk_size == args.chunk_size


@pytest.mark.parametrize("raw_value", [0, -1, -10])
def test_benchmark_config_clamps_invalid_zarr_chunk_size(tmp_path, raw_value):
    base = _base_args(tmp_path)
    base.zarr_chunk_size = raw_value
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="batch",
        topology_type="grid",
    )
    assert args.zarr_chunk_size == args.chunk_size


def test_benchmark_config_respects_explicit_zarr_chunk_size(tmp_path):
    base = _base_args(tmp_path)
    base.zarr_chunk_size = 7
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="batch",
        topology_type="grid",
    )
    assert args.zarr_chunk_size == 7


def test_benchmark_config_chunk_size_override_wins_for_batch(tmp_path):
    base = _base_args(tmp_path)
    base.chunk_size = 123
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="batch",
        topology_type="grid",
    )
    assert args.chunk_size == 123
    assert args.chunk_size_source == "manual_cli_chunk_size"
    assert args.zarr_chunk_size == 123


def test_benchmark_config_chunk_size_override_wins_for_minibatch(tmp_path):
    base = _base_args(tmp_path)
    base.chunk_size = 456
    base.minibatch_chunk_size = 32
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="minibatch",
        topology_type="grid",
    )
    assert args.chunk_size == 456
    assert args.chunk_size_source == "manual_cli_chunk_size"


def test_benchmark_config_minibatch_chunk_used_without_global_override(tmp_path):
    base = _base_args(tmp_path)
    base.minibatch_chunk_size = 77
    args = BenchmarkConfig.create_benchmark_args(
        base_args=base,
        dimension=8,
        num_gpus=1,
        processing_method="minibatch",
        topology_type="grid",
    )
    assert args.chunk_size == 77
    assert args.chunk_size_source == "manual_minibatch_chunk_size"


def test_dataset_cache_key_includes_zarr_chunk_size(tmp_path):
    base = _base_args(tmp_path)
    base.samples = 32
    base.input_dim = 3

    base.zarr_chunk_size = 4
    path_a, _ = generate_or_load_data(base)

    base.zarr_chunk_size = 8
    path_b, _ = generate_or_load_data(base)

    assert path_a != path_b


def test_generate_or_load_data_regenerates_invalid_cached_zarr(tmp_path, monkeypatch):
    base = _base_args(tmp_path)
    base.force_regenerate = False
    base.samples = 64
    base.input_dim = 3
    base.seed = 44
    base.zarr_chunk_size = 5

    config_hash = "random_64_3_44_5".encode()
    expected_hash = hashlib.md5(config_hash).hexdigest()[:8]
    expected_path = Path(base.cache_dir) / f"random_64_3_{expected_hash}.zarr"
    expected_path.mkdir(parents=True, exist_ok=True)
    (expected_path / "zarr.json").write_text("", encoding="utf-8")

    def _raise_invalid_cache(path, mode="r", *args, **kwargs):
        assert str(path) == str(expected_path)
        assert mode == "r"
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(gpu_large_dataset.zarr, "open_array", _raise_invalid_cache)

    regenerate_calls = {"count": 0}

    def _fake_generate_random_zarr_data(
        n_samples,
        input_dim,
        zarr_path,
        zarr_chunk_size,
        seed=None,
        verbose=True,
    ):
        regenerate_calls["count"] += 1
        assert zarr_path == str(expected_path)

        class _FakeArray:
            shape = (n_samples, input_dim)
            dtype = "float32"
            chunks = (zarr_chunk_size, input_dim)

        return _FakeArray()

    monkeypatch.setattr(
        gpu_large_dataset,
        "generate_random_zarr_data",
        _fake_generate_random_zarr_data,
    )

    zarr_path, metadata = generate_or_load_data(base)

    assert regenerate_calls["count"] == 1
    assert zarr_path == str(expected_path)
    assert metadata["shape"] == (64, 3)
    assert not expected_path.exists()

