"""
GPU-dependent smoke tests for the GPU scaling benchmark orchestrator.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..conftest import requires_gpu_and_ray

cp = pytest.importorskip("cupy")
pytest.importorskip("ray")

from floatsom_benchmarks.speed_benchmarks import run_gpu_scaling_benchmark as benchmark
from floatsom_benchmarks.speed_benchmarks.gpu_scaling import runners as gpu_runners

pytestmark = [requires_gpu_and_ray, pytest.mark.gpu_scaling]


@pytest.fixture
def gpu_smoke():
    """Confirm the GPU is reachable for this test module."""
    device_count = cp.cuda.runtime.getDeviceCount()
    assert device_count > 0
    return cp.arange(4, dtype=cp.float32)


@pytest.fixture
def scratch_dir():
    """Create a workspace-local scratch directory (not under /tmp)."""
    base = Path.cwd() / ".bench_scratch"
    unique_dir = base / f"gpu_scaling_{uuid.uuid4().hex}"
    unique_dir.mkdir(parents=True, exist_ok=False)
    yield unique_dir
    shutil.rmtree(unique_dir, ignore_errors=True)
    try:
        base.rmdir()
    except OSError:
        pass


def _make_args(
    mode: str,
    output_dir: Path,
    temp_dir: Path,
    cache_dir: Path,
    safe_cleanup: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        repeats=1,
        dimensions=[8],
        samples=256,
        sample_sizes=[256],
        fixed_dimension=8,
        sampling_method="full",
        sampling_fraction=None,
        gpu_counts=[1],
        topologies=["grid"],
        grid_size=2,
        grid_sizes=[2],
        qe_iterations=[1],
        total_iterations=1,
        initial_learning_rate=0.5,
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        ray_local_storage_path=str(temp_dir),
        resume=False,
        resume_state=None,
        reset_resume_state=False,
        cache_dir=str(cache_dir),
        processing_methods=["batch"],
        chunk_size=None,
        minibatch_chunk_size=32,
        run_timeout_minutes=1,
        dry_run=False,
        skip_existing=False,
        verbose=False,
        seed=123,
        zarr_chunk_size=64,
        force_regenerate=True,
        use_ray=True,
        ray_gpu_count=None,
        safe_cleanup=safe_cleanup,
    )


def _run_benchmark(monkeypatch, args: SimpleNamespace) -> Path:
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    return Path(args.output_dir)


def test_validate_temp_dir_rejects_tmp(gpu_smoke):
    with pytest.raises(ValueError, match="Temporary directory cannot be under /tmp"):
        benchmark._validate_temp_dir("/tmp/gpu_scaling")


def test_filter_numeric_hierarchy_keeps_numeric_keys(gpu_smoke):
    raw = {
        "batch": {
            "10": {"1": {"mean": 1.0}},
            "bad": {"1": {"mean": 2.0}},
            5: {"1": {"mean": 3.0}},
        },
        "junk": "skip",
    }
    filtered = benchmark._filter_numeric_hierarchy(raw)
    assert "batch" in filtered
    assert "10" in filtered["batch"]
    assert 5 in filtered["batch"]
    assert "bad" not in filtered["batch"]


def test_dimension_scaling_gpu_smoke(monkeypatch, tmp_path, scratch_dir, gpu_smoke):
    args = _make_args(
        mode="dimension_scaling",
        output_dir=tmp_path / "dimension_out",
        temp_dir=scratch_dir,
        cache_dir=tmp_path / "cache",
    )
    output_dir = _run_benchmark(monkeypatch, args)
    results_csv = output_dir / "dimension_scaling" / "dimension_scaling_results_grid_batch.csv"
    assert results_csv.exists()
    logs_dir = output_dir / "dimension_scaling" / "logs"
    assert any(logs_dir.glob("*.log"))


def test_sample_scaling_gpu_smoke(monkeypatch, tmp_path, scratch_dir, gpu_smoke):
    args = _make_args(
        mode="sample_scaling",
        output_dir=tmp_path / "sample_out",
        temp_dir=scratch_dir,
        cache_dir=tmp_path / "cache",
    )
    output_dir = _run_benchmark(monkeypatch, args)
    results_csv = output_dir / "sample_scaling" / "sample_scaling_results_grid_batch.csv"
    assert results_csv.exists()
    logs_dir = output_dir / "sample_scaling" / "logs"
    assert any(logs_dir.glob("*.log"))


def test_grid_size_scaling_gpu_smoke(monkeypatch, tmp_path, scratch_dir, gpu_smoke):
    args = _make_args(
        mode="grid_size_scaling",
        output_dir=tmp_path / "grid_out",
        temp_dir=scratch_dir,
        cache_dir=tmp_path / "cache",
    )
    output_dir = _run_benchmark(monkeypatch, args)
    results_csv = output_dir / "grid_size_scaling" / "grid_size_scaling_results_grid_batch.csv"
    assert results_csv.exists()
    logs_dir = output_dir / "grid_size_scaling" / "logs"
    assert any(logs_dir.glob("*.log"))


def test_qe_iterations_gpu_smoke(monkeypatch, tmp_path, scratch_dir, gpu_smoke):
    args = _make_args(
        mode="qe_iterations",
        output_dir=tmp_path / "qe_out",
        temp_dir=scratch_dir,
        cache_dir=tmp_path / "cache",
    )
    output_dir = _run_benchmark(monkeypatch, args)
    qe_dir = output_dir / "qe_iterations"
    assert (qe_dir / "qe_iteration_results.json").exists()
    assert (qe_dir / "qe_convergence_analysis.txt").exists()


@pytest.mark.parametrize("safe_cleanup", [False, True])
def test_sample_scaling_sequential_runs_isolated(
    monkeypatch,
    tmp_path,
    scratch_dir,
    gpu_smoke,
    safe_cleanup,
):
    cache_dir = tmp_path / "cache"
    suffix = "safe" if safe_cleanup else "fast"
    created_dirs = []
    cleaned_dirs = []
    original_create = gpu_runners.BenchmarkRunner._create_run_temp_dir
    original_cleanup = gpu_runners.BenchmarkRunner._cleanup_run_temp_dir

    def create_wrapper(base_run_temp_dir: str, config_name: str) -> str:
        path = original_create(base_run_temp_dir, config_name)
        created_path = Path(path)
        created_dirs.append(created_path)
        assert created_path.parent == Path(base_run_temp_dir)
        assert created_path.name.startswith("run_")
        created_path.joinpath("sentinel.txt").write_text("sentinel", encoding="utf-8")
        return path

    def cleanup_wrapper(run_temp_dir: str) -> None:
        cleaned_dirs.append(Path(run_temp_dir) if run_temp_dir else None)
        original_cleanup(run_temp_dir)
        if run_temp_dir:
            assert not Path(run_temp_dir).exists()

    monkeypatch.setattr(
        gpu_runners.BenchmarkRunner,
        "_create_run_temp_dir",
        staticmethod(create_wrapper),
    )
    monkeypatch.setattr(
        gpu_runners.BenchmarkRunner,
        "_cleanup_run_temp_dir",
        staticmethod(cleanup_wrapper),
    )

    first_args = _make_args(
        mode="sample_scaling",
        output_dir=tmp_path / f"sample_out_{suffix}_1",
        temp_dir=scratch_dir,
        cache_dir=cache_dir,
        safe_cleanup=safe_cleanup,
    )
    first_output = _run_benchmark(monkeypatch, first_args)
    first_results = first_output / "sample_scaling" / "sample_scaling_results_grid_batch.csv"
    assert first_results.exists()
    assert not any(scratch_dir.glob("gpu_scaling_tmp_*"))

    second_args = _make_args(
        mode="sample_scaling",
        output_dir=tmp_path / f"sample_out_{suffix}_2",
        temp_dir=scratch_dir,
        cache_dir=cache_dir,
        safe_cleanup=safe_cleanup,
    )
    second_output = _run_benchmark(monkeypatch, second_args)
    second_results = second_output / "sample_scaling" / "sample_scaling_results_grid_batch.csv"
    assert second_results.exists()
    assert not any(scratch_dir.glob("gpu_scaling_tmp_*"))
    assert created_dirs
    assert len(cleaned_dirs) == len(created_dirs)
    assert all(path is None or not path.exists() for path in cleaned_dirs)
