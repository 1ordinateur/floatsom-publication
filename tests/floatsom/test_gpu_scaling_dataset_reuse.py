"""
Regression tests for dataset reuse in GPU scaling benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from floatsom.benchmarks.speed_benchmarks.gpu_scaling.benchmark_types import SampleScalingBenchmark
from floatsom.benchmarks.speed_benchmarks.gpu_scaling import benchmark_types as gpu_benchmark_types
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.runners import BenchmarkRunner
from floatsom.benchmarks.speed_benchmarks.gpu_scaling import runners as gpu_runners
from floatsom.benchmarks.speed_benchmarks.gpu_scaling.results import ResultsHandler


def _make_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repeats=1,
        topologies=["hexagonal", "mst", "rng"],
        processing_methods=["batch"],
        sample_sizes=[1024],
        gpu_counts=[8],
        fixed_dimension=32,
        grid_size=10,
        seed=123,
        dry_run=False,
        resume_manager=None,
        merge_existing=False,
        verbose=False,
        run_temp_dir=str(tmp_path / "run_temp"),
        temp_dir=str(tmp_path / "temp"),
        cache_dir=str(tmp_path / "cache"),
        samples=1024,
        total_iterations=10,
        sampling_method="full",
        sampling_fraction=None,
        chunk_size=None,
        minibatch_chunk_size=32,
        force_regenerate=False,
        initial_learning_rate=0.5,
        zarr_chunk_size=None,
        use_ray=True,
        ray_gpu_count=8,
        ray_collective_barriers=False,
        force_disk_mode=False,
        skip_existing=False,
        profile=False,
        profile_output="profile.txt",
        safe_cleanup=False,
        profile_workers=False,
        profile_workers_output_dir=None,
        profile_workers_max_stats=50,
    )


def test_sample_scaling_reuses_prepared_dataset_across_topologies(tmp_path, monkeypatch):
    args = _make_args(tmp_path)
    output_dir = tmp_path / "out"

    prepared_datasets = []
    cleanup_payloads = []
    run_payloads = []

    def fake_prepare_dataset_for_execution(**kwargs):
        payload = {
            "execution_zarr_path": f"/tmp/execution_{len(prepared_datasets)}.zarr",
            "metadata": {
                "source_zarr_path": "/tmp/source.zarr",
                "execution_zarr_path": f"/tmp/execution_{len(prepared_datasets)}.zarr",
                "shape": (kwargs["num_samples"], kwargs["dimension"]),
                "dtype": "float32",
                "chunks": (128, kwargs["dimension"]),
                "zarr_path": f"/tmp/execution_{len(prepared_datasets)}.zarr",
            },
            "stage_info": {"staged": True},
            "dataset_stage_dir": str(tmp_path / f"dataset_stage_{len(prepared_datasets)}"),
        }
        prepared_datasets.append(kwargs)
        return payload

    def fake_cleanup_prepared_dataset(prepared_dataset, *, verbose=False):
        cleanup_payloads.append((prepared_dataset, verbose))

    def fake_run_benchmark(**kwargs):
        run_payloads.append(kwargs)
        return {"train_time": 1.5, "training_stats": {"iterations_completed": 10}}

    monkeypatch.setattr(
        BenchmarkRunner,
        "prepare_dataset_for_execution",
        staticmethod(fake_prepare_dataset_for_execution),
    )
    monkeypatch.setattr(
        BenchmarkRunner,
        "cleanup_prepared_dataset",
        staticmethod(fake_cleanup_prepared_dataset),
    )
    monkeypatch.setattr(BenchmarkRunner, "run_benchmark", staticmethod(fake_run_benchmark))
    monkeypatch.setattr(gpu_benchmark_types, "ensure_clean_ray_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_runners, "ensure_clean_ray_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(ResultsHandler, "save_results", staticmethod(lambda *args, **kwargs: None))

    results = SampleScalingBenchmark.run(args, str(output_dir))

    assert len(prepared_datasets) == 1
    assert len(cleanup_payloads) == 1
    assert len(run_payloads) == 3
    assert {payload["topology_type"] for payload in run_payloads} == {"hexagonal", "mst", "rng"}
    assert len({id(payload["prepared_dataset"]) for payload in run_payloads}) == 1
    assert prepared_datasets[0]["processing_method"] == "batch"
    assert prepared_datasets[0]["num_samples"] == 1024
    assert prepared_datasets[0]["seed"] == 123
    assert results["hexagonal"]["batch"][1024][8]["count"] == 1
    assert results["mst"]["batch"][1024][8]["count"] == 1
    assert results["rng"]["batch"][1024][8]["count"] == 1
