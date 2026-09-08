from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from floatsom_benchmarks.speed_benchmarks import harmonize_gpu_scaling_results as harmonize


def _make_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "dimension_scaling").mkdir()
    return path


def test_find_latest_versioned_results_dir_prefers_newest_date(tmp_path: Path) -> None:
    gpu_dir = tmp_path / "4gpu_scaling"
    gpu_dir.mkdir()

    older = _make_results_dir(gpu_dir / "results_20260320_deadbeef")
    newer = _make_results_dir(gpu_dir / "results_20260321_feedface")
    _make_results_dir(gpu_dir / "results_20260319")

    selected = harmonize._find_latest_versioned_results_dir(gpu_dir)

    assert selected == newer
    assert selected != older


def test_find_latest_versioned_results_dir_uses_mtime_for_same_date(tmp_path: Path) -> None:
    gpu_dir = tmp_path / "8gpu_scaling"
    gpu_dir.mkdir()

    earlier = _make_results_dir(gpu_dir / "results_20260321_deadbeef")
    later = _make_results_dir(gpu_dir / "results_20260321_feedface")

    os.utime(earlier, (1, 1))
    os.utime(later, (2, 2))

    selected = harmonize._find_latest_versioned_results_dir(gpu_dir)

    assert selected == later


def test_discover_results_dirs_resolves_latest_results_child_from_gpu_parent(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    gpu_dir = root / "1gpu_scaling"
    gpu_dir.mkdir(parents=True)

    _make_results_dir(gpu_dir / "results_20260320_deadbeef")
    latest = _make_results_dir(gpu_dir / "results_20260321")

    discovered = harmonize.discover_results_dirs([str(gpu_dir)], search_root=None)

    assert discovered == [latest.resolve()]


def test_discover_results_dirs_scans_root_for_latest_versioned_results(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    one_gpu = root / "1gpu_scaling"
    two_gpu = root / "2gpu_scaling"
    one_gpu.mkdir(parents=True)
    two_gpu.mkdir(parents=True)

    latest_one = _make_results_dir(one_gpu / "results_20260321")
    latest_two = _make_results_dir(two_gpu / "results_20260322_abcd1234")
    _make_results_dir(one_gpu / "results_20260320_deadbeef")

    discovered = harmonize.discover_results_dirs([str(root)], search_root=None)

    assert discovered == sorted([latest_one.resolve(), latest_two.resolve()])


def test_build_figure_12_topology_runtime_summary_table_uses_largest_common_axis() -> None:
    merged_modes = {
        "dimension_scaling": {
            ("hexagonal", "batch"): {
                128: {8: {"mean": 10.0, "std": 0.1, "count": 3, "times": [10.0, 10.0, 10.0]}},
                256: {8: {"mean": 20.0, "std": 0.2, "count": 3, "times": [20.0, 20.0, 20.0]}},
            },
            ("mst", "batch"): {
                128: {8: {"mean": 10.2, "std": 0.1, "count": 3, "times": [10.2, 10.2, 10.2]}},
                256: {8: {"mean": 20.1, "std": 0.2, "count": 3, "times": [20.1, 20.1, 20.1]}},
            },
            ("rng", "batch"): {
                128: {8: {"mean": 10.1, "std": 0.1, "count": 3, "times": [10.1, 10.1, 10.1]}},
                256: {8: {"mean": 20.2, "std": 0.2, "count": 3, "times": [20.2, 20.2, 20.2]}},
            },
        },
        "sample_scaling": {
            ("hexagonal", "batch"): {
                1000: {8: {"mean": 5.0, "std": 0.1, "count": 3, "times": [5.0, 5.0, 5.0]}},
                2000: {8: {"mean": 9.0, "std": 0.1, "count": 3, "times": [9.0, 9.0, 9.0]}},
            },
            ("mst", "batch"): {
                1000: {8: {"mean": 5.0, "std": 0.1, "count": 3, "times": [5.0, 5.0, 5.0]}},
                2000: {8: {"mean": 9.1, "std": 0.1, "count": 3, "times": [9.1, 9.1, 9.1]}},
            },
            ("rng", "batch"): {
                1000: {8: {"mean": 5.1, "std": 0.1, "count": 3, "times": [5.1, 5.1, 5.1]}},
                2000: {8: {"mean": 9.2, "std": 0.1, "count": 3, "times": [9.2, 9.2, 9.2]}},
            },
        },
        "grid_size_scaling": {
            ("hexagonal", "batch"): {
                16: {8: {"mean": 3.0, "std": 0.1, "count": 3, "times": [3.0, 3.0, 3.0]}},
                32: {8: {"mean": 8.0, "std": 0.1, "count": 3, "times": [8.0, 8.0, 8.0]}},
            },
            ("mst", "batch"): {
                16: {8: {"mean": 4.0, "std": 0.1, "count": 3, "times": [4.0, 4.0, 4.0]}},
                32: {8: {"mean": 14.0, "std": 0.1, "count": 3, "times": [14.0, 14.0, 14.0]}},
            },
            ("rng", "batch"): {
                16: {8: {"mean": 5.0, "std": 0.1, "count": 3, "times": [5.0, 5.0, 5.0]}},
                32: {8: {"mean": 18.0, "std": 0.1, "count": 3, "times": [18.0, 18.0, 18.0]}},
            },
        },
    }

    rows = harmonize._build_figure_12_topology_runtime_summary_table(
        merged_modes=merged_modes,
        comparison_gpu_count=8,
    )

    assert [row["mode_name"] for row in rows] == [
        "dimension_scaling",
        "sample_scaling",
        "grid_size_scaling",
    ]
    assert rows[0]["axis_value"] == 256.0
    assert rows[0]["fastest_topology"] == "hexagonal"
    assert rows[0]["slowest_topology"] == "rng"
    assert rows[0]["max_pairwise_runtime_spread_pct"] == pytest.approx(1.0)
    assert rows[1]["axis_value"] == 2000.0
    assert rows[1]["max_pairwise_runtime_spread_pct"] == pytest.approx(2.2222222222222223)
    assert rows[2]["axis_value"] == 32.0
    assert rows[2]["fastest_topology"] == "hexagonal"
    assert rows[2]["slowest_topology"] == "rng"
    assert rows[2]["max_pairwise_runtime_spread_pct"] == pytest.approx(125.0)


def test_write_publication_support_tables_builds_rng_scaling_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paper_tables_dir = tmp_path / "paper_assets" / "tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(harmonize, "_resolve_paper_assets_tables_dir", lambda: paper_tables_dir)

    source_dir = tmp_path / "1gpu_scaling" / "results_20260322"
    for mode_name in ["dimension_scaling", "sample_scaling"]:
        (source_dir / mode_name / "logs").mkdir(parents=True, exist_ok=True)

    (source_dir / "dimension_scaling" / "logs" / "rng_dim1000_samples10000000_grid32_gpu1_batch_seed42.log").write_text(
        "Configuration: rng_dim1000_samples10000000_grid32_gpu1_batch_seed42\n"
        "Data staging mode: disk\n",
        encoding="utf-8",
    )
    (source_dir / "sample_scaling" / "logs" / "rng_dim50_samples1000000000_grid32_gpu8_batch_seed42.log").write_text(
        "Configuration: rng_dim50_samples1000000000_grid32_gpu8_batch_seed42\n"
        "Data staging mode: disk\n",
        encoding="utf-8",
    )

    merged_modes = {
        "dimension_scaling": {
            ("rng", "batch"): {
                1000: {
                    1: {"times": [111.0, 113.0], "mean": 112.0, "std": 1.0, "count": 2},
                }
            }
        },
        "sample_scaling": {
            ("rng", "batch"): {
                1_000_000_000: {
                    8: {"times": [540.0, 546.42], "mean": 543.21, "std": 3.21, "count": 2},
                }
            }
        },
        "grid_size_scaling": {},
    }

    warnings: list[str] = []
    harmonize._write_publication_support_tables(
        merged_modes=merged_modes,
        input_dirs=[source_dir],
        output_dir=tmp_path / "harmonized",
        warnings=warnings,
    )

    output_table = (
        tmp_path
        / "harmonized"
        / "publication_figures"
        / "tables"
        / "supp_table_figure_11_rng_scaling_diagnostics.tsv"
    )
    assert output_table.exists()

    with output_table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 2
    dimension_row = next(row for row in rows if row["mode_name"] == "dimension_scaling")
    sample_row = next(row for row in rows if row["mode_name"] == "sample_scaling")

    assert dimension_row["staging_mode"] == "disk"
    assert dimension_row["axis_value"] == "1000"
    assert dimension_row["gpu_count"] == "1"
    assert dimension_row["runtime_mean_s"] == "112.0"

    assert sample_row["staging_mode"] == "disk"
    assert sample_row["axis_value"] == "1000000000"
    assert sample_row["gpu_count"] == "8"
    assert sample_row["runtime_mean_s"] == "543.21"

    synced_table = paper_tables_dir / "supp_table_figure_11_rng_scaling_diagnostics.tsv"
    assert synced_table.exists()
