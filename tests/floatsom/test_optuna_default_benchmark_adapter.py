"""
Tests for default benchmark adapter utilities.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/optuna_results_analysis/modules/parameter_analysis/default_benchmark_adapter.py"
)


@pytest.fixture(scope="module")
def adapter_module():
    if not _MODULE_PATH.exists():
        pytest.fail(f"Missing module under test: {_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("default_benchmark_adapter_test", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for default_benchmark_adapter.py")

    analysis_repo_root = _MODULE_PATH.parents[2]
    if str(analysis_repo_root) not in sys.path:
        sys.path.insert(0, str(analysis_repo_root))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_default_runs_csv_filters_floatsom_and_normalizes(tmp_path, adapter_module):
    csv_path = tmp_path / "defaults.csv"
    pd.DataFrame(
        [
            {
                "method": "xpysom",
                "dataset": "blobs",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "evaluation_split": "both",
                "quantization_error_holdout": 0.6,
                "quantization_error_train": 0.5,
                "balanced_qe_raw": 0.55,
            },
            {
                "method": "floatsom",
                "dataset": "blobs",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "batch_mode": "full",
                "evaluation_split": "both",
                "trial_number": 0,
                "quantization_error_holdout": 0.4,
                "quantization_error_train": 0.3,
                "balanced_qe_raw": 0.35,
            },
        ]
    ).to_csv(csv_path, index=False)

    out = adapter_module.load_default_runs_csv(csv_path, split_policy="both")

    assert len(out) == 1
    row = out.iloc[0]
    assert row["pair_dataset"] == "blobs"
    assert row["pair_topology"] == "hexagonal"
    assert row["pair_sampling"] == "full"
    assert row["pair_processing"] == "batch"
    assert row["pair_batch_mode"] == "full_batch"
    assert row["default_trial_number"] == pytest.approx(0.0)
    assert row["quantization_error_holdout"] == pytest.approx(0.4)


def test_load_default_runs_csv_rejects_duplicate_keys(tmp_path, adapter_module):
    csv_path = tmp_path / "defaults_dupe.csv"
    pd.DataFrame(
        [
            {
                "dataset": "blobs",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "evaluation_split": "both",
                "quantization_error_holdout": 0.5,
            },
            {
                "dataset": "blobs",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "evaluation_split": "both",
                "quantization_error_holdout": 0.4,
            },
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="duplicate rows"):
        adapter_module.load_default_runs_csv(csv_path, split_policy="both")
