"""
Tests for tuned-vs-default paired t-test utilities.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/optuna_results_analysis/modules/parameter_analysis/tuned_vs_default.py"
)


@pytest.fixture(scope="module")
def tuned_default_module():
    if not _MODULE_PATH.exists():
        pytest.fail(f"Missing module under test: {_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("tuned_vs_default_test", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for tuned_vs_default.py")

    analysis_repo_root = _MODULE_PATH.parents[2]
    if str(analysis_repo_root) not in sys.path:
        sys.path.insert(0, str(analysis_repo_root))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_metric_pairs_selects_default_and_best(tuned_default_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.55,
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "trial_number": 1,
                "quantization_error_holdout": 0.45,
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.60,
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 2,
                "trial_number": 2,
                "quantization_error_holdout": 0.50,
            },
        ]
    )

    metric = tuned_default_module.TunedDefaultMetric(
        column="quantization_error_holdout",
        label="QE Holdout",
        slug="qe_holdout",
    )
    pairs, _ = tuned_default_module.build_metric_pairs(df, metric)

    assert len(pairs) == 2
    assert sorted(pairs["default_trial_number"].tolist()) == [0.0, 0.0]
    assert sorted(pairs["tuned_value"].tolist()) == pytest.approx([0.45, 0.50])
    assert sorted(pairs["default_value"].tolist()) == pytest.approx([0.55, 0.60])

    deltas = pairs["delta_tuned_minus_default"].tolist()
    assert all(delta < 0 for delta in deltas)


def test_build_metric_pairs_against_defaults_reports_key_diagnostics(tuned_default_module):
    tuned_df = pd.DataFrame(
        [
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "evaluation_split": "both",
                "trial_number": 0,
                "quantization_error_holdout": 0.60,
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "evaluation_split": "both",
                "trial_number": 2,
                "quantization_error_holdout": 0.40,
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "full",
                "batch_mode": "full_batch",
                "config_topology_type": "mst",
                "seed": 1,
                "evaluation_split": "both",
                "trial_number": 0,
                "quantization_error_holdout": 0.70,
            },
        ]
    )
    default_df = pd.DataFrame(
        [
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "full",
                "pair_batch_mode": "full_batch",
                "pair_topology": "hexagonal",
                "pair_seed": "1",
                "pair_split": "both",
                "default_trial_number": 0.0,
                "quantization_error_holdout": 0.62,
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "full",
                "pair_batch_mode": "full_batch",
                "pair_topology": "rng",
                "pair_seed": "1",
                "pair_split": "both",
                "default_trial_number": 0.0,
                "quantization_error_holdout": 0.58,
            },
        ]
    )

    metric = tuned_default_module.TunedDefaultMetric(
        column="quantization_error_holdout",
        label="QE Holdout",
        slug="qe_holdout",
    )
    pairs, _, tuned_only, default_only = tuned_default_module.build_metric_pairs_against_defaults(
        tuned_df=tuned_df,
        default_df=default_df,
        metric=metric,
        split_policy="both",
    )

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["pair_topology"] == "hexagonal"
    assert row["default_value"] == pytest.approx(0.62)
    assert row["tuned_value"] == pytest.approx(0.40)
    assert len(tuned_only) == 1
    assert tuned_only.iloc[0]["pair_topology"] == "mst"
    assert len(default_only) == 1
    assert default_only.iloc[0]["pair_topology"] == "rng"


def test_apply_split_policy_keeps_rows_for_synthetic_all_marker(tuned_default_module):
    df = pd.DataFrame(
        [
            {"pair_split": "all", "value": 1},
            {"pair_split": "all", "value": 2},
        ]
    )

    out = tuned_default_module._apply_split_policy(df, split_value="both")
    assert len(out) == 2
    assert out["value"].tolist() == [1, 2]


def test_summarize_paired_value_table_includes_pct_improvement_stats(tuned_default_module):
    pairs = pd.DataFrame(
        [
            {
                "pair_dataset": "blobs",
                "default_value": 10.0,
                "tuned_value": 8.0,
                "pct_improvement": 20.0,
            },
            {
                "pair_dataset": "blobs",
                "default_value": 20.0,
                "tuned_value": 18.0,
                "pct_improvement": 10.0,
            },
        ]
    )

    summary = tuned_default_module.summarize_paired_value_table(
        pairs,
        "pair_dataset",
        higher_is_better=False,
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert float(row["mean_delta"]) == pytest.approx(-2.0)
    assert float(row["mean_pct"]) == pytest.approx(15.0)
    assert float(row["mean_pct_improvement"]) == pytest.approx(15.0)
    assert float(row["median_pct_improvement"]) == pytest.approx(15.0)
    assert float(row["ci_low_pct"]) < float(row["mean_pct"])
    assert float(row["ci_high_pct"]) > float(row["mean_pct"])
