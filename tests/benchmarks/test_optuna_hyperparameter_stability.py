"""
Tests for hyperparameter stability utilities.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/optuna/optuna_results_analysis/modules/parameter_analysis/hyperparameter_stability.py"
)


@pytest.fixture(scope="module")
def stability_module():
    if not _MODULE_PATH.exists():
        pytest.fail(f"Missing module under test: {_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("hyperparameter_stability_test", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for hyperparameter_stability.py")

    analysis_repo_root = _MODULE_PATH.parents[2]
    if str(analysis_repo_root) not in sys.path:
        sys.path.insert(0, str(analysis_repo_root))

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_topology_param_pairs_computes_differences(stability_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.5,
                "param_lr": 0.1,
                "param_mode": "a",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "mst",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.6,
                "param_lr": 0.2,
                "param_mode": "b",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.4,
                "param_lr": 0.1,
                "param_mode": "a",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "mst",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.7,
                "param_lr": 0.3,
                "param_mode": "a",
            },
        ]
    )

    metric = stability_module.StabilityMetric(
        column="quantization_error_holdout",
        label="QE Holdout",
        slug="qe_holdout",
    )
    config = stability_module.StabilityConfig()
    working, key_cols, _ = stability_module._resolve_pairing_columns(df, config)
    best_rows = stability_module._select_best_trials(working, metric, key_cols)
    numeric_cols, categorical_cols = stability_module._split_param_columns(
        working, ["param_lr", "param_mode"]
    )
    pairs = stability_module.build_topology_param_pairs(
        best_df=best_rows,
        key_cols=key_cols,
        topology_pair=("hexagonal", "mst"),
        param_columns=["param_lr", "param_mode"],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    assert len(pairs) == 2
    abs_diffs = sorted(pairs["numeric_stability_abs_diff_mean"].tolist())
    assert abs_diffs == pytest.approx([0.1, 0.2])

    mismatch_rates = sorted(pairs["categorical_stability_mismatch_rate"].tolist())
    assert mismatch_rates == pytest.approx([0.0, 1.0])

    # Per-parameter detail columns
    param_abs = sorted(pairs["param_lr_abs_diff"].tolist())
    assert param_abs == pytest.approx([0.1, 0.2])
    param_rel = sorted(pairs["param_lr_rel_diff"].tolist())
    assert param_rel == pytest.approx([0.5, (0.2 / 0.3)])
    param_delta = sorted(pairs["param_lr_delta_left_minus_right"].tolist())
    assert param_delta == pytest.approx([-0.2, -0.1])
    mode_mismatch = sorted(pairs["param_mode_mismatch"].tolist())
    assert mode_mismatch == pytest.approx([0.0, 1.0])


def test_parameter_stability_comparison_table_reports_per_parameter_and_overall_rows(stability_module):
    topology_stability_pairs = pd.DataFrame(
        [
            {
                "pair_dataset": "blobs",
                "pair_topology": "hexagonal",
                "param_lr_rel_diff": 0.2,
                "param_mode_mismatch": 0.0,
            },
            {
                "pair_dataset": "blobs",
                "pair_topology": "hexagonal",
                "param_lr_rel_diff": 0.4,
                "param_mode_mismatch": 1.0,
            },
            {
                "pair_dataset": "blobs",
                "pair_topology": "mst",
                "param_lr_rel_diff": 0.1,
                "param_mode_mismatch": 1.0,
            },
            {
                "pair_dataset": "blobs",
                "pair_topology": "mst",
                "param_lr_rel_diff": 0.3,
                "param_mode_mismatch": 1.0,
            },
        ]
    )
    per_topology = stability_module._parameter_topology_stability_table(
        topology_stability_pairs=topology_stability_pairs,
        group_col="pair_dataset",
        numeric_cols=["param_lr"],
        categorical_cols=["param_mode"],
    )
    summary = stability_module._parameter_stability_comparison_table(
        parameter_topology_stability=per_topology,
        group_col="pair_dataset",
        topology_left="hexagonal",
        topology_right="mst",
    )

    assert not summary.empty
    assert set(summary["parameter"]) == {"param_lr", "param_mode", "OVERALL"}

    numeric_row = summary[summary["parameter"] == "param_lr"].iloc[0]
    categorical_row = summary[summary["parameter"] == "param_mode"].iloc[0]
    overall_row = summary[summary["parameter"] == "OVERALL"].iloc[0]

    assert numeric_row["parameter_type"] == "numeric"
    assert numeric_row["stability_score_mean_left"] == pytest.approx(0.3)
    assert numeric_row["stability_score_mean_right"] == pytest.approx(0.2)
    assert numeric_row["winner"] == "mst"

    assert categorical_row["parameter_type"] == "categorical"
    assert categorical_row["stability_score_mean_left"] == pytest.approx(0.5)
    assert categorical_row["stability_score_mean_right"] == pytest.approx(1.0)
    assert categorical_row["winner"] == "hexagonal"

    # OVERALL row is equal-weight across parameters in the scope.
    assert overall_row["parameter_type"] == "overall"
    assert overall_row["stability_score_mean_left"] == pytest.approx(0.4)
    assert overall_row["stability_score_mean_right"] == pytest.approx(0.6)
    assert overall_row["winner"] == "hexagonal"
    assert int(overall_row["parameters_compared"]) == 2


def test_build_topology_stability_pairs_supports_mst_and_rng(stability_module):
    best_rows = pd.DataFrame(
        [
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "1",
                "pair_topology": "hexagonal",
                "param_lr": 0.10,
                "param_mode": "a",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "2",
                "pair_topology": "hexagonal",
                "param_lr": 0.20,
                "param_mode": "b",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "3",
                "pair_topology": "hexagonal",
                "param_lr": 0.30,
                "param_mode": "b",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "1",
                "pair_topology": "mst",
                "param_lr": 0.45,
                "param_mode": "a",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "2",
                "pair_topology": "mst",
                "param_lr": 0.55,
                "param_mode": "a",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "1",
                "pair_topology": "rng",
                "param_lr": 0.75,
                "param_mode": "c",
            },
            {
                "pair_dataset": "blobs",
                "pair_processing": "batch",
                "pair_sampling": "random",
                "pair_seed": "2",
                "pair_topology": "rng",
                "param_lr": 0.85,
                "param_mode": "c",
            },
        ]
    )

    out = stability_module.build_topology_stability_pairs(
        best_df=best_rows,
        key_cols=["pair_dataset", "pair_processing", "pair_sampling", "pair_seed"],
        topologies=["hexagonal", "mst", "rng"],
        param_columns=["param_lr", "param_mode"],
        numeric_cols=["param_lr"],
        categorical_cols=["param_mode"],
    )

    assert not out.empty
    assert set(out["pair_topology"]) == {"hexagonal", "mst", "rng"}
    assert len(out[out["pair_topology"] == "hexagonal"]) == 3
    assert len(out[out["pair_topology"] == "mst"]) == 1
    assert len(out[out["pair_topology"] == "rng"]) == 1


def test_build_hyperparameter_stability_report_writes_per_parameter_outputs_for_rng(tmp_path, stability_module):
    df = pd.DataFrame(
        [
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.40,
                "param_lr": 0.10,
                "param_mode": "a",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "hexagonal",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.41,
                "param_lr": 0.15,
                "param_mode": "a",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "mst",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.50,
                "param_lr": 0.30,
                "param_mode": "b",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "mst",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.51,
                "param_lr": 0.35,
                "param_mode": "b",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "rng",
                "seed": 1,
                "trial_number": 0,
                "quantization_error_holdout": 0.60,
                "param_lr": 0.70,
                "param_mode": "c",
            },
            {
                "dataset": "blobs",
                "processing_type": "batch",
                "sampling_method": "random",
                "config_topology_type": "rng",
                "seed": 2,
                "trial_number": 0,
                "quantization_error_holdout": 0.61,
                "param_lr": 0.90,
                "param_mode": "d",
            },
        ]
    )

    metric = stability_module.StabilityMetric(
        column="quantization_error_holdout",
        label="QE Holdout",
        slug="qe_holdout",
    )
    report_path = stability_module.build_hyperparameter_stability_report(
        df=df,
        metric=metric,
        output_dir=tmp_path,
        topology_pairs=(("hexagonal", "mst"), ("hexagonal", "rng")),
    )

    assert report_path.exists()
    markdown_text = report_path.read_text(encoding="utf-8")
    assert "Per-Parameter Stability Comparison (Dataset, lower is better)" in markdown_text
    assert "Per-Parameter Stability Comparison (Dataset Type, lower is better)" in markdown_text
    assert "Per-Parameter Stability Comparison (Global, lower is better)" in markdown_text
    assert "Stability Winner (Dataset, lower is better)" not in markdown_text
    assert "Topology-Specific Stability (Dataset, lower is better)" not in markdown_text
    assert "Stability Summary (Dataset)" not in markdown_text

    parameter_global_path = tmp_path / "tables" / stability_module.stability_table_filename(
        "global", "hexagonal", "rng", "qe_holdout"
    )
    assert parameter_global_path.exists()
    parameter_global = pd.read_csv(parameter_global_path)
    assert "stability_score_mean_left" in parameter_global.columns
    assert "stability_score_mean_right" in parameter_global.columns
    assert "mean_value_left" not in parameter_global.columns
    assert "mean_value_right" not in parameter_global.columns
    assert (parameter_global["parameter"] == "OVERALL").any()

    parameter_dataset_path = tmp_path / "tables" / stability_module.stability_table_filename(
        "dataset", "hexagonal", "rng", "qe_holdout"
    )
    parameter_dataset_type_path = tmp_path / "tables" / stability_module.stability_table_filename(
        "dataset_type", "hexagonal", "rng", "qe_holdout"
    )
    assert parameter_dataset_path.exists()
    assert parameter_dataset_type_path.exists()

    winner_rng_path = (
        tmp_path / "tables" / "hyperparam_stability_winner_dataset_hexagonal_vs_rng_qe_holdout.csv"
    )
    topology_dataset_path = (
        tmp_path / "tables" / "hyperparam_stability_topology_dataset_hexagonal_vs_rng_qe_holdout.csv"
    )
    dataset_summary_path = (
        tmp_path / "tables" / "hyperparam_stability_dataset_hexagonal_vs_rng_qe_holdout.csv"
    )
    assert not winner_rng_path.exists()
    assert not topology_dataset_path.exists()
    assert not dataset_summary_path.exists()
