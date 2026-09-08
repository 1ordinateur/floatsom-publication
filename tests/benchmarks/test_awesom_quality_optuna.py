from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark_module():
    return _load_module(
        "benchmarks/optuna/benchmark_awesom_quality.py",
        "benchmark_awesom_quality_test",
    )


@pytest.fixture(scope="module")
def table_module():
    return _load_module(
        "benchmarks/optuna/build_awesom_supplementary_table.py",
        "build_awesom_supplementary_table_test",
    )


def test_iris_loader_matches_publication_standardization(benchmark_module):
    data, feature_names = benchmark_module.load_standardized_dataset("iris")
    assert data.shape == (150, 4)
    assert data.dtype == np.float32
    assert len(feature_names) == 4
    np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(data.std(axis=0), 1.0, atol=1e-6)


def test_split_matches_randomstate_policy(benchmark_module):
    data = np.arange(60, dtype=np.float32).reshape(20, 3)
    train, holdout = benchmark_module.split_train_holdout(data, seed=11780)
    indices = np.random.RandomState(11780).permutation(20)
    np.testing.assert_array_equal(train, data[indices[:14]])
    np.testing.assert_array_equal(holdout, data[indices[14:]])


def test_quantization_error_uses_euclidean_distance(benchmark_module):
    data = np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    weights = np.asarray([[0.0, 0.0]], dtype=np.float32)
    observed = benchmark_module.quantization_error(data, weights, chunk_rows=1)
    assert observed == pytest.approx(2.5)


def test_awesom_rng_adapter_is_reproducible_and_restored(benchmark_module):
    original = np.random.default_rng
    with benchmark_module.deterministic_awesom_randomness(1234):
        first = np.random.default_rng().integers(0, 1000, size=8)
    with benchmark_module.deterministic_awesom_randomness(1234):
        second = np.random.default_rng().integers(0, 1000, size=8)
    np.testing.assert_array_equal(first, second)
    assert np.random.default_rng is original


def _trial_rows(implementation: str, topology: str, dataset: str, seeds):
    rows = []
    for seed in seeds:
        for trial in range(8):
            base = 1.0 if implementation == "aweSOM" else 0.8
            rows.append(
                {
                    "implementation": implementation,
                    "architecture": topology,
                    "dataset": dataset,
                    "seed": seed,
                    "trial_number": trial,
                    "state": "complete",
                    "balanced_qe_raw": base + 0.01 * trial,
                    "quantization_error_holdout": base + 0.02 + 0.01 * trial,
                    "quantization_error_train": base - 0.02 + 0.01 * trial,
                }
            )
    return rows


def test_supplementary_table_uses_top_five_and_matched_effects(table_module):
    awe = pd.DataFrame(
        _trial_rows("aweSOM", "rectangular_chebyshev", "iris", [11780, 24458])
    )
    floatsom = pd.DataFrame(
        _trial_rows("FloatSOM", "hexagonal", "iris", [11780, 24458])
    )
    floatsom["sampling_method_final"] = "full"
    floatsom["processing_type"] = "batch"
    table = table_module.build_table(awe, floatsom=floatsom, top_k=5)
    assert list(table["implementation"]) == ["aweSOM", "FloatSOM"]
    assert list(table["top_k"]) == [5, 5]
    float_row = table[table["implementation"] == "FloatSOM"].iloc[0]
    assert int(float_row["matched_seed_count_vs_awesom"]) == 2
    assert float_row["balanced_qe_improvement_vs_awesom_pct"] > 0
