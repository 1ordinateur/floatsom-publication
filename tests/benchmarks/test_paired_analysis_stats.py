"""
Unit tests for paired-analysis Wilcoxon-compatible interval helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/optuna/optuna_results_analysis/modules/core_analysis/paired_analysis.py"
)


@pytest.fixture(scope="module")
def paired_analysis_module():
    if not _MODULE_PATH.exists():
        pytest.fail(f"Missing module under test: {_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("paired_analysis_test", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for paired_analysis.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wilcoxon_location_ci_degenerate_for_constant_nonzero(paired_analysis_module):
    values = np.array([1.75, 1.75, 1.75], dtype=float)
    location, ci_low, ci_high = paired_analysis_module._wilcoxon_location_ci(values)

    assert location == pytest.approx(1.75)
    assert ci_low == pytest.approx(1.75)
    assert ci_high == pytest.approx(1.75)


def test_compute_delta_stats_uses_wilcoxon_location_ci(monkeypatch, paired_analysis_module):
    captured_inputs: list[np.ndarray] = []

    def _capture_ci(values, alpha=0.05):
        captured_inputs.append(np.asarray(values, dtype=float).copy())
        return 1.25, 0.5, 2.0

    monkeypatch.setattr(paired_analysis_module, "_wilcoxon_location_ci", _capture_ci)

    stats_dict = paired_analysis_module._compute_delta_stats(
        deltas=np.array([1.0, 2.0, np.nan], dtype=float),
        bootstrap_iterations=999,
        seed=123,
    )

    assert len(captured_inputs) == 1
    assert captured_inputs[0] == pytest.approx(np.array([1.0, 2.0], dtype=float))
    assert stats_dict["median_delta"] == pytest.approx(1.25)
    assert stats_dict["median_ci_low"] == pytest.approx(0.5)
    assert stats_dict["median_ci_high"] == pytest.approx(2.0)
