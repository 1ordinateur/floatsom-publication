"""
Regression tests for optuna scenario-id parsing in visualization helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/optuna_results_analysis/modules/core_analysis/visualization_helpers.py"
)


@pytest.fixture(scope="module")
def visualization_helpers_module():
    if not _MODULE_PATH.exists():
        pytest.fail(f"Missing module under test: {_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("visualization_helpers_test", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for visualization_helpers.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_scenario_full_batch_tokenized_variant(visualization_helpers_module):
    parsed = visualization_helpers_module.parse_scenario("blobs_colors_full_full_batch_hexagonal")
    assert parsed == ("blobs", "colors", "full", "hexagonal")


def test_parse_scenario_minibatch_variant(visualization_helpers_module):
    parsed = visualization_helpers_module.parse_scenario("blobs_batch_random_minibatch_rng")
    assert parsed == ("blobs", "batch", "random", "rng")


def test_parse_scenario_legacy_variant(visualization_helpers_module):
    parsed = visualization_helpers_module.parse_scenario("blobs_colors_full_hexagonal")
    assert parsed == ("blobs", "colors", "full", "hexagonal")
