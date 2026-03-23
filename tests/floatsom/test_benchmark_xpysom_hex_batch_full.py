"""
Unit tests for XPySOM-vs-FloatSOM benchmark parameter resolution helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("cupy")
pytest.importorskip("optuna")
pytest.importorskip("pandas")
pytest.importorskip("scipy")


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/benchmark_xpysom_hex_batch_full.py"
)


@pytest.fixture(scope="module")
def xpysom_benchmark_module():
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Missing script under test: {_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("benchmark_xpysom_hex_batch_full_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for benchmark_xpysom_hex_batch_full.py")

    repo_root = _SCRIPT_PATH.parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_floatsom_fixed_param_overrides_uses_matched_runner_precedence(
    tmp_path,
    xpysom_benchmark_module,
):
    by_topology_path = tmp_path / "by_topology.json"
    by_topology_path.write_text(
        '{"rng": {"initial_radius": 4.0}}',
        encoding="utf-8",
    )
    by_sampling_topology_path = tmp_path / "by_sampling_topology.json"
    by_sampling_topology_path.write_text(
        '{"full": {"rng": {"initialization_method": "pca", "use_momentum": true}}}',
        encoding="utf-8",
    )

    manual_fixed_params, by_topology, by_sampling_topology = (
        xpysom_benchmark_module._resolve_floatsom_fixed_param_overrides(
            topologies=["hexagonal", "rng"],
            fixed_param_entries=["radius_decay_type=asymptotic"],
            fixed_params_json=None,
            fixed_params_by_topology_json=str(by_topology_path),
            fixed_params_by_sampling_topology_json=str(by_sampling_topology_path),
        )
    )

    effective = xpysom_benchmark_module._resolve_effective_manual_fixed_params(
        manual_fixed_params=manual_fixed_params,
        topology_fixed_params_by_topology=by_topology,
        sampling_topology_fixed_params=by_sampling_topology,
        topology="rng",
        sampling_method="full",
    )

    assert effective["radius_decay_type"] == "asymptotic"
    assert effective["initial_radius"] == pytest.approx(4.0)
    assert effective["initialization_method"] == "pca"
    assert effective["use_momentum"] is True


def test_resolve_floatsom_fixed_param_overrides_filters_other_sampling_configs(
    tmp_path,
    xpysom_benchmark_module,
):
    by_sampling_topology_path = tmp_path / "by_sampling_topology.json"
    by_sampling_topology_path.write_text(
        (
            "{"
            "\"full\": {\"hexagonal\": {\"initial_radius\": 1.0}}, "
            "\"random\": {\"hexagonal\": {\"initial_radius\": 2.0}}"
            "}"
        ),
        encoding="utf-8",
    )

    _, _, by_sampling_topology = xpysom_benchmark_module._resolve_floatsom_fixed_param_overrides(
        topologies=["hexagonal"],
        fixed_param_entries=None,
        fixed_params_json=None,
        fixed_params_by_topology_json=None,
        fixed_params_by_sampling_topology_json=str(by_sampling_topology_path),
    )

    assert by_sampling_topology == {
        "full": {
            "hexagonal": {
                "initial_radius": pytest.approx(1.0),
            }
        }
    }


def test_floatsom_run_profile_label_matches_matched_runner_semantics(xpysom_benchmark_module):
    assert (
        xpysom_benchmark_module._floatsom_run_profile_label(
            manual_fixed_params={},
            manual_fixed_params_by_topology={},
            manual_fixed_params_by_sampling_topology={},
        )
        == "true_default"
    )
    assert (
        xpysom_benchmark_module._floatsom_run_profile_label(
            manual_fixed_params={"initial_radius": 3.0},
            manual_fixed_params_by_topology={},
            manual_fixed_params_by_sampling_topology={},
        )
        == "manual_fixed"
    )


def test_create_floatsom_batch_topology_preserves_structural_benchmark_settings(
    monkeypatch,
    xpysom_benchmark_module,
):
    captured = {}

    def fake_create_floatsom(params):
        captured["params"] = params
        return object()

    monkeypatch.setattr(xpysom_benchmark_module, "create_floatsom", fake_create_floatsom, raising=True)

    _, params = xpysom_benchmark_module._create_floatsom_batch_topology(
        input_dim=6,
        seed=11780,
        grid_size=10,
        epochs=20,
        topology_type="rng",
        manual_fixed_params={
            "initial_radius": 3.5,
            "initialization_method": "pca",
            "min_iterations": 7,
        },
    )

    assert captured["params"] is params
    assert params.topology_config.topology_type == "rng"
    assert params.topology_config.grid_size == 10
    assert params.sampling_config.method == "full"
    assert params.processing_config.method == "batch"
    assert params.processing_config.batch_mode == "full_batch"
    assert params.total_iterations == 20
    assert params.seed == 11780
    assert params.initial_radius == pytest.approx(3.5)
    assert params.initialization_method == "pca"
    assert params.min_iterations == 7
