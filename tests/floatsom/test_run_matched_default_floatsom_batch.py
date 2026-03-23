"""
Unit tests for matched default/fixed batch runner helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/run_matched_default_floatsom_batch.py"
)


@pytest.fixture(scope="module")
def matched_runner_module():
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Missing script under test: {_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("run_matched_default_floatsom_batch_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for run_matched_default_floatsom_batch.py")

    repo_root = _SCRIPT_PATH.parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_manual_fixed_params_parses_cli_and_json(tmp_path, matched_runner_module):
    json_path = tmp_path / "fixed_params.json"
    json_path.write_text(
        '{"initial_radius": 2.75, "use_momentum": "false"}',
        encoding="utf-8",
    )

    resolved = matched_runner_module._resolve_manual_fixed_params(
        cli_entries=["initialization_method=pca"],
        json_path=str(json_path),
    )

    assert resolved["initial_radius"] == pytest.approx(2.75)
    assert resolved["use_momentum"] is False
    assert resolved["initialization_method"] == "pca"


def test_resolve_manual_fixed_params_rejects_structural_keys(matched_runner_module):
    with pytest.raises(ValueError, match="cannot be provided via fixed params"):
        matched_runner_module._resolve_manual_fixed_params(
            cli_entries=["topology_type=hexagonal"],
            json_path=None,
        )


def test_validate_manual_fixed_params_for_scope_rejects_non_batch_param(matched_runner_module):
    with pytest.raises(ValueError, match="not applicable to processing_method='batch'"):
        matched_runner_module._validate_manual_fixed_params_for_scope(
            {"learning_rate": 0.25},
            processing_method="batch",
            topologies=["hexagonal"],
        )


def test_validate_csv_name_rejects_non_csv(matched_runner_module):
    with pytest.raises(ValueError, match="must end with .csv"):
        matched_runner_module._validate_csv_name("matched_default_runs.txt", arg_name="--runs-csv-name")


def test_resolve_manual_fixed_params_rejects_non_integer_for_int_param(matched_runner_module):
    with pytest.raises(ValueError, match="Invalid integer value for 'chunk_size'"):
        matched_runner_module._resolve_manual_fixed_params(
            cli_entries=["chunk_size=1000.5"],
            json_path=None,
        )


def test_resolve_manual_fixed_params_by_topology_parses_json(tmp_path, matched_runner_module):
    json_path = tmp_path / "by_topology.json"
    json_path.write_text(
        (
            "{"
            "\"hexagonal\": {\"initial_radius\": 3.0, \"use_momentum\": \"true\"}, "
            "\"mst\": {\"topology_variant\": \"planar\"}"
            "}"
        ),
        encoding="utf-8",
    )

    resolved = matched_runner_module._resolve_manual_fixed_params_by_topology(
        topologies=["hexagonal", "mst"],
        json_path=str(json_path),
    )

    assert resolved["hexagonal"]["initial_radius"] == pytest.approx(3.0)
    assert resolved["hexagonal"]["use_momentum"] is True
    assert resolved["mst"]["topology_variant"] == "planar"


def test_resolve_manual_fixed_params_by_topology_rejects_unselected_topology(tmp_path, matched_runner_module):
    json_path = tmp_path / "by_topology.json"
    json_path.write_text(
        "{\"rng\": {\"initial_radius\": 2.0}}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configured for"):
        matched_runner_module._resolve_manual_fixed_params_by_topology(
            topologies=["hexagonal", "mst"],
            json_path=str(json_path),
        )


def test_resolve_effective_manual_fixed_params_applies_topology_override(matched_runner_module):
    effective = matched_runner_module._resolve_effective_manual_fixed_params(
        manual_fixed_params={
            "initial_radius": 2.0,
            "use_momentum": False,
        },
        topology_fixed_params_by_topology={
            "hexagonal": {
                "initial_radius": 5.0,
            }
        },
        sampling_topology_fixed_params={},
        topology="hexagonal",
        sampling_method="full",
    )

    assert effective["initial_radius"] == pytest.approx(5.0)
    assert effective["use_momentum"] is False


def test_resolve_manual_fixed_params_by_sampling_topology_parses_json(tmp_path, matched_runner_module):
    json_path = tmp_path / "sampling_topology.json"
    json_path.write_text(
        (
            "{"
            "\"full\": {\"hexagonal\": {\"initial_radius\": 1.0}}, "
            "\"random\": {\"mst\": {\"use_momentum\": \"false\"}}"
            "}"
        ),
        encoding="utf-8",
    )

    resolved = matched_runner_module._resolve_manual_fixed_params_by_sampling_topology(
        sampling_methods=["full", "random"],
        topologies=["hexagonal", "mst"],
        json_path=str(json_path),
    )

    assert resolved["full"]["hexagonal"]["initial_radius"] == pytest.approx(1.0)
    assert resolved["random"]["mst"]["use_momentum"] is False


def test_resolve_manual_fixed_params_by_sampling_topology_rejects_unselected_sampling(tmp_path, matched_runner_module):
    json_path = tmp_path / "sampling_topology.json"
    json_path.write_text(
        "{\"hdsssom\": {\"hexagonal\": {\"initial_radius\": 1.0}}}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configured for"):
        matched_runner_module._resolve_manual_fixed_params_by_sampling_topology(
            sampling_methods=["full", "random"],
            topologies=["hexagonal"],
            json_path=str(json_path),
        )


def test_resolve_manual_fixed_params_by_sampling_topology_filters_unselected_keys_when_enabled(
    tmp_path,
    matched_runner_module,
):
    json_path = tmp_path / "sampling_topology.json"
    json_path.write_text(
        (
            "{"
            "\"full\": {\"hexagonal\": {\"initial_radius\": 1.0}, \"rng\": {\"use_momentum\": \"true\"}}, "
            "\"random\": {\"hexagonal\": {\"initial_radius\": 2.0}}"
            "}"
        ),
        encoding="utf-8",
    )

    resolved = matched_runner_module._resolve_manual_fixed_params_by_sampling_topology(
        sampling_methods=["full"],
        topologies=["hexagonal"],
        json_path=str(json_path),
        allow_unselected_keys=True,
    )

    assert resolved == {
        "full": {
            "hexagonal": {
                "initial_radius": pytest.approx(1.0),
            }
        }
    }


def test_resolve_effective_manual_fixed_params_applies_sampling_topology_override(matched_runner_module):
    effective = matched_runner_module._resolve_effective_manual_fixed_params(
        manual_fixed_params={
            "initial_radius": 2.0,
            "use_momentum": False,
        },
        topology_fixed_params_by_topology={
            "hexagonal": {
                "initial_radius": 5.0,
            }
        },
        sampling_topology_fixed_params={
            "random": {
                "hexagonal": {
                    "initial_radius": 7.0,
                    "use_momentum": True,
                }
            }
        },
        topology="hexagonal",
        sampling_method="random",
    )

    assert effective["initial_radius"] == pytest.approx(7.0)
    assert effective["use_momentum"] is True


def test_resolve_seeds_from_csv_uses_pair_seed_column(tmp_path, matched_runner_module):
    seeds_csv = tmp_path / "seed_source.csv"
    seeds_csv.write_text(
        "pair_seed,seed,other\n11780,1,x\n24458,2,y\n11780,3,z\n",
        encoding="utf-8",
    )

    resolved = matched_runner_module._resolve_seeds_from_csv(seeds_csv)

    assert resolved == [11780, 24458]


def test_resolve_requested_seeds_rejects_explicit_and_csv(tmp_path, matched_runner_module):
    seeds_csv = tmp_path / "seed_source.csv"
    seeds_csv.write_text("seed\n42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="either --seeds or --seeds-from-csv"):
        matched_runner_module._resolve_requested_seeds(
            explicit_seeds=[1, 2],
            num_seeds=5,
            base_seed=42,
            seeds_from_csv=str(seeds_csv),
        )


def test_dedupe_runs_df_by_key_drops_duplicates(matched_runner_module):
    import pandas as pd

    df = pd.DataFrame(
        [
            {"dataset": "iris", "seed": 1, "architecture": "hexagonal", "sampling_method": "full", "v": 1},
            {"dataset": "iris", "seed": 1, "architecture": "hexagonal", "sampling_method": "full", "v": 2},
            {"dataset": "wine", "seed": 2, "architecture": "mst", "sampling_method": "random", "v": 3},
        ]
    )

    deduped = matched_runner_module._dedupe_runs_df_by_key(df)

    assert len(deduped) == 2
    assert set(deduped["dataset"].tolist()) == {"iris", "wine"}


def test_extract_completed_run_keys_normalizes_case(matched_runner_module):
    import pandas as pd

    df = pd.DataFrame(
        [
            {"dataset": "Iris ", "seed": 42.0, "architecture": "Hexagonal", "sampling_method": " Full "},
            {"dataset": "wine", "seed": 7, "architecture": "mst", "sampling_method": "random"},
        ]
    )

    keys = matched_runner_module._extract_completed_run_keys(df)

    assert ("iris", 42, "hexagonal", "full") in keys
    assert ("wine", 7, "mst", "random") in keys


def test_validate_resume_profile_compatibility_rejects_profile_mismatch(matched_runner_module):
    import pandas as pd

    existing = pd.DataFrame(
        [
            {"run_profile": "true_default", "evaluation_split": "both"},
        ]
    )

    with pytest.raises(ValueError, match="different profile"):
        matched_runner_module._validate_resume_profile_compatibility(
            existing_df=existing,
            expected_profile="manual_fixed",
            expected_split="both",
        )
