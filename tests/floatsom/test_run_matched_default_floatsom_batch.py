"""
Unit tests for matched default/fixed batch runner helpers.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "floatsom/benchmarks/optuna/run_matched_default_floatsom_batch.py"
)
if not _SCRIPT_PATH.exists():
    _SCRIPT_PATH = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/optuna/run_matched_default_floatsom_batch.py"
    )


@pytest.fixture(scope="module")
def matched_runner_module():
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Missing script under test: {_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("run_matched_default_floatsom_batch_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for run_matched_default_floatsom_batch.py")

    repo_root = _SCRIPT_PATH.parents[2] if (_SCRIPT_PATH.parents[2] / "pyproject.toml").exists() else _SCRIPT_PATH.parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    previous_floatsom_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "floatsom" or name.startswith("floatsom.")
    }
    previous_optuna = sys.modules.get("optuna")
    inserted_optuna_stub = False
    for module_name in list(sys.modules):
        if module_name == "floatsom" or module_name.startswith("floatsom."):
            del sys.modules[module_name]
    package = types.ModuleType("floatsom")
    package.__path__ = [str(repo_root)]
    sys.modules["floatsom"] = package
    if "optuna" not in sys.modules and importlib.util.find_spec("optuna") is None:
        optuna_stub = types.ModuleType("optuna")
        optuna_stub.trial = types.SimpleNamespace(
            FrozenTrial=object,
            TrialState=types.SimpleNamespace(COMPLETE="COMPLETE"),
        )
        optuna_stub.Study = object
        sys.modules["optuna"] = optuna_stub
        inserted_optuna_stub = True

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module

    for module_name in list(sys.modules):
        if module_name == "floatsom" or module_name.startswith("floatsom."):
            del sys.modules[module_name]
    sys.modules.update(previous_floatsom_modules)
    if inserted_optuna_stub:
        sys.modules.pop("optuna", None)
    elif previous_optuna is not None:
        sys.modules["optuna"] = previous_optuna


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


def test_parse_additional_fixed_profile_specs_selects_variant_topologies(matched_runner_module):
    profiles = matched_runner_module._parse_additional_fixed_profile_specs(
        ["tuned_rng_random,floatsom_min1000_tuned_defaults_rng_random.json,rng"],
        default_topologies=["hexagonal", "mst", "rng"],
    )

    assert profiles == [
        {
            "label": "tuned_rng_random",
            "json_path": "floatsom_min1000_tuned_defaults_rng_random.json",
            "topologies": ["rng"],
        }
    ]


def test_parse_additional_fixed_profile_specs_rejects_reserved_label(matched_runner_module):
    with pytest.raises(ValueError, match="reserved"):
        matched_runner_module._parse_additional_fixed_profile_specs(
            ["tuned_fixed,floatsom_min1000_tuned_defaults_rng_random.json,rng"],
            default_topologies=["hexagonal", "mst", "rng"],
        )


def test_xpysom_untuned_defaults_json_resolves_for_true_default_profile(matched_runner_module):
    defaults_path = Path(matched_runner_module.__file__).resolve().parents[2] / "xpysom_untuned_defaults.json"

    resolved = matched_runner_module._resolve_manual_fixed_params_by_sampling_topology(
        sampling_methods=["full"],
        topologies=["hexagonal", "mst", "rng"],
        json_path=str(defaults_path),
        allow_unselected_keys=True,
    )

    for topology in ["hexagonal", "mst", "rng"]:
        params = resolved["full"][topology]
        assert params["initial_radius"] == pytest.approx(5.0)
        assert params["initialization_method"] == "random"
        assert params["momentum_init"] == pytest.approx(0.5)
        assert params["radius_decay_type"] == "exponential"
        assert params["use_momentum"] is False


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


def test_build_success_row_includes_mtr_and_balanced_diagnostics(matched_runner_module):
    class Trial:
        number = 0
        params = {}
        user_attrs = {
            "metrics_holdout": {
                "quantization_error": 2.0,
                "mean_tied_rank": 4.0,
                "mean_tied_rank_null_mean": 10.0,
                "mean_tied_rank_null_theoretical_mean": 10.0,
                "mean_tied_rank_null_std": 1.0,
                "mean_tied_rank_null_q025": 8.0,
                "mean_tied_rank_null_q975": 12.0,
                "mean_tied_rank_observed_to_null_ratio": 0.4,
                "mean_tied_rank_null_lower_tail_p": 0.001,
                "mean_tied_rank_null_permutations": 1000,
                "node_utilization": 0.5,
                "dead_node_fraction": 0.5,
                "used_nodes": 5,
                "dead_nodes": 5,
                "total_nodes": 10,
            },
            "metrics_train": {
                "quantization_error": 1.0,
                "mean_tied_rank": 2.0,
                "mean_tied_rank_null_mean": 10.0,
                "mean_tied_rank_null_theoretical_mean": 10.0,
                "mean_tied_rank_null_std": 1.0,
                "mean_tied_rank_null_q025": 8.0,
                "mean_tied_rank_null_q975": 12.0,
                "mean_tied_rank_observed_to_null_ratio": 0.2,
                "mean_tied_rank_null_lower_tail_p": 0.001,
                "mean_tied_rank_null_permutations": 1000,
                "node_utilization": 0.7,
                "dead_node_fraction": 0.3,
                "used_nodes": 7,
                "dead_nodes": 3,
                "total_nodes": 10,
            },
        }

    row = matched_runner_module._build_success_row(
        trial=Trial(),
        dataset="iris",
        seed=42,
        topology="hexagonal",
        sampling_method="full",
        evaluation_split="both",
        forced_params={
            "sampling_method": "full",
            "processing_method": "batch",
            "batch_mode": "full_batch",
            "topology_type": "hexagonal",
        },
        manual_fixed_params={},
        run_profile="true_default",
        run_label="true_default",
    )

    assert row["mean_tied_rank_holdout"] == pytest.approx(4.0)
    assert row["mean_tied_rank_train"] == pytest.approx(2.0)
    assert row["balanced_mean_tied_rank_raw"] == pytest.approx(3.0)
    assert row["balanced_mean_tied_rank_null_mean"] == pytest.approx(10.0)
    assert row["balanced_mean_tied_rank_observed_to_null_ratio"] == pytest.approx(0.3)
    assert row["balanced_node_utilization_raw"] == pytest.approx(0.6)
    assert row["balanced_dead_node_fraction_raw"] == pytest.approx(0.4)
    assert row["used_nodes_holdout"] == pytest.approx(5.0)
    assert row["total_nodes_train"] == pytest.approx(10.0)


def test_build_success_row_rejects_missing_required_diagnostics(matched_runner_module):
    class Trial:
        number = 0
        params = {}
        user_attrs = {
            "metrics_holdout": {
                "quantization_error": 2.0,
                "node_utilization": 0.5,
                "dead_node_fraction": 0.5,
                "used_nodes": 5,
                "dead_nodes": 5,
                "total_nodes": 10,
            },
            "metrics_train": {
                "quantization_error": 1.0,
                "node_utilization": 0.7,
                "dead_node_fraction": 0.3,
                "used_nodes": 7,
                "dead_nodes": 3,
                "total_nodes": 10,
            },
        }

    with pytest.raises(ValueError, match="required matched topology diagnostics"):
        matched_runner_module._build_success_row(
            trial=Trial(),
            dataset="iris",
            seed=42,
            topology="hexagonal",
            sampling_method="full",
            evaluation_split="both",
            forced_params={
                "sampling_method": "full",
                "processing_method": "batch",
                "batch_mode": "full_batch",
                "topology_type": "hexagonal",
            },
            manual_fixed_params={},
            run_profile="true_default",
            run_label="true_default",
        )


def test_resume_validation_rejects_pre_diagnostic_rows(matched_runner_module):
    import pandas as pd

    old_rows = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "quantization_error_holdout": 2.0,
                "quantization_error_train": 1.0,
                "balanced_qe_raw": 1.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required matched topology diagnostic columns"):
        matched_runner_module._validate_required_numeric_columns(
            old_rows,
            required_columns=matched_runner_module.REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS,
            context="Resume CSV test",
        )


def test_report_generation_rejects_pre_diagnostic_manifest_rows(tmp_path, matched_runner_module):
    import json
    import pandas as pd

    old_rows = pd.DataFrame(
        [
            {
                "dataset": "iris",
                "seed": 1,
                "architecture": "hexagonal",
                "sampling_method": "full",
                "quantization_error_holdout": 2.0,
                "quantization_error_train": 1.0,
                "balanced_qe_raw": 1.5,
            }
        ]
    )
    default_csv = tmp_path / "matched_default.csv"
    tuned_csv = tmp_path / "matched_tuned.csv"
    old_rows.to_csv(default_csv, index=False)
    old_rows.to_csv(tuned_csv, index=False)

    manifest_path = tmp_path / "MATCHED_TOPOLOGY_DIAGNOSTICS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default_runs_file": str(default_csv),
                "default_aware_tuned_runs_file": str(tuned_csv),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required matched topology diagnostic columns"):
        matched_runner_module._generate_matched_topology_diagnostic_report(
            manifest_path=manifest_path,
            output_dir=tmp_path,
            markdown_name="DIAGNOSTICS.md",
        )


def test_generate_matched_topology_diagnostic_report_from_manifest(tmp_path, matched_runner_module):
    import json
    import pandas as pd

    def make_row(profile, dataset, seed, topology, qe, mtr, utilization, dead_fraction):
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": topology,
            "sampling_method": "full",
            "run_profile": profile,
            "evaluation_split": "both",
            "quantization_error_holdout": qe + 0.1,
            "quantization_error_train": qe - 0.1,
            "balanced_qe_raw": qe,
            "mean_tied_rank_holdout": mtr + 0.1,
            "mean_tied_rank_train": mtr - 0.1,
            "balanced_mean_tied_rank_raw": mtr,
            "mean_tied_rank_null_mean_holdout": 50.0,
            "mean_tied_rank_null_mean_train": 50.0,
            "balanced_mean_tied_rank_null_mean": 50.0,
            "mean_tied_rank_null_theoretical_mean_holdout": 50.0,
            "mean_tied_rank_null_theoretical_mean_train": 50.0,
            "balanced_mean_tied_rank_null_theoretical_mean": 50.0,
            "mean_tied_rank_null_std_holdout": 1.0,
            "mean_tied_rank_null_std_train": 1.0,
            "mean_tied_rank_null_q025_holdout": 48.0,
            "mean_tied_rank_null_q025_train": 48.0,
            "mean_tied_rank_null_q975_holdout": 52.0,
            "mean_tied_rank_null_q975_train": 52.0,
            "mean_tied_rank_observed_to_null_ratio_holdout": (mtr + 0.1) / 50.0,
            "mean_tied_rank_observed_to_null_ratio_train": (mtr - 0.1) / 50.0,
            "balanced_mean_tied_rank_observed_to_null_ratio": mtr / 50.0,
            "mean_tied_rank_null_lower_tail_p_holdout": 0.001,
            "mean_tied_rank_null_lower_tail_p_train": 0.001,
            "mean_tied_rank_null_permutations_holdout": 1000,
            "mean_tied_rank_null_permutations_train": 1000,
            "node_utilization_holdout": utilization - 0.05,
            "node_utilization_train": utilization + 0.05,
            "balanced_node_utilization_raw": utilization,
            "dead_node_fraction_holdout": dead_fraction + 0.05,
            "dead_node_fraction_train": dead_fraction - 0.05,
            "balanced_dead_node_fraction_raw": dead_fraction,
        }

    default_rows = []
    tuned_rows = []
    for seed in (1, 2):
        default_rows.extend(
            [
                make_row("true_default", "iris", seed, "hexagonal", 10.0, 5.0, 0.50, 0.50),
                make_row("true_default", "iris", seed, "mst", 9.0, 4.0, 0.60, 0.40),
                make_row("true_default", "iris", seed, "rng", 8.0, 3.0, 0.70, 0.30),
            ]
        )
        tuned_rows.extend(
            [
                make_row("manual_fixed", "iris", seed, "hexagonal", 7.0, 4.5, 0.65, 0.35),
                make_row("manual_fixed", "iris", seed, "mst", 6.0, 3.5, 0.75, 0.25),
                make_row("manual_fixed", "iris", seed, "rng", 5.0, 2.5, 0.85, 0.15),
            ]
        )

    default_csv = tmp_path / "matched_default.csv"
    tuned_csv = tmp_path / "matched_tuned.csv"
    pd.DataFrame(default_rows).to_csv(default_csv, index=False)
    pd.DataFrame(tuned_rows).to_csv(tuned_csv, index=False)

    manifest_path = tmp_path / "MATCHED_TOPOLOGY_DIAGNOSTICS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default_runs_file": str(default_csv),
                "default_aware_tuned_runs_file": str(tuned_csv),
            }
        ),
        encoding="utf-8",
    )

    outputs = matched_runner_module._generate_matched_topology_diagnostic_report(
        manifest_path=manifest_path,
        output_dir=tmp_path,
        markdown_name="DIAGNOSTICS.md",
    )

    dataset_summary = pd.read_csv(outputs["diagnostic_dataset_summary_tsv"], sep="\t")
    paired_summary = pd.read_csv(outputs["diagnostic_paired_summary_tsv"], sep="\t")
    mtr_null_by_map = pd.read_csv(outputs["mtr_permutation_null_by_map_tsv"], sep="\t")
    mtr_null_by_dataset = pd.read_csv(outputs["mtr_permutation_null_by_dataset_tsv"], sep="\t")

    assert Path(outputs["diagnostic_markdown_report"]).exists()
    assert len(mtr_null_by_map) == 12
    assert "balanced_mean_tied_rank_observed_to_null_ratio" in mtr_null_by_dataset.columns
    assert set(dataset_summary["profile"]) == {"true_default", "tuned_fixed"}
    assert {"hexagonal", "mst", "rng"}.issubset(set(dataset_summary["topology"]))

    row = paired_summary[
        (paired_summary["comparison"] == "default_hexagonal_vs_default_mst")
        & (paired_summary["metric"] == "balanced_qe_raw")
    ].iloc[0]
    assert row["mean_signed_effect_favoring_comparator"] == pytest.approx(1.0)
    assert row["comparator_wins"] == 2
    assert "raw_p" in paired_summary.columns
    assert "bh_q" not in paired_summary.columns


def test_generate_matched_topology_report_includes_rng_variant_profile(tmp_path, matched_runner_module):
    import json
    import pandas as pd

    def make_row(profile, dataset, seed, topology, qe, mtr, utilization, dead_fraction):
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": topology,
            "sampling_method": "full",
            "run_profile": profile,
            "evaluation_split": "both",
            "quantization_error_holdout": qe + 0.1,
            "quantization_error_train": qe - 0.1,
            "balanced_qe_raw": qe,
            "mean_tied_rank_holdout": mtr + 0.1,
            "mean_tied_rank_train": mtr - 0.1,
            "balanced_mean_tied_rank_raw": mtr,
            "mean_tied_rank_null_mean_holdout": 50.0,
            "mean_tied_rank_null_mean_train": 50.0,
            "balanced_mean_tied_rank_null_mean": 50.0,
            "mean_tied_rank_null_theoretical_mean_holdout": 50.0,
            "mean_tied_rank_null_theoretical_mean_train": 50.0,
            "balanced_mean_tied_rank_null_theoretical_mean": 50.0,
            "mean_tied_rank_null_std_holdout": 1.0,
            "mean_tied_rank_null_std_train": 1.0,
            "mean_tied_rank_null_q025_holdout": 48.0,
            "mean_tied_rank_null_q025_train": 48.0,
            "mean_tied_rank_null_q975_holdout": 52.0,
            "mean_tied_rank_null_q975_train": 52.0,
            "mean_tied_rank_observed_to_null_ratio_holdout": (mtr + 0.1) / 50.0,
            "mean_tied_rank_observed_to_null_ratio_train": (mtr - 0.1) / 50.0,
            "balanced_mean_tied_rank_observed_to_null_ratio": mtr / 50.0,
            "mean_tied_rank_null_lower_tail_p_holdout": 0.001,
            "mean_tied_rank_null_lower_tail_p_train": 0.001,
            "mean_tied_rank_null_permutations_holdout": 1000,
            "mean_tied_rank_null_permutations_train": 1000,
            "node_utilization_holdout": utilization - 0.05,
            "node_utilization_train": utilization + 0.05,
            "balanced_node_utilization_raw": utilization,
            "dead_node_fraction_holdout": dead_fraction + 0.05,
            "dead_node_fraction_train": dead_fraction - 0.05,
            "balanced_dead_node_fraction_raw": dead_fraction,
        }

    default_rows = []
    tuned_rows = []
    variant_rows = []
    for seed in (1, 2):
        default_rows.extend(
            [
                make_row("true_default", "iris", seed, "hexagonal", 10.0, 5.0, 0.50, 0.50),
                make_row("true_default", "iris", seed, "mst", 9.0, 4.0, 0.60, 0.40),
                make_row("true_default", "iris", seed, "rng", 8.0, 3.0, 0.70, 0.30),
            ]
        )
        tuned_rows.extend(
            [
                make_row("tuned_fixed", "iris", seed, "hexagonal", 7.0, 4.5, 0.65, 0.35),
                make_row("tuned_fixed", "iris", seed, "mst", 6.0, 3.5, 0.75, 0.25),
                make_row("tuned_fixed", "iris", seed, "rng", 5.0, 2.5, 0.85, 0.15),
            ]
        )
        variant_rows.append(make_row("tuned_rng_random", "iris", seed, "rng", 4.0, 2.0, 0.90, 0.10))

    default_csv = tmp_path / "matched_default.csv"
    tuned_csv = tmp_path / "matched_tuned.csv"
    variant_csv = tmp_path / "matched_tuned_rng_random.csv"
    pd.DataFrame(default_rows).to_csv(default_csv, index=False)
    pd.DataFrame(tuned_rows).to_csv(tuned_csv, index=False)
    pd.DataFrame(variant_rows).to_csv(variant_csv, index=False)

    manifest_path = tmp_path / "MATCHED_TOPOLOGY_RNG_VARIANTS_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile_runs": [
                    {"profile": "true_default", "runs_csv": str(default_csv)},
                    {"profile": "tuned_fixed", "runs_csv": str(tuned_csv)},
                    {"profile": "tuned_rng_random", "runs_csv": str(variant_csv)},
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = matched_runner_module._generate_matched_topology_diagnostic_report(
        manifest_path=manifest_path,
        output_dir=tmp_path,
        markdown_name="RNG_VARIANTS.md",
    )

    dataset_summary = pd.read_csv(outputs["diagnostic_dataset_summary_tsv"], sep="\t")
    paired_summary = pd.read_csv(outputs["diagnostic_paired_summary_tsv"], sep="\t")

    assert "tuned_rng_random" in set(dataset_summary["profile"])
    assert "tuned_mst_vs_tuned_rng" in set(paired_summary["comparison"])
    row = paired_summary[
        (paired_summary["comparison"] == "tuned_fixed_rng_vs_tuned_rng_random_rng")
        & (paired_summary["metric"] == "balanced_qe_raw")
    ].iloc[0]
    assert row["n_pairs"] == 2
    assert row["mean_signed_effect_favoring_comparator"] == pytest.approx(1.0)
    assert row["comparator_wins"] == 2
