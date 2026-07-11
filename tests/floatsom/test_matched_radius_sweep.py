"""Tests for the matched initial-radius sweep and response analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = _REPO_ROOT / "benchmarks" / "optuna" / "run_matched_radius_sweep.py"
_ANALYZER_PATH = _REPO_ROOT / "benchmarks" / "optuna" / "analyze_matched_radius_sweep.py"
_TWO_GPU_PBS_PATH = _REPO_ROOT / "benchmarks" / "optuna" / "run_matched_radius_sweep_2gpu.pbs.sh"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def radius_runner_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    previous = {name: module for name, module in sys.modules.items() if name == "floatsom" or name.startswith("floatsom.")}
    for module_name in list(previous):
        sys.modules.pop(module_name, None)
    package = types.ModuleType("floatsom")
    package.__path__ = [str(_REPO_ROOT)]
    sys.modules["floatsom"] = package
    inserted_optuna_stub = False
    if "optuna" not in sys.modules and importlib.util.find_spec("optuna") is None:
        optuna_stub = types.ModuleType("optuna")
        optuna_stub.trial = types.SimpleNamespace(
            FrozenTrial=object,
            TrialState=types.SimpleNamespace(COMPLETE="COMPLETE"),
        )
        optuna_stub.Study = object
        sys.modules["optuna"] = optuna_stub
        inserted_optuna_stub = True
    module = _load_module(_RUNNER_PATH, "matched_radius_sweep_runner_test")
    yield module
    for module_name in list(sys.modules):
        if module_name == "floatsom" or module_name.startswith("floatsom."):
            sys.modules.pop(module_name, None)
    sys.modules.update(previous)
    if inserted_optuna_stub:
        sys.modules.pop("optuna", None)


@pytest.fixture(scope="module")
def radius_analyzer_module():
    return _load_module(_ANALYZER_PATH, "matched_radius_sweep_analyzer_test")


def test_default_preset_locks_requested_radius_grid_and_seeds(radius_runner_module):
    preset = radius_runner_module._load_preset(radius_runner_module.DEFAULT_PRESET)
    radii = radius_runner_module._resolve_radii(None, preset["initial_radii"])

    assert radii == pytest.approx([0.5, 0.75, 1.026640962470433, 1.5, 2.0, 3.0, 5.0])
    assert len(preset["seeds"]) == 20
    assert len(set(preset["seeds"])) == 20
    assert preset["fixed_params"]["initialization_method"] == "random"
    assert preset["fixed_params"]["radius_decay_type"] == "asymptotic"


def test_radius_resolution_rejects_duplicate_or_out_of_range(radius_runner_module):
    with pytest.raises(ValueError, match="Duplicate initial radius"):
        radius_runner_module._resolve_radii([1.0, 1.0], [])
    with pytest.raises(ValueError, match="outside"):
        radius_runner_module._resolve_radii([0.49], [])


def test_configuration_invariants_allow_only_topology_and_radius(radius_runner_module):
    fixed = {
        "initialization_method": "random",
        "momentum_init": 0.6068704314721264,
        "normalization": "xpysom",
        "radius_decay_type": "asymptotic",
        "use_momentum": True,
    }
    rows = []
    for topology, radius in [("hexagonal", 0.5), ("mst", 1.5)]:
        rows.append(
            {
                "dataset": "iris",
                "seed": 1,
                "architecture": topology,
                "sampling_method": "full",
                "initial_radius": radius,
                "param_initial_radius": radius,
                "processing_type": "batch",
                "batch_mode": "full_batch",
                "evaluation_split": "both",
                "config_processing_method": "batch",
                "config_sampling_method": "full",
                "config_batch_mode": "full_batch",
                "config_topology_type": topology,
                **{f"param_{name}": value for name, value in fixed.items()},
            }
        )
    frame = pd.DataFrame(rows)
    radius_runner_module._validate_configuration_invariants(
        frame,
        fixed_params=fixed,
        radii=[0.5, 1.5],
        topologies=["hexagonal", "mst"],
    )

    drifted = frame.copy()
    drifted.loc[1, "param_momentum_init"] = 0.7
    with pytest.raises(ValueError, match="Configuration drift"):
        radius_runner_module._validate_configuration_invariants(
            drifted,
            fixed_params=fixed,
            radii=[0.5, 1.5],
            topologies=["hexagonal", "mst"],
        )


def test_outer_ray_worker_reserves_one_whole_gpu(radius_runner_module):
    captured = {}

    class FakeRay:
        @staticmethod
        def remote(**resources):
            captured.update(resources)
            return lambda function: function

    worker = radius_runner_module._create_outer_ray_worker(FakeRay())
    assert worker is radius_runner_module._execute_radius_condition
    assert captured == {"num_gpus": 1}


def test_bounded_outer_ray_keeps_two_tasks_in_flight(radius_runner_module):
    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        def __init__(self):
            self.outstanding = 0
            self.max_outstanding = 0

        def wait(self, references, num_returns):
            assert num_returns == 1
            assert len(references) <= 2
            self.outstanding -= 1
            return [references[-1]], references[:-1]

        @staticmethod
        def get(reference):
            return {"ok": True, "row": {"value": reference.value}}

    fake_ray = FakeRay()

    class FakeWorker:
        def remote(self, condition):
            fake_ray.outstanding += 1
            fake_ray.max_outstanding = max(fake_ray.max_outstanding, fake_ray.outstanding)
            return Ref(condition["value"])

    conditions = [{"value": index} for index in range(5)]
    results = list(
        radius_runner_module._iter_bounded_ray_results(
            ray_module=fake_ray,
            remote_worker=FakeWorker(),
            conditions=conditions,
            max_in_flight=2,
        )
    )

    assert fake_ray.max_outstanding == 2
    assert fake_ray.outstanding == 0
    assert sorted(result["row"]["value"] for _, result in results) == list(range(5))


def test_two_gpu_pbs_requests_and_enables_outer_ray():
    script = _TWO_GPU_PBS_PATH.read_text(encoding="utf-8")
    assert "#PBS -q gpuvolta" in script
    assert "#PBS -l ngpus=2" in script
    assert "#PBS -l ncpus=24" in script
    assert "#PBS -l mem=180GB" in script
    assert "--outer-ray" in script
    assert "--outer-ray-num-gpus 2" in script
    assert "--ray-temp-dir" in script
    assert 'RAY_TEMP_DIR="${RAY_TEMP_DIR:-${PBS_JOBFS}/r}"' in script


def test_runner_checkpoints_and_resumes_by_radius_key(tmp_path, monkeypatch, radius_runner_module):
    anchor = 1.026640962470433
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        json.dumps(
            {
                "profile_name": "test_radius_sweep",
                "source_manifest": "test",
                "anchor_initial_radius": anchor,
                "initial_radii": [anchor, 1.5],
                "seeds": [7],
                "fixed_params": {
                    "initialization_method": "random",
                    "momentum_init": 0.6068704314721264,
                    "normalization": "xpysom",
                    "radius_decay_type": "asymptotic",
                    "use_momentum": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = radius_runner_module.parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--preset",
            str(preset_path),
            "--datasets",
            "iris",
            "--seeds",
            "7",
            "--topologies",
            "hexagonal",
            "--checkpoint-interval",
            "1",
        ]
    )

    calls = []
    monkeypatch.setattr(radius_runner_module.matched, "_run_single_benchmark_lazy", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(radius_runner_module.matched, "_first_completed_trial", lambda study: object())

    def fake_success_row(**kwargs):
        manual = kwargs["manual_fixed_params"]
        topology = kwargs["topology"]
        row = {
            "scenario_id": "iris_batch_full_full_batch_hexagonal",
            "dataset": kwargs["dataset"],
            "seed": kwargs["seed"],
            "architecture": topology,
            "sampling_method": "full",
            "processing_type": "batch",
            "batch_mode": "full_batch",
            "evaluation_split": "both",
            "config_processing_method": "batch",
            "config_sampling_method": "full",
            "config_batch_mode": "full_batch",
            "config_topology_type": topology,
            **{f"param_{name}": value for name, value in manual.items()},
        }
        for metric in radius_runner_module.matched.REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS:
            row[metric] = 1.0
        return row

    monkeypatch.setattr(radius_runner_module.matched, "_build_success_row", fake_success_row)
    first = radius_runner_module.run_sweep(args)
    assert first["complete"] is True
    assert first["successful_runs"] == 2
    assert len(calls) == 2

    runs = pd.read_csv(first["runs_csv"])
    assert len(runs) == 2
    assert runs["scenario_id"].nunique() == 2
    assert set(np.round(runs["initial_radius"], 12)) == {round(anchor, 12), 1.5}

    calls.clear()
    second = radius_runner_module.run_sweep(args)
    assert second["resume_skipped_runs"] == 2
    assert second["executed_runs"] == 0
    assert calls == []


def test_direction_aware_percent_change(radius_analyzer_module):
    candidate = np.asarray([8.0, 12.0])
    anchor = np.asarray([10.0, 10.0])
    lower = radius_analyzer_module._paired_percent_change(candidate, anchor, higher_is_better=False)
    higher = radius_analyzer_module._paired_percent_change(candidate, anchor, higher_is_better=True)

    assert lower == pytest.approx([20.0, -20.0])
    assert higher == pytest.approx([-20.0, 20.0])


def test_benjamini_hochberg_is_monotone_and_not_below_raw_p(radius_analyzer_module):
    p_values = np.asarray([0.01, 0.04, 0.03])
    adjusted = radius_analyzer_module.benjamini_hochberg(p_values)

    assert adjusted == pytest.approx([0.03, 0.04, 0.04])
    assert np.all(adjusted >= p_values)


def _synthetic_sweep_frame(radii, anchor):
    topology_scales = {"hexagonal": 1.0, "mst": 0.9, "rng": 0.8}
    fixed = {
        "initialization_method": "random",
        "momentum_init": 0.6068704314721264,
        "normalization": "xpysom",
        "radius_decay_type": "asymptotic",
        "use_momentum": True,
    }
    rows = []
    for dataset_index, dataset in enumerate(["iris", "wine"]):
        for seed in (11, 22, 33):
            seed_shift = (seed % 7) * 0.002
            for topology, topology_scale in topology_scales.items():
                for radius in radii:
                    radius_shift = float(radius) - float(anchor)
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "architecture": topology,
                            "sampling_method": "full",
                            "processing_type": "batch",
                            "batch_mode": "full_batch",
                            "evaluation_split": "both",
                            "initial_radius": radius,
                            "param_initial_radius": radius,
                            **{f"param_{name}": value for name, value in fixed.items()},
                            "balanced_qe_raw": (
                                1.0 + dataset_index * 0.2 + seed_shift + topology_scale * 0.04 * radius_shift
                            ),
                            "balanced_mean_tied_rank_raw": (
                                5.0 + dataset_index + seed_shift + topology_scale * 0.25 * radius_shift
                            ),
                            "balanced_node_utilization_raw": (
                                0.8 - dataset_index * 0.02 - topology_scale * 0.01 * radius_shift
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_full_analysis_emits_three_topology_response_plots_and_54_q_values(
    tmp_path,
    radius_analyzer_module,
):
    radii = [0.5, 0.75, 1.026640962470433, 1.5, 2.0, 3.0, 5.0]
    anchor = 1.026640962470433
    runs = _synthetic_sweep_frame(radii, anchor)
    runs_path = tmp_path / "matched_radius_sweep_runs.csv"
    runs.to_csv(runs_path, index=False)
    manifest = {
        "runs_csv": str(runs_path),
        "datasets": ["iris", "wine"],
        "seeds": [11, 22, 33],
        "topologies": ["hexagonal", "mst", "rng"],
        "sampling_methods": ["full"],
        "initial_radii": radii,
        "anchor_initial_radius": anchor,
        "fixed_params": {
            "initialization_method": "random",
            "momentum_init": 0.6068704314721264,
            "normalization": "xpysom",
            "radius_decay_type": "asymptotic",
            "use_momentum": True,
        },
    }
    manifest_path = tmp_path / "MATCHED_RADIUS_SWEEP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    outputs = radius_analyzer_module.analyze(
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis",
        dpi=72,
        strict_coverage=True,
    )

    assert outputs["bh_family_size"] == 54
    cross_topology = pd.read_csv(outputs["cross_topology_qe_summary"], sep="\t")
    assert len(cross_topology) == 3 * 7
    assert cross_topology["bh_q_value"].notna().all()
    assert set(cross_topology["n_datasets"]) == {2}
    assert set(cross_topology["n_seed_pairs"]) == {6}
    assert np.isfinite(
        cross_topology[
            ["geometric_mean_qe_ratio", "qe_ratio_ci_low", "qe_ratio_ci_high", "bh_q_value"]
        ].to_numpy(dtype=float)
    ).all()
    assert set(outputs["figures"]) == {"balanced_qe", "balanced_mtr", "balanced_node_utilisation"}
    for figure_path in outputs["figures"].values():
        path = Path(figure_path)
        assert path.exists()
        svg = path.read_text(encoding="utf-8").lower()
        assert "#ff4fa3" in svg
        assert "#8a2be2" in svg
        assert "#00e5ff" in svg
        assert 'id="legend_' not in svg

    pooled = pd.read_csv(outputs["pooled_summary"])
    assert len(pooled) == 3 * 3 * 7
    assert pooled["bh_q_value"].notna().sum() == 54
    assert set(pooled["n_pairs"]) == {6}
    assert {
        "mean_candidate_value",
        "median_candidate_value",
        "candidate_ci_low",
        "candidate_ci_high",
        "geometric_mean_hex_normalized_qe",
        "hex_normalized_qe_ci_low",
        "hex_normalized_qe_ci_high",
        "n_datasets",
    }.issubset(pooled.columns)
    assert np.isfinite(
        pooled[["mean_candidate_value", "candidate_ci_low", "candidate_ci_high"]].to_numpy(dtype=float)
    ).all()
    assert (pooled["candidate_ci_low"] <= pooled["mean_candidate_value"]).all()
    assert (pooled["mean_candidate_value"] <= pooled["candidate_ci_high"]).all()
    anchor_rows = pooled[np.isclose(pooled["initial_radius"], anchor)]
    assert len(anchor_rows) == 9
    assert np.allclose(anchor_rows["mean_pct_change"], 0.0)
    assert anchor_rows["bh_q_value"].isna().all()

    qe_rows = pooled[pooled["metric"] == "balanced_qe_raw"]
    normalized_columns = [
        "geometric_mean_hex_normalized_qe",
        "hex_normalized_qe_ci_low",
        "hex_normalized_qe_ci_high",
    ]
    assert np.isfinite(qe_rows[normalized_columns].to_numpy(dtype=float)).all()
    assert set(qe_rows["n_datasets"]) == {2}
    hex_anchor = qe_rows[
        (qe_rows["architecture"] == "hexagonal")
        & np.isclose(qe_rows["initial_radius"], anchor)
    ]
    assert len(hex_anchor) == 1
    assert hex_anchor.iloc[0]["geometric_mean_hex_normalized_qe"] == pytest.approx(1.0)
