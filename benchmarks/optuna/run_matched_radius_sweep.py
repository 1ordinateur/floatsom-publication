#!/usr/bin/env python3
"""Run a matched initial-radius sweep with all non-structural settings fixed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.machinery
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""} and "floatsom" not in sys.modules:
    _repo_root = Path(__file__).resolve().parents[2]
    _pkg = types.ModuleType("floatsom")
    _pkg.__file__ = str(_repo_root / "__init__.py")
    _pkg.__path__ = [str(_repo_root)]
    _pkg.__package__ = "floatsom"
    _pkg.__spec__ = importlib.machinery.ModuleSpec("floatsom", loader=None, is_package=True)
    _pkg.__spec__.submodule_search_locations = _pkg.__path__
    sys.modules["floatsom"] = _pkg

import numpy as np
import pandas as pd

from floatsom.benchmarks.optuna.config.benchmark_config import Phase3BenchmarkConfig
from floatsom.benchmarks.optuna.config.parameters import PARAMETER_CONFIGS
from floatsom.benchmarks.optuna import run_matched_default_floatsom_batch as matched


SUPPORTED_TOPOLOGIES: Tuple[str, ...] = ("hexagonal", "mst", "rng")
SUPPORTED_SAMPLING_METHODS: Tuple[str, ...] = ("full",)
RUN_KEY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "seed",
    "architecture",
    "sampling_method",
    "initial_radius",
)
DEFAULT_PRESET = Path(__file__).resolve().parent / "config" / "presets" / "radius_sweep_tuned_hex.json"
DEFAULT_RUNS_CSV = "matched_radius_sweep_runs.csv"
DEFAULT_MANIFEST = "MATCHED_RADIUS_SWEEP_MANIFEST.json"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_path(path_value: str | Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    repo_candidate = Path(__file__).resolve().parents[2] / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    raise FileNotFoundError(f"Path does not exist: {path_value}")


def _load_preset(path_value: str | Path) -> Dict[str, Any]:
    path = _resolve_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Radius-sweep preset must be a JSON object: {path}")
    required = {"profile_name", "anchor_initial_radius", "initial_radii", "seeds", "fixed_params"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Radius-sweep preset is missing required keys: {missing}")
    if not isinstance(payload["fixed_params"], dict):
        raise ValueError("Radius-sweep preset fixed_params must be an object.")
    payload["preset_path"] = str(path)
    return payload


def _normalize_values(values: Optional[Iterable[str]]) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        normalized.extend(part.strip() for part in str(value).split(",") if part.strip())
    return list(dict.fromkeys(normalized))


def _resolve_datasets(requested: Optional[Sequence[str]]) -> List[str]:
    available = list(Phase3BenchmarkConfig().datasets or [])
    if not requested:
        return available
    datasets = _normalize_values(requested)
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Available: {available}")
    return datasets


def _resolve_topologies(requested: Optional[Sequence[str]]) -> List[str]:
    topologies = _normalize_values(requested) if requested else list(SUPPORTED_TOPOLOGIES)
    topologies = [value.lower() for value in topologies]
    unknown = sorted(set(topologies) - set(SUPPORTED_TOPOLOGIES))
    if unknown:
        raise ValueError(f"Unsupported topology value(s): {unknown}")
    return topologies


def _resolve_radii(requested: Optional[Sequence[float]], preset_values: Sequence[object]) -> List[float]:
    raw_values = list(requested) if requested else list(preset_values)
    minimum, maximum = PARAMETER_CONFIGS["initial_radius"]["range"]
    radii: List[float] = []
    for raw_value in raw_values:
        radius = float(raw_value)
        if not math.isfinite(radius):
            raise ValueError(f"Initial radius must be finite: {raw_value!r}")
        if not float(minimum) <= radius <= float(maximum):
            raise ValueError(f"Initial radius {radius} is outside [{minimum}, {maximum}].")
        if any(math.isclose(radius, prior, rel_tol=0.0, abs_tol=1e-12) for prior in radii):
            raise ValueError(f"Duplicate initial radius: {radius}")
        radii.append(radius)
    if not radii:
        raise ValueError("At least one initial radius is required.")
    return sorted(radii)


def _resolve_seeds(
    *,
    explicit_seeds: Optional[Sequence[int]],
    seeds_from_csv: Optional[str],
    preset_seeds: Sequence[object],
) -> List[int]:
    if explicit_seeds and seeds_from_csv:
        raise ValueError("Use either --seeds or --seeds-from-csv, not both.")
    if seeds_from_csv:
        return matched._resolve_seeds_from_csv(seeds_from_csv)
    raw_values = list(explicit_seeds) if explicit_seeds else list(preset_seeds)
    seeds = [int(value) for value in raw_values]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _validate_fixed_profile(fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    forbidden = {"initial_radius", "topology_type", "sampling_method", "processing_method", "batch_mode"}
    overlap = sorted(forbidden.intersection(fixed_params))
    if overlap:
        raise ValueError(f"Sweep-controlled parameters cannot appear in fixed_params: {overlap}")
    coerced = {name: matched._coerce_param_value(name, value) for name, value in fixed_params.items()}
    matched._validate_manual_fixed_params_for_scope(
        coerced,
        processing_method="batch",
        topologies=list(SUPPORTED_TOPOLOGIES),
    )
    return coerced


def _normalize_run_key(row: Dict[str, Any] | pd.Series) -> Tuple[str, int, str, str, float]:
    return (
        str(row["dataset"]).strip().lower(),
        int(row["seed"]),
        str(row["architecture"]).strip().lower(),
        str(row["sampling_method"]).strip().lower(),
        round(float(row["initial_radius"]), 12),
    )


def _dedupe_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    missing = [column for column in RUN_KEY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Existing radius-sweep CSV is missing key columns: {missing}")
    working = df.copy()
    working["initial_radius"] = pd.to_numeric(working["initial_radius"], errors="raise").round(12)
    return (
        working.drop_duplicates(list(RUN_KEY_COLUMNS), keep="last")
        .sort_values(list(RUN_KEY_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_configuration_invariants(
    df: pd.DataFrame,
    *,
    fixed_params: Dict[str, Any],
    radii: Sequence[float],
    topologies: Sequence[str],
) -> None:
    if df.empty:
        return
    expected_constants = {
        "processing_type": "batch",
        "sampling_method": "full",
        "batch_mode": "full_batch",
        "evaluation_split": "both",
        "config_processing_method": "batch",
        "config_sampling_method": "full",
        "config_batch_mode": "full_batch",
    }
    for column, expected in expected_constants.items():
        if column not in df.columns:
            raise ValueError(f"Radius-sweep rows are missing invariant column: {column}")
        observed = set(df[column].astype(str).str.strip().str.lower())
        if observed != {expected}:
            raise ValueError(f"Configuration drift in {column}: expected {expected!r}, observed {sorted(observed)}")

    observed_topologies = set(df["architecture"].astype(str).str.strip().str.lower())
    if not observed_topologies.issubset(set(topologies)):
        raise ValueError(f"Unexpected topology values in radius-sweep rows: {sorted(observed_topologies)}")
    configured_topologies = set(df["config_topology_type"].astype(str).str.strip().str.lower())
    if configured_topologies != observed_topologies:
        raise ValueError("architecture and config_topology_type do not match.")

    observed_radii = pd.to_numeric(df["initial_radius"], errors="raise").to_numpy(dtype=float)
    allowed_radii = np.asarray(list(radii), dtype=float)
    if any(not np.isclose(value, allowed_radii, rtol=0.0, atol=1e-12).any() for value in observed_radii):
        raise ValueError("Radius-sweep rows contain an initial radius outside the configured sweep.")
    param_radii = pd.to_numeric(df["param_initial_radius"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(observed_radii, param_radii, rtol=0.0, atol=1e-12):
        raise ValueError("initial_radius and param_initial_radius disagree.")

    for param_name, expected in fixed_params.items():
        column = f"param_{param_name}"
        if column not in df.columns:
            raise ValueError(f"Radius-sweep rows are missing fixed parameter column: {column}")
        series = df[column]
        if isinstance(expected, bool):
            normalized = series.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "false": False, "0": False}
            )
            if normalized.isna().any() or set(normalized) != {expected}:
                raise ValueError(f"Configuration drift in {column}; expected {expected!r}.")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
            if not np.allclose(values, float(expected), rtol=0.0, atol=1e-12):
                raise ValueError(f"Configuration drift in {column}; expected {expected!r}.")
        else:
            observed = set(series.astype(str).str.strip().str.lower())
            if observed != {str(expected).strip().lower()}:
                raise ValueError(f"Configuration drift in {column}; expected {expected!r}, observed {observed}.")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint(
    *,
    existing_df: pd.DataFrame,
    success_rows: Sequence[Dict[str, Any]],
    failure_rows: Sequence[Dict[str, Any]],
    runs_csv: Path,
    failures_csv: Path,
) -> pd.DataFrame:
    frames = [frame for frame in [existing_df, pd.DataFrame(success_rows)] if not frame.empty]
    combined = _dedupe_runs(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    if not combined.empty:
        _write_csv_atomic(combined, runs_csv)
    if failure_rows:
        _write_csv_atomic(pd.DataFrame(failure_rows), failures_csv)
    return combined


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _ensure_ray_worker_import_path(base_dir: Path) -> Path:
    """Expose this checkout as ``floatsom`` to fresh Ray worker interpreters."""
    repo_root = Path(__file__).resolve().parents[2]
    import_root = Path(tempfile.mkdtemp(prefix="radius_sweep_ray_pythonpath_", dir=str(base_dir)))
    os.symlink(repo_root, import_root / "floatsom", target_is_directory=True)
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)
    existing = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if import_root_text not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([import_root_text, *existing])
    return import_root


def _execute_radius_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one single-GPU condition; suitable for local or outer-Ray use."""
    dataset = str(condition["dataset"])
    seed = int(condition["seed"])
    topology = str(condition["topology"])
    initial_radius = float(condition["initial_radius"])
    fixed_params = dict(condition["fixed_params"])
    effective_params = {**fixed_params, "initial_radius": initial_radius}
    forced_params = {
        "sampling_method": "full",
        "processing_method": "batch",
        "batch_mode": "full_batch",
        "topology_type": topology,
        **effective_params,
    }
    ray_gpu_ids: List[int] = []
    try:
        import ray

        if ray.is_initialized():
            ray_gpu_ids = [int(float(value)) for value in ray.get_gpu_ids()]
    except Exception:
        ray_gpu_ids = []
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    try:
        study = matched._run_single_benchmark_lazy(
            algo_type="batch",
            dataset_name=dataset,
            dataset_config=dict(condition["dataset_config"]),
            forced_params=forced_params,
            seed=seed,
            n_trials=1,
            timeout=condition.get("timeout"),
            objectives=["quantization_error"],
            diagnostic_metrics=list(matched.MATCHED_TOPOLOGY_DIAGNOSTIC_METRICS),
            evaluation_split="both",
            output_dir=None,
            use_ray_tune=False,
            max_concurrent=None,
        )
        trial = matched._first_completed_trial(study)
        row = matched._build_success_row(
            trial=trial,
            dataset=dataset,
            seed=seed,
            topology=topology,
            sampling_method="full",
            evaluation_split="both",
            forced_params=forced_params,
            manual_fixed_params=effective_params,
            run_profile=str(condition["profile_name"]),
            run_label=f"radius_{initial_radius:.12g}",
        )
        row["initial_radius"] = initial_radius
        row["anchor_initial_radius"] = float(condition["anchor_radius"])
        radius_tag = f"{initial_radius:.12g}".replace(".", "p").replace("-", "m")
        row["scenario_id"] = f"{row['scenario_id']}_radius_{radius_tag}"
        row["outer_ray_enabled"] = bool(condition.get("outer_ray_enabled", False))
        row["outer_ray_gpu_ids"] = ",".join(str(value) for value in ray_gpu_ids)
        row["cuda_visible_devices"] = cuda_visible_devices

        if bool(condition.get("save_study_json", False)):
            output_dir = Path(str(condition["output_dir"]))
            seed_dir = output_dir / f"seed_{seed}" / f"radius_{radius_tag}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            matched._save_study_json_lazy(
                study=study,
                output_dir=str(seed_dir),
                dataset_name=dataset,
                forced_params=forced_params,
                objectives=matched._expand_objectives_for_split(["quantization_error"], "both"),
                expected_trials=1,
            )
        return {
            "ok": True,
            "row": row,
            "ray_gpu_ids": ray_gpu_ids,
            "cuda_visible_devices": cuda_visible_devices,
        }
    except Exception as exc:
        failure = matched._build_failure_row(
            dataset=dataset,
            seed=seed,
            topology=topology,
            sampling_method="full",
            evaluation_split="both",
            error=exc,
            run_profile=str(condition["profile_name"]),
            run_label=f"radius_{initial_radius:.12g}",
        )
        failure["initial_radius"] = initial_radius
        failure["outer_ray_enabled"] = bool(condition.get("outer_ray_enabled", False))
        failure["outer_ray_gpu_ids"] = ",".join(str(value) for value in ray_gpu_ids)
        failure["cuda_visible_devices"] = cuda_visible_devices
        return {
            "ok": False,
            "failure": failure,
            "error": failure["error"],
            "ray_gpu_ids": ray_gpu_ids,
            "cuda_visible_devices": cuda_visible_devices,
        }


def _iter_bounded_ray_results(
    *,
    ray_module: Any,
    remote_worker: Any,
    conditions: Sequence[Dict[str, Any]],
    max_in_flight: int,
):
    """Yield Ray results while keeping no more than ``max_in_flight`` tasks pending."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive.")
    condition_iter = iter(conditions)
    pending: Dict[Any, Dict[str, Any]] = {}

    def submit_one() -> bool:
        try:
            condition = next(condition_iter)
        except StopIteration:
            return False
        reference = remote_worker.remote(condition)
        pending[reference] = condition
        return True

    while len(pending) < max_in_flight and submit_one():
        pass
    while pending:
        ready, _ = ray_module.wait(list(pending), num_returns=1)
        reference = ready[0]
        condition = pending.pop(reference)
        yield condition, ray_module.get(reference)
        while len(pending) < max_in_flight and submit_one():
            pass


def _create_outer_ray_worker(ray_module: Any) -> Any:
    """Create the outer worker with one whole GPU and no inner Ray Tune."""
    return ray_module.remote(num_gpus=1)(_execute_radius_condition)


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    preset = _load_preset(args.preset)
    datasets = _resolve_datasets(args.datasets)
    topologies = _resolve_topologies(args.topologies)
    radii = _resolve_radii(args.initial_radii, preset["initial_radii"])
    anchor_radius = float(preset["anchor_initial_radius"])
    if not any(math.isclose(anchor_radius, radius, rel_tol=0.0, abs_tol=1e-12) for radius in radii):
        raise ValueError(f"Configured anchor radius {anchor_radius} must be included in the sweep.")
    seeds = _resolve_seeds(
        explicit_seeds=args.seeds,
        seeds_from_csv=args.seeds_from_csv,
        preset_seeds=preset["seeds"],
    )
    fixed_params = _validate_fixed_profile(dict(preset["fixed_params"]))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = output_dir / args.runs_csv_name
    failures_csv = output_dir / f"{Path(args.runs_csv_name).stem}_failures.csv"
    manifest_path = output_dir / args.manifest_name

    existing_df = pd.DataFrame()
    completed_keys: set[Tuple[str, int, str, str, float]] = set()
    if args.resume and runs_csv.exists():
        existing_df = _dedupe_runs(pd.read_csv(runs_csv))
        matched._validate_required_numeric_columns(
            existing_df,
            required_columns=matched.REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS,
            context=f"Resume CSV {runs_csv}",
        )
        _validate_configuration_invariants(
            existing_df,
            fixed_params=fixed_params,
            radii=radii,
            topologies=topologies,
        )
        completed_keys = {_normalize_run_key(row) for _, row in existing_df.iterrows()}

    planned_runs = len(datasets) * len(seeds) * len(topologies) * len(radii)
    print("Running matched initial-radius sweep")
    print(f"Output: {output_dir}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Topologies: {topologies}")
    print(f"Initial radii: {radii}")
    print(f"Anchor radius: {anchor_radius}")
    print(f"Shared fixed params: {fixed_params}")
    print(f"Planned runs: {planned_runs}")

    success_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    skipped_runs = 0
    since_checkpoint = 0
    dataset_config = {"difficulty": str(args.difficulty), "normalize": True}
    conditions: List[Dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for topology in topologies:
                for initial_radius in radii:
                    run_key = (
                        dataset.lower(),
                        int(seed),
                        topology.lower(),
                        "full",
                        round(float(initial_radius), 12),
                    )
                    if run_key in completed_keys:
                        skipped_runs += 1
                        continue
                    conditions.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "topology": topology,
                            "initial_radius": float(initial_radius),
                            "fixed_params": fixed_params,
                            "dataset_config": dataset_config,
                            "timeout": args.timeout,
                            "profile_name": str(preset["profile_name"]),
                            "anchor_radius": anchor_radius,
                            "output_dir": str(output_dir),
                            "save_study_json": bool(args.save_study_json),
                            "outer_ray_enabled": bool(args.outer_ray),
                        }
                    )

    executed_runs = len(conditions)
    observed_ray_gpu_ids: set[int] = set()
    observed_cuda_visible_devices: set[str] = set()

    def process_result(condition: Dict[str, Any], result: Dict[str, Any], completed_index: int) -> None:
        nonlocal since_checkpoint
        print(
            f"[{completed_index}/{executed_runs}] dataset={condition['dataset']} "
            f"seed={condition['seed']} topology={condition['topology']} "
            f"radius={float(condition['initial_radius']):.12g}"
        )
        observed_ray_gpu_ids.update(int(value) for value in result.get("ray_gpu_ids", []))
        visible_devices = str(result.get("cuda_visible_devices", "")).strip()
        if visible_devices:
            observed_cuda_visible_devices.add(visible_devices)
        if bool(result.get("ok")):
            success_rows.append(dict(result["row"]))
        else:
            failure_rows.append(dict(result["failure"]))
            print(f"  FAILED: {result.get('error', 'unknown worker failure')}")
        since_checkpoint += 1
        if since_checkpoint >= int(args.checkpoint_interval):
            _checkpoint(
                existing_df=existing_df,
                success_rows=success_rows,
                failure_rows=failure_rows,
                runs_csv=runs_csv,
                failures_csv=failures_csv,
            )
            since_checkpoint = 0

    ray_resources: Dict[str, float] = {}
    ray_import_root: Optional[Path] = None
    ray_temp_root: Optional[Path] = None
    if args.outer_ray and conditions:
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("--outer-ray requested but Ray is not installed.") from exc

        ray_temp_root = (
            Path(args.ray_temp_dir).expanduser().resolve()
            if args.ray_temp_dir
            else Path(os.environ.get("PBS_JOBFS", str(output_dir / "tmp"))).resolve() / "radius_sweep_outer_ray"
        )
        ray_temp_root.mkdir(parents=True, exist_ok=True)
        ray_import_root = _ensure_ray_worker_import_path(ray_temp_root)
        ray_session_root = ray_temp_root / "ray"
        ray_session_root.mkdir(parents=True, exist_ok=True)
        ray.init(include_dashboard=False, ignore_reinit_error=False, _temp_dir=str(ray_session_root))
        try:
            ray_resources = {key: float(value) for key, value in ray.cluster_resources().items()}
            available_gpus = int(ray_resources.get("GPU", 0))
            requested_gpus = int(args.outer_ray_num_gpus)
            if available_gpus != requested_gpus:
                raise RuntimeError(
                    f"Outer Ray expected exactly {requested_gpus} GPUs but detected {available_gpus}: {ray_resources}"
                )
            print(f"Outer Ray resources: {ray_resources}")
            print(f"Outer Ray worker import root: {ray_import_root}")
            print(f"Outer Ray temp root: {ray_temp_root}")
            print(f"Running {requested_gpus} concurrent one-GPU tasks (nested Ray Tune disabled)")
            remote_worker = _create_outer_ray_worker(ray)
            for completed_index, (condition, result) in enumerate(
                _iter_bounded_ray_results(
                    ray_module=ray,
                    remote_worker=remote_worker,
                    conditions=conditions,
                    max_in_flight=requested_gpus,
                ),
                start=1,
            ):
                process_result(condition, result, completed_index)
        finally:
            ray.shutdown()
    else:
        for completed_index, condition in enumerate(conditions, start=1):
            process_result(condition, _execute_radius_condition(condition), completed_index)

    combined_df = _checkpoint(
        existing_df=existing_df,
        success_rows=success_rows,
        failure_rows=failure_rows,
        runs_csv=runs_csv,
        failures_csv=failures_csv,
    )
    _validate_configuration_invariants(
        combined_df,
        fixed_params=fixed_params,
        radii=radii,
        topologies=topologies,
    )
    successful_keys = {_normalize_run_key(row) for _, row in combined_df.iterrows()} if not combined_df.empty else set()

    metadata: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "output_dir": str(output_dir),
        "preset_path": str(preset["preset_path"]),
        "profile_name": str(preset["profile_name"]),
        "source_manifest": preset.get("source_manifest"),
        "datasets": datasets,
        "seeds": seeds,
        "seed_source_csv": str(_resolve_path(args.seeds_from_csv)) if args.seeds_from_csv else None,
        "topologies": topologies,
        "sampling_methods": list(SUPPORTED_SAMPLING_METHODS),
        "processing_type": "batch",
        "batch_mode": "full_batch",
        "evaluation_split": "both",
        "initial_radii": radii,
        "anchor_initial_radius": anchor_radius,
        "fixed_params": fixed_params,
        "run_key_columns": list(RUN_KEY_COLUMNS),
        "total_planned_runs": planned_runs,
        "resume_enabled": bool(args.resume),
        "resume_existing_rows": int(len(existing_df)),
        "resume_skipped_runs": skipped_runs,
        "executed_runs": executed_runs,
        "newly_successful_runs": len(success_rows),
        "successful_runs": len(successful_keys),
        "failed_runs": len(failure_rows),
        "complete": len(successful_keys) == planned_runs and not failure_rows,
        "runs_csv": str(runs_csv),
        "failures_csv": str(failures_csv) if failure_rows else None,
        "save_study_json": bool(args.save_study_json),
        "outer_ray_enabled": bool(args.outer_ray),
        "outer_ray_num_gpus": int(args.outer_ray_num_gpus) if args.outer_ray else 0,
        "outer_ray_resources": ray_resources,
        "outer_ray_worker_import_root": str(ray_import_root) if ray_import_root else None,
        "outer_ray_temp_root": str(ray_temp_root) if ray_temp_root else None,
        "observed_ray_gpu_ids": sorted(observed_ray_gpu_ids),
        "observed_cuda_visible_devices": sorted(observed_cuda_visible_devices),
        "nested_ray_tune_enabled": False,
    }
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["manifest_json"] = str(manifest_path)
    print(f"Saved {len(combined_df)} successful rows to {runs_csv}")
    print(f"Saved manifest to {manifest_path}")
    if failure_rows:
        print(f"Saved {len(failure_rows)} failures to {failures_csv}")
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=f"Results/radius_sweep_{_timestamp_utc()}")
    parser.add_argument("--preset", default=str(DEFAULT_PRESET))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds-from-csv", default=None)
    parser.add_argument("--topologies", nargs="+", default=list(SUPPORTED_TOPOLOGIES))
    parser.add_argument("--initial-radii", nargs="+", type=float, default=None)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--runs-csv-name", default=DEFAULT_RUNS_CSV)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST)
    parser.add_argument("--scikit-learn-data-home", default=None)
    parser.add_argument("--outer-ray", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outer-ray-num-gpus", type=int, default=2)
    parser.add_argument("--ray-temp-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-study-json", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if args.outer_ray_num_gpus < 1:
        parser.error("--outer-ray-num-gpus must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.scikit_learn_data_home:
        os.environ["SCIKIT_LEARN_DATA"] = str(Path(args.scikit_learn_data_home).expanduser())
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
