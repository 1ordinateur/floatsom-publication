#!/usr/bin/env python3
"""
Export seed-preserving Optuna trials to CSV.

This script reads raw Optuna study JSON outputs from:
  <results-dir>/seed_*/study_*.json

It exports trial-level rows without cross-seed harmonization, so downstream
paired analysis can match Batch vs Colours within the same seed.

Usage:
    python -m floatsom.benchmarks.optuna.harmonization.export_seed_trials_to_csv \
        --results-dir ../data_sifean/optuna_20251021/20251021 \
        --output ../data_sifean/optuna_20251021_seed_trials.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_study_json(json_path: Path) -> Dict[str, Any]:
    """Load a saved study JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_scenario_info(study_data: Dict[str, Any]) -> Dict[str, str]:
    """Extract scenario metadata from the saved study data."""
    scenario_id = study_data.get("scenario_id", "unknown")
    info = {
        "scenario_id": scenario_id,
        "dataset": study_data.get("dataset", "unknown"),
        "processing_type": "unknown",
        "sampling_method": "unknown",
    }

    parts = scenario_id.split("_")
    processing_types = {"batch", "colors"}
    sampling_methods = {"random", "hdsssom", "full"}

    for part in parts:
        if part in processing_types:
            info["processing_type"] = part
            break
    for part in parts:
        if part in sampling_methods:
            info["sampling_method"] = part
            break

    forced_params = study_data.get("forced_params", {}) or {}
    if info["processing_type"] == "unknown":
        info["processing_type"] = (
            forced_params.get("processor_type")
            or forced_params.get("processing")
            or forced_params.get("processing_method")
            or "unknown"
        )
    if info["sampling_method"] == "unknown":
        info["sampling_method"] = (
            forced_params.get("selector_type")
            or forced_params.get("sampling")
            or forced_params.get("selector")
            or forced_params.get("sampling_method")
            or "unknown"
        )

    return info


def parse_seed_from_seed_name(seed_name: Optional[str]) -> Optional[int]:
    """Parse integer seed from names like 'seed_27937'."""
    if not seed_name:
        return None
    match = re.search(r"seed_(\d+)", str(seed_name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_seed_from_study_name(study_name: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Extract seed_name and seed from study_name suffix when present."""
    if not study_name:
        return None, None
    match = re.search(r"(seed_(\d+))$", str(study_name))
    if not match:
        return None, None
    seed_name = match.group(1)
    try:
        seed = int(match.group(2))
    except ValueError:
        seed = None
    return seed_name, seed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _set_split_metric_aliases(row: Dict[str, Any]) -> None:
    """
    Ensure holdout aliases exist when only unsuffixed metric keys are present.
    """
    base_metrics = (
        "quantization_error",
        "distortion_measure",
        "topographic_error",
        "topographic_function",
        "trustworthiness",
        "neighborhood_preservation",
    )
    for metric_name in base_metrics:
        holdout_key = f"{metric_name}_holdout"
        train_key = f"{metric_name}_train"
        if metric_name in row and holdout_key not in row and train_key not in row:
            row[holdout_key] = row[metric_name]


def _flatten_trial_metrics(
    trial: Dict[str, Any],
    objectives: List[str],
) -> Dict[str, Any]:
    """
    Extract all available trial metrics into flat key/value columns.
    """
    flat: Dict[str, Any] = {}

    metrics = trial.get("metrics") or {}
    if isinstance(metrics, dict):
        for metric_name, metric_value in metrics.items():
            if _is_number(metric_value):
                flat[metric_name] = float(metric_value)

    values = trial.get("values") or []
    if isinstance(values, list) and objectives and len(values) == len(objectives):
        for objective_name, objective_value in zip(objectives, values):
            if objective_name not in flat and _is_number(objective_value):
                flat[objective_name] = float(objective_value)

    metrics_holdout = trial.get("metrics_holdout") or {}
    if isinstance(metrics_holdout, dict):
        for base_name, metric_value in metrics_holdout.items():
            if _is_number(metric_value):
                flat[f"{base_name}_holdout"] = float(metric_value)

    metrics_train = trial.get("metrics_train") or {}
    if isinstance(metrics_train, dict):
        for base_name, metric_value in metrics_train.items():
            if _is_number(metric_value):
                flat[f"{base_name}_train"] = float(metric_value)

    _set_split_metric_aliases(flat)
    qe_holdout = flat.get("quantization_error_holdout")
    qe_train = flat.get("quantization_error_train")
    if _is_number(qe_holdout) and _is_number(qe_train) and "balanced_qe_raw" not in flat:
        flat["balanced_qe_raw"] = float((float(qe_holdout) + float(qe_train)) / 2.0)
    return flat


def extract_trial_rows(
    study_data: Dict[str, Any],
    seed_name: Optional[str],
    seed: Optional[int],
    trial_source: str,
) -> List[Dict[str, Any]]:
    """Extract row dictionaries for one study JSON."""
    scenario_info = parse_scenario_info(study_data)
    forced_params = study_data.get("forced_params", {}) or {}
    objectives = study_data.get("objectives", []) or []

    trial_key = "all_trials" if trial_source == "all" else "best_trials"
    raw_trials = study_data.get(trial_key, []) or []

    rows: List[Dict[str, Any]] = []
    for trial in raw_trials:
        if not isinstance(trial, dict):
            continue

        row: Dict[str, Any] = {
            "scenario_id": scenario_info["scenario_id"],
            "dataset": scenario_info["dataset"],
            "processing_type": scenario_info["processing_type"],
            "sampling_method": scenario_info["sampling_method"],
            "seed_name": seed_name,
            "seed": seed,
            "trial_number": trial.get("number", -1),
        }

        user_attrs = trial.get("user_attrs") or {}
        if row["seed_name"] is None:
            candidate_seed_name = user_attrs.get("seed_name")
            if candidate_seed_name:
                row["seed_name"] = candidate_seed_name
        if row["seed"] is None:
            candidate_seed = user_attrs.get("seed")
            if _is_number(candidate_seed):
                row["seed"] = int(candidate_seed)
        if row["seed_name"] is None or row["seed"] is None:
            parsed_seed_name, parsed_seed = extract_seed_from_study_name(user_attrs.get("study_name"))
            if row["seed_name"] is None and parsed_seed_name is not None:
                row["seed_name"] = parsed_seed_name
            if row["seed"] is None and parsed_seed is not None:
                row["seed"] = parsed_seed
        if row["seed"] is None and row["seed_name"] is not None:
            row["seed"] = parse_seed_from_seed_name(row["seed_name"])

        row.update(_flatten_trial_metrics(trial=trial, objectives=objectives))

        params = trial.get("params") or {}
        if isinstance(params, dict):
            for param_name, value in params.items():
                row[f"param_{param_name}"] = value

        for cfg_name, cfg_value in forced_params.items():
            row[f"config_{cfg_name}"] = cfg_value

        rows.append(row)

    return rows


def collect_seed_trial_rows(results_dir: Path, trial_source: str) -> List[Dict[str, Any]]:
    """Collect rows from all seed folders and study files."""
    all_rows: List[Dict[str, Any]] = []

    seed_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        print(f"No seed_* directories found in {results_dir}")
        return all_rows

    print(f"Found {len(seed_dirs)} seed directories")

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        seed = parse_seed_from_seed_name(seed_name)
        study_files = sorted(seed_dir.glob("study_*.json"))
        print(f"  {seed_name}: {len(study_files)} study file(s)")

        for study_file in study_files:
            try:
                study_data = load_study_json(study_file)
                rows = extract_trial_rows(
                    study_data=study_data,
                    seed_name=seed_name,
                    seed=seed,
                    trial_source=trial_source,
                )
                all_rows.extend(rows)
            except Exception as exc:  # pragma: no cover
                print(f"    Error processing {study_file}: {exc}")

    return all_rows


def export_to_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write extracted rows to CSV with stable column ordering."""
    if not rows:
        print("No rows to export")
        return

    df = pd.DataFrame(rows)

    sort_columns = [col for col in ("scenario_id", "seed", "seed_name", "trial_number") if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    base_columns = [
        "scenario_id",
        "dataset",
        "processing_type",
        "sampling_method",
        "seed",
        "seed_name",
        "trial_number",
    ]
    param_columns = sorted(col for col in df.columns if col.startswith("param_"))
    config_columns = sorted(col for col in df.columns if col.startswith("config_"))
    excluded = set(base_columns + param_columns + config_columns)
    metric_columns = sorted(col for col in df.columns if col not in excluded)

    final_columns = [col for col in base_columns if col in df.columns]
    final_columns += metric_columns + param_columns + config_columns
    df = df[final_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nExported {len(df)} rows to {output_path}")
    print("Summary:")
    print(f"  Unique scenarios: {df['scenario_id'].nunique()}")
    print(f"  Unique datasets: {df['dataset'].nunique()}")
    print(f"  Processing types: {sorted(df['processing_type'].dropna().unique())}")
    print(f"  Sampling methods: {sorted(df['sampling_method'].dropna().unique())}")
    if "seed" in df.columns:
        print(f"  Unique seeds: {df['seed'].nunique(dropna=True)}")

    required_metrics = [
        "quantization_error_holdout",
        "quantization_error_train",
        "distortion_measure_holdout",
    ]
    present_required = [metric for metric in required_metrics if metric in df.columns]
    missing_required = [metric for metric in required_metrics if metric not in df.columns]
    print(f"  Required metric columns present: {present_required}")
    if missing_required:
        print(f"  Required metric columns missing: {missing_required}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export seed-preserving Optuna trial results to CSV")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing seed_* subdirectories with study_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--trial-source",
        choices=["all", "best"],
        default="all",
        help="Which trial set to export from each study JSON (default: all)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    if not results_dir.exists():
        print(f"Error: results directory does not exist: {results_dir}")
        return 1

    print(f"Reading raw results from: {results_dir}")
    print(f"Exporting trial source: {args.trial_source}")
    print(f"Output CSV: {output_path}")

    rows = collect_seed_trial_rows(results_dir=results_dir, trial_source=args.trial_source)
    if not rows:
        print("No rows found to export")
        return 1

    export_to_csv(rows=rows, output_path=output_path)
    print("\nSeed-preserving CSV export complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
