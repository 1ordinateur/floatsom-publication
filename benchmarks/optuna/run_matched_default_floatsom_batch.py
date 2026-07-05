#!/usr/bin/env python3
"""
Run matched FloatSOM benchmarks for batch/full_batch scenarios.

This script executes exactly one trial per key:
    (dataset, seed, topology, sampling_method)

By default the single trial is the enqueued default-parameter trial from
FloatSOMParams. Optional manual fixed hyperparameters can be injected so the
same matched runner can produce "global fixed-parameter" runs with the same
output schema as true-default runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""} and "floatsom" not in sys.modules:
    # Support direct execution from a checkout named floatsom-publication.
    _repo_root = Path(__file__).resolve().parents[2]
    _pkg = types.ModuleType("floatsom")
    _pkg.__file__ = str(_repo_root / "__init__.py")
    _pkg.__path__ = [str(_repo_root)]
    _pkg.__package__ = "floatsom"
    _pkg.__spec__ = importlib.machinery.ModuleSpec("floatsom", loader=None, is_package=True)
    _pkg.__spec__.submodule_search_locations = _pkg.__path__
    sys.modules["floatsom"] = _pkg

import numpy as np
import optuna
import pandas as pd

from floatsom.benchmarks.optuna.config.benchmark_config import Phase3BenchmarkConfig
from floatsom.benchmarks.optuna.config.parameters import PARAMETER_CONFIGS


SUPPORTED_TOPOLOGIES: Tuple[str, ...] = ("hexagonal", "mst", "rng")
SUPPORTED_SAMPLING_METHODS: Tuple[str, ...] = ("full", "random", "hdsssom")
SUPPORTED_EXECUTION_MODES: Tuple[str, ...] = ("run", "compare", "run-and-compare")
STRUCTURAL_FORCED_KEYS: Tuple[str, ...] = (
    "sampling_method",
    "processing_method",
    "batch_mode",
    "topology_type",
)
BOOLEAN_TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}
BOOLEAN_FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}
RUN_KEY_COLUMNS: Tuple[str, str, str, str] = ("dataset", "seed", "architecture", "sampling_method")
TOPOLOGY_ONLY_METRICS = {
    "topographic_error",
    "trustworthiness",
    "neighborhood_preservation",
    "distortion_measure",
    "topographic_function",
}
KNOWN_RUNS_CSV_NAMES: Tuple[str, ...] = (
    "matched_default_runs.csv",
    "matched_true_default_runs.csv",
    "matched_tuned_fixed_runs.csv",
    "matched_default_topology_diagnostics_runs.csv",
    "matched_tuned_topology_diagnostics_runs.csv",
)
MATCHED_TOPOLOGY_DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "mean_tied_rank",
    "node_utilization",
    "dead_node_fraction",
    "used_nodes",
    "dead_nodes",
    "total_nodes",
)
BALANCED_DIAGNOSTIC_COLUMNS: Dict[str, str] = {
    "mean_tied_rank": "balanced_mean_tied_rank_raw",
    "node_utilization": "balanced_node_utilization_raw",
    "dead_node_fraction": "balanced_dead_node_fraction_raw",
}
REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS: Tuple[str, ...] = tuple(
    f"{metric_name}_{split}"
    for metric_name in MATCHED_TOPOLOGY_DIAGNOSTIC_METRICS
    for split in ("holdout", "train")
) + tuple(BALANCED_DIAGNOSTIC_COLUMNS.values())
DIAGNOSTIC_REPORT_METRICS: Tuple[str, ...] = (
    "quantization_error_holdout",
    "quantization_error_train",
    "balanced_qe_raw",
    "mean_tied_rank_holdout",
    "mean_tied_rank_train",
    "balanced_mean_tied_rank_raw",
    "node_utilization_holdout",
    "node_utilization_train",
    "balanced_node_utilization_raw",
    "dead_node_fraction_holdout",
    "dead_node_fraction_train",
    "balanced_dead_node_fraction_raw",
)
LOWER_IS_BETTER_REPORT_METRICS = {
    "quantization_error_holdout",
    "quantization_error_train",
    "balanced_qe_raw",
    "mean_tied_rank_holdout",
    "mean_tied_rank_train",
    "balanced_mean_tied_rank_raw",
    "dead_node_fraction_holdout",
    "dead_node_fraction_train",
    "balanced_dead_node_fraction_raw",
}
HIGHER_IS_BETTER_REPORT_METRICS = {
    "node_utilization_holdout",
    "node_utilization_train",
    "balanced_node_utilization_raw",
}
REQUIRED_MATCHED_TOPOLOGY_REPORT_COLUMNS: Tuple[str, ...] = DIAGNOSTIC_REPORT_METRICS


def _resolve_dataset_config() -> Dict[str, Any]:
    return {
        "difficulty": "hard",
        "normalize": True,
    }


def _expand_objectives_for_split(base_objectives: Sequence[str], evaluation_split: str) -> List[str]:
    split = str(evaluation_split).strip().lower()
    if split not in {"both", "holdout", "train"}:
        raise ValueError(
            f"Unsupported evaluation split {evaluation_split!r}. Expected one of ['both', 'holdout', 'train']."
        )

    include_holdout = split in {"both", "holdout"}
    include_train = split in {"both", "train"}
    train_eligible = [metric for metric in base_objectives if metric not in TOPOLOGY_ONLY_METRICS]
    if split == "train" and not train_eligible:
        raise ValueError(
            "Train split requested but none of the objectives support training evaluation. "
            "Select at least one non-topology objective or use 'holdout'/'both'."
        )

    expanded: List[str] = []
    for metric in base_objectives:
        if include_holdout:
            expanded.append(f"{metric}_holdout")
        if include_train and metric not in TOPOLOGY_ONLY_METRICS:
            expanded.append(f"{metric}_train")
    return expanded


def _create_scenario_id(dataset_name: str, forced_params: Dict[str, Any]) -> str:
    parts = [
        dataset_name,
        forced_params.get("processing_method", "unknown"),
        forced_params.get("sampling_method", "unknown"),
        forced_params.get("batch_mode", "unknown"),
        forced_params.get("topology_type", "unknown"),
    ]
    if (
        str(forced_params.get("sampling_method", "")).strip().lower() == "random"
        and "target_proportion" in forced_params
    ):
        target = float(forced_params["target_proportion"])
        target_tag = f"{target:.6g}".replace(".", "p").replace("-", "m")
        parts.append(f"tp{target_tag}")
    return "_".join(str(part) for part in parts)


def _run_single_benchmark_lazy(**kwargs: Any) -> optuna.Study:
    from floatsom.benchmarks.optuna.core.single_benchmark import run_single_benchmark

    return run_single_benchmark(**kwargs)


def _save_study_json_lazy(**kwargs: Any) -> None:
    from floatsom.benchmarks.optuna.run_optuna import save_study_json

    save_study_json(**kwargs)


def _normalize_list(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    seen = set()
    for raw in values:
        for item in str(raw).split(","):
            token = item.strip().lower()
            if not token or token in seen:
                continue
            out.append(token)
            seen.add(token)
    return out


def _resolve_seeds(explicit_seeds: Optional[Sequence[int]], num_seeds: int, base_seed: int) -> List[int]:
    if explicit_seeds:
        return [int(seed) for seed in explicit_seeds]
    return [int(base_seed) + idx for idx in range(int(num_seeds))]


def _resolve_seeds_from_csv(path: str | Path) -> List[int]:
    csv_path = Path(path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Seed source CSV not found: {csv_path}")
    if not csv_path.is_file():
        raise IsADirectoryError(f"Seed source path is not a CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Seed source CSV is empty: {csv_path}")

    seed_col_candidates = ("pair_seed", "seed", "config_seed", "random_seed", "seed_name", "config_seed_name")
    seed_col = next((column for column in seed_col_candidates if column in df.columns), None)
    if seed_col is None:
        raise ValueError(
            f"Seed source CSV must contain one of {list(seed_col_candidates)}. Found columns: {list(df.columns)}"
        )

    numeric = pd.to_numeric(df[seed_col], errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"No parseable seed values were found in column '{seed_col}' from {csv_path}")

    raw_values = numeric.to_numpy(dtype=float)
    rounded = np.round(raw_values)
    if not np.allclose(raw_values, rounded):
        raise ValueError(
            f"Seed column '{seed_col}' in {csv_path} contains non-integer values; cannot derive deterministic seeds."
        )
    return sorted({int(value) for value in rounded.astype(int)})


def _resolve_requested_seeds(
    *,
    explicit_seeds: Optional[Sequence[int]],
    num_seeds: int,
    base_seed: int,
    seeds_from_csv: Optional[str],
) -> List[int]:
    if explicit_seeds and seeds_from_csv:
        raise ValueError("Use either --seeds or --seeds-from-csv, not both.")
    if seeds_from_csv:
        return _resolve_seeds_from_csv(seeds_from_csv)
    return _resolve_seeds(explicit_seeds, num_seeds, base_seed)


def _resolve_datasets(requested: Optional[Sequence[str]]) -> List[str]:
    available = list(Phase3BenchmarkConfig().datasets or [])
    if not requested:
        return available
    normalized = _normalize_list(requested)
    missing = [name for name in normalized if name not in available]
    if missing:
        raise ValueError(f"Unknown dataset(s): {missing}. Available: {available}")
    return normalized


def _resolve_topologies(requested: Optional[Sequence[str]]) -> List[str]:
    if not requested:
        return list(SUPPORTED_TOPOLOGIES)
    normalized = _normalize_list(requested)
    invalid = [name for name in normalized if name not in SUPPORTED_TOPOLOGIES]
    if invalid:
        raise ValueError(
            f"Unsupported topology value(s): {invalid}. Supported: {list(SUPPORTED_TOPOLOGIES)}"
        )
    return normalized


def _resolve_sampling_methods(requested: Optional[Sequence[str]]) -> List[str]:
    if not requested:
        return list(SUPPORTED_SAMPLING_METHODS)
    normalized = _normalize_list(requested)
    invalid = [name for name in normalized if name not in SUPPORTED_SAMPLING_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported sampling method value(s): {invalid}. Supported: {list(SUPPORTED_SAMPLING_METHODS)}"
        )
    return normalized


def _parse_boolean_token(raw_value: object, *, param_name: str) -> bool:
    text = str(raw_value).strip().lower()
    if text in BOOLEAN_TRUE_TOKENS:
        return True
    if text in BOOLEAN_FALSE_TOKENS:
        return False
    raise ValueError(
        f"Invalid boolean value for '{param_name}': {raw_value!r}. "
        f"Accepted true tokens: {sorted(BOOLEAN_TRUE_TOKENS)}; false tokens: {sorted(BOOLEAN_FALSE_TOKENS)}"
    )


def _coerce_param_value(param_name: str, raw_value: object) -> Any:
    config = PARAMETER_CONFIGS.get(param_name)
    if config is None:
        raise ValueError(f"Unknown parameter '{param_name}'.")

    param_type = str(config.get("type", "")).strip().lower()
    if param_type == "categorical":
        choices = list(config.get("choices") or [])
        if not choices:
            raise ValueError(f"Categorical parameter '{param_name}' has no configured choices.")
        if all(isinstance(choice, bool) for choice in choices):
            value = _parse_boolean_token(raw_value, param_name=param_name)
            if value not in choices:
                raise ValueError(f"Invalid value for '{param_name}': {value!r}. Valid choices: {choices}")
            return value
        raw_text = str(raw_value).strip()
        matches = [choice for choice in choices if str(choice).strip().lower() == raw_text.lower()]
        if not matches:
            raise ValueError(f"Invalid value for '{param_name}': {raw_value!r}. Valid choices: {choices}")
        return matches[0]

    if param_type == "int":
        if isinstance(raw_value, bool):
            raise ValueError(f"Invalid integer value for '{param_name}': {raw_value!r}")
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value for '{param_name}': {raw_value!r}") from exc
        if not float(numeric).is_integer():
            raise ValueError(f"Invalid integer value for '{param_name}': {raw_value!r}")
        value = int(numeric)
        min_val, max_val = config["range"]
        if value < int(min_val) or value > int(max_val):
            raise ValueError(f"Value for '{param_name}' out of range [{min_val}, {max_val}]: {value}")
        return value

    if param_type == "float":
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float value for '{param_name}': {raw_value!r}") from exc
        min_val, max_val = config["range"]
        if value < float(min_val) or value > float(max_val):
            raise ValueError(f"Value for '{param_name}' out of range [{min_val}, {max_val}]: {value}")
        return value

    raise ValueError(f"Unsupported parameter type for '{param_name}': {param_type!r}")


def _parse_fixed_param_entries(entries: Optional[Sequence[str]]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for entry in list(entries or []):
        text = str(entry).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Invalid --fixed-param value '{entry}'. Expected key=value (example: --fixed-param initial_radius=3.5)."
            )
        key, raw_value = text.split("=", 1)
        param_name = key.strip()
        if not param_name:
            raise ValueError(f"Invalid --fixed-param value '{entry}': empty parameter name.")
        if param_name in parsed:
            raise ValueError(f"Duplicate --fixed-param provided for '{param_name}'.")
        parsed[param_name] = raw_value.strip()
    return parsed


def _load_fixed_params_json(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    params_path = Path(path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Fixed-params JSON not found: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixed-params JSON must be an object mapping param names to values: {params_path}")
    out: Dict[str, object] = {}
    for key, value in payload.items():
        param_name = str(key).strip()
        if not param_name:
            raise ValueError(f"Fixed-params JSON contains an empty parameter key: {params_path}")
        if param_name in out:
            raise ValueError(f"Duplicate fixed parameter '{param_name}' found in {params_path}")
        out[param_name] = value
    return out


def _resolve_manual_fixed_params(
    cli_entries: Optional[Sequence[str]],
    json_path: Optional[str],
) -> Dict[str, Any]:
    combined_raw = _load_fixed_params_json(json_path)
    cli_raw = _parse_fixed_param_entries(cli_entries)
    overlap = sorted(set(combined_raw).intersection(cli_raw))
    if overlap:
        raise ValueError(
            "Duplicate fixed parameter keys were provided in both --fixed-params-json and --fixed-param: "
            f"{overlap}"
        )
    combined_raw.update(cli_raw)

    resolved: Dict[str, Any] = {}
    for param_name, raw_value in combined_raw.items():
        if param_name in STRUCTURAL_FORCED_KEYS:
            raise ValueError(
                f"'{param_name}' is controlled by matched key dimensions and cannot be provided via fixed params. "
                f"Use --sampling-methods/--topologies instead."
            )
        resolved[param_name] = _coerce_param_value(param_name, raw_value)
    return dict(sorted(resolved.items()))


def _resolve_manual_fixed_params_by_topology(
    *,
    topologies: Sequence[str],
    json_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if not json_path:
        return {}

    params_path = Path(json_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Topology fixed-params JSON not found: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Topology fixed-params JSON must be an object of topology->params mappings: "
            f"{params_path}"
        )

    selected_topologies = {str(topology).strip().lower() for topology in topologies}
    resolved_by_topology: Dict[str, Dict[str, Any]] = {}
    for topology_key, topology_params in payload.items():
        topology = str(topology_key).strip().lower()
        if topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unsupported topology key in topology fixed-params JSON: '{topology_key}'. "
                f"Supported: {list(SUPPORTED_TOPOLOGIES)}"
            )
        if topology not in selected_topologies:
            raise ValueError(
                f"Topology fixed-params JSON contains '{topology}', but this run is configured for "
                f"{sorted(selected_topologies)}."
            )
        if not isinstance(topology_params, dict):
            raise ValueError(
                f"Topology entry '{topology_key}' must map to an object of param->value entries."
            )
        out_params: Dict[str, Any] = {}
        for raw_param_name, raw_value in topology_params.items():
            param_name = str(raw_param_name).strip()
            if not param_name:
                raise ValueError(f"Topology '{topology_key}' contains an empty parameter name.")
            if param_name in out_params:
                raise ValueError(
                    f"Topology '{topology_key}' contains duplicate parameter '{param_name}'."
                )
            if param_name in STRUCTURAL_FORCED_KEYS:
                raise ValueError(
                    f"'{param_name}' is controlled by matched key dimensions and cannot be provided via topology fixed params."
                )
            out_params[param_name] = _coerce_param_value(param_name, raw_value)
        resolved_by_topology[topology] = dict(sorted(out_params.items()))
    return dict(sorted(resolved_by_topology.items()))


def _resolve_manual_fixed_params_by_sampling_topology(
    *,
    sampling_methods: Sequence[str],
    topologies: Sequence[str],
    json_path: Optional[str],
    allow_unselected_keys: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not json_path:
        return {}

    params_path = Path(json_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Sampling-topology fixed-params JSON not found: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Sampling-topology fixed-params JSON must be an object of "
            "sampling_method->topology->params mappings."
        )

    selected_sampling = {str(sampling).strip().lower() for sampling in sampling_methods}
    selected_topologies = {str(topology).strip().lower() for topology in topologies}
    resolved_by_sampling: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sampling_key, topology_map in payload.items():
        sampling = str(sampling_key).strip().lower()
        if sampling not in SUPPORTED_SAMPLING_METHODS:
            raise ValueError(
                f"Unsupported sampling key in sampling-topology fixed-params JSON: '{sampling_key}'. "
                f"Supported: {list(SUPPORTED_SAMPLING_METHODS)}"
            )
        if sampling not in selected_sampling:
            if allow_unselected_keys:
                continue
            raise ValueError(
                f"Sampling-topology fixed-params JSON contains sampling '{sampling}', but this run is configured for "
                f"{sorted(selected_sampling)}."
            )
        if not isinstance(topology_map, dict):
            raise ValueError(
                f"Sampling entry '{sampling_key}' must map to an object of topology->params mappings."
            )
        resolved_topology_map: Dict[str, Dict[str, Any]] = {}
        for topology_key, topology_params in topology_map.items():
            topology = str(topology_key).strip().lower()
            if topology not in SUPPORTED_TOPOLOGIES:
                raise ValueError(
                    f"Unsupported topology key under sampling '{sampling_key}': '{topology_key}'. "
                    f"Supported: {list(SUPPORTED_TOPOLOGIES)}"
                )
            if topology not in selected_topologies:
                if allow_unselected_keys:
                    continue
                raise ValueError(
                    f"Sampling-topology fixed-params JSON contains topology '{topology}' under sampling '{sampling}', "
                    f"but this run is configured for topologies {sorted(selected_topologies)}."
                )
            if not isinstance(topology_params, dict):
                raise ValueError(
                    f"Sampling '{sampling_key}' topology '{topology_key}' must map to an object of param->value entries."
                )
            out_params: Dict[str, Any] = {}
            for raw_param_name, raw_value in topology_params.items():
                param_name = str(raw_param_name).strip()
                if not param_name:
                    raise ValueError(
                        f"Sampling '{sampling_key}' topology '{topology_key}' contains an empty parameter name."
                    )
                if param_name in out_params:
                    raise ValueError(
                        f"Sampling '{sampling_key}' topology '{topology_key}' contains duplicate parameter '{param_name}'."
                    )
                if param_name in STRUCTURAL_FORCED_KEYS:
                    raise ValueError(
                        f"'{param_name}' is controlled by matched key dimensions and cannot be provided via sampling-topology fixed params."
                    )
                out_params[param_name] = _coerce_param_value(param_name, raw_value)
            resolved_topology_map[topology] = dict(sorted(out_params.items()))
        resolved_by_sampling[sampling] = dict(sorted(resolved_topology_map.items()))
    return dict(sorted(resolved_by_sampling.items()))


def _resolve_effective_manual_fixed_params(
    *,
    manual_fixed_params: Dict[str, Any],
    topology_fixed_params_by_topology: Dict[str, Dict[str, Any]],
    sampling_topology_fixed_params: Dict[str, Dict[str, Dict[str, Any]]],
    topology: str,
    sampling_method: str,
) -> Dict[str, Any]:
    topology_key = str(topology).strip().lower()
    sampling_key = str(sampling_method).strip().lower()
    effective = dict(manual_fixed_params)
    topology_overrides = topology_fixed_params_by_topology.get(topology_key, {})
    effective.update(topology_overrides)
    sampling_topology_overrides = sampling_topology_fixed_params.get(sampling_key, {}).get(topology_key, {})
    effective.update(sampling_topology_overrides)
    return dict(sorted(effective.items()))


def _validate_manual_fixed_params_for_scope(
    fixed_params: Dict[str, Any],
    *,
    processing_method: str,
    topologies: Sequence[str],
) -> None:
    processing = str(processing_method).strip().lower()
    topology_set = {str(topology).strip().lower() for topology in topologies}
    for param_name in fixed_params:
        config = PARAMETER_CONFIGS.get(param_name)
        if config is None:
            continue
        applicable_methods = config.get("applicable_processing_methods")
        if applicable_methods and processing not in {str(method).strip().lower() for method in applicable_methods}:
            raise ValueError(
                f"Parameter '{param_name}' is not applicable to processing_method='{processing}'. "
                f"Allowed methods: {list(applicable_methods)}"
            )
        applicable_topologies = config.get("applicable_topologies")
        if applicable_topologies:
            allowed_topologies = {str(topology).strip().lower() for topology in applicable_topologies}
            invalid_topologies = sorted(topology_set.difference(allowed_topologies))
            if invalid_topologies:
                raise ValueError(
                    f"Parameter '{param_name}' is not applicable to topology set {sorted(topology_set)}. "
                    f"Allowed topologies: {sorted(allowed_topologies)}."
                )


def _validate_sampling_topology_override_coverage(
    *,
    fixed_params_by_sampling_topology: Dict[str, Dict[str, Dict[str, Any]]],
    sampling_methods: Sequence[str],
    topologies: Sequence[str],
) -> None:
    if not fixed_params_by_sampling_topology:
        raise ValueError(
            "Expected sampling-topology fixed params for tuned-fixed profile but received an empty mapping."
        )

    selected_sampling = sorted({str(sampling).strip().lower() for sampling in sampling_methods})
    selected_topologies = sorted({str(topology).strip().lower() for topology in topologies})
    missing_pairs: List[str] = []
    for sampling in selected_sampling:
        topology_map = fixed_params_by_sampling_topology.get(sampling, {})
        for topology in selected_topologies:
            if topology not in topology_map:
                missing_pairs.append(f"{sampling}:{topology}")

    if missing_pairs:
        preview = ", ".join(missing_pairs[:12])
        more = f" (+{len(missing_pairs) - 12} more)" if len(missing_pairs) > 12 else ""
        raise ValueError(
            "Sampling-topology fixed params are incomplete for requested sampling/topology combinations. "
            f"Missing entries: {preview}{more}. "
            "Provide entries for every requested sampling/topology pair."
        )


def _validate_csv_name(filename: str, *, arg_name: str) -> str:
    text = str(filename).strip()
    if not text:
        raise ValueError(f"{arg_name} must be a non-empty filename ending with .csv")
    if "/" in text or "\\" in text:
        raise ValueError(f"{arg_name} must be a filename only, not a path: {filename!r}")
    if not text.lower().endswith(".csv"):
        raise ValueError(f"{arg_name} must end with .csv: {filename!r}")
    return text


def _validate_json_name(filename: str, *, arg_name: str) -> str:
    text = str(filename).strip()
    if not text:
        raise ValueError(f"{arg_name} must be a non-empty filename ending with .json")
    if "/" in text or "\\" in text:
        raise ValueError(f"{arg_name} must be a filename only, not a path: {filename!r}")
    if not text.lower().endswith(".json"):
        raise ValueError(f"{arg_name} must end with .json: {filename!r}")
    return text


def _validate_markdown_name(filename: str, *, arg_name: str) -> str:
    text = str(filename).strip()
    if not text:
        raise ValueError(f"{arg_name} must be a non-empty filename ending with .md")
    if "/" in text or "\\" in text:
        raise ValueError(f"{arg_name} must be a filename only, not a path: {filename!r}")
    if not text.lower().endswith(".md"):
        raise ValueError(f"{arg_name} must end with .md: {filename!r}")
    return text


def _validate_subdir_name(value: str, *, arg_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{arg_name} must be a non-empty relative directory name.")
    path_obj = Path(text)
    if path_obj.is_absolute():
        raise ValueError(f"{arg_name} must be a relative directory name, not an absolute path: {value!r}")
    return text


def _resolve_execution_mode(raw_value: str) -> str:
    mode = str(raw_value).strip().lower()
    if mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported execution mode {raw_value!r}. Supported: {list(SUPPORTED_EXECUTION_MODES)}"
        )
    return mode


def _load_runs_df_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Completed runs CSV not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Completed runs CSV path is not a file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Completed runs CSV is empty: {path}")
    return df


def _discover_completed_runs_csv(completed_runs_dir: Path) -> Path:
    metadata_path = completed_runs_dir / "RUN_METADATA.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_runs_csv = metadata.get("runs_csv")
        if metadata_runs_csv:
            candidate = Path(str(metadata_runs_csv)).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate

    csv_candidates = sorted(
        path
        for path in completed_runs_dir.glob("*.csv")
        if path.is_file() and not path.name.endswith("_failures.csv")
    )
    if not csv_candidates:
        raise FileNotFoundError(
            f"No completed runs CSV found in {completed_runs_dir}. "
            "Expected a RUN_METADATA.json pointer or a non-failure CSV."
        )
    if len(csv_candidates) == 1:
        return csv_candidates[0]

    named_candidates = [path for path in csv_candidates if path.name in KNOWN_RUNS_CSV_NAMES]
    if len(named_candidates) == 1:
        return named_candidates[0]

    raise ValueError(
        "Completed runs directory contains multiple candidate CSVs. "
        f"Pass --completed-runs-csv explicitly. Candidates: {[path.name for path in csv_candidates]}"
    )


def _resolve_completed_runs_input(
    *,
    completed_runs_dir: Optional[str],
    completed_runs_csv: Optional[str],
) -> Tuple[pd.DataFrame, Path]:
    if bool(completed_runs_dir) == bool(completed_runs_csv):
        raise ValueError("Provide exactly one of --completed-runs-dir or --completed-runs-csv.")

    if completed_runs_csv:
        runs_csv = Path(completed_runs_csv).resolve()
    else:
        runs_dir = Path(str(completed_runs_dir)).resolve()
        if not runs_dir.exists():
            raise FileNotFoundError(f"Completed runs directory not found: {runs_dir}")
        if not runs_dir.is_dir():
            raise NotADirectoryError(f"Completed runs path is not a directory: {runs_dir}")
        runs_csv = _discover_completed_runs_csv(runs_dir)

    return _load_runs_df_from_csv(runs_csv), runs_csv


def _generate_comparison_report(
    *,
    runs_df: pd.DataFrame,
    output_dir: Path,
    compare_against_csv: str,
    compare_output_subdir: str,
    compare_markdown_name: str,
    compare_split_policy: str,
) -> Tuple[Path, Path]:
    reference_csv = Path(compare_against_csv).resolve()
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV for comparison not found: {reference_csv}")

    from floatsom.benchmarks.optuna.optuna_results_analysis.modules.parameter_analysis.tuned_vs_default import (
        TunedDefaultMetric,
        build_tuned_vs_external_default_report_from_csv,
    )

    comparison_output_dir = output_dir / str(compare_output_subdir).strip()
    metric_specs = [
        TunedDefaultMetric("quantization_error_holdout", "QE Holdout", "qe_holdout"),
        TunedDefaultMetric("quantization_error_train", "QE Train", "qe_train"),
        TunedDefaultMetric("balanced_qe_raw", "Balanced QE", "balanced_qe_raw"),
    ]
    comparison_report = build_tuned_vs_external_default_report_from_csv(
        tuned_df=runs_df,
        default_csv_path=reference_csv,
        metrics=metric_specs,
        output_dir=comparison_output_dir,
        markdown_filename=str(compare_markdown_name),
        split_policy=str(compare_split_policy),
    )
    return comparison_output_dir.resolve(), comparison_report.resolve()


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    try:
        return df.to_markdown(index=False, disable_numparse=True)
    except ImportError:
        return df.to_string(index=False)


def _metric_direction(metric_name: str) -> str:
    if metric_name in LOWER_IS_BETTER_REPORT_METRICS:
        return "lower"
    if metric_name in HIGHER_IS_BETTER_REPORT_METRICS:
        return "higher"
    return "unspecified"


def _signed_effect(reference_values: np.ndarray, comparator_values: np.ndarray, metric_name: str) -> np.ndarray:
    if _metric_direction(metric_name) == "higher":
        return comparator_values - reference_values
    return reference_values - comparator_values


def _paired_ttest_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    if np.allclose(finite, 0.0):
        return 1.0
    if finite.size < 2:
        return float("nan")
    if float(np.std(finite, ddof=1)) == 0.0:
        return 0.0
    try:
        from scipy import stats

        return float(stats.ttest_1samp(finite, popmean=0.0, nan_policy="omit").pvalue)
    except Exception:
        return float("nan")


def _mean_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_value = float(np.mean(finite))
    if finite.size < 2:
        return mean_value, float("nan"), float("nan")
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return mean_value, mean_value, mean_value
    try:
        from scipy import stats

        critical = float(stats.t.ppf(0.975, df=int(finite.size - 1)))
    except Exception:
        critical = 1.96
    half_width = critical * sd / float(np.sqrt(finite.size))
    return mean_value, float(mean_value - half_width), float(mean_value + half_width)


def _cohen_dz(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    mean_value = float(np.mean(finite))
    if sd == 0.0:
        if mean_value == 0.0:
            return 0.0
        return float(np.sign(mean_value) * np.inf)
    return float(mean_value / sd)


def _benjamini_hochberg_qvalues(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([float(value) if value is not None else np.nan for value in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return q.tolist()

    ordered_indices = finite_indices[np.argsort(p[finite_indices])]
    ordered_p = p[ordered_indices]
    m = float(ordered_p.size)
    raw_q = ordered_p * m / np.arange(1, ordered_p.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    q[ordered_indices] = adjusted
    return q.tolist()


def _load_profile_runs_csv(path_value: object, profile: str) -> pd.DataFrame:
    csv_path = Path(str(path_value)).resolve()
    df = _load_runs_df_from_csv(csv_path)
    out = df.copy()
    out["profile"] = str(profile)
    if "architecture" not in out.columns and "config_topology_type" in out.columns:
        out["architecture"] = out["config_topology_type"]
    if "sampling_method" not in out.columns and "config_sampling_method" in out.columns:
        out["sampling_method"] = out["config_sampling_method"]
    out["architecture"] = out["architecture"].astype(str).str.strip().str.lower()
    out["sampling_method"] = out["sampling_method"].astype(str).str.strip().str.lower()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    _validate_required_numeric_columns(
        out,
        required_columns=REQUIRED_MATCHED_TOPOLOGY_REPORT_COLUMNS,
        context=f"Matched topology diagnostic report CSV {csv_path}",
    )
    return out


def _load_manifest_profile_runs(manifest_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    default_csv = manifest.get("default_runs_file") or manifest.get("true_default", {}).get("runs_csv")
    tuned_csv = manifest.get("default_aware_tuned_runs_file") or manifest.get("tuned_fixed", {}).get("runs_csv")
    if not default_csv or not tuned_csv:
        raise ValueError(f"Manifest does not contain both default and tuned runs CSV paths: {manifest_path}")

    default_df = _load_profile_runs_csv(default_csv, "true_default")
    tuned_df = _load_profile_runs_csv(tuned_csv, "tuned_fixed")
    return manifest, pd.concat([default_df, tuned_df], axis=0, ignore_index=True)


def _diagnostic_dataset_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_numeric_columns(
        combined_df,
        required_columns=REQUIRED_MATCHED_TOPOLOGY_REPORT_COLUMNS,
        context="Combined matched topology diagnostic runs",
    )
    metric_columns = [column for column in DIAGNOSTIC_REPORT_METRICS if column in combined_df.columns]
    if not metric_columns:
        raise ValueError("No diagnostic metric columns were found in combined profile runs.")

    group_columns = ["profile", "dataset", "architecture", "sampling_method"]
    aggregation = {column: "median" for column in metric_columns}
    aggregation["seed"] = "nunique"
    summary = (
        combined_df.groupby(group_columns, dropna=False)
        .agg(aggregation)
        .reset_index()
        .rename(columns={"architecture": "topology", "seed": "n_seeds"})
    )
    ordered_columns = ["profile", "dataset", "topology", "sampling_method", "n_seeds"] + metric_columns
    return summary[ordered_columns].sort_values(
        ["profile", "dataset", "topology", "sampling_method"]
    ).reset_index(drop=True)


def _build_metric_pairs(
    *,
    combined_df: pd.DataFrame,
    reference_filter: Dict[str, str],
    comparator_filter: Dict[str, str],
    metric_name: str,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    ref_df = combined_df.copy()
    cmp_df = combined_df.copy()
    for column, value in reference_filter.items():
        ref_df = ref_df[ref_df[column].astype(str).str.lower() == str(value).lower()]
    for column, value in comparator_filter.items():
        cmp_df = cmp_df[cmp_df[column].astype(str).str.lower() == str(value).lower()]

    if metric_name not in ref_df.columns or metric_name not in cmp_df.columns:
        return pd.DataFrame()

    ref = ref_df[list(key_columns) + [metric_name]].rename(columns={metric_name: "reference_value"})
    cmp = cmp_df[list(key_columns) + [metric_name]].rename(columns={metric_name: "comparator_value"})
    pairs = ref.merge(cmp, on=list(key_columns), how="inner")
    if pairs.empty:
        return pairs

    pairs["reference_value"] = pd.to_numeric(pairs["reference_value"], errors="coerce")
    pairs["comparator_value"] = pd.to_numeric(pairs["comparator_value"], errors="coerce")
    pairs = pairs.dropna(subset=["reference_value", "comparator_value"]).copy()
    if pairs.empty:
        return pairs
    pairs["raw_delta_comparator_minus_reference"] = pairs["comparator_value"] - pairs["reference_value"]
    pairs["signed_effect_favoring_comparator"] = _signed_effect(
        pairs["reference_value"].to_numpy(dtype=float),
        pairs["comparator_value"].to_numpy(dtype=float),
        metric_name,
    )
    return pairs


def _diagnostic_paired_summaries(combined_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        {
            "comparison": "default_hexagonal_vs_default_mst",
            "reference_label": "true_default:hexagonal",
            "comparator_label": "true_default:mst",
            "reference_filter": {"profile": "true_default", "architecture": "hexagonal"},
            "comparator_filter": {"profile": "true_default", "architecture": "mst"},
            "key_columns": ("dataset", "seed", "sampling_method"),
        },
        {
            "comparison": "default_hexagonal_vs_default_rng",
            "reference_label": "true_default:hexagonal",
            "comparator_label": "true_default:rng",
            "reference_filter": {"profile": "true_default", "architecture": "hexagonal"},
            "comparator_filter": {"profile": "true_default", "architecture": "rng"},
            "key_columns": ("dataset", "seed", "sampling_method"),
        },
        {
            "comparison": "tuned_hexagonal_vs_tuned_mst",
            "reference_label": "tuned_fixed:hexagonal",
            "comparator_label": "tuned_fixed:mst",
            "reference_filter": {"profile": "tuned_fixed", "architecture": "hexagonal"},
            "comparator_filter": {"profile": "tuned_fixed", "architecture": "mst"},
            "key_columns": ("dataset", "seed", "sampling_method"),
        },
        {
            "comparison": "tuned_hexagonal_vs_tuned_rng",
            "reference_label": "tuned_fixed:hexagonal",
            "comparator_label": "tuned_fixed:rng",
            "reference_filter": {"profile": "tuned_fixed", "architecture": "hexagonal"},
            "comparator_filter": {"profile": "tuned_fixed", "architecture": "rng"},
            "key_columns": ("dataset", "seed", "sampling_method"),
        },
        {
            "comparison": "tuned_vs_default_hexagonal",
            "reference_label": "true_default:hexagonal",
            "comparator_label": "tuned_fixed:hexagonal",
            "reference_filter": {"profile": "true_default", "architecture": "hexagonal"},
            "comparator_filter": {"profile": "tuned_fixed", "architecture": "hexagonal"},
            "key_columns": ("dataset", "seed", "sampling_method", "architecture"),
        },
        {
            "comparison": "tuned_vs_default_mst",
            "reference_label": "true_default:mst",
            "comparator_label": "tuned_fixed:mst",
            "reference_filter": {"profile": "true_default", "architecture": "mst"},
            "comparator_filter": {"profile": "tuned_fixed", "architecture": "mst"},
            "key_columns": ("dataset", "seed", "sampling_method", "architecture"),
        },
        {
            "comparison": "tuned_vs_default_rng",
            "reference_label": "true_default:rng",
            "comparator_label": "tuned_fixed:rng",
            "reference_filter": {"profile": "true_default", "architecture": "rng"},
            "comparator_filter": {"profile": "tuned_fixed", "architecture": "rng"},
            "key_columns": ("dataset", "seed", "sampling_method", "architecture"),
        },
    ]

    metric_columns = [column for column in DIAGNOSTIC_REPORT_METRICS if column in combined_df.columns]
    rows: List[Dict[str, Any]] = []
    for comparison in comparisons:
        for metric_name in metric_columns:
            pairs = _build_metric_pairs(
                combined_df=combined_df,
                reference_filter=comparison["reference_filter"],
                comparator_filter=comparison["comparator_filter"],
                metric_name=metric_name,
                key_columns=comparison["key_columns"],
            )
            effects = (
                pairs["signed_effect_favoring_comparator"].to_numpy(dtype=float)
                if not pairs.empty
                else np.asarray([], dtype=float)
            )
            mean_effect, ci_low, ci_high = _mean_ci95(effects)
            row = {
                "comparison": comparison["comparison"],
                "reference": comparison["reference_label"],
                "comparator": comparison["comparator_label"],
                "metric": metric_name,
                "direction": _metric_direction(metric_name),
                "n_pairs": int(np.isfinite(effects).sum()),
                "mean_signed_effect_favoring_comparator": mean_effect,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "effect_size_cohen_dz": _cohen_dz(effects),
                "raw_p": _paired_ttest_pvalue(effects),
                "comparator_wins": int(np.sum(effects > 0)),
                "reference_wins": int(np.sum(effects < 0)),
                "ties": int(np.sum(effects == 0)),
            }
            rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q"] = _benjamini_hochberg_qvalues(summary["raw_p"].tolist())
    return summary


def _generate_matched_topology_diagnostic_report(
    *,
    manifest_path: Path,
    output_dir: Path,
    markdown_name: str,
) -> Dict[str, str]:
    manifest, combined_df = _load_manifest_profile_runs(manifest_path)
    _ = manifest
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = _diagnostic_dataset_summary(combined_df)
    paired_summary = _diagnostic_paired_summaries(combined_df)

    dataset_summary_path = output_dir / "supp_matched_topology_diagnostics_by_dataset.tsv"
    paired_summary_path = output_dir / "supp_matched_topology_diagnostics_paired_summaries.tsv"
    markdown_path = output_dir / _validate_markdown_name(markdown_name, arg_name="markdown_name")

    dataset_summary.to_csv(dataset_summary_path, sep="\t", index=False)
    paired_summary.to_csv(paired_summary_path, sep="\t", index=False)

    markdown_lines = [
        "# Matched Topology Diagnostics",
        "",
        "Positive signed effects in the paired table favor the comparator after applying metric directionality.",
        "Lower is better for QE, mean tied rank, and dead-node fraction; higher is better for node utilization.",
        "",
        "## Supplementary Diagnostics By Dataset",
        "",
        _to_markdown_table(dataset_summary),
        "",
        "## Paired Summaries",
        "",
        _to_markdown_table(paired_summary),
        "",
    ]
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    return {
        "diagnostic_dataset_summary_tsv": str(dataset_summary_path.resolve()),
        "diagnostic_paired_summary_tsv": str(paired_summary_path.resolve()),
        "diagnostic_markdown_report": str(markdown_path.resolve()),
    }


def _normalize_run_key(dataset: object, seed: object, topology: object, sampling_method: object) -> Tuple[str, int, str, str]:
    dataset_key = str(dataset).strip().lower()
    topology_key = str(topology).strip().lower()
    sampling_key = str(sampling_method).strip().lower()
    if not dataset_key or not topology_key or not sampling_key:
        raise ValueError("Cannot build run key with empty dataset/topology/sampling values.")

    seed_numeric = pd.to_numeric(pd.Series([seed]), errors="coerce").iloc[0]
    if pd.isna(seed_numeric):
        raise ValueError(f"Cannot parse seed value for run key: {seed!r}")
    seed_float = float(seed_numeric)
    if not float(seed_float).is_integer():
        raise ValueError(f"Run key seed must be integer-like, received: {seed!r}")
    seed_key = int(seed_float)
    return dataset_key, seed_key, topology_key, sampling_key


def _extract_completed_run_keys(existing_df: pd.DataFrame) -> set[Tuple[str, int, str, str]]:
    if existing_df.empty:
        return set()
    missing = [column for column in RUN_KEY_COLUMNS if column not in existing_df.columns]
    if missing:
        raise ValueError(
            "Resume CSV is missing required columns for key matching: "
            f"{missing}. Required: {list(RUN_KEY_COLUMNS)}"
        )
    keys: set[Tuple[str, int, str, str]] = set()
    for _, row in existing_df.iterrows():
        keys.add(
            _normalize_run_key(
                row.get("dataset"),
                row.get("seed"),
                row.get("architecture"),
                row.get("sampling_method"),
            )
        )
    return keys


def _dedupe_runs_df_by_key(existing_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return existing_df.copy()
    required = list(RUN_KEY_COLUMNS)
    missing = [column for column in required if column not in existing_df.columns]
    if missing:
        raise ValueError(
            "Resume CSV is missing required columns for deduplication: "
            f"{missing}. Required: {required}"
        )
    deduped = existing_df.drop_duplicates(subset=required, keep="first").copy()
    return deduped.reset_index(drop=True)


def _validate_resume_profile_compatibility(
    *,
    existing_df: pd.DataFrame,
    expected_profile: str,
    expected_split: str,
) -> None:
    if existing_df.empty:
        return

    if "run_profile" in existing_df.columns:
        existing_profiles = (
            existing_df["run_profile"].astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})
        )
        non_empty_profiles = sorted({value for value in existing_profiles.tolist() if value})
        if non_empty_profiles and expected_profile.strip().lower() not in set(non_empty_profiles):
            raise ValueError(
                "Resume CSV appears to belong to a different profile. "
                f"Expected profile '{expected_profile}', found profiles {non_empty_profiles}."
            )

    if "evaluation_split" in existing_df.columns:
        existing_splits = (
            existing_df["evaluation_split"].astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})
        )
        non_empty_splits = sorted({value for value in existing_splits.tolist() if value})
        if non_empty_splits and expected_split.strip().lower() not in set(non_empty_splits):
            raise ValueError(
                "Resume CSV appears to use a different evaluation split. "
                f"Expected split '{expected_split}', found splits {non_empty_splits}."
            )


def _format_missing_or_nonfinite_columns(columns: Sequence[str], *, max_items: int = 12) -> str:
    values = [str(column) for column in columns]
    shown = values[:max_items]
    suffix = "" if len(values) <= max_items else f" ... (+{len(values) - max_items} more)"
    return ", ".join(shown) + suffix


def _validate_required_numeric_values(
    row: Dict[str, Any],
    *,
    required_columns: Sequence[str],
    context: str,
) -> None:
    missing_or_nonfinite: List[str] = []
    for column in required_columns:
        if column not in row:
            missing_or_nonfinite.append(column)
            continue
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            missing_or_nonfinite.append(column)
            continue
        if not np.isfinite(value):
            missing_or_nonfinite.append(column)

    if missing_or_nonfinite:
        raise ValueError(
            f"{context} has missing or non-finite required matched topology diagnostics: "
            f"{_format_missing_or_nonfinite_columns(missing_or_nonfinite)}"
        )


def _validate_required_numeric_columns(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    context: str,
) -> None:
    if df.empty:
        return

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required matched topology diagnostic columns: "
            f"{_format_missing_or_nonfinite_columns(missing)}"
        )

    numeric_df = df[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric_df.to_numpy(dtype=float)
    nonfinite_positions = np.argwhere(~np.isfinite(numeric_values))
    if nonfinite_positions.size == 0:
        return

    examples: List[str] = []
    for row_idx, col_idx in nonfinite_positions[:8]:
        row = df.iloc[int(row_idx)]
        run_key = (
            row.get("dataset", "?"),
            row.get("seed", "?"),
            row.get("architecture", "?"),
            row.get("sampling_method", "?"),
        )
        examples.append(f"row={int(row_idx)} key={run_key} column={required_columns[int(col_idx)]}")
    suffix = "" if len(nonfinite_positions) <= 8 else f" ... (+{len(nonfinite_positions) - 8} more)"
    raise ValueError(
        f"{context} contains non-finite matched topology diagnostics: "
        f"{'; '.join(examples)}{suffix}"
    )


def _first_completed_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trial found in study.")
    return min(completed, key=lambda trial: int(trial.number))


def _extract_metric(trial: optuna.trial.FrozenTrial, metric_name: str, split: str) -> float:
    split_map = {
        "holdout": "metrics_holdout",
        "train": "metrics_train",
    }
    key = split_map[split]
    user_attrs = dict(trial.user_attrs or {})
    nested = user_attrs.get(key, {})
    if isinstance(nested, dict) and metric_name in nested:
        value = nested.get(metric_name)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    fallback = user_attrs.get(f"{metric_name}_{split}")
    if fallback is not None and np.isfinite(float(fallback)):
        return float(fallback)
    raise KeyError(f"Missing metric '{metric_name}_{split}' in trial user_attrs.")


def _extract_optional_metric(trial: optuna.trial.FrozenTrial, metric_name: str, split: str) -> float:
    try:
        return _extract_metric(trial, metric_name, split)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _balanced_optional_metric(holdout_value: float, train_value: float) -> float:
    if np.isfinite(float(holdout_value)) and np.isfinite(float(train_value)):
        return float((float(holdout_value) + float(train_value)) / 2.0)
    return float("nan")


def _build_success_row(
    *,
    trial: optuna.trial.FrozenTrial,
    dataset: str,
    seed: int,
    topology: str,
    sampling_method: str,
    evaluation_split: str,
    forced_params: Dict[str, Any],
    manual_fixed_params: Dict[str, Any],
    run_profile: str,
    run_label: str,
) -> Dict[str, Any]:
    qe_holdout = _extract_metric(trial, "quantization_error", "holdout")
    qe_train = _extract_metric(trial, "quantization_error", "train")
    balanced_qe_raw = float((qe_holdout + qe_train) / 2.0)
    diagnostic_metrics: Dict[str, float] = {}
    for metric_name in MATCHED_TOPOLOGY_DIAGNOSTIC_METRICS:
        holdout_value = _extract_optional_metric(trial, metric_name, "holdout")
        train_value = _extract_optional_metric(trial, metric_name, "train")
        diagnostic_metrics[f"{metric_name}_holdout"] = holdout_value
        diagnostic_metrics[f"{metric_name}_train"] = train_value
        balanced_column = BALANCED_DIAGNOSTIC_COLUMNS.get(metric_name)
        if balanced_column:
            diagnostic_metrics[balanced_column] = _balanced_optional_metric(holdout_value, train_value)

    row: Dict[str, Any] = {
        "scenario_id": _create_scenario_id(dataset, forced_params),
        "dataset": dataset,
        "seed": int(seed),
        "processing_type": "batch",
        "sampling_method": sampling_method,
        "batch_mode": "full_batch",
        "architecture": topology,
        "config_topology_type": topology,
        "config_sampling_method": sampling_method,
        "config_processing_method": "batch",
        "config_batch_mode": "full_batch",
        "evaluation_split": str(evaluation_split).lower().strip(),
        "trial_number": int(trial.number),
        "quantization_error_holdout": qe_holdout,
        "quantization_error_train": qe_train,
        "balanced_qe_raw": balanced_qe_raw,
        **diagnostic_metrics,
        "run_profile": str(run_profile),
        "run_label": str(run_label),
        "status": "ok",
    }
    for param_name, value in dict(trial.params or {}).items():
        row[f"param_{param_name}"] = value
    for param_name, value in manual_fixed_params.items():
        row[f"param_{param_name}"] = value
    _validate_required_numeric_values(
        row,
        required_columns=REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS,
        context=(
            f"Matched topology row dataset={dataset!r} seed={int(seed)} "
            f"topology={topology!r} sampling={sampling_method!r}"
        ),
    )
    return row


def _build_failure_row(
    *,
    dataset: str,
    seed: int,
    topology: str,
    sampling_method: str,
    evaluation_split: str,
    error: Exception,
    run_profile: str,
    run_label: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "processing_type": "batch",
        "sampling_method": sampling_method,
        "batch_mode": "full_batch",
        "architecture": topology,
        "evaluation_split": str(evaluation_split).lower().strip(),
        "run_profile": str(run_profile),
        "run_label": str(run_label),
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def _execute_profile_runs(
    *,
    output_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    topologies: Sequence[str],
    sampling_methods: Sequence[str],
    dataset_config: Dict[str, Any],
    base_objectives: Sequence[str],
    expanded_objectives: Sequence[str],
    timeout: Optional[float],
    evaluation_split: str,
    save_study_json_enabled: bool,
    run_profile: str,
    run_label: str,
    runs_csv_name: str,
    manual_fixed_params: Dict[str, Any],
    manual_fixed_params_by_topology: Dict[str, Dict[str, Any]],
    manual_fixed_params_by_sampling_topology: Dict[str, Dict[str, Dict[str, Any]]],
    compare_against_csv: Optional[str],
    compare_output_subdir: str,
    compare_markdown_name: str,
    compare_split_policy: str,
    resume_enabled: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_csv_name = _validate_csv_name(runs_csv_name, arg_name="runs_csv_name")
    runs_csv = output_dir / runs_csv_name
    planned_runs = int(len(datasets) * len(seeds) * len(topologies) * len(sampling_methods))

    existing_runs_df = pd.DataFrame()
    completed_keys: set[Tuple[str, int, str, str]] = set()
    resume_existing_rows = 0
    resume_skipped_runs = 0
    if resume_enabled and runs_csv.exists():
        existing_runs_df = pd.read_csv(runs_csv)
        if not existing_runs_df.empty:
            _validate_resume_profile_compatibility(
                existing_df=existing_runs_df,
                expected_profile=run_profile,
                expected_split=evaluation_split,
            )
            _validate_required_numeric_columns(
                existing_runs_df,
                required_columns=REQUIRED_MATCHED_TOPOLOGY_ROW_COLUMNS,
                context=f"Resume CSV {runs_csv}",
            )
            deduped_runs_df = _dedupe_runs_df_by_key(existing_runs_df)
            dropped_duplicates = int(len(existing_runs_df) - len(deduped_runs_df))
            if dropped_duplicates > 0:
                print(
                    "Resume detected duplicate completed keys in existing runs CSV; "
                    f"dropping {dropped_duplicates} duplicate row(s)."
                )
            existing_runs_df = deduped_runs_df
            completed_keys = _extract_completed_run_keys(existing_runs_df)
            resume_existing_rows = int(len(existing_runs_df))

    print("Running matched FloatSOM runs")
    print(f"Output: {output_dir}")
    print(f"Datasets ({len(datasets)}): {list(datasets)}")
    print(f"Seeds ({len(seeds)}): {list(seeds)}")
    print(f"Topologies ({len(topologies)}): {list(topologies)}")
    print(f"Sampling methods ({len(sampling_methods)}): {list(sampling_methods)}")
    print(f"Evaluation split: {evaluation_split}")
    print("Processing: batch")
    print("Batch mode: full_batch")
    print(f"Run profile: {run_profile}")
    print(f"Run label: {run_label}")
    print(f"Resume enabled: {bool(resume_enabled)}")
    if resume_enabled:
        print(f"Resume existing rows: {resume_existing_rows}")
    if manual_fixed_params:
        print(f"Manual fixed params ({len(manual_fixed_params)}): {manual_fixed_params}")
    if manual_fixed_params_by_topology:
        print(
            "Manual fixed params by topology: "
            + ", ".join(
                f"{topology}={params}" for topology, params in sorted(manual_fixed_params_by_topology.items())
            )
        )
    if manual_fixed_params_by_sampling_topology:
        print(
            "Manual fixed params by sampling+topology: "
            + ", ".join(
                f"{sampling}={topology_map}"
                for sampling, topology_map in sorted(manual_fixed_params_by_sampling_topology.items())
            )
        )

    success_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    executed_runs = 0

    for dataset in datasets:
        for seed in seeds:
            for topology in topologies:
                for sampling_method in sampling_methods:
                    run_key = _normalize_run_key(dataset, seed, topology, sampling_method)
                    if run_key in completed_keys:
                        resume_skipped_runs += 1
                        continue
                    executed_runs += 1
                    effective_manual_fixed_params = _resolve_effective_manual_fixed_params(
                        manual_fixed_params=manual_fixed_params,
                        topology_fixed_params_by_topology=manual_fixed_params_by_topology,
                        sampling_topology_fixed_params=manual_fixed_params_by_sampling_topology,
                        topology=topology,
                        sampling_method=sampling_method,
                    )
                    forced_params = {
                        "sampling_method": sampling_method,
                        "processing_method": "batch",
                        "batch_mode": "full_batch",
                        "topology_type": topology,
                    }
                    forced_params.update(effective_manual_fixed_params)
                    print(
                        f"[{executed_runs}] dataset={dataset} seed={seed} topology={topology} sampling={sampling_method}"
                    )
                    try:
                        study = _run_single_benchmark_lazy(
                            algo_type="batch",
                            dataset_name=dataset,
                            dataset_config=dataset_config,
                            forced_params=forced_params,
                            seed=int(seed),
                            n_trials=1,
                            timeout=timeout,
                            objectives=list(base_objectives),
                            diagnostic_metrics=list(MATCHED_TOPOLOGY_DIAGNOSTIC_METRICS),
                            evaluation_split=evaluation_split,
                            output_dir=None,
                            use_ray_tune=False,
                            max_concurrent=None,
                        )
                        trial = _first_completed_trial(study)
                        success_rows.append(
                            _build_success_row(
                                trial=trial,
                                dataset=dataset,
                                seed=int(seed),
                                topology=topology,
                                sampling_method=sampling_method,
                                evaluation_split=evaluation_split,
                                forced_params=forced_params,
                                manual_fixed_params=effective_manual_fixed_params,
                                run_profile=run_profile,
                                run_label=run_label,
                            )
                        )
                        if save_study_json_enabled:
                            seed_dir = output_dir / f"seed_{int(seed)}"
                            seed_dir.mkdir(parents=True, exist_ok=True)
                            _save_study_json_lazy(
                                study=study,
                                output_dir=str(seed_dir),
                                dataset_name=dataset,
                                forced_params=forced_params,
                                objectives=list(expanded_objectives),
                                expected_trials=1,
                            )
                    except Exception as exc:
                        failure_rows.append(
                            _build_failure_row(
                                dataset=dataset,
                                seed=int(seed),
                                topology=topology,
                                sampling_method=sampling_method,
                                evaluation_split=evaluation_split,
                                error=exc,
                                run_profile=run_profile,
                                run_label=run_label,
                            )
                        )
                        print(f"  FAILED: {type(exc).__name__}: {exc}")

    if not success_rows and existing_runs_df.empty:
        raise RuntimeError("No successful runs were produced.")

    new_runs_df = pd.DataFrame(success_rows)
    if existing_runs_df.empty:
        runs_df = new_runs_df.copy()
    elif new_runs_df.empty:
        runs_df = existing_runs_df.copy()
    else:
        runs_df = pd.concat([existing_runs_df, new_runs_df], axis=0, ignore_index=True)

    if not runs_df.empty:
        runs_df = _dedupe_runs_df_by_key(runs_df)
        runs_df = runs_df.sort_values(
        ["dataset", "seed", "architecture", "sampling_method"], ascending=[True, True, True, True]
        ).reset_index(drop=True)
    runs_df.to_csv(runs_csv, index=False)

    failures_csv = output_dir / f"{Path(runs_csv_name).stem}_failures.csv"
    failure_df = pd.DataFrame(failure_rows)
    if resume_enabled and failures_csv.exists():
        prior_failures = pd.read_csv(failures_csv)
        if not prior_failures.empty:
            failure_df = pd.concat([prior_failures, failure_df], axis=0, ignore_index=True)
    if not failure_df.empty:
        failure_df.to_csv(failures_csv, index=False)
    elif failures_csv.exists():
        failures_csv.unlink()
        failures_csv = None
    else:
        failures_csv = None

    comparison_report: Optional[Path] = None
    comparison_output_dir: Optional[Path] = None
    if compare_against_csv:
        comparison_output_dir, comparison_report = _generate_comparison_report(
            runs_df=runs_df,
            output_dir=output_dir,
            compare_against_csv=str(compare_against_csv),
            compare_output_subdir=str(compare_output_subdir),
            compare_markdown_name=str(compare_markdown_name),
            compare_split_policy=str(compare_split_policy),
        )

    metadata = {
        "output_dir": str(output_dir.resolve()),
        "evaluation_split": str(evaluation_split),
        "processing_type": "batch",
        "batch_mode": "full_batch",
        "run_profile": run_profile,
        "run_label": run_label,
        "manual_fixed_params": manual_fixed_params,
        "manual_fixed_params_by_topology": manual_fixed_params_by_topology,
        "manual_fixed_params_by_sampling_topology": manual_fixed_params_by_sampling_topology,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "topologies": list(topologies),
        "sampling_methods": list(sampling_methods),
        "total_planned_runs": int(planned_runs),
        "resume_enabled": bool(resume_enabled),
        "resume_existing_rows": int(resume_existing_rows),
        "resume_skipped_runs": int(resume_skipped_runs),
        "executed_runs": int(executed_runs),
        "newly_successful_runs": int(len(success_rows)),
        "successful_runs": int(len(runs_df)),
        "successful_rows_total": int(len(runs_df)),
        "failed_runs": int(len(failure_rows)),
        "runs_csv": str(runs_csv.resolve()),
        "failures_csv": str(failures_csv.resolve()) if failures_csv is not None else None,
        "comparison_reference_csv": str(Path(compare_against_csv).resolve()) if compare_against_csv else None,
        "comparison_split_policy": str(compare_split_policy) if compare_against_csv else None,
        "comparison_output_dir": str(comparison_output_dir.resolve()) if comparison_output_dir else None,
        "comparison_report": str(comparison_report.resolve()) if comparison_report else None,
    }
    metadata_path = output_dir / "RUN_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved run rows to {runs_csv}")
    if resume_enabled:
        print(
            "Resume summary: "
            f"existing_rows={resume_existing_rows}, skipped_keys={resume_skipped_runs}, "
            f"executed_runs={executed_runs}, newly_successful_rows={len(success_rows)}"
        )
    if failures_csv is not None:
        print(f"Saved failure rows to {failures_csv}")
    if comparison_report is not None:
        print(f"Saved comparison report to {comparison_report}")

    metadata["metadata_json"] = str(metadata_path.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched FloatSOM runs for batch/full_batch (true-default or manual-fixed)."
    )
    parser.add_argument(
        "--execution-mode",
        choices=list(SUPPORTED_EXECUTION_MODES),
        default="run-and-compare",
        help=(
            "Execution mode: 'run' executes benchmarks only, 'compare' reuses an existing completed-runs CSV/dir, "
            "and 'run-and-compare' preserves the current integrated behavior."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for CSV/JSON artifacts.")
    parser.add_argument(
        "--completed-runs-dir",
        type=str,
        default=None,
        help="Existing completed-runs directory for --execution-mode compare.",
    )
    parser.add_argument(
        "--completed-runs-csv",
        type=str,
        default=None,
        help="Existing completed-runs CSV for --execution-mode compare.",
    )
    parser.add_argument("--datasets", nargs="+", type=str, default=None, help="Dataset names (space/comma separated).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list.")
    parser.add_argument(
        "--seeds-from-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV used to derive explicit seeds from pair_seed/seed columns. "
            "Cannot be combined with --seeds."
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Seed count when --seeds is not provided.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for generated seed list.")
    parser.add_argument("--topologies", nargs="+", type=str, default=None, help="Topologies to run.")
    parser.add_argument("--sampling-methods", nargs="+", type=str, default=None, help="Sampling methods to run.")
    parser.add_argument("--difficulty", type=str, default="hard", help="Dataset difficulty.")
    parser.add_argument("--evaluation-split", choices=["both", "holdout", "train"], default="both")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds.")
    parser.add_argument(
        "--scikit-learn-data-home",
        type=str,
        default=None,
        help="Optional sklearn dataset cache root (sets SCIKIT_LEARN_DATA).",
    )
    parser.add_argument(
        "--save-study-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist study_*.json in seed_<seed> directories (default: true).",
    )
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=None,
        help=(
            "Manual fixed hyperparameter override in key=value form. "
            "Can be repeated (example: --fixed-param initial_radius=3.5 --fixed-param use_momentum=false)."
        ),
    )
    parser.add_argument(
        "--fixed-params-json",
        type=str,
        default=None,
        help="Optional JSON file containing fixed hyperparameter overrides as a dictionary.",
    )
    parser.add_argument(
        "--fixed-params-by-topology-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing topology-specific fixed overrides as "
            '{"hexagonal": {...}, "mst": {...}, "rng": {...}}. '
            "These override global fixed params for the matching topology."
        ),
    )
    parser.add_argument(
        "--fixed-params-by-sampling-topology-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing sampling+topology-specific fixed overrides as "
            '{"full": {"hexagonal": {...}, "mst": {...}, "rng": {...}}, "random": {...}}. '
            "These override global and topology-specific fixed params for matching sampling+topology."
        ),
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label stored in CSV metadata columns to identify this run set.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume from existing output CSV(s) by skipping already completed "
            "(dataset, seed, topology, sampling_method) keys."
        ),
    )
    parser.add_argument(
        "--run-both-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run true-default and tuned-fixed profiles in one command and emit a manifest with both CSV paths."
        ),
    )
    parser.add_argument(
        "--true-default-output-subdir",
        type=str,
        default="true_default",
        help="Subdirectory under --output-dir for true-default profile when --run-both-profiles is enabled.",
    )
    parser.add_argument(
        "--tuned-fixed-output-subdir",
        type=str,
        default="tuned_fixed",
        help="Subdirectory under --output-dir for tuned-fixed profile when --run-both-profiles is enabled.",
    )
    parser.add_argument(
        "--true-default-runs-csv-name",
        type=str,
        default="matched_true_default_runs.csv",
        help="CSV filename for true-default profile when --run-both-profiles is enabled.",
    )
    parser.add_argument(
        "--tuned-fixed-runs-csv-name",
        type=str,
        default="matched_tuned_fixed_runs.csv",
        help="CSV filename for tuned-fixed profile when --run-both-profiles is enabled.",
    )
    parser.add_argument(
        "--profile-manifest-name",
        type=str,
        default="MATCHED_PROFILE_MANIFEST.json",
        help="Top-level manifest filename for --run-both-profiles outputs.",
    )
    parser.add_argument(
        "--generate-diagnostic-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --run-both-profiles is enabled, generate supplementary diagnostic TSVs and a Markdown report "
            "from the dual-profile manifest (default: true)."
        ),
    )
    parser.add_argument(
        "--diagnostic-report-markdown-name",
        type=str,
        default="MATCHED_TOPOLOGY_DIAGNOSTICS_SUMMARY.md",
        help="Markdown filename for the dual-profile matched topology diagnostic report.",
    )
    parser.add_argument(
        "--require-complete-sampling-topology-overrides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --run-both-profiles, require sampling-topology fixed params to cover every requested "
            "sampling/topology combination (default: true)."
        ),
    )
    parser.add_argument(
        "--runs-csv-name",
        type=str,
        default="matched_default_runs.csv",
        help="Output CSV filename for successful rows (default: matched_default_runs.csv).",
    )
    parser.add_argument(
        "--compare-against-csv",
        type=str,
        default=None,
        help=(
            "Optional reference runs CSV. When provided, generate paired comparison artifacts for this run output "
            "against the reference using existing tuned-vs-default analysis modules."
        ),
    )
    parser.add_argument(
        "--compare-output-subdir",
        type=str,
        default="comparison_vs_reference",
        help="Subdirectory under --output-dir for optional comparison artifacts.",
    )
    parser.add_argument(
        "--compare-markdown-name",
        type=str,
        default="MATCHED_RUNS_VS_REFERENCE_PAIRED_TTEST.md",
        help="Markdown filename for optional comparison summary.",
    )
    parser.add_argument(
        "--compare-split-policy",
        choices=["both", "holdout", "train"],
        default="both",
        help="Split policy for optional paired comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution_mode = _resolve_execution_mode(args.execution_mode)
    if args.scikit_learn_data_home:
        os.environ["SCIKIT_LEARN_DATA"] = str(args.scikit_learn_data_home)

    if execution_mode == "compare":
        if not args.compare_against_csv:
            raise ValueError("--compare-against-csv is required with --execution-mode compare.")
        runs_df, runs_csv = _resolve_completed_runs_input(
            completed_runs_dir=args.completed_runs_dir,
            completed_runs_csv=args.completed_runs_csv,
        )
        compare_base_dir = Path(args.output_dir).resolve() if args.output_dir else runs_csv.parent.resolve()
        compare_base_dir.mkdir(parents=True, exist_ok=True)
        comparison_output_dir, comparison_report = _generate_comparison_report(
            runs_df=runs_df,
            output_dir=compare_base_dir,
            compare_against_csv=str(args.compare_against_csv),
            compare_output_subdir=str(args.compare_output_subdir),
            compare_markdown_name=str(args.compare_markdown_name),
            compare_split_policy=str(args.compare_split_policy),
        )
        metadata = {
            "execution_mode": execution_mode,
            "completed_runs_csv": str(runs_csv),
            "comparison_reference_csv": str(Path(args.compare_against_csv).resolve()),
            "comparison_output_dir": str(comparison_output_dir),
            "comparison_report": str(comparison_report),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        metadata_path = compare_base_dir / "COMPARE_ONLY_METADATA.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Loaded completed runs from {runs_csv}")
        print(f"Saved comparison report to {comparison_report}")
        print(f"Saved comparison metadata to {metadata_path}")
        return 0

    if not args.output_dir:
        raise ValueError("--output-dir is required with --execution-mode run and run-and-compare.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _resolve_datasets(args.datasets)
    seeds = _resolve_requested_seeds(
        explicit_seeds=args.seeds,
        num_seeds=int(args.num_seeds),
        base_seed=int(args.base_seed),
        seeds_from_csv=args.seeds_from_csv,
    )
    topologies = _resolve_topologies(args.topologies)
    sampling_methods = _resolve_sampling_methods(args.sampling_methods)
    manual_fixed_params = _resolve_manual_fixed_params(args.fixed_param, args.fixed_params_json)
    manual_fixed_params_by_topology = _resolve_manual_fixed_params_by_topology(
        topologies=topologies,
        json_path=args.fixed_params_by_topology_json,
    )
    manual_fixed_params_by_sampling_topology = _resolve_manual_fixed_params_by_sampling_topology(
        sampling_methods=sampling_methods,
        topologies=topologies,
        json_path=args.fixed_params_by_sampling_topology_json,
        allow_unselected_keys=True,
    )
    _validate_manual_fixed_params_for_scope(
        manual_fixed_params,
        processing_method="batch",
        topologies=topologies,
    )
    for topology, topology_params in manual_fixed_params_by_topology.items():
        _validate_manual_fixed_params_for_scope(
            topology_params,
            processing_method="batch",
            topologies=[topology],
        )
    for sampling, topology_map in manual_fixed_params_by_sampling_topology.items():
        for topology, topology_params in topology_map.items():
            _validate_manual_fixed_params_for_scope(
                topology_params,
                processing_method="batch",
                topologies=[topology],
            )
    has_any_fixed_overrides = (
        bool(manual_fixed_params)
        or bool(manual_fixed_params_by_topology)
        or bool(manual_fixed_params_by_sampling_topology)
    )
    if (
        bool(args.run_both_profiles)
        and bool(args.require_complete_sampling_topology_overrides)
        and args.fixed_params_by_sampling_topology_json
    ):
        _validate_sampling_topology_override_coverage(
            fixed_params_by_sampling_topology=manual_fixed_params_by_sampling_topology,
            sampling_methods=sampling_methods,
            topologies=topologies,
        )

    dataset_config = _resolve_dataset_config()
    dataset_config["difficulty"] = str(args.difficulty)
    base_objectives = ["quantization_error"]
    expanded_objectives = _expand_objectives_for_split(base_objectives, args.evaluation_split)
    comparison_enabled = execution_mode == "run-and-compare"

    if bool(args.run_both_profiles):
        if not has_any_fixed_overrides:
            raise ValueError(
                "--run-both-profiles requires tuned-fixed overrides. "
                "Provide at least one of --fixed-param/--fixed-params-json/"
                "--fixed-params-by-topology-json/--fixed-params-by-sampling-topology-json."
            )
        if args.compare_against_csv:
            raise ValueError("--compare-against-csv is not supported with --run-both-profiles.")
        if str(args.runs_csv_name).strip() != "matched_default_runs.csv":
            raise ValueError(
                "--runs-csv-name is single-profile only. Use --true-default-runs-csv-name and "
                "--tuned-fixed-runs-csv-name with --run-both-profiles."
            )

        true_default_output_subdir = _validate_subdir_name(
            args.true_default_output_subdir,
            arg_name="--true-default-output-subdir",
        )
        tuned_fixed_output_subdir = _validate_subdir_name(
            args.tuned_fixed_output_subdir,
            arg_name="--tuned-fixed-output-subdir",
        )
        true_default_runs_csv_name = _validate_csv_name(
            args.true_default_runs_csv_name,
            arg_name="--true-default-runs-csv-name",
        )
        tuned_fixed_runs_csv_name = _validate_csv_name(
            args.tuned_fixed_runs_csv_name,
            arg_name="--tuned-fixed-runs-csv-name",
        )
        manifest_name = _validate_json_name(args.profile_manifest_name, arg_name="--profile-manifest-name")

        true_default_output_dir = output_dir / true_default_output_subdir
        tuned_fixed_output_dir = output_dir / tuned_fixed_output_subdir

        print("=== Profile 1/2: true_default ===")
        true_default_result = _execute_profile_runs(
            output_dir=true_default_output_dir,
            datasets=datasets,
            seeds=seeds,
            topologies=topologies,
            sampling_methods=sampling_methods,
            dataset_config=dict(dataset_config),
            base_objectives=base_objectives,
            expanded_objectives=expanded_objectives,
            timeout=args.timeout,
            evaluation_split=str(args.evaluation_split),
            save_study_json_enabled=bool(args.save_study_json),
            run_profile="true_default",
            run_label="true_default",
            runs_csv_name=true_default_runs_csv_name,
            manual_fixed_params={},
            manual_fixed_params_by_topology={},
            manual_fixed_params_by_sampling_topology={},
            compare_against_csv=None,
            compare_output_subdir=str(args.compare_output_subdir),
            compare_markdown_name=str(args.compare_markdown_name),
            compare_split_policy=str(args.compare_split_policy),
            resume_enabled=bool(args.resume),
        )

        print("=== Profile 2/2: tuned_fixed ===")
        tuned_fixed_result = _execute_profile_runs(
            output_dir=tuned_fixed_output_dir,
            datasets=datasets,
            seeds=seeds,
            topologies=topologies,
            sampling_methods=sampling_methods,
            dataset_config=dict(dataset_config),
            base_objectives=base_objectives,
            expanded_objectives=expanded_objectives,
            timeout=args.timeout,
            evaluation_split=str(args.evaluation_split),
            save_study_json_enabled=bool(args.save_study_json),
            run_profile="tuned_fixed",
            run_label="tuned_fixed",
            runs_csv_name=tuned_fixed_runs_csv_name,
            manual_fixed_params=manual_fixed_params,
            manual_fixed_params_by_topology=manual_fixed_params_by_topology,
            manual_fixed_params_by_sampling_topology=manual_fixed_params_by_sampling_topology,
            compare_against_csv=str(true_default_result["runs_csv"]) if comparison_enabled else None,
            compare_output_subdir=str(args.compare_output_subdir),
            compare_markdown_name=str(args.compare_markdown_name),
            compare_split_policy=str(args.compare_split_policy),
            resume_enabled=bool(args.resume),
        )

        manifest = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "output_dir": str(output_dir.resolve()),
            "evaluation_split": str(args.evaluation_split),
            "datasets": list(datasets),
            "seeds": [int(seed) for seed in seeds],
            "topologies": list(topologies),
            "sampling_methods": list(sampling_methods),
            "default_runs_file": str(true_default_result["runs_csv"]),
            "default_aware_tuned_runs_file": str(tuned_fixed_result["runs_csv"]),
            "true_default": true_default_result,
            "tuned_fixed": tuned_fixed_result,
        }
        manifest_path = output_dir / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if bool(args.generate_diagnostic_report):
            diagnostic_report = _generate_matched_topology_diagnostic_report(
                manifest_path=manifest_path,
                output_dir=output_dir,
                markdown_name=str(args.diagnostic_report_markdown_name),
            )
            manifest["diagnostic_report"] = diagnostic_report
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Saved matched topology diagnostic report to {diagnostic_report['diagnostic_markdown_report']}")
        print(f"Saved dual-profile manifest to {manifest_path}")
    else:
        run_profile = "manual_fixed" if has_any_fixed_overrides else "true_default"
        run_label = str(args.run_label).strip() if args.run_label else run_profile
        runs_csv_name = _validate_csv_name(args.runs_csv_name, arg_name="--runs-csv-name")
        if args.compare_against_csv and not comparison_enabled:
            raise ValueError(
                "--compare-against-csv requires --execution-mode run-and-compare or compare."
            )
        _execute_profile_runs(
            output_dir=output_dir,
            datasets=datasets,
            seeds=seeds,
            topologies=topologies,
            sampling_methods=sampling_methods,
            dataset_config=dict(dataset_config),
            base_objectives=base_objectives,
            expanded_objectives=expanded_objectives,
            timeout=args.timeout,
            evaluation_split=str(args.evaluation_split),
            save_study_json_enabled=bool(args.save_study_json),
            run_profile=run_profile,
            run_label=run_label,
            runs_csv_name=runs_csv_name,
            manual_fixed_params=manual_fixed_params,
            manual_fixed_params_by_topology=manual_fixed_params_by_topology,
            manual_fixed_params_by_sampling_topology=manual_fixed_params_by_sampling_topology,
            compare_against_csv=args.compare_against_csv if comparison_enabled else None,
            compare_output_subdir=str(args.compare_output_subdir),
            compare_markdown_name=str(args.compare_markdown_name),
            compare_split_policy=str(args.compare_split_policy),
            resume_enabled=bool(args.resume),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
