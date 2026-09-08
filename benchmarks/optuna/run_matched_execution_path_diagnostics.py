#!/usr/bin/env python3
"""
Run matched FloatSOM diagnostics across local and Ray execution paths.

This is intentionally close to the matched topology diagnostic runner, but the
paired comparison axis is execution profile rather than topology. Each pair
keeps dataset, seed, topology, sampling method, train/holdout split, and fixed
tuned parameters identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.machinery
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


import cupy as cp
import numpy as np
import pandas as pd

from floatsom.base.floatsom_factories import create_floatsom
from floatsom_benchmarks.evaluation.sklearn_datasets import generate_sklearn_dataset
from floatsom_benchmarks.optuna.config.benchmark_config import Phase3BenchmarkConfig
from floatsom_benchmarks.optuna.core.objective import (
    calculate_metrics,
    create_floatsom_params,
    get_metrics_config,
)
from floatsom.floatsom_params import RayConfig


SUPPORTED_TOPOLOGIES: Tuple[str, ...] = ("hexagonal", "mst", "rng")
SUPPORTED_SAMPLING_METHODS: Tuple[str, ...] = ("full",)
EXECUTION_PROFILES: Tuple[str, str] = ("local_cupy", "ray_streaming")
DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "mean_tied_rank",
    "node_utilization",
    "dead_node_fraction",
    "used_nodes",
    "dead_nodes",
    "total_nodes",
)
BALANCED_COLUMNS: Dict[str, str] = {
    "quantization_error": "balanced_qe_raw",
    "mean_tied_rank": "balanced_mean_tied_rank_raw",
    "node_utilization": "balanced_node_utilization_raw",
    "dead_node_fraction": "balanced_dead_node_fraction_raw",
}
REPORT_METRICS: Tuple[str, ...] = (
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
LOWER_IS_BETTER: set[str] = {
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
HIGHER_IS_BETTER: set[str] = {
    "node_utilization_holdout",
    "node_utilization_train",
    "balanced_node_utilization_raw",
}
RUN_KEY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "seed",
    "architecture",
    "sampling_method",
    "execution_profile",
)
PAIR_KEY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "seed",
    "architecture",
    "sampling_method",
)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_ray_worker_import_path(base_dir: Path) -> Path:
    """Expose the installed library and benchmarks to fresh Ray workers."""
    import floatsom
    import floatsom_benchmarks

    import_root = Path(tempfile.mkdtemp(prefix="ray_pythonpath_", dir=str(base_dir)))
    for package in (floatsom, floatsom_benchmarks):
        os.symlink(Path(package.__file__).resolve().parent,
                   import_root / package.__name__, target_is_directory=True)
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)
    existing = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if import_root_text not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([import_root_text, *existing])
    return import_root


def _resolve_datasets(requested: Optional[Sequence[str]]) -> List[str]:
    available = [str(value) for value in (Phase3BenchmarkConfig().datasets or [])]
    if requested is None:
        return available

    requested_values: List[str] = []
    for item in requested:
        requested_values.extend(part.strip() for part in str(item).split(",") if part.strip())

    unknown = sorted(set(requested_values) - set(available))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Available: {available}")
    return requested_values


def _resolve_topologies(requested: Optional[Sequence[str]]) -> List[str]:
    if requested is None:
        return list(SUPPORTED_TOPOLOGIES)
    topologies: List[str] = []
    for item in requested:
        topologies.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    unknown = sorted(set(topologies) - set(SUPPORTED_TOPOLOGIES))
    if unknown:
        raise ValueError(f"Unsupported topology value(s): {unknown}. Supported: {list(SUPPORTED_TOPOLOGIES)}")
    return topologies


def _resolve_sampling_methods(requested: Optional[Sequence[str]]) -> List[str]:
    if requested is None:
        return list(SUPPORTED_SAMPLING_METHODS)
    methods: List[str] = []
    for item in requested:
        methods.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    unknown = sorted(set(methods) - set(SUPPORTED_SAMPLING_METHODS))
    if unknown:
        raise ValueError(
            f"Unsupported sampling method(s) for this audit: {unknown}. "
            f"Supported: {list(SUPPORTED_SAMPLING_METHODS)}"
        )
    return methods


def _resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds and args.seeds_from_csv:
        raise ValueError("--seeds and --seeds-from-csv cannot be combined.")
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    if args.seeds_from_csv:
        df = pd.read_csv(args.seeds_from_csv)
        for column in ("pair_seed", "seed"):
            if column in df.columns:
                seeds = sorted({int(value) for value in pd.to_numeric(df[column], errors="coerce").dropna()})
                if seeds:
                    return seeds
        raise ValueError(f"No pair_seed or seed column with finite values in {args.seeds_from_csv}")
    return [int(args.base_seed) + idx for idx in range(int(args.num_seeds))]


def _resolve_json_path(path: str | Path) -> Path:
    from floatsom_benchmarks.optuna.config.default_profiles import resolve_defaults_path
    resolved = resolve_defaults_path(str(path))
    if not resolved.is_file():
        raise FileNotFoundError(f"Fixed params JSON not found: {path}")
    return resolved


def _load_fixed_params_by_sampling_topology(
    path: str | Path,
    *,
    sampling_methods: Sequence[str],
    topologies: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    resolved = _resolve_json_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixed params JSON must be an object of sampling->topology->params mappings.")

    selected_sampling = {str(value).strip().lower() for value in sampling_methods}
    selected_topologies = {str(value).strip().lower() for value in topologies}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for sampling in selected_sampling:
        raw_topology_map = payload.get(sampling)
        if not isinstance(raw_topology_map, dict):
            raise ValueError(f"Fixed params JSON must contain sampling entry {sampling!r}.")
        out[sampling] = {}
        for topology in selected_topologies:
            raw_params = raw_topology_map.get(topology)
            if not isinstance(raw_params, dict):
                raise ValueError(
                    f"Fixed params JSON must contain params for sampling={sampling!r}, topology={topology!r}."
                )
            out[sampling][topology] = dict(raw_params)
    return out


def _dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "difficulty": str(args.difficulty),
        "normalize": True,
    }


def _split_train_holdout(data: cp.ndarray, seed: int) -> Tuple[cp.ndarray, cp.ndarray]:
    n_samples = int(data.shape[0])
    n_train = int(0.7 * n_samples)
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    holdout_indices = indices[n_train:]
    return data[train_indices], data[holdout_indices]


def _balanced_value(holdout: float, train: float) -> float:
    if np.isfinite(float(holdout)) and np.isfinite(float(train)):
        return float((float(holdout) + float(train)) / 2.0)
    return float("nan")


def _collect_split_metrics(som, train_data: cp.ndarray, holdout_data: cp.ndarray) -> Dict[str, float]:
    metrics_config = get_metrics_config(
        objectives=["quantization_error"],
        evaluation_split="both",
        diagnostic_metrics=list(DIAGNOSTIC_METRICS),
    )
    requested = list(metrics_config["metrics"])
    holdout_metrics = calculate_metrics(som, holdout_data, metrics_config, metrics_subset=requested)
    train_metrics = calculate_metrics(som, train_data, metrics_config, metrics_subset=requested)

    row: Dict[str, float] = {}
    for metric_name in requested:
        holdout_value = float(holdout_metrics.get(metric_name, float("nan")))
        train_value = float(train_metrics.get(metric_name, float("nan")))
        row[f"{metric_name}_holdout"] = holdout_value
        row[f"{metric_name}_train"] = train_value
        balanced_column = BALANCED_COLUMNS.get(metric_name)
        if balanced_column:
            row[balanced_column] = _balanced_value(holdout_value, train_value)
    return row


def _build_ray_config(
    *,
    run_temp_dir: Path,
    local_storage_path: str,
    chunk_size: int,
    ray_gpu_count: int,
    group_name: str,
) -> RayConfig:
    run_temp_dir.mkdir(parents=True, exist_ok=True)
    return RayConfig(
        chunk_size=int(chunk_size),
        storage_path=str(run_temp_dir / "ray_shared"),
        local_storage_path=str(local_storage_path),
        num_gpus=int(ray_gpu_count),
        collective_group_name=str(group_name),
        enable_collective_barriers=False,
        force_disk_mode=False,
        cache_path=str(run_temp_dir / "ray_cache"),
        wipe_local_storage_on_start=True,
    )


def _train_once(
    *,
    train_data: cp.ndarray,
    forced_params: Dict[str, Any],
    seed: int,
    execution_profile: str,
    ray_temp_root: Path,
    ray_local_storage_path: str,
    ray_gpu_count: int,
    group_name: str,
) -> Tuple[Any, Dict[str, Any], float]:
    params = dict(forced_params)
    params["seed"] = int(seed)
    floatsom_params = create_floatsom_params(train_data, params)
    if execution_profile == "ray_streaming":
        floatsom_params.processing_config.ray_config = _build_ray_config(
            run_temp_dir=ray_temp_root,
            local_storage_path=ray_local_storage_path,
            chunk_size=int(floatsom_params.processing_config.chunk_size),
            ray_gpu_count=int(ray_gpu_count),
            group_name=group_name,
        )
    elif execution_profile != "local_cupy":
        raise ValueError(f"Unsupported execution profile: {execution_profile!r}")

    som = create_floatsom(floatsom_params)
    start = time.perf_counter()
    training_stats = som.train(train_data)
    elapsed = float(time.perf_counter() - start)
    return som, dict(training_stats or {}), elapsed


def _run_key(row: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["architecture"]),
        str(row["sampling_method"]),
        str(row["execution_profile"]),
    )


def _load_completed_keys(csv_path: Path) -> Tuple[pd.DataFrame, set[Tuple[str, int, str, str, str]]]:
    if not csv_path.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df, set()
    keys = set()
    for _, row in df.iterrows():
        if str(row.get("status", "ok")) != "ok":
            continue
        keys.add(
            (
                str(row["dataset"]),
                int(row["seed"]),
                str(row["architecture"]),
                str(row["sampling_method"]),
                str(row["execution_profile"]),
            )
        )
    return df, keys


def _write_rows(
    *,
    runs_csv: Path,
    existing_df: pd.DataFrame,
    new_rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    new_df = pd.DataFrame(list(new_rows))
    if existing_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing_df.copy()
    else:
        combined = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
    if not combined.empty:
        combined = (
            combined.drop_duplicates(subset=list(RUN_KEY_COLUMNS), keep="last")
            .sort_values(list(RUN_KEY_COLUMNS))
            .reset_index(drop=True)
        )
    combined.to_csv(runs_csv, index=False)
    return combined


def _mean_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if finite.size < 2:
        return mean, float("nan"), float("nan")
    se = float(np.std(finite, ddof=1) / math.sqrt(finite.size))
    half_width = 1.96 * se
    return mean, mean - half_width, mean + half_width


def _paired_ttest_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    if std == 0.0:
        return 1.0 if float(np.mean(finite)) == 0.0 else 0.0
    try:
        from scipy import stats

        return float(stats.ttest_1samp(finite, 0.0).pvalue)
    except Exception:
        return float("nan")


def _cohen_dz(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(finite) / std)


def _bh_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([float(value) if value is not None else np.nan for value in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite_indices = np.where(finite_mask)[0]
    if finite_indices.size == 0:
        return q.tolist()
    ordered = finite_indices[np.argsort(p[finite_indices])]
    m = float(ordered.size)
    running = 1.0
    for rank, idx in enumerate(ordered[::-1], start=1):
        original_rank = int(ordered.size - rank + 1)
        adjusted = p[idx] * m / original_rank
        running = min(running, adjusted)
        q[idx] = min(running, 1.0)
    return q.tolist()


def _relative_delta(comparator: pd.Series, reference: pd.Series) -> pd.Series:
    denom = reference.abs().replace(0.0, np.nan)
    return (comparator - reference) / denom


def _paired_frames(runs_df: pd.DataFrame) -> pd.DataFrame:
    ok = runs_df.loc[runs_df["status"].astype(str) == "ok"].copy()
    required = set(PAIR_KEY_COLUMNS) | {"execution_profile"} | set(REPORT_METRICS)
    missing = sorted(required - set(ok.columns))
    if missing:
        raise ValueError(f"Runs CSV is missing required columns for report generation: {missing}")
    local = ok.loc[ok["execution_profile"] == "local_cupy"].copy()
    ray = ok.loc[ok["execution_profile"] == "ray_streaming"].copy()
    merged = local.merge(
        ray,
        on=list(PAIR_KEY_COLUMNS),
        how="inner",
        suffixes=("_local_cupy", "_ray_streaming"),
    )
    return merged


def _dataset_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_frames(runs_df)
    rows: List[Dict[str, Any]] = []
    for (dataset, topology), group in paired.groupby(["dataset", "architecture"], dropna=False):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "topology": topology,
            "n_pairs": int(len(group)),
        }
        for metric in REPORT_METRICS:
            local_values = pd.to_numeric(group[f"{metric}_local_cupy"], errors="coerce")
            ray_values = pd.to_numeric(group[f"{metric}_ray_streaming"], errors="coerce")
            delta = ray_values - local_values
            rel = _relative_delta(ray_values, local_values)
            row[f"{metric}_local_cupy_mean"] = float(local_values.mean())
            row[f"{metric}_ray_streaming_mean"] = float(ray_values.mean())
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_abs_delta_max"] = float(delta.abs().max())
            row[f"{metric}_relative_delta_mean"] = float(rel.mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "topology"]).reset_index(drop=True)


def _paired_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_frames(runs_df)
    rows: List[Dict[str, Any]] = []
    for topology, group in paired.groupby("architecture", dropna=False):
        for metric in REPORT_METRICS:
            local_values = pd.to_numeric(group[f"{metric}_local_cupy"], errors="coerce").to_numpy(dtype=float)
            ray_values = pd.to_numeric(group[f"{metric}_ray_streaming"], errors="coerce").to_numpy(dtype=float)
            delta = ray_values - local_values
            finite = delta[np.isfinite(delta)]
            mean_delta, ci_low, ci_high = _mean_ci95(finite)
            rel = _relative_delta(pd.Series(ray_values), pd.Series(local_values)).to_numpy(dtype=float)
            finite_rel = rel[np.isfinite(rel)]
            rows.append(
                {
                    "comparison": "ray_streaming_vs_local_cupy",
                    "topology": topology,
                    "metric": metric,
                    "direction": (
                        "lower"
                        if metric in LOWER_IS_BETTER
                        else "higher"
                        if metric in HIGHER_IS_BETTER
                        else "count"
                    ),
                    "n_pairs": int(finite.size),
                    "mean_delta_ray_minus_local": mean_delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "cohen_dz": _cohen_dz(finite),
                    "p_value": _paired_ttest_pvalue(finite),
                    "mean_relative_delta": float(np.mean(finite_rel)) if finite_rel.size else float("nan"),
                    "max_abs_delta": float(np.max(np.abs(finite))) if finite.size else float("nan"),
                    "ray_higher_count": int(np.sum(finite > 0.0)),
                    "local_higher_count": int(np.sum(finite < 0.0)),
                    "tie_count": int(np.sum(finite == 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    summary["q_value_bh"] = _bh_adjust(summary["p_value"].tolist())
    return summary.sort_values(["topology", "metric"]).reset_index(drop=True)


def _format_float(value: Any, precision: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{precision}g}"


def _write_markdown_report(
    *,
    output_dir: Path,
    markdown_name: str,
    runs_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Path:
    markdown_path = output_dir / markdown_name
    lines: List[str] = [
        "# Matched execution-path diagnostics",
        "",
        f"Generated UTC: {metadata['generated_utc']}",
        f"Execution profiles: {', '.join(EXECUTION_PROFILES)}",
        f"Successful rows: {len(runs_df)}",
        "",
        "Deltas are reported as `ray_streaming - local_cupy`. No pass/fail threshold is applied.",
        "",
        "## Paired summary",
        "",
    ]
    if paired_summary.empty:
        lines.append("No paired rows were available.")
    else:
        preferred_metrics = [
            "balanced_qe_raw",
            "balanced_mean_tied_rank_raw",
            "balanced_node_utilization_raw",
            "balanced_dead_node_fraction_raw",
        ]
        display = paired_summary.loc[paired_summary["metric"].isin(preferred_metrics)].copy()
        if display.empty:
            display = paired_summary.copy()
        lines.extend(
            [
                "| topology | metric | n | mean delta | 95% CI | max abs delta | p | q |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in display.iterrows():
            ci = f"[{_format_float(row['ci95_low'])}, {_format_float(row['ci95_high'])}]"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["topology"]),
                        str(row["metric"]),
                        str(int(row["n_pairs"])),
                        _format_float(row["mean_delta_ray_minus_local"]),
                        ci,
                        _format_float(row["max_abs_delta"]),
                        _format_float(row["p_value"]),
                        _format_float(row["q_value_bh"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- Runs CSV: `{metadata['runs_csv']}`",
            f"- Dataset summary TSV: `{metadata['dataset_summary_tsv']}`",
            f"- Paired summary TSV: `{metadata['paired_summary_tsv']}`",
        ]
    )
    if "failures_csv" in metadata and metadata["failures_csv"]:
        lines.append(f"- Failures CSV: `{metadata['failures_csv']}`")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def _generate_report(
    *,
    output_dir: Path,
    runs_df: pd.DataFrame,
    markdown_name: str,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    dataset_summary = _dataset_summary(runs_df)
    paired_summary = _paired_summary(runs_df)

    dataset_path = output_dir / "supp_matched_execution_path_diagnostics_by_dataset.tsv"
    paired_path = output_dir / "supp_matched_execution_path_diagnostics_paired_summaries.tsv"
    dataset_summary.to_csv(dataset_path, sep="\t", index=False)
    paired_summary.to_csv(paired_path, sep="\t", index=False)

    report_metadata = {
        **metadata,
        "dataset_summary_tsv": str(dataset_path.resolve()),
        "paired_summary_tsv": str(paired_path.resolve()),
    }
    markdown_path = _write_markdown_report(
        output_dir=output_dir,
        markdown_name=markdown_name,
        runs_df=runs_df,
        dataset_summary=dataset_summary,
        paired_summary=paired_summary,
        metadata=report_metadata,
    )
    return {
        "dataset_summary_tsv": str(dataset_path.resolve()),
        "paired_summary_tsv": str(paired_path.resolve()),
        "markdown_report": str(markdown_path.resolve()),
    }


def _sanitize_group_name(*parts: Any) -> str:
    raw = "_".join(str(part) for part in parts)
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
    return safe[:180]


def _run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if args.scikit_learn_data_home:
        os.environ["SCIKIT_LEARN_DATA"] = str(Path(args.scikit_learn_data_home).expanduser())

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(
        f"Results/execution_path_diagnostics_{_timestamp_utc()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = output_dir / args.runs_csv_name
    failures_csv = output_dir / f"{Path(args.runs_csv_name).stem}_failures.csv"
    datasets = _resolve_datasets(args.datasets)
    topologies = _resolve_topologies(args.topologies)
    sampling_methods = _resolve_sampling_methods(args.sampling_methods)
    seeds = _resolve_seeds(args)
    fixed_params = _load_fixed_params_by_sampling_topology(
        args.fixed_params_by_sampling_topology_json,
        sampling_methods=sampling_methods,
        topologies=topologies,
    )
    dataset_config = _dataset_config(args)
    existing_df, completed_keys = _load_completed_keys(runs_csv) if args.resume else (pd.DataFrame(), set())
    run_temp_base = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else output_dir / "tmp"
    run_temp_base.mkdir(parents=True, exist_ok=True)
    ray_import_root = _ensure_ray_worker_import_path(run_temp_base)
    ray_local_storage_path = str(Path(args.ray_local_storage_path).expanduser()) if args.ray_local_storage_path else str(run_temp_base / "ray_local")

    print("Running matched execution-path diagnostics")
    print(f"Output: {output_dir}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Topologies ({len(topologies)}): {topologies}")
    print(f"Sampling methods ({len(sampling_methods)}): {sampling_methods}")
    print(f"Execution profiles: {list(EXECUTION_PROFILES)}")
    print(f"Fixed params JSON: {_resolve_json_path(args.fixed_params_by_sampling_topology_json)}")
    print(f"Ray worker import root: {ray_import_root}")
    print(f"Ray local storage path: {ray_local_storage_path}")
    print(f"Resume enabled: {bool(args.resume)}")

    success_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    total_planned = len(datasets) * len(seeds) * len(topologies) * len(sampling_methods) * len(EXECUTION_PROFILES)
    executed = 0
    skipped = 0

    for dataset in datasets:
        for seed in seeds:
            data, dataset_metadata = generate_sklearn_dataset(
                dataset_name=dataset,
                seed=int(seed),
                **dataset_config,
            )
            train_data, holdout_data = _split_train_holdout(data, int(seed))
            for topology in topologies:
                for sampling_method in sampling_methods:
                    fixed = fixed_params[sampling_method][topology]
                    forced_params = {
                        "sampling_method": sampling_method,
                        "processing_method": "batch",
                        "batch_mode": "full_batch",
                        "topology_type": topology,
                        **fixed,
                    }
                    for execution_profile in EXECUTION_PROFILES:
                        key = (dataset, int(seed), topology, sampling_method, execution_profile)
                        if key in completed_keys:
                            skipped += 1
                            continue
                        executed += 1
                        print(
                            f"[{executed}/{total_planned}] dataset={dataset} seed={seed} "
                            f"topology={topology} sampling={sampling_method} profile={execution_profile}"
                        )
                        run_temp_dir = Path(
                            tempfile.mkdtemp(
                                prefix=_sanitize_group_name(dataset, seed, topology, execution_profile) + "_",
                                dir=str(run_temp_base),
                            )
                        )
                        try:
                            group_name = _sanitize_group_name(
                                "exec_path_diag",
                                dataset,
                                seed,
                                topology,
                                sampling_method,
                                execution_profile,
                                int(time.time_ns()),
                            )
                            som, training_stats, train_time_s = _train_once(
                                train_data=train_data,
                                forced_params=forced_params,
                                seed=int(seed),
                                execution_profile=execution_profile,
                                ray_temp_root=run_temp_dir,
                                ray_local_storage_path=ray_local_storage_path,
                                ray_gpu_count=int(args.ray_gpu_count),
                                group_name=group_name,
                            )
                            metrics = _collect_split_metrics(som, train_data, holdout_data)
                            row: Dict[str, Any] = {
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
                                "execution_profile": execution_profile,
                                "run_profile": "tuned_fixed",
                                "status": "ok",
                                "n_samples": int(data.shape[0]),
                                "n_train_samples": int(train_data.shape[0]),
                                "n_test_samples": int(holdout_data.shape[0]),
                                "train_time_s": float(train_time_s),
                                "iterations_completed": int(training_stats.get("iterations_completed", -1)),
                                "data_staging_mode": str(training_stats.get("data_staging_mode", "")),
                                "data_staging_strategy": str(training_stats.get("data_staging_strategy", "")),
                                "dataset_metadata_name": str(dataset_metadata.get("dataset_name", dataset)),
                                **metrics,
                            }
                            for param_name, value in fixed.items():
                                row[f"param_{param_name}"] = value
                            success_rows.append(row)
                            existing_df = _write_rows(
                                runs_csv=runs_csv,
                                existing_df=existing_df,
                                new_rows=success_rows,
                            )
                            success_rows = []
                        except Exception as exc:
                            failure_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": int(seed),
                                    "sampling_method": sampling_method,
                                    "architecture": topology,
                                    "execution_profile": execution_profile,
                                    "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            print(f"  FAILED: {type(exc).__name__}: {exc}")
                        finally:
                            shutil.rmtree(run_temp_dir, ignore_errors=True)
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    runs_df = _write_rows(runs_csv=runs_csv, existing_df=existing_df, new_rows=success_rows)
    failure_df = pd.DataFrame(failure_rows)
    if not failure_df.empty:
        failure_df.to_csv(failures_csv, index=False)
    elif failures_csv.exists():
        failures_csv.unlink()

    metadata: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": str(output_dir.resolve()),
        "runs_csv": str(runs_csv.resolve()),
        "failures_csv": str(failures_csv.resolve()) if failures_csv.exists() else None,
        "fixed_params_by_sampling_topology_json": str(
            _resolve_json_path(args.fixed_params_by_sampling_topology_json)
        ),
        "datasets": datasets,
        "seeds": [int(seed) for seed in seeds],
        "topologies": topologies,
        "sampling_methods": sampling_methods,
        "execution_profiles": list(EXECUTION_PROFILES),
        "total_planned_runs": int(total_planned),
        "executed_runs": int(executed),
        "resume_skipped_runs": int(skipped),
        "successful_rows_total": int(len(runs_df)),
        "failed_runs": int(len(failure_rows)),
        "ray_gpu_count": int(args.ray_gpu_count),
        "ray_local_storage_path": ray_local_storage_path,
        "ray_worker_import_root": str(ray_import_root),
    }
    report_paths = _generate_report(
        output_dir=output_dir,
        runs_df=runs_df,
        markdown_name=args.report_markdown_name,
        metadata=metadata,
    )
    metadata.update(report_paths)

    manifest_path = output_dir / args.manifest_name
    metadata["manifest_json"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved runs to {runs_csv}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved report to {report_paths['markdown_report']}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched diagnostics comparing local and Ray FloatSOM execution paths."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds-from-csv", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--topologies", nargs="+", type=str, default=None)
    parser.add_argument("--sampling-methods", nargs="+", type=str, default=None)
    parser.add_argument("--difficulty", type=str, default="hard")
    parser.add_argument("--scikit-learn-data-home", type=str, default=None)
    parser.add_argument(
        "--fixed-params-by-sampling-topology-json",
        type=str,
        default="publication",
    )
    parser.add_argument("--ray-gpu-count", type=int, default=1)
    parser.add_argument("--ray-local-storage-path", type=str, default=None)
    parser.add_argument("--temp-dir", type=str, default=None)
    parser.add_argument("--runs-csv-name", type=str, default="matched_tuned_execution_path_diagnostics_runs.csv")
    parser.add_argument("--manifest-name", type=str, default="MATCHED_EXECUTION_PATH_DIAGNOSTICS_MANIFEST.json")
    parser.add_argument("--report-markdown-name", type=str, default="MATCHED_EXECUTION_PATH_DIAGNOSTICS_SUMMARY.md")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    _run_all(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
