#!/usr/bin/env python3
"""Rebuild GPU-scaling result JSON files from CSV artifacts.

This utility restores files like:
  sample_scaling_results_mst_batch.json
from matching CSV files when JSON was accidentally deleted.

Notes:
- This utility also attempts to recover per-repeat `times` from mode log files
  in sibling `logs/` directories and will recompute mean/std/count from those
  times when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


Numeric = Union[int, float]
GPUColMap = Dict[int, Dict[str, int]]
LogRecord = Tuple[str, str, str, int, float]

GPU_COL_RE = re.compile(r"^(?P<gpu>\d+)_GPUs_(?P<metric>Mean|Std|Count)$")
CONFIG_NAME_RE = re.compile(
    r"^(?P<topology>.+?)_dim(?P<dimension>\d+)"
    r"(?:_samples(?P<samples>\d+))?_grid(?P<grid>\d+)"
    r"_gpu(?P<gpu>\d+)_(?P<method>[^_]+)_seed(?P<seed>-?\d+)$"
)
GRID_SIZE_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)")

KNOWN_TOPOLOGIES = {"hex", "hexagonal", "mst", "rng", "grid"}
KNOWN_METHODS = {"batch", "colors", "minibatch", "minisom", "full", "random"}
MODE_TO_AXIS_FIELD = {
    "dimension_scaling": "dimension",
    "sample_scaling": "samples",
    "grid_size_scaling": "grid",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate *_results_*.json files from matching CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files and/or directories containing benchmark CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into input directories (default: true).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned writes without creating files.",
    )
    parser.add_argument(
        "--use_log_times",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover per-repeat times from sibling logs/ files when available (default: true).",
    )
    return parser.parse_args()


def _normalize_topology_name(topology: str) -> str:
    normalized = str(topology).strip().lower()
    if normalized == "hex":
        return "hexagonal"
    return normalized


def _normalize_method_name(method: str) -> str:
    return str(method).strip().lower()


def _parse_axis_value(text: str) -> Optional[Numeric]:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        as_float = float(raw)
    except ValueError:
        return None
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _axis_key(value: Numeric) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def _parse_float(text: str) -> Optional[float]:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_count(text: str) -> Optional[int]:
    value = _parse_float(text)
    if value is None:
        return None
    value = int(value)
    return max(value, 0)


def _parse_gpu_columns(header: List[str]) -> GPUColMap:
    mapping: GPUColMap = {}
    for idx, col_name in enumerate(header[1:], start=1):
        match = GPU_COL_RE.match(str(col_name).strip())
        if not match:
            continue
        gpu = int(match.group("gpu"))
        metric = match.group("metric").lower()
        mapping.setdefault(gpu, {})[metric] = idx
    return {
        gpu: columns
        for gpu, columns in mapping.items()
        if "mean" in columns and "std" in columns and "count" in columns
    }


def _parse_mode_topology_method_from_csv(
    csv_path: Path,
) -> Optional[Tuple[str, str, str]]:
    stem = csv_path.stem
    token = "_results_"
    if token not in stem:
        return None
    mode_name, suffix = stem.split(token, 1)
    if not mode_name or not suffix:
        return None
    parts = [chunk for chunk in suffix.split("_") if chunk]
    if len(parts) < 2:
        return None

    for split_idx in range(1, len(parts)):
        left = "_".join(parts[:split_idx])
        right = "_".join(parts[split_idx:])
        left_topology = _normalize_topology_name(left)
        right_method = _normalize_method_name(right)
        if left_topology in KNOWN_TOPOLOGIES and right_method in KNOWN_METHODS:
            return mode_name, left_topology, right_method
        left_method = _normalize_method_name(left)
        right_topology = _normalize_topology_name(right)
        if left_method in KNOWN_METHODS and right_topology in KNOWN_TOPOLOGIES:
            return mode_name, right_topology, left_method

    topology, method = suffix.rsplit("_", 1)
    topology = _normalize_topology_name(topology)
    method = _normalize_method_name(method)
    if topology and method:
        return mode_name, topology, method
    return None


def _parse_training_time_line(line: str) -> Optional[float]:
    if not line.startswith("Training time:"):
        return None
    raw = line.split(":", 1)[1].strip().rstrip("s")
    return _parse_float(raw)


def _parse_int_after_prefix(line: str, prefix: str, allow_commas: bool = True) -> Optional[int]:
    if not line.startswith(prefix):
        return None
    raw = line.split(":", 1)[1].strip()
    if allow_commas:
        raw = raw.replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_grid_size_line(line: str) -> Optional[int]:
    if not line.startswith("Grid Size:"):
        return None
    match = GRID_SIZE_RE.search(line)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_log_record(log_path: Path, mode_name: str) -> Optional[LogRecord]:
    axis_field = MODE_TO_AXIS_FIELD.get(mode_name)
    if axis_field is None:
        return None

    topology: Optional[str] = None
    method: Optional[str] = None
    axis_value: Optional[int] = None
    gpu_count: Optional[int] = None
    train_time: Optional[float] = None

    stem_match = CONFIG_NAME_RE.match(log_path.stem)
    if stem_match:
        topology = _normalize_topology_name(stem_match.group("topology"))
        method = _normalize_method_name(stem_match.group("method"))
        gpu_count = int(stem_match.group("gpu"))
        if axis_field == "dimension":
            axis_value = int(stem_match.group("dimension"))
        elif axis_field == "samples":
            sample_group = stem_match.group("samples")
            if sample_group is not None:
                axis_value = int(sample_group)
        elif axis_field == "grid":
            axis_value = int(stem_match.group("grid"))

    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if train_time is None:
                    parsed_time = _parse_training_time_line(line)
                    if parsed_time is not None:
                        train_time = float(parsed_time)
                if topology is None and line.startswith("Topology:"):
                    topology = _normalize_topology_name(line.split(":", 1)[1].strip())
                if gpu_count is None:
                    parsed_gpu = _parse_int_after_prefix(line, "GPUs:", allow_commas=False)
                    if parsed_gpu is not None:
                        gpu_count = parsed_gpu
                if axis_value is None:
                    if axis_field == "dimension":
                        parsed_dim = _parse_int_after_prefix(line, "Dimension:")
                        if parsed_dim is not None:
                            axis_value = parsed_dim
                    elif axis_field == "samples":
                        parsed_samples = _parse_int_after_prefix(line, "Samples:")
                        if parsed_samples is not None:
                            axis_value = parsed_samples
                    elif axis_field == "grid":
                        parsed_grid = _parse_grid_size_line(line)
                        if parsed_grid is not None:
                            axis_value = parsed_grid
    except Exception:
        return None

    if topology is None or method is None or axis_value is None or gpu_count is None or train_time is None:
        return None
    return topology, method, _axis_key(axis_value), int(gpu_count), float(train_time)


def _load_mode_log_records(
    logs_dir: Path,
    mode_name: str,
    cache: Dict[Tuple[Path, str], List[LogRecord]],
) -> List[LogRecord]:
    cache_key = (logs_dir.resolve(), mode_name)
    if cache_key in cache:
        return cache[cache_key]
    records: List[LogRecord] = []
    if not logs_dir.is_dir():
        cache[cache_key] = records
        return records
    for log_path in sorted(logs_dir.glob("*.log")):
        record = _parse_log_record(log_path, mode_name=mode_name)
        if record is not None:
            records.append(record)
    cache[cache_key] = records
    return records


def _population_std(values: List[float], mean_val: float) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum((value - mean_val) ** 2 for value in values) / float(len(values)))


def _inject_times_from_logs(
    payload: Dict[str, Dict[str, Dict[str, object]]],
    csv_path: Path,
    mode_name: str,
    topology_name: str,
    method_name: str,
    log_cache: Dict[Tuple[Path, str], List[LogRecord]],
) -> List[str]:
    warnings: List[str] = []
    logs_dir = csv_path.parent / "logs"
    records = _load_mode_log_records(logs_dir=logs_dir, mode_name=mode_name, cache=log_cache)
    if not records:
        warnings.append("no matching logs parsed")
        return warnings

    grouped: Dict[Tuple[str, int], List[float]] = {}
    for rec_topology, rec_method, axis_key, gpu_count, train_time in records:
        if rec_topology != topology_name or rec_method != method_name:
            continue
        grouped.setdefault((axis_key, gpu_count), []).append(float(train_time))

    recovered_points = 0
    for axis_key, gpu_payload in payload.items():
        for gpu_key, stats in gpu_payload.items():
            if not isinstance(stats, dict):
                continue
            try:
                parsed_gpu = int(gpu_key)
            except (TypeError, ValueError):
                continue
            key = (str(axis_key), parsed_gpu)
            times = grouped.get(key, [])
            if not times:
                continue
            mean_val = float(sum(times) / float(len(times)))
            std_val = float(_population_std(times, mean_val))
            stats["times"] = [float(v) for v in times]
            stats["mean"] = mean_val
            stats["std"] = std_val
            stats["count"] = int(len(times))
            recovered_points += 1

    if recovered_points == 0:
        warnings.append("logs found but no axis/GPU matches for this CSV")
    else:
        warnings.append(f"recovered log times for {recovered_points} axis/GPU entries")
    return warnings


def _csv_to_results_payload(csv_path: Path) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], List[str]]:
    warnings: List[str] = []
    payload: Dict[str, Dict[str, Dict[str, object]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        warnings.append("empty file")
        return payload, warnings

    header = rows[0]
    if len(header) < 2:
        warnings.append("header does not contain GPU columns")
        return payload, warnings

    gpu_cols = _parse_gpu_columns(header)
    if not gpu_cols:
        warnings.append("no '<gpu>_GPUs_(Mean|Std|Count)' columns found")
        return payload, warnings

    for row_idx, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        axis = _parse_axis_value(row[0] if len(row) > 0 else "")
        if axis is None:
            warnings.append(f"row {row_idx}: non-numeric axis '{row[0] if row else ''}'")
            continue
        axis_entry: Dict[str, Dict[str, object]] = {}
        for gpu, cols in sorted(gpu_cols.items()):
            mean_idx = cols["mean"]
            std_idx = cols["std"]
            count_idx = cols["count"]
            mean_val = _parse_float(row[mean_idx] if mean_idx < len(row) else "")
            std_val = _parse_float(row[std_idx] if std_idx < len(row) else "")
            count_val = _parse_count(row[count_idx] if count_idx < len(row) else "")
            if mean_val is None:
                continue
            if std_val is None:
                std_val = 0.0
            if count_val is None:
                count_val = 0
            axis_entry[str(gpu)] = {
                "times": [],
                "mean": float(mean_val),
                "std": float(std_val),
                "count": int(count_val),
            }
        if axis_entry:
            payload[_axis_key(axis)] = axis_entry

    if not payload:
        warnings.append("parsed rows but found no usable numeric entries")
    return payload, warnings


def _iter_candidate_csvs(inputs: Iterable[str], recursive: bool) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        elif recursive:
            candidates = path.rglob("*_results*.csv")
        else:
            candidates = path.glob("*_results*.csv")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found)


def main() -> None:
    args = _parse_args()
    csv_files = _iter_candidate_csvs(args.inputs, recursive=bool(args.recursive))
    if not csv_files:
        raise FileNotFoundError("No candidate CSV files found.")

    written = 0
    skipped = 0
    failed = 0
    log_cache: Dict[Tuple[Path, str], List[LogRecord]] = {}
    for csv_path in csv_files:
        json_path = csv_path.with_suffix(".json")
        if json_path.exists() and not args.overwrite:
            skipped += 1
            print(f"skip (exists): {json_path}")
            continue

        payload, parse_warnings = _csv_to_results_payload(csv_path)
        if not payload:
            failed += 1
            warning_text = "; ".join(parse_warnings) if parse_warnings else "no data parsed"
            print(f"fail: {csv_path} ({warning_text})")
            continue

        if args.use_log_times:
            parsed_csv_meta = _parse_mode_topology_method_from_csv(csv_path)
            if parsed_csv_meta is None:
                parse_warnings.append("could not infer mode/topology/method from CSV filename; skipping log-time recovery")
            else:
                mode_name, topology_name, method_name = parsed_csv_meta
                parse_warnings.extend(
                    _inject_times_from_logs(
                        payload=payload,
                        csv_path=csv_path,
                        mode_name=mode_name,
                        topology_name=topology_name,
                        method_name=method_name,
                        log_cache=log_cache,
                    )
                )

        if args.dry_run:
            print(f"dry-run write: {json_path}")
            if parse_warnings:
                print(f"  warnings: {'; '.join(parse_warnings)}")
            written += 1
            continue

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote: {json_path}")
        if parse_warnings:
            print(f"  warnings: {'; '.join(parse_warnings)}")
        written += 1

    print(
        f"Done. csv_files={len(csv_files)}, wrote={written}, skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
