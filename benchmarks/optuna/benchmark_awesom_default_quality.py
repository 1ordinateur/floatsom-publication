#!/usr/bin/env python3
"""Run one matched, untuned aweSOM-default quality benchmark unit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from contextlib import redirect_stdout
import sys
import time
from typing import Any, Dict, Optional, Sequence

import pandas as pd

try:
    from .benchmark_awesom_quality import (
        DATASETS,
        PUBLICATION_SEEDS,
        _add_awesom_source_root,
        deterministic_awesom_randomness,
        load_standardized_dataset,
        quantization_error,
        split_train_holdout,
        trial_random_seed,
    )
except ImportError:
    from benchmark_awesom_quality import (
        DATASETS,
        PUBLICATION_SEEDS,
        _add_awesom_source_root,
        deterministic_awesom_randomness,
        load_standardized_dataset,
        quantization_error,
        split_train_holdout,
        trial_random_seed,
    )


RUNS_FILENAME = "awesom_default_runs.csv"
MANIFEST_FILENAME = "AWESOM_DEFAULT_MANIFEST.json"
DEFAULT_TRAIN_PASSES = 50


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-passes", type=int, default=DEFAULT_TRAIN_PASSES)
    parser.add_argument("--metric-chunk-rows", type=int, default=256)
    parser.add_argument("--sklearn-data-home", default=None)
    parser.add_argument("--awesom-source-root", default=None)
    parser.add_argument("--verbose-awesom", action="store_true")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, row: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame([row]).to_csv(temporary, index=False)
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.seed) not in PUBLICATION_SEEDS:
        raise ValueError(f"Seed must be one of {list(PUBLICATION_SEEDS)}")
    if int(args.train_passes) <= 0:
        raise ValueError("--train-passes must be positive.")
    if int(args.metric_chunk_rows) <= 0:
        raise ValueError("--metric-chunk-rows must be positive.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / RUNS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    started_utc = _utc_now()

    _add_awesom_source_root(args.awesom_source_root)
    awesom = importlib.import_module("aweSOM")
    lattice_class = getattr(awesom, "Lattice")
    data, feature_names = load_standardized_dataset(
        args.dataset,
        sklearn_data_home=args.sklearn_data_home,
    )
    train_data, holdout_data = split_train_holdout(data, seed=int(args.seed))
    train_steps = int(args.train_passes) * int(train_data.shape[0])
    realization_seed = trial_random_seed(int(args.seed), 0)

    # Only the training budget is supplied. All learning and initialization
    # arguments retain aweSOM 1.1.0's public Lattice defaults.
    lattice = lattice_class(train=train_steps)
    train_start = time.perf_counter()
    try:
        with deterministic_awesom_randomness(realization_seed):
            if args.verbose_awesom:
                lattice.train_lattice(train_data, feature_names, labels=None)
            else:
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    with redirect_stdout(sink):
                        lattice.train_lattice(train_data, feature_names, labels=None)
        train_time = time.perf_counter() - train_start
        metric_start = time.perf_counter()
        qe_holdout = quantization_error(
            holdout_data,
            lattice.lattice,
            chunk_rows=int(args.metric_chunk_rows),
        )
        qe_train = quantization_error(
            train_data,
            lattice.lattice,
            chunk_rows=int(args.metric_chunk_rows),
        )
        metric_time = time.perf_counter() - metric_start
    finally:
        del lattice
        gc.collect()

    row: Dict[str, Any] = {
        "implementation": "aweSOM",
        "profile": "default_untuned",
        "architecture": "rectangular_chebyshev",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "quantization_error_holdout": float(qe_holdout),
        "quantization_error_train": float(qe_train),
        "balanced_qe_raw": float((qe_holdout + qe_train) / 2.0),
        "train_time_s": float(train_time),
        "metric_time_s": float(metric_time),
        "trial_seed": int(realization_seed),
        "n_samples": int(data.shape[0]),
        "n_train": int(train_data.shape[0]),
        "n_holdout": int(holdout_data.shape[0]),
        "n_features": int(data.shape[1]),
        "xdim": 10,
        "ydim": 10,
        "n_nodes": 100,
        "train_passes": int(args.train_passes),
        "train_steps": int(train_steps),
        "alpha_0": 0.3,
        "alpha_type": "decay",
        "sampling_type": "sampling",
        "status": "complete",
    }
    _write_csv(runs_path, row)

    manifest: Dict[str, Any] = {
        "benchmark": "awesom_default_vs_untuned_floatsom_hex",
        "implementation": "aweSOM",
        "profile": "default_untuned",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "status": "complete",
        "runs_file": str(runs_path),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "protocol": {
            "preprocessing": "StandardScaler fit on full dataset",
            "split": "70/30 RandomState(seed) permutation",
            "map": "10x10 aweSOM rectangular Chebyshev lattice",
            "training_budget": f"{int(args.train_passes)} full passes",
            "defaults": {
                "alpha_0": 0.3,
                "alpha_type": "decay",
                "sampling_type": "sampling",
            },
            "randomness": "deterministic matched-seed adapter",
        },
        "provenance": {
            "generated_utc": _utc_now(),
            "hostname": platform.node(),
            "python": sys.version,
            "aweSOM_version": _package_version("aweSOM"),
            "aweSOM_file": str(getattr(awesom, "__file__", "")),
            "pandas_version": pd.__version__,
            "pbs_jobid": os.environ.get("PBS_JOBID", ""),
            "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX", ""),
        },
    }
    _write_json(manifest_path, manifest)
    print(f"Wrote {runs_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
