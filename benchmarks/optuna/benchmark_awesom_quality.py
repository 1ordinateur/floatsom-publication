#!/usr/bin/env python3
"""Optuna quality benchmark for aweSOM on the five feasible labelled datasets.

The protocol mirrors the FloatSOM quality campaign where the implementations
have comparable quantities:

* StandardScaler is fitted to the full dataset before splitting;
* the split is the publication's deterministic 70/30 seed permutation;
* map size is fixed at 10 x 10 (100 prototypes);
* training uses 50 complete online passes over the training partition; and
* Optuna minimizes holdout and train quantization error jointly.

Only aweSOM parameters exposed by its public Lattice API are optimized:
``alpha_0``, ``alpha_type``, and ``sampling_type``.  The native aweSOM map is a
regular rectangular lattice using Chebyshev neighborhood distance; it is not
described as a hexagonal map.

"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_olivetti_faces,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.preprocessing import StandardScaler


DATASETS = (
    "iris",
    "wine",
    "digits",
    "breast_cancer",
    "olivetti_faces",
)

# Seeds used by the existing 10-seed external-calibration campaign.
PUBLICATION_SEEDS = (
    11780,
    24458,
    27760,
    33080,
    39252,
    48049,
    69281,
    88014,
    89580,
    90744,
)

DEFAULT_TRIALS = 200
DEFAULT_TRAIN_PASSES = 50
DEFAULT_XDIM = 10
DEFAULT_YDIM = 10
TRIALS_FILENAME = "awesom_optuna_trials.csv"
MANIFEST_FILENAME = "AWESOM_OPTUNA_MANIFEST.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--train-passes", type=int, default=DEFAULT_TRAIN_PASSES)
    parser.add_argument("--xdim", type=int, default=DEFAULT_XDIM)
    parser.add_argument("--ydim", type=int, default=DEFAULT_YDIM)
    parser.add_argument("--metric-chunk-rows", type=int, default=256)
    parser.add_argument("--sklearn-data-home", default=None)
    parser.add_argument("--awesom-source-root", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose-awesom", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _module_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _add_awesom_source_root(source_root: Optional[str]) -> None:
    if not source_root:
        return
    root = Path(source_root).expanduser().resolve()
    if root.name == "aweSOM" and (root / "__init__.py").exists():
        import_root = root.parent
    else:
        src = root / "src"
        import_root = src if src.exists() else root
    # Append so the active environment's compiled NumPy/Numba dependencies
    # take precedence over packages beside an externally supplied aweSOM tree.
    sys.path.append(str(import_root))


def load_standardized_dataset(
    dataset_name: str,
    *,
    sklearn_data_home: Optional[str] = None,
) -> tuple[np.ndarray, List[str]]:
    """Load one of the five datasets exactly once and standardize all features."""
    if dataset_name == "iris":
        bundle = load_iris(return_X_y=False, as_frame=False)
    elif dataset_name == "wine":
        bundle = load_wine(return_X_y=False, as_frame=False)
    elif dataset_name == "digits":
        bundle = load_digits(return_X_y=False, as_frame=False)
    elif dataset_name == "breast_cancer":
        bundle = load_breast_cancer(return_X_y=False, as_frame=False)
    elif dataset_name == "olivetti_faces":
        bundle = fetch_olivetti_faces(
            data_home=sklearn_data_home,
            shuffle=False,
            random_state=0,
            download_if_missing=False,
            return_X_y=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    raw = np.asarray(bundle.data)
    data = StandardScaler().fit_transform(raw).astype(np.float32, copy=False)
    names = getattr(bundle, "feature_names", None)
    if names is None or len(names) != data.shape[1]:
        feature_names = [f"feature_{index}" for index in range(data.shape[1])]
    else:
        feature_names = [str(value) for value in names]
    return data, feature_names


def split_train_holdout(
    data: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same RandomState permutation used in the publication."""
    n_train = int(float(train_fraction) * int(data.shape[0]))
    indices = np.random.RandomState(int(seed)).permutation(int(data.shape[0]))
    return data[indices[:n_train]], data[indices[n_train:]]


def quantization_error(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_rows: int = 256,
) -> float:
    """Mean Euclidean distance to the nearest prototype using bounded chunks."""
    observations = np.asarray(data, dtype=np.float32)
    prototypes = np.asarray(weights, dtype=np.float32).reshape((-1, observations.shape[1]))
    prototype_norms = np.einsum("ij,ij->i", prototypes, prototypes)
    distance_sum = 0.0
    for start in range(0, observations.shape[0], int(chunk_rows)):
        batch = observations[start : start + int(chunk_rows)]
        batch_norms = np.einsum("ij,ij->i", batch, batch)
        squared = (
            batch_norms[:, None]
            + prototype_norms[None, :]
            - 2.0 * (batch @ prototypes.T)
        )
        np.maximum(squared, 0.0, out=squared)
        distance_sum += float(
            np.sqrt(np.min(squared, axis=1)).sum(dtype=np.float64)
        )
    return distance_sum / max(int(observations.shape[0]), 1)


def trial_random_seed(dataset_seed: int, trial_number: int) -> int:
    """Derive a reproducible uint32 seed for each Optuna realization."""
    sequence = np.random.SeedSequence([int(dataset_seed), int(trial_number)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@contextmanager
def deterministic_awesom_randomness(seed: int) -> Iterator[None]:
    """Seed aweSOM initialization and its otherwise unseeded sample-order RNG.

    aweSOM 1.1.0 calls ``np.random.default_rng()`` without an argument inside
    ``Lattice.fast_som`` and exposes no training seed parameter.  During one
    trial, this adapter supplies the recorded trial seed only to that no-arg
    call.  No update equation or schedule is changed.
    """
    original_default_rng = np.random.default_rng
    np.random.seed(int(seed))

    def _seeded_default_rng(value: Any = None):
        return original_default_rng(int(seed) if value is None else value)

    np.random.default_rng = _seeded_default_rng  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng  # type: ignore[assignment]


def _provenance(awesom_module: Optional[Any] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": _module_version("scikit-learn"),
        "scipy_version": _module_version("scipy"),
        "numba_version": _module_version("numba"),
        "optuna_version": _module_version("optuna"),
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX", ""),
        "pbs_ncpus": os.environ.get("PBS_NCPUS", ""),
    }
    if awesom_module is not None:
        payload.update(
            {
                "awesom_version": _module_version("aweSOM"),
                "awesom_file": str(getattr(awesom_module, "__file__", "")),
            }
        )
    return payload


def _study_rows(study: Any, *, dataset: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        state = str(getattr(trial.state, "name", trial.state)).lower()
        values = list(trial.values) if trial.values is not None else []
        attributes = dict(trial.user_attrs)
        row: Dict[str, Any] = {
            "implementation": "aweSOM",
            "architecture": "rectangular_chebyshev",
            "dataset": dataset,
            "seed": int(seed),
            "trial_number": int(trial.number),
            "state": state,
            "quantization_error_holdout": (
                float(values[0]) if len(values) > 0 else float("nan")
            ),
            "quantization_error_train": (
                float(values[1]) if len(values) > 1 else float("nan")
            ),
            "balanced_qe_raw": float(attributes.get("balanced_qe_raw", float("nan"))),
            "train_time_s": float(attributes.get("train_time_s", float("nan"))),
            "metric_time_s": float(attributes.get("metric_time_s", float("nan"))),
            "trial_seed": attributes.get("trial_seed", ""),
            "n_samples": attributes.get("n_samples", ""),
            "n_train": attributes.get("n_train", ""),
            "n_holdout": attributes.get("n_holdout", ""),
            "n_features": attributes.get("n_features", ""),
            "xdim": attributes.get("xdim", ""),
            "ydim": attributes.get("ydim", ""),
            "n_nodes": attributes.get("n_nodes", ""),
            "train_passes": attributes.get("train_passes", ""),
            "train_steps": attributes.get("train_steps", ""),
            "param_alpha_0": trial.params.get("alpha_0", ""),
            "param_alpha_type": trial.params.get("alpha_type", ""),
            "param_sampling_type": trial.params.get("sampling_type", ""),
            "error_type": attributes.get("error_type", ""),
            "error_message": attributes.get("error_message", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _write_trials_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed not in PUBLICATION_SEEDS:
        raise ValueError(
            f"Seed {args.seed} is not in the prespecified publication set "
            f"{list(PUBLICATION_SEEDS)}."
        )
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.train_passes <= 0:
        raise ValueError("--train-passes must be positive.")
    if args.xdim < 4 or args.ydim < 4:
        raise ValueError("aweSOM requires xdim and ydim to be at least 4.")
    if args.metric_chunk_rows <= 0:
        raise ValueError("--metric-chunk-rows must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / TRIALS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    study_name = args.study_name or f"awesom_{args.dataset}_seed_{int(args.seed)}"
    study_path = output_dir / "awesom_optuna.sqlite3"

    protocol: Dict[str, Any] = {
        "benchmark": "awesom_optuna_quality",
        "protocol_version": 1,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "publication_seed_set": list(PUBLICATION_SEEDS),
        "n_trials_target": int(args.n_trials),
        "train_passes": int(args.train_passes),
        "xdim": int(args.xdim),
        "ydim": int(args.ydim),
        "n_nodes": int(args.xdim) * int(args.ydim),
        "objectives": [
            "quantization_error_holdout",
            "quantization_error_train",
        ],
        "directions": ["minimize", "minimize"],
        "search_space": {
            "alpha_0": {"type": "float", "low": 0.01, "high": 1.0},
            "alpha_type": {"type": "categorical", "choices": ["decay", "static"]},
            "sampling_type": {
                "type": "categorical",
                "choices": ["sampling", "uniform"],
            },
        },
        "fixed_settings": {
            "preprocessing": "StandardScaler fit on full dataset",
            "train_holdout_split": "70/30 RandomState(seed) permutation",
            "map": "aweSOM native rectangular Chebyshev lattice",
            "train_steps": "train_passes * n_train",
            "randomness": (
                "adapter supplies a recorded trial seed to aweSOM's otherwise "
                "unseeded no-argument default_rng call"
            ),
        },
        "study_name": study_name,
        "study_path": str(study_path),
        "trials_file": str(trials_path),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "started_utc": _utc_now(),
        "provenance": _provenance(),
    }
    _write_json(manifest_path, protocol)
    if args.dry_run:
        print(json.dumps(protocol, indent=2, default=_json_default))
        return 0

    if study_path.exists() and not args.resume:
        raise FileExistsError(
            f"Study database already exists: {study_path}. Use --resume or a new output directory."
        )

    if args.sklearn_data_home:
        os.environ["SCIKIT_LEARN_DATA"] = str(
            Path(args.sklearn_data_home).expanduser().resolve()
        )
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    _add_awesom_source_root(args.awesom_source_root)
    awesom = importlib.import_module("aweSOM")
    lattice_class = getattr(awesom, "Lattice")
    optuna = importlib.import_module("optuna")

    data, feature_names = load_standardized_dataset(
        args.dataset,
        sklearn_data_home=args.sklearn_data_home,
    )
    train_data, holdout_data = split_train_holdout(data, seed=int(args.seed))
    train_steps = int(args.train_passes) * int(train_data.shape[0])

    storage = f"sqlite:///{study_path}"
    sampler = optuna.samplers.NSGAIISampler(seed=int(args.seed))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        directions=["minimize", "minimize"],
        load_if_exists=bool(args.resume),
    )
    study.set_user_attr("protocol", protocol)
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "alpha_0": 0.5,
                "alpha_type": "decay",
                "sampling_type": "sampling",
            }
        )

    def objective(trial: Any) -> tuple[float, float]:
        alpha_0 = trial.suggest_float("alpha_0", 0.01, 1.0)
        alpha_type = trial.suggest_categorical("alpha_type", ["decay", "static"])
        sampling_type = trial.suggest_categorical(
            "sampling_type",
            ["sampling", "uniform"],
        )
        realization_seed = trial_random_seed(int(args.seed), int(trial.number))
        trial.set_user_attr("trial_seed", realization_seed)
        trial.set_user_attr("n_samples", int(data.shape[0]))
        trial.set_user_attr("n_train", int(train_data.shape[0]))
        trial.set_user_attr("n_holdout", int(holdout_data.shape[0]))
        trial.set_user_attr("n_features", int(data.shape[1]))
        trial.set_user_attr("xdim", int(args.xdim))
        trial.set_user_attr("ydim", int(args.ydim))
        trial.set_user_attr("n_nodes", int(args.xdim) * int(args.ydim))
        trial.set_user_attr("train_passes", int(args.train_passes))
        trial.set_user_attr("train_steps", train_steps)

        lattice = lattice_class(
            xdim=int(args.xdim),
            ydim=int(args.ydim),
            alpha_0=float(alpha_0),
            train=train_steps,
            alpha_type=str(alpha_type),
            sampling_type=str(sampling_type),
        )
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
            trial.set_user_attr("train_time_s", float(train_time))
            trial.set_user_attr("metric_time_s", float(metric_time))
            trial.set_user_attr("quantization_error_holdout", float(qe_holdout))
            trial.set_user_attr("quantization_error_train", float(qe_train))
            trial.set_user_attr(
                "balanced_qe_raw",
                float((qe_holdout + qe_train) / 2.0),
            )
            return float(qe_holdout), float(qe_train)
        except BaseException as exc:
            trial.set_user_attr("error_type", type(exc).__name__)
            trial.set_user_attr("error_message", str(exc))
            trial.set_user_attr("traceback", traceback.format_exc())
            raise
        finally:
            del lattice
            gc.collect()

    completed_before = sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = max(0, int(args.n_trials) - completed_before)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
            gc_after_trial=True,
            show_progress_bar=False,
        )

    frame = _study_rows(study, dataset=args.dataset, seed=int(args.seed))
    _write_trials_csv(trials_path, frame)
    completed = int((frame["state"] == "complete").sum()) if not frame.empty else 0
    failed = int((frame["state"] == "fail").sum()) if not frame.empty else 0
    protocol.update(
        {
            "finished_utc": _utc_now(),
            "completed_trials": completed,
            "failed_trials": failed,
            "total_study_trials": int(len(frame)),
            "provenance": _provenance(awesom),
        }
    )
    _write_json(manifest_path, protocol)
    print(f"Wrote {trials_path}")
    print(f"Wrote {manifest_path}")
    print(f"Completed trials: {completed}/{args.n_trials}; failed: {failed}")
    return 0 if completed >= int(args.n_trials) else 2


if __name__ == "__main__":
    raise SystemExit(main())
