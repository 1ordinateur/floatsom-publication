"""
Smoke-test sklearn dataset availability/cache for FloatSOM benchmark datasets.

Usage:
    python floatsom/benchmarks/evaluation/smoke_sklearn_dataset_cache.py \
        --data-home /g/data/eu59/SIFEAN/sfa/sklearn_data
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List, Tuple

from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    fetch_kddcup99,
    fetch_lfw_people,
    fetch_olivetti_faces,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)


def _checks(data_home: str) -> List[Tuple[str, Callable[[], tuple]]]:
    return [
        ("swiss_roll", lambda: make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)[0].shape),
        ("s_curve", lambda: make_s_curve(n_samples=1000, noise=0.1, random_state=42)[0].shape),
        ("moons", lambda: make_moons(n_samples=1000, noise=0.1, random_state=42)[0].shape),
        ("circles", lambda: make_circles(n_samples=1000, noise=0.1, random_state=42)[0].shape),
        ("blobs", lambda: make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)[0].shape),
        ("breast_cancer", lambda: load_breast_cancer().data.shape),
        ("wine", lambda: load_wine().data.shape),
        ("iris", lambda: load_iris().data.shape),
        ("digits", lambda: load_digits().data.shape),
        ("diabetes", lambda: load_diabetes().data.shape),
        ("olivetti_faces", lambda: fetch_olivetti_faces(data_home=data_home, download_if_missing=True).data.shape),
        ("california_housing", lambda: fetch_california_housing(data_home=data_home, download_if_missing=True).data.shape),
        ("covertype", lambda: fetch_covtype(data_home=data_home, download_if_missing=True).data.shape),
        # Use a smaller subset for smoke tests.
        ("kddcup99", lambda: fetch_kddcup99(data_home=data_home, subset="http", download_if_missing=True).data.shape),
        ("lfw_people", lambda: fetch_lfw_people(data_home=data_home, min_faces_per_person=20, download_if_missing=True).data.shape),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test sklearn dataset download/cache for FloatSOM datasets.")
    parser.add_argument(
        "--data-home",
        type=str,
        required=True,
        help="Path to sklearn cache directory (sets SCIKIT_LEARN_DATA for this process).",
    )
    args = parser.parse_args()

    os.environ["SCIKIT_LEARN_DATA"] = args.data_home
    os.makedirs(args.data_home, exist_ok=True)

    failures: List[Tuple[str, str]] = []
    for dataset_name, run_check in _checks(args.data_home):
        try:
            shape = run_check()
            print(f"OK   {dataset_name:20s} shape={shape}")
        except Exception as exc:
            failures.append((dataset_name, str(exc)))
            print(f"FAIL {dataset_name:20s} {exc}")

    print(f"\nDone. ok={15 - len(failures)} fail={len(failures)}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
