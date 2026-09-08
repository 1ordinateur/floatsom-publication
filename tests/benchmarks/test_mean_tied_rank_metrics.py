"""
Unit tests for Mean Tied Rank topology diagnostics.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _install_cupy_stub_if_needed() -> bool:
    try:
        import cupy  # noqa: F401
        return False
    except ImportError:
        pass

    cupy_stub = types.ModuleType("cupy")
    cupy_stub.ndarray = np.ndarray
    cupy_stub.float32 = np.float32
    cupy_stub.int32 = np.int32
    cupy_stub.inf = np.inf
    cupy_stub.newaxis = np.newaxis
    cupy_stub.asarray = np.asarray
    cupy_stub.array = np.asarray
    cupy_stub.asnumpy = np.asarray
    cupy_stub.zeros = np.zeros
    cupy_stub.full = np.full
    cupy_stub.arange = np.arange
    cupy_stub.stack = np.stack
    cupy_stub.einsum = np.einsum
    cupy_stub.argpartition = np.argpartition
    cupy_stub.argsort = np.argsort
    cupy_stub.take_along_axis = np.take_along_axis
    cupy_stub.concatenate = np.concatenate
    cupy_stub.bincount = np.bincount
    cupy_stub.logical_or = np.logical_or
    cupy_stub.count_nonzero = np.count_nonzero
    cupy_stub.isfinite = np.isfinite
    sys.modules["cupy"] = cupy_stub
    return True


_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRICS_PATH = _REPO_ROOT / "benchmarks/evaluation/metrics.py"
if not _METRICS_PATH.exists():
    _METRICS_PATH = _REPO_ROOT / "benchmarks/evaluation/metrics.py"


@pytest.fixture(scope="module")
def metrics_module():
    if not _METRICS_PATH.exists():
        pytest.fail(f"Missing metrics module under test: {_METRICS_PATH}")

    inserted_cupy_stub = _install_cupy_stub_if_needed()
    spec = importlib.util.spec_from_file_location("metrics_under_test", _METRICS_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("Unable to create import spec for metrics.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if inserted_cupy_stub:
        sys.modules.pop("cupy", None)
    return module


def test_tied_rank_shells_use_average_ordinal_rank(metrics_module):
    graph_distances = np.asarray(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [2, 3, 3, 0],
        ],
        dtype=np.float32,
    )

    tied_ranks = metrics_module.calculate_tied_rank_table(graph_distances)

    assert tied_ranks[0, 1] == pytest.approx(1.5)
    assert tied_ranks[0, 2] == pytest.approx(1.5)
    assert tied_ranks[0, 3] == pytest.approx(3.0)


def test_hexagonal_adjacency_has_six_neighbor_interior_not_square_eight(metrics_module):
    adjacency = metrics_module.build_hexagonal_adjacency_list(3)
    center = 4

    assert len(adjacency[center]) == 6
    assert set(adjacency[center]) == {0, 1, 3, 5, 6, 7}
    assert 2 not in adjacency[center]
    assert 8 not in adjacency[center]


def test_disconnected_graph_raises_explicit_error(metrics_module):
    with pytest.raises(ValueError, match="disconnected"):
        metrics_module._adjacency_list_to_distance_matrix(
            {0: [1], 1: [0], 2: []},
            total_nodes=3,
        )


def test_calculate_mean_tied_rank_on_hand_built_graph(metrics_module):
    class Wrapper:
        weights = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
        topology_type = "mst"
        adjacency_list = {0: [1], 1: [0, 2], 2: [1]}
        topology = None

        def get_weights(self):
            return self.weights

    data = np.asarray([[0.1], [1.9]], dtype=np.float32)

    assert metrics_module.calculate_mean_tied_rank(
        Wrapper(),
        data,
        use_gpu=False,
        batch_size=1,
    ) == pytest.approx(1.0)


def test_mtr_permutation_null_freezes_graph_and_has_expected_midrank_mean(metrics_module):
    graph_distances = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.float32,
    )
    tied_ranks = metrics_module.calculate_tied_rank_table(graph_distances)

    # Deliberately concentrate all observations in two ordered BMU pairs. The
    # exact permutation-null expectation remains P/2 despite this imbalance.
    bmu1 = np.asarray([0] * 90 + [2] * 10, dtype=np.int64)
    bmu2 = np.asarray([1] * 90 + [3] * 10, dtype=np.int64)
    stats = metrics_module.calculate_mean_tied_rank_permutation_null_from_bmus(
        bmu1,
        bmu2,
        tied_ranks,
        n_permutations=5_000,
        random_seed=123,
    )

    assert stats["mean_tied_rank"] == pytest.approx(1.05)
    assert stats["mean_tied_rank_null_theoretical_mean"] == pytest.approx(2.0)
    assert stats["mean_tied_rank_null_mean"] == pytest.approx(2.0, abs=0.05)
    assert stats["mean_tied_rank_observed_to_null_ratio"] == pytest.approx(0.525, abs=0.02)
    assert stats["mean_tied_rank_null_permutations"] == pytest.approx(5_000)
    assert 0.0 < stats["mean_tied_rank_null_lower_tail_p"] < 0.5


def test_mtr_permutation_null_is_reproducible(metrics_module):
    graph_distances = np.asarray(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [2, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    tied_ranks = metrics_module.calculate_tied_rank_table(graph_distances)
    bmu1 = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    bmu2 = np.asarray([1, 2, 3, 3, 1], dtype=np.int64)

    first = metrics_module.calculate_mean_tied_rank_permutation_null_from_bmus(
        bmu1,
        bmu2,
        tied_ranks,
        n_permutations=100,
        random_seed=77,
    )
    second = metrics_module.calculate_mean_tied_rank_permutation_null_from_bmus(
        bmu1,
        bmu2,
        tied_ranks,
        n_permutations=100,
        random_seed=77,
    )

    assert first == second
