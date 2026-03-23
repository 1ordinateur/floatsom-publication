"""Tests for chunked graph-distance precompute in MST/RNG topologies."""

from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest

from floatsom.topology import mst_topology as mst_topology_module
from floatsom.topology.mst_topology import MSTTopology, _resolve_fw_row_chunk_size
from floatsom.topology.rng_topology import RNGTopology


def _tree_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (4, 5),
    ]


def _dense_reference_fw(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(distances, 0.0)
    for u, v in edges:
        distances[u, v] = 1.0
        distances[v, u] = 1.0

    for k in range(num_nodes):
        distances = np.minimum(distances, distances[:, k:k + 1] + distances[k:k + 1, :])

    return distances


def _build_mst_topology(num_nodes: int) -> MSTTopology:
    topology = MSTTopology(
        num_nodes=num_nodes,
        input_dim=3,
        mst_update_frequency=None,
        initialization_method="random",
        seed=42,
        verbose=False,
    )
    topology.mst_edges = _tree_edges()
    topology._build_adjacency_list()
    return topology


def _to_numpy(matrix) -> np.ndarray:
    if isinstance(matrix, cp.ndarray):
        return cp.asnumpy(matrix)
    return np.asarray(matrix)


def test_chunked_fw_matches_dense_reference():
    """T-FEAT-001 / INV-003: chunked precompute matches dense FW reference."""
    topology = _build_mst_topology(num_nodes=6)
    topology._precompute_graph_distances()

    got = _to_numpy(topology.graph_distances)
    expected = _dense_reference_fw(6, _tree_edges())

    np.testing.assert_allclose(got, expected)


def test_chunked_fw_preserves_symmetry_and_zero_diagonal():
    """T-INV-002: precomputed matrix remains symmetric with zero diagonal."""
    topology = _build_mst_topology(num_nodes=6)
    topology._precompute_graph_distances()

    got = _to_numpy(topology.graph_distances)
    np.testing.assert_allclose(got, got.T)
    np.testing.assert_allclose(np.diag(got), np.zeros(got.shape[0], dtype=np.float32))


def test_cpu_backed_chunked_fw_matches_dense_reference():
    """Large-graph storage mode should remain equivalent to dense FW semantics."""
    topology = _build_mst_topology(num_nodes=6)
    topology._cpu_storage_threshold = 1
    topology._precompute_graph_distances()

    got = _to_numpy(topology.graph_distances)
    expected = _dense_reference_fw(6, _tree_edges())

    assert isinstance(topology.graph_distances, np.ndarray)
    np.testing.assert_allclose(got, expected)


def test_mst_precompute_uses_chunked_helper():
    """T-INV-001: MST precompute must call chunked Floyd-Warshall helper."""
    topology = _build_mst_topology(num_nodes=6)

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        wraps=mst_topology_module._run_chunked_floyd_warshall_gpu,
    ) as helper_mock:
        topology._precompute_graph_distances()

    helper_mock.assert_called_once()


def test_mst_cpu_storage_precompute_uses_chunked_helper():
    """CPU storage branch should also use the same GPU chunked helper."""
    topology = _build_mst_topology(num_nodes=6)
    topology._cpu_storage_threshold = 1

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        wraps=mst_topology_module._run_chunked_floyd_warshall_gpu,
    ) as helper_mock:
        topology._precompute_graph_distances()

    helper_mock.assert_called_once()


def test_rng_precompute_uses_chunked_mst_helper():
    """RNG path should inherit/use the same chunked MST FW implementation."""
    topology = RNGTopology(
        num_nodes=6,
        input_dim=3,
        mst_update_frequency=None,
        initialization_method="random",
        seed=42,
        verbose=False,
    )
    topology.mst_edges = _tree_edges()
    topology._build_adjacency_list()

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        wraps=mst_topology_module._run_chunked_floyd_warshall_gpu,
    ) as helper_mock:
        topology._precompute_graph_distances()

    helper_mock.assert_called_once()


def test_precompute_fail_hard_on_gpu_oom():
    """T-FEAT-002: OOM must propagate (no automatic fallback)."""
    topology = _build_mst_topology(num_nodes=6)

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        side_effect=cp.cuda.memory.OutOfMemoryError("forced oom"),
    ):
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            topology._precompute_graph_distances()


def test_cpu_storage_precompute_fail_hard_on_gpu_oom():
    """CPU storage branch must also propagate OOM (no fallback)."""
    topology = _build_mst_topology(num_nodes=6)
    topology._cpu_storage_threshold = 1

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        side_effect=cp.cuda.memory.OutOfMemoryError("forced oom"),
    ):
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            topology._precompute_graph_distances()


def test_resolver_respects_override_and_clamps_to_num_nodes():
    """Resolver should honor explicit override within [1, num_nodes]."""
    assert _resolve_fw_row_chunk_size(10, 4, requested_row_chunk_size=7) == 7
    assert _resolve_fw_row_chunk_size(10, 4, requested_row_chunk_size=100) == 10
    assert _resolve_fw_row_chunk_size(10, 4, requested_row_chunk_size=0) == 1


def test_resolver_uses_target_bytes_and_temp_multiplier():
    """Auto chunk sizing should account for target bytes and temp multiplier."""
    chunk = _resolve_fw_row_chunk_size(
        64,
        4,
        target_bytes=2048,
        temp_multiplier=2,
    )
    assert chunk == 4


def test_mst_params_override_applies_to_chunk_size():
    """`mst_params.fw_row_chunk_size` should drive helper chunk size."""
    topology = MSTTopology(
        num_nodes=6,
        input_dim=3,
        mst_update_frequency=None,
        initialization_method="random",
        seed=42,
        verbose=False,
        mst_params={"fw_row_chunk_size": 3},
    )
    topology.mst_edges = _tree_edges()
    topology._build_adjacency_list()

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        wraps=mst_topology_module._run_chunked_floyd_warshall_gpu,
    ) as helper_mock:
        topology._precompute_graph_distances()

    assert helper_mock.call_args.kwargs["row_chunk_size"] == 3


def test_mst_params_target_bytes_applies_to_chunk_size():
    """`mst_params.fw_row_target_bytes` should influence auto chunk size."""
    topology = MSTTopology(
        num_nodes=6,
        input_dim=3,
        mst_update_frequency=None,
        initialization_method="random",
        seed=42,
        verbose=False,
        mst_params={"fw_row_target_bytes": 32},
    )
    topology.mst_edges = _tree_edges()
    topology._build_adjacency_list()
    expected_chunk = _resolve_fw_row_chunk_size(
        topology.num_nodes,
        int(np.dtype(np.float32).itemsize),
        target_bytes=32,
    )

    with patch(
        "floatsom.topology.mst_topology._run_chunked_floyd_warshall_gpu",
        wraps=mst_topology_module._run_chunked_floyd_warshall_gpu,
    ) as helper_mock:
        topology._precompute_graph_distances()

    assert helper_mock.call_args.kwargs["row_chunk_size"] == expected_chunk


def test_mst_params_invalid_chunk_tuning_raises():
    """Invalid chunk tuning params should fail fast."""
    with pytest.raises(ValueError):
        MSTTopology(
            num_nodes=6,
            input_dim=3,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False,
            mst_params={"fw_row_chunk_size": 0},
        )

    with pytest.raises(ValueError):
        MSTTopology(
            num_nodes=6,
            input_dim=3,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False,
            mst_params={"fw_row_target_bytes": 0},
        )
