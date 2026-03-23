"""
Tests for RNG topology.
"""

from collections import deque

import cupy as cp
import numpy as np

from floatsom.floatsom_params import FloatSOMParams, ProcessingConfig, TopologyConfig
from floatsom.topology.rng_topology import RNGTopology, calculate_rng_cpu, calculate_rng_gpu
from floatsom.topology.topology_factory import create_topology


def _reachable_node_count(edges, num_nodes: int, start_node: int = 0) -> int:
    adjacency = {node: [] for node in range(num_nodes)}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = set()
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return len(visited)


def test_calculate_rng_cpu_matches_canonical_lune_criterion():
    """The middle point should block the longer edge under the strict RNG rule."""
    distances = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert set(calculate_rng_cpu(distances)) == {(0, 1), (1, 2)}


def test_calculate_rng_cpu_uses_strict_inequality():
    """Equal blocker distances must not remove an edge from the RNG."""
    distances = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert set(calculate_rng_cpu(distances)) == {(0, 1), (0, 2), (1, 2)}


def test_calculate_rng_cpu_repairs_pathological_disconnected_distance_matrix():
    """Fallback is a defensive guard for non-Euclidean/pathological inputs."""
    distances = np.asarray(
        [
            [0.0, 1.0, 5.0, 8.0],
            [1.0, 0.0, 8.0, 5.0],
            [5.0, 8.0, 0.0, 1.0],
            [8.0, 5.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    raw_edges = calculate_rng_cpu(distances, ensure_connected=False)
    repaired_edges = calculate_rng_cpu(distances, ensure_connected=True)

    assert _reachable_node_count(raw_edges, 4) == 2
    assert _reachable_node_count(repaired_edges, 4) == 4
    assert set(raw_edges) == {(0, 1), (2, 3)}


def test_rng_topology_initialization_builds_graph_state():
    """RNG topology should initialize adjacency and finite graph distances."""
    topology = RNGTopology(
        num_nodes=16,
        input_dim=6,
        mst_update_frequency=None,
        initialization_method="random",
        seed=42,
        verbose=False,
    )
    data = cp.random.randn(64, 6).astype(cp.float32)
    topology.initialize_weights(data)

    assert topology.adjacency_list is not None
    assert len(topology.adjacency_list) == 16
    assert topology.graph_distances is not None

    graph_dist = topology.graph_distances
    if isinstance(graph_dist, cp.ndarray):
        assert not bool(cp.any(cp.isinf(graph_dist)))
    else:
        assert not bool(np.any(np.isinf(graph_dist)))


def test_calculate_rng_gpu_matches_cpu_exact_edges():
    """GPU RNG construction should match CPU RNG edges exactly."""
    rng = np.random.RandomState(7)
    points = rng.uniform(-1.0, 1.0, (20, 5)).astype(np.float32)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sum(diff * diff, axis=2).astype(np.float32)

    cpu_edges = set(calculate_rng_cpu(distances))
    gpu_edges = set(
        calculate_rng_gpu(
            cp.asarray(distances),
            ensure_connected=True,
            block_chunk_size=8,
        )
    )

    assert gpu_edges == cpu_edges


def test_rng_topology_uses_periodic_update_schedule():
    """RNG topology should expose MST-like periodic update behavior."""
    topology = RNGTopology(
        num_nodes=9,
        input_dim=4,
        mst_update_frequency=3,
        initialization_method="random",
        seed=42,
        verbose=False,
        dynamic_mst_frequency=False,
    )

    assert topology.should_update_topology(0) is True
    assert topology.should_update_topology(1) is False
    assert topology.should_update_topology(2) is True
    assert topology.should_update_topology(3) is False


def test_topology_factory_creates_rng_topology():
    """Factory should route topology_type='rng' to RNGTopology."""
    params = FloatSOMParams(
        input_dim=6,
        topology_config=TopologyConfig(topology_type="rng", num_nodes=25),
        processing_config=ProcessingConfig(chunk_size=1000),
    )
    topology = create_topology(params)
    assert isinstance(topology, RNGTopology)
