"""
Test topology invariants.

Tests graph/topology structure guarantees:
- MST has exactly N-1 edges for N nodes
- MST is connected (all nodes reachable)
- MST is acyclic
- Adjacency list is bidirectional
- Unique node coordinates
"""

import pytest
import numpy as np
from collections import deque

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# Import topology functions (these are CPU-based, so no GPU check needed)
from floatsom.topology.mst_topology import (
    calculate_mst_cpu,
    build_adjacency_list,
)


# =============================================================================
# MST Edge Count Invariants
# =============================================================================

class TestMSTEdgeCount:
    """Test that MST has exactly N-1 edges."""

    @pytest.mark.parametrize("num_nodes", [5, 10, 20, 50, 100])
    def test_mst_has_n_minus_one_edges(self, num_nodes):
        """MST for N nodes should have exactly N-1 edges."""
        rng = np.random.RandomState(42)

        # Create random distance matrix
        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)

        expected_edges = num_nodes - 1
        assert len(mst_edges) == expected_edges, \
            f"Expected {expected_edges} edges, got {len(mst_edges)}"

    def test_mst_single_node_zero_edges(self):
        """MST with 1 node should have 0 edges."""
        distances = np.array([[0.0]])
        mst_edges = calculate_mst_cpu(distances)

        assert len(mst_edges) == 0, f"Single node MST should have 0 edges, got {len(mst_edges)}"

    def test_mst_two_nodes_one_edge(self):
        """MST with 2 nodes should have 1 edge."""
        distances = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        mst_edges = calculate_mst_cpu(distances)

        assert len(mst_edges) == 1, f"Two node MST should have 1 edge, got {len(mst_edges)}"


# =============================================================================
# MST Connectivity Invariants
# =============================================================================

class TestMSTConnectivity:
    """Test that MST is connected."""

    def _bfs_reachable_nodes(self, adjacency_list, start_node):
        """Return set of all nodes reachable from start_node via BFS."""
        visited = set()
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for neighbor in adjacency_list.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    @pytest.mark.parametrize("num_nodes", [5, 10, 20])
    def test_mst_all_nodes_reachable(self, num_nodes):
        """All nodes should be reachable from node 0."""
        rng = np.random.RandomState(42)

        # Create distance matrix
        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        reachable = self._bfs_reachable_nodes(adjacency_list, 0)

        assert len(reachable) == num_nodes, \
            f"Only {len(reachable)}/{num_nodes} nodes reachable from node 0"

    def test_mst_connectivity_any_start(self):
        """All nodes should be reachable from any starting node."""
        num_nodes = 15
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        # Test from multiple start nodes
        for start in [0, num_nodes // 2, num_nodes - 1]:
            reachable = self._bfs_reachable_nodes(adjacency_list, start)
            assert len(reachable) == num_nodes, \
                f"From node {start}: only {len(reachable)}/{num_nodes} nodes reachable"


# =============================================================================
# MST Acyclicity Invariants
# =============================================================================

class TestMSTAcyclicity:
    """Test that MST has no cycles."""

    def _has_cycle(self, adjacency_list, num_nodes):
        """Check if the graph has a cycle using DFS."""
        visited = set()

        def dfs(node, parent):
            visited.add(node)
            for neighbor in adjacency_list.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True  # Back edge found = cycle
            return False

        # Start from all components (handles disconnected graphs)
        for node in range(num_nodes):
            if node not in visited:
                if dfs(node, -1):
                    return True
        return False

    @pytest.mark.parametrize("num_nodes", [5, 10, 20])
    def test_mst_no_cycles(self, num_nodes):
        """MST should have no cycles."""
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        assert not self._has_cycle(adjacency_list, num_nodes), "MST contains a cycle"


# =============================================================================
# Adjacency List Invariants
# =============================================================================

class TestAdjacencyListInvariants:
    """Test adjacency list properties."""

    @pytest.mark.parametrize("num_nodes", [5, 10, 20])
    def test_adjacency_bidirectional(self, num_nodes):
        """If v is in adj[u], then u must be in adj[v]."""
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        for u, neighbors in adjacency_list.items():
            for v in neighbors:
                assert u in adjacency_list[v], \
                    f"Edge ({u}, {v}) not bidirectional: {u} not in adj[{v}]"

    def test_adjacency_list_all_nodes_present(self):
        """Adjacency list should have entries for all nodes."""
        num_nodes = 10
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        for node in range(num_nodes):
            assert node in adjacency_list, f"Node {node} missing from adjacency list"

    def test_adjacency_no_self_loops(self):
        """No node should be its own neighbor."""
        num_nodes = 10
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)
        adjacency_list = build_adjacency_list(mst_edges, num_nodes)

        for node, neighbors in adjacency_list.items():
            assert node not in neighbors, f"Node {node} has self-loop"


# =============================================================================
# MST Edge Validity
# =============================================================================

class TestMSTEdgeValidity:
    """Test MST edge properties."""

    def test_all_edges_use_valid_nodes(self):
        """All edge endpoints should be valid node indices."""
        num_nodes = 20
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)

        for u, v in mst_edges:
            assert 0 <= u < num_nodes, f"Invalid node index u={u}"
            assert 0 <= v < num_nodes, f"Invalid node index v={v}"

    def test_no_duplicate_edges(self):
        """MST should not contain duplicate edges."""
        num_nodes = 20
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)

        # Normalize edge representation (smaller node first)
        normalized_edges = set()
        for u, v in mst_edges:
            edge = (min(u, v), max(u, v))
            assert edge not in normalized_edges, f"Duplicate edge: {edge}"
            normalized_edges.add(edge)

    def test_edges_have_distinct_endpoints(self):
        """Edges should connect different nodes (no self-edges)."""
        num_nodes = 20
        rng = np.random.RandomState(42)

        points = rng.uniform(0, 10, (num_nodes, 2))
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))

        mst_edges = calculate_mst_cpu(distances)

        for u, v in mst_edges:
            assert u != v, f"Self-edge found: ({u}, {v})"


# =============================================================================
# Grid Topology Invariants
# =============================================================================

class TestGridTopologyInvariants:
    """Test grid topology coordinate properties."""

    @pytest.mark.parametrize("grid_size", [5, 8, 10])
    def test_grid_unique_coordinates(self, grid_size):
        """Grid topology should have unique coordinates for all nodes."""
        num_nodes = grid_size * grid_size

        # Generate grid coordinates
        coordinates = []
        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            coordinates.append((row, col))

        # Check uniqueness
        unique_coords = set(coordinates)
        assert len(unique_coords) == num_nodes, \
            f"Expected {num_nodes} unique coordinates, got {len(unique_coords)}"

    @pytest.mark.parametrize("grid_size", [5, 8, 10])
    def test_grid_coordinates_bounded(self, grid_size):
        """Grid coordinates should be within bounds."""
        num_nodes = grid_size * grid_size

        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size

            assert 0 <= row < grid_size, f"Row {row} out of bounds [0, {grid_size})"
            assert 0 <= col < grid_size, f"Col {col} out of bounds [0, {grid_size})"

    def test_grid_coordinates_complete(self):
        """Grid should have all (row, col) combinations."""
        grid_size = 5
        num_nodes = grid_size * grid_size

        expected_coords = set()
        for r in range(grid_size):
            for c in range(grid_size):
                expected_coords.add((r, c))

        actual_coords = set()
        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            actual_coords.add((row, col))

        assert actual_coords == expected_coords, "Grid coordinates incomplete"


# =============================================================================
# Distance Matrix Input Invariants
# =============================================================================

class TestDistanceMatrixInputInvariants:
    """Test invariants about distance matrix inputs to MST."""

    def test_distance_matrix_symmetric(self, distance_matrix_for_mst):
        """Distance matrix should be symmetric."""
        np.testing.assert_allclose(
            distance_matrix_for_mst,
            distance_matrix_for_mst.T,
            rtol=1e-10,
            err_msg="Distance matrix not symmetric"
        )

    def test_distance_matrix_zero_diagonal(self, distance_matrix_for_mst):
        """Distance matrix diagonal should be zero."""
        diagonal = np.diag(distance_matrix_for_mst)
        np.testing.assert_allclose(
            diagonal,
            np.zeros_like(diagonal),
            atol=1e-10,
            err_msg="Distance matrix diagonal not zero"
        )

    def test_distance_matrix_non_negative(self, distance_matrix_for_mst):
        """Distance matrix values should be non-negative."""
        assert np.all(distance_matrix_for_mst >= 0), \
            f"Negative values in distance matrix: min={distance_matrix_for_mst.min()}"

    def test_distance_matrix_square(self, distance_matrix_for_mst):
        """Distance matrix should be square."""
        assert distance_matrix_for_mst.shape[0] == distance_matrix_for_mst.shape[1], \
            f"Distance matrix not square: {distance_matrix_for_mst.shape}"
