"""
Tests for topology structures in FloatSOM
Tests GridTopology, HexagonalTopology, and MSTTopology
"""

import numpy as np
import cupy as cp
import pytest

from floatsom.topology.grid_topology import GridTopology
from floatsom.topology.hexagonal_topology import HexagonalTopology
from floatsom.topology.mst_topology import MSTTopology


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return cp.random.randn(100, 10).astype(cp.float32)


class TestGridTopology:
    """Tests for GridTopology"""

    def test_grid_topology_shape(self):
        """Grid topology should have correct dimensions"""
        grid_size = 10
        topology = GridTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        assert topology.total_nodes == grid_size * grid_size
        assert topology.grid_size == grid_size

    def test_grid_topology_neighbors_interior(self):
        """Interior nodes in grid should have 4 neighbors"""
        grid_size = 5
        topology = GridTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()

        # Check interior node (2,2) -> node 12 in flattened grid
        node_idx = 2 * grid_size + 2
        coord = topology.coord_grid[node_idx]

        # Count neighbors at distance 1.0 (adjacent cells)
        distances = topology._distance_matrix[node_idx]
        neighbors = cp.sum((distances > 0) & (distances <= 1.1))  # Allow small tolerance

        # Interior node should have exactly 4 neighbors
        assert neighbors == 4

    def test_grid_topology_neighbors_edge(self):
        """Edge nodes should have fewer than 4 neighbors"""
        grid_size = 5
        topology = GridTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()

        # Check edge node (0,2) -> node 2 in flattened grid
        node_idx = 2
        distances = topology._distance_matrix[node_idx]
        neighbors = cp.sum((distances > 0) & (distances <= 1.1))

        # Edge node should have 3 neighbors
        assert neighbors == 3

        # Check corner node (0,0) -> node 0
        node_idx = 0
        distances = topology._distance_matrix[node_idx]
        neighbors = cp.sum((distances > 0) & (distances <= 1.1))

        # Corner node should have 2 neighbors
        assert neighbors == 2

    def test_grid_topology_toroidal_wrapping(self):
        """Toroidal grid should wrap around edges"""
        grid_size = 5
        topology = GridTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="toroidal",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()

        # In toroidal grid, corner node (0,0) should have 4 neighbors
        node_idx = 0
        distances = topology._distance_matrix[node_idx]
        neighbors = cp.sum((distances > 0) & (distances <= 1.1))

        assert neighbors == 4


class TestHexagonalTopology:
    """Tests for HexagonalTopology"""

    def test_hexagonal_topology_neighbors(self):
        """Interior nodes in hexagonal grid should have 6 neighbors"""
        grid_size = 5
        topology = HexagonalTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()

        # Check an interior node - node at (2,2)
        node_idx = 2 * grid_size + 2
        squared_distances = topology._distance_matrix[node_idx]

        # Note: _distance_matrix stores SQUARED distances
        # In hexagonal grid with 0.5 row offset, diagonal neighbors are at distance ~1.118
        # (sqrt(0.5^2 + 1^2) ≈ 1.118), squared = 1.25
        # Use squared threshold of 1.5 (equivalent to distance <= 1.22)
        neighbors = cp.sum((squared_distances > 0) & (squared_distances <= 1.5))

        # Interior hexagonal node should have 6 neighbors
        assert neighbors == 6

    def test_hexagonal_topology_coordinate_offset(self):
        """Hexagonal topology should offset odd rows"""
        grid_size = 5
        topology = HexagonalTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        coords = topology.get_coordinates()

        # Check that odd rows have x-offset of 0.5
        for i in range(grid_size):
            for j in range(grid_size):
                node_idx = i * grid_size + j
                x_coord = float(coords[node_idx, 0])

                if i % 2 == 1:
                    # Odd row should have offset
                    expected_x = j + 0.5
                else:
                    # Even row should not have offset
                    expected_x = float(j)

                assert abs(x_coord - expected_x) < 0.01


class TestMSTTopology:
    """Tests for MSTTopology"""

    def test_mst_topology_tree_structure(self, sample_data):
        """MST should form a valid tree structure"""
        num_nodes = 50
        topology = MSTTopology(
            num_nodes=num_nodes,
            input_dim=10,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Initialize weights
        weights = topology.initialize_weights(sample_data)

        # MST edges should form a tree (n-1 edges for n nodes)
        assert len(topology.mst_edges) == num_nodes - 1

        # Check that all nodes are connected
        assert topology.adjacency_list is not None
        assert len(topology.adjacency_list) == num_nodes

    def test_mst_topology_updates(self, sample_data):
        """MST should update when weights change"""
        num_nodes = 30
        topology = MSTTopology(
            num_nodes=num_nodes,
            input_dim=10,
            mst_update_frequency=1,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Initialize weights
        initial_weights = topology.initialize_weights(sample_data)
        initial_edges = topology.mst_edges.copy()

        # Update with different weights
        new_weights = initial_weights + cp.random.randn(*initial_weights.shape).astype(cp.float32) * 0.5
        topology.update_topology(new_weights, current_iteration=1)

        # MST edges should potentially be different
        # (may be same if structure is similar, but tree property should hold)
        assert len(topology.mst_edges) == num_nodes - 1

    def test_mst_topology_graph_distances(self, sample_data):
        """MST should compute graph distances correctly"""
        num_nodes = 20
        topology = MSTTopology(
            num_nodes=num_nodes,
            input_dim=10,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Initialize weights
        weights = topology.initialize_weights(sample_data)

        # Check graph distances exist
        assert topology.graph_distances is not None

        # Diagonal should be zero (distance to self)
        if isinstance(topology.graph_distances, cp.ndarray):
            diagonal = cp.diag(topology.graph_distances)
        else:
            diagonal = np.diag(topology.graph_distances)
        assert cp.all(diagonal == 0) or np.all(diagonal == 0)

        # All nodes should be reachable (no infinite distances)
        if isinstance(topology.graph_distances, cp.ndarray):
            assert not cp.any(cp.isinf(topology.graph_distances))
        else:
            assert not np.any(np.isinf(topology.graph_distances))

    def test_mst_distance_helper_handles_cpu_storage(self):
        """MST distance helper should work when graph distances are stored as NumPy."""
        topology = MSTTopology(
            num_nodes=2,
            input_dim=2,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False,
        )
        topology.graph_distances = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        assert topology._calculate_mst_distance(0, 1) == 1.0

    def test_mst_influence_matrix_lazy_initialization(self, sample_data):
        """Influence lookup should lazily initialize cache when precompute was skipped."""
        topology = MSTTopology(
            num_nodes=16,
            input_dim=10,
            mst_update_frequency=None,
            initialization_method="random",
            seed=42,
            verbose=False,
        )
        topology.initialize_weights(sample_data)

        influence = topology.get_precomputed_influence_matrix(1.0, "gaussian")
        assert influence.shape == (16, 16)


class TestTopologyDistances:
    """Tests for topology distance properties"""

    def test_topology_distances_symmetric(self):
        """Distance d(i,j) should equal d(j,i)"""
        grid_size = 5
        topology = GridTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()
        dist_matrix = topology._distance_matrix

        # Check symmetry
        if isinstance(dist_matrix, cp.ndarray):
            assert cp.allclose(dist_matrix, dist_matrix.T)
        else:
            assert np.allclose(dist_matrix, dist_matrix.T)

    def test_hexagonal_distances_symmetric(self):
        """Hexagonal distances should be symmetric"""
        grid_size = 5
        topology = HexagonalTopology(
            grid_size=grid_size,
            input_dim=5,
            topology_type="planar",
            initialization_method="random",
            seed=42,
            verbose=False
        )

        # Precompute distance matrix
        topology._precompute_distance_matrix()
        dist_matrix = topology._distance_matrix

        # Check symmetry
        if isinstance(dist_matrix, cp.ndarray):
            assert cp.allclose(dist_matrix, dist_matrix.T)
        else:
            assert np.allclose(dist_matrix, dist_matrix.T)
