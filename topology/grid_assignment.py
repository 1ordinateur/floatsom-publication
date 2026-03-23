"""
Grid assignment module for transforming MST topology to grid structures.
Uses Delaunay triangulation with Force Atlas 2 layout to assign nodes to grid cells with minimal overlap.
"""

import cupy as cp
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import warnings
import cugraph
import cudf
from scipy.optimize import linear_sum_assignment

# WARNING: CuPy's Delaunay implementation may be deprecated in newer versions
# Attempting to import from cupyx.scipy.spatial first, with fallback to local copy
try:
    import cupyx.scipy.spatial as cuspatial
    from cupyx.scipy.spatial import Delaunay
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported Delaunay from cupyx.scipy.spatial")
except (ImportError, AttributeError) as e:
    # Fallback to local copy of Delaunay implementation
    warnings.warn(
        f"CuPy's Delaunay import failed ({e}). Using local fallback implementation. "
        "This may be due to version deprecation in your CuPy environment. "
        "Consider updating CuPy or using the local implementation permanently.",
        UserWarning
    )
    logger = logging.getLogger(__name__)
    logger.warning("Using local Delaunay implementation due to CuPy import failure")
    
    # Import from local copy
    from .delaunay_local import Delaunay


class GridAssigner(ABC):
    """Abstract base class for grid assignment using Delaunay triangulation with Force Atlas 2 layout."""
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize grid assigner.
        
        Args:
            chunk_size: Max nodes to process at once for distance matrix computation
        """
        self.chunk_size = chunk_size
    
    def assign_nodes_to_grid(self, weights: cp.ndarray, num_nodes: int) -> Tuple[Dict, Dict]:
        """
        Assign nodes to grid positions using Delaunay triangulation and Force Atlas 2.
        
        Args:
            weights: MST node weights (num_nodes, input_dim)
            num_nodes: Number of nodes
            
        Returns:
            Tuple of (node_to_grid_mapping, grid_adjacency_list)
        """
        # Derive grid dimensions
        grid_size = int(np.sqrt(num_nodes))
        if grid_size * grid_size < num_nodes:
            grid_size += 1
        
        logger.debug(f"Assigning {num_nodes} nodes to {grid_size}x{grid_size} grid using Delaunay method")
        
        # Project to 2D
        nodes_2d = self._project_to_2d(weights)
        
        # Build Delaunay triangulation and get graph
        G = self._build_delaunay_graph(nodes_2d)
        
        # Apply Force Atlas 2 layout to preserve topology
        positions = self._apply_force_atlas2_layout(G, nodes_2d)
        
        # Generate grid positions
        grid_positions = self.generate_grid_positions(grid_size)
        
        # Normalize both to [0, 1] range
        self._normalize_inplace(positions)
        self._normalize_inplace(grid_positions)
        
        # Scale positions to grid dimensions
        positions[:, 0] *= (grid_size - 1)
        positions[:, 1] *= (grid_size - 1)
        grid_positions[:, 0] *= (grid_size - 1)
        grid_positions[:, 1] *= (grid_size - 1)
        
        # Final assignment using scipy's linear_sum_assignment
        assignments = self._assign_to_grid_positions(positions, grid_positions)
        
        # Build mapping and adjacency
        node_to_grid = self._build_node_to_grid_mapping(assignments, grid_size)
        adjacency = self.build_grid_adjacency(node_to_grid, grid_size)
        
        return node_to_grid, adjacency
    
    def _build_delaunay_graph(self, nodes_2d: cp.ndarray) -> cugraph.Graph:
        """
        Build Delaunay triangulation and convert to cuGraph graph.
        
        Args:
            nodes_2d: 2D projected node positions (num_nodes, 2) on GPU
            
        Returns:
            cuGraph Graph object with Delaunay edges
        """
        logger.debug("Building Delaunay triangulation on GPU")
        
        # Ensure nodes_2d is on GPU and contiguous
        if not isinstance(nodes_2d, cp.ndarray):
            nodes_2d = cp.asarray(nodes_2d, dtype=cp.float32)
        
        # Create Delaunay triangulation on GPU
        delaunay = Delaunay(nodes_2d)
        
        # Extract edges from simplices
        simplices = delaunay.simplices  # Shape: (n_triangles, 3)
        edges = []
        
        # Extract unique edges from triangles
        for i in range(3):
            # Each triangle contributes 3 edges
            src = simplices[:, i]
            dst = simplices[:, (i + 1) % 3]
            edges.append(cp.stack([src, dst], axis=1))
        
        # Combine all edges
        all_edges = cp.vstack(edges)
        
        # Make edges undirected by adding reverse edges
        reverse_edges = cp.stack([all_edges[:, 1], all_edges[:, 0]], axis=1)
        all_edges = cp.vstack([all_edges, reverse_edges])
        
        # Remove duplicates by sorting each edge and then using unique
        sorted_edges = cp.sort(all_edges, axis=1)
        unique_edges = cp.unique(sorted_edges, axis=0)
        
        # Create cuGraph DataFrame
        edge_df = cudf.DataFrame({
            'src': unique_edges[:, 0].get(),  # Transfer to CPU for cuDF
            'dst': unique_edges[:, 1].get()
        })
        
        # Create cuGraph Graph
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')
        
        logger.debug(f"Delaunay graph created with {G.number_of_edges()} edges")
        return G
    
    def _apply_force_atlas2_layout(self, G: cugraph.Graph, initial_pos: cp.ndarray) -> cp.ndarray:
        """
        Apply Force Atlas 2 layout to preserve topology without crossings.
        
        Args:
            G: cuGraph Graph object
            initial_pos: Initial 2D positions (num_nodes, 2) on GPU
            
        Returns:
            Optimized 2D positions (num_nodes, 2) on GPU
        """
        logger.debug("Applying Force Atlas 2 layout for topology preservation")
        
        # Convert initial positions to DataFrame for cuGraph
        n_nodes = len(initial_pos)

        if isinstance(initial_pos, cp.ndarray):
            x = initial_pos[:, 0].get()
            y = initial_pos[:, 1].get()
        else:
            x = initial_pos[:, 0]
            y = initial_pos[:, 1]

        pos_df = cudf.DataFrame({
            'vertex': cp.arange(n_nodes).get(),
            'x': x,
            'y': y
        })
        
        # Apply Force Atlas 2 with appropriate parameters
        pos_result = cugraph.layout.force_atlas2(
            G,
            max_iter=500,  # Iterations for layout optimization
            pos_list=pos_df,  # Use initial positions
            outbound_attraction_distribution=True,
            lin_log_mode=False,
            edge_weight_influence=1.0,
            jitter_tolerance=1.0,
            barnes_hut_optimize=True,  # Use Barnes-Hut for efficiency
            barnes_hut_theta=1.0,
            scaling_ratio=2.0,
            strong_gravity_mode=False,
            gravity=1.0,
            verbose=False
        )
        
        # Extract positions and convert back to GPU array
        x = cp.asarray(pos_result['x'].to_numpy())
        y = cp.asarray(pos_result['y'].to_numpy())
        positions = cp.stack([x, y], axis=1)
        
        logger.debug("Force Atlas 2 layout completed")
        return positions
    
    def _assign_to_grid_positions(self, positions: cp.ndarray, 
                                  grid_positions: cp.ndarray) -> cp.ndarray:
        """
        Final assignment of nodes to grid positions using linear sum assignment.
        
        Args:
            positions: Optimized node positions (num_nodes, 2) on GPU
            grid_positions: Grid positions (num_grid_positions, 2) on GPU
            
        Returns:
            Grid assignments for each node on GPU
        """
        logger.debug("Performing final grid assignment")
        
        # Compute distance matrix between positions and grid
        # Transfer to CPU for scipy (this is the only CPU step)
        positions_cpu = positions.get()
        grid_positions_cpu = grid_positions.get()
        
        # Compute cost matrix on CPU
        from scipy.spatial.distance import cdist
        cost_matrix = cdist(positions_cpu, grid_positions_cpu, metric='euclidean')
        
        # Use Hungarian algorithm for final assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Convert back to GPU array
        assignments = cp.asarray(col_ind, dtype=cp.int32)
        
        total_cost = cost_matrix[row_ind, col_ind].sum()
        logger.debug(f"Grid assignment completed with total cost: {total_cost:.4f}")
        
        return assignments
    
    def _compute_cost_matrix_efficient(self, nodes_2d: cp.ndarray, 
                                       grid_positions: cp.ndarray) -> cp.ndarray:
        """
        Compute cost matrix using efficient einsum operations from utils.py pattern.
        Uses the same approach as EuclideanDistance.calculate_distances().
        
        Args:
            nodes_2d: 2D projected node positions (num_nodes, 2)
            grid_positions: Grid positions (num_positions, 2)
            
        Returns:
            Cost matrix (num_nodes, num_positions)
        """
        n_nodes = len(nodes_2d)
        n_positions = len(grid_positions)
        
        # Process in chunks on GPU to manage memory
        if n_nodes > self.chunk_size:
            logger.debug(f"Computing {n_nodes}x{n_positions} cost matrix in chunks on GPU")
            
            # Allocate result on GPU
            cost_matrix = cp.zeros((n_nodes, n_positions), dtype=cp.float32)
            
            # Precompute grid positions squared norms once
            grid_sq = cp.einsum('jk,jk->j', grid_positions, grid_positions)
            
            for i in range(0, n_nodes, self.chunk_size):
                end_i = min(i + self.chunk_size, n_nodes)
                nodes_chunk = nodes_2d[i:end_i]
                
                # Use the efficient distance calculation pattern from utils.py
                cost_matrix[i:end_i] = self._calculate_distances_efficient(nodes_chunk, grid_positions, grid_sq)
                
                # Free GPU memory periodically
                if i % (self.chunk_size * 5) == 0:
                    cp.get_default_memory_pool().free_all_blocks()
            
            return cost_matrix
        else:
            # Small enough - compute directly on GPU using efficient pattern
            return self._calculate_distances_efficient(nodes_2d, grid_positions)
    
    def _calculate_distances_efficient(self, batch: cp.ndarray, weights: cp.ndarray, 
                                      weights_sq: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Calculate Euclidean distances using the optimized pattern from utils.py.
        Based on EuclideanDistance.calculate_distances() implementation.
        
        Args:
            batch: First set of points (batch_size, 2)
            weights: Second set of points (n_points, 2)
            weights_sq: Optional precomputed squared norms of weights
            
        Returns:
            Distance matrix (batch_size, n_points)
        """
        # Calculate squared norms
        batch_sq = cp.einsum('ik,ik->i', batch, batch)
        if weights_sq is None:
            weights_sq = cp.einsum('jk,jk->j', weights, weights)
        
        # Calculate dot products and reuse as main array
        distances = cp.einsum('ik,jk->ij', batch, weights)
        
        # In-place operations to save memory
        distances *= -2
        distances += batch_sq[:, cp.newaxis]
        distances += weights_sq[cp.newaxis, :]
        cp.maximum(distances, 0, out=distances)  # Ensure non-negative
        cp.sqrt(distances, out=distances)
        
        return distances
    
    def _project_to_2d(self, weights: cp.ndarray) -> cp.ndarray:
        """
        Project high-dimensional weights to 2D using PCA.
        
        Args:
            weights: High-dimensional weights (num_nodes, input_dim)
            
        Returns:
            2D projected positions (num_nodes, 2)
        """
        # Use cuML PCA (GPU-accelerated) - required for reform_grid
        from cuml.decomposition import PCA
        logger.debug("Using cuML PCA for 2D projection")
        pca = PCA(n_components=2)
        return pca.fit_transform(weights)
    
    def _normalize_inplace(self, coords: cp.ndarray) -> None:
        """
        Normalize coordinates to [0, 1] range in-place.
        
        Args:
            coords: Coordinates to normalize (n_points, 2)
        """
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        
        # Avoid division by zero
        ranges = cp.maximum(ranges, 1e-10)
        
        # Normalize in-place
        coords -= mins
        coords /= ranges
    
    def _build_node_to_grid_mapping(self, assignments: cp.ndarray, 
                                   grid_size: int) -> Dict[int, Tuple[int, int]]:
        """
        Convert linear assignments to (row, col) grid positions.
        
        Args:
            assignments: Linear grid position assignments from Hungarian algorithm (GPU array)
            grid_size: Size of the grid
            
        Returns:
            Mapping from node index to (row, col) grid position
        """
        # Convert to host array for the final mapping dictionary
        # This is necessary because Python dicts need Python int keys/values
        assignments_cpu = cp.asnumpy(assignments)
        
        mapping = {}
        for node_idx, grid_idx in enumerate(assignments_cpu):
            if grid_idx < grid_size * grid_size:  # Valid position
                row = grid_idx // grid_size
                col = grid_idx % grid_size
                mapping[node_idx] = (int(row), int(col))
        return mapping
    
    @abstractmethod
    def generate_grid_positions(self, grid_size: int) -> cp.ndarray:
        """
        Generate grid positions for the specific grid type.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            
        Returns:
            Grid positions array (grid_size^2, 2)
        """
        pass
    
    @abstractmethod
    def build_grid_adjacency(self, node_to_grid: Dict[int, Tuple[int, int]], 
                           grid_size: int) -> Dict[int, List[int]]:
        """
        Build adjacency list for the specific grid type.
        
        Args:
            node_to_grid: Mapping from node index to grid position
            grid_size: Size of the grid
            
        Returns:
            Adjacency list dictionary
        """
        pass


class RegularGridAssigner(GridAssigner):
    """Regular rectangular grid assigner."""
    
    def generate_grid_positions(self, grid_size: int) -> cp.ndarray:
        """
        Generate regular grid positions normalized to [0, 1].
        
        Args:
            grid_size: Size of the grid
            
        Returns:
            Grid positions (grid_size^2, 2)
        """
        # Create evenly spaced grid
        x = cp.linspace(0, 1, grid_size, dtype=cp.float32)
        y = cp.linspace(0, 1, grid_size, dtype=cp.float32)
        
        # Create meshgrid without copying
        xx, yy = cp.meshgrid(x, y, copy=False)
        
        # Stack and flatten
        positions = cp.stack([xx.ravel(), yy.ravel()], axis=1)
        return positions
    
    def build_grid_adjacency(self, node_to_grid: Dict[int, Tuple[int, int]], 
                            grid_size: int) -> Dict[int, List[int]]:
        """
        Build 4-connected grid adjacency (up, down, left, right).
        
        Args:
            node_to_grid: Mapping from node index to grid position
            grid_size: Size of the grid
            
        Returns:
            Adjacency list for 4-connected grid
        """
        adjacency = {i: [] for i in node_to_grid.keys()}
        
        # Create reverse mapping for O(1) lookups
        grid_to_node = {pos: node for node, pos in node_to_grid.items()}
        
        for node_idx, (row, col) in node_to_grid.items():
            # 4-connected neighbors (up, down, left, right)
            neighbors = [
                (row - 1, col),  # up
                (row + 1, col),  # down
                (row, col - 1),  # left
                (row, col + 1)   # right
            ]
            
            for nr, nc in neighbors:
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbor_pos = (nr, nc)
                    if neighbor_pos in grid_to_node:
                        adjacency[node_idx].append(grid_to_node[neighbor_pos])
        
        return adjacency


class HexagonalGridAssigner(GridAssigner):
    """Hexagonal grid assigner with pointy-top layout."""
    
    def generate_grid_positions(self, grid_size: int) -> cp.ndarray:
        """
        Generate hexagonal grid positions (pointy-top layout).
        
        Args:
            grid_size: Size of the grid
            
        Returns:
            Hexagonal grid positions (grid_size^2, 2)
        """
        positions = []
        sqrt3 = np.sqrt(3)
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Pointy-top hexagon layout
                # Columns are spaced 1.5 units apart
                # Odd columns are offset by sqrt(3)/2 in y
                x = col * 1.5
                y = row * sqrt3 + (col % 2) * (sqrt3 / 2)
                positions.append([x, y])
        
        positions = cp.array(positions, dtype=cp.float32)
        return positions
    
    def build_grid_adjacency(self, node_to_grid: Dict[int, Tuple[int, int]], 
                            grid_size: int) -> Dict[int, List[int]]:
        """
        Build 6-connected hexagonal adjacency.
        
        Args:
            node_to_grid: Mapping from node index to grid position
            grid_size: Size of the grid
            
        Returns:
            Adjacency list for 6-connected hexagonal grid
        """
        adjacency = {i: [] for i in node_to_grid.keys()}
        
        # Create reverse mapping for O(1) lookups
        grid_to_node = {pos: node for node, pos in node_to_grid.items()}
        
        for node_idx, (row, col) in node_to_grid.items():
            # 6 neighbors for hexagonal grid (pointy-top layout)
            # The neighbor pattern depends on whether the column is even or odd
            if col % 2 == 0:  # Even column
                neighbors = [
                    (row - 1, col),      # top
                    (row - 1, col + 1),  # top-right
                    (row, col + 1),      # right
                    (row + 1, col),      # bottom
                    (row, col - 1),      # left
                    (row - 1, col - 1)   # top-left
                ]
            else:  # Odd column (offset down by sqrt(3)/2)
                neighbors = [
                    (row, col - 1),      # left
                    (row - 1, col),      # top
                    (row, col + 1),      # right
                    (row + 1, col + 1),  # bottom-right
                    (row + 1, col),      # bottom
                    (row + 1, col - 1)   # bottom-left
                ]
            
            # Add valid neighbors
            for nr, nc in neighbors:
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbor_pos = (nr, nc)
                    if neighbor_pos in grid_to_node:
                        adjacency[node_idx].append(grid_to_node[neighbor_pos])
        
        return adjacency
