"""
Shared utility functions for topology implementations
"""

import numpy as np
import cupy as cp
from typing import Any, Dict, Iterator, List, Tuple, Union


class DisjointSetUnion:
    """
    Union-find data structure with path compression and union by rank.

    Used by graph topologies to keep connectivity logic centralized.
    """

    def __init__(self, size: int):
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        self.parent = list(range(size))
        self.rank = [0] * size
        self.component_count = size

    def find(self, node: int) -> int:
        """Find the root of `node` with path compression."""
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, node_a: int, node_b: int) -> bool:
        """Union two sets. Returns True only when a merge happened."""
        root_a = self.find(node_a)
        root_b = self.find(node_b)
        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        self.component_count -= 1
        return True


def iter_chunk_ranges(total_size: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """Yield half-open [start, stop) chunk ranges over `total_size`."""
    if total_size < 0:
        raise ValueError(f"total_size must be non-negative, got {total_size}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    for start in range(0, total_size, chunk_size):
        yield start, min(start + chunk_size, total_size)


def calculate_euclidean_distance(pos1, pos2, toroidal: bool = False, grid_size: int = None):
    """
    Calculate Euclidean distance between two positions
    
    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)
        toroidal: Whether to use toroidal (wrap-around) distance
        grid_size: Grid size for toroidal calculations
        
    Returns:
        Euclidean distance
    """
    if toroidal and grid_size is not None:
        # Calculate toroidal distance
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        dx = min(dx, grid_size - dx)
        dy = min(dy, grid_size - dy)
        return (dx ** 2 + dy ** 2) ** 0.5
    else:
        # Calculate regular Euclidean distance
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx ** 2 + dy ** 2) ** 0.5


def create_distance_matrix(positions, toroidal: bool = False, grid_size: int = None):
    """
    Create distance matrix for array of positions
    
    Args:
        positions: Array of (row, col) positions
        toroidal: Whether to use toroidal distances
        grid_size: Grid size for toroidal calculations
        
    Returns:
        Distance matrix
    """
    n_positions = len(positions)
    xp = cp if isinstance(positions, cp.ndarray) else np
    
    # Create distance matrix
    distance_matrix = xp.zeros((n_positions, n_positions), dtype=xp.float32)
    
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            dist = calculate_euclidean_distance(positions[i], positions[j], 
                                               toroidal=toroidal, grid_size=grid_size)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    
    return distance_matrix


def get_circular_neighbors(center, radius, shape, toroidal: bool = False):
    """
    Get neighbors within circular radius
    
    Args:
        center: Center position (row, col)
        radius: Circular radius
        shape: Grid shape (height, width)
        toroidal: Whether to use toroidal wrapping
        
    Returns:
        List of neighbor positions
    """
    center_row, center_col = center
    height, width = shape
    neighbors = []
    
    # Search in a square region around the center
    search_radius = int(radius) + 1
    
    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            row = center_row + dr
            col = center_col + dc
            
            if toroidal:
                # Apply toroidal wrapping
                row, col = apply_toroidal_wrap((row, col), shape)
                # Calculate toroidal distance
                distance = calculate_euclidean_distance((center_row, center_col), (row, col), 
                                                      toroidal=True, grid_size=min(height, width))
            else:
                # Check bounds
                if row < 0 or row >= height or col < 0 or col >= width:
                    continue
                # Calculate regular distance
                distance = calculate_euclidean_distance((center_row, center_col), (row, col))
            
            if distance <= radius:
                neighbors.append((row, col))
    
    return neighbors


def get_manhattan_neighbors(center, radius, shape, toroidal: bool = False):
    """
    Get neighbors within Manhattan distance
    
    Args:
        center: Center position (row, col)
        radius: Manhattan radius
        shape: Grid shape (height, width)
        toroidal: Whether to use toroidal wrapping
        
    Returns:
        List of neighbor positions
    """
    center_row, center_col = center
    height, width = shape
    neighbors = []
    
    # Manhattan distance uses L1 norm
    search_radius = int(radius)
    
    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            # Calculate Manhattan distance
            manhattan_dist = abs(dr) + abs(dc)
            
            if manhattan_dist <= radius:
                row = center_row + dr
                col = center_col + dc
                
                if toroidal:
                    # Apply toroidal wrapping
                    row, col = apply_toroidal_wrap((row, col), shape)
                    neighbors.append((row, col))
                else:
                    # Check bounds
                    if 0 <= row < height and 0 <= col < width:
                        neighbors.append((row, col))
    
    return neighbors


def apply_toroidal_wrap(position, shape):
    """
    Apply toroidal wrapping to position
    
    Args:
        position: Position (row, col)
        shape: Grid shape (height, width)
        
    Returns:
        Wrapped position
    """
    row, col = position
    height, width = shape
    
    # Apply modulo wrapping
    wrapped_row = row % height
    wrapped_col = col % width
    
    return (wrapped_row, wrapped_col)


def validate_topology_shape(shape, topology_type):
    """
    Validate shape for topology type
    
    Args:
        shape: Proposed shape (height, width) or (grid_size,)
        topology_type: Type of topology ('grid', 'hexagonal', 'mst', 'rng')
        
    Returns:
        bool: True if valid
    """
    if topology_type in ['grid', 'rectangular']:
        # Grid topology supports square and rectangular shapes
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            height, width = shape
            return height > 0 and width > 0
        elif isinstance(shape, (int, float)):
            return shape > 0  # Square grid
        return False
    
    elif topology_type == 'hexagonal':
        # Hexagonal topology typically uses square grids
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            height, width = shape
            return height > 0 and width > 0
        elif isinstance(shape, (int, float)):
            return shape > 0  # Square hexagonal grid
        return False
    
    elif topology_type in {'mst', 'rng'}:
        # Graph topologies just need number of nodes
        if isinstance(shape, (int, float)):
            return shape > 1  # Need at least 2 nodes for graph topology
        return False
    
    # Unknown topology type
    return False


def interpolate_weights_linear(start_weights, end_weights, positions, shape, alpha: float = 0.5):
    """
    Linear interpolation of weights across topology
    
    Args:
        start_weights: Starting weight values (array)
        end_weights: Ending weight values (array)
        positions: Positions for interpolation (not used in linear interpolation)
        shape: Topology shape (not used in linear interpolation)
        alpha: Interpolation factor (0.0 = start_weights, 1.0 = end_weights)
        
    Returns:
        Interpolated weights
    """
    xp = cp if isinstance(start_weights, cp.ndarray) else np
    
    # Simple linear interpolation: result = (1-alpha) * start + alpha * end
    return (1 - alpha) * start_weights + alpha * end_weights


def deduplicate_radii_list(radii_list: List[float], radius_threshold: float = 0.1) -> Tuple[List[float], Dict[float, float]]:
    """
    Deduplicate radii list by combining radii within threshold percentage.
    
    Args:
        radii_list: List of radius values to deduplicate
        radius_threshold: Percentage threshold (0.1 = 10%)
        
    Returns:
        unique_radii: List of radii to actually precompute
        radius_mapping: Maps original_radius -> precomputed_radius
    """
    unique_radii = []
    radius_mapping = {}
    
    for radius in sorted(radii_list):
        # Find if any existing unique radius is within threshold
        found_match = False
        for unique_radius in unique_radii:
            if abs(radius - unique_radius) / unique_radius <= radius_threshold:
                # Use existing radius
                radius_mapping[round(float(radius), 3)] = round(float(unique_radius), 3)
                found_match = True
                break
        
        if not found_match:
            # New unique radius
            unique_radii.append(radius)
            radius_mapping[round(float(radius), 3)] = round(float(radius), 3)
    
    return unique_radii, radius_mapping


def calculate_gaussian_influence(coord_grid: Union[cp.ndarray, np.ndarray], 
                                radius: float, 
                                distance_matrix: Union[cp.ndarray, np.ndarray] = None,
                                toroidal: bool = False,
                                grid_size: int = None,
                                std_coeff: float = 0.5,
                                compact_support: bool = False) -> Union[cp.ndarray, np.ndarray]:
    """
    Calculate Gaussian influence matrix for weight updates with in-place operations.
    
    Args:
        coord_grid: Coordinate grid for topology
        radius: Current neighborhood radius
        distance_matrix: Precomputed distance matrix (optional, preferred)
        toroidal: Whether to use toroidal (wrap-around) distances
        grid_size: Grid size for toroidal calculations (required if toroidal=True and distance_matrix=None)
        std_coeff: Sigma coefficient (sigma = radius * std_coeff)
        compact_support: Whether to zero influence beyond radius
        
    Returns:
        influence_matrix: Gaussian influence matrix
    """
    xp = cp if isinstance(coord_grid, cp.ndarray) else np
    radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
    std_coeff_scalar = float(std_coeff)
    if std_coeff_scalar <= 0:
        raise ValueError(f"std_coeff must be positive, got {std_coeff_scalar}")
    
    if distance_matrix is not None:
        # Use precomputed distances with in-place operations
        effective_sigma = (radius_scalar * std_coeff_scalar) ** 2
        # Make a copy since we'll modify it (distance_matrix might be reused)
        influence = distance_matrix.copy()
        influence /= -(2 * effective_sigma)  # In-place division and negation
        xp.exp(influence, out=influence)  # In-place exponential
        if compact_support:
            mask = distance_matrix <= radius_scalar ** 2
            influence *= mask
        return influence
    else:
        # Calculate distances on-demand with memory-efficient operations
        total_nodes = len(coord_grid)
        coords_expanded1 = coord_grid[:, None, :]  # Shape: (total_nodes, 1, coord_dim)
        coords_expanded2 = coord_grid[None, :, :]  # Shape: (1, total_nodes, coord_dim)
        
        if toroidal and grid_size is not None:
            # Calculate toroidal distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            xp.abs(diff, out=diff)  # In-place absolute value
            wrap_diff = grid_size - diff
            xp.minimum(diff, wrap_diff, out=diff)  # In-place minimum (reuse diff)
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        else:
            # Calculate regular Euclidean distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        
        # Convert squared distances to Gaussian influence in-place
        effective_sigma = (radius_scalar * std_coeff_scalar) ** 2
        if compact_support:
            mask = sq_dist <= radius_scalar ** 2
        sq_dist /= -(2 * effective_sigma)  # In-place division and negation
        xp.exp(sq_dist, out=sq_dist)  # In-place exponential
        if compact_support:
            sq_dist *= mask
        return sq_dist


def calculate_bubble_influence(coord_grid: Union[cp.ndarray, np.ndarray], 
                              radius: float, 
                              distance_matrix: Union[cp.ndarray, np.ndarray] = None,
                              toroidal: bool = False,
                              grid_size: int = None) -> Union[cp.ndarray, np.ndarray]:
    """
    Calculate bubble (step) influence matrix for weight updates with in-place operations.
    
    Args:
        coord_grid: Coordinate grid for topology
        radius: Current neighborhood radius
        distance_matrix: Precomputed distance matrix (optional, preferred)
        toroidal: Whether to use toroidal (wrap-around) distances
        grid_size: Grid size for toroidal calculations (required if toroidal=True and distance_matrix=None)
        
    Returns:
        influence_matrix: Bubble influence matrix
    """
    xp = cp if isinstance(coord_grid, cp.ndarray) else np
    
    if distance_matrix is not None:
        # Use precomputed distances (already handles toroidal if computed that way)
        # Ensure radius is a scalar
        radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
        return (distance_matrix <= radius_scalar ** 2).astype(xp.float32)
    else:
        # Calculate distances on-demand with in-place operations
        total_nodes = len(coord_grid)
        coords_expanded1 = coord_grid[:, None, :]
        coords_expanded2 = coord_grid[None, :, :]
        
        if toroidal and grid_size is not None:
            # Calculate toroidal distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            xp.abs(diff, out=diff)  # In-place absolute value
            wrap_diff = grid_size - diff
            xp.minimum(diff, wrap_diff, out=diff)  # In-place minimum (reuse diff)
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        else:
            # Calculate regular Euclidean distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        
        # Ensure radius is a scalar
        radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
        return (sq_dist <= radius_scalar ** 2).astype(xp.float32)


def calculate_mexican_hat_influence(coord_grid: Union[cp.ndarray, np.ndarray], 
                                   radius: float, 
                                   distance_matrix: Union[cp.ndarray, np.ndarray] = None,
                                   toroidal: bool = False,
                                   grid_size: int = None,
                                   std_coeff: float = 0.5,
                                   compact_support: bool = False) -> Union[cp.ndarray, np.ndarray]:
    """
    Calculate Mexican hat influence matrix for weight updates with in-place operations.
    
    Args:
        coord_grid: Coordinate grid for topology
        radius: Current neighborhood radius
        distance_matrix: Precomputed distance matrix (optional, preferred)
        toroidal: Whether to use toroidal (wrap-around) distances
        grid_size: Grid size for toroidal calculations (required if toroidal=True and distance_matrix=None)
        std_coeff: Sigma coefficient (sigma = radius * std_coeff)
        compact_support: Whether to zero influence beyond radius
        
    Returns:
        influence_matrix: Mexican hat influence matrix
    """
    xp = cp if isinstance(coord_grid, cp.ndarray) else np
    radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
    std_coeff_scalar = float(std_coeff)
    if std_coeff_scalar <= 0:
        raise ValueError(f"std_coeff must be positive, got {std_coeff_scalar}")
    
    if distance_matrix is not None:
        # Use precomputed distances - make a copy to avoid modifying the original
        # We'll reuse this array for all operations
        result = xp.sqrt(distance_matrix.copy())
    else:
        # Calculate distances on-demand with in-place operations
        total_nodes = len(coord_grid)
        coords_expanded1 = coord_grid[:, None, :]
        coords_expanded2 = coord_grid[None, :, :]
        
        if toroidal and grid_size is not None:
            # Calculate toroidal distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            xp.abs(diff, out=diff)  # In-place absolute value
            wrap_diff = grid_size - diff
            xp.minimum(diff, wrap_diff, out=diff)  # In-place minimum (reuse diff)
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        else:
            # Calculate regular Euclidean distances with in-place operations
            diff = coords_expanded1 - coords_expanded2
            diff **= 2  # In-place square
            sq_dist = xp.sum(diff, axis=2)
        
        result = xp.sqrt(sq_dist, out=sq_dist)  # In-place sqrt, reuse sq_dist array
    
    # Mexican hat: (1 - (d/sigma)^2) * exp(-(d/sigma)^2 / 2)
    # Optimize by reusing arrays and doing operations in-place
    sigma = radius_scalar * std_coeff_scalar
    if compact_support:
        mask = result <= radius_scalar
    
    # Reuse 'result' array for all operations
    result /= sigma  # In-place normalize: result = d/sigma
    result **= 2     # In-place square: result = (d/sigma)^2
    
    # Now result contains normalized_sq
    # Compute exp part: exp(-normalized_sq / 2)
    exp_part = result.copy()  # Need a copy for the exp calculation
    exp_part /= -2  # In-place: exp_part = -normalized_sq / 2
    xp.exp(exp_part, out=exp_part)  # In-place exponential
    
    # Compute (1 - normalized_sq) and multiply with exp_part
    result *= -1  # In-place: result = -normalized_sq
    result += 1   # In-place: result = 1 - normalized_sq
    result *= exp_part  # In-place: result = (1 - normalized_sq) * exp(-normalized_sq / 2)
    if compact_support:
        result *= mask
    
    return result


def build_sparse_influence_from_distance_matrix(
    distance_matrix: Union[cp.ndarray, np.ndarray],
    radius: float,
    influence_function: str,
    *,
    std_coeff: float = 0.5,
    compact_support: bool = False,
    distance_is_squared: bool = True,
) -> Dict[str, Any]:
    """
    Build a CSR influence payload from a (possibly squared) distance matrix.

    Returns a dict with keys: format, indptr, indices, data, shape.
    """
    xp = cp if isinstance(distance_matrix, cp.ndarray) else np
    radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
    std_coeff_scalar = float(std_coeff)
    if std_coeff_scalar <= 0:
        raise ValueError(f"std_coeff must be positive, got {std_coeff_scalar}")

    influence_function = str(influence_function or "").lower()
    if influence_function not in {"gaussian", "bubble", "mexican_hat"}:
        raise ValueError(f"Unsupported influence_function for sparse map: {influence_function}")
    if influence_function in {"gaussian", "mexican_hat"} and not compact_support:
        raise ValueError(
            "Sparse influence requires compact_support=True for gaussian or mexican_hat."
        )

    if distance_is_squared:
        mask = distance_matrix <= radius_scalar ** 2
    else:
        mask = distance_matrix <= radius_scalar

    row_idx, col_idx = xp.where(mask)
    if row_idx.size == 0:
        n_nodes = int(distance_matrix.shape[0])
        indptr = xp.zeros(n_nodes + 1, dtype=xp.int32)
        indices = xp.zeros(0, dtype=xp.int32)
        data = xp.zeros(0, dtype=xp.float32)
        return {
            "format": "csr",
            "indptr": indptr,
            "indices": indices,
            "data": data,
            "shape": (n_nodes, n_nodes),
        }

    if influence_function == "bubble":
        data = xp.ones(row_idx.shape[0], dtype=xp.float32)
    else:
        sigma = radius_scalar * std_coeff_scalar
        if distance_is_squared:
            dist_sq = distance_matrix[mask]
            normalized_sq = dist_sq / (sigma ** 2)
        else:
            dist = distance_matrix[mask]
            normalized = dist / sigma
            normalized_sq = normalized ** 2
        if influence_function == "gaussian":
            data = xp.exp(-normalized_sq / 2)
        else:
            data = (1 - normalized_sq) * xp.exp(-normalized_sq / 2)
        data = data.astype(xp.float32, copy=False)

    row_idx = row_idx.astype(xp.int32, copy=False)
    col_idx = col_idx.astype(xp.int32, copy=False)
    counts = xp.bincount(row_idx, minlength=distance_matrix.shape[0]).astype(xp.int32, copy=False)
    indptr = xp.zeros(int(distance_matrix.shape[0]) + 1, dtype=xp.int32)
    xp.cumsum(counts, out=indptr[1:])

    return {
        "format": "csr",
        "indptr": indptr,
        "indices": col_idx,
        "data": data,
        "shape": tuple(distance_matrix.shape),
    }
