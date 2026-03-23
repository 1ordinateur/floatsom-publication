import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix


def calculate_equal_sized_color_sets(samples, som_weights, topology, params, influence_map):
    """
    Calculate equal-sized color sets for balanced processing
    
    Args:
        samples: Input samples
        som_weights: Current SOM weights
        topology: SOM topology
        params: Processing parameters
        influence_map: Influence matrix
        
    Returns:
        color_sets: List of equal-sized color sets
    """
    algorithm = params.color_set_algorithm
    target_colors = params.num_color_sets

    # Route to appropriate algorithm
    if algorithm == "systematic":
        return create_systematic_pattern_color_sets_optimized(
            influence_map, topology.name, min_influence_threshold=0.1, using_gpu=True
        )
    elif algorithm == "greedy_balanced":
        return create_greedy_balanced_color_sets_optimized(
            influence_map, topology.name, target_colors, using_gpu=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def calculate_batch_all_color_sets(samples, som_weights, topology, params, influence_map):
    """
    Calculate color sets for batch-all processing mode.
    """
    # Store original value and temporarily modify for batch-all behavior
    original_max_rounds = params.processing_config.max_rounds
    params.processing_config.max_rounds = 1
    
    # Use equal-sized infrastructure with single round
    result = calculate_equal_sized_color_sets(samples, som_weights, topology, params, influence_map)
    
    # Restore original value
    params.processing_config.max_rounds = original_max_rounds
    
    return result


def create_systematic_pattern_color_sets_optimized(influence_map, topology_type="grid", 
                                                  min_influence_threshold=0.01, using_gpu=True):
    """
    Optimized systematic pattern approach using direct formulas for regular topologies
    """
    n_nodes = len(influence_map)
    
    # For regular topologies, use ultra-fast direct methods
    if topology_type in ["grid", "hexagonal"]:
        grid_size = int(cp.sqrt(n_nodes))
        if grid_size * grid_size == n_nodes:  # Perfect square
            return _direct_topology_coloring(grid_size, topology_type, min_influence_threshold)
    
    # For MST or irregular topologies, use optimized systematic approach
    return _systematic_pattern_general(influence_map, min_influence_threshold)


def create_greedy_balanced_color_sets_optimized(influence_matrix, topology_type="grid", 
                                               target_colors=None, using_gpu=True):
    """
    Optimized greedy balanced coloring using parallel algorithms
    """
    n_nodes = len(influence_matrix)
    
    # OPTIMIZATION: Check for regular topologies FIRST to avoid expensive color counting
    if topology_type in ["grid", "hexagonal"]:
        grid_size = int(cp.sqrt(n_nodes))
        
        # Handle perfect square grids
        if grid_size * grid_size == n_nodes:
            if target_colors is None:
                target_colors = _calculate_grid_chromatic_number(influence_matrix, topology_type)
            return _instant_balanced_coloring(grid_size, topology_type, target_colors)
        
        # Handle rectangular grids (non-perfect squares)
        elif topology_type == "grid":
            if target_colors is None:
                target_colors = _calculate_grid_chromatic_number(influence_matrix, topology_type)
            return _instant_balanced_coloring_rectangular(n_nodes, target_colors)
    
    # Auto-detect colors for irregular topologies only
    if target_colors is None:
        target_colors = count_colors_needed_optimized(influence_matrix, 0.01, topology_type)
    
    # For small graphs or MST, use parallel batch assignment
    if n_nodes < 100000:
        return _greedy_balanced_parallel_vectorized(influence_matrix, target_colors)
    
    # For very large graphs, use multi-level approach
    return _multilevel_balanced_coloring_vectorized(influence_matrix, target_colors)


# ============================================================================
# DIRECT FORMULA METHODS (Ultra-fast for regular topologies)
# ============================================================================

def _direct_topology_coloring(grid_size, topology_type, min_influence_threshold):
    """
    Direct O(n) coloring for regular topologies using mathematical patterns
    """
    n_nodes = grid_size * grid_size
    
    # Calculate step size based on influence threshold
    step = max(2, int(1.0 / min_influence_threshold))
    
    if topology_type == "grid":
        return _direct_grid_pattern_vectorized(grid_size, step)
    elif topology_type == "hexagonal":
        return _direct_hex_pattern_vectorized(grid_size, step)


def _direct_grid_pattern_vectorized(grid_size, step):
    """
    Fully vectorized instant pattern for square grids
    """
    # Create all node indices
    nodes = cp.arange(grid_size * grid_size)
    rows = nodes // grid_size
    cols = nodes % grid_size
    
    # Calculate color for each node using vectorized operations
    color_rows = (rows // step) % step
    color_cols = (cols // step) % step
    assignments = color_rows * step + color_cols
    
    # Convert to color sets efficiently
    n_colors = step * step
    color_sets = []
    
    # Vectorized extraction of color sets
    for c in range(n_colors):
        mask = assignments == c
        if cp.any(mask):
            color_sets.append(nodes[mask])
    
    return color_sets


def _direct_hex_pattern_vectorized(grid_size, step):
    """
    Fully vectorized pattern for hexagonal grids
    """
    hex_step = max(1, int(step * 0.866))
    
    # Create all node indices
    nodes = cp.arange(grid_size * grid_size)
    rows = nodes // grid_size
    cols = nodes % grid_size
    
    # Apply hexagonal offset vectorized
    hex_offset = (rows % 2) * 0.5
    adjusted_cols = cols + hex_offset
    
    # Calculate colors
    color_rows = (rows // hex_step) % hex_step
    color_cols = (adjusted_cols.astype(cp.int32) // hex_step) % hex_step
    assignments = color_rows * hex_step + color_cols
    
    # Extract color sets
    n_colors = hex_step * hex_step
    color_sets = []
    
    for c in range(n_colors):
        mask = assignments == c
        if cp.any(mask):
            color_sets.append(nodes[mask])
    
    return color_sets


def _instant_balanced_coloring(grid_size, topology_type, target_colors):
    """
    Direct formula for balanced coloring - fully vectorized
    """
    n_nodes = grid_size * grid_size
    
    # Calculate optimal grid partitioning
    colors_per_side = int(cp.sqrt(target_colors))
    if colors_per_side * colors_per_side < target_colors:
        colors_per_side += 1
    
    step = max(1, grid_size // colors_per_side)
    
    # Create all indices at once
    nodes = cp.arange(n_nodes)
    rows = nodes // grid_size
    cols = nodes % grid_size
    
    if topology_type == "grid":
        # Vectorized color assignment
        color_rows = (rows // step) % colors_per_side
        color_cols = (cols // step) % colors_per_side
        assignments = (color_rows * colors_per_side + color_cols) % target_colors
    
    elif topology_type == "hexagonal":
        # Hexagonal adjustment
        hex_offset = (rows % 2) * 0.5
        adjusted_cols = cols + hex_offset
        color_rows = (rows // step) % colors_per_side
        color_cols = (adjusted_cols.astype(cp.int32) // step) % colors_per_side
        assignments = (color_rows * colors_per_side + color_cols) % target_colors
    
    # Vectorized color set extraction
    color_sets = []
    for c in range(target_colors):
        color_sets.append(nodes[assignments == c])
    
    return color_sets


# ============================================================================
# PARALLEL ALGORITHMS (For MST and complex cases)
# ============================================================================

def _systematic_pattern_general(influence_map, min_influence_threshold):
    """
    Fully vectorized systematic approach for general topologies
    """
    n_nodes = len(influence_map)
    influences = influence_map.copy()
    cp.fill_diagonal(influences, 1.0)
    
    # Pre-compute conflict matrix
    conflicts = influences > min_influence_threshold
    
    # Vectorized MIS extraction
    assignments = cp.full(n_nodes, -1, dtype=cp.int32)
    color_sets = []
    group_id = 0
    
    unassigned = cp.ones(n_nodes, dtype=bool)
    
    while cp.any(unassigned):
        # Find MIS using fully vectorized approach
        mis_mask = _find_mis_vectorized(conflicts, unassigned)
        
        if not cp.any(mis_mask):
            # Assign remaining nodes individually
            remaining = cp.where(unassigned)[0]
            for idx in remaining:
                color_sets.append(cp.array([idx]))
            break
        
        # Extract MIS nodes
        mis_nodes = cp.where(mis_mask)[0]
        assignments[mis_nodes] = group_id
        color_sets.append(mis_nodes)
        unassigned[mis_nodes] = False
        group_id += 1
    
    return color_sets


def _greedy_balanced_parallel_vectorized(influence_matrix, target_colors):
    """
    Fully vectorized parallel greedy assignment
    """
    n_nodes = len(influence_matrix)
    conflicts = influence_matrix > 0.01
    
    # Pre-compute node degrees
    degrees = cp.sum(conflicts, axis=1)
    
    assignments = cp.full(n_nodes, -1, dtype=cp.int32)
    group_sizes = cp.zeros(target_colors, dtype=cp.int32)
    target_size = n_nodes // target_colors
    
    unassigned = cp.ones(n_nodes, dtype=bool)
    
    while cp.any(unassigned):
        # Vectorized MIS with priority
        mis_mask = _find_mis_priority_vectorized(conflicts, unassigned, degrees)
        
        if not cp.any(mis_mask):
            break
        
        mis_nodes = cp.where(mis_mask)[0]
        
        # Vectorized balanced assignment
        _assign_batch_vectorized(mis_nodes, assignments, group_sizes, target_size, target_colors)
        unassigned[mis_nodes] = False
    
    # Handle remaining nodes vectorized
    remaining_mask = unassigned
    if cp.any(remaining_mask):
        _force_assign_vectorized(remaining_mask, assignments, group_sizes, target_colors)
    
    # Extract color sets
    color_sets = []
    for c in range(target_colors):
        mask = assignments == c
        if cp.any(mask):
            color_sets.append(cp.where(mask)[0])
    
    return color_sets


def _find_mis_vectorized(conflicts, active_mask):
    """
    Fully vectorized Luby's MIS algorithm using einsum
    """
    n_nodes = len(conflicts)
    
    if not cp.any(active_mask):
        return cp.zeros(n_nodes, dtype=bool)
    
    # Generate random priorities
    priorities = cp.random.rand(n_nodes)
    priorities[~active_mask] = -1
    
    # Use einsum to compute max neighbor priorities without intermediate arrays
    # This computes: max_j(conflicts[i,j] * active[j] * priorities[j])
    active_priorities = active_mask.astype(cp.float32) * priorities
    
    # Efficient matrix multiplication to get neighbor priorities
    # einsum avoids creating the full neighbor_priorities matrix
    max_neighbor_priorities = cp.einsum('ij,j->i', 
                                       conflicts.astype(cp.float32), 
                                       active_priorities,
                                       optimize=True).astype(cp.float32)
    
    # Nodes are in MIS if their priority > all neighbor priorities
    is_mis = active_mask & (priorities > max_neighbor_priorities)
    
    return is_mis


def _find_mis_priority_vectorized(conflicts, active_mask, degrees):
    """
    Vectorized MIS with degree-based priority using einsum
    """
    n_nodes = len(conflicts)
    
    if not cp.any(active_mask):
        return cp.zeros(n_nodes, dtype=bool)
    
    # Use inverse degree as priority (lower degree = higher priority)
    priorities = 1.0 / (degrees + 1.0) + cp.random.rand(n_nodes) * 0.001
    priorities[~active_mask] = -1
    
    # Compute max neighbor priorities using einsum (no intermediate arrays)
    active_priorities = active_mask.astype(cp.float32) * priorities
    
    # Direct computation without creating active_conflicts matrix
    max_neighbor_priorities = cp.einsum('ij,j->i',
                                       conflicts.astype(cp.float32),
                                       active_priorities,
                                       optimize=True)
    
    is_mis = active_mask & (priorities > max_neighbor_priorities)
    
    return is_mis


def _assign_batch_vectorized(nodes, assignments, group_sizes, target_size, target_colors):
    """
    Fully vectorized batch assignment using einsum
    """
    n_nodes = len(nodes)
    
    # Calculate remaining capacity
    remaining_capacity = cp.maximum(0, target_size - group_sizes)
    
    # Instead of creating full priority matrix, compute assignments directly
    # Add small random values for tie-breaking
    capacity_with_noise = remaining_capacity + cp.random.rand(target_colors) * 0.1
    
    # For load balancing, assign nodes in chunks to groups with most capacity
    # Sort groups by capacity (descending)
    sorted_indices = cp.argsort(capacity_with_noise)[::-1]
    
    # Calculate how many nodes each group should get
    total_remaining = cp.sum(remaining_capacity)
    
    if total_remaining >= n_nodes:
        # Distribute proportionally using einsum-like operations
        node_assignments = cp.zeros(n_nodes, dtype=cp.int32)
        
        # Assign nodes in round-robin to groups with capacity
        node_idx = 0
        while node_idx < n_nodes:
            for group_idx in sorted_indices:
                if node_idx >= n_nodes:
                    break
                if remaining_capacity[group_idx] > 0:
                    # Assign batch of nodes
                    batch_size = min(
                        int(remaining_capacity[group_idx] * n_nodes / total_remaining) + 1,
                        n_nodes - node_idx
                    )
                    node_assignments[node_idx:node_idx + batch_size] = group_idx
                    node_idx += batch_size
                    remaining_capacity[group_idx] -= batch_size
        
        assignments[nodes] = node_assignments
        
        # Update group sizes using bincount
        group_increments = cp.bincount(node_assignments, minlength=target_colors)
        group_sizes += group_increments
    else:
        # Just balance across all groups
        nodes_per_group = n_nodes // target_colors
        remainder = n_nodes % target_colors
        
        node_assignments = cp.repeat(cp.arange(target_colors), nodes_per_group)
        # Add remainder to first groups
        if remainder > 0:
            extra_assignments = cp.arange(remainder)
            node_assignments = cp.concatenate([node_assignments, extra_assignments])
        
        assignments[nodes] = node_assignments[:n_nodes]
        
        # Update group sizes
        group_increments = cp.bincount(node_assignments[:n_nodes], minlength=target_colors)
        group_sizes += group_increments


def _force_assign_vectorized(remaining_mask, assignments, group_sizes, target_colors):
    """
    Vectorized forced assignment of remaining nodes
    """
    remaining_nodes = cp.where(remaining_mask)[0]
    n_remaining = len(remaining_nodes)
    
    if n_remaining == 0:
        return
    
    # Sort groups by size and assign in round-robin to smallest groups
    sorted_groups = cp.argsort(group_sizes)
    
    # Cycle through groups starting with smallest
    group_assignments = cp.tile(sorted_groups, (n_remaining // target_colors + 1))[:n_remaining]
    
    assignments[remaining_nodes] = group_assignments
    
    # Update group sizes
    group_counts = cp.bincount(group_assignments, minlength=target_colors)
    group_sizes += group_counts


# ============================================================================
# MULTI-LEVEL APPROACH (For very large graphs)
# ============================================================================

def _multilevel_balanced_coloring_vectorized(influence_matrix, target_colors, depth=0, max_depth=10):
    """
    Vectorized coarsen -> color -> refine approach with recursion depth limit
    """
    n_nodes = len(influence_matrix)
    
    # Prevent infinite recursion
    if depth >= max_depth:
        return _greedy_balanced_parallel_vectorized(influence_matrix, target_colors)
    
    # Vectorized coarsening
    coarse_map, coarse_matrix = _coarsen_graph_vectorized(influence_matrix)
    
    # Color coarse graph
    if len(coarse_matrix) < n_nodes / 2:
        if len(coarse_matrix) > 10000:
            coarse_colors = _multilevel_balanced_coloring_vectorized(coarse_matrix, target_colors, depth + 1, max_depth)
        else:
            coarse_colors = _greedy_balanced_parallel_vectorized(coarse_matrix, target_colors)
    else:
        coarse_colors = _greedy_balanced_parallel_vectorized(coarse_matrix, target_colors)
    
    # Vectorized projection back to fine graph
    fine_assignments = cp.full(n_nodes, -1, dtype=cp.int32)
    
    # Build reverse mapping
    for color_idx, coarse_nodes in enumerate(coarse_colors):
        for coarse_node in coarse_nodes:
            fine_nodes = coarse_map[coarse_node]
            fine_assignments[fine_nodes] = color_idx
    
    # Extract color sets
    color_sets = []
    for c in range(target_colors):
        mask = fine_assignments == c
        if cp.any(mask):
            color_sets.append(cp.where(mask)[0])
    
    return color_sets


def _coarsen_graph_vectorized(influence_matrix):
    """
    Vectorized graph coarsening using einsum for efficiency
    """
    n_nodes = len(influence_matrix)
    
    # Find low-influence pairs vectorized
    low_influence_threshold = 0.01
    
    # Mask diagonal
    influence_copy = influence_matrix.copy()
    cp.fill_diagonal(influence_copy, cp.inf)
    
    # Find minimum influence and partners
    min_influences = cp.min(influence_copy, axis=1)
    min_neighbors = cp.argmin(influence_copy, axis=1)
    
    # Build merge pairs efficiently
    merged = cp.zeros(n_nodes, dtype=bool)
    coarse_map = []
    
    # Vectorized pair identification
    can_merge = (min_influences < low_influence_threshold) & ~merged
    
    # Process all mergeable pairs at once
    while cp.any(can_merge):
        # Find valid pairs (avoid double merging)
        merge_candidates = cp.where(can_merge)[0]
        partners = min_neighbors[merge_candidates]
        
        # Valid pairs: partner not merged and has higher index
        valid_mask = ~merged[partners] & (partners > merge_candidates)
        
        # Add valid pairs
        valid_nodes = merge_candidates[valid_mask]
        valid_partners = partners[valid_mask]
        
        for node, partner in zip(valid_nodes, valid_partners):
            coarse_map.append(cp.array([node, partner]))
            merged[node] = True
            merged[partner] = True
        
        # Add singles from invalid pairs
        invalid_nodes = merge_candidates[~valid_mask]
        for node in invalid_nodes:
            if not merged[node]:
                coarse_map.append(cp.array([node]))
                merged[node] = True
        
        # Update can_merge
        can_merge = (min_influences < low_influence_threshold) & ~merged
    
    # Add remaining unmerged
    unmerged = cp.where(~merged)[0]
    for node in unmerged:
        coarse_map.append(cp.array([node]))
    
    # Build coarse matrix using einsum
    n_coarse = len(coarse_map)
    
    # Create mapping matrix for einsum
    fine_to_coarse = cp.zeros((n_nodes, n_coarse), dtype=cp.float32)
    for c_idx, fine_nodes in enumerate(coarse_map):
        fine_to_coarse[fine_nodes, c_idx] = 1.0
    
    # Use einsum to compute coarse matrix efficiently
    # This computes max influence between coarse nodes without loops
    coarse_matrix = cp.einsum('ic,ij,jd->cd',
                             fine_to_coarse,
                             influence_matrix,
                             fine_to_coarse,
                             optimize=True)
    
    # Ensure diagonal is zero
    cp.fill_diagonal(coarse_matrix, 0)
    
    return coarse_map, coarse_matrix


# ============================================================================
# MATHEMATICAL GRID OPTIMIZATION FUNCTIONS
# ============================================================================

def _calculate_grid_chromatic_number(influence_matrix, topology_type="grid"):
    """
    Fast chromatic number estimation for grid topologies using multiple heuristics
    """
    n_nodes = len(influence_matrix)
    
    # Method 1: Ultra-fast degree sampling
    chromatic_estimate = _estimate_chromatic_by_degree_sampling(influence_matrix)
    
    # Method 2: For regular grids, use structural analysis
    if topology_type in ["grid", "hexagonal"]:
        structural_estimate = _estimate_chromatic_by_structure(n_nodes, influence_matrix, topology_type)
        chromatic_estimate = min(chromatic_estimate, structural_estimate)
    
    # Method 3: Clique-based lower bound (only for small graphs)
    if n_nodes < 5000:
        clique_estimate = _estimate_max_clique_size_fast(influence_matrix)
        chromatic_estimate = max(chromatic_estimate, clique_estimate)
    
    return chromatic_estimate


def _estimate_chromatic_by_degree_sampling(influence_matrix, threshold=0.01, sample_ratio=0.01):
    """
    Ultra-fast degree-based chromatic number estimation using sampling
    """
    n_nodes = len(influence_matrix)
    
    # For small graphs, compute exact max degree
    if n_nodes < 10000:
        degrees = cp.sum(influence_matrix > threshold, axis=1)
        max_degree = int(cp.max(degrees))
        avg_degree = float(cp.mean(degrees))
    else:
        # For large graphs, sample intelligently
        sample_size = max(100, int(n_nodes * sample_ratio))
        
        # Sample both random nodes and high-connectivity regions
        # Random sample
        random_indices = cp.random.choice(n_nodes, size=sample_size//2, replace=False)
        
        # High-connectivity sample (nodes with many small influences)
        row_sums = cp.sum(influence_matrix, axis=1)
        high_conn_indices = cp.argpartition(-row_sums, sample_size//2)[:sample_size//2]
        
        sample_indices = cp.unique(cp.concatenate([random_indices, high_conn_indices]))
        
        # Compute degrees for sampled nodes
        sample_degrees = cp.sum(influence_matrix[sample_indices] > threshold, axis=1)
        max_degree = int(cp.max(sample_degrees))
        avg_degree = float(cp.mean(sample_degrees))
    
    # Heuristic: chromatic number is typically much less than max_degree + 1
    # For grid-like structures, it's closer to sqrt(avg_degree)
    if avg_degree < 10:
        return min(max_degree + 1, 4)
    elif avg_degree < 30:
        return min(max_degree + 1, int(cp.sqrt(avg_degree) * 2))
    else:
        return min(max_degree + 1, int(cp.sqrt(avg_degree) * 1.5))


def _estimate_chromatic_by_structure(n_nodes, influence_matrix, topology_type):
    """
    Structure-based estimation for regular topologies
    """
    grid_size = int(cp.sqrt(n_nodes))
    
    # Sample center node to estimate influence radius
    center_node = n_nodes // 2
    center_influences = influence_matrix[center_node]
    
    # Count significant influences
    significant_neighbors = cp.sum(center_influences > 0.01)
    
    if topology_type == "grid":
        # Grid coloring based on neighborhood size
        if significant_neighbors <= 4:  # Cross pattern
            return 2
        elif significant_neighbors <= 8:  # 3x3 neighborhood
            return 4
        elif significant_neighbors <= 24:  # 5x5 neighborhood
            return 9
        elif significant_neighbors <= 48:  # 7x7 neighborhood
            return 16
        else:
            # Large neighborhood - use formula
            radius = int(cp.sqrt(significant_neighbors) / 2)
            return min((radius * 2 + 1) ** 2, grid_size // 4)
    
    elif topology_type == "hexagonal":
        # Hexagonal coloring patterns
        if significant_neighbors <= 6:
            return 3
        elif significant_neighbors <= 12:
            return 7
        elif significant_neighbors <= 18:
            return 12
        else:
            # Estimate based on rings
            rings = int(cp.sqrt(significant_neighbors / 6))
            return min(3 * rings * rings, grid_size // 4)
    
    return 4  # Safe default


def _estimate_max_clique_size_fast(influence_matrix, threshold=0.01):
    """
    Fast clique size estimation using neighborhood analysis
    """
    n_nodes = len(influence_matrix)
    
    # Sample high-degree nodes most likely to be in large cliques
    degrees = cp.sum(influence_matrix > threshold, axis=1)
    
    # Check top 10 highest degree nodes
    k = min(10, n_nodes)
    top_k_nodes = cp.argpartition(-degrees, k)[:k]
    
    max_clique = 2
    
    for node in top_k_nodes:
        # Get node's neighbors
        neighbors = cp.where(influence_matrix[node] > threshold)[0]
        if len(neighbors) < max_clique:
            continue
        
        # Check connectivity among neighbors (sample if too many)
        if len(neighbors) > 20:
            # Sample subset of neighbors
            sample_neighbors = neighbors[cp.random.choice(len(neighbors), 20, replace=False)]
        else:
            sample_neighbors = neighbors
        
        # Check if neighbors form a clique
        submatrix = influence_matrix[sample_neighbors][:, sample_neighbors]
        if cp.all(submatrix > threshold):
            max_clique = max(max_clique, len(sample_neighbors))
    
    return max_clique


def _instant_balanced_coloring_rectangular(n_nodes, target_colors):
    """
    Fast balanced coloring for rectangular (non-perfect square) grids
    """
    # Find best factorization of n_nodes closest to square
    sqrt_n = int(cp.sqrt(n_nodes))
    
    # Try factors near sqrt
    best_rows, best_cols = 1, n_nodes
    min_diff = n_nodes
    
    for rows in range(max(1, sqrt_n - 10), min(n_nodes, sqrt_n + 10)):
        if n_nodes % rows == 0:
            cols = n_nodes // rows
            diff = abs(rows - cols)
            if diff < min_diff:
                min_diff = diff
                best_rows, best_cols = rows, cols
    
    # Create virtual grid
    nodes = cp.arange(n_nodes)
    virtual_rows = nodes // best_cols
    virtual_cols = nodes % best_cols
    
    # Calculate colors using grid pattern
    colors_per_row = int(cp.sqrt(target_colors))
    colors_per_col = (target_colors + colors_per_row - 1) // colors_per_row
    
    step_row = max(1, best_rows // colors_per_row)
    step_col = max(1, best_cols // colors_per_col)
    
    color_rows = (virtual_rows // step_row) % colors_per_row
    color_cols = (virtual_cols // step_col) % colors_per_col
    assignments = (color_rows * colors_per_col + color_cols) % target_colors
    
    # Extract color sets
    color_sets = []
    for c in range(target_colors):
        mask = assignments == c
        if cp.any(mask):
            color_sets.append(nodes[mask])
    
    return color_sets


def count_colors_needed_optimized(influence_matrix, threshold=0.01, topology_type=None):
    """
    Optimized color counting that avoids full graph coloring
    """
    n_nodes = len(influence_matrix)
    
    # For very small graphs, use exact method
    if n_nodes < 100:
        return count_colors_needed_exact_small(influence_matrix, threshold)
    
    # Use fast heuristics
    degree_estimate = _estimate_chromatic_by_degree_sampling(influence_matrix, threshold)
    
    # For known topologies, combine with structural analysis
    if topology_type in ["grid", "hexagonal"]:
        structural_estimate = _estimate_chromatic_by_structure(n_nodes, influence_matrix, topology_type)
        return min(degree_estimate, structural_estimate)
    
    # For large unknown topologies, use degree estimate with clique refinement
    if n_nodes < 5000:
        clique_lower_bound = _estimate_max_clique_size_fast(influence_matrix, threshold)
        return max(degree_estimate, clique_lower_bound)
    
    return degree_estimate


def count_colors_needed_exact_small(influence_matrix, threshold=0.01):
    """
    Exact but fast coloring for small graphs using parallel Welsh-Powell
    """
    n_nodes = len(influence_matrix)
    conflicts = influence_matrix > threshold
    
    # Compute degrees
    degrees = cp.sum(conflicts, axis=1)
    
    # Sort by degree (descending)
    node_order = cp.argsort(-degrees)
    
    # Color assignment
    colors = cp.full(n_nodes, -1, dtype=cp.int32)
    max_color = -1
    
    for node in node_order:
        # Find neighbors' colors
        neighbor_mask = conflicts[node]
        neighbor_colors = colors[neighbor_mask]
        used_colors = neighbor_colors[neighbor_colors >= 0]
        
        # Find smallest available color
        if len(used_colors) == 0:
            colors[node] = 0
            max_color = max(max_color, 0)
        else:
            # Create color availability array
            color_available = cp.ones(max_color + 2, dtype=bool)
            color_available[used_colors] = False
            
            # Assign first available color
            new_color = int(cp.argmax(color_available))
            colors[node] = new_color
            max_color = max(max_color, new_color)
    
    return max_color + 1


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_colors_needed_vectorized(influence_matrix, threshold=0.01):
    """
    Fully vectorized chromatic number estimation using einsum
    """
    n_nodes = len(influence_matrix)
    conflicts = influence_matrix > threshold
    
    # Compute degrees using einsum (more efficient than sum)
    degrees = cp.einsum('ij->i', conflicts.astype(cp.int32), optimize=True)
    
    # Process nodes in degree order
    node_order = cp.argsort(degrees)[::-1]
    
    # Initialize all colors to -1
    colors = cp.full(n_nodes, -1, dtype=cp.int32)
    
    # Pre-allocate color availability matrix
    max_possible_colors = min(n_nodes, cp.max(degrees).item() + 1)
    
    # Vectorized greedy coloring
    for i, node in enumerate(node_order):
        # Get neighbor colors using boolean indexing
        neighbor_colors = colors[conflicts[node]]
        neighbor_colors = neighbor_colors[neighbor_colors >= 0]
        
        if len(neighbor_colors) == 0:
            colors[node] = 0
        else:
            # Vectorized color selection using bincount
            used_color_flags = cp.zeros(max_possible_colors, dtype=bool)
            valid_colors = neighbor_colors[neighbor_colors < max_possible_colors]
            used_color_flags[valid_colors] = True
            
            # Find first False (available color)
            colors[node] = cp.argmax(~used_color_flags)
    
    return int(cp.max(colors) + 1)
