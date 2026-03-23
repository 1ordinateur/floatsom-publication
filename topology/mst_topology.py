"""
Minimum Spanning Tree (MST) topology for FloatSOM
"""

import cupy as cp
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

from .som_topology import SOMTopology
from .pca_initialization import pca_weights_init, pca_sampling_init, pca_sampling_init_snake, pca_density_init
from .utils import DisjointSetUnion, iter_chunk_ranges

_DEFAULT_FW_ROW_CHUNK_SIZE = 256
_FW_ROW_TARGET_BYTES = 64 * 1024 * 1024
_FW_ROW_TEMP_MULTIPLIER = 2


def _resolve_fw_row_chunk_size(
    num_nodes: int,
    item_size: int,
    requested_row_chunk_size: Optional[int] = None,
    target_bytes: Optional[int] = None,
    temp_multiplier: int = _FW_ROW_TEMP_MULTIPLIER,
) -> int:
    """Pick a Floyd-Warshall row chunk size that bounds temporary GPU memory."""
    if requested_row_chunk_size is not None:
        return max(1, min(num_nodes, int(requested_row_chunk_size)))

    effective_target_bytes = _FW_ROW_TARGET_BYTES if target_bytes is None else int(target_bytes)
    effective_target_bytes = max(1, effective_target_bytes)

    effective_multiplier = max(1, int(temp_multiplier))
    bytes_per_row = max(1, num_nodes * item_size * effective_multiplier)
    auto_chunk = effective_target_bytes // bytes_per_row
    auto_chunk = max(1, int(auto_chunk))
    return max(1, min(num_nodes, min(_DEFAULT_FW_ROW_CHUNK_SIZE, auto_chunk)))


def _run_chunked_floyd_warshall_gpu(graph_distances: cp.ndarray, row_chunk_size: int) -> None:
    """Run in-place Floyd-Warshall with bounded row-tiled temporaries on GPU."""
    num_nodes = int(graph_distances.shape[0])
    if num_nodes <= 1:
        return

    for k in range(num_nodes):
        kth_row = graph_distances[k:k + 1, :]
        for start_idx, end_idx in iter_chunk_ranges(num_nodes, row_chunk_size):
            row_chunk = graph_distances[start_idx:end_idx, :]
            dist_through_k = graph_distances[start_idx:end_idx, k:k + 1] + kth_row
            cp.minimum(row_chunk, dist_through_k, out=row_chunk)

        if k % 100 == 0 and k > 0:
            cp.get_default_memory_pool().free_all_blocks()


def calculate_mst_cpu(distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Calculate MST using CPU Kruskal's algorithm (optimized for SOM sizes)
    
    Args:
        distance_matrix: Square distance matrix between all nodes
        
    Returns:
        edges: List of (source, dest) edge tuples in the MST
    """
    n = int(distance_matrix.shape[0])
    if n <= 1:
        return []

    dsu = DisjointSetUnion(n)
    
    mst_edges = []
    
    # Create list of all edges with weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance_matrix[i, j], i, j))
    
    # Sort edges by weight
    edges.sort()
    
    # Kruskal's algorithm
    for _, u, v in edges:
        if dsu.union(u, v):
            mst_edges.append((u, v))
            if len(mst_edges) == n - 1:
                break
    
    return mst_edges


def build_adjacency_list(edges: List[Tuple[int, int]], num_nodes: int) -> Dict[int, List[int]]:
    """
    Build adjacency list from MST edges
    
    Args:
        edges: List of MST edges
        num_nodes: Total number of nodes
        
    Returns:
        adjacency_list: Dictionary mapping node -> list of neighbors
    """
    adjacency_list = {i: [] for i in range(num_nodes)}
    
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    return adjacency_list


class MSTTopology(SOMTopology):
    """
    Minimum Spanning Tree topology
    Creates topology based on data structure rather than fixed grid
    """
    
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int,
                 mst_update_frequency: int = 50,
                 initialization_method: str = "random",
                 influence_function: str = "gaussian",
                 seed: Optional[int] = None,
                 verbose: bool = False,
                 mst_params=None,
                 dynamic_mst_frequency: bool = True,
                 mst_decay_function: str = "exponential",
                 initial_mst_frequency: int = 1,
                 final_mst_frequency: int = 10,
                 total_iterations: Optional[int] = None):
        """
        Initialize MST topology
        
        Args:
            num_nodes: Number of nodes in the SOM
            input_dim: Dimensionality of input data
            mst_update_frequency: Frequency of MST recalculation (used when dynamic_mst_frequency=False)
            initialization_method: Method to initialize weights
            influence_function: Type of influence function to use ('gaussian', 'bubble', 'mexican_hat')
            seed: Random seed for reproducibility
            verbose: Whether to print verbose output
            mst_params: MST-specific parameters (compatibility)
            dynamic_mst_frequency: Whether to use dynamic MST update frequency
            mst_decay_function: Type of decay function for dynamic frequency
            initial_mst_frequency: Initial update frequency (update every N iterations)
            final_mst_frequency: Final update frequency (update every N iterations)
            total_iterations: Total training iterations (required for dynamic frequency)
        """
        super().__init__()  # Call parent init for reformation tracking
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.mst_update_frequency = mst_update_frequency
        self.initialization_method = initialization_method
        self.influence_function = influence_function
        self.seed = seed
        self.verbose = verbose
        self.mst_params = mst_params if isinstance(mst_params, dict) else {}
        
        # Dynamic MST frequency parameters
        self.dynamic_mst_frequency = dynamic_mst_frequency
        self.mst_decay_function = mst_decay_function
        self.initial_mst_frequency = initial_mst_frequency
        self.final_mst_frequency = final_mst_frequency
        self.total_iterations = total_iterations
        
        # FloatSOM is fully GPU
        self.xp = cp
        
        # MST state
        self.mst_edges = None
        self.adjacency_list = None
        self.iteration_count = 0
        self.graph_distances = None  # Precomputed shortest path distances
        
        # Cache for influence maps with 10% radius threshold
        self._influence_map_cache = {}  # Maps function_type -> (cached_radius, influence_matrix)
        self._radius_threshold = 0.1  # 10% threshold for recalculation
        self.precomputed_radii = None
        self._influence_params: Dict[str, object] = {}
        self._use_sparse_influence = False
        self._use_cpu_storage = False  # Flag for CPU offloading for large grids
        self._cpu_storage_threshold = 2500  # Node count threshold for CPU storage (50x50 grid)

        # Floyd-Warshall chunk tuning.
        fw_row_chunk_size = self.mst_params.get("fw_row_chunk_size")
        self._fw_row_chunk_size_override = (
            int(fw_row_chunk_size) if fw_row_chunk_size is not None else None
        )
        fw_target_bytes = self.mst_params.get("fw_row_target_bytes")
        self._fw_row_target_bytes = (
            int(fw_target_bytes) if fw_target_bytes is not None else _FW_ROW_TARGET_BYTES
        )

        if self._fw_row_chunk_size_override is not None and self._fw_row_chunk_size_override <= 0:
            raise ValueError("mst_params['fw_row_chunk_size'] must be positive")
        if self._fw_row_target_bytes <= 0:
            raise ValueError("mst_params['fw_row_target_bytes'] must be positive")
        
        # Set random seed if provided
        if seed is not None:
            cp.random.seed(seed)
            np.random.seed(seed)
        
        # Prepare grid coordinates used during initialization
        self._set_initial_grid()
    
    @property
    def name(self) -> str:
        """Get the name of this topology strategy"""
        return "MST"
    
    @property
    def total_nodes(self) -> int:
        """Get the total number of nodes in this topology"""
        return self.num_nodes
    
    def initialize_weights(self, data: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Initialize weights for MST topology

        Args:
            data: Optional input data for PCA initialization

        Returns:
            weights: Initialized weight vectors
        """
        if self.verbose:
            logger.info(f"Initializing MST topology with {self.num_nodes} nodes, input dim {self.input_dim}")

        total_nodes = self.total_nodes
        grid_shape = getattr(self, "grid_shape", (total_nodes,))
        grid_coords = getattr(self, "_initialization_coord_grid", None)
        grid_slots = getattr(self, "_grid_slot_count", total_nodes)

        data_for_init = data
        if data is not None:
            if hasattr(data, 'get'):
                data_np = data.get()
            else:
                data_np = np.asarray(data)
            if data_np.shape[0] == 0:
                data_for_init = data_np
            else:
                if grid_slots > data_np.shape[0]:
                    repeat_factor = int(np.ceil(grid_slots / data_np.shape[0]))
                    data_np = np.tile(data_np, (repeat_factor, 1))
                data_for_init = data_np[:grid_slots]

        if self.initialization_method == "pca" and data_for_init is not None:
            if self.verbose:
                logger.info("Using PCA weights initialization for MST topology")

            if data_for_init.size == 0:
                raise ValueError("Data provided for PCA initialization is empty")
            else:
                weights_grid = pca_weights_init(data_for_init, grid_shape, xp=self.xp, grid_coords=grid_coords)
                weights = weights_grid.reshape(-1, self.input_dim)[:total_nodes]
                weights = weights.astype(self.xp.float32)

                if self.verbose:
                    logger.info("SUCCESS: PCA weights initialization completed for MST topology!")

        elif self.initialization_method in ["pca_sampling", "pca_sampling_snake"] and data_for_init is not None:
            if self.verbose:
                logger.info(f"Using {self.initialization_method} initialization for MST topology")

            if self.initialization_method == "pca_sampling":
                weights_grid = pca_sampling_init(data_for_init, grid_shape, n_iterations=10, xp=self.xp, grid_coords=grid_coords)
            else:
                weights_grid = pca_sampling_init_snake(data_for_init, grid_shape, n_iterations=10, xp=self.xp, grid_coords=grid_coords)

            weights = weights_grid.reshape(-1, self.input_dim)[:total_nodes]
            weights = weights.astype(self.xp.float32)

            if self.verbose:
                logger.info(f"SUCCESS: {self.initialization_method} initialization completed for MST topology!")

        elif self.initialization_method == "pca_density" and data_for_init is not None:
            if self.verbose:
                logger.info("Using density-based PCA initialization for MST topology")

            weights_grid = pca_density_init(data_for_init, grid_shape, xp=self.xp, verbose=self.verbose, grid_coords=grid_coords)
            weights = weights_grid.reshape(-1, self.input_dim)[:total_nodes]
            weights = weights.astype(self.xp.float32)

            if self.verbose:
                logger.info("SUCCESS: density-based PCA initialization completed for MST topology!")

        else:
            weights = (self.xp.random.rand(total_nodes, self.input_dim) * 2 - 1).astype(self.xp.float32)
            norms = self.xp.linalg.norm(weights, axis=1, keepdims=True)
            norms = self.xp.maximum(norms, self.xp.float32(1e-12))
            weights = weights / norms
            if self.verbose:
                logger.info("Using random initialization for MST topology")

        # Update coordinate grid slice in case initialization grid changed
        if hasattr(self, "_initialization_coord_grid"):
            self.coord_grid = self._initialization_coord_grid[: self.num_nodes]

        # Calculate initial MST
        self._update_mst(weights)

        return weights

    def build_mst_from_data(self, data, n_nodes):
        """
        Build MST structure from input data
        
        Args:
            data: Input dataset
            n_nodes: Number of SOM nodes
            
        Returns:
            MST adjacency structure
        """
        # Initialize weights from data
        weights = self.initialize_weights(data)
        
        # MST is built during initialization
        return self.adjacency_list
    

    def get_coordinates(self) -> cp.ndarray:
        """
        Get the coordinate grid for this topology
        
        Returns:
            coord_grid: Coordinate grid
        """
        return self.coord_grid
    

    def _set_initial_grid(self, max_slots: Optional[int] = None) -> None:
        """Create and cache grid metadata used during initialization routines."""
        grid_height, grid_width = self._determine_grid_shape(self.num_nodes, max_slots=max_slots)
        grid_slots = grid_height * grid_width

        self.grid_shape = (grid_height, grid_width)
        self._grid_slot_count = grid_slots
        self._initialization_coord_grid = self._create_rectangular_coordinates(grid_height, grid_width, grid_slots)
        self.coord_grid = self._initialization_coord_grid[: self.num_nodes]

    def _determine_grid_shape(self, total_nodes: int, max_slots: Optional[int] = None) -> Tuple[int, int]:
        """Determine a near-rectangular grid shape that can host all nodes."""
        if total_nodes < 1:
            raise ValueError("MST topology requires at least one node")

        max_slots = max(total_nodes, max_slots) if max_slots is not None else None
        ideal_height = int(np.ceil(np.sqrt(total_nodes)))

        best_height = 1
        best_width = total_nodes
        best_product = total_nodes

        for height in range(ideal_height, 0, -1):
            width = int(np.ceil(total_nodes / height))
            product = height * width

            if product < total_nodes:
                continue
            if max_slots is not None and product > max_slots:
                continue

            best_height = height
            best_width = width
            best_product = product
            break

        if max_slots is not None and best_product > max_slots:
            best_height = 1
            best_width = total_nodes

        return best_height, best_width

    def _create_rectangular_coordinates(self, grid_height: int, grid_width: int, total_points: int) -> cp.ndarray:
        """Create a simple rectangular coordinate grid for initialization routines."""
        coords_np = np.zeros((total_points, 2), dtype=np.float32)

        for idx in range(total_points):
            y = idx // grid_width
            x = idx % grid_width
            coords_np[idx] = (x, y)

        return self.xp.asarray(coords_np)

    def _get_update_frequency(self, iteration_index: int) -> Tuple[Optional[int], str]:
        """Return update frequency and mode for the given iteration index."""
        if self.mst_update_frequency is not None:
            return self.mst_update_frequency, "fixed"

        if self.dynamic_mst_frequency and self.total_iterations is not None:
            from ..floatsom_params import calculate_mst_update_frequency

            current_frequency = calculate_mst_update_frequency(
                iteration_index,
                self.total_iterations,
                self.initial_mst_frequency,
                self.final_mst_frequency,
                self.mst_decay_function,
            )
            return current_frequency, "dynamic"

        return None, "fallback"

    def _should_update_mst(self, iteration_index: int) -> Tuple[bool, Optional[int], str]:
        """Determine whether MST should update for the given iteration index."""
        current_frequency, mode = self._get_update_frequency(iteration_index)
        if current_frequency is None:
            return True, None, mode

        iteration_number = iteration_index + 1
        should_update = (iteration_number == 1) or (iteration_number % current_frequency == 0)
        return should_update, current_frequency, mode

    def should_update_mst(self, current_iteration: int) -> bool:
        """Check whether MST should update without mutating internal counters."""
        should_update, _, _ = self._should_update_mst(current_iteration)
        return should_update

    def requires_topology_updates(self) -> bool:
        """MST is dynamic and needs periodic topology refreshes."""
        return True

    def should_update_topology(self, current_iteration: int) -> bool:
        """Generic update hook used by FloatSOM."""
        return self.should_update_mst(current_iteration)

    def finalize_topology(self, weights: cp.ndarray) -> None:
        """Finalize MST structure with the latest weights."""
        self._update_mst(weights)
    
    def update_topology(self, weights: cp.ndarray, current_iteration: int = None) -> None:
        """
        Update the MST based on current weights
        
        Args:
            weights: Current SOM weights
            current_iteration: Current training iteration
        """
        iteration_index = self.iteration_count if current_iteration is None else current_iteration
        self.iteration_count = iteration_index + 1

        should_update, current_frequency, mode = self._should_update_mst(iteration_index)
        iteration_number = iteration_index + 1

        if self.verbose and should_update:
            if mode == "fixed":
                logger.info(
                    "Updating MST at iteration %d (fixed frequency: every %d iterations)",
                    iteration_number,
                    current_frequency,
                )
            elif mode == "dynamic":
                logger.info(
                    "Updating MST at iteration %d (dynamic frequency: every %d iterations)",
                    iteration_number,
                    current_frequency,
                )
            else:
                logger.info(
                    "Updating MST at iteration %d (no frequency specified, updating every iteration)",
                    iteration_number,
                )
        
        # Update MST if needed
        if should_update:
            self._update_mst(weights)
    
    def _calculate_distance_matrix(self, weights: cp.ndarray) -> cp.ndarray:
        """Calculate pairwise distances without materializing a 3D tensor.

        The naive broadcast implementation builds an intermediate tensor with
        shape (num_nodes, num_nodes, input_dim), which explodes GPU memory for
        high-dimensional weights. Instead we use the Gram-matrix identity:
            ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b

        Note: For MST construction we only need a monotone distance measure, so
        we return squared distances (avoids an extra sqrt pass). Kruskal's MST
        is identical under any strictly monotone transform.
        """
        w = weights.astype(cp.float32, copy=False)

        # (n, 1) squared norms
        norms = cp.sum(w * w, axis=1, keepdims=True)

        # (n, n) squared distances via GEMM
        sq_dist = -2.0 * (w @ w.T)
        sq_dist += norms
        sq_dist += norms.T

        # Clamp small negatives from floating point error
        cp.maximum(sq_dist, 0.0, out=sq_dist)
        return sq_dist
    
    def _calculate_mst_edges(self, distance_matrix: cp.ndarray) -> List[Tuple[int, int]]:
        """Calculate MST edges using CPU algorithm"""
        # Transfer to CPU for MST calculation
        distance_matrix_cpu = distance_matrix.get()
        return calculate_mst_cpu(distance_matrix_cpu)
    
    def _update_mst(self, weights: cp.ndarray) -> None:
        """
        Calculate MST from current weights

        Args:
            weights: Current SOM weights
        """
        if self.verbose:
            start_time = time.time()
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(weights)
        
        # Calculate MST edges
        self.mst_edges = self._calculate_mst_edges(distance_matrix)
        
        # Build adjacency list
        self._build_adjacency_list()

        # Precompute graph distances
        self._precompute_graph_distances()

        # Precompute influence maps if we have radii
        if hasattr(self, '_radius_mapping') and self._radius_mapping:
            unique_radii = list(set(self._radius_mapping.values()))
            self._precompute_optimized_influence_maps(unique_radii)
        
        if self.verbose:
            end_time = time.time()
            num_edges = len(self.mst_edges)
            logger.info(f"MST updated in {end_time - start_time:.3f}s with {num_edges} edges")

    def _build_adjacency_list(self) -> None:
        """Build adjacency list from MST edges"""
        self.adjacency_list = build_adjacency_list(self.mst_edges, self.num_nodes)

    def get_neighbors(self, node_idx: int) -> List[int]:
        """Return neighboring node indices for a given node."""
        if self.adjacency_list is None:
            if self.mst_edges is None:
                raise ValueError("MST not initialized. Call update_topology first.")
            self._build_adjacency_list()
        return self.adjacency_list.get(int(node_idx), [])

    def _precompute_graph_distances(self) -> None:
        """
        Precompute all-pairs shortest path distances on the MST using Floyd-Warshall
        """
        self._use_cpu_storage = self.num_nodes > self._cpu_storage_threshold
        if self._use_cpu_storage and self.verbose:
            logger.info(
                "MST has %d nodes (>%d), computing graph distances with CPU storage",
                self.num_nodes,
                self._cpu_storage_threshold,
            )

        row_chunk_size = _resolve_fw_row_chunk_size(
            self.num_nodes,
            int(np.dtype(np.float32).itemsize),
            requested_row_chunk_size=self._fw_row_chunk_size_override,
            target_bytes=self._fw_row_target_bytes,
        )
        if self.verbose:
            logger.info(
                "Floyd-Warshall row chunk size=%d (target_bytes=%d, override=%s)",
                row_chunk_size,
                self._fw_row_target_bytes,
                self._fw_row_chunk_size_override,
            )

        graph_dist_gpu = cp.full((self.num_nodes, self.num_nodes), cp.inf, dtype=cp.float32)
        cp.fill_diagonal(graph_dist_gpu, 0)

        for u, v in self.mst_edges:
            graph_dist_gpu[u, v] = 1
            graph_dist_gpu[v, u] = 1

        _run_chunked_floyd_warshall_gpu(graph_dist_gpu, row_chunk_size=row_chunk_size)

        if self._use_cpu_storage:
            self.graph_distances = cp.asnumpy(graph_dist_gpu)
            del graph_dist_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            self.graph_distances = graph_dist_gpu
    
    def _calculate_mst_distance(self, node1, node2):
        """
        Calculate distance along MST path
        
        Args:
            node1: First node index
            node2: Second node index
            
        Returns:
            distance: MST hop distance
        """
        if self.graph_distances is None:
            raise ValueError("MST not initialized. Call update_topology first.")

        distance_value = self.graph_distances[node1, node2]
        if hasattr(distance_value, "get"):
            distance_value = distance_value.get()
        if hasattr(distance_value, "item"):
            distance_value = distance_value.item()
        return float(distance_value)
    
    def _find_shortest_path(self, start, end):
        """
        Find shortest path between nodes in MST
        
        Args:
            start: Start node index
            end: End node index
            
        Returns:
            path: List of node indices forming shortest path
        """
        if self.adjacency_list is None:
            raise ValueError("MST not initialized. Call update_topology first.")
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            
            if node == end:
                return path
            
            for neighbor in self.adjacency_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def precompute_topology_data(self, som_params, radii_list: List[float]) -> None:
        """
        Precompute optimized influence maps with radius deduplication.
        Only stores matrices for radii that differ by >10%.
        
        Args:
            som_params: SOM parameters (compatibility)
            radii_list: List of radius values
        """
        if self.verbose:
            logger.info(f"Precomputing {self.name} topology data with radius deduplication...")

        if som_params is not None:
            processing_config = getattr(som_params, "processing_config", None)
            influence_params = getattr(processing_config, "influence_params", None) or {}
            self._influence_params = dict(influence_params)
            self._use_sparse_influence = bool(getattr(processing_config, "use_sparse_influence", False))
        
        # Store original radii list for reference
        self.precomputed_radii = radii_list
        
        # Deduplicate radii list by 10% threshold
        from .utils import deduplicate_radii_list
        unique_radii, self._radius_mapping = deduplicate_radii_list(radii_list, self._radius_threshold)
        
        if self.verbose:
            logger.info(f"Deduplicated {len(radii_list)} radii to {len(unique_radii)} unique matrices")
            logger.info(f"Memory savings: {(1 - len(unique_radii)/len(radii_list))*100:.1f}%")
        
        # If MST is already built (from initialization), precompute influence maps now
        if self.graph_distances is not None:
            self._precompute_optimized_influence_maps(unique_radii)
        # Otherwise, influence maps will be precomputed after MST is built in update_topology
        
        if self.verbose:
            logger.info(f"MST topology precomputation setup complete")
    
    def get_distance_matrix_for_color_sets(self, current_iteration: int = None) -> cp.ndarray:
        """
        Get precomputed distance matrix for color set calculation.
        Returns MST graph distances - algorithms apply radius thresholds as needed.
        
        Args:
            current_iteration: Current training iteration (optional)
            
        Returns:
            distance_matrix: Precomputed MST graph distance matrix between all nodes
        """
        if self.graph_distances is None:
            raise ValueError("MST not initialized. Call update_topology with weights first.")
        
        # Convert from CPU to GPU if needed
        graph_distances = self.graph_distances
        if self._use_cpu_storage and isinstance(self.graph_distances, np.ndarray):
            graph_distances = cp.asarray(self.graph_distances)
        
        # Return squared distances for consistency with other topologies
        return graph_distances ** 2
    
    
    def get_precomputed_influence_matrix(self, radius: float, influence_function: str = 'gaussian', return_radius: bool = False) -> cp.ndarray:
        """
        Get cached continuous influence map for weight updates with optimized precomputation.
        Continuous functions: Gaussian, bubble, Mexican hat adapted for MST graph distances.
        Uses O(1) lookup with radius deduplication - assumes cache always finds.
        
        Args:
            radius: Current neighborhood radius
            influence_function: Type of influence function ('gaussian', 'bubble', 'mexican_hat')
            
        Returns:
            weight_influence_map: Precomputed continuous influence matrix for weight updates
        """
        # Validate parameters
        super().get_precomputed_influence_matrix(radius, influence_function)
        
        if self.graph_distances is None:
            raise ValueError("MST not initialized. Call update_topology with weights first.")

        # Lazily initialize radius mapping when direct callers skip precompute_topology_data.
        radius_key = round(float(radius), 3)
        if not hasattr(self, "_radius_mapping") or self._radius_mapping is None:
            self._radius_mapping = {}

        if radius_key not in self._radius_mapping:
            from .utils import deduplicate_radii_list

            tracked_radii = list(self._radius_mapping.keys())
            tracked_radii.append(radius_key)
            unique_radii, self._radius_mapping = deduplicate_radii_list(
                tracked_radii, self._radius_threshold
            )
            self._precompute_optimized_influence_maps(unique_radii)
        
        # Map requested radius to precomputed radius using deduplication mapping
        cached_radius = self._radius_mapping[radius_key]
        cached_key = (cached_radius, influence_function)
        
        # Ensure requested influence function exists for this radius.
        if cached_key not in self._influence_map_cache:
            self._compute_mst_influence_map(cached_radius, influence_function)
        influence_map = self._influence_map_cache[cached_key]
        
        # Convert from CPU to GPU if needed
        if isinstance(influence_map, dict) and influence_map.get("format") == "csr":
            pass
        elif self._use_cpu_storage and isinstance(influence_map, np.ndarray):
            influence_map = cp.asarray(influence_map)
        
        if return_radius:
            return influence_map, cached_radius
        else:
            return influence_map
    
    def _compute_mst_influence_map(self, radius: float, influence_function: str) -> None:
        """
        Compute influence map for MST topology using graph distances.
        
        Args:
            radius: Current neighborhood radius
            influence_function: Type of influence function
        """
        if self.graph_distances is None:
            raise ValueError("MST not initialized. Call update_topology with weights first.")
        
        xp = self.xp
        
        # Convert from CPU to GPU if needed for calculation
        distances = self.graph_distances
        if self._use_cpu_storage and isinstance(self.graph_distances, np.ndarray):
            distances = cp.asarray(self.graph_distances)
        
        # Ensure radius is a scalar
        radius_scalar = float(radius.item() if hasattr(radius, 'item') else radius)
        std_coeff = float(self._influence_params.get("std_coeff", 0.5))
        if std_coeff <= 0:
            raise ValueError(f"std_coeff must be positive, got {std_coeff}")
        compact_support = bool(self._influence_params.get("compact_support", False))
        
        if self._use_sparse_influence:
            from .utils import build_sparse_influence_from_distance_matrix
            influence_map = build_sparse_influence_from_distance_matrix(
                distances,
                radius_scalar,
                influence_function,
                std_coeff=std_coeff,
                compact_support=compact_support,
                distance_is_squared=False,
            )
        else:
            if influence_function == 'gaussian':
                # Gaussian influence adapted for MST
                effective_sigma = (radius_scalar * std_coeff) ** 2
                influence_map = xp.exp(-distances ** 2 / (2 * effective_sigma))
                if compact_support:
                    influence_map *= distances <= radius_scalar
            elif influence_function == 'bubble':
                # Bubble influence adapted for MST
                influence_map = (distances <= radius_scalar).astype(xp.float32)
            elif influence_function == 'mexican_hat':
                # Mexican hat influence adapted for MST
                sigma = radius_scalar * std_coeff
                normalized_dist = distances / sigma
                normalized_sq = normalized_dist ** 2
                influence_map = (1 - normalized_sq) * xp.exp(-normalized_sq / 2)
                if compact_support:
                    influence_map *= distances <= radius_scalar
            else:
                raise ValueError(f"Unknown influence function: {influence_function}")
        
        # Store on CPU if using CPU storage
        if self._use_cpu_storage and not isinstance(influence_map, dict):
            influence_map = cp.asnumpy(influence_map)
        
        # Store in cache
        cache_key = (round(float(radius_scalar), 3), influence_function)
        self._influence_map_cache[cache_key] = influence_map
        
        # Free GPU memory if using CPU storage
        if self._use_cpu_storage:
            cp.get_default_memory_pool().free_all_blocks()
    
    def _precompute_optimized_influence_maps(self, unique_radii: List[float]) -> None:
        """
        Precompute influence maps for unique radii and the specified influence function.
        Called after MST is built.
        
        Args:
            unique_radii: List of deduplicated radii to precompute
        """
        if self.graph_distances is None:
            if self.verbose:
                logger.info("MST not yet built - influence maps will be computed on-demand")
            return
        
        # Only precompute the specified influence function
        influence_functions = [self.influence_function]
        
        if self.verbose:
            logger.info(f"Precomputing {self.influence_function} MST influence maps for {len(unique_radii)} radii...")
        
        for radius in unique_radii:
            for func_name in influence_functions:
                self._compute_mst_influence_map(radius, func_name)
        
        if self.verbose:
            total_matrices = len(self._influence_map_cache)
            matrix_size = self.num_nodes * self.num_nodes * 4  # float32
            total_memory = total_matrices * matrix_size / (1024**2)  # MB
            logger.info(f"Precomputed {total_matrices} MST influence maps using {total_memory:.1f} MB of memory")
    
    def set_precomputed_radii(self, radii_list: List[float]) -> None:
        """
        Set precomputed radii list
        
        Args:
            radii_list: List of radius values
        """
        self.precomputed_radii = radii_list
        if self.verbose:
            logger.info(f"MST topology received {len(radii_list)} pre-computed radii")
    
    def _calculate_influence_map(self, coord_grid: cp.ndarray, radius: float, influence_function: str) -> cp.ndarray:
        """
        Calculate influence map for given radius and function (fallback method).
        
        Args:
            coord_grid: Coordinate grid for topology (unused for MST)
            radius: Current neighborhood radius
            influence_function: Type of influence function
            
        Returns:
            influence_map: Calculated influence matrix
        """
        self._compute_mst_influence_map(radius, influence_function)
        cache_key = (round(float(radius), 3), influence_function)
        return self._influence_map_cache[cache_key]
