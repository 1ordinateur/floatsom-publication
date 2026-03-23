"""
Hexagonal topology for FloatSOM - hexagonal grid with offset rows
"""

import cupy as cp
import numpy as np
import logging
from typing import Any, Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

from .som_topology import SOMTopology
from .pca_initialization import pca_weights_init, pca_sampling_init, pca_sampling_init_snake, pca_density_init
from .utils import iter_chunk_ranges


class HexagonalTopology(SOMTopology):
    """
    Hexagonal grid topology where odd rows are offset by 0.5 units
    Creates more natural neighborhood relationships than rectangular grids
    """
    
    def __init__(self,
                 grid_size: int,
                 input_dim: int,
                 topology_type: str = "planar",
                 initialization_method: str = "random",
                 influence_function: str = "gaussian",
                 seed: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize hexagonal topology
        
        Args:
            grid_size: Size of the hexagonal grid (grid_size x grid_size)
            input_dim: Dimensionality of input data
            topology_type: Type of topology ('planar' or 'toroidal')
            initialization_method: Method to initialize weights ('random' or 'pca')
            influence_function: Type of influence function to use ('gaussian', 'bubble', 'mexican_hat')
            seed: Random seed for reproducibility
            verbose: Whether to print verbose output
        """
        super().__init__()  # Call parent init for reformation tracking
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.topology_type = topology_type
        self.initialization_method = initialization_method
        self.influence_function = influence_function
        self.seed = seed
        self.verbose = verbose
        
        # FloatSOM is fully GPU
        self.xp = cp
        
        # Set random seed if provided
        if seed is not None:
            cp.random.seed(seed)
            np.random.seed(seed)
        
        # Create hexagonal coordinate grid
        self.coord_grid = self._create_hexagonal_coordinates()
        
        # Cache for precomputed data
        self._distance_matrix = None
        self.precomputed_radii = None
        
        # Cache for influence maps with 10% radius threshold
        self._influence_map_cache = {}  # Maps function_type -> (cached_radius, influence_matrix)
        self._radius_threshold = 0.1  # 10% threshold for recalculation
        self._topology_initialized_for_training = False
        self._use_cpu_storage = False  # Flag for CPU offloading for large grids
        self._cpu_storage_threshold = 2500  # Node count threshold for CPU storage (50x50 grid)
        self._use_sparse_influence = False
    
    @property
    def name(self) -> str:
        """Get the name of this topology strategy"""
        base_name = "Hexagonal"
        if self.topology_type == "toroidal":
            return f"Toroidal-{base_name}"
        return base_name
    
    @property
    def total_nodes(self) -> int:
        """Get the total number of nodes in this topology"""
        return self.grid_size * self.grid_size
    
    def initialize_weights(self, data: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Initialize weights for hexagonal topology
        
        Args:
            data: Optional input data for PCA initialization
            
        Returns:
            weights: Initialized weight vectors
        """
        if self.verbose:
            logger.info(f"Initializing {self.name} grid with {self.total_nodes} nodes, input dim {self.input_dim}")
        
        total_nodes = self.total_nodes
        
        # Check if data is provided for PCA initialization
        if self.initialization_method == "pca" and data is not None:
            if self.verbose:
                logger.info(f"Using PCA weights initialization for hexagonal topology")
            
            if data.size == 0:
                raise ValueError("Data provided for PCA initialization is empty")
            
            # Use 2D grid shape with hexagonal coordinates
            grid_shape = (self.grid_size, self.grid_size)
            
            # Call PCA initialization with hexagonal coordinates
            weights_grid = pca_weights_init(data, grid_shape, xp=self.xp, grid_coords=self.coord_grid)
            
            # Reshape to match expected format
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: PCA weights initialization completed for hexagonal topology!")
        
        elif self.initialization_method in ["pca_sampling", "pca_sampling_snake"]:
            if self.verbose:
                logger.info(f"Using {self.initialization_method} initialization for hexagonal topology")
            
            grid_shape = (self.grid_size, self.grid_size)
            
            if self.initialization_method == "pca_sampling":
                weights_grid = pca_sampling_init(data, grid_shape, n_iterations=10, xp=self.xp, grid_coords=self.coord_grid)
            else:  # pca_sampling_snake
                weights_grid = pca_sampling_init_snake(data, grid_shape, n_iterations=10, xp=self.xp, grid_coords=self.coord_grid)
            
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: {self.initialization_method} initialization completed for hexagonal topology!")
        
        elif self.initialization_method == "pca_density":
            if self.verbose:
                logger.info(f"Using density-based PCA initialization for hexagonal topology")
            
            grid_shape = (self.grid_size, self.grid_size)
            weights_grid = pca_density_init(data, grid_shape, xp=self.xp, verbose=self.verbose, grid_coords=self.coord_grid)
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: density-based PCA initialization completed for hexagonal topology!")
        
        else:
            # Fallback to random initialization
            weights = (self.xp.random.rand(total_nodes, self.input_dim) * 2 - 1).astype(self.xp.float32)
            norms = self.xp.linalg.norm(weights, axis=1, keepdims=True)
            norms = self.xp.maximum(norms, self.xp.float32(1e-12))
            weights = weights / norms
            if self.verbose:
                logger.info(f"Using random initialization for hexagonal topology")
        
        return weights
    
    
    
    def get_coordinates(self) -> cp.ndarray:
        """
        Get the coordinate grid for this topology
        
        Returns:
            coord_grid: Hexagonal coordinate grid
        """
        return self.coord_grid
    
    def update_topology(self, weights: cp.ndarray, current_iteration: int = None) -> None:
        """
        Update the topology structure
        
        For fixed hexagonal topologies, this is a no-op
        """
        pass
    
    def _create_hexagonal_coordinates(self) -> cp.ndarray:
        """
        Create hexagonal coordinate grid with row offsets
        Odd rows: x positions offset by 0.5 units
        
        Returns:
            coord_grid: Hexagonal coordinate grid
        """
        # Match XPySOM row-offset parity: offset rows with parity of (grid_size - 1).
        # This means odd rows for even grids, and even rows for odd grids.
        offset_parity = (self.grid_size - 1) % 2
        xp = self.xp

        x = xp.arange(self.grid_size, dtype=xp.float32)
        y = xp.arange(self.grid_size, dtype=xp.float32)
        xx, yy = xp.meshgrid(x, y)

        row_parity = (yy.astype(xp.int32) % 2)
        offset_mask = (row_parity == offset_parity).astype(xp.float32)
        xx = xx + 0.5 * offset_mask

        coord_grid = xp.stack([xx.ravel(), yy.ravel()], axis=1)
        return coord_grid
    
    def _precompute_distance_matrix(self) -> None:
        """
        Precompute squared distance matrix between all nodes in the hexagonal grid
        """
        total_nodes = self.total_nodes
        
        if self.verbose:
            topology_str = f"{self.topology_type} hexagonal"
            logger.info(f"Precomputing {topology_str} distance matrix for {total_nodes} nodes...")
        
        # Decide whether to use CPU storage based on grid size
        if self.total_nodes > self._cpu_storage_threshold:
            self._use_cpu_storage = True
            if self.verbose:
                logger.info(f"Grid has {self.total_nodes} nodes (>{self._cpu_storage_threshold}), using chunked computation with CPU storage")
            
            # Pre-allocate distance matrix on CPU
            self._distance_matrix = np.zeros((total_nodes, total_nodes), dtype=np.float32)
            
            # Process in chunks to avoid GPU memory exhaustion
            chunk_size = min(100, total_nodes)  # Process 100 rows at a time
            
            for start_idx, end_idx in iter_chunk_ranges(total_nodes, chunk_size):
                
                if self.topology_type == "toroidal":
                    # Extract coordinates for toroidal calculation
                    x_coords = self.coord_grid[:, 0]
                    y_coords = self.coord_grid[:, 1]
                    
                    # Get chunk of first coordinates
                    x_chunk = x_coords[start_idx:end_idx]
                    y_chunk = y_coords[start_idx:end_idx]
                    coords1_chunk = self.xp.stack([x_chunk, y_chunk], axis=1)
                    coords1_expanded = coords1_chunk[:, None, :]  # Shape: (chunk_size, 1, 2)
                    
                    # All second coordinates
                    coords2 = self.xp.stack([x_coords, y_coords], axis=1)
                    coords2_expanded = coords2[None, :, :]  # Shape: (1, total_nodes, 2)
                    
                    # Calculate absolute differences for this chunk
                    diff = self.xp.abs(coords1_expanded - coords2_expanded)
                    
                    # Apply toroidal wrapping
                    wrap_diff = self.grid_size - diff
                    min_diff = self.xp.minimum(diff, wrap_diff)
                    
                    # Calculate squared distances for this chunk
                    chunk_distances = self.xp.sum(min_diff ** 2, axis=2)
                else:
                    # Regular Euclidean distances
                    coords1_chunk = self.coord_grid[start_idx:end_idx]
                    coords1_expanded = coords1_chunk[:, None, :]
                    coords2_expanded = self.coord_grid[None, :, :]
                    
                    diff = coords1_expanded - coords2_expanded
                    chunk_distances = self.xp.sum(diff ** 2, axis=2)
                
                # Transfer chunk to CPU and store
                self._distance_matrix[start_idx:end_idx, :] = cp.asnumpy(chunk_distances)
                
                # Free GPU memory periodically
                del chunk_distances
                if 'diff' in locals():
                    del diff
                if 'min_diff' in locals():
                    del min_diff
                if start_idx % (chunk_size * 10) == 0:
                    cp.get_default_memory_pool().free_all_blocks()
            
            matrix_size = self._distance_matrix.nbytes / (1024**2)  # MB
            if self.verbose:
                logger.info(f"Hexagonal distance matrix computed and stored on CPU: {matrix_size:.1f} MB")
        else:
            # Small grid - compute all at once on GPU
            self._use_cpu_storage = False
            
            # Calculate all pairwise distances using hexagonal coordinates
            coords_expanded1 = self.coord_grid[:, None, :]  # Shape: (total_nodes, 1, coord_dim)
            coords_expanded2 = self.coord_grid[None, :, :]  # Shape: (1, total_nodes, coord_dim)
            
            if self.topology_type == "toroidal":
                # Extract x and y coordinates from hexagonal coordinate grid  
                x_coords = self.coord_grid[:, 0]
                y_coords = self.coord_grid[:, 1]
                
                # Create expanded coordinate arrays for pairwise distance calculation
                coords_expanded1 = self.xp.stack([x_coords, y_coords], axis=1)[:, None, :]
                coords_expanded2 = self.xp.stack([x_coords, y_coords], axis=1)[None, :, :]
                
                # Calculate absolute differences
                diff = self.xp.abs(coords_expanded1 - coords_expanded2)
                
                # Apply toroidal wrapping: min(diff, grid_size - diff)
                wrap_diff = self.grid_size - diff
                min_diff = self.xp.minimum(diff, wrap_diff)
                
                # Calculate squared distances
                self._distance_matrix = self.xp.sum(min_diff ** 2, axis=2)
            else:
                # Regular Euclidean distances in hexagonal coordinate space
                diff = coords_expanded1 - coords_expanded2
                self._distance_matrix = self.xp.sum(diff ** 2, axis=2)
            
            if self.verbose:
                matrix_size = self._distance_matrix.nbytes / (1024**2)  # MB
                logger.info(f"Hexagonal distance matrix computed and stored on GPU: {matrix_size:.1f} MB")
    
    def precompute_topology_data(self, som_params, radii_list: List[float]) -> None:
        """
        Precompute influence maps for each scheduled radius (no deduplication).
        
        Args:
            som_params: SOM parameters (compatibility)
            radii_list: List of radius values
        """
        if self.verbose:
            logger.info(f"Precomputing {self.name} topology data without radius deduplication...")

        influence_params = {}
        self._use_sparse_influence = False
        if som_params is not None:
            processing_config = getattr(som_params, "processing_config", None)
            influence_params = getattr(processing_config, "influence_params", None) or {}
            self._use_sparse_influence = bool(getattr(processing_config, "use_sparse_influence", False))
        
        # Store original radii list for reference
        self.precomputed_radii = radii_list
        exact_radii = [float(radius) for radius in radii_list]
        
        if not self._use_sparse_influence:
            # Precompute distance matrix for dense influence calculations
            self._precompute_distance_matrix()
        
        # Precompute influence maps for every scheduled radius
        if self._use_sparse_influence:
            self._precompute_sparse_influence_maps(exact_radii, influence_params)
        else:
            self._precompute_optimized_influence_maps(exact_radii, influence_params)
        
        if self.verbose:
            logger.info(f"Hexagonal topology precomputation complete")
    
    def get_distance_matrix_for_color_sets(self, current_iteration: int = None) -> cp.ndarray:
        """
        Get precomputed distance matrix for color set calculation.
        Returns raw squared distances - algorithms apply radius thresholds as needed.
        
        Args:
            current_iteration: Current training iteration (optional)
            
        Returns:
            distance_matrix: Precomputed squared distance matrix between all nodes
        """
        # Always use precomputed distance matrix - no fallback
        # Convert from CPU to GPU if needed
        if self._use_cpu_storage and isinstance(self._distance_matrix, np.ndarray):
            return cp.asarray(self._distance_matrix)
        return self._distance_matrix
    
    
    def get_precomputed_influence_matrix(self, radius: float, influence_function: str = 'gaussian', return_radius: bool = False) -> cp.ndarray:
        """
        Get cached continuous influence map for weight updates with optimized precomputation.
        Continuous functions: Gaussian, bubble, Mexican hat for weight update calculations.
        Uses O(1) lookup with radius deduplication - assumes cache always finds.
        
        Args:
            radius: Current neighborhood radius
            influence_function: Type of influence function ('gaussian', 'bubble', 'mexican_hat')
            
        Returns:
            weight_influence_map: Precomputed continuous influence matrix for weight updates
        """
        # Validate parameters
        super().get_precomputed_influence_matrix(radius, influence_function)
        
        cached_radius = float(radius)
        cached_key = (cached_radius, influence_function)
        
        # Fast O(1) lookup - assumes cache always hits
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
    
    def set_precomputed_radii(self, radii_list: List[float]) -> None:
        """
        Set precomputed radii list
        
        Args:
            radii_list: List of radius values
        """
        self.precomputed_radii = radii_list
        if self.verbose:
            logger.info(f"Hexagonal topology received {len(radii_list)} pre-computed radii")
    
    def _precompute_optimized_influence_maps(self, radii: List[float], influence_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Precompute influence maps for all scheduled radii and the specified influence function.
        
        Args:
            radii: List of radii to precompute
            influence_params: Optional parameters for influence kernels
        """
        from .utils import calculate_gaussian_influence, calculate_bubble_influence, calculate_mexican_hat_influence
        
        # Only precompute the specified influence function
        all_influence_functions = {
            'gaussian': calculate_gaussian_influence,
            'bubble': calculate_bubble_influence, 
            'mexican_hat': calculate_mexican_hat_influence
        }
        
        # Select only the function we need
        influence_functions = {
            self.influence_function: all_influence_functions[self.influence_function]
        }
        
        influence_params = influence_params or {}
        std_coeff = float(influence_params.get("std_coeff", 0.5))
        compact_support = bool(influence_params.get("compact_support", False))

        if self.verbose:
            logger.info(f"Precomputing {self.influence_function} hexagonal influence maps for {len(radii)} radii...")
        
        coord_grid = self.get_coordinates()
        
        for radius in radii:
            for func_name, func in influence_functions.items():
                # Calculate influence map using precomputed distance matrix
                # Pass toroidal parameter for hexagonal topology
                
                # Convert distance matrix to GPU if needed for calculation
                distance_matrix_for_calc = self._distance_matrix
                if self._use_cpu_storage and isinstance(self._distance_matrix, np.ndarray):
                    distance_matrix_for_calc = cp.asarray(self._distance_matrix)
                
                if func_name in {"gaussian", "mexican_hat"}:
                    influence_map = func(
                        coord_grid,
                        radius,
                        distance_matrix_for_calc,
                        toroidal=(self.topology_type == "toroidal"),
                        grid_size=self.grid_size,
                        std_coeff=std_coeff,
                        compact_support=compact_support,
                    )
                else:
                    influence_map = func(
                        coord_grid,
                        radius,
                        distance_matrix_for_calc,
                        toroidal=(self.topology_type == "toroidal"),
                        grid_size=self.grid_size,
                    )
                
                # Store on CPU if using CPU storage
                if self._use_cpu_storage:
                    influence_map = cp.asnumpy(influence_map)
                
                # Store with (radius, function) key
                cache_key = (float(radius), func_name)
                self._influence_map_cache[cache_key] = influence_map
                
                # Free GPU memory if using CPU storage
                if self._use_cpu_storage:
                    cp.get_default_memory_pool().free_all_blocks()
        
        if self.verbose:
            total_matrices = len(self._influence_map_cache)
            matrix_size = self.total_nodes * self.total_nodes * 4  # float32
            total_memory = total_matrices * matrix_size / (1024**2)  # MB
            logger.info(f"Precomputed {total_matrices} hexagonal influence maps using {total_memory:.1f} MB of memory")

    def _precompute_sparse_influence_maps(self, radii: List[float], influence_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Precompute sparse influence maps for all scheduled radii.

        Args:
            radii: List of radii to precompute
            influence_params: Optional parameters for influence kernels
        """
        self._precompute_distance_matrix()

        from .utils import build_sparse_influence_from_distance_matrix

        influence_params = influence_params or {}
        std_coeff = float(influence_params.get("std_coeff", 0.5))
        compact_support = bool(influence_params.get("compact_support", False))

        if self.verbose:
            logger.info(f"Precomputing sparse {self.influence_function} influence maps for {len(radii)} radii...")

        for radius in radii:
            distance_matrix_for_calc = self._distance_matrix
            if self._use_cpu_storage and isinstance(self._distance_matrix, np.ndarray):
                distance_matrix_for_calc = self._distance_matrix

            influence_map = build_sparse_influence_from_distance_matrix(
                distance_matrix_for_calc,
                radius,
                self.influence_function,
                std_coeff=std_coeff,
                compact_support=compact_support,
                distance_is_squared=True,
            )

            cache_key = (float(radius), self.influence_function)
            self._influence_map_cache[cache_key] = influence_map

            if self._use_cpu_storage:
                cp.get_default_memory_pool().free_all_blocks()
    
