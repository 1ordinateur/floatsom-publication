"""
Grid topology for FloatSOM - standard rectangular grid
"""

import cupy as cp
import numpy as np
import logging
from typing import Any, Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

from .som_topology import SOMTopology
from .pca_initialization import pca_weights_init, pca_sampling_init, pca_sampling_init_snake, pca_density_init
from .utils import iter_chunk_ranges


class GridTopology(SOMTopology):
    """
    Standard rectangular grid topology
    Most common SOM topology with regular grid structure
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
        Initialize rectangular grid topology
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
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
        
        # Create coordinate grid
        self.coord_grid = self._create_coordinate_grid()
        
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
        base_name = "Rectangular"
        if self.topology_type == "toroidal":
            return f"Toroidal-{base_name}"
        return base_name
    
    @property
    def total_nodes(self) -> int:
        """Get the total number of nodes in this topology"""
        return self.grid_size * self.grid_size
    
    def initialize_weights(self, data: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Initialize weight vectors for rectangular grid
        
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
                logger.info(f"Using PCA weights initialization")
            
            if data.size == 0:
                raise ValueError("Data provided for PCA initialization is empty")
            
            # Use 2D grid shape for rectangular grid
            grid_shape = (self.grid_size, self.grid_size)
            
            # Call PCA initialization
            weights_grid = pca_weights_init(data, grid_shape, xp=self.xp)
            
            # Reshape to match expected format
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: PCA weights initialization completed!")
                logger.info(f"  - Final weights shape: {weights.shape}")
                logger.info(f"  - Weight range: [{self.xp.min(weights):.6f}, {self.xp.max(weights):.6f}]")
        
        elif self.initialization_method in ["pca_sampling", "pca_sampling_snake"]:
            if self.verbose:
                logger.info(f"Using {self.initialization_method} initialization")
            
            grid_shape = (self.grid_size, self.grid_size)
            
            if self.initialization_method == "pca_sampling":
                weights_grid = pca_sampling_init(data, grid_shape, n_iterations=10, xp=self.xp)
            else:  # pca_sampling_snake
                weights_grid = pca_sampling_init_snake(data, grid_shape, n_iterations=10, xp=self.xp)
            
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: {self.initialization_method} initialization completed!")
        
        elif self.initialization_method == "pca_density":
            if self.verbose:
                logger.info(f"Using density-based PCA initialization")
            
            grid_shape = (self.grid_size, self.grid_size)
            weights_grid = pca_density_init(data, grid_shape, xp=self.xp, verbose=self.verbose)
            weights = weights_grid.reshape(total_nodes, self.input_dim)
            weights = weights.astype(self.xp.float32)
            
            if self.verbose:
                logger.info(f"SUCCESS: density-based PCA initialization completed!")
        
        else:
            # Fallback to random initialization
            weights = (self.xp.random.rand(total_nodes, self.input_dim) * 2 - 1).astype(self.xp.float32)
            norms = self.xp.linalg.norm(weights, axis=1, keepdims=True)
            norms = self.xp.maximum(norms, self.xp.float32(1e-12))
            weights = weights / norms
            if self.verbose:
                logger.info(f"Using random initialization")
        
        return weights
    
    
    def get_coordinates(self) -> cp.ndarray:
        """
        Get the coordinate grid for this topology
        
        Returns:
            coord_grid: Coordinate grid
        """
        return self.coord_grid
    
    def update_topology(self, weights: cp.ndarray, current_iteration: int = None) -> None:
        """
        Update the topology structure
        
        For fixed grid topologies, this is a no-op
        """
        pass
    
    def _create_coordinate_grid(self) -> cp.ndarray:
        """
        Create a coordinate grid for rectangular SOM nodes
        
        Returns:
            coord_grid: Coordinate grid
        """
        # Create rectangular coordinate grid
        x = self.xp.arange(self.grid_size)
        y = self.xp.arange(self.grid_size)
        xx, yy = self.xp.meshgrid(x, y)
        
        # Reshape to get array of coordinates (x, y)
        coord_grid = self.xp.vstack([xx.ravel(), yy.ravel()]).T.astype(self.xp.float32)
        
        return coord_grid
    
    
    
    def _precompute_distance_matrix(self) -> None:
        """
        Precompute squared distance matrix between all nodes in the rectangular grid
        """
        total_nodes = self.total_nodes
        
        if self.verbose:
            topology_str = f"{self.topology_type} rectangular"
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
                
                # Get chunk of first coordinates
                coords1_chunk = self.coord_grid[start_idx:end_idx]
                coords1_expanded = coords1_chunk[:, None, :]  # Shape: (chunk_size, 1, coord_dim)
                coords2_expanded = self.coord_grid[None, :, :]  # Shape: (1, total_nodes, coord_dim)
                
                if self.topology_type == "toroidal":
                    # Vectorized toroidal distance calculation for this chunk
                    diff = self.xp.abs(coords1_expanded - coords2_expanded)
                    wrap_diff = self.grid_size - diff
                    min_diff = self.xp.minimum(diff, wrap_diff)
                    chunk_distances = self.xp.sum(min_diff ** 2, axis=2)
                else:
                    # Regular Euclidean distances for this chunk
                    diff = coords1_expanded - coords2_expanded
                    chunk_distances = self.xp.sum(diff ** 2, axis=2)
                
                # Transfer chunk to CPU and store
                self._distance_matrix[start_idx:end_idx, :] = cp.asnumpy(chunk_distances)
                
                # Free GPU memory periodically
                del chunk_distances, diff
                if 'min_diff' in locals():
                    del min_diff
                if start_idx % (chunk_size * 10) == 0:
                    cp.get_default_memory_pool().free_all_blocks()
            
            matrix_size = self._distance_matrix.nbytes / (1024**2)  # MB
            if self.verbose:
                logger.info(f"Distance matrix computed and stored on CPU: {matrix_size:.1f} MB")
        else:
            # Small grid - compute all at once on GPU
            self._use_cpu_storage = False
            
            # Calculate all pairwise distances
            coords_expanded1 = self.coord_grid[:, None, :]  # Shape: (total_nodes, 1, coord_dim)
            coords_expanded2 = self.coord_grid[None, :, :]  # Shape: (1, total_nodes, coord_dim)
            
            if self.topology_type == "toroidal":
                # Vectorized toroidal distance calculation
                diff = self.xp.abs(coords_expanded1 - coords_expanded2)
                wrap_diff = self.grid_size - diff
                min_diff = self.xp.minimum(diff, wrap_diff)
                self._distance_matrix = self.xp.sum(min_diff ** 2, axis=2)
            else:
                # Regular Euclidean distances
                diff = coords_expanded1 - coords_expanded2
                self._distance_matrix = self.xp.sum(diff ** 2, axis=2)
            
            if self.verbose:
                matrix_size = self._distance_matrix.nbytes / (1024**2)  # MB
                logger.info(f"Distance matrix computed and stored on GPU: {matrix_size:.1f} MB")
    
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
        
        # Precompute influence maps for every scheduled radius
        if self._use_sparse_influence:
            self._precompute_sparse_influence_maps(exact_radii, influence_params)
        else:
            self._precompute_optimized_influence_maps(exact_radii, influence_params)
        
        if self.verbose:
            logger.info(f"Rectangular topology precomputation complete")
    
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
        Uses O(1) lookup with radius deduplication.
        
        Args:
            radius: Current neighborhood radius
            influence_function: Type of influence function ('gaussian', 'bubble', 'mexican_hat')
            
        Returns:
            weight_influence_map: Precomputed continuous influence matrix for weight updates
        """
        # Validate parameters via parent interface
        super().get_precomputed_influence_matrix(radius, influence_function)

        # Lazily initialize radius mapping and influence cache if they have
        # not been created via precompute_topology_data. Some call sites in
        # tests invoke this method directly without precomputation.
        if not self._influence_map_cache:
            # som_params is unused by precompute_topology_data; provide None
            self.precompute_topology_data(som_params=None, radii_list=[radius])
        
        cached_radius = float(radius)
        cached_key = (cached_radius, influence_function)
        
        # Fast O(1) lookup
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
            logger.info(f"Rectangular grid topology received {len(radii_list)} pre-computed radii")
    
    def _precompute_optimized_influence_maps(self, radii: List[float], influence_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Precompute influence maps for all scheduled radii and the specified influence function.
        
        Args:
            radii: List of radii to precompute
            influence_params: Optional parameters for influence kernels
        """
        # Precompute distance matrix first (needed for influence calculations)
        self._precompute_distance_matrix()
        
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
            logger.info(f"Precomputing {self.influence_function} influence maps for {len(radii)} radii...")
        
        coord_grid = self.get_coordinates()
        
        for radius in radii:
            for func_name, func in influence_functions.items():
                # Calculate influence map using precomputed distance matrix
                # Pass toroidal parameter for grid topology
                is_toroidal = (self.topology_type == "toroidal")
                
                # Convert distance matrix to GPU if needed for calculation
                distance_matrix_for_calc = self._distance_matrix
                if self._use_cpu_storage and isinstance(self._distance_matrix, np.ndarray):
                    distance_matrix_for_calc = cp.asarray(self._distance_matrix)
                
                if func_name in {"gaussian", "mexican_hat"}:
                    influence_map = func(
                        coord_grid,
                        radius,
                        distance_matrix_for_calc,
                        toroidal=is_toroidal,
                        grid_size=self.grid_size,
                        std_coeff=std_coeff,
                        compact_support=compact_support,
                    )
                else:
                    influence_map = func(
                        coord_grid,
                        radius,
                        distance_matrix_for_calc,
                        toroidal=is_toroidal,
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
            logger.info(f"Precomputed {total_matrices} influence maps using {total_memory:.1f} MB of memory")

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
