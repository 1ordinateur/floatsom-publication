"""
PCA-based weight initialization for SOM topologies
"""

import numpy as np
import logging
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def pca_weights_init(data: Union[np.ndarray, 'cp.ndarray'], 
                    grid_shape: Tuple[int, ...], 
                    xp = None,
                    grid_coords: Union[np.ndarray, 'cp.ndarray'] = None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Initialize SOM weights to span the first two principal components.
    
    This initialization doesn't depend on random processes and
    makes the training process converge faster. This method follows
    the MiniSom approach of creating a regular grid in PC space.
    
    It is strongly recommended to normalize the data before initializing
    the weights and use the same normalization for the training data.
    
    Parameters:
    -----------
    data : array_like
        Input data of shape (n_samples, n_features)
    grid_shape : tuple
        Shape of the SOM grid (height, width) or (nodes,) for 1D
    xp : module
        Array module (numpy or cupy)
    grid_coords : array_like, optional
        Grid coordinates for each node. If provided, enables spatially-aware
        initialization (e.g., for hexagonal grids)
    
    Returns:
    --------
    weights : array_like
        Initialized weights of shape (height, width, n_features) or (nodes, n_features)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA weights initialization. "
                         "Install with: pip install scikit-learn")
    
    if xp is None:
        xp = np
    
    if data is None or data.size == 0:
        raise ValueError("Data required for PCA weights initialization")
    
    # Handle different grid shapes
    if len(grid_shape) == 1:
        grid_h = grid_shape[0]
        grid_w = 1
        is_1d = True
    elif len(grid_shape) == 2:
        grid_h, grid_w = grid_shape
        is_1d = False
    else:
        raise ValueError("Grid shape must be 1D or 2D")
    
    n_features = data.shape[1]
    
    if n_features == 1:
        raise ValueError("The data needs at least 2 features for PCA initialization")
    
    # Convert to numpy for sklearn if needed
    if hasattr(data, 'get'):  # CuPy array
        data_np = data.get()
    else:
        data_np = data
    
    # Fit PCA and get principal components
    pca = PCA(n_components=2)
    pca.fit(data_np)
    
    # Get the principal components (eigenvectors)
    pc = pca.components_  # shape: (2, n_features)
    
    # Initialize weights array
    if is_1d:
        weights = xp.zeros((grid_h, n_features))
        # For 1D, only use the first principal component
        for i, c1 in enumerate(np.linspace(-1, 1, grid_h)):
            weights[i] = xp.asarray(c1 * pc[0])
    else:
        weights = xp.zeros((grid_h, grid_w, n_features))
        
        if grid_coords is not None:
            # Spatially-aware initialization using provided coordinates
            # Convert to numpy if needed for calculations
            if hasattr(grid_coords, 'get'):
                coords_np = grid_coords.get()
            else:
                coords_np = grid_coords
            
            # Normalize coordinates to [-1, 1] range
            coords_norm = coords_np.copy()
            coords_min = coords_norm.min(axis=0)
            coords_max = coords_norm.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1  # Avoid division by zero
            coords_norm = 2 * (coords_norm - coords_min) / coords_range - 1
            
            # Assign weights based on normalized spatial coordinates
            for idx, (cx, cy) in enumerate(coords_norm):
                row = idx // grid_w
                col = idx % grid_w
                # Use spatial coordinates as coefficients for PCs
                weights[row, col] = xp.asarray(cx * pc[0] + cy * pc[1])
        else:
            # Original rectangular grid initialization
            for i, c2 in enumerate(np.linspace(-1, 1, grid_h)):
                for j, c1 in enumerate(np.linspace(-1, 1, grid_w)):
                    # Linear combination of the two principal components
                    weights[i, j] = xp.asarray(c1 * pc[0] + c2 * pc[1])
    
    return weights


def pca_sampling_init(data: Union[np.ndarray, 'cp.ndarray'], 
                     grid_shape: Tuple[int, ...], 
                     n_iterations: int = 1,
                     xp = None,
                     grid_coords: Union[np.ndarray, 'cp.ndarray'] = None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Initialize SOM weights by:
    1. Randomly sampling n points from the data
    2. Projecting to PC1-PC2 space
    3. Ordering them in a grid-like fashion
    4. Averaging over multiple iterations
    
    Parameters:
    -----------
    data : array_like
        Input data of shape (n_samples, n_features)
    grid_shape : tuple
        Shape of the SOM grid (height, width) or (nodes,) for 1D
    n_iterations : int
        Number of sampling iterations to average
    xp : module
        Array module (numpy or cupy)
    grid_coords : array_like, optional
        Grid coordinates for each node. If provided, enables spatially-aware
        initialization (e.g., for hexagonal grids)
    
    Returns:
    --------
    weights : array_like
        Initialized weights of shape (height, width, n_features) or (nodes, n_features)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA sampling initialization. "
                         "Install with: pip install scikit-learn")
    
    if xp is None:
        xp = np
    
    if data is None or data.size == 0:
        raise ValueError("Data required for PCA sampling initialization")
    
    # Handle different grid shapes
    if len(grid_shape) == 1:
        grid_h = grid_shape[0]
        grid_w = 1
        is_1d = True
    elif len(grid_shape) == 2:
        grid_h, grid_w = grid_shape
        is_1d = False
    else:
        raise ValueError("Grid shape must be 1D or 2D")
    
    n_nodes = grid_h * grid_w
    n_features = data.shape[1]
    
    # Convert to numpy for sklearn if needed
    if hasattr(data, 'get'):  # CuPy array
        data_np = data.get()
    else:
        data_np = data
    
    # Fit PCA once on all data for consistent projection
    pca = PCA(n_components=2)
    pca.fit(data_np)
    
    # Initialize accumulator for averaging
    if is_1d:
        accumulated_weights = xp.zeros((grid_h, n_features))
    else:
        accumulated_weights = xp.zeros((grid_h, grid_w, n_features))
    
    for iteration in range(n_iterations):
        # Step 1: Random sample from data (without replacement)
        indices = np.random.choice(len(data_np), n_nodes, replace=False)
        samples = data_np[indices]
        
        # Step 2: Project samples to PC1-PC2 space
        samples_2d = pca.transform(samples)
        
        # Step 3: Grid assignment based on spatial or lexicographic ordering
        if grid_coords is not None:
            # Spatially-aware assignment: match samples to grid positions
            # Convert coordinates to numpy if needed
            if hasattr(grid_coords, 'get'):
                coords_np = grid_coords.get()
            else:
                coords_np = grid_coords
            
            # Normalize both sample projections and grid coordinates
            samples_norm = samples_2d.copy()
            samples_norm -= samples_norm.min(axis=0)
            if samples_norm.max(axis=0).max() > 0:
                samples_norm /= samples_norm.max(axis=0)
            
            coords_norm = coords_np.copy()
            coords_min = coords_norm.min(axis=0)
            coords_max = coords_norm.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1
            coords_norm = (coords_norm - coords_min) / coords_range
            
            # Assign samples to nearest grid positions
            from scipy.spatial.distance import cdist
            distances = cdist(samples_norm, coords_norm)
            assignment = distances.argmin(axis=1)
            
            for sample_idx, grid_idx in enumerate(assignment):
                if is_1d:
                    accumulated_weights[grid_idx] += xp.asarray(samples[sample_idx])
                else:
                    row = grid_idx // grid_w
                    col = grid_idx % grid_w
                    accumulated_weights[row, col] += xp.asarray(samples[sample_idx])
        else:
            # Original lexicographic assignment
            samples_norm = samples_2d.copy()
            samples_norm -= samples_norm.min(axis=0)
            if samples_norm.max(axis=0).max() > 0:
                samples_norm /= samples_norm.max(axis=0)
            
            # Sort by PC2 first (rows), then PC1 (columns)
            sorted_indices = np.lexsort((samples_norm[:, 0], samples_norm[:, 1]))
            
            # Step 4: Assign samples to grid positions
            if is_1d:
                for idx, sample_idx in enumerate(sorted_indices):
                    accumulated_weights[idx] += xp.asarray(samples[sample_idx])
            else:
                for idx, sample_idx in enumerate(sorted_indices):
                    row = idx // grid_w
                    col = idx % grid_w
                    accumulated_weights[row, col] += xp.asarray(samples[sample_idx])
    
    # Step 5: Average across iterations
    final_weights = accumulated_weights / n_iterations
    
    return final_weights


def pca_sampling_init_snake(data: Union[np.ndarray, 'cp.ndarray'], 
                           grid_shape: Tuple[int, ...], 
                           n_iterations: int = 10,
                           xp = None,
                           grid_coords: Union[np.ndarray, 'cp.ndarray'] = None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Alternative implementation using snake/boustrophedon pattern
    for better edge continuity.
    
    Parameters:
    -----------
    data : array_like
        Input data of shape (n_samples, n_features)
    grid_shape : tuple
        Shape of the SOM grid (height, width) or (nodes,) for 1D
    n_iterations : int
        Number of sampling iterations to average
    xp : module
        Array module (numpy or cupy)
    grid_coords : array_like, optional
        Grid coordinates for each node. If provided, enables spatially-aware
        initialization (e.g., for hexagonal grids)
    
    Returns:
    --------
    weights : array_like
        Initialized weights of shape (height, width, n_features) or (nodes, n_features)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA sampling initialization. "
                         "Install with: pip install scikit-learn")
    
    if xp is None:
        xp = np
    
    if data is None or data.size == 0:
        raise ValueError("Data required for PCA sampling initialization")
    
    # Handle different grid shapes
    if len(grid_shape) == 1:
        grid_h = grid_shape[0]
        grid_w = 1
        is_1d = True
    elif len(grid_shape) == 2:
        grid_h, grid_w = grid_shape
        is_1d = False
    else:
        raise ValueError("Grid shape must be 1D or 2D")
    
    n_nodes = grid_h * grid_w
    n_features = data.shape[1]
    
    # Convert to numpy for sklearn if needed
    if hasattr(data, 'get'):  # CuPy array
        data_np = data.get()
    else:
        data_np = data
    
    # Fit PCA once
    pca = PCA(n_components=2)
    pca.fit(data_np)
    
    # Initialize accumulator
    if is_1d:
        accumulated_weights = xp.zeros((grid_h, n_features))
    else:
        accumulated_weights = xp.zeros((grid_h, grid_w, n_features))
    
    for iteration in range(n_iterations):
        # Sample and project
        indices = np.random.choice(len(data_np), n_nodes, replace=False)
        samples = data_np[indices]
        samples_2d = pca.transform(samples)
        
        # Assignment based on spatial or snake pattern
        if grid_coords is not None:
            # Use same spatial assignment as pca_sampling_init
            if hasattr(grid_coords, 'get'):
                coords_np = grid_coords.get()
            else:
                coords_np = grid_coords
            
            # Normalize both sample projections and grid coordinates
            samples_norm = samples_2d.copy()
            samples_norm -= samples_norm.min(axis=0)
            if samples_norm.max(axis=0).max() > 0:
                samples_norm /= samples_norm.max(axis=0)
            
            coords_norm = coords_np.copy()
            coords_min = coords_norm.min(axis=0)
            coords_max = coords_norm.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1
            coords_norm = (coords_norm - coords_min) / coords_range
            
            # Assign samples to nearest grid positions
            from scipy.spatial.distance import cdist
            distances = cdist(samples_norm, coords_norm)
            assignment = distances.argmin(axis=1)
            
            for sample_idx, grid_idx in enumerate(assignment):
                if is_1d:
                    accumulated_weights[grid_idx] += xp.asarray(samples[sample_idx])
                else:
                    row = grid_idx // grid_w
                    col = grid_idx % grid_w
                    accumulated_weights[row, col] += xp.asarray(samples[sample_idx])
        else:
            # Original snake pattern
            samples_norm = samples_2d.copy()
            samples_norm -= samples_norm.min(axis=0)
            if samples_norm.max(axis=0).max() > 0:
                samples_norm /= samples_norm.max(axis=0)
            
            if is_1d:
                # For 1D, just sort by PC1
                col_order = samples_norm[:, 0].argsort()
                for idx, sample_idx in enumerate(col_order):
                    accumulated_weights[idx] += xp.asarray(samples[sample_idx])
            else:
                # Sort by PC2 (rows)
                row_order = samples_norm[:, 1].argsort()
                
                # Assign with snake pattern
                for row in range(grid_h):
                    # Get samples for this row
                    row_indices = row_order[row * grid_w:(row + 1) * grid_w]
                    row_samples_2d = samples_norm[row_indices]
                    row_samples = samples[row_indices]
                    
                    # Sort by PC1, alternating direction
                    if row % 2 == 0:
                        col_order = row_samples_2d[:, 0].argsort()
                    else:
                        col_order = row_samples_2d[:, 0].argsort()[::-1]
                    
                    # Assign to grid
                    for col, sample_idx in enumerate(col_order):
                        accumulated_weights[row, col] += xp.asarray(row_samples[sample_idx])
    
    final_weights = accumulated_weights / n_iterations
    return final_weights


def pca_density_init(data: Union[np.ndarray, 'cp.ndarray'], 
                     grid_shape: Tuple[int, ...], 
                     xp=None, 
                     verbose: bool = False,
                     grid_coords: Union[np.ndarray, 'cp.ndarray'] = None,
                     chunk_size: Optional[int] = None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Initialize SOM weights using density-based PCA initialization.
    
    This method combines data density awareness with PCA structure preservation:
    1. Estimates data density using KDE on a 10K subsample
    2. Samples actual data points with density-weighted probabilities
    3. Orders sampled points using PCA projection
    4. Assigns directly to grid positions without averaging
    
    This avoids the mean reversion problem of traditional averaging methods
    while placing nodes where data actually exists.
    
    Parameters:
    -----------
    data : array_like
        Input data of shape (n_samples, n_features)
    grid_shape : tuple
        Shape of the SOM grid (height, width) or (nodes,) for 1D
    xp : module
        Array module (numpy or cupy)
    verbose : bool
        Whether to print progress information
    grid_coords : array_like, optional
        Grid coordinates for each node. If provided, enables spatially-aware
        initialization (e.g., for hexagonal grids)
        
    Returns:
    --------
    weights : array_like
        Initialized weights of shape (height, width, n_features) or (nodes, n_features)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for density-based PCA initialization. "
                         "Install with: pip install scikit-learn")
    
    if xp is None:
        xp = np
    
    if data is None or data.size == 0:
        raise ValueError("Data required for density-based PCA initialization")
    
    # Handle different grid shapes
    if len(grid_shape) == 1:
        grid_h = grid_shape[0]
        grid_w = 1
        is_1d = True
    elif len(grid_shape) == 2:
        grid_h, grid_w = grid_shape
        is_1d = False
    else:
        raise ValueError("Grid shape must be 1D or 2D")
    
    n_nodes = grid_h * grid_w
    n_samples, n_features = data.shape
    
    if n_samples < n_nodes:
        raise ValueError(f"Not enough data points ({n_samples}) for grid size ({n_nodes})")
    
    # Convert to numpy for sklearn operations
    if hasattr(data, 'get'):  # CuPy array
        data_np = data.get()
    else:
        data_np = data
    
    if verbose:
        logger.info(f"Density-based PCA initialization: {n_samples} samples -> {n_nodes} nodes")
    
    # Cap very large inputs by random subsampling to keep density scoring tractable.
    max_density_samples = 100000
    if n_samples > max_density_samples:
        sample_idx = np.random.choice(n_samples, max_density_samples, replace=False)
        data_np = data_np[sample_idx]
        n_samples = data_np.shape[0]
        if verbose:
            logger.info(
                "Density init subsampled data from %d to %d samples",
                data.shape[0],
                n_samples,
            )

    # Step 1: Fit PCA for ordering (subsample for very large datasets)
    pca_sample_size = min(50000, n_samples)
    if n_samples > pca_sample_size:
        pca_sample_idx = np.random.choice(n_samples, pca_sample_size, replace=False)
        pca_data = data_np[pca_sample_idx]
    else:
        pca_data = data_np
    
    pca = PCA(n_components=2)
    pca.fit(pca_data)
    
    if verbose:
        logger.info(f"PCA fitted on {len(pca_data)} samples")
    
    # Step 2: Subsample for density estimation (fixed 10K as specified)
    kde_sample_size = min(10000, n_samples)
    if n_samples > kde_sample_size:
        kde_sample_idx = np.random.choice(n_samples, kde_sample_size, replace=False)
        kde_data = data_np[kde_sample_idx]
    else:
        kde_data = data_np
        kde_sample_idx = np.arange(n_samples)
    
    if verbose:
        logger.info(f"KDE estimation using {len(kde_data)} samples")
    
    # Step 3: Estimate data density using KDE
    kde = KernelDensity(bandwidth='scott')
    kde.fit(kde_data)
    
    # Step 4: Compute density for all data points in the capped working set.
    log_density = kde.score_samples(data_np)
    
    # Convert to probabilities (numerical stability)
    density_weights = np.exp(log_density - log_density.max())
    density_weights = density_weights / density_weights.sum()
    
    if verbose:
        logger.info(f"Density range: {density_weights.min():.6f} to {density_weights.max():.6f}")
    
    # Step 5: Density-weighted sampling WITHOUT replacement
    selected_indices = np.random.choice(
        n_samples, 
        size=n_nodes, 
        replace=False,
        p=density_weights
    )
    selected_points = data_np[selected_indices]
    
    # Step 6: Order by PCA projection or spatial assignment
    selected_projected = pca.transform(selected_points)
    
    # Step 7: Assign to grid
    if is_1d:
        weights = xp.zeros((grid_h, n_features))
        if grid_coords is not None:
            # Spatial assignment for 1D - use PC1 directly
            pc1_values = selected_projected[:, 0]
            sorted_order = pc1_values.argsort()
            for idx, point_idx in enumerate(sorted_order):
                weights[idx] = xp.asarray(selected_points[point_idx])
        else:
            # Original lexicographic ordering for 1D
            sorted_order = selected_projected[:, 0].argsort()
            for idx, point_idx in enumerate(sorted_order):
                weights[idx] = xp.asarray(selected_points[point_idx])
    else:
        weights = xp.zeros((grid_h, grid_w, n_features))
        
        if grid_coords is not None:
            # Spatially-aware assignment using coordinates
            if hasattr(grid_coords, 'get'):
                coords_np = grid_coords.get()
            else:
                coords_np = grid_coords
            
            # Normalize both projections and coordinates
            proj_norm = selected_projected.copy()
            proj_norm -= proj_norm.min(axis=0)
            if proj_norm.max(axis=0).max() > 0:
                proj_norm /= proj_norm.max(axis=0)
            
            coords_norm = coords_np.copy()
            coords_min = coords_norm.min(axis=0)
            coords_max = coords_norm.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1
            coords_norm = (coords_norm - coords_min) / coords_range
            
            # Assign to nearest grid positions
            from scipy.spatial.distance import cdist
            distances = cdist(proj_norm, coords_norm)
            assignment = distances.argmin(axis=1)
            
            for point_idx, grid_idx in enumerate(assignment):
                row = grid_idx // grid_w
                col = grid_idx % grid_w
                weights[row, col] = xp.asarray(selected_points[point_idx])
        else:
            # Original lexicographic ordering
            sorted_order = np.lexsort((
                selected_projected[:, 0],  # PC1 (secondary sort)
                selected_projected[:, 1]   # PC2 (primary sort)
            ))
            
            for idx, point_idx in enumerate(sorted_order):
                row = idx // grid_w
                col = idx % grid_w
                weights[row, col] = xp.asarray(selected_points[point_idx])
    
    if verbose:
        logger.info("Density-based initialization complete")
    
    return weights


class PCAInitializer:
    """
    PCA-based initialization for SOM weights
    Initializes weights along principal components of the data
    """
    
    def __init__(self, n_components=2):
        """
        Initialize PCA initializer
        
        Args:
            n_components: Number of principal components to use
        """
        self.n_components = n_components
        self.pca = None
        
    def initialize_weights_pca(self, shape, data, topology_type="grid"):
        """
        Initialize SOM weights using PCA
        
        Args:
            shape: SOM dimensions (height, width)
            data: Input dataset for PCA analysis
            topology_type: Type of topology ("grid", "mst", or "rng")
            
        Returns:
            PCA-initialized weights
        """
        if topology_type == "grid":
            return pca_weights_init(data, shape)
        elif topology_type in {"mst", "rng"}:
            # Graph topologies use a linear node arrangement for PCA initialization.
            return pca_weights_init(data, (shape[0] * shape[1],))
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
    
    def fit_pca(self, data):
        """
        Fit PCA to data
        
        Args:
            data: Input dataset
            
        Returns:
            Fitted PCA components
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA initialization")
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(data)
        return self.pca.components_
    
    def get_explained_variance_ratio(self):
        """
        Get explained variance ratio from fitted PCA
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.explained_variance_ratio_
