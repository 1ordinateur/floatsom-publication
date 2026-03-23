"""
Topology preservation metrics for SOM evaluation.
"""

import numpy as np
import cupy as cp
from typing import Optional, Union, Tuple, Any
from scipy.spatial import cKDTree
from ..base import Metric


class TopographicError(Metric):
    """
    Topographic Error metric for SOM evaluation.
    
    Measures the proportion of data points for which the first and second
    best matching units (BMUs) are not adjacent in the SOM grid.
    
    Range: [0, 1], lower is better
    """
    
    name = "topographic_error"
    
    def __init__(self, use_gpu: bool = True, chunk_size: int = 5000):
        """
        Initialize TopographicError metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            chunk_size: Number of samples to process at once
        """
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.chunk_size = chunk_size
        self._adjacency_cache = {}
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som, data: Union[np.ndarray, cp.ndarray], **kwargs) -> float:
        """
        Compute topographic error using vectorized operations.
        
        Args:
            som: Trained SOM instance with get_bmu() method
            data: Input data
            
        Returns:
            Topographic error value
        """
        # Convert data to appropriate format
        if isinstance(data, np.ndarray):
            data = cp.array(data)
        
        weights = som.get_weights()
        if isinstance(weights, np.ndarray):
            weights = cp.array(weights)
        
        n_samples = len(data)
        
        # Get grid dimensions from SOM object
        if hasattr(som, 'grid_size'):
            n_cols = som.grid_size
            n_rows = n_cols
        elif hasattr(som, 'shape'):
            n_rows, n_cols = som.shape
        elif hasattr(som, 'n_rows') and hasattr(som, 'n_cols'):
            n_rows, n_cols = som.n_rows, som.n_cols
        else:
            # Fallback: try to infer from weights shape
            if weights.ndim == 3:
                n_rows, n_cols = weights.shape[:2]
            else:
                # Assume square grid
                n_neurons = weights.shape[0]
                n_rows = n_cols = int(np.sqrt(n_neurons))
        
        n_neurons = n_rows * n_cols
        weights_flat = weights.reshape(n_neurons, -1)
        xp = cp if self.use_gpu else np
        
        # Get or create adjacency matrix
        grid_key = (n_rows, n_cols)
        if grid_key not in self._adjacency_cache:
            self._adjacency_cache[grid_key] = self._create_adjacency_matrix(n_rows, n_cols)
        adjacency_matrix = self._adjacency_cache[grid_key]
        
        total_errors = 0
        
        # Process in batches for memory efficiency
        for i in range(0, n_samples, self.chunk_size):
            batch = data[i:i+self.chunk_size]
            current_batch_size = len(batch)
            
            # Vectorized distance computation
            distances = xp.linalg.norm(
                weights_flat[xp.newaxis, :, :] - batch[:, xp.newaxis, :], 
                axis=2
            )
            
            # Get top 2 BMU indices for all samples in batch
            if distances.shape[1] >= 2:
                top2_indices = xp.argpartition(distances, 1, axis=1)[:, :2]
                
                # Get the actual indices of the two smallest values
                rows = xp.arange(current_batch_size)[:, xp.newaxis]
                top2_distances = distances[rows, top2_indices]
                sorted_idx = xp.argsort(top2_distances, axis=1)
                top2_indices = top2_indices[rows, sorted_idx].squeeze()
                
                # Extract BMU indices
                bmu1_indices = top2_indices[:, 0]
                bmu2_indices = top2_indices[:, 1]
                
                # Vectorized adjacency check
                is_adjacent = adjacency_matrix[bmu1_indices, bmu2_indices]
                batch_errors = xp.sum(~is_adjacent)
                
                if self.use_gpu:
                    total_errors += int(batch_errors.get())
                else:
                    total_errors += int(batch_errors)
        
        return total_errors / n_samples
    
    def _get_two_best_bmus(self, som, sample) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get the two best matching units for a sample."""
        # This will need to be adapted based on the SOM implementation
        # For now, assume som has a method to get distances to all neurons
        
        distances = som.get_bmu_distances(sample)
        
        # Get indices of two smallest distances (always use numpy)
        if isinstance(distances, cp.ndarray):
            distances = distances.get()
        
        distances = np.asarray(distances)
        flat_indices = np.argpartition(distances.flatten(), min(2, distances.size))[:2]
        indices = np.unravel_index(flat_indices, distances.shape)
        bmu1 = (int(indices[0][0]), int(indices[1][0]))
        bmu2 = (int(indices[0][1]), int(indices[1][1]))
        
        return bmu1, bmu2
    
    def _are_adjacent(self, som, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent in the SOM grid."""
        # Check if positions are adjacent (including diagonal neighbors)
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Adjacent if distance is 1 in manhattan distance or diagonal
        return (dx <= 1 and dy <= 1) and not (dx == 0 and dy == 0)
    
    def _create_adjacency_matrix(self, n_rows: int, n_cols: int):
        """Create boolean adjacency matrix for the SOM grid using vectorized operations."""
        n_neurons = n_rows * n_cols
        xp = cp if self.use_gpu else np
        
        # Create grid coordinates for all neurons
        neuron_rows = xp.arange(n_neurons) // n_cols
        neuron_cols = xp.arange(n_neurons) % n_cols
        
        # Compute pairwise distances
        row_diff = xp.abs(neuron_rows[:, xp.newaxis] - neuron_rows[xp.newaxis, :])
        col_diff = xp.abs(neuron_cols[:, xp.newaxis] - neuron_cols[xp.newaxis, :])
        
        # Adjacent if both row and col differences are <= 1, but not both 0
        adjacency = (row_diff <= 1) & (col_diff <= 1) & ((row_diff != 0) | (col_diff != 0))
        
        return adjacency


class Trustworthiness(Metric):
    """
    Trustworthiness metric adapted for SOM evaluation.
    
    Measures how well k-nearest neighbors in the original space
    are preserved in the SOM representation.
    
    Range: [0, 1], higher is better
    """
    
    name = "trustworthiness"
    
    def __init__(self, k: int = 7):
        """
        Initialize Trustworthiness metric.
        
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
    
    def compute(self, som, data: Union[np.ndarray, cp.ndarray], **kwargs) -> float:
        """
        Compute trustworthiness metric.
        
        Args:
            som: Trained SOM instance
            data: Input data
            
        Returns:
            Trustworthiness value
        """
        # Convert to numpy for sklearn compatibility
        if isinstance(data, cp.ndarray):
            data_np = data.get()
        else:
            data_np = data
        
        # Get SOM embedding (BMU positions)
        som_embedding = self._get_som_embedding(som, data_np)

        return self._compute_trustworthiness_custom(data_np, som_embedding)
    
    def _get_som_embedding(self, som, data: np.ndarray) -> np.ndarray:
        """Get 2D SOM grid positions for each data point."""
        n_samples = len(data)
        embedding = np.zeros((n_samples, 2))
        
        for i, sample in enumerate(data):
            # Get BMU position
            if hasattr(som, 'get_bmu'):
                bmu_pos = som.get_bmu(sample)
            else:
                # Fallback: compute BMU manually
                weights = som.weights if hasattr(som, 'weights') else som.get_weights()
                
                # Ensure sample is numpy array
                if isinstance(sample, cp.ndarray):
                    sample_np = sample.get()
                else:
                    sample_np = np.asarray(sample)
                
                # Ensure weights is numpy array
                if isinstance(weights, cp.ndarray):
                    weights_np = weights.get()
                else:
                    weights_np = np.asarray(weights)
                
                distances = np.linalg.norm(weights_np - sample_np, axis=-1)
                bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
                bmu_pos = bmu_idx
            
            embedding[i] = bmu_pos
        
        return embedding
    
    def _compute_trustworthiness_custom(self, X: np.ndarray, X_embedded: np.ndarray) -> float:
        """Custom trustworthiness computation."""
        n_samples = X.shape[0]
        
        # Compute distance matrices
        dist_X = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        dist_embedded = np.linalg.norm(X_embedded[:, None] - X_embedded[None, :], axis=2)
        
        # Get k-nearest neighbors in embedded space
        embedded_nn = np.argsort(dist_embedded, axis=1)[:, 1:self.k+1]
        
        # Compute trustworthiness
        trustworthiness_sum = 0.0
        
        for i in range(n_samples):
            # Get ranks in original space
            original_ranks = np.argsort(dist_X[i])
            rank_dict = {idx: rank for rank, idx in enumerate(original_ranks)}
            
            # Check embedded neighbors
            for j in embedded_nn[i]:
                rank_j = rank_dict[j]
                if rank_j > self.k:
                    trustworthiness_sum += rank_j - self.k
        
        # Normalize
        normalizer = n_samples * self.k * (2 * n_samples - 3 * self.k - 1) / 2
        return 1 - (2 * trustworthiness_sum) / normalizer


class NeighborhoodPreservation(Metric):
    """
    Neighborhood Preservation metric for SOM evaluation.
    
    Measures the overlap between k-nearest neighbors in the original
    space and k-nearest neighbors in the SOM representation.
    
    Range: [0, 1], higher is better
    """
    
    name = "neighborhood_preservation"
    
    def __init__(self, k: int = 7, use_approximate: bool = True, 
                 sample_size: int = 5000, chunk_size: int = 5000):
        """
        Initialize NeighborhoodPreservation metric.
        
        Args:
            k: Number of nearest neighbors to consider
            use_approximate: Use sampling for very large datasets
            sample_size: Maximum number of samples for approximation
            chunk_size: Chunk size for BMU computation
        """
        self.k = k
        self.use_approximate = use_approximate
        self.sample_size = sample_size
        self.chunk_size = chunk_size
    
    def compute(self, som, data: Union[np.ndarray, cp.ndarray], **kwargs) -> float:
        """
        Compute neighborhood preservation with memory optimization.
        
        Args:
            som: Trained SOM instance
            data: Input data
            
        Returns:
            Neighborhood preservation value
        """
        # Convert to numpy (KDTree requires numpy)
        if isinstance(data, cp.ndarray):
            data_np = data.get()
        else:
            data_np = data
        
        n_samples = len(data_np)
        
        # Use sampling for very large datasets
        if self.use_approximate and n_samples > self.sample_size:
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            data_sample = data_np[indices]
        else:
            data_sample = data_np
        
        # Get SOM embedding using batched computation
        som_embedding = self._get_som_embedding_batched(som, data_sample)
        
        return self._compute_preservation_kdtree(data_sample, som_embedding)
    
    def _get_som_embedding_batched(self, som, data: np.ndarray) -> np.ndarray:
        """Get 2D SOM grid positions using batched operations."""
        weights = som.get_weights()
        
        # Get grid dimensions from SOM object
        if hasattr(som, 'grid_size'):
            n_cols = som.grid_size
            n_rows = n_cols
        elif hasattr(som, 'shape'):
            n_rows, n_cols = som.shape
        elif hasattr(som, 'n_rows') and hasattr(som, 'n_cols'):
            n_rows, n_cols = som.n_rows, som.n_cols
        else:
            # Fallback: try to infer from weights shape
            if weights.ndim == 3:
                n_rows, n_cols = weights.shape[:2]
            else:
                # Assume square grid
                n_neurons = weights.shape[0]
                n_rows = n_cols = int(np.sqrt(n_neurons))
        
        n_neurons = n_rows * n_cols
        weights_flat = weights.reshape(n_neurons, -1)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(data), self.chunk_size):
            batch = data[i:i+self.chunk_size]
            
            # Compute distances for batch
            distances = np.linalg.norm(
                weights[np.newaxis, :, :] - batch[:, np.newaxis, :], 
                axis=2
            )
            bmu_flat_indices = np.argmin(distances, axis=1)
            
            # Convert flat indices to 2D coordinates
            bmu_rows = bmu_flat_indices // som.grid_size
            bmu_cols = bmu_flat_indices % som.grid_size
            
            embeddings.append(np.column_stack([bmu_rows, bmu_cols]))
        
        return np.vstack(embeddings).astype(np.float64)
    
    def _compute_preservation_kdtree(self, data: np.ndarray, embedding: np.ndarray) -> float:
        """Compute preservation using efficient KD-tree neighbor search."""
        # Build KD-trees for efficient neighbor search
        tree_original = cKDTree(data)
        tree_som = cKDTree(embedding)
        
        # Find k-nearest neighbors (k+1 to exclude self)
        _, nn_original = tree_original.query(data, k=self.k+1)
        _, nn_som = tree_som.query(embedding, k=self.k+1)
        
        # Remove self from neighbors (first column)
        nn_original = nn_original[:, 1:]
        nn_som = nn_som[:, 1:]
        
        # Vectorized overlap computation using broadcasting
        n_samples = len(data)
        total_overlap = 0
        
        # Process in chunks for memory efficiency
        chunk_size = 1000
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            for j in range(i, end_idx):
                # Use numpy intersect1d for efficient set intersection
                overlap = len(np.intersect1d(nn_original[j], nn_som[j]))
                total_overlap += overlap
        
        return total_overlap / (n_samples * self.k)
    


class DistortionMeasure(Metric):
    """
    Distortion Measure (DM) metric for SOM evaluation.
    
    Combines quantization error with neighborhood structure using 
    distance-based weighting factor for the quantization error.
    
    Formula: DM = (1/k) * Σᵢ Σⱼ hᵦᵢⱼ * ||xᵢ - wⱼ||²
    
    Range: [0, ∞), lower is better
    """
    
    name = "distortion_measure"
    
    def __init__(self, use_gpu: bool = True, sigma: float = 1.0, chunk_size: int = 1000):
        """
        Initialize DistortionMeasure metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            sigma: Neighborhood function parameter (controls spread)
            chunk_size: Number of samples to process at once
        """
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.sigma = sigma
        self.chunk_size = chunk_size
        self._grid_distance_cache = {}
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som, data: Union[np.ndarray, cp.ndarray], **kwargs) -> float:
        """
        Compute distortion measure using fully vectorized operations.
        
        Args:
            som: Trained SOM instance
            data: Input data
            
        Returns:
            Distortion measure value
        """
        # Convert data to appropriate format
        if isinstance(data, np.ndarray):
            data = cp.array(data)
        
        weights = som.get_weights()
        if isinstance(weights, np.ndarray):
            weights = cp.array(weights)
        
        # Get grid dimensions from SOM object
        if hasattr(som, 'grid_size'):
            n_cols = som.grid_size
            n_rows = n_cols
        elif hasattr(som, 'shape'):
            n_rows, n_cols = som.shape
        elif hasattr(som, 'n_rows') and hasattr(som, 'n_cols'):
            n_rows, n_cols = som.n_rows, som.n_cols
        else:
            # Fallback: try to infer from weights shape
            if weights.ndim == 3:
                n_rows, n_cols = weights.shape[:2]
            else:
                # Assume square grid
                n_neurons = weights.shape[0]
                n_rows = n_cols = int(np.sqrt(n_neurons))
        
        n_neurons = n_rows * n_cols
        weights_flat = weights.reshape(n_neurons, -1)
        xp = cp if self.use_gpu else np
        
        # Get or create grid distance matrix
        grid_key = (som.grid_size, som.grid_size)
        if grid_key not in self._grid_distance_cache:
            self._grid_distance_cache[grid_key] = self._compute_grid_distances(som.grid_size, som.grid_size)
        grid_distances = self._grid_distance_cache[grid_key]
        
        total_distortion = 0.0
        n_samples = len(data)
        
        # Process in batches
        for i in range(0, n_samples, self.chunk_size):
            batch = data[i:i+self.chunk_size]
            
            # Compute distances to all neurons for the batch
            # Shape: (batch_size, n_neurons)
            distances = xp.linalg.norm(
                weights[xp.newaxis, :, :] - batch[:, xp.newaxis, :], 
                axis=2
            )
            
            # Find BMUs for the batch
            bmu_indices = xp.argmin(distances, axis=1)
            
            # Compute neighborhood values for each sample's BMU
            # Shape: (, n_neurons)
            h_values = xp.exp(
                -(grid_distances[bmu_indices]**2) / (2 * self.sigma**2)
            )
            
            # Compute weighted errors
            weighted_errors = xp.sum(h_values * (distances**2), axis=1)
            total_distortion += xp.sum(weighted_errors)
        
        # Average over all data points
        distortion_measure = total_distortion / n_samples
        
        if self.use_gpu and isinstance(distortion_measure, cp.ndarray):
            distortion_measure = float(distortion_measure.get())
        
        return float(distortion_measure)
    
    def _get_bmus(self, som, data):
        """Get BMU indices for all data points."""
        if hasattr(som, 'map_vectors'):
            return som.map_vectors(data)
        else:
            # Fallback: compute BMUs manually
            weights = som.get_weights()
            if isinstance(data, cp.ndarray):
                data_cp = data
                weights_cp = cp.array(weights) if isinstance(weights, np.ndarray) else weights
            else:
                data_cp = cp.array(data) if self.use_gpu else data
                weights_cp = cp.array(weights) if self.use_gpu and isinstance(weights, np.ndarray) else weights
            
            # Compute distances and find BMUs
            if self.use_gpu:
                distances = cp.linalg.norm(weights_cp[cp.newaxis, :, :] - data_cp[:, cp.newaxis, :], axis=2)
                bmu_indices = cp.argmin(distances, axis=1)
                return cp.unravel_index(bmu_indices, weights.shape[:2])
            else:
                distances = np.linalg.norm(weights_cp[np.newaxis, :, :] - data_cp[:, np.newaxis, :], axis=2)
                bmu_indices = np.argmin(distances, axis=1)
                return np.unravel_index(bmu_indices, weights.shape[:2])
    
    def _compute_grid_distances(self, n_rows: int, n_cols: int):
        """Pre-compute pairwise distances between grid positions."""
        xp = cp if self.use_gpu else np
        
        # Generate grid coordinates
        i_coords, j_coords = xp.meshgrid(
            xp.arange(n_rows), xp.arange(n_cols), indexing='ij'
        )
        coord_grid = xp.stack([i_coords, j_coords], axis=-1).reshape(-1, 2)
        
        # Compute pairwise distances
        # Shape: (n_neurons, n_neurons)
        grid_distances = xp.linalg.norm(
            coord_grid[:, xp.newaxis, :] - coord_grid[xp.newaxis, :, :], 
            axis=2
        )
        
        return grid_distances


class TopographicFunction(Metric):
    """
    Topographic Function (TF) metric for SOM evaluation.
    
    Measures topology preservation by comparing distance matrices
    from grid space and data connectivity space.
    
    Formula: TF = (1/2k) * Σᵢ (|P₁ᵢ - Q₁ᵢ| + |P₂ᵢ - Q₂ᵢ|)
    
    Range: [0, 1], lower is better (0 = perfect topology preservation)
    """
    
    name = "topographic_function"
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize TopographicFunction metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and cp.cuda.is_available()
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som, data: Union[np.ndarray, cp.ndarray], **kwargs) -> float:
        """
        Compute topographic function.
        
        Args:
            som: Trained SOM instance
            data: Input data
            
        Returns:
            Topographic function value
        """
        # Convert data to appropriate format
        if isinstance(data, np.ndarray):
            data = cp.array(data)
        
        # Compute distance matrices
        R_grid = self._compute_grid_distance_matrix(som)
        R_data = self._compute_data_distance_matrix(som, data)
        
        # Compute topographic function using ranking correlation
        return self._compute_topographic_function(R_grid, R_data, len(data))
    
    def _compute_grid_distance_matrix(self, som):
        """Compute R^grid distance matrix between neurons on the grid."""
        # Get grid coordinates - ensure they match weights dimensions
        weights = som.get_weights()
        n_rows, n_cols = weights.shape[:2]
        
        if hasattr(som, 'coord_grid') and som.coord_grid is not None:
            coord_grid = som.coord_grid
            # Validate dimensions match weights
            if coord_grid.shape[:2] != (n_rows, n_cols):
                # Dimensions don't match, regenerate coord_grid
                coord_grid = None
        else:
            coord_grid = None
            
        if coord_grid is None:
            # Generate rectangular grid coordinates
            i_coords, j_coords = (cp if self.use_gpu else np).meshgrid(
                (cp if self.use_gpu else np).arange(n_rows),
                (cp if self.use_gpu else np).arange(n_cols),
                indexing='ij'
            )
            coord_grid = (cp if self.use_gpu else np).stack([i_coords, j_coords], axis=-1)
        
        if isinstance(coord_grid, np.ndarray) and self.use_gpu:
            coord_grid = cp.array(coord_grid)
        
        # Flatten coordinates
        coords_flat = coord_grid.reshape(-1, 2)
        n_neurons = len(coords_flat)
        
        # Compute pairwise distances
        if self.use_gpu:
            R_grid = cp.linalg.norm(
                coords_flat[:, cp.newaxis, :] - coords_flat[cp.newaxis, :, :], axis=2
            )
        else:
            R_grid = np.linalg.norm(
                coords_flat[:, np.newaxis, :] - coords_flat[np.newaxis, :, :], axis=2
            )
        
        return R_grid
    
    def _compute_data_distance_matrix(self, som, data):
        """Compute R^data distance matrix using connectivity from 1st/2nd winners."""
        # Build connectivity matrix
        C = self._build_connectivity_matrix(som, data)
        
        # Convert connectivity to distance matrix
        # Distance = 1 / (connectivity + small_epsilon) for connected pairs
        # Distance = large_value for unconnected pairs
        epsilon = 1e-8
        large_value = 1000.0
        
        if self.use_gpu:
            R_data = cp.where(C > epsilon, 1.0 / (C + epsilon), large_value)
        else:
            R_data = np.where(C > epsilon, 1.0 / (C + epsilon), large_value)
        
        # Make diagonal zero (distance from neuron to itself)
        if self.use_gpu:
            cp.fill_diagonal(R_data, 0.0)
        else:
            np.fill_diagonal(R_data, 0.0)
        
        return R_data
    
    def _build_connectivity_matrix(self, som, data):
        """Build connectivity matrix C using 1st and 2nd winner relationships."""
        weights = som.get_weights()
        n_neurons = weights.shape[0] * weights.shape[1]
        
        if self.use_gpu:
            C = cp.zeros((n_neurons, n_neurons))
        else:
            C = np.zeros((n_neurons, n_neurons))
        
        for sample in data:
            # Get two best matching units
            bmu1_idx, bmu2_idx = self._get_two_best_bmus(som, sample)
            
            # Convert 2D indices to flat indices if needed
            if isinstance(bmu1_idx, tuple):
                bmu1_flat = bmu1_idx[0] * weights.shape[1] + bmu1_idx[1]
                bmu2_flat = bmu2_idx[0] * weights.shape[1] + bmu2_idx[1]
            else:
                bmu1_flat = bmu1_idx
                bmu2_flat = bmu2_idx
            
            # Update connectivity matrix (symmetric)
            C[bmu1_flat, bmu2_flat] += 1
            C[bmu2_flat, bmu1_flat] += 1
        
        return C
    
    def _get_two_best_bmus(self, som, sample):
        """Get the two best matching units for a sample."""
        weights = som.get_weights()
        
        # Ensure sample is in correct format
        if isinstance(sample, np.ndarray) and self.use_gpu:
            sample = cp.array(sample)
        
        # Ensure weights is in correct format
        if isinstance(weights, np.ndarray) and self.use_gpu:
            weights = cp.array(weights)
        
        # Compute distances to all neurons
        weights_flat = weights.reshape(-1, weights.shape[-1])
        distances = (cp if self.use_gpu else np).linalg.norm(
            weights_flat - sample, axis=1
        )
        
        # Get indices of two smallest distances
        if self.use_gpu:
            sorted_indices = cp.argpartition(distances, min(2, len(distances)))[:2]
            bmu1_idx = int(sorted_indices[0].get())
            bmu2_idx = int(sorted_indices[1].get())
        else:
            sorted_indices = np.argpartition(distances, min(2, len(distances)))[:2]
            bmu1_idx = int(sorted_indices[0])
            bmu2_idx = int(sorted_indices[1])
        
        # Convert flat indices back to 2D if needed
        n_rows, n_cols = weights.shape[:2]
        bmu1_2d = (bmu1_idx // n_cols, bmu1_idx % n_cols)
        bmu2_2d = (bmu2_idx // n_cols, bmu2_idx % n_cols)
        
        return bmu1_2d, bmu2_2d
    
    def _compute_topographic_function(self, R_grid, R_data, k):
        """Compute TF using ranking correlation between distance matrices."""
        # Get rankings for both matrices
        if self.use_gpu:
            # Flatten matrices and get rankings
            grid_ranks = cp.argsort(cp.argsort(R_grid.flatten()))
            data_ranks = cp.argsort(cp.argsort(R_data.flatten()))
            
            # Compute absolute differences
            rank_diffs = cp.abs(grid_ranks - data_ranks)
            
            # Sum differences and normalize
            total_diff = cp.sum(rank_diffs)
            tf_value = float(total_diff.get()) / (2 * k * len(R_grid.flatten()))
        else:
            # Flatten matrices and get rankings
            grid_ranks = np.argsort(np.argsort(R_grid.flatten()))
            data_ranks = np.argsort(np.argsort(R_data.flatten()))
            
            # Compute absolute differences
            rank_diffs = np.abs(grid_ranks - data_ranks)
            
            # Sum differences and normalize
            total_diff = np.sum(rank_diffs)
            tf_value = float(total_diff) / (2 * k * len(R_grid.flatten()))
        
        return tf_value
