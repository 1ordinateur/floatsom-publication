#!/usr/bin/env python3
"""
SOM evaluation metrics for FloatSOM benchmarks.
"""

import numpy as np
import cupy as cp
from typing import Union, Tuple, Optional

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _to_positive_int(value) -> Optional[int]:
    """Best-effort conversion to positive integer."""
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def _to_non_negative_int(value) -> Optional[int]:
    """Best-effort conversion to non-negative integer."""
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int >= 0 else None


def _is_graph_topology(som_wrapper) -> bool:
    """Return True when wrapper topology is graph-based (MST/RNG)."""
    topology_type = getattr(som_wrapper, 'topology_type', None)
    if isinstance(topology_type, str) and topology_type.lower() in {'mst', 'rng'}:
        return True

    topology = getattr(som_wrapper, 'topology', None)
    topology_name = getattr(topology, 'name', None)
    if isinstance(topology_name, str) and topology_name.lower() in {'mst', 'rng'}:
        return True

    return False


def _is_reformed_topology(som_wrapper) -> bool:
    """Return True when graph topology has been reformed to a grid mapping."""
    grid_node_mapping = getattr(som_wrapper, 'grid_node_mapping', None)
    if isinstance(grid_node_mapping, dict) and len(grid_node_mapping) > 0:
        return True

    topology = getattr(som_wrapper, 'topology', None)
    return bool(getattr(topology, 'is_reformed', False))


def _flatten_som_weights(weights: Union[np.ndarray, cp.ndarray]) -> Tuple[Union[np.ndarray, cp.ndarray], int]:
    """Flatten SOM weights to (n_neurons, features) and return neuron count."""
    if weights.ndim == 0:
        raise ValueError("SOM weights must have at least one dimension")
    if weights.ndim == 1:
        return weights.reshape(1, -1), 1
    if weights.ndim == 2:
        n_neurons = int(weights.shape[0])
        return weights.reshape(n_neurons, -1), n_neurons

    n_neurons = int(np.prod(weights.shape[:-1]))
    return weights.reshape(n_neurons, -1), n_neurons


def _extract_grid_node_mapping(som_wrapper, n_neurons: int) -> Optional[np.ndarray]:
    """Extract per-neuron 2D coordinates from grid-node mapping when available."""
    grid_node_mapping = getattr(som_wrapper, 'grid_node_mapping', None)
    if not isinstance(grid_node_mapping, dict):
        return None

    coords = np.zeros((n_neurons, 2), dtype=np.float32)
    for node_idx in range(n_neurons):
        position = grid_node_mapping.get(node_idx)
        if not isinstance(position, (tuple, list)) or len(position) < 2:
            return None
        row = _to_non_negative_int(position[0])
        col = _to_non_negative_int(position[1])
        if row is None or col is None:
            return None
        coords[node_idx, 0] = float(row)
        coords[node_idx, 1] = float(col)

    return coords


def _infer_grid_shape(
    som_wrapper,
    weights: Union[np.ndarray, cp.ndarray],
    n_neurons: int,
) -> Tuple[int, int]:
    """Infer robust 2D layout dimensions for flattened neurons."""
    allow_grid_from_size = (not _is_graph_topology(som_wrapper)) or _is_reformed_topology(som_wrapper)

    candidates = []

    if weights.ndim >= 3:
        n_rows = _to_positive_int(weights.shape[0])
        n_cols = _to_positive_int(int(np.prod(weights.shape[1:-1])))
        if n_rows is not None and n_cols is not None:
            candidates.append((n_rows, n_cols))

    shape = getattr(som_wrapper, 'shape', None)
    if isinstance(shape, (tuple, list)) and len(shape) >= 2:
        n_rows = _to_positive_int(shape[0])
        n_cols = _to_positive_int(shape[1])
        if n_rows is not None and n_cols is not None:
            candidates.append((n_rows, n_cols))

    n_rows_attr = _to_positive_int(getattr(som_wrapper, 'n_rows', None))
    n_cols_attr = _to_positive_int(getattr(som_wrapper, 'n_cols', None))
    if n_rows_attr is not None and n_cols_attr is not None:
        candidates.append((n_rows_attr, n_cols_attr))

    grid_size = _to_positive_int(getattr(som_wrapper, 'grid_size', None))
    if allow_grid_from_size and grid_size is not None:
        candidates.append((grid_size, grid_size))

    for n_rows, n_cols in candidates:
        if n_rows * n_cols == n_neurons:
            return int(n_rows), int(n_cols)

    if not _is_graph_topology(som_wrapper):
        square_dim = int(np.sqrt(n_neurons))
        if square_dim * square_dim == n_neurons:
            return square_dim, square_dim

    # Fallback for graph topologies and non-square maps: treat as 1D chain.
    return n_neurons, 1


def _get_som_coordinates(
    som_wrapper,
    weights: Union[np.ndarray, cp.ndarray],
    xp,
) -> Tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray], Optional[Tuple[int, int]]]:
    """
    Return flattened weights and a coordinate table for each neuron.

    Coordinates are sourced from `grid_node_mapping` when present (reformed graph
    topologies), otherwise from inferred rectangular layout.
    """
    weights_flat, n_neurons = _flatten_som_weights(weights)

    mapped_coords = _extract_grid_node_mapping(som_wrapper, n_neurons)
    if mapped_coords is not None:
        return weights_flat, xp.asarray(mapped_coords, dtype=xp.float32), None

    n_rows, n_cols = _infer_grid_shape(som_wrapper, weights, n_neurons)
    neuron_rows = xp.arange(n_neurons, dtype=xp.int32) // n_cols
    neuron_cols = xp.arange(n_neurons, dtype=xp.int32) % n_cols
    coords = xp.stack([neuron_rows, neuron_cols], axis=1).astype(xp.float32, copy=False)
    return weights_flat, coords, (n_rows, n_cols)


def calculate_quantization_error(data: Union[np.ndarray, cp.ndarray], weights: Union[np.ndarray, cp.ndarray]) -> float:
    """
    Calculate the average quantization error for a SOM (GPU-accelerated implementation)
    
    Args:
        data: Input data (samples x features)
        weights: SOM weights (neurons x features)
        
    Returns:
        error: Average quantization error
    """
    # Convert to GPU arrays
    data_gpu = cp.array(data) if isinstance(data, np.ndarray) else data
    weights_gpu = cp.array(weights) if isinstance(weights, np.ndarray) else weights
    
    # Reshape weights to match format (total_neurons, features)
    if len(weights_gpu.shape) > 2:
        total_neurons = weights_gpu.shape[0] * weights_gpu.shape[1]
        weights_gpu = weights_gpu.reshape(total_neurons, -1)
    
    n_samples = data_gpu.shape[0]
    batch_size = min(500_000, n_samples)

    if batch_size >= n_samples:
        # Memory-efficient einsum-based calculation
        data_sqnorms = cp.einsum('ij,ij->i', data_gpu, data_gpu)
        weights_sqnorms = cp.einsum('ij,ij->i', weights_gpu, weights_gpu)
        cross_term = cp.einsum('ik,jk->ij', data_gpu, weights_gpu)
        squared_distances = data_sqnorms[:, cp.newaxis] + weights_sqnorms[cp.newaxis, :] - 2 * cross_term
        squared_distances = cp.maximum(squared_distances, 0.0)  # clamp numerical negatives
        min_distances = cp.sqrt(cp.min(squared_distances, axis=1))
    
        return float(cp.mean(min_distances).get())

    # Batched calculation for large datasets
    all_min_distances = []
    weights_sqnorms = cp.einsum('ij,ij->i', weights_gpu, weights_gpu)

    for i in range(0, n_samples, batch_size):
        batch_data = data_gpu[i:i + batch_size]

        data_sqnorms = cp.einsum('ij,ij->i', batch_data, batch_data)
        cross_term = cp.einsum('ik,jk->ij', batch_data, weights_gpu)

        squared_distances = data_sqnorms[:, cp.newaxis] + weights_sqnorms[cp.newaxis, :] - 2 * cross_term
        squared_distances = cp.maximum(squared_distances, 0.0)  # clamp numerical negatives
        min_distances_batch = cp.sqrt(cp.min(squared_distances, axis=1))
        all_min_distances.append(min_distances_batch)

    min_distances = cp.concatenate(all_min_distances)
    return float(cp.mean(min_distances).get())


class QuantizationError:
    """
    Quantization Error metric for SOM evaluation.
    
    Measures the average distance between each data point and its
    best matching unit (BMU) in the SOM.
    
    Range: [0, ∞), lower is better
    """
    
    def __init__(self, use_optimized: bool = True):
        """
        Initialize QuantizationError metric.
        
        Args:
            use_optimized: Whether to use the optimized vectorized implementation
        """
        self.use_optimized = use_optimized
        self.name = 'quantization_error'
    
    @property
    def requires_gpu(self) -> bool:
        return True
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute quantization error.
        
        Args:
            som_wrapper: Trained SOM instance
            data: Input data
            
        Returns:
            Quantization error value
        """
        # Get SOM weights
        weights = som_wrapper.weights if hasattr(som_wrapper, 'weights') else som_wrapper.get_weights()
        
        if self.use_optimized:
            # Use GPU implementation
            return calculate_quantization_error(data, weights)
        else:
            # Simple GPU implementation
            data_gpu = cp.array(data) if isinstance(data, np.ndarray) else data
            weights_gpu = cp.array(weights) if isinstance(weights, np.ndarray) else weights
            return self._compute_simple_gpu(data_gpu, weights_gpu)
    
    def _compute_simple_gpu(self, data: cp.ndarray, weights: cp.ndarray) -> float:
        """Simple GPU implementation."""
        if len(weights.shape) > 2:
            total_neurons = weights.shape[0] * weights.shape[1]
            weights = weights.reshape(total_neurons, -1)
        
        total_error = 0.0
        
        for sample in data:
            distances = cp.linalg.norm(weights - sample, axis=1)
            min_distance = cp.min(distances)
            total_error += float(min_distance.get())
        
        return total_error / len(data)

class TopographicError:
    """
    Topographic Error metric for SOM evaluation.
    
    Measures the proportion of data points for which the first and second
    best matching units (BMUs) are not adjacent in the SOM grid.
    
    Range: [0, 1], lower is better
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 5000):
        """
        Initialize TopographicError metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            batch_size: Number of samples to process at once
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._adjacency_cache = {}
        self.name = 'topographic_error'
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute topographic error using vectorized operations.
        
        Args:
            som_wrapper: Trained SOM instance with get_weights() method
            data: Input data
            
        Returns:
            Topographic error value
        """
        # Convert data to appropriate format
        if isinstance(data, np.ndarray) and self.use_gpu:
            data = cp.array(data)
        
        weights = som_wrapper.get_weights()
        if isinstance(weights, np.ndarray) and self.use_gpu:
            weights = cp.array(weights)
        
        n_samples = len(data)

        xp = cp if self.use_gpu else np

        # Determine neuron layout and flatten weights
        if weights.ndim >= 3:
            n_rows, n_cols = weights.shape[:2]
            n_neurons = n_rows * n_cols
            weights_flat = weights.reshape(n_neurons, -1)
        else:
            n_rows = n_cols = None
            n_neurons = weights.shape[0]
            weights_flat = weights.reshape(n_neurons, -1)

        # Obtain adjacency information
        adjacency_matrix = None

        if hasattr(som_wrapper, 'get_adjacency_matrix'):
            adjacency_matrix = som_wrapper.get_adjacency_matrix()
            if self.use_gpu and isinstance(adjacency_matrix, np.ndarray):
                adjacency_matrix = cp.array(adjacency_matrix)

        if adjacency_matrix is None:
            adjacency_list = None
            if hasattr(som_wrapper, 'adjacency_list') and som_wrapper.adjacency_list:
                adjacency_list = som_wrapper.adjacency_list
            elif hasattr(som_wrapper, 'topology') and getattr(som_wrapper.topology, 'adjacency_list', None):
                adjacency_list = som_wrapper.topology.adjacency_list

            if adjacency_list is not None:
                adjacency_matrix = self._adjacency_from_list(adjacency_list, n_neurons, xp)
            elif n_rows is not None and n_cols is not None:
                grid_key = (n_rows, n_cols)
                if grid_key not in self._adjacency_cache:
                    self._adjacency_cache[grid_key] = self._create_adjacency_matrix(n_rows, n_cols)
                adjacency_matrix = self._adjacency_cache[grid_key]
            else:
                adjacency_matrix = xp.ones((n_neurons, n_neurons), dtype=bool)

        if adjacency_matrix.shape[0] != n_neurons:
            raise ValueError(f"Adjacency matrix size {adjacency_matrix.shape} does not match neuron count {n_neurons}")
        
        total_errors = 0
        
        # Process in batches for memory efficiency
        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i+self.batch_size]
            batch_size = len(batch)
            
            # Memory-efficient einsum distance computation
            batch_sqnorms = xp.einsum('ij,ij->i', batch, batch)
            weights_sqnorms = xp.einsum('ij,ij->i', weights_flat, weights_flat)
            cross_term = xp.einsum('ik,jk->ij', batch, weights_flat)
            distances = xp.sqrt(batch_sqnorms[:, xp.newaxis] + weights_sqnorms[xp.newaxis, :] - 2 * cross_term)
            
            # Get top 2 BMU indices for all samples in batch
            if distances.shape[1] >= 2:
                top2_indices = xp.argpartition(distances, 1, axis=1)[:, :2]
                
                # Get the actual indices of the two smallest values
                rows = xp.arange(batch_size)[:, xp.newaxis]
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

    def _adjacency_from_list(self, adjacency_list, n_neurons, xp):
        """Convert adjacency representations (dict/list) to a dense adjacency matrix."""
        matrix = xp.zeros((n_neurons, n_neurons), dtype=bool)

        if isinstance(adjacency_list, dict):
            items = adjacency_list.items()
        else:
            items = enumerate(adjacency_list)

        for node, neighbors in items:
            if neighbors is None:
                continue
            if hasattr(neighbors, 'tolist'):
                neighbors = neighbors.tolist()
            for neighbor in neighbors:
                if 0 <= node < n_neurons and 0 <= neighbor < n_neurons:
                    matrix[node, neighbor] = True
                    matrix[neighbor, node] = True

        xp.fill_diagonal(matrix, False)
        return matrix

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


class Trustworthiness:
    """
    Trustworthiness metric adapted for SOM evaluation.
    
    Measures how well k-nearest neighbors in the original space
    are preserved in the SOM representation.
    
    Range: [0, 1], higher is better
    """
    
    def __init__(self, k: int = 7):
        """
        Initialize Trustworthiness metric.
        
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.name = 'trustworthiness'
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute trustworthiness metric.
        
        Args:
            som_wrapper: Trained SOM instance
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
        som_embedding = self._get_som_embedding(som_wrapper, data_np)
        
        return self._compute_trustworthiness_custom(data_np, som_embedding)
    
    def _get_som_embedding(self, som_wrapper, data: np.ndarray) -> np.ndarray:
        """Get 2D SOM grid positions for each data point using efficient batched computation."""
        weights = som_wrapper.weights if hasattr(som_wrapper, 'weights') else som_wrapper.get_weights()
        
        # Ensure everything is NumPy for this computation (KDTree requires NumPy anyway)
        if isinstance(weights, cp.ndarray):
            weights = weights.get()
        if isinstance(data, cp.ndarray):
            data = data.get()
        weights_flat, neuron_coords, _ = _get_som_coordinates(som_wrapper, weights, cp)
        
        # GPU-accelerated einsum BMU computation
        data_gpu = cp.array(data) if isinstance(data, np.ndarray) else data
        weights_gpu = cp.array(weights_flat) if isinstance(weights_flat, np.ndarray) else weights_flat
        
        data_sqnorms = cp.einsum('ij,ij->i', data_gpu, data_gpu)
        weights_sqnorms = cp.einsum('ij,ij->i', weights_gpu, weights_gpu)
        cross_term = cp.einsum('ik,jk->ij', data_gpu, weights_gpu)
        distances = cp.sqrt(data_sqnorms[:, cp.newaxis] + weights_sqnorms[cp.newaxis, :] - 2 * cross_term)
        bmu_flat_indices = cp.argmin(distances, axis=1)
        return neuron_coords[bmu_flat_indices].astype(cp.float64).get()
    
    def _compute_trustworthiness_custom(self, X: np.ndarray, X_embedded: np.ndarray) -> float:
        """Memory-efficient chunked GPU trustworthiness computation."""
        X_gpu = cp.array(X)
        X_embedded_gpu = cp.array(X_embedded)
        n_samples = X.shape[0]
        
        # Process in chunks to avoid memory explosion
        chunk_size = min(1000, n_samples)  # Adjust based on available memory
        trustworthiness_sum = 0.0
        
        # Pre-compute squared norms
        X_sqnorms = cp.einsum('ij,ij->i', X_gpu, X_gpu)
        emb_sqnorms = cp.einsum('ij,ij->i', X_embedded_gpu, X_embedded_gpu)
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk_X = X_gpu[i:end_idx]
            chunk_emb = X_embedded_gpu[i:end_idx]
            chunk_size_actual = end_idx - i
            
            # Compute embedded space distances for this chunk
            chunk_emb_sqnorms = emb_sqnorms[i:end_idx]
            emb_cross_term = cp.einsum('ik,jk->ij', chunk_emb, X_embedded_gpu)
            dist_embedded_chunk = cp.sqrt(chunk_emb_sqnorms[:, cp.newaxis] + emb_sqnorms[cp.newaxis, :] - 2 * emb_cross_term)
            embedded_nn_chunk = cp.argsort(dist_embedded_chunk, axis=1)[:, 1:self.k+1]
            
            # Compute original space distances for this chunk
            chunk_X_sqnorms = X_sqnorms[i:end_idx]
            X_cross_term = cp.einsum('ik,jk->ij', chunk_X, X_gpu)
            dist_X_chunk = cp.sqrt(chunk_X_sqnorms[:, cp.newaxis] + X_sqnorms[cp.newaxis, :] - 2 * X_cross_term)
            original_ranks_chunk = cp.argsort(dist_X_chunk, axis=1)
            
            # Vectorized rank lookup for entire chunk
            rank_matrix = cp.zeros((chunk_size_actual, n_samples), dtype=cp.int32)
            for local_idx in range(chunk_size_actual):
                rank_matrix[local_idx, original_ranks_chunk[local_idx]] = cp.arange(n_samples)
            
            # Vectorized neighbor rank lookup
            chunk_indices = cp.arange(chunk_size_actual)[:, cp.newaxis]
            neighbor_ranks = rank_matrix[chunk_indices, embedded_nn_chunk]
            
            # Vectorized violation computation
            violations = neighbor_ranks > self.k
            penalties = cp.where(violations, neighbor_ranks - self.k, 0)
            trustworthiness_sum += float(cp.sum(penalties).get())
        
        # Normalize
        normalizer = n_samples * self.k * (2 * n_samples - 3 * self.k - 1) / 2
        return 1 - (2 * trustworthiness_sum) / normalizer


class NeighborhoodPreservation:
    """
    Neighborhood Preservation metric for SOM evaluation.
    
    Measures the overlap between k-nearest neighbors in the original
    space and k-nearest neighbors in the SOM representation.
    
    Range: [0, 1], higher is better
    """
    
    def __init__(self, k: int = 7, use_approximate: bool = True, 
                 sample_size: int = 5000, batch_size: int = 5000):
        """
        Initialize NeighborhoodPreservation metric.
        
        Args:
            k: Number of nearest neighbors to consider
            use_approximate: Use sampling for very large datasets
            sample_size: Maximum number of samples for approximation
            batch_size: Batch size for BMU computation
        """
        self.k = k
        self.use_approximate = use_approximate
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.name = 'neighborhood_preservation'
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute neighborhood preservation with memory optimization.
        
        Args:
            som_wrapper: Trained SOM instance
            data: Input data
            
        Returns:
            Neighborhood preservation value
        """
        # Keep on GPU as long as possible
        data_gpu = cp.array(data) if isinstance(data, np.ndarray) else data
        
        n_samples = len(data_gpu)
        
        # Use sampling for very large datasets
        if self.use_approximate and n_samples > self.sample_size:
            indices = cp.random.choice(n_samples, self.sample_size, replace=False)
            data_sample = data_gpu[indices]
        else:
            data_sample = data_gpu
        
        # Get SOM embedding using batched GPU computation
        som_embedding = self._get_som_embedding_batched(som_wrapper, data_sample.get())
        
        if SCIPY_AVAILABLE:
            # Use efficient KD-tree based computation (requires CPU)
            return self._compute_preservation_kdtree(data_sample.get(), som_embedding)
        else:
            # GPU chunk-based computation
            return self._compute_preservation_chunked_gpu(data_sample, som_embedding)
    
    def _get_som_embedding_batched(self, som_wrapper, data: np.ndarray) -> np.ndarray:
        """Get 2D SOM grid positions using batched operations."""
        weights = som_wrapper.get_weights()
        
        # Ensure everything is NumPy for this computation (KDTree requires NumPy anyway)
        if isinstance(weights, cp.ndarray):
            weights = weights.get()
        if isinstance(data, cp.ndarray):
            data = data.get()
        weights_flat, neuron_coords, _ = _get_som_coordinates(som_wrapper, weights, cp)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            
            # GPU einsum distance computation for batch
            batch_gpu = cp.array(batch) if isinstance(batch, np.ndarray) else batch
            weights_gpu = cp.array(weights_flat) if isinstance(weights_flat, np.ndarray) else weights_flat
            
            batch_sqnorms = cp.einsum('ij,ij->i', batch_gpu, batch_gpu)
            weights_sqnorms = cp.einsum('ij,ij->i', weights_gpu, weights_gpu)
            cross_term = cp.einsum('ik,jk->ij', batch_gpu, weights_gpu)
            distances = cp.sqrt(batch_sqnorms[:, cp.newaxis] + weights_sqnorms[cp.newaxis, :] - 2 * cross_term)
            bmu_flat_indices = cp.argmin(distances, axis=1)

            embeddings.append(neuron_coords[bmu_flat_indices].get())
        
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
    
    def _compute_preservation_chunked(self, data: np.ndarray, embedding: np.ndarray) -> float:
        """Fallback method when scipy is not available."""
        n_samples = len(data)
        chunk_size = min(100, n_samples)
        
        # Pre-compute embedded space distances (small, 2D)
        dist_embedded = np.linalg.norm(embedding[:, None] - embedding[None, :], axis=2)
        nn_som = np.argsort(dist_embedded, axis=1)[:, 1:self.k+1]
        
        total_overlap = 0
        
        # Process original space in chunks
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            # Compute distances for this chunk vs all
            dist_chunk = np.linalg.norm(
                data[i:end_idx, None] - data[None, :], 
                axis=2
            )
            
            for j, idx in enumerate(range(i, end_idx)):
                nn_original = np.argsort(dist_chunk[j])[1:self.k+1]
                overlap = len(set(nn_original) & set(nn_som[idx]))
                total_overlap += overlap
        
        return total_overlap / (n_samples * self.k)
    
    def _compute_preservation_chunked_gpu(self, data: cp.ndarray, embedding: np.ndarray) -> float:
        """GPU-accelerated chunked computation when scipy is not available."""
        n_samples = len(data)
        chunk_size = min(100, n_samples)
        
        # Convert embedding to GPU and compute embedded space distances
        embedding_gpu = cp.array(embedding)
        
        # Pre-compute embedded space distances using einsum
        emb_sqnorms = cp.einsum('ij,ij->i', embedding_gpu, embedding_gpu)
        emb_cross = cp.einsum('ik,jk->ij', embedding_gpu, embedding_gpu)
        dist_embedded = cp.sqrt(emb_sqnorms[:, cp.newaxis] + emb_sqnorms[cp.newaxis, :] - 2 * emb_cross)
        nn_som = cp.argsort(dist_embedded, axis=1)[:, 1:self.k+1]
        
        total_overlap = 0
        
        # Process original space in chunks using einsum
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = data[i:end_idx]
            
            # GPU einsum distance computation for chunk
            chunk_sqnorms = cp.einsum('ij,ij->i', chunk, chunk)
            data_sqnorms = cp.einsum('ij,ij->i', data, data)
            chunk_cross = cp.einsum('ik,jk->ij', chunk, data)
            dist_chunk = cp.sqrt(chunk_sqnorms[:, cp.newaxis] + data_sqnorms[cp.newaxis, :] - 2 * chunk_cross)
            
            for j, idx in enumerate(range(i, end_idx)):
                nn_original = cp.argsort(dist_chunk[j])[1:self.k+1]
                overlap = len(cp.intersect1d(nn_original, nn_som[idx]).get())
                total_overlap += overlap
        
        return total_overlap / (n_samples * self.k)


class DistortionMeasure:
    """
    Distortion Measure (DM) metric for SOM evaluation.
    
    Combines quantization error with neighborhood structure using 
    distance-based weighting factor for the quantization error.
    
    Formula: DM = (1/k) * Σᵢ Σⱼ hᵦᵢⱼ * ||xᵢ - wⱼ||²
    
    Range: [0, ∞), lower is better
    """
    
    def __init__(self, use_gpu: bool = True, sigma: float = 1.0, batch_size: int = 1000):
        """
        Initialize DistortionMeasure metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            sigma: Neighborhood function parameter (controls spread)
            batch_size: Number of samples to process at once
        """
        self.use_gpu = use_gpu
        self.sigma = sigma
        self.batch_size = batch_size
        self._grid_distance_cache = {}
        self.name = 'distortion_measure'
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute distortion measure using fully vectorized operations.
        
        Args:
            som_wrapper: Trained SOM instance
            data: Input data
            
        Returns:
            Distortion measure value
        """
        # Convert data to appropriate format
        if isinstance(data, np.ndarray) and self.use_gpu:
            data = cp.array(data)
        
        weights = som_wrapper.get_weights()
        if isinstance(weights, np.ndarray) and self.use_gpu:
            weights = cp.array(weights)
        xp = cp if self.use_gpu else np
        weights_flat, neuron_coords, grid_key = _get_som_coordinates(som_wrapper, weights, xp)
        
        # Get or create grid distance matrix
        if grid_key is None:
            grid_distances = self._compute_grid_distances(neuron_coords)
        else:
            if grid_key not in self._grid_distance_cache:
                self._grid_distance_cache[grid_key] = self._compute_grid_distances(neuron_coords)
            grid_distances = self._grid_distance_cache[grid_key]
        
        total_distortion = 0.0
        n_samples = len(data)
        
        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i+self.batch_size]
            
            # GPU einsum distance computation for batch
            batch_sqnorms = xp.einsum('ij,ij->i', batch, batch)
            weights_sqnorms = xp.einsum('ij,ij->i', weights_flat, weights_flat)
            cross_term = xp.einsum('ik,jk->ij', batch, weights_flat)
            distances = xp.sqrt(batch_sqnorms[:, xp.newaxis] + weights_sqnorms[xp.newaxis, :] - 2 * cross_term)
            
            # Find BMUs for the batch
            bmu_indices = xp.argmin(distances, axis=1)
            
            # Compute neighborhood values for each sample's BMU
            # Shape: (batch_size, n_neurons)
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
    
    def _compute_grid_distances(self, neuron_coords):
        """Pre-compute pairwise distances between grid positions."""
        xp = cp if self.use_gpu else np
        return xp.linalg.norm(
            neuron_coords[:, xp.newaxis, :] - neuron_coords[xp.newaxis, :, :],
            axis=2,
        )


class TopographicFunction:
    """
    Topographic Function (TF) metric for SOM evaluation.
    
    Measures topology preservation by comparing distance matrices
    from grid space and data connectivity space.
    
    Formula: TF = (1/2k) * Σᵢ (|P₁ᵢ - Q₁ᵢ| + |P₂ᵢ - Q₂ᵢ|)
    
    Range: [0, 1], lower is better (0 = perfect topology preservation)
    """
    
    def __init__(self, use_gpu: bool = True, max_neurons: int = 10000, 
                 sample_ratio: float = 0.1):
        """
        Initialize TopographicFunction metric.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            max_neurons: Maximum neurons before switching to approximation
            sample_ratio: Fraction of neuron pairs to sample for large grids
        """
        self.use_gpu = use_gpu
        self.max_neurons = max_neurons
        self.sample_ratio = sample_ratio
        self.name = 'topographic_function'
    
    @property
    def requires_gpu(self) -> bool:
        return self.use_gpu
    
    def compute(self, som_wrapper, data: Union[np.ndarray, cp.ndarray]) -> float:
        """
        Compute topographic function, skipping for very large grids.
        
        Args:
            som_wrapper: Trained SOM instance
            data: Input data
            
        Returns:
            Topographic function value or -1 if skipped due to size
        """
        # Get grid size
        weights = som_wrapper.get_weights()
        _, n_neurons = _flatten_som_weights(weights)
        
        # Skip computation for very large grids to avoid memory issues
        if n_neurons > self.max_neurons:
            return -1.0  # Indicate computation was skipped
        
        # Convert data to appropriate format
        if isinstance(data, np.ndarray) and self.use_gpu:
            data = cp.array(data)
        
        # Compute distance matrices for smaller grids
        R_grid = self._compute_grid_distance_matrix(som_wrapper)
        R_data = self._compute_data_distance_matrix(som_wrapper, data)
        
        # Compute topographic function using ranking correlation
        return self._compute_topographic_function(R_grid, R_data, len(data))
    
    def _compute_grid_distance_matrix(self, som_wrapper):
        """Compute R^grid distance matrix between neurons on the grid using memory-efficient direct formula."""
        weights = som_wrapper.get_weights()
        xp = cp if self.use_gpu else np
        _, neuron_coords, _ = _get_som_coordinates(som_wrapper, weights, xp)
        n_neurons = int(neuron_coords.shape[0])
        
        # Compute Euclidean distances with chunking for large maps
        if n_neurons > 5000:
            # Chunked computation for memory efficiency
            R_grid = xp.zeros((n_neurons, n_neurons), dtype=xp.float32)
            chunk_size = min(1000, n_neurons // 4)
            
            for i in range(0, n_neurons, chunk_size):
                end_i = min(i + chunk_size, n_neurons)
                coord_diff = neuron_coords[i:end_i, xp.newaxis, :] - neuron_coords[xp.newaxis, :, :]
                R_grid[i:end_i, :] = xp.linalg.norm(coord_diff.astype(xp.float32), axis=2)
        else:
            # Direct computation for smaller grids
            coord_diff = neuron_coords[:, xp.newaxis, :] - neuron_coords[xp.newaxis, :, :]
            R_grid = xp.linalg.norm(coord_diff.astype(xp.float32), axis=2)
        
        return R_grid
    
    def _compute_data_distance_matrix(self, som_wrapper, data):
        """Compute R^data distance matrix using connectivity from 1st/2nd winners."""
        # Build connectivity matrix
        C = self._build_connectivity_matrix(som_wrapper, data)
        
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
    
    def _build_connectivity_matrix(self, som_wrapper, data):
        """Build connectivity matrix C using 1st and 2nd winner relationships."""
        weights = som_wrapper.get_weights()
        _, n_neurons = _flatten_som_weights(weights)
        
        if self.use_gpu:
            C = cp.zeros((n_neurons, n_neurons))
        else:
            C = np.zeros((n_neurons, n_neurons))
        
        for sample in data:
            # Get two best matching units
            bmu1_idx, bmu2_idx = self._get_two_best_bmus(som_wrapper, sample)

            # Update connectivity matrix (symmetric)
            C[bmu1_idx, bmu2_idx] += 1
            C[bmu2_idx, bmu1_idx] += 1
        
        return C
    
    def _get_two_best_bmus(self, som_wrapper, sample):
        """Get the two best matching units for a sample."""
        weights = som_wrapper.get_weights()
        
        # Ensure sample is in correct format
        if isinstance(sample, np.ndarray) and self.use_gpu:
            sample = cp.array(sample)
        
        # Ensure weights is in correct format
        if isinstance(weights, np.ndarray) and self.use_gpu:
            weights = cp.array(weights)
        
        # Compute distances to all neurons
        weights_flat, _ = _flatten_som_weights(weights)
        n_neurons = int(weights_flat.shape[0])
        distances = (cp if self.use_gpu else np).linalg.norm(
            weights_flat - sample, axis=1
        )
        
        # Get indices of two smallest distances
        kth = 1 if n_neurons > 1 else 0
        if self.use_gpu:
            sorted_indices = cp.argpartition(distances, kth)[:2]
            bmu1_idx = int(sorted_indices[0].get())
            bmu2_idx = int(sorted_indices[1].get()) if len(sorted_indices) > 1 else bmu1_idx
        else:
            sorted_indices = np.argpartition(distances, kth)[:2]
            bmu1_idx = int(sorted_indices[0])
            bmu2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else bmu1_idx

        return bmu1_idx, bmu2_idx
    
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
