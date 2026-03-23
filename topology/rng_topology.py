"""
Relative Neighborhood Graph (RNG) topology for FloatSOM.
"""

from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np

from .mst_topology import MSTTopology, calculate_mst_cpu
from .utils import DisjointSetUnion, iter_chunk_ranges

_DEFAULT_GPU_BLOCK_CHUNK_SIZE = 32
_GPU_BLOCK_TARGET_BYTES = 128 * 1024 * 1024


def _resolve_gpu_block_chunk_size(
    num_nodes: int,
    item_size: int,
    requested_chunk_size: Optional[int],
) -> int:
    """Pick a chunk size that bounds temporary 3D tensor memory."""
    if requested_chunk_size is not None:
        return max(1, min(num_nodes, int(requested_chunk_size)))

    bytes_per_k = max(1, num_nodes * num_nodes * (item_size + 1))
    auto_chunk = _GPU_BLOCK_TARGET_BYTES // bytes_per_k
    auto_chunk = max(1, int(auto_chunk))
    return max(1, min(num_nodes, min(_DEFAULT_GPU_BLOCK_CHUNK_SIZE, auto_chunk)))


def _connect_components_with_mst(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    distance_matrix_provider: Callable[[], np.ndarray],
) -> List[Tuple[int, int]]:
    """Defensively connect pathological disconnected RNG outputs.

    For exact Euclidean distances, the RNG contains the MST and should already
    be connected. We keep this repair path only as a guard against numerical or
    otherwise non-Euclidean/pathological distance matrices so downstream graph
    distance code never sees disconnected components.
    """
    if num_nodes <= 1:
        return []

    if not edges:
        return calculate_mst_cpu(distance_matrix_provider())

    dsu = DisjointSetUnion(num_nodes)
    for u, v in edges:
        dsu.union(u, v)

    if dsu.component_count == 1:
        return edges

    mst_edges = calculate_mst_cpu(distance_matrix_provider())
    for u, v in mst_edges:
        if dsu.union(u, v):
            edges.append((u, v))
            if dsu.component_count == 1:
                break

    return edges


def calculate_rng_cpu(
    distance_matrix: np.ndarray,
    ensure_connected: bool = True,
    tolerance: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Calculate Relative Neighborhood Graph (RNG) edges from a distance matrix.

    Edge (i, j) exists iff no third node k is closer to both i and j than i and j
    are to each other:
        max(d(i, k), d(j, k)) < d(i, j)

    Args:
        distance_matrix: Square pairwise distance matrix.
        ensure_connected: Whether to defensively connect disconnected outputs.
        tolerance: Optional additive slack applied to the blocking comparison.

    Returns:
        List of undirected edges as (u, v) tuples with u < v.
    """
    n = int(distance_matrix.shape[0])
    if n <= 1:
        return []

    rng_edges: List[Tuple[int, int]] = []
    for i in range(n):
        dist_i = distance_matrix[i]
        for j in range(i + 1, n):
            dij = float(distance_matrix[i, j])
            if not np.isfinite(dij):
                continue

            blocking = np.maximum(dist_i, distance_matrix[j]) < (dij - tolerance)
            blocking[i] = False
            blocking[j] = False
            if not np.any(blocking):
                rng_edges.append((i, j))

    if not ensure_connected:
        return rng_edges

    return _connect_components_with_mst(n, rng_edges, lambda: distance_matrix)


def calculate_rng_gpu(
    distance_matrix: cp.ndarray,
    ensure_connected: bool = True,
    tolerance: float = 0.0,
    block_chunk_size: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Calculate Relative Neighborhood Graph (RNG) edges directly on GPU.

    This is mathematically equivalent to ``calculate_rng_cpu`` and applies the
    canonical RNG lune test directly on the supplied distance matrix.
    """
    n = int(distance_matrix.shape[0])
    if n <= 1:
        return []

    chunk_size = _resolve_gpu_block_chunk_size(
        n,
        int(distance_matrix.dtype.itemsize),
        block_chunk_size,
    )

    finite = cp.isfinite(distance_matrix)
    upper_triangle = cp.triu(cp.ones((n, n), dtype=cp.bool_), k=1)
    rng_edges_mask = cp.zeros((n, n), dtype=cp.bool_)
    candidates = upper_triangle & finite

    has_candidates = bool(cp.any(candidates))
    if has_candidates:
        # Broadcast view only; avoids allocating an additional full n x n tensor.
        pairwise_distances = distance_matrix[:, :, None]

        for start, stop in iter_chunk_ranges(n, chunk_size):
            blocking = cp.maximum(
                distance_matrix[:, None, start:stop],
                distance_matrix[None, :, start:stop],
            )
            if tolerance != 0.0:
                blocking += tolerance
            blocking = blocking < pairwise_distances

            # Exclude k == i and k == j exactly like the CPU implementation.
            for local_idx, node_idx in enumerate(range(start, stop)):
                blocking[node_idx, :, local_idx] = False
                blocking[:, node_idx, local_idx] = False

            blocked_pairs = cp.any(blocking, axis=2)
            candidates &= ~blocked_pairs

        rng_edges_mask |= candidates

    edge_u, edge_v = cp.where(rng_edges_mask)
    edge_u_cpu = cp.asnumpy(edge_u)
    edge_v_cpu = cp.asnumpy(edge_v)
    rng_edges = [(int(u), int(v)) for u, v in zip(edge_u_cpu, edge_v_cpu)]

    if not ensure_connected:
        return rng_edges

    return _connect_components_with_mst(n, rng_edges, distance_matrix.get)


class RNGTopology(MSTTopology):
    """
    Relative Neighborhood Graph topology.

    This topology reuses MSTTopology's scheduling and influence-map machinery,
    but uses RNG edges instead of MST edges. Exact Euclidean RNGs should
    already be connected; the MST repair path is kept only as a defensive guard
    for pathological numerical inputs.
    """

    @property
    def name(self) -> str:
        return "RNG"

    def _calculate_mst_edges(self, distance_matrix: cp.ndarray) -> List[Tuple[int, int]]:
        """Calculate RNG edges on GPU and transfer only the final edge list."""
        return calculate_rng_gpu(distance_matrix, ensure_connected=True)

