"""
FloatSOM-specific test fixtures.

This module provides pytest fixtures specific to FloatSOM testing,
including fixtures for SOM training parameters, weight matrices,
and distance matrices for MST topology tests.
"""

import warnings
import pytest
import numpy as np

from ..conftest import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp


def pytest_configure(config):
    warnings.filterwarnings(
        "ignore",
        message=r"Normalization is set to none\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"cupyx\.jit\.rawkernel is experimental\..*",
        category=FutureWarning,
    )


@pytest.fixture
def momentum_schedule_params():
    """
    Create parameters for momentum schedule testing.

    Returns:
        dict: Momentum schedule parameters including initial/final values
              and total iterations for schedule computation.
    """
    return {
        "initial_momentum": 0.9,
        "final_momentum": 0.1,
        "total_iterations": 100,
    }


@pytest.fixture
def som_weight_matrix():
    """
    Create a SOM weight matrix for testing.

    Returns:
        cp.ndarray: Weight matrix of shape (64, 16) representing
                   an 8x8 SOM grid with 16-dimensional weights.
    """
    if not GPU_AVAILABLE:
        pytest.skip("GPU required for this test")

    np.random.seed(42)
    weights = np.random.randn(64, 16).astype(np.float32)
    return cp.asarray(weights)


@pytest.fixture
def batch_for_weight_updates():
    """
    Create a batch of samples for weight update testing.

    Returns:
        cp.ndarray: Sample batch of shape (100, 16) with normalized
                   values suitable for SOM weight updates.
    """
    if not GPU_AVAILABLE:
        pytest.skip("GPU required for this test")

    np.random.seed(42)
    batch = np.random.randn(100, 16).astype(np.float32)
    return cp.asarray(batch)


@pytest.fixture
def weights_for_updates():
    """
    Create SOM weights for weight update contract testing.

    Returns:
        cp.ndarray: Weight matrix of shape (64, 16) with normalized
                   initial values for testing weight update bounds.
    """
    if not GPU_AVAILABLE:
        pytest.skip("GPU required for this test")

    np.random.seed(42)
    weights = np.random.randn(64, 16).astype(np.float32)
    # Normalize to reasonable range
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    return cp.asarray(weights)


@pytest.fixture
def distance_matrix_for_mst():
    """
    Create a valid distance matrix for MST topology testing.

    The matrix satisfies distance matrix invariants:
    - Square shape
    - Symmetric
    - Zero diagonal
    - Non-negative values

    Returns:
        np.ndarray: Distance matrix of shape (20, 20) suitable for
                   MST construction and topology tests.
    """
    np.random.seed(42)
    n = 20

    # Generate random points and compute Euclidean distances
    points = np.random.rand(n, 5).astype(np.float32)

    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Ensure exact symmetry and zero diagonal
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)

    return dist.astype(np.float32)
