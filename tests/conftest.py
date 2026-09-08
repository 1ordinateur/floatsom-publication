"""Shared test helpers; no dependency on the parent PIBLO repository."""
import importlib.util
import os

import numpy as np
import pytest

try:
    import cupy as cp
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    if GPU_COUNT:
        cp.zeros(1).sum().item()
except Exception:
    GPU_COUNT = 0

GPU_AVAILABLE = GPU_COUNT > 0
RAY_AVAILABLE = importlib.util.find_spec("ray") is not None
requires_multi_gpu = pytest.mark.skipif(GPU_COUNT < 2, reason="Requires at least two usable CUDA GPUs")
requires_gpu_and_ray = pytest.mark.skipif(not (GPU_AVAILABLE and RAY_AVAILABLE), reason="Requires CUDA and Ray")


@pytest.fixture
def gpu_parity_tolerance():
    return {"rtol": 1e-4, "atol": 1e-6}


@pytest.fixture
def strict_tolerance():
    return {"rtol": 1e-5, "atol": 1e-8}


@pytest.fixture
def ray_context():
    ray = pytest.importorskip("ray")
    started_here = not ray.is_initialized()
    try:
        if started_here:
            address = os.environ.get("RAY_ADDRESS")
            if address:
                ray.init(address=address)
            else:
                ray.init(num_gpus=GPU_COUNT, object_store_memory=256 * 1024**2)
        if GPU_AVAILABLE and ray.cluster_resources().get("GPU", 0) < 1:
            pytest.skip("Ray has no GPU resources available for GPU actor tests")
        yield ray
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()


@pytest.fixture
def gpu_count(ray_context):
    return min(GPU_COUNT, int(ray_context.cluster_resources().get("GPU", 0)))


@pytest.fixture(scope="session")
def multi_gpu_available():
    return GPU_COUNT >= 2


def assert_arrays_equivalent(arr1, arr2, tolerance, name="arrays"):
    def host(a):
        return a.get() if hasattr(a, "get") else a
    np.testing.assert_allclose(host(arr1), host(arr2), **tolerance, err_msg=f"{name} differ beyond tolerance")
