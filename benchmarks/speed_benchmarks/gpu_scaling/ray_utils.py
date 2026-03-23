"""Ray/NCCL cleanup utilities for GPU scaling benchmarks."""

from __future__ import annotations

import time


def _free_gpu_memory_pools() -> None:
    """Best-effort release of CuPy memory pools."""

    try:
        import cupy as cp  # type: ignore

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        # CuPy may not be installed or GPU not in use; ignore
        pass


def ensure_clean_ray_state(*, initial: bool = False, sleep_seconds: float = 2.0, free_gpu_memory: bool = True) -> None:
    """Tear down Ray collectives, clusters, and GPU pools between runs."""

    try:
        import ray  # type: ignore

        try:
            from ray.util import collective  # type: ignore

            for group_name in ("default", "som_training"):
                try:
                    collective.destroy_collective_group(group_name=group_name)
                except Exception:
                    pass
        except Exception:
            # Collective utilities may not be available; ignore
            pass

        if ray.is_initialized():
            try:
                ray.shutdown()
            finally:
                time.sleep(sleep_seconds)
    except Exception:
        # Ray may not be installed; ignore
        pass

    if free_gpu_memory:
        _free_gpu_memory_pools()
