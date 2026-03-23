"""
Shared resource-limit detection helpers for Ray pipelines.

This module intentionally avoids importing GPU libraries at import time so it can
be used from lightweight contexts and unit tests.
"""

from __future__ import annotations

import multiprocessing
import os
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence


_HPC_CPU_ENV_VARS = (
    "PBS_NCPUS",
    "NCPUS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
)


def parse_memory_value(value: str) -> Optional[int]:
    """Convert scheduler memory strings (e.g. '120GB', '4096') to bytes."""
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.match(r"^(?P<num>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[a-zA-Z]*)$", text)
    if not match:
        return None

    number = float(match.group("num"))
    unit = match.group("unit").lower()

    unit_multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }

    multiplier = unit_multipliers.get(unit)
    if multiplier is None:
        return None

    return int(number * multiplier)


def get_job_memory_limit_bytes(
    env: Optional[Mapping[str, str]] = None,
    cgroup_paths: Optional[Sequence[Path]] = None,
) -> Optional[int]:
    """Best-effort detection of job-level memory limits (scheduler or cgroup)."""
    effective_env = os.environ if env is None else env

    env_keys = [
        "PBS_JOBMEM",
        "PBS_MEM",
        "PBS_VMEM",
        "PBS_RESOURCE_LIST_MEM",
        "PBS_RESOURCE_LIST_mem",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_GPU",
        "SLURM_MEM_PER_CPU",
    ]
    for key in env_keys:
        raw = effective_env.get(key)
        if not raw:
            continue
        value = parse_memory_value(raw)
        if value:
            # SLURM provides per-core values as plain integers (MB)
            if key in {"SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU"} and str(raw).strip().isdigit():
                value = int(str(raw).strip()) * 1024**2
            return int(value)

    if cgroup_paths is None:
        cgroup_paths = (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        )

    for path in cgroup_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text().strip()
        except OSError:
            continue
        if not content or content.lower() == "max":
            continue
        try:
            limit = int(content)
        except ValueError:
            continue
        if 0 < limit < (1 << 62):
            return int(limit)

    return None


def get_cgroup_memory_usage_bytes(
    usage_paths: Optional[Sequence[Path]] = None,
) -> Optional[int]:
    """Return current cgroup memory usage in bytes if available."""
    if usage_paths is None:
        usage_paths = (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        )

    for path in usage_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text().strip()
        except OSError:
            continue
        if not content:
            continue
        try:
            usage = int(content)
        except ValueError:
            continue
        if usage >= 0:
            return int(usage)
    return None


def detect_assigned_cpu_count() -> int:
    """
    Best-effort detection of how many CPUs this process should use.

    - Prefers Ray runtime context when available.
    - Falls back to scheduler env vars and OS CPU count.
    - Clamps to CPU affinity when available.
    """
    affinity_limit = None
    try:
        affinity_limit = len(os.sched_getaffinity(0))
    except Exception:
        affinity_limit = None

    try:
        import ray  # type: ignore

        context = ray.get_runtime_context()
        if hasattr(context, "get_assigned_resources"):
            assigned = context.get_assigned_resources() or {}
            cpu = assigned.get("CPU")
            if cpu is not None:
                detected = max(1, int(cpu))
                if affinity_limit is not None:
                    return max(1, min(detected, int(affinity_limit)))
                return detected
        if hasattr(context, "get_resource_ids"):
            resource_ids = context.get_resource_ids() or {}
            cpu_ids = resource_ids.get("CPU")
            if cpu_ids:
                total = 0.0
                for _, quantity in cpu_ids:
                    total += float(quantity)
                if total > 0:
                    detected = max(1, int(total))
                    if affinity_limit is not None:
                        return max(1, min(detected, int(affinity_limit)))
                    return detected
    except Exception:
        pass

    for key in _HPC_CPU_ENV_VARS:
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            detected = parsed
            if affinity_limit is not None:
                return max(1, min(detected, int(affinity_limit)))
            return detected

    detected = max(1, multiprocessing.cpu_count() or 1)
    if affinity_limit is not None:
        return max(1, min(detected, int(affinity_limit)))
    return detected


def estimate_workers_on_node(total_workers: int) -> int:
    """
    Best-effort estimate of how many GPU workers share a single node.

    For multi-node Ray, `ray.cluster_resources()` reports cluster-wide totals, so we derive
    a per-node estimate from `ray.nodes()` instead (e.g., max GPUs on any alive node).
    """
    total_workers = max(1, int(total_workers))
    try:
        import ray  # type: ignore

        if not ray.is_initialized():
            return total_workers
        nodes = ray.nodes()
    except Exception:
        return total_workers

    alive_nodes = [node for node in nodes if node.get("Alive", False)]
    if not alive_nodes:
        return total_workers

    max_gpus_per_node = 0
    for node in alive_nodes:
        resources = node.get("Resources") or {}
        try:
            node_gpus = int(resources.get("GPU", 0) or 0)
        except (TypeError, ValueError):
            node_gpus = 0
        max_gpus_per_node = max(max_gpus_per_node, node_gpus)

    if max_gpus_per_node <= 0:
        return total_workers

    return max(1, min(total_workers, max_gpus_per_node))


def detect_gpu_workers_on_node() -> int:
    """
    Best-effort estimate of how many GPU workers share the current node.

    In this pipeline we schedule 1 GPU per worker actor, so we treat the node's
    GPU count as the worker-sharing factor.
    """
    try:
        import ray  # type: ignore

        if not ray.is_initialized():
            return 1
    except Exception:
        return 1

    node_id = None
    try:
        ctx = ray.get_runtime_context()
        get_node_id = getattr(ctx, "get_node_id", None)
        if callable(get_node_id):
            node_id = get_node_id()
        else:
            node_id = getattr(ctx, "node_id", None)
    except Exception:
        node_id = None

    node_ip = None
    try:
        from ray.util import get_node_ip_address  # type: ignore

        node_ip = get_node_ip_address()
    except Exception:
        node_ip = None

    try:
        nodes = ray.nodes()
    except Exception:
        nodes = []

    alive_nodes = [node for node in nodes if node.get("Alive", False)]
    if not alive_nodes:
        return 1

    def _extract_gpu_count(node: Mapping[str, object]) -> int:
        resources = node.get("Resources") or {}
        try:
            return max(0, int(resources.get("GPU", 0) or 0))
        except (TypeError, ValueError):
            return 0

    if node_id is not None:
        for node in alive_nodes:
            if node.get("NodeID") == node_id:
                gpus = _extract_gpu_count(node)
                return max(1, gpus or 1)

    if node_ip is not None:
        for node in alive_nodes:
            if node.get("NodeManagerAddress") == node_ip:
                gpus = _extract_gpu_count(node)
                return max(1, gpus or 1)

    max_gpus = 0
    for node in alive_nodes:
        max_gpus = max(max_gpus, _extract_gpu_count(node))
    return max(1, max_gpus or 1)

