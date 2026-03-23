"""
Numexpr thread configuration for Ray workers.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Optional


_HPC_CPU_ENV_VARS = (
    "PBS_NCPUS",
    "NCPUS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
)


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _detect_available_cpus() -> int:
    for key in _HPC_CPU_ENV_VARS:
        parsed = _parse_positive_int(os.environ.get(key))
        if parsed is not None:
            return parsed
    return max(1, multiprocessing.cpu_count() or 1)


def configure_numexpr_threads() -> None:
    """
    Align numexpr max/requested threads to allocated CPU capacity.
    """
    available = _detect_available_cpus()

    os.environ["NUMEXPR_MAX_THREADS"] = str(available)

    requested = _parse_positive_int(os.environ.get("NUMEXPR_NUM_THREADS"))
    if requested is None or requested > available:
        os.environ["NUMEXPR_NUM_THREADS"] = str(available)
