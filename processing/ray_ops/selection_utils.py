"""Utilities for handling selector-provided sample index payloads."""

from typing import Any, Optional

import cupy as cp
import numpy as np


def extract_selected_indices(samples: Any) -> Optional[np.ndarray]:
    """
    Normalize selector-provided sample indices into a 1D int64 NumPy array.

    Returns None when input cannot be interpreted as a selector index payload.
    """
    if samples is None:
        return None

    candidate = samples
    if hasattr(samples, "indices"):
        candidate = getattr(samples, "indices")

    if candidate is None:
        return None

    if isinstance(candidate, cp.ndarray):
        if candidate.ndim != 1:
            return None
        candidate = cp.asnumpy(candidate)

    if isinstance(candidate, np.ndarray):
        if candidate.ndim != 1:
            return None
        if not np.issubdtype(candidate.dtype, np.integer):
            return None
        return candidate.astype(np.int64, copy=False)

    if isinstance(candidate, (list, tuple)):
        if len(candidate) == 0:
            return np.empty(0, dtype=np.int64)
        if isinstance(candidate[0], (int, np.integer)):
            return np.asarray(candidate, dtype=np.int64)
        return None

    if isinstance(candidate, (int, np.integer)):
        return np.asarray([int(candidate)], dtype=np.int64)

    return None
