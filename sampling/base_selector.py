"""
Base abstract class for sample selection strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    """
    Canonical selector payload consumed by training processors.

    Attributes:
        samples: Selected sample payload for in-memory processors. May be None when
            selection is index-only (e.g., Ray random subsampling over file-backed data).
        indices: Global selected row indices when available.
    """
    samples: Any
    indices: Optional[np.ndarray]


class SampleSelector(ABC):
    """
    Abstract base class for sample selection strategies
    """
    
    @staticmethod
    def _looks_like_index_payload(payload: Any) -> bool:
        """
        Heuristic guard to reject index-only payloads that are incorrectly returned
        as sample data under the SelectionResult contract.
        """
        if payload is None:
            return False
        if isinstance(payload, (Integral, np.integer)):
            return True
        if isinstance(payload, np.ndarray):
            return payload.ndim == 1 and np.issubdtype(payload.dtype, np.integer)
        if isinstance(payload, (list, tuple)):
            if not payload:
                return False
            return all(isinstance(item, (Integral, np.integer)) for item in payload)
        # Handle array-likes from non-NumPy backends (e.g. CuPy) without importing them.
        ndim = getattr(payload, "ndim", None)
        dtype = getattr(payload, "dtype", None)
        if ndim == 1 and dtype is not None:
            dtype_kind = getattr(dtype, "kind", None)
            if dtype_kind in {"i", "u"}:
                return True
        return False

    @abstractmethod
    def select_samples(self, dataset):
        """
        Select samples from dataset for training iteration
        
        Args:
            dataset: Full dataset to select from
            
        Returns:
            Selected samples subset
        """
        pass

    def get_selected_indices(self, dataset, selected_samples):
        """
        Optional hook for selectors that can provide global selected row indices.
        """
        return None

    def select(self, dataset) -> SelectionResult:
        """
        Canonical selection entry point.

        Default behavior wraps select_samples() output into SelectionResult. Selectors
        that can compute indices should override get_selected_indices() or select().
        """
        selected_samples = self.select_samples(dataset)
        selected_indices = self.get_selected_indices(dataset, selected_samples)
        if selected_indices is None and self._looks_like_index_payload(selected_samples):
            raise TypeError(
                f"{type(self).__name__}.select_samples() returned index-like payload without "
                "declaring selected indices. Implement get_selected_indices() or override select()."
            )
        return SelectionResult(samples=selected_samples, indices=selected_indices)
    
    def initialize(self, dataset):
        """
        Optional initialization with full dataset
        """
        pass
    
    def update_metadata(self, selected_samples, bmus, distances=None):
        """
        Optional metadata update after BMU calculation
        """
        pass
    
    def initialize_with_data_source(self, data_source):
        """
        Initialize selector with DataSource abstraction
        Default implementation delegates to initialize()

        Args:
            data_source: DataSource instance
        """
        # Default: get reference and call initialize
        data_reference = data_source.get_reference()
        self.initialize(data_reference)
