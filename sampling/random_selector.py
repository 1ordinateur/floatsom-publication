"""
Random sample selector for stochastic training approaches
"""

import cupy as cp
import numpy as np
import logging
from typing import Optional, Union
from .base_selector import SampleSelector, SelectionResult

logger = logging.getLogger(__name__)


class RandomSelector(SampleSelector):
    """
    Random sample selection for stochastic training variants
    """
    
    def __init__(self, sample_config=None):
        """
        Initialize random selector
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.random_seed = sample_config.random_seed 
        self.whole_chunk_random = bool(getattr(sample_config, "whole_chunk_random", False))
        self.total_samples = None  # Will be set during initialization
        self.samples_per_epoch = sample_config.samples_per_epoch 
        if self.samples_per_epoch is None:
            self.target_proportion = sample_config.target_proportion 
        else:
            self.target_proportion = None
        if self.random_seed is not None:
            cp.random.seed(self.random_seed)

    def initialize_with_data_source(self, data_source):
        """
        Initialize selector with DataSource to get total sample count
        """
        # Resolve total sample count directly from the data source shape
        shape = data_source.get_shape()
        if not shape or shape[0] is None:
            raise ValueError("Data source did not provide a valid shape; subsampling requires sample count")
        self.total_samples = int(shape[0])

        if self.target_proportion is not None and self.total_samples:
            # target_proportion is defined as a fractional dataset share (e.g. 0.25 => 25%).
            self.samples_per_epoch = max(
                1,
                int(self.total_samples * float(self.target_proportion)),
            )

    def _draw_indices(
        self,
        total_count: int,
        *,
        use_cupy: bool,
    ) -> Optional[Union[np.ndarray, cp.ndarray]]:
        """
        Draw random row indices for the current epoch in backend-native format.
        """
        if self.samples_per_epoch is None or self.samples_per_epoch >= total_count:
            return None

        if use_cupy:
            return cp.random.choice(total_count, size=self.samples_per_epoch, replace=False)

        indices = np.random.choice(total_count, size=self.samples_per_epoch, replace=False)
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _to_numpy_indices(indices: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """
        Normalize backend-native indices into NumPy int64 for SelectionResult.indices.
        """
        if isinstance(indices, cp.ndarray):
            return cp.asnumpy(indices).astype(np.int64, copy=False)
        return np.asarray(indices, dtype=np.int64)

    def select(self, dataset) -> SelectionResult:
        """
        Canonical selector contract with both sample payload and global row indices.
        """
        if isinstance(dataset, str) or dataset is None:
            if self.total_samples is None:
                return SelectionResult(samples=None, indices=None)
            raw_indices = self._draw_indices(self.total_samples, use_cupy=False)
            if raw_indices is None:
                return SelectionResult(samples=None, indices=None)
            indices = self._to_numpy_indices(raw_indices)
            logger.debug("indices=%s", indices)
            return SelectionResult(samples=None, indices=indices)

        total_count = int(len(dataset))
        use_cupy = isinstance(dataset, cp.ndarray)
        raw_indices = self._draw_indices(total_count, use_cupy=use_cupy)
        if raw_indices is None:
            return SelectionResult(samples=dataset, indices=None)

        gathered = dataset[raw_indices]
        indices = self._to_numpy_indices(raw_indices)
        logger.debug("indices=%s", indices)
        return SelectionResult(samples=gathered, indices=indices)
        
    def select_samples(self, dataset):
        """
        Select random subset of samples
        
        For Ray processing: returns indices as numpy array
        For regular processing: returns actual selected data
        """
        if isinstance(dataset, str) or dataset is None:
            if self.total_samples is None:
                return None
            raw_indices = self._draw_indices(self.total_samples, use_cupy=False)
            if raw_indices is None:
                return None
            indices = self._to_numpy_indices(raw_indices)
            logger.debug("indices=%s", indices)
            return indices

        total_count = int(len(dataset))
        use_cupy = isinstance(dataset, cp.ndarray)
        raw_indices = self._draw_indices(total_count, use_cupy=use_cupy)
        if raw_indices is None:
            return dataset

        logger.debug("indices=%s", raw_indices)
        return dataset[raw_indices]
