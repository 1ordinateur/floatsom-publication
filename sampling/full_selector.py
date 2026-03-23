"""
Full sample selector - processes all samples (current behavior)
Supports both in-memory arrays and file paths for GDS streaming
"""

from .base_selector import SampleSelector
from typing import Union, Tuple
import numpy as np


class FullSelector(SampleSelector):
    """
    Select all samples - maintains existing behavior for compatibility
    When given a file path, passes it through for the processor to handle via GDS
    """
    
    def __init__(self):
        """
        Initialize full selector
        """
        self.is_file_path = False
        self.dataset_shape = None
        self.data_path = None
    
    def initialize(self, dataset):
        """
        Initialize selector with dataset or file path
        
        Args:
            dataset: numpy/cupy array or file path string
        """
        if isinstance(dataset, str):
            # File path mode
            self.is_file_path = True
            self.data_path = dataset
        else:
            # Array mode (existing behavior)
            self.is_file_path = False
            super().initialize(dataset)
    
    def initialize_with_metadata(self, dataset_shape: Tuple[int, int], data_path: str):
        """
        Initialize with metadata for file-based processing
        
        Args:
            dataset_shape: Tuple of (n_samples, n_features)
            data_path: Path to data file
        """
        self.is_file_path = True
        self.dataset_shape = dataset_shape
        self.data_path = data_path
    
    def select_samples(self, dataset):
        """
        Return entire dataset or file path
        
        For file paths, the processor will handle GDS streaming
        For arrays, returns the entire array (existing behavior)
        """
        # If dataset is a file path, pass it through
        if isinstance(dataset, str):
            return dataset
        
        # Otherwise, return the entire array (existing behavior)
        return dataset