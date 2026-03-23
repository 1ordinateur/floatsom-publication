"""
Array-based data source implementation
Handles in-memory numpy/cupy arrays
"""

from typing import Union, Tuple, Iterator, Optional
import numpy as np
import cupy as cp
from .base import DataSource


class ArrayDataSource(DataSource):
    """
    Data source for in-memory arrays
    Single Responsibility: Manages access to array-based data
    """
    
    def __init__(self, data: Union[np.ndarray, cp.ndarray]):
        """
        Initialize with array data
        
        Args:
            data: Input array (numpy or cupy)
        """
        self.data = data
        self.is_gpu = isinstance(data, cp.ndarray)
        self._shape = data.shape
        
    def get_shape(self) -> Tuple[int, int]:
        """Get dataset shape without computation"""
        return self._shape
    
    def get_initialization_sample(self, 
                                 max_samples: Optional[int] = None) -> cp.ndarray:
        """
        Get sample for initialization
        For arrays, returns the full array or a subset
        
        Args:
            max_samples: Maximum samples to return
        Returns:
            Sample data
        """
        # Determine how many samples to return
        n_samples = self._shape[0]
        if max_samples is not None and max_samples < n_samples:
            sample_data = self.data[:max_samples]
        else:
            sample_data = self.data
        
        return cp.asarray(sample_data)
    
    def iterate_chunks(self, 
                      chunk_size: int) -> Iterator[cp.ndarray]:
        """
        Iterate over array in chunks
        
        Args:
            chunk_size: Size of each chunk
        Yields:
            Data chunks
        """
        n_samples = self._shape[0]
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = self.data[start_idx:end_idx]
            
            yield cp.asarray(chunk)
    
    def get_reference(self) -> Union[np.ndarray, cp.ndarray]:
        """
        Return the array itself as reference
        
        Returns:
            The underlying array
        """
        return self.data
    
    def supports_streaming(self) -> bool:
        """
        Arrays don't need streaming (already in memory)
        
        Returns:
            False
        """
        return False
    
    def get_memory_requirement(self) -> int:
        """
        Calculate memory requirement for the array
        
        Returns:
            Memory in bytes
        """
        # Assuming float32 (4 bytes per element)
        return self.data.nbytes if hasattr(self.data, 'nbytes') else self.data.size * 4
