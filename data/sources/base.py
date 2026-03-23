"""
Data source abstraction for FloatSOM - SOLID compliant data access layer
Implements Strategy pattern for different data sources
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Iterator, Optional, Any
import numpy as np
import cupy as cp


class DataSource(ABC):
    """
    Abstract base class for data sources
    Follows Single Responsibility: Only responsible for data access
    Follows Dependency Inversion: High-level modules depend on this abstraction
    """
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the dataset without loading it
        
        Returns:
            Tuple of (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def get_initialization_sample(self, 
                                 max_samples: Optional[int] = None) -> cp.ndarray:
        """
        Get a sample of data for initialization (e.g., PCA)
        
        Args:
            max_samples: Maximum number of samples to return
            
        Returns:
            Sample of data for initialization
        """
        pass
    
    @abstractmethod
    def iterate_chunks(self, 
                      chunk_size: int) -> Iterator[cp.ndarray]:
        """
        Iterate over data in chunks
        
        Args:
            chunk_size: Size of each chunk
            
        Yields:
            Data chunks
        """
        pass
    
    @abstractmethod
    def get_reference(self) -> Any:
        """
        Get a reference to the data that can be passed to selectors/processors
        For arrays: returns the array itself
        For files: returns the file path
        
        Returns:
            Data reference (array or path)
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if this data source supports streaming
        
        Returns:
            True if streaming is supported
        """
        pass
    
    @abstractmethod
    def get_memory_requirement(self) -> int:
        """
        Estimate memory requirement in bytes
        
        Returns:
            Estimated memory requirement in bytes
        """
        pass
    
    def cleanup(self) -> None:
        """
        Optional cleanup method for resource management
        """
        pass
