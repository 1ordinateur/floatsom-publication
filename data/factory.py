"""
Factory for creating appropriate data sources
Implements Factory pattern for SOLID compliance
"""

from typing import Union, Optional, Dict, Any
import numpy as np
import cupy as cp
from pathlib import Path
import logging

from .sources.base import DataSource
from .sources.array import ArrayDataSource
from .sources.file import FileDataSource
from ..floatsom_params import RayConfig

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """
    Factory for creating data sources
    Open/Closed Principle: New data sources can be added without modifying this class
    Single Responsibility: Only responsible for creating data sources
    """
    
    # Registry of data source creators
    _creators: Dict[str, Any] = {}
    
    @classmethod
    def register_creator(cls, data_type: str, creator):
        """
        Register a new data source creator
        Allows extending with new data sources without modifying the factory
        
        Args:
            data_type: Type identifier for the data source
            creator: Callable that creates a DataSource instance
        """
        cls._creators[data_type] = creator
    
    @staticmethod
    def create(data_input: Union[str, np.ndarray, cp.ndarray],
              ray_config: Optional[RayConfig] = None) -> DataSource:
        """
        Create appropriate data source based on input type
        
        Args:
            data_input: Data input (file path or array)
            ray_config: Optional RayConfig configuration for file sources
            
        Returns:
            Appropriate DataSource implementation
            
        Raises:
            ValueError: If data input type is not supported
        """
        # Check for custom registered creators first
        data_type = DataSourceFactory._identify_type(data_input)
        if data_type in DataSourceFactory._creators:
            return DataSourceFactory._creators[data_type](data_input, ray_config)

        # Default creation logic
        if isinstance(data_input, str):
            path = Path(data_input)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {data_input}")

            # Directory-backed stores (FastArrayStore, Zarr) are allowed even without a suffix.
            if path.is_dir():
                logger.info(f"Creating FileDataSource for directory store: {data_input}")
                return FileDataSource(data_input, ray_config)

            # File path - create FileDataSource, but guard unsupported extensions
            if DataSourceFactory._identify_type(data_input) == "generic_file":
                raise ValueError(f"Unsupported file extension: {path.suffix or '<none>'}")
            logger.info(f"Creating FileDataSource for: {data_input}")
            return FileDataSource(data_input, ray_config)
            
        elif isinstance(data_input, (np.ndarray, cp.ndarray)):
            # Array - create ArrayDataSource
            logger.info(f"Creating ArrayDataSource for array of shape: {data_input.shape}")
            return ArrayDataSource(data_input)
            
        else:
            raise ValueError(
                f"Unsupported data input type: {type(data_input)}. "
                "Expected file path (str) or array (numpy/cupy)"
            )
    
    @staticmethod
    def _identify_type(data_input: Any) -> str:
        """
        Identify the type of data input
        
        Args:
            data_input: Input data
            
        Returns:
            Type identifier string
        """
        if isinstance(data_input, str):
            # Could further differentiate by file extension
            path = Path(data_input)
            suffix = path.suffix.lower()
            if suffix in {'.npy', '.npz'}:
                return 'numpy_file'
            if suffix == '.parquet':
                return 'parquet_file'
            if suffix in {'.csv', '.txt'}:
                return 'csv_file'
            if suffix == '.zarr':
                return 'zarr_file'
            if suffix == '.fast':
                return 'fast_array_store'
            return 'generic_file'
        elif isinstance(data_input, np.ndarray):
            return 'numpy_array'
        elif isinstance(data_input, cp.ndarray):
            return 'cupy_array'
        else:
            return 'unknown'
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DataSource:
        """
        Create data source from configuration dictionary
        Useful for loading from config files
        
        Args:
            config: Configuration dictionary with 'type' and parameters
            
        Returns:
            Configured DataSource instance
        """
        source_type = config.get('type', 'array')
        
        if source_type == 'array':
            # Load array from file if path provided
            if 'path' in config:
                data = np.load(config['path'])
            else:
                raise ValueError("Array source requires 'path' in config")
            return ArrayDataSource(data)
            
        elif source_type == 'file':
            if 'path' not in config:
                raise ValueError("File source requires 'path' in config")
            
            # Create file source without RayConfig
            # For config-based creation, we don't have RayConfig yet
            return FileDataSource(config['path'], None)
            
        else:
            raise ValueError(f"Unknown data source type: {source_type}")


# Optional: Register custom creators for specific file types
def _create_numpy_file_source(file_path: str, ray_config: Optional[RayConfig]) -> DataSource:
    """Specialized creator for NumPy files"""
    logger.debug(f"Creating optimized NumPy file source for: {file_path}")
    return FileDataSource(file_path, ray_config)


def _create_parquet_file_source(file_path: str, ray_config: Optional[RayConfig]) -> DataSource:
    """Specialized creator for Parquet files"""
    logger.debug(f"Creating Parquet file source for: {file_path}")
    # Could have specialized Parquet handling here
    return FileDataSource(file_path, ray_config)


# Register specialized creators
DataSourceFactory.register_creator('numpy_file', _create_numpy_file_source)
DataSourceFactory.register_creator('parquet_file', _create_parquet_file_source)
