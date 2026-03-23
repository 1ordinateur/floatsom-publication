#!/usr/bin/env python3
"""
Fast Array IO using memory-mapped files
A high-performance alternative to Zarr for large array reading on HPC systems
"""

import numpy as np
import cupy as cp
import time
import os
import json
import mmap
from typing import Tuple, Optional, Dict, Any

class FastArrayStore:
    """
    High-performance array storage using memory-mapped files.
    Designed to minimize overhead for GPU data loading.
    """
    
    def __init__(self, path: str, mode: str = 'r'):
        """
        Initialize the array store.
        
        Args:
            path: Directory path for the array store
            mode: 'r' for reading, 'w' for writing
        """
        self.path = path
        self.mode = mode
        self.metadata_file = os.path.join(path, 'array_metadata.json')
        self.data_file = os.path.join(path, 'array_data.bin')
        
        if mode == 'r':
            self._load_metadata()
            self._open_for_reading()
        elif mode == 'w':
            os.makedirs(path, exist_ok=True)
    
    def _load_metadata(self):
        """Load array metadata from JSON file"""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.shape = tuple(metadata['shape'])
        self.dtype = np.float32  # Always use float32
        self.chunks = tuple(metadata.get('chunks', [1000000, self.shape[1] if len(self.shape) > 1 else 1]))
        self.n_samples = self.shape[0]
        self.n_features = self.shape[1] if len(self.shape) > 1 else 1
        self.chunk_samples = self.chunks[0]
        self.itemsize = self.dtype.itemsize
    
    def _open_for_reading(self):
        """Open the data file for memory-mapped reading"""
        # Standard memory-mapped array
        self.mmap_array = np.memmap(
            self.data_file,
            dtype=np.float32,  # Always use float32
            mode='r',
            shape=self.shape
        )
        
        # Also open with mmap for advanced options
        self.file_handle = open(self.data_file, 'rb')
        self.mmap_handle = mmap.mmap(
            self.file_handle.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
        
        # Pre-advise the kernel about our access pattern
        if hasattr(self.mmap_handle, 'madvise'):
            try:
                self.mmap_handle.madvise(2)  # MADV_SEQUENTIAL = 2 on Linux
            except:
                pass
    
    def create(self, shape: Tuple[int, ...], dtype: Any = None, chunks: Optional[Tuple[int, ...]] = None):
        """Create a new array store (dtype parameter ignored, always uses float32)"""
        if self.mode != 'w':
            raise ValueError("Cannot create array in read mode")
        
        self.shape = shape
        self.dtype = np.float32  # Always use float32
        
        # Default chunks if not provided
        if chunks is None:
            if len(shape) > 1:
                chunks = (min(1000000, shape[0]), shape[1])
            else:
                chunks = (min(1000000, shape[0]),)
        self.chunks = chunks
        
        # Save metadata
        metadata = {
            'shape': list(shape),
            'dtype': 'float32',  # Always save as float32
            'chunks': list(chunks),
            'version': '1.0'
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create memory-mapped array for writing
        self.mmap_array = np.memmap(
            self.data_file,
            dtype=np.float32,  # Always use float32
            mode='w+',
            shape=self.shape
        )
        
        return self.mmap_array
    
    def close(self):
        """Close the memory-mapped file"""
        if hasattr(self, 'mmap_array'):
            del self.mmap_array
        if hasattr(self, 'mmap_handle'):
            self.mmap_handle.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()