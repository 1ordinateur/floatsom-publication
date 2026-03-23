"""
Base topology interface for FloatSOM
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Union
import cupy as cp
import numpy as np


class SOMTopology(ABC):
    """
    Abstract base class for SOM topology strategies
    """
    
    def __init__(self):
        """Initialize base topology attributes"""
        # Reformation tracking (set by FloatSOM when grid reformation occurs)
        self.is_reformed = False
        self.reformed_type = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this topology strategy"""
        pass
    
    @property
    @abstractmethod
    def total_nodes(self) -> int:
        """Get the total number of nodes in this topology"""
        pass
    
    @abstractmethod
    def initialize_weights(self, data: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Initialize weight vectors according to topology
        
        Args:
            data: Optional input data for initialization (e.g., PCA)
            
        Returns:
            Initialized weight vectors
        """
        pass
    
    
    
    @abstractmethod
    def get_coordinates(self) -> cp.ndarray:
        """Get the coordinate grid for this topology"""
        pass
    
    @abstractmethod
    def update_topology(self, weights: cp.ndarray, current_iteration: int = None) -> None:
        """Update the topology structure if dynamic (e.g., MST)"""
        pass
    
    @abstractmethod
    def precompute_topology_data(self, som_params, radii_list: List[float]) -> None:
        """
        Precompute all topology-related distance and influence matrices.
        This should be called once before training starts.
        
        Args:
            som_params: SOM parameters containing training parameters
            radii_list: List of radius values to precompute for
        """
        pass
    
    @abstractmethod
    def get_precomputed_influence_matrix(self, radius: float, influence_function: str = 'gaussian') -> cp.ndarray:
        """
        Get cached continuous influence map for weight updates with optimized precomputation.
        
        Args:
            radius: Current neighborhood radius
            influence_function: Type of influence function ('gaussian', 'bubble', 'mexican_hat')
            
        Returns:
            weight_influence_map: Precomputed continuous influence matrix for weight updates
            
        Raises:
            ValueError: If radius is not positive
            ValueError: If influence_function is not supported
        """
        # Basic validation
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        
        if influence_function not in ['gaussian', 'bubble', 'mexican_hat']:
            raise ValueError(f"Unsupported influence function: {influence_function}. Supported: gaussian, bubble, mexican_hat")
        
        # Subclasses implement the actual computation
        pass
    
    @abstractmethod
    def set_precomputed_radii(self, radii_list: List[float]) -> None:
        """
        Set precomputed radii list
        
        Args:
            radii_list: List of radius values
        """
        pass
    
    @abstractmethod
    def get_distance_matrix_for_color_sets(self, current_iteration: int = None) -> cp.ndarray:
        """
        Get precomputed distance matrix for color set calculation.
        Returns raw squared distances - algorithms apply radius thresholds as needed.
        
        Args:
            current_iteration: Current training iteration (optional)
            
        Returns:
            distance_matrix: Precomputed squared distance matrix between all nodes
        """
        pass
    
    def requires_topology_updates(self) -> bool:
        """
        Whether this topology needs periodic structure updates during training.
        """
        return False

    def should_update_topology(self, current_iteration: int) -> bool:
        """
        Check whether topology should update at the current iteration.
        """
        return self.requires_topology_updates()

    def finalize_topology(self, weights: cp.ndarray) -> None:
        """
        Optional finalize hook for topologies that need a final structure refresh.
        """
        return None
    
    def validate_shape(self, shape):
        """
        Validate SOM shape for this topology
        """
        pass
