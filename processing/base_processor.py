"""
Base abstract class for sample processing strategies
"""

from abc import ABC, abstractmethod
from typing import Optional
import cupy as cp


class ProcessingMethod(ABC):
    """
    Abstract base class for sample processing strategies
    """
    
    @abstractmethod
    def process_samples(self, samples, som_weights, topology, params):
        """
        Process samples and return updated weights
        
        Args:
            samples: Selected samples for this iteration
            som_weights: Current SOM weights
            topology: SOMTopology implementation
            params: Training parameters
            
        Returns:
            Updated SOM weights
        """
        pass
    
    def initialize(self, som_weights, topology, params):
        """
        Optional initialization with SOM configuration
        """
        pass
    
    def finalize(self):
        """
        Optional cleanup after training
        """
        pass
    
    def set_selector_callback(self, callback):
        """
        Set callback function for selector metadata updates (used by HDSSSOM)
        
        Args:
            callback: Function to call with (samples, bmus, distances) for metadata updates
        """
        pass

    def supports_selection_result(self) -> bool:
        """
        Whether process_samples accepts sampling.base_selector.SelectionResult directly.
        """
        return False

    def prefers_processor_local_sampling(self, data_reference, params) -> bool:
        """
        Whether the processor wants to own stochastic sampling locally instead of
        receiving a selector-materialized SelectionResult payload.
        """
        return False
