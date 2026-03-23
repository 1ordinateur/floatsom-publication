"""
BMU scheduler for colour set processing
Manages BMU recalculation during colour set rounds
"""

import numpy as np
import cupy as cp
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class BMUScheduler:
    """
    Schedules BMU recalculations during colour set processing
    GPU-only implementation for FloatSOM
    """
    
    def __init__(self, enabled: bool = False, initial_frequency: Optional[int] = None, 
                 decay_type: str = 'exponential', min_samples: int = 1000,
                 verbose: bool = False):
        """
        Initialize BMU scheduler
        
        Args:
            enabled: Enable adaptive BMU recalculation
            initial_frequency: Initial number of recalculations per iteration
            decay_type: Decay function for recalculation frequency
            min_samples: Minimum samples required to enable adaptive recalc
            verbose: Whether to print debug information
        """
        self.enabled = enabled
        self.initial_frequency = initial_frequency
        self.decay_type = decay_type
        self.min_samples = min_samples
        self.verbose = verbose
        
        # Tracking state - GPU arrays only
        self.processed_mask = None
        self.total_samples = 0
        self.recalc_count = 0
    
    def initialize_iteration(self, num_samples: int, num_color_sets: int):
        """
        Initialize tracking for a new iteration.
        
        Args:
            num_samples: Total number of samples
            num_color_sets: Total number of color sets
        """
        self.total_samples = num_samples
        self.recalc_count = 0
        
        # Set initial frequency if not specified
        if self.initial_frequency is None:
            self.initial_frequency = num_color_sets
            
        # Disable for small datasets
        if num_samples < self.min_samples and self.enabled:
            self.enabled = False
            if self.verbose:
                logger.info(f"Adaptive BMU disabled: {num_samples} samples < {self.min_samples} threshold")
        
        # Initialize processed mask on GPU
        if self.enabled:
            self.processed_mask = cp.zeros(num_samples, dtype=cp.bool_)
        else:
            self.processed_mask = None
    
    def get_recalc_frequency(self, iteration: int, total_iterations: int) -> int:
        """
        Calculate the number of BMU recalculations for the current iteration.
        Uses the same decay functions as learning rate and radius for consistency.
        
        Args:
            iteration: Current iteration number (0-based)
            total_iterations: Total number of iterations
            
        Returns:
            Number of times to recalculate BMUs in this iteration
        """
        if not self.enabled or self.initial_frequency is None:
            return 1  # Only initial calculation
            
        # Use the same decay logic as learning rate and radius
        if self.decay_type == 'exponential':
            # Exponential decay - more aggressive for faster reduction
            decay_factor = np.exp(-iteration / (total_iterations / 8))
            frequency = max(1, int(self.initial_frequency * decay_factor))
        elif self.decay_type == 'linear':
            # Linear decay - same as LR/radius
            progress = iteration / total_iterations
            frequency = max(1, int(self.initial_frequency * (1 - progress)))
        elif self.decay_type == 'sigmoid':
            # Sigmoid decay - sharper transition for faster reduction
            half_point = total_iterations / 2
            sigmoid_scale = total_iterations / 20
            decay_factor = 1 / (1 + np.exp((iteration - half_point) / sigmoid_scale))
            frequency = max(1, int(self.initial_frequency * decay_factor))
        elif self.decay_type == 'gaussian':
            # Gaussian decay - narrower curve for faster reduction
            decay_factor = np.exp(-iteration**2 / (2 * (total_iterations/6)**2))
            frequency = max(1, int(self.initial_frequency * decay_factor))
        elif self.decay_type == 'asymptotic':
            # Asymptotic decay - steeper for faster reduction
            decay_factor = 1 / (1 + iteration / (total_iterations / 5))
            frequency = max(1, int(self.initial_frequency * decay_factor))
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return frequency
    
    def should_recalculate_round(self, 
                                round_idx: int, 
                                total_rounds: int,
                                iteration: int,
                                total_iterations: int) -> bool:
        """
        Determine if BMUs should be recalculated at the start of this round.
        
        Args:
            round_idx: Current round index (0-based)
            total_rounds: Total number of rounds
            iteration: Current iteration number
            total_iterations: Total number of iterations
            
        Returns:
            True if BMUs should be recalculated
        """
        if not self.enabled:
            return round_idx == 0  # Only recalculate at start
            
        # Get target frequency for this iteration
        target_frequency = self.get_recalc_frequency(iteration, total_iterations)
        
        # Calculate rounds per recalculation
        rounds_per_recalc = max(1, total_rounds // target_frequency)
        
        # Check if it's time to recalculate
        # This will naturally handle the first recalc (when round_idx == 0)
        # and subsequent recalcs based on the frequency
        should_recalc = (round_idx % rounds_per_recalc == 0) and (self.recalc_count < target_frequency)
        
        if should_recalc:
            self.recalc_count += 1
            
        return should_recalc
    
    def mark_samples_processed(self, sample_indices: Union[cp.ndarray, np.ndarray]):
        """
        Mark samples as processed to avoid recalculating their BMUs.
        
        Args:
            sample_indices: Indices of samples that have been processed
        """
        if self.processed_mask is None or not self.enabled:
            return
        
        # Ensure indices are on GPU
        if isinstance(sample_indices, np.ndarray):
            sample_indices = cp.asarray(sample_indices)
        
        # Mark as processed
        self.processed_mask[sample_indices] = True
        
        if self.verbose:
            processed_count = int(cp.sum(self.processed_mask))
            logger.debug(f"Marked {len(sample_indices)} samples as processed. "
                  f"Total processed: {processed_count}/{self.total_samples}")
    
    def get_unprocessed_mask(self) -> cp.ndarray:
        """
        Get a boolean mask for samples that haven't been processed yet.
        
        Returns:
            Boolean mask where True indicates unprocessed samples
        """
        if self.processed_mask is None or not self.enabled:
            # All samples are unprocessed
            return cp.ones(self.total_samples, dtype=cp.bool_)
        
        # Return inverse of processed mask
        return ~self.processed_mask
    
    def get_unprocessed_indices(self) -> cp.ndarray:
        """
        Get indices of samples that haven't been processed yet.
        
        Returns:
            Array of unprocessed sample indices on GPU
        """
        if self.processed_mask is None or not self.enabled:
            # All samples are unprocessed
            return cp.arange(self.total_samples, dtype=cp.int32)
        
        # Get unprocessed indices
        unprocessed_mask = ~self.processed_mask
        return cp.where(unprocessed_mask)[0].astype(cp.int32)
    
    def has_unprocessed_samples(self) -> bool:
        """
        Check if there are any unprocessed samples remaining.
        
        Returns:
            True if there are unprocessed samples
        """
        if self.processed_mask is None or not self.enabled:
            return True
            
        return bool(cp.any(~self.processed_mask))
    
    def reset(self):
        """Reset the scheduler state for a new iteration."""
        if self.processed_mask is not None:
            self.processed_mask.fill(False)
        self.recalc_count = 0
        
    def get_stats(self) -> dict:
        """Get current scheduler statistics."""
        if self.processed_mask is not None:
            processed_count = int(cp.sum(self.processed_mask))
        else:
            processed_count = 0
            
        return {
            'enabled': self.enabled,
            'decay_type': self.decay_type,
            'recalc_count': self.recalc_count,
            'processed_samples': processed_count,
            'total_samples': self.total_samples,
            'completion_percentage': (processed_count / self.total_samples * 100) 
                                   if self.total_samples > 0 else 0,
            'device': 'GPU'
        }