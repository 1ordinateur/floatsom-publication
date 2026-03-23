"""
HDSSSOM (Hierarchical Dynamic Subset Selection) intelligent sample selector
Based on "Speeding up the Self-Organizing Feature Map Using Dynamic Subset Selection" by Wetmore et al. (2005)
Uses fully vectorized GPU operations for maximum efficiency
"""

import cupy as cp
import numpy as np
from typing import Union, Optional

from .base_selector import SampleSelector
from .hdsssom_components import HDSSSOMMetadata


class HDSSSOMSelector(SampleSelector):
    """
    HDSSSOM intelligent sample selection using difficulty and age metrics
    
    Two-level hierarchical selection:
    1. Block selection: 90% difficulty, 10% age weighting
    2. Exemplar selection: 70% difficulty, 30% age weighting
    """
    
    def __init__(self, config):
        """
        Initialize HDSSSOM selector
        
        Args:
            config: Configuration object with HDSSSOM parameters
        """
        # HDSSSOM parameters from config
        self.block_size = config.block_size
        self.p_block_difficulty = config.p_block_difficulty
        self.p_exemplar_difficulty = config.p_exemplar_difficulty
        self.alpha = config.alpha  # Difficulty smoothing factor
        
        # Selection parameters
        self.samples_per_epoch = config.samples_per_epoch
        self.target_proportion = None  # Initialize to None
        if self.samples_per_epoch is None:
            self.target_proportion = config.target_proportion
        self.min_blocks_to_select = config.min_blocks_to_select

        
        # State variables
        self.dataset = None
        self.metadata = None
        self.dataset_size = 0
        self.initialized = False
        
        # Performance tracking
        self.selection_stats = {
            'total_selections': 0,
            'avg_difficulty': 0.0,
            'avg_age': 0.0
        }
        
    def initialize(self, dataset: Union[cp.ndarray, np.ndarray]):
        """
        Initialize metadata tracking for the dataset
        
        Args:
            dataset: Training dataset (n_samples, n_features)
        """
        # Ensure dataset is on GPU
        if isinstance(dataset, np.ndarray):
            self.dataset = cp.asarray(dataset)
        else:
            self.dataset = dataset
            
        self.dataset_size = len(self.dataset)

        if self.target_proportion is not None and self.samples_per_epoch is None:
            self.samples_per_epoch = max(1, int(self.dataset_size * self.target_proportion))
        else: 
            # If target_proportion is not set, then we are using samples_per_epoch
            pass 

        # Initialize vectorized metadata tracking
        self.metadata = HDSSSOMMetadata(self.dataset_size, self.block_size)
                
        self.initialized = True
        
    def select_samples(self, dataset: Union[cp.ndarray, np.ndarray]) -> cp.ndarray:
        """
        Two-level HDSSSOM selection: blocks then exemplars
        
        Args:
            dataset: Full dataset (should match initialized dataset)
            target_size: Number of samples to select (uses samples_per_epoch if None)
            
        Returns:
            Selected samples as CuPy array (target_size, n_features)
        """
        if not self.initialized:
            raise ValueError("HDSSSOMSelector must be initialized before use")
            
        # Determine selection size
        if self.samples_per_epoch <= 0:
            return cp.empty((0, self.dataset.shape[1]), dtype=self.dataset.dtype)
        
        # Calculate selection strategy
        num_blocks_to_select, samples_per_block = self._calculate_selection_strategy(self.samples_per_epoch)
        
        # Phase 1: Block selection (90% difficulty, 10% age)
        selected_block_indices = self.metadata.select_blocks_vectorized(
            num_blocks_to_select, self.p_block_difficulty
        )
        
        # Phase 2: Exemplar selection (70% difficulty, 30% age)
        if len(selected_block_indices) == 0:
            return cp.empty((0, self.dataset.shape[1]), dtype=self.dataset.dtype)
        
        # Use the fully vectorized exemplar selection
        selected_sample_indices = self.metadata.select_exemplars_fully_vectorized(
            selected_block_indices, samples_per_block, self.p_exemplar_difficulty
        )
        
        # Limit to target size (in case we selected more due to block boundaries)
        if len(selected_sample_indices) > self.samples_per_epoch:
            # Use random subsampling to reach exact target size
            subsample_indices = cp.random.choice(
                len(selected_sample_indices), self.samples_per_epoch, replace=False
            )
            selected_sample_indices = selected_sample_indices[subsample_indices]
        
        # Store selection indices for metadata updates
        self.last_selected_indices = selected_sample_indices
        
        # Update selection statistics
        self._update_selection_stats(selected_sample_indices)
        
        # Return selected samples
        if len(selected_sample_indices) == 0:
            return cp.empty((0, self.dataset.shape[1]), dtype=self.dataset.dtype)
        
        return self.dataset[selected_sample_indices]

    def get_selected_indices(self, dataset, selected_samples):
        """
        Provide selector indices for the canonical SelectionResult contract.
        """
        del dataset, selected_samples
        selected = getattr(self, "last_selected_indices", None)
        if selected is None:
            return None
        if isinstance(selected, cp.ndarray):
            return cp.asnumpy(selected).astype(np.int64, copy=False)
        selected_arr = np.asarray(selected)
        if selected_arr.ndim != 1:
            return None
        return selected_arr.astype(np.int64, copy=False)
        
    def update_metadata(self, selected_samples: cp.ndarray, bmus: cp.ndarray, distances: cp.ndarray):
        """
        Update difficulty and age metadata after BMU calculation
        Following Wetmore et al. - updates difficulty for PROCESSED samples, age for ALL samples
        
        Args:
            selected_samples: Samples that were selected and processed
            bmus: BMU indices for selected samples  
            distances: Distances to BMUs for selected samples
        """
        if not self.initialized or not hasattr(self, 'last_selected_indices'):
            return
            
        # Create selection mask for processed samples
        selected_mask = cp.zeros(self.dataset_size, dtype=cp.bool_)
        
        # Fill in mask for selected samples
        if len(distances) > 0 and len(self.last_selected_indices) > 0:
            valid_count = min(len(distances), len(self.last_selected_indices))
            valid_indices = self.last_selected_indices[:valid_count]
            selected_mask[valid_indices] = True
            
            # Calculate time-varying alpha (decaying over epochs)
            current_alpha = self._calculate_time_varying_alpha()
            
            # Update difficulties for PROCESSED samples using Wetmore et al. formula:
            # d(v) = α(t) * ||x - w_i|| + (1 - α(t)) * d(v-1)
            self.metadata.difficulties[valid_indices] = (
                current_alpha * distances[:valid_count] + 
                (1.0 - current_alpha) * self.metadata.difficulties[valid_indices]
            )
        
        # Update ages for ALL samples (Wetmore et al. approach)
        # Selected samples: reset age to 1 (not 0)
        # Non-selected samples: increment age by 1
        self.metadata.update_ages(selected_mask)
        
        # Update global selection statistics
        self.selection_stats['total_selections'] += 1
        
    def _calculate_time_varying_alpha(self) -> float:
        """
        Calculate time-varying alpha as per Wetmore et al.
        α(t) decays over time to give less weight to current distance, more to history
        
        Returns:
            Current alpha value
        """
        if not hasattr(self, 'total_iterations_estimate'):
            # Estimate based on typical SOM training (can be overridden)
            self.total_iterations_estimate = 1000
            
        current_iteration = self.selection_stats['total_selections']
        
        # Exponential decay: α(t) = α_initial * exp(-λt) where λ controls decay rate
        # Following Wetmore et al. principle of decreasing influence of current distance
        decay_rate = 3.0 / self.total_iterations_estimate  # Decay to ~5% by end
        current_alpha = self.alpha * cp.exp(-decay_rate * current_iteration)
        
        # Ensure minimum alpha to prevent complete history dominance
        min_alpha = 0.1
        return max(float(current_alpha), min_alpha)
        
    def _calculate_selection_strategy(self, target_size: int) -> tuple[int, int]:
        """
        Calculate optimal block and exemplar selection strategy
        
        Args:
            target_size: Total number of samples to select
            
        Returns:
            num_blocks_to_select: Number of blocks to select
            samples_per_block: Average samples to select per block
        """
        # Ensure we select at least minimum blocks
        max_possible_blocks = self.metadata.num_blocks
        
        if target_size <= self.block_size:
            # Select from single block or few blocks
            num_blocks_to_select = min(max(1, target_size // (self.block_size // 2)), max_possible_blocks)
            samples_per_block = (target_size + num_blocks_to_select - 1) // num_blocks_to_select
        else:
            # Select from multiple blocks
            # Aim for roughly equal distribution across blocks
            num_blocks_to_select = min(max(self.min_blocks_to_select, target_size // self.block_size + 1), max_possible_blocks)
            samples_per_block = (target_size + num_blocks_to_select - 1) // num_blocks_to_select
        
        return num_blocks_to_select, samples_per_block
        
    def _update_selection_stats(self, selected_indices: cp.ndarray):
        """
        Update selection statistics for monitoring
        
        Args:
            selected_indices: Indices of selected samples
        """
        if len(selected_indices) == 0:
            return
            
        # Calculate average difficulty and age of selected samples
        selected_difficulties = self.metadata.difficulties[selected_indices]
        selected_ages = self.metadata.ages[selected_indices]
        
        avg_difficulty = float(cp.mean(selected_difficulties))
        avg_age = float(cp.mean(selected_ages.astype(cp.float32)))
        
        # Exponential moving average of statistics
        alpha_stats = 0.1
        self.selection_stats['avg_difficulty'] = (
            alpha_stats * avg_difficulty + (1 - alpha_stats) * self.selection_stats['avg_difficulty']
        )
        self.selection_stats['avg_age'] = (
            alpha_stats * avg_age + (1 - alpha_stats) * self.selection_stats['avg_age']
        )
        
    def get_selection_stats(self) -> dict:
        """
        Get current selection statistics
        
        Returns:
            Dictionary with selection statistics
        """
        stats = self.selection_stats.copy()
        
        if self.initialized:
            stats.update({
                'dataset_size': self.dataset_size,
                'block_size': self.block_size,
                'num_blocks': self.metadata.num_blocks,
                'samples_per_epoch': self.samples_per_epoch,
                'current_avg_difficulty': float(cp.mean(self.metadata.difficulties)),
                'current_avg_age': float(cp.mean(self.metadata.ages.astype(cp.float32))),
                'difficulty_std': float(cp.std(self.metadata.difficulties)),
                'age_std': float(cp.std(self.metadata.ages.astype(cp.float32)))
            })
            
        return stats
        
    def reset_metadata(self):
        """
        Reset all metadata to initial state (for debugging/testing)
        """
        if self.initialized:
            self.metadata = HDSSSOMMetadata(self.dataset_size, self.block_size)
            self.selection_stats = {
                'total_selections': 0,
                'avg_difficulty': 0.0,
                'avg_age': 0.0
            }
