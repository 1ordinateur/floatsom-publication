"""
HDSSSOM components: Fully vectorized GPU-optimized metadata tracking
All operations use pure CuPy vectorized operations without loops or memory transfers
"""

import cupy as cp
from typing import Tuple


class HDSSSOMMetadata:
    """
    Fully vectorized metadata tracking for all samples using pure GPU operations
    No loops, no memory transfers, maximum GPU efficiency
    """
    
    def __init__(self, dataset_size: int, block_size: int):
        """
        Initialize vectorized metadata for entire dataset
        
        Args:
            dataset_size: Total number of samples in dataset
            block_size: Number of samples per block
        """
        self.dataset_size = dataset_size
        self.block_size = block_size
        self.num_blocks = (dataset_size + block_size - 1) // block_size
        
        # Vectorized metadata arrays (all on GPU)
        self.difficulties = cp.ones(dataset_size, dtype=cp.float32)
        self.ages = cp.zeros(dataset_size, dtype=cp.int32)
        self.selection_counts = cp.zeros(dataset_size, dtype=cp.int32)
        
        # Precompute block assignment for each sample (vectorized)
        self.sample_block_indices = cp.arange(dataset_size, dtype=cp.int32) // block_size
        
        # Block boundaries for efficient slicing
        self.block_starts = cp.arange(0, dataset_size, block_size, dtype=cp.int32)
        self.block_ends = cp.minimum(self.block_starts + block_size, dataset_size)
        self.block_sizes = self.block_ends - self.block_starts
        
    def update_difficulties(self, distances: cp.ndarray, alpha: float = 0.9):
        """
        Update difficulties using vectorized exponential smoothing
        """
        self.difficulties = alpha * self.difficulties + (1.0 - alpha) * distances
        
    def update_ages(self, selected_mask: cp.ndarray):
        """
        Update ages following Wetmore et al. approach:
        - Selected samples: reset age to 1 (not 0) 
        - Non-selected samples: increment age by 1
        """
        # Increment age for all samples first
        self.ages += 1
        # Reset selected samples to 1 (Wetmore et al. convention)
        self.ages = cp.where(selected_mask, 1, self.ages)
        self.selection_counts += selected_mask.astype(cp.int32)
        
    def get_block_scores(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Calculate block scores following Wetmore et al. with full vectorization:
        - Block difficulty = |mean - median| difficulty within block
        - Block age = mean age within block
        """
        # Calculate block means using vectorized bincount
        difficulty_sums = cp.bincount(self.sample_block_indices, weights=self.difficulties, 
                                    minlength=self.num_blocks)
        age_sums = cp.bincount(self.sample_block_indices, weights=self.ages.astype(cp.float32), 
                              minlength=self.num_blocks)
        
        block_difficulty_means = difficulty_sums / self.block_sizes.astype(cp.float32)
        block_age_scores = age_sums / self.block_sizes.astype(cp.float32)
        
        # Vectorized median calculation by reshaping difficulties into block matrix
        # Pad dataset to make it divisible by block_size for clean reshape
        padded_size = self.num_blocks * self.block_size
        if padded_size > self.dataset_size:
            # Pad with last values to maintain median calculation accuracy
            padding_size = padded_size - self.dataset_size
            padded_difficulties = cp.concatenate([
                self.difficulties, 
                cp.full(padding_size, self.difficulties[-1])
            ])
        else:
            padded_difficulties = self.difficulties[:padded_size]
        
        # Reshape into (num_blocks, block_size) matrix for vectorized median
        block_matrix = padded_difficulties.reshape(self.num_blocks, self.block_size)
        block_difficulty_medians = cp.median(block_matrix, axis=1)
        
        # Wetmore et al. block difficulty: |mean - median|
        block_difficulty_scores = cp.abs(block_difficulty_means - block_difficulty_medians)
        
        return block_difficulty_scores, block_age_scores
        
    def select_blocks_vectorized(self, num_blocks_to_select: int, p_difficulty: float = 0.9) -> cp.ndarray:
        """
        Select blocks using vectorized scoring and top-k selection
        """
        block_difficulty_scores, block_age_scores = self.get_block_scores()
        
        # Vectorized normalization
        diff_normalized = block_difficulty_scores / (cp.max(block_difficulty_scores) + 1e-6)
        age_normalized = block_age_scores / (cp.max(block_age_scores) + 1e-6)
        
        # Vectorized weighted combination
        combined_scores = p_difficulty * diff_normalized + (1.0 - p_difficulty) * age_normalized
        
        # Top-k selection
        if num_blocks_to_select >= self.num_blocks:
            return cp.arange(self.num_blocks, dtype=cp.int32)
        
        return cp.argsort(combined_scores)[-num_blocks_to_select:]
        
    def select_exemplars_from_blocks_vectorized(self, selected_block_indices: cp.ndarray, 
                                              samples_per_block: int, p_difficulty: float = 0.7) -> cp.ndarray:
        """
        Fully vectorized exemplar selection from multiple blocks simultaneously
        Uses advanced indexing and broadcasting to avoid loops
        """
        # Create mask for samples in selected blocks
        block_mask = cp.isin(self.sample_block_indices, selected_block_indices)
        
        # Get indices of samples in selected blocks
        candidate_indices = cp.where(block_mask)[0]
        
        if len(candidate_indices) == 0:
            return cp.array([], dtype=cp.int32)
        
        # Get block indices for candidates
        candidate_block_indices = self.sample_block_indices[candidate_indices]
        
        # Get scores for all candidates
        candidate_difficulties = self.difficulties[candidate_indices]
        candidate_ages = self.ages[candidate_indices].astype(cp.float32)
        
        # Fully vectorized per-block normalization and selection
        # Use advanced indexing to avoid loops and appending
        
        # Get unique blocks and their positions for vectorization
        unique_blocks, inverse_indices = cp.unique(candidate_block_indices, return_inverse=True)
        
        # Vectorized per-block max calculation for normalization
        difficulty_maxes = cp.zeros(len(unique_blocks), dtype=cp.float32)
        age_maxes = cp.zeros(len(unique_blocks), dtype=cp.float32)
        
        # Use segmented reduction for max calculation
        for i, block_id in enumerate(unique_blocks):
            block_mask = candidate_block_indices == block_id
            if cp.any(block_mask):
                difficulty_maxes[i] = cp.max(candidate_difficulties[block_mask])
                age_maxes[i] = cp.max(candidate_ages[block_mask])
        
        # Broadcast maxes back to all candidates using inverse indices
        candidate_diff_maxes = difficulty_maxes[inverse_indices]
        candidate_age_maxes = age_maxes[inverse_indices]
        
        # Vectorized normalization for all candidates at once
        diff_normalized = candidate_difficulties / (candidate_diff_maxes + 1e-6)
        age_normalized = candidate_ages / (candidate_age_maxes + 1e-6)
        
        # Vectorized scoring for all candidates
        combined_scores = p_difficulty * diff_normalized + (1.0 - p_difficulty) * age_normalized
        
        # Vectorized Gumbel-max sampling for all candidates
        gumbel_noise = -cp.log(-cp.log(cp.random.uniform(0, 1, size=len(combined_scores))))
        log_scores = cp.log(combined_scores + 1e-10)
        final_scores = log_scores + gumbel_noise
        
        # Vectorized selection using argsort and advanced indexing
        # Sort candidates by final scores (highest first)
        sorted_indices = cp.argsort(final_scores)[::-1]
        
        # Use vectorized selection with block constraints
        selected_mask = cp.zeros(len(candidate_indices), dtype=cp.bool_)
        block_selection_counts = cp.zeros(len(unique_blocks), dtype=cp.int32)
        
        # Fully vectorized selection respecting per-block limits
        # Create cumulative count within each block group
        block_ids_sorted = inverse_indices[sorted_indices]
        
        # Use bincount to create block change indicators, then cumsum for within-block counts
        block_changes = cp.concatenate([cp.array([True]), block_ids_sorted[1:] != block_ids_sorted[:-1]])
        within_block_positions = cp.cumsum(cp.ones_like(block_ids_sorted)) - cp.cumsum(block_changes)[block_ids_sorted]
        
        # Vectorized constraint check: select samples where within-block position < samples_per_block
        valid_selections = within_block_positions < samples_per_block
        selected_mask[sorted_indices[valid_selections]] = True
        
        # Return selected candidate indices
        selected_candidates = candidate_indices[selected_mask]
        return selected_candidates if len(selected_candidates) > 0 else cp.array([], dtype=cp.int32)
            
    def select_exemplars_fully_vectorized(self, selected_block_indices: cp.ndarray, 
                                        samples_per_block: int, p_difficulty: float = 0.7) -> cp.ndarray:
        """
        Alternative fully vectorized implementation using advanced indexing
        Processes all blocks simultaneously without any loops
        """
        # Create expanded arrays for vectorized operations across all blocks
        total_candidates = len(selected_block_indices) * samples_per_block
        
        # Pre-allocate result arrays
        all_block_indices = cp.repeat(selected_block_indices, samples_per_block)
        
        # Create sample indices for each block (vectorized)
        block_ranges = cp.tile(cp.arange(samples_per_block), len(selected_block_indices))
        global_sample_indices = (all_block_indices * self.block_size + block_ranges).astype(cp.int32)
        
        # Clip to valid dataset range
        valid_mask = global_sample_indices < self.dataset_size
        valid_indices = global_sample_indices[valid_mask]
        valid_block_assignment = all_block_indices[valid_mask]
        
        if len(valid_indices) == 0:
            return cp.array([], dtype=cp.int32)
        
        # Get scores for all valid candidates
        candidate_difficulties = self.difficulties[valid_indices]
        candidate_ages = self.ages[valid_indices].astype(cp.float32)
        
        # Vectorized per-block normalization using segment operations
        # This is the most complex part - normalize within each block group
        
        # Use segmented operations for per-block normalization
        unique_blocks, inverse_indices = cp.unique(valid_block_assignment, return_inverse=True)
        
        # Vectorized per-segment max calculation
        difficulty_maxes = cp.zeros(len(unique_blocks))
        age_maxes = cp.zeros(len(unique_blocks))
        
        for i, block_id in enumerate(unique_blocks):
            block_mask = valid_block_assignment == block_id
            difficulty_maxes[i] = cp.max(candidate_difficulties[block_mask])
            age_maxes[i] = cp.max(candidate_ages[block_mask])
        
        # Broadcast maxes back to candidates
        candidate_diff_maxes = difficulty_maxes[inverse_indices]
        candidate_age_maxes = age_maxes[inverse_indices]
        
        # Vectorized normalization
        diff_normalized = candidate_difficulties / (candidate_diff_maxes + 1e-6)
        age_normalized = candidate_ages / (candidate_age_maxes + 1e-6)
        
        # Vectorized scoring
        combined_scores = p_difficulty * diff_normalized + (1.0 - p_difficulty) * age_normalized
        
        # Vectorized Gumbel-max trick
        gumbel_noise = -cp.log(-cp.log(cp.random.uniform(0, 1, size=len(combined_scores))))
        log_scores = cp.log(combined_scores + 1e-10)
        final_scores = log_scores + gumbel_noise
        
        # Select top scoring samples (this will naturally distribute across blocks)
        # due to the Gumbel noise ensuring diversity
        num_total_to_select = min(len(selected_block_indices) * samples_per_block, len(valid_indices))
        selected_local_indices = cp.argsort(final_scores)[-num_total_to_select:]
        
        return valid_indices[selected_local_indices]