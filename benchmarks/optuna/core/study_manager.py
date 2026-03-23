"""
Study configuration and management for Optuna benchmarks.

This module provides utilities for creating, configuring, and managing
Optuna studies for FloatSOM hyperparameter optimization.
"""

import optuna
from typing import Dict, Any, List, Optional
import logging

class StudyManager:
    """
    Optuna study manager for GPU execution.
    
    Manages study creation and coordination for GPU workers
    using in-memory storage.
    """
    
    def __init__(self):
        """
        Initialize study manager.
        """
        self.logger = logging.getLogger(__name__)
    
    
    def create_study(
        self, 
        study_name: str, 
        directions: List[str], 
        sampler_config: Dict[str, Any], 
        load_if_exists: bool = True,
        multi_objective: Optional[bool] = None
    ) -> optuna.Study:
        """
        Create or load Optuna study for execution.
        
        Args:
            study_name: Name of the study to create/load
            directions: Optimization directions ('minimize' or 'maximize')
            sampler_config: Configuration for the sampler
            load_if_exists: Whether to load existing study or create new one
            multi_objective: Whether to use multi-objective optimization (NSGA-II).
                           If None, automatically determined from number of directions.
            
        Returns:
            Created or loaded Optuna study
        """
        # Auto-determine multi-objective from number of directions if not specified
        if multi_objective is None:
            multi_objective = len(directions) > 1
        
        # Create sampler based on optimization type
        if multi_objective:
            # Multi-objective optimization with NSGA-II
            sampler = optuna.samplers.TPESampler(**sampler_config)
            # No pruner for multi-objective optimization
            pruner = optuna.pruners.NopPruner()
        else:
            # Single objective optimization with TPE
            sampler = optuna.samplers.TPESampler(**sampler_config)
            
            # Create pruner from default config
            pruner_config = create_default_pruner_config()
            if pruner_config['type'] == 'median':
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=pruner_config['n_startup_trials'],
                    n_warmup_steps=pruner_config['n_warmup_steps'],
                    interval_steps=pruner_config['interval_steps']
                )
            else:
                pruner = optuna.pruners.NopPruner()
        
        optimization_type = "multi-objective" if multi_objective else "single-objective"
        self.logger.info(f"Creating {optimization_type} study: {study_name} ({len(directions)} objectives)")
        
        return optuna.create_study(
            study_name=study_name,
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists
        )
    
    def load_study(self, study_name: str) -> optuna.Study:
        """
        Load existing Optuna study for execution.
        
        Args:
            study_name: Name of the study to load
            
        Returns:
            Loaded Optuna study
        """
        return optuna.load_study(
            study_name=study_name
        )
    
    def list_studies(self) -> List[str]:
        """List all study names in storage."""
        try:
            studies = optuna.study.get_all_study_summaries()
            return [study.study_name for study in studies]
        except Exception:
            return []
    
    def close(self):
        """Clean up resources."""
        pass

def create_benchmark_study_name(
    algo_type: str, 
    dataset_name: str, 
    forced_params: Dict[str, Any], 
    seed: int
) -> str:
    """
    Create standardized study name for benchmark configuration.
    
    Args:
        algo_type: Algorithm variant ('colors', 'batch')
        dataset_name: Name of dataset
        forced_params: Dictionary of forced parameters
        seed: Random seed
        
    Returns:
        Standardized study name
    """
    # Parameter abbreviations for common SOM parameters
    param_abbrev = {
        'topology': 'topo',
        'grid_height': 'h',
        'grid_width': 'w', 
        'learning_rate': 'lr',
        'initial_learning_rate': 'lr0',
        'final_learning_rate': 'lrf',
        'sigma': 'sig',
        'initial_sigma': 'sig0',
        'final_sigma': 'sigf',
        'epochs': 'ep',
        'neighborhood_function': 'nbf',
        'topology_type': 'topo',
        'initialization': 'init',
        'distance_metric': 'dist',
        'decay_function': 'decay'
    }
    
    def format_value(value):
        """Format parameter values for compact representation."""
        if isinstance(value, float):
            # Remove trailing zeros and decimal point if integer
            formatted = f"{value:.6f}".rstrip('0').rstrip('.')
            return formatted
        elif isinstance(value, str):
            # Abbreviate common values
            abbrev_values = {
                'rectangular': 'rect',
                'hexagonal': 'hex', 
                'gaussian': 'gauss',
                'bubble': 'bub',
                'random': 'rand',
                'pca': 'pca',
                'linear': 'lin',
                'exponential': 'exp'
            }
            return abbrev_values.get(value.lower(), value)
        else:
            return str(value)
    
    # Process forced parameters into readable format
    param_parts = []
    sorted_params = sorted(forced_params.items())
    
    for key, value in sorted_params:
        # Use abbreviation if available, otherwise use original key
        abbrev_key = param_abbrev.get(key, key)
        formatted_value = format_value(value)
        
        # Handle grid dimensions specially
        if key in ['grid_height', 'grid_width']:
            # Skip individual height/width if both are present - will be handled as size
            if 'grid_height' in forced_params and 'grid_width' in forced_params:
                if key == 'grid_height':
                    h = format_value(forced_params['grid_height'])
                    w = format_value(forced_params['grid_width'])
                    param_parts.append(f"size-{h}x{w}")
                # Skip grid_width as it's handled with grid_height
                continue
            else:
                param_parts.append(f"{abbrev_key}-{formatted_value}")
        else:
            param_parts.append(f"{abbrev_key}-{formatted_value}")
    
    # Create the readable parameter string
    if param_parts:
        param_str = "_".join(param_parts)
        # Sanitize for filename safety
        param_str = param_str.replace('/', '-').replace('\\', '-').replace(':', '-')
        
        # If the parameter string is too long, fall back to hash
        if len(param_str) > 80:
            import hashlib
            sorted_params = sorted(forced_params.items())
            hash_input = "_".join(f"{k}={v}" for k, v in sorted_params)
            param_str = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    else:
        param_str = "default"
    
    return f"{algo_type}_{dataset_name}_{param_str}_seed{seed}"

def create_default_sampler_config(seed: int = 42) -> Dict[str, Any]:
    """
    Create default sampler configuration for benchmarks.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Sampler configuration dictionary
    """
    return {
        'multivariate': True,
        'n_startup_trials': 10,  # Number of random trials before TPE kicks in
        'n_ei_candidates': 24,   # Number of candidates for Expected Improvement
        'seed': seed,
        'prior_weight': 1.0,     # Weight of prior distribution
        'consider_prior': True,  # Whether to consider prior distribution
        'consider_magic_clip': True,  # Improve sampling near boundaries
        'consider_endpoints': False,  # Whether to consider endpoints
    }


def create_default_pruner_config() -> Dict[str, Any]:
    """
    Create default pruner configuration for benchmarks.
    
    Returns:
        Pruner configuration dictionary
    """
    return {
        'type': 'median',
        'n_startup_trials': 5,    # Number of trials before pruning starts
        'n_warmup_steps': 10,     # Number of steps before pruning evaluation
        'interval_steps': 1       # Pruning evaluation interval
    }
