"""
Parameter configuration system for Optuna benchmarking.

This module defines all hyperparameters that can be optimized by Optuna,
including their types, ranges, and whether they are forced parameters
for specific benchmark configurations.
"""

from typing import Dict, Any, List, Union

# Parameter configurations extracted from existing FloatSOM benchmarks
PARAMETER_CONFIGS = {
    # Core SOM Parameters
    'initial_radius': {
        'type': 'float',
        'range': (0.5, 10.0),
        'forced': False,
        'description': 'Initial neighborhood radius (None = auto-calculated)'
    },
    
    # Decay Parameters
    'radius_decay_type': {
        'type': 'categorical',
        'choices': ['exponential', 'linear', 'asymptotic'],
        'forced': False,
        'description': 'Radius decay type'
    },

    'learning_rate': {
        'type': 'float',
        'range': (0.01, 1.0),
        'forced': True,
        'description': 'Initial learning rate',
        'applicable_processing_methods': ['colors']
    },

    'lr_decay_type': {
        'type': 'categorical',
        'choices': ['exponential', 'linear', 'sigmoid', 'gaussian', 'asymptotic', 'fixed'],
        'forced': True,
        'description': 'Learning rate decay type',
        'applicable_processing_methods': ['colors']
    },

    'lr_decay_factor': {
        'type': 'float',
        'range': (0.05, 1.0),
        'forced': True,
        'description': 'Learning rate decay factor',
        'applicable_processing_methods': ['colors']
    },
    
    # Sampling Configuration
    'sampling_method': {
        'type': 'categorical',
        'choices': ['full', 'random', 'hdsssom'],
        'forced': True,  # This will be set as forced parameter combination
        'description': 'Sample selection method'
    },

    'target_proportion': {
        'type': 'float',
        'range': (0.01, 1.0),
        'forced': True,
        'description': 'Target subsampling proportion for random sampling'
    },
    
    # Processing Configuration - method is forced
    'processing_method': {
        'type': 'categorical',
        'choices': ['batch', 'colors'],
        'forced': True,  # This will be set as forced parameter combination
        'description': 'Processing method (batch or colors)'
    },
    
    'batch_mode': {
        'type': 'categorical',
        'choices': ['full_batch', 'minibatch'],
        'forced': True,
        'description': 'Batch processing mode (forced per scenario)'
    },


    'chunk_size': {
        'type': 'int',
        'range': (500, 500000),
        'forced': True,
        'description': 'Chunk size for batch processing (fixed, not optimized)'
    },
    # Momentum Parameters  
    'use_momentum': {
        'type': 'categorical',
        'choices': [True, False],
        'forced': False,
        'description': 'Enable momentum in weight updates'
    },
    
    'momentum_init': {
        'type': 'float',
        'range': (0.1, 0.9),
        'forced': False,
        'description': 'Initial momentum coefficient'
    },
    
    # Normalization Parameters
    'normalization': {
        'type': 'categorical',
        'choices': ['xpysom'],
        'forced': True,
        'description': 'Normalization method for weight updates (fixed to xpysom)'
    },
    
    # Topology Configuration - type is forced
    'topology_type': {
        'type': 'categorical',
        'choices': ['hexagonal', 'grid', 'mst', 'rng'],
        'forced': True,  # This will be set as forced parameter combination
        'description': 'Type of topology'
    },
    
    'topology_variant': {
        'type': 'categorical',
        'choices': ['planar'],
        'forced': True,
        'description': 'Topology variant (fixed to planar)',
        'applicable_topologies': ['grid', 'hexagonal']
    },
    
    # Initialization Parameters
    'initialization_method': {
        'type': 'categorical',
        'choices': ['random', 'pca', 'pca_sampling', 'pca_sampling_snake', 'pca_density'],
        'forced': False,
        'description': 'Method to initialize SOM weights'
    }

}


def get_forced_parameters() -> List[str]:
    """Get list of parameter names that are forced (not optimized by Optuna)."""
    return [name for name, config in PARAMETER_CONFIGS.items() if config['forced']]


def get_optimizable_parameters() -> List[str]:
    """Get list of parameter names that can be optimized by Optuna."""
    return [name for name, config in PARAMETER_CONFIGS.items() if not config['forced']]


def get_parameter_config(param_name: str) -> Dict[str, Any]:
    """Get configuration for a specific parameter."""
    if param_name not in PARAMETER_CONFIGS:
        raise ValueError(f"Unknown parameter: {param_name}")
    return PARAMETER_CONFIGS[param_name].copy()


def validate_forced_params(forced_params: Dict[str, Any]) -> None:
    """Validate that forced parameters contain valid values."""
    for param_name, value in forced_params.items():
        if param_name not in PARAMETER_CONFIGS:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        config = PARAMETER_CONFIGS[param_name]
        
        if config['type'] == 'categorical':
            if value not in config['choices']:
                raise ValueError(f"Invalid value {value} for {param_name}. Valid choices: {config['choices']}")
        elif config['type'] == 'float':
            min_val, max_val = config['range']
            if not (min_val <= value <= max_val):
                raise ValueError(f"Value {value} for {param_name} not in range [{min_val}, {max_val}]")
        elif config['type'] == 'int':
            min_val, max_val = config['range']
            if not (min_val <= value <= max_val):
                raise ValueError(f"Value {value} for {param_name} not in range [{min_val}, {max_val}]")


def get_conditional_parameters() -> Dict[str, List[str]]:
    """
    Get parameters that are only relevant when certain other parameters have specific values.
    
    Returns:
        Dictionary mapping condition to list of dependent parameters
    """
    return {}


def get_categorical_parameters() -> List[str]:
    """Get list of all categorical parameter names."""
    return [name for name, config in PARAMETER_CONFIGS.items() if config['type'] == 'categorical']


def get_discrete_parameters() -> List[str]:
    """Get list of all discrete/categorical parameter names (including int choices)."""
    return [name for name, config in PARAMETER_CONFIGS.items() if config['type'] in ['categorical', 'int']]


def get_algorithm_variants() -> List[str]:
    """Get the algorithm variants to be benchmarked."""
    return ['batch', 'colors']


def get_default_forced_combinations(topology_types: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get default forced parameter combinations for comprehensive benchmarking.
    
    This creates all combinations of the three main categorical parameters:
    - sampling_method: ['full', 'random']
    - processing_method: ['batch', 'colors']
    - batch_mode: ['full_batch', 'minibatch']
    - topology_type: ['hexagonal'] by default, or specified list
    
    Args:
        topology_types: List of topology types to use. Defaults to ['hexagonal']
    """
    forced_combinations = []
    
    sampling_methods = ['full', 'random', 'hdsssom']
    processing_methods = ['batch', 'colors']
    batch_modes = ['full_batch', 'minibatch']
    if topology_types is None:
        topology_types = ['hexagonal']
    
    for sampling in sampling_methods:
        for processing in processing_methods:
            for topology in topology_types:
                if processing == 'batch':
                    for batch_mode in batch_modes:
                        combination = {
                            'sampling_method': sampling,
                            'processing_method': processing,
                            'batch_mode': batch_mode,
                            'topology_type': topology
                        }
                        forced_combinations.append(combination)
                else:
                    combination = {
                        'sampling_method': sampling,
                        'processing_method': processing,
                        'batch_mode': 'full_batch',
                        'topology_type': topology
                    }
                    forced_combinations.append(combination)
    
    return forced_combinations
