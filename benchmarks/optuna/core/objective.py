"""
Core objective function factory for Optuna optimization.

This module provides the main objective function that integrates with FloatSOM
training and evaluation metrics for hyperparameter optimization.
"""

import time
import numpy as np
import cupy as cp
from typing import Dict, Any, Callable, Union, Optional, List, Iterable
import optuna

# FloatSOM imports
from floatsom.base.floatsom import FloatSOM
from floatsom.base.floatsom_factories import create_floatsom
from floatsom.floatsom_params import FloatSOMParams, SamplingConfig, ProcessingConfig, TopologyConfig

# Evaluation metrics from existing benchmarks
from floatsom.benchmarks.evaluation.metrics import (
    QuantizationError,
    TopographicError, 
    Trustworthiness, 
    NeighborhoodPreservation,
    DistortionMeasure,
    TopographicFunction
)

# sklearn datasets
from floatsom.benchmarks.evaluation.sklearn_datasets import generate_sklearn_dataset

# Parameter configuration
from ..config.parameters import PARAMETER_CONFIGS, get_conditional_parameters


# Metrics that reflect SOM topology only and therefore do not need separate
# train/holdout evaluations.
TOPOLOGY_ONLY_METRICS = {
    'topographic_error',
    'trustworthiness',
    'neighborhood_preservation',
    'distortion_measure',
    'topographic_function'
}


def _build_default_floatsom_for_context(input_dim: int, params: Dict[str, Any]) -> FloatSOMParams:
    """Build contextual FloatSOM defaults for the requested sampling/topology pair."""
    sampling_config = create_sampling_config(params)
    topology_config = create_topology_config(params)
    processing_config = ProcessingConfig(
        method=params['processing_method'],
        batch_mode=params.get('batch_mode', 'full_batch'),
        chunk_size=params.get('chunk_size', 500000),
        max_rounds=1,
        normalization='xpysom',
        use_gpu=True,
    )
    return FloatSOMParams(
        input_dim=int(input_dim),
        sampling_config=sampling_config,
        processing_config=processing_config,
        topology_config=topology_config,
        use_gpu=True,
    )


def sanitize_objective_value(value: float, objective_name: str) -> float:
    """
    Convert inf/NaN values to large finite penalties appropriate for the objective.
    
    Args:
        value: The objective value to sanitize
        objective_name: Name of the objective metric
        
    Returns:
        Sanitized finite value
    """
    # Define which objectives should be maximized (higher is better)
    maximize_objectives = {'trustworthiness', 'neighborhood_preservation'}
    
    # Check if value is invalid (NaN or Inf)
    if np.isnan(value) or np.isinf(value):
        # Use large penalty values that are finite
        # For minimization objectives: use large positive value
        # For maximization objectives: use large negative value
        if objective_name in maximize_objectives:
            return -1e10  # Large negative penalty for maximization
        else:
            return 1e10   # Large positive penalty for minimization
    
    return float(value)


class FloatSOMWrapper:
    """Wrapper to make FloatSOM compatible with metric interfaces."""
    
    def __init__(self, som: FloatSOM):
        self.floatsom = som
        self.weights = som.get_weights()
        
        # Handle different SOM types
        if hasattr(som, 'params'):
            # FloatSOM
            self.grid_size = som.params.topology_config.grid_size
        elif hasattr(som, 'topology'):
            # Has topology attribute
            self.grid_size = som.topology.grid_size if hasattr(som.topology, 'grid_size') else 10
        else:
            # Fallback
            self.grid_size = 10
    
    def get_weights(self):
        return self.weights
    
    def get_bmu(self, sample):
        """Get best matching unit for a single sample."""
        sample_batch = sample.reshape(1, -1)
        bmu_idx = self.floatsom.predict(sample_batch)[0]
        
        # Convert linear index to grid coordinates if needed
        if hasattr(self.floatsom.topology, 'grid_size'):
            grid_size = self.floatsom.topology.grid_size
            bmu_coords = (bmu_idx // grid_size, bmu_idx % grid_size)
            return bmu_coords
        else:
            return bmu_idx
    
    def get_bmu_distances(self, sample):
        """Get distances to all neurons for a single sample."""
        weights = self.weights
        if isinstance(sample, cp.ndarray):
            sample_np = cp.asnumpy(sample)
        else:
            sample_np = sample
            
        if isinstance(weights, cp.ndarray):
            weights_np = cp.asnumpy(weights)
        else:
            weights_np = weights
        
        # Compute distances to all neurons
        distances = np.linalg.norm(weights_np - sample_np, axis=-1)
        
        # Reshape to grid if applicable
        if hasattr(self.floatsom.topology, 'grid_size'):
            grid_size = self.floatsom.topology.grid_size
            distances = distances.reshape(grid_size, grid_size)
        
        return distances


def create_objective(
    algo_type: str, 
    dataset_name: str,
    dataset_config: Dict[str, Any],
    forced_params: Dict[str, Any], 
    metrics_config: Dict[str, Any],
    seed: int = 42
) -> Callable[[optuna.Trial], float]:
    """
    Factory function to create objective functions with specific configurations.
    
    Args:
        algo_type: Algorithm variant ('colors', 'batch')
        dataset_name: Name of dataset to use
        dataset_config: Dataset configuration parameters
        forced_params: Dict of parameters that are fixed for this run
        metrics_config: Dict specifying which metrics to optimize
        seed: Random seed for reproducibility
        
    Returns:
        Objective function for Optuna optimization
    """
        
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function that trains FloatSOM and returns metric to optimize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Primary objective value (lower is better for minimization)
        """
        # Generate dataset
        data, metadata = generate_sklearn_dataset(
            dataset_name=dataset_name,
            seed=seed,
            **dataset_config
        )
        
        # Perform 70/30 train/test split
        n_samples = data.shape[0]
        n_train = int(0.7 * n_samples)
        
        # Create random indices for shuffling (using consistent seed for reproducibility)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        
        # Split indices into train and test
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Create train and test datasets
        train_data = data[train_indices]
        test_data = data[test_indices]
        
        # Start with forced parameters
        params = forced_params.copy()
        
        # Add parameters suggested by Optuna
        conditional_params = get_conditional_parameters()
        added_params = []
        
        for param_name, config in PARAMETER_CONFIGS.items():
            # Skip if parameter is forced for this run
            if param_name in forced_params:
                continue
            
            # Skip if parameter is marked as forced in config
            if config.get('forced', False):
                continue

            applicable_topologies = config.get('applicable_topologies')
            if applicable_topologies:
                if params.get('topology_type') not in set(applicable_topologies):
                    continue

            applicable_methods = config.get('applicable_processing_methods')
            if applicable_methods:
                if params.get('processing_method') not in set(applicable_methods):
                    continue
                
            # Graph topologies do not support toroidal variants.
            if param_name == 'topology_variant' and params.get('topology_type') in {'mst', 'rng'}:
                continue
                
            # Check if this parameter has conditional dependencies
            should_add = True
            for condition, dependent_params in conditional_params.items():
                if param_name in dependent_params:
                    # Parse condition (e.g., 'use_momentum=True')
                    cond_param, cond_value = condition.split('=')
                    cond_value_normalized = cond_value.strip()
                    if cond_value_normalized.lower() == 'true':
                        cond_value_normalized = True
                    elif cond_value_normalized.lower() == 'false':
                        cond_value_normalized = False
                    # Only add if condition is met
                    if params.get(cond_param) != cond_value_normalized:
                        should_add = False
                        break
            
            if not should_add:
                continue
                
            # Add parameter based on type
            if config['type'] == 'float':
                min_val, max_val = config['range']
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                added_params.append(param_name)
            elif config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config['choices'])
                added_params.append(param_name)
            elif config['type'] == 'int':
                min_val, max_val = config['range']
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                added_params.append(param_name)
                
        # Train algorithm on training data only
        som, training_stats, train_time = train_floatsom_algorithm(
            train_data, params
        )
        
        # Calculate metrics on test data for validation (holdout)
        holdout_metrics = calculate_metrics(som, test_data, metrics_config)

        # Only evaluate training metrics for data-dependent metrics
        train_metric_names = metrics_config.get('train_metrics', metrics_config.get('metrics', []))
        if train_metric_names:
            train_metrics = calculate_metrics(
                som,
                train_data,
                metrics_config,
                metrics_subset=train_metric_names
            )
        else:
            train_metrics = {}

        # Persist the full metric dictionaries for downstream consumers
        trial.set_user_attr('metrics_holdout', {name: float(value) for name, value in holdout_metrics.items()})
        trial.set_user_attr('metrics_train', {name: float(value) for name, value in train_metrics.items()})

        base_objectives = metrics_config.get('base_objectives', metrics_config.get('objectives', ['quantization_error']))

        # Store holdout/train metrics with clear naming
        for name, value in holdout_metrics.items():
            trial.set_user_attr(name, float(value))
        for base_name in base_objectives:
            if base_name in holdout_metrics:
                trial.set_user_attr(f"{base_name}_holdout", float(holdout_metrics[base_name]))
            if base_name in train_metrics:
                trial.set_user_attr(f"{base_name}_train", float(train_metrics[base_name]))

        # Store training info
        trial.set_user_attr('train_time', train_time)
        trial.set_user_attr('iterations_completed', training_stats['iterations_completed'])
        trial.set_user_attr('dataset_name', dataset_name)
        trial.set_user_attr('algo_type', algo_type)
        trial.set_user_attr('n_train_samples', n_train)
        trial.set_user_attr('n_test_samples', len(test_indices))
        trial.set_user_attr('train_test_split', 0.7)
        
        # Store discrete parameters (both forced and optimized by Optuna)
        from ..config.parameters import get_discrete_parameters
        discrete_param_names = get_discrete_parameters()
        
        for param_name in discrete_param_names:
            if param_name in params:
                trial.set_user_attr(param_name, params[param_name])
        
        # Return multiple objectives for multi-objective optimization
        expanded_objectives = metrics_config.get('objectives', base_objectives)
        include_train = metrics_config.get('include_train_objectives', False)

        def _base_metric_name(metric_name: str) -> str:
            if metric_name.endswith('_holdout'):
                return metric_name[:-8]
            if metric_name.endswith('_train'):
                return metric_name[:-6]
            return metric_name

        sanitized_values = []
        for obj_name in expanded_objectives:
            base_name = _base_metric_name(obj_name)
            if obj_name.endswith('_train'):
                value = train_metrics.get(base_name, float('inf'))
            else:
                value = holdout_metrics.get(base_name, float('inf'))
            sanitized_values.append(sanitize_objective_value(value, base_name))

        if len(sanitized_values) == 1:
            return sanitized_values[0]
        return tuple(sanitized_values)
    
    return objective


def create_sampling_config(params: Dict[str, Any]) -> SamplingConfig:
    """Create sampling configuration from parameters."""
    config_params = {
        'method': params['sampling_method'],
    }
    if params.get('sampling_method') == 'random' and 'target_proportion' in params:
        config_params['target_proportion'] = float(params['target_proportion'])
    if 'seed' in params:
        config_params['random_seed'] = params['seed']
    
    return SamplingConfig(**config_params)


def create_processing_config(
    params: Dict[str, Any],
    default_processing_config: Optional[ProcessingConfig] = None,
) -> ProcessingConfig:
    """Create processing configuration from parameters."""
    normalization = 'xpysom'
    norm_alpha = params.get('norm_alpha')
    norm_clamp_factor = params.get('norm_clamp_factor')
    norm_percentile = params.get('norm_percentile')
    default_chunk_size = 500000
    default_enable_momentum = False
    default_initial_momentum = 0.5

    if default_processing_config is not None:
        if default_processing_config.chunk_size is not None:
            default_chunk_size = int(default_processing_config.chunk_size)
        if default_processing_config.enable_momentum is not None:
            default_enable_momentum = bool(default_processing_config.enable_momentum)
        if default_processing_config.initial_momentum is not None:
            default_initial_momentum = float(default_processing_config.initial_momentum)

    if normalization == 'hybrid' and norm_alpha is None:
        norm_alpha = 0.5
    if normalization == 'clamped_weighted' and norm_clamp_factor is None:
        norm_clamp_factor = 2.0
    if normalization == 'local' and norm_percentile is None:
        norm_percentile = 85.0

    return ProcessingConfig(
        method=params['processing_method'],
        batch_mode=params.get('batch_mode', 'full_batch'),
        chunk_size=params.get('chunk_size', default_chunk_size),
        max_rounds=1,
        enable_momentum=params.get('use_momentum', default_enable_momentum),
        initial_momentum=params.get('momentum_init', default_initial_momentum),
        normalization=normalization,
        norm_alpha=norm_alpha,
        norm_clamp_factor=norm_clamp_factor,
        norm_percentile=norm_percentile,
        use_gpu=True
    )


def create_topology_config(params: Dict[str, Any]) -> TopologyConfig:
    """Create topology configuration from parameters.""" 
    topology_variant = 'planar'
    
    return TopologyConfig(
        topology_type=params['topology_type'],
        topology_variant=topology_variant,
        grid_size=params.get('grid_size', 10),
        grid_dim=2
    )


def create_floatsom_params(data: cp.ndarray, params: Dict[str, Any]) -> FloatSOMParams:
    """Create FloatSOM parameters from input data and parameter dictionary."""
    input_dim = data.shape[1]

    default_params = _build_default_floatsom_for_context(input_dim=int(input_dim), params=params)
    sampling_config = create_sampling_config(params)
    topology_config = create_topology_config(params)
    processing_config = create_processing_config(
        params,
        default_processing_config=default_params.processing_config,
    )

    return FloatSOMParams(
        input_dim=input_dim,
        total_iterations=params.get('iterations', 50),
        initial_learning_rate=params.get('learning_rate', default_params.initial_learning_rate),
        initial_radius=params.get('initial_radius', default_params.initial_radius),
        lr_decay_type=params.get('lr_decay_type', default_params.lr_decay_type),
        radius_decay_type=params.get('radius_decay_type', default_params.radius_decay_type),
        lr_decay_factor=params.get('lr_decay_factor', default_params.lr_decay_factor),
        radius_decay_factor=params.get('radius_decay_factor', default_params.radius_decay_factor),
        sampling_config=sampling_config,
        processing_config=processing_config,
        topology_config=topology_config,
        convergence_threshold=params.get('convergence_threshold', default_params.convergence_threshold),
        min_iterations=params.get('min_iterations', default_params.min_iterations),
        verbose=False,
        use_gpu=True,
        seed=params.get('seed', default_params.seed if default_params.seed is not None else 42),
        store_history=False,
        initialization_method=params.get('initialization_method', default_params.initialization_method)
    )


def train_floatsom_algorithm(
    data: cp.ndarray, 
    params: Dict[str, Any], 
) -> tuple:
    """
    Train FloatSOM algorithm with given parameters.
    
    Args:
        data: Training data
        params: Complete parameter dictionary
        
    Returns:
        Tuple of (trained_som, training_stats, train_time)
    """
    # Use standard FloatSOM
    floatsom_params = create_floatsom_params(data, params)
    som = create_floatsom(floatsom_params)
    data_for_training = data
    
    # Train the SOM
    start_time = time.time()
    training_stats = som.train(data_for_training)
    train_time = time.time() - start_time
    
    return som, training_stats, train_time


def calculate_metrics(
    som: FloatSOM,
    data: Union[cp.ndarray, np.ndarray], 
    metrics_config: Dict[str, Any],
    metrics_subset: Optional[Iterable[str]] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for trained SOM.
    
    Args:
        som: Trained FloatSOM instance
        data: Training/test data
        metrics_config: Configuration for which metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    
    metrics = {}
    use_gpu = True
    
    # Initialize metrics based on configuration
    available_metrics = {
        'quantization_error': QuantizationError(use_optimized=use_gpu),
        'topographic_error': TopographicError(use_gpu=use_gpu),
        'trustworthiness': Trustworthiness(k=metrics_config.get('topology_k', 7)),
        'neighborhood_preservation': NeighborhoodPreservation(k=metrics_config.get('topology_k', 7)),
        'distortion_measure': DistortionMeasure(use_gpu=use_gpu, sigma=1.0),
        'topographic_function': TopographicFunction(use_gpu=use_gpu)
    }
    
    # Default metrics to compute if configuration is missing specific guidance
    default_metrics = ['quantization_error', 'distortion_measure']
    requested_metrics = list(metrics_subset) if metrics_subset is not None else metrics_config.get('metrics', default_metrics)
    
    # Create wrapper for metric compatibility
    som_wrapper = FloatSOMWrapper(som)
    
    # Compute each requested metric
    for metric_name in requested_metrics:
        if metric_name in available_metrics:
            try:
                metric_calculator = available_metrics[metric_name]
                metric_value = metric_calculator.compute(som_wrapper, data)
                # Sanitize the metric value to ensure it's finite
                metrics[metric_name] = sanitize_objective_value(float(metric_value), metric_name)
            except Exception as e:
                # If metric calculation fails, set a penalty value
                print(f"Warning: Metric '{metric_name}' failed with error: {str(e)}")
                metrics[metric_name] = sanitize_objective_value(float('inf'), metric_name)
        else:
            print(f"Warning: Unknown metric '{metric_name}', skipping")
    
    return metrics

def get_metrics_config(
    objectives: Optional[List[str]] = None,
    include_train_objectives: Optional[bool] = None,
    evaluation_split: str = 'both'
) -> Dict[str, Any]:
    """
    Get metrics configuration for specific objectives.


    Args:
        objectives: List of objective metrics to optimize. If None, defaults to ['quantization_error']
        include_train_objectives: Whether to include training metrics in the optimization objective.
            If None, this is inferred from `evaluation_split`.
        evaluation_split: Which dataset split(s) should contribute to the optimization objective.

    Returns:
        Metrics configuration dictionary
    """

    if objectives is None:
        objectives = ['quantization_error']

    base_objectives = list(objectives)

    if include_train_objectives is None:
        include_train_objectives = evaluation_split in ('both', 'train')

    include_holdout_objectives = evaluation_split in ('both', 'holdout')

    expanded_objectives: List[str] = []
    holdout_objectives: List[str] = []
    train_objectives: List[str] = []

    train_eligible = [obj for obj in base_objectives if obj not in TOPOLOGY_ONLY_METRICS]
    if evaluation_split == 'train' and not train_eligible:
        raise ValueError(
            "Train split requested but none of the objectives support training evaluation. "
            "Select at least one non-topology objective or use 'holdout'/'both'."
        )

    for obj in base_objectives:
        if include_holdout_objectives:
            suffix = f"{obj}_holdout"
            expanded_objectives.append(suffix)
            holdout_objectives.append(suffix)

        if include_train_objectives:
            if obj in TOPOLOGY_ONLY_METRICS:
                if evaluation_split == 'train':
                    raise ValueError(
                        f"Objective '{obj}' cannot be evaluated on the training split. "
                        "Choose a compatible objective or enable holdout evaluation."
                    )
                continue
            suffix = f"{obj}_train"
            expanded_objectives.append(suffix)
            train_objectives.append(suffix)

    if not expanded_objectives:
        raise ValueError(
            f"No objectives configured for evaluation_split='{evaluation_split}'. "
            "Check requested objectives and splits."
        )

    if not base_objectives:
        raise ValueError("At least one objective metric must be specified.")

    # Only request the metrics explicitly supplied by the caller, preserving order
    metrics_to_compute: List[str] = []
    for metric in base_objectives:
        if metric not in metrics_to_compute:
            metrics_to_compute.append(metric)

    train_metrics = [metric for metric in metrics_to_compute if metric not in TOPOLOGY_ONLY_METRICS]

    config = {
        'metrics': metrics_to_compute,
        'objectives': expanded_objectives,
        'base_objectives': base_objectives,
        'include_train_objectives': include_train_objectives,
        'include_holdout_objectives': include_holdout_objectives,
        'evaluation_split': evaluation_split,
        'holdout_objectives': holdout_objectives,
        'train_objectives': train_objectives,
        'topology_k': 7,
        'train_metrics': train_metrics,
        'topology_only_metrics': [metric for metric in metrics_to_compute if metric in TOPOLOGY_ONLY_METRICS]
    }


    if len(objectives) == 1:
        config['primary_metric'] = objectives[0]


    return config
