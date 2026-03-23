"""
FloatSOM - Flexible Self-Organizing Map with configurable sampling and processing strategies
"""

import cupy as cp
import numpy as np
import time
import logging
import warnings
from typing import Optional, Union, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

from ..sampling.base_selector import SampleSelector, SelectionResult
from ..processing.base_processor import ProcessingMethod
from ..topology.som_topology import SOMTopology
from ..floatsom_params import (
    FloatSOMParams,
    precompute_all_momentum_coefficients, 
    calculate_adaptive_momentum_coefficient
)
from ..data.sources.base import DataSource
from ..data.factory import DataSourceFactory
from ..processing.utils import find_bmus
from .inference import InferenceMixin


class FloatSOM(InferenceMixin):
    """
    Main FloatSOM class that combines sample selection and processing strategies
    
    Architecture (SOLID Compliant):
    - SampleSelector: Chooses which samples to process each iteration
    - ProcessingMethod: Determines how samples update SOM weights  
    - SOMTopology: Manages topology structure and neighborhood relationships
    - DataSource: Abstraction for data access (array or file-based)
    
    Follows:
    - Single Responsibility: Each component has one clear responsibility
    - Open/Closed: New data sources can be added without modification
    - Dependency Inversion: Depends on abstractions, not concrete implementations
    """
    
    def __init__(self, 
                 selector: SampleSelector, 
                 processor: ProcessingMethod, 
                 topology: SOMTopology, 
                 params: FloatSOMParams,
                 data_source_factory: Optional[DataSourceFactory] = None):
        """
        Initialize FloatSOM with pluggable components
        
        Args:
            selector: SampleSelector implementation (e.g., random, HDSSSOM)
            processor: ProcessingMethod implementation (e.g., colors, batch)
            topology: SOMTopology implementation (e.g., grid, hexagonal, MST)
            params: FloatSOMParams configuration
            data_source_factory: Optional factory for creating data sources (DI)
        """
        self.selector = selector
        self.processor = processor
        self.topology = topology
        self.params = params
        
        # Dependency injection for factory (default if not provided)
        self.data_source_factory = data_source_factory or DataSourceFactory()
        
        # Training state
        self.weights = None
        self.is_trained = False
        self.current_iteration = 0
        self.training_history = []
        
        # Momentum support
        self.delta_weights = None
        self.use_momentum = params.processing_config.enable_momentum
        self.use_adaptive_momentum = params.processing_config.enable_adaptive_momentum
        self.previous_weight_change = 0.0
        
        # Performance tracking
        self.verbose = params.verbose
        self.use_gpu = True

        # FloatSOM uses CuPy exclusively for execution.
        self.xp = cp
        
        # Grid reformation tracking (post-training)
        self.adjacency_list = None  # Current adjacency list (grid after reformation)
        self.original_adjacency_list = None  # Preserve original topology adjacency
        self.grid_node_mapping = None  # Node-to-grid position mapping
        self.reform_grid = params.reform_grid
        self.reform_grid_type = params.reform_grid_type
        
        if self.verbose:
            logger.info(f"FloatSOM initialized:")
            logger.info(f"  - Selector: {type(selector).__name__}")
            logger.info(f"  - Processor: {type(processor).__name__}")  
            logger.info(f"  - Topology: {topology.name}")
            logger.info("  - Device: GPU (CuPy)")
            if self.use_momentum:
                momentum_type = "adaptive" if self.use_adaptive_momentum else "scheduled"
                logger.info(f"  - Momentum: enabled ({momentum_type})")
    
    def train(self, data_input: Union[np.ndarray, cp.ndarray, str]) -> Dict[str, Any]:
        """
        Main training loop using pluggable components with momentum support
        SOLID compliant: Uses DataSource abstraction instead of type checking
        
        Args:
            data_input: Training data (array or file path)
            
        Returns:
            training_stats: Dictionary with training statistics
        """
        start_time = time.time() if self.verbose else None

        perf_total_start = time.perf_counter()

        perf_setup_start = time.perf_counter()
        data_source, schedules = self._setup_training(data_input)
        perf_setup_end = time.perf_counter()

        # Initialize training statistics
        training_stats = self._initialize_training_stats()

        perf_iter_start = time.perf_counter()
        try:
            training_stats = self._execute_training_loop(data_source, schedules, training_stats)
            perf_iter_end = time.perf_counter()
        except KeyboardInterrupt:
            perf_iter_end = time.perf_counter()
            if self.verbose:
                logger.warning(f"Training interrupted at iteration {self.current_iteration}")
            perf_finalize_start = time.perf_counter()
            final_stats = self._finalize_training(training_stats, start_time)
            perf_finalize_end = time.perf_counter()
            self._attach_training_timing(
                final_stats,
                perf_total_start=perf_total_start,
                perf_setup_start=perf_setup_start,
                perf_setup_end=perf_setup_end,
                perf_iter_start=perf_iter_start,
                perf_iter_end=perf_iter_end,
                perf_finalize_start=perf_finalize_start,
                perf_finalize_end=perf_finalize_end,
            )
            return final_stats
        except Exception:
            perf_iter_end = time.perf_counter()
            perf_finalize_start = time.perf_counter()
            final_stats = training_stats
            finalize_error = None
            try:
                # Preserve the original training exception by avoiding a best-effort
                # final weight fetch when iteration processing has already failed.
                final_stats = self._finalize_training(
                    training_stats,
                    start_time,
                    fetch_final_weights=False,
                )
            except Exception as exc:
                finalize_error = exc
            perf_finalize_end = time.perf_counter()
            self._attach_training_timing(
                final_stats,
                perf_total_start=perf_total_start,
                perf_setup_start=perf_setup_start,
                perf_setup_end=perf_setup_end,
                perf_iter_start=perf_iter_start,
                perf_iter_end=perf_iter_end,
                perf_finalize_start=perf_finalize_start,
                perf_finalize_end=perf_finalize_end,
            )
            if finalize_error is not None:
                logger.warning(
                    "Best-effort cleanup after training failure raised an additional exception: %s",
                    finalize_error,
                )
            raise

        perf_finalize_start = time.perf_counter()
        final_stats = self._finalize_training(training_stats, start_time)
        perf_finalize_end = time.perf_counter()

        self._attach_training_timing(
            final_stats,
            perf_total_start=perf_total_start,
            perf_setup_start=perf_setup_start,
            perf_setup_end=perf_setup_end,
            perf_iter_start=perf_iter_start,
            perf_iter_end=perf_iter_end,
            perf_finalize_start=perf_finalize_start,
            perf_finalize_end=perf_finalize_end,
        )

        return final_stats

    def _attach_training_timing(
        self,
        training_stats: Dict[str, Any],
        *,
        perf_total_start: float,
        perf_setup_start: float,
        perf_setup_end: float,
        perf_iter_start: float,
        perf_iter_end: float,
        perf_finalize_start: float,
        perf_finalize_end: float,
    ) -> None:
        """Attach timing diagnostics to the training statistics dict."""
        if not isinstance(training_stats, dict):
            return

        training_stats["time_total_s"] = float(perf_finalize_end - perf_total_start)
        training_stats["time_setup_s"] = float(perf_setup_end - perf_setup_start)
        training_stats["time_iteration_s"] = float(perf_iter_end - perf_iter_start)
        training_stats["time_finalize_s"] = float(perf_finalize_end - perf_finalize_start)

        # Attach Ray data-staging metrics when available.
        worker_manager = getattr(self.processor, "worker_manager", None)
        staging_stats = getattr(worker_manager, "last_staging_stats", None)
        if isinstance(staging_stats, dict):
            training_stats.update(staging_stats)

        # Attach Ray per-iteration timing breakdown (sampled) when available.
        iteration_timing = getattr(worker_manager, "iteration_timing_history", None)
        if isinstance(iteration_timing, list) and iteration_timing:
            training_stats["ray_iteration_timing"] = iteration_timing
        else:
            last_iter_timing = getattr(worker_manager, "last_iteration_timing_stats", None)
            if isinstance(last_iter_timing, dict):
                training_stats["ray_iteration_timing"] = [last_iter_timing]
    
    def _setup_training(self, data_input: Union[np.ndarray, cp.ndarray, str]) -> Tuple[Any, Dict]:
        """Setup training components and compute schedules"""
        if self.verbose:
            logger.info("Starting FloatSOM training...")
        
        # Create data source using factory
        ray_config = self.params.processing_config.ray_config if hasattr(self.params.processing_config, 'ray_config') else None
        if isinstance(data_input, str):
            data_source = self.data_source_factory.create(data_input, ray_config)
        elif isinstance(data_input, np.ndarray) or isinstance(data_input, cp.ndarray): 
            data_source = self.data_source_factory.create(data_input, ray_config)
        elif isinstance(data_input, DataSource):
            data_source = data_input
        else:
            raise ValueError(f"Unsupported data input type: {type(data_input)}")
        if self.verbose:
            logger.info(f"Data source: {type(data_source).__name__}")
            logger.info(f"Data shape: {data_source.get_shape()}")
            if data_source.supports_streaming():
                logger.info(f"Streaming mode enabled")
        
        # Initialize training with data source abstraction
        self._initialize_with_data_source(data_source)
        
        # Store reference for training loop
        self.data_source = data_source
        
        # Get training parameters
        total_iterations = self.params.total_iterations
        initial_radius = self.params.initial_radius
        initial_learning_rate = self.params.initial_learning_rate
        
        # Precompute schedules
        schedules = {
            'radii': self._compute_radius_schedule(total_iterations, initial_radius),
            'learning_rates': self._compute_learning_rate_schedule(total_iterations, initial_learning_rate),
            'momentum': None,
            'total_iterations': total_iterations
        }
        
        # Precompute momentum coefficient schedule if needed
        if self.use_momentum and not self.use_adaptive_momentum:
            schedules['momentum'] = precompute_all_momentum_coefficients(
                total_iterations,
                self.params.processing_config.initial_momentum,
                self.params.processing_config.final_momentum,
                self.params.processing_config.momentum_decay_type
            )
        
        return data_source, schedules
    
    def _initialize_training_stats(self) -> Dict[str, Any]:
        """Initialize training statistics dictionary"""
        return {
            'iterations_completed': 0,
            'total_samples_processed': 0,
            'training_time': 0.0,
            'final_weights_norm': 0.0,
            'convergence_detected': False,
            'momentum_used': self.use_momentum
        }
    
    def _execute_training_loop(self, data_source, schedules: Dict, training_stats: Dict) -> Dict[str, Any]:
        """Execute the main training loop"""
        data_reference = data_source.get_reference()
        total_iterations = schedules['total_iterations']
        weight_change = 0.0
        
        for iteration in range(total_iterations):
            weight_change = self._process_training_iteration(
                iteration, data_source, data_reference, schedules, 
                weight_change, training_stats
            )
            
            if self._check_convergence(weight_change, iteration):
                training_stats['convergence_detected'] = True
                if self.verbose:
                    logger.info(f"Convergence detected at iteration {iteration}")
                break
        
        return training_stats
    
    def _process_training_iteration(self, iteration: int, data_source, data_reference,
                                   schedules: Dict, weight_change: float, 
                                   training_stats: Dict) -> float:
        """Process a single training iteration"""
        self.current_iteration = iteration
        
        # Get current parameters from schedules
        current_radius = schedules['radii'][iteration]
        current_lr = schedules['learning_rates'][iteration]
        
        # Calculate current momentum coefficient
        current_momentum = self._calculate_momentum_coefficient(
            iteration, schedules['total_iterations'], weight_change, schedules['momentum']
        )
        
        # Sample selection: only request SelectionResult when the processor consumes it.
        uses_selection_result = self._processor_supports_selection_result()
        if uses_selection_result:
            if self._processor_prefers_local_sampling(data_reference):
                selection = SelectionResult(samples=data_reference, indices=None)
            else:
                selection = self.selector.select(data_reference)
                if not isinstance(selection, SelectionResult):
                    raise TypeError(
                        f"{type(self.selector).__name__}.select() must return SelectionResult, "
                        f"got {type(selection).__name__}"
                    )
            selected_payload = selection
        else:
            selected_samples = self.selector.select_samples(data_reference)
            selection = SelectionResult(samples=selected_samples, indices=None)
            selected_payload = selected_samples

        # Determine sample count for logging
        sample_count = self._determine_sample_count(selection, data_source)
        
        if self.verbose and iteration % 1 == 0:
            logger.info(f"Iteration {iteration:4d}: radius={current_radius:.3f}, "
                        f"lr={current_lr:.6f}, samples={sample_count}")
        
        # Create iteration parameters without mutation
        iteration_params = self.params
        iteration_params.current_radius = current_radius
        iteration_params.current_learning_rate = current_lr
        iteration_params.current_iteration = iteration
        iteration_params.current_momentum = current_momentum
        iteration_params.delta_weights = self.delta_weights
        
        # Process samples
        result = self.processor.process_samples(
            selected_payload,
            self.weights, 
            self.topology, 
            iteration_params
        )
        
        # Handle result and update weights
        inplace_update = isinstance(result, dict) and result.get('update_type') == 'inplace'
        if inplace_update:
            # Weights are updated in-place on workers; use reported scalar for convergence
            new_weight_change = float(result.get('weight_change_norm', 0.0))
            # Maintain previous weight change for adaptive momentum bookkeeping
            self.previous_weight_change = weight_change
            # Do not touch self.weights here; they remain on workers during training.
            # When topology is dynamic, fetch weights sparingly and update explicitly.
            requires_updates_fn = getattr(self.topology, "requires_topology_updates", None)
            requires_updates = bool(requires_updates_fn()) if callable(requires_updates_fn) else False
            if requires_updates:
                should_update = True
                should_update_fn = getattr(self.topology, "should_update_topology", None)
                if callable(should_update_fn):
                    should_update = bool(should_update_fn(iteration))

                if should_update and hasattr(self.processor, 'get_current_weights'):
                    current_weights = self.processor.get_current_weights()
                    # Unpack if processor returns (weights, delta)
                    if isinstance(current_weights, tuple):
                        current_weights = current_weights[0]
                    self.weights = self._ensure_backend_array(current_weights)
                    # Update topology using fresh weights
                    self.topology.update_topology(self.weights, iteration)
                    # For distributed (in-place) training, only ship topology to workers
                    # when it actually changes.
                    update_topology_fn = getattr(self.processor, "update_topology", None)
                    if callable(update_topology_fn):
                        update_topology_fn(self.topology)
        else:
            new_weights = self._handle_processor_result(result)
            new_weight_change = self._update_weights(new_weights, weight_change)
            # Update topology if dynamic only when driver weights are updated locally
            self.topology.update_topology(self.weights, iteration)
        
        # Update statistics
        training_stats['iterations_completed'] = iteration + 1
        if inplace_update:
            training_stats['total_samples_processed'] += int(result.get('samples_processed', sample_count))
        else:
            training_stats['total_samples_processed'] += sample_count
        
        # Store training history if needed
        if self.params.store_history:
            self._record_training_history(iteration, current_radius, current_lr, 
                                         new_weight_change, sample_count)
        
        return new_weight_change
    
    def _ensure_backend_array(self, array):
        """Convert arrays to CuPy for GPU execution."""
        if array is None:
            return None
        if isinstance(array, cp.ndarray):
            return array
        return cp.asarray(array)

    def _handle_processor_result(self, result):
        """Handle processor result and manage momentum"""
        if isinstance(result, tuple):
            new_weights, delta_weights = result
            new_weights = self._ensure_backend_array(new_weights)
            self.delta_weights = self._ensure_backend_array(delta_weights)
        else:
            new_weights = self._ensure_backend_array(result)
            # Initialize delta_weights if needed
            if self.use_momentum and self.delta_weights is None:
                self.delta_weights = self.xp.zeros_like(self.weights)
        return new_weights

    def _update_weights(self, new_weights, current_weight_change: float) -> float:
        """Update weights and track changes"""
        current_weights = self._ensure_backend_array(self.weights)
        new_weights = self._ensure_backend_array(new_weights)
        self.weights = current_weights
        new_weight_change = self._calculate_weight_change(current_weights, new_weights)
        self.previous_weight_change = current_weight_change
        self.weights = new_weights
        return new_weight_change
    
    def _record_training_history(self, iteration: int, radius: float, lr: float, 
                                weight_change: float, sample_count: int):
        """Record training history if enabled"""
        self.training_history.append({
            'iteration': iteration,
            'radius': radius,
            'learning_rate': lr,
            'weight_change': float(weight_change),
            'samples_processed': sample_count
        })
    
    def _finalize_training(
        self,
        training_stats: Dict,
        start_time: Optional[float],
        *,
        fetch_final_weights: bool = True,
    ) -> Dict[str, Any]:
        """Finalize training and cleanup"""
        self.is_trained = True
        
        if start_time and self.verbose:
            training_stats['training_time'] = time.time() - start_time
        
        # Get final weights from processor unless this is a failure path cleanup.
        self._finalize_processor(fetch_final_weights=fetch_final_weights)

        # Give dynamic topologies a chance to finalize their structure.
        finalize_topology_fn = getattr(self.topology, "finalize_topology", None)
        if callable(finalize_topology_fn):
            if self.verbose and self.topology.requires_topology_updates():
                logger.info(
                    "Recalculating %s with final weights for optimal structure...",
                    self.topology.name,
                )
            finalize_topology_fn(self.weights)
            if self.verbose and self.topology.requires_topology_updates():
                logger.info("Final %s structure updated with optimal edges", self.topology.name)
        
        # Calculate final weights norm
        training_stats['final_weights_norm'] = float(self.xp.linalg.norm(self.weights))
                
        if self.reform_grid and self.topology.name not in ['grid', 'hexagonal']:
            if self.verbose:
                logger.info(f"Reforming {self.topology.name} topology to {self.reform_grid_type} grid...")
            self.reform_to_grid(grid_type=self.reform_grid_type)
            training_stats['topology_reformed'] = True
            training_stats['reformed_to'] = self.reform_grid_type
        
        # Cleanup components
        if hasattr(self, 'data_source'):
            self.data_source.cleanup()
        
        if self.verbose:
            logger.info(f"Training completed: {training_stats['iterations_completed']} iterations, "
                        f"{training_stats['training_time']:.2f}s")
        
        return training_stats
    
    def predict(self, data: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """
        Find Best Matching Units (BMUs) for new data
        
        Args:
            data: Input data to map (n_samples, n_features)
            
        Returns:
            bmu_indices: BMU indices for each input sample
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before prediction")
        data_gpu = cp.asarray(data, dtype=cp.float32) if not isinstance(data, cp.ndarray) else data.astype(cp.float32, copy=False)
        self.weights = cp.asarray(self.weights, dtype=cp.float32) if not isinstance(self.weights, cp.ndarray) else self.weights.astype(cp.float32, copy=False)
        node_chunk_size = min(900, self.weights.shape[0]) if self.weights.size else 1
        return find_bmus(
            batch=data_gpu,
            weights=self.weights,
            verbose=self.verbose,
            return_distances=False,
            node_chunk_size=node_chunk_size,
        )
    
    def get_weights(self, as_numpy: bool = True) -> Union[np.ndarray, cp.ndarray]:
        """
        Get current SOM weights

        Args:
            as_numpy: If True, convert CuPy arrays to NumPy (default: True for backward compatibility)

        Returns:
            weights: Current weight vectors (n_nodes, n_features)
        """
        if self.weights is None:
            raise ValueError("SOM weights not initialized. Call train() first.")

        if as_numpy and isinstance(self.weights, cp.ndarray):
            return self.weights.get()
        return self.weights
    
    def get_topology_info(self) -> Dict[str, Any]:
        """
        Get information about the current topology
        
        Returns:
            topology_info: Dictionary with topology details
        """
        return {
            'name': self.topology.name,
            'total_nodes': self.topology.total_nodes,
            'coordinates': self.topology.get_coordinates()
        }
    
    def map_vectors(self, data: Union[np.ndarray, cp.ndarray]) -> Tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """
        Map input vectors to BMUs and return both indices and distances
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            bmu_indices: BMU indices for each sample
            bmu_distances: Distances to BMUs for each sample
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before mapping")

        data = cp.asarray(data) if not isinstance(data, cp.ndarray) else data
        
        # Calculate distances
        data_expanded = data[:, None, :]
        weights_expanded = self.weights[None, :, :]
        distances = self.xp.sum((data_expanded - weights_expanded) ** 2, axis=2)
        
        # Find BMUs and their distances
        bmu_indices = self.xp.argmin(distances, axis=1)
        bmu_distances = self.xp.sqrt(distances[self.xp.arange(len(data)), bmu_indices])
        
        return bmu_indices, bmu_distances
    

    
    def _initialize_with_data_source(self, data_source: DataSource) -> None:
        """
        Initialize training components using DataSource abstraction
        SOLID compliant: Works with abstraction, not concrete types
        
        Args:
            data_source: DataSource instance providing data access
        """
        if self.verbose:
            logger.info(f"Initializing training components with {type(data_source).__name__}")
        
        # Get dataset shape from data source
        dataset_shape = data_source.get_shape()
        n_samples, n_features = dataset_shape
                
        # Initialize selector with data reference
        data_reference = data_source.get_reference()
        self._validate_whole_chunk_random_runtime(data_reference)
        if hasattr(self.selector, 'initialize_with_data_source'):
            self.selector.initialize_with_data_source(data_source)
        else:
            self.selector.initialize(data_reference)
        
        # Initialize weights using data source
        if 'pca' in self.topology.initialization_method:
            # Get initialization sample from data source
            init_sample = data_source.get_initialization_sample()
            
            if self.verbose:
                logger.info(f"PCA initialization with {len(init_sample):,} samples")
            
            # Initialize weights with PCA
            self.weights = self._ensure_backend_array(
                self.topology.initialize_weights(init_sample)
            )
            
            # Clean up initialization sample
            del init_sample
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # For random initialization, create minimal dummy data
            dummy_data = cp.zeros((2, n_features), dtype=cp.float32)
            
            self.weights = self._ensure_backend_array(
                self.topology.initialize_weights(dummy_data)
            )
            del dummy_data
        
        # Initialize momentum if enabled
        if self.use_momentum:
            self.delta_weights = self.xp.zeros_like(self.weights)
            if self.verbose:
                logger.info("Momentum initialized")
        
        # Precompute topology data if needed (must be done before processor initialization)
        if hasattr(self.params, 'total_iterations'):
            total_iterations = self.params.total_iterations
            initial_radius = self.params.initial_radius
            radii_list = self._compute_radius_schedule(total_iterations, initial_radius)
            self.topology.precompute_topology_data(self.params, radii_list)
        
        # Pass selector to processor if it supports it (for Ray processing)
        if hasattr(self.processor, 'set_selector'):
            self.processor.set_selector(self.selector)
        
        # Initialize processor with SOM configuration
        self.processor.initialize(self.weights, self.topology, self.params, data_source)
        self._warn_on_whole_chunk_random_chunk_size(n_samples)
        
        # Update selector's total_samples if needed (for Ray processing)
        if hasattr(self.selector, 'total_samples') and hasattr(self.processor, 'worker_manager'):
            if hasattr(self.processor.worker_manager, 'total_samples'):
                self.selector.total_samples = self.processor.worker_manager.total_samples
        
        # Set up HDSSSOM callback if selector supports it
        if self.params.sampling_config.method == "hdsssom":
            if hasattr(self.processor, 'set_selector_callback'):
                self.processor.set_selector_callback(self.selector.update_metadata)
        
        if self.verbose:
            logger.info(f"Initialized {self.topology.total_nodes} nodes with {n_features} features")
            if data_source.supports_streaming():
                logger.info(f"Data will be streamed from: {data_source.get_reference()}")
    
    
    def _compute_radius_schedule(self, total_iterations: int, initial_radius: float) -> Union[np.ndarray, cp.ndarray]:
        """Compute radius decay schedule using the unified calculation function"""
        from floatsom.floatsom_params import calculate_radius
        
        schedule = []
        for iteration in range(total_iterations):
            radius = calculate_radius(
                current_iteration=iteration,
                total_iterations=total_iterations,
                initial_radius=initial_radius,
                decay_type=self.params.radius_decay_type,
                min_radius=self.params.final_radius,
                radius_decay_factor=self.params.radius_decay_factor,
                radius_warmup_iters=self.params.radius_warmup_iters,
                use_minisom=False
            )
            schedule.append(radius)
        
        return self.xp.array(schedule)
    
    def _compute_learning_rate_schedule(self, total_iterations: int, initial_lr: float) -> Union[np.ndarray, cp.ndarray]:
        """Compute learning rate decay schedule using the unified calculation function"""
        from floatsom.floatsom_params import calculate_learning_rate

        schedule = []
        for iteration in range(total_iterations):
            lr = calculate_learning_rate(
                current_iteration=iteration,
                total_iterations=total_iterations,
                initial_learning_rate=initial_lr,
                decay_type=self.params.lr_decay_type,
                final_learning_rate=self.params.final_learning_rate,
                lr_decay_factor=self.params.lr_decay_factor,
                use_minisom=False
            )
            schedule.append(lr)

        return self.xp.array(schedule)
    
    def _calculate_weight_change(self, old_weights: Union[np.ndarray, cp.ndarray], 
                               new_weights: Union[np.ndarray, cp.ndarray]) -> float:
        """Calculate magnitude of weight change"""
        diff = self.xp.asarray(new_weights) - self.xp.asarray(old_weights)
        return float(self.xp.linalg.norm(diff))
    
    def _finalize_processor(self, *, fetch_final_weights: bool = True) -> None:
        """Unified processor finalization handling both Ray and non-Ray processors"""
        if hasattr(self.processor, 'worker_manager'):
            final_result = None
            try:
                if fetch_final_weights:
                    final_result = self.processor.worker_manager.finalize()
                elif self.verbose:
                    logger.warning(
                        "Skipping final Ray weight fetch because training terminated with an exception."
                    )
            finally:
                try:
                    self.processor.worker_manager.cleanup()
                except Exception as exc:
                    if self.verbose:
                        logger.warning(f"Failed to fully clean up Ray worker manager: {exc}")
            self._apply_final_result(final_result)
        else:
            final_result = self.processor.finalize()
            self._apply_final_result(final_result)
    
    def _apply_final_result(self, final_result) -> None:
        """Apply finalization result to weights and delta_weights"""
        if final_result is not None:
            if isinstance(final_result, tuple):
                weights, delta_weights = final_result
                self.weights = self._ensure_backend_array(weights)
                self.delta_weights = self._ensure_backend_array(delta_weights)
            else:
                self.weights = self._ensure_backend_array(final_result)
    
    def _calculate_momentum_coefficient(self, iteration: int, total_iterations: int, 
                                       weight_change: float, momentum_schedule: Optional[List]) -> float:
        """Calculate momentum coefficient for current iteration"""
        if not self.use_momentum:
            return 0.0
        
        if self.use_adaptive_momentum:
            return calculate_adaptive_momentum_coefficient(
                iteration, total_iterations, weight_change,
                self.previous_weight_change, 
                self.params.processing_config.initial_momentum
            )
        else:
            if momentum_schedule is None:
                return 0.0
            # momentum_schedule may be a list or NumPy array; both support indexing
            return float(momentum_schedule[iteration])
    
    def _processor_supports_selection_result(self) -> bool:
        """
        Whether the active processor accepts a SelectionResult payload directly.
        """
        checker = getattr(self.processor, "supports_selection_result", None)
        if callable(checker):
            return bool(checker())
        return False

    def _processor_prefers_local_sampling(self, data_reference) -> bool:
        """
        Whether the active processor wants to own stochastic sampling locally.
        """
        checker = getattr(self.processor, "prefers_processor_local_sampling", None)
        if callable(checker):
            return bool(checker(data_reference, self.params))
        return False

    def _whole_chunk_random_requested(self) -> bool:
        sampling_cfg = getattr(self.params, "sampling_config", None)
        return bool(getattr(sampling_cfg, "whole_chunk_random", False))

    def _validate_whole_chunk_random_runtime(self, data_reference) -> None:
        """
        Reject whole-chunk random runs that are not file-backed.
        """
        if not self._whole_chunk_random_requested():
            return
        if isinstance(data_reference, str):
            return
        raise ValueError(
            "whole_chunk_random requires file-backed random batch training so the "
            "processor can own chunk loading locally."
        )

    def _warn_on_whole_chunk_random_chunk_size(self, total_samples: int) -> None:
        """
        Warn when whole-chunk random operates at very fine chunk granularity.
        """
        if not self._whole_chunk_random_requested():
            return
        if total_samples is None or int(total_samples) <= 0:
            return

        worker_manager = getattr(self.processor, "worker_manager", None)
        if worker_manager is None:
            return

        dataset_samples = int(getattr(worker_manager, "total_samples", total_samples) or total_samples)
        if dataset_samples <= 0:
            return

        loader_chunk_size = int(getattr(worker_manager, "loader_chunk_size", 0) or 0)
        if loader_chunk_size <= 0:
            return

        chunk_fraction = loader_chunk_size / float(dataset_samples)
        if chunk_fraction >= 0.01:
            return

        warnings.warn(
            "whole_chunk_random is enabled and one chunk represents less than 1% "
            "of the dataset; random chunk selection may lead to unstable training.",
            UserWarning,
        )

    def _determine_sample_count(self, selection: SelectionResult, data_source) -> int:
        """Determine sample count for logging purposes"""
        selected_samples = selection.samples
        if isinstance(selected_samples, str):
            # File path - use configured samples per epoch or total samples
            return (self.params.sampling_config.samples_per_epoch or 
                    data_source.get_shape()[0])
        if selected_samples is None and selection.indices is not None:
            return int(len(selection.indices))
        if selected_samples is None:
            # Entire dataset selected
            return data_source.get_shape()[0]
        return len(selected_samples)
    
    def _check_convergence(self, weight_change: float, iteration: int) -> bool:
        """Check if training has converged"""
        convergence_threshold = self.params.convergence_threshold
        min_iterations = self.params.min_iterations
        
        return (iteration > min_iterations and weight_change < convergence_threshold)
    
    def reform_to_grid(self, grid_type: str = 'regular', chunk_size: int = 1000) -> None:
        """
        Reform current topology to grid structure post-training.
        Only works for non-grid topologies (e.g., MST).
        
        Args:
            grid_type: 'regular' or 'hexagonal'
            chunk_size: Chunk size for Hungarian algorithm computation
        """
        if self.topology.name in ['grid', 'hexagonal']:
            if self.verbose:
                logger.warning(f"Cannot reform {self.topology.name} topology to grid (already a grid)")
            return
        
        if not self.is_trained:
            raise ValueError("SOM must be trained before reforming to grid")
        
        # Import grid assigner
        from ..topology.grid_assignment import RegularGridAssigner, HexagonalGridAssigner
        
        # Select appropriate assigner
        if grid_type == 'regular':
            assigner = RegularGridAssigner(chunk_size=chunk_size)
        elif grid_type == 'hexagonal':
            assigner = HexagonalGridAssigner(chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        # Preserve original adjacency
        self.original_adjacency_list = self.topology.adjacency_list.copy() if self.topology.adjacency_list else None
        
        # Also preserve MST edges if available
        if hasattr(self.topology, 'mst_edges'):
            self.original_mst_edges = self.topology.mst_edges.copy() if self.topology.mst_edges else None
        
        # Perform grid assignment using final weights
        node_to_grid, grid_adjacency = assigner.assign_nodes_to_grid(
            self.weights, self.topology.total_nodes
        )
        
        # Update topology's adjacency to use grid
        self.topology.adjacency_list = grid_adjacency
        self.adjacency_list = grid_adjacency  # Also store at FloatSOM level
        self.grid_node_mapping = node_to_grid
        
        # Mark topology as reformed
        self.topology.is_reformed = True
        self.topology.reformed_type = grid_type
        
        # Add grid_size attribute for metrics compatibility
        grid_size = int(np.ceil(np.sqrt(self.topology.total_nodes)))
        self.topology.grid_size = grid_size
        
        if self.verbose:
            logger.info(f"Successfully reformed {self.topology.name} to {grid_type} grid")
            logger.info(f"Original adjacency preserved in self.original_adjacency_list")
            if hasattr(self, 'original_mst_edges'):
                logger.info(f"Original MST edges preserved in self.original_mst_edges")
    
    def get_adjacency_info(self) -> Dict[str, Any]:
        """
        Get current and original adjacency information.
        
        Returns:
            info: Dictionary with adjacency details and reformation status
        """
        info = {
            'current_adjacency': self.topology.adjacency_list,
            'original_adjacency': self.original_adjacency_list,
            'is_reformed': getattr(self.topology, 'is_reformed', False),
            'reformed_type': getattr(self.topology, 'reformed_type', None),
            'grid_mapping': self.grid_node_mapping,
            'topology_name': self.topology.name
        }
        
        # Include MST edges if available
        if hasattr(self, 'original_mst_edges'):
            info['original_mst_edges'] = self.original_mst_edges
        if hasattr(self.topology, 'mst_edges'):
            info['current_mst_edges'] = self.topology.mst_edges
        
        return info
