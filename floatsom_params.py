"""
FloatSOM Parameters - Master configuration for modular SOM architecture
Extends the original FlexibleSOMParams with sampling and processing method support
"""

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
import logging
import numpy as np
import cupy
from floatsom.processing.processing_params import (
    AsyncLoadingConfig,
    CleanupConfig,
    WorkerProfileConfig,
    DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS,
    validate_multi_buffering_buffers,
)
# Note: chunk_size must now be explicitly provided - no defaults

logger = logging.getLogger(__name__)

_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH = Path(__file__).with_name("floatsom_min1000_tuned_defaults.json")
_CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS = {"full", "random"}
_CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS = {"hexagonal", "mst", "rng"}
_VALID_DECAY_TYPES = {"exponential", "linear", "sigmoid", "gaussian", "asymptotic", "fixed"}
_VALID_INITIALIZATION_METHODS = {"random", "pca", "pca_sampling", "pca_sampling_snake", "pca_density"}


@lru_cache(maxsize=1)
def load_contextual_floatsom_defaults() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load and validate sampling/topology-specific FloatSOM defaults."""
    try:
        payload = json.loads(_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Contextual FloatSOM defaults JSON not found: {_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Contextual FloatSOM defaults JSON is not valid JSON: {_CONTEXTUAL_FLOATSOM_DEFAULTS_PATH}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Contextual FloatSOM defaults JSON must be an object of "
            "sampling_method->topology->params mappings."
        )

    normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_sampling_key, topology_map in payload.items():
        sampling_key = str(raw_sampling_key).strip().lower()
        if sampling_key not in _CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS:
            raise ValueError(
                f"Unsupported sampling key in contextual FloatSOM defaults: '{raw_sampling_key}'. "
                f"Supported: {sorted(_CONTEXTUAL_FLOATSOM_DEFAULT_SAMPLING_KEYS)}"
            )
        if not isinstance(topology_map, dict):
            raise ValueError(
                f"Sampling entry '{raw_sampling_key}' must map to an object of topology->params mappings."
            )

        normalized_topology_map: Dict[str, Dict[str, Any]] = {}
        for raw_topology_key, raw_param_map in topology_map.items():
            topology_key = str(raw_topology_key).strip().lower()
            if topology_key not in _CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS:
                raise ValueError(
                    f"Unsupported topology key in contextual FloatSOM defaults: '{raw_topology_key}'. "
                    f"Supported: {sorted(_CONTEXTUAL_FLOATSOM_DEFAULT_TOPOLOGY_KEYS)}"
                )
            if not isinstance(raw_param_map, dict):
                raise ValueError(
                    f"Topology entry '{raw_sampling_key}:{raw_topology_key}' must map to an object of param->value entries."
                )

            normalized_params: Dict[str, Any] = {}
            for raw_param_name, raw_value in raw_param_map.items():
                param_name = str(raw_param_name).strip()
                if param_name == "initial_radius":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:initial_radius' "
                            f"must be numeric, got {type(raw_value).__name__}."
                        )
                    normalized_params["initial_radius"] = float(raw_value)
                elif param_name == "radius_decay_type":
                    if not isinstance(raw_value, str) or raw_value not in _VALID_DECAY_TYPES:
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:radius_decay_type' "
                            f"must be one of {sorted(_VALID_DECAY_TYPES)}."
                        )
                    normalized_params["radius_decay_type"] = raw_value
                elif param_name == "initialization_method":
                    if not isinstance(raw_value, str) or raw_value not in _VALID_INITIALIZATION_METHODS:
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:initialization_method' "
                            f"must be one of {sorted(_VALID_INITIALIZATION_METHODS)}."
                        )
                    normalized_params["initialization_method"] = raw_value
                elif param_name == "use_momentum":
                    if not isinstance(raw_value, bool):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:use_momentum' must be boolean."
                        )
                    normalized_params["enable_momentum"] = raw_value
                elif param_name == "momentum_init":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(
                            f"Contextual default '{raw_sampling_key}:{raw_topology_key}:momentum_init' "
                            f"must be numeric, got {type(raw_value).__name__}."
                        )
                    normalized_params["initial_momentum"] = float(raw_value)
                else:
                    raise ValueError(
                        f"Unsupported contextual FloatSOM default key '{param_name}' "
                        f"under '{raw_sampling_key}:{raw_topology_key}'."
                    )

            normalized_topology_map[topology_key] = normalized_params
        normalized[sampling_key] = normalized_topology_map

    return normalized


def resolve_contextual_floatsom_defaults(
    sampling_method: str,
    topology_type: str,
) -> Dict[str, Any]:
    """Return contextual defaults for an exact sampling/topology pair."""
    sampling_key = str(sampling_method).strip().lower()
    topology_key = str(topology_type).strip().lower()
    defaults = load_contextual_floatsom_defaults()
    return dict(defaults.get(sampling_key, {}).get(topology_key, {}))

@dataclass
class SamplingConfig:
    """Configuration for sample selection methods"""
    # General sampling parameters
    method: str = "full"  # Options: "full", "random", "hdsssom"
    samples_per_epoch: Optional[int] = None  # Auto-calculate as % of dataset if None
    target_proportion: float = 0.1 # If we are subsampling, this is the target size as a fraction of the dataset. Only used if samples_per_epoch is not specified
    random_seed: Optional[int] = None
    whole_chunk_random: bool = False
    
    # HDSSSOM-specific parameters
    block_size: int = 1000
    p_block_difficulty: float = 0.9  # Probability of difficulty-based block selection
    p_exemplar_difficulty: float = 0.7  # Probability of difficulty-based exemplar selection
    alpha: float = 0.9  # Difficulty smoothing factor for exponential moving average
    min_blocks_to_select: int = 1  # Minimum number of blocks to select
    
    # Dynamic learning rate based on difficulty
    enable_dynamic_lr: bool = False
    lr_min: float = 0.001
    lr_max: float = 0.1
    
    # Convergence detection
    enable_adaptive_stopping: bool = False
    convergence_threshold: float = 0.001
    convergence_window: int = 100

@dataclass
class RayConfig:
    """Configuration for Ray multi-GPU processing"""
    chunk_size: int  # Must be explicitly provided
    storage_path: str  # Shared directory for manager-side FastArrayStore staging
    local_storage_path: str  # Node-local directory for per-worker FastArrayStore copies
    num_gpus: Optional[int] = None  # Auto-detect if None
    collective_group_name: str = 'default'  # NCCL collective group name
    ray_address: Optional[str] = None  # Ray cluster address
    enable_collective_barriers: bool = True  # Extra collective.barrier sync points (debug/stability; hurts scaling)

    # Data access configuration
    force_copy: bool = False  # Force local copy even if source is accessible
    force_disk_mode: bool = False  # Force disk staging and disable RAM-mode heuristics
    cache_path: Optional[str] = None  # Path for cached zarr copies (default: temp dir)

    # Failure recovery / watchdog configuration
    # NOTE: This is a *stall* watchdog (no progress). Non-positive values fall
    # back to a processor-defined safe default timeout to avoid unbounded waits.
    # Increase for legitimately slow iterations.
    iteration_timeout_s: float = 0.0
    # Local per-stage timeout used by colour workers for gather/sync waits.
    # Non-positive values inherit `iteration_timeout_s`; if that is also
    # non-positive, workers use a safe internal default.
    color_stage_timeout_s: float = 0.0
    iteration_timeout_max_retries: int = 1  # Number of hard-reset retries before failing

    # Hygiene: ensure worker-local scratch does not accumulate across runs.
    wipe_local_storage_on_start: bool = True

    # Timing diagnostics (driver-side INFO logs)
    # Log Ray iteration timing breakdown every N iterations. Values <= 0 disable
    # periodic logging (iteration 0 is still logged when instrumentation is enabled).
    timing_log_every_n_iterations: int = 2

    # Warm up NCCL collectives once during initialization. This forces the
    # NCCL communicator to initialize before the first real iteration so the
    # iteration-0 timing doesn't get dominated by NCCL startup.
    # Set `collective_warmup_iters=0` to disable.
    collective_warmup_iters: int = 1
    collective_warmup_tensor_elements: int = 1024

    def __post_init__(self):
        """Validate configuration.

        IMPORTANT: When running on a Ray cluster, the driver process only "sees"
        GPUs on its own node via CuPy. For multi-node jobs we therefore treat
        `num_gpus=None` as "auto-detect from the Ray cluster" and defer the
        actual GPU count resolution to `RayWorkerManager.initialize_workers()`,
        which queries `ray.cluster_resources()` after connecting to the cluster.
        """
        if not self.storage_path:
            raise ValueError("storage_path must be specified for RayConfig")
        if not self.local_storage_path:
            raise ValueError("local_storage_path must be specified for RayConfig")
            
@dataclass
class ProcessingConfig:
    """Configuration for processing methods"""
    # General processing parameters
    chunk_size: int  # Falls back to 50_000 if not provided
    method: str = "batch"  # Options: "colors", "batch", "minisom"
    
    # Distance metric configuration
    distance_metric: str = "euclidean"  # Options: "euclidean", "cosine", "manhattan", "norm_p"
    distance_metric_params: Dict[str, Any] = field(default_factory=dict)  # e.g., {"p": 3} for norm_p
    
    # Batch processing parameters
    batch_mode: str = "full_batch"  # Options: "full_batch", "minibatch"
    
    # Momentum parameters
    enable_momentum: Optional[bool] = None
    initial_momentum: Optional[float] = None
    final_momentum: float = 0.0
    momentum_decay_type: str = "exponential"  # Options: "exponential", "linear", "fixed", "inverse", "sigmoid"
    enable_adaptive_momentum: bool = False  # Adaptive momentum based on weight change consistency
    
    # Colors processing parameters
    use_gpu: bool = True
    processing_mode: str = "equal_sized"  # Options: "equal_sized", "batch_all"
    sample_order: str = "random"  # Options: "random", "strided"
    max_rounds: int = 1
    num_color_sets: Optional[int] = None  # Auto-detect if None
    color_set_algorithm: str = "greedy_balanced"  # Options: "systematic", "greedy_balanced"
    # When True, scale each color-set update by (color_set_samples / iteration_samples)
    # to emulate minibatch-equivalent aggregate update magnitude.
    # Set False for legacy non-scaled color updates.
    scale_color_updates_by_fraction: bool = True
    # Recompute BMUs before each color-set update (uses latest weights).
    # This reduces stale-BMU drift but increases runtime.
    recompute_bmus_per_color_set: bool = False
    
    
    enable_adaptive_bmu: bool = True
    bmu_recalc_initial: Optional[int] = None  # Initial number of recalcs per iteration
    bmu_recalc_decay_type: str = 'exponential'  # Options: "exponential", "linear", "sigmoid", "gaussian", "asymptotic"
    bmu_recalc_min_samples: int = 1000  # Minimum samples to enable adaptive BMU
    
    # Influence function parameters
    influence_type: str = "gaussian"  # Options: "gaussian", "bubble", "mexican_hat", "triangle"
    influence_params: Dict[str, Any] = field(default_factory=dict)  # e.g., {"std_coeff": 0.5, "compact_support": True}
    # Sparse influence storage for compact-support neighborhoods (batch only).
    use_sparse_influence: bool = False
    
    # Ray multi-GPU configuration (generic for all processing methods)
    ray_config: Optional[RayConfig] = None  # Ray configuration for multi-GPU processing
    
    # Weight update normalization
    normalization: str = "xpysom"  # Options: "count_based", "weighted", "hybrid", "clamped_weighted", "local", "adaptive", "none", "minisom_weighted", "xpysom"
    
    # Normalization parameters (no defaults - must be specified)
    norm_alpha: Optional[float] = None  # For hybrid normalization (blend ratio)
    norm_clamp_factor: Optional[float] = None  # For clamped_weighted normalization
    norm_percentile: Optional[float] = None  # For local normalization (percentile threshold)
    norm_max_update_threshold: Optional[float] = None  # For extreme value validation
    training_progress: Optional[float] = None  # For adaptive normalization (0-1 range)
    current_epoch: Optional[int] = None  # Alternative to training_progress
    total_epochs: Optional[int] = None  # For calculating adaptive progress
    virtual_ratio: float = 0.5  # Virtual samples ratio for count_based normalization (0.2=fast, 0.5=balanced, 1.0=stable)
    
    # Multi-buffering for GPU data transfers (CPU->GPU overlapped transfers).
    # Interpreted as the number of GPU buffers to allocate for the transfer/compute pipeline.
    # Values <= 1 disable buffering; 2=double buffering; 3=triple buffering.
    enable_multi_buffering: int = DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS

    # Async loading configuration (RAM mode distribution)
    async_loading_config: Optional[AsyncLoadingConfig] = None

    # Ray cleanup configuration
    cleanup_config: Optional[CleanupConfig] = None

    # Ray worker profiling configuration
    worker_profile_config: Optional[WorkerProfileConfig] = None

    # Note: When not provided, chunk_size defaults to 50_000

    def __post_init__(self):
        if self.chunk_size is None:
            self.chunk_size = 50_000

        if self.cleanup_config is None:
            self.cleanup_config = CleanupConfig()

        self.enable_multi_buffering = validate_multi_buffering_buffers(self.enable_multi_buffering)

        method = (self.method or "").lower()
        if method == 'colors' and self.chunk_size is not None and self.chunk_size > 1_000_000:
            original_chunk_size = self.chunk_size
            self.chunk_size = 1_000_000
            # Guard runtime stability by preventing enormous color processing batches
            logger.info(
                "Capped colors chunk_size from %d to %d to respect deployment limit",
                original_chunk_size,
                self.chunk_size,
            )

        if self.use_sparse_influence:
            if method != "batch":
                raise ValueError(
                    "Sparse influence is only supported for 'batch' processing."
                )
            influence_type = (self.influence_type or "").lower()
            if influence_type not in {"gaussian", "bubble", "mexican_hat"}:
                raise ValueError(
                    "Sparse influence only supports gaussian, bubble, or mexican_hat kernels."
                )
            if influence_type in {"gaussian", "mexican_hat"}:
                compact_support = bool(self.influence_params.get("compact_support", False))
                if not compact_support:
                    raise ValueError(
                        "Sparse influence requires influence_params['compact_support']=True "
                        "for gaussian or mexican_hat kernels."
                    )

@dataclass
class TopologyConfig:
    """Configuration for topology methods"""
    # Topology type
    topology_type: str = "grid"  # Options: "grid", "hexagonal", "mst", "rng"
    topology_variant: str = "planar"  # Options: "planar", "toroidal" (for grid/hexagonal only)
    
    # Grid/Hexagonal parameters
    grid_size: int = 32
    grid_dim: int = 2  # 1 for 1D chain, 2 for 2D grid
    
    # Influence function for weight updates
    influence_function: str = "gaussian"  # Options: "gaussian", "bubble", "mexican_hat"
    
    # Graph topology parameters (MST/RNG)
    num_nodes: Optional[int] = None  # For MST/RNG topology, overrides grid_size
    mst_update_frequency: Optional[int] = None
    dynamic_mst_frequency: bool = True
    mst_decay_function: str = "exponential"
    initial_mst_frequency: int = 1
    final_mst_frequency: int = 10

@dataclass
class FloatSOMParams:
    """
    Master parameters for FloatSOM - modular SOM architecture
    Combines sampling, processing, and topology configurations
    """
    # Core SOM parameters
    input_dim: Optional[int] = None
    total_iterations: int = 100
    min_iterations: int = 20
    # Match XPySOM default behavior (no convergence-based early stop).
    # A zero threshold effectively disables convergence stopping.
    convergence_threshold: float = 0.0
    initial_learning_rate: float = 0.5
    final_learning_rate: float = 0.01
    initial_radius: Optional[float] = None
    final_radius: float = 1.0
    
    # Decay parameters
    decay_type: str = "exponential"  # Options: "exponential", "linear", "sigmoid", "gaussian", "asymptotic", "fixed"
    radius_decay_type: Optional[str] = None  # If None, uses decay_type
    lr_decay_type: Optional[str] = None  # If None, uses decay_type
    lr_decay_factor: float = 0.3
    radius_decay_factor: Optional[float] = 1.0
    radius_warmup_iters: int = 0
    
    # Modular configurations
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    processing_config: ProcessingConfig = field(
        default_factory=lambda: ProcessingConfig(
            chunk_size=None,
            method="batch",
            batch_mode="full_batch",
            normalization="xpysom",
            use_sparse_influence=False,
        )
    )
    topology_config: TopologyConfig = field(default_factory=TopologyConfig)
    
    # Weight initialization
    initialization_method: Optional[str] = None  # Options: "random", "pca", "pca_sampling", "pca_sampling_snake", "pca_density"
    seed: Optional[int] = None
    
    # Grid reformation parameters (for MST and other non-grid topologies)
    reform_grid: bool = False  # Whether to reform topology to grid structure after training
    reform_grid_type: str = "regular"  # Options: "regular", "hexagonal"
    
    # System parameters
    verbose: bool = False
    enable_profiling: bool = False
    use_gpu: bool = True
    store_history: bool = False
    
    # Advanced parameters
    bitmap_recalc_threshold: float = 0.1
    toy_dataset_mode: bool = False
    
    # MiniSOM integration parameters
    minisom_defaults: bool = False  # Use MiniSOM default hyperparameters instead of FloatSOM defaults
    
    # Dynamic runtime attributes (set during training)
    current_radius: Optional[float] = None
    current_learning_rate: Optional[float] = None
    current_iteration: Optional[int] = None
    current_momentum: Optional[float] = None
    delta_weights: Optional[Any] = None
    selector_callback: Optional[Any] = None
    bmu_scheduler: Optional[Any] = None
    
    def __post_init__(self):
        """Validate and initialize parameters"""
        is_dynamic_graph_topology = self.topology_config.topology_type in {"mst", "rng"}
        contextual_defaults = resolve_contextual_floatsom_defaults(
            self.sampling_config.method,
            self.topology_config.topology_type,
        )

        if is_dynamic_graph_topology:
            if self.topology_config.num_nodes is None:
                fallback_nodes = (self.topology_config.grid_size ** 2
                                   if self.topology_config.grid_dim == 2
                                   else self.topology_config.grid_size)
                self.topology_config.num_nodes = fallback_nodes
            self.total_nodes = self.topology_config.num_nodes
        else:
            self.total_nodes = (self.topology_config.grid_size ** 2
                                if self.topology_config.grid_dim == 2
                                else self.topology_config.grid_size)

        # Set default initial radius based on topology
        if self.initial_radius is None:
            if "initial_radius" in contextual_defaults:
                self.initial_radius = float(contextual_defaults["initial_radius"])
            elif is_dynamic_graph_topology:
                self.initial_radius = max(1, int(np.sqrt(self.total_nodes)))
            else:
                # Match XPySOM: sigma defaults to min(x, y) / 2 (float).
                self.initial_radius = float(self.topology_config.grid_size) / 2.0

        if self.initialization_method is None:
            self.initialization_method = contextual_defaults.get("initialization_method", "random")

        if self.processing_config.enable_momentum is None:
            self.processing_config.enable_momentum = bool(contextual_defaults.get("enable_momentum", False))

        if self.processing_config.initial_momentum is None:
            self.processing_config.initial_momentum = float(contextual_defaults.get("initial_momentum", 0.5))

        # Set default radius_decay_factor based on topology type
        if self.radius_decay_factor is None:
            self.radius_decay_factor = 1.0

        # Set specific decay types if not specified
        if self.radius_decay_type is None:
            self.radius_decay_type = contextual_defaults.get("radius_decay_type", self.decay_type)
        if self.lr_decay_type is None:
            self.lr_decay_type = self.decay_type

        # Validate decay types
        valid_decay_types = sorted(_VALID_DECAY_TYPES)
        if self.decay_type not in valid_decay_types:
            raise ValueError(f"Invalid decay_type: {self.decay_type}. Must be one of {valid_decay_types}")
        if self.radius_decay_type not in valid_decay_types:
            raise ValueError(f"Invalid radius_decay_type: {self.radius_decay_type}. Must be one of {valid_decay_types}")
        if self.lr_decay_type not in valid_decay_types:
            raise ValueError(f"Invalid lr_decay_type: {self.lr_decay_type}. Must be one of {valid_decay_types}")

        if self.initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate must be positive")
        if self.final_learning_rate < 0:
            raise ValueError("final_learning_rate must be non-negative")
        if self.final_radius <= 0:
            raise ValueError("final_radius must be positive")
        
        # Validate sampling method
        valid_sampling_methods = ["full", "random", "hdsssom"]
        if self.sampling_config.method not in valid_sampling_methods:
            raise ValueError(f"Invalid sampling method: {self.sampling_config.method}. Must be one of {valid_sampling_methods}")
        
        # Validate processing method
        valid_processing_methods = ["colors", "batch", "minisom"]
        if self.processing_config.method not in valid_processing_methods:
            raise ValueError(f"Invalid processing method: {self.processing_config.method}. Must be one of {valid_processing_methods}")
        
        # Validate batch mode for batch processor
        if self.processing_config.method == "batch":
            valid_batch_modes = ["full_batch", "minibatch"]
            if self.processing_config.batch_mode not in valid_batch_modes:
                raise ValueError(f"Invalid batch_mode: {self.processing_config.batch_mode}. Must be one of {valid_batch_modes}")
        
        # Validate topology type
        valid_topology_types = ["grid", "hexagonal", "mst", "rng"]
        if self.topology_config.topology_type not in valid_topology_types:
            raise ValueError(f"Invalid topology type: {self.topology_config.topology_type}. Must be one of {valid_topology_types}")
        
        # Validate topology variant
        valid_topology_variants = ["planar", "toroidal"]
        if self.topology_config.topology_variant not in valid_topology_variants:
            raise ValueError(f"Invalid topology variant: {self.topology_config.topology_variant}. Must be one of {valid_topology_variants}")
        
        # Toroidal only valid for grid and hexagonal
        if (
            self.topology_config.topology_variant == "toroidal"
            and self.topology_config.topology_type in {"mst", "rng"}
        ):
            topology_upper = self.topology_config.topology_type.upper()
            raise ValueError(f"Toroidal topology is not supported for {topology_upper} topology")
        
        # Validate HDSSSOM parameters
        if self.sampling_config.method == "hdsssom":
            if not (0 <= self.sampling_config.p_block_difficulty <= 1):
                raise ValueError("p_block_difficulty must be between 0 and 1")
            if not (0 <= self.sampling_config.p_exemplar_difficulty <= 1):
                raise ValueError("p_exemplar_difficulty must be between 0 and 1")
            if not (0 < self.sampling_config.alpha <= 1):
                raise ValueError("alpha must be between 0 and 1")

        if self.sampling_config.whole_chunk_random:
            if self.sampling_config.method != "random":
                raise ValueError(
                    "whole_chunk_random requires sampling_config.method='random'"
                )
            if self.processing_config.method != "batch":
                raise ValueError(
                    "whole_chunk_random requires processing_config.method='batch'"
                )
        
        # Validate processing config
        if self.processing_config.method == "colors":
            valid_processing_modes = ["equal_sized", "batch_all"]
            if self.processing_config.processing_mode not in valid_processing_modes:
                raise ValueError(f"Invalid processing_mode: {self.processing_config.processing_mode}")
            
            valid_sample_orders = ["random", "strided"]
            if self.processing_config.sample_order not in valid_sample_orders:
                raise ValueError(f"Invalid sample_order: {self.processing_config.sample_order}")
            
            # Validate BMU recalculation parameters
            if self.processing_config.enable_adaptive_bmu:
                valid_bmu_decay_types = ["exponential", "linear", "sigmoid", "gaussian", "asymptotic"]
                if self.processing_config.bmu_recalc_decay_type not in valid_bmu_decay_types:
                    raise ValueError(f"Invalid bmu_recalc_decay_type: {self.processing_config.bmu_recalc_decay_type}")
                
                if self.processing_config.bmu_recalc_min_samples < 1:
                    raise ValueError("bmu_recalc_min_samples must be >= 1")
            
            # Validate color set algorithm
            valid_color_algorithms = ["systematic", "greedy_balanced"]
            if self.processing_config.color_set_algorithm not in valid_color_algorithms:
                raise ValueError(f"Invalid color_set_algorithm: {self.processing_config.color_set_algorithm}")
            
            # Validate max_rounds
            if self.processing_config.max_rounds < 1:
                raise ValueError("max_rounds must be >= 1")
        
        # Validate normalization
        valid_normalizations = [
            "count_based",
            "weighted",
            "hybrid",
            "clamped_weighted",
            "local",
            "adaptive",
            "none",
            "minisom_weighted",
            "xpysom",
        ]
        if self.processing_config.normalization not in valid_normalizations:
            raise ValueError(f"Invalid normalization: {self.processing_config.normalization}")
        
        # Validate virtual_ratio parameter
        if not (0.0 < self.processing_config.virtual_ratio <= 2.0):
            raise ValueError("virtual_ratio must be between 0.0 and 2.0")
                
        # Validate normalization parameters
        if self.processing_config.normalization == "hybrid":
            if self.processing_config.norm_alpha is None:
                raise ValueError("norm_alpha must be specified for hybrid normalization")
            if not (0 <= self.processing_config.norm_alpha <= 1):
                raise ValueError("norm_alpha must be between 0 and 1")
        
        if self.processing_config.normalization == "clamped_weighted":
            if self.processing_config.norm_clamp_factor is None:
                raise ValueError("norm_clamp_factor must be specified for clamped_weighted normalization")
            if self.processing_config.norm_clamp_factor <= 0:
                raise ValueError("norm_clamp_factor must be positive")
        
        if self.processing_config.normalization == "local":
            if self.processing_config.norm_percentile is None:
                raise ValueError("norm_percentile must be specified for local normalization")
            if not (0 < self.processing_config.norm_percentile <= 100):
                raise ValueError("norm_percentile must be between 0 and 100")
        
        if self.processing_config.normalization == "adaptive":
            # Set defaults for adaptive normalization if not provided
            if (self.processing_config.training_progress is None and 
                (self.processing_config.current_epoch is None or self.processing_config.total_epochs is None)):
                # Set default values for adaptive normalization
                self.processing_config.current_epoch = 0
                self.processing_config.total_epochs = self.total_iterations
            
            if self.processing_config.training_progress is not None:
                if not (0 <= self.processing_config.training_progress <= 1):
                    raise ValueError("training_progress must be between 0 and 1")
            if self.processing_config.current_epoch is not None and self.processing_config.current_epoch < 0:
                raise ValueError("current_epoch must be non-negative")
            if self.processing_config.total_epochs is not None and self.processing_config.total_epochs <= 0:
                raise ValueError("total_epochs must be positive")
        
        if self.processing_config.norm_max_update_threshold is not None:
            if self.processing_config.norm_max_update_threshold <= 0:
                raise ValueError("norm_max_update_threshold must be positive")
        
        # Validate topology-algorithm compatibility
        if self.processing_config.method == "colors":
            # Check if color set algorithm is compatible with topology
            if self.topology_config.topology_type == "hexagonal":
                # Greedy balanced works best for hexagonal due to topology-agnostic approach
                if self.processing_config.color_set_algorithm == "systematic":
                    warnings.warn(
                        "Systematic color set algorithm may not be optimal for hexagonal topology. "
                        "Consider using 'greedy_balanced' for better performance with hexagonal grids."
                    )
            elif self.topology_config.topology_type == "mst":
                # MST topology requires special handling
                warnings.warn(
                    "Color set processing with MST topology is experimental. "
                    "The dynamic nature of MST may affect color set stability."
                )
            elif self.topology_config.topology_type == "rng":
                warnings.warn(
                    "Color set processing with RNG topology is experimental. "
                    "The dynamic nature of RNG may affect color set stability."
                )
        
        # Validate momentum parameters
        if self.processing_config.enable_momentum:
            if not (0 <= self.processing_config.initial_momentum <= 1):
                raise ValueError("initial_momentum must be between 0 and 1")
            if not (0 <= self.processing_config.final_momentum <= 1):
                raise ValueError("final_momentum must be between 0 and 1")
            
            valid_momentum_decay_types = ["exponential", "linear", "fixed", "inverse", "sigmoid"]
            if self.processing_config.momentum_decay_type not in valid_momentum_decay_types:
                raise ValueError(f"Invalid momentum_decay_type: {self.processing_config.momentum_decay_type}")
        
        # Validate distance metric
        valid_distance_metrics = ["euclidean", "cosine", "manhattan", "norm_p"]
        if self.processing_config.distance_metric not in valid_distance_metrics:
            raise ValueError(f"Invalid distance_metric: {self.processing_config.distance_metric}. Must be one of {valid_distance_metrics}")
        
        # Validate distance metric parameters
        if self.processing_config.distance_metric == "norm_p":
            if "p" not in self.processing_config.distance_metric_params:
                self.processing_config.distance_metric_params["p"] = 3.0  # Default p value
            elif self.processing_config.distance_metric_params["p"] < 1:
                raise ValueError("p value for norm_p distance must be >= 1")
        
        # Validate influence type
        valid_influence_types = ["gaussian", "bubble", "mexican_hat", "triangle"]
        if self.processing_config.influence_type not in valid_influence_types:
            raise ValueError(f"Invalid influence_type: {self.processing_config.influence_type}. Must be one of {valid_influence_types}")
        
        # FloatSOM runtime is GPU-only.
        if not bool(self.use_gpu) or not bool(self.processing_config.use_gpu):
            raise ValueError(
                "CPU execution paths have been removed. "
                "Set use_gpu=True in both FloatSOMParams and ProcessingConfig."
            )
        self.use_gpu = True
        self.processing_config.use_gpu = True

        # Warn about normalization=none
        if self.processing_config.normalization == "none":
            warnings.warn("Normalization is set to none. Scale LR to be higher, weight updates normed by dividing by number of samples")
    
    def get_architecture_summary(self) -> str:
        """Get summary of the selected FloatSOM architecture"""
        return (f"FloatSOM: {self.sampling_config.method.title()}Sampling × "
                f"{self.processing_config.method.title()}Processing × "
                f"{self.topology_config.topology_type.title()}Topology "
                f"({self.topology_config.topology_variant}) | "
                f"Distance: {self.processing_config.distance_metric}, "
                f"Influence: {self.processing_config.influence_type}")
    
# Utility functions for decay calculations (preserved from original)
def calculate_radius(current_iteration: int, total_iterations: int, initial_radius: float,
                    decay_type: str, min_radius: float = 0.01, use_minisom: bool = False,
                    radius_warmup_iters: int = 0, radius_decay_factor: float = 1.0, 
                    radius_decay_type: Optional[str] = None) -> float:
    """Calculate the current radius based on iteration and decay type
    
    The radius always decays from initial_radius to min_radius over the training period.
    The radius_decay_factor controls the shape/slope of the decay curve:
    - Higher values (e.g., 3.0) make the decay slower/gentler
    - Lower values (e.g., 0.5) make the decay faster/steeper
    """
    import inspect
    
    # Auto-detect MiniSom context if not explicitly specified
    if not use_minisom:
        for frame in inspect.stack():
            if 'minisom' in frame.filename.lower():
                use_minisom = True
                break
    
    # During warmup period, keep radius constant
    if current_iteration < radius_warmup_iters:
        return initial_radius
    
    # Use radius_decay_type if provided, otherwise use decay_type
    if radius_decay_type is not None:
        decay_type = radius_decay_type
    
    # Special handling for 'fixed' learning rate mode
    if decay_type == 'fixed':
        return initial_radius
    
    # Adjust iteration and total for the actual decay period
    adjusted_iteration = current_iteration - radius_warmup_iters
    adjusted_total = total_iterations - radius_warmup_iters
    
    if adjusted_total <= 0:
        return initial_radius
    
    # XPySOM-compatible decays for shared decay names.
    # Keep custom FloatSOM-only decays below for non-overlapping names.
    max_iter = int(adjusted_total)
    curr_iter = int(np.clip(adjusted_iteration, 0, max(0, max_iter - 1)))
    if decay_type == "exponential":
        if min_radius == 0:
            diff = -np.log(0.1) / max_iter
        else:
            diff = -np.log(min_radius / initial_radius) / max_iter
        radius = initial_radius * np.exp(-curr_iter * diff)
        return float(np.clip(radius, min(min_radius, initial_radius), max(min_radius, initial_radius)))
    elif decay_type == "linear":
        if max_iter != 1:
            radius = initial_radius + (min_radius - initial_radius) * curr_iter / (max_iter - 1)
        else:
            radius = initial_radius
        return float(radius)
    elif decay_type == "asymptotic":
        return float(initial_radius / (1 + 2 * curr_iter / max_iter))

    # Calculate progress (0.0 to 1.0) over the decay period
    progress = adjusted_iteration / adjusted_total
    progress = np.clip(progress, 0.0, 1.0)
    
    # Apply decay factor to modify the progress curve
    # Higher decay factor = slower/gentler curve, lower = faster/steeper
    if decay_type == "sigmoid":
        # Sigmoid decay with decay factor controlling transition steepness
        steepness = 10.0 / radius_decay_factor
        midpoint = 0.5
        normalized_decay = 1.0 / (1.0 + np.exp(steepness * (progress - midpoint)))
    elif decay_type == "gaussian":
        # Gaussian decay with decay factor controlling width
        sigma = 0.5 * radius_decay_factor  # Higher factor = wider/slower decay
        normalized_decay = np.exp(-(progress**2) / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    
    # Scale the normalized decay to go from initial_radius to min_radius
    radius = min_radius + (initial_radius - min_radius) * normalized_decay
    
    # Ensure we never go below min_radius or above initial_radius
    return np.clip(radius, min_radius, initial_radius)

def calculate_learning_rate(current_iteration: int, total_iterations: int, initial_learning_rate: float,
                           decay_type: str, final_learning_rate: float = 0.01,
                           use_minisom: bool = False, lr_decay_factor: float = 8.0,
                           lr_decay_type: Optional[str] = None) -> float:
    """Calculate the current learning rate based on iteration and decay type"""
    
    # Use lr_decay_type if provided, otherwise use decay_type
    if lr_decay_type is not None:
        decay_type = lr_decay_type
    
    # Special case for MiniSom compatibility
    if use_minisom:
        t = current_iteration / total_iterations
        
        if decay_type == "exponential":
            return max(initial_learning_rate * np.exp(-5 * t), 0.01)
        elif decay_type == "linear":
            return max(initial_learning_rate * (1 - t), 0.01)
        elif decay_type == "asymptotic":
            return max(initial_learning_rate / (1 + t), 0.01)
        else:
            return max(initial_learning_rate / (1 + t), 0.01)
    
    # FloatSOM implementations with configurable decay
    if decay_type == "exponential":
        max_iter = max(1, int(total_iterations))
        curr_iter = int(np.clip(current_iteration, 0, max(0, max_iter - 1)))
        if final_learning_rate == 0:
            diff = -np.log(0.1) / max_iter
        else:
            diff = -np.log(final_learning_rate / initial_learning_rate) / max_iter
        return float(initial_learning_rate * np.exp(-curr_iter * diff))
    elif decay_type == "linear":
        max_iter = max(1, int(total_iterations))
        curr_iter = int(np.clip(current_iteration, 0, max(0, max_iter - 1)))
        if max_iter != 1:
            return float(
                initial_learning_rate
                + (final_learning_rate - initial_learning_rate) * curr_iter / (max_iter - 1)
            )
        return float(initial_learning_rate)
    elif decay_type == "sigmoid":
        half_point = total_iterations / 2
        return max(
            initial_learning_rate / (1 + np.exp((current_iteration - half_point) / (total_iterations / (lr_decay_factor * 2.5)))),
            final_learning_rate,
        )
    elif decay_type == "gaussian":
        return max(
            initial_learning_rate * np.exp(-current_iteration**2 / (2 * (total_iterations / lr_decay_factor)**2)),
            final_learning_rate,
        )
    elif decay_type == "asymptotic":
        max_iter = max(1, int(total_iterations))
        curr_iter = int(np.clip(current_iteration, 0, max(0, max_iter - 1)))
        return float(initial_learning_rate / (1 + 2 * curr_iter / max_iter))
    elif decay_type == "fixed":
        return initial_learning_rate
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

def precompute_all_radii(total_iterations: int, initial_radius: float, decay_type: str, 
                        use_minisom: bool = False, radius_decay_factor: float = 1.0, 
                        radius_warmup_iters: int = 0) -> List[float]:
    """Pre-compute all radius values for consistent topology caching"""
    radii_list = []
    for iteration in range(total_iterations):
        radius = calculate_radius(iteration, total_iterations, initial_radius, decay_type, 
                                 use_minisom=use_minisom, radius_decay_factor=radius_decay_factor, 
                                 radius_warmup_iters=radius_warmup_iters)
        radii_list.append(round(radius, 3))
    return radii_list

def precompute_all_learning_rates(total_iterations: int, initial_learning_rate: float,
                                 decay_type: str, final_learning_rate: float = 0.01,
                                 use_minisom: bool = False,
                                 lr_decay_factor: float = 8.0) -> List[float]:
    """Pre-compute all learning rate values for the entire training process"""
    learning_rates_list = []
    for iteration in range(total_iterations):
        lr = calculate_learning_rate(iteration, total_iterations, initial_learning_rate, 
                                   decay_type, final_learning_rate=final_learning_rate,
                                   use_minisom=use_minisom, lr_decay_factor=lr_decay_factor)
        learning_rates_list.append(lr)
    return learning_rates_list


def calculate_momentum_coefficient(current_iteration: int, total_iterations: int, 
                                 initial_momentum: float, final_momentum: float = 0.0,
                                 momentum_decay_type: str = "exponential") -> float:
    """
    Calculate the current momentum coefficient based on iteration and decay type
    
    Args:
        current_iteration: Current training iteration
        total_iterations: Total number of training iterations
        initial_momentum: Starting momentum coefficient (typically 0.5-0.9)
        final_momentum: Final momentum coefficient (typically 0.0-0.1)
        momentum_decay_type: Type of decay ("exponential", "linear", "fixed", "inverse")
        
    Returns:
        Current momentum coefficient
    """
    if total_iterations <= 0:
        return initial_momentum
    
    # Clamp iteration to valid range
    current_iteration = max(0, min(current_iteration, total_iterations - 1))
    
    if momentum_decay_type == "fixed":
        return initial_momentum
    
    # Calculate progress ratio (0.0 to 1.0)
    # Use (total_iterations - 1) so that the final iteration
    # reaches progress == 1.0, matching linear schedule tests.
    progress = current_iteration / max(1, total_iterations - 1)
    
    if momentum_decay_type == "linear":
        # Linear decay from initial to final
        return initial_momentum * (1 - progress) + final_momentum * progress
    
    elif momentum_decay_type == "exponential":
        # Exponential decay with configurable steepness
        decay_factor = 3.0  # Controls steepness of decay
        return final_momentum + (initial_momentum - final_momentum) * np.exp(-decay_factor * progress)
    
    elif momentum_decay_type == "inverse":
        # Inverse decay (similar to asymptotic)
        return final_momentum + (initial_momentum - final_momentum) / (1 + progress * 2)
    
    elif momentum_decay_type == "sigmoid":
        # Sigmoid decay for smooth transition
        midpoint = 0.5  # Where the transition happens fastest
        steepness = 10.0  # Controls transition sharpness
        sigmoid_val = 1 / (1 + np.exp(steepness * (progress - midpoint)))
        return final_momentum + (initial_momentum - final_momentum) * sigmoid_val

    elif momentum_decay_type == "cosine":
        # Cosine annealing from initial to final momentum.
        # progress = 0   -> coefficient = initial_momentum
        # progress = 1   -> coefficient = final_momentum
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return final_momentum + (initial_momentum - final_momentum) * cosine_factor
    
    else:
        raise ValueError(f"Unknown momentum decay type: {momentum_decay_type}")


def precompute_all_momentum_coefficients(total_iterations: int, initial_momentum: float,
                                       final_momentum: float = 0.0, 
                                       momentum_decay_type: str = "exponential") -> List[float]:
    """
    Pre-compute all momentum coefficient values for the entire training process
    
    Args:
        total_iterations: Total number of training iterations
        initial_momentum: Starting momentum coefficient
        final_momentum: Final momentum coefficient
        momentum_decay_type: Type of decay
        
    Returns:
        List of momentum coefficients for each iteration
    """
    momentum_coefficients = []
    for iteration in range(total_iterations):
        momentum_coeff = calculate_momentum_coefficient(
            iteration, total_iterations, initial_momentum, 
            final_momentum, momentum_decay_type
        )
        momentum_coefficients.append(float(round(momentum_coeff, 6)))
    # Return as NumPy array so schedule comparisons in tests use
    # elementwise operations (e.g., schedule >= 0) without type errors.
    return np.asarray(momentum_coefficients, dtype=float)


def calculate_mst_update_frequency(current_iteration: int, total_iterations: int,
                                  initial_frequency: int = 1, final_frequency: int = 10,
                                  decay_type: str = "exponential") -> int:
    """
    Calculate the current MST update frequency based on iteration and decay type.
    
    The frequency starts low (frequent updates) and increases (less frequent updates)
    as training progresses. This is the opposite of radius/learning rate decay.
    
    Args:
        current_iteration: Current training iteration
        total_iterations: Total number of training iterations
        initial_frequency: Initial update frequency (update every N iterations) - typically 1
        final_frequency: Final update frequency (update every N iterations) - typically 10-50
        decay_type: Type of decay function ("exponential", "linear", "sigmoid", "gaussian", "asymptotic")
        
    Returns:
        Current MST update frequency (update every N iterations)
    """
    if total_iterations <= 0:
        return initial_frequency
    
    # Calculate progress (0.0 to 1.0)
    progress = min(current_iteration / total_iterations, 1.0)
    
    if decay_type == "exponential":
        # Exponential growth from initial to final frequency
        decay_rate = 5.0  # Controls the steepness - increased for faster decay
        growth_factor = np.exp(decay_rate * progress)
        max_growth = np.exp(decay_rate)
        normalized_growth = (growth_factor - 1) / (max_growth - 1)
        
    elif decay_type == "linear":
        # Linear growth from initial to final
        normalized_growth = progress
        
    elif decay_type == "sigmoid":
        # Sigmoid growth for smooth transition
        steepness = 15.0  # Increased for faster transition
        midpoint = 0.5
        normalized_growth = 1.0 / (1.0 + np.exp(-steepness * (progress - midpoint)))
        
    elif decay_type == "gaussian":
        # Gaussian-based growth (slow start, fast middle, slow end)
        # Using inverted gaussian for growth
        sigma = 0.3
        normalized_growth = 1.0 - np.exp(-(progress - 0.5)**2 / (2 * sigma**2))
        # Rescale to [0, 1]
        max_val = 1.0 - np.exp(-0.25 / (2 * sigma**2))
        normalized_growth = normalized_growth / max_val
        
    elif decay_type == "asymptotic":
        # Asymptotic growth (fast initial growth, slow later)
        steepness = 5.0  # Increased for faster initial growth
        normalized_growth = (steepness * progress) / (1.0 + steepness * progress)
        
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
    
    # Scale the normalized growth to go from initial_frequency to final_frequency
    frequency = initial_frequency + (final_frequency - initial_frequency) * normalized_growth
    
    # Round to nearest integer and ensure bounds
    frequency = max(initial_frequency, min(final_frequency, round(frequency)))
    
    return frequency


def precompute_all_mst_frequencies(total_iterations: int, initial_frequency: int = 1,
                                  final_frequency: int = 10, decay_type: str = "exponential") -> List[int]:
    """
    Pre-compute all MST update frequencies for the entire training process
    
    Args:
        total_iterations: Total number of training iterations
        initial_frequency: Starting update frequency (update every N iterations)
        final_frequency: Final update frequency (update every N iterations)
        decay_type: Type of decay function
        
    Returns:
        List of MST update frequencies for each iteration
    """
    frequencies = []
    for iteration in range(total_iterations):
        freq = calculate_mst_update_frequency(
            iteration, total_iterations, initial_frequency,
            final_frequency, decay_type
        )
        frequencies.append(freq)
    return frequencies


def calculate_adaptive_momentum_coefficient(current_iteration: int, total_iterations: int,
                                          weight_change: float, previous_weight_change: float,
                                          base_momentum: float = 0.5, 
                                          min_momentum: float = 0.1, 
                                          max_momentum: float = 0.9) -> float:
    """
    Calculate adaptive momentum coefficient based on weight change dynamics
    
    Higher momentum when weight changes are consistent (same direction)
    Lower momentum when weight changes are inconsistent (oscillating)
    
    Args:
        current_iteration: Current training iteration
        total_iterations: Total number of training iterations
        weight_change: Current magnitude of weight change
        previous_weight_change: Previous magnitude of weight change
        base_momentum: Base momentum coefficient
        min_momentum: Minimum allowed momentum
        max_momentum: Maximum allowed momentum
        
    Returns:
        Adaptive momentum coefficient
    """
    if current_iteration == 0 or previous_weight_change == 0:
        return base_momentum
    
    # Calculate change consistency (ratio of current to previous change)
    change_ratio = weight_change / previous_weight_change
    
    # Adaptive adjustment based on consistency
    if 0.8 <= change_ratio <= 1.2:
        # Consistent changes - increase momentum
        momentum_adjustment = 0.1
    elif 0.5 <= change_ratio <= 2.0:
        # Moderately consistent - keep base momentum
        momentum_adjustment = 0.0
    else:
        # Inconsistent changes - reduce momentum
        momentum_adjustment = -0.2
    
    # Apply progress-based decay
    progress = current_iteration / total_iterations
    decay_factor = 1.0 - 0.3 * progress  # Gradually reduce momentum over time
    
    # Calculate final momentum
    adaptive_momentum = (base_momentum + momentum_adjustment) * decay_factor
    
    # Clamp to valid range
    return max(min_momentum, min(adaptive_momentum, max_momentum))
