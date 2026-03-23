"""
Benchmark configuration management for Phase 1 and Phase 3 implementation.

This module provides configuration classes and utilities for managing
benchmark scenarios and dataset configurations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from .parameters import get_default_forced_combinations
import itertools


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    difficulty: str = 'hard'
    normalize: bool = True
    seed: Optional[int] = None
    n_features: Optional[int] = None  # For blobs dataset
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        config = {
            'difficulty': self.difficulty,
            'normalize': self.normalize
        }
        if self.seed is not None:
            config['seed'] = self.seed
        if self.n_features is not None:
            config['n_features'] = self.n_features
        return config


@dataclass 
class BenchmarkScenario:
    """Configuration for a single benchmark scenario."""
    name: str
    dataset: DatasetConfig
    forced_params: Dict[str, Any]
    n_trials: int = 100
    timeout: Optional[float] = None


@dataclass
class Phase1BenchmarkConfig:
    """
    Configuration class for Phase 1 benchmarks.
    
    This simplified configuration focuses on single benchmark execution
    and basic dataset testing before moving to full orchestration.
    """
    
    # Available datasets for Phase 1 testing
    datasets: List[str] = None
    
    # Number of trials per benchmark
    n_trials_per_benchmark: int = 100
    
    # GPU configuration
    use_gpu: bool = True
    gpu_id: Optional[int] = 0
    
    # Storage configuration
    storage_url: str = "sqlite:///floatsom_optuna_phase1.db"
    
    # Output configuration
    output_dir: Optional[str] = None
    
    # Metrics configuration
    topology_k: int = 7
    
    # Timeout per benchmark (seconds)
    timeout_per_benchmark: Optional[float] = None
    
    def __post_init__(self):
        """Set default datasets if not provided."""
        if self.datasets is None:
            self.datasets = [
                'swiss_roll',
                'moons', 
                'circles',
                'blobs',
                's_curve',
                'breast_cancer',
                'wine',
                'iris',
                'digits',
                'olivetti_faces',
                'diabetes',
                'california_housing',
                'covertype',
                'kddcup99'
            ]
    
    def get_dataset_configs(self) -> List[DatasetConfig]:
        """Get dataset configurations for all specified datasets."""
        configs = []
        for dataset_name in self.datasets:
            config = DatasetConfig(
                name=dataset_name,
                difficulty='hard',  # Use hard difficulty for challenging benchmarks
                normalize=True
            )
            configs.append(config)
        return configs
    
    def create_test_scenarios(self, seed: int = 42) -> List[BenchmarkScenario]:
        """
        Create a small set of test scenarios for Phase 1 validation.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            List of benchmark scenarios for testing
        """
        scenarios = []
        
        # Get first few forced parameter combinations for testing
        forced_combinations = get_default_forced_combinations()
        test_combinations = forced_combinations[:3]  # Just test first 3 combinations
        
        # Use first dataset for initial testing
        test_dataset = DatasetConfig(
            name='swiss_roll',
            difficulty='hard',  # Use hard difficulty for challenging benchmarks
            normalize=True,
            seed=seed
        )
        
        for i, forced_params in enumerate(test_combinations):
            scenario = BenchmarkScenario(
                name=f"test_scenario_{i+1}",
                dataset=test_dataset,
                forced_params=forced_params,
                n_trials=20,  # Small number for quick testing
                timeout=300  # 5 minute timeout
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def create_single_algorithm_scenarios(
        self, 
        algorithm: str = 'batch',
        seed: int = 42
    ) -> List[BenchmarkScenario]:
        """
        Create scenarios for testing a single algorithm across datasets.
        
        Args:
            algorithm: Algorithm to test ('batch')
            seed: Random seed
            
        Returns:
            List of benchmark scenarios
        """
        scenarios = []
        
        # Find forced combinations for this algorithm
        forced_combinations = get_default_forced_combinations()
        algo_combinations = [
            combo for combo in forced_combinations 
            if combo['processing_method'] == algorithm
        ]
        
        # Test on subset of datasets (full config includes additional datasets)
        test_datasets = ['swiss_roll', 'moons', 'blobs']
        
        for dataset_name in test_datasets:
            dataset_config = DatasetConfig(
                name=dataset_name,
                difficulty='hard',
                normalize=True,
                seed=seed
            )
            
            # Use first combination for this algorithm
            if algo_combinations:
                forced_params = algo_combinations[0]
                
                scenario = BenchmarkScenario(
                    name=f"{algorithm}_{dataset_name}_test",
                    dataset=dataset_config,
                    forced_params=forced_params,
                    n_trials=self.n_trials_per_benchmark,
                    timeout=self.timeout_per_benchmark
                )
                scenarios.append(scenario)
        
        return scenarios


@dataclass
class AlgorithmVariant:
    """Configuration for a single algorithm variant."""
    name: str                      # 'batch'
    processing_method: str         # Maps to existing forced params
    description: str               # Human-readable description
    default_params: Dict[str, Any] # Algorithm-specific defaults
    supports_gpu: bool = True      # Whether this algorithm can use GPU


@dataclass 
class ExtendedBenchmarkScenario:
    """Extended benchmark scenario configuration for Phase 3."""
    name: str
    algorithm: str                 # Algorithm variant name 
    dataset: DatasetConfig
    forced_params: Dict[str, Any]
    n_trials: int = 200
    n_seeds: int = 5
    gpu_id: Optional[int] = None   # Assigned GPU for this scenario


@dataclass
class Phase3BenchmarkConfig:
    """
    Configuration class for Phase 3 full orchestration benchmarks.
    
    This comprehensive configuration manages all benchmarks across
    batch-only algorithms × datasets × forced parameter combinations
    (including batch_mode splits). Scenario counts are computed dynamically.
    """
    
    # Core configuration - dynamically set from parameters.py
    algorithms: List[str] = None
    datasets: List[str] = None
    
    # Execution parameters
    n_seeds: int = 5
    n_trials_per_benchmark: int = 200
    n_gpus: Optional[int] = None
    
    # Storage and resource management
    max_concurrent_benchmarks: int = 8
    checkpoint_interval: int = 50
    
    # Output configuration
    output_dir: str = "./phase3_results"
    
    def __post_init__(self):
        """Set default algorithms and datasets if not provided."""
        if self.algorithms is None:
            # Import here to avoid circular imports
            from .parameters import get_algorithm_variants
            self.algorithms = get_algorithm_variants()
        
        if self.datasets is None:
            # 14 sklearn datasets for comprehensive benchmarking
            self.datasets = [
                'swiss_roll',
                'moons', 
                'circles',
                'blobs',
                's_curve',
                'breast_cancer',
                'wine',
                'iris',
                'digits',
                'olivetti_faces',
                'diabetes',
                'california_housing',
                'covertype',
                'kddcup99'
            ]
    
    @property
    def algorithm_variants(self) -> List[AlgorithmVariant]:
        """Get algorithm variant configurations using defaults from FloatSOMParams."""
        # Import FloatSOM parameter configs to get sensible defaults
        from ....floatsom_params import ProcessingConfig, FloatSOMParams
        
        # Derive a default chunk size from FloatSOM defaults to satisfy required param
        default_chunk_size = FloatSOMParams().processing_config.chunk_size

        # Get default processing configs for each method with explicit chunk_size
        batch_config = ProcessingConfig(method='batch', chunk_size=default_chunk_size)
        
        variants = {
            'batch': AlgorithmVariant(
                name='batch',
                processing_method='batch',
                description='Batch processing with configurable batch sizes and modes',
                default_params={
                    'batch_mode': batch_config.batch_mode,
                    'chunk_size': batch_config.chunk_size,
                    'use_momentum': batch_config.enable_momentum,
                    'momentum_init': batch_config.initial_momentum,
                    'normalization': batch_config.normalization
                },
                supports_gpu=True
            )
        }
        
        return [variants[name] for name in self.algorithms if name in variants]
    
    def get_algorithm_variant(self, name: str) -> AlgorithmVariant:
        """Get specific algorithm variant by name."""
        for variant in self.algorithm_variants:
            if variant.name == name:
                return variant
        raise ValueError(f"Algorithm variant '{name}' not found in configuration")
    
    def get_dataset_configs(self) -> List[DatasetConfig]:
        """Get dataset configurations for all specified datasets."""
        configs = []
        for dataset_name in self.datasets:
            # Use hard difficulty for all datasets for challenging benchmarks
            difficulty = 'hard'
            
            config = DatasetConfig(
                name=dataset_name,
                difficulty=difficulty,
                normalize=True
            )
            configs.append(config)
        return configs
    
    def generate_all_scenarios(self, seed: int = 42) -> List[ExtendedBenchmarkScenario]:
        """
        Generate all benchmark scenarios using dynamic forced combinations.
        
        Scenario counts are computed from forced combinations and configured datasets.
        
        Returns:
            List of all benchmark scenarios to be executed
        """
        scenarios = []
        scenario_id = 1
        
        dataset_configs = self.get_dataset_configs()
        # Use the existing forced combinations from parameters.py
        forced_combinations = get_default_forced_combinations()
        
        for algorithm in self.algorithms:
            for dataset_config in dataset_configs:
                # Filter forced combinations for this algorithm
                algo_combinations = [
                    combo for combo in forced_combinations 
                    if combo['processing_method'] == algorithm
                ]
                
                for forced_params in algo_combinations:
                    scenario = ExtendedBenchmarkScenario(
                        name=(
                            f"scenario_{scenario_id:03d}_{algorithm}_{dataset_config.name}_"
                            f"{forced_params['sampling_method']}_{forced_params['batch_mode']}_"
                            f"{forced_params['topology_type']}"
                        ),
                        algorithm=algorithm,
                        dataset=dataset_config,
                        forced_params=forced_params,
                        n_trials=self.n_trials_per_benchmark,
                        n_seeds=self.n_seeds
                    )
                    scenarios.append(scenario)
                    scenario_id += 1
        
        return scenarios
    
    def validate_configuration(self) -> bool:
        """Validate the benchmark configuration."""
        try:
            # Import here to avoid circular imports
            from .parameters import get_algorithm_variants
            
            # Check that we have valid algorithms
            valid_algorithms = get_algorithm_variants()
            for algo in self.algorithms:
                if algo not in valid_algorithms:
                    raise ValueError(f"Invalid algorithm: {algo}")
            
            # Check that we have valid datasets
            if len(self.datasets) == 0:
                raise ValueError("No datasets specified")
            
            # Check resource constraints
            if self.n_gpus is not None and self.n_gpus < 0:
                raise ValueError("Invalid GPU count")
            
            if self.n_trials_per_benchmark <= 0:
                raise ValueError("Must have at least 1 trial per benchmark")
            
            if self.n_seeds <= 0:
                raise ValueError("Must have at least 1 seed")
            
            # Validate that we get expected number of scenarios
            scenarios = self.generate_all_scenarios()
            
            # Calculate expected count dynamically based on actual forced combinations
            forced_combinations = get_default_forced_combinations()
            combinations_per_algorithm = len([c for c in forced_combinations if c['processing_method'] == self.algorithms[0]])
            expected_count = len(self.algorithms) * len(self.datasets) * combinations_per_algorithm
            
            if len(scenarios) != expected_count:
                raise ValueError(f"Expected {expected_count} scenarios, got {len(scenarios)}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get_total_benchmark_count(self) -> int:
        """Get total number of individual benchmarks that will be run."""
        forced_combinations = get_default_forced_combinations()
        combinations_per_algorithm = len([c for c in forced_combinations if c['processing_method'] == self.algorithms[0]])
        return len(self.algorithms) * len(self.datasets) * combinations_per_algorithm
    
    def get_total_trials_count(self) -> int:
        """Get total number of Optuna trials across all benchmarks and seeds."""
        return self.get_total_benchmark_count() * self.n_seeds * self.n_trials_per_benchmark
    
    def get_storage_url_for_gpu(self, gpu_id: int) -> str:
        """Get database storage URL for a specific GPU."""
        if self.use_postgresql:
            return f"{self.postgresql_url}_gpu{gpu_id}"
        else:
            return f"sqlite:///floatsom_optuna_phase3_gpu{gpu_id}.db"
    
    def get_estimated_runtime_hours(self, trials_per_hour: int = 50) -> float:
        """Estimate total runtime in hours based on trials per hour."""
        total_trials = self.get_total_trials_count()
        # Account for parallel execution across GPUs
        if self.n_gpus is not None:
            effective_trials = total_trials / min(self.n_gpus, self.max_concurrent_benchmarks)
        else:
            effective_trials = total_trials / self.max_concurrent_benchmarks
        return effective_trials / trials_per_hour


def create_phase3_development_config(
    n_gpus: Optional[int] = None,
    output_dir: Optional[str] = None,
    n_trials_per_benchmark: Optional[int] = None,
    n_seeds: Optional[int] = None
) -> Phase3BenchmarkConfig:
    """Create configuration for Phase 3 development testing with reduced scope."""
    config_params = {
        'algorithms': ['batch'],
        'datasets': ['swiss_roll', 'moons', 'blobs'],  # Test 3 datasets
    }
    
    if n_gpus is not None:
        config_params['n_gpus'] = n_gpus
    else:
        # Auto-detect number of GPUs using CuPy
        try:
            import cupy as cp
            config_params['n_gpus'] = cp.cuda.runtime.getDeviceCount()
        except:
            config_params['n_gpus'] = 0  # No GPUs detected
    
    if output_dir is not None:
        config_params['output_dir'] = output_dir
    else:
        config_params['output_dir'] = "./phase3_dev_results"
    
    if n_trials_per_benchmark is not None:
        config_params['n_trials_per_benchmark'] = n_trials_per_benchmark
    else:
        config_params['n_trials_per_benchmark'] = 50  # Reduced trials for faster testing
    
    if n_seeds is not None:
        config_params['n_seeds'] = n_seeds
    else:
        config_params['n_seeds'] = 2  # Reduced seeds for faster testing
    
    return Phase3BenchmarkConfig(**config_params)


def create_phase3_full_config(
    n_gpus: Optional[int] = None,
    output_dir: Optional[str] = None,
    n_trials_per_benchmark: Optional[int] = None,
    n_seeds: Optional[int] = None
) -> Phase3BenchmarkConfig:
    """Create configuration for full Phase 3 production benchmarking."""
    config_params = {}
    
    if n_gpus is not None:
        config_params['n_gpus'] = n_gpus
    else:
        # Auto-detect number of GPUs using CuPy
        try:
            import cupy as cp
            config_params['n_gpus'] = cp.cuda.runtime.getDeviceCount()
        except:
            config_params['n_gpus'] = 0  # No GPUs detected
    
    if output_dir is not None:
        config_params['output_dir'] = output_dir
    else:
        config_params['output_dir'] = "./phase3_full_results"
    
    if n_trials_per_benchmark is not None:
        config_params['n_trials_per_benchmark'] = n_trials_per_benchmark
    
    if n_seeds is not None:
        config_params['n_seeds'] = n_seeds
    
    return Phase3BenchmarkConfig(**config_params)


def create_phase3_production_config() -> Phase3BenchmarkConfig:
    """Create configuration for full Phase 3 production benchmarking."""
    return Phase3BenchmarkConfig(
        # Uses defaults: batch-only algorithms, 16 datasets, 5 seeds, 200 trials
        output_dir="./phase3_production_results"
    )


def create_quick_test_config() -> Phase1BenchmarkConfig:
    """Create configuration for quick testing."""
    return Phase1BenchmarkConfig(
        datasets=['swiss_roll'],
        n_trials_per_benchmark=10,
        use_gpu=True,
        gpu_id=0,
        timeout_per_benchmark=180  # 3 minutes
    )


def create_development_config() -> Phase1BenchmarkConfig:
    """Create configuration for development testing."""
    return Phase1BenchmarkConfig(
        datasets=['swiss_roll', 'moons', 'blobs', 'diabetes'],
        n_trials_per_benchmark=50,
        use_gpu=True,
        gpu_id=0,
        timeout_per_benchmark=600  # 10 minutes
    )


def create_full_phase1_config() -> Phase1BenchmarkConfig:
    """Create configuration for full Phase 1 benchmarking."""
    return Phase1BenchmarkConfig(
        datasets=[
            'swiss_roll',
            'moons',
            'circles',
            'blobs',
            's_curve',
            'breast_cancer',
            'wine',
            'iris',
            'digits',
            'olivetti_faces',
            'diabetes',
            'california_housing',
            'covertype',
            'kddcup99'
        ],
        n_trials_per_benchmark=100,
        use_gpu=True,
        gpu_id=0,
        timeout_per_benchmark=1800  # 30 minutes
    )
