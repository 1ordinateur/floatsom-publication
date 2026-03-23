# Optuna-Based Benchmarking System for FloatSOM Algorithms

## Overview

This document outlines a phased implementation plan for a comprehensive benchmarking system using Optuna to optimize hyperparameters across multiple FloatSOM algorithm variants, datasets, and forced parameter combinations. The system will leverage multi-GPU parallelization and generate harmonized Pareto fronts from multiple seed runs.

## Goals

1. **Compare 3 FloatSOM algorithm variants** across hyperparameter spaces
2. **Test on 10 different datasets** to ensure generalization
3. **Evaluate forced parameter combinations** (e.g., HDSSSOM on/off, sampling strategies, topologies)
4. **Generate robust Pareto fronts** by harmonizing results from 5 different random seeds
5. **Maximize computational efficiency** through multi-GPU parallelization

## Architecture Overview

```
Benchmarking System
├── Algorithm Variants (3)
│   ├── Standard FloatSOM
│   ├── Variant A
│   └── Variant B
├── Datasets (10)
│   └── Various sizes and characteristics
├── Forced Parameters
│   ├── HDSSSOM: [on, off]
│   ├── Sampling: [random, full]
│   └── Topology: [mst, hexagonal]
└── Seeds (5 per configuration)
    └── Harmonized Pareto Front
```

## Directory Structure

All Optuna benchmarking code will be organized under:

```
floatsom/benchmarks/optuna/
├── __init__.py
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── parameters.py         # Parameter definitions with types and ranges
│   └── benchmark_config.py   # Benchmark configuration management
├── core/                     # Core execution components
│   ├── __init__.py
│   ├── objective.py          # Core objective function factory
│   ├── gpu_manager.py        # GPU management for benchmarks
│   ├── study_manager.py      # Study configuration and storage
│   └── single_benchmark.py   # Single benchmark execution
├── harmonization/            # Multi-seed result merging
│   ├── __init__.py
│   ├── pareto_utils.py       # Pareto front extraction utilities
│   ├── harmonization.py      # Seed merging strategies
│   └── seed_coordinator.py   # Multi-seed execution coordination
├── orchestration/            # Full benchmark orchestration
│   ├── __init__.py
│   └── orchestrator.py       # Full benchmark orchestration
├── analysis/                 # Results analysis and visualization
│   ├── __init__.py
│   ├── analysis.py           # Results analysis
│   └── visualization.py      # Plotting and visualization
└── utils/                    # Shared utilities
    ├── __init__.py
    └── utils.py              # Helper utilities
```

## Parameter Configuration System

All parameters are defined in a centralized configuration file:

```python
# floatsom/benchmarks/optuna/config/parameters.py

PARAMETER_CONFIGS = {
    # Example continuous parameter
    'learning_rate': {
        'type': 'float',
        'range': (0.1, 3.0),
        'forced': False,
        'description': 'Learning rate for SOM training'
    },
    
    # Example categorical parameter  
    'topology': {
        'type': 'categorical',
        'choices': ['mst', 'hexagonal', 'grid'],
        'forced': True,  # This will be set as forced parameter combination
        'description': 'SOM topology type'
    },
}
```

## Phase 1: Single Optuna Benchmark Infrastructure

### 1.1 Core Objective Function

Create a flexible objective function that can handle all algorithm variants and parameter combinations:

```python
# floatsom/benchmarks/optuna/core/objective.py
def create_objective(algo_type, dataset, forced_params, metrics_config):
    """
    Factory function to create objective functions with specific configurations.
    
    Args:
        algo_type: 'standard', 'variant_a', 'variant_b'
        dataset: Dataset instance
        forced_params: Dict of parameters that are fixed for this run
        metrics_config: Dict specifying which metrics to optimize
    """
    def objective(trial):
        # Merge forced parameters with Optuna suggestions
        params = {**forced_params}
        
        # Add parameters based on configuration
        from floatsom.benchmarks.optuna.config.parameters import PARAMETER_CONFIGS
        
        for param_name, config in PARAMETER_CONFIGS.items():
            # Skip if parameter is forced for this run
            if param_name in forced_params:
                continue
                
            # Add parameter based on type
            if config['type'] == 'float':
                min_val, max_val = config['range']
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config['choices'])
            elif config['type'] == 'int':
                min_val, max_val = config['range']
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
        
        # Train algorithm
        result = train_algorithm(algo_type, dataset, params)
        
        # Calculate metrics
        metrics = calculate_metrics(result, dataset, metrics_config)
        
        # Store all metrics for later analysis
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        
        # Return primary objective (or weighted combination)
        return metrics['primary_objective']
    
    return objective
```

### 1.2 Basic Study Configuration

Set up single-GPU execution for individual benchmarks:

```python
# floatsom/benchmarks/optuna/core/gpu_manager.py
def create_single_gpu_objective(base_objective, gpu_id=0):
    """Wrap objective function to use a specific GPU."""
    def gpu_objective(trial):
        with cp.cuda.Device(gpu_id):
            trial.set_user_attr('gpu_id', gpu_id)
            return base_objective(trial)
    
    return gpu_objective
```

### 1.3 Study Configuration and Storage

Set up basic study configuration for single benchmark execution:

```python
# floatsom/benchmarks/optuna/core/study_manager.py
class StudyManager:
    def __init__(self, storage_url="sqlite:///floatsom_optuna.db"):
        self.storage_url = storage_url
    
    def create_study(self, study_name, directions=['minimize'], sampler_config=None):
        """Create a new study for single benchmark execution."""
        
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=20,
            **(sampler_config or {})
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            directions=directions,
            sampler=sampler,
            load_if_exists=False,
        )
        
        return study
```

### 1.4 Single Benchmark Execution

Complete pipeline for running a single benchmark:

```python
# floatsom/benchmarks/optuna/core/single_benchmark.py
def run_single_benchmark(algo_type, dataset, forced_params, seed, n_trials=200, gpu_id=0):
    """Run a single Optuna benchmark with specific configuration."""
    
    # Create unique study name
    study_name = f"{algo_type}_{dataset.name}_{hash_params(forced_params)}_seed{seed}"
    
    # Set random seed
    set_global_seed(seed)
    
    # Create study
    study_manager = StudyManager()
    study = study_manager.create_study(study_name)
    
    # Create objective
    objective = create_objective(algo_type, dataset, forced_params, metrics_config)
    
    # Wrap with single GPU management
    gpu_objective = create_single_gpu_objective(objective, gpu_id)
    
    # Run optimization (single-threaded for Phase 1)
    study.optimize(
        gpu_objective,
        n_trials=n_trials,
        catch=(Exception,),  # Continue on failures
    )
    
    # Save results
    save_study_results(study, study_name)
    
    return study
```

## Phase 2: Seed Merging and Pareto Front Harmonization

### 2.1 Pareto Front Extraction

Extract Pareto-optimal solutions from each seed:

```python
# floatsom/benchmarks/optuna/harmonization/pareto_utils.py
def extract_pareto_front(study):
    """Extract Pareto-optimal trials from a study."""
    
    # Get all completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Extract objective values
    if len(study.directions) == 1:
        # Single objective - just get best
        return [study.best_trial]
    else:
        # Multi-objective - compute Pareto front
        return compute_pareto_optimal_trials(trials)

def compute_pareto_optimal_trials(trials):
    """Compute Pareto-optimal trials using non-dominated sorting."""
    # Implementation of non-dominated sorting
    # Returns list of Pareto-optimal trials
    pass
```

### 2.2 Harmonization Strategy

Merge Pareto fronts from multiple seeds:

```python
# floatsom/benchmarks/optuna/harmonization/harmonization.py
class ParetoHarmonizer:
    def __init__(self, aggregation_strategy='union'):
        """
        Args:
            aggregation_strategy: 'union', 'average', or 'robust'
        """
        self.strategy = aggregation_strategy
    
    def harmonize_fronts(self, seed_studies):
        """Harmonize Pareto fronts from multiple seed runs."""
        
        # Extract Pareto fronts from each seed
        pareto_fronts = [extract_pareto_front(study) for study in seed_studies]
        
        if self.strategy == 'union':
            # Take union of all Pareto points and recompute dominance
            all_trials = [trial for front in pareto_fronts for trial in front]
            return compute_pareto_optimal_trials(all_trials)
        
        elif self.strategy == 'average':
            # Group similar parameter configurations and average metrics
            return self._average_similar_configs(pareto_fronts)
        
        elif self.strategy == 'robust':
            # Only keep solutions that appear in multiple seeds
            return self._find_robust_solutions(pareto_fronts)
    
    def _average_similar_configs(self, pareto_fronts):
        """Group trials with similar parameters and average their metrics."""
        # Implementation details
        pass
    
    def _find_robust_solutions(self, pareto_fronts, min_appearances=3):
        """Find solutions that appear across multiple seeds."""
        # Implementation details
        pass
```

### 2.3 Seed Coordination

Manage multiple seed runs:

```python
# floatsom/benchmarks/optuna/harmonization/seed_coordinator.py
class SeedCoordinator:
    def __init__(self, n_seeds=5):
        self.n_seeds = n_seeds
    
    def run_seeded_benchmark(self, algo_type, dataset, forced_params, **kwargs):
        """Run benchmark across multiple seeds and harmonize results."""
        
        seed_studies = []
        
        # Run benchmarks for each seed
        for seed in range(self.n_seeds):
            print(f"Running seed {seed}/{self.n_seeds}")
            study = run_single_benchmark(
                algo_type, dataset, forced_params, seed, **kwargs
            )
            seed_studies.append(study)
        
        # Harmonize results
        harmonizer = ParetoHarmonizer(aggregation_strategy='union')
        harmonized_front = harmonizer.harmonize_fronts(seed_studies)
        
        # Save harmonized results
        save_harmonized_results(
            harmonized_front,
            algo_type, dataset, forced_params
        )
        
        return harmonized_front
```

## Phase 3: Full Benchmark Orchestration

### 3.1 Configuration Management

Define all benchmark configurations:

```python
# floatsom/benchmarks/optuna/config/benchmark_config.py
@dataclass
class BenchmarkConfig:
    algorithms = ['standard', 'variant_a', 'variant_b']
    
    datasets = [
        'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5',
        'dataset_6', 'dataset_7', 'dataset_8', 'dataset_9', 'dataset_10'
    ]
    
    forced_param_combinations = [
        {'hdsssom': True, 'sampling': 'random', 'topology': 'mst'},
        {'hdsssom': True, 'sampling': 'random', 'topology': 'hexagonal'},
        {'hdsssom': True, 'sampling': 'full', 'topology': 'mst'},
        {'hdsssom': True, 'sampling': 'full', 'topology': 'hexagonal'},
        {'hdsssom': False, 'sampling': 'random', 'topology': 'mst'},
        {'hdsssom': False, 'sampling': 'random', 'topology': 'hexagonal'},
        {'hdsssom': False, 'sampling': 'full', 'topology': 'mst'},
        {'hdsssom': False, 'sampling': 'full', 'topology': 'hexagonal'},
    ]
    
    n_seeds = 5
    n_trials_per_benchmark = 200
    n_gpus = 4
```

### 3.2 Full Orchestrator

Main entry point for complete benchmark:

```python
# floatsom/benchmarks/optuna/orchestration/orchestrator.py
class BenchmarkOrchestrator:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.seed_coordinator = SeedCoordinator(config.n_seeds)
    
    def run_full_benchmark(self):
        """Execute complete benchmarking suite."""
        
        results = {}
        
        # Total number of benchmarks
        total_benchmarks = (
            len(self.config.algorithms) * 
            len(self.config.datasets) * 
            len(self.config.forced_param_combinations)
        )
        
        benchmark_idx = 0
        
        for algo in self.config.algorithms:
            results[algo] = {}
            
            for dataset_name in self.config.datasets:
                dataset = load_dataset(dataset_name)
                results[algo][dataset_name] = {}
                
                for forced_params in self.config.forced_param_combinations:
                    benchmark_idx += 1
                    print(f"\nBenchmark {benchmark_idx}/{total_benchmarks}")
                    print(f"Algorithm: {algo}")
                    print(f"Dataset: {dataset_name}")
                    print(f"Forced params: {forced_params}")
                    
                    # Run seeded benchmark
                    harmonized_front = self.seed_coordinator.run_seeded_benchmark(
                        algo, dataset, forced_params,
                        n_trials=self.config.n_trials_per_benchmark,
                        gpu_id=benchmark_idx % self.config.n_gpus  # Round-robin GPU assignment for multiple instances
                    )
                    
                    # Store results
                    param_key = self._params_to_key(forced_params)
                    results[algo][dataset_name][param_key] = harmonized_front
        
        # Generate final report
        self.generate_benchmark_report(results)
        
        return results
    
    def _params_to_key(self, params):
        """Convert parameter dict to string key."""
        return "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
```

### 3.3 Results Analysis and Visualization

Tools for analyzing benchmark results:

```python
# floatsom/benchmarks/optuna/analysis/analysis.py
class BenchmarkAnalyzer:
    def __init__(self, results):
        self.results = results
    
    def generate_comparison_plots(self):
        """Generate plots comparing algorithms across datasets."""
        # Implementation for various plot types:
        # - Pareto front comparisons
        # - Parameter importance analysis
        # - Performance across datasets
        # - Stability analysis across seeds
        pass
    
    def export_best_configurations(self):
        """Export best parameter configurations for each scenario."""
        best_configs = {}
        
        for algo in self.results:
            for dataset in self.results[algo]:
                for params in self.results[algo][dataset]:
                    front = self.results[algo][dataset][params]
                    best_trial = self._select_best_from_front(front)
                    
                    key = f"{algo}_{dataset}_{params}"
                    best_configs[key] = best_trial.params
        
        return best_configs
```

## Implementation Timeline

### Week 1: Phase 1 Implementation
- [ ] Create `floatsom/benchmarks/optuna/` directory structure with subdirectories
- [ ] Parameter configuration system (`config/parameters.py`)
- [ ] Core objective function framework (`core/objective.py`)
- [ ] Single GPU management system (`core/gpu_manager.py`)
- [ ] Study configuration (`core/study_manager.py`)
- [ ] Single benchmark execution pipeline (`core/single_benchmark.py`)
- [ ] Basic utilities (`utils/utils.py`)

### Week 2: Phase 2 Implementation
- [ ] Pareto front extraction (`harmonization/pareto_utils.py`)
- [ ] Harmonization strategies (`harmonization/harmonization.py`)
- [ ] Seed coordination system (`harmonization/seed_coordinator.py`)
- [ ] Harmonized result storage

### Week 3: Phase 3 Implementation
- [ ] Configuration management (`config/benchmark_config.py`)
- [ ] Full orchestration system (`orchestration/orchestrator.py`)
- [ ] Progress tracking and resumption
- [ ] Analysis tools (`analysis/analysis.py`)
- [ ] Visualization tools (`analysis/visualization.py`)

### Week 4: Testing and Refinement
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Visualization tools
- [ ] Documentation and examples

## Key Design Decisions

1. **Storage Backend**: SQLite for Phase 1 (single benchmarks), PostgreSQL for Phase 3 (multiple parallel instances)
2. **GPU Assignment**: Single GPU per benchmark in Phase 1, round-robin assignment across benchmarks in Phase 3
3. **Harmonization Default**: Union strategy to preserve diversity
4. **Checkpointing**: Enable resume capability for long-running benchmarks
5. **Metric Flexibility**: Support both single and multi-objective optimization

## Usage Example

```python
# Located in floatsom/benchmarks/optuna/
from floatsom.benchmarks.optuna.config.benchmark_config import BenchmarkConfig
from floatsom.benchmarks.optuna.orchestration.orchestrator import BenchmarkOrchestrator
from floatsom.benchmarks.optuna.analysis.analysis import BenchmarkAnalyzer

# Run complete benchmark
config = BenchmarkConfig()
orchestrator = BenchmarkOrchestrator(config)
results = orchestrator.run_full_benchmark()

# Analyze results
analyzer = BenchmarkAnalyzer(results)
analyzer.generate_comparison_plots()
best_configs = analyzer.export_best_configurations()
```

## Monitoring and Debugging

1. **Optuna Dashboard**: Monitor progress in real-time
   ```bash
   # Phase 1: SQLite storage
   optuna-dashboard sqlite:///floatsom_optuna.db
   
   # Phase 3: PostgreSQL storage (for multiple parallel instances)
   optuna-dashboard postgresql://localhost/floatsom_optuna
   ```

2. **Progress Tracking**: Regular checkpoints and ETA calculations

3. **Error Handling**: Graceful failure recovery and trial retry logic

4. **Resource Monitoring**: GPU utilization and memory tracking

## Future Extensions

1. **Adaptive Sampling**: Adjust n_trials based on convergence
2. **Transfer Learning**: Use results from one dataset to warm-start others
3. **Distributed Execution**: Support for multi-node clusters
4. **AutoML Integration**: Automatic algorithm selection based on dataset characteristics