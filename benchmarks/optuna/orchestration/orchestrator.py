"""
Core orchestration engine for Phase 3 benchmarking.

This module implements the BenchmarkOrchestrator that coordinates execution
of all benchmark scenarios across single GPU configuration.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import optuna
from pathlib import Path

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..config.benchmark_config import Phase3BenchmarkConfig, ExtendedBenchmarkScenario
from ..core.single_benchmark import run_single_benchmark


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution."""
    scenario_name: str
    seed: Any  # Can be int (single seed) or List[int] (multiple seeds)
    status: str  # 'completed', 'failed', 'timeout'
    harmonized_front: List[optuna.trial.FrozenTrial] = field(default_factory=list)
    n_trials: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    study_name: Optional[str] = None


@dataclass
class ExecutionProgress:
    """Track execution progress across all scenarios."""
    total_scenarios: int
    completed_scenarios: int = 0
    failed_scenarios: int = 0
    total_seeds: int = 0
    completed_seeds: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_seeds == 0:
            return 0.0
        return (self.completed_seeds / self.total_seeds) * 100.0
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        if self.start_time is None:
            return timedelta(0)
        return datetime.now() - self.start_time


class BenchmarkScheduler:
    """Manages the execution queue and scheduling of benchmark scenarios."""
    
    def __init__(self, config: Phase3BenchmarkConfig):
        self.config = config
        self.pending_tasks: List[Tuple[ExtendedBenchmarkScenario, int]] = []
        self.running_tasks: List[Tuple[ExtendedBenchmarkScenario, int]] = []
        self.completed_tasks: List[BenchmarkResult] = []
        self.failed_tasks: List[BenchmarkResult] = []
        
    def initialize_queue(self, scenarios: List[ExtendedBenchmarkScenario]) -> None:
        """Initialize the execution queue with all scenario-seed combinations."""
        self.pending_tasks.clear()
        
        for scenario in scenarios:
            for seed in range(scenario.n_seeds):
                self.pending_tasks.append((scenario, seed))
        
        logging.info(f"Initialized queue with {len(self.pending_tasks)} tasks")
    
    def get_next_task(self) -> Optional[Tuple[ExtendedBenchmarkScenario, int]]:
        """Get the next task to execute."""
        if not self.pending_tasks:
            return None
        
        task = self.pending_tasks.pop(0)
        self.running_tasks.append(task)
        return task
    
    def mark_completed(self, result: BenchmarkResult) -> None:
        """Mark a task as completed."""
        # Find and remove from running tasks
        for i, (scenario, seed) in enumerate(self.running_tasks):
            if scenario.name == result.scenario_name and seed == result.seed:
                self.running_tasks.pop(i)
                break
        
        if result.status == 'completed':
            self.completed_tasks.append(result)
        else:
            self.failed_tasks.append(result)
    
    def get_progress(self) -> Tuple[int, int, int]:
        """Get progress: (completed, running, pending)."""
        return len(self.completed_tasks), len(self.running_tasks), len(self.pending_tasks)


class ProgressTracker:
    """Tracks and persists execution progress with checkpointing."""
    
    def __init__(self, config: Phase3BenchmarkConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.output_dir / "execution_progress.json"
        self.results_file = self.output_dir / "benchmark_results.json"
        
        self.progress = ExecutionProgress(total_scenarios=0)
        self.all_results: List[BenchmarkResult] = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / "orchestration.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_progress(self, scenarios: List[ExtendedBenchmarkScenario]) -> None:
        """Initialize progress tracking."""
        total_seeds = sum(scenario.n_seeds for scenario in scenarios)
        
        self.progress = ExecutionProgress(
            total_scenarios=len(scenarios),
            total_seeds=total_seeds,
            start_time=datetime.now()
        )
        
        self.save_progress()
        self.logger.info(f"Initialized progress tracking for {len(scenarios)} scenarios, {total_seeds} total executions")
    
    def update_progress(self, result: BenchmarkResult) -> None:
        """Update progress with completed result."""
        self.all_results.append(result)
        
        if result.status == 'completed':
            self.progress.completed_seeds += 1
        else:
            self.progress.failed_scenarios += 1
        
        # Update estimated completion time
        if self.progress.completed_seeds > 0:
            elapsed = self.progress.elapsed_time
            avg_time_per_seed = elapsed.total_seconds() / self.progress.completed_seeds
            remaining_seeds = self.progress.total_seeds - self.progress.completed_seeds
            remaining_seconds = remaining_seeds * avg_time_per_seed
            self.progress.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
        
        # Save progress periodically
        if self.progress.completed_seeds % self.config.checkpoint_interval == 0:
            self.save_progress()
            self.save_results()
        
        self.log_progress()
    
    def save_progress(self) -> None:
        """Save progress to file."""
        import json
        
        progress_data = {
            'total_scenarios': self.progress.total_scenarios,
            'completed_scenarios': self.progress.completed_scenarios,
            'failed_scenarios': self.progress.failed_scenarios,
            'total_seeds': self.progress.total_seeds,
            'completed_seeds': self.progress.completed_seeds,
            'start_time': self.progress.start_time.isoformat() if self.progress.start_time else None,
            'estimated_completion': self.progress.estimated_completion.isoformat() if self.progress.estimated_completion else None,
            'completion_percentage': self.progress.completion_percentage
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def save_results(self) -> None:
        """Save all results to file."""
        import json
        
        results_data = []
        for result in self.all_results:
            results_data.append({
                'scenario_name': result.scenario_name,
                'seed': result.seed,
                'status': result.status,
                'n_trials': result.n_trials,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'study_name': result.study_name
            })
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def log_progress(self) -> None:
        """Log current progress."""
        self.logger.info(
            f"Progress: {self.progress.completed_seeds}/{self.progress.total_seeds} "
            f"({self.progress.completion_percentage:.1f}%) - "
            f"Elapsed: {self.progress.elapsed_time} - "
            f"ETA: {self.progress.estimated_completion.strftime('%Y-%m-%d %H:%M:%S') if self.progress.estimated_completion else 'Unknown'}"
        )


class BenchmarkOrchestrator:
    """
    Main orchestrator for Phase 3 benchmark execution.
    
    Coordinates execution of all benchmark scenarios with proper scheduling,
    progress tracking, and checkpointing for single GPU configuration.
    """
    
    def __init__(self, config: Phase3BenchmarkConfig, objectives: Optional[List[str]] = None, 
                 max_gpus: Optional[int] = None, max_concurrent: Optional[int] = None):
        """
        Initialize the benchmark orchestrator.
        
        Args:
            config: Phase 3 benchmark configuration
            objectives: List of objective metrics to optimize
            max_gpus: Maximum number of GPUs to use (None = use all available)
            max_concurrent: Maximum concurrent trials for Ray Tune (None = auto-detect)
        """
        self.config = config
        self.objectives = objectives or ['quantization_error']
        self.max_gpus = max_gpus
        self.max_concurrent = max_concurrent
        
        # Initialize components
        self.progress_tracker = ProgressTracker(config, config.output_dir)
        
        self.logger = logging.getLogger(__name__)
    
    def execute_all_benchmarks(self, use_ray: bool = True) -> Dict[str, Any]:
        """
        Execute all benchmark scenarios.
        
        Args:
            use_ray: Whether to use Ray for parallel scenario execution (default: True)
        
        Returns:
            Dictionary with execution summary and results
        """
        self.logger.info("Starting Phase 3 benchmark orchestration")
        
        # Generate all scenarios
        scenarios = self.config.generate_all_scenarios()
        self.logger.info(f"Generated {len(scenarios)} benchmark scenarios")
        
        # Initialize progress tracker
        self.progress_tracker.initialize_progress(scenarios)
        
        # Execute scenarios
        if use_ray and RAY_AVAILABLE:
            self.logger.info("Using Ray for parallel scenario execution")
            results = self._execute_scenarios_parallel(scenarios)
        else:
            if use_ray and not RAY_AVAILABLE:
                self.logger.info("Ray requested but not available, falling back to sequential execution")
            self.logger.info("Using sequential scenario execution")
            results = self._execute_scenarios_serial(scenarios)
        
        # Save final results
        self.progress_tracker.save_progress()
        self.progress_tracker.save_results()
        
        # Generate summary
        summary = self._generate_execution_summary(results)
        self.logger.info("Phase 3 benchmark orchestration completed")
        
        return summary
    
    def _execute_scenarios_serial(self, scenarios: List[ExtendedBenchmarkScenario]) -> List[BenchmarkResult]:
        """Execute scenarios serially."""
        results = []
        for i, scenario in enumerate(scenarios):
            self.logger.info(f"Executing scenario {i+1}/{len(scenarios)}: {scenario.name}")
            
            # Execute scenario directly with single seed
            result = self._execute_single_scenario(scenario)
            
            # Update tracking
            self.progress_tracker.update_progress(result)
            results.append(result)
        
        return results
    
    def _execute_scenarios_parallel(self, scenarios: List[ExtendedBenchmarkScenario]) -> List[BenchmarkResult]:
        """Execute scenarios in parallel using Ray."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            try:
                # Try to connect to existing cluster first
                ray.init(address='auto', ignore_reinit_error=True)
                self.logger.info("Connected to existing Ray cluster")
            except:
                # Fall back to local Ray
                ray.init(ignore_reinit_error=True)
                self.logger.info("Started local Ray instance")
        
        # Get available resources
        resources = ray.available_resources()
        available_gpus = int(resources.get('GPU', 0))
        available_cpus = int(resources.get('CPU', 0))
        
        self.logger.info(f"Ray cluster resources: {available_gpus} GPUs, {available_cpus} CPUs")
        
        if available_gpus == 0:
            self.logger.warning("No GPUs detected by Ray, falling back to sequential execution")
            return self._execute_scenarios_serial(scenarios)
        
        self.logger.info(f"Will run scenarios in parallel across {available_gpus} GPUs")
        
        # Create Ray remote function for scenario execution
        @ray.remote(num_gpus=1)
        def execute_scenario_remote(scenario, objectives, config_output_dir, scenario_index, total_scenarios, seed, max_concurrent):
            """Execute a single scenario."""
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"[Ray Task] Executing scenario {scenario_index+1}/{total_scenarios}: {scenario.name} with seed {seed}")
            
            start_time = time.time()
            
            try:
                # Create scenario-specific output directory
                scenario_output_dir = None
                if config_output_dir:
                    scenario_output_dir = os.path.join(config_output_dir, scenario.name)
                    os.makedirs(scenario_output_dir, exist_ok=True)
                    logger.info(f"[Ray Task] Scenario output directory: {scenario_output_dir}")
                
                # Run single benchmark directly
                study = run_single_benchmark(
                    algo_type=scenario.algorithm,
                    dataset_name=scenario.dataset.name,
                    dataset_config=scenario.dataset.to_dict(),
                    forced_params=scenario.forced_params,
                    seed=seed,
                    n_trials=scenario.n_trials,
                    objectives=objectives,
                    max_concurrent=max_concurrent
                )
                
                execution_time = time.time() - start_time
                
                # Log scenario completion
                if scenario_output_dir:
                    logger.info(f"[Ray Task] Scenario {scenario.name} completed. Reports saved to: {scenario_output_dir}")
                else:
                    logger.info(f"[Ray Task] Scenario {scenario.name} completed (no output directory specified)")
                
                # Extract best trials for the result
                best_trials = study.best_trials if hasattr(study, 'best_trials') else []
                
                return BenchmarkResult(
                    scenario_name=scenario.name,
                    seed=seed,
                    status='completed',
                    harmonized_front=best_trials,
                    n_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    execution_time=execution_time,
                    study_name=study.study_name if hasattr(study, 'study_name') else scenario.name
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[Ray Task] Scenario {scenario.name} failed: {str(e)}")
                
                return BenchmarkResult(
                    scenario_name=scenario.name,
                    seed=seed,
                    status='failed',
                    execution_time=execution_time,
                    error_message=str(e)
                )
        
        # Submit all scenarios as Ray tasks
        self.logger.info(f"Submitting {len(scenarios)} scenarios to Ray for parallel execution")
        futures = []
        seed = self._get_seed()  # Get seed for this run
        for i, scenario in enumerate(scenarios):
            future = execute_scenario_remote.remote(
                scenario, 
                self.objectives,
                self.config.output_dir,
                i,
                len(scenarios),
                seed,
                self.max_concurrent
            )
            futures.append(future)
        
        # Process results as they complete
        results = []
        self.logger.info(f"Waiting for {len(scenarios)} scenario tasks to complete...")
        
        # Use ray.wait to process results as they complete
        pending = futures
        while pending:
            # Wait for at least one task to complete
            ready, pending = ray.wait(pending, num_returns=1)
            
            # Get the completed result
            result = ray.get(ready[0])
            results.append(result)
            
            # Update progress tracker
            self.progress_tracker.update_progress(result)
            
            self.logger.info(f"Completed {len(results)}/{len(scenarios)} scenarios")
        
        return results
    
    def _execute_single_scenario(self, scenario: ExtendedBenchmarkScenario) -> BenchmarkResult:
        """Execute a single benchmark scenario."""
        start_time = time.time()
        
        try:
            # Create scenario-specific output directory
            scenario_output_dir = None
            if self.config.output_dir:
                scenario_output_dir = os.path.join(self.config.output_dir, scenario.name)
                os.makedirs(scenario_output_dir, exist_ok=True)
                self.logger.info(f"Scenario output directory: {scenario_output_dir}")
            
            # Get seed for this run
            seed = self._get_seed()
            
            # Run single benchmark directly
            study = run_single_benchmark(
                algo_type=scenario.algorithm,
                dataset_name=scenario.dataset.name,
                dataset_config=scenario.dataset.to_dict(),
                forced_params=scenario.forced_params,
                seed=seed,
                n_trials=scenario.n_trials,
                objectives=self.objectives,
                max_concurrent=self.max_concurrent
            )
            
            execution_time = time.time() - start_time
            
            # Log scenario completion with report location
            if scenario_output_dir:
                self.logger.info(f"Scenario {scenario.name} completed. Reports saved to: {scenario_output_dir}")
            else:
                self.logger.info(f"Scenario {scenario.name} completed (no output directory specified)")
            
            # Extract best trials for the result
            best_trials = study.best_trials if hasattr(study, 'best_trials') else []
            
            return BenchmarkResult(
                scenario_name=scenario.name,
                seed=seed,
                status='completed',
                harmonized_front=best_trials,
                n_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                execution_time=execution_time,
                study_name=study.study_name if hasattr(study, 'study_name') else scenario.name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Scenario {scenario.name} failed: {str(e)}")
            
            return BenchmarkResult(
                scenario_name=scenario.name,
                seed=None,
                status='failed',
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_seed(self) -> int:
        """Get seed for the current run from environment or use default."""
        import os
        # Try to get seed from environment variable (useful for batch jobs)
        seed_str = os.environ.get('OPTUNA_SEED', '42')
        try:
            return int(seed_str)
        except ValueError:
            self.logger.warning(f"Invalid seed value '{seed_str}', using default 42")
            return 42
    
    def _generate_execution_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate execution summary."""
        completed_results = [r for r in results if r.status == 'completed']
        failed_results = [r for r in results if r.status == 'failed']
        
        total_execution_time = sum(r.execution_time for r in results)
        avg_execution_time = total_execution_time / len(results) if results else 0
        
        summary = {
            'total_executions': len(results),
            'completed_executions': len(completed_results),
            'failed_executions': len(failed_results),
            'success_rate': len(completed_results) / len(results) if results else 0,
            'total_execution_time_hours': total_execution_time / 3600,
            'average_execution_time_minutes': avg_execution_time / 60,
            'total_trials': sum(r.n_trials for r in completed_results),
            'scenarios_completed': len(set(r.scenario_name for r in completed_results)),
        }
        
        if completed_results:
            total_front_size = sum(len(r.harmonized_front) for r in completed_results)
            summary['total_pareto_solutions'] = total_front_size
            summary['average_front_size'] = total_front_size / len(completed_results) if completed_results else 0
        
        return summary
    
    def resume_execution(self) -> Dict[str, Any]:
        """Resume execution from checkpoint (placeholder for future implementation)."""
        self.logger.info("Resume functionality not yet implemented")
        return self.execute_all_benchmarks()
