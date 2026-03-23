"""
Results analysis system for Phase 3 benchmarking.

This module provides comprehensive analysis of benchmark results including
statistical analysis, performance comparisons, and multi-seed harmonization.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict
import scipy.stats as stats
from datetime import datetime


@dataclass
class ScenarioResults:
    """Results for a single scenario across all seeds."""
    scenario_name: str
    algorithm: str
    dataset: str
    forced_params: Dict[str, Any]
    seed_results: List[float] = field(default_factory=list)
    seed_trials: List[int] = field(default_factory=list)
    seed_times: List[float] = field(default_factory=list)
    
    @property
    def mean_performance(self) -> float:
        """Mean performance across seeds."""
        return np.mean(self.seed_results) if self.seed_results else float('inf')
    
    @property
    def std_performance(self) -> float:
        """Standard deviation of performance."""
        return np.std(self.seed_results) if len(self.seed_results) > 1 else 0.0
    
    @property
    def best_performance(self) -> float:
        """Best performance across seeds."""
        return min(self.seed_results) if self.seed_results else float('inf')
    
    @property
    def worst_performance(self) -> float:
        """Worst performance across seeds."""
        return max(self.seed_results) if self.seed_results else float('inf')
    
    @property
    def median_performance(self) -> float:
        """Median performance across seeds."""
        return np.median(self.seed_results) if self.seed_results else float('inf')
    
    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """95% confidence interval for performance."""
        if len(self.seed_results) < 2:
            return (self.mean_performance, self.mean_performance)
        
        confidence = 0.95
        n = len(self.seed_results)
        mean = self.mean_performance
        std_err = self.std_performance / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * std_err
        
        return (mean - margin, mean + margin)
    
    @property
    def stability_score(self) -> float:
        """Stability score (lower std relative to mean is more stable)."""
        if self.mean_performance == 0:
            return float('inf')
        return self.std_performance / abs(self.mean_performance)


@dataclass
class AlgorithmPerformance:
    """Performance summary for an algorithm across all scenarios."""
    algorithm_name: str
    scenario_results: List[ScenarioResults] = field(default_factory=list)
    
    @property
    def overall_mean(self) -> float:
        """Overall mean performance across all scenarios."""
        all_results = []
        for scenario in self.scenario_results:
            all_results.extend(scenario.seed_results)
        return np.mean(all_results) if all_results else float('inf')
    
    @property
    def overall_std(self) -> float:
        """Overall standard deviation."""
        all_results = []
        for scenario in self.scenario_results:
            all_results.extend(scenario.seed_results)
        return np.std(all_results) if len(all_results) > 1 else 0.0
    
    @property
    def wins_count(self) -> int:
        """Number of scenarios where this algorithm performed best."""
        return len([s for s in self.scenario_results if hasattr(s, '_is_winner') and s._is_winner])
    
    @property
    def mean_stability(self) -> float:
        """Mean stability across scenarios."""
        stabilities = [s.stability_score for s in self.scenario_results if s.stability_score != float('inf')]
        return np.mean(stabilities) if stabilities else float('inf')


class ResultsAggregator:
    """Aggregates results from multiple seeds and scenarios."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_results_from_file(self, results_file: Path) -> List[Dict[str, Any]]:
        """Load benchmark results from JSON file."""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            self.logger.info(f"Loaded {len(results)} benchmark results from {results_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results from {results_file}: {e}")
            return []
    
    def aggregate_by_scenario(self, raw_results: List[Dict[str, Any]]) -> Dict[str, ScenarioResults]:
        """Aggregate results by scenario across all seeds."""
        scenario_map = defaultdict(ScenarioResults)
        
        for result in raw_results:
            if result['status'] != 'completed' or result['best_value'] is None:
                continue
            
            scenario_name = result['scenario_name']
            
            # Parse scenario name to extract components
            parts = scenario_name.split('_')
            if len(parts) >= 4:
                algorithm = parts[1]  # scenario_001_colors_swiss_roll_full_grid
                dataset = parts[2]
            else:
                algorithm = "unknown"
                dataset = "unknown"
            
            # Initialize scenario result if not exists
            if scenario_name not in scenario_map:
                scenario_map[scenario_name] = ScenarioResults(
                    scenario_name=scenario_name,
                    algorithm=algorithm,
                    dataset=dataset,
                    forced_params={}  # Will be populated from first result
                )
            
            scenario_result = scenario_map[scenario_name]
            scenario_result.seed_results.append(result['best_value'])
            scenario_result.seed_trials.append(result['n_trials'])
            scenario_result.seed_times.append(result['execution_time'])
        
        self.logger.info(f"Aggregated results for {len(scenario_map)} scenarios")
        return dict(scenario_map)
    
    def group_by_algorithm(self, scenario_results: Dict[str, ScenarioResults]) -> Dict[str, AlgorithmPerformance]:
        """Group scenario results by algorithm."""
        algorithm_map = defaultdict(list)
        
        for scenario_result in scenario_results.values():
            algorithm_map[scenario_result.algorithm].append(scenario_result)
        
        # Create AlgorithmPerformance objects
        algorithm_performance = {}
        for algorithm, scenarios in algorithm_map.items():
            algorithm_performance[algorithm] = AlgorithmPerformance(
                algorithm_name=algorithm,
                scenario_results=scenarios
            )
        
        self.logger.info(f"Grouped results for {len(algorithm_performance)} algorithms")
        return algorithm_performance


class PerformanceComparator:
    """Compares performance across algorithms and scenarios."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def rank_algorithms_overall(self, algorithm_performance: Dict[str, AlgorithmPerformance]) -> List[Tuple[str, float, float]]:
        """Rank algorithms by overall performance."""
        rankings = []
        
        for algo_name, performance in algorithm_performance.items():
            rankings.append((
                algo_name,
                performance.overall_mean,
                performance.overall_std
            ))
        
        # Sort by mean performance (lower is better)
        rankings.sort(key=lambda x: x[1])
        
        self.logger.info(f"Ranked {len(rankings)} algorithms by overall performance")
        return rankings
    
    def find_scenario_winners(self, scenario_results: Dict[str, ScenarioResults]) -> Dict[str, str]:
        """Find the winning algorithm for each scenario."""
        winners = {}
        
        # Group scenarios by dataset and forced params
        scenario_groups = defaultdict(list)
        for scenario_name, result in scenario_results.items():
            group_key = f"{result.dataset}_{hash(str(result.forced_params))}"
            scenario_groups[group_key].append((scenario_name, result))
        
        for group_key, scenarios in scenario_groups.items():
            if len(scenarios) < 2:
                continue
            
            # Find best performing algorithm in this group
            best_performance = float('inf')
            best_algorithm = None
            
            for scenario_name, result in scenarios:
                if result.mean_performance < best_performance:
                    best_performance = result.mean_performance
                    best_algorithm = result.algorithm
            
            # Mark winners
            for scenario_name, result in scenarios:
                winners[scenario_name] = best_algorithm
                result._is_winner = (result.algorithm == best_algorithm)
        
        self.logger.info(f"Identified winners for {len(winners)} scenarios")
        return winners
    
    def statistical_significance_test(self, results1: List[float], results2: List[float]) -> Tuple[float, float]:
        """Perform statistical significance test between two sets of results."""
        if len(results1) < 2 or len(results2) < 2:
            return float('nan'), float('nan')
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(results1, results2, equal_var=False)
        
        return t_stat, p_value
    
    def effect_size_cohens_d(self, results1: List[float], results2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(results1) < 2 or len(results2) < 2:
            return float('nan')
        
        mean1, mean2 = np.mean(results1), np.mean(results2)
        std1, std2 = np.std(results1, ddof=1), np.std(results2, ddof=1)
        n1, n2 = len(results1), len(results2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return float('nan')
        
        return (mean1 - mean2) / pooled_std


class ResultsAnalyzer:
    """
    Main results analysis system for Phase 3 benchmarking.
    
    Provides comprehensive analysis of benchmark results including
    statistical analysis, performance comparisons, and reporting.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        self.aggregator = ResultsAggregator()
        self.comparator = PerformanceComparator()
        
        self.scenario_results: Dict[str, ScenarioResults] = {}
        self.algorithm_performance: Dict[str, AlgorithmPerformance] = {}
        self.scenario_winners: Dict[str, str] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def load_and_analyze_results(self) -> Dict[str, Any]:
        """Load results and perform comprehensive analysis."""
        self.logger.info("Starting comprehensive results analysis")
        
        # Load raw results
        results_file = self.results_dir / "benchmark_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        raw_results = self.aggregator.load_results_from_file(results_file)
        
        if not raw_results:
            raise ValueError("No valid results found to analyze")
        
        # Aggregate by scenario
        self.scenario_results = self.aggregator.aggregate_by_scenario(raw_results)
        
        # Group by algorithm
        self.algorithm_performance = self.aggregator.group_by_algorithm(self.scenario_results)
        
        # Find winners
        self.scenario_winners = self.comparator.find_scenario_winners(self.scenario_results)
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary()
        
        self.logger.info("Results analysis completed")
        return analysis_summary
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        # Overall algorithm rankings
        algorithm_rankings = self.comparator.rank_algorithms_overall(self.algorithm_performance)
        
        # Calculate win rates
        total_scenarios = len(self.scenario_results)
        win_counts = defaultdict(int)
        for winner in self.scenario_winners.values():
            win_counts[winner] += 1
        
        win_rates = {algo: count / total_scenarios for algo, count in win_counts.items()}
        
        # Statistical analysis
        pairwise_comparisons = self._perform_pairwise_comparisons()
        
        # Performance by dataset
        dataset_performance = self._analyze_performance_by_dataset()
        
        # Performance by parameter combinations
        parameter_performance = self._analyze_performance_by_parameters()
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_scenarios': total_scenarios,
            'total_algorithms': len(self.algorithm_performance),
            'algorithm_rankings': [
                {
                    'algorithm': algo,
                    'mean_performance': mean_perf,
                    'std_performance': std_perf,
                    'win_rate': win_rates.get(algo, 0.0),
                    'stability_score': self.algorithm_performance[algo].mean_stability
                }
                for algo, mean_perf, std_perf in algorithm_rankings
            ],
            'pairwise_comparisons': pairwise_comparisons,
            'dataset_performance': dataset_performance,
            'parameter_performance': parameter_performance,
            'best_overall_algorithm': algorithm_rankings[0][0] if algorithm_rankings else None,
            'most_stable_algorithm': self._find_most_stable_algorithm()
        }
        
        return summary
    
    def _perform_pairwise_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """Perform pairwise statistical comparisons between algorithms."""
        algorithms = list(self.algorithm_performance.keys())
        comparisons = {}
        
        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms[i+1:], i+1):
                # Collect all results for both algorithms
                results1 = []
                results2 = []
                
                for scenario in self.algorithm_performance[algo1].scenario_results:
                    results1.extend(scenario.seed_results)
                
                for scenario in self.algorithm_performance[algo2].scenario_results:
                    results2.extend(scenario.seed_results)
                
                # Perform statistical tests
                t_stat, p_value = self.comparator.statistical_significance_test(results1, results2)
                effect_size = self.comparator.effect_size_cohens_d(results1, results2)
                
                comparison_key = f"{algo1}_vs_{algo2}"
                comparisons[comparison_key] = {
                    'algorithm_1': algo1,
                    'algorithm_2': algo2,
                    'mean_1': np.mean(results1) if results1 else float('inf'),
                    'mean_2': np.mean(results2) if results2 else float('inf'),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size_cohens_d': effect_size,
                    'significant_at_0.05': p_value < 0.05 if not np.isnan(p_value) else False,
                    'better_algorithm': algo1 if np.mean(results1) < np.mean(results2) else algo2
                }
        
        return comparisons
    
    def _analyze_performance_by_dataset(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance broken down by dataset."""
        dataset_performance = defaultdict(lambda: defaultdict(list))
        
        for scenario in self.scenario_results.values():
            dataset_performance[scenario.dataset][scenario.algorithm].extend(scenario.seed_results)
        
        # Calculate statistics
        dataset_stats = {}
        for dataset, algorithms in dataset_performance.items():
            dataset_stats[dataset] = {}
            for algorithm, results in algorithms.items():
                dataset_stats[dataset][algorithm] = {
                    'mean': np.mean(results),
                    'std': np.std(results),
                    'median': np.median(results),
                    'best': min(results),
                    'worst': max(results),
                    'count': len(results)
                }
        
        return dict(dataset_stats)
    
    def _analyze_performance_by_parameters(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by forced parameter combinations."""
        param_performance = defaultdict(lambda: defaultdict(list))
        
        for scenario in self.scenario_results.values():
            # Create parameter signature
            param_sig = f"{scenario.forced_params.get('sampling_method', 'unknown')}_{scenario.forced_params.get('topology_type', 'unknown')}"
            param_performance[param_sig][scenario.algorithm].extend(scenario.seed_results)
        
        # Calculate statistics
        param_stats = {}
        for param_combo, algorithms in param_performance.items():
            param_stats[param_combo] = {}
            for algorithm, results in algorithms.items():
                param_stats[param_combo][algorithm] = {
                    'mean': np.mean(results),
                    'std': np.std(results),
                    'count': len(results)
                }
        
        return dict(param_stats)
    
    def _find_most_stable_algorithm(self) -> Optional[str]:
        """Find the most stable algorithm (lowest relative standard deviation)."""
        if not self.algorithm_performance:
            return None
        
        stabilities = [
            (algo, perf.mean_stability)
            for algo, perf in self.algorithm_performance.items()
            if perf.mean_stability != float('inf')
        ]
        
        if not stabilities:
            return None
        
        stabilities.sort(key=lambda x: x[1])
        return stabilities[0][0]
    
    def export_detailed_results(self, output_file: Optional[Path] = None) -> Path:
        """Export detailed results to CSV format."""
        if output_file is None:
            output_file = self.results_dir / "detailed_analysis.csv"
        
        # Prepare data for export
        rows = []
        
        for scenario_name, scenario in self.scenario_results.items():
            for i, (result, trials, exec_time) in enumerate(zip(
                scenario.seed_results, scenario.seed_trials, scenario.seed_times
            )):
                rows.append({
                    'scenario_name': scenario_name,
                    'algorithm': scenario.algorithm,
                    'dataset': scenario.dataset,
                    'sampling_method': scenario.forced_params.get('sampling_method', 'unknown'),
                    'topology_type': scenario.forced_params.get('topology_type', 'unknown'),
                    'seed_index': i,
                    'best_value': result,
                    'n_trials': trials,
                    'execution_time': exec_time,
                    'scenario_mean': scenario.mean_performance,
                    'scenario_std': scenario.std_performance,
                    'is_scenario_winner': self.scenario_winners.get(scenario_name) == scenario.algorithm
                })
        
        # Create DataFrame and export
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Exported detailed results to {output_file}")
        return output_file