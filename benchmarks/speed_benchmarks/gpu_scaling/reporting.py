#!/usr/bin/env python3
"""
Report generation for GPU scaling benchmarks.

This module handles comprehensive report generation and analysis.
Single Responsibility: Generate benchmark reports and summaries.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    @staticmethod
    def create_summary_report(
        dim_results: Optional[Dict],
        sample_results: Optional[Dict],
        grid_results: Optional[Dict],
        args,
        output_dir: str,
    ):
        """
        Create a comprehensive summary report.
        
        Args:
            dim_results: Dimension scaling results
            sample_results: Sample scaling results
            grid_results: Grid size scaling results
            args: Command line arguments
            output_dir: Output directory
        """
        report_file = f"{output_dir}/overall_benchmark_summary.txt"
        
        with open(report_file, 'w') as f:
            # Write header
            ReportGenerator._write_header(f, args)
            
            # Write dimension scaling results
            if dim_results:
                ReportGenerator._write_dimension_results(f, dim_results, args)
            
            # Write sample scaling results
            if sample_results:
                ReportGenerator._write_sample_results(f, sample_results, args)
            
            # Write grid size scaling results
            if grid_results:
                ReportGenerator._write_grid_results(f, grid_results, args)
            
            # Write overall efficiency analysis
            ReportGenerator._write_efficiency_analysis(f, dim_results, sample_results, grid_results, args)
        
        print(f"Summary report saved to: {report_file}")

    @staticmethod
    def _collect_axis_values(results: Dict) -> list:
        """Collect sorted axis values across all topologies/methods."""
        values = set()
        for topology in results:
            for method in results[topology]:
                values.update(results[topology][method].keys())
        return sorted(values)

    @staticmethod
    def _collect_gpu_counts(results: Dict) -> list:
        """Collect sorted GPU counts across all topologies/methods/axis values."""
        gpus = set()
        for topology in results:
            for method in results[topology]:
                for axis_val in results[topology][method]:
                    gpus.update(results[topology][method][axis_val].keys())
        return sorted(gpus)
    
    @staticmethod
    def _write_header(f, args):
        """Write report header."""
        f.write(f"FloatSOM GPU Scaling Benchmark Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Mode: {args.mode}\n")
        f.write(f"  Repeats per configuration: {args.repeats}\n")
        f.write(f"  GPU counts tested: {args.gpu_counts}\n")
        f.write(f"  SOM grid size: {args.grid_size}×{args.grid_size}\n")
        f.write(f"  Iterations: {args.total_iterations}\n")
        f.write(f"  Base seed: {args.seed}\n")
        f.write(f"\n")
    
    @staticmethod
    def _write_dimension_results(f, results: Dict, args):
        """Write dimension scaling results."""
        dimensions = args.dimensions
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            dimensions = ReportGenerator._collect_axis_values(results)
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write(f"DIMENSION SCALING RESULTS:\n")
        f.write(f"{'='*80}\n")
        f.write(f"  Fixed samples: {args.samples:,}\n")
        f.write(f"  Dimensions tested: {dimensions}\n")
        f.write(f"  Topologies tested: {args.topologies}\n\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"Topology: {topology}, Method: {method}\n")
                f.write("-" * 60 + "\n")
                
                # Create formatted table
                f.write(f"{'Dimension':<12}")
                for gpu in gpu_counts:
                    f.write(f"{f'{gpu} GPU(s)':<20}")
                f.write("\n")
                f.write("-" * (12 + 20 * len(gpu_counts)) + "\n")
                
                for dim in sorted(results[topology][method].keys()):
                    f.write(f"{dim:<12}")
                    for gpu in gpu_counts:
                        if gpu in results[topology][method][dim]:
                            stats = results[topology][method][dim][gpu]
                            f.write(f"{stats['mean']:.2f}±{stats['std']:.2f}s".ljust(20))
                        else:
                            f.write(f"{'N/A':<20}")
                    f.write("\n")
                f.write("\n")
    
    @staticmethod
    def _write_sample_results(f, results: Dict, args):
        """Write sample scaling results."""
        sample_sizes = args.sample_sizes
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            sample_sizes = ReportGenerator._collect_axis_values(results)
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write(f"SAMPLE SIZE SCALING RESULTS:\n")
        f.write(f"{'='*80}\n")
        f.write(f"  Fixed dimension: {args.fixed_dimension}\n")
        f.write(f"  Sample sizes tested: {[f'{s:,}' for s in sample_sizes]}\n")
        f.write(f"  Topologies tested: {args.topologies}\n\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"Topology: {topology}, Method: {method}\n")
                f.write("-" * 60 + "\n")
                
                # Create formatted table
                f.write(f"{'Samples':<15}")
                for gpu in gpu_counts:
                    f.write(f"{f'{gpu} GPU(s)':<20}")
                f.write("\n")
                f.write("-" * (15 + 20 * len(gpu_counts)) + "\n")
                
                for samples in sorted(results[topology][method].keys()):
                    f.write(f"{samples:<15,}")
                    for gpu in gpu_counts:
                        if gpu in results[topology][method][samples]:
                            stats = results[topology][method][samples][gpu]
                            f.write(f"{stats['mean']:.2f}±{stats['std']:.2f}s".ljust(20))
                        else:
                            f.write(f"{'N/A':<20}")
                    f.write("\n")
                f.write("\n")
    
    @staticmethod
    def _write_grid_results(f, results: Dict, args):
        """Write grid size scaling results."""
        grid_sizes = args.grid_sizes
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            grid_sizes = ReportGenerator._collect_axis_values(results)
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write(f"GRID SIZE SCALING RESULTS:\n")
        f.write(f"{'='*80}\n")
        f.write(f"  Fixed dimension: {args.fixed_dimension}\n")
        f.write(f"  Fixed samples: {args.samples:,}\n")
        f.write(f"  Grid sizes tested: {grid_sizes}\n")
        f.write(f"  Topologies tested: {args.topologies}\n\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"Topology: {topology}, Method: {method}\n")
                f.write("-" * 60 + "\n")
                
                # Create formatted table
                f.write(f"{'Grid Size':<12}")
                for gpu in gpu_counts:
                    f.write(f"{f'{gpu} GPU(s)':<20}")
                f.write("\n")
                f.write("-" * (12 + 20 * len(gpu_counts)) + "\n")
                
                for grid_size in sorted(results[topology][method].keys()):
                    f.write(f"{grid_size:<12}")
                    for gpu in gpu_counts:
                        if gpu in results[topology][method][grid_size]:
                            stats = results[topology][method][grid_size][gpu]
                            f.write(f"{stats['mean']:.2f}±{stats['std']:.2f}s".ljust(20))
                        else:
                            f.write(f"{'N/A':<20}")
                    f.write("\n")
                f.write("\n")
    
    @staticmethod
    def _write_efficiency_analysis(f, dim_results: Optional[Dict], 
                                  sample_results: Optional[Dict],
                                  grid_results: Optional[Dict], args):
        """Write overall efficiency analysis."""
        f.write(f"{'='*80}\n")
        f.write(f"SCALING EFFICIENCY ANALYSIS:\n\n")
        
        if dim_results:
            ReportGenerator._write_dimension_efficiency(f, dim_results, args)
        
        if sample_results:
            ReportGenerator._write_sample_efficiency(f, sample_results, args)
        
        if grid_results:
            ReportGenerator._write_grid_efficiency(f, grid_results, args)
    
    @staticmethod
    def _write_dimension_efficiency(f, results: Dict, args):
        """Write dimension scaling efficiency analysis."""
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write("Dimension Scaling Efficiency:\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"  {topology} topology, {method} method:\n")
                
                for gpu in gpu_counts:
                    if gpu == 1:
                        continue
                    
                    efficiencies = []
                    for dim in results[topology][method].keys():
                        if 1 in results[topology][method][dim] and gpu in results[topology][method][dim]:
                            baseline = results[topology][method][dim][1]['mean']
                            gpu_time = results[topology][method][dim][gpu]['mean']
                            speedup = baseline / gpu_time
                            efficiency = (speedup / gpu) * 100
                            efficiencies.append(efficiency)
                    
                    if efficiencies:
                        avg_efficiency = np.mean(efficiencies)
                        std_efficiency = np.std(efficiencies)
                        f.write(f"    {gpu} GPUs: {avg_efficiency:.1f}% ± {std_efficiency:.1f}% average efficiency\n")
        
        f.write("\n")
    
    @staticmethod
    def _write_sample_efficiency(f, results: Dict, args):
        """Write sample scaling efficiency analysis."""
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write("Sample Size Scaling Efficiency:\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"  {topology} topology, {method} method:\n")
                
                for gpu in gpu_counts:
                    if gpu == 1:
                        continue
                    
                    efficiencies = []
                    for samples in results[topology][method].keys():
                        if 1 in results[topology][method][samples] and gpu in results[topology][method][samples]:
                            baseline = results[topology][method][samples][1]['mean']
                            gpu_time = results[topology][method][samples][gpu]['mean']
                            speedup = baseline / gpu_time
                            efficiency = (speedup / gpu) * 100
                            efficiencies.append(efficiency)
                    
                    if efficiencies:
                        avg_efficiency = np.mean(efficiencies)
                        std_efficiency = np.std(efficiencies)
                        f.write(f"    {gpu} GPUs: {avg_efficiency:.1f}% ± {std_efficiency:.1f}% average efficiency\n")
        
        f.write("\n")
    
    @staticmethod
    def _write_grid_efficiency(f, results: Dict, args):
        """Write grid size scaling efficiency analysis."""
        gpu_counts = args.gpu_counts
        if getattr(args, "merge_existing", False):
            gpu_counts = ReportGenerator._collect_gpu_counts(results)

        f.write("Grid Size Scaling Efficiency:\n")
        
        for topology in results:
            for method in results[topology]:
                f.write(f"  {topology} topology, {method} method:\n")
                
                for gpu in gpu_counts:
                    if gpu == 1:
                        continue
                    
                    efficiencies = []
                    for grid_size in results[topology][method].keys():
                        if 1 in results[topology][method][grid_size] and gpu in results[topology][method][grid_size]:
                            baseline = results[topology][method][grid_size][1]['mean']
                            gpu_time = results[topology][method][grid_size][gpu]['mean']
                            speedup = baseline / gpu_time
                            efficiency = (speedup / gpu) * 100
                            efficiencies.append(efficiency)
                    
                    if efficiencies:
                        avg_efficiency = np.mean(efficiencies)
                        std_efficiency = np.std(efficiencies)
                        f.write(f"    {gpu} GPUs: {avg_efficiency:.1f}% ± {std_efficiency:.1f}% average efficiency\n")
    
    @staticmethod
    def print_final_summary(args, output_dir: str):
        """Print final summary with file locations."""
        print(f"\n{'='*60}")
        print(f"Benchmark complete!")
        print(f"Results saved to: {output_dir}")
        
        if args.mode in ['dimension_scaling', 'both']:
            print(f"\nDimension scaling results:")
            print(f"  - dimension_scaling/dimension_scaling_results_*.csv: Raw timing data per topology/method")
            print(f"  - dimension_scaling/dimension_scaling_results_*.json: JSON format")
            print(f"  - dimension_scaling/dimension_scaling_performance.svg: Performance graph")
            print(f"  - dimension_scaling/dimension_scaling_speedup.svg: Speedup graph")
            print(f"  - dimension_scaling/dimension_scaling_speedup_analysis_*.txt: Detailed analysis")
        
        if args.mode in ['sample_scaling', 'both']:
            print(f"\nSample size scaling results:")
            print(f"  - sample_scaling/sample_scaling_results_*.csv: Raw timing data per topology/method")
            print(f"  - sample_scaling/sample_scaling_results_*.json: JSON format")
            print(f"  - sample_scaling/sample_scaling_performance.svg: Performance graph")
            print(f"  - sample_scaling/sample_scaling_speedup.svg: Speedup graph")
            print(f"  - sample_scaling/sample_scaling_speedup_analysis_*.txt: Detailed analysis")
        
        if args.mode == 'grid_size_scaling':
            print(f"\nGrid size scaling results:")
            print(f"  - grid_size_scaling/grid_size_scaling_results_*.csv: Raw timing data per topology/method")
            print(f"  - grid_size_scaling/grid_size_scaling_results_*.json: JSON format")
            print(f"  - grid_size_scaling/grid_size_scaling_performance.svg: Performance graph")
            print(f"  - grid_size_scaling/grid_size_scaling_speedup.svg: Speedup graph")
            print(f"  - grid_size_scaling/grid_size_scaling_speedup_analysis_*.txt: Detailed analysis")
        
        print(f"\n  - overall_benchmark_summary.txt: Comprehensive report")
