#!/usr/bin/env python3
"""
Result handling and statistics for GPU scaling benchmarks.

This module handles result storage, statistics calculation, and data persistence.
Single Responsibility: Manage benchmark results and statistics.
"""

import os
import csv
import json
import numpy as np
from typing import Dict


class ResultsHandler:
    """Handle benchmark results storage and statistics."""

    @staticmethod
    def _set_actual_iteration_ticks(ax, iterations: list[float]) -> None:
        """Use exact sampled iteration counts on a linear x-axis."""
        if not iterations:
            return

        unique_sorted = sorted(
            {
                float(value)
                for value in iterations
                if value is not None and np.isfinite(float(value))
            }
        )
        if not unique_sorted:
            return

        tick_positions = [
            int(value) if float(value).is_integer() else float(value)
            for value in unique_sorted
        ]
        tick_labels = [
            str(int(value)) if float(value).is_integer() else f"{value:g}"
            for value in unique_sorted
        ]

        ax.set_xscale('linear')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        if len(unique_sorted) > 1:
            min_iter = float(min(unique_sorted))
            max_iter = float(max(unique_sorted))
            pad = max(1.0, (max_iter - min_iter) * 0.04)
            ax.set_xlim(min_iter - pad, max_iter + pad)
        else:
            center = float(unique_sorted[0])
            ax.set_xlim(center - 1.0, center + 1.0)
    
    @staticmethod
    def save_results(
        results: Dict,
        output_dir: str,
        mode: str = 'dimension',
        method_suffix: str = '',
        merge_existing: bool = False,
    ):
        """
        Save results to CSV and JSON files.
        
        Args:
            results: Dictionary with results including statistics
            output_dir: Directory to save results
            mode: 'dimension', 'sample', or 'grid_size' to determine labeling
            method_suffix: Optional suffix for file names (e.g., '_batch' or '_colors')
            merge_existing: If True, merge new results into existing JSON (replacing collisions)
        """
        # Determine labels based on mode
        x_label, prefix = ResultsHandler._get_mode_labels(mode)

        if merge_existing:
            existing = ResultsHandler._load_existing_results(output_dir, prefix, method_suffix)
            if existing:
                merged = ResultsHandler._merge_results(existing, results)
                results.clear()
                results.update(merged)
        
        # Save as CSV with statistics
        ResultsHandler._save_csv(results, output_dir, prefix, method_suffix, x_label)
        
        # Save as JSON for easy loading
        ResultsHandler._save_json(results, output_dir, prefix, method_suffix)
        
        # Calculate and save speedup analysis
        ResultsHandler._save_speedup_analysis(results, output_dir, prefix, method_suffix, mode)
    
    @staticmethod
    def _get_mode_labels(mode: str) -> tuple:
        """Get labels and prefix based on mode."""
        if mode == 'dimension':
            return 'Dimension', 'dimension_scaling'
        elif mode == 'sample':
            return 'Sample_Size', 'sample_scaling'
        elif mode == 'grid_size':
            return 'Grid_Size', 'grid_size_scaling'
        else:
            return 'Value', 'scaling'

    @staticmethod
    def _coerce_numeric_key(key):
        """Best-effort conversion of JSON keys to numeric types."""
        if isinstance(key, (int, float)):
            if isinstance(key, float) and key.is_integer():
                return int(key)
            return key
        if isinstance(key, str):
            stripped = key.strip()
            if not stripped:
                return key
            try:
                value = float(stripped)
            except ValueError:
                return key
            if value.is_integer():
                return int(value)
            return value
        return key

    @staticmethod
    def _normalize_stats(stats: Dict) -> Dict:
        """Ensure stats dict contains numeric times/mean/std/count."""
        raw_times = stats.get('times', [])
        times = [float(t) for t in raw_times if t is not None]
        mean = stats.get('mean')
        std = stats.get('std')
        count = stats.get('count')

        if mean is None:
            mean = float(np.mean(times)) if times else 0.0
        else:
            mean = float(mean)

        if std is None:
            std = float(np.std(times)) if times else 0.0
        else:
            std = float(std)

        if count is None:
            count = len(times)
        count = int(count)

        return {
            'times': times,
            'mean': mean,
            'std': std,
            'count': count,
        }

    @staticmethod
    def _load_existing_results(output_dir: str, prefix: str, method_suffix: str) -> Dict:
        """Load existing JSON results, coercing keys to numeric types."""
        json_file = os.path.join(output_dir, f'{prefix}_results{method_suffix}.json')
        if not os.path.exists(json_file):
            return {}

        try:
            with open(json_file, 'r') as f:
                raw = json.load(f)
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        results: Dict = {}
        for axis_key, axis_results in raw.items():
            if not isinstance(axis_results, dict):
                continue
            parsed_axis = ResultsHandler._coerce_numeric_key(axis_key)
            results.setdefault(parsed_axis, {})
            for gpu_key, stats in axis_results.items():
                if not isinstance(stats, dict):
                    continue
                parsed_gpu = ResultsHandler._coerce_numeric_key(gpu_key)
                results[parsed_axis][parsed_gpu] = ResultsHandler._normalize_stats(stats)

        return results

    @staticmethod
    def _merge_results(existing: Dict, new: Dict) -> Dict:
        """Merge result dictionaries, replacing collisions with new values."""
        merged: Dict = {}
        for axis_key, axis_results in existing.items():
            if isinstance(axis_results, dict):
                merged[axis_key] = dict(axis_results)
        for axis_key, axis_results in new.items():
            if not isinstance(axis_results, dict):
                continue
            merged.setdefault(axis_key, {})
            for gpu_key, stats in axis_results.items():
                merged[axis_key][gpu_key] = stats
        return merged
    
    @staticmethod
    def _save_csv(results: Dict, output_dir: str, prefix: str, method_suffix: str, x_label: str):
        """Save results as CSV file."""
        csv_file = os.path.join(output_dir, f'{prefix}_results{method_suffix}.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Get all GPU counts
            gpu_counts = sorted(set(gpu for x_results in results.values() 
                                   for gpu in x_results.keys()))
            
            # Header
            header = [x_label]
            for gpu in gpu_counts:
                header.extend([f'{gpu}_GPUs_Mean', f'{gpu}_GPUs_Std', f'{gpu}_GPUs_Count'])
            writer.writerow(header)
            
            # Data
            for x_val in sorted(results.keys()):
                row = [x_val]
                for gpu in gpu_counts:
                    if gpu in results[x_val]:
                        stats = results[x_val][gpu]
                        row.extend([stats['mean'], stats['std'], stats['count']])
                    else:
                        row.extend(['', '', ''])
                writer.writerow(row)
        
        print(f"Results saved to: {csv_file}")
    
    @staticmethod
    def _save_json(results: Dict, output_dir: str, prefix: str, method_suffix: str):
        """Save results as JSON file."""
        json_file = os.path.join(output_dir, f'{prefix}_results{method_suffix}.json')
        
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for x_val, x_results in results.items():
                json_results[str(x_val)] = {}
                for gpu, stats in x_results.items():
                    json_results[str(x_val)][str(gpu)] = {
                        'times': [float(t) for t in stats['times']],
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'count': int(stats['count'])
                    }
            json.dump(json_results, f, indent=2)
    
    @staticmethod
    def _save_speedup_analysis(results: Dict, output_dir: str, prefix: str, method_suffix: str, mode: str):
        """Calculate and save speedup analysis."""
        speedup_file = os.path.join(output_dir, f'{prefix}_speedup_analysis{method_suffix}.txt')
        
        with open(speedup_file, 'w') as f:
            f.write(f"GPU Scaling Speedup Analysis ({mode.title()} Mode)\n")
            f.write(f"{'='*60}\n\n")
            
            for x_val in sorted(results.keys()):
                if 1 not in results[x_val]:
                    continue
                
                baseline = results[x_val][1]['mean']
                baseline_std = results[x_val][1]['std']
                
                if mode == 'dimension':
                    f.write(f"Dimension {x_val}:\n")
                elif mode == 'sample':
                    f.write(f"Sample Size {x_val:,}:\n")
                elif mode == 'grid_size':
                    f.write(f"Grid Size {x_val}×{x_val}:\n")
                else:
                    f.write(f"Value {x_val}:\n")
                
                f.write(f"  Baseline (1 GPU): {baseline:.2f}s ± {baseline_std:.2f}s\n")
                
                for gpu in sorted(results[x_val].keys()):
                    if gpu == 1:
                        continue
                    stats = results[x_val][gpu]
                    speedup = baseline / stats['mean']
                    efficiency = (speedup / gpu) * 100
                    f.write(f"  {gpu} GPUs: {stats['mean']:.2f}s ± {stats['std']:.2f}s "
                           f"(speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)\n")
                f.write("\n")
        
        print(f"Speedup analysis saved to: {speedup_file}")
    
