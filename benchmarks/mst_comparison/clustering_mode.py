"""
Clustering evaluation mode for MST comparison benchmarks.

This module evaluates how well different SOM topologies preserve cluster structure
by applying agglomerative clustering to SOM nodes and comparing to ground truth labels.

Adjustments:
- Many-to-one accuracy implemented via contingency_matrix (robust to label domains).
- Added diagnostics: homogeneity, completeness, V-measure, cluster purity, class coverage.
- Added B-cubed Precision/Recall/F1 (robust when K_pred != K_true).
- Persist diagnostics in per-run JSON and final CSV/report.
- NEW: Line plots for ALL metrics (core + diagnostics) across offsets (ΔK) and across centers (K at ΔK=0).
"""

import os
import numpy as np
import cupy as cp
import pandas as pd
import logging
from typing import Dict, Union, List, Tuple, DefaultDict
from collections import defaultdict

# Safe headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from floatsom.base.floatsom import FloatSOM
# Use the benchmarks metrics implementation which does not rely on a missing Metric protocol
from floatsom.benchmarks.evaluation.metrics import QuantizationError

logger = logging.getLogger(__name__)

# Import clustering libraries
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_score,
        completeness_score,
        v_measure_score,
    )
    from sklearn.metrics.cluster import contingency_matrix
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logger.warning("sklearn clustering libraries not available")


# ----------------------------
# Metrics helpers (new/updated)
# ----------------------------

def many_to_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Many-to-one clustering accuracy. Each predicted cluster is mapped to the
    single true class most frequent within it. Multiple predicted clusters
    may map to the same true class. Robust to label domains and K mismatch.
    """
    C = contingency_matrix(y_true, y_pred)  # rows=true classes, cols=pred clusters
    total = C.sum()
    if total == 0:
        return 0.0
    correct = C.max(axis=0).sum()
    return float(correct) / float(total)


def clustering_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute merge/split fingerprints and purity-like summaries that remain meaningful
    when K_pred != K_true.
    - homogeneity: low => clusters mix many true classes (mixing).
    - completeness: low => true classes are fragmented across clusters (splitting).
    - v_measure: harmonic mean of the two.
    - cluster_purity: many-to-one-style purity over predicted clusters.
    - class_coverage: many-to-one-style "inverse purity" over true labels.
    """
    C = contingency_matrix(y_true, y_pred).astype(float)
    N = C.sum()
    cluster_purity = (C.max(axis=0).sum() / N) if N else 0.0
    class_coverage = (C.max(axis=1).sum() / N) if N else 0.0
    h = homogeneity_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    v = v_measure_score(y_true, y_pred)
    return {
        'cluster_purity': float(cluster_purity),
        'class_coverage': float(class_coverage),
        'homogeneity': float(h),
        'completeness': float(c),
        'v_measure': float(v),
    }


def bcubed_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    B-cubed Precision/Recall/F1 for hard partitions using the contingency matrix.
    Especially informative under K mismatch and many-to-one evaluation preferences.
    """
    C = contingency_matrix(y_true, y_pred).astype(float)
    N = C.sum()
    if N == 0:
        return {'bcubed_precision': 0.0, 'bcubed_recall': 0.0, 'bcubed_f1': 0.0}
    # Precision: for each predicted cluster, probability two points in it share the same true label
    denom_prec = np.maximum(1.0, C.sum(axis=0))  # col sums
    bc_prec = ((C ** 2).sum(axis=0) / denom_prec).sum() / N
    # Recall: for each true class, probability two points in it share the same predicted cluster
    denom_rec = np.maximum(1.0, C.sum(axis=1))  # row sums
    bc_rec = ((C ** 2).sum(axis=1) / denom_rec).sum() / N
    bc_f1 = (2 * bc_prec * bc_rec / (bc_prec + bc_rec)) if (bc_prec + bc_rec) > 0 else 0.0
    return {
        'bcubed_precision': float(bc_prec),
        'bcubed_recall': float(bc_rec),
        'bcubed_f1': float(bc_f1),
    }


# ----------------------------
# Plotting helpers (NEW)
# ----------------------------

def _aggregate_by_centers_and_delta(per_run_records: List[Dict], metric_key: str
                                    ) -> Tuple[DefaultDict[Tuple[int,int], List[float]],
                                               List[int], List[int]]:
    """
    Aggregate metric values per (centers, delta_k) for both methods.

    Returns:
      values[(centers, delta_k)] -> dict with 'grid': [..], 'mst': [..]
      sorted unique centers list, sorted unique delta_k list
    """
    values: DefaultDict[Tuple[int,int], Dict[str, List[float]]] = defaultdict(
        lambda: {'grid': [], 'direct_mst': []}
    )
    centers_set, deltas_set = set(), set()
    for rec in per_run_records:
        centers = int(rec['centers'])
        delta_k = int(rec['delta_k'])
        centers_set.add(centers)
        deltas_set.add(delta_k)
        g = rec['grid'].get(metric_key, np.nan)
        m = rec['direct_mst'].get(metric_key, np.nan)
        values[(centers, delta_k)]['grid'].append(g)
        values[(centers, delta_k)]['direct_mst'].append(m)
    return values, sorted(centers_set), sorted(deltas_set)

def _mean_std(arr: List[float]) -> Tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1)) if a.size > 1 else 0.0

def create_clustering_offset_line_plots_extended(
    per_run_records: List[Dict],
    output_dir: str,
    metrics_to_plot: List[str],
    label_map: Dict[str, str],
    *,
    centers_palette: List[str] = None
):
    """
    Create two families of plots for each metric:
      (A) metric_vs_delta_by_centers_<metric>.png
          x-axis: ΔK, separate line per centers, two panels (Grid vs MST) stacked vertically.
      (B) metric_vs_centers_at_delta0_<metric>.png
          x-axis: centers, evaluated only at ΔK = 0 (when available), with errorbars.

    Saves PNGs into <output_dir>/line_plots/.
    """
    out_dir = os.path.join(output_dir, "line_plots")
    os.makedirs(out_dir, exist_ok=True)

    # palette (just let matplotlib choose default cycle if not provided)
    if centers_palette is None:
        centers_palette = None  # use default

    # collect unique centers and deltas once
    # (we’ll still re-compute aggregates per metric to keep code simple)
    for metric in metrics_to_plot:
        values, centers_list, deltas_list = _aggregate_by_centers_and_delta(per_run_records, metric)

        # --- Plot (A): vs ΔK (stacked Grid/MST) ---
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)
        method_names = [('grid', 'Grid SOM'), ('direct_mst', 'Direct MST-SOM')]

        for ax_idx, (method_key, method_label) in enumerate(method_names):
            ax = axes[ax_idx]
            # color cycle reset per subplot for consistent centers colors
            if centers_palette is not None:
                ax.set_prop_cycle(color=centers_palette)

            for centers in centers_list:
                means, stds = [], []
                for d in deltas_list:
                    m, s = _mean_std(values[(centers, d)][method_key])
                    means.append(m)
                    stds.append(s)
                ax.errorbar(deltas_list, means, yerr=stds, marker='o', linestyle='-',
                            label=f"K={centers}")
            ax.set_title(f"{label_map.get(metric, metric)} vs ΔK — {method_label}")
            ax.set_xlabel("ΔK (agglomerative_clusters − true_centers)")
            ax.set_ylabel(label_map.get(metric, metric))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncols=2, fontsize=9)

        plt.tight_layout()
        fnameA = os.path.join(out_dir, f"metric_vs_delta_by_centers_{metric}.png")
        plt.savefig(fnameA, dpi=160, bbox_inches='tight')
        plt.close(fig)

        # --- Plot (B): vs centers at ΔK = 0 ---
        if 0 in deltas_list:
            means_grid, stds_grid, means_mst, stds_mst = [], [], [], []
            for centers in centers_list:
                g_mean, g_std = _mean_std(values[(centers, 0)]['grid'])
                m_mean, m_std = _mean_std(values[(centers, 0)]['direct_mst'])
                means_grid.append(g_mean); stds_grid.append(g_std)
                means_mst.append(m_mean);  stds_mst.append(m_std)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            width = 0.35
            x = np.arange(len(centers_list))
            ax2.bar(x - width/2, means_grid, width, yerr=stds_grid, label='Grid SOM')
            ax2.bar(x + width/2, means_mst, width, yerr=stds_mst, label='Direct MST-SOM')
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(c) for c in centers_list])
            ax2.set_xlabel("True centers K (ΔK = 0)")
            ax2.set_ylabel(label_map.get(metric, metric))
            ax2.set_title(f"{label_map.get(metric, metric)} vs K at ΔK=0")
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.legend(loc='best')
            plt.tight_layout()
            fnameB = os.path.join(out_dir, f"metric_vs_centers_at_delta0_{metric}.png")
            plt.savefig(fnameB, dpi=160, bbox_inches='tight')
            plt.close(fig2)
        else:
            logger.info(f"No ΔK=0 entries found for metric '{metric}'; skipping centers-at-delta0 plot.")


# ----------------------------
# Main routines (original + edits)
# ----------------------------

def run_clustering_stats(args):
    """Run repeated clustering evaluation on sklearn_blobs and perform paired t-tests.

    Collects per-repeat Acc (many-to-one), ARI, NMI, QE, plus diagnostics (H/C/V, purity,
    class coverage, B-cubed). Performs paired t-tests on core metrics and writes a report.
    Also generates line plots for all metrics across ΔK and across K at ΔK=0.
    """
    if not CLUSTERING_AVAILABLE:
        logger.error("Clustering evaluation requires sklearn. Install with: pip install scikit-learn")
        return None

    # Local imports to avoid circular dependencies
    from .training import train_hexagonal_som, train_direct_mst_som
    from .utils import generate_data
    from .visualization import create_clustering_metrics_violin_plot
    from scipy import stats
    import numpy as np
    import os

    # Force blobs dataset
    logger.info("="*60)
    logger.info("CLUSTERING STATS MODE (paired t-tests on sklearn_blobs)")
    logger.info("="*60)
    args.data_type = 'sklearn_blobs'

    # Parse cluster counts list
    clusters_list = []
    raw = getattr(args, 'clusters_list', None)
    if raw:
        try:
            if isinstance(raw, (list, tuple)):
                clusters_list = [int(x) for x in raw]
            elif isinstance(raw, str):
                clusters_list = [int(x.strip()) for x in raw.split(',') if x.strip()]
            else:
                clusters_list = [3, 4, 5]
        except Exception:
            logger.warning(f"Could not parse --clusters_list='{raw}', falling back to defaults [3,4,5]")
            clusters_list = [3, 4, 5]
    if not clusters_list:
        if hasattr(args, 'clusters') and args.clusters is not None:
            clusters_list = [int(args.clusters)]
        else:
            clusters_list = [3, 4, 5]
    logger.info(f"Cluster counts to evaluate: {clusters_list}")

    # Metrics to collect (core first; extras later)
    core_metrics = ['accuracy', 'adjusted_rand_score', 'normalized_mutual_info', 'quantization_error']
    extra_metrics = [
        'homogeneity', 'completeness', 'v_measure',
        'cluster_purity', 'class_coverage',
        'bcubed_precision', 'bcubed_recall', 'bcubed_f1'
    ]
    metrics = core_metrics + extra_metrics

    grid_values = {m: [] for m in metrics}
    mst_values = {m: [] for m in metrics}

    # Parse relative agglomerative offsets
    offsets = [-3, 0, 3]
    raw_offsets = getattr(args, 'agglomerative_offsets', None)
    if raw_offsets is not None:
        try:
            if isinstance(raw_offsets, (list, tuple)):
                offsets = [int(x) for x in raw_offsets]
            elif isinstance(raw_offsets, str):
                offsets = [int(x.strip()) for x in raw_offsets.split(',') if x.strip()]
            else:
                offsets = [-3, 0, 3]
        except Exception:
            logger.warning(f"Could not parse --agglomerative_offsets='{raw_offsets}', using default [-3,0,3]")
            offsets = [-3, 0, 3]
    logger.info(f"Agglomerative cluster offsets: {offsets}")

    base_seed = args.seed
    rng = None if base_seed is not None else np.random.default_rng()
    n_repeats = getattr(args, 'repeats', 10)

    # Store per-run records for new line plots
    per_run_records = []

    # Prepare QE metric once
    qe_metric = QuantizationError(use_optimized=True)

    for centers in clusters_list:
        logger.info("-" * 60)
        logger.info(f"Centers (clusters): {centers}")
        for offset in offsets:
            agglom_k = centers + offset
            if agglom_k < 2:
                logger.info(f"  Skipping offset {offset} (agglomerative clusters {agglom_k} < 2)")
                continue
            logger.info(f"  Offset {offset} -> agglomerative_clusters={agglom_k}")
            for r in range(n_repeats):
                # Determine run-level seed
                if base_seed is not None:
                    run_seed = int(base_seed + r)
                else:
                    run_seed = int(rng.integers(0, 2**31 - 1))
                args.seed = run_seed
                # set per-iteration centers override used by generate_data()
                args.blobs_centers = centers
                logger.info(f"    Repeat {r+1}/{n_repeats} (seed={run_seed}, centers={centers}, ΔK={agglom_k - centers:+d}, K={agglom_k})")

                # Generate data with labels
                data, metadata = generate_data(args)
                if 'labels' not in metadata:
                    logger.error("Dataset does not contain ground truth labels!")
                    return None

                true_labels = metadata['labels']
                if hasattr(true_labels, 'get'):
                    true_labels = true_labels.get()

                # Train Grid SOM (hexagonal)
                if base_seed is None:
                    grid_seed = int(rng.integers(0, 2**31 - 1))
                    prev_seed = args.seed
                    args.seed = grid_seed
                grid_som, _, _ = train_hexagonal_som(data, args)
                if base_seed is None:
                    args.seed = prev_seed
                grid_method = getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')
                grid_metaclusters = perform_metacluster_on_som(grid_som, agglom_k, args, method_override=grid_method)
                grid_predictions = assign_data_to_metaclusters(data, grid_som, grid_metaclusters)
                grid_scores = evaluate_clustering(true_labels, grid_predictions)
                # Quantization Error (lower is better)
                grid_qe = float(qe_metric.compute(grid_som, data))
                grid_scores['quantization_error'] = grid_qe

                # Train Direct MST-SOM
                if base_seed is None:
                    mst_seed = int(rng.integers(0, 2**31 - 1))
                    prev_seed = args.seed
                    args.seed = mst_seed
                mst_som, _, _ = train_direct_mst_som(data, args)
                if base_seed is None:
                    args.seed = prev_seed
                mst_method = getattr(args, 'mst_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')
                mst_metaclusters = perform_metacluster_on_som(mst_som, agglom_k, args, method_override=mst_method)
                mst_predictions = assign_data_to_metaclusters(data, mst_som, mst_metaclusters)
                mst_scores = evaluate_clustering(true_labels, mst_predictions)
                # Quantization Error (lower is better)
                mst_qe = float(qe_metric.compute(mst_som, data))
                mst_scores['quantization_error'] = mst_qe

                # Collect
                for k in metrics:
                    if k in grid_scores:
                        grid_values[k].append(grid_scores[k])
                    else:
                        grid_values[k].append(np.nan)
                    if k in mst_scores:
                        mst_values[k].append(mst_scores[k])
                    else:
                        mst_values[k].append(np.nan)

                # Record for line plots / JSON
                per_run_records.append({
                    'seed': int(args.seed),
                    'centers': int(centers),
                    'agglomerative_clusters': int(agglom_k),
                    'delta_k': int(agglom_k - centers),
                    'grid': grid_scores,
                    'direct_mst': mst_scores
                })

                # Per-run logging and visualization
                per_run_dir = os.path.join(
                    args.output_dir,
                    'runs',
                    f'centers_{centers}_agglom_{agglom_k}',
                    f'seed_{run_seed}'
                )
                os.makedirs(per_run_dir, exist_ok=True)

                # Save per-run metrics
                run_record = {
                    'mode': 'clustering_stats',
                    'seed': int(run_seed),
                    'centers': int(centers),
                    'agglomerative_clusters': int(agglom_k),
                    'delta_k': int(agglom_k - centers),
                    'samples': int(metadata['shape'][0]),
                    'features': int(metadata['shape'][1]),
                    'grid_size': int(getattr(args, 'grid_size', 0)),
                    'iterations': int(getattr(args, 'iterations', 0)),
                    'method_labels': {
                        'direct_mst': 'Direct MST-SOM',
                        'grid': 'Grid SOM'
                    },
                    'metrics': {
                        'grid': grid_scores,
                        'direct_mst': mst_scores
                    }
                }
                try:
                    import json
                    with open(os.path.join(per_run_dir, 'metrics.json'), 'w') as f:
                        json.dump(run_record, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save per-run metrics: {e}")

                # Save per-run visualization (optional)
                if getattr(args, 'visualize', False):
                    try:
                        from .visualization import visualize_clustering_results
                        visualize_clustering_results(
                            grid_som=grid_som,
                            mst_som=mst_som,
                            data=data,
                            true_labels=true_labels,
                            grid_predictions=grid_predictions,
                            mst_predictions=mst_predictions,
                            grid_metaclusters=grid_metaclusters,
                            mst_metaclusters=mst_metaclusters,
                            metadata=metadata,
                            output_dir=per_run_dir,
                            grid_method_name=(getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')),
                            mst_method_name=(getattr(args, 'mst_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative'))
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save per-run visualization: {e}")

    # Paired t-tests on core metrics only
    from scipy import stats
    def paired_t_and_ci(direct_list, grid_list):
        diffs = np.asarray(direct_list) - np.asarray(grid_list)
        n = diffs.size
        if n < 2:
            return {
                't_statistic': np.nan,
                'df': n - 1,
                'p_value': np.nan,
                'mean_diff': np.nan,
                'std_diff': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan
            }
        t_stat, p_val = stats.ttest_rel(direct_list, grid_list, alternative='two-sided')
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        se = std_diff / np.sqrt(n) if n > 0 else np.nan
        t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
        ci_low = mean_diff - t_crit * se if np.isfinite(t_crit) else np.nan
        ci_high = mean_diff + t_crit * se if np.isfinite(ci_low) else np.nan
        return {
            't_statistic': float(t_stat),
            'df': int(n - 1),
            'p_value': float(p_val),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'ci_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
            'ci_high': float(ci_high) if np.isfinite(ci_high) else np.nan
        }

    tests = {}
    for metric in core_metrics:
        tests[metric] = paired_t_and_ci(mst_values[metric], grid_values[metric])

    # Prepare report
    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, 'clustering_stats_report.txt')
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLUSTERING STATS REPORT (sklearn_blobs)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Repeats: {n_repeats}\n")
        f.write(f"Seeds: {args.seed}\n")
        f.write(f"Grid size: {getattr(args, 'grid_size', 0)} | Iterations: {getattr(args, 'total_iterations', 0)} | LR: {getattr(args, 'learning_rate', 'n/a')}\n")
        f.write(f"GPU: {'enabled' if getattr(args, 'use_gpu', False) else 'disabled'}\n\n")

        label_map = {
            'accuracy': 'Accuracy (Many-to-One)',
            'adjusted_rand_score': 'Adjusted Rand Index (ARI)',
            'normalized_mutual_info': 'Normalized Mutual Information (NMI)',
            'quantization_error': 'Quantization Error (lower is better)',
            'homogeneity': 'Homogeneity',
            'completeness': 'Completeness',
            'v_measure': 'V-measure',
            'cluster_purity': 'Cluster Purity (many-to-one)',
            'class_coverage': 'Class Coverage (many-to-one)',
            'bcubed_precision': 'B-cubed Precision',
            'bcubed_recall': 'B-cubed Recall',
            'bcubed_f1': 'B-cubed F1',
        }
        higher_better = {
            'accuracy': True,
            'adjusted_rand_score': True,
            'normalized_mutual_info': True,
            'quantization_error': False,
            'homogeneity': True,
            'completeness': True,
            'v_measure': True,
            'cluster_purity': True,
            'class_coverage': True,
            'bcubed_precision': True,
            'bcubed_recall': True,
            'bcubed_f1': True,
        }

        # Summaries for core metrics with t-tests
        for metric in core_metrics:
            t = tests[metric]
            direct_mean = float(np.mean(mst_values[metric])) if len(mst_values[metric]) else np.nan
            grid_mean = float(np.mean(grid_values[metric])) if len(grid_values[metric]) else np.nan

            if higher_better.get(metric, True):
                winner = 'Direct MST-SOM' if direct_mean > grid_mean else 'Grid SOM'
                denom = grid_mean if winner == 'Direct MST-SOM' else direct_mean
                denom = denom if denom != 0 else 1.0
                win_margin = (abs(direct_mean - grid_mean) / denom) * 100
            else:
                # Lower is better (QE)
                winner = 'Direct MST-SOM' if direct_mean < grid_mean else 'Grid SOM'
                if winner == 'Direct MST-SOM':
                    denom = grid_mean if grid_mean != 0 else 1.0
                    win_margin = ((grid_mean - direct_mean) / denom) * 100
                else:
                    denom = direct_mean if direct_mean != 0 else 1.0
                    win_margin = ((direct_mean - grid_mean) / denom) * 100

            f.write(f"{label_map[metric]}:\n")
            f.write(f"  Direct MST-SOM mean ± SD: {direct_mean:.6f} ± {np.std(mst_values[metric], ddof=1) if len(mst_values[metric])>1 else 0.0:.6f}\n")
            f.write(f"  Grid SOM mean ± SD:       {grid_mean:.6f} ± {np.std(grid_values[metric], ddof=1) if len(grid_values[metric])>1 else 0.0:.6f}\n")
            f.write(f"  Paired t({t['df']}): {t['t_statistic']:.4f}, p-value: {t['p_value']:.4f}\n")
            f.write(f"  95% CI of diff (Direct - Grid): [{t['ci_low']:.6f}, {t['ci_high']:.6f}]\n")
            f.write(f"  Winner: {winner} ({win_margin:.1f}% difference)\n")
            if t['p_value'] < 0.05:
                f.write("  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)\n")
            f.write("\n")

        # Also dump the extra diagnostics (means only)
        f.write("-" * 80 + "\n")
        f.write("Additional diagnostics (means across runs)\n")
        f.write("-" * 80 + "\n")
        for metric in extra_metrics:
            direct_mean = float(np.mean(mst_values[metric])) if len(mst_values[metric]) else np.nan
            grid_mean = float(np.mean(grid_values[metric])) if len(grid_values[metric]) else np.nan
            f.write(f"{label_map[metric]}:\n")
            f.write(f"  Direct MST-SOM mean: {direct_mean:.6f}\n")
            f.write(f"  Grid SOM mean:       {grid_mean:.6f}\n\n")

        f.write("Note: No multiple-comparisons adjustment applied across the t-tests.\n")

    logger.info(f"Clustering stats report saved to {report_file}")

    # Create violin plot (kept for Acc/ARI/NMI)
    try:
        from .visualization import create_qe_violin_plot, create_clustering_metrics_violin_plot
        create_clustering_metrics_violin_plot(
            mst_values, grid_values, args.output_dir,
            grid_method=getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative'),
            mst_method=getattr(args, 'mst_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')
        )
        create_qe_violin_plot(
            mst_values, grid_values, args.output_dir,
            grid_method=getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative'),
            mst_method=getattr(args, 'mst_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')
        )
        logger.info("Violin plots saved")
    except Exception as e:
        logger.warning(f"Violin plotting skipped or failed: {e}")

    logger.info("Generating extended line plots for all metrics...")
    label_map_all = {
        'accuracy': 'Accuracy (Many-to-One)',
        'adjusted_rand_score': 'Adjusted Rand Index (ARI)',
        'normalized_mutual_info': 'Normalized Mutual Information (NMI)',
        'quantization_error': 'Quantization Error (lower is better)',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'v_measure': 'V-measure',
        'cluster_purity': 'Cluster Purity (many-to-one)',
        'class_coverage': 'Class Coverage (many-to-one)',
        'bcubed_precision': 'B-cubed Precision',
        'bcubed_recall': 'B-cubed Recall',
        'bcubed_f1': 'B-cubed F1',
    }
    # Generate plots for all metrics we collected
    create_clustering_offset_line_plots_extended(
        per_run_records,
        args.output_dir,
        metrics_to_plot=metrics,
        label_map=label_map_all
    )
    logger.info("Extended line plots saved")

    return {
        'grid_values': grid_values,
        'mst_values': mst_values,
        'tests': tests,
        'n_repeats': n_repeats,
        'base_seed': base_seed,
        'report_file': report_file
    }


def perform_agglomerative_on_som(som: FloatSOM, n_clusters: int, use_gpu: bool = True) -> np.ndarray:
    """Apply agglomerative clustering to SOM weights."""
    weights = som.get_weights()  # Shape: (n_nodes, n_features)

    # Convert to CPU if needed for sklearn
    if hasattr(weights, 'get'):  # CuPy array
        weights_cpu = weights.get()
    else:
        weights_cpu = weights

    # Check if cuML is available for GPU acceleration
    if use_gpu:
        try:
            from cuml.cluster import AgglomerativeClustering as AgglomerativeClusteringGPU
            weights_gpu = cp.asarray(weights_cpu)
            clusterer = AgglomerativeClusteringGPU(n_clusters=n_clusters)
            metacluster_labels = clusterer.fit_predict(weights_gpu)
            metacluster_labels = cp.asnumpy(metacluster_labels)
        except ImportError:
            # Fall back to CPU
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            metacluster_labels = clusterer.fit_predict(weights_cpu)
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        metacluster_labels = clusterer.fit_predict(weights_cpu)

    return metacluster_labels


def perform_consensus_on_som(
    som: FloatSOM,
    n_clusters: int,
    *,
    consensus_runs: int = 20,
    consensus_subsample: float = 0.8,
    consensus_linkage: str = 'average',
    seed: int = None,
) -> np.ndarray:
    """Consensus clustering on SOM weights via feature subsampling and co-association."""
    weights = som.get_weights()
    X = weights.get() if hasattr(weights, 'get') else weights  # (n_nodes, n_features)
    n_nodes, n_features = X.shape

    rng = np.random.default_rng(seed)
    runs = max(1, int(consensus_runs))
    p = float(consensus_subsample)
    p = min(max(p, 0.1), 1.0)

    co = np.zeros((n_nodes, n_nodes), dtype=float)

    for _ in range(runs):
        k = max(1, int(round(p * n_features)))
        cols = rng.choice(n_features, size=k, replace=False) if k < n_features else np.arange(n_features)
        X_sub = X[:, cols]
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_sub)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            co[np.ix_(idx, idx)] += 1.0

    co /= runs

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    d = 1.0 - co
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method=consensus_linkage)
    labels_final = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    print("Performed consensus on SOM")
    return labels_final.astype(int)


def perform_metacluster_on_som(som: FloatSOM, n_clusters: int, args, method_override: str = None) -> np.ndarray:
    """Dispatch metaclustering based on configuration."""
    method = method_override or getattr(args, 'metacluster_method', 'agglomerative')
    if method == 'consensus':
        return perform_consensus_on_som(
            som,
            n_clusters=n_clusters,
            consensus_runs=getattr(args, 'consensus_runs', 20),
            consensus_subsample=getattr(args, 'consensus_subsample', 0.8),
            consensus_linkage=getattr(args, 'consensus_linkage', 'average'),
            seed=getattr(args, 'seed', None),
        )
    # default: agglomerative
    return perform_agglomerative_on_som(som, n_clusters=n_clusters, use_gpu=getattr(args, 'use_gpu', True))


def assign_data_to_metaclusters(data: Union[np.ndarray, cp.ndarray],
                                som: FloatSOM,
                                metacluster_labels: np.ndarray) -> np.ndarray:
    """Map data points to metaclusters via their BMUs."""
    weights = som.get_weights()

    # Ensure both data and weights are on same device
    if hasattr(data, 'get'):  # data is CuPy
        if not hasattr(weights, 'get'):  # weights is numpy
            weights = cp.asarray(weights)
    else:  # data is numpy
        if hasattr(weights, 'get'):  # weights is CuPy
            data = cp.asarray(data)

    # Find BMU for each data point
    bmu_indices = []
    batch_size = 1000  # Process in batches for memory efficiency

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        # Compute distances to all SOM nodes
        if hasattr(data, 'get'):  # Using CuPy
            distances = cp.sum((weights[None, :, :] - batch[:, None, :])**2, axis=2)
            batch_bmus = cp.argmin(distances, axis=1)
            bmu_indices.extend(batch_bmus.get().tolist())
        else:  # Using NumPy
            distances = np.sum((weights[None, :, :] - batch[:, None, :])**2, axis=2)
            batch_bmus = np.argmin(distances, axis=1)
            bmu_indices.extend(batch_bmus.tolist())

    # Map BMU index to metacluster label
    data_metaclusters = metacluster_labels[bmu_indices]
    return data_metaclusters


def evaluate_clustering(true_labels: np.ndarray,
                        predicted_labels: np.ndarray) -> Dict[str, float]:
    """
    Compare predicted metaclusters to true labels and return a dictionary of metrics.

    Core:
        - adjusted_rand_score (ARI)
        - normalized_mutual_info (NMI)
        - accuracy (many-to-one)
    Extras:
        - homogeneity, completeness, v_measure
        - cluster_purity, class_coverage
        - B-cubed precision/recall/f1
    """
    # Core metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    acc = many_to_one_accuracy(true_labels, predicted_labels)

    # Diagnostics
    diag = clustering_diagnostics(true_labels, predicted_labels)
    b3 = bcubed_scores(true_labels, predicted_labels)

    out = {
        'adjusted_rand_score': float(ari),
        'normalized_mutual_info': float(nmi),
        'accuracy': float(acc),
        # diagnostics
        **diag,
        **b3,
    }
    return out


def print_clustering_comparison(results: Dict[str, Dict[str, float]]):
    """Print formatted clustering evaluation results (includes diagnostics)."""
    logger.info("\n" + "="*60)
    logger.info("CLUSTERING EVALUATION RESULTS")
    logger.info("="*60)

    def fmt_block(name: str, m: Dict[str, float]):
        logger.info(f"\n{name}:")
        logger.info(f"  Adjusted Rand Index: {m.get('adjusted_rand_score', float('nan')):.4f}")
        logger.info(f"  Normalized Mutual Info: {m.get('normalized_mutual_info', float('nan')):.4f}")
        logger.info(f"  Accuracy (Many-to-One): {m.get('accuracy', float('nan')):.4f}")
        if 'quantization_error' in m:
            logger.info(f"  Quantization Error (lower is better): {m['quantization_error']:.6f}")
        # Diagnostics
        logger.info(f"  Homogeneity: {m.get('homogeneity', float('nan')):.4f} | "
                    f"Completeness: {m.get('completeness', float('nan')):.4f} | "
                    f"V-measure: {m.get('v_measure', float('nan')):.4f}")
        logger.info(f"  Cluster Purity: {m.get('cluster_purity', float('nan')):.4f} | "
                    f"Class Coverage: {m.get('class_coverage', float('nan')):.4f}")
        logger.info(f"  B-cubed P/R/F1: {m.get('bcubed_precision', float('nan')):.4f} / "
                    f"{m.get('bcubed_recall', float('nan')):.4f} / "
                    f"{m.get('bcubed_f1', float('nan')):.4f}")

    fmt_block("Grid SOM", results['grid'])
    fmt_block("Direct MST-SOM", results['direct_mst'])

    # Calculate average of core metrics (exclude QE which is lower-better)
    keys_for_avg = ['adjusted_rand_score', 'normalized_mutual_info', 'accuracy']
    grid_avg = float(np.mean([results['grid'][k] for k in keys_for_avg]))
    mst_avg = float(np.mean([results['direct_mst'][k] for k in keys_for_avg]))

    logger.info("\n" + "-"*60)
    logger.info("SUMMARY (mean of Acc/ARI/NMI)")
    logger.info("-"*60)
    logger.info(f"Grid SOM average score: {grid_avg:.4f}")
    logger.info(f"Direct MST-SOM average score: {mst_avg:.4f}")

    if mst_avg > grid_avg:
        improvement = ((mst_avg - grid_avg) / grid_avg) * 100 if grid_avg != 0 else float('inf')
        logger.info(f"\n✓ Direct MST-SOM performs better for clustering ({improvement:.1f}% improvement)")
    elif grid_avg > mst_avg:
        improvement = ((grid_avg - mst_avg) / mst_avg) * 100 if mst_avg != 0 else float('inf')
        logger.info(f"\n✓ Grid SOM performs better for clustering ({improvement:.1f}% improvement)")
    else:
        logger.info(f"\n≈ Both approaches perform similarly for clustering")


def run_clustering_evaluation(args):
    """Run clustering evaluation mode.

    Compares Grid SOM vs Direct MST-SOM for preserving cluster structure.
    """
    if not CLUSTERING_AVAILABLE:
        logger.error("Clustering evaluation requires sklearn. Install with: pip install scikit-learn")
        return

    # Import here to avoid circular dependency
    from .training import train_hexagonal_som, train_direct_mst_som
    from .utils import generate_data

    # Force blobs dataset for true labels
    logger.info("="*60)
    logger.info("CLUSTERING EVALUATION MODE")
    logger.info("="*60)
    logger.info("\nForcing sklearn_blobs dataset for ground truth labels...")
    args.data_type = 'sklearn_blobs'
    # If user provided --clusters, use it to set the number of blobs centers
    if hasattr(args, 'clusters') and args.clusters is not None:
        args.blobs_centers = args.clusters

    # Generate data with labels (respects blobs centers override and --clusters)
    logger.info("Generating labeled dataset...")
    # If no seed provided, pick a random one for data generation
    if getattr(args, 'seed', None) is None:
        temp_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        args.seed = temp_seed
        data, metadata = generate_data(args)
        # Keep run-level seed for logging, but allow per-method seeds below
        run_seed = temp_seed
    else:
        run_seed = args.seed
        data, metadata = generate_data(args)

    if 'labels' not in metadata:
        logger.error("Dataset does not contain ground truth labels!")
        return

    true_labels = metadata['labels']
    # Ensure labels are numpy array on CPU for sklearn metrics and logging
    if hasattr(true_labels, 'get'):
        true_labels = true_labels.get()
    n_clusters = metadata['n_clusters']
    # Allow overriding agglomerative target clusters via hyperparameter
    agglom_k = getattr(args, 'agglomerative_clusters', None) or n_clusters

    logger.info(f"Dataset: {metadata['dataset_name']}")
    logger.info(f"Shape: {metadata['shape']}")
    logger.info(f"Number of true clusters: {n_clusters}")
    logger.info(f"Unique labels: {np.unique(true_labels)}")

    results = {}

    # Train Grid SOM (hexagonal topology)
    logger.info("\n" + "="*60)
    logger.info("TRAINING GRID SOM")
    logger.info("="*60)
    grid_som, grid_stats, grid_time = train_hexagonal_som(data, args)
    logger.info(f"Training completed in {grid_time:.3f}s")

    # Apply metaclustering to Grid SOM
    grid_method = getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')
    logger.info(f"\nApplying {grid_method} metaclustering to Grid SOM nodes...")
    grid_metaclusters = perform_metacluster_on_som(grid_som, agglom_k, args, method_override=grid_method)
    logger.info(f"Created {len(np.unique(grid_metaclusters))} metaclusters")

    # Assign data points to metaclusters
    logger.info("Assigning data points to Grid SOM metaclusters...")
    grid_predictions = assign_data_to_metaclusters(data, grid_som, grid_metaclusters)

    # Evaluate Grid SOM
    results['grid'] = evaluate_clustering(true_labels, grid_predictions)
    logger.info(f"Grid SOM evaluation complete")

    # Train Direct MST-SOM
    logger.info("\n" + "="*60)
    logger.info("TRAINING DIRECT MST-SOM")
    logger.info("="*60)
    # If seed was None, use a fresh seed for MST training
    if getattr(args, 'seed', None) == run_seed and 'temp_seed' in locals():
        prev_seed = args.seed
        args.seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        mst_som, mst_stats, mst_time = train_direct_mst_som(data, args)
        args.seed = prev_seed
    else:
        mst_som, mst_stats, mst_time = train_direct_mst_som(data, args)
    logger.info(f"Training completed in {mst_time:.3f}s")

    # Apply metaclustering to MST-SOM
    logger.info(f"\nApplying {grid_method} metaclustering to MST-SOM nodes...")
    mst_metaclusters = perform_metacluster_on_som(mst_som, agglom_k, args, method_override=grid_method)
    logger.info(f"Created {len(np.unique(mst_metaclusters))} metaclusters")

    # Assign data points to metaclusters
    logger.info("Assigning data points to MST-SOM metaclusters...")
    mst_predictions = assign_data_to_metaclusters(data, mst_som, mst_metaclusters)

    # Evaluate MST-SOM
    results['direct_mst'] = evaluate_clustering(true_labels, mst_predictions)

    # Compute and attach Quantization Error for both methods
    try:
        qe_metric = QuantizationError(use_optimized=True)
        grid_qe = float(qe_metric.compute(grid_som, data))
        mst_qe = float(qe_metric.compute(mst_som, data))
        results['grid']['quantization_error'] = grid_qe
        results['direct_mst']['quantization_error'] = mst_qe
        logger.info(f"\nQuantization Error (lower is better):")
        logger.info(f"  Grid SOM QE: {grid_qe:.6f}")
        logger.info(f"  Direct MST-SOM QE: {mst_qe:.6f}")
    except Exception as e:
        logger.warning(f"Failed to compute Quantization Error: {e}")
    logger.info(f"Direct MST-SOM evaluation complete")

    # Display results
    print_clustering_comparison(results)

    # Visualize clustering assignments if requested
    if getattr(args, 'visualize', False):
        from .visualization import visualize_clustering_results
        visualize_clustering_results(
            grid_som=grid_som,
            mst_som=mst_som,
            data=data,
            true_labels=true_labels,
            grid_predictions=grid_predictions,
            mst_predictions=mst_predictions,
            grid_metaclusters=grid_metaclusters,
            mst_metaclusters=mst_metaclusters,
            metadata=metadata,
            output_dir=args.output_dir,
            grid_method_name=(getattr(args, 'grid_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative')),
            mst_method_name=(getattr(args, 'mst_metacluster_method', None) or getattr(args, 'metacluster_method', 'agglomerative'))
        )
        # Additionally plot a simple QE comparison
        try:
            from .visualization import create_qe_bar_plot
            create_qe_bar_plot(results['direct_mst'].get('quantization_error'),
                               results['grid'].get('quantization_error'),
                               args.output_dir)
        except Exception as e:
            logger.warning(f"Failed to create QE bar plot: {e}")

    # Save results to file (now includes diagnostics)
    output_file = os.path.join(args.output_dir, 'clustering_evaluation_results.csv')
    df = pd.DataFrame(results).T
    df.to_csv(output_file)
    logger.info(f"\nResults saved to {output_file}")

    return results