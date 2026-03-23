"""
MST Clustering Quality Metrics
==============================
Compare MST quality for downstream clustering tasks, independent of node positions.
Focuses on edge weight distributions and structural properties that indicate good clustering.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
import warnings


class MSTClusteringMetrics:
    """
    A collection of metrics to evaluate MST quality for clustering purposes.
    These metrics focus on edge weight distributions and structural properties
    rather than node positions.
    """
    
    @staticmethod
    def gap_statistic(edge_weights: np.ndarray) -> Dict[str, float]:
        """
        Compute gap statistics for edge weights.
        Larger gaps indicate clearer cluster separation.
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
            
        Returns:
        --------
        dict : Dictionary containing:
            - max_gap: Maximum gap between consecutive sorted weights
            - gap_cv: Coefficient of variation of gaps
            - gap_ratio: Ratio of largest gap to average gap
        """
        sorted_weights = np.sort(edge_weights)
        gaps = np.diff(sorted_weights)
        
        if len(gaps) == 0:
            return {'max_gap': 0, 'gap_cv': 0, 'gap_ratio': 0}
        
        max_gap = np.max(gaps)
        mean_gap = np.mean(gaps)
        
        # Avoid division by zero
        gap_cv = np.std(gaps) / mean_gap if mean_gap > 0 else 0
        gap_ratio = max_gap / mean_gap if mean_gap > 0 else 0
        
        return {
            'max_gap': max_gap,
            'gap_cv': gap_cv,
            'gap_ratio': gap_ratio  # Higher = better for clustering
        }
    
    @staticmethod
    def edge_heterogeneity(edge_weights: np.ndarray) -> Dict[str, float]:
        """
        Measure heterogeneity of edge weights.
        Higher heterogeneity indicates mix of intra-cluster and inter-cluster edges.
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
            
        Returns:
        --------
        dict : Dictionary containing:
            - gini: Gini coefficient (0=equal, 1=unequal)
            - entropy: Shannon entropy of weight distribution
            - cv: Coefficient of variation
        """
        if len(edge_weights) == 0:
            return {
                'gini': 0,
                'entropy': 0,
                'cv': 0,
            }

        sorted_weights = np.sort(edge_weights)
        n = len(sorted_weights)
        
        # Gini coefficient
        if n == 0 or np.sum(sorted_weights) == 0:
            gini = 0
        else:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        
        # Entropy of edge weight distribution
        if len(edge_weights) > 1:
            hist, bins = np.histogram(edge_weights, bins=min(20, len(edge_weights)))
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist_norm = hist / hist.sum()
                weight_entropy = entropy(hist_norm)
            else:
                weight_entropy = 0
        else:
            weight_entropy = 0
        
        # Coefficient of variation
        cv = np.std(edge_weights) / np.mean(edge_weights) if np.mean(edge_weights) > 0 else 0
        
        return {
            'gini': gini,  # Higher = better for clustering
            'entropy': weight_entropy,  # Higher = more diverse weights
            'cv': cv  # Higher = more variation
        }
    
    @staticmethod
    def edge_length_ratio(edge_weights: np.ndarray) -> Dict[str, float]:
        """
        Compute ratios of long to short edges.
        Higher ratios indicate clearer cluster separation.
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
            
        Returns:
        --------
        dict : Dictionary containing:
            - max_min_ratio: Ratio of maximum to minimum edge weight
            - top_10_percent_ratio: Ratio of top 10% to bottom 10% average weights
            - percentile_90_10: Ratio of 90th to 10th percentile
        """
        if len(edge_weights) == 0:
            return {'max_min_ratio': 1, 'top_10_percent_ratio': 1, 'percentile_90_10': 1}
            
        sorted_weights = np.sort(edge_weights)
        
        # Basic max/min ratio
        if sorted_weights[0] > 0:
            max_min_ratio = sorted_weights[-1] / sorted_weights[0]
        else:
            max_min_ratio = np.inf
        
        # Top/bottom k ratio (more robust)
        k = max(1, len(sorted_weights) // 10)  # At least 1, at most 10% of edges
        if np.mean(sorted_weights[:k]) > 0:
            top_k_ratio = np.mean(sorted_weights[-k:]) / np.mean(sorted_weights[:k])
        else:
            top_k_ratio = np.inf
        
        # Percentile ratio
        p90 = np.percentile(edge_weights, 90)
        p10 = np.percentile(edge_weights, 10)
        percentile_ratio = p90 / p10 if p10 > 0 else np.inf
        
        return {
            'max_min_ratio': max_min_ratio,
            'top_10_percent_ratio': top_k_ratio,  # Higher = better separation
            'percentile_90_10': percentile_ratio
        }
    
    @staticmethod
    def bimodality_coefficient(edge_weights: np.ndarray) -> float:
        """
        Compute bimodality coefficient.
        Values > 0.555 suggest bimodal distribution (good for clustering).
        
        BC = (skewness^2 + 1) / (kurtosis + 3)
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
            
        Returns:
        --------
        float : Bimodality coefficient
        """
        if len(edge_weights) < 4:
            return 0
        
        # Compute moments
        mean = np.mean(edge_weights)
        std = np.std(edge_weights, ddof=1)
        
        if std == 0:
            return 0
        
        # Skewness
        skewness = np.mean(((edge_weights - mean) / std) ** 3)
        
        # Kurtosis (excess kurtosis)
        kurtosis = np.mean(((edge_weights - mean) / std) ** 4) - 3
        
        # Bimodality coefficient
        bc = (skewness ** 2 + 1) / (kurtosis + 3)
        
        return bc
    
    @staticmethod
    def clustering_score_range(edge_weights: np.ndarray, 
                              n_clusters_range: List[int] = [2, 3, 4, 5]) -> Dict[str, float]:
        """
        Simulate clustering quality across different numbers of clusters.
        Based on edge weight distribution characteristics.
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
        n_clusters_range : List[int]
            Range of cluster numbers to test
            
        Returns:
        --------
        dict : Dictionary containing quality scores
        """
        sorted_weights = np.sort(edge_weights)
        scores = []
        
        for k in n_clusters_range:
            if k > len(edge_weights):
                continue
                
            # Find the k-1 cut points (largest gaps)
            if k - 1 <= len(edge_weights):
                # Get the k-1 largest edges (cut points)
                cut_weights = sorted_weights[-(k-1):] if k > 1 else []
                
                if len(cut_weights) > 0:
                    # Measure separation quality: ratio of cut edges to non-cut edges
                    non_cut_weights = sorted_weights[:-(k-1)]
                    if len(non_cut_weights) > 0:
                        separation_score = np.mean(cut_weights) / np.mean(non_cut_weights)
                    else:
                        separation_score = 1
                else:
                    separation_score = 1
                    
                scores.append(separation_score)
        
        if len(scores) == 0:
            return {'mean_separation': 1, 'max_separation': 1}
            
        return {
            'mean_separation': np.mean(scores),
            'max_separation': np.max(scores)
        }
    
    @staticmethod
    def cut_stability_proxy(edge_weights: np.ndarray) -> Dict[str, float]:
        """
        Estimate cut stability based on edge weight distribution.
        More uniform consecutive gaps = more stable cuts.
        
        Parameters:
        -----------
        edge_weights : np.ndarray
            Array of MST edge weights
            
        Returns:
        --------
        dict : Dictionary containing stability metrics
        """
        sorted_weights = np.sort(edge_weights)
        gaps = np.diff(sorted_weights)
        
        if len(gaps) < 2:
            return {'gap_uniformity': 1, 'relative_gap_variance': 0}
        
        # Measure uniformity of gaps (lower variance = more stable)
        gap_variance = np.var(gaps)
        gap_mean = np.mean(gaps)
        
        # Relative variance (normalized)
        relative_variance = gap_variance / (gap_mean ** 2) if gap_mean > 0 else 0
        
        # Gap uniformity score (inverse of coefficient of variation)
        gap_uniformity = 1 / (1 + np.std(gaps) / gap_mean) if gap_mean > 0 else 0
        
        return {
            'gap_uniformity': gap_uniformity,  # Higher = more uniform = more stable
            'relative_gap_variance': relative_variance  # Lower = more stable
        }
    
    @staticmethod
    def comprehensive_comparison(mst1_weights: np.ndarray, 
                                mst2_weights: np.ndarray,
                                weight_importance: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Comprehensive comparison of two MSTs for clustering quality.
        
        Parameters:
        -----------
        mst1_weights : np.ndarray
            Edge weights from first MST
        mst2_weights : np.ndarray
            Edge weights from second MST
        weight_importance : Dict[str, float]
            Optional weights for different metrics (default: equal weights)
            
        Returns:
        --------
        dict : Comprehensive comparison results
        """
        if weight_importance is None:
            weight_importance = {
                'gap_ratio': 1.0,
                'gini': 1.0,
                'edge_ratio': 1.0,
                'bimodality': 0.5,
                'separation': 1.0,
                'stability': 0.5
            }
        
        metrics = MSTClusteringMetrics()
        
        # Compute all metrics for both MSTs
        results = {
            'MST1': {},
            'MST2': {},
            'comparison': {},
            'recommendation': None
        }
        
        # Gap statistics
        gap1 = metrics.gap_statistic(mst1_weights)
        gap2 = metrics.gap_statistic(mst2_weights)
        results['MST1']['gap_ratio'] = gap1['gap_ratio']
        results['MST2']['gap_ratio'] = gap2['gap_ratio']
        
        # Heterogeneity
        het1 = metrics.edge_heterogeneity(mst1_weights)
        het2 = metrics.edge_heterogeneity(mst2_weights)
        results['MST1']['gini'] = het1['gini']
        results['MST2']['gini'] = het2['gini']
        results['MST1']['entropy'] = het1['entropy']
        results['MST2']['entropy'] = het2['entropy']
        
        # Edge ratios
        ratio1 = metrics.edge_length_ratio(mst1_weights)
        ratio2 = metrics.edge_length_ratio(mst2_weights)
        results['MST1']['edge_ratio'] = ratio1['top_10_percent_ratio']
        results['MST2']['edge_ratio'] = ratio2['top_10_percent_ratio']
        
        # Bimodality
        results['MST1']['bimodality'] = metrics.bimodality_coefficient(mst1_weights)
        results['MST2']['bimodality'] = metrics.bimodality_coefficient(mst2_weights)
        
        # Clustering separation
        sep1 = metrics.clustering_score_range(mst1_weights)
        sep2 = metrics.clustering_score_range(mst2_weights)
        results['MST1']['separation'] = sep1['mean_separation']
        results['MST2']['separation'] = sep2['mean_separation']
        
        # Stability
        stab1 = metrics.cut_stability_proxy(mst1_weights)
        stab2 = metrics.cut_stability_proxy(mst2_weights)
        results['MST1']['stability'] = stab1['gap_uniformity']
        results['MST2']['stability'] = stab2['gap_uniformity']
        
        # Compute weighted scores
        mst1_score = 0
        mst2_score = 0
        
        for metric in ['gap_ratio', 'gini', 'edge_ratio', 'bimodality', 'separation', 'stability']:
            if metric in weight_importance:
                weight = weight_importance[metric]
                
                # Normalize metrics to [0, 1] range for fair comparison
                val1 = results['MST1'].get(metric, 0)
                val2 = results['MST2'].get(metric, 0)
                
                # Handle infinite values
                if np.isinf(val1):
                    val1 = val2 * 2 if not np.isinf(val2) else 1
                if np.isinf(val2):
                    val2 = val1 * 2 if not np.isinf(val1) else 1
                
                max_val = max(val1, val2)
                if max_val > 0:
                    norm_val1 = val1 / max_val
                    norm_val2 = val2 / max_val
                else:
                    norm_val1 = norm_val2 = 0.5
                
                mst1_score += weight * norm_val1
                mst2_score += weight * norm_val2
                
                # Store comparison
                results['comparison'][metric] = {
                    'MST1': val1,
                    'MST2': val2,
                    'better': 'MST1' if val1 > val2 else 'MST2'
                }
        
        # Overall scores
        results['overall_scores'] = {
            'MST1': mst1_score,
            'MST2': mst2_score
        }
        
        # Recommendation
        score_diff = abs(mst1_score - mst2_score)
        total_weight = sum(weight_importance.values())
        relative_diff = score_diff / total_weight if total_weight > 0 else 0
        
        if relative_diff < 0.05:
            results['recommendation'] = "Both MSTs are similarly suitable for clustering"
        elif mst1_score > mst2_score:
            results['recommendation'] = f"MST1 is better for clustering (score: {mst1_score:.3f} vs {mst2_score:.3f})"
        else:
            results['recommendation'] = f"MST2 is better for clustering (score: {mst2_score:.3f} vs {mst1_score:.3f})"
        
        return results


def print_comparison_report(comparison_results: Dict[str, Any]):
    """
    Pretty print the comparison results.
    
    Parameters:
    -----------
    comparison_results : dict
        Output from comprehensive_comparison
    """
    print("=" * 60)
    print("MST CLUSTERING QUALITY COMPARISON")
    print("=" * 60)
    
    print("\nDETAILED METRICS:")
    print("-" * 40)
    
    for metric, values in comparison_results['comparison'].items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  MST1: {values['MST1']:.4f}")
        print(f"  MST2: {values['MST2']:.4f}")
        print(f"  Better: {values['better']}")
    
    print("\n" + "=" * 60)
    print("OVERALL SCORES:")
    print(f"  MST1: {comparison_results['overall_scores']['MST1']:.4f}")
    print(f"  MST2: {comparison_results['overall_scores']['MST2']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"RECOMMENDATION: {comparison_results['recommendation']}")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Example: Create synthetic edge weights for two MSTs
    np.random.seed(42)
    
    # MST1: Clear cluster structure (bimodal distribution)
    cluster1_edges = np.random.normal(1.0, 0.2, 30)
    cluster2_edges = np.random.normal(1.2, 0.2, 25)
    inter_cluster_edges = np.random.normal(5.0, 0.5, 5)
    mst1_weights = np.concatenate([cluster1_edges, cluster2_edges, inter_cluster_edges])
    
    # MST2: More uniform distribution (poorer clustering)
    mst2_weights = np.random.uniform(0.5, 3.0, 60)
    
    # Initialize metrics calculator
    metrics = MSTClusteringMetrics()
    
    # Run comprehensive comparison
    comparison = metrics.comprehensive_comparison(mst1_weights, mst2_weights)
    
    # Print report
    print_comparison_report(comparison)
    
    # You can also access individual metrics
    print("\n\nINDIVIDUAL METRIC EXAMPLES:")
    print(f"MST1 Gap Statistic: {metrics.gap_statistic(mst1_weights)}")
    print(f"MST1 Bimodality: {metrics.bimodality_coefficient(mst1_weights):.4f}")
    print(f"MST1 Edge Heterogeneity: {metrics.edge_heterogeneity(mst1_weights)}")
