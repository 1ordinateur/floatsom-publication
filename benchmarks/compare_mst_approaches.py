#!/usr/bin/env python3
"""
Compare MST Quality: Direct MST-SOM vs Hexagonal-to-MST Conversion
===================================================================

This is the refactored version using modular structure.
All functionality has been split into organized modules in mst_comparison/.
"""

import os
import sys
import time
import logging
import numpy as np
import cupy as cp
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

# Import from refactored modules
from mst_comparison.utils import parse_args, generate_data, save_results
from mst_comparison.training import (
    train_direct_mst_som, 
    train_hexagonal_som,
    convert_hexagonal_to_mst,
    get_mst_edge_weights
)
from mst_comparison.clustering_mode import run_clustering_evaluation, run_clustering_stats
from mst_comparison.statistical_mode import run_statistical_evaluation
from mst_comparison.visualization import visualize_msts

# Import MST clustering metrics
from floatsom.evaluation.mst_clustering_metrics import MSTClusteringMetrics, print_comparison_report


def main():
    """Main function for comparing MST approaches."""
    args = parse_args()
    
    # Route to appropriate mode
    if args.mode == 'clustering':
        # Run clustering evaluation mode
        run_clustering_evaluation(args)
        return
    
    elif args.mode == 'stats_eval' or args.stats_eval:
        # Run statistical evaluation across all datasets
        run_statistical_evaluation(args)
        return
    elif args.mode == 'clustering_stats':
        # Run repeated clustering stats (paired t-tests on blobs)
        run_clustering_stats(args)
        return
    
    # Default: Single comparison mode
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data
    logger.info("Generating dataset...")
    data, metadata = generate_data(args)
    logger.info(f"Dataset: {metadata['dataset_name']}, shape: {metadata['shape']}")
    
    # Train direct MST-SOM
    logger.info("\n" + "="*60)
    logger.info("APPROACH 1: Direct MST-SOM")
    logger.info("="*60)
    direct_mst_som, direct_stats, direct_train_time = train_direct_mst_som(data, args)
    logger.info(f"Training completed in {direct_train_time:.3f}s")
    logger.info(f"Iterations: {direct_stats['iterations_completed']}")
    
    # Train hexagonal SOM
    logger.info("\n" + "="*60)
    logger.info("APPROACH 2: Hexagonal SOM (to be converted to MST)")
    logger.info("="*60)
    hexagonal_som, hex_stats, hex_train_time = train_hexagonal_som(data, args)
    logger.info(f"Training completed in {hex_train_time:.3f}s")
    logger.info(f"Iterations: {hex_stats['iterations_completed']}")
    
    # Convert hexagonal to MST
    logger.info("\nConverting hexagonal SOM to MST...")
    hex_mst_edges, hex_mst_weights = convert_hexagonal_to_mst(hexagonal_som)
    logger.info(f"Created MST with {len(hex_mst_edges)} edges")
    
    # Get edge weights from direct MST
    logger.info("\nExtracting edge weights from direct MST-SOM...")
    direct_mst_weights = get_mst_edge_weights(direct_mst_som)
    logger.info(f"Direct MST has {len(direct_mst_weights)} edges")
    
    # Compare MST quality using clustering metrics
    logger.info("\n" + "="*60)
    logger.info("COMPARING MST QUALITY")
    logger.info("="*60)
    
    metrics = MSTClusteringMetrics()
    comparison = metrics.comprehensive_comparison(
        direct_mst_weights,
        hex_mst_weights,
        weight_importance={
            'gap_ratio': 1.0,      # Important for cluster separation
            'gini': 1.0,           # Important for edge heterogeneity
            'edge_ratio': 1.5,     # Very important for clustering
            'bimodality': 0.5,     # Moderate importance
            'separation': 1.5,     # Very important for clustering
            'stability': 0.5       # Moderate importance
        }
    )
    
    # Print comparison report
    print_comparison_report(comparison)
    
    # Save results
    save_results(comparison, direct_train_time, hex_train_time, args, args.output_dir)
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("\nCreating visualizations...")
        visualize_msts(direct_mst_som, hexagonal_som, hex_mst_edges, 
                      data, args.output_dir)
        logger.info(f"Visualizations saved to {args.output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*60)
    
    return comparison


if __name__ == "__main__":
    # Match original environment behavior for reproducibility/performance
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
