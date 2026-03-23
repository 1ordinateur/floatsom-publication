# Pareto Front Analysis Pipeline

A comprehensive analysis framework for comparing Self-Organizing Map (SOM) algorithms across different architectures, datasets, and sampling methods.

## 🎯 Overview

This pipeline analyzes SOM performance by comparing two algorithms (batch and colors) across multiple datasets and sampling methods. It evaluates both quantization error (QE) and topology preservation, combining them into a normalized overall score for comprehensive performance assessment.

## 📁 Project Structure

```
.
├── Data/                                    # Input and processed data
│   ├── pareto_front_results_v3.csv        # Input data file
│   ├── processed_data_with_pca_{arch}.csv # Processed data per architecture
│   └── best_performers_by_sampling_{arch}.csv
├── Results/                                 # All analysis outputs
│   ├── ANALYSIS_SUMMARY.md                # Main comprehensive report
│   ├── plots/                             # Visualizations
│   │   ├── algorithm_comparison.png
│   │   └── mann_whitney_all_comparisons.png
│   └── tables/                            # CSV outputs
│       ├── best_performers_compact_{arch}.csv
│       └── best_performers_complete_records_{arch}.csv
├── modules/                                # Core analysis modules
│   ├── core_analysis/
│   │   ├── clean.py                      # Data cleaning
│   │   ├── filter_best.py                # Top performer filtering
│   │   ├── normalization.py              # Metric normalization
│   │   ├── pca.py                        # PCA computation
│   │   ├── statistical_analysis.py       # Mann-Whitney U tests
│   │   ├── visualization_helpers.py      # Plotting functions
│   │   └── unified_summary.py            # Report generation
│   └── overall_score_analysis/
│       ├── calculate_overall_score.py    # Overall score computation
│       └── mann_whitney_normalized.py    # Normalized score tests
├── scripts/                                # Analysis scripts
│   ├── analysis/                         # Additional analyses
│   └── utils/                            # Utility scripts
└── main.py                                # Main orchestrator with CLI

```

## 🚀 Usage

### Basic Run
```bash
# Analyze both architectures with best performers
python main.py --data-file Data/pareto_front_results_v3.csv --best-performers --by-architecture

# Standard analysis (all architectures together)
python main.py --data-file Data/pareto_front_results_v3.csv
```

### Command-Line Options
```bash
--data-file, -d       Path to input CSV file (default: Data/pareto_front_results.csv)
--percentile, -p      Percentile cutoff for filtering (default: 20, keeps top 20%)
--no-filter           Skip filtering entirely (use all data)
--best-performers     Calculate overall scores and identify best performers
--no-stats            Skip Mann-Whitney U statistical tests
--by-architecture     Analyze each architecture separately (MST, hexagonal)
```

### Examples
```bash
# Strict filtering (top 5% performers only)
python main.py --data-file Data/pareto_front_results_v3.csv --percentile 5 --by-architecture

# Analyze without filtering
python main.py --data-file Data/pareto_front_results_v3.csv --no-filter --by-architecture

# Quick run without statistical tests
python main.py --data-file Data/pareto_front_results_v3.csv --no-stats --by-architecture
```

## 📊 Analysis Pipeline

### Step 1: Data Cleaning
- Remove rows with NaN or Inf values
- Remove rows where topographic_error = 0 (invalid topology measurements)
- Parse scenario IDs to extract algorithm, sampling method, and architecture

### Step 2: Data Filtering (Top Performers)
- Keep only the top X% performers (lowest quantization error) per dataset
- Default: 20% (configurable via --percentile)
- Purpose: Focus analysis on best-performing configurations

### Step 3: Normalization
- Min-Max normalization (0-1 range) within each dataset for:
  - Quantization error
  - All topology measures (topographic_error, neighborhood_preservation, etc.)
  - Distance to origin
- Ensures fair comparison across different datasets

### Step 4: PCA on Topology Measures
- Standardize 4 topology measures
- Perform PCA to create single topology score (PC1)
- Normalize PC1 within each dataset (0-1 range)
- PC1 typically explains ~60-70% of variance

### Step 5: Calculate Overall Score
- Formula: `Overall_Score = √(QE_normalized² + PC1_normalized²)`
- Interpretation: Euclidean distance combining both objectives
- Range: [0, √2] where lower is better

### Step 6: Statistical Analysis
- Mann-Whitney U tests comparing algorithms
- Tests performed for each sampling method
- Non-parametric test (no distribution assumptions)

### Step 7: Identify Best Performers
- Find best configuration for each processing type and sampling method
- Extract complete records for detailed analysis

## 📈 Output Files

### When using `--by-architecture`:
Files will have architecture suffixes (e.g., `_mst` or `_hexagonal`)

### Main Outputs:
- **Results/ANALYSIS_SUMMARY.md**: Comprehensive analysis report with all statistics
- **Results/plots/algorithm_comparison.png**: Multi-panel visualization
- **Results/plots/mann_whitney_all_comparisons.png**: Statistical test visualizations
- **Results/tables/best_performers_*.csv**: Best performing configurations

## 🔍 Architectures Supported
- **MST**: Minimum Spanning Tree architecture
- **Hexagonal**: Hexagonal grid architecture

## 📝 Interpretation Guide

### Overall Score
- **Lower is better** (0 = perfect, √2 ≈ 1.414 = worst)
- Balances quantization error and topology preservation
- Normalized to allow fair comparison across datasets

### Mann-Whitney U Test Results
- **p-value < 0.05**: Significant difference between algorithms
- **Effect size**: Magnitude of difference (small: 0.1, medium: 0.3, large: 0.5)
- **Negative effect size**: Batch performs better
- **Positive effect size**: Colors performs better

### Best Performers
- Identified per combination of processing type and sampling method
- Based on highest normalized overall score
- Complete records extracted for detailed analysis

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.