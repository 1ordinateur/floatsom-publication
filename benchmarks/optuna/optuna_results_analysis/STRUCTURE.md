# Project Structure Explanation

## Directory Organization

### `modules/` - Core Pipeline Components
Contains the essential functionality used by `main.py` for the primary analysis pipeline.

#### `modules/core_analysis/`
- **clean.py** - Data cleaning and validation
- **filter_best.py** - Filter top performers by quantization error
- **normalization.py** - Normalize metrics within datasets
- **pca.py** - PCA computation for topology measures
- **statistical_analysis.py** - Mann-Whitney U tests and statistical comparisons
- **visualization_helpers.py** - Plotting and visualization functions
- **unified_summary.py** - Generate comprehensive analysis reports
- **extract_best_performers.py** - Extract and summarize best performing configurations

#### `modules/overall_score_analysis/`
- **calculate_overall_score.py** - Compute combined performance scores
- **calculate_normalized_overall_score.py** - Normalized score calculations
- **mann_whitney_normalized.py** - Statistical tests on normalized scores
- **report_best_performers.py** - Generate best performer reports

### `scripts/` - Standalone Analysis Scripts
Contains independent scripts for additional analyses that can be run separately from the main pipeline.

#### `scripts/analysis/`
- **analyze_individual_topology.py** - Deep dive into individual topology metrics
- **analyze_topology_metrics_breakdown.py** - Detailed topology metric analysis
- **analyze_topology_patterns.py** - Pattern analysis across topology measures
- **generate_algorithm_comparison_tables.py** - Create detailed comparison tables

## Key Distinction

- **`modules/`**: Imported by `main.py`, part of the core pipeline
- **`scripts/`**: Standalone scripts for additional/specialized analyses

## Usage

### Main Pipeline
```bash
python main.py --data-file Data/pareto_front_results_v3.csv --best-performers --by-architecture
```

### Additional Analyses
```bash
python scripts/analysis/analyze_topology_patterns.py
python scripts/analysis/analyze_topology_metrics_breakdown.py
```