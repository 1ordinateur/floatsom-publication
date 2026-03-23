"""
Report generation system for Phase 3 benchmark analysis.

This module provides comprehensive report generation including
HTML reports, summary tables, and statistical analysis reports.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd


class ReportGenerator:
    """
    Generates comprehensive analysis reports from benchmark results.
    
    Creates HTML reports, summary tables, and statistical analysis
    reports for easy interpretation of benchmark results.
    """
    
    def __init__(self, results_dir: Path, analysis_data: Dict[str, Any]):
        self.results_dir = Path(results_dir)
        self.analysis_data = analysis_data
        self.logger = logging.getLogger(__name__)
        
    def generate_html_report(self, output_file: Optional[Path] = None) -> Path:
        """Generate comprehensive HTML report."""
        if output_file is None:
            output_file = self.results_dir / "analysis_report.html"
            
        html_content = self._build_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Generated HTML report: {output_file}")
        return output_file
    
    def generate_summary_table(self, output_file: Optional[Path] = None) -> Path:
        """Generate algorithm performance summary table."""
        if output_file is None:
            output_file = self.results_dir / "algorithm_summary.csv"
            
        # Extract algorithm rankings data
        rankings = self.analysis_data.get('algorithm_rankings', [])
        
        df = pd.DataFrame([
            {
                'Algorithm': ranking['algorithm'],
                'Mean Performance': f"{ranking['mean_performance']:.6f}",
                'Std Performance': f"{ranking['std_performance']:.6f}",
                'Win Rate': f"{ranking['win_rate']:.3f}",
                'Stability Score': f"{ranking['stability_score']:.6f}" if ranking['stability_score'] != float('inf') else 'N/A'
            }
            for ranking in rankings
        ])
        
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated summary table: {output_file}")
        return output_file
    
    def generate_pairwise_comparison_report(self, output_file: Optional[Path] = None) -> Path:
        """Generate pairwise comparison report."""
        if output_file is None:
            output_file = self.results_dir / "pairwise_comparisons.csv"
            
        comparisons = self.analysis_data.get('pairwise_comparisons', {})
        
        rows = []
        for comp_name, comp_data in comparisons.items():
            rows.append({
                'Comparison': comp_name,
                'Algorithm 1': comp_data['algorithm_1'],
                'Algorithm 2': comp_data['algorithm_2'],
                'Mean 1': f"{comp_data['mean_1']:.6f}",
                'Mean 2': f"{comp_data['mean_2']:.6f}",
                'P-Value': f"{comp_data['p_value']:.6f}" if not pd.isna(comp_data['p_value']) else 'N/A',
                'Effect Size (Cohen\'s d)': f"{comp_data['effect_size_cohens_d']:.4f}" if not pd.isna(comp_data['effect_size_cohens_d']) else 'N/A',
                'Significant (α=0.05)': comp_data['significant_at_0.05'],
                'Better Algorithm': comp_data['better_algorithm']
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated pairwise comparison report: {output_file}")
        return output_file
    
    def generate_dataset_performance_report(self, output_file: Optional[Path] = None) -> Path:
        """Generate dataset-specific performance report."""
        if output_file is None:
            output_file = self.results_dir / "dataset_performance.csv"
            
        dataset_performance = self.analysis_data.get('dataset_performance', {})
        
        rows = []
        for dataset_name, algorithms in dataset_performance.items():
            for algorithm, stats in algorithms.items():
                rows.append({
                    'Dataset': dataset_name,
                    'Algorithm': algorithm,
                    'Mean': f"{stats['mean']:.6f}",
                    'Std': f"{stats['std']:.6f}",
                    'Median': f"{stats['median']:.6f}",
                    'Best': f"{stats['best']:.6f}",
                    'Worst': f"{stats['worst']:.6f}",
                    'Count': stats['count']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated dataset performance report: {output_file}")
        return output_file
    
    def _build_html_report(self) -> str:
        """Build comprehensive HTML report."""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FloatSOM Phase 3 Benchmark Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .winner {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .significant {{
            background-color: #fff3cd;
        }}
        .metric {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .timestamp {{
            color: #666;
            font-style: italic;
        }}
        .section {{
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FloatSOM Phase 3 Benchmark Analysis Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {self._build_executive_summary()}
        {self._build_algorithm_rankings_section()}
        {self._build_pairwise_comparisons_section()}
        {self._build_dataset_analysis_section()}
        {self._build_parameter_analysis_section()}
        {self._build_recommendations_section()}
        
    </div>
</body>
</html>
"""
        return html
    
    def _build_executive_summary(self) -> str:
        """Build executive summary section."""
        best_algorithm = self.analysis_data.get('best_overall_algorithm', 'Unknown')
        most_stable = self.analysis_data.get('most_stable_algorithm', 'Unknown')
        total_scenarios = self.analysis_data.get('total_scenarios', 0)
        total_algorithms = self.analysis_data.get('total_algorithms', 0)
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Best Overall Algorithm</h3>
                    <p class="metric">{best_algorithm}</p>
                </div>
                <div class="summary-card">
                    <h3>Most Stable Algorithm</h3>
                    <p class="metric">{most_stable}</p>
                </div>
                <div class="summary-card">
                    <h3>Total Scenarios</h3>
                    <p class="metric">{total_scenarios}</p>
                </div>
                <div class="summary-card">
                    <h3>Algorithms Tested</h3>
                    <p class="metric">{total_algorithms}</p>
                </div>
            </div>
        </div>
        """
    
    def _build_algorithm_rankings_section(self) -> str:
        """Build algorithm rankings section."""
        rankings = self.analysis_data.get('algorithm_rankings', [])
        
        if not rankings:
            return "<div class='section'><h2>Algorithm Rankings</h2><p>No ranking data available.</p></div>"
        
        table_rows = ""
        for i, ranking in enumerate(rankings):
            row_class = "winner" if i == 0 else ""
            stability = ranking['stability_score']
            stability_str = f"{stability:.6f}" if stability != float('inf') else 'N/A'
            
            table_rows += f"""
            <tr class="{row_class}">
                <td>{i + 1}</td>
                <td>{ranking['algorithm']}</td>
                <td>{ranking['mean_performance']:.6f}</td>
                <td>{ranking['std_performance']:.6f}</td>
                <td>{ranking['win_rate']:.3f}</td>
                <td>{stability_str}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Algorithm Performance Rankings</h2>
            <p>Algorithms ranked by mean performance across all scenarios (lower is better).</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Algorithm</th>
                        <th>Mean Performance</th>
                        <th>Std Performance</th>
                        <th>Win Rate</th>
                        <th>Stability Score</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _build_pairwise_comparisons_section(self) -> str:
        """Build pairwise comparisons section."""
        comparisons = self.analysis_data.get('pairwise_comparisons', {})
        
        if not comparisons:
            return "<div class='section'><h2>Pairwise Comparisons</h2><p>No comparison data available.</p></div>"
        
        table_rows = ""
        for comp_name, comp_data in comparisons.items():
            row_class = "significant" if comp_data['significant_at_0.05'] else ""
            p_value = f"{comp_data['p_value']:.6f}" if not pd.isna(comp_data['p_value']) else 'N/A'
            effect_size = f"{comp_data['effect_size_cohens_d']:.4f}" if not pd.isna(comp_data['effect_size_cohens_d']) else 'N/A'
            
            table_rows += f"""
            <tr class="{row_class}">
                <td>{comp_data['algorithm_1']} vs {comp_data['algorithm_2']}</td>
                <td>{comp_data['mean_1']:.6f}</td>
                <td>{comp_data['mean_2']:.6f}</td>
                <td>{p_value}</td>
                <td>{effect_size}</td>
                <td>{'Yes' if comp_data['significant_at_0.05'] else 'No'}</td>
                <td>{comp_data['better_algorithm']}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Statistical Pairwise Comparisons</h2>
            <p>Statistical significance tests between algorithm pairs. Highlighted rows show statistically significant differences (α=0.05).</p>
            <table>
                <thead>
                    <tr>
                        <th>Comparison</th>
                        <th>Mean 1</th>
                        <th>Mean 2</th>
                        <th>P-Value</th>
                        <th>Effect Size (Cohen's d)</th>
                        <th>Significant</th>
                        <th>Better Algorithm</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _build_dataset_analysis_section(self) -> str:
        """Build dataset-specific analysis section."""
        dataset_performance = self.analysis_data.get('dataset_performance', {})
        
        if not dataset_performance:
            return "<div class='section'><h2>Dataset Analysis</h2><p>No dataset analysis available.</p></div>"
        
        dataset_sections = ""
        for dataset_name, algorithms in dataset_performance.items():
            table_rows = ""
            
            # Sort algorithms by mean performance for this dataset
            sorted_algorithms = sorted(algorithms.items(), key=lambda x: x[1]['mean'])
            
            for i, (algorithm, stats) in enumerate(sorted_algorithms):
                row_class = "winner" if i == 0 else ""
                table_rows += f"""
                <tr class="{row_class}">
                    <td>{algorithm}</td>
                    <td>{stats['mean']:.6f}</td>
                    <td>{stats['std']:.6f}</td>
                    <td>{stats['median']:.6f}</td>
                    <td>{stats['best']:.6f}</td>
                    <td>{stats['worst']:.6f}</td>
                    <td>{stats['count']}</td>
                </tr>
                """
            
            dataset_sections += f"""
            <h3>{dataset_name}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Median</th>
                        <th>Best</th>
                        <th>Worst</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """
        
        return f"""
        <div class="section">
            <h2>Dataset-Specific Performance Analysis</h2>
            <p>Performance breakdown by dataset. Best performing algorithm for each dataset is highlighted.</p>
            {dataset_sections}
        </div>
        """
    
    def _build_parameter_analysis_section(self) -> str:
        """Build parameter combination analysis section."""
        parameter_performance = self.analysis_data.get('parameter_performance', {})
        
        if not parameter_performance:
            return "<div class='section'><h2>Parameter Analysis</h2><p>No parameter analysis available.</p></div>"
        
        table_rows = ""
        for param_combo, algorithms in parameter_performance.items():
            # Find best algorithm for this parameter combination
            best_algo = min(algorithms.items(), key=lambda x: x[1]['mean'])[0]
            
            for algorithm, stats in algorithms.items():
                row_class = "winner" if algorithm == best_algo else ""
                table_rows += f"""
                <tr class="{row_class}">
                    <td>{param_combo}</td>
                    <td>{algorithm}</td>
                    <td>{stats['mean']:.6f}</td>
                    <td>{stats['std']:.6f}</td>
                    <td>{stats['count']}</td>
                </tr>
                """
        
        return f"""
        <div class="section">
            <h2>Parameter Combination Analysis</h2>
            <p>Performance by sampling method and topology combinations. Best performing algorithm for each combination is highlighted.</p>
            <table>
                <thead>
                    <tr>
                        <th>Parameter Combination</th>
                        <th>Algorithm</th>
                        <th>Mean Performance</th>
                        <th>Std Performance</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _build_recommendations_section(self) -> str:
        """Build recommendations section."""
        best_algorithm = self.analysis_data.get('best_overall_algorithm', 'Unknown')
        most_stable = self.analysis_data.get('most_stable_algorithm', 'Unknown')
        
        recommendations = []
        
        if best_algorithm != 'Unknown':
            recommendations.append(f"<strong>Overall Performance:</strong> {best_algorithm} shows the best overall performance across all scenarios.")
        
        if most_stable != 'Unknown':
            recommendations.append(f"<strong>Stability:</strong> {most_stable} demonstrates the most consistent performance across different runs.")
        
        # Analyze pairwise comparisons for significant differences
        comparisons = self.analysis_data.get('pairwise_comparisons', {})
        significant_comparisons = [comp for comp in comparisons.values() if comp['significant_at_0.05']]
        
        if significant_comparisons:
            recommendations.append(f"<strong>Statistical Significance:</strong> {len(significant_comparisons)} algorithm pairs show statistically significant performance differences.")
        
        recommendations_html = "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
        
        return f"""
        <div class="section">
            <h2>Recommendations</h2>
            {recommendations_html}
            <p><strong>Note:</strong> These recommendations are based on the specific datasets and parameter combinations tested. 
            Consider your specific use case requirements when selecting an algorithm.</p>
        </div>
        """