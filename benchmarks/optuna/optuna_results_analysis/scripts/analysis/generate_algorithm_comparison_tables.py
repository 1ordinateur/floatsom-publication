#!/usr/bin/env python3
"""
Generate structured comparison tables showing algorithm similarity for best performers.
Creates clear tables showing how similar batch and colors are for each dataset,
organized by sampling method.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def tabulate(data, headers=None, tablefmt='simple', stralign='left', numalign='left'):
    """
    Simple table formatter to replace tabulate module.
    """
    if not data:
        return ""
    
    # Calculate column widths
    col_widths = []
    if headers:
        col_widths = [len(str(h)) for h in headers]
    
    for row in data:
        for i, cell in enumerate(row):
            width = len(str(cell))
            if i >= len(col_widths):
                col_widths.append(width)
            else:
                col_widths[i] = max(col_widths[i], width)
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    
    # Build table
    lines = []
    
    if tablefmt == 'grid':
        # Top border
        lines.append("+" + "+".join("-" * w for w in col_widths) + "+")
        
        # Headers
        if headers:
            header_row = "|"
            for i, h in enumerate(headers):
                header_row += f" {str(h):^{col_widths[i]-2}} |"
            lines.append(header_row)
            lines.append("+" + "+".join("=" * w for w in col_widths) + "+")
        
        # Data rows
        for row in data:
            data_row = "|"
            for i, cell in enumerate(row):
                if stralign == 'center':
                    data_row += f" {str(cell):^{col_widths[i]-2}} |"
                else:
                    data_row += f" {str(cell):<{col_widths[i]-2}} |"
            lines.append(data_row)
        
        # Bottom border
        lines.append("+" + "+".join("-" * w for w in col_widths) + "+")
    
    else:  # simple format
        # Headers
        if headers:
            header_row = ""
            for i, h in enumerate(headers):
                header_row += f"{str(h):<{col_widths[i]}}"
            lines.append(header_row)
            lines.append("-" * sum(col_widths))
        
        # Data rows
        for row in data:
            data_row = ""
            for i, cell in enumerate(row):
                data_row += f"{str(cell):<{col_widths[i]}}"
            lines.append(data_row)
    
    return "\n".join(lines)

def load_data():
    """Load the complete records of best performers."""
    # First try Results folder, then Data folder for backward compatibility
    results_file = Path("Results/tables/best_performers_complete_records.csv")
    data_file = Path("Data/best_performers_complete_records.csv")
    
    if results_file.exists():
        return pd.read_csv(results_file)
    elif data_file.exists():
        return pd.read_csv(data_file)
    else:
        print("Error: best_performers_complete_records.csv not found")
        print("Please run: python3 scripts/utils/extract_best_performers.py first")
        return None

def calculate_similarity(percent_diff):
    """
    Categorize similarity based on percentage difference.
    Returns symbol and category.
    """
    abs_diff = abs(percent_diff)
    if abs_diff < 5:
        return "++", "IDENTICAL"
    elif abs_diff < 10:
        return "+", "VERY SIMILAR"
    elif abs_diff < 20:
        return "~", "SIMILAR"
    elif abs_diff < 40:
        return "!", "DIFFERENT"
    else:
        return "X", "VERY DIFFERENT"

def create_comparison_table_by_sampling(df, sampling_method):
    """
    Create a comparison table for a specific sampling method.
    """
    # Filter for this sampling method
    sampling_df = df[df['sampling_method'] == sampling_method]
    
    if sampling_df.empty:
        return None
    
    metrics = [
        ('QE', 'quantization_error', 'lower'),
        ('TE', 'topographic_error', 'lower'),
        ('NP', 'neighborhood_preservation', 'higher'),
        ('DM', 'distortion_measure', 'lower'),
        ('TF', 'topographic_function', 'context')
    ]
    
    datasets = sorted(sampling_df['dataset'].unique())
    
    # Create comparison data
    table_data = []
    
    for dataset in datasets:
        batch_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                 (sampling_df['processing_type'] == 'batch')]
        colors_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                  (sampling_df['processing_type'] == 'colors')]
        
        if not batch_data.empty and not colors_data.empty:
            row = [dataset.upper()]
            
            for metric_abbr, metric_name, better_direction in metrics:
                if metric_name in batch_data.columns:
                    batch_val = batch_data[metric_name].iloc[0]
                    colors_val = colors_data[metric_name].iloc[0]
                    
                    if not pd.isna(batch_val) and not pd.isna(colors_val):
                        percent_diff = ((colors_val - batch_val) / batch_val * 100) if batch_val != 0 else 0
                        symbol, category = calculate_similarity(percent_diff)
                        
                        # Determine which is better
                        if better_direction == 'lower':
                            better = 'B' if batch_val < colors_val else 'C'
                        elif better_direction == 'higher':
                            better = 'B' if batch_val > colors_val else 'C'
                        else:
                            better = '-'
                        
                        # Format: symbol (% diff) [better]
                        cell = f"{symbol} ({percent_diff:+.0f}%) [{better}]"
                        row.append(cell)
                    else:
                        row.append("N/A")
                else:
                    row.append("--")
            
            table_data.append(row)
    
    # Create headers
    headers = ['Dataset'] + [f"{abbr}" for abbr, _, _ in metrics]
    
    return table_data, headers

def create_summary_matrix(df):
    """
    Create a summary matrix showing overall similarity patterns.
    """
    metrics = [
        ('quantization_error', 'QE'),
        ('topographic_error', 'TE'),
        ('neighborhood_preservation', 'NP'),
        ('distortion_measure', 'DM'),
        ('topographic_function', 'TF')
    ]
    
    sampling_methods = ['full', 'random', 'hdsssom']
    datasets = sorted(df['dataset'].unique())
    
    # Create summary for each sampling method
    summary_data = {}
    
    for sampling in sampling_methods:
        sampling_df = df[df['sampling_method'] == sampling]
        sampling_summary = []
        
        for dataset in datasets:
            batch_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                     (sampling_df['processing_type'] == 'batch')]
            colors_data = sampling_df[(sampling_df['dataset'] == dataset) & 
                                      (sampling_df['processing_type'] == 'colors')]
            
            if not batch_data.empty and not colors_data.empty:
                dataset_similarity = []
                
                for metric_name, _ in metrics:
                    if metric_name in batch_data.columns:
                        batch_val = batch_data[metric_name].iloc[0]
                        colors_val = colors_data[metric_name].iloc[0]
                        
                        if not pd.isna(batch_val) and not pd.isna(colors_val):
                            percent_diff = ((colors_val - batch_val) / batch_val * 100) if batch_val != 0 else 0
                            _, category = calculate_similarity(percent_diff)
                            
                            # Convert to numeric score
                            if category == "IDENTICAL":
                                score = 5
                            elif category == "VERY SIMILAR":
                                score = 4
                            elif category == "SIMILAR":
                                score = 3
                            elif category == "DIFFERENT":
                                score = 2
                            else:  # VERY DIFFERENT
                                score = 1
                            
                            dataset_similarity.append(score)
                
                # Calculate average similarity
                avg_similarity = np.mean(dataset_similarity) if dataset_similarity else 0
                sampling_summary.append((dataset, avg_similarity))
        
        summary_data[sampling] = sampling_summary
    
    return summary_data

def generate_report(df):
    """
    Generate comprehensive comparison report with tables in Markdown format.
    """
    results_dir = Path("Results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "algorithm_comparison_tables.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Algorithm Comparison Tables for Best Performers\n")
        f.write("## Batch vs Colors Processing\n\n")
        
        f.write("## Legend\n\n")
        f.write("### Similarity Symbols\n")
        f.write("| Symbol | Meaning | Difference Range |\n")
        f.write("|--------|---------|------------------|\n")
        f.write("| ++ | Identical | <5% |\n")
        f.write("| + | Very Similar | 5-10% |\n")
        f.write("| ~ | Similar | 10-20% |\n")
        f.write("| ! | Different | 20-40% |\n")
        f.write("| X | Very Different | >40% |\n\n")
        
        f.write("### Metrics\n")
        f.write("| Abbreviation | Full Name | Optimization |\n")
        f.write("|--------------|-----------|-------------|\n")
        f.write("| QE | Quantization Error | Lower is better |\n")
        f.write("| TE | Topographic Error | Lower is better |\n")
        f.write("| NP | Neighborhood Preservation | Higher is better |\n")
        f.write("| DM | Distortion Measure | Lower is better |\n")
        f.write("| TF | Topographic Function | Context-dependent |\n\n")
        
        f.write("**Format**: Symbol (% diff) [B=Batch better, C=Colors better]\n\n")
        f.write("---\n\n")
        
        # Create tables for each sampling method
        for sampling in ['full', 'random', 'hdsssom']:
            f.write(f"## Sampling Method: {sampling.upper()}\n\n")
            
            table_data, headers = create_comparison_table_by_sampling(df, sampling)
            
            if table_data:
                # Create Markdown table
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["-"*max(15, len(h)) for h in headers]) + "|\n")
                
                for row in table_data:
                    f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
                
                f.write("\n")
                
                # Add summary statistics
                f.write("### Summary Statistics\n\n")
                
                # Count similarity levels for each metric
                for col_idx, metric in enumerate(['QE', 'TE', 'NP', 'DM', 'TF']):
                    identical = sum(1 for row in table_data if '++' in row[col_idx + 1])
                    very_similar = sum(1 for row in table_data if '+' in row[col_idx + 1] and '++' not in row[col_idx + 1])
                    similar = sum(1 for row in table_data if '~' in row[col_idx + 1])
                    different = sum(1 for row in table_data if '!' in row[col_idx + 1])
                    very_different = sum(1 for row in table_data if 'X' in row[col_idx + 1])
                    
                    total = len(table_data)
                    
                    f.write(f"**{metric}:**\n")
                    stats = []
                    if identical > 0:
                        stats.append(f"Identical: {identical}/{total} ({identical/total*100:.0f}%)")
                    if very_similar > 0:
                        stats.append(f"Very Similar: {very_similar}/{total} ({very_similar/total*100:.0f}%)")
                    if similar > 0:
                        stats.append(f"Similar: {similar}/{total} ({similar/total*100:.0f}%)")
                    if different > 0:
                        stats.append(f"Different: {different}/{total} ({different/total*100:.0f}%)")
                    if very_different > 0:
                        stats.append(f"Very Different: {very_different}/{total} ({very_different/total*100:.0f}%)")
                    
                    if stats:
                        f.write("- " + "\n- ".join(stats) + "\n\n")
            else:
                f.write("*No data available for this sampling method*\n\n")
        
        # Create overall metric summary across all sampling methods
        f.write("---\n\n")
        f.write("## Overall Metric Summary (All Sampling Methods Combined)\n\n")
        f.write("Summary of how similar Batch and Colors are for each metric across all sampling methods.\n\n")
        
        # Collect all comparisons across all sampling methods
        all_metric_stats = {'QE': [], 'TE': [], 'NP': [], 'DM': [], 'TF': []}
        
        for sampling in ['full', 'random', 'hdsssom']:
            table_data, headers = create_comparison_table_by_sampling(df, sampling)
            if table_data:
                for col_idx, metric in enumerate(['QE', 'TE', 'NP', 'DM', 'TF']):
                    for row in table_data:
                        cell = row[col_idx + 1]
                        if '++' in cell:
                            all_metric_stats[metric].append('Identical')
                        elif '+' in cell and '++' not in cell:
                            all_metric_stats[metric].append('Very Similar')
                        elif '~' in cell:
                            all_metric_stats[metric].append('Similar')
                        elif '!' in cell:
                            all_metric_stats[metric].append('Different')
                        elif 'X' in cell:
                            all_metric_stats[metric].append('Very Different')
        
        # Create summary table
        f.write("| Metric | Identical | Very Similar | Similar | Different | Very Different | Total | Similarity Score |\n")
        f.write("|--------|-----------|--------------|---------|-----------|----------------|-------|------------------|\n")
        
        metric_names = {
            'QE': 'Quantization Error',
            'TE': 'Topographic Error',
            'NP': 'Neighborhood Preservation',
            'DM': 'Distortion Measure',
            'TF': 'Topographic Function'
        }
        
        for metric_abbr, similarities in all_metric_stats.items():
            if similarities:
                total = len(similarities)
                identical = similarities.count('Identical')
                very_similar = similarities.count('Very Similar')
                similar = similarities.count('Similar')
                different = similarities.count('Different')
                very_different = similarities.count('Very Different')
                
                # Calculate weighted similarity score (5=identical, 1=very different)
                score = (identical*5 + very_similar*4 + similar*3 + different*2 + very_different*1) / total if total > 0 else 0
                
                # Determine overall assessment
                if score >= 4:
                    assessment = "Highly Similar"
                elif score >= 3:
                    assessment = "Moderately Similar"
                elif score >= 2:
                    assessment = "Mixed"
                else:
                    assessment = "Very Different"
                
                f.write(f"| **{metric_abbr}** | {identical} ({identical/total*100:.0f}%) | "
                       f"{very_similar} ({very_similar/total*100:.0f}%) | "
                       f"{similar} ({similar/total*100:.0f}%) | "
                       f"{different} ({different/total*100:.0f}%) | "
                       f"{very_different} ({very_different/total*100:.0f}%) | "
                       f"{total} | {score:.2f} - {assessment} |\n")
        
        f.write("\n")
        f.write("**Key Insights:**\n")
        
        # Find most and least similar metrics
        metric_scores = []
        for metric_abbr, similarities in all_metric_stats.items():
            if similarities:
                total = len(similarities)
                score = (similarities.count('Identical')*5 + 
                        similarities.count('Very Similar')*4 + 
                        similarities.count('Similar')*3 + 
                        similarities.count('Different')*2 + 
                        similarities.count('Very Different')*1) / total
                metric_scores.append((metric_abbr, score))
        
        if metric_scores:
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"- **Most Similar Metric**: {metric_names[metric_scores[0][0]]} (score: {metric_scores[0][1]:.2f})\n")
            f.write(f"- **Least Similar Metric**: {metric_names[metric_scores[-1][0]]} (score: {metric_scores[-1][1]:.2f})\n")
            
            # Count metrics by similarity level
            highly_similar = sum(1 for _, score in metric_scores if score >= 4)
            moderately_similar = sum(1 for _, score in metric_scores if 3 <= score < 4)
            mixed = sum(1 for _, score in metric_scores if 2 <= score < 3)
            very_different = sum(1 for _, score in metric_scores if score < 2)
            
            f.write(f"- **Overall Pattern**: ")
            if highly_similar >= 3:
                f.write("Algorithms are generally very similar across most metrics\n")
            elif moderately_similar >= 3:
                f.write("Algorithms show moderate similarity across metrics\n")
            elif mixed >= 3:
                f.write("Algorithms show mixed behavior - similar on some metrics, different on others\n")
            else:
                f.write("Algorithms behave quite differently across metrics\n")
        
        f.write("\n")
        
        # Create overall summary matrix
        f.write("---\n\n")
        f.write("## Overall Similarity Matrix\n\n")
        f.write("Average similarity score across all 5 metrics for each dataset and sampling method.\n\n")
        
        summary_data = create_summary_matrix(df)
        
        # Create Markdown matrix table
        f.write("| Dataset | Full | Random | HDSSSOM |\n")
        f.write("|---------|------|--------|----------|\n")
        
        datasets = sorted(df['dataset'].unique())
        for dataset in datasets:
            row = [dataset.upper()]
            for sampling in ['full', 'random', 'hdsssom']:
                # Find similarity score for this dataset/sampling
                sampling_scores = summary_data.get(sampling, [])
                score = next((s for d, s in sampling_scores if d == dataset), 0)
                
                if score >= 4.5:
                    symbol = "++"
                elif score >= 3.5:
                    symbol = "+"
                elif score >= 2.5:
                    symbol = "~"
                elif score >= 1.5:
                    symbol = "!"
                elif score > 0:
                    symbol = "X"
                else:
                    symbol = "--"
                
                row.append(f"{symbol} ({score:.1f})")
            
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n")
        f.write("**Score Key:**\n")
        f.write("- 5 = Identical\n")
        f.write("- 4 = Very Similar\n")
        f.write("- 3 = Similar\n")
        f.write("- 2 = Different\n")
        f.write("- 1 = Very Different\n\n")
        f.write("*(Average across all 5 metrics)*\n")
    
    print(f"Report saved to {output_file}")
    return output_file

def create_console_output(df):
    """
    Create formatted console output for immediate viewing.
    """
    print("\n" + "="*80)
    print("ALGORITHM SIMILARITY COMPARISON (BATCH vs COLORS)")
    print("="*80)
    
    for sampling in ['full', 'random', 'hdsssom']:
        print(f"\n{'='*70}")
        print(f"SAMPLING: {sampling.upper()}")
        print(f"{'='*70}")
        
        table_data, headers = create_comparison_table_by_sampling(df, sampling)
        
        if table_data:
            # Simplify for console output
            simplified_data = []
            for row in table_data:
                simplified_row = [row[0][:10]]  # Truncate dataset name
                for cell in row[1:]:
                    if '++' in cell:
                        simplified_row.append('SAME')
                    elif '+' in cell and '++' not in cell:
                        simplified_row.append('VSIM')
                    elif '~' in cell:
                        simplified_row.append('SIM')
                    elif '!' in cell:
                        simplified_row.append('DIFF')
                    elif 'X' in cell:
                        simplified_row.append('VDIFF')
                    else:
                        simplified_row.append('--')
                simplified_data.append(simplified_row)
            
            print(tabulate(simplified_data, headers=headers, tablefmt='simple'))
            
            # Quick summary
            total_comparisons = len(table_data) * 5  # 5 metrics
            very_similar_count = sum(1 for row in table_data for cell in row[1:] if '+' in cell)
            different_count = sum(1 for row in table_data for cell in row[1:] if '!' in cell or 'X' in cell)
            
            print(f"\nQuick Summary:")
            print(f"  Similar or better: {very_similar_count}/{total_comparisons} ({very_similar_count/total_comparisons*100:.0f}%)")
            print(f"  Different: {different_count}/{total_comparisons} ({different_count/total_comparisons*100:.0f}%)")

def main():
    """
    Main function to generate comparison tables.
    """
    print("="*80)
    print("GENERATING ALGORITHM COMPARISON TABLES")
    print("="*80)
    
    # Load data
    print("\nLoading best performer data...")
    df = load_data()
    
    if df is None:
        return
    
    print(f"Loaded {len(df)} records")
    
    # Generate report
    report_file = generate_report(df)
    
    # Show console output
    create_console_output(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nDetailed Markdown report saved to: {report_file}")
    print("\nKey: SAME=Identical, VSIM=Very Similar, SIM=Similar, DIFF=Different, VDIFF=Very Different")

if __name__ == "__main__":
    main()