"""
Unified markdown summary replicating the legacy layout while incorporating new metric requirements.
"""

from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd

from floatsom_benchmarks.optuna.optuna_results_analysis.modules.core_analysis import dataset_groups


def _resolve(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _format_value(value):
    if pd.isna(value):
        return 'N/A'
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _processing_display(row, primary_col):
    candidates = []
    if primary_col:
        candidates.append(primary_col)
    candidates.extend(['config_processing_method', 'processing_type'])

    seen = []
    for col in candidates:
        if col and col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                seen.append(val)

    if not seen:
        return 'N/A'

    for val in seen:
        lower = val.lower()
        if 'mini' in lower:
            return 'Mini-batch'
        if 'batch' in lower:
            return 'Batch'
    fallback = seen[0].replace('_', ' ').strip()
    return fallback.title() if fallback else 'N/A'


def _format_algorithm(value):
    if pd.isna(value):
        return 'N/A'
    text = str(value).strip()
    if not text:
        return 'N/A'
    return text.replace('_', ' ').title()


def _format_sampling(value):
    if pd.isna(value):
        return 'N/A'
    text = str(value).strip()
    if not text:
        return 'N/A'
    text = text.replace('_', ' ')
    return text.title()


def _format_batch_mode(value):
    if pd.isna(value):
        return 'N/A'
    text = str(value).strip()
    if not text:
        return 'N/A'
    lower = text.lower()
    if 'mini' in lower:
        return 'Mini-batch'
    if 'full' in lower or 'batch' in lower:
        return 'Full batch'
    return text.replace('_', ' ').title()


def _format_architecture(value):
    if pd.isna(value):
        return 'N/A'
    text = str(value).strip()
    if not text:
        return 'N/A'
    return text.replace('_', ' ').title()


def _format_dataset_group(value):
    group = dataset_groups.dataset_group(value)
    if group == "synthetic":
        return "Synthetic"
    if group == "real":
        return "Real"
    return "Global"


def _format_metric_summary(df, metrics):
    rows = ["| Metric | Min | Max | Mean | Std Dev |", "|--------|-----|-----|------|---------|"]
    for col, label in metrics:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        rows.append(f"| {label} | {series.min():.4f} | {series.max():.4f} | {series.mean():.4f} | {series.std():.4f} |")
    return "\n".join(rows)


def _best_group_table(df, metric_config, group_cols, extra_cols):
    metric_col = metric_config['norm_col']
    metric_label_norm = metric_config['norm_label']
    raw_col = metric_config.get('raw_col')
    raw_label = metric_config.get('raw_label')

    if metric_col not in df.columns:
        return pd.DataFrame()

    metric_df = df.dropna(subset=[metric_col]).copy()
    if metric_df.empty:
        return pd.DataFrame()

    idx = metric_df.groupby(group_cols, dropna=False)[metric_col].idxmin()
    best = metric_df.loc[idx].copy()
    best = best.sort_values(group_cols)
    best[metric_label_norm] = best[metric_col].apply(_format_value)
    has_raw = raw_col and raw_label and raw_col in best.columns
    if has_raw:
        best[raw_label] = best[raw_col].apply(_format_value)

    ordered_cols = []
    for col in list(group_cols) + list(extra_cols or []):
        if col in best.columns and col not in ordered_cols:
            ordered_cols.append(col)
    if has_raw and raw_label not in ordered_cols:
        ordered_cols.append(raw_label)
    if metric_label_norm not in ordered_cols:
        ordered_cols.append(metric_label_norm)

    return best[ordered_cols]


def _summary_table(best_records, metrics):
    if best_records.empty:
        return ""

    columns = ['Dataset']
    for metric in metrics:
        columns.append(metric['norm_label'])

    summary_rows = []
    datasets = dataset_groups.ordered_dataset_labels(best_records['Dataset'].unique())
    for dataset in datasets:
        row = {'Dataset': dataset}
        dataset_rows = best_records[best_records['Dataset'] == dataset]
        for metric in metrics:
            label = metric['norm_label']
            match = dataset_rows[dataset_rows['Metric'] == label]
            if not match.empty:
                info = match.iloc[0]
                parts = [
                    info.get('Algorithm', 'N/A'),
                    info.get('Processing Mode', 'N/A'),
                    info.get('Sampling', 'N/A'),
                    info.get('Batch Mode', 'N/A'),
                    info.get('Architecture', 'N/A'),
                    f"raw={_format_value(info.get('Raw Value', float('nan')))}",
                    f"norm={_format_value(info.get('Value', float('nan')))}"
                ]
                row[label] = " / ".join(parts)
            else:
                row[label] = "N/A"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=columns).to_markdown(index=False)


def _sample_counts(df, grouping):
    count_df = (
        df.groupby(grouping)
        .size()
        .reset_index(name='Count')
        .sort_values(grouping)
    )
    return count_df.to_markdown(index=False)

def _format_category(df, column, formatter, default_value='N/A'):
    if column and column in df.columns:
        return df[column].apply(formatter)
    return pd.Series([default_value] * len(df), index=df.index)

def _compute_best_by_group(df, group_cols, value_col):
    if not all(col in df.columns for col in group_cols + [value_col]):
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    try:
        best_indices = grouped.idxmin()
    except ValueError:
        return pd.DataFrame()
    best = df.loc[best_indices].copy()
    best['__raw_value__'] = best[value_col]
    return best

def _winner_summary(best_df, classification_col, value_col='__raw_value__', label_formatter=lambda x: x, float_tolerance=1e-9):
    dataset_col = 'Dataset'
    if dataset_col not in best_df.columns or classification_col not in best_df.columns or value_col not in best_df.columns:
        return "", ""

    summary_rows = []
    win_counts = defaultdict(float)

    for dataset, group in best_df.groupby(dataset_col):
        if group.empty:
            continue
        values = pd.to_numeric(group[value_col], errors='coerce').dropna()
        if values.empty:
            continue
        min_value = values.min()
        winners = group[pd.to_numeric(group[value_col], errors='coerce') == min_value][classification_col].tolist()
        winners_formatted = [label_formatter(w) for w in winners] if winners else []
        if winners_formatted:
            increment = 1.0 / len(winners_formatted)
            for winner in winners_formatted:
                win_counts[winner] += increment
        summary_rows.append({
            'Dataset': dataset,
            'Winner(s)': ", ".join(winners_formatted) if winners_formatted else 'N/A',
            'Best Score': f"{min_value:.4f}"
        })

    dataset_summary = pd.DataFrame(summary_rows, columns=['Dataset', 'Winner(s)', 'Best Score'])
    win_table = pd.DataFrame(
        [{'Category': key, 'Win Count': f"{value:.2f}"} for key, value in sorted(win_counts.items())],
        columns=['Category', 'Win Count']
    )
    return dataset_summary.to_markdown(index=False) if not dataset_summary.empty else "", \
        win_table.to_markdown(index=False) if not win_table.empty else ""

def _sampling_section(df, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col):
    if not sampling_col or dataset_col not in df.columns:
        return []

    metric_specs = [
        ('quantization_error_train_normalized', 'QE Train (norm)'),
        ('quantization_error_holdout_normalized', 'QE Holdout (norm)'),
        ('balanced_qe_normalized', 'Balanced QE (norm)'),
        (score_col, 'Combined Distance (norm)')
    ]

    available_metrics = [(col, label) for col, label in metric_specs if col and col in df.columns]
    if not available_metrics:
        return []

    base = df.dropna(subset=[dataset_col, sampling_col]).copy()
    if base.empty:
        return []

    lines = ["## Sampling Method Comparisons", ""]

    for metric_col, metric_label in available_metrics:
        metric_base = base.dropna(subset=[metric_col])
        if metric_base.empty:
            continue

        best_metric = _compute_best_by_group(metric_base, [dataset_col, sampling_col], metric_col)
        if best_metric.empty:
            continue

        display = best_metric.copy()
        display['Dataset'] = display[dataset_col]
        display['Sampling'] = display[sampling_col].apply(_format_sampling)
        display[metric_label] = display[metric_col].apply(_format_value)

        columns = ['Dataset', 'Sampling', metric_label]
        if architecture_col and architecture_col in display.columns:
            display['Winning Architecture'] = display[architecture_col].apply(_format_architecture)
            columns.append('Winning Architecture')
        if algorithm_col and algorithm_col in display.columns:
            display['Winning Algorithm'] = display[algorithm_col].apply(_format_algorithm)
            columns.append('Winning Algorithm')
        if batch_mode_col and batch_mode_col in display.columns:
            display['Batch Mode'] = display[batch_mode_col].apply(_format_batch_mode)
            columns.append('Batch Mode')

        lines.append(f"### {metric_label} Winners by Dataset")
        lines.append(display[columns].sort_values(['Dataset', 'Sampling']).to_markdown(index=False))
        lines.append("")

        dataset_summary_md, win_counts_md = _winner_summary(display[['Dataset', 'Sampling', '__raw_value__']].copy(), 'Sampling')
        if dataset_summary_md:
            lines.append(f"#### Dataset Winners – {metric_label}")
            lines.append(dataset_summary_md)
            lines.append("")
        if win_counts_md:
            lines.append(f"#### Sampling Win Counts – {metric_label}")
            lines.append(win_counts_md)
            lines.append("")

        if architecture_col and architecture_col in metric_base.columns:
            best_arch = _compute_best_by_group(metric_base, [dataset_col, architecture_col, sampling_col], metric_col)
            if not best_arch.empty:
                lines.append(f"#### {metric_label} by Architecture")
                for arch_value in sorted(best_arch[architecture_col].dropna().unique()):
                    subset = best_arch[best_arch[architecture_col] == arch_value].copy()
                    subset['Dataset'] = subset[dataset_col]
                    subset['Sampling'] = subset[sampling_col].apply(_format_sampling)
                    subset[metric_label] = subset[metric_col].apply(_format_value)
                    subset['Architecture'] = subset[architecture_col].apply(_format_architecture)
                    columns_arch = ['Dataset', 'Sampling', metric_label, 'Architecture']
                    if algorithm_col and algorithm_col in subset.columns:
                        subset['Winning Algorithm'] = subset[algorithm_col].apply(_format_algorithm)
                        columns_arch.append('Winning Algorithm')
                    if batch_mode_col and batch_mode_col in subset.columns:
                        subset['Batch Mode'] = subset[batch_mode_col].apply(_format_batch_mode)
                        columns_arch.append('Batch Mode')

                    lines.append(f"##### Architecture: {_format_architecture(arch_value)}")
                    lines.append(subset[columns_arch].sort_values(['Dataset', 'Sampling']).to_markdown(index=False))
                    lines.append("")

                    subset_summary = subset[['Dataset', 'Sampling', '__raw_value__']].copy()
                    ds_summary_md, count_summary_md = _winner_summary(subset_summary, 'Sampling')
                    if ds_summary_md:
                        lines.append(f"###### Dataset Winners – {metric_label} ({_format_architecture(arch_value)})")
                        lines.append(ds_summary_md)
                        lines.append("")
                    if count_summary_md:
                        lines.append(f"###### Sampling Win Counts – {metric_label} ({_format_architecture(arch_value)})")
                        lines.append(count_summary_md)
                        lines.append("")

    return lines

def _architecture_section(df, dataset_col, architecture_col, algorithm_col):
    metric_columns = [
        ('quantization_error_train_normalized', 'QE Train (norm)'),
        ('quantization_error_holdout_normalized', 'QE Holdout (norm)')
    ]
    available_metrics = [(col, label) for col, label in metric_columns if col in df.columns]
    if not available_metrics or architecture_col not in df.columns or dataset_col not in df.columns:
        return []

    lines = ["## Architecture Comparisons (QE Metrics)", ""]

    base = df.dropna(subset=[dataset_col, architecture_col]).copy()
    if base.empty:
        return []

    arch_summary = base.groupby([dataset_col, architecture_col], dropna=False)[[col for col, _ in available_metrics]].min().reset_index()
    if arch_summary.empty:
        return []

    display = arch_summary.copy()
    display['Dataset'] = display[dataset_col]
    display['Architecture'] = display[architecture_col].apply(_format_architecture)
    for col, label in available_metrics:
        display[label] = display[col].apply(_format_value)

    lines.append("### Best QE per Architecture")
    lines.append(display[['Dataset', 'Architecture'] + [label for _, label in available_metrics]].sort_values(['Dataset', 'Architecture']).to_markdown(index=False))
    lines.append("")

    winner_rows = []
    win_counts = defaultdict(float)
    for dataset, dataset_df in arch_summary.groupby(dataset_col):
        for col, label in available_metrics:
            metric_values = dataset_df[[architecture_col, col]].dropna()
            if metric_values.empty:
                continue
            min_value = metric_values[col].min()
            winners = metric_values[metric_values[col] == min_value][architecture_col].tolist()
            winners_formatted = [_format_architecture(w) for w in winners]
            if winners_formatted:
                increment = 1.0 / len(winners_formatted)
                for winner in winners_formatted:
                    win_counts[(label, winner)] += increment
            winner_rows.append({
                'Dataset': dataset,
                'Metric': label,
                'Winner(s)': ", ".join(winners_formatted) if winners_formatted else 'N/A',
                'Best Score': f"{min_value:.4f}"
            })

    dataset_winner_table = pd.DataFrame(winner_rows, columns=['Dataset', 'Metric', 'Winner(s)', 'Best Score'])
    if not dataset_winner_table.empty:
        lines.append("#### Architecture Winners by Dataset")
        lines.append(dataset_winner_table.sort_values(['Dataset', 'Metric']).to_markdown(index=False))
        lines.append("")

    if win_counts:
        win_rows = []
        for (metric_label, arch_label), count in sorted(win_counts.items()):
            win_rows.append({'Metric': metric_label, 'Architecture': arch_label, 'Win Count': f"{count:.2f}"})
        lines.append("#### Architecture Win Counts")
        lines.append(pd.DataFrame(win_rows, columns=['Metric', 'Architecture', 'Win Count']).to_markdown(index=False))
        lines.append("")

    if algorithm_col and algorithm_col in base.columns:
        algo_summary = base.groupby([dataset_col, algorithm_col, architecture_col], dropna=False)[[col for col, _ in available_metrics]].min().reset_index()
        if not algo_summary.empty:
            for algorithm_value in sorted(algo_summary[algorithm_col].dropna().unique()):
                subset = algo_summary[algo_summary[algorithm_col] == algorithm_value].copy()
                subset['Dataset'] = subset[dataset_col]
                subset['Algorithm'] = subset[algorithm_col].apply(_format_algorithm)
                subset['Architecture'] = subset[architecture_col].apply(_format_architecture)
                for col, label in available_metrics:
                    subset[label] = subset[col].apply(_format_value)

                lines.append(f"### Algorithm: {_format_algorithm(algorithm_value)}")
                lines.append(subset[['Dataset', 'Architecture'] + [label for _, label in available_metrics]].sort_values(['Dataset', 'Architecture']).to_markdown(index=False))
                lines.append("")

    return lines

def _batch_mode_section(df, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col):
    if not batch_mode_col or dataset_col not in df.columns:
        return []

    metric_specs = [
        ('quantization_error_train_normalized', 'QE Train (norm)'),
        ('quantization_error_holdout_normalized', 'QE Holdout (norm)'),
        (score_col, 'Combined Distance (norm)')
    ]

    available_metrics = [(col, label) for col, label in metric_specs if col and col in df.columns]
    if not available_metrics:
        return []

    base = df.dropna(subset=[dataset_col, batch_mode_col]).copy()
    if base.empty:
        return []

    lines = ["## Batch Mode Comparisons", ""]

    for metric_col, metric_label in available_metrics:
        metric_base = base.dropna(subset=[metric_col])
        if metric_base.empty:
            continue

        best_metric = _compute_best_by_group(metric_base, [dataset_col, batch_mode_col], metric_col)
        if best_metric.empty:
            continue

        display = best_metric.copy()
        display['Dataset'] = display[dataset_col]
        display['Batch Mode'] = display[batch_mode_col].apply(_format_batch_mode)
        display[metric_label] = display[metric_col].apply(_format_value)

        columns = ['Dataset', 'Batch Mode', metric_label]
        if architecture_col and architecture_col in display.columns:
            display['Winning Architecture'] = display[architecture_col].apply(_format_architecture)
            columns.append('Winning Architecture')
        if sampling_col and sampling_col in display.columns:
            display['Sampling Method'] = display[sampling_col].apply(_format_sampling)
            columns.append('Sampling Method')
        if algorithm_col and algorithm_col in display.columns:
            display['Winning Algorithm'] = display[algorithm_col].apply(_format_algorithm)
            columns.append('Winning Algorithm')

        lines.append(f"### {metric_label} Winners by Dataset")
        lines.append(display[columns].sort_values(['Dataset', 'Batch Mode']).to_markdown(index=False))
        lines.append("")

        dataset_summary_md, win_counts_md = _winner_summary(display[['Dataset', 'Batch Mode', '__raw_value__']].copy(), 'Batch Mode')
        if dataset_summary_md:
            lines.append(f"#### Dataset Winners – {metric_label}")
            lines.append(dataset_summary_md)
            lines.append("")
        if win_counts_md:
            lines.append(f"#### Batch Mode Win Counts – {metric_label}")
            lines.append(win_counts_md)
            lines.append("")

        if architecture_col and architecture_col in metric_base.columns:
            best_arch = _compute_best_by_group(metric_base, [dataset_col, architecture_col, batch_mode_col], metric_col)
            if not best_arch.empty:
                lines.append(f"#### {metric_label} by Architecture")
                for arch_value in sorted(best_arch[architecture_col].dropna().unique()):
                    subset = best_arch[best_arch[architecture_col] == arch_value].copy()
                    subset['Dataset'] = subset[dataset_col]
                    subset['Batch Mode'] = subset[batch_mode_col].apply(_format_batch_mode)
                    subset[metric_label] = subset[metric_col].apply(_format_value)
                    subset['Architecture'] = subset[architecture_col].apply(_format_architecture)
                    columns_arch = ['Dataset', 'Batch Mode', metric_label, 'Architecture']
                    if sampling_col and sampling_col in subset.columns:
                        subset['Sampling Method'] = subset[sampling_col].apply(_format_sampling)
                        columns_arch.append('Sampling Method')
                    if algorithm_col and algorithm_col in subset.columns:
                        subset['Winning Algorithm'] = subset[algorithm_col].apply(_format_algorithm)
                        columns_arch.append('Winning Algorithm')

                    lines.append(f"##### Architecture: {_format_architecture(arch_value)}")
                    lines.append(subset[columns_arch].sort_values(['Dataset', 'Batch Mode']).to_markdown(index=False))
                    lines.append("")

                    subset_summary = subset[['Dataset', 'Batch Mode', '__raw_value__']].copy()
                    ds_summary_md, count_summary_md = _winner_summary(subset_summary, 'Batch Mode')
                    if ds_summary_md:
                        lines.append(f"###### Dataset Winners – {metric_label} ({_format_architecture(arch_value)})")
                        lines.append(ds_summary_md)
                        lines.append("")
                    if count_summary_md:
                        lines.append(f"###### Batch Mode Win Counts – {metric_label} ({_format_architecture(arch_value)})")
                        lines.append(count_summary_md)
                        lines.append("")

        if sampling_col and sampling_col in metric_base.columns:
            best_sampling = _compute_best_by_group(metric_base, [dataset_col, sampling_col, batch_mode_col], metric_col)
            if not best_sampling.empty:
                lines.append(f"#### {metric_label} by Sampling Method")
                for sampling_value in sorted(best_sampling[sampling_col].dropna().unique()):
                    subset = best_sampling[best_sampling[sampling_col] == sampling_value].copy()
                    subset['Dataset'] = subset[dataset_col]
                    subset['Sampling Method'] = subset[sampling_col].apply(_format_sampling)
                    subset['Batch Mode'] = subset[batch_mode_col].apply(_format_batch_mode)
                    subset[metric_label] = subset[metric_col].apply(_format_value)
                    columns_sampling = ['Dataset', 'Sampling Method', 'Batch Mode', metric_label]
                    if architecture_col and architecture_col in subset.columns:
                        subset['Architecture'] = subset[architecture_col].apply(_format_architecture)
                        columns_sampling.append('Architecture')
                    if algorithm_col and algorithm_col in subset.columns:
                        subset['Winning Algorithm'] = subset[algorithm_col].apply(_format_algorithm)
                        columns_sampling.append('Winning Algorithm')

                    lines.append(f"##### Sampling: {_format_sampling(sampling_value)}")
                    lines.append(subset[columns_sampling].sort_values(['Dataset', 'Batch Mode']).to_markdown(index=False))
                    lines.append("")

                    subset_summary = subset[['Dataset', 'Batch Mode', '__raw_value__']].copy()
                    ds_summary_md, count_summary_md = _winner_summary(subset_summary, 'Batch Mode')
                    if ds_summary_md:
                        lines.append(f"###### Dataset Winners – {metric_label} ({_format_sampling(sampling_value)})")
                        lines.append(ds_summary_md)
                        lines.append("")
                    if count_summary_md:
                        lines.append(f"###### Batch Mode Win Counts – {metric_label} ({_format_sampling(sampling_value)})")
                        lines.append(count_summary_md)
                        lines.append("")

    return lines

def _build_tables_markdown(df, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col):
    lines = ["# Summary Comparison Tables", "", f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ""]

    sections = [
        _sampling_section(df, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col),
        _architecture_section(df, dataset_col, architecture_col, algorithm_col),
        _batch_mode_section(df, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col)
    ]

    for section in sections:
        if section:
            lines.extend(section)
            if section[-1] != "":
                lines.append("")

    return "\n".join(lines).strip() + "\n"

def create_unified_summary(df_final,
                           mann_whitney_results=None,
                           mann_whitney_normalized_results=None,
                           best_performers_df=None,
                           pca_info=None,
                           output_file='Results/ANALYSIS_SUMMARY.md'):
    dataset_col = 'dataset' if 'dataset' in df_final.columns else None
    sampling_col = _resolve(df_final, ['sampling_method_parsed', 'sampling_method_final', 'sampling_method'])
    processing_col = _resolve(df_final, ['config_processing_method', 'processing_type'])
    architecture_col = _resolve(df_final, ['architecture', 'map_type'])
    algorithm_col = _resolve(df_final, ['algorithm', 'processing_type'])
    batch_mode_col = _resolve(df_final, ['param_batch_mode', 'batch_mode'])
    scenario_col = 'scenario_id' if 'scenario_id' in df_final.columns else None
    score_col = 'Normalized_Overall_Score'

    # Ensure QE-only helper metric exists for both MST and hexagonal (and unified) summaries.
    if (
        'balanced_qe_normalized' not in df_final.columns
        and 'quantization_error_holdout_normalized' in df_final.columns
        and 'quantization_error_train_normalized' in df_final.columns
    ):
        df_final['balanced_qe_normalized'] = df_final[
            ['quantization_error_holdout_normalized', 'quantization_error_train_normalized']
        ].mean(axis=1)

    if (
        'balanced_qe_raw' not in df_final.columns
        and 'quantization_error_holdout' in df_final.columns
        and 'quantization_error_train' in df_final.columns
    ):
        df_final['balanced_qe_raw'] = df_final[
            ['quantization_error_holdout', 'quantization_error_train']
        ].mean(axis=1)

    base_metric_configs = [
        {
            'label': 'QE Holdout',
            'norm_col': 'quantization_error_holdout_normalized',
            'norm_label': 'QE Holdout (norm)',
            'raw_col': 'quantization_error_holdout',
            'raw_label': 'QE Holdout (raw)'
        },
        {
            'label': 'QE Train',
            'norm_col': 'quantization_error_train_normalized',
            'norm_label': 'QE Train (norm)',
            'raw_col': 'quantization_error_train',
            'raw_label': 'QE Train (raw)'
        },
        {
            'label': 'Balanced QE',
            'norm_col': 'balanced_qe_normalized',
            'norm_label': 'Balanced QE (norm)',
            'raw_col': 'balanced_qe_raw',
            'raw_label': 'Balanced QE (raw)'
        },
        {
            'label': 'Combined Distance Score',
            'norm_col': score_col,
            'norm_label': 'Combined Distance (norm)',
            'raw_col': 'distance_to_origin',
            'raw_label': 'Distance to Origin (raw)'
        }
    ]

    is_mst_only = False
    if architecture_col:
        architectures = df_final[architecture_col].dropna().astype(str).str.lower().unique()
        if len(architectures) > 0 and all(arch == 'mst' for arch in architectures):
            is_mst_only = True

    if is_mst_only:
        base_metric_configs = [
            cfg for cfg in base_metric_configs
            if cfg['label'] not in {'Combined Distance Score'}
        ]

    metric_configs = [cfg for cfg in base_metric_configs if cfg['norm_col'] in df_final.columns]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# 📊 Complete Analysis Summary\n\n')
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write('---\n\n')
        f.write('## 📋 Table of Contents\n\n')
        f.write('1. [Dataset Overview](#dataset-overview)\n')
        f.write('2. [PCA Analysis](#pca-analysis)\n')
        f.write('3. [Dataset-by-Dataset Results](#dataset-by-dataset-results)\n')
        f.write('4. [Mann-Whitney U Test Results](#mann-whitney-u-test-results)\n')
        f.write('5. [Key Findings](#key-findings)\n')
        f.write('6. [Interactive Pareto Fronts](#interactive-pareto-fronts)\n')
        f.write('7. [Sample Counts](#sample-counts)\n\n')
        f.write('---\n\n')

        f.write('## 📈 Dataset Overview\n\n')
        f.write(f"- **Total configurations analysed**: {len(df_final):,}\n")
        f.write(f"- **Datasets**: {df_final['dataset'].nunique()}\n")
        if dataset_col:
            dataset_values = dataset_groups.ordered_dataset_labels(df_final[dataset_col].dropna().unique())
            synthetic_count = sum(
                dataset_groups.dataset_group(name) == "synthetic" for name in dataset_values
            )
            real_count = sum(
                dataset_groups.dataset_group(name) == "real" for name in dataset_values
            )
            f.write(f"- **Synthetic datasets**: {synthetic_count}\n")
            f.write(f"- **Real datasets**: {real_count}\n")
        if architecture_col:
            f.write(f"- **Architectures**: {', '.join(sorted(df_final[architecture_col].dropna().unique()))}\n")
        if processing_col:
            f.write(f"- **Processing modes**: {', '.join(sorted(df_final[processing_col].dropna().unique()))}\n")
        if sampling_col:
            f.write(f"- **Sampling methods**: {', '.join(sorted(df_final[sampling_col].dropna().unique()))}\n")
        f.write('\n')

        normalized_pairs = [(cfg['norm_col'], cfg['norm_label']) for cfg in metric_configs if cfg['norm_col'] in df_final.columns]
        raw_pairs = [(cfg['raw_col'], cfg['raw_label']) for cfg in metric_configs if cfg.get('raw_col') and cfg['raw_col'] in df_final.columns]
        if normalized_pairs:
            f.write('### Normalised Metrics Summary\n\n')
            f.write(_format_metric_summary(df_final, normalized_pairs))
            f.write('\n\n')
        if raw_pairs:
            f.write('### Raw Metrics Summary\n\n')
            f.write(_format_metric_summary(df_final, raw_pairs))
            f.write('\n\n')
        f.write('---\n\n')

        f.write('## 🔍 PCA Analysis\n\n')
        if pca_info:
            f.write(f"- PC1 explains **{pca_info['explained_variance'][0]:.1%}** of the variance.\n")
            f.write('| Measure | Loading |\n|---------|---------|\n')
            for measure, loading in zip(pca_info['measures'], pca_info['loadings']):
                f.write(f"| {measure.replace('_', ' ').title()} | {loading:+.4f} |\n")
            f.write('\n')
        else:
            f.write('_PCA skipped: new schema exposes a single topology metric._\n\n')
        f.write('---\n\n')

        f.write('## 🗂️ Dataset-by-Dataset Results\n\n')
        metric_winner_records = []
        algo_win_counts = {}
        architecture_compare_records = []
        architecture_win_records = []
        for dataset in dataset_groups.ordered_dataset_labels(df_final['dataset'].dropna().unique()):
            f.write(f"### Dataset: {dataset}\n\n")
            subset = df_final[df_final['dataset'] == dataset].copy()
            if subset.empty:
                f.write('_No records for this dataset._\n\n---\n\n')
                continue

            if algorithm_col and algorithm_col in subset.columns:
                subset['Algorithm'] = subset[algorithm_col].map(_format_algorithm)
            elif 'algorithm' in subset.columns:
                subset['Algorithm'] = subset['algorithm'].map(_format_algorithm)
            else:
                subset['Algorithm'] = 'N/A'

            if architecture_col and architecture_col in subset.columns:
                subset['Architecture'] = subset[architecture_col].map(_format_architecture)
            else:
                subset['Architecture'] = 'N/A'

            subset['Processing Mode'] = subset.apply(
                lambda row: _processing_display(row, processing_col) if processing_col else 'N/A',
                axis=1
            )

            if sampling_col and sampling_col in subset.columns:
                subset['Sampling'] = subset[sampling_col].map(_format_sampling)
            else:
                subset['Sampling'] = 'N/A'

            if batch_mode_col and batch_mode_col in subset.columns:
                subset['Batch Mode'] = subset[batch_mode_col].map(_format_batch_mode)
            else:
                subset['Batch Mode'] = 'N/A'

            mask_colors = subset['Algorithm'].astype(str).str.lower() == 'colors'
            subset.loc[mask_colors, 'Batch Mode'] = 'All (ignored)'

            if scenario_col and scenario_col in subset.columns:
                subset['Scenario'] = subset[scenario_col].fillna('N/A')
            else:
                subset['Scenario'] = 'N/A'

            for metric_cfg in metric_configs:
                norm_col = metric_cfg['norm_col']
                metric_data = subset.dropna(subset=[norm_col]).copy()
                if metric_data.empty:
                    continue

                f.write(f"#### {metric_cfg['label']}\n\n")

                overall_table = _best_group_table(
                    metric_data,
                    metric_cfg,
                    ['Algorithm'],
                    ['Processing Mode', 'Sampling', 'Batch Mode', 'Architecture', 'Scenario']
                )
                if overall_table.empty:
                    f.write('_No valid records for this metric._\n\n')
                    continue

                f.write('**Overall Batch vs. Colors**\n\n')
                f.write(overall_table.to_markdown(index=False))
                f.write('\n\n')

                if sampling_col:
                    sampling_table = _best_group_table(
                        metric_data,
                        metric_cfg,
                        ['Sampling', 'Algorithm'],
                        ['Processing Mode', 'Batch Mode', 'Architecture', 'Scenario']
                    )
                    if not sampling_table.empty:
                        f.write('**By Sampling Method**\n\n')
                        f.write(sampling_table.to_markdown(index=False))
                        f.write('\n\n')

                if batch_mode_col:
                    batch_table = _best_group_table(
                        metric_data,
                        metric_cfg,
                        ['Batch Mode', 'Algorithm'],
                        ['Processing Mode', 'Sampling', 'Architecture', 'Scenario']
                    )
                    if not batch_table.empty:
                        f.write('**By Batch Mode**\n\n')
                        f.write(batch_table.to_markdown(index=False))
                        f.write('\n\n')

                    batch_algo_data = metric_data[metric_data['Algorithm'].astype(str).str.lower() == 'batch'].copy()
                    if not batch_algo_data.empty:
                        batch_algo_table = _best_group_table(
                            batch_algo_data,
                            metric_cfg,
                            ['Sampling', 'Batch Mode'],
                            ['Processing Mode', 'Architecture', 'Scenario']
                        )
                        if not batch_algo_table.empty:
                            f.write('**Batch Algorithm: Mini vs. Full**\n\n')
                            f.write(batch_algo_table.to_markdown(index=False))
                            f.write('\n\n')

                if architecture_col:
                    architecture_table = _best_group_table(
                        metric_data,
                        metric_cfg,
                        ['Architecture', 'Algorithm'],
                        ['Processing Mode', 'Sampling', 'Batch Mode', 'Scenario']
                    )
                    if not architecture_table.empty:
                        f.write('**By Architecture**\n\n')
                        f.write(architecture_table.to_markdown(index=False))
                        f.write('\n\n')
                        arch_copy = architecture_table.copy()
                        arch_copy['Dataset'] = dataset
                        arch_copy['Metric'] = metric_cfg['norm_label']
                        architecture_compare_records.append(arch_copy)

                best_idx = metric_data[norm_col].idxmin()
                best_row = metric_data.loc[best_idx]
                best_algorithm = best_row.get('Algorithm', 'N/A')
                raw_val = float(best_row[metric_cfg['raw_col']]) if metric_cfg.get('raw_col') and metric_cfg['raw_col'] in metric_data.columns else float('nan')
                architecture_win_records.append({
                    'Dataset': dataset,
                    'Metric': metric_cfg['norm_label'],
                    'Algorithm': best_algorithm,
                    'Processing Mode': best_row.get('Processing Mode', 'N/A'),
                    'Sampling': best_row.get('Sampling', 'N/A'),
                    'Batch Mode': best_row.get('Batch Mode', 'N/A'),
                    'Architecture': best_row.get('Architecture', 'N/A'),
                    'Value': float(best_row[norm_col])
                })
                metric_winner_records.append({
                    'Dataset': dataset,
                    'Dataset Group': _format_dataset_group(dataset),
                    'Metric': metric_cfg['norm_label'],
                    'Algorithm': best_algorithm,
                    'Processing Mode': best_row.get('Processing Mode', 'N/A'),
                    'Sampling': best_row.get('Sampling', 'N/A'),
                    'Batch Mode': best_row.get('Batch Mode', 'N/A'),
                    'Architecture': best_row.get('Architecture', 'N/A'),
                    'Scenario': best_row.get('Scenario', 'N/A'),
                    'Raw Value': raw_val,
                    'Value': float(best_row[norm_col])
                })
                algo_win_counts.setdefault(metric_cfg['norm_label'], {}).setdefault(best_algorithm, 0)
                algo_win_counts[metric_cfg['norm_label']][best_algorithm] += 1

            f.write('---\n\n')

        if architecture_compare_records:
            f.write('## 🧭 Architecture Comparisons\n\n')
            arch_df = pd.concat(architecture_compare_records, ignore_index=True)
            for metric_label, group_df in arch_df.groupby('Metric'):
                f.write(f"### {metric_label}\n\n")
                base_cols = ['Dataset', 'Architecture', 'Algorithm', 'Processing Mode', 'Sampling', 'Batch Mode', 'Scenario']
                present_base = [col for col in base_cols if col in group_df.columns]
                value_cols = [col for col in group_df.columns if col not in set(present_base + ['Metric'])]
                display_df = group_df[present_base + value_cols]
                f.write(display_df.to_markdown(index=False))
                f.write('\n\n')
            f.write('---\n\n')
            if architecture_win_records:
                summary_columns = ['Dataset', 'Metric', 'Architecture', 'Algorithm', 'Sampling', 'Batch Mode', 'Processing Mode']
                arch_win_df = pd.DataFrame(architecture_win_records)
                subset_cols = [col for col in summary_columns if col in arch_win_df.columns]
                grouped_arch = arch_win_df.groupby(['Metric', 'Architecture']).size().reset_index(name='Wins')
                f.write('### Architecture Win Summary\n\n')
                f.write(grouped_arch.to_markdown(index=False))
                f.write('\n\n')
                if 'Algorithm' in arch_win_df.columns:
                    grouped_arch_algo = arch_win_df.groupby(['Metric', 'Architecture', 'Algorithm']).size().reset_index(name='Wins')
                    f.write('### Architecture Win Summary by Algorithm\n\n')
                    f.write(grouped_arch_algo.to_markdown(index=False))
                    f.write('\n\n')
                f.write('---\n\n')

        f.write('## 📊 Mann-Whitney U Test Results\n\n')
        if mann_whitney_results:
            f.write('### Individual Metrics\n\n')
            f.write('| Sampling | Metric | Batch Mean | Colors Mean | P-value | Significant | Winner |\n')
            f.write('|----------|--------|------------|-------------|---------|-------------|--------|\n')
            for sampling, results in sorted(mann_whitney_results.items()):
                for metric, res in sorted(results.items()):
                    winner = '-'
                    if res['significance'] == 'significant':
                        winner = '**Batch**' if res['group1_mean'] < res['group2_mean'] else '**Colors**'
                    label = metric.replace('_normalized', '').replace('_', ' ').title()
                    f.write(f"| {sampling} | {label} | {res['group1_mean']:.4f} | {res['group2_mean']:.4f} | "
                            f"{res['p_value']:.4f} | {res['significance']} | {winner} |\n")
            f.write('\n')
        else:
            f.write('_Individual metric comparisons were not generated for this run._\n\n')

        if mann_whitney_normalized_results:
            f.write('### Normalised Overall Score (3D Euclidean)\n\n')
            f.write('| Sampling | Batch Mean | Colors Mean | P-value | Significant | Effect Size | Winner |\n')
            f.write('|----------|------------|-------------|---------|-------------|-------------|--------|\n')
            for sampling, results in sorted(mann_whitney_normalized_results.items()):
                res = results.get('Normalized_Overall_Score')
                if not res:
                    continue
                winner = '-'
                if res['significance'] == 'significant':
                    winner = '**Batch** ✓' if res['group1_mean'] < res['group2_mean'] else '**Colors** ✓'
                f.write(f"| {sampling} | {res['group1_mean']:.4f} | {res['group2_mean']:.4f} | "
                        f"{res['p_value']:.4f} | {res['significance']} | {res['effect_size']:.3f} | {winner} |\n")
            f.write('\n')
        else:
            f.write('_Normalised overall score comparisons were not generated for this run._\n\n')
        f.write('---\n\n')

        if metric_winner_records:
            summary_df = pd.DataFrame(metric_winner_records)
            summary_table = _summary_table(summary_df, metric_configs)
            f.write('### Summary of Winners by Dataset\n\n')
            if summary_table:
                f.write(summary_table)
                f.write('\n')
            else:
                f.write('_Unable to build dataset summary table._\n\n')
        else:
            f.write('_No best-performer information available for metric summaries._\n\n')

        if algo_win_counts:
            algorithms = sorted({algo for counts in algo_win_counts.values() for algo in counts})
            if algorithms:
                win_rows = []
                stratified_totals = {}
                for metric_cfg in metric_configs:
                    label = metric_cfg['norm_label']
                    counts = algo_win_counts.get(label, {})

                    row = {'Metric': label}
                    for algo in algorithms:
                        key = 'overall_' + (algo.lower() if algo.lower() != 'colors' else 'colours')
                        row[key] = counts.get(algo, 0)
                    win_rows.append(row)

                win_df = pd.DataFrame(win_rows)
                f.write('### Algorithm Win Summary\n\n')
                f.write(win_df.to_markdown(index=False))
                f.write('\n')

                if metric_winner_records:
                    winner_df = pd.DataFrame(metric_winner_records)
                    if 'Dataset Group' in winner_df.columns:
                        group_counts = (
                            winner_df.groupby(['Metric', 'Dataset Group', 'Algorithm'])
                            .size()
                            .reset_index(name='Wins')
                        )
                        if not group_counts.empty:
                            f.write('### Algorithm Win Summary by Dataset Group\n\n')
                            group_order = {'Synthetic': 0, 'Real': 1, 'Global': 2}
                            for metric_cfg in metric_configs:
                                metric_label = metric_cfg['norm_label']
                                subset = group_counts[group_counts['Metric'] == metric_label].copy()
                                if subset.empty:
                                    continue
                                subset['__order__'] = subset['Dataset Group'].map(group_order).fillna(99)
                                subset = subset.sort_values(['__order__', 'Dataset Group', 'Algorithm']).drop(columns='__order__')
                                f.write(f"#### {metric_label}\n\n")
                                f.write(subset.to_markdown(index=False))
                                f.write('\n\n')

                # Stratified counts across all runs per sampling method
                stratified_totals = {}
                stratified_totals = {}
                full_run_totals = {}
                if metric_winner_records:
                    for record in metric_winner_records:
                        metric_label = record.get('Metric')
                        sampling_label = record.get('Sampling')
                        algo_label = record.get('Algorithm')
                        dataset_label = record.get('Dataset')
                        if pd.isna(metric_label) or pd.isna(sampling_label) or pd.isna(algo_label):
                            continue

                        metric_key = str(metric_label)
                        sampling_key = str(sampling_label)
                        algo_key = str(algo_label)

                        stratified_totals.setdefault(metric_key, {}).setdefault(sampling_key, {})
                        stratified_totals[metric_key][sampling_key][algo_key] = (
                            stratified_totals[metric_key][sampling_key].get(algo_key, 0) + 1
                        )

                        if sampling_key.lower() == 'full' and pd.notna(dataset_label):
                            dataset_key = str(dataset_label)
                            full_run_totals.setdefault(metric_key, {}).setdefault(dataset_key, {'batch': 0, 'colours': 0})
                            algo_normalized = 'colours' if algo_key.lower() == 'colors' else 'batch'
                            full_run_totals[metric_key][dataset_key][algo_normalized] += 1

                if stratified_totals:
                    f.write('### Algorithm Win Summary by Sampling Method (All Runs)\n\n')
                    for metric_label, sampling_map in stratified_totals.items():
                        f.write(f"#### {metric_label}\n\n")
                        rows = []
                        algo_columns = set()
                        for sampling_label, algo_map in sampling_map.items():
                            row = {'Sampling': sampling_label}
                            for algo_name, win_count in algo_map.items():
                                normalized = algo_name.lower()
                                column_name = 'colours' if normalized == 'colors' else normalized
                                row[column_name] = row.get(column_name, 0) + win_count
                                algo_columns.add(column_name)
                            rows.append(row)
                        if rows:
                            ordered_columns = ['Sampling'] + sorted(algo_columns)
                            df_rows = pd.DataFrame(rows)
                            for col in ordered_columns:
                                if col != 'Sampling' and col not in df_rows.columns:
                                    df_rows[col] = 0
                            df_rows = df_rows[ordered_columns].fillna(0).sort_values('Sampling')
                            f.write(df_rows.to_markdown(index=False))
                            f.write('\n\n')

                if full_run_totals:
                    f.write('### Algorithm Win Summary for Full Sampling by Dataset\n\n')
                    for metric_label, dataset_map in full_run_totals.items():
                        f.write(f"#### {metric_label}\n\n")
                        rows = []
                        for dataset_label, counts in sorted(dataset_map.items()):
                            row = {'Dataset': dataset_label, 'full_batch': counts.get('batch', 0), 'full_colours': counts.get('colours', 0)}
                            rows.append(row)
                        if rows:
                            df_rows = pd.DataFrame(rows)
                            df_rows = df_rows[['Dataset', 'full_batch', 'full_colours']]
                            f.write(df_rows.sort_values('Dataset').to_markdown(index=False))
                            f.write('\n\n')
        else:
            f.write('_No algorithm win summary available._\n\n')
        f.write('---\n\n')

        f.write('## 🔑 Key Findings\n\n')
        findings = []
        if score_col in df_final.columns and not df_final[score_col].dropna().empty:
            best_idx = df_final[score_col].idxmin()
            best_row = df_final.loc[best_idx]
            findings.append(
                f"Lowest overall score: **{best_row[score_col]:.4f}** "
                f"(Dataset: {best_row['dataset']}, Architecture: {best_row.get(architecture_col, 'N/A')}, "
                f"Processing: {best_row.get(processing_col, 'N/A')}, Sampling: {best_row.get(sampling_col, 'N/A')})."
            )
        if 'balanced_qe_normalized' in df_final.columns and not df_final['balanced_qe_normalized'].dropna().empty:
            best_idx = df_final['balanced_qe_normalized'].idxmin()
            best_row = df_final.loc[best_idx]
            findings.append(
                f"Lowest balanced QE (train+holdout): **{best_row['balanced_qe_normalized']:.4f}** "
                f"(Dataset: {best_row['dataset']}, Architecture: {best_row.get(architecture_col, 'N/A')}, "
                f"Processing: {best_row.get(processing_col, 'N/A')}, Sampling: {best_row.get(sampling_col, 'N/A')})."
            )
        if mann_whitney_results:
            sig_metrics = []
            for sampling, results in sorted(mann_whitney_results.items()):
                for metric, res in results.items():
                    if res['significance'] == 'significant':
                        winner = 'Batch' if res['group1_mean'] < res['group2_mean'] else 'Colors'
                        sig_metrics.append(
                            f"{metric.replace('_normalized', '').replace('_', ' ').title()} "
                            f"({sampling}): {winner} (p={res['p_value']:.4f})"
                        )
            if sig_metrics:
                findings.append("Significant individual metric differences:\n  - " + "\n  - ".join(sig_metrics))
        if mann_whitney_normalized_results:
            sig_overall = []
            for sampling, results in sorted(mann_whitney_normalized_results.items()):
                res = results.get('Normalized_Overall_Score')
                if res and res['significance'] == 'significant':
                    winner = 'Batch' if res['group1_mean'] < res['group2_mean'] else 'Colors'
                    sig_overall.append(
                        f"{sampling}: {winner} (p={res['p_value']:.4f}, effect={res['effect_size']:.3f})"
                    )
            if sig_overall:
                findings.append("Significant overall-score differences:\n  - " + "\n  - ".join(sig_overall))
        if not findings:
            findings.append("No statistically significant differences detected in this run.")
        for item in findings:
            f.write(f"- {item}\n")
        f.write('\n---\n\n')

        f.write('## 🕹️ Interactive Pareto Fronts\n\n')
        plots_dir = Path(output_file).parent.parent / 'plots'
        if plots_dir.exists():
            html_files = sorted(p.name for p in plots_dir.glob('pareto_front_3d_*.html'))
            if html_files:
                for name in html_files:
                    f.write(f"- {name}\n")
            else:
                f.write('_No interactive Pareto front files generated in this run._\n')
        else:
            f.write('_Plots directory not found._\n')
        f.write('\n---\n\n')

        f.write('## 📦 Sample Counts\n\n')
        grouping = ['dataset']
        if architecture_col:
            grouping.append(architecture_col)
        if processing_col:
            grouping.append(processing_col)
        if sampling_col:
            grouping.append(sampling_col)
        f.write(_sample_counts(df_final, grouping))
        f.write('\n')

    if dataset_col:
        tables_path = Path(output_file).with_name('tables.md')
        tables_md = _build_tables_markdown(df_final, dataset_col, sampling_col, architecture_col, algorithm_col, batch_mode_col, score_col)
        tables_path.write_text(tables_md, encoding='utf-8')
        print(f"Comparison tables saved to {tables_path}")
    else:
        print("Dataset column not found; skipping tables.md generation.")

    print(f"Unified summary saved to {output_file}")
