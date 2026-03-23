from __future__ import annotations

import argparse
import atexit
import json
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .constants import *
from .helpers import *
from .plots import *
from .svg_postprocess import _normalize_publication_svg_assets
from .workflows import *


def _run_block(
    *,
    suite_name: str,
    block_df: pd.DataFrame,
    block_output_dir: Path,
    block_label: str,
    expected_pairs_value: Optional[int],
    run_algorithm_comparison: bool,
    run_mst_rng_pair: bool,
    emit_block_legend: bool,
    metrics_to_run: Sequence[str],
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    strict_pairing: bool,
    inline_legends: bool,
    generated_files: List[str],
    diagnostics: Dict[str, object],
) -> Optional[str]:
    (block_output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (block_output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    print(f"\n--- {suite_name}/{block_label} ---")
    print(f"Output: {block_output_dir}")

    diagnostics_prefix = f"{suite_name}.{block_label}."
    legend_registry: Dict[str, Dict[str, object]] = OrderedDict()
    for metric in metrics_to_run:
        metric_label = METRIC_LABELS.get(metric, metric)
        print(f"\nAnalyzing metric: {metric} ({metric_label})")

        if run_algorithm_comparison:
            _analyze_hex_algorithms(
                df=block_df,
                metric=metric,
                metric_label=metric_label,
                output_dir=block_output_dir,
                top_k=top_k,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                dpi=dpi,
                alpha=alpha,
                strict_pairing=bool(strict_pairing),
                expected_pairs_per_dataset=expected_pairs_value,
                generated_files=generated_files,
                diagnostics=diagnostics,
                diagnostics_prefix=diagnostics_prefix,
                inline_legends=inline_legends,
                legend_registry=legend_registry,
            )
        _analyze_topology(
            df=block_df,
            metric=metric,
            metric_label=metric_label,
            output_dir=block_output_dir / 'topology_main',
            top_k=top_k,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            dpi=dpi,
            alpha=alpha,
            strict_pairing=bool(strict_pairing),
            expected_pairs_per_dataset=expected_pairs_value,
            generated_files=generated_files,
            diagnostics=diagnostics,
            diagnostics_prefix=diagnostics_prefix,
            figure_scope=TOPOLOGY_SCOPE_MAIN,
            inline_legends=inline_legends,
            legend_registry=legend_registry,
        )
        _analyze_topology_pair_basic(
            df=block_df,
            metric=metric,
            metric_label=metric_label,
            output_dir=block_output_dir / 'topology_main',
            top_k=top_k,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            dpi=dpi,
            alpha=alpha,
            strict_pairing=bool(strict_pairing),
            expected_pairs_per_dataset=expected_pairs_value,
            generated_files=generated_files,
            diagnostics=diagnostics,
            diagnostics_prefix=diagnostics_prefix,
            value_a='hexagonal',
            value_b='rng',
            pair_slug='hex_vs_rng',
            reference_value='hexagonal',
            inline_legends=inline_legends,
            legend_registry=legend_registry,
        )
        if run_mst_rng_pair:
            _analyze_topology_pair_basic(
                df=block_df,
                metric=metric,
                metric_label=metric_label,
                output_dir=block_output_dir / 'topology_main',
                top_k=top_k,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                dpi=dpi,
                alpha=alpha,
                strict_pairing=bool(strict_pairing),
                expected_pairs_per_dataset=expected_pairs_value,
                generated_files=generated_files,
                diagnostics=diagnostics,
                diagnostics_prefix=diagnostics_prefix,
                value_a='mst',
                value_b='rng',
                pair_slug='mst_vs_rng',
                reference_value='mst',
                inline_legends=inline_legends,
                legend_registry=legend_registry,
            )

    if inline_legends or not emit_block_legend:
        return None

    legend_key_path = block_output_dir / 'figures' / 'legend_key.svg'
    if _render_legend_key(legend_registry=legend_registry, output_path=legend_key_path, dpi=dpi):
        generated_files.append(str(legend_key_path.resolve()))
        return str(legend_key_path.resolve())
    return None


def _run_suite(
    *,
    suite_name: str,
    suite_metrics_list: Sequence[str],
    df: pd.DataFrame,
    run_output_dir: Path,
    suite_outputs: Dict[str, Dict[str, object]],
    suite_metrics: Dict[str, List[str]],
    top_k: int,
    bootstrap_iterations: int,
    seed: int,
    dpi: int,
    alpha: float,
    strict_pairing: bool,
    inline_legends: bool,
    full_suite: bool,
    sampling_comparison_focus_key: Optional[str],
    generated_files: List[str],
    diagnostics: Dict[str, object],
) -> None:
    suite_output_dir = run_output_dir / suite_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    suite_outputs[suite_name] = {
        'root': str(suite_output_dir.resolve()),
        'sampling': {},
        'sampling_comparison': '',
        'publication_figures': '',
        'legend_keys': {},
    }
    suite_metrics[suite_name] = list(suite_metrics_list)

    print(f"\n=== Running {suite_name} suite ===")
    print(f"Output: {suite_output_dir}")

    sampling_series = df['pair_sampling'].astype(str).str.lower().str.strip()
    sampling_values = _canonical_sampling_modes(sampling_series.dropna().unique().tolist())
    sampling_block_dirs: Dict[str, Path] = {}
    publication_topology_metrics: List[str] = [
        metric_name
        for metric_name in [
            'balanced_qe_raw',
            'balanced_qe_normalized',
            'quantization_error_holdout',
            'quantization_error_holdout_normalized',
            'quantization_error_train',
            'quantization_error_train_normalized',
        ]
        if metric_name in suite_metrics_list
    ]
    if not publication_topology_metrics:
        publication_topology_metrics = list(suite_metrics_list)
    if full_suite:
        for sampling in sampling_values:
            sampling_slug = _slug(sampling)
            sampling_output_dir = suite_output_dir / 'sampling' / sampling_slug
            subset = df[sampling_series == sampling].copy()
            if subset.empty:
                continue
            sampling_block_dirs[sampling] = sampling_output_dir
            suite_outputs[suite_name]['sampling'][sampling] = str(sampling_output_dir.resolve())
            sampling_legend = _run_block(
                suite_name=suite_name,
                block_df=subset,
                block_output_dir=sampling_output_dir,
                block_label=f'sampling.{sampling_slug}',
                expected_pairs_value=None,
                run_algorithm_comparison=True,
                run_mst_rng_pair=True,
                emit_block_legend=True,
                metrics_to_run=suite_metrics_list,
                top_k=top_k,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                dpi=dpi,
                alpha=alpha,
                strict_pairing=strict_pairing,
                inline_legends=inline_legends,
                generated_files=generated_files,
                diagnostics=diagnostics,
            )
            if sampling_legend:
                suite_outputs[suite_name]['legend_keys'][f'sampling.{sampling_slug}'] = sampling_legend

    available_sampling = {str(v).strip().lower() for v in sampling_values}
    if 'full' in available_sampling and not full_suite:
        full_only_output_dir = suite_output_dir / 'sampling' / 'full'
        full_only_subset = df[sampling_series == 'full'].copy()
        if not full_only_subset.empty:
            suite_outputs[suite_name]['sampling']['full'] = str(full_only_output_dir.resolve())
            _run_block(
                suite_name=suite_name,
                block_df=full_only_subset,
                block_output_dir=full_only_output_dir,
                block_label='sampling.full',
                expected_pairs_value=None,
                run_algorithm_comparison=False,
                run_mst_rng_pair=True,
                emit_block_legend=False,
                metrics_to_run=publication_topology_metrics,
                top_k=top_k,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                dpi=dpi,
                alpha=alpha,
                strict_pairing=strict_pairing,
                inline_legends=inline_legends,
                generated_files=generated_files,
                diagnostics=diagnostics,
            )

    sampling_compare_dir = suite_output_dir / 'sampling_comparison'
    suite_outputs[suite_name]['sampling_comparison'] = str(sampling_compare_dir.resolve())
    sampling_compare_metrics: Sequence[str] = list(suite_metrics_list)
    if not full_suite:
        sampling_compare_metrics = list(publication_topology_metrics)
    for metric in sampling_compare_metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        _analyze_sampling_mode_outcomes(
            df=df,
            metric=metric,
            metric_label=metric_label,
            output_dir=sampling_compare_dir,
            top_k=top_k,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            dpi=dpi,
            alpha=alpha,
            generated_files=generated_files,
            diagnostics=diagnostics,
            diagnostics_prefix=f'{suite_name}.sampling_comparison.',
            comparison_focus_key=sampling_comparison_focus_key,
        )

    _compose_sampling_stratified_publication_figures(
        suite_output_dir=suite_output_dir,
        suite_metrics_list=suite_metrics_list,
        sampling_dirs=sampling_block_dirs,
        full_suite=full_suite,
        alpha=alpha,
        dpi=dpi,
        generated_files=generated_files,
        diagnostics=diagnostics,
        diagnostics_prefix=f'{suite_name}.',
        sampling_primary_comparison_key=sampling_comparison_focus_key,
    )
    suite_outputs[suite_name]['publication_figures'] = str((suite_output_dir / 'publication_figures').resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate publication paired comparison figures (seaborn)')
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Input CSV (recommended: processed_data_with_overall_score.csv from analysis output)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='Results/publication_figures',
        help='Base output directory for publication run folders',
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Optional explicit run folder name (default: timestamped)',
    )
    parser.add_argument(
        '--default-runs-file',
        type=str,
        default=None,
        help=(
            'Optional true-default run CSV (for example matched_default_runs.csv). '
            'When provided, runs integrated tuned-vs-default and hyperparameter stability analyses.'
        ),
    )
    parser.add_argument(
        '--default-aware-manifest',
        type=str,
        default=None,
        help=(
            'Optional manifest JSON from run_matched_default_floatsom_batch --run-both-profiles. '
            'When provided, resolves --default-runs-file and --default-aware-tuned-runs-file automatically.'
        ),
    )
    parser.add_argument(
        '--default-aware-tuned-runs-file',
        type=str,
        default=None,
        help=(
            'Optional rerun tuned/global-parameter CSV used for tuned-vs-default pairing. '
            'If omitted, pairing uses --data-file rows.'
        ),
    )
    parser.add_argument(
        '--default-aware-output-subdir',
        type=str,
        default='default_aware_analysis',
        help='Subdirectory under run output for default-aware analysis artifacts.',
    )
    parser.add_argument(
        '--mst-xpysom-defaults-csv',
        type=str,
        default=None,
        help=(
            'Optional explicit self-contained MST XPySOM benchmark CSV for Supplementary Figure S1. '
            'This CSV must already contain both method=floatsom and method=xpysom rows for the MST comparison.'
        ),
    )
    parser.add_argument(
        '--rng-xpysom-defaults-csv',
        type=str,
        default=None,
        help=(
            'Optional explicit self-contained RNG XPySOM benchmark CSV for Supplementary Figure S2. '
            'This CSV must already contain both method=floatsom and method=xpysom rows for the RNG comparison.'
        ),
    )
    parser.add_argument(
        '--hexagonal-xpysom-defaults-csv',
        type=str,
        default=None,
        help=(
            'Optional explicit self-contained hexagonal XPySOM benchmark CSV for Supplementary Figure S3. '
            'This CSV must already contain both method=floatsom and method=xpysom rows for the hexagonal comparison.'
        ),
    )
    parser.add_argument(
        '--xpysom-default-runs-file',
        type=str,
        default=None,
        help=(
            'Optional XPySOM benchmark CSV or benchmark output directory used for the dedicated '
            'XPySOM-vs-RNG publication figure. If a directory is provided, the workflow resolves '
            '<dir>/xpysom_batch_topology_sweep_runs.csv automatically. Panels A-C compare this '
            'XPySOM benchmark export directly against --xpysom-rng-runs-file by keyed run matching, '
            'so the two inputs must describe the same executed configuration set even if row order differs.'
        ),
    )
    parser.add_argument(
        '--xpysom-rng-runs-file',
        type=str,
        default=None,
        help=(
            'Optional FloatSOM RNG benchmark CSV used for the dedicated XPySOM-vs-RNG publication figure. '
            'Panels A-C compare this benchmark export directly against --xpysom-default-runs-file by keyed '
            'run matching on dataset, seed, sampling, processing, batch mode, and split. '
            'No Optuna trial-number pairing is used in this path.'
        ),
    )
    parser.add_argument(
        '--xpysom-hexagonal-runs-file',
        type=str,
        default=None,
        help=(
            'Optional explicit self-contained XPySOM-vs-tuned-FloatSOM hexagonal benchmark CSV '
            'used for the supplementary tripanel figure. This CSV must already contain both '
            'method=floatsom and method=xpysom rows for the hexagonal comparison.'
        ),
    )
    parser.add_argument(
        '--xpysom-mst-runs-file',
        type=str,
        default=None,
        help=(
            'Optional explicit self-contained XPySOM-vs-tuned-FloatSOM MST benchmark CSV used '
            'for the supplementary tripanel figure. This CSV must already contain both '
            'method=floatsom and method=xpysom rows for the MST comparison.'
        ),
    )
    parser.add_argument(
        '--xpysom-rng-scaling-csv',
        type=str,
        default=None,
        help=(
            'Optional scaling CSV generated by run_xpysom_rng_scaling_comparison.py for panel D. '
            'If omitted, panel D is rendered as a placeholder.'
        ),
    )
    parser.add_argument(
        '--xpysom-rng-output-subdir',
        type=str,
        default='xpysom_rng_publication',
        help='Subdirectory under the run output for the dedicated XPySOM-vs-RNG figure.',
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=list(DEFAULT_METRICS),
        help='Normalized metric columns to analyze',
    )
    parser.add_argument(
        '--raw-metrics',
        nargs='+',
        default=list(RAW_METRICS),
        help='Raw metric columns to analyze for the raw-output suite',
    )
    parser.add_argument(
        '--include-normalized-set',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Generate normalized-output suite (default: false)',
    )
    parser.add_argument(
        '--include-raw-set',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate raw-output suite (default: true)',
    )
    parser.add_argument(
        '--full-suite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Generate supplementary/non-publication outputs in addition to manuscript-focused figures '
            '(default: false).'
        ),
    )
    parser.add_argument(
        '--sampling-full-vs-hdsssom-only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Restrict sampling-comparison publication outputs to Full vs HDSSOM only '
            '(default: false; include all available sampling comparisons).'
        ),
    )
    parser.add_argument(
        '--inline-legends',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Render legends inside each figure (default: false; standalone legend key per block is generated)',
    )
    parser.add_argument(
        '--strict-pairing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Fail if required pairing columns are missing (default: true)',
    )
    parser.add_argument(
        '--expected-pairs-per-dataset',
        type=int,
        default=None,
        help='Optional expected n_pairs per dataset; error if mismatched',
    )
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance threshold for coloring/annotation')
    parser.add_argument('--top-k', type=int, default=5, help='Use top-k trials per unit/group (median as robust score)')
    parser.add_argument(
        '--bootstrap-iterations',
        type=int,
        default=2000,
        help='Deprecated compatibility argument (ignored; paired t CI is used).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Deprecated compatibility argument (currently unused by paired t CI summaries).',
    )
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    args = parser.parse_args()

    data_file = Path(args.data_file)
    base_output_dir = Path(args.output_dir)
    full_suite = bool(args.full_suite)
    manuscript_only = not full_suite
    transient_workspace: Optional[TemporaryDirectory[str]] = None
    if manuscript_only:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        transient_workspace = TemporaryDirectory(
            prefix="publication_figures_",
            dir=str(base_output_dir.resolve()),
        )
        atexit.register(transient_workspace.cleanup)
        run_output_dir = Path(transient_workspace.name).resolve()
    else:
        run_output_dir = _resolve_run_output_dir(base_output_dir, args.run_name)
        run_output_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f'Input data file does not exist: {data_file}')
    if not data_file.is_file():
        raise IsADirectoryError(
            f'Input path is not a file: {data_file}. '
            'Pass a CSV file path, e.g. .../unified/processed_data_with_overall_score.csv'
        )

    df = pd.read_csv(data_file)
    df = _prepare_dataframe(df)
    df, excluded_minibatch_reporting_rows = _exclude_minibatch_rows_for_reporting(df)

    include_normalized = False
    include_raw = True
    inline_legends = bool(args.inline_legends)
    sampling_comparison_focus_key: Optional[str] = (
        'full_vs_hdsssom' if bool(args.sampling_full_vs_hdsssom_only) else None
    )
    if bool(args.include_normalized_set):
        print('Note: --include-normalized-set is ignored; this workflow runs raw metrics only.')

    metrics_normalized: List[str] = []
    metrics_raw: List[str] = []
    has_distortion_raw = any(
        col in df.columns
        for col in ['distortion_measure_holdout', 'distortion_measure_train', 'distortion_measure']
    )
    has_distortion_normalized = any(
        col in df.columns
        for col in ['distortion_measure_holdout_normalized', 'distortion_measure_train_normalized']
    )
    if include_normalized:
        requested_normalized = list(args.metrics)
        if not has_distortion_normalized:
            requested_normalized = [m for m in requested_normalized if 'distortion_measure' not in m]
            print('Note: distortion normalized metrics not found; skipping distortion in normalized suite.')
        metrics_normalized = _resolve_metric_columns(df, requested_normalized, strict_mode=bool(args.strict_pairing))
        if not metrics_normalized:
            raise ValueError('No normalized metrics were resolved from --metrics.')
    if include_raw:
        requested_raw = list(args.raw_metrics)
        if not has_distortion_raw:
            requested_raw = [m for m in requested_raw if 'distortion_measure' not in m]
            print('Note: distortion raw metrics not found; skipping distortion in raw suite.')
        metrics_raw = _resolve_metric_columns(df, requested_raw, strict_mode=bool(args.strict_pairing))
        if not metrics_raw:
            raise ValueError('No raw metrics were resolved from --raw-metrics.')

    print(f'Loaded data: {data_file}')
    print(f'Rows: {len(df)}')
    if excluded_minibatch_reporting_rows > 0:
        print(f'Excluded minibatch rows for reporting: {excluded_minibatch_reporting_rows}')
    if include_normalized:
        print(f'Resolved normalized metrics: {metrics_normalized}')
    if include_raw:
        print(f'Resolved raw metrics: {metrics_raw}')
    print(f'Full suite enabled: {full_suite}')
    print(f'Manuscript-only mode: {manuscript_only}')
    print(f'Inline legends: {inline_legends}')
    print(
        'Sampling comparison focus: '
        f"{sampling_comparison_focus_key or 'all available comparisons'}"
    )
    print(f'Run output directory: {run_output_dir}')

    generated_files: List[str] = []
    diagnostics: Dict[str, object] = {
        'rows_excluded_minibatch_reporting': int(excluded_minibatch_reporting_rows),
    }

    suite_outputs: Dict[str, Dict[str, object]] = {}
    suite_metrics: Dict[str, List[str]] = {}

    if include_normalized:
        _run_suite(
            suite_name='normalized',
            suite_metrics_list=metrics_normalized,
            df=df,
            run_output_dir=run_output_dir,
            suite_outputs=suite_outputs,
            suite_metrics=suite_metrics,
            top_k=args.top_k,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            dpi=args.dpi,
            alpha=args.alpha,
            strict_pairing=bool(args.strict_pairing),
            inline_legends=inline_legends,
            full_suite=full_suite,
            sampling_comparison_focus_key=sampling_comparison_focus_key,
            generated_files=generated_files,
            diagnostics=diagnostics,
        )
    if include_raw:
        _run_suite(
            suite_name='raw',
            suite_metrics_list=metrics_raw,
            df=df,
            run_output_dir=run_output_dir,
            suite_outputs=suite_outputs,
            suite_metrics=suite_metrics,
            top_k=args.top_k,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            dpi=args.dpi,
            alpha=args.alpha,
            strict_pairing=bool(args.strict_pairing),
            inline_legends=inline_legends,
            full_suite=full_suite,
            sampling_comparison_focus_key=sampling_comparison_focus_key,
            generated_files=generated_files,
            diagnostics=diagnostics,
        )

    diagnostics['paper_assets_sync_raw'] = _sync_sampling_comparison_assets_to_paper(
        suite_outputs=suite_outputs,
        generated_files=generated_files,
    )
    diagnostics['manuscript_sampling_regression_stats_sync'] = _sync_sampling_regression_stats_to_manuscript(
        suite_outputs=suite_outputs,
    )
    diagnostics['topology_pvalue_sync'] = _sync_topology_pvalue_summary_to_paper_and_manuscript(
        suite_outputs=suite_outputs,
        generated_files=generated_files,
    )
    diagnostics['systems_scaling_sync'] = _sync_systems_scaling_stats_to_manuscript()
    optimal_parameter_report: Dict[str, object] = {
        "generated": False,
        "reason": "Optimal-parameter reporting is disabled in manuscript-only mode.",
    }
    if not manuscript_only:
        optimal_parameter_report = _run_optimal_parameter_reporting(
            tuned_df=df,
            run_output_dir=run_output_dir,
            generated_files=generated_files,
            metric_col=OPTIMAL_PARAMETER_DEFAULT_METRIC,
        )
    default_aware_analysis: Dict[str, object] = {'enabled': False}
    xpysom_calibration_publication: Dict[str, object] = {'enabled': False}
    xpysom_rng_publication: Dict[str, object] = {'enabled': False}
    xpysom_topology_tripanel_publication: Dict[str, object] = {'enabled': False}
    default_aware_pairing_df: Optional[pd.DataFrame] = None
    default_aware_pairing_source: str = ''
    default_runs_file_arg: Optional[str] = args.default_runs_file
    default_aware_tuned_runs_file_arg: Optional[str] = args.default_aware_tuned_runs_file
    default_aware_manifest_source: Optional[str] = None
    if args.default_aware_manifest:
        if args.default_runs_file or args.default_aware_tuned_runs_file:
            raise ValueError(
                '--default-aware-manifest cannot be combined with --default-runs-file or '
                '--default-aware-tuned-runs-file.'
            )
        (
            manifest_default_runs_path,
            manifest_tuned_runs_path,
            manifest_source,
        ) = _resolve_default_aware_paths_from_manifest(args.default_aware_manifest)
        default_runs_file_arg = str(manifest_default_runs_path)
        default_aware_tuned_runs_file_arg = manifest_tuned_runs_path
        default_aware_manifest_source = manifest_source

    if default_runs_file_arg:
        default_runs_path = Path(default_runs_file_arg)
        if not default_runs_path.exists():
            raise FileNotFoundError(f'Default runs CSV does not exist: {default_runs_path}')
        pairing_df, pairing_source = _resolve_default_aware_pairing_input(
            base_df=df,
            data_file=data_file,
            override_csv_path=default_aware_tuned_runs_file_arg,
        )
        default_aware_pairing_df = pairing_df.copy()
        default_aware_pairing_source = str(pairing_source)
        default_aware_analysis = _run_default_aware_analysis(
            tuned_df=df,
            tuned_vs_default_df=pairing_df,
            tuned_vs_default_source=pairing_source,
            tuned_vs_default_base_source=str(data_file.resolve()),
            default_runs_file=default_runs_path,
            run_output_dir=run_output_dir,
            output_subdir=str(args.default_aware_output_subdir),
            generated_files=generated_files,
            dpi=int(args.dpi),
            alpha=float(args.alpha),
            manuscript_only=manuscript_only,
        )
        if default_aware_manifest_source:
            default_aware_analysis['manifest'] = default_aware_manifest_source
    diagnostics['paper_assets_sync_default_aware'] = _sync_default_aware_assets_to_paper(
        default_aware_analysis=default_aware_analysis,
        generated_files=generated_files,
    )
    diagnostics['manuscript_default_aware_stats_sync'] = _sync_default_aware_stats_to_manuscript(
        default_aware_analysis=default_aware_analysis,
    )
    diagnostics['manuscript_default_aware_stability_regression_sync'] = (
        _sync_default_aware_stability_regression_stats_to_manuscript(
            default_aware_analysis=default_aware_analysis,
        )
    )
    diagnostics['manuscript_figure10_topology_runtime_sync'] = (
        _sync_figure10_topology_runtime_stats_to_manuscript()
    )
    diagnostics['manuscript_figure11_deployment_runtime_sync'] = (
        _sync_figure11_deployment_runtime_stats_to_manuscript()
    )

    xpysom_calibration_csvs = {
        'mst': args.mst_xpysom_defaults_csv,
        'rng': args.rng_xpysom_defaults_csv,
        'hexagonal': args.hexagonal_xpysom_defaults_csv,
    }
    if any(bool(value) for value in xpysom_calibration_csvs.values()):
        xpysom_calibration_publication = _render_xpysom_default_calibration_publication_figures(
            topology_csvs=xpysom_calibration_csvs,
            output_dir=run_output_dir / 'xpysom_calibration_publication',
            generated_files=generated_files,
            manuscript_only=manuscript_only,
        )
    diagnostics['paper_assets_sync_xpysom_calibration_publication'] = _sync_xpysom_calibration_assets_to_paper(
        xpysom_calibration_publication=xpysom_calibration_publication,
        generated_files=generated_files,
    )

    xpysom_topology_tripanel_csvs = {
        'hexagonal': args.xpysom_hexagonal_runs_file,
        'mst': args.xpysom_mst_runs_file,
    }
    if any(bool(value) for value in xpysom_topology_tripanel_csvs.values()):
        xpysom_topology_tripanel_publication = _render_xpysom_topology_tripanel_publication_figures(
            topology_csvs=xpysom_topology_tripanel_csvs,
            output_dir=run_output_dir / 'xpysom_topology_tripanel_publication',
            generated_files=generated_files,
            dpi=int(args.dpi),
            alpha=float(args.alpha),
            manuscript_only=manuscript_only,
        )
    diagnostics['paper_assets_sync_xpysom_topology_tripanel_publication'] = (
        _sync_xpysom_topology_tripanel_assets_to_paper(
            xpysom_topology_tripanel_publication=xpysom_topology_tripanel_publication,
            generated_files=generated_files,
        )
    )

    xpysom_rng_requested = any(
        bool(value)
        for value in [
            args.xpysom_default_runs_file,
            args.xpysom_rng_runs_file,
            args.xpysom_rng_scaling_csv,
        ]
    )
    if xpysom_rng_requested:
        if not args.xpysom_default_runs_file:
            raise ValueError(
                '--xpysom-default-runs-file is required for the dedicated XPySOM-vs-RNG publication figure.'
            )
        if not args.xpysom_rng_runs_file:
            raise ValueError(
                '--xpysom-rng-runs-file is required for the dedicated XPySOM-vs-RNG publication figure. '
                'Provide the matched FloatSOM RNG benchmark CSV explicitly.'
            )
        xpysom_rng_publication = _render_xpysom_rng_scaling_publication_figure(
            xpysom_default_runs_file=str(args.xpysom_default_runs_file),
            matched_rng_runs_file=str(args.xpysom_rng_runs_file),
            scaling_csv_file=(
                str(args.xpysom_rng_scaling_csv) if args.xpysom_rng_scaling_csv else None
            ),
            output_dir=run_output_dir / str(args.xpysom_rng_output_subdir),
            generated_files=generated_files,
            dpi=int(args.dpi),
            manuscript_only=manuscript_only,
        )
    diagnostics['paper_assets_sync_xpysom_rng_publication'] = _sync_xpysom_rng_publication_assets_to_paper(
        xpysom_rng_publication=xpysom_rng_publication,
        generated_files=generated_files,
    )
    paper_assets_dir = _resolve_paper_assets_dir()
    if paper_assets_dir is None:
        diagnostics['paper_assets_svg_postprocess'] = {
            'updated': False,
            'reason': 'Could not locate floatsom/paper/assets from script path.',
        }
    else:
        diagnostics['paper_assets_svg_postprocess'] = _normalize_publication_svg_assets(
            paper_assets_dir / 'figures',
        )

    print('\nPublication figure generation complete.')
    if manuscript_only:
        print('Transient workspace cleaned after manuscript sync.')
    else:
        print(f'Run directory: {run_output_dir}')
    print(f"Optimal parameter report: {optimal_parameter_report.get('markdown_report')}")
    if default_aware_analysis.get('enabled'):
        variant_count = 0
        variants = default_aware_analysis.get('tuned_vs_default_variants', [])
        if isinstance(variants, list):
            variant_count = len([item for item in variants if isinstance(item, dict)])
        print(
            'Default-aware analysis: '
            f"{default_aware_analysis.get('output_dir')} "
            f'(tuned-vs-default variants={variant_count})'
        )
    if xpysom_calibration_publication.get('enabled'):
        print(
            'XPySOM calibration publication figures: '
            f"{xpysom_calibration_publication.get('output_dir')}"
        )
    if xpysom_rng_publication.get('enabled'):
        print(
            'XPySOM RNG publication figure: '
            f"{xpysom_rng_publication.get('combined_figure')}"
        )
    if xpysom_topology_tripanel_publication.get('enabled'):
        print(
            'XPySOM topology supplementary tripanels: '
            f"{xpysom_topology_tripanel_publication.get('output_dir')}"
        )
    if not manuscript_only:
        manifest = {
            'generated_utc': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'data_file': str(data_file.resolve()),
            'run_output_dir': str(run_output_dir.resolve()),
            'params': {
                'suite_metrics': suite_metrics,
                'include_normalized_set': include_normalized,
                'include_raw_set': include_raw,
                'full_suite': full_suite,
                'top_k': int(args.top_k),
                'bootstrap_iterations': int(args.bootstrap_iterations),
                'seed': int(args.seed),
                'dpi': int(args.dpi),
                'alpha': float(args.alpha),
                'strict_pairing': bool(args.strict_pairing),
                'expected_pairs_per_dataset': args.expected_pairs_per_dataset,
                'inline_legends': inline_legends,
            },
            'suite_outputs': suite_outputs,
            'optimal_parameter_report': optimal_parameter_report,
            'default_aware_analysis': default_aware_analysis,
            'xpysom_calibration_publication': xpysom_calibration_publication,
            'xpysom_rng_publication': xpysom_rng_publication,
            'xpysom_topology_tripanel_publication': xpysom_topology_tripanel_publication,
            'diagnostics': diagnostics,
            'generated_files': sorted(set(generated_files)),
        }
        manifest_path = run_output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Manifest: {manifest_path}')
    if transient_workspace is not None:
        transient_workspace.cleanup()
    return 0


__all__ = [
    name
    for name in globals()
    if ((name.startswith("_") and not name.startswith("__")) or name.isupper() or name == "main")
]
