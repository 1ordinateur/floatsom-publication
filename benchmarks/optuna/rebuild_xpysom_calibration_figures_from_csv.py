#!/usr/bin/env python3
"""
Rebuild XPySOM calibration figures/tables from existing CSV outputs.

This script does not rerun training. It only regenerates summary tables,
forest stats, and SVG figures by reusing the existing benchmark figure code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from benchmark_xpysom_hex_batch_full import (
    MST_VS_RNG_FIGURE_FILENAME,
    MST_VS_RNG_FOREST_STATS_FILENAME,
    _build_summary_table,
    _figure_filename_for_topology,
    _forest_stats_filename_for_topology,
    _generate_mst_vs_rng_final_figure,
    _generate_xpysom_calibration_figure,
    _publish_to_paper_assets,
    _resolve_floatsom_topologies,
    _table_filename_for_topology,
)


SUMMARY_COLUMNS: List[str] = [
    "dataset",
    "metric",
    "split",
    "wins_floatsom",
    "wins_xpysom",
    "ties",
    "win_rate_floatsom",
    "median_delta_raw",
    "median_pct_improvement",
    "mean_pct_improvement",
    "ci_low_pct",
    "ci_high_pct",
    "p_value",
    "n_pairs",
    "notes",
    "effect_size_signed",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild XPySOM calibration figures from existing benchmark CSV files."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Directory containing topology subfolders and/or combined runs CSV.",
    )
    parser.add_argument(
        "--combined-runs-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit combined runs CSV path. "
            "If omitted, defaults to <input-root>/xpysom_batch_topology_sweep_runs.csv when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for regenerated artifacts. Default: <input-root>/rebuilt_xpysom_figures",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        type=str,
        default=["hexagonal", "mst", "rng"],
        help="Topology list to regenerate.",
    )
    parser.add_argument(
        "--skip-mst-vs-rng-final",
        action="store_true",
        help="Skip regeneration of the MST-vs-RNG final figure.",
    )
    parser.add_argument(
        "--publish-paper-assets",
        action="store_true",
        default=True,
        help="Copy regenerated files into floatsom/paper/assets.",
    )
    parser.add_argument(
        "--no-publish-paper-assets",
        action="store_false",
        dest="publish_paper_assets",
        help="Do not copy regenerated files into paper assets.",
    )
    return parser.parse_args()


def _load_combined_runs_frame(input_root: Path, combined_csv_arg: Optional[str]) -> Optional[pd.DataFrame]:
    combined_path = Path(combined_csv_arg).expanduser().resolve() if combined_csv_arg else (
        input_root / "xpysom_batch_topology_sweep_runs.csv"
    )
    if not combined_path.exists():
        return None
    frame = pd.read_csv(combined_path)
    if frame.empty:
        return None
    return frame


def _load_topology_frame(
    *,
    topology: str,
    input_root: Path,
    combined_frame: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if combined_frame is not None and not combined_frame.empty:
        if "comparison_topology" in combined_frame.columns:
            subset = combined_frame[
                combined_frame["comparison_topology"].astype(str).str.lower().str.strip() == str(topology).lower()
            ].copy()
            if not subset.empty:
                return subset.sort_values(["dataset", "seed", "method"]).reset_index(drop=True)

    topology_csv = input_root / str(topology) / f"xpysom_{topology}_batch_full_runs.csv"
    if topology_csv.exists():
        frame = pd.read_csv(topology_csv)
        if not frame.empty:
            return frame.sort_values(["dataset", "seed", "method"]).reset_index(drop=True)
    return None


def _write_topology_outputs(
    *,
    topology: str,
    runs_df: pd.DataFrame,
    output_dir: Path,
    publish_paper_assets: bool,
) -> Dict[str, Path]:
    topology_output_dir = output_dir / str(topology)
    topology_output_dir.mkdir(parents=True, exist_ok=True)

    topology_runs_csv = topology_output_dir / f"xpysom_{topology}_batch_full_runs.csv"
    runs_df.to_csv(topology_runs_csv, index=False)

    summary_df = _build_summary_table(runs_df)
    summary_df = summary_df[SUMMARY_COLUMNS]
    summary_tsv_path = topology_output_dir / _table_filename_for_topology(topology)
    summary_df.to_csv(summary_tsv_path, sep="\t", index=False)

    figure_path = topology_output_dir / _figure_filename_for_topology(topology)
    forest_stats_path = topology_output_dir / _forest_stats_filename_for_topology(topology)
    _generate_xpysom_calibration_figure(
        runs_df=runs_df,
        topology=topology,
        output_path=figure_path,
        stats_output_path=forest_stats_path,
    )

    outputs: Dict[str, Path] = {
        "runs_csv": topology_runs_csv,
        "summary_tsv": summary_tsv_path,
        "forest_stats_csv": forest_stats_path,
        "figure": figure_path,
    }
    if publish_paper_assets:
        published = _publish_to_paper_assets(
            topology=topology,
            summary_tsv_path=summary_tsv_path,
            figure_path=figure_path,
        )
        outputs["paper_table"] = published["table"]
        outputs["paper_figure"] = published["figure"]
        if "figure_main" in published:
            outputs["paper_main_figure"] = published["figure_main"]
    return outputs


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        input_root / "rebuilt_xpysom_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    topologies = _resolve_floatsom_topologies(args.topologies)
    combined_frame = _load_combined_runs_frame(input_root, args.combined_runs_csv)
    frames_by_topology: Dict[str, pd.DataFrame] = {}

    print("Rebuilding XPySOM calibration figures from CSV")
    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Topologies: {topologies}")
    print(f"Publish to paper assets: {bool(args.publish_paper_assets)}")
    print(f"Combined runs CSV detected: {bool(combined_frame is not None)}")

    for topology in topologies:
        topology_df = _load_topology_frame(
            topology=topology,
            input_root=input_root,
            combined_frame=combined_frame,
        )
        if topology_df is None or topology_df.empty:
            print(f"[{topology}] skipped: no usable runs CSV found.")
            continue

        frames_by_topology[topology] = topology_df.copy()
        outputs = _write_topology_outputs(
            topology=topology,
            runs_df=topology_df,
            output_dir=output_dir,
            publish_paper_assets=bool(args.publish_paper_assets),
        )

        print(f"[{topology}] runs CSV: {outputs['runs_csv']}")
        print(f"[{topology}] summary TSV: {outputs['summary_tsv']}")
        print(f"[{topology}] forest stats CSV: {outputs['forest_stats_csv']}")
        print(f"[{topology}] figure: {outputs['figure']}")
        if "paper_table" in outputs:
            print(f"[{topology}] paper table: {outputs['paper_table']}")
        if "paper_figure" in outputs:
            print(f"[{topology}] paper figure: {outputs['paper_figure']}")
        if "paper_main_figure" in outputs:
            print(f"[{topology}] paper main figure: {outputs['paper_main_figure']}")

    if not bool(args.skip_mst_vs_rng_final):
        if {"mst", "rng"}.issubset(set(frames_by_topology.keys())):
            mst_rng_runs = pd.concat(
                [frames_by_topology["mst"], frames_by_topology["rng"]],
                ignore_index=True,
            )
            mst_rng_figure_path = output_dir / MST_VS_RNG_FIGURE_FILENAME
            mst_rng_stats_path = output_dir / MST_VS_RNG_FOREST_STATS_FILENAME
            _generate_mst_vs_rng_final_figure(
                runs_df=mst_rng_runs,
                output_path=mst_rng_figure_path,
                stats_output_path=mst_rng_stats_path,
            )
            print(f"[mst_vs_rng] figure: {mst_rng_figure_path}")
            print(f"[mst_vs_rng] forest stats CSV: {mst_rng_stats_path}")
        else:
            print("[mst_vs_rng] skipped: both MST and RNG runs are required.")

    print("Rebuild complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
