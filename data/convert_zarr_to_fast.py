#!/usr/bin/env python3
"""Convert existing Zarr arrays to FastArrayStore format"""

import numpy as np
import argparse
import logging
import os
import sys
import time
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .fast_array_store import FastArrayStore
from .zarr_utils import open_array_read
from .zarr_to_fast_shard_mp import convert_zarr_slice_to_fast_shard_mp
# chunk_size must now be explicitly provided

def convert_zarr_to_fast(zarr_path: str, output_path: str, chunk_size: int,
                         num_workers: Optional[int] = None):
    """
    Convert Zarr to FastArrayStore for 15x performance improvement.
    
    Args:
        zarr_path: Input Zarr path
        output_path: Output FastArrayStore path
        chunk_size: Chunk size for the FastArrayStore (must be explicitly provided)
        num_workers: Optional number of worker processes (default: all available CPUs)
    """
    logger.info(f"Converting {zarr_path} to FastArrayStore...")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Chunk size: {chunk_size:,} samples")
    
    # Open Zarr
    z = open_array_read(zarr_path).array
    logger.info(f"Source shape: {z.shape}, dtype: {z.dtype}")
    
    # Calculate total size
    total_bytes = z.shape[0] * (z.shape[1] if len(z.shape) > 1 else 1) * 4  # float32
    logger.info(f"Total data size: {total_bytes / (1024**3):.2f} GB")
    
    # Create FastArrayStore
    store = FastArrayStore(output_path, mode='w')
    n_features = z.shape[1] if len(z.shape) > 1 else 1
    chunks = (chunk_size, n_features) if len(z.shape) > 1 else (chunk_size,)
    arr = store.create(
        shape=z.shape,
        dtype=np.float32,  # Always use float32
        chunks=chunks,
    )
    arr.flush()

    total_samples = z.shape[0]
    data_file = store.data_file
    del arr
    store.close()

    tasks = [
        (start, min(start + chunk_size, total_samples))
        for start in range(0, total_samples, chunk_size)
    ]
    available_workers = num_workers or os.cpu_count() or 1
    worker_count = max(1, min(len(tasks), int(available_workers)))

    start_time = time.time()

    def _progress_cb(rows_done: int, total_rows: int, elapsed_s: float) -> None:
        progress = (rows_done / total_rows * 100.0) if total_rows else 100.0
        throughput = (rows_done / elapsed_s) if elapsed_s > 0 else 0.0
        logger.info(
            f"  Progress: {rows_done:,}/{total_rows:,} samples "
            f"({progress:.1f}%) - {throughput:.0f} samples/sec"
        )

    logger.info(f"Using {worker_count} worker process(es) for conversion")
    rows_done, elapsed_s = convert_zarr_slice_to_fast_shard_mp(
        zarr_path=zarr_path,
        out_data_file=data_file,
        shard_shape=z.shape,
        global_start=0,
        tasks=tasks,
        num_procs=worker_count,
        progress_cb=_progress_cb,
    )
    samples_copied = int(rows_done)
    
    # Report completion
    elapsed = time.time() - start_time
    throughput_gb = total_bytes / (1024**3) / elapsed if elapsed > 0 else 0
    logger.info(f"\nConversion complete in {elapsed:.1f} seconds")
    logger.info(f"Average throughput: {throughput_gb:.2f} GB/s")
    logger.info(f"Output: {output_path}")
    
    # Verify the output
    logger.info("\nVerifying output...")
    verify_store = FastArrayStore(output_path, mode='r')
    assert verify_store.shape == z.shape, "Shape mismatch!"
    assert verify_store.n_samples == z.shape[0], "Sample count mismatch!"
    verify_store.close()
    logger.info("Verification successful!")


def batch_convert(input_dir: str, output_dir: str, pattern: str = "*.zarr", chunk_size: int = None,
                  num_workers: Optional[int] = None):
    """
    Convert all Zarr files in a directory to FastArrayStore format.
    
    Args:
        input_dir: Directory containing Zarr files
        output_dir: Directory for FastArrayStore outputs
        pattern: Glob pattern for finding Zarr files
        chunk_size: Chunk size for FastArrayStore (must be provided)
        num_workers: Optional number of worker processes per conversion
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zarr_files = list(input_path.glob(pattern))
    
    if not zarr_files:
        logger.info(f"No files matching pattern '{pattern}' found in {input_dir}")
        return
    
    logger.info(f"Found {len(zarr_files)} Zarr files to convert")

    if chunk_size is None:
        raise ValueError("chunk_size must be provided for batch conversion")
    
    for i, zarr_file in enumerate(zarr_files, 1):
        logger.info(f"\n[{i}/{len(zarr_files)}] Converting {zarr_file.name}")
        
        # Create output path with .fast extension
        output_name = zarr_file.stem + '.fast'
        output_file = output_path / output_name
        
        convert_zarr_to_fast(
            str(zarr_file),
            str(output_file),
            chunk_size,
            num_workers=num_workers,
        )
    
    logger.info(f"\nBatch conversion complete. Converted {len(zarr_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Zarr arrays to FastArrayStore format for 15x performance improvement"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single file conversion
    single_parser = subparsers.add_parser('convert', help='Convert a single Zarr file')
    single_parser.add_argument('zarr_path', help='Input Zarr path')
    single_parser.add_argument('output_path', help='Output FastArrayStore path')
    single_parser.add_argument('--chunk-size', type=int, required=True,
                              help='Chunk size (must be explicitly provided)')
    single_parser.add_argument('--workers', type=int, default=None,
                              help='Number of worker processes (default: all available)')
    
    # Batch conversion
    batch_parser = subparsers.add_parser('batch', help='Convert all Zarr files in a directory')
    batch_parser.add_argument('input_dir', help='Input directory containing Zarr files')
    batch_parser.add_argument('output_dir', help='Output directory for FastArrayStore files')
    batch_parser.add_argument('--pattern', default='*.zarr',
                             help='Glob pattern for Zarr files (default: *.zarr)')
    batch_parser.add_argument('--chunk-size', type=int, required=True,
                             help='Chunk size (must be explicitly provided)')
    batch_parser.add_argument('--workers', type=int, default=None,
                             help='Number of worker processes (default: all available)')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        convert_zarr_to_fast(args.zarr_path, args.output_path, args.chunk_size,
                             num_workers=args.workers)
    elif args.command == 'batch':
        batch_convert(args.input_dir, args.output_dir, args.pattern,
                      args.chunk_size, num_workers=args.workers)
    else:
        parser.print_help()
