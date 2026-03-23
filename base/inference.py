import logging
from typing import Optional, Union

import cupy as cp
import numpy as np

from ..data.sources.array import ArrayDataSource

logger = logging.getLogger(__name__)


class InferenceMixin:
    """Mixin providing inference routines for FloatSOM."""

    def infer(
        self,
        data_input: Union[np.ndarray, cp.ndarray, str],
        output_path: Optional[str] = None,
        chunk_size: int = 100000,
        return_in_memory: bool = False,
        use_cuda_streams: bool = True,
        verbose: Optional[bool] = None,
    ) -> Union[np.ndarray, str]:
        if not self.is_trained:
            raise ValueError("SOM must be trained before inference")

        verbose = verbose if verbose is not None else self.verbose
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

        data_source = self._create_inference_data_source(data_input)
        n_samples, _ = data_source.get_shape()

        if verbose:
            logger.info(f"Starting inference on {n_samples:,} samples")
            logger.info(f"Chunk size: {chunk_size:,}, Output: {output_path or 'memory'}")

        output_array = None
        zarr_output_array = None
        zarr_output_path = None

        if output_path:
            from ..data.zarr_utils import create_sharded_array

            zarr_output_path = str(output_path)
            output_chunk = max(1, min(chunk_size, max(1, n_samples)))
            zarr_output_array = create_sharded_array(
                zarr_output_path,
                array_key="assignments",
                shape=(n_samples,),
                dtype="i4",
                chunks=(output_chunk,),
                overwrite=True,
            ).array
            if verbose:
                logger.info(f"Writing results to zarr: {output_path}")
        else:
            output_array = np.empty(n_samples, dtype=np.int32)
            if verbose:
                logger.info("Collecting results in memory")

        result = self._infer_gpu(
            data_source,
            output_array,
            zarr_output_array,
            zarr_output_path,
            chunk_size,
            use_cuda_streams,
            verbose,
        )

        data_source.cleanup()
        return result

    def _create_inference_data_source(self, data_input):
        if isinstance(data_input, str):
            data_source = self.data_source_factory.create(data_input, None)
        elif isinstance(data_input, (np.ndarray, cp.ndarray)):
            data_source = self.data_source_factory.create(data_input, None)
        else:
            raise ValueError(f"Unsupported data input type: {type(data_input)}")
        return data_source

    def _infer_gpu(
        self,
        data_source,
        output_array,
        zarr_output_array,
        zarr_output_path,
        chunk_size,
        use_cuda_streams,
        verbose,
    ):
        with cp.cuda.Device(0):
            if not isinstance(self.weights, cp.ndarray):
                self.weights = cp.asarray(self.weights)

            n_samples = data_source.get_shape()[0]
            n_features = self.weights.shape[1]
            original_chunk_size = chunk_size
            n_nodes = self.weights.shape[0]
            itemsize = self.weights.dtype.itemsize
            if n_nodes > 0 and n_features > 0 and chunk_size > 0:
                free_mem, _ = cp.cuda.runtime.memGetInfo()
                node_chunk = min(900, n_nodes) if n_nodes > 0 else 1
                float32_size = np.dtype(np.float32).itemsize
                int32_size = np.dtype(np.int32).itemsize
                stream_buffers = 2 if use_cuda_streams else 1
                per_sample_bytes = (
                    stream_buffers * ((n_features * itemsize) + int32_size)
                    + node_chunk * float32_size
                )
                per_sample_bytes = max(
                    per_sample_bytes,
                    (n_features * itemsize) + node_chunk * float32_size + int32_size,
                )
                safe_mem = int(free_mem * 0.45)
                if per_sample_bytes > 0 and safe_mem > 0:
                    max_chunk = max(1, int(safe_mem / per_sample_bytes))
                    if max_chunk < chunk_size:
                        if verbose:
                            logger.info(
                                f"Reducing inference chunk size from {chunk_size:,} to {max_chunk:,} to fit GPU memory"
                            )
                        chunk_size = max_chunk
            chunk_size = min(chunk_size, n_samples)
            chunk_size = max(1, chunk_size)
            if chunk_size != original_chunk_size and verbose:
                logger.info(f"Using inference chunk size {chunk_size:,} (requested {original_chunk_size:,})")
            n_chunks = (n_samples + chunk_size - 1) // chunk_size

            if isinstance(data_source, ArrayDataSource) and data_source.is_gpu:
                return self._infer_gpu_resident(
                    data_source,
                    output_array,
                    zarr_output_array,
                    zarr_output_path,
                    chunk_size,
                    n_samples,
                    verbose,
                )

            if use_cuda_streams and n_chunks > 1:
                return self._infer_gpu_streamed(
                    data_source,
                    output_array,
                    zarr_output_array,
                    zarr_output_path,
                    chunk_size,
                    n_samples,
                    n_features,
                    verbose,
                )
            else:
                return self._infer_gpu_simple(
                    data_source,
                    output_array,
                    zarr_output_array,
                    zarr_output_path,
                    chunk_size,
                    n_samples,
                    verbose,
                )

    def _infer_gpu_streamed(
        self,
        data_source,
        output_array,
        zarr_output_array,
        zarr_output_path,
        chunk_size,
        n_samples,
        n_features,
        verbose,
    ):
        compute_stream = cp.cuda.Stream(non_blocking=True)
        transfer_stream = cp.cuda.Stream(non_blocking=True)

        buffers = [
            cp.empty((chunk_size, n_features), dtype=cp.float32),
            cp.empty((chunk_size, n_features), dtype=cp.float32),
        ]
        result_buffers = [
            cp.empty(chunk_size, dtype=cp.int32),
            cp.empty(chunk_size, dtype=cp.int32),
        ]
        use_pinned = zarr_output_array is not None
        if use_pinned:
            pinned_stride = np.dtype(np.int32).itemsize * chunk_size
            host_buffers = [cp.cuda.alloc_pinned_memory(pinned_stride) for _ in range(2)]
            host_views = [
                np.frombuffer(host_buffers[idx], dtype=np.int32, count=chunk_size)
                for idx in range(2)
            ]

        from ..data.cpugpu_fast_loader import CPUGPUFastLoader

        if hasattr(data_source, "get_path"):
            loader = CPUGPUFastLoader(
                array_path=data_source.get_path(),
                chunk_size=chunk_size,
                randomize_chunks=False,
            )
        else:
            loader = CPUGPUFastLoader(
                ram_data=data_source.get_reference(),
                ram_mode=True,
                chunk_size=chunk_size,
                randomize_chunks=False,
            )

        total_chunks = (n_samples + chunk_size - 1) // chunk_size

        with transfer_stream:
            chunk_data, _ = loader.get_chunk(
                0,
                use_randomized_order=False,
                advance_cursor=True,
            )
            actual_chunk_size = min(chunk_size, n_samples)
            buffers[0][:actual_chunk_size] = chunk_data[:actual_chunk_size]

        for chunk_idx in range(total_chunks):
            buf_idx = chunk_idx % 2
            next_buf_idx = (chunk_idx + 1) % 2

            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            actual_size = end_idx - start_idx

            if chunk_idx > 0:
                # Chunk `chunk_idx` was prefetched previously with advance_cursor=False
                # and becomes consumed at this point in the pipeline.
                loader.mark_chunk_consumed(chunk_idx)

            if chunk_idx + 1 < total_chunks:
                with transfer_stream:
                    next_chunk, _ = loader.get_chunk(
                        chunk_idx + 1,
                        use_randomized_order=False,
                        advance_cursor=False,
                    )
                    next_size = min(chunk_size, n_samples - (chunk_idx + 1) * chunk_size)
                    buffers[next_buf_idx][:next_size] = next_chunk[:next_size]

            with compute_stream:
                chunk_buffer = buffers[buf_idx][:actual_size]
                bmu_indices = self.predict(chunk_buffer)
                result_buffers[buf_idx][:actual_size] = bmu_indices
                del bmu_indices

            compute_stream.synchronize()

            if use_pinned:
                with transfer_stream:
                    cp.cuda.runtime.memcpyAsync(
                        host_buffers[buf_idx].ptr,
                        result_buffers[buf_idx].data.ptr,
                        actual_size * host_views[buf_idx].itemsize,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        transfer_stream.ptr,
                    )
                transfer_stream.synchronize()
                result_cpu = host_views[buf_idx][:actual_size]
            else:
                result_cpu = cp.asnumpy(result_buffers[buf_idx][:actual_size])
            if output_array is not None:
                output_array[start_idx:end_idx] = result_cpu
            if zarr_output_array is not None:
                zarr_output_array[start_idx:end_idx] = result_cpu

            if verbose and (chunk_idx + 1) % 10 == 0:
                logger.info(f"Processed {start_idx + actual_size:,}/{n_samples:,} samples")

        if hasattr(loader, "cleanup"):
            loader.cleanup()

        return output_array if output_array is not None else zarr_output_path

    def _infer_gpu_simple(
        self,
        data_source,
        output_array,
        zarr_output_array,
        zarr_output_path,
        chunk_size,
        n_samples,
        verbose,
    ):
        from ..data.cpugpu_fast_loader import CPUGPUFastLoader

        if hasattr(data_source, "get_path"):
            loader = CPUGPUFastLoader(
                array_path=data_source.get_path(),
                chunk_size=chunk_size,
                randomize_chunks=False,
            )
        else:
            loader = CPUGPUFastLoader(
                ram_data=data_source.get_reference(),
                ram_mode=True,
                chunk_size=chunk_size,
                randomize_chunks=False,
            )

        total_chunks = (n_samples + chunk_size - 1) // chunk_size
        use_pinned = zarr_output_array is not None
        if use_pinned:
            pinned_stride = np.dtype(np.int32).itemsize * chunk_size
            host_buffer = cp.cuda.alloc_pinned_memory(pinned_stride)
            host_view = np.frombuffer(host_buffer, dtype=np.int32, count=chunk_size)

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)

            gpu_chunk, _ = loader.get_chunk(
                chunk_idx,
                use_randomized_order=False,
                advance_cursor=True,
            )
            actual_size = end_idx - start_idx
            gpu_chunk = gpu_chunk[:actual_size]

            bmu_indices = self.predict(gpu_chunk)
            if use_pinned:
                cp.cuda.runtime.memcpyAsync(
                    host_buffer.ptr,
                    bmu_indices.data.ptr,
                    actual_size * host_view.itemsize,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    0,
                )
                cp.cuda.Stream.null.synchronize()
                bmu_indices_cpu = host_view[:actual_size]
            else:
                bmu_indices_cpu = cp.asnumpy(bmu_indices)

            if output_array is not None:
                output_array[start_idx:end_idx] = bmu_indices_cpu
            if zarr_output_array is not None:
                zarr_output_array[start_idx:end_idx] = bmu_indices_cpu

            if verbose and (chunk_idx + 1) % 10 == 0:
                logger.info(f"Processed chunk {chunk_idx + 1}/{total_chunks}")

        if hasattr(loader, "cleanup"):
            loader.cleanup()
        return output_array if output_array is not None else zarr_output_path

    def _infer_gpu_resident(
        self,
        data_source,
        output_array,
        zarr_output_array,
        zarr_output_path,
        chunk_size,
        n_samples,
        verbose,
    ):
        gpu_data = data_source.get_reference()
        if chunk_size <= 0:
            chunk_size = 1
        total_chunks = (n_samples + chunk_size - 1) // chunk_size if n_samples > 0 else 0

        collect_on_device = output_array is not None and zarr_output_array is None
        gpu_result = None
        if collect_on_device:
            gpu_result = cp.empty(n_samples, dtype=cp.int32)

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            if end_idx <= start_idx:
                continue

            chunk = gpu_data[start_idx:end_idx]
            bmu_indices = self.predict(chunk)

            if collect_on_device:
                gpu_result[start_idx:end_idx] = bmu_indices
            else:
                bmu_indices_cpu = cp.asnumpy(bmu_indices)
                if output_array is not None:
                    output_array[start_idx:end_idx] = bmu_indices_cpu
                if zarr_output_array is not None:
                    zarr_output_array[start_idx:end_idx] = bmu_indices_cpu

            if verbose and (chunk_idx + 1) % 10 == 0:
                logger.info(f"Processed chunk {chunk_idx + 1}/{total_chunks} (GPU-resident data)")

        if collect_on_device and gpu_result is not None:
            output_array[:] = cp.asnumpy(gpu_result)
            return output_array

        return output_array if output_array is not None else zarr_output_path

