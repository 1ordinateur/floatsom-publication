"""
ProcessorFactory - Factory pattern for creating appropriate processors
Eliminates circular dependencies and provides clean instantiation
"""

from typing import Optional, Tuple
import logging

from .base_processor import ProcessingMethod
from ..floatsom_params import RayConfig, ProcessingConfig
from .processing_params import BatchConfig, ColorsConfig, ChunkingConfig

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    Factory for creating processing method instances based on configuration.
    Follows SOLID principles by depending on abstractions (ProcessingMethod).
    """

    @staticmethod
    def _assert_local_gpu_available(config: ProcessingConfig) -> None:
        """Ensure a local GPU is available for FloatSOM runtime processing."""

        try:
            import cupy as cp  # type: ignore
        except Exception as exc:  # pragma: no cover - best effort import guard
            raise RuntimeError(
                "GPU processing requested but CuPy is not available. "
                "Install CuPy."
            ) from exc

        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:  # pragma: no cover - CUDA runtime errors vary by platform
            raise RuntimeError(
                "GPU processing requested but CUDA devices could not be detected."
            ) from exc

        if gpu_count <= 0:
            raise RuntimeError(
                "GPU processing requested but no CUDA devices were detected."
            )
    
    @staticmethod
    def create(processing_config: ProcessingConfig,
               n_features: Optional[int] = None,
               n_nodes: Optional[int] = None) -> ProcessingMethod:
        """
        Create appropriate processor based on configuration.
        
        Args:
            processing_config: ProcessingConfig with method and parameters
            n_features: Optional number of features for chunk size calculation
            n_nodes: Optional number of SOM nodes for chunk size calculation
            
        Returns:
            ProcessingMethod implementation (BatchProcessor, ColorsProcessor, etc.)
            
        Raises:
            ValueError: If processing method is not supported
        """
        method = processing_config.method
        
        if method == "batch":
            return ProcessorFactory._create_batch_processor(processing_config, n_features, n_nodes)
        elif method == "colors":
            return ProcessorFactory._create_colors_processor(processing_config, n_features, n_nodes)
        else:
            raise ValueError(f"Unknown processing method: {method}")
    
    @staticmethod
    def _create_batch_processor(config: ProcessingConfig,
                               n_features: Optional[int] = None,
                               n_nodes: Optional[int] = None) -> ProcessingMethod:
        """
        Create batch processor - either single-GPU or multi-GPU Ray version.
        
        Args:
            config: ProcessingConfig with batch parameters and optional ray_config
            n_features: Optional number of features for chunk size calculation
            n_nodes: Optional number of SOM nodes for chunk size calculation
            
        Returns:
            BatchProcessor or RayBatchProcessor instance
        """
        # Chunk size must be explicitly provided
        chunk_size = config.chunk_size
        if chunk_size is None:
            raise ValueError("chunk_size must be explicitly provided in ProcessingConfig")
        
        # Create BatchConfig with guaranteed chunk_size
        batch_config = BatchConfig(
            batch_mode=config.batch_mode,
            chunk_size=chunk_size,  # Always set
            weight_update_frequency=getattr(config, 'weight_update_frequency', 10)
        )
        
        # Check if Ray multi-GPU is requested
        if config.ray_config is not None:
            # Import Ray processor - fail if not available
            from .ray_ops.ray_batch_processor import RayBatchProcessor
            logger.info("Creating RayBatchProcessor for multi-GPU training")
            return RayBatchProcessor(batch_config, config.ray_config, config)
        
        # Create standard single-GPU processor
        ProcessorFactory._assert_local_gpu_available(config)
        from .batch_processor import BatchProcessor
        logger.info("Creating BatchProcessor for single-GPU training")
        return BatchProcessor(batch_config)
    
    @staticmethod
    def _create_colors_processor(config: ProcessingConfig,
                                n_features: Optional[int] = None,
                                n_nodes: Optional[int] = None) -> ProcessingMethod:
        """
        Create colors processor - either single-GPU or multi-GPU Ray version.
        
        Args:
            config: ProcessingConfig with colors parameters
            n_features: Optional number of features for chunk size calculation
            n_nodes: Optional number of SOM nodes for chunk size calculation
            
        Returns:
            ColorsProcessor or RayColorsProcessor instance
        """
        # Chunk size must be explicitly provided
        chunk_size = config.chunk_size
        if chunk_size is None:
            raise ValueError("chunk_size must be explicitly provided in ProcessingConfig")

        # Create ColorsConfig from ProcessingConfig with guaranteed chunk_size
        colors_config = ColorsConfig(
            processing_mode=config.processing_mode,
            max_rounds=config.max_rounds,
            sample_order=config.sample_order,
            color_set_algorithm=config.color_set_algorithm,
            num_color_sets=config.num_color_sets,
            chunk_size=chunk_size,
            enable_adaptive_bmu=config.enable_adaptive_bmu,
            bmu_recalc_initial=config.bmu_recalc_initial,
            bmu_recalc_decay_type=config.bmu_recalc_decay_type,
            bmu_recalc_min_samples=config.bmu_recalc_min_samples,
            recompute_bmus_per_color_set=config.recompute_bmus_per_color_set,
        )
        
        # Check if Ray multi-GPU is requested
        if config.ray_config is not None:
            # Import Ray processor - fail if not available
            from .ray_ops.ray_colors_processor import RayColorsProcessor
            logger.info("Creating RayColorsProcessor for multi-GPU training")
            return RayColorsProcessor(colors_config, config.ray_config, config)
        
        # Create standard colors processor
        ProcessorFactory._assert_local_gpu_available(config)
        from .colors_processor import ColorsProcessor
        logger.info("Creating ColorsProcessor for single-GPU training")
        return ColorsProcessor(colors_config)
    
    @staticmethod
    def create_from_params(params) -> ProcessingMethod:
        """
        Convenience method to create processor from FloatSOMParams.
        
        Args:
            params: FloatSOMParams instance
            
        Returns:
            ProcessingMethod implementation
        """
        if not hasattr(params, 'processing_config'):
            raise ValueError("params must have processing_config attribute")
        
        return ProcessorFactory.create(params.processing_config)
