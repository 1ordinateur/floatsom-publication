"""
Factory for creating SOM topology instances
"""

from typing import Optional, Union, Dict, Any
import cupy as cp
import numpy as np
import logging

logger = logging.getLogger(__name__)

from floatsom.topology.som_topology import SOMTopology
from floatsom.topology.grid_topology import GridTopology
from floatsom.topology.hexagonal_topology import HexagonalTopology
from floatsom.topology.mst_topology import MSTTopology
from floatsom.topology.rng_topology import RNGTopology


def create_topology(som_params, data: Optional[cp.ndarray] = None) -> SOMTopology:
    """
    Factory function to create the appropriate topology strategy.
    
    Args:
        som_params: SOM parameters containing topology configuration
        data: Optional input data for initialization (e.g., PCA)
        
    Returns:
        topology: SOM topology strategy instance
    """
    # Extract parameters from som_params object
    topology_type = som_params.topology_config.topology_type
    verbose = som_params.verbose
    
    if verbose and data is not None:
        logger.debug(f"Data shape for topology initialization: {data.shape}")
    
    resolved_input_dim = som_params.input_dim
    if resolved_input_dim is None:
        if data is None:
            raise ValueError(
                "input_dim must be provided when data is unavailable for topology initialization"
            )
        if getattr(data, "ndim", 0) < 2:
            raise ValueError(
                f"Cannot infer input_dim from data with ndim={getattr(data, 'ndim', None)}"
            )
        resolved_input_dim = int(data.shape[1])

    resolved_input_dim = int(resolved_input_dim)
    if resolved_input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {resolved_input_dim}")

    if topology_type == "mst":
        topology = MSTTopology(
            num_nodes=som_params.topology_config.num_nodes,
            input_dim=resolved_input_dim,
            mst_update_frequency=som_params.topology_config.mst_update_frequency,
            initialization_method=som_params.initialization_method,
            influence_function=som_params.topology_config.influence_function,
            seed=som_params.seed,
            verbose=verbose,
            mst_params=None,
            dynamic_mst_frequency=som_params.topology_config.dynamic_mst_frequency,
            mst_decay_function=som_params.topology_config.mst_decay_function,
            initial_mst_frequency=som_params.topology_config.initial_mst_frequency,
            final_mst_frequency=som_params.topology_config.final_mst_frequency,
            total_iterations=som_params.total_iterations
        )
    elif topology_type == "rng":
        topology = RNGTopology(
            num_nodes=som_params.topology_config.num_nodes,
            input_dim=resolved_input_dim,
            mst_update_frequency=som_params.topology_config.mst_update_frequency,
            initialization_method=som_params.initialization_method,
            influence_function=som_params.topology_config.influence_function,
            seed=som_params.seed,
            verbose=verbose,
            mst_params=None,
            dynamic_mst_frequency=som_params.topology_config.dynamic_mst_frequency,
            mst_decay_function=som_params.topology_config.mst_decay_function,
            initial_mst_frequency=som_params.topology_config.initial_mst_frequency,
            final_mst_frequency=som_params.topology_config.final_mst_frequency,
            total_iterations=som_params.total_iterations
        )
    elif topology_type in ["grid", "rectangular", "planar", "toroidal"]:
        # Determine topology type for grid - check both topology_type and topology_variant
        topology_config = som_params.topology_config
        topo_variant = getattr(topology_config, 'topology_variant', None)
        topo_type = "toroidal" if (topology_type == "toroidal" or topo_variant == "toroidal") else "planar"

        topology = GridTopology(
            grid_size=som_params.topology_config.grid_size,
            input_dim=resolved_input_dim,
            topology_type=topo_type,
            initialization_method=som_params.initialization_method,
            influence_function=som_params.topology_config.influence_function,
            seed=som_params.seed,
            verbose=verbose
        )
    elif topology_type == "hexagonal":
        # Get topology variant and grid size from config
        topology_config = som_params.topology_config
        topo_variant = topology_config.topology_variant
        grid_size = topology_config.grid_size
            
        topology = HexagonalTopology(
            grid_size=grid_size,
            input_dim=resolved_input_dim,
            topology_type=topo_variant,
            initialization_method=som_params.initialization_method,
            influence_function=som_params.topology_config.influence_function,
            seed=som_params.seed,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return topology


class TopologyFactory:
    """
    Factory for creating different SOM topology types
    """
    
    AVAILABLE_TOPOLOGIES = ["grid", "rectangular", "hexagonal", "mst", "rng", "planar", "toroidal"]
    
    @staticmethod
    def create_topology(topology_type: str, **kwargs) -> SOMTopology:
        """
        Create topology instance
        
        Args:
            topology_type: Type of topology ("grid", "hexagonal", "mst", "rng", "rectangular", "planar", "toroidal")
            **kwargs: Topology-specific parameters
            
        Returns:
            SOMTopology instance
        """
        if topology_type not in TopologyFactory.AVAILABLE_TOPOLOGIES:
            raise ValueError(f"Unknown topology type: {topology_type}. Available: {TopologyFactory.AVAILABLE_TOPOLOGIES}")
        
        if topology_type == "mst":
            return TopologyFactory.create_mst_topology(**kwargs)
        elif topology_type == "rng":
            return TopologyFactory.create_rng_topology(**kwargs)
        elif topology_type in ["grid", "rectangular", "planar", "toroidal"]:
            toroidal = (topology_type == "toroidal")
            return TopologyFactory.create_grid_topology(toroidal=toroidal, **kwargs)
        elif topology_type == "hexagonal":
            return TopologyFactory.create_hexagonal_topology(**kwargs)
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")
    
    @staticmethod
    def get_available_topologies():
        """
        Get list of available topology types
        
        Returns:
            List of topology type names
        """
        return TopologyFactory.AVAILABLE_TOPOLOGIES.copy()
    
    @staticmethod
    def create_grid_topology(toroidal: bool = False, **kwargs) -> GridTopology:
        """
        Create rectangular grid topology with specified parameters
        
        Args:
            toroidal: Whether to use toroidal (wrap-around) topology
            **kwargs: Grid topology parameters
            
        Returns:
            GridTopology instance
        """
        # Extract grid-specific parameters
        grid_size = kwargs.get('grid_size', 10)
        input_dim = kwargs.get('input_dim', 1)
        initialization_method = kwargs.get('initialization_method', 'random')
        seed = kwargs.get('seed', None)
        verbose = kwargs.get('verbose', False)
        
        topology_type = "toroidal" if toroidal else "planar"
        
        return GridTopology(
            grid_size=grid_size,
            input_dim=input_dim,
            topology_type=topology_type,
            initialization_method=initialization_method,
            seed=seed,
            verbose=verbose
        )
    
    @staticmethod
    def create_hexagonal_topology(toroidal: bool = False, **kwargs) -> HexagonalTopology:
        """
        Create hexagonal grid topology with specified parameters
        
        Args:
            toroidal: Whether to use toroidal (wrap-around) topology
            **kwargs: Hexagonal topology parameters
            
        Returns:
            HexagonalTopology instance
        """
        # Extract hexagonal-specific parameters
        grid_size = kwargs.get('grid_size', 10)
        input_dim = kwargs.get('input_dim', 1)
        initialization_method = kwargs.get('initialization_method', 'random')
        seed = kwargs.get('seed', None)
        verbose = kwargs.get('verbose', False)
        
        topology_type = "toroidal" if toroidal else "planar"
        
        return HexagonalTopology(
            grid_size=grid_size,
            input_dim=input_dim,
            topology_type=topology_type,
            initialization_method=initialization_method,
            seed=seed,
            verbose=verbose
        )
    
    @staticmethod
    def create_mst_topology(mst_params: Optional[Dict[str, Any]] = None, **kwargs) -> MSTTopology:
        """
        Create MST topology with specified parameters
        
        Args:
            mst_params: MST-specific parameters dictionary
            **kwargs: MST topology parameters
            
        Returns:
            MSTTopology instance
        """
        # Extract MST-specific parameters
        num_nodes = kwargs.get('num_nodes', 100)
        input_dim = kwargs.get('input_dim', 1)
        mst_update_frequency = kwargs.get('mst_update_frequency', 50)
        initialization_method = kwargs.get('initialization_method', 'random')
        seed = kwargs.get('seed', None)
        verbose = kwargs.get('verbose', False)
        dynamic_mst_frequency = kwargs.get('dynamic_mst_frequency', True)
        mst_decay_function = kwargs.get('mst_decay_function', 'exponential')
        initial_mst_frequency = kwargs.get('initial_mst_frequency', 1)
        final_mst_frequency = kwargs.get('final_mst_frequency', 10)
        total_iterations = kwargs.get('total_iterations', None)
        influence_function = kwargs.get('influence_function', 'gaussian')
        
        return MSTTopology(
            num_nodes=num_nodes,
            input_dim=input_dim,
            mst_update_frequency=mst_update_frequency,
            initialization_method=initialization_method,
            influence_function=influence_function,
            seed=seed,
            verbose=verbose,
            mst_params=mst_params,
            dynamic_mst_frequency=dynamic_mst_frequency,
            mst_decay_function=mst_decay_function,
            initial_mst_frequency=initial_mst_frequency,
            final_mst_frequency=final_mst_frequency,
            total_iterations=total_iterations
        )

    @staticmethod
    def create_rng_topology(mst_params: Optional[Dict[str, Any]] = None, **kwargs) -> RNGTopology:
        """
        Create RNG topology with specified parameters.

        Uses the same update-frequency controls as MST for dynamic graph updates.
        """
        num_nodes = kwargs.get('num_nodes', 100)
        input_dim = kwargs.get('input_dim', 1)
        mst_update_frequency = kwargs.get('mst_update_frequency', 50)
        initialization_method = kwargs.get('initialization_method', 'random')
        seed = kwargs.get('seed', None)
        verbose = kwargs.get('verbose', False)
        dynamic_mst_frequency = kwargs.get('dynamic_mst_frequency', True)
        mst_decay_function = kwargs.get('mst_decay_function', 'exponential')
        initial_mst_frequency = kwargs.get('initial_mst_frequency', 1)
        final_mst_frequency = kwargs.get('final_mst_frequency', 10)
        total_iterations = kwargs.get('total_iterations', None)
        influence_function = kwargs.get('influence_function', 'gaussian')

        return RNGTopology(
            num_nodes=num_nodes,
            input_dim=input_dim,
            mst_update_frequency=mst_update_frequency,
            initialization_method=initialization_method,
            influence_function=influence_function,
            seed=seed,
            verbose=verbose,
            mst_params=mst_params,
            dynamic_mst_frequency=dynamic_mst_frequency,
            mst_decay_function=mst_decay_function,
            initial_mst_frequency=initial_mst_frequency,
            final_mst_frequency=final_mst_frequency,
            total_iterations=total_iterations
        )
    
    @staticmethod
    def validate_topology_params(topology_type: str, params) -> bool:
        """
        Validate parameters for topology type
        
        Args:
            topology_type: Type of topology to validate
            params: FloatSOMParams object to validate
            
        Returns:
            is_valid: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if topology_type not in TopologyFactory.AVAILABLE_TOPOLOGIES:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        # Common validation
        if hasattr(params, 'input_dim') and params.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        
        # Topology-specific validation
        if topology_type in ["grid", "rectangular", "planar", "toroidal"]:
            if hasattr(params, 'topology_config') and hasattr(params.topology_config, 'grid_size') and params.topology_config.grid_size <= 0:
                raise ValueError("grid_size must be positive")
        
        elif topology_type in {"mst", "rng"}:
            if hasattr(params, 'topology_config') and hasattr(params.topology_config, 'num_nodes') and params.topology_config.num_nodes <= 0:
                raise ValueError("num_nodes must be positive")
            if (
                hasattr(params, 'topology_config')
                and hasattr(params.topology_config, 'mst_update_frequency')
                and params.topology_config.mst_update_frequency is not None
                and params.topology_config.mst_update_frequency <= 0
            ):
                raise ValueError("mst_update_frequency must be positive")
        
        return True
