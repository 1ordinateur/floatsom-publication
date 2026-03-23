"""
Factory functions for creating FloatSOM with different sampling and processing combinations
"""

from floatsom.base.floatsom import FloatSOM
from floatsom.topology.topology_factory import create_topology
from floatsom.sampling.full_selector import FullSelector
from floatsom.sampling.random_selector import RandomSelector
from floatsom.sampling.hdsssom_selector import HDSSSOMSelector
from floatsom.processing.processor_factory import ProcessorFactory

# Import MiniSOM adapter
try:
    from ..adapters.minisom_adapter import MiniSOMAdapter, MINISOM_AVAILABLE
except ImportError:
    MINISOM_AVAILABLE = False


def create_floatsom(params, data=None):
    """
    Create FloatSOM with components based on params configuration
    
    Args:
        params: FloatSOMParams object containing all configuration
        data_shape: Tuple of (n_samples, n_features) for chunk size calculation (required)
        data: Optional data for topology initialization
        
    Returns:
        FloatSOM instance or MiniSOMAdapter configured according to params
    """
    # Extract configurations
    sampling_method = params.sampling_config.method
    
    # Handle MiniSOM processing specially - it's a complete SOM implementation
    if params.processing_config.method == 'minisom':
        if not MINISOM_AVAILABLE:
            raise ImportError("MiniSOM package not available. Please install it with 'pip install minisom'")

        # Validate topology compatibility
        if params.topology_config.topology_type not in ['grid', 'hexagonal']:
            raise ValueError(f"MiniSOM only supports 'grid' and 'hexagonal' topologies, got: {params.topology_config.topology_type}")
        
        # Get minisom_defaults flag from params if available
        use_minisom_defaults = getattr(params, 'minisom_defaults', False)
        
        # Return MiniSOM adapter directly
        return MiniSOMAdapter(params, use_default_hyperparams=use_minisom_defaults)
    
    # Create topology based on params
    topology = create_topology(params, data)
    
    # Create sample selector
    if sampling_method == 'full':
        selector = FullSelector()
    elif sampling_method == 'random':
        selector = RandomSelector(params.sampling_config)
    elif sampling_method == 'hdsssom':
        selector = HDSSSOMSelector(params.sampling_config)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Create processor using the factory, passing dimensions for chunk size calculation
    processor = ProcessorFactory.create(
        params.processing_config, 
        params.input_dim, 
        params.total_nodes)
    
    # Create and return FloatSOM instance
    return FloatSOM(selector, processor, topology, params)


def get_available_sampling_methods():
    """
    Return list of available sampling methods
    """
    return ['full', 'random', 'hdsssom']


def get_available_processing_methods():
    """
    Return list of available processing methods
    """
    return ['colors', 'batch', 'minisom']


def get_available_topology_types():
    """
    Return list of available topology types
    """
    return ['grid', 'hexagonal', 'mst', 'rng']


def create_floatsom_matrix(base_params, data=None):
    """
    Create all 9 combinations of sampling × processing methods for testing
    
    Args:
        base_params: Base FloatSOMParams to use as template
        data: Optional data for topology initialization
        
    Returns:
        Dictionary mapping config names to FloatSOM instances
    """
    som_matrix = {}
    
    for sampling in get_available_sampling_methods():
        for processing in get_available_processing_methods():
            # Clone params and update methods
            params = type(base_params)(**vars(base_params))
            params.sampling_config.method = sampling
            params.processing_config.method = processing
            
            # Create FloatSOM instance
            config_name = f"{sampling}_{processing}"
            som_matrix[config_name] = create_floatsom(params, data)
    
    return som_matrix
