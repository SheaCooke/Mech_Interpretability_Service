"""
model_processor package

Public interface — import from here, not from internal modules.
Internal module structure can change without breaking callers.
"""

from .model_processor import Model_Processor
from .factory         import ModelStrategyFactory
from .types           import ModelMetadata, DataRecord, InferenceRecord, LayerInfo
from .dataset_loader  import load_dataset
from .summariser      import summarise_results

__all__ = [
    'Model_Processor',
    'ModelStrategyFactory',
    'ModelMetadata',
    'DataRecord',
    'InferenceRecord',
    'LayerInfo',
    'load_dataset',
    'summarise_results',
]