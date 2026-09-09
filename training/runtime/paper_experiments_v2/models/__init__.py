"""Model components required by the released PPD-EHR workflow."""

from .enhanced_cdt import (
    EnhancedCausalDigitalTwin,
    EnhancedCausalDigitalTwinNoDiff,
    EnhancedCausalDigitalTwinNoGNN,
    SparseFeatureProcessor,
    FeatureGNN,
    DiffusionAugmentation
)
from .data_utils import (
    load_event_data,
    collate_events,
    extract_flat_features
)
from .enhanced_cdt_v5 import EnhancedCausalDigitalTwinV5

__all__ = [
    # V3 模型
    'EnhancedCausalDigitalTwin',
    'EnhancedCausalDigitalTwinNoDiff',
    'EnhancedCausalDigitalTwinNoGNN',
    'SparseFeatureProcessor',
    'FeatureGNN',
    'DiffusionAugmentation',
    # 数据工具
    'load_event_data',
    'collate_events',
    'extract_flat_features',
    # Current external-validation model
    'EnhancedCausalDigitalTwinV5',
]
