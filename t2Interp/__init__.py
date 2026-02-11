# =============================================================================
# T2I Interp Toolkit
# =============================================================================

# Core classes
from t2Interp.accessors import IOType, ModuleAccessor
from t2Interp.blocks import SAEBlock, TransformerBlock, UnetTransformerBlock

# Interventions
from t2Interp.intervention import (
    AddVectorIntervention,
    DiffusionIntervention,
    FeatureIntervention,
    ReplaceIntervention,
    SAEFeatureScalingIntervention,
    ScalingAttentionIntervention,
    run_intervention,
)

# SAE Management - OLD: hook-based approach (deprecated)
from t2Interp.sae import SAEManager

# SAE Management - NEW: .edit() based approach (recommended)
from t2Interp.sae_edit import SAEManagerEdit, SAEModule, add_sae_to_model

# Model wrapper
from t2Interp.T2I import T2IModel

__all__ = [
    # Core
    "IOType",
    "ModuleAccessor",
    "SAEBlock",
    "TransformerBlock",
    "UnetTransformerBlock",
    # SAE (new)
    "SAEManagerEdit",
    "SAEModule",
    "add_sae_to_model",
    # SAE (old)
    "SAEManager",
    # Model
    "T2IModel",
    # Interventions
    "DiffusionIntervention",
    "AddVectorIntervention",
    "ReplaceIntervention",
    "ScalingAttentionIntervention",
    "FeatureIntervention",
    "SAEFeatureScalingIntervention",
    "run_intervention",
]
