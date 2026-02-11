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
    "run_intervention",
]

# =============================================================================
# Legacy envoy extensions (commented out - may not be needed)
# =============================================================================
# import nnsight.intervention.envoy as envoy_mod
# from weakref import WeakKeyDictionary
# from typing import Union

# Envoy = envoy_mod.Envoy

# _KIND = WeakKeyDictionary()   # env -> "input"/"output"
# _RET  = WeakKeyDictionary()   # env -> bool

# # ---- io_type property (accepts Enum with .value or a plain str)
# def _get_io(self):
#     return _KIND.get(self, "input")
# def _set_io(self, v):
#     _KIND[self] = getattr(v, "value", v)
# def _del_io(self):
#     _KIND.pop(self, None)

# # ---- returns_tuple property
# def _get_rt(self):
#     return _RET.get(self, False)
# def _set_rt(self, v: bool):
#     _RET[self] = bool(v)
# def _del_rt(self):
#     _RET.pop(self, None)

# # ---- value property (uses self.io_type / self.returns_tuple)
# def _get_value(self):
#     kind = _get_io(self)
#     tgt = self.input if kind == "input" else self.output  # NOTE: output requires interleaving or fake output
#     return tgt[0] if _get_rt(self) and isinstance(tgt, tuple) else tgt

# def _set_value(self, new):
#     kind = _get_io(self)
#     if kind == "input":
#         old = self.input
#         if _get_rt(self):
#             rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
#             self.input = (new, *rest) if rest else (new,)
#         else:
#             self.input = new
#     else:
#         old = self.output
#         if _get_rt(self):
#             rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
#             self.output = (new, *rest) if rest else (new,)
#         else:
#             self.output = new

# # ---- install (don’t clobber if upstream ever adds these)
# if not hasattr(Envoy, "io_type"):
#     Envoy.io_type = property(_get_io, _set_io, _del_io)
# if not hasattr(Envoy, "returns_tuple"):
#     Envoy.returns_tuple = property(_get_rt, _set_rt, _del_rt)
# if not hasattr(Envoy, "value"):
#     Envoy.value = property(_get_value, _set_value)
