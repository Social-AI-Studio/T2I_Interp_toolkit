from typing import Dict, Iterable, Optional, List
from t2Interp.accessors import IOType,ModuleAccessor, AttentionAccessor
from dataclasses import dataclass

@dataclass
class TransformerBlock:
    """
    Holds accessors for one transformer layer.

    - For non-attention parts, store individual ModuleAccessor objects:
        in_, mlp_in, mlp_out, out_
    - For attention, use a *single* AttentionAccessor shared by
      attn_in / attn_q / attn_k / attn_v / attn_out.
    """
    # layer: int
    in_: Optional[ModuleAccessor] = None
    mlp_in: Optional[ModuleAccessor] = None
    mlp_out: Optional[ModuleAccessor] = None
    out_: Optional[ModuleAccessor] = None
    attn_in: Optional[ModuleAccessor] = None  
    attn_out: Optional[ModuleAccessor] = None  
    q_in: Optional[ModuleAccessor] = None  
    q_out: Optional[ModuleAccessor] = None
    k_in: Optional[ModuleAccessor] = None
    k_out: Optional[ModuleAccessor] = None
    v_in: Optional[ModuleAccessor] = None
    v_out: Optional[ModuleAccessor] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"TransformerBlock has no attribute '{k}'")
            setattr(self, k, v)
            
    # ---- helpers ----
    def summary(self) -> str:
        return (
            f"[L{self.layer:02d}] "
            f"in={bool(self.in_)} | attn={bool(self.attention)} | "
            f"mlp_in={bool(self.mlp_in)} | mlp_out={bool(self.mlp_out)} | out={bool(self.out_)}"
        )
        
@dataclass
class UnetTransformerBlock:
    """
    Holds accessors for one transformer layer.

    - For non-attention parts, store individual ModuleAccessor objects:
        in_, mlp_in, mlp_out, out_
    - For attention, use a *single* AttentionAccessor shared by
      attn_in / attn_q / attn_k / attn_v / attn_out.
    """
    in_: Optional[ModuleAccessor] = None
    mlp_in: Optional[ModuleAccessor] = None
    mlp_out: Optional[ModuleAccessor] = None
    out_: Optional[ModuleAccessor] = None
    cross_attn_in: Optional[ModuleAccessor] = None  
    cross_attn_out: Optional[ModuleAccessor] = None  
    cross_q_in: Optional[ModuleAccessor] = None  
    cross_q_out: Optional[ModuleAccessor] = None
    cross_k_in: Optional[ModuleAccessor] = None
    cross_k_out: Optional[ModuleAccessor] = None
    cross_v_in: Optional[ModuleAccessor] = None
    cross_v_out: Optional[ModuleAccessor] = None
    self_attn_in: Optional[ModuleAccessor] = None  
    self_attn_out: Optional[ModuleAccessor] = None  
    self_q_in: Optional[ModuleAccessor] = None  
    self_q_out: Optional[ModuleAccessor] = None
    self_k_in: Optional[ModuleAccessor] = None
    self_k_out: Optional[ModuleAccessor] = None
    self_v_in: Optional[ModuleAccessor] = None
    self_v_out: Optional[ModuleAccessor] = None
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"TransformerBlock has no attribute '{k}'")
            setattr(self, k, v)
            
    # ---- helpers ----
    def summary(self) -> str:
        return (
            f"in | self_attn_in | self_attn_out | "
            f"cross_attn_in | cross_attn_out | "
            f"mlp_in | mlp_out | out | "
            f"cross_q_in | cross_k_in | cross_v_in | "
            f"self_q_in | self_k_in | self_v_in |"
            f"cross_q_out | cross_k_out | cross_v_out | "
            f"self_q_out | self_k_out | self_v_out"
        )   
        
    def __repr__(self):
        return self.summary()      
