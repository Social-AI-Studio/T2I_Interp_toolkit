import torch as th
from accessors import ModuleAccessor, AttentionAccessor
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple
from accessors import IOType
from blocks import TransformerBlock


@dataclass
class SingleStreamBlock(TransformerBlock):
    """
    Flux single-stream transformer block.

    Matches modules like:
        FluxSingleTransformerBlock(
            norm, proj_mlp, act_mlp, proj_out, attn
        )

    YAML mapping you shared:
        single_stream_blocks:
          discover: "**.single_transformer_blocks.*"
          slots:
            mlp:
              up:        "proj_mlp"   # 3072 -> 12288
              act:       "act_mlp"    # GELU
              down:      "proj_out"   # (attn ⊕ mlp_act) -> 3072

    Fields:
      - mlp_up         ← maps to 'proj_mlp'
      - mlp_act        ← maps to 'act_mlp'
      - mlp_down       ← maps to 'proj_out'
      - attn.q         ← maps to 'attn.to_q'
      - attn.k         ← maps to 'attn.to_k'
      - attn.v         ← maps to 'attn.to_v'
    """
    in_: Optional[ModuleAccessor] = None
    out_: Optional[ModuleAccessor] = None
    
    # MLP slots
    mlp_up_in: Optional[ModuleAccessor] = None 
    mlp_up_out: Optional[ModuleAccessor] = None    
    mlp_act_in: Optional[ModuleAccessor] = None   
    mlp_act_out: Optional[ModuleAccessor] = None       
    mlp_down_in: Optional[ModuleAccessor] = None 
    mlp_down_out: Optional[ModuleAccessor] = None
    out_: Optional[ModuleAccessor] = None
    in_: Optional[ModuleAccessor] = None

    # Attention slots
    attn_in: Optional[ModuleAccessor] = None
    attn_out: Optional[ModuleAccessor] = None
    q_in: Optional[ModuleAccessor] = None
    q_out: Optional[ModuleAccessor] = None
    k_in: Optional[ModuleAccessor] = None
    k_out: Optional[ModuleAccessor] = None
    v_in: Optional[ModuleAccessor] = None
    v_out: Optional[ModuleAccessor] = None
    
    def __init__(self, module: th.nn.Module):
        self.in_ = ModuleAccessor(module,"single_stream_input",IOType.INPUT)
        self.out_ = ModuleAccessor(module,"single_stream_output",IOType.OUTPUT)
        self.mlp_up_in = ModuleAccessor(module.mlp_up,"mlp_up_input",IOType.INPUT)
        self.mlp_up_out = ModuleAccessor(module.mlp_up,"mlp_up_output",IOType.OUTPUT)
        self.mlp_act_in = ModuleAccessor(module.mlp_act,"mlp_act_input",IOType.INPUT)
        self.mlp_act_out = ModuleAccessor(module.mlp_act,"mlp_act_output",IOType.OUTPUT)
        self.mlp_down_in = ModuleAccessor(module.mlp_down,"mlp_down_input",IOType.INPUT)
        self.mlp_down_out = ModuleAccessor(module.mlp_down,"mlp_down_output",IOType.OUTPUT)
        self.attn_in = ModuleAccessor(module.attn,"attn_input",IOType.INPUT)
        self.attn_out = ModuleAccessor(module.attn,"attn_output",IOType.OUTPUT)
        self.q_in = ModuleAccessor(module.attn.q,"attn_q_input",IOType.INPUT)
        self.q_out = ModuleAccessor(module.attn.q,"attn_q_output",IOType.OUTPUT)
        self.k_in = ModuleAccessor(module.attn.k,"attn_k_input",IOType.INPUT)
        self.k_out = ModuleAccessor(module.attn.k,"attn_k_output",IOType.OUTPUT)
        self.v_in = ModuleAccessor(module.attn.v,"attn_v_input",IOType.INPUT)
        self.v_out = ModuleAccessor(module.attn.v,"attn_v_output",IOType.OUTPUT)
        
    def summary(self) -> str:
        return (
            f"SingleStreamBlock(layer={self.layer}, "
            f"in_={self.in_}, out_={self.out_}, "
            f"mlp_up_in={self.mlp_up_in}, mlp_up_out={self.mlp_up_out}, "
            f"mlp_act_in={self.mlp_act_in}, mlp_act_out={self.mlp_act_out}, "
            f"mlp_down_in={self.mlp_down_in}, mlp_down_out={self.mlp_down_out}, "
            f"attn_in={self.attn_in}, attn_out={self.attn_out}, "
            f"q_in={self.q_in}, q_out={self.q_out}, "
            f"k_in={self.k_in}, k_out={self.k_out}, "
            f"v_in={self.v_in}, v_out={self.v_out})"
        )
        

