from typing import Dict, Iterable, Optional, List
from accessors import IOType,ModuleAccessor, AttentionAccessor
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
    layer: int
    in_: Optional[ModuleAccessor] = None
    mlp_in: Optional[ModuleAccessor] = None
    mlp_out: Optional[ModuleAccessor] = None
    out_: Optional[ModuleAccessor] = None
    attention: Optional[AttentionAccessor] = None  # single shared accessor

    @property
    def attn_in(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_in.setter
    def attn_in(self, v: Optional[AttentionAccessor]): self.attention = v

    @property
    def attn_q_in(self) -> Optional[ModuleAccessor]:
        return self.attn_q_in
    @attn_q_in.setter
    def attn_q_in(self, inp: Optional[ModuleAccessor]): self.attn_q_in = inp

    @property
    def attn_q_out(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_q_out.setter
    def attn_q_out(self, v: Optional[ModuleAccessor]): self.attention = v

    @property
    def attn_k_in(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_k_in.setter
    def attn_k_in(self, v: Optional[ModuleAccessor]): self.attention = v

    @property
    def attn_k_out(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_k_out.setter
    def attn_k_out(self, v: Optional[ModuleAccessor]): self.attention = v

    @property
    def attn_v_in(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_v_in.setter
    def attn_v_in(self, v: Optional[ModuleAccessor]): self.attention = v

    @property
    def attn_v_out(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_v_out.setter
    def attn_v_out(self, v: Optional[ModuleAccessor]): self.attention = v
    
    @property
    def attn_out(self) -> Optional[ModuleAccessor]:
        return self.attention
    @attn_out.setter
    def attn_out(self, v: Optional[ModuleAccessor]): self.attention = v

    # ---- helpers ----
    def summary(self) -> str:
        return (
            f"[L{self.layer:02d}] "
            f"in={bool(self.in_)} | attn={bool(self.attention)} | "
            f"mlp_in={bool(self.mlp_in)} | mlp_out={bool(self.mlp_out)} | out={bool(self.out_)}"
        )
