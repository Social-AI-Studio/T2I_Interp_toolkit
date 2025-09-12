import torch as th
from t2Interp.accessors import ModuleAccessor, AttentionAccessor, IOType
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List
from t2Interp.blocks import TransformerBlock

@dataclass
class T5Block(TransformerBlock):
    """
    Holds accessors for one transformer layer.

    - For non-attention parts, store individual ModuleAccessor objects:
        in_, mlp_in, mlp_out, out_
    - For attention, use a *single* AttentionAccessor shared by
      attn_in / attn_q / attn_k / attn_v / attn_out.
    """
    up_a_in: Optional[ModuleAccessor] = None
    up_a_out: Optional[ModuleAccessor] = None
    up_b_in: Optional[ModuleAccessor] = None
    up_b_out: Optional[ModuleAccessor] = None
    down_in: Optional[ModuleAccessor] = None
    down_out: Optional[ModuleAccessor] = None

    @property
    def up_a_in(self) -> Optional[ModuleAccessor]:
        return self.up_a_in
    @up_a_in.setter
    def up_a_in(self, v: Optional[ModuleAccessor]): self.up_a_in = v
    @property
    def up_a_out(self) -> Optional[ModuleAccessor]:
        return self.up_a_out
    @up_a_out.setter
    def up_a_out(self, v: Optional[ModuleAccessor]): self.up_a_out = v
    @property
    def up_b_in(self) -> Optional[ModuleAccessor]:
        return self.up_b_in
    @up_b_in.setter
    def up_b_in(self, v: Optional[ModuleAccessor]): self.up_b_in = v
    @property
    def up_b_out(self) -> Optional[ModuleAccessor]:
        return self.up_b_out
    @up_b_out.setter
    def up_b_out(self, v: Optional[ModuleAccessor]): self.up_b_out = v
    @property
    def down_in(self) -> Optional[ModuleAccessor]:
        return self.down_in
    @down_in.setter
    def down_in(self, v: Optional[ModuleAccessor]): self.down_in = v
    @property
    def down_out(self) -> Optional[ModuleAccessor]:
        return self.down_out
    @down_out.setter
    def down_out(self, v: Optional[ModuleAccessor]): self.down_out = v

    # ---- helpers ----
    def summary(self) -> str:
        return (
            f"[L{self.layer:02d}] "
            f"in={bool(self.in_)} | attn={bool(self.attention)} | "
            f"mlp_in={bool(self.mlp_in)} | mlp_out={bool(self.mlp_out)} | out={bool(self.out_)}"
        )
  
class T5Encoder:
    """
    Container of TransformerBlock objects with convenient getters/setters and
    the ability to add/remove layers. By default, initializes `num_layers`
    empty blocks indexed [0..num_layers-1].

    Key methods:
      - get_block(i) / set_block(i, block)
      - summary()
    """
    def __init__(self, encoder: th.nn.Module):
        self._blocks: Dict[int, TransformerBlock] = {
            i: TransformerBlock(
                layer=i, in_=ModuleAccessor(encoder.layers[i],"t5_encoder_input",IOType.INPUT),
                out_=ModuleAccessor(encoder.layers[i],"t5_encoder_output",IOType.OUTPUT),
                attn_in=AttentionAccessor(encoder.layers[i].self_attn,"t5_encoder_attn",IOType.INPUT),
                attn_out=AttentionAccessor(encoder.layers[i].self_attn,"t5_encoder_attn",IOType.OUTPUT),
                mlp_in=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_out",IOType.OUTPUT),
                ) for i in range(int(len(encoder.layers)))
        }

    def summary(self) -> str:
        lines = [blk.summary() for blk in self]
        return "\n".join(lines)
        
        