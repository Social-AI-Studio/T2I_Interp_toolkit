import torch as th
from accessors import ModuleAccessor, AttentionAccessor
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List
from accessors import IOType
from blocks import TransformerBlock
  
class Unet:
    """
    Container of TransformerBlock objects with convenient getters/setters and
    the ability to add/remove layers. By default, initializes `num_layers`
    empty blocks indexed [0..num_layers-1].

    Key methods:
      - get_block(i) / set_block(i, block)
      - summary()
    """
    def __init__(self, encoder: th.nn.Module):
        self.blocks: Dict[int, TransformerBlock] = {
            i: {"self_attn": TransformerBlock(
                layer=i, in_=ModuleAccessor(encoder.layers[i],"t5_encoder_input",IOType.INPUT),
                out_=ModuleAccessor(encoder.layers[i],"t5_encoder_output",IOType.OUTPUT),
                attn_in=AttentionAccessor(encoder.layers[i].self_attn,"t5_encoder_attn",IOType.INPUT),
                attn_out=AttentionAccessor(encoder.layers[i].self_attn,"t5_encoder_attn",IOType.OUTPUT),
                mlp_in=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_out",IOType.OUTPUT),
                ),
                "cross_attn": TransformerBlock(
                    layer=i, in_=ModuleAccessor(encoder.layers[i],"t5_encoder_input",IOType.INPUT),
                    out_=ModuleAccessor(encoder.layers[i],"t5_encoder_output",IOType.OUTPUT),
                    attn_in=AttentionAccessor(encoder.layers[i].cross_attn,"t5_encoder_attn",IOType.INPUT),
                    attn_out=AttentionAccessor(encoder.layers[i].cross_attn,"t5_encoder_attn",IOType.OUTPUT),
                    mlp_in=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_in",IOType.INPUT),
                    mlp_out=ModuleAccessor(encoder.layers[i].mlp,"t5_encoder_mlp_out",IOType.OUTPUT),
                )} for i in range(int(len(encoder.layers)))
        }

    def summary(self) -> str:
        lines = [blk.summary() for blk in self]
        return "\n".join(lines)
        
        