import torch as th
from t2Interp.accessors import ModuleAccessor, AttentionAccessor
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List
from t2Interp.accessors import IOType
from t2Interp.blocks import TransformerBlock

class ClipEncoder:
    """
    Container of TransformerBlock objects with convenient getters/setters and
    the ability to add/remove layers. By default, initializes `num_layers`
    empty blocks indexed [0..num_layers-1].

    Key methods:
      - get_block(i) / set_block(i, block)
      - summary()
    """
    def __init__(self, clip_text_model: th.nn.Module):
        encoder=clip_text_model.text_model.encoder
        self.blocks: Dict[int, TransformerBlock] = {
            i: TransformerBlock(
                in_=ModuleAccessor(encoder.layers[i],"clip_encoder_input",IOType.INPUT),
                out_=ModuleAccessor(encoder.layers[i],"clip_encoder_output",IOType.OUTPUT),
                attn_in=ModuleAccessor(encoder.layers[i].self_attn,"clip_encoder_attn",IOType.INPUT),
                attn_out=ModuleAccessor(encoder.layers[i].self_attn,"clip_encoder_attn",IOType.OUTPUT),
                mlp_in=ModuleAccessor(encoder.layers[i].mlp,"clip_encoder_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(encoder.layers[i].mlp,"clip_encoder_mlp_out",IOType.OUTPUT),
                ) for i in range(int(len(encoder.layers)))
        }

    def summary(self) -> str:
        lines = [blk.summary() for blk in self]
        return "\n".join(lines)
        
        