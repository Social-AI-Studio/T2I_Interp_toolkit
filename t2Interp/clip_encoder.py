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
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                in_=ModuleAccessor(encoder.layers[i],f"clip_encoder_block_{i}_input",IOType.INPUT),
                out_=ModuleAccessor(encoder.layers[i],f"clip_encoder_block_{i}_output",IOType.OUTPUT, returns_tuple=True),
                attn_in=ModuleAccessor(encoder.layers[i].self_attn,f"clip_encoder_block_{i}_self_attn_in",IOType.INPUT),
                attn_out=ModuleAccessor(encoder.layers[i].self_attn,f"clip_encoder_block_{i}_self_attn_out",IOType.OUTPUT),
                WO_in=ModuleAccessor(encoder.layers[i].self_attn.out_proj,f"clip_encoder_block_{i}_WO_in",IOType.INPUT),
                WO_out=ModuleAccessor(encoder.layers[i].self_attn.out_proj,f"clip_encoder_block_{i}_WO_out",IOType.OUTPUT, returns_tuple=True),
                mlp_in=ModuleAccessor(encoder.layers[i].mlp,f"clip_encoder_block_{i}_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(encoder.layers[i].mlp,f"clip_encoder_block_{i}_mlp_out",IOType.OUTPUT,returns_tuple=True),
                ) for i in range(int(len(encoder.layers)))
        ]
        self.final_layer_norm_in = ModuleAccessor(clip_text_model.text_model.final_layer_norm,"clip_encoder_final_layer_norm",IOType.INPUT)
        self.out_ = ModuleAccessor(clip_text_model.text_model.encoder.layers[-1],"clip_encoder_out",IOType.OUTPUT)

    def summary(self) -> str:
        return "blocks:\n" + "".join(
            f"{i}: {block}\n" for i, block in enumerate(self.blocks)
        )
    
    def __repr__(self):
        return self.summary()
        
        