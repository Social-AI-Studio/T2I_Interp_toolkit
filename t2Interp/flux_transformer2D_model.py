import torch as th
from t2Interp.accessors import ModuleAccessor, AttentionAccessor, IOType
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List 
from t2Interp.blocks import TransformerBlock
from t2Interp.flux_single_stream_block import SingleStreamBlock
from t2Interp.flux_dual_stream_block import DualStreamBlock

class FluxTransformer:
    """
    Container of TransformerBlock objects with convenient getters/setters and
    the ability to add/remove layers. By default, initializes `num_layers`
    empty blocks indexed [0..num_layers-1].

    Key methods:
      - summary()
    """
    def __init__(self, transformer: th.nn.Module):
        
        if not hasattr(transformer, 'dual_stream_blocks') or not hasattr(transformer, 'single_stream_blocks'):
            raise ValueError("Blocks missing in module.")
        
        single_stream_blocks = transformer.single_stream_blocks
        dual_stream_blocks = transformer.transformer_blocks
        
        self.dual_stream_blocks: Dict[int, DualStreamBlock] = {
            i: DualStreamBlock(
                layer=i, in_=ModuleAccessor(dual_stream_blocks[i],"clip_encoder_input",IOType.INPUT),
                out_=ModuleAccessor(dual_stream_blocks[i],"clip_encoder_output",IOType.OUTPUT),
                attn_in=ModuleAccessor(dual_stream_blocks[i].self_attn,"clip_encoder_attn",IOType.INPUT),
                attn_out=ModuleAccessor(dual_stream_blocks[i].self_attn,"clip_encoder_attn",IOType.OUTPUT),
                mlp_in=ModuleAccessor(dual_stream_blocks[i].mlp,"clip_encoder_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(dual_stream_blocks[i].mlp,"clip_encoder_mlp_out",IOType.OUTPUT),
                ) for i in range(int(len(dual_stream_blocks)))
        }
        self.single_stream_blocks: Dict[int, SingleStreamBlock] = {
            i: SingleStreamBlock(
                layer=i, in_=ModuleAccessor(single_stream_blocks[i],"clip_encoder_input",IOType.INPUT),
                out_=ModuleAccessor(single_stream_blocks[i],"clip_encoder_output",IOType.OUTPUT),
                attn_in=ModuleAccessor(single_stream_blocks[i].self_attn,"clip_encoder_attn",IOType.INPUT),
                attn_out=ModuleAccessor(single_stream_blocks[i].self_attn,"clip_encoder_attn",IOType.OUTPUT),
                mlp_in=ModuleAccessor(single_stream_blocks[i].mlp,"clip_encoder_mlp_in",IOType.INPUT),
                mlp_out=ModuleAccessor(single_stream_blocks[i].mlp,"clip_encoder_mlp_out",IOType.OUTPUT),
                ) for i in range(int(len(single_stream_blocks)))
        }

    def summary(self) -> str:
        lines = [blk.summary() for blk in self.dual_Stream_blocks.values()]
        lines += [blk.summary() for blk in self.single_Stream_blocks.values()]
        return "\n".join(lines)
        
        