import torch as th
from t2Interp.accessors import ModuleAccessor, ModuleAccessor, IOType
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List
from t2Interp.blocks import TransformerBlock, UnetTransformerBlock
# from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D 

class UnetDownBlock:
    down_blocks: List[UnetTransformerBlock]
    
    def __init__(self, down_blocks: List[UnetTransformerBlock]):
        self.down_blocks = down_blocks
    
class UnetMidBlock:
    mid_blocks: List[UnetTransformerBlock]
    
    def __init__(self, mid_blocks: List[UnetTransformerBlock]):
        self.mid_blocks = mid_blocks
    
class UnetUpBlock:
    up_blocks: List[UnetTransformerBlock]    
    
    def __init__(self, up_blocks: List[UnetTransformerBlock]):
        self.up_blocks = up_blocks
    
class Unet:
    """
    Container of TransformerBlock objects with convenient getters/setters and
    the ability to add/remove layers. By default, initializes `num_layers`
    empty blocks indexed [0..num_layers-1].

    Key methods:
      - summary()
    """
    def __init__(self, unet: th.nn.Module):
        setattr(self, "down_blocks", [])
        setattr(self, "mid_block", None)
        setattr(self, "up_blocks", [])
        
        for down_block in unet.down_blocks:
            if down_block._get_name()=='CrossAttnDownBlock2D':
                for attention in down_block.attentions:
                    for block in attention.transformer_blocks:
                        self.down_blocks.append(UnetTransformerBlock(
                                in_=ModuleAccessor(block,"unet_down_block_in",IOType.INPUT),
                                out_=ModuleAccessor(block,"unet_down_block_out",IOType.OUTPUT),
                                self_attn_in=ModuleAccessor(block.attn1,"unet_down_block_self_attn_in",IOType.INPUT),
                                self_attn_out=ModuleAccessor(block.attn1,"unet_down_block_self_attn_out",IOType.OUTPUT),
                                cross_attn_in=ModuleAccessor(block.attn2,"unet_down_block_cross_attn_in",IOType.INPUT),
                                cross_attn_out=ModuleAccessor(block.attn2,"unet_down_block_cross_attn_out",IOType.OUTPUT),
                                mlp_in=ModuleAccessor(block.ff,"unet_down_block_mlp_in",IOType.INPUT),
                                mlp_out=ModuleAccessor(block.ff,"unet_down_block_mlp_out",IOType.OUTPUT),
                                ))
        for attention in unet.mid_block.attentions:
            for block in attention.transformer_blocks:
                self.mid_block = UnetTransformerBlock(
                        in_=ModuleAccessor(block,"unet_mid_block_in",IOType.INPUT),
                        out_=ModuleAccessor(block,"unet_mid_block_out",IOType.OUTPUT),
                        self_attn_in=ModuleAccessor(block.attn1,"unet_mid_block_self_attn_in",IOType.INPUT),
                        self_attn_out=ModuleAccessor(block.attn1,"unet_mid_block_self_attn_out",IOType.OUTPUT),
                        cross_attn_in=ModuleAccessor(block.attn2,"unet_mid_block_cross_attn_in",IOType.INPUT),
                        cross_attn_out=ModuleAccessor(block.attn2,"unet_mid_block_cross_attn_out",IOType.OUTPUT),
                        mlp_in=ModuleAccessor(block.ff,"unet_mid_block_mlp_in",IOType.INPUT),
                        mlp_out=ModuleAccessor(block.ff,"unet_mid_block_mlp_out",IOType.OUTPUT),
                        ) 
        for up_block in unet.up_blocks:
            if up_block._get_name()=='CrossAttnUpBlock2D':
                for attention in up_block.attentions:
                    for block in attention.transformer_blocks:
                        self.up_blocks.append(UnetTransformerBlock(
                                in_=ModuleAccessor(block,"unet_up_block_in",IOType.INPUT),
                                out_=ModuleAccessor(block,"unet_up_block_out",IOType.OUTPUT),
                                self_attn_in=ModuleAccessor(block.attn1,"unet_up_block_self_attn_in",IOType.INPUT),
                                self_attn_out=ModuleAccessor(block.attn1,"unet_up_block_self_attn_out",IOType.OUTPUT),
                                cross_attn_in=ModuleAccessor(block.attn2,"unet_up_block_cross_attn_in",IOType.INPUT),
                                cross_attn_out=ModuleAccessor(block.attn2,"unet_up_block_cross_attn_out",IOType.OUTPUT),
                                mlp_in=ModuleAccessor(block.ff,"unet_up_block_mlp_in",IOType.INPUT),
                                mlp_out=ModuleAccessor(block.ff,"unet_up_block_mlp_out",IOType.OUTPUT),
                                ))

    def summary(self) -> str:
        s = "UNet:\n"
        s += f"  Down Blocks: {len(self.down_blocks)}\n"
        for i, block in enumerate(self.down_blocks):
            s += f"    [{i:02d}] {block.summary()}\n"
        s += f"  Mid Block:\n"
        if self.mid_block:
            s += f"    {self.mid_block.summary()}\n"
        else:
            s += "    None\n"
        s += f"  Up Blocks: {len(self.up_blocks)}\n"
        for i, block in enumerate(self.up_blocks):
            s += f"    [{i:02d}] {block.summary()}\n"
        return s
    
    def __repr__(self):
        return self.summary()
        
        