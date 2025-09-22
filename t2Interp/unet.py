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
        setattr(self, "down_attn_blocks", [])
        setattr(self, "mid_attn_block", None)
        setattr(self, "up_attn_blocks", [])
        
        for i,down_block in enumerate(unet.down_blocks):
            if down_block._get_name()=='CrossAttnDownBlock2D':
                for j,attention in enumerate(down_block.attentions):
                    for k,block in enumerate(attention.transformer_blocks):
                        self.down_attn_blocks.append(UnetTransformerBlock(
                                in_=ModuleAccessor(block,f"unet_down_block_attn{i+j+k}_in",IOType.INPUT),
                                out_=ModuleAccessor(block,f"unet_down_block_attn{i+j+k}_out",IOType.OUTPUT),
                                self_attn_in=ModuleAccessor(block.attn1,f"unet_down_block_attn{i+j+k}_self_attn_in",IOType.INPUT),
                                self_attn_out=ModuleAccessor(block.attn1,f"unet_down_block_attn{i+j+k}_self_attn_out",IOType.OUTPUT),
                                cross_attn_in=ModuleAccessor(block.attn2,f"unet_down_block_attn{i+j+k}_cross_attn_in",IOType.INPUT),
                                cross_attn_out=ModuleAccessor(block.attn2,f"unet_down_block_attn{i+j+k}_cross_attn_out",IOType.OUTPUT),
                                mlp_in=ModuleAccessor(block.ff,f"unet_down_block_attn{i+j+k}_mlp_in",IOType.INPUT),
                                mlp_out=ModuleAccessor(block.ff,f"unet_down_block_attn{i+j+k}_mlp_out",IOType.OUTPUT),
                                self_attn_WO_in=ModuleAccessor(block.attn1.to_out[0],f"unet_down_block_attn{i+j+k}_self_attn_WO_in",IOType.INPUT),
                                self_attn_WO_out=ModuleAccessor(block.attn1.to_out[0],f"unet_down_block_attn{i+j+k}_self_attn_WO_out",IOType.OUTPUT, returns_tuple=True),
                                cross_attn_WO_in=ModuleAccessor(block.attn2.to_out[0],f"unet_down_block_attn{i+j+k}_cross_attn_WO_in",IOType.INPUT),
                                cross_attn_WO_out=ModuleAccessor(block.attn2.to_out[0],f"unet_down_block_attn{i+j+k}_cross_attn_WO_out",IOType.OUTPUT, returns_tuple=True),
                                ))
        for attention in unet.mid_block.attentions:
            for block in attention.transformer_blocks:
                self.mid_attn_block = UnetTransformerBlock(
                        in_=ModuleAccessor(block,"unet_mid_block_in",IOType.INPUT),
                        out_=ModuleAccessor(block,"unet_mid_block_out",IOType.OUTPUT),
                        self_attn_in=ModuleAccessor(block.attn1,"unet_mid_block_self_attn_in",IOType.INPUT),
                        self_attn_out=ModuleAccessor(block.attn1,"unet_mid_block_self_attn_out",IOType.OUTPUT),
                        cross_attn_in=ModuleAccessor(block.attn2,"unet_mid_block_cross_attn_in",IOType.INPUT),
                        cross_attn_out=ModuleAccessor(block.attn2,"unet_mid_block_cross_attn_out",IOType.OUTPUT),
                        mlp_in=ModuleAccessor(block.ff,"unet_mid_block_mlp_in",IOType.INPUT),
                        mlp_out=ModuleAccessor(block.ff,"unet_mid_block_mlp_out",IOType.OUTPUT),
                        self_attn_WO_in=ModuleAccessor(block.attn1.to_out[0],"unet_mid_block_self_attn_WO_in",IOType.INPUT),
                        self_attn_WO_out=ModuleAccessor(block.attn1.to_out[0],"unet_mid_block_self_attn_WO_out",IOType.OUTPUT),
                        cross_attn_WO_in=ModuleAccessor(block.attn2.to_out[0],"unet_mid_block_cross_attn_WO_in",IOType.INPUT),
                        cross_attn_WO_out=ModuleAccessor(block.attn2.to_out[0],"unet_mid_block_cross_attn_WO_out",IOType.OUTPUT),
                        ) 
        for i,up_block in enumerate(unet.up_blocks):
            if up_block._get_name()=='CrossAttnUpBlock2D':
                for j,attention in enumerate(up_block.attentions):
                    for k,block in enumerate(attention.transformer_blocks):
                        self.up_attn_blocks.append(UnetTransformerBlock(
                                in_=ModuleAccessor(block,f"unet_up_block_attn{i+j+k}_in",IOType.INPUT),
                                out_=ModuleAccessor(block,f"unet_up_block_attn{i+j+k}_out",IOType.OUTPUT),
                                self_attn_in=ModuleAccessor(block.attn1,f"unet_up_block_attn{i+j+k}_self_attn_in",IOType.INPUT),
                                self_attn_out=ModuleAccessor(block.attn1,f"unet_up_block_attn{i+j+k}_self_attn_out",IOType.OUTPUT),
                                cross_attn_in=ModuleAccessor(block.attn2,f"unet_up_block_attn{i+j+k}_cross_attn_in",IOType.INPUT),
                                cross_attn_out=ModuleAccessor(block.attn2,f"unet_up_block_attn{i+j+k}_cross_attn_out",IOType.OUTPUT),
                                mlp_in=ModuleAccessor(block.ff,f"unet_up_block_attn{i+j+k}_mlp_in",IOType.INPUT),
                                mlp_out=ModuleAccessor(block.ff,f"unet_up_block_attn{i+j+k}_mlp_out",IOType.OUTPUT),
                                self_attn_WO_in=ModuleAccessor(block.attn1.to_out[0],f"unet_up_block_attn{i+j+k}_self_attn_WO_in",IOType.INPUT),
                                self_attn_WO_out=ModuleAccessor(block.attn1.to_out[0],f"unet_up_block_attn{i+j+k}_self_attn_WO_out",IOType.OUTPUT, returns_tuple=True),
                                cross_attn_WO_in=ModuleAccessor(block.attn2.to_out[0],f"unet_up_block_attn{i+j+k}_cross_attn_WO_in",IOType.INPUT),
                                cross_attn_WO_out=ModuleAccessor(block.attn2.to_out[0],f"unet_up_block_attn{i+j+k}_cross_attn_WO_out",IOType.OUTPUT, returns_tuple=True),
                                ))

    def summary(self) -> str:
        s = "UNet:\n"
        s += f"down_attn_blocks: {len(self.down_attn_blocks)}\n"
        for i, block in enumerate(self.down_attn_blocks):
            s += f"    [{i:01d}] {block.summary()}\n"
        s += f"mid_attn_block:\n"
        if self.mid_attn_block:
            s += f"    {self.mid_attn_block.summary()}\n"
        else:
            s += "    None\n"
        s += f"up_attn_blocks: {len(self.up_attn_blocks)}\n"
        for i, block in enumerate(self.up_attn_blocks):
            s += f"    [{i:01d}] {block.summary()}\n"
        return s
    
    def __repr__(self):
        return self.summary()
        
        