import torch as th

from t2Interp.accessors import IOType, ModuleAccessor
from t2Interp.blocks import UnetTransformerBlock

# from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D


class UnetDownBlock:
    down_blocks: list[UnetTransformerBlock]

    def __init__(self, down_blocks: list[UnetTransformerBlock]):
        self.down_blocks = down_blocks


class UnetMidBlock:
    mid_blocks: list[UnetTransformerBlock]

    def __init__(self, mid_blocks: list[UnetTransformerBlock]):
        self.mid_blocks = mid_blocks


class UnetUpBlock:
    up_blocks: list[UnetTransformerBlock]

    def __init__(self, up_blocks: list[UnetTransformerBlock]):
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
        self.down_attn_blocks = []
        self.mid_attn_block = None
        self.up_attn_blocks = []

        in_ = ModuleAccessor(unet.conv_in, "unet_in", IOType.INPUT)
        out_ = ModuleAccessor(unet.conv_out, "unet_out", IOType.OUTPUT)
        self.in_ = in_
        self.out_ = out_

        i = 0
        for down_block in unet.down_blocks:
            if down_block._get_name() == "CrossAttnDownBlock2D":
                for attention in down_block.attentions:
                    for block in attention.transformer_blocks:
                        self.down_attn_blocks.append(
                            UnetTransformerBlock(
                                in_=ModuleAccessor(
                                    block, f"unet_down_block_attn{i}_in", IOType.INPUT
                                ),
                                out_=ModuleAccessor(
                                    block, f"unet_down_block_attn{i}_out", IOType.OUTPUT
                                ),
                                self_attn_in=ModuleAccessor(
                                    block.attn1,
                                    f"unet_down_block_attn{i}_self_attn_in",
                                    IOType.INPUT,
                                ),
                                self_attn_out=ModuleAccessor(
                                    block.attn1,
                                    f"unet_down_block_attn{i}_self_attn_out",
                                    IOType.OUTPUT,
                                ),
                                cross_attn_in=ModuleAccessor(
                                    block.attn2,
                                    f"unet_down_block_attn{i}_cross_attn_in",
                                    IOType.INPUT,
                                ),
                                cross_attn_out=ModuleAccessor(
                                    block.attn2,
                                    f"unet_down_block_attn{i}_cross_attn_out",
                                    IOType.OUTPUT,
                                ),
                                mlp_in=ModuleAccessor(
                                    block.ff, f"unet_down_block_attn{i}_mlp_in", IOType.INPUT
                                ),
                                mlp_out=ModuleAccessor(
                                    block.ff, f"unet_down_block_attn{i}_mlp_out", IOType.OUTPUT
                                ),
                                self_attn_WO_in=ModuleAccessor(
                                    block.attn1.to_out[0],
                                    f"unet_down_block_attn{i}_self_attn_WO_in",
                                    IOType.INPUT,
                                ),
                                self_attn_WO_out=ModuleAccessor(
                                    block.attn1.to_out[0],
                                    f"unet_down_block_attn{i}_self_attn_WO_out",
                                    IOType.OUTPUT,
                                    returns_tuple=True,
                                ),
                                cross_attn_WO_in=ModuleAccessor(
                                    block.attn2.to_out[0],
                                    f"unet_down_block_attn{i}_cross_attn_WO_in",
                                    IOType.INPUT,
                                ),
                                cross_attn_WO_out=ModuleAccessor(
                                    block.attn2.to_out[0],
                                    f"unet_down_block_attn{i}_cross_attn_WO_out",
                                    IOType.OUTPUT,
                                    returns_tuple=True,
                                ),
                            )
                        )
                        i += 1
        for attention in unet.mid_block.attentions:
            for block in attention.transformer_blocks:
                self.mid_attn_block = UnetTransformerBlock(
                    in_=ModuleAccessor(block, "unet_mid_block_in", IOType.INPUT),
                    out_=ModuleAccessor(block, "unet_mid_block_out", IOType.OUTPUT),
                    self_attn_in=ModuleAccessor(
                        block.attn1, "unet_mid_block_self_attn_in", IOType.INPUT
                    ),
                    self_attn_out=ModuleAccessor(
                        block.attn1, "unet_mid_block_self_attn_out", IOType.OUTPUT
                    ),
                    cross_attn_in=ModuleAccessor(
                        block.attn2, "unet_mid_block_cross_attn_in", IOType.INPUT
                    ),
                    cross_attn_out=ModuleAccessor(
                        block.attn2, "unet_mid_block_cross_attn_out", IOType.OUTPUT
                    ),
                    mlp_in=ModuleAccessor(block.ff, "unet_mid_block_mlp_in", IOType.INPUT),
                    mlp_out=ModuleAccessor(block.ff, "unet_mid_block_mlp_out", IOType.OUTPUT),
                    self_attn_WO_in=ModuleAccessor(
                        block.attn1.to_out[0], "unet_mid_block_self_attn_WO_in", IOType.INPUT
                    ),
                    self_attn_WO_out=ModuleAccessor(
                        block.attn1.to_out[0], "unet_mid_block_self_attn_WO_out", IOType.OUTPUT
                    ),
                    cross_attn_WO_in=ModuleAccessor(
                        block.attn2.to_out[0], "unet_mid_block_cross_attn_WO_in", IOType.INPUT
                    ),
                    cross_attn_WO_out=ModuleAccessor(
                        block.attn2.to_out[0], "unet_mid_block_cross_attn_WO_out", IOType.OUTPUT
                    ),
                )
        i = 0
        for up_block in unet.up_blocks:
            if up_block._get_name() == "CrossAttnUpBlock2D":
                for attention in up_block.attentions:
                    for block in attention.transformer_blocks:
                        self.up_attn_blocks.append(
                            UnetTransformerBlock(
                                in_=ModuleAccessor(
                                    block, f"unet_up_block_attn{i}_in", IOType.INPUT
                                ),
                                out_=ModuleAccessor(
                                    block, f"unet_up_block_attn{i}_out", IOType.OUTPUT
                                ),
                                self_attn_in=ModuleAccessor(
                                    block.attn1, f"unet_up_block_attn{i}_self_attn_in", IOType.INPUT
                                ),
                                self_attn_out=ModuleAccessor(
                                    block.attn1,
                                    f"unet_up_block_attn{i}_self_attn_out",
                                    IOType.OUTPUT,
                                ),
                                cross_attn_in=ModuleAccessor(
                                    block.attn2,
                                    f"unet_up_block_attn{i}_cross_attn_in",
                                    IOType.INPUT,
                                ),
                                cross_attn_out=ModuleAccessor(
                                    block.attn2,
                                    f"unet_up_block_attn{i}_cross_attn_out",
                                    IOType.OUTPUT,
                                ),
                                mlp_in=ModuleAccessor(
                                    block.ff, f"unet_up_block_attn{i}_mlp_in", IOType.INPUT
                                ),
                                mlp_out=ModuleAccessor(
                                    block.ff, f"unet_up_block_attn{i}_mlp_out", IOType.OUTPUT
                                ),
                                self_attn_WO_in=ModuleAccessor(
                                    block.attn1.to_out[0],
                                    f"unet_up_block_attn{i}_self_attn_WO_in",
                                    IOType.INPUT,
                                ),
                                self_attn_WO_out=ModuleAccessor(
                                    block.attn1.to_out[0],
                                    f"unet_up_block_attn{i}_self_attn_WO_out",
                                    IOType.OUTPUT,
                                    returns_tuple=True,
                                ),
                                cross_attn_WO_in=ModuleAccessor(
                                    block.attn2.to_out[0],
                                    f"unet_up_block_attn{i}_cross_attn_WO_in",
                                    IOType.INPUT,
                                ),
                                cross_attn_WO_out=ModuleAccessor(
                                    block.attn2.to_out[0],
                                    f"unet_up_block_attn{i}_cross_attn_WO_out",
                                    IOType.OUTPUT,
                                    returns_tuple=True,
                                ),
                            )
                        )
                        i += 1

    def summary(self) -> str:
        s = "UNet:\n"
        s += f"down_attn_blocks: {len(self.down_attn_blocks)}\n"
        for i, block in enumerate(self.down_attn_blocks):
            s += f"    [{i:01d}] {block.summary()}\n"
        s += "mid_attn_block:\n"
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
