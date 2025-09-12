import torch as th
from t2Interp.accessors import ModuleAccessor, AttentionAccessor, IOType
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple
from t2Interp.blocks import TransformerBlock

@dataclass
class DualStreamBlock(TransformerBlock):
    """
    Flux-style dual-stream transformer block:
      - Per-stream pre/post norms
      - Per-stream MLP (up / act / down)
      - Single shared AttentionAccessor via base class `attention`

    Field naming follows slot config:
      - image_pre_norm   -> maps to 'norm1'
      - text_pre_norm    -> maps to 'norm1_context'
      - image_post_norm  -> maps to 'norm2'
      - text_post_norm   -> maps to 'norm2_context'
      - mlp_image_up     -> maps to 'ff.net.0.proj'
      - mlp_image_act    -> maps to 'ff.net.0'
      - mlp_image_down   -> maps to 'ff.net.2'
      - mlp_text_up      -> maps to 'ff_context.net.0.proj'
      - mlp_text_act     -> maps to 'ff_context.net.0'
      - mlp_text_down    -> maps to 'ff_context.net.2'
    """

    # per-stream norms
    image_pre_norm_in:  Optional[ModuleAccessor] = None
    image_pre_norm_out: Optional[ModuleAccessor] = None
    text_pre_norm_in:   Optional[ModuleAccessor] = None
    text_pre_norm_out:  Optional[ModuleAccessor] = None
    image_post_norm_in: Optional[ModuleAccessor] = None
    image_post_norm_out:Optional[ModuleAccessor] = None
    text_post_norm_in:  Optional[ModuleAccessor] = None
    

    # image-stream MLP slots
    mlp_image_up_in:   Optional[ModuleAccessor] = None
    mlp_image_up_out:  Optional[ModuleAccessor] = None
    mlp_image_act_in:  Optional[ModuleAccessor] = None
    mlp_image_act_out: Optional[ModuleAccessor] = None
    mlp_image_down_in: Optional[ModuleAccessor] = None
    mlp_image_down_out:Optional[ModuleAccessor] = None

    # text-stream MLP slots
    mlp_text_up_in:   Optional[ModuleAccessor] = None
    mlp_text_up_out:  Optional[ModuleAccessor] = None
    mlp_text_act_in:  Optional[ModuleAccessor] = None
    mlp_text_act_out: Optional[ModuleAccessor] = None
    mlp_text_down_in: Optional[ModuleAccessor] = None
    mlp_text_down_out: Optional[ModuleAccessor] = None

    def __init__(self, module: th.nn.Module):
        self.in_ = ModuleAccessor(module,"dual_stream_input",IOType.INPUT)
        self.out_ = ModuleAccessor(module,"dual_stream_output",IOType.OUTPUT)
        
        # norms
        self.image_pre_norm_in = ModuleAccessor(module.norm1, "dual_stream_image_pre_norm", IOType.INPUT)
        self.image_pre_norm_out = ModuleAccessor(module.norm1, "dual_stream_image_pre_norm", IOType.OUTPUT)
        self.text_pre_norm_in = ModuleAccessor(module.norm1_context, "dual_stream_text_pre_norm", IOType.INPUT)
        self.text_pre_norm_out = ModuleAccessor(module.norm1_context, "dual_stream_text_pre_norm", IOType.OUTPUT)
        self.image_post_norm_in = ModuleAccessor(module.norm2, "dual_stream_image_post_norm", IOType.INPUT)
        self.image_post_norm_out = ModuleAccessor(module.norm2, "dual_stream_image_post_norm", IOType.OUTPUT)
        self.text_post_norm_in = ModuleAccessor(module.norm2_context, "dual_stream_text_post_norm", IOType.INPUT)
        self.text_post_norm_out = ModuleAccessor(module.norm2_context, "dual_stream_text_post_norm", IOType.OUTPUT)
        
        # image MLP
        self.mlp_image_up_in = ModuleAccessor(module.mlp_image.up, "dual_stream_mlp_image_up", IOType.INPUT)
        self.mlp_image_up_out = ModuleAccessor(module.mlp_image.up.proj, "dual_stream_mlp_image_up", IOType.OUTPUT)
        self.mlp_image_act_in = ModuleAccessor(module.mlp_image.act, "dual_stream_mlp_image_act", IOType.INPUT)
        self.mlp_image_act_out = ModuleAccessor(module.mlp_image.act, "dual_stream_mlp_image_act", IOType.OUTPUT)
        self.mlp_image_down_in = ModuleAccessor(module.mlp_image.down, "dual_stream_mlp_image_down", IOType.INPUT)
        self.mlp_image_down_out = ModuleAccessor(module.mlp_image.down, "dual_stream_mlp_image_down", IOType.OUTPUT)
        
        # text MLP
        self.mlp_text_up_in = ModuleAccessor(module.mlp_text.up, "dual_stream_mlp_text_up", IOType.INPUT)
        self.mlp_text_up_out = ModuleAccessor(module.mlp_text.up, "dual_stream_mlp_text_up", IOType.OUTPUT)
        self.mlp_text_act_in = ModuleAccessor(module.mlp_text.act, "dual_stream_mlp_text_act", IOType.INPUT)
        self.mlp_text_act_out = ModuleAccessor(module.mlp_text.act, "dual_stream_mlp_text_act", IOType.OUTPUT)
        self.mlp_text_down_in = ModuleAccessor(module.mlp_text.down, "dual_stream_mlp_text_down", IOType.INPUT)
        self.mlp_text_down_out = ModuleAccessor(module.mlp_text.down, "dual_stream_mlp_text_down", IOType.OUTPUT)
      

    def summary(self) -> str:
        return (
            f"in_={self.in_}, out_={self.out_}, "
            f"image_pre_norm_in={self.image_pre_norm_in}, image_pre_norm_out={self.image_pre_norm_out}, "
            f"text_pre_norm_in={self.text_pre_norm_in}, text_pre_norm_out={self.text_pre_norm_out}, "
            f"image_post_norm_in={self.image_post_norm_in}, image_post_norm_out={self.image_post_norm_out}, "
            f"text_post_norm_in={self.text_post_norm_in}, text_post_norm_out={self.text_post_norm_out}, "
            f"mlp_image_up_in={self.mlp_image_up_in}, mlp_image_up_out={self.mlp_image_up_out}, "
            f"mlp_image_act_in={self.mlp_image_act_in}, mlp_image_act_out={self.mlp_image_act_out}, "
            f"mlp_image_down_in={self.mlp_image_down_in}, mlp_image_down_out={self.mlp_image_down_out}, "
            f"mlp_text_up_in={self.mlp_text_up_in}, mlp_text_up_out={self.mlp_text_up_out}, "
            f"mlp_text_act_in={self.mlp_text_act_in}, mlp_text_act_out={self.mlp_text_act_out}, "
            f"mlp_text_down_in={self.mlp_text_down_in}, mlp_text_down_out={self.mlp_text_down_out}, "
            f"attn_in={self.attn_in}, attn_out={self.attn_out})"
        )
