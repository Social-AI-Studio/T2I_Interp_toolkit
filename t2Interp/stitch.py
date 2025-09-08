# stitcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import copy
import inspect
import re
import torch as th
import torch.nn as nn
from T2I import T2IModel


@dataclass
class StitchResult:
    model: Optional[T2IModel] = None  
    info: Dict[str, Any]


class MaskSpec:
    module:nn.Module
    value: Any
    
    def __repr__(self):
        return f"MaskSpec(module={self.module}, value={self.value})"
    
# ModuleSpec = Union[str, nn.Module]
# MaybeModels = Union[nn.Module, Sequence[nn.Module]]

class Stitcher:
    """
    Tools for:
      1) skip_module(model, module, replacement)
      2) run_diffusion_lens(model, ...)
      3) stitch(model_a, model_b, module_a, module_b, mode=...)
      4) map(model_or_models, target_modules, mapper)

    All return (model_or_models, info) where info includes restore handles and metadata.
    """

    def skip_module(
        self,
        prompt: str,
        model: T2IModel,
        module_to_skip: nn.Module,
        replacement: nn.Module,
    ) -> StitchResult:
        """
        Replace a submodule with:
          - 'identity' -> returns input
          - 'zero'     -> zeros_like(input)
          - nn.Module  -> your custom replacement (deepcopied by default)
        `module` can be a dotted path or a module object (path recommended).
        """
        with model.trace(prompt):
            module_to_skip.skip(replacement.output)
            skipped_out=model.output.save()

        info = {"skipped_module": str(module_to_skip), "skipped_With": str(replacement), "skipped_output": skipped_out}
        return StitchResult(info=info)

    
    def run_diffusion_lens(
        self,
        prompt: str,
        model: T2IModel,
        module_list: List[nn.Module],
        final_module: nn.Module,
        # masked_module: Optional[nn.Module] = None,
        masks: Optional[List[MaskSpec]] = None,
        **kwargs,
    ) -> StitchResult:
        """
        Runs a Diffusers-style pipeline with a callback that captures per-step latents,
        optionally decodes them (simple diffusion lens).

        Requirements:
          - `model` has a __call__ that accepts `callback` and `callback_steps` (most HF diffusion pipelines).
          - For decoded collection, model.vae must be present or you pass `decode_fn(model, latents)`.
        """     
        info=[]    
        for m in module_list:
            with th.no_grad():
                with model.generate(
                    prompt,
                    **kwargs,
                ):
                    if masks:
                        for mask in masks:
                            mask.module.output = mask.value

                    final_module.input = m.output[0]

                    image = model.output.images[0].save()
                    info.append({"module": m, "image": image, "masks": masks})
        return StitchResult(info=info)

    def graft(
        self,
        model_a: T2IModel,
        model_b: T2IModel,
        module_a: str,
        module_b: str,
    ) -> StitchResult:
        """
        Join two models at specified modules.
        - model_a: first model (gets module_b inserted)
        - model_b: second model (source of module_b)"""
        
        # implement "Distilling Diversity and Control in Diffusion Models"
        # code to be adapted from - https://github.com/rohitgandikota/distillation/blob/32f59eaba3f04a73c53462f291805adcbf0354e3/utils/utils.py

        info = {
            "model_a_path": module_a,
            "model_b_path": module_b,
           
        }
        return StitchResult(info=info)
    
    def map(
        self,
        model_a: T2IModel,
        model_b: Optional[T2IModel],
        module_a: str,
        module_b: str,
        mapper: nn.Module,
        prompt: str,
        **kwargs,
    ) -> StitchResult:
        
        if model_b:
            with model_a.generate(prompt, **kwargs) as tracer:
                output_a = model_a.output.save()
                tracer.stop()
            input_b = mapper(output_a)
            with model_b.trace(prompt):
                module_b.input = input_b
                model_out = model_b.output.save()
        else:
            with model_a.generate(prompt, **kwargs) as tracer:
                output_a = model_a.output.save()
                input_b= mapper(output_a)
                module_b.input = input_b
                model_out = model_a.output.save()
            
        return StitchResult(info={"mapped_output": model_out})



# if __name__ == "__main__":
#     # Minimal smoke test on toy modules (no diffusers dependency).
#     class ToyBlock(nn.Module):
#         def __init__(self, d=8): super().__init__(); self.lin = nn.Linear(d, d)
#         def forward(self, x): return self.lin(x)

#     class ToyModel(nn.Module):
#         def __init__(self, d=8):
#             super().__init__()
#             self.backbone = nn.Sequential(ToyBlock(d), ToyBlock(d))
#         def forward(self, x, **kw): return self.backbone(x)

#     A = ToyModel()
#     B = ToyModel()
#     st = Stitcher()

#     # skip_module
#     r1 = st.skip_module(A, "backbone.0.lin", "identity")
#     y = A(torch.randn(2, 8))
#     r1.info["restore"]()

#     # stitch replace
#     r2 = st.stitch(A, B, "backbone.1", "backbone.0", mode="replace")
#     y = A(torch.randn(2, 8))
#     r2.info["restore"]()

#     # map with a mapper
#     mapper = nn.Sequential(nn.ReLU())
#     r3 = st.map(A, "backbone.0", mapper)
#     y = A(torch.randn(2, 8))
#     r3.info["restore"]()
