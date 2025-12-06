# stitcher.py
from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import tee
from typing import Any

import torch as th
import torch.nn as nn

from dictionary_learning.utils import hf_dataset_to_generator
from t2Interp.T2I import T2IModel
from utils.buffer import t2IActivationBuffer
from utils.output import Output
from utils.runningstats import TrainUpdate


@dataclass
class StitchResult(Output):
    model: T2IModel | None = None
    info: dict[str, Any] = None


class MaskSpec:
    module: nn.Module
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
            skipped_out = model.output.save()

        info = {"skipped_module": str(module_to_skip), "skipped_With": str(replacement)}
        return StitchResult(info=info, preds=skipped_out)

    def run_diffusion_lens(
        self,
        prompt: str,
        model: T2IModel,
        module_list: list[nn.Module],
        final_module: nn.Module,
        # masked_module: Optional[nn.Module] = None,
        masks: list[MaskSpec] | None = None,
        final_norm: nn.Module | None = None,
        **kwargs,
    ) -> StitchResult:
        """
        Runs a Diffusers-style pipeline with a callback that captures per-step latents,
        optionally decodes them (simple diffusion lens).

        Requirements:
          - `model` has a __call__ that accepts `callback` and `callback_steps` (most HF diffusion pipelines).
          - For decoded collection, model.vae must be present or you pass `decode_fn(model, latents)`.
        """
        # info=[]
        preds = []
        for m in module_list:
            with model.generate(
                prompt,
                **kwargs,
            ):
                if masks:
                    for mask in masks:
                        mask.module.output = mask.value

                if final_norm:
                    final_norm.value = m.value  # m.output[0]
                else:
                    final_module.value = m.value  # m.output[0]

                output = model.output.save()
                # info.append({"module": m, "image": image, "masks": masks})
                preds.extend(output.images)
        return Output(preds=preds)

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
        model_b: T2IModel | None,
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
                input_b = mapper(output_a)
                module_b.input = input_b
                model_out = model_a.output.save()

        return StitchResult(info={"mapped_output": model_out})

    def train_mapper(
        self,
        hf_dataset,
        steps: int,
        model_a: T2IModel,
        module_a: str,
        module_b: str,
        mapper: nn.Module,
        optimizers: list[th.optim.Optimizer],
        model_b: T2IModel | None = None,
        training_device: str = "cuda",
        data_device: str = "cpu",
        autocast_dtype: th.dtype = th.float32,
        loss_fn: Callable | None = None,
        **kwargs,
    ) -> Generator[TrainUpdate, None, nn.Module]:
        generator = hf_dataset_to_generator(hf_dataset, **kwargs)
        gen_a, gen_b = tee(generator)
        d_sub = kwargs.pop("d_submodule_a", kwargs.pop("d_submodule", None))
        buffer_a = t2IActivationBuffer(gen_a, model_a, module_a, d_submodule=d_sub, **kwargs)
        d_sub = kwargs.pop("d_submodule_b", d_sub)
        buffer_b = t2IActivationBuffer(
            gen_b,
            model_b if model_b is not None else model_a,
            module_b,
            d_submodule=d_sub,
            **kwargs,
        )

        log_steps = kwargs.get("log_steps", 100)
        autocast_context = (
            nullcontext()
            if training_device == "cpu"
            else th.autocast(device_type=training_device, dtype=autocast_dtype)
        )

        for step, (act_a, act_b) in enumerate(zip(buffer_a, buffer_b, strict=False)):
            with autocast_context:
                # if loss_fn:
                #     mapped,loss = mapper(act_a,loss_fn=loss_fn)
                # else:
                mapped = mapper(act_a)
                if loss_fn:
                    loss = loss_fn(mapped, act_b)
                else:
                    loss = th.nn.functional.mse_loss(mapped, act_b)
                loss.backward()

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()

                if log_steps and step % log_steps == 0:
                    update = TrainUpdate(step=step, parts={"loss": loss.item()})
                    yield update

            if step >= steps:
                break
        return mapper


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
