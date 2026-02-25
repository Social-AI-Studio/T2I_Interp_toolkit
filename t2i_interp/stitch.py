# stitcher.py
from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch as th
import torch.nn as nn

from t2i_interp.t2i import T2IModel
from t2i_interp.utils.output import Output
from t2i_interp.utils.runningstats import TrainUpdate, Update
from t2i_interp.utils.T2I.hook import CaptureHook, UNetAlterHook
from t2i_interp.utils.trace import TraceDict
from functools import partial

class Stitcher:
    """
    Tools for:
      1) skip_module(model, module, replacement)
      2) run_diffusion_lens(model, ...)
      3) graft(model_a, model_b, module_a, module_b)
      4) map(model_a, model_b, module_a, module_b, mapper, prompt)
      5) train_mapper(train_loader, mapper, ...)
    """

    def skip_module(
        self,
        prompt: str,
        model: T2IModel,
        module_to_skip: nn.Module,
        replacement: nn.Module,
    ) -> "StitchResult":
        """
        Replace a submodule with:
          - 'identity' -> returns input
          - 'zero'     -> zeros_like(input)
          - nn.Module  -> your custom replacement (deepcopied by default)
        `module` can be a dotted path or a module object (path recommended).
        """
        pass

    def run_diffusion_lens(
        self,
        prompt: str,
        model: T2IModel,
        module_list: list[nn.Module],
        final_module: nn.Module,
        masks: list | None = None,
        final_norm: nn.Module | None = None,
        **kwargs,
    ) -> Output:
        """
        Runs a Diffusers-style pipeline with a callback that captures per-step
        latents, optionally decodes them (simple diffusion lens).
        """
        preds = []
        for m in module_list:
            with model.generate(prompt, **kwargs):
                if masks:
                    for mask in masks:
                        mask.module.output = mask.value
                if final_norm:
                    final_norm.value = m.value
                else:
                    final_module.value = m.value
                output = model.output.save()
                preds.extend(output.images)
        return Output(preds=preds)

    def graft(
        self,
        model_a: T2IModel,
        model_b: T2IModel,
        module_a: str,
        module_b: str,
    ) -> "StitchResult":
        """
        Join two models at specified modules.
        - model_a: first model (gets module_b inserted)
        - model_b: second model (source of module_b)
        """
        pass

    def map(
        self,
        model: T2IModel,
        module_a: str,                  # accessor path for the source module (e.g. "text_encoder.text_model.final_layer_norm")
        module_b: str,                  # accessor path for the target module (e.g. "unet.conv_out")
        mapper: nn.Module,
        prompts: list[str],
        device: str = "cuda",
        num_inference_steps: int = 30,
        inject_steps: list[int] | None = None,  # which UNet steps to inject; None=all, default=[0]
        **pipeline_kwargs,
    ) -> list:
        """Run the pipeline with the mapper stitching module_a → module_b.

        1. A :class:`CaptureHook` on ``module_a`` records its output on the
           first call (text encoding step).
        2. An :class:`AlterHook` on ``module_b`` intercepts its output and
           replaces it with ``mapper(act_a.reshape(B,-1)).reshape(original_shape)``.

        Args:
            model: The :class:`T2IModel` wrapping the diffusion pipeline.
            module_a: Accessor path for the source activation module.
            module_b: Accessor path for the target activation module.
            mapper: Trained :class:`nn.Module` (e.g. :class:`MLPMapper`).
            prompts: List of text prompts.
            device: Compute device.
            num_inference_steps: Diffusion steps.
            inject_steps: Which UNet denoising steps to apply the injection on.
                ``None`` → inject on every step.  Default ``[0]`` → step 0 only
                (matches the ``capture_step_index=0`` used during latent collection).
            **pipeline_kwargs: Extra kwargs forwarded to the pipeline.

        Returns:
            List of PIL images.
        """

        pipe = model.pipeline
        pipe.set_progress_bar_config(disable=True)

        mod_a = model.resolve_accessor(module_a).module
        mod_b = model.resolve_accessor(module_b).module

        mapper = mapper.eval().to(device)

        # --- Step 1: run text encoder pass to capture act_a ---
        capture = CaptureHook(capture="output")

        _inject_steps = inject_steps if inject_steps is not None else list(range(num_inference_steps))
        act_a_cache: list[th.Tensor] = []   # populated once after the text encoder fires
        step_counter   = [0]   # mutable so the closure can mutate it

        def _inject_policy(act_b: th.Tensor, **_) -> th.Tensor:
            """Replace act_b with mapper(act_a) on the configured steps."""
            current_step = step_counter[0]
            step_counter[0] += 1
            if not act_a_cache or current_step not in _inject_steps:
                return act_b                         # pass through
            act_a = act_a_cache[0].to(device=act_b.device, dtype=act_b.dtype)
            B     = act_b.shape[0]
            # handle CFG doubling: act_a may be (B,…) while act_b is (2B,…)
            if act_a.shape[0] < B:
                act_a = act_a.repeat(B // act_a.shape[0], *([1] * (act_a.ndim - 1)))
            flat  = act_a.reshape(act_a.shape[0], -1)
            with th.no_grad():
                mapped_flat = mapper(flat)           # (B, output_dim)
            return mapped_flat.reshape(act_b.shape) if mapped_flat.numel() == act_b.numel() \
                   else mapped_flat.reshape(B, *act_b.shape[1:])

        alter = UNetAlterHook(policy=_inject_policy)

        # Register both hooks and run
        with th.no_grad():
            with TraceDict([mod_a, mod_b], {mod_a: capture, mod_b: alter}):
                # After the text encoder fires, populate act_a_cache
                def _on_capture(module, inputs, output):
                    if capture.last is not None and not act_a_cache:
                        act_a_cache.append(capture.last.detach().cpu())
                mod_a.register_forward_hook(_on_capture)        # one-shot capture helper
                imgs = pipe(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    **pipeline_kwargs,
                ).images

        return imgs

    def train_mapper(
        self,
        train_loader,                              # pre-built DataLoader: yields (act_a, act_b) or {"a": ..., "b": ...}
        mapper: nn.Module,
        optimizers: list[th.optim.Optimizer],
        val_loader=None,                           # optional DataLoader – same format as train_loader
        num_steps: int = 1_000,
        loss_fn: Callable | None = None,
        training_device: str = "cuda",
        autocast_dtype: th.dtype = th.float32,
        log_steps: int = 100,
        model: T2IModel | None = None,             # kept for API compatibility; not used internally
        **kwargs,
    ) -> Generator[Update, None, nn.Module]:
        """Train a mapper network between two activation spaces.

        The mapper is trained to minimise ``loss_fn(mapper(act_a), act_b)``,
        defaulting to MSE.  A validation sweep is performed every
        ``log_steps`` steps when ``val_loader`` is provided; the best
        checkpoint (lowest val loss) is restored before returning.

        Args:
            train_loader: Pre-built DataLoader.  Each batch must be either:
                * a 2-tuple/list ``(act_a, act_b)``, or
                * a dict with keys ``"a"`` and ``"b"``.
            mapper: ``nn.Module`` to train.
            optimizers: List of optimizers (e.g. ``[Adam(mapper.parameters())]``).
            val_loader: Optional validation DataLoader (same format).
            num_steps: Total training steps.
            loss_fn: Loss callable ``(pred, target) -> scalar``.
                     Defaults to ``F.mse_loss``.
            training_device: Device to move activations to.
            autocast_dtype: ``torch.dtype`` for ``autocast``; ignored on CPU.
            log_steps: Yield a :class:`TrainUpdate` every this many steps.
            model: Unused; kept for TrainingSpec compatibility.
            **kwargs: Ignored extra kwargs from TrainingSpec.

        Yields:
            :class:`Update` / :class:`TrainUpdate` progress events, then the
            trained mapper as the final yielded value.

        Returns:
            Best :class:`nn.Module` checkpoint (or final if no val_loader).
        """
        if loss_fn is None:
            loss_fn = th.nn.functional.mse_loss

        mapper = mapper.to(device=training_device, dtype=autocast_dtype)
        autocast_ctx = (
            nullcontext()
            if training_device == "cpu"
            else th.autocast(device_type=training_device.split(":")[0], dtype=autocast_dtype)
        )

        yield Update(
            info=f"Starting Stitcher mapper training: steps={num_steps}, "
                 f"device={training_device}, val={'yes' if val_loader is not None else 'no'}"
        )

        best_val       = float("inf")
        best_mapper_sd = deepcopy(mapper.state_dict())

        def _unpack(batch):
            """Return (act_a, act_b) from either a tuple or a dict batch."""
            if isinstance(batch, (tuple, list)):
                return batch[0], batch[1]
            return batch["a"], batch["b"]

        def _run_val():
            mapper.eval()
            val_loss, n = 0.0, 0
            with th.no_grad():
                if hasattr(val_loader, "reset"):
                    val_loader.reset()
                val_it = val_loader.iterate() if hasattr(val_loader, "iterate") else iter(val_loader)
                for vbatch in val_it:
                    va, vb = _unpack(vbatch)
                    va = va.to(training_device, dtype=autocast_dtype).reshape(va.shape[0], -1)
                    vb = vb.to(training_device, dtype=autocast_dtype).reshape(vb.shape[0], -1)
                    val_loss += float(loss_fn(mapper(va), vb))
                    n += 1
            mapper.train()
            return val_loss / max(n, 1)

        step = 0
        # Support both standard DataLoader (iter) and custom loaders with .iterate()
        def _make_iter(loader):
            if hasattr(loader, "reset"):
                loader.reset()
            return loader.iterate() if hasattr(loader, "iterate") else iter(loader)

        it   = _make_iter(train_loader)
        mapper.train()

        while step < num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it    = _make_iter(train_loader)
                batch = next(it)

            act_a, act_b = _unpack(batch)
            act_a = act_a.to(training_device, dtype=autocast_dtype).reshape(act_a.shape[0], -1)
            act_b = act_b.to(training_device, dtype=autocast_dtype).reshape(act_b.shape[0], -1)

            with autocast_ctx:
                mapped = mapper(act_a)
                loss   = loss_fn(mapped, act_b)

            loss.backward()
            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            if log_steps and (step % log_steps == 0):
                if val_loader is not None:
                    val_loss = _run_val()
                    if val_loss < best_val:
                        best_val       = val_loss
                        best_mapper_sd = deepcopy(mapper.state_dict())
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item()), "val_loss": val_loss})
                else:
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item())})

            step += 1

        # Restore best checkpoint
        if val_loader is not None:
            mapper.load_state_dict(best_mapper_sd)
            yield Update(info=f"Mapper training done. Best val_loss={best_val:.6f}")
        else:
            yield Update(info="Mapper training done.")

        yield mapper
