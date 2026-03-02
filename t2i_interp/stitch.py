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
        module_a: str,                  # accessor path for the source module (e.g. "text_encoder.text_model.final_layer_norm")
        module_b: str,                  # accessor path for the target module (e.g. "unet.conv_out")
        mapper: nn.Module,
        prompts: list[str],
        model_a: T2IModel,              # model that owns module_a (source)
        model_b: T2IModel | None = None,  # model that owns module_b (target); None → same as model_a
        device: str = "cuda",
        num_inference_steps: int = 30,
        inject_steps: list[int] | None = None,  # which UNet steps to inject; None=all
        **pipeline_kwargs,
    ) -> list:
        """Run the pipeline with the mapper stitching module_a → module_b.

        Supports both single-model and cross-model stitching:

        * **Single-model** – pass only ``model_a`` (or set ``model_b=None``).
          ``module_a`` is captured lazily during ``model_a.pipeline``'s forward pass
          and the mapped activation is injected into ``module_b`` on the configured
          denoising steps.

        * **Cross-model** – pass distinct ``model_a`` and ``model_b``.
          ``module_a``'s activation is pre-captured by running ``model_a.pipeline``
          for a single step (fast), then ``model_b.pipeline`` is run for
          ``num_inference_steps`` with the mapped activation injected into
          ``module_b``.

        Args:
            module_a: Accessor path for the source activation module.
            module_b: Accessor path for the target activation module.
            mapper: Trained :class:`nn.Module` (e.g. :class:`MLPMapper`).
            prompts: List of text prompts.
            model_a: Model that owns ``module_a``.
            model_b: Model that owns ``module_b`` and drives generation.
                     ``None`` → ``module_b`` is also resolved from ``model_a``
                     (single-model mode).
            device: Compute device.
            num_inference_steps: Diffusion denoising steps.
            inject_steps: Which denoising steps to apply the injection on.
                ``None`` → every step.
            **pipeline_kwargs: Extra kwargs forwarded to the generation pipeline.

        Returns:
            List of PIL images.
        """
        # If model_b is not provided, fall back to model_a (single-model mode)
        model_b = model_b or model_a

        cross_model = model_a is not model_b

        mod_a = model_a.resolve_accessor(module_a).module
        mod_b = model_b.resolve_accessor(module_b).module

        mapper = mapper.eval().to(device)

        _inject_steps = inject_steps if inject_steps is not None else list(range(num_inference_steps))
        act_a_cache: list[th.Tensor] = []

        # ------------------------------------------------------------------
        # Cross-model: pre-capture act_a from model_a with a 1-step run
        # ------------------------------------------------------------------
        if cross_model:
            capture = CaptureHook(capture="output")
            with th.no_grad():
                with TraceDict([mod_a], {mod_a: capture}):
                    model_a.pipeline.set_progress_bar_config(disable=True)
                    model_a.pipeline(prompts, num_inference_steps=1, **pipeline_kwargs)
            if capture.last is not None:
                act_a_cache.append(capture.last.detach().cpu())

        # ------------------------------------------------------------------
        # Same-model: capture act_a lazily inside the generation pipeline
        # ------------------------------------------------------------------
        else:
            capture = CaptureHook(capture="output")

        # ------------------------------------------------------------------
        # Injection policy for module_b
        # ------------------------------------------------------------------
        step_counter = [0]

        def _inject_policy(act_b: th.Tensor, **_) -> th.Tensor:
            current_step = step_counter[0]
            step_counter[0] += 1
            if not act_a_cache or current_step not in _inject_steps:
                return act_b
            act_a = act_a_cache[0].to(device=act_b.device, dtype=act_b.dtype)
            B = act_b.shape[0]
            if act_a.shape[0] < B:
                act_a = act_a.repeat(B // act_a.shape[0], *([1] * (act_a.ndim - 1)))
            flat = act_a.reshape(act_a.shape[0], -1)
            with th.no_grad():
                mapped_flat = mapper(flat)
            return mapped_flat.reshape(act_b.shape) if mapped_flat.numel() == act_b.numel() \
                   else mapped_flat.reshape(B, *act_b.shape[1:])

        alter = UNetAlterHook(policy=_inject_policy)

        # ------------------------------------------------------------------
        # Generation pass on model_b
        # ------------------------------------------------------------------
        pipe_b = model_b.pipeline
        pipe_b.set_progress_bar_config(disable=True)

        trace_modules = [mod_b] if cross_model else [mod_a, mod_b]
        trace_hooks   = {mod_b: alter} if cross_model else {mod_a: capture, mod_b: alter}

        with th.no_grad():
            with TraceDict(trace_modules, trace_hooks):
                if not cross_model:
                    def _on_capture(module, inputs, output):
                        if capture.last is not None and not act_a_cache:
                            act_a_cache.append(capture.last.detach().cpu())
                    mod_a.register_forward_hook(_on_capture)
                imgs = pipe_b(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    **pipeline_kwargs,
                ).images

        return imgs

    def steer_transfer(
        self,
        steering_vector: th.Tensor,
        module_b: str,
        mapper: nn.Module,
        model_b: T2IModel,
        prompts: list[str],
        device: str = "cuda",
        alpha: float = 1.0,
        ref_activation: th.Tensor | None = None,
        num_inference_steps: int = 50,
        inject_steps: list[int] | None = None,
        also_generate_baseline: bool = True,
        **pipeline_kwargs,
    ) -> list:
        """Apply a steering vector from model_a's activation space into model_b via the trained mapper.

        The mapper is used to translate a direction vector from model_a's space into
        model_b's space. The transferred delta is then added to model_b's activation at
        ``module_b`` during generation — without replacing it (unlike ``map()``).

        Transfer formula:
            delta_b = mapper(ref_a + alpha * v) - mapper(ref_a)

        Args:
            steering_vector: Flat direction in model_a's activation space, shape ``(flat_dim,)``.
            module_b: Accessor path to the target layer in model_b.
            mapper: Trained :class:`~t2i_interp.mapper.MLPMapper` (model_a → model_b).
            model_b: Model to generate images with.
            prompts: List of text prompts.
            device: Compute device.
            alpha: Steering strength. ``1.0`` = full trained direction.
            ref_activation: Reference activation in model_a's space for computing the
                delta. Shape ``(flat_dim,)``. If ``None``, uses zeros (works well when
                the mapper is approximately linear around zero).
            num_inference_steps: Denoising steps for model_b's pipeline.
            inject_steps: Which denoising steps to inject the delta at. ``None`` = all.
            also_generate_baseline: If ``True``, also generate baseline images (no steering)
                and return them interleaved: ``[steered_0, baseline_0, steered_1, ...]``.
            **pipeline_kwargs: Extra kwargs forwarded to the pipeline.

        Returns:
            List of PIL images. If ``also_generate_baseline=True``, returns pairs
            ``[steered_0, baseline_0, steered_1, baseline_1, ...]``.
        """
        mapper = mapper.eval().to(device)
        v = steering_vector.to(device=device, dtype=th.float32).flatten()

        # Compute reference activation in model_a space
        if ref_activation is None:
            ref_a = th.zeros_like(v)
        else:
            ref_a = ref_activation.to(device=device, dtype=th.float32).flatten()

        # Transfer direction: mapper(ref + alpha*v) - mapper(ref)
        with th.no_grad():
            delta_b = mapper(ref_a.unsqueeze(0) + alpha * v.unsqueeze(0)) \
                    - mapper(ref_a.unsqueeze(0))
            delta_b = delta_b.squeeze(0)  # (flat_dim_b,)

        mod_b = model_b.resolve_accessor(module_b).module
        _inject_steps = inject_steps if inject_steps is not None else list(range(num_inference_steps))
        pipe_b = model_b.pipeline
        pipe_b.set_progress_bar_config(disable=True)

        def _make_inject_hook():
            step_counter = [0]

            def _policy(act_b: th.Tensor, **_) -> th.Tensor:
                current_step = step_counter[0]
                step_counter[0] += 1
                if current_step not in _inject_steps:
                    return act_b
                d = delta_b.to(device=act_b.device, dtype=act_b.dtype)
                # Broadcast delta to match act_b spatial shape
                B = act_b.shape[0]
                d_expanded = d.reshape(act_b.shape[1:]).unsqueeze(0).expand(B, *act_b.shape[1:])
                return act_b + d_expanded

            return UNetAlterHook(policy=_policy)

        # --- Steered generation ---
        with th.no_grad():
            with TraceDict([mod_b], {mod_b: _make_inject_hook()}):
                steered_imgs = pipe_b(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    **pipeline_kwargs,
                ).images

        if not also_generate_baseline:
            return steered_imgs

        # --- Baseline generation (no injection) ---
        with th.no_grad():
            baseline_imgs = pipe_b(
                prompts,
                num_inference_steps=num_inference_steps,
                **pipeline_kwargs,
            ).images

        # Interleave: steered_0, baseline_0, steered_1, baseline_1, ...
        pairs = []
        for s, b in zip(steered_imgs, baseline_imgs):
            pairs.extend([s, b])
        return pairs

    def steer_contrast(
        self,
        pos_prompts: list[str],
        neg_prompts: list[str],
        apply_prompts: list[str],
        module_a: str,
        module_b: str,
        mapper: nn.Module,
        model_a: T2IModel,
        model_b: T2IModel,
        device: str = "cuda",
        alpha: float = 1.0,
        num_inference_steps: int = 50,
        inject_steps: list[int] | None = None,
        also_generate_baseline: bool = True,
        **pipeline_kwargs,
    ) -> list:
        """Compute a steering direction from contrasting prompts on model_a, transfer
        via the trained mapper, and inject into model_b during generation.

        Steps:
          1. Run model_a for 1 step on ``pos_prompts`` and ``neg_prompts`` →
             capture activations at ``module_a``.
          2. Compute steering direction:
             ``v = mean(pos_acts) - mean(neg_acts)``, L2-normalised.
          3. Transfer direction: ``delta_b = alpha * mapper(v)``.
          4. At generation time: ``final_act_b = act_b_natural(apply_prompt) + delta_b``.
          5. Generate images with model_b on ``apply_prompts``, adding ``delta_b``
             additively to ``module_b`` at every target denoising step.

        Args:
            pos_prompts: Prompts representing the positive concept
                (e.g. ``["photo of a black man"]``).
            neg_prompts: Prompts representing the negative/baseline concept
                (e.g. ``["photo of a white man"]``).
            apply_prompts: Prompts to generate steered images for
                (e.g. ``["photo of a man"]``).
            module_a: Accessor path to the layer in model_a to capture from.
            module_b: Accessor path to the layer in model_b to inject into.
            mapper: Trained :class:`~t2i_interp.mapper.MLPMapper`.
            model_a: Source model used to compute the steering direction.
            model_b: Target model used for generation.
            device: Compute device.
            alpha: Steering strength. Negative values reverse the direction.
            num_inference_steps: Denoising steps for model_b.
            inject_steps: Which steps to inject at. ``None`` = all.
            also_generate_baseline: Also run an unsteered baseline generation
                and interleave results ``[steered_0, baseline_0, ...]``.
            **pipeline_kwargs: Extra kwargs forwarded to pipelines.

        Returns:
            List of PIL images. If ``also_generate_baseline=True``, pairs are
            interleaved ``[steered_0, baseline_0, steered_1, baseline_1, ...]``.
        """
        mapper = mapper.eval().to(device)
        mod_a = model_a.resolve_accessor(module_a).module

        # ── Step 1: capture activations from model_a for pos and neg prompts ──
        def _capture_acts(prompts_list: list[str]) -> th.Tensor:
            """1-step forward pass on model_a; returns mean activation (flat)."""
            captured = []
            capture_hook = CaptureHook(capture="output")
            model_a.pipeline.set_progress_bar_config(disable=True)
            with th.no_grad():
                with TraceDict([mod_a], {mod_a: capture_hook}):
                    model_a.pipeline(
                        prompts_list, num_inference_steps=1, **pipeline_kwargs
                    )
            if capture_hook.last is not None:
                captured.append(capture_hook.last.detach().cpu())
            if not captured:
                raise RuntimeError("No activations captured from model_a.")
            acts = th.cat(captured, dim=0)  # (N, ...)
            return acts.reshape(acts.shape[0], -1).mean(dim=0)  # (flat_dim,)

        print("[steer_contrast] Capturing positive activations from model_a...")
        mean_pos = _capture_acts(pos_prompts).to(device=device, dtype=th.float32)
        print("[steer_contrast] Capturing negative activations from model_a...")
        mean_neg = _capture_acts(neg_prompts).to(device=device, dtype=th.float32)

        # ── Step 2: steering direction in model_a's space ─────────────────────
        v = mean_pos - mean_neg
        v = v / (v.norm(p=2) + 1e-12)  # L2-normalise

        # ── Step 3: transfer direction through mapper, scale by alpha ─────────
        # delta_b = alpha * mapper(v): map the unit steering direction into
        # model_b's space and scale. At inference the hook adds this to
        # model_b's natural activation for the apply prompt.
        with th.no_grad():
            delta_b = (alpha * mapper(v.unsqueeze(0))).squeeze(0)  # (flat_dim_b,)

        print(f"[steer_contrast] Steering direction norm (model_a): {v.norm():.4f}")
        print(f"[steer_contrast] Transferred delta norm (model_b):  {delta_b.norm():.4f}")

        # ── Step 5: inject delta_b into model_b during generation ─────────────
        mod_b = model_b.resolve_accessor(module_b).module
        _inject_steps = inject_steps if inject_steps is not None else list(range(num_inference_steps))
        pipe_b = model_b.pipeline
        pipe_b.set_progress_bar_config(disable=True)

        def _make_inject_hook():
            step_counter = [0]

            def _policy(act_b: th.Tensor, **_) -> th.Tensor:
                step = step_counter[0]
                step_counter[0] += 1
                if step not in _inject_steps:
                    return act_b
                d = delta_b.to(device=act_b.device, dtype=act_b.dtype)
                B = act_b.shape[0]
                d_expanded = d.reshape(act_b.shape[1:]).unsqueeze(0).expand(B, *act_b.shape[1:])
                return act_b + d_expanded

            return UNetAlterHook(policy=_policy)

        with th.no_grad():
            with TraceDict([mod_b], {mod_b: _make_inject_hook()}):
                steered_imgs = pipe_b(
                    apply_prompts,
                    num_inference_steps=num_inference_steps,
                    **pipeline_kwargs,
                ).images

        if not also_generate_baseline:
            return steered_imgs

        # Baseline (no injection)
        with th.no_grad():
            baseline_imgs = pipe_b(
                apply_prompts,
                num_inference_steps=num_inference_steps,
                **pipeline_kwargs,
            ).images

        pairs = []
        for s, b in zip(steered_imgs, baseline_imgs):
            pairs.extend([s, b])
        return pairs

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
