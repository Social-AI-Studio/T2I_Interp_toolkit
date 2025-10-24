import torch as t
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from utils.buffer import t2IActivationBuffer
from utils.text_img_util import  _prep_prompts_images, CaptureInputHook, CaptureOutputHook, flatten_batch, run_with_hook
from t2Interp.T2I import T2IModel
from t2Interp.accessors import ModuleAccessor, IOType

Tensor = t.Tensor
# ---- Your class imports assumed available ----
# from t2Interp.T2I import T2IModel
# from dictionary_learning.buffer import NNsightActivationBuffer

class TextImageActivationBuffer(t2IActivationBuffer):
    """
    Activation buffer for Diffusers pipelines using plain PyTorch hooks (no NNsight).
    - Accepts batches that may contain prompts, images, or both.
    - Captures activations from a specified submodule via forward/pre hook.
    - Supports picking a specific UNet call index (denoising step) to capture.
    """

    def __init__(
        self,
        data,                              # iterator / generator yielding samples
        model: T2IModel,
        submodule: ModuleAccessor,
        d_submodule: Optional[int] = None,
        n_ctxs: int = int(3e4),
        refresh_batch_size: int = 512,
        out_batch_size: int = 512,
        data_device: str = "cpu",
        denoiser_steps: list[int] =[0],     # 0-based call index; None = capture on any (first/last managed by policy)
        reduce_fn: Optional[Callable[[Tensor], Tensor]] = None,
        **kwargs: Optional[Dict[str, Any]],  # default kwargs passed to pipeline(...)
    ):
        super().__init__(
            data=data, model=model, submodule=submodule, d_submodule=d_submodule,
            n_ctxs=n_ctxs, refresh_batch_size=refresh_batch_size,
            out_batch_size=out_batch_size, data_device=data_device
        )
        self.capture = submodule.io_type
        self.denoiser_steps= denoiser_steps
        self.reduce_fn = reduce_fn
        self.pipeline_kwargs = kwargs if kwargs is not None else {}

        # Resolve the target module to hook (string path or module instance)
        self._target_module = submodule.module

        # buffer storage
        self.activations: Tensor = t.empty(0, device=self.device)
        self.read = t.zeros(0, dtype=t.bool, device=self.device)

    def token_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.refresh_batch_size
        items = []
        for _ in range(batch_size):
            try:
                items.append(next(self.data))
            except StopIteration:
                break
        if not items:
            raise StopIteration("End of data stream reached")
        return items

    def refresh(self):
        """
        Run the Diffusers pipeline with hooks, capture the chosen submodule's activation
        at the requested UNet call (denoising step), flatten per-sample, and append.
        """
        # keep only unread activations
        self.activations = self.activations[~self.read]
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

        while len(self.activations) < self.refresh_batch_size:
            batch_raw = self.token_batch()
            io = _prep_prompts_images(batch_raw)

            pipe_kwargs = dict(self.pipeline_kwargs)

            # If all prompts empty, avoid CFG pulling toward text by mistake
            if "prompt" in io and isinstance(io["prompt"], list) and all(p == "" for p in io["prompt"]):
                pipe_kwargs.setdefault("guidance_scale", 1.0)

            # build the appropriate capture hook
            counter = {"n": 0}
            common = dict(
                call_counter=counter,
                denoiser_steps=self.denoiser_steps,
                device=self.device,
                reduce_fn=self.reduce_fn,
            )
            
            cap = CaptureOutputHook(**common) if self.capture == IOType.OUTPUT else CaptureInputHook(**common)

            # run pipeline once with the hook; we don't use the return value here
            _ = run_with_hook(
                model=self.model,
                batch=batch_raw,
                module=self._target_module,
                hook_obj=cap,
                io_type=self.capture,            # IOType.OUTPUT or IOType.INPUT
                **self.pipeline_kwargs,          # forwarded as-is
            )

            if cap.last is None:
                # Probably wrong submodule or step_index out of range for this submodule in this call.
                # Skip this mini-batch and continue.
                continue

            acts = flatten_batch(cap.last, device=self.device)
            if self.activations.numel() == 0:
                self.activations = acts
            else:
                self.activations = t.cat([self.activations, acts], dim=0)

            # reset read mask
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

