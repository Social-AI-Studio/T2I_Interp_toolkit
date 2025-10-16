import torch as t
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from utils.buffer import t2IActivationBuffer
from t2Interp.T2I import T2IModel
from t2Interp.accessors import ModuleAccessor, IOType

# ---- Your class imports assumed available ----
# from t2Interp.T2I import T2IModel
# from dictionary_learning.buffer import NNsightActivationBuffer

class TextImageActivationBuffer(t2IActivationBuffer):
    """
    Activation buffer for Diffusers pipelines using plain PyTorch hooks (no NNsight).
    - Accepts batches that may contain prompts, images, or both.
    - Captures activations from a specified submodule via forward hook.
    - Supports picking a specific UNet call index (denoising step) to capture.
    """

    def __init__(
        self,
        data,                              # iterator / generator yielding samples (see token_batch)
        model: T2IModel,
        submodule: ModuleAccessor,  
        d_submodule: Optional[int] = None,
        n_ctxs: int = int(3e4),
        refresh_batch_size: int = 512,
        out_batch_size: int = 512,
        data_device: str = "cpu",
        step_index: Optional[int] = 0,  # which UNet call (0-based) to capture per pipeline run; None = last call
        reduce_fn: Optional[Callable[[t.Tensor], t.Tensor]] = None,  # optional transform on captured tensor (per batch)
        **kwargs: Optional[Dict[str, Any]] ,  # default kwargs passed to pipeline(...)
    ):
        super().__init__(
            data=data, model=model, submodule=submodule, d_submodule=d_submodule,
            n_ctxs=n_ctxs, refresh_batch_size=refresh_batch_size,
            out_batch_size=out_batch_size, data_device=data_device
        )
        self.capture = submodule.io_type 
        self.step_index = step_index
        self.reduce_fn = reduce_fn
        self.pipeline_kwargs = kwargs if kwargs is not None else {}

        # Resolve the target module to hook (string path or module instance)
        self._target_module = submodule.module

        # Internal scratch
        self._last_captured: Optional[t.Tensor] = None

    def _flatten_batch_acts(self, tensor: t.Tensor) -> t.Tensor:
        """(B, ...) -> (B, -1) on self.device"""
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if not isinstance(tensor, t.Tensor):
            raise TypeError(f"Captured value is not a Tensor: {type(tensor)}")
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        B = tensor.shape[0] if tensor.dim() >= 1 else 1
        tensor = tensor.view(B, -1).to(self.device)
        return tensor

    def _capture_hook(self, call_counter: dict):
        """kind: 'output' for forward_hook, 'input' for forward_pre_hook"""
        kind = self.capture

        if kind == IOType.OUTPUT:
            # forward hook: (module, inputs, output)
            def hook(module, input, output):
                call_counter["n"] += 1
                take_it = (
                    True if self.step_index is None
                    else (call_counter["n"] - 1) == self.step_index
                )
                if not take_it:
                    return
                val = output
                if isinstance(val, (tuple, list)):
                    val = val[0]
                if isinstance(val, t.Tensor):
                    self._last_captured = val.detach().to(self.device)
            return hook

        else:
            # forward pre-hook: (module, inputs)
            def hook(module, inputs):
                call_counter["n"] += 1
                take_it = (
                    True if self.step_index is None
                    else (call_counter["n"] - 1) == self.step_index
                )
                if not take_it:
                    return
                # inputs is a tuple of positional args; pick the first tensor-like
                val = None
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    val = inputs[0]
                    if isinstance(val, (tuple, list)):
                        val = val[0]
                if isinstance(val, t.Tensor):
                    self._last_captured = val.detach().to(self.device)
            return hook


    def _prep_prompts_images(
        self, batch: Union[List[Any], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Accepts:
          - list of mixed samples (str prompts and/or images tensors/PIL) -> split into dict
          - dict with keys 'prompt' and/or 'image' -> used directly
        Ensures at least an empty prompt list if only images are provided.
        """
        # Case 1: batch is a dict already
        if isinstance(batch, dict):
            prompts = batch.get("prompt", None)
            images = batch.get("image", None)

            # Normalize prompts
            if prompts is None:
                if images is not None:
                    bs = len(images) if hasattr(images, "__len__") else 1
                    prompts = [""] * bs
            elif isinstance(prompts, str):
                prompts = [prompts]

            out = {}
            if prompts is not None:
                out["prompt"] = prompts
            if images is not None:
                out["image"] = images
            return out

        # Case 2: batch is a list of mixed entries
        prompts: List[str] = []
        images: List[Any] = []
        for x in batch:
            if isinstance(x, str):
                prompts.append(x)
            else:
                images.append(x)

        if not prompts and images:
            prompts = [""] * len(images)

        out: Dict[str, Any] = {}
        if prompts:
            out["prompt"] = prompts
        if images:
            out["image"] = images
        return out

    # ---------- main overrides ----------

    def token_batch(self, batch_size=None):
        """
        You can keep your upstream data the same (strings/tensors/PIL or a dict).
        This override just returns the raw list (or dict), to be normalized in refresh().
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            # Preserve original behavior: gather a list of `batch_size` samples
            return [next(self.data) for _ in range(batch_size)]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def refresh(self):
        """
        Run the Diffusers pipeline with hooks, capture the chosen submodule's activation
        at the requested UNet call (denoising step), flatten per-sample, and append.
        """
        # Keep only unread activations
        self.activations = self.activations[~self.read]
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
        
        while len(self.activations) < self.refresh_batch_size:
            batch_raw = self.token_batch()
            io = self._prep_prompts_images(batch_raw)

            # Diffusers broadcasting rules handle single prompt or single image
            # (image can be PIL or CHW/BCHW float32 on CPU in [0,1])
            pipe_kwargs = dict(self.pipeline_kwargs)

            # Ensure guidance_scale sane if you truly want minimal text pull w/ empty prompts
            if "prompt" in io and isinstance(io["prompt"], list) and all(p == "" for p in io["prompt"]):
                pipe_kwargs.setdefault("guidance_scale", 1.0)

            # Register hook
            call_counter = {"n": 0}
            handle = (
                self._target_module.register_forward_hook(self._capture_hook(call_counter))
                if self.capture == IOType.OUTPUT
                else self._target_module.register_forward_pre_hook(self._capture_hook(call_counter))  # type: ignore
            )

            try:
                with t.no_grad():
                    # Call the pipeline; we don't need returned images—just trigger the forward
                    _ = self.model.pipeline(**io, **pipe_kwargs)
            finally:
                handle.remove()

            if self._last_captured is None:
                # Could happen if step_index is out-of-range for this submodule (too high),
                # or the hook never fired (wrong submodule). Skip this mini-batch.
                continue

            acts = self._last_captured
            self._last_captured = None  # reset for next loop

            if self.reduce_fn is not None:
                acts = self.reduce_fn(acts)

            acts = self._flatten_batch_acts(acts)
            self.activations = t.cat([self.activations, acts], dim=0)

            # refresh read mask to match new activations length
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
