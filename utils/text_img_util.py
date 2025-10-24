from typing import Callable, Optional, Union, Tuple, List, Any, Dict
import torch as t
from t2Interp.accessors import ModuleAccessor, IOType
from t2Interp.T2I import T2IModel
from utils.utils import reshape_like

Tensor = t.Tensor

def _to_dtype_device(x: Tensor, ref: Tensor) -> Tensor:
    # keep gradients, but match dtype/device for numerical safety
    if x.dtype != ref.dtype:
        x = x.to(ref.dtype)
    if x.device != ref.device:
        x = x.to(ref.device)
    return x

# def reshape_like(vec, x):
#     """
#     vec: tensor with total elements == x.numel()
#     x:   target tensor (e.g., (B, C, H, W))
#     """
#     v = vec.to(dtype=x.dtype, device=x.device)
#     if v.numel() != x.numel():
#         raise ValueError(f"vec.numel()={v.numel()} != x.numel()={x.numel()}")
#     # .reshape handles non-contiguous; .view requires contiguity
#     return v.reshape(x.shape)


class OutputAlterHook:
    """
    Modifies the **output** of a module. Works with:
    - plain Tensor outputs
    - tuple/list of tensors (modifies the first tensor)
    - objects with `.sample` (e.g., diffusers' UNet2DConditionOutput)
    Supports optional step gating with an external call_counter.
    If `guidance=True`, applies policy only to the second half of batch (CFG cond branch).
    """
    def __init__(
        self,
        policy: Callable[[Tensor, t.nn.Module], Tensor],
        call_counter: Optional[dict] = None,
        step_index: Optional[int] = None,     # apply only on the Nth call (0-index)
        device: Optional[str] = None,
        guidance: bool = True,               # NEW: apply only to latter half if CFG duplication is used
    ):
        self.policy = policy
        self.call_counter = call_counter if call_counter is not None else {"n": 0}
        self.step_index = step_index
        self.device = device
        self.guidance = guidance
        self._handle = None

    def _take_it(self) -> bool:
        n = self.call_counter["n"]
        self.call_counter["n"] += 1
        return (self.step_index is None) or (n == self.step_index)

    def _apply(self, x: Tensor, module: t.nn.Module) -> Tensor:
        """Apply policy either to entire batch or only to the CFG conditional half."""
        if self.guidance and x.dim() >= 1 and x.size(0) % 2 == 0 and x.size(0) > 1:
            B2 = x.size(0)
            B = B2 // 2
            uncond = x[:B]
            cond   = x[B:]
            cond_new = self.policy(cond, module)
            out = t.cat([uncond, cond_new], dim=0)
        else:
            out = self.policy(x, module)

        if self.device is not None and out.device != t.device(self.device):
            out = out.to(self.device)
        return out

    def hook(self, module: t.nn.Module, inputs, output):
        if not self._take_it():
            return None  # no modification
        out = output

        # Case A: output is a Tensor
        if isinstance(out, t.Tensor):
            return self._apply(out, module)

        # Case B: output is a (tuple/list); alter first tensor-like and rebuild the container
        if isinstance(out, (tuple, list)) and len(out) > 0:
            head = out[0]
            if isinstance(head, t.Tensor):
                new_head = self._apply(head, module)
                if isinstance(out, tuple):
                    return (new_head,) + tuple(out[1:])
                else:
                    out = list(out)
                    out[0] = new_head
                    return out
            return None

        # # Case C: diffusers object with `.sample` tensor (e.g., UNet2DConditionOutput)
        # if hasattr(out, "sample") and isinstance(out.sample, t.Tensor):
        #     new_sample = self._apply(out.sample, module)
        #     # Recreate same type, preserving other attrs
        #     return type(out)(sample=new_sample, **{k: v for k, v in out.__dict__.items() if k != "sample"})

        # Unknown output type
        return None

    def register(self, module: t.nn.Module):
        self._handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

class InputAlterHook:
    """
    Modifies the **first positional tensor input** of a module.
    If your module's meaningful input isn't the first tensor, adapt the selection logic.
    """
    def __init__(
        self,
        policy: Callable[[Tensor, t.nn.Module], Tensor],
        call_counter: Optional[dict] = None,
        step_index: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.policy = policy
        self.call_counter = call_counter if call_counter is not None else {"n": 0}
        self.step_index = step_index
        self.device = device
        self._handle = None

    def _take_it(self) -> bool:
        n = self.call_counter["n"]
        self.call_counter["n"] += 1
        return (self.step_index is None) or (n == self.step_index)

    def hook(self, module: t.nn.Module, inputs):
        if not self._take_it():
            return None
        if not isinstance(inputs, tuple) or len(inputs) == 0:
            return None

        # Find first tensor-like positional arg
        idx = None
        for i, item in enumerate(inputs):
            if isinstance(item, t.Tensor):
                idx = i
                break
            if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], t.Tensor):
                idx = i
                break
        if idx is None:
            return None

        # Prepare a new tuple with modified element
        new_inputs = list(inputs)
        target = new_inputs[idx]
        if isinstance(target, t.Tensor):
            new_t = self.policy(target, module)
        elif isinstance(target, (tuple, list)) and len(target) > 0 and isinstance(target[0], t.Tensor):
            # modify only the first tensor inside this nested structure
            head = target[0]
            new_head = self.policy(head, module)
            target = (new_head,) + tuple(target[1:]) if isinstance(target, tuple) else [new_head] + list(target[1:])
            new_t = target
        else:
            return None

        if self.device is not None and isinstance(new_t, t.Tensor):
            new_t = new_t.to(self.device)

        new_inputs[idx] = new_t
        return tuple(new_inputs)

    def register(self, module: t.nn.Module):
        self._handle = module.register_forward_pre_hook(self.hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

def scale_policy(factor: Union[float, Tensor]) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    factor: float or tensor. If tensor, it will be broadcast to x.
    """
    if not t.is_tensor(factor):
        factor = t.tensor(factor)

    def _f(x: Tensor, module: t.nn.Module) -> Tensor:
        f = reshape_like(factor, x)
        return x * f
    return _f

def add_vector_policy(vec: Tensor, *, channel_dim: Optional[int] = None) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    vec: tensor to add; can be 1D (e.g., channels) or any shape broadcastable to x.
    channel_dim: set if you want to force which dimension the 1D vec maps to.
    """
    def _f(x: Tensor, module: t.nn.Module) -> Tensor:
        v = reshape_like(vec, x)
        return x + v
    return _f

def replace_policy(
    value: Union[Tensor, Callable[[Tensor, t.nn.Module], Tensor]]
) -> Callable[[Tensor, t.nn.Module], Tensor]:
    """
    value: 
      - Tensor: used directly (broadcast to x if needed)
      - Callable: called as value(x, module) -> Tensor
    """
    def _f(x: Tensor, module: t.nn.Module) -> Tensor:
        new = value(x, module) if callable(value) else value
        new = reshape_like(new, x)
        return new
    return _f

class BaseCaptureHook:
    """
    Base class with step gating & reduce.
    Stores the most recent captured tensor in self.last (on chosen device).
    """
    def __init__(
        self,
        *,
        call_counter: Optional[dict] = None,
        denoiser_steps: Optional[list[int]] = [0],  # 0-based call index; None = last (handled by caller by running once per step) or "any"
        device: Optional[Union[str, t.device]] = None,
        reduce_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.call_counter = call_counter if call_counter is not None else {"n": 0}
        self.denoiser_steps = denoiser_steps
        self.device = t.device(device) if device is not None else None
        self.reduce_fn = reduce_fn
        self.last: Optional[Tensor] = None

    def _gate(self) -> bool:
        n = self.call_counter["n"]
        self.call_counter["n"] += 1
        return (self.denoiser_steps is None) or (n in self.denoiser_steps)

    def _post(self, x: Tensor) -> None:
        if self.reduce_fn is not None:
            x = self.reduce_fn(x)
        if self.device is not None:
            x = x.to(self.device)
        self.last = x


class CaptureOutputHook(BaseCaptureHook):
    """
    Captures the (tensor) output of a module.
    Handles: Tensor, (Tensor, ...), and objects with `.sample`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._handle = None

    def _extract(self, output: Any) -> Optional[Tensor]:
        val = output
        if isinstance(val, (tuple, list)) and len(val) > 0:
            val = val[0]
        if isinstance(val, Tensor):
            return val
        if hasattr(output, "sample") and isinstance(output.sample, Tensor):
            return output.sample
        return None

    def hook(self, module: t.nn.Module, inputs, output):
        if not self._gate():
            return None
        tensor = self._extract(output)
        if tensor is not None:
            self._post(tensor)
        return None  # capture-only

    def register(self, module: t.nn.Module):
        self._handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class CaptureInputHook(BaseCaptureHook):
    """
    Captures the first tensor-like positional input of a module.
    (You can customize the selection logic if your target arg differs.)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._handle = None

    def _pick(self, inputs: Any) -> Optional[Tensor]:
        if not isinstance(inputs, tuple) or len(inputs) == 0:
            return None
        val = inputs[0]
        if isinstance(val, (tuple, list)) and len(val) > 0:
            val = val[0]
        if isinstance(val, Tensor):
            return val
        return None

    def hook(self, module: t.nn.Module, inputs):
        if not self._gate():
            return None
        tensor = self._pick(inputs)
        if tensor is not None:
            self._post(tensor)
        return None  # capture-only

    def register(self, module: t.nn.Module):
        self._handle = module.register_forward_pre_hook(self.hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None



def _prep_prompts_images(
        batch: Union[List[Any], Dict[str, Any]]
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

def flatten_batch(acts: Tensor, device: Optional[Union[str, t.device]] = None) -> Tensor:
    """
    (B, ...) -> (B, -1)
    """
    if isinstance(acts, tuple):
        acts = acts[0]
    if not isinstance(acts, Tensor):
        raise TypeError(f"Captured value is not a Tensor: {type(acts)}")
    if acts.dim() == 0:
        acts = acts.unsqueeze(0)
    B = acts.shape[0] if acts.dim() >= 1 else 1
    if device is not None:
        acts = acts.to(device)
    return acts.view(B, -1)

def _register(module: t.nn.Module, io_type:IOType, fn: Callable):
    if io_type == IOType.INPUT:
        return module.register_forward_pre_hook(fn)
    else:
        return module.register_forward_hook(fn)

def run_with_hook(
    model:T2IModel,
    batch,
    module: t.nn.Module,
    hook_obj: Any, 
    io_type: IOType,
    **pipe_kwargs,
    ) -> Tuple[Any, Any]:
        """
        Register hook_obj.hook on `module`, run `runner()`, then remove the hook.
        Returns (hook_obj, result).
        """
        handle = _register(module, io_type, hook_obj.hook)
        try:
            io = _prep_prompts_images(batch)
            # If all prompts empty, avoid CFG pulling toward text by mistake
            if "prompt" in io and isinstance(io["prompt"], list) and all(p == "" for p in io["prompt"]):
                pipe_kwargs.setdefault("guidance_scale", 1.0)
            result = model.pipeline(**io, **pipe_kwargs)
        finally:
            handle.remove()
        return result
