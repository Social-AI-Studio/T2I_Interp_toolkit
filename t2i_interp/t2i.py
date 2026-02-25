from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from loguru import logger

from t2i_interp.accessors.accessor import IOType, ModuleAccessor, ModelWrapper
from t2i_interp.utils.T2I.hook import CaptureHook
from t2i_interp.utils.trace import TraceDict
from pathlib import Path
import os

CONFIG_PATH = Path(__file__).parent / "config" / "modules_to_pick.yaml"


def _parse_dtype(x):
    """
    Parses a logical input (string, type, tensor) into a torch.dtype.

    Args:
        x: Input to parse. Can be None, torch.dtype, str, or object with .dtype.

    Returns:
        torch.dtype or None: The parsed data type.
    """
    _DTYPE_ALIASES = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "single": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    if x is None or isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        return _DTYPE_ALIASES.get(x.lower())
    return x


def _parse_device(x):
    """
    Parses a logical input (string, int, tensor) into a torch.device.

    Args:
        x: Input to parse. Can be None, torch.device, tensor, int (cuda index), or str.

    Returns:
        torch.device or None: The parsed device.
    """
    if x is None:
        return None
    if isinstance(x, torch.device):
        return x
    if torch.is_tensor(x):
        return x.device
    if isinstance(x, int):
        return torch.device(f"cuda:{x}")
    if isinstance(x, str):
        if x.lower() == "auto":  # people sometimes pass device_map through here by mistake
            return None
        try:
            return torch.device(x)  # "cuda:0", "cpu", "cuda"
        except Exception:
            return None
    return None


class T2IModel:
    """
    A wrapper around Diffusers pipelines to support interpretation hooks.
    """
    def __init__(
        self,
        model: str,
        automodal: DiffusionPipeline = StableDiffusionImg2ImgPipeline,
        device: str | dict[str, int] | None = "cpu",
        dtype: torch.dtype | None = torch.float16,
        trust_remote_code: bool = False,
        safety_checker=None,
        **kwargs,
    ):
        """
        Initializes the T2IModel.
        """
        # Check if attention implementation is supported for attention pattern tracing
        if "attn_implementation" in kwargs:
            impl = kwargs.pop("attn_implementation", None)
            if impl != "eager":
                logger.warning(
                    f"Attention implementation {impl} is not supported using standard hooks if specific attention pattern access is needed. "
                    "For full compatibility, consider using eager."
                )
        else:
             impl = "eager"

        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        
        self.device = _parse_device(device) or torch.device("cpu")
        self.dtype = _parse_dtype(dtype) or torch.float16

        pipeline_kwargs = {
             "use_safetensors": True,
             "trust_remote_code": trust_remote_code,
             "safety_checker": safety_checker,
        }
        if "token" in kwargs:
             pipeline_kwargs["token"] = kwargs.pop("token")

        logger.info(f"Loading model {model}...")
        try:
            # Try loading with the provided class
            if isinstance(automodal, type) or (hasattr(automodal, "from_pretrained")):
                 self.pipeline = automodal.from_pretrained(
                    model,
                    torch_dtype=self.dtype,
                    **pipeline_kwargs,
                    **kwargs
                 )
            else:
                 # If automodal is already an instance? (Rare but possible)
                 self.pipeline = automodal
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        if self.device:
            self.pipeline.to(self.device)

        self.map_properties()

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.dtype):
            self.pipeline.to(dtype=device_or_dtype)
            self.dtype = device_or_dtype
        else:
            self.pipeline.to(device=device_or_dtype)
            self.device = torch.device(device_or_dtype)
        return self

    def eval(self):
        """Sets the model to evaluation mode."""
        # Config-based pipelines might not have components dict populated the same way
        # But Diffusers usually exposes .components
        if hasattr(self.pipeline, "components"):
            for _, component in self.pipeline.components.items():
                if isinstance(component, nn.Module):
                    component.eval()
        return self

    def train(self, mode: bool = True):
        """Sets the model to training mode."""
        if hasattr(self.pipeline, "components"):
            for _, component in self.pipeline.components.items():
                if isinstance(component, nn.Module):
                    component.train(mode)
        return self

    def parameters(self, recurse: bool = True):
        """Returns an iterator over module parameters."""
        if hasattr(self.pipeline, "components"):
            for _, component in self.pipeline.components.items():
                if isinstance(component, nn.Module):
                    yield from component.parameters(recurse=recurse)

    def map_properties(self):
        """
        Inspects the underlying pipeline components and maps them to `ModelWrapper`.
        """
        self._wrappers = {}
        
        # Identify components
        if hasattr(self.pipeline, "components"):
            comps = self.pipeline.components
        elif hasattr(self.pipeline, "config"):
             # Fallback: try to guess from config keys that map to attributes
             comps = {k: getattr(self.pipeline, k) for k in self.pipeline.config.keys() if hasattr(self.pipeline, k)}
        else:
            comps = {}

        for k, v in comps.items():
            # We only wrap nn.Module components (e.g., unet, text_encoder)
            if isinstance(v, nn.Module):
                wrapper = ModelWrapper(v, config_path=CONFIG_PATH)
                setattr(self, k, wrapper)
                self._wrappers[k] = wrapper

    @contextlib.contextmanager
    def edit(self, hooks: dict[nn.Module, Any] | None = None):
        """
        Context manager for applying temporary hooks (e.g. for SAEs or interventions).
        
        Args:
            hooks: Optional initial dictionary of hooks.
        """
        if hooks is None:
            hooks = {}
            
        with TraceDict(list(hooks.keys()), hooks):
            yield self
    
    def run_with_cache(
        self, 
        prompt, 
        accessors: list[ModuleAccessor] | None = None, 
        all_steps: bool = True,
        return_output: bool = False,
        reduce_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], Any]:
        """
        Runs the model generation with the given prompt and caches the activations of specified accessors.
        Can also apply custom hooks (e.g. for interventions).

        Args:
            prompt (str or list[str]): The input prompt(s) for generation.
            accessors (list[ModuleAccessor]): A list of accessors identifying the modules to cache.
            all_steps (bool): If True, returns the full history of activations (step-wise).
            return_output (bool): If True, returns the pipeline output (e.g. images) along with activations.
            reduce_fn (Callable): Optional function to reduce/process captured tensors (e.g. slice for CFG).
            **kwargs: Additional arguments for generation.

        Returns:
            dict or tuple: 
                - If return_output is False: A dictionary mapping accessor attribute names to their cached values.
                - If return_output is True: (saved_values, pipeline_output)
        """
        saved_values = {}
        hooks = {}
        
        # We also need to keep track of which hook corresponds to which accessor to retrieve values later
        accessor_hook_map = {}

        if accessors:
            for acc in accessors:
                if acc.io_type == IOType.INPUT:
                    hooks[acc.module] = CaptureHook(capture="input", reduce_fn=reduce_fn)
                else:
                    hooks[acc.module] = CaptureHook(capture="output", reduce_fn=reduce_fn)
                
                accessor_hook_map[acc.attr_name] = hooks[acc.module]

        with torch.no_grad():
            with TraceDict(list([acc.module for acc in accessors]), hooks):
                output = self.pipeline(prompt, **kwargs)

        for name, hook in accessor_hook_map.items():
            # Retrieve value. 
            # If all_steps is True, we try to get 'cache' (dict of step -> tensor).
            # If False, we get 'last'.
            
            val = None
            if all_steps:
                if hasattr(hook, "cache"):
                    val = hook.cache
            else:
                if hasattr(hook, "last"):
                    val = hook.last
            
            # Detach/cpu if needed
            # If val is a dict (cache), we iterate
            if val is not None:
                if isinstance(val, dict):
                     # it's a step cache
                     safe_val = {}
                     for step, v in val.items():
                         if hasattr(v, "detach"):
                             safe_val[step] = v.detach().cpu()
                         else:
                             safe_val[step] = v
                     saved_values[name] = safe_val
                else:
                    if hasattr(val, "detach"):
                        saved_values[name] = val.detach().cpu()
                    else:
                        saved_values[name] = val
        
        if return_output:
            return saved_values, output
        return saved_values

    def resolve_accessor(self, path: str, io_type: IOType = IOType.OUTPUT) -> ModuleAccessor:
        """
        Resolves a dot-notation path (e.g. 'unet.down_blocks.0') to a ModuleAccessor.
        """
        if "." in path:
            comp_name, subpath = path.split(".", 1)
        else:
            comp_name, subpath = path, ""
            
        component = getattr(self.pipeline, comp_name, None)
        if component is None:
            raise ValueError(f"Component '{comp_name}' not found in pipeline.")
            
        if not isinstance(component, nn.Module):
             # If it's not a module (e.g. scheduler?), we can't really hook it standardly
             # But let's try to proceed if it has modules?
             pass

        if subpath:
            # simple attribute lookup for the rest, utilizing torch's get_submodule if available
            # or manual traversal for things that might not be pure modules? 
            # Actually get_submodule is robust for nn.Module hierarchies.
            try:
                module = component.get_submodule(subpath)
            except AttributeError:
                 # Fallback for non-module components or manual traversal if needed
                 curr = component
                 for p in subpath.split("."):
                     if p.isdigit():
                         curr = curr[int(p)]
                     else:
                         curr = getattr(curr, p)
                 module = curr
        else:
            module = component
            
        return ModuleAccessor(module=module, attr_name=path, io_type=io_type)
