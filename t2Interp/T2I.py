from __future__ import annotations
from loguru import logger
import torch.nn as nn
import torch
from nnsight.modeling.diffusion import DiffusionModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict, List, Optional, Union
from t2Interp.clip_encoder import ClipEncoder
from t2Interp.flux_transformer2D_model import FluxTransformer
from t2Interp.t5_encoder import T5Encoder
from t2Interp.unet import Unet
from t2Interp.accessors import ModuleAccessor, IOType
from transformers import CLIPTextModel, T5EncoderModel
from diffusers.models import UNet2DConditionModel
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from utils.config_loader import build_module_mapper
import warnings
from utils.utils import FunctionModule
from diffusers import DiffusionPipeline


def _by_type(type_):
    def pred(m): 
        try:
            return isinstance(m, type_)
        except Exception:
            return False
    return pred

# def _by_name(s):
#     s = s.lower()
#     return lambda m: s == m.__class__.__name__.lower()

# def _fixed(name):
#     return lambda comp_key, module: name

def _from_comp_key(default_name):
    return lambda comp_key, module: default_name if comp_key is None else comp_key

# CONFIG_MAP = {
#     ["stable-diffusion"] : "../config/unet.yaml",
#     ["stable-diffusion","flux","pixart"] : "../config/CLIPEncoder.yaml",
#     ["flux"] : "../config/T5EncoderModel.yaml",    
#     ["flux"] : "../config/FluxTransformer2DModel.yaml"    
#     }

CONFIG_MAP = [(_by_type(CLIPTextModel), "./config/CLIPEncoder.yaml"),
                 (_by_type(T5EncoderModel), "./config/T5EncoderModel.yaml"),
                 (_by_type(UNet2DConditionModel), "./config/unet.yaml"),
                (_by_type(FluxTransformer2DModel), "./config/FluxTransformer2DModel.yaml"),
                 ]

REGISTRY = []
REGISTRY.extend([(_by_type(CLIPTextModel), _from_comp_key("clip_encoder"), ClipEncoder),
                 (_by_type(T5EncoderModel), _from_comp_key("t5_encoder"), T5Encoder),
                 (_by_type(UNet2DConditionModel), _from_comp_key("unet"), Unet),
                (_by_type(FluxTransformer2DModel), _from_comp_key("flux_transformer"), FluxTransformer),
                 ])

def _parse_dtype(x):
    _DTYPE_ALIASES = {
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32, "single": torch.float32,
        "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    }
    if x is None or isinstance(x, torch.dtype): return x
    if isinstance(x, str): return _DTYPE_ALIASES.get(x.lower())
    return x

def _parse_device(x):
    if x is None: return None
    if isinstance(x, torch.device): return x
    if torch.is_tensor(x): return x.device
    if isinstance(x, int): return torch.device(f"cuda:{x}")
    if isinstance(x, str):
        if x.lower() == "auto":   # people sometimes pass device_map through here by mistake
            return None
        try:
            return torch.device(x)   # "cuda:0", "cpu", "cuda"
        except Exception:
            return None
    return None

class T2IModel(DiffusionModel):
    __methods__ = {"generate": "_generate"}
    def __init__(
        self,
        model: str | DiffusionPipeline,
        device: Optional[Union[str, Dict[str, int]]] = "cpu",
        dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = False,
        check_renaming: bool = True,
        allow_dispatch: bool = True,
        rename_config: Dict[str, str] | None = None,
        **kwargs,
    ):
        self._model: Diffuser = None
        # Check if attention implementation is supported for attention pattern tracing
        if "attn_implementation" in kwargs:
            impl = kwargs.pop("attn_implementation", None)
            if impl != "eager":
                logger.warning(
                    f"Attention implementation {impl} is not supported for attention pattern tracing. Please use eager attention implementation if you plan to access attention patterns."
                )
        else:
            logger.info(
                "Enforcing eager attention implementation for attention pattern tracing. The HF default would be to use sdpa if available. To use sdpa, set attn_implementation='sdpa' or None to use the HF default."
            )
            impl = "eager"
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        if type(model) == str:
            super().__init__(
                model,
                dispatch=True,
                attn_implementation=impl,
                tokenizer_kwargs=tokenizer_kwargs,
                trust_remote_code=trust_remote_code,
                rename=rename_config,
                **kwargs,
            )
        else:
            raise NotImplementedError("Only model path is supported for T2IModel")
        
        device= _parse_device(device)
        dtype = _parse_dtype(dtype)

        if device:
            self.to(device)
        if dtype:
            self.to(dtype)                

        self.map_properties()
       
    def map_properties(self):
        comps = getattr(self.pipeline, "components", None)
        visited = set()
        rename_config={}
        self._wrappers = {}
        def consider(comp_key, module):
            if module in visited:
                return
            visited.add(module)
            for pred, name_fn, ctor in REGISTRY:
                if pred(module):
                    attr_name = name_fn(comp_key, module)
                    # Disambiguate duplicates by suffixing the attr_name when names collide
                    if hasattr(self, attr_name):
                        wrapper = ctor(getattr(self, attr_name)) # wrap existing
                        i = 2
                        while hasattr(self, f"{attr_name}_{i}"):
                            i += 1
                        attr_name = f"{attr_name}_{i}"
                    else:
                        # wrapper = ctor(module)    
                        # log skip
                        continue
                    setattr(self, attr_name, wrapper)
                    self._wrappers[attr_name] = wrapper
        
        # creating properties dynamically based on what classes exist in the model            
        if isinstance(comps, dict):
            for k, v in comps.items():
                # v can be nn.Module, tokenizer, scheduler, etc.
                is_mod = isinstance(v, nn.Module)
                if is_mod:
                    consider(k, v)       
        del visited      
    
    def run_with_cache(self, prompt, accessors: List[ModuleAccessor], **kwargs) -> List[ModuleAccessor, torch.Tensor]:
        modules = [m.module for m in accessors]
    
        num_inference_steps = kwargs.get("num_inference_steps", None)
        seed = kwargs.get("seed", None)
        if num_inference_steps is None or seed is None:
            raise ValueError("num_inference_steps and seed must be provided in kwargs")
        
        generate_kwargs = {}
        if "guidance_scale" in kwargs:
            generate_kwargs["guidance_scale"] = kwargs["guidance_scale"]        
        with self.generate(prompt, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs) as tracer:
            dtype,device,detach = kwargs.get("dtype", None), kwargs.get("device", "cpu"), kwargs.get("detach", True)
            cache = tracer.cache(modules=modules, include_output = True, include_inputs=False, dtype=dtype, detach=detach, device=device)

        # for mod in modules:
        #     if type(mod) == FunctionModule:
        #         mod.bound_kwargs.clear()          
        return {accessors[i].attr_name:v for i,(k,v) in enumerate(cache.items())}  