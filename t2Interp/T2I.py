from __future__ import annotations
from loguru import logger
import torch.nn as nn
from nnsight.modeling.diffusion import DiffusionModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict, List, Optional, Union
from clip_encoder import ClipEncoder
from flux_transformer2D_model import FluxTransformer
from t5_encoder import T5Encoder
from unet import Unet
from transformers import CLIPTextModel, T5EncoderModel
from diffusers.models import UNet2DConditionModel
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from utils.config_loader import build_module_mapper

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

CONFIG_MAP = [(_by_type(CLIPTextModel), "../config/CLIPEncoder.yaml"),
                 (_by_type(T5EncoderModel), "../config/T5EncoderModel.yaml"),
                 (_by_type(UNet2DConditionModel), "../config/unet.yaml"),
                (_by_type(FluxTransformer2DModel), "../config/FluxTransformer2DModel.yaml"),
                 ]

REGISTRY = []
REGISTRY.extend([(_by_type(CLIPTextModel), _from_comp_key("clip_encoder"), ClipEncoder),
                 (_by_type(T5EncoderModel), _from_comp_key("t5_encoder"), T5Encoder),
                 (_by_type(UNet2DConditionModel), _from_comp_key("unet"), Unet),
                (_by_type(FluxTransformer2DModel), _from_comp_key("flux_transformer"), FluxTransformer),
                 ])

class T2IModel(DiffusionModel):
    def __init__(
        self,
        model: str | nn.Module,
        trust_remote_code: bool = False,
        check_renaming: bool = True,
        allow_dispatch: bool = True,
        rename_config: Dict[str, str] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("device_map", "auto")
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
        # user_rename = kwargs.pop("rename", None)
        # # if user_rename is not None:
        # #     logger.info(
        # #         f"Updating default rename with user-provided rename: {user_rename}"
        # #     )
        # #     rename_config.update(user_rename)
        
        super().__init__(
            model,
            attn_implementation=impl,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename=rename_config,
            **kwargs,
        )
        
        # if isinstance(model, str):
        #     model_name = model
        # else:
        #     model_name = model.__class__.__name__
        
    def map_properties(self):
        comps = getattr(self.pipeline, "components", None)
        visited = set()

        def consider(comp_key, module):
            if module in visited:
                return
            visited.add(module)
            for pred, name_fn, ctor in REGISTRY:
                try:
                    if pred(module):
                        attr_name = name_fn(comp_key, module)
                        # Disambiguate duplicates by suffixing the attr_name when names collide
                        if hasattr(self, attr_name):
                            i = 2
                            while hasattr(self, f"{attr_name}_{i}"):
                                i += 1
                            attr_name = f"{attr_name}_{i}"
                        wrapper = ctor(module, name=attr_name)
                        setattr(self, attr_name, wrapper)
                        self._wrappers[attr_name] = wrapper
                        # if verbose:
                        #     print(f"[T2I] attached {attr_name}: {module.__class__.__name__}") # logger
                        return
                except Exception:
                    continue  # keep scanning other rules
                    
        # renaming modules for consistency
        if isinstance(comps, dict):
            rename_config={}
            for k, v in comps.items():
                # v can be nn.Module, tokenizer, scheduler, etc.
                is_mod = isinstance(v, nn.Module)
                if is_mod:
                    config_applicable = [cfg for pred, cfg  in CONFIG_MAP if pred(v)]
                    rename_config.update({build_module_mapper(self,cfg) for cfg in config_applicable})
            super().__init__(
                self._model,
                rename=rename_config,
            )        
        
        # creating properties dynamically based on what classes exist in the model            
        if isinstance(comps, dict):
            for k, v in comps.items():
                # v can be nn.Module, tokenizer, scheduler, etc.
                is_mod = isinstance(v, nn.Module)
                if is_mod:
                    consider(k, v)       
        del visited      
        # log

    def run_with_cache(self, prompt, modules: List[nn.Module]):
        with self.trace(prompt) as tracer:
            cache = tracer.cache(modules=modules).save()
        return cache    