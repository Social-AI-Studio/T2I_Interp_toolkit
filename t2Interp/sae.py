import torch as t
from typing import List, Tuple, Any
import torch.nn as nn
from t2Interp.T2I import T2IModel
from t2Interp.accessors import ModuleAccessor
from t2Interp.blocks import SAEBlock
# from datasets import load_dataset
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from train_config import sae_trainer_config
from dictionary_learning.dictionary import Dictionary
from utils.buffer import t2IActivationBuffer
from utils.utils import FunctionModule
from t2Interp.accessors import IOType
# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import copy
from typing import Dict
from functools import partial
from t2Interp.accessors import ModuleAccessor
            
class SAEManager:
    def __init__(self, model):
        self.model: T2IModel = model
        self.registry_mod = {}        # dict: hook_name -> SAE module
        # self.handles = {}     # dict: hook_name -> hook handle

    def train(self, hf_dataset, module:ModuleAccessor, **kwargs):
        generator = hf_dataset_to_generator(hf_dataset)
        buffer = t2IActivationBuffer(generator, self.model, module, **kwargs)
        trainer_config = sae_trainer_config(**kwargs)
        
        save_dir=kwargs.pop("save_dir", None)
        if save_dir:
            save_dir=os.path.join(save_dir, module.attr_name.replace(".","_"))
            
        trainSAE(
            data=buffer,
            trainer_configs=trainer_config,
            save_dir=save_dir,
            **kwargs,
        )

    def clear(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
    
    def add_saes_to_model(self, sae_list:List[Tuple[ModuleAccessor,Dictionary,str]]):
        def process(x, accessor:ModuleAccessor):
            ae = accessor.ae
            kwargs = ae[0].bound_kwargs
            accessor_input = accessor.module.input
            diff = kwargs.get("diff",False)
            if diff:
                sae_in = x - accessor_input
                enc_out = ae[0](sae_in)
            else:
                enc_out = ae[0](x)    
                
            mask = kwargs.get("mask", False)
            error = kwargs.get("error", False)
            
            if mask is not None:
                masked_out = enc_out * mask.to(x.device,x.dtype)
            if error:    
                enc_out = t.cat([enc_out,masked_out], dim=0)
     
            dec_out = ae[1](enc_out)
            
            if diff:
                out = dec_out + accessor_input.expand((dec_out.shape[0], accessor_input.shape[1:]))
                
            if error:
                error = x - out[0]
                out = out[1] + error
            return out
        
        def sae_pre_hook(module, inputs, accessor:ModuleAccessor):
            (x,) = inputs
            out = process(x, accessor=accessor)
            return (out,)
            
        def sae_forward_hook(module, input, output, accessor:ModuleAccessor):
            if type(output) == tuple:
                (x,) = output
            else:
                x = output
            out = process(x, accessor=accessor) 
            if type(output) == tuple:
                return (out,)
            return out 
    
        for accessor, sae, parent_attr_name in sae_list:
            encode = FunctionModule(sae.encode)
            decode = FunctionModule(sae.decode)
            ae = nn.ModuleList([encode, decode])
            ae = SAEBlock(**{"encoder_in":ModuleAccessor(ae[0], parent_attr_name + "_encode_in", io_type=IOType.INPUT),
                            "encoder_out":ModuleAccessor(ae[0], parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
                            "decoder_in":ModuleAccessor(ae[1], parent_attr_name + "_decode_in", io_type=IOType.INPUT),
                            "decoder_out":ModuleAccessor(ae[1], parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)})
            accessor.ae = ae
            if accessor.io_type == IOType.OUTPUT:
                h = accessor.module.register_forward_hook(partial(sae_forward_hook,accessor=accessor))
            elif accessor.io_type == IOType.INPUT:
                h = accessor.module.register_forward_pre_hook(partial(sae_pre_hook,accessor=accessor))
            else:
                raise ValueError(f"IOType {accessor.io_type} not supported for SAE insertion")    
            self._handles = getattr(self, "_handles", [])
            self._handles.append(h)
    
    def add_encoder_hook(self, accessor:ModuleAccessor, cache:Dict[str,t.Tensor]):
        def cache_hook(module, input, output):
            if type(output) == tuple:
                (x,) = output
            else:
                x = output
            cache[accessor.attr_name] = x.detach().cpu()
        h = accessor.ae.module.register_forward_hook(cache_hook)
        self._handles = getattr(self, "_handles", [])
        self._handles.append(h)
        
    def run_with_cache(self, prompt, accessors: List[ModuleAccessor], **kwargs) -> List[Tuple[ModuleAccessor, dict]]:
        for accessor in accessors:
            if not hasattr(accessor, "ae"):
                raise ValueError(f"Accessor {accessor.attr_name} does not have an SAE attached")
        cache = {}
        for accessor in accessors:
            self.add_encoder_hook(accessor, cache)
        self.model.generate(prompt, **kwargs)
        return [(accessor, cache[accessor.attr_name]) for accessor in accessors]
    
    # def ScaleIntervention(self, accessor:ModuleAccessor, factor:float, **kwargs):
    #     pass     
    
    # def PatchIntervention(self, accessor:ModuleAccessor, patch_tensor:t.Tensor, **kwargs):
    #     pass
      
    # def add_saes_to_model(self, sae_list:List[Tuple[ModuleAccessor,Dictionary,str]], edit=True):
    #     """
    #     sae_dict: {hook_name: sae_module}
    #     hook_name: dotted path to module output where SAE should attach
    #     sae_module: nn.Module implementing encode/decode
    #     """
        
    #     # self.model.ae_store = getattr(self.model, "ae_store", nn.ModuleDict()) 
    #     # for accessor, sae, parent_attr_name in sae_list:
    #     #     if accessor.attr_name not in self.model.ae_store:
    #     #         self.model.ae_store[accessor.attr_name] = nn.Identity() # placeholder
                
    #     with self.model.edit(inplace=edit) as edited: #inplace = True returns and edited model, inplace = False returns a copy
    #         for accessor, sae, parent_attr_name in sae_list:
    #             encode = FunctionModule(sae.encode)
    #             decode = FunctionModule(sae.decode)
    #             ae = nn.ModuleList([encode, decode])
                
    #             # ae = SAEBlock(**{"encoder_in":ModuleAccessor(encode, parent_attr_name + "_encode_in", io_type=IOType.INPUT),
    #             #             "encoder_out":ModuleAccessor(encode, parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
    #             #             "decoder_in":ModuleAccessor(decode, parent_attr_name + "_decode_in", io_type=IOType.INPUT),
    #             #             "decoder_out":ModuleAccessor(decode, parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)})
                
    #             # self.registry_mod[accessor.attr_name] = ae
    #             # self.model.ae_store[accessor.attr_name].ae = ae
    #             accessor.module.ae = ae
                
    #             kwargs = ae[0].bound_kwargs
    #             accessor_input = accessor.module.input
    #             clone = copy.deepcopy(accessor.module)
    #             accessor_output = clone(accessor_input)
    #             # accessor_output = accessor.module.output
    #             diff = kwargs.get("diff",False)
    #             if diff:
    #                 out = ae[0](accessor_output - accessor_input)
    #             else:
    #                 out = ae[0](accessor_output)    
                    
    #             mask = kwargs.get("mask", False)
    #             error = kwargs.get("error", False)
                
    #             if mask and error:
    #                 out = t.cat([out,out * mask], dim=0) 
    #             out = ae[1](out)
                
    #             if diff:
    #                 out = out + accessor.module.input.expand((out.shape[0], accessor.module.input.shape[1:]))
                    
    #             if error:
    #                 error = accessor_output - out[0]
    #                 out = out[1] + error
    #             accessor.module.output = out  
                
    #     for accessor, sae, parent_attr_name in sae_list:
    #         ae = SAEBlock(**{"encoder_in":ModuleAccessor(accessor.module.ae[0], parent_attr_name + "_encode_in", io_type=IOType.INPUT),
    #                         "encoder_out":ModuleAccessor(accessor.module.ae[0], parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
    #                         "decoder_in":ModuleAccessor(accessor.module.ae[1], parent_attr_name + "_decode_in", io_type=IOType.INPUT),
    #                         "decoder_out":ModuleAccessor(accessor.module.ae[1], parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)})  
    #         accessor.ae = ae                                      
    #     return edited
    
    def extract_activation(self,site):
        pass
    
    @t.no_grad()
    def evaluate(self, data_loader, site):
        pass    