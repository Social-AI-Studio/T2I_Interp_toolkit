import torch as t
from typing import List, Tuple
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

class SAEManager:
    def __init__(self, model):
        self.model: T2IModel = model
        # self.saes = {}        # dict: hook_name -> SAE module
        # self.handles = {}     # dict: hook_name -> hook handle

    def train(self, hf_dataset, module:ModuleAccessor, **kwargs):
        generator = hf_dataset_to_generator(hf_dataset)
        buffer = t2IActivationBuffer(generator, self.model, **kwargs)
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

    def add_saes_to_model(self, sae_list:List[Tuple[ModuleAccessor,Dictionary]], edit=True):
        """
        sae_dict: {hook_name: sae_module}
        hook_name: dotted path to module output where SAE should attach
        sae_module: nn.Module implementing encode/decode
        """
        self.saes = sae_list
        for envoy, sae, parent_attr_name in sae_list:
            encode = FunctionModule(sae.encode)
            decode = FunctionModule(sae.decode)
            envoy.module.ae = nn.ModuleList([encode, decode])
            envoy.ae = SAEBlock(**{"encoder_in":ModuleAccessor(None, parent_attr_name + "_encode_in", io_type=IOType.INPUT),
                            "encoder_out":ModuleAccessor(None, parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
                            "decoder_in":ModuleAccessor(None, parent_attr_name + "_decode_in", io_type=IOType.INPUT),
                            "decoder_out":ModuleAccessor(None, parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)})  
            
            
        with self.model.edit(inplace=edit) as edited: #inplace = True returns and edited model, inplace = False returns a copy
            for envoy, sae, _ in sae_list:
                args,kwargs = envoy.module.inputs
                
                diff = kwargs.get("diff",False)
                if diff:
                    out = envoy.module.ae[0](envoy.module.output - envoy.module.input)
                else:
                    out = envoy.module.ae[0](envoy.module.output)    
                    
                mask = kwargs.get("mask", False)
                error = kwargs.get("error", False)
                
                if mask and error:
                    out = t.cat([out,out * mask], dim=0) 
                out = envoy.module.ae[1](out)
                
                if diff:
                    out = out + envoy.module.input.expand((out.shape[0], envoy.module.input.shape[1:]))
                    
                if error:
                    error = envoy.module.output - out[0]
                    out = out[1] + error
                envoy.module.output = out 
        
        for envoy, sae, _ in sae_list:        
            envoy.ae.encoder_in.module = envoy.module.ae[0]
            envoy.ae.encoder_out.module = envoy.module.ae[0]
            envoy.ae.decoder_in.module = envoy.module.ae[1]
            envoy.ae.decoder_out.module = envoy.module.ae[1]
                       
        return edited
    
    def extract_activation(self,site):
        pass
    
    @t.no_grad()
    def evaluate(self, data_loader, site):
        pass