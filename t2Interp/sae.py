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
        for accessor, sae, parent_attr_name in sae_list:
            encode = FunctionModule(sae.encode)
            decode = FunctionModule(sae.decode)
            ae = nn.ModuleList([encode, decode])
            accessor.module.add_module("ae", ae)
            
            accessor.ae = SAEBlock(**{"encoder_in":ModuleAccessor(None, parent_attr_name + "_encode_in", io_type=IOType.INPUT),
                            "encoder_out":ModuleAccessor(None, parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
                            "decoder_in":ModuleAccessor(None, parent_attr_name + "_decode_in", io_type=IOType.INPUT),
                            "decoder_out":ModuleAccessor(None, parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)}) 
            
        with self.model.edit(inplace=edit) as edited: #inplace = True returns and edited model, inplace = False returns a copy
            for accessor, sae, parent_attr_name in sae_list:
                # encode = FunctionModule(sae.encode)
                # decode = FunctionModule(sae.decode)
                # ae = nn.ModuleList([encode, decode])
                # accessor.module.add_module("sae", ae)
                # accessor.ae = SAEBlock(**{"encoder_in":ModuleAccessor(ae[0], parent_attr_name + "_encode_in", io_type=IOType.INPUT),
                #             "encoder_out":ModuleAccessor(ae[0], parent_attr_name + "_encode_out", io_type=IOType.OUTPUT),
                #             "decoder_in":ModuleAccessor(ae[1], parent_attr_name + "_decode_in", io_type=IOType.INPUT),
                #             "decoder_out":ModuleAccessor(ae[1], parent_attr_name + "_decode_out", io_type=IOType.OUTPUT)})
            
                kwargs = accessor.module.ae[0].bound_kwargs
                
                diff = kwargs.get("diff",False)
                if diff:
                    out = accessor.module.ae[0](accessor.module.output - accessor.module.input)
                else:
                    out = accessor.module.ae[0](accessor.module.output)    
                    
                mask = kwargs.get("mask", False)
                error = kwargs.get("error", False)
                
                if mask and error:
                    out = t.cat([out,out * mask], dim=0) 
                out = accessor.module.ae[1](out)
                
                if diff:
                    out = out + accessor.module.input.expand((out.shape[0], accessor.module.input.shape[1:]))
                    
                if error:
                    error = accessor.module.output - out[0]
                    out = out[1] + error
                accessor.module.ae[1].output = out 
        
        for accessor, sae, _ in sae_list:        
            accessor.ae.encoder_in.module = accessor.module.ae[0]
            accessor.ae.encoder_out.module = accessor.module.ae[0]
            accessor.ae.decoder_in.module = accessor.module.ae[1]
            accessor.ae.decoder_out.module = accessor.module.ae[1]
                           
        return edited
    
    def extract_activation(self,site):
        pass
    
    @t.no_grad()
    def evaluate(self, data_loader, site):
        pass