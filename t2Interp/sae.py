import torch
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
import torch.nn as nn
from T2I import T2IModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class SAEManager:
    def __init__(self, model):
        self.model: T2IModel = model
        self.saes = {}        # dict: hook_name -> SAE module
        self.handles = {}     # dict: hook_name -> hook handle

    def train(self, dataset, modules, save_data=False):
        

        cfg = TrainConfig(SaeConfig(), batch_size=16)
        trainer = Trainer(cfg, tokenized, gpt)

        trainer.fit()

    def _load(self, model:T2IModel, sae_dict:dict[str,nn.module], edit=False):
        """
        sae_dict: {hook_name: sae_module}
        hook_name: dotted path to module output where SAE should attach
        sae_module: nn.Module implementing encode/decode
        """
        self.saes = sae_dict
        
        # if edit, edit the model to include SAE modules
        with model.edit():
            for name, sae in sae_dict.items():
                pass

    def extract_activation(self,site):
        pass
        
    # def intervene(self, name, features, mode="scale", value=1.0):
    #     """
    #     name: which SAE site to intervene on
    #     features: indices of features to manipulate
    #     mode: "scale", "ablate", or "set"
    #     value: scalar or tensor value
    #     """
        
    #     sae = self.saes[name]
    #     # shape has to be managed based on the module
    #     def hook_fn(mod, inp, out):
    #         f = sae.encode(out)
    #         if mode == "scale":
    #             f[:, features] *= value
    #         elif mode == "ablate":
    #             f[:, features] = 0.0
    #         elif mode == "set":
    #             f[:, features] = value
    #         return sae.decode(f)

    def evaluate(self, data_loader, site):
        pass
    
    def autointerp(self):
        pass