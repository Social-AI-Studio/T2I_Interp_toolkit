# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from collections import defaultdict
from functools import partial

import torch as t
import torch.nn as nn

from dictionary_learning.dictionary import Dictionary
from dictionary_learning.training import trainSAE

# from datasets import load_dataset
from dictionary_learning.utils import hf_dataset_to_generator
from t2Interp.accessors import IOType, ModuleAccessor
from t2Interp.blocks import SAEBlock
from t2Interp.T2I import T2IModel
from train_config import sae_trainer_config
from utils.buffer import t2IActivationBuffer
from utils.output import Output
from utils.utils import FunctionModule


class SAEManager:
    def __init__(self, model):
        self.model: T2IModel = model
        self.registry_mod = {}  # dict: hook_name -> SAE module
        self._handles = []  # list of hook handles
        self.sae_activations = defaultdict(list)  # key -> list of tensors

    def clear_activations(self):
        self.sae_activations = defaultdict(list)

    def train(self, hf_dataset, module: ModuleAccessor, **kwargs):
        generator = hf_dataset_to_generator(hf_dataset)
        buffer = t2IActivationBuffer(generator, self.model, module, **kwargs)
        trainer_config = sae_trainer_config(**kwargs)

        save_dir = kwargs.pop("save_dir", None)
        if save_dir:
            save_dir = os.path.join(save_dir, module.attr_name.replace(".", "_"))

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

    def add_saes_to_model(self, sae_list: list[tuple[ModuleAccessor, Dictionary, str]], **kwargs):
        def process(x, accessor: ModuleAccessor):
            ae = accessor.ae
            # if hasattr(cache, "parent_out") and hasattr(cache, "parent_in"):
            # parent_out = cache["parent_out"].to(x.device,x.dtype)
            # parent_in = cache["parent_in"].to(x.device,x.dtype)
            if (
                accessor.attr_name + "_out" in self.sae_activations
                and accessor.attr_name + "_in" in self.sae_activations
            ):
                parent_out = self.sae_activations[accessor.attr_name + "_out"].to(x.device, x.dtype)
                parent_in = self.sae_activations[accessor.attr_name + "_in"].to(x.device, x.dtype)
                sae_in = parent_out - parent_in
                print(62, parent_in.shape, parent_out.shape, sae_in.shape)
            else:
                sae_in = x
            enc_out = accessor.ae.encoder_out.module(
                sae_in, return_topk=False, use_threshold=False, remove_bias=False
            )
            return enc_out

        # def sae_pre_hook(module, inputs, accessor:ModuleAccessor, cache={}):
        #     (x,) = inputs
        #     out = process(x, accessor=accessor, cache=cache)
        #     self.sae_activations[accessor.attr_name].append(out.detach().cpu())
        #     return (out,)

        def sae_forward_hook(module, input, output, accessor: ModuleAccessor):
            if type(output) == tuple:
                (x,) = output
            else:
                x = output
            enc_out = process(x, accessor=accessor)
            self.sae_activations[accessor.ae.encoder_out.attr_name] = enc_out.detach().cpu()
            # if type(output) == tuple:
            #     return (out,)
            return output

        def parent_out_hook(module, input, output, accessor: ModuleAccessor = None):
            # cache["parent_out"] = output.detach().cpu()
            self.sae_activations[accessor.attr_name + "_out"] = output.detach().cpu()
            if accessor is not None:
                accessor.ae.encoder_out.module(output)
            return output

        def parent_in_hook(module, inputs):
            (x,) = inputs
            # cache["parent_in"] = x.detach().cpu()
            self.sae_activations[accessor.attr_name + "_in"] = x.detach().cpu()
            return inputs

        for accessor, sae, parent_attr_name in sae_list:
            encode = FunctionModule(sae.encode)
            decode = FunctionModule(sae.decode)
            ae = nn.ModuleList([encode, decode])
            ae = SAEBlock(
                **{
                    "encoder_in": ModuleAccessor(
                        ae[0], parent_attr_name + "_encode_in", io_type=IOType.INPUT
                    ),
                    "encoder_out": ModuleAccessor(
                        ae[0], parent_attr_name + "_encode_out", io_type=IOType.OUTPUT
                    ),
                    "decoder_in": ModuleAccessor(
                        ae[1], parent_attr_name + "_decode_in", io_type=IOType.INPUT
                    ),
                    "decoder_out": ModuleAccessor(
                        ae[1], parent_attr_name + "_decode_out", io_type=IOType.OUTPUT
                    ),
                }
            )
            accessor.ae = ae
            # cache={}
            self._handles = getattr(self, "_handles", [])
            if kwargs.pop("diff", False):
                h = accessor.module.register_forward_hook(
                    partial(parent_out_hook, accessor=accessor)
                )
                self._handles.append(h)
                h = accessor.module.register_forward_pre_hook(partial(parent_in_hook))
                self._handles.append(h)
            # if accessor.io_type == IOType.OUTPUT:
            #     h = accessor.ae.encoder_out.module.register_forward_hook(partial(sae_forward_hook,accessor=accessor.ae.encoder_out, cache=cache))
            #     self._handles.append(h)
            # elif accessor.io_type == IOType.INPUT:
            #     h = accessor.ae.encoder_in.module.register_forward_pre_hook(partial(sae_pre_hook,accessor=accessor.ae.encoder_in, cache=cache))
            #     self._handles.append(h)
            h = accessor.module.register_forward_hook(partial(sae_forward_hook, accessor=accessor))
            # else:
            #     raise ValueError(f"IOType {accessor.io_type} not supported for SAE insertion")

            self._handles.append(h)

    # def add_encoder_hook(self, accessor:ModuleAccessor, cache:Dict[str,t.Tensor]):
    #     def cache_hook(module, input, output):
    #         if type(output) == tuple:
    #             (x,) = output
    #         else:
    #             x = output
    #         cache[accessor.attr_name] = x.detach().cpu()
    #     h = accessor.module.register_forward_hook(cache_hook)
    #     self._handles = getattr(self, "_handles", [])
    #     self._handles.append(h)

    def run_with_cache(self, prompt, **kwargs) -> Output:
        # for accessor in accessors:
        #     if not hasattr(accessor, "ae"):
        #         raise ValueError(f"Accessor {accessor.attr_name} does not have an SAE attached")
        # cache = {}
        # for accessor in accessors:
        #     self.add_encoder_hook(accessor, cache)
        try:
            self.model.generate(prompt, **kwargs)
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
        output = Output()
        output.preds = self.sae_activations
        return output

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

    # def extract_activation(self,site):
    #     pass

    @t.no_grad()
    def evaluate(self, data_loader, site):
        pass
