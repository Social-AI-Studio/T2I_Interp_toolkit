from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from packaging import version
from typing import Callable, Union, Type
import torch as th
from nnsight import Envoy
from nnsight.intervention.tracing.globals import Object
from typing import Literal

TraceTensor = Union[th.Tensor, Object]

class IOType(Enum):
    """Enum to specify input or output access"""

    INPUT = "input"
    OUTPUT = "output"


class ModuleAccessor:
    """I/O accessor that provides input/output access with setter"""

    def __init__(
        self,
        module: th.nn.Module | Envoy,
        attr_name: str | None,
        io_type: IOType | None,
        returns_tuple: bool = False,
    ):

        self.module = module
        self.attr_name = attr_name
        self.io_type = io_type
        self.returns_tuple = returns_tuple
        # self.module.io_type = io_type
        # self.module.returns_tuple = returns_tuple
        
        # self.module = self.attach_io_property_to_envoy(
        #     module,
        #     kind=self.io_type if self.io_type else IOType.INPUT,
        #     returns_tuple=self.returns_tuple,
        #     prop_name="value",
        #     overwrite=True,
        # )
    
    @property
    def value(self) -> TraceTensor | Envoy:
        if self.io_type is None:
            # name = self.attr_name or "layers"
            raise ValueError(
                f"Cannot get the value of a module accessor."
            )
        if self.io_type.value == "input":
            target = self.module.input
        elif self.io_type.value == "output":
            target = self.module.output
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")
        if self.returns_tuple:
            return target[0]
        else:
            return target
        
    @value.setter
    def value(self, new):
        if self.io_type is None:
            raise ValueError("Cannot set the value of a module accessor.")
        kind = getattr(self.io_type, "value", self.io_type)

        if kind == "input":
            if self.returns_tuple:
                # keep any extra tuple elements intact
                old = getattr(self.module, "input")
                rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
                self.module.input = (new, *rest)
            else:
                self.module.input = new
        elif kind == "output":
            if self.returns_tuple:
                old = getattr(self.module, "output")
                rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
                self.module.output = (new, *rest)
            else:
                self.module.output = new
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")
    
    @property
    def heads(self) -> int:
        return getattr(self.module, "heads", None)
        
    # def __call__(self) -> TraceTensor | Envoy:
    #     return self.value

    # def attach_io_property_to_envoy(
    #     self,
    #     env,
    #     kind: IOType,
    #     returns_tuple: bool = False,
    #     prop_name: str = "value",
    #     overwrite: bool = False,
    # ):
    #     """
    #     Attach a property (default name='value') to THIS envoy instance only,
    #     with getter/setter that read/write envoy.input/envoy.output accordingly.
    #     """
    #     # collide safely
    #     Base = env.__class__
    #     if hasattr(Base, prop_name) and not overwrite:
    #         raise AttributeError(
    #             f"{Base.__name__} already has '{prop_name}'. "
    #             "Pass overwrite=True or choose a different prop_name (e.g. 'mi_value')."
    #         )

    #     def fget(self):
    #         tgt = env.input if self.io_type == IOType.INPUT else env.output
    #         if self.returns_tuple and isinstance(tgt, tuple):
    #             return tgt[0]
    #         return tgt

    #     def fset(self, new):
    #         if self.io_type == IOType.INPUT:
    #             old = env.input
    #             if self.returns_tuple:
    #                 rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
    #                 env.input = (new, *rest) if rest else (new,)
    #             else:
    #                 env.input = new
    #         else:
    #             old = env.output
    #             if self.returns_tuple:
    #                 rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
    #                 env.output = (new, *rest) if rest else (new,)
    #             else:
    #                 env.output = new

    #     # per-instance dynamic subclass so only THIS env is affected
    #     attrs = {prop_name: property(fget, fset)}
    #     env.__class__ = type(Base.__name__, (Base,), attrs)
    #     return env
        
class AttentionAccessor:
    def __init__(self):
        pass

# class AttentionAccessor:
#     def __init__(self, model, rename_config: RenameConfig | None = None):
#         self.model = model
#         if rename_config is not None and rename_config.attn_prob_source is not None:
#             self.source_attr = rename_config.attn_prob_source
#         elif isinstance(model._model, BloomForCausalLM):
#             self.source_attr = bloom_attention_prob_source
#         elif isinstance(model._model, GPT2LMHeadModel):
#             self.source_attr = gpt2_attention_prob_source
#         elif isinstance(model._model, GPTJForCausalLM):
#             self.source_attr = gptj_attention_prob_source
#         else:
#             self.source_attr = default_attention_prob_source
#         self.enabled = True

#     def disable(self):
#         self.enabled = False

#     def __getitem__(self, layer: int) -> TraceTensor:
#         if not self.enabled:
#             raise RenamingError("Attention probabilities are disabled for this model.")
#         return self.source_attr(self.model.layers[layer].self_attn).output

#     def __setitem__(self, layer: int, value: TraceTensor):
#         if not self.enabled:
#             raise RenamingError("Attention probabilities are disabled for this model.")
#         self.source_attr(self.model.layers[layer].self_attn).output = value

#     def check_source(
#         self, layer: int = 0, allow_dispatch: bool = True, use_trace: bool = True
#     ):
#         if self.model.num_heads is None:
#             raise RenamingError(
#                 f"Can't check the shapes of the model internals because the number of attention heads is not available in {self.model.repo_id} architecture."
#                 "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
#             )

#         def test_prob_source():
#             batch_size, seq_len = self.model.input_size
#             num_heads = self.model.num_heads
#             probs = self[layer]
#             if probs.shape != (batch_size, num_heads, seq_len, seq_len):
#                 raise RenamingError(
#                     f"Attention probabilities have shape {probs.shape} != {(batch_size, num_heads, seq_len, seq_len)} (batch_size, n_head, seq_len, seq_len) in {self.model.repo_id} architecture. This means it's not properly initialized."
#                 )
#             rnd = th.randn_like(probs).abs()
#             rnd = rnd / rnd.sum(dim=-1, keepdim=True)
#             self[layer] = rnd
#             if probs.device != th.device("meta"):
#                 sum_last = probs.sum(dim=-1)
#                 if not th.allclose(sum_last, th.ones_like(sum_last)):
#                     raise RenamingError("Attention probabilities do not sum to 1.")

#         if use_trace:
#             with self.model.trace(dummy_inputs()):
#                 test_prob_source()
#                 corr_logits = self.model.logits.save()
#             with self.model.trace(dummy_inputs()):
#                 clean_logits = self.model.logits.save()

#             if th.allclose(corr_logits, clean_logits):
#                 raise RenamingError(
#                     "Attention probabilities are not properly initialized: changing the attention probabilities should change the logits."
#                 )
#             return

#         try_with_scan(
#             self.model,
#             test_prob_source,
#             RenamingError(
#                 "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
#             ),
#             allow_dispatch=allow_dispatch,
#             errors_to_raise=(RenamingError,),
#         )

#     def print_source(self, layer: int = 0, allow_dispatch: bool = True):
#         in_notebook = is_notebook()
#         if in_notebook:
#             markdown_text = "## Accessing attention probabilities from:\n"
#         else:
#             print("Accessing attention probabilities from:")

#         def print_hook_source():
#             nonlocal markdown_text
#             source = self.source_attr(self.model.layers[layer].self_attn)
#             if in_notebook:
#                 markdown_text += f"```py\n{source}\n```"
#             else:
#                 print(source)

#         used_scan = try_with_scan(
#             self.model,
#             print_hook_source,
#             RenamingError(
#                 "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
#             ),
#             allow_dispatch=allow_dispatch,
#         )
#         if in_notebook:
#             markdown_text += "\n\n## Full module source:\n"
#         else:
#             print("\n\nFull module source:")

#         def print_attn_source():
#             nonlocal markdown_text
#             source = str(
#                 self.source_attr(
#                     self.model.layers[layer].self_attn, return_module_source=True
#                 )
#             )
#             if in_notebook:
#                 markdown_text += f"```py\n{source}\n```"
#             else:
#                 print(source)

#         try_with_scan(
#             self.model,
#             print_attn_source,
#             RenamingError(
#                 "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
#             ),
#             allow_dispatch=allow_dispatch,
#             warn_if_scan_fails=used_scan,
#         )

#         if in_notebook:
#             display_markdown(markdown_text)


# def get_ignores(model, rename_config: RenameConfig | None = None) -> list[str]:
#     ignores = []
#     if isinstance(model, IGNORE_MLP_MODELS):
#         message = f"{model.__class__.__name__} does not have a mlp module."
#         if isinstance(model, OPTForCausalLM):
#             message += " You'll have to manually use layers.fc1 and layers.fc2 instead."
#         logger.warning(message)
#         ignores.append("mlp")
#     if rename_config is not None:
#         if rename_config.ignore_mlp:
#             ignores.append("mlp")
#         if rename_config.ignore_attn:
#             ignores.append("attention")
#     return ignores


# def mlp_returns_tuple(model, rename_config: RenameConfig | None = None) -> bool:
#     if rename_config is not None and rename_config.mlp_returns_tuple is not None:
#         return rename_config.mlp_returns_tuple
#     return isinstance(model, MLP_RETURNS_TUPLE_MODELS)


# def layer_returns_tuple(model, rename_config: RenameConfig | None = None) -> bool:
#     if rename_config is not None and rename_config.layer_returns_tuple is not None:
#         return rename_config.layer_returns_tuple
#     if version.parse(TRANSFORMERS_VERSION) >= version.parse("4.54") and isinstance(
#         model, LAYER_RETURNS_TENSOR_AFTER_454_MODELS
#     ):
#         return False
#     else:
#         return True


# def check_io(std_model, model_name: str, ignores: list[str]):
#     batch_size, seq_len = std_model.input_size
#     hidden_size = std_model.hidden_size
#     if hidden_size is None:
#         raise RenamingError(
#             f"Can't check the shapes of the model internals because the hidden size is not available in {model_name} architecture."
#             "You should pass the hidden size as an integer or look at the config and pass the key in the hidden_size_config_key argument of a RenameConfig."
#         )
#     token_embeddings = std_model.token_embeddings
#     if not isinstance(token_embeddings, th.Tensor):
#         raise ValueError(
#             f"token_embeddings is not a tensor in {model_name} architecture. Found type {type(token_embeddings)}. This means it's not properly initialized."
#         )
#     if token_embeddings.shape != (batch_size, seq_len, hidden_size):
#         raise ValueError(
#             f"token_embeddings has shape {token_embeddings.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#         )
#     layer_input = std_model.layers_input[0]
#     if not isinstance(layer_input, th.Tensor):
#         raise ValueError(
#             f"layers_input[0] is not a tensor in {model_name} architecture. Found type {type(layer_input)}. This means it's not properly initialized."
#         )
#     if layer_input.shape != (batch_size, seq_len, hidden_size):
#         raise ValueError(
#             f"layers_input[0] has shape {layer_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#         )
#     if "attention" not in ignores:
#         attention_input = std_model.attentions_input[0]
#         attention_output = std_model.attentions_output[0]
#         if not isinstance(attention_input, th.Tensor):
#             raise ValueError(
#                 f"attentions_input[0] is not a tensor in {model_name} architecture. Found type {type(attention_input)}. This means it's not properly initialized."
#             )
#         if attention_input.shape != (batch_size, seq_len, hidden_size):
#             raise ValueError(
#                 f"attentions_input[0] has shape {attention_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#             )
#         if not isinstance(attention_output, th.Tensor):
#             raise ValueError(
#                 f"attentions_output[0] is not a tensor in {model_name} architecture. Found type {type(attention_output)}. This means it's not properly initialized."
#             )
#         if attention_output.shape != (
#             batch_size,
#             seq_len,
#             hidden_size,
#         ):
#             raise ValueError(
#                 f"attentions_output[0] has shape {attention_output.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#             )
#     if "mlp" not in ignores:
#         mlp_input = std_model.mlps_input[0]
#         mlp_output = std_model.mlps_output[0]
#         if not isinstance(mlp_input, th.Tensor):
#             raise ValueError(
#                 f"mlps_input[0] is not a tensor in {model_name} architecture. Found type {type(mlp_input)}. This means it's not properly initialized."
#             )
#         if mlp_input.shape != (batch_size, seq_len, hidden_size):
#             raise ValueError(
#                 f"mlps_input[0] has shape {mlp_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#             )
#         if not isinstance(mlp_output, th.Tensor):
#             raise ValueError(
#                 f"mlps_output[0] is not a tensor in {model_name} architecture. Found type {type(mlp_output)}. This means it's not properly initialized."
#             )
#         if mlp_output.shape != (batch_size, seq_len, hidden_size):
#             raise ValueError(
#                 f"mlps_output[0] has shape {mlp_output.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#             )
#     layer_output = std_model.layers_output[0]
#     if not isinstance(layer_output, th.Tensor):
#         raise ValueError(
#             f"layers_output[0] is not a tensor in {model_name} architecture. Found type {type(layer_output)}. This means it's not properly initialized."
#         )
#     if layer_output.shape != (batch_size, seq_len, hidden_size):
#         raise ValueError(
#             f"layers_output[0] has shape {layer_output.shape} != {(batch_size, seq_len, std_model.config.hidden_size)} in {model_name} architecture. This means it's not properly initialized."
#         )


# def check_model_renaming(
#     std_model,
#     model_name: str,
#     ignores: list[str],
#     allow_dispatch: bool,
# ):

#     if not hasattr(std_model, "layers"):
#         raise RenamingError(
#             f"Could not find layers module in {model_name} architecture. This means that it was not properly renamed.\n"
#             "Please pass the name of the layers module to the layers_rename argument."
#         )
#     if not hasattr(std_model, "ln_final"):
#         raise RenamingError(
#             f"Could not find ln_final module in {model_name} architecture. This means that it was not properly renamed.\n"
#             "Please pass the name of the ln_final module to the ln_final_rename argument."
#         )
#     if not hasattr(std_model, "lm_head"):
#         raise RenamingError(
#             f"Could not find lm_head module in {model_name} architecture. This means that it was not properly renamed.\n"
#             "Please pass the name of the lm_head module to the lm_head_rename argument."
#         )
#     if "attention" not in ignores:
#         if not hasattr(std_model.layers[0], "self_attn"):
#             raise RenamingError(
#                 f"Could not find self_attn module in {model_name} architecture. This means that it was not properly renamed.\n"
#                 "Please pass the name of the self_attn module to the attn_rename argument."
#             )
#     if "mlp" not in ignores:
#         if not hasattr(std_model.layers[0], "mlp"):
#             raise RenamingError(
#                 f"Could not find mlp module in {model_name} architecture. This means that it was not properly renamed.\n"
#                 "Please pass the name of the mlp module to the mlp_rename argument."
#             )

#     try_with_scan(
#         std_model,
#         lambda: check_io(std_model, model_name, ignores),
#         RenamingError(f"Could not check the IO of {model_name}"),
#         allow_dispatch,
#         errors_to_raise=(RenamingError,),
#     )
