from typing import Dict, List, Tuple
import torch
from t2Interp.T2I import T2IModel
from nnsight import NNsight
from nnsight import Envoy, trace
from t2Interp.T2I import T2IModel
from utils.utils import encode_prompt, FieldModel
from utils.output import Output
# from ray_runner import RayWorker
# import ray

class DiffusionIntervention:

    def __init__(
        self,
        model: T2IModel,
        envoys: List[Envoy],
        selection: Dict[str, List[int]] = None,
        start_step: int = 0,
        end_step: int = 50,
    ) -> None:

        self.model = model
        self.envoys = envoys
        self.selection = selection
        self.start_step = start_step
        self.end_step = end_step
        
    def intervene(self, envoy: Envoy):
        pass

    def __call__(self):
        for envoy in self.envoys:
        #     with envoy.iter[self.start_step:self.end_step]:
            self.intervene(envoy)

    @classmethod
    def fields(cls):
        return []
    
class EncoderAttentionIntervention(DiffusionIntervention):

    def __init__(self, replacement_text, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if replacement_text is None:
            replacement_text = ""
            
        self.replacement = encode_prompt(replacement_text, self.model)

    def intervene(self, envoy: Envoy):

        if self.selection:
            
            selection = self.selection["spatial_location"], self.selection["heads"]
            envoy.input = torch.cat([envoy.input, envoy.input[1:]])
            
            envoy.inputs[1]['encoder_hidden_states'] = torch.cat([envoy.inputs[1]['encoder_hidden_states'], self.replacement[1:]])
            
            hidden_states = envoy.to_out[0].input
            
            spatial_dim = hidden_states.shape[1]
            n_heads = envoy.heads

            hidden_states = hidden_states.view(
                (hidden_states.shape[0], spatial_dim, n_heads, -1)
            )

            hidden_states[1, selection[0], selection[1]] = hidden_states[2, selection[0], selection[1]]
            
            envoy.to_out[0].input = envoy.to_out[0].input[:2]

    @classmethod
    def fields(cls):
        return [FieldModel(name="Text", type=FieldModel.FieldType.string)]    

class ScalingAttentionIntervention(DiffusionIntervention):

    def __init__(self, factor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factor = factor

    def intervene(self, attn: Envoy, **kwargs):

        # envoy = attn_envoy.to_out[0]

        # (batch, spatial * spatial, heads * dim)
        hidden_states: torch.Tensor = attn.value

        # if attn_envoy.path in self.selections:
        #     selection = self.selections[attn_envoy.path]
            
        # if self.selection:
            # selection = self.selection["spatial_location"], self.selection["heads"]
        # apply(hidden_states, self.factor, self.selection, attn)
        
        spatial_dim = hidden_states.shape[1]
        if hasattr(attn, 'heads'):
            n_heads = attn.heads
        else:
            n_heads=kwargs.pop('n_heads',None)
        assert n_heads is not None, "n_heads must be provided either as an attribute of the envoy or as a kwarg"        

        hidden_states = hidden_states.view(
            (hidden_states.shape[0], spatial_dim, n_heads, -1)
        )
        
        spatial_idx, head_idx = self.selection["spatial_location"], self.selection["heads"]
        
        if self.selection is not None:
            # THis is effecting both the cond and uncond
            if not torch.is_tensor(spatial_idx):
                spatial_idx = torch.tensor(spatial_idx, dtype=torch.long, device=hidden_states.device)
            else:
                spatial_idx = spatial_idx.to(hidden_states.device).long()
            if not torch.is_tensor(head_idx):
                head_idx = torch.tensor(head_idx, dtype=torch.long, device=hidden_states.device)
            else:
                head_idx = head_idx.to(hidden_states.device).long()
            hidden_states[1, spatial_idx[:, None], head_idx[None, :], :] *= self.factor
        else:
            hidden_states *= self.factor

    @classmethod
    def fields(cls):
        return [FieldModel(name="Factor", type=FieldModel.FieldType.float)]

def run_intervention(model:T2IModel, prompt:str, n_steps:int = 50, start_step:int=0, end_step:int=50, seed:int=40, interventions: List[DiffusionIntervention] = [], **kwargs) -> Output:
    with model.generate(prompt, num_inference_steps=n_steps, seed=seed, validate=False, scan=False, **kwargs) as tracer:
        with tracer.iter[start_step:end_step]:
            for intervention in interventions:
                intervention()    
            output = model.output.save()
    return Output(preds=output.images) 

# @ray.remote(num_gpus=1)
# class InterventionRunner(RayWorker):
#     def __init__(self, model: T2IModel):
#         self.model = model
        
#     def run(self, prompt:str, n_steps:int = 50, seed:int=40, interventions: List[DiffusionIntervention] = []):
#         with self.model.generate(prompt, num_inference_steps=n_steps, seed=seed, validate=False, scan=False) as tracer:
#             for intervention in interventions:
#                 intervention()
#             output = self.model.output.save()
#         return {"preds":output.images[0]}
