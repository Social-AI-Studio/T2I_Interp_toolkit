from typing import Dict, List, Tuple, Any
import torch
from t2Interp.T2I import T2IModel
from nnsight import NNsight
from nnsight import Envoy, trace
from t2Interp.T2I import T2IModel
from utils.utils import encode_prompt, FieldModel, reshape_like
from utils.output import Output
from t2Interp.accessors import ModuleAccessor
# from ray_runner import RayWorker
# import ray

class DiffusionIntervention:

    def __init__(
        self,
        model: T2IModel,
        accessors: List[ModuleAccessor],
        selection: Dict[str, List[int]]| None = None,
        start_step: int = 0,
        end_step: int = 50,
    ) -> None:

        self.model = model
        self.accessors = accessors
        self.selection = selection
        self.start_step = start_step
        self.end_step = end_step
        
    def intervene(self, accessor: ModuleAccessor, **kwargs):
        pass

    def __call__(self,**kwargs):
        for accessor in self.accessors:
        #     with accessor.iter[self.start_step:self.end_step]:
            self.intervene(accessor,**kwargs)

    @classmethod
    def fields(cls):
        return []

class AddVectorIntervention(DiffusionIntervention):
    def __init__(self, steering_vec:torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec
        
    def intervene(self, accessor: ModuleAccessor, **kwargs):
        alpha = kwargs.get("alpha", 1.0)
        accessor.value = accessor.value + alpha * self.steering_vec
    
class ReplaceIntervention(DiffusionIntervention):
    def __init__(self, steering_vec:torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec
        self.guidance = kwargs.get("guidance", True)
        
    def intervene(self, accessor: ModuleAccessor, **kwargs):
        if self.guidance and accessor.value.dim() >= 1 and accessor.value.size(0) % 2 == 0 and accessor.value.size(0) > 1:
            B2 = accessor.value.size(0)
            B = B2 // 2
            uncond = accessor.value[:B]
            cond   = accessor.value[B:]
            cond_new = reshape_like(self.steering_vec,cond)
            out = torch.cat([uncond, cond_new], dim=0)
        else:
            out = reshape_like(self.steering_vec,accessor.value)

        if accessor.module.device is not None:
            out = out.to(accessor.module.device)
            
        accessor.value = out
             
class EncoderAttentionIntervention(DiffusionIntervention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def intervene(self, attn: Envoy, **kwargs):
        
        hs: torch.Tensor = attn.value          # expected shapes: (B, S, D) or (S, D)

        # heads may be on the envoy or passed in
        n_heads = kwargs.get("n_heads", getattr(attn, "heads", None))
        assert n_heads is not None, "n_heads must be provided (kwarg or attn.heads)"

        # original shape bookkeeping
        orig_shape = hs.shape
        orig_ndim  = hs.ndim

        assert orig_ndim ==3 , f"Unexpected hidden_states ndim: {orig_ndim}"
        S = hs.shape[-2]

        assert hs.shape[0] >=2 , f"Expected batch size >=2, got {hs.shape[0]}"
        
        # ----- build indices -----
        sel = getattr(self, "selection", None) or {}
        spatial_idx = sel.get("spatial_location", None)
        head_idx    = sel.get("heads", None)

        device = hs.device

        def to_index(idx, length):
            if idx is None:
                return torch.arange(length, device=device)
            if isinstance(idx, slice):
                return torch.arange(length, device=device)[idx]
            if torch.is_tensor(idx):
                return idx.to(device=device, dtype=torch.long)
            if isinstance(idx, (int,)):
                return torch.tensor([idx], device=device, dtype=torch.long)
            # assume sequence of ints
            return torch.tensor(list(idx), device=device, dtype=torch.long)

        spatial_idx = to_index(spatial_idx, S)
        head_idx    = to_index(head_idx, n_heads)

        # (B, S, H, d)
        hs = hs.view(hs.shape[0], S, n_heads, -1)
        if hs.shape[0] ==2:
            hs[0, spatial_idx[:, None], head_idx[None, :], :] = hs[1, spatial_idx[:, None], head_idx[None, :], :]
        elif hs.shape[0] ==4:
            hs[1, spatial_idx[:, None], head_idx[None, :], :] = hs[3, spatial_idx[:, None], head_idx[None, :], :]
            
        hs = hs.view(orig_shape[0], S, -1)
        attn.value = hs
            
        # if self.selection:
        #     selection = self.selection["spatial_location"], self.selection["heads"]
        #     envoy.input = torch.cat([envoy.input, envoy.input[1:]])
            
        #     envoy.inputs[1]['encoder_hidden_states'] = torch.cat([envoy.inputs[1]['encoder_hidden_states'], self.replacement[1:]])
            
        #     hidden_states = envoy.to_out[0].input
            
        #     spatial_dim = hidden_states.shape[1]
        #     n_heads = envoy.heads

        #     hidden_states = hidden_states.view(
        #         (hidden_states.shape[0], spatial_dim, n_heads, -1)
        #     )

        #     hidden_states[1, selection[0], selection[1]] = hidden_states[2, selection[0], selection[1]]
            
        #     envoy.to_out[0].input = envoy.to_out[0].input[:2]   

class ScalingAttentionIntervention(DiffusionIntervention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.factor = factor

    def intervene(self, attn: Envoy, **kwargs):
        hs: torch.Tensor = attn.value          # expected shapes: (B, S, D) or (S, D)
        factor = kwargs.get("factor", getattr(attn, "factor", None))
        assert factor is not None, "factor must be provided (kwarg or attn.factor)"
        
        # heads may be on the envoy or passed in
        n_heads = kwargs.get("n_heads", getattr(attn, "heads", None))
        assert n_heads is not None, "n_heads must be provided (kwarg or attn.heads)"

        # original shape bookkeeping
        orig_shape = hs.shape
        orig_ndim  = hs.ndim

        # infer spatial length from the sequence dimension before head-splitting
        # (B,S,D) -> S at dim=-2 ; (S,D) -> S at dim=-2 as well
        S = hs.shape[-2]

        # reshape to expose heads
        if orig_ndim == 3:       # (B, S, D) -> (B, S, H, d)
            hs = hs.view(hs.shape[0], S, n_heads, -1)
        elif orig_ndim == 2:     # (S, D)    -> (S, H, d)
            hs = hs.view(S, n_heads, -1)
        else:
            raise ValueError(f"Unexpected hidden_states shape: {tuple(orig_shape)}")

        # ----- build indices -----
        sel = getattr(self, "selection", None) or {}
        spatial_idx = sel.get("spatial_location", None)
        head_idx    = sel.get("heads", None)

        if type(head_idx) == dict:
            head_idx = head_idx.get(attn.attr_name, None)
            
        device = hs.device

        def to_index(idx, length):
            if idx is None:
                return torch.arange(length, device=device)
            if isinstance(idx, slice):
                return torch.arange(length, device=device)[idx]
            if torch.is_tensor(idx):
                return idx.to(device=device, dtype=torch.long)
            if isinstance(idx, (int,)):
                return torch.tensor([idx], device=device, dtype=torch.long)
            # assume sequence of ints
            return torch.tensor(list(idx), device=device, dtype=torch.long)

        spatial_idx = to_index(spatial_idx, S)
        head_idx    = to_index(head_idx, n_heads)

        # no-op if any index set is empty
        if spatial_idx.numel() == 0 or head_idx.numel() == 0 or factor == 1.0:
            # write back original and return
            attn.value = attn.value  # noop for clarity
            return

        # ----- apply intervention -----
        if hs.ndim == 4:
            # (B, S, H, d)
            hs[:, spatial_idx[:, None], head_idx[None, :], :] *= factor
        else:
            # (S, H, d)
            hs[spatial_idx[:, None], head_idx[None, :], :] *= factor

        # reshape back and write result into the Envoy value
        if orig_ndim == 3:
            attn.value = hs.view(orig_shape[0], S, -1)
        else:
            attn.value = hs.view(S, -1)

    # @classmethod
    # def fields(cls):
    #     return [FieldModel(name="Factor", type=FieldModel.FieldType.float)]

class FeatureIntervention(DiffusionIntervention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)    
            
    def intervene(self, accessor: ModuleAccessor, **kwargs):   
        feature_indices = kwargs.get("feature_indices", None)
        scale = kwargs.get("factor", 0.0)
        assert feature_indices is not None, "feature_indices must be provided as kwargs"
        def scale_hook(module, input, output):
            # ablate last indices
            if output.dim() == 2:
                output[:, feature_indices] *= scale
            elif output.dim() ==3:
                output[:, :, feature_indices] *= scale
            elif output.dim() ==4:
                output[:, :, :, feature_indices] *= scale
            else:
                raise ValueError(f"Unexpected output dim: {output.dim()}")
            return output
        accessor.module.register_forward_hook(scale_hook) 
    
def run_intervention(model:T2IModel, prompts:List[str], interventions: List[DiffusionIntervention] = [], **kwargs) -> Output:
    start_step = kwargs.get("denoiser_step", 0)
    start_step = 0
    end_step   = start_step + 1
    with model.generate(prompts, validate=False, scan=False, **kwargs) as tracer:
        with tracer.iter[start_step:end_step]:
            for intervention in interventions:
                intervention(**kwargs)    
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
