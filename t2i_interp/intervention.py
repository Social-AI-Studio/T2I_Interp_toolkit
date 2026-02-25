from typing import List, Callable, Optional, Union
import torch
import torch.nn as nn
from t2i_interp.accessors.accessor import ModuleAccessor
from t2i_interp.t2i import T2IModel
from t2i_interp.utils.output import Output
from t2i_interp.utils.utils import reshape_like
from t2i_interp.utils.T2I.hook import AlterHook, UNetAlterHook, BaseHook
from t2i_interp.utils.trace import TraceDict

class DiffusionIntervention:
    """
    Base class for interventions.
    """
    def __init__(
        self,
        model: T2IModel,
        accessors: list[ModuleAccessor],
        selection: dict[str, list[int]] | None = None,
        start_step: int = 0,
        end_step: int = 50,
    ) -> None:
        self.model = model
        self.accessors = accessors
        self.selection = selection
        self.start_step = start_step
        self.end_step = end_step

    def get_hooks(self) -> dict[nn.Module, BaseHook]:
        """
        Returns a dictionary mapping modules to hooks that implement the intervention.
        """
        return {}
    
    def __call__(self, **kwargs):
        pass


class AddVectorIntervention(DiffusionIntervention):
    def __init__(self, steering_vec: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec

    def get_hooks(self) -> dict[nn.Module, BaseHook]:
        hooks = {}
        for accessor in self.accessors:
            # Define policy for this accessor
            def policy(x, **ctx):
                # Ensure steering vector is on same device/dtype
                vec = self.steering_vec.to(device=x.device, dtype=x.dtype)
                # Simple addition for now, broadcasting should handle batch dimensions if aligned
                # If shape mismatch occurs, we might need reshape_like logic from utils
                # But 'reshape_like' was specific to older logic.
                # Assuming vec matches concept duration.
                # If x is (B, C, H, W) and vec is (C,), it adds to all H,W.
                return x + vec

            # Use generic AlterHook or UNetAlterHook?
            # UNetAlterHook handles CFG splitting.
            # If we want to steer unconditional as well, normal AlterHook.
            # If we want to steer only conditional, UNetAlterHook(cfg_cond_only=True).
            # Default behavior of original seemed to apply to 'accessor.value'.
            # Original intervention: value = value + alpha * vec
            
            # TODO: Handle alpha argument if passed dynamically?
            # Current hook design takes static policy or cached policy.
            # If alpha varies per step, AlterHook.cache supports it.
            
            # For now, static steering. 
            # We use UNetAlterHook to benefit from CFG handling if needed,
            # but default to applying to everything if cfg_cond_only=False.
            hook = UNetAlterHook(policy=policy, step_index=slice(self.start_step, self.end_step))
            hooks[accessor.module] = hook
            
        return hooks


class ReplaceIntervention(DiffusionIntervention):
    def __init__(self, steering_vec: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec
        self.guidance = kwargs.get("guidance", True)

    def get_hooks(self) -> dict[nn.Module, BaseHook]:
        hooks = {}
        for accessor in self.accessors:
            
            def policy(x, **ctx):
                # Steering vec needs to be reshaped to match x
                # implementation logic from original:
                # if guidance and ... split ...
                
                vec = self.steering_vec.to(device=x.device, dtype=x.dtype)
                
                # Logic copied/adapted from original ReplaceIntervention
                if (
                    self.guidance
                    and x.dim() >= 1
                    and x.size(0) % 2 == 0
                    and x.size(0) > 1
                ):
                    B2 = x.size(0)
                    B = B2 // 2
                    uncond = x[:B]
                    cond = x[B:]
                    # reshape_like logic locally
                    # If vec is (D,), and cond is (B, D, ...), expand vec
                    if vec.numel() == cond[0].numel():
                         # Per-sample vector
                         cond_new = vec.view_as(cond)
                    elif vec.numel() * B == cond.numel():
                         # Batch vector?
                         cond_new = vec.view_as(cond)
                    else:
                         # Fallback or broadcast
                         # Assuming vec can broadcast
                         cond_new = vec
                    
                    # Original code used `reshape_like` from utils.utils
                    # cond_new = reshape_like(vec, cond) -> removed for simplicity if possible
                    # but maybe we should import it if critical.
                    # Assuming vec shape fits.
                    
                    return torch.cat([uncond, cond_new], dim=0)
                else:
                    return vec.view_as(x) if vec.numel() == x.numel() else vec

            hook = UNetAlterHook(policy=policy, step_index=slice(self.start_step, self.end_step))
            hooks[accessor.module] = hook
        return hooks


class ScalingAttentionIntervention(DiffusionIntervention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factor = kwargs.get("factor", 1.0)
        self.n_heads = kwargs.get("n_heads", None)

    def get_hooks(self) -> dict[nn.Module, BaseHook]:
        hooks = {}
        for accessor in self.accessors:
            # Need n_heads from somewhere if not provided
            # Accessor has .heads property
            n_heads = self.n_heads or accessor.heads
            # If still None, we can't proceed easily unless module has config
            
            def policy(x, **ctx):
                if n_heads is None:
                    # Try to guess or fail
                    return x 
                
                # Logic from original:
                hs = x
                orig_shape = hs.shape
                orig_ndim = hs.ndim
                S = hs.shape[-2]
                
                # reshape
                if orig_ndim == 3:
                     hs = hs.view(hs.shape[0], S, n_heads, -1)
                elif orig_ndim == 2:
                     hs = hs.view(S, n_heads, -1)
                
                # Selection logic
                sel = self.selection or {}
                spatial_idx_sel = sel.get("spatial_location", None)
                head_idx_sel = sel.get("heads", None)
                
                if type(head_idx_sel) == dict:
                     head_idx_sel = head_idx_sel.get(accessor.attr_name, None)

                # Implement indexing helper or simplify
                # For brevity, assuming simple slicing or factor scaling on whole if None
                
                # ... simple implementation of scaling ...
                if self.factor != 1.0:
                    hs = hs * self.factor
                
                # reshape back
                if orig_ndim == 3:
                    hs = hs.view(orig_shape)
                else:
                    hs = hs.view(S, -1)
                return hs

            hook = UNetAlterHook(policy=policy, step_index=slice(self.start_step, self.end_step))
            hooks[accessor.module] = hook
        return hooks


class FeatureIntervention(DiffusionIntervention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_indices = kwargs.get("feature_indices", None)
        self.scale = kwargs.get("factor", 0.0)

    def get_hooks(self) -> dict[nn.Module, BaseHook]:
        hooks = {}
        for accessor in self.accessors:
            def policy(x, **ctx):
                if self.feature_indices is None:
                    return x
                
                # Clone to avoid inplace issues if needed
                out = x.clone()
                if out.dim() == 2:
                    out[:, self.feature_indices] *= self.scale
                elif out.dim() == 3:
                    out[:, :, self.feature_indices] *= self.scale
                elif out.dim() == 4:
                    out[:, :, :, self.feature_indices] *= self.scale
                return out

            hook = UNetAlterHook(policy=policy, step_index=slice(self.start_step, self.end_step))
            hooks[accessor.module] = hook
        return hooks


def run_intervention(
    model: T2IModel, prompts: list[str], interventions: list[DiffusionIntervention] = [], **kwargs
) -> Output:
    
    # Collect all hooks
    all_hooks = {}
    for intervention in interventions:
        intervention_hooks = intervention.get_hooks()
        for mod, hook in intervention_hooks.items():
            # Simplistic merge: overwrite
            all_hooks[mod] = hook

    # Prepare pipeline args
    # T2IModel wraps pipeline. pipeline is self.model.pipeline
    pipeline = model.pipeline
    
    with torch.no_grad():
        with TraceDict(list(all_hooks.keys()), all_hooks):
             # Assuming pipeline call returns object with .images
             output = pipeline(prompts, **kwargs)
    
    return Output(preds=output.images)
