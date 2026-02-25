from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch as t
import torch.nn as nn

from dictionary_learning.dictionary import Dictionary
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator
from t2i_interp.accessors.accessor import IOType, ModuleAccessor
# from t2i_interp.accessors.blocks import SAEBlock # removed if unused
from t2i_interp.t2i import T2IModel
from t2i_interp.config.train_config import sae_trainer_config
from t2i_interp.utils.buffer import t2IActivationBuffer
from t2i_interp.utils.output import Output
from t2i_interp.utils.utils import FunctionModule
from t2i_interp.utils.generic import _extract_tensor_and_rebuild


@dataclass(frozen=True)
class SAEHookSpec:
    """
    Configuration for hooking one module site with one SAE.
    """
    accessor: Any                # should have .module and .attr_name
    sae: nn.Module               # should have encoder/decoder (recommended)
    name: str                    # key to store in _sae_bank and cache
    use_delta: bool = False      # True if SAE trained on (output - input)


class SAEManager:
    def __init__(self, model: T2IModel):
        """
        model: your T2I model wrapper. If it provides model.edit(...), we'll use it as a
               context boundary; hooks are still registered natively on nn.Modules.
        """
        self.model = model

    def clear(self):
        if hasattr(self.model, "clear_edits"):
            self.model.clear_edits()

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

    # -----------------------
    # internal helpers
    # -----------------------
    def _ensure_bank(self, edited_model: Any) -> nn.ModuleDict:
        """
        Ensure SAEs are registered as children modules so external hook/steering utilities
        can discover sae.encoder/decoder via PyTorch traversal.
        """
        if not hasattr(edited_model, "_sae_bank"):
            setattr(edited_model, "_sae_bank", nn.ModuleDict())
        bank = getattr(edited_model, "_sae_bank")
        if not isinstance(bank, nn.ModuleDict):
            raise TypeError("edited_model._sae_bank exists but is not nn.ModuleDict")
        return bank

    def _to_device_dtype(self, sae: nn.Module, ref_module: nn.Module):
        """
        Move SAE once per context entry, not inside the hook (hooks should be cheap).
        """
        p = next(ref_module.parameters(), None)
        if p is not None:
            sae.to(device=p.device, dtype=p.dtype)
        sae.eval()

    def _get_sae_dim(self, sae: nn.Module) -> int:
        if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
             # linear layer: weight is [out, in] usually? or [in, out]?
             # nn.Linear(in, out) -> weight is [out, in]. 
             # So in_features is weight.shape[1].
             return sae.encoder.weight.shape[1]
        # Fallback/Assumption if not standard linear
        # Try to guess from config or attribute if exists
        if hasattr(sae, "config") and hasattr(sae.config, "d_in"):
            return sae.config.d_in
        if hasattr(sae, "d_in"):
            return sae.d_in
        # Fallback: assume last dim is correct? No, that's what caused the bug.
        # We need to know what SAE expects.
        # If we can't determine, raise/warn?
        raise ValueError("Cannot determine input dimension of SAE. Ensure it has .encoder.weight or .config.d_in or .d_in")

    def _flatten_for_sae(self, x: t.Tensor, sae: nn.Module) -> Tuple[t.Tensor, Callable[[t.Tensor], t.Tensor]]:
        """
        Detects correct feature dimension in x based on sae input dim.
        Permutes x to (..., d_in), flattens to (N, d_in).
        Returns (x_flat, restore_fn).
        """
        d_in = self._get_sae_dim(sae)
        
        # Check if d_in matches x.shape[-1] (no permute needed)
        if x.shape[-1] == d_in:
            orig_shape = x.shape
            x_flat = x.reshape(-1, d_in)
            def restore(flat):
                return flat.reshape(orig_shape)
            return x_flat, restore
            
        # Check other dims
        dims = [i for i, d in enumerate(x.shape) if d == d_in]
        if not dims:
            raise RuntimeError(f"SAE expects input dim {d_in}, but input shape is {x.shape}. No dimension matches.")
        
        # If multiple, logic? Heuristic: usually channels is index 1 for (B,C,H,W) or index 2 for (B,S,C)?
        # For now, pick the first one? Or assert unique?
        # In (B, C, H, W), C is index 1.
        # In (B, S, D), D is index 2 (-1).
        # Let's pick dim. If multiple, ambiguous.
        if len(dims) > 1:
             # If last dim is one of them, prefer last?
             if (len(x.shape) - 1) in dims:
                 target_dim = len(x.shape) - 1
             else:
                 # Prefer index 1 (channels) if present?
                 if 1 in dims:
                     target_dim = 1
                 else:
                     target_dim = dims[0] # Fallback
        else:
            target_dim = dims[0]
            
        # Permute target_dim to last
        # Construct permutation
        ndim = x.ndim
        perm = list(range(ndim))
        perm.pop(target_dim)
        perm.append(target_dim)
        
        x_perm = x.permute(*perm)
        orig_perm_shape = x_perm.shape
        x_flat = x_perm.reshape(-1, d_in)
        
        def restore(flat):
            # flat: (N, d_in)
            # unflatten to permuted shape
            unflat = flat.reshape(orig_perm_shape)
            # Inverse permute
            # We want to put last dim back to target_dim
            # Current: [0, 1, ... ndim-2, last] corresponds to [original_non_target..., original_target]
            # We want to insert 'last' at 'target_dim'
            
            # Inverse of perm:
            # The permutation mapped src_idx -> dst_idx.
            # perm[i] is the original index at position i.
            # We want to map back.
            # inv_perm[perm[i]] = i
            inv_perm = [0] * ndim
            for dst_idx, src_idx in enumerate(perm):
                inv_perm[src_idx] = dst_idx
            
            return unflat.permute(*inv_perm)
            
        return x_flat, restore

    def _encode_decode(
        self,
        sae: nn.Module,
        x_flat: t.Tensor,
        z_alter_fn: Optional[Callable[[t.Tensor], t.Tensor]] = None,
    ) -> Tuple[t.Tensor, Optional[t.Tensor]]:
        """
        Preferred SAE API:
            z = sae.encoder(x_flat)
            if z_alter_fn: z = z_alter_fn(z)
            recon = sae.decoder(z)
            return recon, z

        Fallback for capture_identity:
            out = sae(x_flat)
            if out is (recon, z, ...): returns recon, z
            else: returns out, None
        """
        if hasattr(sae, "encoder") and hasattr(sae, "decoder"):
            z = sae.encoder(x_flat)
            if z_alter_fn is not None:
                z = z_alter_fn(z)
            recon = sae.decoder(z)
            return recon, z

        # fallback
        out = sae(x_flat)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        return out, None

    def _convert_list_to_specs(
        self,
        sae_list: Union[List[SAEHookSpec], List[Tuple[ModuleAccessor, nn.Module, str]]],
        use_delta: Union[bool, Dict[str, bool]]
    ) -> List[SAEHookSpec]:
        specs = []
        for item in sae_list:
            if isinstance(item, SAEHookSpec):
                specs.append(item)
            else:
                accessor, sae, name = item
                # Determine use_delta for this specific SAE
                if isinstance(use_delta, dict):
                    delta = use_delta.get(name, False)
                else:
                    delta = use_delta
                
                specs.append(SAEHookSpec(accessor=accessor, sae=sae, name=name, use_delta=delta))
        return specs

    # -----------------------
    # Mode 1: identity + capture
    # -----------------------
    def capture_identity(
        self,
        specs: List[SAEHookSpec],
        *,
        cache: Optional[Dict[str, List[Dict[str, t.Tensor]]]] = None,
        pick_input: Optional[Callable[[Tuple[Any, ...]], t.Tensor]] = None,
    ):
        """
        Attach SAEs and capture {target, recon, err, z} but return original module output.
        """
        @contextlib.contextmanager
        def cm():
            nonlocal cache
            if cache is None:
                cache = {}

            # per-module input stash (needed for delta)
            input_stash: Dict[nn.Module, List[t.Tensor]] = {}
            handles: List[Any] = []

            # Optional boundary context if your model supports it
            edit_ctx = getattr(self.model, "edit", None)
            if callable(edit_ctx):
                boundary = edit_ctx(hooks={})
            else:
                boundary = contextlib.nullcontext(self.model)

            with boundary as edited_model:
                bank = self._ensure_bank(edited_model)
                setattr(edited_model, "_sae_cache", cache)

                for spec in specs:
                    accessor, sae, name, use_delta = spec.accessor, spec.sae, spec.name, spec.use_delta
                    mod: nn.Module = accessor.module
                    self._to_device_dtype(sae, mod)
                    bank[name] = sae
                    input_stash.setdefault(mod, [])

                    # Identify IOType
                    io_type = getattr(accessor, "io_type", IOType.OUTPUT) # Default to OUTPUT if missing

                    if io_type == IOType.OUTPUT:
                        # Standard behavior: pre-hook stashes input, fwd-hook processes output
                        def _pre_hook(module, inputs, _use_pick=pick_input):
                            x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]
                            input_stash[module].append(x_in.detach()) # Detach stash? Or keep grad? usually detached for delta calc if just for error?
                            # If use_delta is for Reconstruction, we might need grad if we were training? But here is inference/steering.
                            # Existing code didn't detach stash in pre_hook, but did detach in capture.
                            # Let's stick to existing behavior: input_stash[module].append(x_in)

                        def _fwd_hook(module, inputs, output, _sae=sae, _name=name, _use_delta=use_delta, _use_pick=pick_input):
                            out_tensor, rebuild = _extract_tensor_and_rebuild(output)
                            
                            # Retrieve stash
                            if input_stash[module]:
                                x_in = input_stash[module].pop()
                            else:
                                # Fallback if pre-hook didn't run or list empty? Should not happen if registered correctly.
                                # Or if we are in a weird re-entrant state. 
                                x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]
                            
                            target = (out_tensor - x_in) if _use_delta else out_tensor
                            
                            x_flat, restore_fn = self._flatten_for_sae(target, _sae)

                            recon_flat, z = self._encode_decode(_sae, x_flat, z_alter_fn=None)
                            recon = restore_fn(recon_flat)
                            err = target - recon

                            cache.setdefault(_name, []).append({
                                "target": target.detach(),
                                "recon": recon.detach(),
                                "err": err.detach(),
                                "z": None if z is None else z.detach(),
                            })

                            # identity: return unmodified output
                            return output

                        handles.append(mod.register_forward_pre_hook(_pre_hook))
                        handles.append(mod.register_forward_hook(_fwd_hook))

                    elif io_type == IOType.INPUT:
                        if use_delta:
                            # Warn? Delta on input usually doesn't make sense unless defining delta wrt something else.
                            # For now, ignore use_delta or treat as identity.
                            pass

                        def _pre_hook_input(module, inputs, _sae=sae, _name=name, _use_pick=pick_input):
                            # inputs is a tuple
                            x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]
                            # We might need to handle tuple rebuild if we were modifying, but here is capture_identity.
                            
                            target = x_in
                            x_flat, restore_fn = self._flatten_for_sae(target, _sae)

                            recon_flat, z = self._encode_decode(_sae, x_flat, z_alter_fn=None)
                            recon = restore_fn(recon_flat)
                            err = target - recon

                            cache.setdefault(_name, []).append({
                                "target": target.detach(),
                                "recon": recon.detach(),
                                "err": err.detach(),
                                "z": None if z is None else z.detach(),
                            })
                            
                            return None # Check if pre-hook returning None means "no change"

                        handles.append(mod.register_forward_pre_hook(_pre_hook_input))

                try:
                    yield edited_model
                finally:
                    for h in reversed(handles):
                        try:
                            h.remove()
                        except Exception:
                            pass
                    for k in input_stash:
                        input_stash[k].clear()

        return cm()

    # -----------------------
    # Mode 2: edit output via encoder alteration
    # -----------------------
    def edit_with_encoder_alter(
        self,
        specs: List[SAEHookSpec],
        *,
        z_alter_fns: Optional[Dict[str, Callable[[t.Tensor], t.Tensor]]] = None,
        cache: Optional[Dict[str, List[Dict[str, t.Tensor]]]] = None,
        pick_input: Optional[Callable[[Tuple[Any, ...]], t.Tensor]] = None,
    ):
        """
        Attach SAEs, optionally alter encoder latents z, and replace module output with an edited value.
        """
        @contextlib.contextmanager
        def cm():
            nonlocal cache
            if cache is None:
                cache = {}

            input_stash: Dict[nn.Module, List[t.Tensor]] = {}
            handles: List[Any] = []

            edit_ctx = getattr(self.model, "edit", None)
            if callable(edit_ctx):
                boundary = edit_ctx(hooks={})
            else:
                boundary = contextlib.nullcontext(self.model)

            with boundary as edited_model:
                bank = self._ensure_bank(edited_model)
                setattr(edited_model, "_sae_cache", cache)

                for spec in specs:
                    accessor, sae, name, use_delta = spec.accessor, spec.sae, spec.name, spec.use_delta
                    mod: nn.Module = accessor.module
                    self._to_device_dtype(sae, mod)
                    bank[name] = sae
                    input_stash.setdefault(mod, [])

                    z_fn = (z_alter_fns or {}).get(name, None)

                    # Enforce encoder/decoder for meaningful z edits
                    needs_encdec = (z_fn is not None) 
                    if needs_encdec and not (hasattr(sae, "encoder") and hasattr(sae, "decoder")):
                        raise ValueError(
                            f"SAE '{name}' must expose .encoder and .decoder"
                        )

                    # Identify IOType
                    io_type = getattr(accessor, "io_type", IOType.OUTPUT)

                    if io_type == IOType.OUTPUT:
                        def _pre_hook(module, inputs, _use_pick=pick_input):
                            x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]
                            input_stash[module].append(x_in.detach())

                        def _fwd_hook(
                            module,
                            inputs,
                            output,
                            _sae=sae,
                            _name=name,
                            _use_delta=use_delta,
                            _zfn=z_fn,
                            _use_pick=pick_input
                        ):
                            out_tensor, rebuild = _extract_tensor_and_rebuild(output)
                            
                            if input_stash[module]:
                                x_in = input_stash[module].pop()
                            else:
                                x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]

                            target = (out_tensor - x_in) if _use_delta else out_tensor
                            
                            x_flat, restore_fn = self._flatten_for_sae(target, _sae)
                            N = x_flat.shape[0]

                            # Double batch for stable baseline error on second half
                            x2 = t.cat([x_flat, x_flat], dim=0)          # [2N, D]
                            z2 = _sae.encoder(x2)                        # [2N, ...]
                            z1 = z2[:N]
                            z0 = z2[N:]

                            if _zfn is not None:
                                z1 = _zfn(z1)

                            r1 = _sae.decoder(z1)                        # [N, D]
                            r0 = _sae.decoder(z0)                        # [N, D]
                            e0 = x_flat - r0                             # [N, D]
                            edited_flat = r1 + e0                        # [N, D]

                            # capture
                            cache.setdefault(_name, []).append({
                                "target": target.detach(),
                                "recon_altered": restore_fn(r1).detach(),
                                "recon_base": restore_fn(r0).detach(),
                                "err_base": restore_fn(e0).detach(),
                                "z_altered": z1.detach(),
                                "z_base": z0.detach(),
                            })

                            edited = restore_fn(edited_flat)

                            # Map back to module output space
                            new_out_tensor = (x_in + edited) if _use_delta else edited
                            return rebuild(new_out_tensor)

                        handles.append(mod.register_forward_pre_hook(_pre_hook))
                        handles.append(mod.register_forward_hook(_fwd_hook))

                    elif io_type == IOType.INPUT:
                        def _pre_hook_input(
                            module, 
                            inputs, 
                            _sae=sae, 
                            _name=name, 
                            _zfn=z_fn, 
                            _use_pick=pick_input
                        ):
                            # inputs is tuple
                            x_in = _use_pick(inputs) if _use_pick is not None else inputs[0]
                            
                            # For Input, target is x_in
                            target = x_in
                            x_flat, restore_fn = self._flatten_for_sae(target, _sae)
                            N = x_flat.shape[0]

                            # Double batch strategy
                            x2 = t.cat([x_flat, x_flat], dim=0)
                            z2 = _sae.encoder(x2)
                            z1 = z2[:N]
                            z0 = z2[N:]

                            if _zfn is not None:
                                z1 = _zfn(z1)

                            r1 = _sae.decoder(z1)
                            r0 = _sae.decoder(z0)
                            e0 = x_flat - r0
                            edited_flat = r1 + e0

                            cache.setdefault(_name, []).append({
                                "target": target.detach(),
                                "recon_altered": restore_fn(r1).detach(),
                                "recon_base": restore_fn(r0).detach(),
                                "err_base": restore_fn(e0).detach(),
                                "z_altered": z1.detach(),
                                "z_base": z0.detach(),
                            })
                            
                            edited = restore_fn(edited_flat)
                            
                            # If inputs was a tuple, we need to rebuild it
                            # Assuming inputs is (x,) or (x, ...)
                            if isinstance(inputs, tuple):
                                new_inputs = list(inputs)
                                new_inputs[0] = edited # Asumming index 0 is the tensor
                                return tuple(new_inputs)
                            return edited

                        handles.append(mod.register_forward_pre_hook(_pre_hook_input))

                try:
                    yield edited_model
                finally:
                    for h in reversed(handles):
                        try:
                            h.remove()
                        except Exception:
                            pass
                    for k in input_stash:
                        input_stash[k].clear()

        return cm()


    # -----------------------
    # Wrappers
    # -----------------------
    def capture_activations(
        self,
        sae_list: List[Tuple[ModuleAccessor, nn.Module, str]],
        prompt: Union[str, List[str]],
        *,
        use_delta: Union[bool, Dict[str, bool]] = False,
        pick_input: Optional[Callable[[Tuple[Any, ...]], t.Tensor]] = None,
        return_images: bool = True,
        **kwargs # passed to pipeline/generation
    ) -> Any:
        """
        Runs the model/pipeline with SAEs attached in capture mode.
        Returns preds dict {sae_name: z} if return_images=False.
        Returns (images, preds) if return_images=True.
        
        Args:
            sae_list: List of (accessor, sae, name) tuples.
            prompt: Text prompt(s).
            use_delta: Boolean or dict mapping name -> bool.
            kwargs: Passed to self.model.pipeline(...) or self.model(...)
        """
        specs = self._convert_list_to_specs(sae_list, use_delta)
        cache = {}
        with self.capture_identity(
            specs, 
            cache=cache, 
            pick_input=pick_input
        ) as edited_model:
            # Use pipeline to generate if available (standard specific to T2IModel)
            if hasattr(edited_model, "pipeline"):
                 out = edited_model.pipeline(prompt, **kwargs)
            else:
                 # Fallback to simple call
                 out = edited_model(prompt, **kwargs)
        
        # Construct preds dict
        preds = {}
        for name, captures in cache.items():
            if captures:
                # Store the z latents from the last step (assuming single step or end usage)
                preds[name] = captures[-1]["z"]
        
        if return_images:
            images = getattr(out, "images", out)
            return images, preds
        return preds


    def run_with_steering(
        self,
        sae_list: List[Tuple[ModuleAccessor, nn.Module, str]],
        prompt: Union[str, List[str]],
        *,
        z_alter_fns: Optional[Dict[str, Callable[[t.Tensor], t.Tensor]]] = None,
        use_delta: Union[bool, Dict[str, bool]] = False,
        pick_input: Optional[Callable[[Tuple[Any, ...]], t.Tensor]] = None,
        return_images: bool = True,
        **kwargs
    ) -> Any:
        """
        Runs the model/pipeline with SAE steering (encoder alteration).
        Returns the generation output (e.g. images) if return_images=True.
        """
        specs = self._convert_list_to_specs(sae_list, use_delta)
        cache = {} # Optional capture even during steering
        with self.edit_with_encoder_alter(
            specs, 
            z_alter_fns=z_alter_fns, 
            cache=cache,
            pick_input=pick_input
        ) as edited_model:
            if hasattr(edited_model, "pipeline"):
                 out = edited_model.pipeline(prompt, **kwargs)
            else:
                 out = edited_model(prompt, **kwargs)
        
        if return_images:
            return getattr(out, "images", out)
        return out
