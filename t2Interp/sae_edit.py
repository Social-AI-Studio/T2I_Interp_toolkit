"""
SAE Manager using .edit() instead of hooks.

This is a cleaner approach that:
1. Registers SAE as a model attribute (making it an Envoy)
2. Uses .edit() to wire SAE into the computation flow
3. Allows direct tracing/intervention on SAE encoder/decoder

Key insight: When you set an nn.Module as an attribute on an Envoy,
nnsight automatically wraps it via _add_envoy(), making it fully traceable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from nnsight import Envoy

from t2Interp.accessors import IOType, ModuleAccessor
from t2Interp.blocks import SAEBlock

if TYPE_CHECKING:
    from dictionary_learning.dictionary import Dictionary

    from t2Interp.T2I import T2IModel


class SAEModule(nn.Module):
    """
    Wrapper around a Dictionary (SAE) that exposes encoder/decoder as submodules.

    This allows nnsight to wrap encoder/decoder as separate Envoys when
    the SAEModule is registered as a model attribute.
    """

    def __init__(self, sae: Dictionary, name: str):
        super().__init__()
        self.name = name
        self._sae = sae

        # Create encoder/decoder as proper nn.Module submodules
        # so they become Envoys when parent is registered
        self.encoder = _EncoderModule(sae)
        self.decoder = _DecoderModule(sae)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full SAE forward: encode then decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Just encode."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Just decode."""
        return self.decoder(z)


class _EncoderModule(nn.Module):
    """Encoder wrapper that calls sae.encode()"""

    def __init__(self, sae: Dictionary):
        super().__init__()
        self._sae = sae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sae.encode(x)


class _DecoderModule(nn.Module):
    """Decoder wrapper that calls sae.decode()"""

    def __init__(self, sae: Dictionary):
        super().__init__()
        self._sae = sae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(z)


class SAEManagerEdit:
    """
    SAE Manager that uses .edit() instead of hooks.

    Usage:
        manager = SAEManagerEdit(model)

        # Add SAE to a target module
        sae_block = manager.add_sae(
            target_accessor=model.unet.conv_out,  # or a ModuleAccessor
            sae=my_trained_sae,
            name="conv_out_sae"
        )

        # Now you can trace/intervene on SAE internals
        with manager.edited_model.generate(...) as tracer:
            # Access SAE encoder output
            enc_out = sae_block.encoder_out.value

            # Intervene on SAE features
            sae_block.encoder_out.value = enc_out * 0.5
    """

    def __init__(self, model: T2IModel):
        self.model = model
        self._sae_blocks: dict[str, SAEBlock] = {}
        self._sae_modules: dict[str, SAEModule] = {}
        self._edited_model = None
        self._edit_applied = False

    def add_sae(
        self,
        target_accessor: ModuleAccessor | Envoy,
        sae: Dictionary,
        name: str,
        io_type: IOType = IOType.OUTPUT,
    ) -> SAEBlock:
        """
        Add an SAE at a target module location.

        Args:
            target_accessor: The module to attach SAE to (ModuleAccessor or Envoy)
            sae: The trained SAE (Dictionary object with encode/decode methods)
            name: Unique name for this SAE (used as attribute name)
            io_type: Whether to intercept INPUT or OUTPUT of target module

        Returns:
            SAEBlock with accessors for encoder_in, encoder_out, decoder_in, decoder_out
        """
        if name in self._sae_modules:
            raise ValueError(f"SAE with name '{name}' already exists")

        # Get the target module (Envoy)
        if isinstance(target_accessor, ModuleAccessor):
            target_envoy = target_accessor.module
            target_io_type = target_accessor.io_type or io_type
        else:
            target_envoy = target_accessor
            target_io_type = io_type

        # Create SAE module wrapper
        sae_module = SAEModule(sae, name)

        # Move SAE to same device/dtype as model
        if hasattr(self.model, "_module"):
            # Get device from model
            try:
                model_device = next(self.model._module.parameters()).device
                model_dtype = next(self.model._module.parameters()).dtype
                sae_module = sae_module.to(model_device).to(model_dtype)
            except StopIteration:
                pass

        # Register SAE as attribute on parent module
        # This triggers nnsight's _add_envoy() automatically!
        parent_envoy = self._get_parent_envoy(target_envoy)
        setattr(parent_envoy, name, sae_module)

        # Store reference
        self._sae_modules[name] = sae_module

        # Get the SAE envoy (now properly wrapped)
        sae_envoy = getattr(parent_envoy, name)

        # Create SAEBlock with accessors
        sae_block = SAEBlock(
            encoder_in=ModuleAccessor(
                sae_envoy.encoder, f"{name}_encoder_in", io_type=IOType.INPUT
            ),
            encoder_out=ModuleAccessor(
                sae_envoy.encoder, f"{name}_encoder_out", io_type=IOType.OUTPUT
            ),
            decoder_in=ModuleAccessor(
                sae_envoy.decoder, f"{name}_decoder_in", io_type=IOType.INPUT
            ),
            decoder_out=ModuleAccessor(
                sae_envoy.decoder, f"{name}_decoder_out", io_type=IOType.OUTPUT
            ),
        )

        # Store the block
        self._sae_blocks[name] = sae_block

        # Store info for wiring
        sae_block._target_envoy = target_envoy
        sae_block._target_io_type = target_io_type
        sae_block._sae_envoy = sae_envoy
        sae_block._parent_envoy = parent_envoy

        return sae_block

    def _get_parent_envoy(self, envoy: Envoy) -> Envoy:
        """Get the parent envoy to attach SAE to."""
        # For now, attach to the same level as target
        # Could be made smarter to find appropriate parent
        path = envoy.path if hasattr(envoy, "path") else ""
        parts = path.split(".")

        if len(parts) <= 1:
            return self.model

        # Navigate to parent
        parent = self.model
        for part in parts[:-1]:
            if part and hasattr(parent, part):
                parent = getattr(parent, part)

        return parent

    def apply_edits(self) -> Envoy:
        """
        Apply .edit() to wire all SAEs into the computation flow.

        This method is diff-mode aware - if any SAE was added with diff=True,
        it will process (output - input) and add input back.

        Returns:
            The edited model ready for generation
        """
        return self.apply_edits_with_diff()

    @property
    def edited_model(self) -> Envoy:
        """Get the edited model, applying edits if not already done."""
        if not self._edit_applied:
            return self.apply_edits()
        return self._edited_model

    def get_sae(self, name: str) -> SAEBlock | None:
        """Get an SAEBlock by name."""
        return self._sae_blocks.get(name)

    def get_sae_module(self, name: str) -> SAEModule | None:
        """Get the underlying SAE module by name."""
        return self._sae_modules.get(name)

    @property
    def sae_names(self) -> list[str]:
        """List all registered SAE names."""
        return list(self._sae_blocks.keys())

    def clear(self):
        """Remove all SAEs and reset state."""
        self._sae_blocks.clear()
        self._sae_modules.clear()
        self._edited_model = None
        self._edit_applied = False

    # =========================================================================
    # Backward-compatible methods (matching old SAEManager interface)
    # =========================================================================

    def add_saes_to_model(
        self,
        sae_list: list[tuple],
        diff: bool = False,
        **kwargs,
    ) -> list[SAEBlock]:
        """
        Add multiple SAEs to the model (backward-compatible with old SAEManager).

        Args:
            sae_list: List of tuples (target_accessor, sae, name)
                - target_accessor: ModuleAccessor or Envoy for target module
                - sae: Dictionary object with encode/decode methods
                - name: Unique name for this SAE
            diff: If True, SAE processes (output - input) instead of just output.
                  This matches how some SAEs are trained (on residuals).
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            List of SAEBlock objects for each added SAE

        Example (matching old interface):
            sae_manager.add_saes_to_model(
                sae_list=[(model.unet.conv_out, sae, "conv_out_sae")],
                diff=True
            )
        """
        sae_blocks = []
        for target_accessor, sae, name in sae_list:
            sae_block = self.add_sae(
                target_accessor=target_accessor,
                sae=sae,
                name=name,
                io_type=IOType.OUTPUT,
            )
            # Store diff mode for this SAE
            sae_block._diff_mode = diff
            sae_blocks.append(sae_block)

        return sae_blocks

    def apply_edits_with_diff(self) -> Envoy:
        """
        Apply .edit() to wire all SAEs, supporting diff mode.

        In diff mode, SAE processes (output - input) and adds back input:
            residual = output - input
            sae_out = sae(residual)
            final = sae_out + input

        Returns:
            The edited model ready for generation
        """
        if self._edit_applied:
            return self._edited_model

        with self.model.edit() as edited:
            for _name, sae_block in self._sae_blocks.items():
                target_envoy = sae_block._target_envoy
                sae_envoy = sae_block._sae_envoy
                io_type = sae_block._target_io_type
                diff_mode = getattr(sae_block, "_diff_mode", False)

                if io_type == IOType.OUTPUT:
                    if diff_mode:
                        # Diff mode: SAE processes (output - input)
                        target_input = target_envoy.input
                        target_output = target_envoy.output
                        # Handle tuple inputs
                        if hasattr(target_input, "__getitem__"):
                            try:
                                inp = target_input[0]
                            except (IndexError, TypeError):
                                inp = target_input
                        else:
                            inp = target_input
                        residual = target_output - inp
                        sae_output = sae_envoy(residual)
                        target_envoy.output = sae_output + inp
                    else:
                        # Normal mode: SAE processes output directly
                        activation = target_envoy.output
                        sae_output = sae_envoy(activation)
                        target_envoy.output = sae_output
                else:
                    # Input mode (no diff support)
                    activation = target_envoy.input
                    sae_output = sae_envoy(activation)
                    target_envoy.input = sae_output

        self._edited_model = edited
        self._edit_applied = True
        return edited

    def run_with_cache(
        self,
        prompt: str,
        accessors: list[ModuleAccessor] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Run the model and cache SAE activations (backward-compatible).

        Args:
            prompt: Text prompt for generation
            accessors: List of ModuleAccessors to cache (default: all SAE encoder outputs)
            **kwargs: Additional generation arguments (num_inference_steps, seed, etc.)

        Returns:
            Dict mapping accessor names to cached activation tensors

        Example:
            output = sae_manager.run_with_cache(
                prompt="A mountain",
                accessors=[sae_block.encoder_out],
                num_inference_steps=1,
                seed=42,
            )
        """
        # Apply edits if not done (use diff-aware version)
        if not self._edit_applied:
            self.apply_edits_with_diff()

        # Default to all SAE encoder outputs
        if accessors is None:
            accessors = [block.encoder_out for block in self._sae_blocks.values()]

        # Extract generation kwargs
        num_inference_steps = kwargs.get("num_inference_steps", 1)
        seed = kwargs.get("seed", None)
        guidance_scale = kwargs.get("guidance_scale", 0.0)

        # Prepare generation kwargs
        gen_kwargs = {
            "num_inference_steps": num_inference_steps,
            "validate": False,
            "scan": False,
        }
        if seed is not None:
            gen_kwargs["seed"] = seed
        if guidance_scale is not None:
            gen_kwargs["guidance_scale"] = guidance_scale

        # Run with caching
        cache = {}
        with self._edited_model.generate(prompt, **gen_kwargs):
            for accessor in accessors:
                cache[accessor.attr_name] = accessor.value.save()

        # Convert proxies to tensors
        return {k: v.value if hasattr(v, "value") else v for k, v in cache.items()}

    @property
    def sae_activations(self) -> dict[str, torch.Tensor]:
        """Property for backward compatibility with old SAEManager."""
        return {}


# =============================================================================
# Convenience functions
# =============================================================================


def add_sae_to_model(
    model: T2IModel,
    target_path: str,
    sae: Dictionary,
    name: str | None = None,
    io_type: IOType = IOType.OUTPUT,
) -> tuple[SAEManagerEdit, SAEBlock]:
    """
    Convenience function to add a single SAE to a model.

    Args:
        model: The T2IModel
        target_path: Dot-separated path to target module (e.g., "unet.conv_out")
        sae: The trained SAE
        name: Name for the SAE (defaults to path-based name)
        io_type: Whether to intercept INPUT or OUTPUT

    Returns:
        Tuple of (SAEManagerEdit, SAEBlock)

    Example:
        manager, sae_block = add_sae_to_model(
            model,
            "unet.conv_out",
            my_sae,
            name="conv_sae"
        )

        with manager.edited_model.generate(...) as tracer:
            features = sae_block.encoder_out.value.save()
    """
    manager = SAEManagerEdit(model)

    # Navigate to target
    target = model
    for attr in target_path.split("."):
        target = getattr(target, attr)

    # Generate name if not provided
    if name is None:
        name = target_path.replace(".", "_") + "_sae"

    sae_block = manager.add_sae(
        target_accessor=target,
        sae=sae,
        name=name,
        io_type=io_type,
    )

    return manager, sae_block
