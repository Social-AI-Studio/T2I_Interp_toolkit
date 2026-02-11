"""
Intervention classes for diffusion model interpretability.

Provides various intervention strategies that can be applied to model activations
during generation. All interventions work with ModuleAccessor objects for
consistent input/output access.

Usage with SAEManagerEdit (recommended):
    sae_block = manager.add_sae(target, sae, "my_sae")
    intervention = SAEFeatureScalingIntervention(
        model=model,
        accessors=[sae_block.encoder_out],
        feature_indices=[42, 128, 256],
        scale=2.0,
    )

Usage with run_intervention:
    output = run_intervention(model, prompts, interventions=[intervention])
"""

import torch
from nnsight import Envoy

from t2Interp.accessors import ModuleAccessor
from t2Interp.T2I import T2IModel
from utils.output import Output
from utils.utils import reshape_like


class DiffusionIntervention:
    """
    Base class for all diffusion model interventions.

    Interventions modify model activations at specific modules during generation.
    Each intervention operates on a list of ModuleAccessor targets.

    Args:
        model: The T2IModel instance.
        accessors: List of ModuleAccessor objects specifying where to intervene.
        selection: Optional dict for spatial/head selection (e.g., {"spatial_location": [...], "heads": [...]}).
        start_step: First diffusion step to apply intervention (inclusive).
        end_step: Last diffusion step to apply intervention (exclusive).
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

    def intervene(self, accessor: ModuleAccessor, **kwargs):
        """Apply the intervention to a single accessor. Override in subclasses."""
        pass

    def __call__(self, **kwargs):
        """Apply intervention to all accessors."""
        for accessor in self.accessors:
            self.intervene(accessor, **kwargs)

    @classmethod
    def fields(cls):
        """Return field descriptors for UI integration."""
        return []


class AddVectorIntervention(DiffusionIntervention):
    """
    Add a steering vector to activations.

    Computes: activation = activation + alpha * steering_vec

    Args:
        steering_vec: The steering vector to add.
        alpha: Scaling factor for the steering vector (passed via kwargs at call time).

    Example:
        intervention = AddVectorIntervention(
            steering_vec=my_vector,
            model=model,
            accessors=[accessor],
        )
        intervention(alpha=2.0)
    """

    def __init__(self, steering_vec: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec

    def intervene(self, accessor: ModuleAccessor, **kwargs):
        alpha = kwargs.get("alpha", 1.0)
        accessor.value = accessor.value + alpha * self.steering_vec


class ReplaceIntervention(DiffusionIntervention):
    """
    Replace activations with a steering vector.

    When guidance is enabled (default), only replaces the conditional portion
    of the batch, preserving the unconditional guidance.

    Args:
        steering_vec: The replacement vector.
        guidance: Whether to handle classifier-free guidance splitting (default: True).
    """

    def __init__(self, steering_vec: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steering_vec = steering_vec
        self.guidance = kwargs.get("guidance", True)

    def intervene(self, accessor: ModuleAccessor, **kwargs):
        if (
            self.guidance
            and accessor.value.dim() >= 1
            and accessor.value.size(0) % 2 == 0
            and accessor.value.size(0) > 1
        ):
            B2 = accessor.value.size(0)
            B = B2 // 2
            uncond = accessor.value[:B]
            cond = accessor.value[B:]
            cond_new = reshape_like(self.steering_vec, cond)
            out = torch.cat([uncond, cond_new], dim=0)
        else:
            out = reshape_like(self.steering_vec, accessor.value)

        if accessor.module.device is not None:
            out = out.to(accessor.module.device)

        accessor.value = out


class EncoderAttentionIntervention(DiffusionIntervention):
    """
    Intervene on encoder attention hidden states.

    Copies selected spatial positions and heads from the conditional batch
    element to the unconditional element. Supports both batch size 2
    (standard guidance) and batch size 4 (extended guidance).

    Selection dict keys:
        - "spatial_location": indices of spatial positions to intervene on.
        - "heads": indices of attention heads to intervene on.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def intervene(self, attn: Envoy, **kwargs):
        hs: torch.Tensor = attn.value

        n_heads = kwargs.get("n_heads", getattr(attn, "heads", None))
        assert n_heads is not None, "n_heads must be provided (kwarg or attn.heads)"

        orig_shape = hs.shape
        assert hs.ndim == 3, f"Unexpected hidden_states ndim: {hs.ndim}"
        S = hs.shape[-2]
        assert hs.shape[0] >= 2, f"Expected batch size >=2, got {hs.shape[0]}"

        sel = getattr(self, "selection", None) or {}
        spatial_idx = sel.get("spatial_location", None)
        head_idx = sel.get("heads", None)
        device = hs.device

        def to_index(idx, length):
            if idx is None:
                return torch.arange(length, device=device)
            if isinstance(idx, slice):
                return torch.arange(length, device=device)[idx]
            if torch.is_tensor(idx):
                return idx.to(device=device, dtype=torch.long)
            if isinstance(idx, int):
                return torch.tensor([idx], device=device, dtype=torch.long)
            return torch.tensor(list(idx), device=device, dtype=torch.long)

        spatial_idx = to_index(spatial_idx, S)
        head_idx = to_index(head_idx, n_heads)

        hs = hs.view(hs.shape[0], S, n_heads, -1)
        if hs.shape[0] == 2:
            hs[0, spatial_idx[:, None], head_idx[None, :], :] = hs[
                1, spatial_idx[:, None], head_idx[None, :], :
            ]
        elif hs.shape[0] == 4:
            hs[1, spatial_idx[:, None], head_idx[None, :], :] = hs[
                3, spatial_idx[:, None], head_idx[None, :], :
            ]

        hs = hs.view(orig_shape[0], S, -1)
        attn.value = hs


class ScalingAttentionIntervention(DiffusionIntervention):
    """
    Scale attention activations by a factor at selected spatial positions and heads.

    Args:
        factor: Scaling factor (passed via kwargs at call time).
        n_heads: Number of attention heads (passed via kwargs or from attn.heads).

    Selection dict keys:
        - "spatial_location": indices of spatial positions to scale.
        - "heads": indices (or dict of attn_name -> indices) of heads to scale.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def intervene(self, attn: Envoy, **kwargs):
        hs: torch.Tensor = attn.value
        factor = kwargs.get("factor", getattr(attn, "factor", None))
        assert factor is not None, "factor must be provided (kwarg or attn.factor)"

        n_heads = kwargs.get("n_heads", getattr(attn, "heads", None))
        assert n_heads is not None, "n_heads must be provided (kwarg or attn.heads)"

        orig_shape = hs.shape
        orig_ndim = hs.ndim
        S = hs.shape[-2]

        if orig_ndim == 3:
            hs = hs.view(hs.shape[0], S, n_heads, -1)
        elif orig_ndim == 2:
            hs = hs.view(S, n_heads, -1)
        else:
            raise ValueError(f"Unexpected hidden_states shape: {tuple(orig_shape)}")

        sel = getattr(self, "selection", None) or {}
        spatial_idx = sel.get("spatial_location", None)
        head_idx = sel.get("heads", None)

        if isinstance(head_idx, dict):
            head_idx = head_idx.get(attn.attr_name, None)

        device = hs.device

        def to_index(idx, length):
            if idx is None:
                return torch.arange(length, device=device)
            if isinstance(idx, slice):
                return torch.arange(length, device=device)[idx]
            if torch.is_tensor(idx):
                return idx.to(device=device, dtype=torch.long)
            if isinstance(idx, int):
                return torch.tensor([idx], device=device, dtype=torch.long)
            return torch.tensor(list(idx), device=device, dtype=torch.long)

        spatial_idx = to_index(spatial_idx, S)
        head_idx = to_index(head_idx, n_heads)

        if spatial_idx.numel() == 0 or head_idx.numel() == 0 or factor == 1.0:
            return

        if hs.ndim == 4:
            hs[:, spatial_idx[:, None], head_idx[None, :], :] *= factor
        else:
            hs[spatial_idx[:, None], head_idx[None, :], :] *= factor

        if orig_ndim == 3:
            attn.value = hs.view(orig_shape[0], S, -1)
        else:
            attn.value = hs.view(S, -1)


class FeatureIntervention(DiffusionIntervention):
    """
    Scale specific feature indices using forward hooks.

    .. deprecated::
        This uses register_forward_hook() which is incompatible with
        SAEManagerEdit's .edit() approach. Use SAEFeatureScalingIntervention
        instead for the new workflow.

    Args:
        feature_indices: Indices of features to scale (passed via kwargs).
        factor: Scaling factor (passed via kwargs, default 0.0 for ablation).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def intervene(self, accessor: ModuleAccessor, **kwargs):
        feature_indices = kwargs.get("feature_indices", None)
        scale = kwargs.get("factor", 0.0)
        assert feature_indices is not None, "feature_indices must be provided as kwargs"

        def scale_hook(module, input, output):
            if output.dim() == 2:
                output[:, feature_indices] *= scale
            elif output.dim() == 3:
                output[:, :, feature_indices] *= scale
            elif output.dim() == 4:
                output[:, :, :, feature_indices] *= scale
            else:
                raise ValueError(f"Unexpected output dim: {output.dim()}")
            return output

        accessor.module.register_forward_hook(scale_hook)


class SAEFeatureScalingIntervention(DiffusionIntervention):
    """
    Scale specific SAE feature indices using the value-based approach.

    Compatible with SAEManagerEdit's .edit() workflow. Operates directly on
    ModuleAccessor values (typically sae_block.encoder_out) without hooks.

    This is the recommended way to do feature scaling/ablation with SAEs.

    Args:
        model: The T2IModel instance.
        accessors: List of ModuleAccessor objects (e.g., [sae_block.encoder_out]).
        feature_indices: List of feature indices to scale.
        scale: Scaling factor (default 0.0 for ablation, >1.0 for amplification).

    Example - Ablate features:
        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[sae_block.encoder_out],
            feature_indices=[42, 128, 256],
            scale=0.0,  # ablate
        )

    Example - Amplify features:
        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[sae_block.encoder_out],
            feature_indices=[42],
            scale=5.0,  # 5x amplification
        )

    Example - Use with run_intervention:
        output = run_intervention(
            model, prompts,
            interventions=[intervention],
        )
    """

    def __init__(
        self,
        model: T2IModel,
        accessors: list[ModuleAccessor],
        feature_indices: list[int] | torch.Tensor,
        scale: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, accessors=accessors, **kwargs)
        if isinstance(feature_indices, list):
            self.feature_indices = torch.tensor(feature_indices, dtype=torch.long)
        else:
            self.feature_indices = feature_indices.long()
        self.scale = scale

    def intervene(self, accessor: ModuleAccessor, **kwargs):
        """Scale the specified feature indices on the accessor's value."""
        scale = kwargs.get("scale", self.scale)
        value = accessor.value

        indices = self.feature_indices
        if value.device != indices.device:
            indices = indices.to(value.device)

        if value.dim() == 1:
            # (D,) - single feature vector
            value[indices] = value[indices] * scale
        elif value.dim() == 2:
            # (B, D) or (S, D) - batch of feature vectors
            value[:, indices] = value[:, indices] * scale
        elif value.dim() == 3:
            # (B, S, D) - batch with spatial dimension
            value[:, :, indices] = value[:, :, indices] * scale
        elif value.dim() == 4:
            # (B, S, H, D) or similar
            value[:, :, :, indices] = value[:, :, :, indices] * scale
        else:
            raise ValueError(
                f"Unexpected activation shape: {value.shape} (ndim={value.dim()}). "
                "SAEFeatureScalingIntervention supports 1D-4D tensors."
            )

        accessor.value = value


def run_intervention(
    model: T2IModel,
    prompts: list[str],
    interventions: list[DiffusionIntervention] | None = None,
    **kwargs,
) -> Output:
    """
    Run the model with interventions applied during generation.

    Args:
        model: The T2IModel instance.
        prompts: List of text prompts.
        interventions: List of DiffusionIntervention objects to apply.
        **kwargs: Additional generation kwargs (num_inference_steps, seed, etc.).

    Returns:
        Output object containing generated images.

    Example:
        output = run_intervention(
            model,
            ["A mountain landscape"],
            interventions=[my_intervention],
            num_inference_steps=50,
            seed=42,
        )
    """
    if interventions is None:
        interventions = []

    start_step = 0
    end_step = start_step + 1
    with model.generate(prompts, validate=False, scan=False, **kwargs) as tracer:
        with tracer.iter[start_step:end_step]:
            for intervention in interventions:
                intervention(**kwargs)
            output = model.output.save()
    return Output(preds=output.images)
