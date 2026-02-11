"""
Example: Using SAEManagerEdit for SAE interventions

This example shows the new .edit() based approach for:
1. Adding SAEs to a model
2. Tracing SAE activations
3. Intervening on SAE features

Run with: python examples/sae_edit_example.py
"""

# Add parent to path
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from t2Interp import (
    IOType,
    SAEManagerEdit,
    T2IModel,
    add_sae_to_model,
)

# =============================================================================
# Mock SAE for demo (replace with your trained SAE)
# =============================================================================


class DemoSAE:
    """Demo SAE - identity transform for testing."""

    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def to(self, device_or_dtype):
        return self


# =============================================================================
# Example 1: Basic SAE addition and tracing
# =============================================================================


def example_basic_tracing():
    """
    Basic example: Add SAE and trace its activations.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic SAE tracing")
    print("=" * 60)

    # 1. Load model
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)
    print(f"  Loaded model on {device}")

    # 2. Create SAE manager
    manager = SAEManagerEdit(model)

    # 3. Add SAE to a target layer
    sae = DemoSAE()
    sae_block = manager.add_sae(
        target_accessor=model.unet.conv_out,
        sae=sae,
        name="conv_out_sae",
        io_type=IOType.OUTPUT,
    )
    print("  Added SAE to unet.conv_out")

    # 4. Apply edits (wires SAE into computation)
    edited_model = manager.apply_edits()

    # 5. Generate with tracing
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    with edited_model.generate(
        "A beautiful mountain",
        image=test_image,
        num_inference_steps=2,
        seed=42,
        validate=False,
        scan=False,
    ):
        # Access SAE activations
        encoder_input = sae_block.encoder_in.value.save()
        encoder_output = sae_block.encoder_out.value.save()
        decoder_output = sae_block.decoder_out.value.save()

    print(f"  Encoder input shape:  {encoder_input.shape}")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Decoder output shape: {decoder_output.shape}")
    print("  SUCCESS!")


# =============================================================================
# Example 2: Intervening on SAE features
# =============================================================================


def example_intervention():
    """
    Example: Intervene on SAE encoder output (scale features).
    """
    print("\n" + "=" * 60)
    print("Example 2: SAE intervention")
    print("=" * 60)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)

    # Use convenience function
    manager, sae_block = add_sae_to_model(
        model, target_path="unet.conv_out", sae=DemoSAE(), name="my_sae"
    )

    edited_model = manager.edited_model
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    # Generate with intervention
    with edited_model.generate(
        "A beautiful mountain",
        image=test_image,
        num_inference_steps=2,
        seed=42,
        validate=False,
        scan=False,
    ):
        # Get encoder output
        enc_out = sae_block.encoder_out.value

        # Intervene: scale by 0.5
        sae_block.encoder_out.value = enc_out * 0.5

        # Save final output
        _ = edited_model.output.save()

    print("  Generated image with scaled features")
    print("  SUCCESS!")


# =============================================================================
# Example 3: Feature-level intervention (ablation)
# =============================================================================


def example_feature_ablation():
    """
    Example: Ablate specific features in SAE latent space.
    """
    print("\n" + "=" * 60)
    print("Example 3: Feature ablation")
    print("=" * 60)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)

    manager, sae_block = add_sae_to_model(
        model, target_path="unet.conv_out", sae=DemoSAE(), name="my_sae"
    )

    edited_model = manager.edited_model
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    with edited_model.generate(
        "A beautiful mountain",
        image=test_image,
        num_inference_steps=2,
        seed=42,
        validate=False,
        scan=False,
    ):
        # Get encoder output (SAE features)
        features = sae_block.encoder_out.value

        # Ablate first channel (set to zero)
        # For real SAE: this would be specific feature indices
        ablated = features.clone()
        ablated[:, 0, :, :] = 0  # Zero out first channel

        # Apply ablation
        sae_block.encoder_out.value = ablated

        _ = edited_model.output.save()

    print("  Ablated feature channel 0")
    print("  SUCCESS!")


# =============================================================================
# Example 4: Multiple SAEs
# =============================================================================


def example_multiple_saes():
    """
    Example: Add multiple SAEs to different layers.
    """
    print("\n" + "=" * 60)
    print("Example 4: Multiple SAEs")
    print("=" * 60)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)

    manager = SAEManagerEdit(model)

    # Add SAE to conv_out
    sae_block_1 = manager.add_sae(
        target_accessor=model.unet.conv_out,
        sae=DemoSAE(),
        name="conv_out_sae",
    )

    # Add SAE to conv_in
    sae_block_2 = manager.add_sae(
        target_accessor=model.unet.conv_in,
        sae=DemoSAE(),
        name="conv_in_sae",
    )

    print(f"  Registered SAEs: {manager.sae_names}")

    edited_model = manager.apply_edits()
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    with edited_model.generate(
        "A beautiful mountain",
        image=test_image,
        num_inference_steps=2,
        seed=42,
        validate=False,
        scan=False,
    ):
        # Access both SAE outputs
        conv_out_features = sae_block_1.encoder_out.value.save()
        conv_in_features = sae_block_2.encoder_out.value.save()

    print(f"  conv_out SAE features: {conv_out_features.shape}")
    print(f"  conv_in SAE features:  {conv_in_features.shape}")
    print("  SUCCESS!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SAEManagerEdit Examples")
    print("=" * 60)

    try:
        example_basic_tracing()
        example_intervention()
        example_feature_ablation()
        example_multiple_saes()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
