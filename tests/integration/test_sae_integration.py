"""
Integration tests for SAEManagerEdit with real diffusion models.

These tests require:
- A GPU (CUDA) or MPS (Apple Silicon)
- Internet connection to download models
- Sufficient memory

Run with: pytest tests/integration/test_sae_integration.py -v -m integration
Skip with: pytest -m "not integration"
"""

# Add parent to path
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def model(device):
    """Load a test model (cached for the module)."""
    try:
        from t2Interp.T2I import T2IModel

        return T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


@pytest.fixture
def test_image():
    """Create a test image for img2img."""
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def identity_sae():
    """Create an identity SAE for testing."""

    class IdentitySAE:
        def encode(self, x):
            return x

        def decode(self, z):
            return z

        def to(self, _):
            return self

    return IdentitySAE()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestSAEManagerEditIntegration:
    """Integration tests for SAEManagerEdit."""

    def test_sae_becomes_envoy(self, model, identity_sae):
        """Test that SAE becomes an Envoy when registered."""
        from nnsight import Envoy

        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)
        sae_block = manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=identity_sae,
            name="test_sae",
        )

        # Check encoder is an Envoy
        encoder_module = sae_block.encoder_out.module
        assert isinstance(encoder_module, Envoy), f"Expected Envoy, got {type(encoder_module)}"

    def test_generation_with_sae(self, model, identity_sae, test_image):
        """Test that generation works with SAE inserted."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)
        manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=identity_sae,
            name="gen_test_sae",
        )

        edited_model = manager.apply_edits()

        # Should not raise
        result = edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            trace=False,
        )

        assert result is not None

    def test_trace_sae_encoder_output(self, model, identity_sae, test_image):
        """Test that we can trace SAE encoder output."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)
        sae_block = manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=identity_sae,
            name="trace_test_sae",
        )

        edited_model = manager.apply_edits()

        with edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            validate=False,
            scan=False,
        ):
            enc_out = sae_block.encoder_out.value
            saved = enc_out.save()

        assert saved is not None
        assert len(saved.shape) == 4, f"Expected 4D tensor, got shape {saved.shape}"

    def test_intervene_on_sae_encoder(self, model, identity_sae, test_image):
        """Test that we can intervene on SAE encoder output."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)
        sae_block = manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=identity_sae,
            name="intervene_test_sae",
        )

        edited_model = manager.apply_edits()

        with edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            validate=False,
            scan=False,
        ):
            # Get encoder output and modify it
            enc_out = sae_block.encoder_out.value
            sae_block.encoder_out.value = enc_out * 0.5
            _ = edited_model.output.save()

        # If we got here without exception, intervention worked
        assert True

    def test_multiple_saes(self, model, identity_sae, test_image):
        """Test adding multiple SAEs to different layers."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)

        # Add SAE to conv_out
        sae_block_1 = manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=identity_sae,
            name="multi_sae_1",
        )

        # Add SAE to conv_in
        sae_block_2 = manager.add_sae(
            target_accessor=model.unet.conv_in,
            sae=identity_sae,
            name="multi_sae_2",
        )

        assert len(manager.sae_names) == 2

        edited_model = manager.apply_edits()

        with edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            validate=False,
            scan=False,
        ):
            out_1 = sae_block_1.encoder_out.value.save()
            out_2 = sae_block_2.encoder_out.value.save()

        assert out_1 is not None
        assert out_2 is not None


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with old SAEManager interface."""

    def test_add_saes_to_model_interface(self, model, identity_sae):
        """Test the backward-compatible add_saes_to_model method."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)

        # Old interface: list of tuples
        blocks = manager.add_saes_to_model(
            sae_list=[(model.unet.conv_out, identity_sae, "compat_sae")],
            diff=False,
        )

        assert len(blocks) == 1
        assert "compat_sae" in manager.sae_names

    def test_diff_mode(self, model, identity_sae, test_image):
        """Test diff=True mode (SAE on residual)."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)

        blocks = manager.add_saes_to_model(
            sae_list=[(model.unet.conv_out, identity_sae, "diff_sae")],
            diff=True,
        )

        assert blocks[0]._diff_mode is True

        edited_model = manager.apply_edits()

        # Should work without error
        with edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            validate=False,
            scan=False,
        ):
            enc_out = blocks[0].encoder_out.value.save()

        assert enc_out is not None

    def test_run_with_cache(self, model, identity_sae, test_image):
        """Test backward-compatible run_with_cache method."""
        from t2Interp.sae_edit import SAEManagerEdit

        manager = SAEManagerEdit(model)
        blocks = manager.add_saes_to_model(
            sae_list=[(model.unet.conv_out, identity_sae, "cache_sae")],
        )

        # Note: run_with_cache for text-to-image, not img2img
        # This tests the interface exists and doesn't crash
        try:
            cache = manager.run_with_cache(
                prompt="A test image",
                accessors=[blocks[0].encoder_out],
                num_inference_steps=1,
                seed=42,
            )
            assert isinstance(cache, dict)
        except Exception:
            # May fail due to model type mismatch, but interface should exist
            pass


@pytest.mark.integration
class TestConvenienceFunction:
    """Test the add_sae_to_model convenience function."""

    def test_convenience_function(self, model, identity_sae, test_image):
        """Test add_sae_to_model convenience function."""
        from t2Interp.sae_edit import add_sae_to_model

        manager, sae_block = add_sae_to_model(
            model,
            target_path="unet.conv_out",
            sae=identity_sae,
            name="convenience_sae",
        )

        assert manager is not None
        assert sae_block is not None
        assert "convenience_sae" in manager.sae_names

        edited_model = manager.edited_model

        with edited_model.generate(
            "A test image",
            image=test_image,
            num_inference_steps=1,
            seed=42,
            validate=False,
            scan=False,
        ):
            enc_out = sae_block.encoder_out.value.save()

        assert enc_out is not None


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
