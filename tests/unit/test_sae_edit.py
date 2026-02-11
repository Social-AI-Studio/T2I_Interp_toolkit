"""
Unit tests for SAEManagerEdit - the .edit() based SAE management.

Run with: pytest tests/unit/test_sae_edit.py -v
"""

# Add parent to path for imports
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from t2Interp.accessors import IOType, ModuleAccessor
from t2Interp.blocks import SAEBlock
from t2Interp.sae_edit import (
    SAEManagerEdit,
    SAEModule,
    _DecoderModule,
    _EncoderModule,
    add_sae_to_model,
)

# =============================================================================
# Mock classes for unit testing (no model loading required)
# =============================================================================


class MockDictionary:
    """Mock SAE that mimics dictionary_learning.Dictionary interface."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 16):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._encoder = nn.Linear(input_dim, latent_dim)
        self._decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return torch.relu(self._encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self._decoder(z)

    def to(self, device_or_dtype):
        """Move to device/dtype."""
        self._encoder = self._encoder.to(device_or_dtype)
        self._decoder = self._decoder.to(device_or_dtype)
        return self


class IdentityDictionary:
    """Identity SAE for testing (pass-through)."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to(self, device_or_dtype):
        return self


# =============================================================================
# Unit Tests: SAEModule
# =============================================================================


class TestSAEModule:
    """Tests for SAEModule wrapper class."""

    def test_sae_module_creation(self):
        """Test SAEModule properly wraps a Dictionary."""
        sae = IdentityDictionary()
        module = SAEModule(sae, name="test_sae")

        assert hasattr(module, "encoder"), "Should have encoder submodule"
        assert hasattr(module, "decoder"), "Should have decoder submodule"
        assert isinstance(module.encoder, nn.Module), "Encoder should be nn.Module"
        assert isinstance(module.decoder, nn.Module), "Decoder should be nn.Module"
        assert module.name == "test_sae"

    def test_sae_module_forward(self):
        """Test SAEModule forward pass."""
        sae = IdentityDictionary()
        module = SAEModule(sae, name="test")

        x = torch.randn(2, 4)
        out = module(x)

        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"
        assert torch.allclose(out, x), "Identity SAE should return input unchanged"

    def test_sae_module_encode_decode(self):
        """Test encode/decode methods."""
        sae = IdentityDictionary()
        module = SAEModule(sae, name="test")

        x = torch.randn(2, 4)
        encoded = module.encode(x)
        decoded = module.decode(encoded)

        assert torch.allclose(encoded, x)
        assert torch.allclose(decoded, x)

    def test_sae_module_with_real_sae(self):
        """Test SAEModule with a real (non-identity) SAE."""
        sae = MockDictionary(input_dim=8, latent_dim=32)
        module = SAEModule(sae, name="real_sae")

        x = torch.randn(4, 8)
        encoded = module.encode(x)
        decoded = module.decode(encoded)
        full_out = module(x)

        assert encoded.shape == (4, 32), f"Encoded shape wrong: {encoded.shape}"
        assert decoded.shape == (4, 8), f"Decoded shape wrong: {decoded.shape}"
        assert full_out.shape == x.shape


class TestEncoderDecoderModules:
    """Tests for _EncoderModule and _DecoderModule."""

    def test_encoder_module(self):
        """Test _EncoderModule calls sae.encode()."""
        sae = MockDictionary(input_dim=4, latent_dim=8)
        encoder = _EncoderModule(sae)

        x = torch.randn(2, 4)
        out = encoder(x)

        assert out.shape == (2, 8)
        # Should match direct encode call
        expected = sae.encode(x)
        assert torch.allclose(out, expected)

    def test_decoder_module(self):
        """Test _DecoderModule calls sae.decode()."""
        sae = MockDictionary(input_dim=4, latent_dim=8)
        decoder = _DecoderModule(sae)

        z = torch.randn(2, 8)
        out = decoder(z)

        assert out.shape == (2, 4)
        # Should match direct decode call
        expected = sae.decode(z)
        assert torch.allclose(out, expected)


# =============================================================================
# Unit Tests: SAEBlock
# =============================================================================


class TestSAEBlock:
    """Tests for SAEBlock dataclass."""

    def test_sae_block_creation(self):
        """Test SAEBlock can be created with accessors."""
        # Create mock accessors
        mock_module = nn.Linear(4, 4)

        block = SAEBlock(
            encoder_in=ModuleAccessor(mock_module, "enc_in", IOType.INPUT),
            encoder_out=ModuleAccessor(mock_module, "enc_out", IOType.OUTPUT),
            decoder_in=ModuleAccessor(mock_module, "dec_in", IOType.INPUT),
            decoder_out=ModuleAccessor(mock_module, "dec_out", IOType.OUTPUT),
        )

        assert block.encoder_in is not None
        assert block.encoder_out is not None
        assert block.decoder_in is not None
        assert block.decoder_out is not None

    def test_sae_block_summary(self):
        """Test SAEBlock summary method."""
        block = SAEBlock()
        summary = block.summary()

        assert "encoder_in" in summary
        assert "encoder_out" in summary
        assert "decoder_in" in summary
        assert "decoder_out" in summary


# =============================================================================
# Unit Tests: ModuleAccessor
# =============================================================================


class TestModuleAccessor:
    """Tests for ModuleAccessor class."""

    def test_accessor_creation(self):
        """Test ModuleAccessor creation."""
        module = nn.Linear(4, 4)
        accessor = ModuleAccessor(module, "test_module", IOType.OUTPUT)

        assert accessor.module is module
        assert accessor.attr_name == "test_module"
        assert accessor.io_type == IOType.OUTPUT
        assert accessor.returns_tuple is False

    def test_accessor_with_tuple(self):
        """Test ModuleAccessor with returns_tuple=True."""
        module = nn.Linear(4, 4)
        accessor = ModuleAccessor(module, "test", IOType.OUTPUT, returns_tuple=True)

        assert accessor.returns_tuple is True


# =============================================================================
# Unit Tests: IOType
# =============================================================================


class TestIOType:
    """Tests for IOType enum."""

    def test_io_type_values(self):
        """Test IOType enum values."""
        assert IOType.INPUT.value == "input"
        assert IOType.OUTPUT.value == "output"

    def test_io_type_comparison(self):
        """Test IOType can be compared."""
        assert IOType.INPUT == IOType.INPUT
        assert IOType.OUTPUT == IOType.OUTPUT
        assert IOType.INPUT != IOType.OUTPUT


# =============================================================================
# Integration Tests (require model - marked as slow)
# =============================================================================


@pytest.mark.slow
class TestSAEManagerEditIntegration:
    """Integration tests that require loading a model."""

    @pytest.fixture
    def model(self):
        """Load a test model."""
        try:
            from t2Interp.T2I import T2IModel

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            return T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

    def test_sae_manager_creation(self, model):
        """Test SAEManagerEdit can be created."""
        manager = SAEManagerEdit(model)

        assert manager.model is model
        assert len(manager._sae_blocks) == 0
        assert len(manager._sae_modules) == 0
        assert manager._edited_model is None

    def test_add_sae(self, model):
        """Test adding an SAE to the model."""
        manager = SAEManagerEdit(model)
        sae = IdentityDictionary()

        sae_block = manager.add_sae(
            target_accessor=model.unet.conv_out,
            sae=sae,
            name="test_sae",
        )

        assert sae_block is not None
        assert "test_sae" in manager.sae_names
        assert manager.get_sae("test_sae") is sae_block

    def test_add_saes_to_model_backward_compat(self, model):
        """Test backward-compatible add_saes_to_model method."""
        manager = SAEManagerEdit(model)
        sae = IdentityDictionary()

        blocks = manager.add_saes_to_model(
            sae_list=[(model.unet.conv_out, sae, "compat_sae")],
            diff=False,
        )

        assert len(blocks) == 1
        assert "compat_sae" in manager.sae_names

    def test_add_saes_with_diff_mode(self, model):
        """Test adding SAE with diff=True mode."""
        manager = SAEManagerEdit(model)
        sae = IdentityDictionary()

        blocks = manager.add_saes_to_model(
            sae_list=[(model.unet.conv_out, sae, "diff_sae")],
            diff=True,
        )

        assert len(blocks) == 1
        assert blocks[0]._diff_mode is True

    def test_clear(self, model):
        """Test clearing all SAEs."""
        manager = SAEManagerEdit(model)
        sae = IdentityDictionary()

        manager.add_sae(model.unet.conv_out, sae, "test_sae")
        assert len(manager.sae_names) == 1

        manager.clear()
        assert len(manager.sae_names) == 0
        assert manager._edited_model is None


@pytest.mark.slow
class TestConvenienceFunction:
    """Tests for add_sae_to_model convenience function."""

    @pytest.fixture
    def model(self):
        """Load a test model."""
        try:
            from t2Interp.T2I import T2IModel

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            return T2IModel("segmind/tiny-sd", device=device, dtype=torch.float16)
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

    def test_add_sae_to_model(self, model):
        """Test add_sae_to_model convenience function."""
        sae = IdentityDictionary()

        manager, sae_block = add_sae_to_model(
            model,
            target_path="unet.conv_out",
            sae=sae,
            name="conv_sae",
        )

        assert manager is not None
        assert sae_block is not None
        assert "conv_sae" in manager.sae_names

    def test_add_sae_to_model_auto_name(self, model):
        """Test auto-generated name when name=None."""
        sae = IdentityDictionary()

        manager, sae_block = add_sae_to_model(
            model,
            target_path="unet.conv_out",
            sae=sae,
        )

        assert "unet_conv_out_sae" in manager.sae_names


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
