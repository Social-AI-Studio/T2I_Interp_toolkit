"""
Unit tests for intervention classes.

Run with: pytest tests/unit/test_interventions.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from t2Interp.accessors import IOType
from t2Interp.intervention import (
    AddVectorIntervention,
    DiffusionIntervention,
    SAEFeatureScalingIntervention,
)


# =============================================================================
# Mock classes
# =============================================================================


class MockAccessor:
    """Mock accessor that stores a tensor as .value for testing."""

    def __init__(self, tensor: torch.Tensor, attr_name: str = "test"):
        self._value = tensor
        self.attr_name = attr_name
        self.module = MagicMock()
        self.io_type = IOType.OUTPUT

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @value.setter
    def value(self, new: torch.Tensor):
        self._value = new


class MockModel:
    """Minimal mock model for intervention construction."""

    pass


# =============================================================================
# Tests: DiffusionIntervention base class
# =============================================================================


class TestDiffusionIntervention:
    """Tests for the base DiffusionIntervention class."""

    def test_creation(self):
        model = MockModel()
        accessor = MockAccessor(torch.randn(4))
        intervention = DiffusionIntervention(model=model, accessors=[accessor])

        assert intervention.model is model
        assert len(intervention.accessors) == 1
        assert intervention.start_step == 0
        assert intervention.end_step == 50

    def test_fields_returns_empty(self):
        assert DiffusionIntervention.fields() == []

    def test_call_iterates_accessors(self):
        model = MockModel()
        accessors = [MockAccessor(torch.randn(4)) for _ in range(3)]
        intervention = DiffusionIntervention(model=model, accessors=accessors)

        # Base class intervene is a no-op, should not raise
        intervention()


# =============================================================================
# Tests: SAEFeatureScalingIntervention
# =============================================================================


class TestSAEFeatureScalingIntervention:
    """Tests for SAEFeatureScalingIntervention."""

    def test_ablation_2d(self):
        """Test feature ablation (scale=0) on 2D tensor (B, D)."""
        tensor = torch.ones(2, 8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[0, 3, 7],
            scale=0.0,
        )
        intervention()

        result = accessor.value
        # Ablated features should be 0
        assert result[0, 0].item() == 0.0
        assert result[0, 3].item() == 0.0
        assert result[0, 7].item() == 0.0
        # Non-ablated features should remain 1
        assert result[0, 1].item() == 1.0
        assert result[0, 2].item() == 1.0

    def test_amplification_2d(self):
        """Test feature amplification (scale=5) on 2D tensor."""
        tensor = torch.ones(2, 8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[2, 5],
            scale=5.0,
        )
        intervention()

        result = accessor.value
        assert result[0, 2].item() == 5.0
        assert result[0, 5].item() == 5.0
        assert result[0, 0].item() == 1.0  # unchanged

    def test_ablation_3d(self):
        """Test feature ablation on 3D tensor (B, S, D)."""
        tensor = torch.ones(2, 4, 8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[1, 6],
            scale=0.0,
        )
        intervention()

        result = accessor.value
        assert result[0, 0, 1].item() == 0.0
        assert result[1, 3, 6].item() == 0.0
        assert result[0, 0, 0].item() == 1.0  # unchanged

    def test_ablation_1d(self):
        """Test feature ablation on 1D tensor (D,)."""
        tensor = torch.ones(8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[0, 4],
            scale=0.0,
        )
        intervention()

        result = accessor.value
        assert result[0].item() == 0.0
        assert result[4].item() == 0.0
        assert result[1].item() == 1.0

    def test_multiple_accessors(self):
        """Test intervention applied to multiple accessors."""
        tensors = [torch.ones(2, 8) for _ in range(3)]
        accessors = [MockAccessor(t, f"acc_{i}") for i, t in enumerate(tensors)]
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=accessors,
            feature_indices=[0],
            scale=0.0,
        )
        intervention()

        for acc in accessors:
            assert acc.value[0, 0].item() == 0.0
            assert acc.value[0, 1].item() == 1.0

    def test_scale_override_via_kwargs(self):
        """Test that scale can be overridden via kwargs."""
        tensor = torch.ones(2, 8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[0],
            scale=0.0,  # default is ablation
        )
        intervention(scale=3.0)  # override to amplification

        assert accessor.value[0, 0].item() == 3.0

    def test_tensor_feature_indices(self):
        """Test passing feature_indices as a tensor."""
        tensor = torch.ones(2, 8)
        accessor = MockAccessor(tensor)
        model = MockModel()

        indices = torch.tensor([2, 5])
        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=indices,
            scale=0.0,
        )
        intervention()

        assert accessor.value[0, 2].item() == 0.0
        assert accessor.value[0, 5].item() == 0.0
        assert accessor.value[0, 0].item() == 1.0

    def test_identity_scale(self):
        """Test that scale=1.0 leaves values unchanged."""
        tensor = torch.randn(2, 8)
        original = tensor.clone()
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = SAEFeatureScalingIntervention(
            model=model,
            accessors=[accessor],
            feature_indices=[0, 1, 2, 3],
            scale=1.0,
        )
        intervention()

        assert torch.allclose(accessor.value, original)


# =============================================================================
# Tests: AddVectorIntervention
# =============================================================================


class TestAddVectorIntervention:
    """Tests for AddVectorIntervention."""

    def test_add_vector_default_alpha(self):
        """Test adding steering vector with default alpha=1.0."""
        tensor = torch.zeros(2, 4)
        steering = torch.ones(2, 4)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = AddVectorIntervention(
            steering_vec=steering,
            model=model,
            accessors=[accessor],
        )
        intervention()

        assert torch.allclose(accessor.value, steering)

    def test_add_vector_custom_alpha(self):
        """Test adding steering vector with custom alpha."""
        tensor = torch.zeros(2, 4)
        steering = torch.ones(2, 4)
        accessor = MockAccessor(tensor)
        model = MockModel()

        intervention = AddVectorIntervention(
            steering_vec=steering,
            model=model,
            accessors=[accessor],
        )
        intervention(alpha=2.5)

        expected = 2.5 * steering
        assert torch.allclose(accessor.value, expected)


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
