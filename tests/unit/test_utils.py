"""Unit tests for utility functions."""

import torch

from utils.output import Output


def test_output_creation():
    """Test Output dataclass creation."""
    output = Output()
    assert output.preds is None
    assert output.baselines is None
    assert output.run_metadata is None
    assert output.best_ckpt is None


def test_output_with_data():
    """Test Output with actual data."""
    preds = [1, 2, 3]
    baselines = [0, 0, 0]
    metadata = {"lr": 1e-5, "steps": 100}

    output = Output(preds=preds, baselines=baselines, run_metadata=metadata)

    assert output.preds == preds
    assert output.baselines == baselines
    assert output.run_metadata == metadata


def test_output_checkpoint_storage():
    """Test storing checkpoint in Output."""
    checkpoint = {"model_state": torch.randn(10, 10)}
    output = Output(best_ckpt=checkpoint)

    assert output.best_ckpt == checkpoint
    assert "model_state" in output.best_ckpt


# Add more utility tests as needed
class TestBatchIterator:
    """Tests for BatchIterator utility."""

    def test_batch_iterator_basic(self):
        """Test basic batch iteration."""
        # TODO: Implement when BatchIterator is properly exposed
        pass

    def test_batch_iterator_uneven_batches(self):
        """Test batch iterator with data not divisible by batch size."""
        # TODO: Implement
        pass


class TestActivationConfig:
    """Tests for ActivationConfig."""

    def test_config_creation(self):
        """Test creating ActivationConfig."""
        # TODO: Implement
        pass
