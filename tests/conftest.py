"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Return CPU device for testing (avoid GPU requirements)."""
    return torch.device("cpu")


@pytest.fixture
def sample_config():
    """Return a sample training configuration."""
    return {
        "train_steps": 10,
        "lr": 1e-5,
        "batch_size": 2,
        "autocast_dtype": torch.float32,
        "training_device": "cpu",
        "data_device": "cpu",
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test runs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_dataset():
    """Return a minimal mock dataset for testing."""
    return {
        "train": [
            {"image": "path/to/image1.png", "race": "white"},
            {"image": "path/to/image2.png", "race": "black"},
        ],
        "val": [
            {"image": "path/to/image3.png", "race": "asian"},
        ],
    }
