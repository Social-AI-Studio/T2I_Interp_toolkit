import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from t2i_interp.scripts.run_steer import main
import torch

@pytest.fixture
def steer_cfg():
    return OmegaConf.create({
        "model_key": "test_model",
        "device": "cpu",
        "dtype": "float32",
        "dataset_name": "test_dataset",
        "prompt_col": "prompt",
        "layer_name": "unet.down_blocks",
        "save_dir": "./tmp_latents",
        "batch_size": 1,
        "train_steps": 1,
        "lr": 1e-4,
        "alpha": 1,
        "steer_steps": 1,
        "output_dir": "./tmp_output",
        "prompts": ["test prompt"],
        "max_samples": 2,
        "steer_type": "caa"
    })

@patch("t2i_interp.scripts.run_steer.os.makedirs")
@patch("t2i_interp.linear_steering.CAA")
@patch("t2i_interp.utils.T2I.buffer.ActivationsDataloader")
@patch("t2i_interp.utils.T2I.collect_latents.collect_latents")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_steer_caa(mock_wandb, mock_t2imodel, mock_load_dataset, mock_collect, mock_dataloader, mock_caa, mock_makedirs, steer_cfg):
    # Mock dataset
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10
    
    mock_load_dataset.return_value = {"train": mock_ds}
    
    # Mock dataloader iterate
    mock_loader_instance = MagicMock()
    # Batch data: (act, label)
    act = torch.randn(2, 10)
    label = torch.tensor([1, 0])
    mock_loader_instance.iterate.return_value = iter([(act, label)])
    mock_dataloader.return_value = mock_loader_instance
    
    # Mock CAA
    mock_caa_instance = MagicMock()
    mock_caa_instance.steer.return_value = [MagicMock()] # Return a mock image
    mock_caa.return_value = mock_caa_instance

    main.__wrapped__(steer_cfg)
    
    mock_t2imodel.assert_called_once()
    mock_load_dataset.assert_called_once_with("test_dataset")
    mock_collect.assert_called()
    mock_caa_instance.fit.assert_called_once()
    mock_caa_instance.steer.assert_called_once()

@patch("t2i_interp.scripts.run_steer.os.makedirs")
@patch("t2i_interp.linear_steering.KSteer")
@patch("t2i_interp.mapper.MLPMapper")
@patch("t2i_interp.utils.training.Training")
@patch("t2i_interp.utils.T2I.buffer.ActivationsDataloader")
@patch("t2i_interp.utils.T2I.collect_latents.collect_latents")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
def test_run_steer_ksteer(mock_t2imodel, mock_load_dataset, mock_collect, mock_dataloader, mock_training, mock_mapper, mock_ksteer, mock_makedirs, steer_cfg):
    steer_cfg.steer_type = "ksteer"
    
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10
    mock_load_dataset.return_value = {"train": mock_ds}
    
    mock_loader_instance = MagicMock()
    act = torch.randn(2, 10)
    label = torch.tensor([1, 0])
    mock_loader_instance.iterate.return_value = iter([(act, label)])
    mock_dataloader.return_value = mock_loader_instance
    
    mock_ksteer_instance = MagicMock()
    mock_ksteer_instance.steer.return_value = [MagicMock()]
    mock_ksteer.return_value = mock_ksteer_instance
    
    main.__wrapped__(steer_cfg)
    
    mock_ksteer.assert_called_once()
    mock_training.assert_called_once()
    mock_ksteer_instance.steer.assert_called_once()
