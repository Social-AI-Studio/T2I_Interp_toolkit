import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from t2i_interp.scripts.run_stitch import main

@pytest.fixture
def stitch_cfg():
    return OmegaConf.create({
        "model_key": "test_model",
        "device": "cpu",
        "dtype": "float32",
        "dataset_name": "test_dataset",
        "layer_a": "layer_a",
        "layer_b": "layer_b",
        "prompt_col_a": "prompt_a",
        "prompt_col_b": "prompt_b",
        "save_dir": "./tmp",
        "batch_size": 1,
        "guidance_scale": 7.5,
        "conditional_only": True,
        "input_dim": 10,
        "hidden_dim": 10,
        "output_dim": 10,
        "num_steps": 1,
        "lr": 1e-4,
        "log_steps": 1,
        "inject_steps": [0],
        "num_inference_steps": 2,
        "output_dir": "./tmp_out",
        "prompts": ["test prompt"],
        "max_samples": 2
    })

@patch("t2i_interp.scripts.run_stitch.os.makedirs")
@patch("t2i_interp.stitch.Stitcher")
@patch("t2i_interp.utils.training.Training")
@patch("t2i_interp.utils.inference.Inference")
@patch("t2i_interp.mapper.MLPMapper")
@patch("t2i_interp.utils.T2I.buffer.PairedLoader")
@patch("t2i_interp.utils.T2I.buffer.ActivationsDataloader")
@patch("t2i_interp.utils.T2I.collect_latents.collect_latents")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_stitch(mock_wandb, mock_t2imodel, mock_load_dataset, mock_collect, mock_dataloader, mock_pairedloader, mock_mapper, mock_inference, mock_training, mock_stitcher, mock_makedirs, stitch_cfg):
    # Mock dataset
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10
    mock_load_dataset.return_value = {"train": mock_ds}
    
    mock_stitcher_instance = MagicMock()
    mock_stitcher.return_value = mock_stitcher_instance
    
    mock_inference_instance = MagicMock()
    mock_inference_instance.run_inference.return_value = MagicMock(preds=([MagicMock()], []))
    mock_inference.return_value = mock_inference_instance
    
    main.__wrapped__(stitch_cfg)
    
    mock_t2imodel.assert_called_once()
    mock_load_dataset.assert_called_once()
    mock_collect.assert_called()
    mock_pairedloader.assert_called()
    mock_stitcher_instance.train_mapper.assert_called_once()
    mock_inference_instance.run_inference.assert_called()
