import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from t2i_interp.scripts.run_localisation import main

@pytest.fixture
def localisation_cfg():
    return OmegaConf.create({
        "model_key": "test_model",
        "device": "cpu",
        "dtype": "float32",
        "target_heads": [0],
        "target_layer": "test_layer_out",
        "factor": 0.5,
        "start_step": 0,
        "end_step": 2,
        "num_inference_steps": 2,
        "prompt": "test prompt",
        "seed": 42,
        "guidance_scale": 7.5,
        "output_dir": "./tmp_out",
        "sweep_all_layers": False
    })

@patch("t2i_interp.scripts.run_localisation.os.makedirs")
@patch("t2i_interp.scripts.run_localisation.plt")
@patch("t2i_interp.utils.inference.Inference")
@patch("diffusers.StableDiffusionPipeline")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_localisation(mock_wandb, mock_t2imodel, mock_pipeline, mock_inference, mock_plt, mock_makedirs, localisation_cfg):
    import torch
    
    mock_model_instance = MagicMock()
    mock_acc = MagicMock()
    mock_acc.module.heads = 8
    
    # Mock accessors dict
    mock_model_instance.unet.accessors = {
        "test_layer_out": mock_acc,
        "other_layer_out": mock_acc
    }
    
    # Mock baseline image
    mock_model_instance.pipeline.return_value = MagicMock(images=[MagicMock()])
    
    mock_t2imodel.return_value = mock_model_instance

    mock_inference_instance = MagicMock()
    mock_inference_instance.run_inference.return_value = MagicMock(preds=[MagicMock()])
    mock_inference.return_value = mock_inference_instance
    
    main.__wrapped__(localisation_cfg)
    
    mock_t2imodel.assert_called_once()
    mock_inference.assert_called()
    mock_plt.subplots.assert_called()
    mock_plt.savefig.assert_called()
