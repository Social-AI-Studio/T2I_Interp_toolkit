import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from t2i_interp.scripts.run_sae import main

@pytest.fixture
def sae_cfg():
    return OmegaConf.create({
        "model_key": "test_model",
        "device": "cpu",
        "dtype": "float32",
        "saes": "test_saes",
        "target_sae": "test_sae_key",
        "n_top_features": 2,
        "n_features_to_plot": 2,
        "strengths": [1, 2],
        "output_dir": "./tmp_out",
        "prompt": "test prompt",
        "num_inference_steps": 2,
        "guidance_scale": 7.5,
        "seed": 42,
        "spatial_h": 16,
        "spatial_w": 16
    })

@patch("t2i_interp.scripts.run_sae.os.makedirs")
@patch("t2i_interp.scripts.run_sae.plt")
@patch("t2i_interp.utils.inference.Inference")
@patch("t2i_interp.build_sae.build_sae_manager")
@patch("diffusers.AutoPipelineForText2Image")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_sae(mock_wandb, mock_t2imodel, mock_pipeline, mock_build_sae, mock_inference, mock_plt, mock_makedirs, sae_cfg):
    import torch
    
    mock_model_instance = MagicMock()
    mock_t2imodel.return_value = mock_model_instance

    mock_sae_manager = MagicMock()
    # Mock activate run_with_steering
    mock_sae_manager.run_with_steering.return_value = [MagicMock()]
    mock_build_sae.return_value = (mock_sae_manager, MagicMock())
    
    # Mock captured sparse_maps
    mock_inference_instance = MagicMock()
    mock_inference.return_value = mock_inference_instance
    mock_inference_instance.run_inference.return_value = MagicMock(
        preds={"test_sae_key": torch.ones(1, 16 * 16, 10)}
    )

    main.__wrapped__(sae_cfg)
    
    mock_t2imodel.assert_called_once()
    mock_build_sae.assert_called_once()
    mock_inference.assert_called()
    mock_plt.subplots.assert_called()
    mock_plt.savefig.assert_called()
