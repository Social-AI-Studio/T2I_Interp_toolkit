"""End-to-end test of the t2i-sae entry point with all heavy dependencies mocked."""

import matplotlib.pyplot  # noqa: F401 — needed so @patch("matplotlib.pyplot") resolves
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from t2i_interp.scripts.run_sae import main


@pytest.fixture
def sae_cfg(tmp_path):
    """Mirror the real sae/run.yaml schema closely enough that run_sae's
    OmegaConf.to_container + fingerprint code can resolve all fields."""
    return OmegaConf.create(
        {
            "model_key": "test_model",
            "device": "cpu",
            "dtype": "float32",
            # `saes` is a Map[str, SaeConfig] in the real config; matches what
            # build_sae_manager + fingerprint OmegaConf.to_container expect.
            "saes": {
                "model_unet_down_blocks_2_attentions_1_out": {
                    "path": "test.pth",
                    "k": 10,
                    "hidden_dim": 5120,
                }
            },
            "target_sae": "model_unet_down_blocks_2_attentions_1_out",
            "n_top_features": 2,
            "n_features_to_plot": 2,
            "strengths": [-1, 1],
            "output_dir": str(tmp_path / "out"),
            "prompt": "test prompt",
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "seed": 42,
            "spatial_h": 16,
            "spatial_w": 16,
            # wandb subkey: present-but-empty so the script's `cfg.wandb.get("project")`
            # check evaluates falsy and skips W&B init.
            "wandb": {"project": None},
        }
    )


@patch("matplotlib.pyplot")
@patch("t2i_interp.utils.inference.Inference")
@patch("t2i_interp.build_sae.build_sae_manager")
@patch("diffusers.AutoPipelineForText2Image")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_sae(
    mock_wandb,
    mock_t2imodel,
    mock_pipeline,
    mock_build_sae,
    mock_inference,
    mock_plt,
    sae_cfg,
):
    mock_t2imodel.return_value = MagicMock()
    mock_sae_manager = MagicMock()
    mock_build_sae.return_value = (mock_sae_manager, MagicMock())

    # plt.subplots returns (fig, axes) — make the MagicMock unpack-able.
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # The script does:
    #   sparse_maps = output.preds[sae_key].view(cfg.spatial_h, cfg.spatial_w, -1)
    # So preds must be a dict whose value is a tensor that reshapes cleanly to
    # (spatial_h, spatial_w, *) — provide a (256, 10) tensor (256 = 16*16).
    mock_inference_instance = MagicMock()
    mock_inference_instance.run_inference.return_value = MagicMock(
        preds={"model_unet_down_blocks_2_attentions_1_out": torch.ones(256, 10)}
    )
    mock_inference.return_value = mock_inference_instance

    main.__wrapped__(sae_cfg)

    mock_t2imodel.assert_called_once()
    mock_build_sae.assert_called_once()
    mock_inference.assert_called()
    mock_plt.subplots.assert_called()
    mock_plt.savefig.assert_called()
