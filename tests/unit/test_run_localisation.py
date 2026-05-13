import matplotlib.pyplot  # noqa: F401 — needed so @patch("matplotlib.pyplot") resolves
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from t2i_interp.scripts.run_localisation import main


@pytest.fixture
def localisation_cfg(tmp_path):
    return OmegaConf.create(
        {
            "model_key": "test_model",
            "device": "cpu",
            "dtype": "float32",
            "target_heads": [0],
            # The script filters accessors by `"attn2" in name and name.endswith("_out")`
            # then by `cfg.target_layer in name`. Use a realistic SD1.x attn2 path.
            "target_layer": "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
            "factor": 0.5,
            "start_step": 0,
            "end_step": 2,
            "num_inference_steps": 2,
            "prompt": "test prompt",
            "seed": 42,
            "guidance_scale": 7.5,
            "output_dir": str(tmp_path / "out"),
            "sweep_all_layers": False,
            "wandb": {"project": None},
        }
    )


@patch("matplotlib.pyplot")
@patch("t2i_interp.utils.inference.Inference")
@patch("diffusers.StableDiffusionPipeline")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_localisation(
    mock_wandb,
    mock_t2imodel,
    mock_pipeline,
    mock_inference,
    mock_plt,
    localisation_cfg,
):
    # Realistic accessor: name contains "attn2", ends with "_out", module has heads.
    mock_acc = MagicMock()
    mock_acc.module.heads = 8

    mock_model_instance = MagicMock()
    mock_model_instance.unet.accessors = {
        "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out": mock_acc,
    }
    mock_model_instance.pipeline.return_value = MagicMock(images=[MagicMock()])
    mock_t2imodel.return_value = mock_model_instance

    mock_inference_instance = MagicMock()
    mock_inference_instance.run_inference.return_value = MagicMock(preds=[MagicMock()])
    mock_inference.return_value = mock_inference_instance

    # plt.subplots returns (fig, [ax, ax, ...]) — make both unpackable + indexable.
    fake_axes = [MagicMock() for _ in range(8)]
    mock_plt.subplots.return_value = (MagicMock(), fake_axes)

    main.__wrapped__(localisation_cfg)

    mock_t2imodel.assert_called_once()
    mock_inference.assert_called()
    mock_plt.subplots.assert_called()
    mock_plt.savefig.assert_called()
