"""End-to-end tests of t2i-steer for CAA + KSteer with heavy deps mocked."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from t2i_interp.scripts.run_steer import main


@pytest.fixture
def steer_cfg(tmp_path):
    """Mirrors the steer/run.yaml schema (layer_names + label_col + pos/neg)."""
    return OmegaConf.create(
        {
            "model_key": "test_model",
            "device": "cpu",
            "dtype": "float32",
            "dataset_name": "test_dataset",
            "prompt_col": "prompt",
            # Provide BOTH layer_name (singular, fallback) and layer_names (list).
            "layer_name": "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2",
            "layer_names": [
                "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2",
            ],
            "label_col": None,  # skip the str→int label remapping branch
            "pos_labels": [1],
            "neg_labels": [0],
            "save_dir": str(tmp_path / "cache"),
            "batch_size": 1,
            "guidance_scale": 7.5,
            "conditional_only": True,
            "num_inference_steps": 1,
            "capture_step_index": 0,
            "shuffle": False,
            "train_steps": 1,
            "lr": 1e-4,
            "log_steps": 1,
            "alpha": 1,
            "steer_steps": 1,
            "output_dir": str(tmp_path / "out"),
            "prompts": ["test prompt"],
            "max_samples": 4,
            "steer_type": "caa",
            "use_baseline": False,
            "delete_cache": False,
            "metrics": {},
            "wandb": {"project": None},
        }
    )


def _make_loader_with_iterate():
    """Build a mock loader whose .iterate() returns reusable batches.

    The script calls `next(loader.iterate())` once to peek (input_dim
    detection), then iterates again to collect pos/neg samples. Provide
    enough labelled batches that both runs succeed.
    """

    def _gen():
        for label_val in (1, 0, 1, 0):
            yield (torch.randn(1, 10), torch.tensor([label_val]))

    loader = MagicMock()
    loader.iterate.side_effect = lambda: _gen()
    loader.reset = MagicMock()
    return loader


@patch("t2i_interp.utils.T2I.buffer.PairedLoader")
@patch("t2i_interp.linear_steering.CAA")
@patch("t2i_interp.utils.T2I.buffer.ActivationsDataloader")
@patch("t2i_interp.utils.T2I.collect_latents.collect_latents")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_steer_caa(
    mock_wandb,
    mock_t2imodel,
    mock_load_dataset,
    mock_collect,
    mock_dataloader,
    mock_caa,
    mock_paired,
    steer_cfg,
    tmp_path,
):
    # Dataset with both train + validation splits (script does ds_full.get).
    fake_ds = MagicMock()
    fake_ds.__len__.return_value = 10
    fake_ds.__getitem__.return_value = {"prompt": "x"}  # for the label-col str check
    fake_ds.select.return_value = fake_ds
    mock_load_dataset.return_value = {"train": fake_ds, "validation": fake_ds}

    # Make the tar files the script's _find_tars() expects actually exist
    # so the `all_exist` short-circuit fires and collect_latents is skipped.
    cache_dir = tmp_path / "cache"
    (cache_dir / "train").mkdir(parents=True, exist_ok=True)
    (cache_dir / "val").mkdir(parents=True, exist_ok=True)
    tar_name = "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2_prompt.tar"
    (cache_dir / "train" / tar_name).touch()
    (cache_dir / "val" / tar_name).touch()

    loader = _make_loader_with_iterate()
    mock_dataloader.return_value = loader

    # CAA fit/steer mocks
    mock_caa_instance = MagicMock()
    mock_caa_instance.steering_vecs = {
        steer_cfg.layer_names[0]: torch.randn(10),
    }
    mock_caa_instance.steer.return_value = [MagicMock()]  # one fake image
    mock_caa.return_value = mock_caa_instance

    main.__wrapped__(steer_cfg)

    mock_t2imodel.assert_called_once()
    mock_load_dataset.assert_called_once_with("test_dataset")
    mock_caa_instance.fit.assert_called()
    mock_caa_instance.steer.assert_called()


@patch("t2i_interp.utils.T2I.buffer.PairedLoader")
@patch("t2i_interp.linear_steering.KSteer")
@patch("t2i_interp.mapper.MLPMapper")
@patch("t2i_interp.utils.training.Training")
@patch("t2i_interp.utils.T2I.buffer.ActivationsDataloader")
@patch("t2i_interp.utils.T2I.collect_latents.collect_latents")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_steer_ksteer(
    mock_wandb,
    mock_t2imodel,
    mock_load_dataset,
    mock_collect,
    mock_dataloader,
    mock_training,
    mock_mapper,
    mock_ksteer,
    mock_paired,
    steer_cfg,
    tmp_path,
):
    steer_cfg.steer_type = "ksteer"

    fake_ds = MagicMock()
    fake_ds.__len__.return_value = 10
    fake_ds.__getitem__.return_value = {"prompt": "x"}
    fake_ds.select.return_value = fake_ds
    mock_load_dataset.return_value = {"train": fake_ds, "validation": fake_ds}

    # Pre-create the tar files so the latent-collection short-circuit fires.
    cache_dir = tmp_path / "cache"
    (cache_dir / "train").mkdir(parents=True, exist_ok=True)
    (cache_dir / "val").mkdir(parents=True, exist_ok=True)
    tar_name = "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2_prompt.tar"
    (cache_dir / "train" / tar_name).touch()
    (cache_dir / "val" / tar_name).touch()

    mock_dataloader.return_value = _make_loader_with_iterate()

    mock_ksteer_instance = MagicMock()
    mock_ksteer_instance.steer.return_value = [MagicMock()]
    mock_ksteer.return_value = mock_ksteer_instance

    main.__wrapped__(steer_cfg)

    mock_ksteer.assert_called_once()
    mock_training.assert_called()
    mock_ksteer_instance.steer.assert_called()
