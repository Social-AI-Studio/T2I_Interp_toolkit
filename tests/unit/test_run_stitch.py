"""End-to-end test of the t2i-stitch entry point with heavy deps mocked.

Uses the `collect_to_memory=True` code path so the test doesn't need to
materialize tar files on disk.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from t2i_interp.scripts.run_stitch import main


@pytest.fixture
def stitch_cfg(tmp_path):
    return OmegaConf.create(
        {
            "model_key": "test_model",
            "device": "cpu",
            "dtype": "float32",
            "dataset_name": "test_dataset",
            "layer_a": "layer_a",
            "layer_b": "layer_b",
            "prompt_col_a": "prompt_a",
            "prompt_col_b": "prompt_b",
            "save_dir": str(tmp_path / "cache"),
            "batch_size": 1,
            "guidance_scale": 7.5,
            "conditional_only": True,
            "capture_step_index": 0,
            "num_inference_steps_a": 1,
            "num_inference_steps_b": 1,
            "input_dim": 10,
            "hidden_dim": 10,
            "output_dim": 10,
            "num_steps": 1,
            "lr": 1e-4,
            "log_steps": 1,
            "loader_batch_size": 2,
            "collect_to_memory": True,  # ← skip tar-file path
            "use_gpu_cache": False,
            "gpu_cache_dtype": "float32",
            "inject_steps": [0],
            "num_inference_steps": 2,
            "output_dir": str(tmp_path / "out"),
            "prompts": ["test prompt"],
            "max_samples": 4,
            "mode": "train",
            "wandb": {"project": None},
        }
    )


@patch("t2i_interp.utils.T2I.collect_latents.collect_latents_inmemory")
@patch("t2i_interp.utils.inference.Inference")
@patch("t2i_interp.utils.training.Training")
@patch("t2i_interp.stitch.Stitcher")
@patch("datasets.load_dataset")
@patch("t2i_interp.t2i.T2IModel")
@patch("wandb.init")
def test_run_stitch(
    mock_wandb,
    mock_t2imodel,
    mock_load_dataset,
    mock_stitcher,
    mock_training,
    mock_inference,
    mock_collect_inmem,
    stitch_cfg,
):
    # Dataset with both required splits so the `if ds_val is None: ...` branch
    # doesn't fire (script does ds_full.get("validation") or ds_full.get("test")
    # then a train_test_split fallback).
    fake_ds = MagicMock()
    fake_ds.__len__.return_value = 10
    # .select() returns a sliced dataset (used by max_samples logic).
    fake_ds.select.return_value = fake_ds
    mock_load_dataset.return_value = {"train": fake_ds, "validation": fake_ds}

    # In-memory activation collector returns {layer: tensor} for each call.
    # 4 calls expected: model_a train/val, model_b train/val.
    fake_acts = {
        "layer_a": torch.randn(4, 10),
        "layer_b": torch.randn(4, 10),
    }
    mock_collect_inmem.return_value = fake_acts

    # Stitcher.train_mapper is called via Training(spec).run_trainer() — that
    # returns an Output object whose .preds is the trained mapper. The script
    # then does th.save(trained_mapper.state_dict(), ...) — so state_dict()
    # must return something pickleable (an empty dict works for this smoke test).
    mock_trained_mapper = MagicMock()
    mock_trained_mapper.state_dict.return_value = {}
    mock_training_instance = MagicMock()
    mock_training_instance.run_trainer.return_value = MagicMock(preds=mock_trained_mapper)
    mock_training.return_value = mock_training_instance

    # Inference (stitched generation) returns a list of fake images.
    mock_inference_instance = MagicMock()
    mock_inference_instance.run_inference.return_value = MagicMock(
        preds=[MagicMock()]  # one fake PIL image
    )
    mock_inference.return_value = mock_inference_instance

    main.__wrapped__(stitch_cfg)

    mock_t2imodel.assert_called()
    mock_load_dataset.assert_called_once_with("test_dataset")
    assert mock_collect_inmem.call_count >= 2  # at least train+val collections
    mock_training.assert_called()
