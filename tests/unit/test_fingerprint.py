"""Unit tests for the run-fingerprint module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from t2i_interp.reporting.fingerprint import RunFingerprint, seed_everything


def _make_cfg(**overrides):
    base = {
        "model_key": "stabilityai/sdxl-turbo",
        "dataset_name": "nirmalendu01/spectacles-bias-prompts",
        "seed": 42,
        "guidance_scale": 0.0,
        "num_inference_steps": 4,
        "alpha": 10.0,
        "layer_names": [
            "unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2",
        ],
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _intervention():
    return {
        "steer_type": "loreft",
        "layer_names": ["unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2"],
        "alpha": 10.0,
        "steer_steps": 4,
    }


def test_from_cfg_captures_required_paper_fields():
    """Paper §2 fields: model id, dataset, seed, intervention, full config."""
    cfg = _make_cfg()
    fp = RunFingerprint.from_cfg(cfg, workflow="steer", intervention=_intervention())

    assert fp.workflow == "steer"
    assert fp.model_id == "stabilityai/sdxl-turbo"
    assert fp.dataset_id == "nirmalendu01/spectacles-bias-prompts"
    assert fp.seed == 42
    assert fp.intervention["steer_type"] == "loreft"
    assert fp.intervention["alpha"] == 10.0
    # Full config snapshot present.
    assert fp.config["guidance_scale"] == 0.0
    # Engineering essentials present (values may be None on machines without git).
    assert fp.python_version
    assert fp.platform
    assert fp.hostname
    assert fp.timestamp


def test_hash_is_stable_across_volatile_field_changes():
    """The 16-char hash must ignore timestamp / hostname / git_dirty."""
    fp1 = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=_intervention())
    # Construct a sibling with different volatile fields.
    fp2 = RunFingerprint(
        workflow=fp1.workflow,
        model_id=fp1.model_id,
        model_revision=fp1.model_revision,
        dataset_id=fp1.dataset_id,
        dataset_split=fp1.dataset_split,
        seed=fp1.seed,
        intervention=dict(fp1.intervention),
        config=dict(fp1.config),
        git_sha=fp1.git_sha,
        git_dirty=not fp1.git_dirty,  # flipped
        python_version=fp1.python_version,
        platform=fp1.platform,
        hostname="some-other-host",  # different
        timestamp="1970-01-01T00:00:00Z",  # different
    )
    assert fp1.hash() == fp2.hash()
    assert len(fp1.hash()) == 16


def test_hash_changes_when_seed_changes():
    """Two runs with different seeds must produce different fingerprints."""
    fp_a = RunFingerprint.from_cfg(
        _make_cfg(seed=1), workflow="steer", intervention=_intervention()
    )
    fp_b = RunFingerprint.from_cfg(
        _make_cfg(seed=2), workflow="steer", intervention=_intervention()
    )
    assert fp_a.hash() != fp_b.hash()


def test_hash_changes_when_intervention_changes():
    """Different alpha → different fingerprint (paper Fig 2 sweep relies on this)."""
    iv1 = _intervention()
    iv2 = {**iv1, "alpha": 20.0}
    fp_a = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=iv1)
    fp_b = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=iv2)
    assert fp_a.hash() != fp_b.hash()


def test_write_creates_parent_dirs_and_round_trips_json(tmp_path: Path):
    """Fingerprint.write should create dirs and produce parseable JSON."""
    fp = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=_intervention())
    out = tmp_path / "nested" / "subdir" / "fingerprint.json"
    written = fp.write(out)

    assert written == out
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["workflow"] == "steer"
    assert data["model_id"] == "stabilityai/sdxl-turbo"
    assert data["seed"] == 42
    assert data["fingerprint_hash"] == fp.hash()


def test_seed_everything_is_noop_for_none():
    """seed_everything(None) must not raise even without numpy/torch installed."""
    # Should simply return without touching any RNG.
    seed_everything(None)


def test_seed_everything_makes_python_random_deterministic():
    import random

    seed_everything(123)
    a = [random.random() for _ in range(4)]
    seed_everything(123)
    b = [random.random() for _ in range(4)]
    assert a == b


def test_seed_everything_makes_torch_deterministic():
    """If torch is available, seeded calls should produce identical tensors."""
    torch = pytest.importorskip("torch")
    seed_everything(7)
    a = torch.randn(8)
    seed_everything(7)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_log_to_wandb_records_summary_fields():
    """log_to_wandb should call run.log_artifact and populate summary keys."""
    import sys
    import types
    from unittest.mock import MagicMock

    # Stub the `wandb` module so `Artifact(...)` returns a context-managing mock.
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = lambda name, type: MagicMock()  # noqa: A002 (kw matches wandb API)
    sys.modules["wandb"] = fake_wandb

    try:
        fp = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=_intervention())
        run = MagicMock()
        run.summary = {}
        fp.log_to_wandb(run)

        assert run.summary["fingerprint/hash"] == fp.hash()
        assert run.summary["fingerprint/workflow"] == "steer"
        assert run.summary["fingerprint/model_id"] == "stabilityai/sdxl-turbo"
        assert run.summary["fingerprint/seed"] == 42
        run.log_artifact.assert_called_once()
    finally:
        del sys.modules["wandb"]
