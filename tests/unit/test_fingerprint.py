"""Unit tests for the run-fingerprint module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from t2i_interp.reporting.fingerprint import (
    RunFingerprint,
    mark_run_completed,
    record_wandb_run,
    seed_everything,
)


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


def test_hash_is_machine_independent_across_device_dtype_paths():
    """The paper's central reproducibility claim: same logical experiment
    from a laptop (MPS, bfloat16, /Users/...) and a CUDA cluster (cuda:0,
    float16, /home/...) must hash identically. device/dtype/local-path
    keys are stripped from the embedded config before hashing.
    """
    fp_mac = RunFingerprint.from_cfg(
        _make_cfg(
            device="mps",
            dtype="bfloat16",
            output_dir="/Users/alice/runs/spectacles",
            save_dir="/Users/alice/cache/latents",
            inline_pairs_file="/Users/alice/pairs.json",
            wandb={"project": "p", "entity": "alice"},
        ),
        workflow="steer",
        intervention=_intervention(),
    )
    fp_cluster = RunFingerprint.from_cfg(
        _make_cfg(
            device="cuda:0",
            dtype="float16",
            output_dir="/home/bob/runs/spectacles",
            save_dir="/scratch/bob/latents",
            inline_pairs_file="/home/bob/pairs.json",
            wandb={"project": "p", "entity": "bob"},
        ),
        workflow="steer",
        intervention=_intervention(),
    )
    assert fp_mac.hash() == fp_cluster.hash()


def test_hash_still_changes_when_logical_inputs_differ():
    """Strip-list shouldn't accidentally hide logical changes — model_key,
    dataset, seed, intervention, hyperparams must still drive the hash."""
    base = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=_intervention())
    diff_model = RunFingerprint.from_cfg(
        _make_cfg(model_key="other/model"), workflow="steer", intervention=_intervention()
    )
    assert base.hash() != diff_model.hash()


def test_record_wandb_run_writes_payload(tmp_path: Path):
    """record_wandb_run persists the live W&B run's URL + IDs so the
    Streamlit Results panel can render the link button + iframe embed."""
    import types

    fake_run = types.SimpleNamespace(
        url="https://wandb.ai/alice/dream-reader/runs/abc123",
        entity="alice",
        project="dream-reader",
        name="spectacles-alpha=10",
        id="abc123",
    )
    out = tmp_path / "run_dir"
    path = record_wandb_run(out, fake_run)
    assert path == out / "wandb_run.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["url"] == fake_run.url
    assert data["entity"] == "alice"
    assert data["project"] == "dream-reader"
    assert data["name"] == "spectacles-alpha=10"
    assert data["id"] == "abc123"


def test_record_wandb_run_skips_when_url_missing(tmp_path: Path):
    """W&B offline mode produces a run with url=None. record_wandb_run
    should no-op rather than write a half-formed payload."""
    import types

    fake_run = types.SimpleNamespace(url=None, entity="alice", project="p", name=None, id="abc")
    out = tmp_path / "run_dir"
    path = record_wandb_run(out, fake_run)
    assert path is None
    assert not (out / "wandb_run.json").exists()


def test_record_wandb_run_skips_when_run_is_none(tmp_path: Path):
    """Cleanly no-ops on a None run (wandb disabled)."""
    assert record_wandb_run(tmp_path, None) is None


def test_mark_run_completed_writes_marker(tmp_path: Path):
    """mark_run_completed writes a _RUN_COMPLETE.json that downstream code
    can use to distinguish finished runs from crashed-mid-train ones."""
    out = tmp_path / "run_dir"
    marker = mark_run_completed(out, workflow="steer", extra={"alpha": 10.0})

    assert marker == out / "_RUN_COMPLETE.json"
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["workflow"] == "steer"
    assert data["alpha"] == 10.0
    assert "completed_at" in data


def test_log_to_wandb_records_summary_fields(monkeypatch):
    """log_to_wandb should call run.log_artifact and populate summary keys."""
    import types
    from unittest.mock import MagicMock

    # Stub `wandb` via monkeypatch — auto-restored after the test, so a real
    # `wandb` import elsewhere in the session is not affected.
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = lambda name, type: MagicMock()  # noqa: A002 (matches wandb API)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    fp = RunFingerprint.from_cfg(_make_cfg(), workflow="steer", intervention=_intervention())
    run = MagicMock()
    run.summary = {}
    fp.log_to_wandb(run)

    assert run.summary["fingerprint/hash"] == fp.hash()
    assert run.summary["fingerprint/workflow"] == "steer"
    assert run.summary["fingerprint/model_id"] == "stabilityai/sdxl-turbo"
    assert run.summary["fingerprint/seed"] == 42
    run.log_artifact.assert_called_once()
