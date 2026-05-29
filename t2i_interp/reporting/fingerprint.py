"""Run fingerprinting for reproducibility.

Per DreamReader paper §2: "Every run logs a fingerprint consisting of the
model identifier, dataset version, random seeds, train/eval splits, full
intervention specifications (hook sites, features, scales, schedules).
Artifacts such as images and trained mappers may be materialized to disk
or exported to W&B."

This module produces a canonical, hashable record of every reproducibility-
relevant input to a run, plus a stable 16-char SHA hash that identifies the
logical experiment. The hash deliberately excludes wall-clock fields
(timestamp, hostname, dirty-flag) so the same logical experiment on two
machines yields the same fingerprint.

Typical use inside a Hydra-decorated entry point::

    fp = RunFingerprint.from_cfg(cfg, workflow="steer", intervention={
        "steer_type": cfg.steer_type,
        "layer_names": list(cfg.layer_names),
        "alpha": cfg.alpha,
    })
    fp.write(os.path.join(cfg.output_dir, "fingerprint.json"))
    if wandb_run is not None:
        fp.log_to_wandb(wandb_run)
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import DictConfig, OmegaConf


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _git_dirty() -> bool:
    # Untracked files (build artifacts, local notes, downloaded weights) don't
    # change the logical run, so `--untracked-files=no` keeps dirty true to its
    # actual meaning: tracked code differs from HEAD.
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


@dataclass(frozen=True)
class RunFingerprint:
    """Canonical record of every reproducibility-relevant input to a run.

    Fields map to the paper's stated requirements:
      - model identifier & revision  → `model_id`, `model_revision`
      - dataset version & splits     → `dataset_id`, `dataset_split`
      - random seeds                  → `seed`
      - intervention specifications  → `intervention`
        (hook sites, features, scales, schedules)
      - full config snapshot          → `config`

    Engineering essentials:
      - `git_sha`, `git_dirty`        → repro check against repo state
      - `python_version`, `platform`, `hostname`, `timestamp`

    The 16-char `hash()` is computed over the canonical content (excluding
    `_VOLATILE_FIELDS` — timestamp, hostname, dirty-flag, python version,
    platform) so re-running the same logical experiment from different
    machines produces an identical hash.
    """

    workflow: str
    model_id: str
    model_revision: str | None
    dataset_id: str | None
    dataset_split: str | None
    seed: int | None
    intervention: dict[str, Any]
    config: dict[str, Any]
    git_sha: str | None
    git_dirty: bool
    python_version: str
    platform: str
    hostname: str
    timestamp: str

    # Fields excluded from the stable hash (volatile / machine-local).
    # `ClassVar` so dataclass treats it as a class-level constant — it does
    # not become an instance field and does not appear in `asdict()` output.
    _VOLATILE_FIELDS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "hostname",
        "git_dirty",
        "python_version",
        "platform",
    )

    # Config keys stripped before hashing so the hash is machine-independent.
    # device/dtype select the runtime path (CUDA float16 vs MPS bfloat16) but
    # represent the *same* logical experiment. Local paths (output_dir, save_dir,
    # inline_pairs_file) are absolutised by Hydra entry points and naturally
    # differ between machines. The wandb block is user-specific. Hydra's own
    # block is bookkeeping. None of these should change the identity of a run.
    _NON_REPRODUCIBLE_CONFIG_KEYS: ClassVar[tuple[str, ...]] = (
        "device",
        "dtype",
        "output_dir",
        "save_dir",
        "inline_pairs_file",
        "wandb",
        "hydra",
    )

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        workflow: str,
        intervention: dict[str, Any] | None = None,
    ) -> RunFingerprint:
        """Build a fingerprint from a resolved Hydra config.

        Args:
            cfg: Resolved DictConfig from a Hydra-decorated entry point.
            workflow: One of 'steer', 'stitch', 'sae', 'localisation'.
            intervention: Optional structured summary of the intervention
                (hook sites, alphas, schedules). If None, an empty dict is
                stored — callers should populate this for it to be useful.
        """
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # cfg_dict is always Dict[str, Any] for a top-level DictConfig
        if not isinstance(cfg_dict, dict):
            cfg_dict = {}

        return cls(
            workflow=workflow,
            model_id=str(cfg.get("model_key", "")),
            model_revision=cfg.get("model_revision"),
            dataset_id=cfg.get("dataset_name"),
            dataset_split=cfg.get("split") or cfg.get("dataset_split"),
            seed=cfg.get("seed"),
            intervention=dict(intervention or {}),
            config=cfg_dict,
            git_sha=_git_sha(),
            git_dirty=_git_dirty(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            hostname=socket.gethostname(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def hash(self) -> str:
        """Stable 16-char SHA-256 hex digest of the canonical content.

        Same logical experiment from a laptop (MPS, bfloat16, /Users/...) and a
        CUDA cluster (cuda:0, float16, /home/...) produces the same hash —
        device/dtype/local-path keys are stripped from the embedded config
        before hashing. See `_VOLATILE_FIELDS` and `_NON_REPRODUCIBLE_CONFIG_KEYS`.
        """
        d = asdict(self)
        for k in self._VOLATILE_FIELDS:
            d.pop(k, None)
        if isinstance(d.get("config"), dict):
            d["config"] = {
                k: v for k, v in d["config"].items() if k not in self._NON_REPRODUCIBLE_CONFIG_KEYS
            }
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint_hash"] = self.hash()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, sort_keys=False)

    def write(self, path: str | Path) -> Path:
        """Write the fingerprint JSON to `path` (creating parents as needed)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    def log_to_wandb(self, run: Any, name: str | None = None) -> None:
        """Log this fingerprint as a W&B Artifact and surface key fields in summary.

        Args:
            run: An active `wandb.sdk.wandb_run.Run`.
            name: Optional artifact name. Defaults to `fingerprint-<hash>`.
        """
        import wandb

        h = self.hash()
        artifact_name = name or f"fingerprint-{h}"
        art = wandb.Artifact(artifact_name, type="fingerprint")
        with art.new_file("fingerprint.json", mode="w") as f:
            f.write(self.to_json())
        run.log_artifact(art)

        # Surface in run summary for fast filtering in the W&B UI.
        run.summary["fingerprint/hash"] = h
        run.summary["fingerprint/workflow"] = self.workflow
        run.summary["fingerprint/model_id"] = self.model_id
        if self.git_sha:
            run.summary["fingerprint/git_sha"] = self.git_sha
        if self.seed is not None:
            run.summary["fingerprint/seed"] = self.seed


def mark_run_completed(
    output_dir: str | Path,
    *,
    workflow: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `_RUN_COMPLETE.json` into `output_dir` at the end of a successful run.

    Lets downstream consumers (Fingerprints page, crash-detection scripts)
    distinguish a finished run from one that left a fingerprint behind but
    crashed mid-train. The file's mere existence is the signal; the contents
    record when and which workflow finished.

    Returns the path written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if workflow:
        payload["workflow"] = workflow
    if extra:
        payload.update(extra)
    marker = out / "_RUN_COMPLETE.json"
    marker.write_text(json.dumps(payload, indent=2))
    return marker


def seed_everything(seed: int | None) -> None:
    """Set deterministic seeds across torch, numpy, and Python's random.

    No-op when `seed` is None. Returns nothing — call once at the top of a
    Hydra entry point, before any model loads or stochastic ops.
    """
    if seed is None:
        return

    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # MPS shares the global torch RNG (torch.manual_seed seeds it too) but
        # we set it explicitly anyway so future torch versions that introduce
        # a separate MPS generator stay covered. The `if available` guard
        # avoids a UserWarning on non-Apple-Silicon machines.
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_module = getattr(torch, "mps", None)
            if mps_module is not None and hasattr(mps_module, "manual_seed"):
                mps_module.manual_seed(seed)
    except ImportError:
        pass
