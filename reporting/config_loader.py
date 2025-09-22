from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import os
import json

def _read_yaml(p: Path) -> Dict[str, Any]:
    import yaml  # pip install pyyaml
    return yaml.safe_load(p.read_text()) or {}

def _read_toml(p: Path) -> Dict[str, Any]:
    import tomllib  # py3.11+
    return tomllib.loads(p.read_text())

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix in {".yml", ".yaml"}:
        cfg = _read_yaml(p)
    elif p.suffix == ".toml":
        cfg = _read_toml(p)
    elif p.suffix == ".json":
        cfg = _read_json(p)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")

    # Optional: environment variable overrides (e.g., WANDB_PROJECT, WANDB_MODE)
    w = cfg.setdefault("wandb", {})
    w["project"] = os.getenv("WANDB_PROJECT", w.get("project"))
    w["entity"]  = os.getenv("WANDB_ENTITY",  w.get("entity"))
    w["mode"]    = os.getenv("WANDB_MODE",    w.get("mode", "online"))
    # You can extend with run name, tags, etc., as needed.

    return cfg

def wandb_init_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a dict of kwargs for wandb.init from the loaded config."""
    w = cfg.get("wandb", {})
    kw: Dict[str, Any] = {}
    if w.get("project") is None:
        raise ValueError("wandb.project is required")
    kw["project"] = w["project"]
    if w.get("entity") is not None:
        kw["entity"] = w["entity"]
    if w.get("run_name") is not None:
        kw["name"] = w["run_name"]
    if w.get("job_type") is not None:
        kw["job_type"] = w["job_type"]
    if w.get("tags") is not None:
        kw["tags"] = w["tags"]
    if w.get("group") is not None:
        kw["group"] = w["group"]
    if w.get("mode") is not None:
        kw["mode"] = w["mode"]
    extra = w.get("extra_init_kwargs") or {}
    kw.update(extra)
    return kw
