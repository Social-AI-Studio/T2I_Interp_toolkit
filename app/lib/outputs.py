"""Find and load run artifacts (images + fingerprint.json) for display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_images(dir_: str | Path) -> list[Path]:
    """Recursively list image files under `dir_` (best-effort, ignores missing)."""
    d = Path(dir_)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*.png") if p.is_file())


def load_fingerprint(dir_: str | Path) -> dict[str, Any] | None:
    """Find and parse the closest fingerprint.json under `dir_`, or None."""
    d = Path(dir_)
    if not d.exists():
        return None
    candidates = sorted(d.rglob("fingerprint.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def scan_fingerprints(roots: list[str | Path]) -> list[dict[str, Any]]:
    """Walk given roots, parse every fingerprint.json found. Returns rows
    suitable for st.dataframe (sorted by timestamp descending)."""
    rows: list[dict[str, Any]] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for fp_file in p.rglob("fingerprint.json"):
            try:
                fp = json.loads(fp_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(
                {
                    "hash": fp.get("fingerprint_hash", "?"),
                    "workflow": fp.get("workflow", "?"),
                    "model": fp.get("model_id", "?"),
                    "dataset": fp.get("dataset_id") or "-",
                    "seed": fp.get("seed"),
                    "git_sha": (fp.get("git_sha") or "")[:8],
                    "git_dirty": fp.get("git_dirty", False),
                    "timestamp": fp.get("timestamp", ""),
                    "path": str(fp_file.parent),
                }
            )
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return rows
