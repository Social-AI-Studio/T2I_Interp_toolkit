"""Find and load run artifacts (images + fingerprint.json) for display."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def collect_images(dir_: str | Path, *, include_prefix_siblings: bool = False) -> list[Path]:
    """Recursively list image files under `dir_` (best-effort, ignores missing).

    With `include_prefix_siblings=True`, also walks sibling directories whose
    name starts with `dir_.name`. The Steering workflow uses this because
    `run_steer.py` appends `_<block>_alpha=<alpha>` to the configured
    output_dir, so the real outputs land next to (not under) the directory
    the Streamlit page created. Workflows that don't rewrite their output
    path (Localisation, Stitching, SAE) don't need to opt in.
    """
    d = Path(dir_)
    images: list[Path] = []
    if d.exists():
        images.extend(p for p in d.rglob("*.png") if p.is_file())
    if include_prefix_siblings and d.parent.exists():
        prefix = d.name
        for sibling in d.parent.iterdir():
            if sibling == d:
                continue  # already walked above
            if sibling.is_dir() and sibling.name.startswith(prefix):
                images.extend(p for p in sibling.rglob("*.png") if p.is_file())
    return sorted(images)


def pair_baseline_modified(
    images: list[Path],
    *,
    modified_kinds: tuple[str, ...] = ("steered", "modified", "head", "ablated"),
    label_prefix: str = "prompt",
) -> list[tuple[str, Path | None, Path | None]]:
    """Group output images into (label, baseline, modified) triples.

    Recognises filenames like `baseline_0.png`, `baseline.png`, `steered_0.png`,
    `modified_5.png`, etc. The `baseline_<idx>` form pairs with the matching
    `<kind>_<idx>` modified image. A single unindexed `baseline.png` is
    treated as the shared baseline for every modified index (the
    Localisation case, where one baseline is shared across head ablations).

    Anything that doesn't match either pattern is returned as a leftover
    triple `(name, None, image)` so it still gets displayed.
    """
    indexed_baselines: dict[str, Path] = {}
    shared_baseline: Path | None = None
    modified: dict[str, Path] = {}
    leftovers: list[Path] = []

    modified_alt = "|".join(re.escape(k) for k in modified_kinds)
    modified_re = re.compile(rf"(?:{modified_alt})_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    indexed_baseline_re = re.compile(r"baseline_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    bare_baseline_re = re.compile(r"baseline\.(?:png|jpg|jpeg)$", re.IGNORECASE)

    for img in images:
        name = img.name
        if m := indexed_baseline_re.search(name):
            indexed_baselines[m.group(1)] = img
        elif bare_baseline_re.search(name):
            shared_baseline = img
        elif m := modified_re.search(name):
            modified[m.group(1)] = img
        else:
            leftovers.append(img)

    indices = sorted(set(indexed_baselines) | set(modified), key=int)
    out: list[tuple[str, Path | None, Path | None]] = []
    for idx in indices:
        baseline = indexed_baselines.get(idx, shared_baseline)
        out.append((f"{label_prefix} {idx}", baseline, modified.get(idx)))
    # If there are no modified-with-index images but we have a shared baseline
    # alone, still show it once so the user sees what they got.
    if not indices and shared_baseline is not None:
        out.append((f"{label_prefix} (baseline only)", shared_baseline, None))
    for img in leftovers:
        out.append((img.name, None, img))
    return out


def load_fingerprint(
    dir_: str | Path, *, include_prefix_siblings: bool = False
) -> dict[str, Any] | None:
    """Find and parse the closest fingerprint.json under `dir_`, or None.

    With `include_prefix_siblings=True`, also searches sibling directories
    whose name starts with `dir_.name`. Same rationale as `collect_images`.
    """
    d = Path(dir_)
    candidates: list[Path] = []
    if d.exists():
        candidates.extend(d.rglob("fingerprint.json"))
    if include_prefix_siblings and d.parent.exists():
        prefix = d.name
        for sibling in d.parent.iterdir():
            if sibling == d:
                continue
            if sibling.is_dir() and sibling.name.startswith(prefix):
                candidates.extend(sibling.rglob("fingerprint.json"))
    candidates.sort()
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
