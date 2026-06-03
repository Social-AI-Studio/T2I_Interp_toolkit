"""Shared helpers for run_steer / run_stitch inline-pair plumbing.

Both workflows accept a `cfg.inline_pairs` (list literal in YAML) or
`cfg.inline_pairs_file` (path to a JSON list) that lets the Streamlit
playground / quick demos skip the HF dataset and train on prompt pairs
the user just typed. The two scripts shape rows differently (CAA wants
`caption` + binary `label`, LoReFT wants `base_prompt` + `teacher_prompt`,
stitching wants `prompt_a` + `prompt_b`), but the read-and-validate and
shuffle-and-split steps are identical. This module owns those steps.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from omegaconf import OmegaConf


def load_inline_pairs(cfg: Any) -> list | None:
    """Read `cfg.inline_pairs` (list) or `cfg.inline_pairs_file` (path).

    Returns the raw list when present (caller is responsible for shape
    validation); None when neither knob is set or the file is missing.
    """
    raw = getattr(cfg, "inline_pairs", None)
    pairs = OmegaConf.to_container(raw, resolve=True) if raw else []
    if not pairs:
        path = getattr(cfg, "inline_pairs_file", None)
        if path and os.path.exists(path):
            with open(path) as f:
                pairs = json.load(f)
    return pairs or None


def make_disjoint_split(
    rows: list, *, seed: int | None, val_frac: float = 0.2
) -> tuple[list, list]:
    """Split rows into disjoint train/val with shuffling.

    Very small datasets (≤3 rows) reuse train as val — useful for smoke
    tests; val metrics are meaningless either way. Otherwise shuffles
    (seeded for reproducibility) before slicing the tail off for val.

    The shuffle matters for CAA, where rows alternate `[pos, neg, pos, neg]`
    — an unshuffled tail-slice gives adjacent pos/neg pairs to val and
    biases both halves.
    """
    if len(rows) <= 3:
        return list(rows), list(rows)
    rng = random.Random(seed if seed is not None else 0)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[:-n_val], shuffled[-n_val:]
