"""Unit tests for the shared inline-pairs load/split helpers.

run_steer.py (CAA / KSteer / LoReFT) and run_stitch.py both build
in-memory datasets from `cfg.inline_pairs` or `cfg.inline_pairs_file`.
The load + shuffle-and-split steps moved into this module; these tests
pin its behaviour so a future change to the splitter can't silently
break both workflows at once.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from t2i_interp.utils.inline_pairs import load_inline_pairs, make_disjoint_split


def test_load_inline_pairs_returns_none_for_empty_cfg():
    cfg = OmegaConf.create({})
    assert load_inline_pairs(cfg) is None


def test_load_inline_pairs_from_cfg_list():
    pairs = [{"pos": "a", "neg": "b"}, {"pos": "c", "neg": "d"}]
    cfg = OmegaConf.create({"inline_pairs": pairs})
    out = load_inline_pairs(cfg)
    assert out == pairs


def test_load_inline_pairs_from_file(tmp_path: Path):
    pairs = [{"pos": "a", "neg": "b"}]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(pairs))
    cfg = OmegaConf.create({"inline_pairs_file": str(p)})
    out = load_inline_pairs(cfg)
    assert out == pairs


def test_load_inline_pairs_prefers_cfg_over_file(tmp_path: Path):
    """cfg.inline_pairs (literal) takes precedence over inline_pairs_file."""
    cfg_pairs = [{"pos": "from_cfg", "neg": "from_cfg"}]
    file_pairs = [{"pos": "from_file", "neg": "from_file"}]
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(file_pairs))
    cfg = OmegaConf.create({"inline_pairs": cfg_pairs, "inline_pairs_file": str(p)})
    assert load_inline_pairs(cfg) == cfg_pairs


def test_make_disjoint_split_reuses_train_for_tiny_dataset():
    rows = [{"x": 1}, {"x": 2}]
    train, val = make_disjoint_split(rows, seed=42)
    # ≤3 rows: val mirrors train rather than empty-out train.
    assert train == rows
    assert val == rows


def test_make_disjoint_split_shuffles_before_slice():
    """For CAA, rows alternate [pos, neg, pos, neg, ...]. An unshuffled
    tail-slice gave adjacent pos/neg pairs to val. The splitter must
    shuffle first."""
    rows = [{"label": i % 2} for i in range(20)]
    train, val = make_disjoint_split(rows, seed=42)
    # Disjoint: every val row absent from train.
    assert len(train) + len(val) == len(rows)
    train_ids = [id(r) for r in train]
    for r in val:
        assert id(r) not in train_ids
    # If the splitter still did unshuffled tail-slicing, val would be
    # the *last* 20% — which is rows[16:20] = labels [0,1,0,1]. With
    # the seeded shuffle, val labels should NOT match that exact slice.
    unshuffled_tail_labels = [r["label"] for r in rows[16:20]]
    val_labels = [r["label"] for r in val]
    assert val_labels != unshuffled_tail_labels or len(val) != 4


def test_make_disjoint_split_is_seed_deterministic():
    rows = [{"i": i} for i in range(20)]
    a_train, a_val = make_disjoint_split(rows, seed=7)
    b_train, b_val = make_disjoint_split(rows, seed=7)
    assert a_train == b_train
    assert a_val == b_val


def test_make_disjoint_split_different_seeds_yield_different_splits():
    rows = [{"i": i} for i in range(20)]
    _, a_val = make_disjoint_split(rows, seed=1)
    _, b_val = make_disjoint_split(rows, seed=2)
    assert a_val != b_val


def test_make_disjoint_split_handles_none_seed():
    """seed=None must not raise; deterministic via a fixed fallback."""
    rows = [{"i": i} for i in range(20)]
    train, val = make_disjoint_split(rows, seed=None)
    assert len(train) + len(val) == len(rows)
