"""Unit tests for app.lib.runner helpers.

`render_workflow_run` is Streamlit-bound and tested via AppTest in the
integration suite; this module covers the pure helpers (tempdir sweep
+ basic run_workflow plumbing).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from app.lib.runner import sweep_old_streamlit_tempdirs


def test_sweep_removes_old_dirs_with_matching_prefix(monkeypatch, tmp_path: Path):
    """Old directory with the matching prefix is removed; everything else
    is left alone (in-flight runs, dirs with other prefixes, plain files)."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    old_dir = tmp_path / "streamlit_loc_OLD"
    fresh_dir = tmp_path / "streamlit_loc_FRESH"
    other_prefix = tmp_path / "streamlit_steer_OLD"
    plain_file = tmp_path / "streamlit_loc_NOTADIR"

    old_dir.mkdir()
    fresh_dir.mkdir()
    other_prefix.mkdir()
    plain_file.write_text("a")

    # Backdate the "old" dirs past the cutoff.
    past = time.time() - 7200
    os.utime(old_dir, (past, past))
    os.utime(other_prefix, (past, past))

    removed = sweep_old_streamlit_tempdirs("streamlit_loc_", max_age_seconds=3600)

    assert removed == 1
    assert not old_dir.exists()
    assert fresh_dir.exists()
    assert other_prefix.exists()
    assert plain_file.exists()


def test_sweep_returns_zero_when_no_matches(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert sweep_old_streamlit_tempdirs("streamlit_nothing_") == 0


def test_sweep_returns_zero_on_empty_prefix(monkeypatch, tmp_path: Path):
    """Empty prefix is a no-op (defensive guard against accidental wipe)."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    (tmp_path / "anything").mkdir()
    assert sweep_old_streamlit_tempdirs("") == 0
    assert (tmp_path / "anything").exists()
