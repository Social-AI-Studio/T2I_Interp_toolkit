"""Unit tests for scan_fingerprints — the walker the Fingerprints page uses
to build its dataframe.

The `completed` column added in the reproducibility batch reflects the
presence of `_RUN_COMPLETE.json` next to each fingerprint.json. The
Fingerprints page rendered this via st.dataframe which paints to canvas,
so the e2e Playwright text-scrape couldn't observe it. These tests
verify the data path directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.lib.outputs import scan_fingerprints


def _fp(workflow="steer", hash_="abc123def4567890"):
    return {
        "workflow": workflow,
        "model_id": "stabilityai/sdxl-turbo",
        "dataset_id": "nirmalendu01/spectacles",
        "seed": 42,
        "git_sha": "deadbeefcafe",
        "git_dirty": False,
        "timestamp": "2026-06-02T19:00:00Z",
        "fingerprint_hash": hash_,
    }


def test_scan_finds_one_completed_one_partial(tmp_path: Path):
    """A directory holding a fingerprint.json + _RUN_COMPLETE.json marker
    is completed; one without the marker is partial."""
    a = tmp_path / "run_A"
    b = tmp_path / "run_B"
    a.mkdir()
    b.mkdir()
    (a / "fingerprint.json").write_text(json.dumps(_fp(hash_="aaaa111122223333")))
    (a / "_RUN_COMPLETE.json").write_text('{"completed_at": "2026-06-02T19:01:00Z"}')
    (b / "fingerprint.json").write_text(json.dumps(_fp(hash_="bbbb111122223333")))
    # NOTE: no _RUN_COMPLETE.json in b — that's the "crashed mid-train" case.

    rows = scan_fingerprints([str(tmp_path)])
    by_hash = {r["hash"]: r for r in rows}
    assert by_hash["aaaa111122223333"]["completed"] is True
    assert by_hash["bbbb111122223333"]["completed"] is False


def test_scan_skips_unreadable_fingerprints(tmp_path: Path):
    """Malformed JSON shouldn't crash the scan; the row is silently dropped."""
    a = tmp_path / "broken"
    a.mkdir()
    (a / "fingerprint.json").write_text("not valid json {{{")
    rows = scan_fingerprints([str(tmp_path)])
    assert rows == []


def test_scan_sorts_newest_first(tmp_path: Path):
    """Rows come back sorted by timestamp descending."""
    a = tmp_path / "older"
    b = tmp_path / "newer"
    a.mkdir()
    b.mkdir()
    fp_a = _fp(hash_="aaaa")
    fp_a["timestamp"] = "2024-01-01T00:00:00Z"
    fp_b = _fp(hash_="bbbb")
    fp_b["timestamp"] = "2026-06-02T19:00:00Z"
    (a / "fingerprint.json").write_text(json.dumps(fp_a))
    (b / "fingerprint.json").write_text(json.dumps(fp_b))
    rows = scan_fingerprints([str(tmp_path)])
    assert [r["hash"] for r in rows] == ["bbbb", "aaaa"]


def test_completed_column_present_when_run_complete_missing(tmp_path: Path):
    """Older fingerprints without the _RUN_COMPLETE.json convention still
    return a `completed` field (False) — the Fingerprints page expects
    every row to have it."""
    a = tmp_path / "legacy"
    a.mkdir()
    (a / "fingerprint.json").write_text(json.dumps(_fp()))
    rows = scan_fingerprints([str(tmp_path)])
    assert len(rows) == 1
    assert "completed" in rows[0]
    assert rows[0]["completed"] is False
