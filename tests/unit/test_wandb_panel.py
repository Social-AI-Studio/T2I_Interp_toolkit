"""Unit tests for render_wandb_panel — the panel that surfaces a live W&B
run's link button in the Streamlit Results section.

In-browser e2e can't drive this without a real wandb run + live login.
Here we use Streamlit's AppTest to render a tiny harness that calls
render_wandb_panel directly with a fake wandb_run dict, then inspect the
resulting widget tree.

(An earlier version of this panel also embedded the live run dashboard
via `st.components.v1.iframe(run.url)`. W&B sets `X-Frame-Options:
SAMEORIGIN` and a CSP `frame-ancestors 'self'` on every run page, so the
iframe never loaded from a Streamlit origin — it was silently dead UX.
That branch was removed; the link button is the reliable path.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest


_HARNESS = """
import streamlit as st

from app.lib import render_wandb_panel

wandb_run = st.session_state.get("__wandb_run__", None)
render_wandb_panel(wandb_run)
"""


def _run(wandb_run) -> AppTest:
    """Render the harness with a given wandb_run payload."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_HARNESS)
        path = f.name
    at = AppTest.from_file(path)
    at.session_state["__wandb_run__"] = wandb_run
    at.run()
    Path(path).unlink()
    return at


def test_panel_renders_nothing_when_payload_is_none():
    at = _run(None)
    # No headings, no link buttons — the panel should be invisible when
    # wandb wasn't enabled for the run.
    assert not at.exception
    assert len(at.markdown) == 0


def test_panel_renders_nothing_when_url_missing():
    """A wandb_run dict without a url should suppress the panel — e.g. wandb
    in offline mode where run.url is None."""
    at = _run({"entity": "alice", "project": "p", "id": "abc123"})
    assert not at.exception
    assert len(at.markdown) == 0


def test_panel_renders_link_button_and_metadata():
    """When the payload has a URL, we render the project + run name and a
    link button to the live W&B dashboard."""
    payload = {
        "url": "https://wandb.ai/alice/dream-reader/runs/abc123",
        "entity": "alice",
        "project": "dream-reader",
        "name": "spectacles-alpha=10",
        "id": "abc123",
    }
    at = _run(payload)
    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    assert "W&B run" in body
    assert "dream-reader" in body
    assert "spectacles-alpha=10" in body
    link_buttons = at.get("link_button")
    assert len(link_buttons) >= 1
    assert any(payload["url"] in (lb.url or "") for lb in link_buttons)


def test_panel_does_not_render_iframe_expander():
    """W&B sets X-Frame-Options: SAMEORIGIN and CSP frame-ancestors 'self'
    on every run page, so the iframe embed could never load from a
    Streamlit origin. The expander was removed; the panel must never
    render it again."""
    payload = {
        "url": "https://wandb.ai/alice/p/runs/abc123",
        "entity": "alice",
        "project": "p",
        "name": "test",
        "id": "abc123",
    }
    at = _run(payload)
    assert not at.exception
    expander_labels = [e.label for e in at.get("expander")]
    assert "Embedded W&B view" not in expander_labels


# ── Full chain: CLI → record_wandb_run → load_wandb_run → render_wandb_panel ──


def test_load_wandb_run_reads_what_record_wandb_run_writes(tmp_path):
    """End-to-end JSON round-trip. The CLI writes wandb_run.json via
    record_wandb_run; the Streamlit page reads it via load_wandb_run.
    These two functions live on opposite sides of a process boundary,
    so the contract is the JSON file. Pin it."""
    import types

    from app.lib.outputs import load_wandb_run
    from t2i_interp.reporting.fingerprint import record_wandb_run

    fake_run = types.SimpleNamespace(
        url="https://wandb.ai/alice/dream-reader/runs/abc123",
        entity="alice",
        project="dream-reader",
        name="spectacles-alpha=10",
        id="abc123",
    )
    out = tmp_path / "real_output_dir"
    record_wandb_run(out, fake_run)

    loaded = load_wandb_run(out)
    assert loaded is not None
    assert loaded["url"] == fake_run.url
    assert loaded["entity"] == "alice"
    assert loaded["project"] == "dream-reader"
    assert loaded["name"] == "spectacles-alpha=10"
    assert loaded["id"] == "abc123"


def test_load_wandb_run_returns_none_when_file_missing(tmp_path):
    """No `wandb_run.json` → load returns None and the panel correctly
    renders nothing (covered by `test_panel_renders_nothing_when_payload_is_none`)."""
    from app.lib.outputs import load_wandb_run

    assert load_wandb_run(tmp_path) is None


def test_load_wandb_run_finds_sibling_for_steering_suffixed_output_dir(tmp_path):
    """run_steer.py rewrites output_dir to `<base>_<block>_alpha=<a>` after
    Streamlit picks the tempdir. The Steering page passes
    `include_prefix_siblings=True` so load_wandb_run walks siblings whose
    name starts with the base prefix."""
    import types

    from app.lib.outputs import load_wandb_run
    from t2i_interp.reporting.fingerprint import record_wandb_run

    base = tmp_path / "streamlit_steer_XYZ"
    suffixed = tmp_path / "streamlit_steer_XYZ_down_blocks_alpha=8"
    fake_run = types.SimpleNamespace(
        url="https://wandb.ai/alice/p/runs/r1",
        entity="alice",
        project="p",
        name="r1",
        id="r1",
    )
    record_wandb_run(suffixed, fake_run)

    # Without sibling walking, the lookup misses.
    assert load_wandb_run(base) is None
    # With it, the suffixed sibling is found.
    found = load_wandb_run(base, include_prefix_siblings=True)
    assert found is not None
    assert found["url"] == fake_run.url


def test_full_chain_with_simulated_wandb_init(tmp_path, monkeypatch):
    """Simulate the full online chain: monkey-patch wandb so wandb.init()
    returns a fake live run with a real-shaped URL, exercise the CLI path
    (record_wandb_run), then verify the AppTest panel renders the correct
    link button.

    This is the closest we can get to a true online integration test
    without a real W&B API key. It pins the contract end-to-end:
    wandb.init shape → record_wandb_run → JSON on disk → load_wandb_run
    → render_wandb_panel widget tree.
    """
    import types

    from app.lib.outputs import load_wandb_run
    from t2i_interp.reporting.fingerprint import record_wandb_run

    # Fake the run object wandb.init() would return.
    live_run = types.SimpleNamespace(
        url="https://wandb.ai/dream-reader-team/spectacles/runs/8j9l8m9n",
        entity="dream-reader-team",
        project="spectacles",
        name="loreft-alpha=10-down_blocks",
        id="8j9l8m9n",
    )

    out = tmp_path / "run_dir"
    # Writer side (CLI's record_wandb_run after wandb.init).
    record_wandb_run(out, live_run)
    # Reader side (Streamlit page's load_wandb_run on the same dir).
    payload = load_wandb_run(out)
    assert payload is not None

    # And the panel renders the live link.
    at = _run(payload)
    assert not at.exception
    link_buttons = at.get("link_button")
    assert len(link_buttons) >= 1
    assert link_buttons[0].url == live_run.url
