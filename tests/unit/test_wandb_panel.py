"""Unit tests for render_wandb_panel — the panel that surfaces a live W&B
run's link button + iframe in the Streamlit Results section.

In-browser e2e can't drive these without a real wandb run + live login.
Here we use Streamlit's AppTest to render a tiny harness that calls
render_wandb_panel directly with a fake wandb_run dict, then inspect the
resulting widget tree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest


_HARNESS = """
import streamlit as st

from app.lib import render_wandb_panel

wandb_run = st.session_state.get("__wandb_run__", None)
embed = st.session_state.get("__embed__", True)
render_wandb_panel(wandb_run, embed=embed)
"""


def _run(wandb_run, *, embed=True) -> AppTest:
    """Render the harness with a given wandb_run payload."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_HARNESS)
        path = f.name
    at = AppTest.from_file(path)
    at.session_state["__wandb_run__"] = wandb_run
    at.session_state["__embed__"] = embed
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
    at = _run(payload, embed=True)
    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    assert "W&B run" in body
    assert "dream-reader" in body
    assert "spectacles-alpha=10" in body
    # The link_button surfaces as one of LinkButton elements.
    link_buttons = at.get("link_button")
    assert len(link_buttons) >= 1
    assert any(payload["url"] in (lb.url or "") for lb in link_buttons)


def test_panel_embed_off_skips_iframe():
    """embed=False keeps the link but suppresses the iframe — useful when
    the user wants a compact panel."""
    payload = {
        "url": "https://wandb.ai/alice/p/runs/abc123",
        "entity": "alice",
        "project": "p",
        "name": "test",
        "id": "abc123",
    }
    at = _run(payload, embed=False)
    assert not at.exception
    # link button still present
    assert len(at.get("link_button")) >= 1
    # No expander labeled "Embedded W&B view"
    expander_labels = [e.label for e in at.get("expander")]
    assert "Embedded W&B view" not in expander_labels


def test_panel_embed_on_creates_expander():
    payload = {
        "url": "https://wandb.ai/alice/p/runs/abc123",
        "entity": "alice",
        "project": "p",
        "name": "test",
        "id": "abc123",
    }
    at = _run(payload, embed=True)
    assert not at.exception
    expander_labels = [e.label for e in at.get("expander")]
    assert "Embedded W&B view" in expander_labels
