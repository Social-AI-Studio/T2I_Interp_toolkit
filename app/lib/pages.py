"""Shared helpers for the workflow pages.

Each of the four workflow pages (Localisation, Steering, Stitching, SAE)
shares the same scaffolding:

1. Initialise session_state with defaults.
2. Consume a recipe payload (if one was dropped by the Recipes page).
3. Render a Step 1 scenario radio + Apply preset button.
4. Render a sidebar Run-label text input.

The helpers here turn that scaffolding into one-liners so the pages stay
small and the pattern stays consistent.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

SessionState = MutableMapping[str, Any]


def apply_payload(
    state: SessionState,
    *,
    prefix: str,
    defaults: dict[str, Any],
    workflow_name: str,
) -> None:
    """Initialise `state` with `defaults`, then consume any matching
    `state["recipe_payload"]` left by the Recipes page.

    `state` is normally `st.session_state`, but any dict-like works (which
    makes the function trivially unit-testable).

    For each key in `defaults`, `state.setdefault(key, value)` runs once.
    If `state["recipe_payload"]` exists and its `workflow` matches
    `workflow_name`, the payload is popped and applied:

    - `payload["goal"]` (if non-empty) goes to `state[f"{prefix}_goal"]`.
    - For each `(fk, fv)` in `payload["fields"]`, sets
      `state[f"{prefix}_{fk}"] = fv` provided that key exists in `defaults`.
      Unknown keys are silently ignored so the pipeline tolerates schema
      drift.

    Payloads for a different workflow are left untouched so the matching
    page can pick them up later.
    """
    for k, v in defaults.items():
        state.setdefault(k, v)
    payload = state.get("recipe_payload")
    if not (payload and payload.get("workflow") == workflow_name):
        return
    del state["recipe_payload"]
    if payload.get("goal"):
        state[f"{prefix}_goal"] = payload["goal"]
    for fk, fv in payload.get("fields", {}).items():
        sk = f"{prefix}_{fk}"
        if sk in defaults:
            state[sk] = fv


def scenario_radio(
    *,
    presets: dict[str, dict[str, Any]],
    prefix: str,
    apply_keys: list[str],
    container_help_caption: str = "Pick a scenario. The settings below get pre-filled.",
) -> None:
    """Render a Step-1 scenario picker (radio + Apply button).

    Each entry in `presets` must have `label` (display heading) and `hint`
    (caption string). It must also include one entry per key in
    `apply_keys`. On Apply, sets `st.session_state[f"{prefix}_{key}"]` for
    each apply_key, sets `st.session_state[f"{prefix}_goal"]` to the
    scenario's label, then triggers `st.rerun()`.
    """
    st.caption(container_help_caption)
    chosen = st.radio(
        "Scenario",
        list(presets),
        index=0,
        key=f"{prefix}_scenario_label",
        label_visibility="collapsed",
    )
    meta = presets[chosen]
    st.markdown(f"**{meta['label']}**")
    st.caption(str(meta["hint"]))
    if st.button(
        "Apply scenario",
        help="Drops the scenario's values into the form below.",
        key=f"{prefix}_apply_btn",
        use_container_width=True,
    ):
        for k in apply_keys:
            if k in meta:
                st.session_state[f"{prefix}_{k}"] = meta[k]
        st.session_state[f"{prefix}_goal"] = meta["label"]
        st.rerun()


def render_run_label_sidebar(*, key: str) -> None:
    """Sidebar `Run label (optional)` text input shared by every workflow."""
    st.sidebar.text_input(
        "Run label (optional)",
        help=(
            "Free-text label saved in the fingerprint and shown in the "
            "results panel. Set automatically when you Apply a scenario. "
            "Does not drive the run."
        ),
        key=key,
    )


def render_app_footer() -> None:
    """Sidebar footer shared by every page. Right now: a Clear cache button.

    The bare `C` keyboard shortcut for clearing caches was popping up a
    confirmation dialog any time a user pressed C (e.g. trying to copy
    text). We disable that shortcut in `.streamlit/config.toml` by setting
    `client.toolbarMode = "viewer"`, and surface the action here so users
    still have an explicit way to invoke it.
    """
    st.sidebar.divider()
    if st.sidebar.button(
        "Clear cache",
        help=(
            "Drops every @st.cache_data and @st.cache_resource entry. "
            "Useful if the app shows stale content after a code change."
        ),
        key="__app_clear_cache_btn",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cleared.")
