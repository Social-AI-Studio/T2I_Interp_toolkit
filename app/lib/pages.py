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


def render_metrics_panel(
    metrics: dict[str, Any] | None,
    *,
    featured_prefixes: tuple[tuple[str, str], ...] = (
        ("baseline/clip", "CLIP (baseline)"),
        ("steered/clip", "CLIP (steered)"),
        ("baseline/fid", "FID (baseline)"),
        ("steered/fid", "FID (steered)"),
        ("baseline/lpips", "LPIPS (baseline)"),
        ("steered/lpips", "LPIPS (steered)"),
    ),
) -> None:
    """Render the Metrics container — tile callouts for featured scalars,
    JSON expander for everything else, info banner when empty.

    Keys in `metrics.json` look like `baseline/clip/clip_score` — match by
    prefix so per-scorer suffixes don't tie the UI to a specific module
    path. List values (per-prompt scores) are averaged; NaN renders as "—".
    """
    if metrics is None:
        return
    import math

    def _fmt(v):
        if isinstance(v, list):
            v = [x for x in v if isinstance(x, int | float) and not math.isnan(x)]
            if not v:
                return "—"
            v = sum(v) / len(v)
        if not isinstance(v, int | float) or math.isnan(v):
            return "—"
        return f"{v:.3f}"

    with st.container(border=True):
        st.markdown("##### Metrics")
        if not metrics:
            st.info(
                "metrics.json is empty — the workflow finished but no "
                "per-spec scorer returned a value. Check the run logs above "
                "for per-metric failure messages."
            )
            return
        shown: list[tuple[str, str]] = []
        for prefix, label in featured_prefixes:
            match = next((v for k, v in metrics.items() if k.startswith(prefix + "/")), None)
            if match is None and prefix in metrics:
                match = metrics[prefix]
            if match is not None:
                shown.append((label, _fmt(match)))
        if shown:
            st.caption(
                "Lower FID/LPIPS = closer to baseline distribution. Higher "
                "CLIP = better prompt alignment. Paper Figure 2 reports "
                "these three."
            )
            cols = st.columns(min(len(shown), 3))
            for i, (label, value) in enumerate(shown):
                cols[i % len(cols)].metric(label, value)
        with st.expander("All metrics (full JSON)", expanded=False):
            st.json(metrics)


def render_wandb_panel(wandb_run: dict[str, Any] | None) -> None:
    """Render a W&B run panel: project + run-name metadata + an "Open in W&B"
    link button.

    `wandb_run` is the payload `outputs.load_wandb_run` returns (None when
    wandb wasn't enabled for the run). An earlier version also embedded the
    live run dashboard via `st.components.v1.iframe(run.url)` — that was
    silently dead UX. W&B sets `X-Frame-Options: SAMEORIGIN` and a CSP
    `frame-ancestors 'self'` on every run page, so the iframe never
    actually loaded from a Streamlit origin. The link button is the
    reliable path; the metric tiles in the Metrics panel already surface
    the values the workflow logged via `wandb.log(...)`.
    """
    if not wandb_run:
        return
    url = wandb_run.get("url")
    if not url:
        return
    with st.container(border=True):
        st.markdown("##### W&B run")
        label = wandb_run.get("name") or wandb_run.get("id") or "Open in W&B"
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**Project:** `{wandb_run.get('project', '?')}`")
            st.markdown(f"**Run name:** `{label}`")
        with cols[1]:
            st.link_button(
                "Open in W&B ↗",
                url,
                use_container_width=True,
                help="Opens the live W&B dashboard with plots, tables, "
                "media panels, and the fingerprint artifact.",
            )
        st.caption(
            "Per-step charts and per-image media panels live on W&B "
            "itself — W&B blocks iframe embedding, so use the button "
            "above. The Metrics panel right below surfaces the final "
            "scalar values from this run."
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
