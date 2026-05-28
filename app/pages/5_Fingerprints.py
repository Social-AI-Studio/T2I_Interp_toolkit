"""Fingerprint browser. Lists every past run's reproducibility hash."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.lib import render_app_footer, scan_fingerprints

st.set_page_config(page_title="Fingerprints • T2I-Interp", layout="wide")

st.title("Run fingerprints")
st.caption("Browse past runs by their reproducibility hash.")

st.markdown(
    "Every workflow writes a `fingerprint.json` next to its output images. "
    "This page walks the default output directories and lists them. The "
    "16-character **hash** is the canonical identifier for a logical "
    "experiment. Same model, dataset, seed, and intervention produce the "
    "same hash on any machine."
)

# ── Scan roots ────────────────────────────────────────────────────────────────
default_roots = [
    "./output_images",
    "./notebook_runs",
    "./test_run",
    "./cli_verify",
    "/tmp",  # picks up streamlit_loc_, streamlit_steer_, etc. ephemeral dirs
]
extra_roots_input = st.text_input(
    "Additional directories (comma-separated)",
    value="",
    help="Walked recursively for any **/fingerprint.json files.",
)
extra_roots = [r.strip() for r in extra_roots_input.split(",") if r.strip()]
all_roots = default_roots + extra_roots

rows = scan_fingerprints(all_roots)

st.caption(f"Scanned: {', '.join(str(Path(r).resolve()) for r in all_roots if Path(r).exists())}")

if not rows:
    st.info("No fingerprints found yet. Run a workflow from one of the sidebar pages first.")
    st.stop()

# ── Filter widgets ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    wf_filter = st.multiselect(
        "Workflow",
        sorted({r["workflow"] for r in rows}),
        default=sorted({r["workflow"] for r in rows}),
    )
with c2:
    model_filter = st.multiselect(
        "Model",
        sorted({r["model"] for r in rows}),
        default=sorted({r["model"] for r in rows}),
    )
with c3:
    hide_dirty = st.checkbox(
        "Hide dirty-git runs", value=False, help="Exclude runs from uncommitted code."
    )

filtered = [
    r
    for r in rows
    if r["workflow"] in wf_filter
    and r["model"] in model_filter
    and (not hide_dirty or not r["git_dirty"])
]

st.markdown(f"**{len(filtered)} run(s)**, sorted newest first.")
df = pd.DataFrame(filtered)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Inspect one run in detail ────────────────────────────────────────────────
if filtered:
    pick = st.selectbox(
        "Inspect a run",
        [f"{r['hash']} · {r['workflow']} · {r['timestamp']}" for r in filtered],
    )
    if pick:
        chosen = filtered[
            [f"{r['hash']} · {r['workflow']} · {r['timestamp']}" for r in filtered].index(pick)
        ]
        st.markdown(f"### `{chosen['hash']}` ({chosen['workflow']})")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Model", chosen["model"])
            st.metric("Dataset", chosen["dataset"] or "-")
            st.metric("Seed", str(chosen["seed"]))
            st.metric("Git SHA", chosen["git_sha"] + (" (dirty)" if chosen["git_dirty"] else ""))
            st.code(f"cat '{chosen['path']}/fingerprint.json'", language="bash")
        with c2:
            # Inline thumbnails of the run's output images.
            from app.lib import collect_images

            imgs = collect_images(chosen["path"])
            if imgs:
                st.markdown(f"**Output images ({len(imgs)}):**")
                for img in imgs[:6]:
                    st.image(str(img), caption=img.name, use_container_width=True)
                if len(imgs) > 6:
                    st.caption(f"...and {len(imgs) - 6} more in `{chosen['path']}`")

render_app_footer()
