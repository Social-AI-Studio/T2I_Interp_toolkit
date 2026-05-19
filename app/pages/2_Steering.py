"""Steering playground — train a concept direction and inject it during generation."""

from __future__ import annotations

import tempfile
import time

import streamlit as st

from app.lib import (
    collect_images,
    device_dtype_picker,
    load_fingerprint,
    model_preset_picker,
    run_workflow,
)

st.set_page_config(page_title="Steering • T2I-Interp", layout="wide")

st.title("Steering — concept direction injection")

st.markdown(
    "Trains a steering vector (CAA), classifier (K-Steer), or low-rank "
    "adapter (LoReFT) from paired positive/negative prompts in a dataset, "
    "then injects it during generation. The headline figure of the paper "
    "uses **LoReFT + SDXL-Turbo** to add spectacles to character prompts."
)

# ── Quick presets ───────────────────────────────────────────────────────────
if "steer_preset" not in st.session_state:
    st.session_state.steer_preset = None

c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button(
        "Reproduce Figure 2", help="LoReFT + SDXL-Turbo + spectacles prompts, paper-style"
    ):
        st.session_state.steer_preset = "fig2"
with c2:
    if st.button("Quick smoke run", help="Tiny scale just to confirm the wiring works"):
        st.session_state.steer_preset = "smoke"

PRESET_DEFAULTS = {
    "fig2": {
        "steer_type": "loreft",
        "prompts": "A photo of Jack Sparrow\nA photo of Simba",
        "alpha": 10.0,
        "max_samples": 200,
        "train_steps": 50,
        "model_preset": "sdxl_turbo",
    },
    "smoke": {
        "steer_type": "loreft",
        "prompts": "A photo of a cat",
        "alpha": 5.0,
        "max_samples": 10,
        "train_steps": 2,
        "model_preset": "sdxl_turbo",
    },
}
PD = PRESET_DEFAULTS.get(st.session_state.steer_preset, {})

# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(default=PD.get("model_preset", "sdxl_turbo"))

_steer_opts = ["loreft", "caa", "ksteer"]
steer_type = st.sidebar.selectbox(
    "Steering method",
    _steer_opts,
    index=_steer_opts.index(PD.get("steer_type", "loreft")),
)
prompts_raw = st.sidebar.text_area(
    "Prompts (one per line)",
    value=PD.get("prompts", "A photo of Jack Sparrow\nA photo of Simba"),
    help="Generated once as baseline, once steered.",
)
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
alpha = st.sidebar.slider(
    "Alpha (steering strength)",
    0.0,
    30.0,
    float(PD.get("alpha", 10.0)),
    0.5,
    help="0.0 = no steering. Higher = stronger. SDXL-Turbo + LoReFT works well around 10-20.",
)
max_samples = st.sidebar.slider("Training samples", 10, 1000, PD.get("max_samples", 100))
train_steps = st.sidebar.slider("Training steps", 2, 500, PD.get("train_steps", 50))

# ── Build overrides ──────────────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_steer_")
overrides = [
    f"--config-name=steer/{steer_type}",
    f"device={device}",
    f"dtype={dtype}",
    f"alpha={alpha}",
    f"max_samples={max_samples}",
    f"train_steps={train_steps}",
    f"prompts=[{','.join(prompts)}]",
    f"save_dir={out_dir}/cache",
    f"output_dir={out_dir}",
    f"hydra.run.dir={out_dir}/.hydra",
    "wandb.project=null",
]
if preset:
    overrides.append(f"model={preset}")

st.subheader("CLI equivalent")
st.code("t2i-steer " + " ".join(overrides[:8]) + " …", language="bash")

# ── Run ───────────────────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status(f"Training {steer_type.upper()} + generating…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-steer", overrides, output_dir=out_dir):
            if isinstance(event, str):
                recent.append(event)
                line_box.code("\n".join(recent[-20:]))
            else:
                result = event
        elapsed = time.time() - start
        if result is not None and result.returncode == 0:
            status.update(label=f"Done in {elapsed:.1f}s", state="complete")
        else:
            status.update(label="Run failed — see logs above", state="error")

    st.divider()

    images = collect_images(out_dir)
    if images:
        st.subheader(f"Output images ({len(images)})")
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.warning("No images produced — check logs above.")

    fp = load_fingerprint(out_dir)
    if fp:
        st.subheader("Run fingerprint")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Hash", fp["fingerprint_hash"])
            st.metric("Workflow", fp["workflow"])
            st.metric("Alpha", str(fp["intervention"].get("alpha", "—")))
        with c2:
            st.json(fp, expanded=False)
