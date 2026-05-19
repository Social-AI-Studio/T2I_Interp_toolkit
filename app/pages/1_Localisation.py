"""Localisation playground — scale a cross-attention head and see the effect."""

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

st.set_page_config(page_title="Localisation • T2I-Interp", layout="wide")

st.title("Localisation — head ablation sweeps")

st.markdown(
    "Picks a single cross-attention head in the UNet and scales its output "
    "by `factor` for a chosen step range. Compare the generated image against "
    "the unaltered baseline to localise which head carries which concept."
)

# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(default="(use config default)", options=("sd15", "sdxl_turbo"))

prompt = st.sidebar.text_input("Prompt", value="a unicorn")
target_layer = st.sidebar.text_input(
    "Target layer",
    value="down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
    help="Underscore-sanitised UNet path. Run a generation first then check the "
    "logs for available layer names if unsure.",
)
target_head = st.sidebar.selectbox("Target head index", list(range(8)), index=0)
factor = st.sidebar.slider(
    "Scale factor",
    -5.0,
    5.0,
    0.0,
    0.5,
    help="0.0 = zero-ablate the head; 1.0 = no change; >1.0 amplifies; "
    "negative = inverts the head's contribution.",
)
n_steps = st.sidebar.slider("Inference steps", 4, 50, 15)
seed = st.sidebar.number_input("Seed", value=42, step=1)

# ── Build the override list ──────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_loc_")
overrides = [
    f"device={device}",
    f"dtype={dtype}",
    f"prompt={prompt}",
    f"target_layer={target_layer}",
    f"target_heads=[{target_head}]",
    f"factor={factor}",
    f"num_inference_steps={n_steps}",
    f"seed={seed}",
    f"output_dir={out_dir}",
    f"hydra.run.dir={out_dir}/.hydra",
    "wandb.project=null",
]
if preset:
    overrides.append(f"model={preset}")

st.subheader("CLI equivalent")
st.code("t2i-localise " + " ".join(overrides[:7]), language="bash")

# ── Run + show output ─────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status("Running localisation…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-localise", overrides, output_dir=out_dir):
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
        with c2:
            st.json(fp, expanded=False)
