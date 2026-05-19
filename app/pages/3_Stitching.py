"""Stitching playground — train an MLP mapper across two activation spaces."""

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

st.set_page_config(page_title="Stitching • T2I-Interp", layout="wide")

st.title("Stitching — cross-layer activation mapper")

st.markdown(
    "Trains a small MLP that maps activations from `layer_a` (typically a "
    "text-encoder output) into `layer_b`'s space (typically a UNet block). "
    "At generation time, captured activations from `layer_a` are passed "
    "through the mapper and injected at `layer_b`. Fig 4 in the paper."
)

# ── Sidebar config ───────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(default="sd15")

prompts_raw = st.sidebar.text_area("Prompts", value="a photo of a person")
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]

st.sidebar.markdown("**Quick-mode params** (drop hidden_dim/samples for speed)")
hidden_dim = st.sidebar.slider("Mapper hidden_dim", 64, 1024, 256, step=64)
max_samples = st.sidebar.slider("Train samples", 10, 500, 50)
num_steps = st.sidebar.slider("Mapper training steps", 5, 1000, 50)
num_inference_steps = st.sidebar.slider("Inference steps", 4, 50, 15)

# ── Build overrides ──────────────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_stitch_")
overrides = [
    f"device={device}",
    f"dtype={dtype}",
    f"hidden_dim={hidden_dim}",
    f"max_samples={max_samples}",
    f"num_steps={num_steps}",
    f"num_inference_steps={num_inference_steps}",
    f"prompts=[{','.join(prompts)}]",
    f"save_dir={out_dir}/cache",
    f"output_dir={out_dir}",
    f"hydra.run.dir={out_dir}/.hydra",
    "wandb.project=null",
]
if preset:
    overrides.append(f"model={preset}")

st.subheader("CLI equivalent")
st.code("t2i-stitch " + " ".join(overrides[:7]) + " …", language="bash")

# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status("Training mapper + generating…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-stitch", overrides, output_dir=out_dir):
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
