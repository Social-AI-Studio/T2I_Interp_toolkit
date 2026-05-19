"""SAE playground — discover sparse features and modulate them at generation time."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from app.lib import (
    collect_images,
    device_dtype_picker,
    load_fingerprint,
    model_preset_picker,
    run_workflow,
)

st.set_page_config(page_title="SAE • T2I-Interp", layout="wide")

st.title("Sparse Autoencoders — feature discovery + modulation")

st.markdown(
    "Loads pretrained sparse autoencoders trained on SDXL-Turbo UNet activations, "
    "captures the SAE latents for your prompt, picks the top-activating features, "
    "and re-generates with each feature scaled by a set of `strengths`. "
    "Output is a grid: rows = features, columns = strengths."
)

# Surface missing checkpoints early — sae.ipynb / t2i-sae need these.
ckpt_dir = Path("./sdxl-unbox/checkpoints")
if not ckpt_dir.exists():
    st.error(
        "Missing SAE checkpoints at `./sdxl-unbox/checkpoints/`. "
        "Run `t2i-migrate-sae --checkpoint-dir ./sdxl-unbox/checkpoints` after "
        "downloading from `anonymous-author-129/sdxl-unbox-saes` on HuggingFace, "
        "or follow `notebooks/sae.ipynb` for the full setup flow."
    )
    st.stop()

# ── Sidebar config ───────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(default="sdxl_turbo")

prompt = st.sidebar.text_input("Prompt", value="a red apple")

st.sidebar.markdown("**Strengths to modulate each feature by**")
strength_lo = st.sidebar.slider("Min strength", -20.0, 0.0, -5.0, 0.5)
strength_hi = st.sidebar.slider("Max strength", 0.0, 20.0, 5.0, 0.5)
strengths = sorted({strength_lo, 0.0, strength_hi})  # always include baseline

n_features_to_plot = st.sidebar.slider("Top features to modulate", 1, 6, 2)
n_top_features = st.sidebar.slider("Capture top-K features", 2, 20, 10)

# ── Build overrides ──────────────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_sae_")
overrides = [
    f"device={device}",
    f"dtype={dtype}",
    f"prompt={prompt}",
    f"strengths=[{','.join(str(s) for s in strengths)}]",
    f"n_features_to_plot={n_features_to_plot}",
    f"n_top_features={n_top_features}",
    f"output_dir={out_dir}",
    f"hydra.run.dir={out_dir}/.hydra",
    "wandb.project=null",
]
if preset:
    overrides.append(f"model={preset}")

st.subheader("CLI equivalent")
st.code("t2i-sae " + " ".join(overrides[:6]) + " …", language="bash")

# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status("Capturing activations + modulating features…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-sae", overrides, output_dir=out_dir):
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
        st.subheader(f"Feature modulation grid ({len(images)} image(s))")
        for img in images:
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
