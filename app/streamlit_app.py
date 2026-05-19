"""T2I-Interp Toolkit — Streamlit playground.

Run with `make app` or `uv run streamlit run app/streamlit_app.py`.

The home page describes the four workflows; each workflow has its own
page in the sidebar (Streamlit picks them up from `app/pages/`).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="T2I-Interp Toolkit",
    page_icon=None,
    layout="wide",
)

st.title("T2I-Interp Toolkit — interactive playground")

st.markdown(
    """
This is the companion app for the [**DreamReader: An Interpretability Toolkit
for Text-to-Image Models**](https://arxiv.org/abs/2603.13299) paper. Each
sidebar page is a runnable demo of one of the toolkit's four workflows —
no code required.

Pick a workflow from the sidebar, configure it via the widgets, hit
**Run**, and inspect the generated images + the reproducibility
fingerprint.
"""
)

st.divider()

st.subheader("Workflows")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Localisation")
    st.markdown(
        "Scale individual attention heads in cross-attention layers and "
        "see how the generated image changes. Useful for identifying "
        "*where* in the UNet a specific concept is bound. → §3.1 of the paper."
    )

    st.markdown("### Stitching")
    st.markdown(
        "Train an MLP mapper that translates activations from model A "
        "into model B's space — then inject mapped activations during "
        "generation. Foundation for cross-model transfer. → §3.3."
    )

with c2:
    st.markdown("### Steering")
    st.markdown(
        "Train a concept-direction vector (CAA), classifier-guided probe "
        "(K-Steer), or low-rank ReFT adapter (LoReFT) on paired prompts, "
        "then inject it at generation time. Headline figure of the paper. → §3.2."
    )

    st.markdown("### SAEs (Sparse Autoencoders)")
    st.markdown(
        "Decompose dense activations into sparse, interpretable features. "
        "Modulate individual features to see what concept each one binds. → §3.4."
    )

st.divider()

st.subheader("Behind every run")

st.markdown(
    """
- **Every run writes a `fingerprint.json`** — a stable 16-char hash of the
  model + dataset + seed + intervention. Same logical experiment on any
  machine produces the same hash. Browse all past runs in the
  **Fingerprints** sidebar page.
- **Device + dtype auto-detect.** This app picks CUDA → MPS → CPU based on
  what your machine supports.
- **Model presets.** Switch between SD 1.5, SDXL, and SDXL-Turbo with one
  dropdown (compose-equivalent to `t2i-steer model=sdxl_turbo` etc. on the CLI).
"""
)

st.subheader("Equivalent CLI")
st.markdown(
    "Every demo here is a thin wrapper around one of these terminal commands "
    "— see [README.md](https://github.com/Social-AI-Studio/T2I_Interp_toolkit#readme) for details."
)
st.code(
    """t2i-localise model=sd15 device=mps dtype=bfloat16
t2i-steer --config-name=steer/loreft model=sdxl_turbo
t2i-stitch model=sd15 num_inference_steps=15
t2i-sae model=sdxl_turbo strengths='[-5,5]'""",
    language="bash",
)
