"""T2I-Interp Toolkit — Streamlit playground (home page).

Run with `make app` or `uv run streamlit run app/streamlit_app.py`.

The home page frames *why* the toolkit exists, what research questions it
answers, and shows a sample output per workflow. Each sidebar page is a
runnable demo of one of the toolkit's four workflows — no code required.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="T2I-Interp Toolkit",
    page_icon=None,
    layout="wide",
)

# ── Hero ──────────────────────────────────────────────────────────────────────

st.title("T2I-Interp Toolkit")
st.markdown(
    "**A unified workflow toolkit for understanding *why* text-to-image diffusion "
    "models produce the images they do — and for steering them without retraining.**"
)
st.markdown(
    "Companion app for [**DreamReader: An Interpretability Toolkit for "
    "Text-to-Image Models**](https://arxiv.org/abs/2603.13299). Every page in "
    "the sidebar is a runnable demo; pick any one and click **Run**."
)

st.divider()

# ── Why this toolkit ─────────────────────────────────────────────────────────

st.subheader("Why interpretability for diffusion models?")
st.markdown(
    """
Diffusion T2I models like Stable Diffusion, SDXL, and FLUX are remarkable
at producing images from prompts — but they're **black boxes**. When a
generation comes out biased, off-distribution, or just wrong, you can't
open the model weights and read off the reason. The 4-billion-parameter
UNet doesn't have a `bias_for_doctor_is_male` flag inside it; that
information is *spread across* attention heads, time-steps, and feature
maps.

This toolkit gives researchers and engineers **four ways to look inside**:
locate where concepts live (Localisation), nudge generation in a direction
without retraining (Steering), transfer a behavior from one model to
another (Stitching), and discover the model's internal vocabulary of
features (SAEs).

Everything written here as a one-line CLI command (`t2i-steer`, `t2i-sae`,
…) is also runnable from the sidebar pages, with the same Hydra config
surface. The headline result of the paper — adding *spectacles* to
SDXL-Turbo via a low-rank adapter — can be reproduced in one click from
the **Steering** page.
"""
)

st.divider()

# ── Research questions ──────────────────────────────────────────────────────

st.subheader("What research questions does this answer?")
st.markdown(
    "Each row maps a concrete problem you might be trying to solve to the "
    "workflow that addresses it."
)
qa_rows = [
    (
        "My model produces stereotyped images (e.g. 'doctor' → always white men). "
        "*Where* in the UNet is the bias concentrated?",
        "**Localisation** — scale individual cross-attention heads to test "
        "whether ablating them removes the bias.",
    ),
    (
        "I want all my generations to lean toward concept Y (younger faces, "
        "spectacles, a painterly style) without fine-tuning the model.",
        "**Steering** — train a CAA / K-Steer / LoReFT direction once, inject "
        "at every generation. Paper §3.2.",
    ),
    (
        "I have a base SD 1.5 and an SD 1.5 LoRA fine-tune. Can I transfer the "
        "fine-tune's behavior into a different model via the activation space?",
        "**Stitching** — train a small MLP mapper that translates between "
        "the two models' activations. Paper §3.3.",
    ),
    (
        "What concepts does my model represent internally? Is there a 'whiskers' "
        "feature or a 'rim lighting' feature I can amplify?",
        "**Sparse Autoencoders** — decompose dense activations into "
        "sparse, monosemantic features. Modulate them and see what changes.",
    ),
    (
        "I ran 200 sweep configs last week. Which exact (model, dataset, seed, "
        "intervention) combination produced *this particular* image?",
        "**Fingerprints** — every run drops a 16-char hash + full config "
        "next to its outputs. Sidebar → Fingerprints to browse.",
    ),
]
for question, answer in qa_rows:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Q:** {question}")
    with c2:
        st.markdown(f"**A:** {answer}")
    st.markdown("")

st.divider()

# ── Four workflows with sample images ────────────────────────────────────────

st.subheader("The four workflows")
st.markdown(
    "Sample outputs are from real runs on this machine; click into a workflow "
    "to reproduce or modify."
)

SHOTS = Path(__file__).parent / "static" / "screenshots"

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Localisation")
    st.markdown(
        "Scale individual attention heads in cross-attention layers and see "
        "how the generated image changes. Useful for identifying *where* in "
        "the UNet a specific concept is bound. → Paper §3.1."
    )
    loc_img = SHOTS / "localisation.jpg"
    if loc_img.exists():
        st.image(
            str(loc_img),
            caption="Head ablation sweep at down/mid/up blocks (15-step SD 1.4 on MPS)",
        )

    st.markdown("### Stitching")
    st.markdown(
        "Train a small MLP that translates activations from model A's space "
        "into model B's space. At generation time, captured activations from "
        "A flow through the mapper and into B. → Paper §3.3."
    )
    stitch_img = SHOTS / "stitch.jpg"
    if stitch_img.exists():
        st.image(
            str(stitch_img), caption="Mapper-stitched generation (text-encoder → unet.conv_out)"
        )

with c2:
    st.markdown("### Steering")
    st.markdown(
        "Train a concept-direction vector (CAA), classifier-guided probe "
        "(K-Steer), or low-rank ReFT adapter (LoReFT) on paired prompts, "
        "then inject it at generation time. **The headline figure of the "
        "paper.** → Paper §3.2 / Fig 2."
    )
    steer_img = SHOTS / "steer.jpg"
    if steer_img.exists():
        st.image(str(steer_img), caption="Baseline vs steered (LoReFT on SDXL-Turbo)")

    st.markdown("### Sparse Autoencoders (SAEs)")
    st.markdown(
        "Decompose dense activations into sparse, interpretable features. "
        "Each modulation grid below shows what happens when one feature is "
        "scaled up or down at generation time. → Paper §3.4."
    )
    sae_img = SHOTS / "sae.jpg"
    if sae_img.exists():
        st.image(str(sae_img), caption="SAE feature modulation grid (SDXL-Turbo)")

st.divider()

# ── Reproducibility & infra ──────────────────────────────────────────────────

st.subheader("What ships with every run")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("##### Fingerprint")
    st.markdown(
        "A stable 16-char SHA that uniquely identifies the (model + dataset + "
        "seed + intervention) tuple. Same logical experiment on any machine = "
        "same hash. Browse all past runs on the **Fingerprints** page."
    )
with c2:
    st.markdown("##### Auto device + dtype")
    st.markdown(
        "Each playground page detects what your machine supports — CUDA → MPS "
        "→ CPU — and picks a sensible dtype default. Override in the sidebar "
        "if you need to."
    )
with c3:
    st.markdown("##### Model presets")
    st.markdown(
        "Switch between **SD 1.5**, **SDXL**, and **SDXL-Turbo** with one "
        "dropdown. Each preset bundles the model's CFG scale, denoising steps, "
        "and dtype defaults. Same as `model=sdxl_turbo` on the CLI."
    )

st.divider()

# ── CLI equivalent ──────────────────────────────────────────────────────────

st.subheader("Same tools, four equivalent terminal commands")
st.markdown(
    "Every demo here is a thin wrapper around one of these. Build the "
    "intuition in the GUI, then automate / sweep via the CLI."
)
st.code(
    """t2i-localise model=sd15 device=mps dtype=bfloat16
t2i-steer --config-name=steer/loreft model=sdxl_turbo
t2i-stitch model=sd15 num_inference_steps=15
t2i-sae model=sdxl_turbo strengths='[-5,5]'""",
    language="bash",
)

st.divider()

# ── Get started CTA ─────────────────────────────────────────────────────────

st.subheader("Start here")
st.markdown(
    """
- **Not sure which tool you need?** → Click **Recipes** in the sidebar.
  Each card is a *goal* (add an attribute, find where a concept lives,
  discover internal features, …) with the matching workflow + suggested
  config.
- **Already know which workflow you want?** → Click straight to
  **Localisation**, **Steering**, **Stitching**, or **SAE**. Each page
  has its own "common goals this serves" expander and a free-form
  *What are you trying to achieve?* text box that gets shown back in
  the results.
- **Want to reproduce the paper's headline result?** → Steering →
  *Reproduce Figure 2*.
- **Looking for past runs?** → **Fingerprints**.
- **Need a vocab refresher** (CFG, alpha, attn2, LoReFT, …)? → **Glossary**.
"""
)

st.caption(
    "Built on Streamlit + Hydra + diffusers. Source: "
    "[github.com/Social-AI-Studio/T2I_Interp_toolkit](https://github.com/Social-AI-Studio/T2I_Interp_toolkit)"
)
