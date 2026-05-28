"""T2I-Interp Toolkit Streamlit playground (home page).

Run with `make app` or `uv run streamlit run app/streamlit_app.py`.
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
st.markdown("**See why a text-to-image model produced an image, and steer it without retraining.**")
st.markdown(
    "This is the companion app for "
    "[DreamReader: An Interpretability Toolkit for Text-to-Image Models]"
    "(https://arxiv.org/abs/2603.13299). "
    "Every page in the sidebar is a runnable demo. Pick one and press Run."
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Browse recipes", type="primary", use_container_width=True):
        st.switch_page("pages/0_Recipes.py")
with c2:
    if st.button("Reproduce paper Fig 2", use_container_width=True):
        st.switch_page("pages/2_Steering.py")
with c3:
    if st.button("Read the glossary", use_container_width=True):
        st.switch_page("pages/6_Glossary.py")

st.divider()

# ── Why this toolkit ─────────────────────────────────────────────────────────

st.subheader("Why interpretability for diffusion models?")
st.markdown(
    """
Models like Stable Diffusion, SDXL, and FLUX produce great images, but
they are black boxes. When a generation comes out biased, off, or just
wrong, you can't open the weights and read off the reason. There is no
`bias_for_doctor_is_male` flag inside the 4-billion-parameter UNet.
That information is spread across attention heads, time-steps, and
feature maps.

This toolkit gives you four ways to look inside:
"""
)

c1, c2, c3, c4 = st.columns(4)
for col, name, blurb in zip(
    [c1, c2, c3, c4],
    ["Localisation", "Steering", "Stitching", "SAE"],
    [
        "Find where a concept lives by scaling one attention head at a time.",
        "Nudge generation toward (or away from) a concept without retraining.",
        "Move behaviour from one model into another through an MLP mapper.",
        "Decompose dense activations into a sparse vocabulary of features.",
    ],
    strict=True,
):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(blurb)

st.markdown("")
st.markdown(
    "Each demo is a thin wrapper around a CLI command (`t2i-localise`, "
    "`t2i-steer`, `t2i-stitch`, `t2i-sae`). Build intuition in the browser, "
    "then sweep via the CLI for paper-scale runs."
)

st.divider()

# ── Research questions ──────────────────────────────────────────────────────

st.subheader("Common problems and which workflow helps")

qa_rows = [
    (
        "My model generates stereotyped images "
        "(e.g. 'doctor' always white men). "
        "Where in the UNet is the bias?",
        "**Localisation.** Ablate one attention head at a time and watch which "
        "ones change the bias.",
    ),
    (
        "I want all my generations to lean toward a concept "
        "(younger faces, spectacles, painterly style) without fine-tuning.",
        "**Steering.** Train a direction once, add it at every generation. Paper §3.2.",
    ),
    (
        "I have a base SD 1.5 and a LoRA fine-tune. Can I transfer the "
        "fine-tune's behaviour into a different model through activations?",
        "**Stitching.** Train an MLP that translates between the two models' "
        "activation spaces. Paper §3.3.",
    ),
    (
        "What concepts does my model represent internally? "
        "Is there a 'whiskers' feature I can amplify?",
        "**SAE.** Decompose dense activations into sparse, monosemantic "
        "features. Push them up or down at generation time.",
    ),
    (
        "I ran 200 sweep configs last week. "
        "Which (model, dataset, seed, intervention) made *this* image?",
        "**Fingerprints.** Every run writes a 16-char hash next to its outputs. "
        "Browse them on the Fingerprints page.",
    ),
]
for question, answer in qa_rows:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**You ask:** {question}")
    with c2:
        st.markdown(f"**Use:** {answer}")
    st.markdown("")

st.divider()

# ── Four workflows with sample images ────────────────────────────────────────

st.subheader("Sample outputs from each workflow")
st.caption("Real runs on this machine. Click a workflow to reproduce or tweak.")

SHOTS = Path(__file__).parent / "static" / "screenshots"

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("### Localisation")
        st.markdown(
            "Scale one attention head at a time and see how the image changes. "
            "Tells you where a concept is bound in the UNet. Paper §3.1."
        )
        loc_img = SHOTS / "localisation.jpg"
        if loc_img.exists():
            st.image(
                str(loc_img),
                caption="Head ablation sweep, 15-step SD 1.4 on MPS",
            )
        if st.button("Open Localisation", key="open_loc"):
            st.switch_page("pages/1_Localisation.py")

    with st.container(border=True):
        st.markdown("### Stitching")
        st.markdown(
            "Train a small MLP that translates activations from model A into "
            "model B's space. At generation time, A's activations flow through "
            "the mapper into B. Paper §3.3."
        )
        stitch_img = SHOTS / "stitch.jpg"
        if stitch_img.exists():
            st.image(
                str(stitch_img),
                caption="Mapper-stitched generation (text-encoder to unet.conv_out)",
            )
        if st.button("Open Stitching", key="open_stitch"):
            st.switch_page("pages/3_Stitching.py")

with c2:
    with st.container(border=True):
        st.markdown("### Steering")
        st.markdown(
            "Train a concept direction (CAA), classifier (K-Steer), or low-rank "
            "adapter (LoReFT) on paired prompts, then inject it during "
            "generation. The paper's headline result. Paper §3.2 and Fig 2."
        )
        steer_img = SHOTS / "steer.jpg"
        if steer_img.exists():
            st.image(
                str(steer_img),
                caption="Baseline vs steered (LoReFT on SDXL-Turbo)",
            )
        if st.button("Open Steering", key="open_steer", type="primary"):
            st.switch_page("pages/2_Steering.py")

    with st.container(border=True):
        st.markdown("### Sparse Autoencoders (SAE)")
        st.markdown(
            "Decompose dense activations into sparse, interpretable features. "
            "The grid shows what happens when one feature is scaled up or down "
            "at generation time. Paper §3.4."
        )
        sae_img = SHOTS / "sae.jpg"
        if sae_img.exists():
            st.image(
                str(sae_img),
                caption="SAE feature modulation grid (SDXL-Turbo)",
            )
        if st.button("Open SAE", key="open_sae"):
            st.switch_page("pages/4_SAE.py")

st.divider()

# ── Reproducibility & infra ──────────────────────────────────────────────────

st.subheader("What ships with every run")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("##### Fingerprint")
        st.markdown(
            "A 16-character SHA that uniquely identifies the (model, dataset, "
            "seed, intervention) tuple. Same logical experiment on any "
            "machine produces the same hash. Past runs live on the "
            "Fingerprints page."
        )
with c2:
    with st.container(border=True):
        st.markdown("##### Auto device and dtype")
        st.markdown(
            "Each page detects what your machine supports (CUDA, MPS, or CPU) "
            "and picks a sensible dtype default. Override from the sidebar "
            "if you want to."
        )
with c3:
    with st.container(border=True):
        st.markdown("##### Model presets")
        st.markdown(
            "Switch between **SD 1.5**, **SDXL**, and **SDXL-Turbo** with one "
            "dropdown. Each preset bundles the model's CFG scale, denoising "
            "steps, and dtype. Same as `model=sdxl_turbo` on the CLI."
        )

st.divider()

# ── CLI equivalent ──────────────────────────────────────────────────────────

st.subheader("The same things from the terminal")
st.caption("Every demo here is a thin wrapper. Use these for sweeps and automation.")
st.code(
    """t2i-localise model=sd15 device=mps dtype=bfloat16
t2i-steer --config-name=steer/loreft model=sdxl_turbo
t2i-stitch model=sd15 num_inference_steps=15
t2i-sae model=sdxl_turbo strengths='[-5,5]'""",
    language="bash",
)

st.divider()

# ── Get started CTA ─────────────────────────────────────────────────────────

st.subheader("Where to start")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("##### Not sure which tool you need?")
        st.markdown(
            "Open **Recipes**. Each card is a goal "
            "(add an attribute, find where a concept lives, discover internal "
            "features). One click pre-fills the matching workflow."
        )
        if st.button("Browse recipes", key="cta_recipes"):
            st.switch_page("pages/0_Recipes.py")

    with st.container(border=True):
        st.markdown("##### Already know what you want?")
        st.markdown(
            "Jump straight to **Localisation**, **Steering**, **Stitching**, "
            "or **SAE**. Each page has a *What are you trying to achieve?* "
            "field at the top that gets stored with the run."
        )

with c2:
    with st.container(border=True):
        st.markdown("##### Want the paper's headline result?")
        st.markdown(
            "Open **Steering** and press *Reproduce Figure 2*. LoReFT plus "
            "SDXL-Turbo adds spectacles to character prompts in about a minute."
        )
        if st.button("Reproduce Fig 2", key="cta_fig2", type="primary"):
            st.switch_page("pages/2_Steering.py")

    with st.container(border=True):
        st.markdown("##### Need a vocab refresher?")
        st.markdown(
            "Open **Glossary**. Plain-English explanations of CFG, alpha, "
            "attn2, LoReFT, K-Steer, CAA, and the rest of the toolkit's jargon."
        )
        if st.button("Open glossary", key="cta_glossary"):
            st.switch_page("pages/6_Glossary.py")

st.caption(
    "Built on Streamlit, Hydra, and diffusers. Source on "
    "[GitHub](https://github.com/Social-AI-Studio/T2I_Interp_toolkit)."
)
