"""T2I-Interp Toolkit Streamlit playground (home page).

Run with `make app` or `uv run streamlit run app/streamlit_app.py`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.lib import FIG2_SPECTACLES_PAYLOAD

st.set_page_config(
    page_title="DreamReader Demo",
    page_icon=None,
    layout="wide",
)

SHOTS = Path(__file__).parent / "static" / "screenshots"


# ── Hero ──────────────────────────────────────────────────────────────────────

st.title("DreamReader")
st.markdown(
    "##### An interpretability toolkit for text-to-image diffusion models. "
    "See why a model produced an image, and steer it without retraining."
)

st.markdown(
    "Companion app for "
    "[DreamReader: An Interpretability Toolkit for Text-to-Image Models]"
    "(https://arxiv.org/abs/2603.13299) (Prakash et al., 2026). "
    "Source on "
    "[GitHub](https://github.com/Social-AI-Studio/T2I_Interp_toolkit)."
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button(
        "▶ Reproduce paper Fig 2",
        type="primary",
        use_container_width=True,
        help="LoReFT adds spectacles to SDXL-Turbo character prompts. Drops you on Steering with everything pre-filled.",
    ):
        # Pre-fill Steering with the spectacles payload so the user lands on
        # a form already configured for the headline result. One more click
        # (Run) and they have Fig 2.
        st.session_state["recipe_payload"] = FIG2_SPECTACLES_PAYLOAD
        st.switch_page("pages/2_Steering.py")
with c2:
    if st.button("Browse recipes", use_container_width=True):
        st.switch_page("pages/0_Recipes.py")
with c3:
    if st.button("Glossary of terms", use_container_width=True):
        st.switch_page("pages/6_Glossary.py")

st.divider()


# ── How an experiment works (paper Figure 1) ─────────────────────────────────

st.subheader("How a DreamReader experiment works")
st.caption("Every workflow follows the same four steps (paper, Figure 1).")

steps = [
    ("1. Configure", "YAML or sidebar form. Pick model, dataset, layers, prompts."),
    ("2. Execute", "One CLI command or one Run button."),
    ("3. Workflow", "Localisation, Steering, Stitching, or SAE."),
    ("4. Report", "Side-by-side images, metrics, and a reproducibility fingerprint."),
]
cols = st.columns(4)
for col, (title, body) in zip(cols, steps, strict=True):
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)

st.divider()


# ── Four workflows with sample outputs ───────────────────────────────────────

st.subheader("The four workflows")
st.caption("Sample outputs from real runs on this machine. Click a card to open it.")


def _workflow_card(
    *,
    title: str,
    blurb: str,
    image: Path,
    caption: str,
    page: str,
    button_key: str,
    primary: bool = False,
) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(blurb)
        if image.exists():
            st.image(str(image), caption=caption, use_container_width=True)
        if st.button(
            f"Open {title}",
            key=button_key,
            type="primary" if primary else "secondary",
            use_container_width=True,
        ):
            st.switch_page(page)


c1, c2 = st.columns(2)
with c1:
    _workflow_card(
        title="Steering",
        blurb=(
            "Train a concept direction (CAA, K-Steer, or LoReFT) on paired "
            "prompts, then add it during generation. The paper's headline "
            "result (Fig 2 spectacles)."
        ),
        image=SHOTS / "steer.jpg",
        caption="Baseline vs steered (LoReFT on SDXL-Turbo)",
        page="pages/2_Steering.py",
        button_key="open_steer",
        primary=True,
    )
    _workflow_card(
        title="Stitching",
        blurb=(
            "Train a small MLP that translates activations from model A "
            "into model B. Used for cross-model behaviour transfer "
            "(paper §3.3)."
        ),
        image=SHOTS / "stitch.jpg",
        caption="Mapper-stitched generation (text encoder to UNet)",
        page="pages/3_Stitching.py",
        button_key="open_stitch",
    )

with c2:
    _workflow_card(
        title="Localisation",
        blurb=(
            "Scale one attention head at a time and watch what changes. "
            "Tells you where in the UNet a concept is bound (paper §3.1)."
        ),
        image=SHOTS / "localisation.jpg",
        caption="Head ablation sweep, 15-step SD 1.4 on MPS",
        page="pages/1_Localisation.py",
        button_key="open_loc",
    )
    _workflow_card(
        title="Sparse Autoencoders",
        blurb=(
            "Decompose dense activations into a sparse vocabulary of "
            "features. Push individual features up or down at generation "
            "time (paper §3.4)."
        ),
        image=SHOTS / "sae.jpg",
        caption="SAE feature modulation grid (SDXL-Turbo)",
        page="pages/4_SAE.py",
        button_key="open_sae",
    )

st.divider()


# ── Common problems and which workflow helps ─────────────────────────────────

st.subheader("Common problems and which workflow helps")

qa_rows = [
    (
        "My model produces stereotyped images "
        "(e.g. 'doctor' always white men). "
        "Where in the UNet is the bias?",
        "**Localisation.** Ablate one attention head at a time and see which ones change the bias.",
    ),
    (
        "I want all my generations to lean toward a concept "
        "(younger faces, spectacles, painterly style) without fine-tuning.",
        "**Steering.** Train a direction once, add it at every generation (paper §3.2).",
    ),
    (
        "I have a base SD 1.5 and a LoRA fine-tune. Can I transfer the "
        "fine-tune's behaviour into a different model through activations?",
        "**Stitching.** Train an MLP that translates between the two "
        "models' activation spaces (paper §3.3).",
    ),
    (
        "What concepts does my model represent internally? "
        "Is there a 'whiskers' feature I can amplify?",
        "**SAE.** Decompose dense activations into sparse, monosemantic features (paper §3.4).",
    ),
    (
        "I ran 200 sweep configs last week. "
        "Which (model, dataset, seed, intervention) produced this image?",
        "**Fingerprints.** Every run writes a 16-character hash next to its outputs.",
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


# ── What ships with every run ────────────────────────────────────────────────

st.subheader("What ships with every run")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("##### Fingerprint")
        st.markdown(
            "A 16-character SHA that uniquely identifies the (model, "
            "dataset, seed, intervention) tuple. Same logical experiment "
            "on any machine produces the same hash. Browse past runs on "
            "the Fingerprints page."
        )
with c2:
    with st.container(border=True):
        st.markdown("##### Auto device and dtype")
        st.markdown(
            "Each page detects what your machine supports (CUDA, MPS, or "
            "CPU) and picks a sensible dtype default. Override from the "
            "sidebar if you want to."
        )
with c3:
    with st.container(border=True):
        st.markdown("##### Model presets")
        st.markdown(
            "Switch between **SD 1.5**, **SDXL**, and **SDXL-Turbo** with "
            "one dropdown. Each preset bundles the model's CFG scale, "
            "denoising steps, and dtype. Same as `model=sdxl_turbo` on "
            "the CLI."
        )


st.divider()


# ── CLI equivalent ──────────────────────────────────────────────────────────

st.subheader("Same things from the terminal")
st.caption("Every demo here is a thin wrapper. Use these for sweeps and automation.")
st.code(
    """t2i-localise model=sd15 device=mps dtype=bfloat16
t2i-steer --config-name=steer/loreft model=sdxl_turbo
t2i-stitch model=sd15 num_inference_steps=15
t2i-sae model=sdxl_turbo strengths='[-5,5]'""",
    language="bash",
)


st.caption(
    "Built on Streamlit, Hydra, and diffusers. "
    "DreamReader is a research toolkit. Interpretability results are "
    "approximate. Read the paper's Ethics Statement before using outputs "
    "to make claims about individuals or groups."
)
