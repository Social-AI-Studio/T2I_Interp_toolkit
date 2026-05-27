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

# ── Defaults + recipe-payload intake ─────────────────────────────────────────
_SAE_DEFAULTS: dict[str, object] = {
    "sae_goal": "",
    "sae_prompt": "a red apple",
    "sae_strength_lo": -5.0,
    "sae_strength_hi": 5.0,
    "sae_n_features_to_plot": 2,
    "sae_n_top_features": 10,
    "sae_model_preset": "sdxl_turbo",
}
for _k, _v in _SAE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_payload = st.session_state.get("recipe_payload")
if _payload and _payload.get("workflow") == "SAE":
    del st.session_state["recipe_payload"]
    if _payload.get("goal"):
        st.session_state["sae_goal"] = _payload["goal"]
    for _fk, _fv in _payload.get("fields", {}).items():
        _sk = f"sae_{_fk}"
        if _sk in _SAE_DEFAULTS:
            st.session_state[_sk] = _fv

# ── Page body ────────────────────────────────────────────────────────────────

st.title("Sparse Autoencoders — feature discovery + modulation")

st.markdown(
    "Loads pretrained sparse autoencoders trained on SDXL-Turbo UNet activations, "
    "captures the SAE latents for your prompt, picks the top-activating features, "
    "and re-generates with each feature scaled by a set of `strengths`. "
    "Output is a grid: rows = features, columns = strengths."
)

with st.expander("**Common goals this page serves**", expanded=False):
    st.markdown(
        """
- **Discover what features your model uses for a given prompt.**
- **Find a feature that controls a specific visual property** (shininess,
  texture, colour, object part).
- **Amplify or suppress a known feature index** to bias all generations.

See the **Recipes** page for one-click presets — clicking *Open* there
will pre-fill the form below.
"""
    )

st.text_input(
    "What are you trying to achieve? (optional)",
    placeholder='e.g. "Find a feature that controls shininess in fruit images"',
    help=(
        "Stored in the run fingerprint and shown back in the results panel. "
        "Pre-filled automatically if you arrived from a Recipe."
    ),
    key="sae_goal",
)

st.info(
    """
**How this affects the picture.** An SAE expresses the model's dense
activations as a sparse combination of ~5,000 *features* — each one
ideally corresponds to a single interpretable concept (a texture, a
color, an object part). Negative `strength` values *suppress* that
feature in the activation; positive values *amplify* it. The grid below
shows the same prompt regenerated with each top feature scaled to a
range of strengths, so you can read off what concept each one encodes
by watching what changes across each row.
""",
    icon="ℹ️",
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
preset = model_preset_picker(
    default=str(st.session_state.get("sae_model_preset", "sdxl_turbo")),
    key="sae_model_preset",
)

st.sidebar.text_input("Prompt", key="sae_prompt")

st.sidebar.markdown("**Strengths to modulate each feature by**")
st.sidebar.slider("Min strength", -20.0, 0.0, step=0.5, key="sae_strength_lo")
st.sidebar.slider("Max strength", 0.0, 20.0, step=0.5, key="sae_strength_hi")

st.sidebar.slider("Top features to modulate", 1, 6, key="sae_n_features_to_plot")
st.sidebar.slider("Capture top-K features", 2, 20, key="sae_n_top_features")

prompt = str(st.session_state["sae_prompt"])
strength_lo = float(st.session_state["sae_strength_lo"])
strength_hi = float(st.session_state["sae_strength_hi"])
strengths = sorted({strength_lo, 0.0, strength_hi})  # always include baseline
n_features_to_plot = int(st.session_state["sae_n_features_to_plot"])
n_top_features = int(st.session_state["sae_n_top_features"])
goal = str(st.session_state["sae_goal"])

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

    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        st.subheader(f"Feature modulation grid ({len(images)} image(s))")
        for img in images:
            st.image(str(img), caption=img.name, use_container_width=True)

        st.markdown("##### How to read these results")
        st.markdown(
            """
- **Each row = one feature** (e.g. feature `#1338`). The Top-K features
  were the ones most active for your prompt.
- **Each column = one strength value.** Left columns = the feature
  suppressed; right columns = amplified.
- **Across a row, look for what changes consistently.** If amplifying
  feature 1338 progressively adds shininess to your subject across the
  row → "1338" encodes a *shininess* concept. If amplifying it makes
  the image redder → it encodes redness or warm tones.
- **Compare different rows.** Different features should change different
  visual properties. Two rows changing the same thing means the SAE
  hasn't fully disentangled the concept.
- **If amplifying breaks the image** → that feature wasn't really
  meaningful for this prompt (or the strength was over-scaled).
- **The leftmost (negative-strength) column** often reveals what the
  feature was *suppressing* — sometimes more informative than the
  amplification.
"""
        )
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
