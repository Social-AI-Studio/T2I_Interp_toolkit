"""SAE playground. Discover sparse features and modulate them at generation time."""

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


# ── Page header ──────────────────────────────────────────────────────────────

st.title("Sparse Autoencoders")
st.markdown("##### Decompose activations into interpretable features. Push them up or down.")
st.caption(
    "Paper §3.4. Loads pretrained SAEs trained on SDXL-Turbo UNet "
    "activations. Captures the latents for your prompt, picks the top "
    "active features, and re-generates with each scaled by a range of "
    "strengths."
)

# Surface missing checkpoints early.
ckpt_dir = Path("./sdxl-unbox/checkpoints")
if not ckpt_dir.exists():
    st.error(
        "Missing SAE checkpoints at `./sdxl-unbox/checkpoints/`. Run "
        "`t2i-migrate-sae --checkpoint-dir ./sdxl-unbox/checkpoints` after "
        "downloading from `anonymous-author-129/sdxl-unbox-saes` on "
        "HuggingFace. Or follow `notebooks/sae.ipynb` for the full setup."
    )
    st.stop()


# ── Step 1: What you want ────────────────────────────────────────────────────
# Hardcoded scenarios that pick prompt + strengths + n_features for the
# three common use cases.

_SAE_PRESETS: dict[str, dict[str, object]] = {
    "Discover what features fire for my prompt": {
        "label": "Discover features for a prompt",
        "prompt": "a red apple on a wooden table",
        "n_top_features": 10,
        "n_features_to_plot": 4,
        "strength_lo": -5.0,
        "strength_hi": 5.0,
        "hint": (
            "Captures the top 10 most-active features for the prompt and "
            "modulates the top 4 across a -5 to +5 strength sweep. The "
            "easiest entry point. Look across each row to see what changes "
            "consistently."
        ),
    },
    "Hunt for a feature that controls a property": {
        "label": "Hunt for a feature controlling a property",
        "prompt": "a glossy red apple",
        "n_top_features": 15,
        "n_features_to_plot": 6,
        "strength_lo": -8.0,
        "strength_hi": 8.0,
        "hint": (
            "Wider strength sweep (-8 to +8) and more features (6 rows). "
            "Use a prompt where your target property is prominent (e.g. "
            "'glossy' for shininess). The wider sweep makes per-feature "
            "behaviour easier to read."
        ),
    },
    "Amplify a known feature (single, fixed)": {
        "label": "Amplify a known feature index",
        "prompt": "a red apple on a wooden table",
        "n_top_features": 5,
        "n_features_to_plot": 1,
        "strength_lo": 0.0,
        "strength_hi": 5.0,
        "hint": (
            "Once you've identified a feature you like, this preset "
            "narrows to a single row (one feature) at a positive-only "
            "strength range. Useful for confirming the feature behaves "
            "the way you think it does."
        ),
    },
}

with st.container(border=True):
    st.markdown("### Step 1 · What you want to do")
    st.caption("Pick a scenario. Prompt, feature counts, and strength range get pre-filled.")

    chosen = st.radio(
        "Scenario",
        list(_SAE_PRESETS),
        index=0,
        key="sae_scenario_label",
        label_visibility="collapsed",
    )
    preset_meta = _SAE_PRESETS[chosen]
    st.markdown(f"**{preset_meta['label']}**")
    st.caption(str(preset_meta["hint"]))

    if st.button(
        "Apply scenario",
        help="Drops the scenario's values into Steps 2 and 3 below.",
        use_container_width=True,
    ):
        st.session_state["sae_prompt"] = preset_meta["prompt"]
        st.session_state["sae_n_top_features"] = preset_meta["n_top_features"]
        st.session_state["sae_n_features_to_plot"] = preset_meta["n_features_to_plot"]
        st.session_state["sae_strength_lo"] = preset_meta["strength_lo"]
        st.session_state["sae_strength_hi"] = preset_meta["strength_hi"]
        st.session_state["sae_goal"] = preset_meta["label"]
        st.rerun()

    st.text_input(
        "Prompt to probe",
        help=(
            "The SAE captures features active for this prompt, then "
            "re-generates with each one scaled."
        ),
        key="sae_prompt",
    )


# ── Step 2: How to modulate ──────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 2 · How to modulate")
    st.caption("How many features to probe and what strengths to test.")

    c_n, c_top = st.columns(2)
    with c_n:
        st.slider(
            "Top features to modulate",
            1,
            6,
            help="One row per feature in the output grid.",
            key="sae_n_features_to_plot",
        )
    with c_top:
        st.slider(
            "Capture top-K features",
            2,
            20,
            help="How many features to look at before picking the modulation targets.",
            key="sae_n_top_features",
        )

    c_lo, c_hi = st.columns(2)
    with c_lo:
        st.slider(
            "Min strength (suppress)",
            -20.0,
            0.0,
            step=0.5,
            help="Negative strengths suppress the feature.",
            key="sae_strength_lo",
        )
    with c_hi:
        st.slider(
            "Max strength (amplify)",
            0.0,
            20.0,
            step=0.5,
            help="Positive strengths amplify the feature.",
            key="sae_strength_hi",
        )


# ── Step 3: Run config (mostly sidebar) ──────────────────────────────────────

st.sidebar.header("Hardware")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(
    default=str(st.session_state.get("sae_model_preset", "sdxl_turbo")),
    key="sae_model_preset",
)
st.sidebar.text_input(
    "Run label (optional)",
    help=(
        "Free-text label saved in the fingerprint and shown in the "
        "results panel. Set automatically when you Apply a scenario. "
        "Does not drive the run."
    ),
    key="sae_goal",
)

prompt = str(st.session_state["sae_prompt"])
strength_lo = float(st.session_state["sae_strength_lo"])
strength_hi = float(st.session_state["sae_strength_hi"])
strengths = sorted({strength_lo, 0.0, strength_hi})  # always include baseline
n_features_to_plot = int(st.session_state["sae_n_features_to_plot"])
n_top_features = int(st.session_state["sae_n_top_features"])
goal = str(st.session_state["sae_goal"])


# ── Step 4: Run ──────────────────────────────────────────────────────────────

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

with st.container(border=True):
    st.markdown("### Step 3 · Run")
    st.markdown(
        f"Will capture activations for **{prompt!r}**, pick the top "
        f"**{n_features_to_plot}** features, and re-generate at "
        f"strengths {strengths}."
    )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-sae " + " \\\n  ".join(overrides), language="bash")
    run_clicked = st.button(
        "▶ Capture and modulate",
        type="primary",
        use_container_width=True,
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    with st.status("Capturing activations and modulating features...", expanded=True) as status:
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
            status.update(label="Run failed. See logs above.", state="error")

    st.divider()
    st.subheader("Results")
    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        st.markdown(f"**Feature modulation grid** ({len(images)} image(s))")
        for img in images:
            st.image(str(img), caption=img.name, use_container_width=True)

        with st.expander("How to read these results", expanded=False):
            st.markdown(
                """
- **Each row is one feature** (e.g. feature `#1338`). The top-K features
  were the ones most active for your prompt.
- **Each column is one strength value.** Left columns suppress the
  feature. Right columns amplify it.
- **Across a row, look for what changes consistently.** If amplifying
  feature 1338 progressively adds shininess to your subject across the
  row, then 1338 encodes a shininess concept.
- **Compare different rows.** Different features should change
  different visual properties. Two rows changing the same thing means
  the SAE hasn't fully disentangled the concept.
- **If amplifying breaks the image**: that feature wasn't really
  meaningful for this prompt, or the strength was over-scaled.
- **The leftmost (negative-strength) column** often reveals what the
  feature was suppressing. Sometimes more informative than amplification.
"""
            )
    else:
        st.warning("No images produced. Check logs above.")

    fp = load_fingerprint(out_dir)
    if fp:
        with st.container(border=True):
            st.markdown("##### Run fingerprint")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Hash", fp["fingerprint_hash"])
                st.metric("Workflow", fp["workflow"])
            with c2:
                st.json(fp, expanded=False)
