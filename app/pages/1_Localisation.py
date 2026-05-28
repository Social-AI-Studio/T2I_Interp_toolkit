"""Localisation playground. Scale a cross-attention head and see the effect."""

from __future__ import annotations

import re
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

# ── Defaults + recipe-payload intake ─────────────────────────────────────────
_LOC_DEFAULTS: dict[str, object] = {
    "loc_goal": "",
    "loc_prompt": "a unicorn",
    "loc_target_layer": "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
    "loc_target_head": 0,
    "loc_factor": 0.0,
    "loc_n_steps": 15,
    "loc_seed": 42,
    "loc_model_preset": "(use config default)",
}
for _k, _v in _LOC_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_payload = st.session_state.get("recipe_payload")
if _payload and _payload.get("workflow") == "Localisation":
    del st.session_state["recipe_payload"]
    if _payload.get("goal"):
        st.session_state["loc_goal"] = _payload["goal"]
    for _fk, _fv in _payload.get("fields", {}).items():
        _sk = f"loc_{_fk}"
        if _sk in _LOC_DEFAULTS:
            st.session_state[_sk] = _fv


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pair_baseline_modified(images: list) -> list[tuple[str, object | None, object | None]]:
    """Group output images into (label, baseline, modified) triples."""
    pairs: dict[str, dict[str, object]] = {}
    leftovers: list = []
    for img in images:
        name = img.name.lower()
        if name.startswith("baseline"):
            pairs.setdefault("0", {})["baseline"] = img
        elif m := re.match(r"(?:modified|head|layer|ablated)_(\d+)\.", name):
            pairs.setdefault(m.group(1), {})["modified"] = img
        else:
            leftovers.append(img)
    out: list[tuple[str, object | None, object | None]] = []
    for idx, both in sorted(pairs.items(), key=lambda kv: int(kv[0])):
        out.append((f"head {idx}", both.get("baseline"), both.get("modified")))
    for img in leftovers:
        out.append((img.name, None, img))
    return out


# ── Page header ──────────────────────────────────────────────────────────────

st.title("Localisation")
st.markdown("##### Scale one attention head and watch what changes in the image.")
st.caption(
    "Paper §3.1. Tells you where in the UNet a concept is bound. "
    "Sweep all heads with `factor=0.0` to find which ones carry the concept, "
    "or test a hypothesis by ablating one specific head."
)


# ── Step 1: What you want ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 1 · What you want")
    st.text_input(
        "Your goal (optional)",
        placeholder='e.g. "Test whether head 3 of mid_block carries the unicorn-ness"',
        help="A label for your run. Saved in the fingerprint and shown in the results panel.",
        key="loc_goal",
    )
    st.text_input(
        "Prompt to test",
        help=("Pick something where you have a hypothesis about which head carries which concept."),
        key="loc_prompt",
    )


# ── Step 2: Where to look ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 2 · Where to look")
    st.caption("Which layer and head to scale.")

    c_layer, c_head = st.columns([2, 1])
    with c_layer:
        st.text_input(
            "Target layer (UNet path)",
            help=(
                "`down_blocks_1` is early (rough composition). `mid_block` "
                "is mid (object identity). `up_blocks_X` is late (textures "
                "and fine detail). Suffix `_attn2_out` is the output of a "
                "cross-attention layer (image to text)."
            ),
            key="loc_target_layer",
        )
    with c_head:
        st.selectbox(
            "Head index (0-7)",
            list(range(8)),
            help=(
                "SD 1.x cross-attn layers have 8 parallel heads. Iterate "
                "over them to find the one that carries your concept."
            ),
            key="loc_target_head",
        )

    st.slider(
        "Scale factor",
        -5.0,
        5.0,
        step=0.5,
        help=(
            "0.0 zero-ablates the head. 1.0 is no change. Above 1.0 "
            "amplifies. Negative values invert the head's contribution."
        ),
        key="loc_factor",
    )


# ── Step 3: Run config (sidebar + small main controls) ───────────────────────

st.sidebar.header("Hardware")
device, dtype = device_dtype_picker(default_device="mps")
st.sidebar.header("Less-used knobs")
st.sidebar.slider(
    "Inference steps",
    4,
    50,
    help=(
        "Diffusion denoising steps. 15 is plenty to see whether a head "
        "matters. Bump to 30 or more for paper-quality."
    ),
    key="loc_n_steps",
)
st.sidebar.number_input(
    "Seed",
    step=1,
    help=("Same seed means same initial noise, so comparisons isolate the head's effect."),
    key="loc_seed",
)
preset = model_preset_picker(
    default=str(st.session_state.get("loc_model_preset", "(use config default)")),
    options=("sd15", "sdxl_turbo"),
    key="loc_model_preset",
)

# Pull session-state values for the override list
prompt = str(st.session_state["loc_prompt"])
target_layer = str(st.session_state["loc_target_layer"])
target_head = int(st.session_state["loc_target_head"])
factor = float(st.session_state["loc_factor"])
n_steps = int(st.session_state["loc_n_steps"])
seed = int(st.session_state["loc_seed"])
goal = str(st.session_state["loc_goal"])

# ── Step 4: Run ──────────────────────────────────────────────────────────────

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

with st.container(border=True):
    st.markdown("### Step 3 · Run")
    st.markdown(
        f"Will generate the prompt twice. Once unmodified (baseline), once "
        f"with head **{target_head}** of `{target_layer}` scaled by "
        f"**{factor:g}**."
    )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-localise " + " \\\n  ".join(overrides), language="bash")
    run_clicked = st.button(
        "▶ Run head ablation",
        type="primary",
        use_container_width=True,
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    with st.status("Running localisation...", expanded=True) as status:
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
            status.update(label="Run failed. See logs above.", state="error")

    st.divider()
    st.subheader("Results")
    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        triples = _pair_baseline_modified(images)
        for label, baseline, modified in triples:
            with st.container(border=True):
                st.markdown(f"##### {label}")
                c_b, c_m = st.columns(2)
                with c_b:
                    st.markdown("**Baseline** (head at factor = 1)")
                    if baseline is not None:
                        st.image(str(baseline), use_container_width=True)
                    else:
                        st.caption("(missing)")
                with c_m:
                    st.markdown(f"**Modified** (head at factor = {factor:g})")
                    if modified is not None:
                        st.image(str(modified), use_container_width=True)
                    else:
                        st.caption("(missing)")

        with st.expander("How to read these results", expanded=False):
            st.markdown(
                """
- **Baseline** is the model's default response to your prompt.
- **Modified** is the same prompt with one head scaled by your `factor`.
  Any difference is caused by that head.
- **They look identical**: the head wasn't carrying your concept in
  this context. Try another head or another layer.
- **A clear visual property changed** (object disappears, colour shifts,
  shape distorts, composition breaks): you've localised it. That head
  controls that property.
- **The image becomes noise**: the head was critical for the whole
  forward pass. Try a less aggressive factor (0.5 instead of 0.0).
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
