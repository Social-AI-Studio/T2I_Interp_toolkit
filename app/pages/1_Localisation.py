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

# ── Page body ────────────────────────────────────────────────────────────────

st.title("Localisation — head ablation sweeps")

st.markdown(
    "Picks a single cross-attention head in the UNet and scales its output "
    "by `factor` for a chosen step range. Compare the generated image against "
    "the unaltered baseline to localise which head carries which concept."
)

with st.expander("**Common goals this page serves**", expanded=False):
    st.markdown(
        """
- **Find where a concept lives in the UNet.** Sweep all heads with `factor=0.0`
  and watch which ablations break the concept.
- **Test a hypothesis** that a *specific* head carries a specific behaviour
  (e.g. head 3 of `mid_block` binds colour words).
- **Compare early vs late UNet layers** to map their responsibilities.

See the **Recipes** page (sidebar) for one-click presets — clicking *Open*
there will pre-fill the form below.
"""
    )

st.text_input(
    "What are you trying to achieve? (optional)",
    placeholder='e.g. "Test whether head 3 of mid_block carries the unicorn-ness"',
    help=(
        "Stored in the run fingerprint and shown back in the result panel. "
        "Pre-filled automatically if you arrived from a Recipe."
    ),
    key="loc_goal",
)

st.info(
    """
**How this affects the picture.** A *head* is a small slice of an attention
layer that specializes in one type of relationship between the prompt and
the image (e.g. binding color words to color regions, or shape words to
object outlines). Scaling a head by `factor=0` blanks out its contribution
entirely; `factor=2.0` doubles it; negative values invert it. The baseline
image is what the model produces normally; the *modified* image shows what
happens when one specific head no longer functions. Comparing the two
tells you what that head was doing.
""",
    icon="ℹ️",
)

# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(
    default=str(st.session_state.get("loc_model_preset", "(use config default)")),
    options=("sd15", "sdxl_turbo"),
    key="loc_model_preset",
)

st.sidebar.text_input(
    "Prompt",
    help="The text prompt the model is conditioning on. Pick something where "
    "you have a hypothesis about which heads carry which concept.",
    key="loc_prompt",
)
st.sidebar.text_input(
    "Target layer",
    help="Underscore-sanitised UNet path. `down_blocks_1` = early in the UNet "
    "(rough composition); `mid_block` = mid (object identity); "
    "`up_blocks_X` = late (textures + fine detail). Suffix `_attn2_out` "
    "is the output of a cross-attention layer (image→text).",
    key="loc_target_layer",
)
st.sidebar.selectbox(
    "Target head index",
    list(range(8)),
    help="SD 1.x cross-attn layers have 8 parallel heads. Each is a "
    "specialist on some aspect of the image-text binding; iterate over "
    "them to find the one that carries your concept.",
    key="loc_target_head",
)
st.sidebar.slider(
    "Scale factor",
    -5.0,
    5.0,
    step=0.5,
    help="0.0 = zero-ablate the head; 1.0 = no change; >1.0 amplifies; "
    "negative = inverts the head's contribution.",
    key="loc_factor",
)
st.sidebar.slider(
    "Inference steps",
    4,
    50,
    help="Diffusion denoising steps. More = sharper output, but slower. "
    "15 is plenty to see whether a head matters; bump to 30+ for paper-quality.",
    key="loc_n_steps",
)
st.sidebar.number_input(
    "Seed",
    step=1,
    help="Same seed = same initial noise → comparisons isolate the head's "
    "effect (not different starting points).",
    key="loc_seed",
)

# Pull session-state values for the override list
prompt = str(st.session_state["loc_prompt"])
target_layer = str(st.session_state["loc_target_layer"])
target_head = int(st.session_state["loc_target_head"])
factor = float(st.session_state["loc_factor"])
n_steps = int(st.session_state["loc_n_steps"])
seed = int(st.session_state["loc_seed"])
goal = str(st.session_state["loc_goal"])

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

    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        st.subheader(f"Output images ({len(images)})")
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(str(img), caption=img.name, use_container_width=True)

        st.markdown("##### How to read these results")
        st.markdown(
            """
- **`baseline.png`** is the unmodified output — the model's default response
  to your prompt at this seed.
- **The other image(s)** are the same prompt with one head scaled by your
  `factor`. Differences between them and the baseline are *caused by that
  head*, holding everything else constant.
- **If they look identical** → the head wasn't carrying your concept in
  this context. Try another head or another layer.
- **If a clear visual property changed** (object disappears, color shifts,
  shape distorts, composition breaks) → that property was being controlled
  by that head. You've localised it.
- **If the image becomes noise** → the head was critical for the whole
  forward pass; you need a more surgical scale (try `factor=0.5` instead
  of `0.0`).
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
