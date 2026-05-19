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

st.title("Localisation — head ablation sweeps")

st.markdown(
    "Picks a single cross-attention head in the UNet and scales its output "
    "by `factor` for a chosen step range. Compare the generated image against "
    "the unaltered baseline to localise which head carries which concept."
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
preset = model_preset_picker(default="(use config default)", options=("sd15", "sdxl_turbo"))

prompt = st.sidebar.text_input(
    "Prompt",
    value="a unicorn",
    help="The text prompt the model is conditioning on. Pick something where "
    "you have a hypothesis about which heads carry which concept.",
)
target_layer = st.sidebar.text_input(
    "Target layer",
    value="down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
    help="Underscore-sanitised UNet path. `down_blocks_1` = early in the UNet "
    "(rough composition); `mid_block` = mid (object identity); "
    "`up_blocks_X` = late (textures + fine detail). Suffix `_attn2_out` "
    "is the output of a cross-attention layer (image→text).",
)
target_head = st.sidebar.selectbox(
    "Target head index",
    list(range(8)),
    index=0,
    help="SD 1.x cross-attn layers have 8 parallel heads. Each is a "
    "specialist on some aspect of the image-text binding; iterate over "
    "them to find the one that carries your concept.",
)
factor = st.sidebar.slider(
    "Scale factor",
    -5.0,
    5.0,
    0.0,
    0.5,
    help="0.0 = zero-ablate the head; 1.0 = no change; >1.0 amplifies; "
    "negative = inverts the head's contribution.",
)
n_steps = st.sidebar.slider(
    "Inference steps",
    4,
    50,
    15,
    help="Diffusion denoising steps. More = sharper output, but slower. "
    "15 is plenty to see whether a head matters; bump to 30+ for paper-quality.",
)
seed = st.sidebar.number_input(
    "Seed",
    value=42,
    step=1,
    help="Same seed = same initial noise → comparisons isolate the head's "
    "effect (not different starting points).",
)

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
