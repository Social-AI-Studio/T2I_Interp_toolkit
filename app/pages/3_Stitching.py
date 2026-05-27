"""Stitching playground — train an MLP mapper across two activation spaces."""

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

st.set_page_config(page_title="Stitching • T2I-Interp", layout="wide")

# ── Defaults + recipe-payload intake ─────────────────────────────────────────
_STITCH_DEFAULTS: dict[str, object] = {
    "stitch_goal": "",
    "stitch_prompts": "a photo of a person",
    "stitch_hidden_dim": 256,
    "stitch_max_samples": 50,
    "stitch_num_steps": 50,
    "stitch_num_inference_steps": 15,
    "stitch_model_preset": "sd15",
}
for _k, _v in _STITCH_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_payload = st.session_state.get("recipe_payload")
if _payload and _payload.get("workflow") == "Stitching":
    del st.session_state["recipe_payload"]
    if _payload.get("goal"):
        st.session_state["stitch_goal"] = _payload["goal"]
    for _fk, _fv in _payload.get("fields", {}).items():
        _sk = f"stitch_{_fk}"
        if _sk in _STITCH_DEFAULTS:
            st.session_state[_sk] = _fv

# ── Page body ────────────────────────────────────────────────────────────────

st.title("Stitching — cross-layer activation mapper")

st.markdown(
    "Trains a small MLP that maps activations from `layer_a` (typically a "
    "text-encoder output) into `layer_b`'s space (typically a UNet block). "
    "At generation time, captured activations from `layer_a` are passed "
    "through the mapper and injected at `layer_b`. Fig 4 in the paper."
)

with st.expander("**Common goals this page serves**", expanded=False):
    st.markdown(
        """
- **Transfer a behaviour between two models** (e.g. base SD 1.5 ↔
  fine-tuned variant) without retraining. The paper's §4 case study.
- **Check whether two layers encode comparable information.** If a small
  mapper can stitch them, the two activations carry the same kind of
  content; if not, they don't.
- **Move a steering direction across models.** Train a mapper, then
  apply a steering vector learned in model A inside model B's activation
  space.

See the **Recipes** page for concrete walkthroughs — clicking *Open*
there will pre-fill the form below.
"""
    )

st.text_input(
    "What are you trying to achieve? (optional)",
    placeholder='e.g. "Can SD1.5 text-encoder output stand in for unet.conv_out?"',
    help=(
        "Stored in the run fingerprint and shown back in the results panel. "
        "Pre-filled automatically if you arrived from a Recipe."
    ),
    key="stitch_goal",
)

st.info(
    """
**How this affects the picture.** Two parts of the model (or two different
models) live in *different activation spaces* — their internal tensors
have different shapes, semantics, and meanings. The mapper learns a
translation between them: "given an activation at layer A that means
something, what would the equivalent activation at layer B look like?"

At inference, the model's normal forward pass is *re-routed* through the
mapper at layer B. The generated image is what the model produces when
its information flow has been rewired this way — a way to test whether
two layers / models encode comparable information, and to transfer
behavior between them without retraining.
""",
    icon="ℹ️",
)

# ── Sidebar config ───────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(
    default=str(st.session_state.get("stitch_model_preset", "sd15")),
    key="stitch_model_preset",
)

st.sidebar.text_area("Prompts", key="stitch_prompts")
st.sidebar.markdown("**Quick-mode params** (drop hidden_dim/samples for speed)")
st.sidebar.slider("Mapper hidden_dim", 64, 1024, step=64, key="stitch_hidden_dim")
st.sidebar.slider("Train samples", 10, 500, key="stitch_max_samples")
st.sidebar.slider("Mapper training steps", 5, 1000, key="stitch_num_steps")
st.sidebar.slider("Inference steps", 4, 50, key="stitch_num_inference_steps")

prompts_raw = str(st.session_state["stitch_prompts"])
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
hidden_dim = int(st.session_state["stitch_hidden_dim"])
max_samples = int(st.session_state["stitch_max_samples"])
num_steps = int(st.session_state["stitch_num_steps"])
num_inference_steps = int(st.session_state["stitch_num_inference_steps"])
goal = str(st.session_state["stitch_goal"])

# ── Build overrides ──────────────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_stitch_")
overrides = [
    f"device={device}",
    f"dtype={dtype}",
    f"hidden_dim={hidden_dim}",
    f"max_samples={max_samples}",
    f"num_steps={num_steps}",
    f"num_inference_steps={num_inference_steps}",
    f"prompts=[{','.join(prompts)}]",
    f"save_dir={out_dir}/cache",
    f"output_dir={out_dir}",
    f"hydra.run.dir={out_dir}/.hydra",
    "wandb.project=null",
]
if preset:
    overrides.append(f"model={preset}")

st.subheader("CLI equivalent")
st.code("t2i-stitch " + " ".join(overrides[:7]) + " …", language="bash")

# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status("Training mapper + generating…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-stitch", overrides, output_dir=out_dir):
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
- **`mapper.pt`** is the trained mapper checkpoint — reusable across runs.
- **`stitched_*.png`** is the prompt generated with the mapper rewiring
  activations from `layer_a` into `layer_b`.
- **If you get a coherent image related to the prompt** → the mapper
  learned a useful translation between the two activation spaces. The
  two layers do encode comparable information.
- **If you get noise or unrelated content** → mapper didn't converge.
  Try more training steps, a bigger `hidden_dim`, or more samples.
- **If the stitched image looks identical to the baseline** → the
  mapper is a no-op (rare); check that `inject_steps` actually fires
  on the early step you chose.
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
