"""Stitching playground. Train an MLP mapper across two activation spaces."""

from __future__ import annotations

import json
import os
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
    "stitch_inline_pairs": "",
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_inline_stitch(raw: str) -> tuple[list[dict[str, str] | str], list[int]]:
    """Parse 'prompt' or 'a | b' lines. Returns (entries, skipped_line_numbers)."""
    out: list[dict[str, str] | str] = []
    skipped: list[int] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            a, b = (s.strip() for s in line.split("|", 1))
            if a and b:
                out.append({"a": a, "b": b})
            else:
                skipped.append(idx)
        else:
            out.append(line)
    return out, skipped


# ── Page header ──────────────────────────────────────────────────────────────

st.title("Stitching")
st.markdown("##### Train an MLP that translates activations between two layers or two models.")
st.caption(
    "Paper §3.3. Tests whether two activation spaces carry comparable "
    "information. If a small mapper can stitch them, they do. If not, "
    "they don't. Used in the paper's §4 cross-model case study."
)


# ── Step 1: What you want ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 1 · What you want")
    st.text_input(
        "Your goal (optional)",
        placeholder='e.g. "Can SD1.5 text-encoder output stand in for unet.conv_out?"',
        help="A label for your run. Saved in the fingerprint and shown in the results panel.",
        key="stitch_goal",
    )


# ── Step 2: Training data ────────────────────────────────────────────────────

_pre_inline, _pre_skipped = _parse_inline_stitch(
    str(st.session_state.get("stitch_inline_pairs", ""))
)

with st.container(border=True):
    st.markdown("### Step 2 · Training data")
    if _pre_inline:
        st.success(
            f"**Training mapper on {len(_pre_inline)} inline prompts.** "
            "No network call for data. Edit the textarea below to change them.",
            icon="✅",
        )
    else:
        st.info(
            "**Training mapper on HuggingFace dataset "
            "`nirmalendu01/spectacles-bias-prompts`** (default). Paste "
            "prompts below to train on your own content instead.",
            icon="🌐",
        )

    st.text_area(
        "Mapper training prompts, one per line",
        help=(
            "Each line is fed to BOTH models. The mapper learns the "
            "translation between their activations on that prompt. To pair "
            "different prompts (concept transfer), use `prompt_a | prompt_b` "
            "syntax. Leave empty to use the HF dataset."
        ),
        placeholder=("a photo of a person\na photo of a cat\na photo of a landscape"),
        height=180,
        key="stitch_inline_pairs",
    )
    _live_parsed, _live_skipped = _parse_inline_stitch(
        str(st.session_state.get("stitch_inline_pairs", ""))
    )
    if _live_parsed:
        st.caption(
            f"**{len(_live_parsed)} valid prompt(s)** parsed"
            + (f" · {len(_live_skipped)} skipped" if _live_skipped else "")
        )

    with st.expander("Tips for picking training prompts", expanded=False):
        st.markdown(
            """
- **What kind of prompts?** Generic, diverse content (people, objects,
  scenes, styles). The mapper learns a translation between the two
  layers' activation spaces. Varied prompts cover more of that space.
- **Same prompt for both models?** Yes, by default. One prompt per line.
- **Different prompts per model (concept transfer)?** Use
  `prompt_a | prompt_b` syntax on a line.
- **How many?** 10 to 30 prompts is plenty for a small `hidden_dim` mapper.
- **`hidden_dim` choice.** Small (128 to 256) keeps the test honest.
  Big (512 and up) can paper over real incompatibility.
- **Failure modes.** Noise: mapper didn't converge. Stitched looks
  identical to baseline: `inject_steps` didn't fire (rare).
"""
        )


# ── Step 3: Run config ───────────────────────────────────────────────────────

st.sidebar.header("Hardware")
device, dtype = device_dtype_picker(default_device="mps")
st.sidebar.header("Less-used knobs")
st.sidebar.slider("Mapper hidden_dim", 64, 1024, step=64, key="stitch_hidden_dim")
st.sidebar.slider("Train samples", 10, 500, key="stitch_max_samples")
st.sidebar.slider("Mapper training steps", 5, 1000, key="stitch_num_steps")
st.sidebar.slider("Inference steps", 4, 50, key="stitch_num_inference_steps")
preset = model_preset_picker(
    default=str(st.session_state.get("stitch_model_preset", "sd15")),
    key="stitch_model_preset",
)

with st.container(border=True):
    st.markdown("### Step 3 · Run config")
    st.text_area(
        "Inference prompts (one per line)",
        key="stitch_prompts",
        height=110,
        help="What gets generated with the mapper rewiring activations.",
    )

prompts_raw = str(st.session_state["stitch_prompts"])
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
hidden_dim = int(st.session_state["stitch_hidden_dim"])
max_samples = int(st.session_state["stitch_max_samples"])
num_steps = int(st.session_state["stitch_num_steps"])
num_inference_steps = int(st.session_state["stitch_num_inference_steps"])
goal = str(st.session_state["stitch_goal"])

# Re-parse from session_state in case the textarea was edited after the top-of-page parse.
inline_pairs, _inline_skipped = _parse_inline_stitch(
    str(st.session_state.get("stitch_inline_pairs", ""))
)
if str(st.session_state.get("stitch_inline_pairs", "")).strip() and not inline_pairs:
    st.sidebar.warning(
        "Inline prompts textarea is non-empty but no valid lines were found. "
        "The HF dataset will be used instead.",
        icon="⚠️",
    )
elif _inline_skipped:
    st.sidebar.warning(
        f"Skipped {len(_inline_skipped)} malformed line(s) "
        f"(line(s) {', '.join(map(str, _inline_skipped))}).",
        icon="⚠️",
    )


# ── Step 4: Run ──────────────────────────────────────────────────────────────

out_dir = tempfile.mkdtemp(prefix="streamlit_stitch_")

inline_pairs_file: str | None = None
if inline_pairs:
    inline_pairs_file = os.path.join(out_dir, "inline_pairs.json")
    with open(inline_pairs_file, "w") as f:
        json.dump(inline_pairs, f)

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
if inline_pairs_file:
    overrides.append(f"inline_pairs_file={inline_pairs_file}")

with st.container(border=True):
    st.markdown("### Step 4 · Run")
    st.markdown(
        f"Will train an MLP mapper (`hidden_dim={hidden_dim}`) on "
        f"{'inline prompts' if inline_pairs else 'the HF dataset'} for "
        f"`{num_steps}` steps, then generate **{len(prompts)} prompt(s)** "
        "with the mapper rewiring activations."
    )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-stitch " + " \\\n  ".join(overrides), language="bash")
    run_clicked = st.button(
        "▶ Train mapper and generate",
        type="primary",
        use_container_width=True,
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    with st.status("Training mapper and generating...", expanded=True) as status:
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
            status.update(label="Run failed. See logs above.", state="error")

    st.divider()
    st.subheader("Results")
    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        st.markdown(f"**{len(images)} image(s)** produced.")
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(str(img), caption=img.name, use_container_width=True)

        with st.expander("How to read these results", expanded=False):
            st.markdown(
                """
- **`mapper.pt`** is the trained mapper checkpoint. Reusable across runs.
- **`stitched_*.png`** is the prompt generated with the mapper rewiring
  activations from `layer_a` into `layer_b`.
- **Coherent image related to the prompt**: the mapper learned a useful
  translation. The two layers do encode comparable information.
- **Noise or unrelated content**: mapper didn't converge. Try more
  training steps, a bigger `hidden_dim`, or more samples.
- **Stitched image looks identical to baseline**: the mapper is a no-op
  (rare). Check that `inject_steps` actually fires on the early step
  you chose.
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
