"""Stitching playground — train an MLP mapper across two activation spaces."""

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
    # Inline mapper training prompts. One per line; lines containing `|`
    # become paired prompts (prompt_a | prompt_b); lines without become
    # same-prompt-both-models (typical cross-model transfer).
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
        "A label for your run — stored in the fingerprint and echoed in the "
        "results panel. **Does not** drive training; see the 'Training data' "
        "section below for the prompts the mapper actually learns from."
    ),
    key="stitch_goal",
)


# ── Training data source banner ──────────────────────────────────────────────


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


_pre_inline, _pre_skipped = _parse_inline_stitch(
    str(st.session_state.get("stitch_inline_pairs", ""))
)

st.subheader("Training data")
if _pre_inline:
    st.success(
        f"**Training mapper on {len(_pre_inline)} inline prompt(s)** — no "
        "network call for data. Edit the prompts in the sidebar's *Training "
        "data (inline prompts)* section.",
        icon="✅",
    )
else:
    st.info(
        "**Training mapper on HuggingFace dataset "
        "`nirmalendu01/spectacles-bias-prompts`** (default). Paste prompts "
        "into the sidebar textarea below to train on your own content instead.",
        icon="🌐",
    )


with st.expander(
    "**Tips for picking mapper training prompts**",
    expanded=False,
):
    st.markdown(
        """
- **What kind of prompts?** Generic, *diverse* content (people, objects,
  scenes, styles). The mapper learns a translation between the two layers'
  activation spaces — varied prompts cover more of that space.
- **Same prompt for both models?** Yes, by default. One prompt per line.
  Both models see the same prompt and the mapper learns to translate
  activation_a → activation_b for that prompt.
- **Different prompts per model (concept transfer)?** Use
  `prompt_a | prompt_b` syntax on a line. Useful when you want the mapper
  to encode a *concept difference* (e.g. `a photo of X | a painterly X`).
- **How many?** 10–30 prompts is plenty for a small `hidden_dim` mapper.
  Push to 100+ if you bump `hidden_dim` to 1024 or train for many steps.
- **`hidden_dim` choice.** Small (128–256) keeps the test honest — a tiny
  mapper that converges is real evidence the two layers carry comparable
  information. Big (512+) can paper over real incompatibility.
- **Failure modes.** *Stitched image is noise* → mapper didn't converge:
  more steps or more prompts. *Stitched looks identical to baseline* →
  `inject_steps` didn't fire (rare).
"""
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

st.sidebar.text_area("Inference prompts", key="stitch_prompts")
st.sidebar.markdown("**Quick-mode params** (drop hidden_dim/samples for speed)")
st.sidebar.slider("Mapper hidden_dim", 64, 1024, step=64, key="stitch_hidden_dim")
st.sidebar.slider("Train samples", 10, 500, key="stitch_max_samples")
st.sidebar.slider("Mapper training steps", 5, 1000, key="stitch_num_steps")
st.sidebar.slider("Inference steps", 4, 50, key="stitch_num_inference_steps")

with st.sidebar.expander(
    "Training data (inline prompts)",
    expanded=bool(st.session_state.get("stitch_inline_pairs", "").strip()),
):
    st.text_area(
        "Mapper training prompts — one per line",
        help=(
            "Each line is fed to BOTH models and the mapper learns the "
            "translation between their activations on that prompt. To pair "
            "different prompts across the two models (concept transfer), "
            "use `prompt_a | prompt_b` syntax on the line.\n\nLeave empty "
            "to use the workflow's default HuggingFace dataset."
        ),
        placeholder=("a photo of a person\na photo of a cat\na photo of a landscape"),
        height=200,
        key="stitch_inline_pairs",
    )
    _live_parsed, _live_skipped = _parse_inline_stitch(
        str(st.session_state.get("stitch_inline_pairs", ""))
    )
    if _live_parsed:
        st.caption(
            f"✅ **{len(_live_parsed)} valid prompt(s)** parsed"
            + (f" · {len(_live_skipped)} skipped" if _live_skipped else "")
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

# ── Build overrides ──────────────────────────────────────────────────────────
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

st.subheader("CLI equivalent")
st.code("t2i-stitch " + " ".join(overrides[:7]) + " …", language="bash")
if inline_pairs:
    st.caption(
        f"Mapper training on **{len(inline_pairs)} inline prompt(s)** — the HF dataset is skipped."
    )

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
