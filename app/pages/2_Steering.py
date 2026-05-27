"""Steering playground — train a concept direction and inject it during generation."""

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

st.set_page_config(page_title="Steering • T2I-Interp", layout="wide")

# ── Defaults + recipe-payload intake ─────────────────────────────────────────
# Every recipe-controllable widget on this page is keyed into session_state
# so the Recipes page can pre-fill it via st.session_state["recipe_payload"].
# Order matters: set defaults first (idempotent), then overwrite with payload
# *before* any widgets render.

_STEER_DEFAULTS: dict[str, object] = {
    "steer_goal": "",
    "steer_method": "loreft",
    "steer_model_preset": "sdxl_turbo",
    "steer_prompts": "A photo of Jack Sparrow\nA photo of Simba",
    "steer_alpha": 10.0,
    "steer_max_samples": 100,
    "steer_train_steps": 50,
    # Inline training pairs (positive | negative, one per line). When empty,
    # the page falls back to the workflow's default HuggingFace dataset
    # (currently the spectacles dataset). When set, run_steer.py builds an
    # in-memory dataset from these pairs and trains on them directly.
    "steer_inline_pairs": "",
}
for _k, _v in _STEER_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_payload = st.session_state.get("recipe_payload")
if _payload and _payload.get("workflow") == "Steering":
    del st.session_state["recipe_payload"]
    if _payload.get("goal"):
        st.session_state["steer_goal"] = _payload["goal"]
    for _fk, _fv in _payload.get("fields", {}).items():
        _sk = f"steer_{_fk}"
        if _sk in _STEER_DEFAULTS:
            st.session_state[_sk] = _fv

# ── Page body ────────────────────────────────────────────────────────────────

st.title("Steering — concept direction injection")

st.markdown(
    "Trains a steering vector (CAA), classifier (K-Steer), or low-rank "
    "adapter (LoReFT) from paired positive/negative prompts in a dataset, "
    "then injects it during generation. The headline figure of the paper "
    "uses **LoReFT + SDXL-Turbo** to add spectacles to character prompts."
)

with st.expander("**Common goals this page serves**", expanded=False):
    st.markdown(
        """
- **Add an attribute** to existing prompts (paper Fig 2 — spectacles).
- **Shift outputs toward a specific demographic** (paper Fig 3 — "photo of
  a man" → Black).
- **Suppress / erase an unwanted concept** (use negative alpha — subtract
  the direction rather than add it).
- **Apply a style** (painterly, impressionist, photorealistic) without a LoRA.

See the **Recipes** page (sidebar) for one-click presets — clicking *Open*
there will pre-fill the form below.
"""
    )

st.text_input(
    "What are you trying to achieve? (optional)",
    placeholder='e.g. "Add spectacles to portraits" or "Reduce gender bias for \'doctor\'"',
    help=(
        "Stored in the run fingerprint and shown back in the results panel. "
        "Pre-filled automatically if you arrived from a Recipe."
    ),
    key="steer_goal",
)

st.info(
    """
**How this affects the picture.** From a dataset of paired prompts (the
'positive' has the target concept, the 'negative' doesn't), the toolkit
learns a *direction in activation space* that, when added to a layer's
output, biases generation toward the positive concept. At inference time,
this direction is multiplied by `alpha` and added at the chosen layer.
Higher `alpha` = stronger push toward the concept. Same prompt + same
seed will now produce an image leaning toward the trained attribute,
without retraining the model itself.
""",
    icon="ℹ️",
)

# ── Quick presets (in-page, not via Recipes) ─────────────────────────────────
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button(
        "Reproduce Figure 2", help="LoReFT + SDXL-Turbo + spectacles prompts, paper-style"
    ):
        st.session_state["steer_method"] = "loreft"
        st.session_state["steer_model_preset"] = "sdxl_turbo"
        st.session_state["steer_prompts"] = "A photo of Jack Sparrow\nA photo of Simba"
        st.session_state["steer_alpha"] = 10.0
        st.session_state["steer_max_samples"] = 200
        st.session_state["steer_train_steps"] = 50
        st.rerun()
with c2:
    if st.button("Quick smoke run", help="Tiny scale just to confirm the wiring works"):
        st.session_state["steer_method"] = "loreft"
        st.session_state["steer_model_preset"] = "sdxl_turbo"
        st.session_state["steer_prompts"] = "A photo of a cat"
        st.session_state["steer_alpha"] = 5.0
        st.session_state["steer_max_samples"] = 10
        st.session_state["steer_train_steps"] = 2
        st.rerun()

# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
device, dtype = device_dtype_picker(default_device="mps")
preset = model_preset_picker(
    default=str(st.session_state.get("steer_model_preset", "sdxl_turbo")),
    key="steer_model_preset",
)

st.sidebar.selectbox(
    "Steering method",
    ["loreft", "caa", "ksteer"],
    key="steer_method",
)
st.sidebar.text_area(
    "Prompts (one per line)",
    help="Generated once as baseline, once steered.",
    key="steer_prompts",
)
st.sidebar.slider(
    "Alpha (steering strength)",
    -30.0,
    30.0,
    step=0.5,
    help="0.0 = no steering. Higher = stronger. Negative = subtract the "
    "direction (suppression). SDXL-Turbo + LoReFT works well around 10-20.",
    key="steer_alpha",
)
st.sidebar.slider(
    "Training samples",
    10,
    1000,
    key="steer_max_samples",
)
st.sidebar.slider(
    "Training steps",
    2,
    500,
    key="steer_train_steps",
)

# Inline training pairs — open by default if pre-filled by a recipe, else collapsed.
with st.sidebar.expander(
    "Training data (inline pairs)",
    expanded=bool(st.session_state.get("steer_inline_pairs", "").strip()),
):
    st.text_area(
        "Prompt pairs — one per line, `positive | negative`",
        help=(
            "When set, trains on these inline pairs instead of the workflow's "
            "default HuggingFace dataset (currently the spectacles dataset). "
            "For CAA each `positive` becomes a label=1 caption and each "
            "`negative` becomes a label=0 caption; for LoReFT each pair becomes "
            "one (base=negative, teacher=positive) row.\n\nLeave empty to use "
            "the configured HF dataset."
        ),
        placeholder=(
            "photo of a Black man | photo of a man\nphoto of a Black woman | photo of a woman"
        ),
        height=160,
        key="steer_inline_pairs",
    )

steer_type = str(st.session_state["steer_method"])
prompts_raw = str(st.session_state["steer_prompts"])
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
alpha = float(st.session_state["steer_alpha"])
max_samples = int(st.session_state["steer_max_samples"])
train_steps = int(st.session_state["steer_train_steps"])
goal = str(st.session_state["steer_goal"])
inline_pairs_text = str(st.session_state.get("steer_inline_pairs", ""))


def _parse_inline_pairs(raw: str) -> list[dict[str, str]]:
    """Parse 'pos | neg' lines into [{'pos': ..., 'neg': ...}, ...]."""
    out: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        left, right = line.split("|", 1)
        pos, neg = left.strip(), right.strip()
        if pos and neg:
            out.append({"pos": pos, "neg": neg})
    return out


inline_pairs = _parse_inline_pairs(inline_pairs_text)

# ── Build overrides ──────────────────────────────────────────────────────────
out_dir = tempfile.mkdtemp(prefix="streamlit_steer_")

# Inline pairs go via a JSON sidecar file — Hydra's list-of-dict override
# syntax is awkward for prompts containing spaces/commas.
inline_pairs_file: str | None = None
if inline_pairs:
    inline_pairs_file = os.path.join(out_dir, "inline_pairs.json")
    with open(inline_pairs_file, "w") as f:
        json.dump(inline_pairs, f)

overrides = [
    f"--config-name=steer/{steer_type}",
    f"device={device}",
    f"dtype={dtype}",
    f"alpha={alpha}",
    f"max_samples={max_samples}",
    f"train_steps={train_steps}",
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
st.code("t2i-steer " + " ".join(overrides[:8]) + " …", language="bash")
if inline_pairs:
    st.caption(f"Training on **{len(inline_pairs)} inline pair(s)** — the HF dataset is skipped.")

# ── Run ───────────────────────────────────────────────────────────────────────
if st.button("Run", type="primary"):
    with st.status(f"Training {steer_type.upper()} + generating…", expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = time.time()
        result = None
        for event in run_workflow("t2i-steer", overrides, output_dir=out_dir):
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
- **`baseline_*`** images are generated **without** the steering vector —
  the same prompt and seed the model would produce normally.
- **`steered_*`** images apply the trained direction at `alpha`. They
  should preserve the prompt's content while leaning toward the trained
  concept.
- **If steered ≈ baseline** → alpha is too low, or you trained on the
  wrong layer. Push alpha up to 15-20.
- **If steered looks like garbage / noise** → alpha is too high or the
  adapter overfit a tiny dataset. Lower alpha or train on more samples.
- **If steered shows the target concept but the original prompt is gone**
  (e.g. you wanted "Jack Sparrow + spectacles" and got just "spectacles")
  → alpha overpowered the prompt; reduce it.
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
            st.metric("Alpha", str(fp["intervention"].get("alpha", "—")))
        with c2:
            st.json(fp, expanded=False)
