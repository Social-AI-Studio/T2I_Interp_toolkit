"""Steering playground — train a concept direction and inject it during generation."""

from __future__ import annotations

import json
import os
import tempfile
import time

import streamlit as st

from app.lib import (
    INTENT_TEMPLATES,
    collect_images,
    device_dtype_picker,
    generate_inline_pairs,
    load_fingerprint,
    model_preset_picker,
    run_workflow,
)
from app.lib import is_available as llm_is_available

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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_inline_pairs(raw: str) -> tuple[list[dict[str, str]], list[int]]:
    """Parse 'pos | neg' lines. Returns (pairs, skipped_line_numbers)."""
    out: list[dict[str, str]] = []
    skipped: list[int] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            skipped.append(idx)
            continue
        left, right = line.split("|", 1)
        pos, neg = left.strip(), right.strip()
        if pos and neg:
            out.append({"pos": pos, "neg": neg})
        else:
            skipped.append(idx)
    return out, skipped


# Per-method default HF dataset (matches the YAML configs).
_METHOD_DEFAULT_DATASETS = {
    "caa": "nirmalendu01/spectacles-bias-prompts-headshot-captioned",
    "loreft": "nirmalendu01/spectacles-bias-prompts-headshot",
    "ksteer": "nirmalendu01/spectacles-bias-prompts-headshot-captioned",
}


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
        "A label for your run — stored in the fingerprint and echoed in the "
        "results panel. **Does not** drive training; see the 'Training data' "
        "section below for the prompts the model actually learns from."
    ),
    key="steer_goal",
)


# ── Training data source banner ──────────────────────────────────────────────
# Parse pairs early so we can show the active source before any other UI.
inline_pairs_text = str(st.session_state.get("steer_inline_pairs", ""))
inline_pairs, _inline_skipped = _parse_inline_pairs(inline_pairs_text)
_current_method = str(st.session_state.get("steer_method", "loreft"))

st.subheader("Training data")
if inline_pairs:
    st.success(
        f"**Training on {len(inline_pairs)} inline prompt pair(s)** — no "
        "network call for data. Edit the pairs in the sidebar's *Training "
        "data (inline pairs)* section.",
        icon="✅",
    )
else:
    _hf_default = _METHOD_DEFAULT_DATASETS.get(_current_method, "(none)")
    st.info(
        f"**Training on HuggingFace dataset `{_hf_default}`** "
        f"(the default for `{_current_method}`). Paste pairs into the sidebar "
        "textarea below to train on your own concept instead.",
        icon="🌐",
    )


# ── Custom-scenario builder ──────────────────────────────────────────────────


with st.expander(
    "**Build a custom scenario** — generate training pairs for your own concept",
    expanded=not inline_pairs,
):
    st.markdown(
        "Pick what kind of change you want, then either edit the template or "
        "let Claude generate pairs for your concept. The pairs land in the "
        "sidebar textarea, ready to Run."
    )

    intent_label_to_key = {
        "Add an attribute (spectacles, beard, long hair…)": "add_attribute",
        "Suppress a concept (cigarettes, weapons, NSFW…)": "suppress_concept",
        "Shift toward a demographic (Black, women, older adults…)": "shift_demographic",
        "Apply an art style (painterly, watercolor, anime…)": "apply_style",
    }
    chosen_label = st.radio(
        "I'm trying to…",
        list(intent_label_to_key),
        index=0,
        horizontal=False,
        key="steer_intent_label",
    )
    intent_key = intent_label_to_key[chosen_label]
    tmpl = INTENT_TEMPLATES[intent_key]

    st.markdown(f"**Format**: {tmpl['format_hint']}")
    st.caption(tmpl["tip"])

    # Template starter pairs ----------------------------------------------------
    starter_text = "\n".join(tmpl["starter_pairs"])  # type: ignore[arg-type]
    with st.expander(f"Show template starter ({len(tmpl['starter_pairs'])} pairs)", expanded=False):
        st.code(starter_text, language="text")
    if st.button(
        "Use template as starter pairs",
        help=(
            "Pastes the 8 template pairs into the sidebar textarea. Replace "
            "the placeholder (<ATTRIBUTE> / <CONCEPT> / <DEMO> / <STYLE>) with "
            "your concept, then Run."
        ),
        key="steer_use_template_btn",
    ):
        st.session_state["steer_inline_pairs"] = starter_text
        st.session_state["steer_method"] = str(tmpl["method_hint"])
        st.session_state["steer_alpha"] = float(tmpl["alpha_hint"])  # type: ignore[arg-type]
        st.rerun()

    st.divider()

    # Claude-driven pair generation --------------------------------------------
    if llm_is_available():
        st.markdown("##### Or have Claude write pairs for your concept")
        concept = st.text_input(
            "Describe your concept",
            placeholder=(
                "e.g. 'tattoos on the arms', 'wearing a chef's hat', "
                "'older adult faces', 'oil-painting style'"
            ),
            key="steer_claude_concept",
        )
        c_btn, c_n = st.columns([2, 1])
        with c_n:
            n_pairs = st.number_input(
                "Pairs", min_value=4, max_value=16, value=8, step=1, key="steer_claude_n"
            )
        with c_btn:
            ask = st.button(
                "Generate pairs with Claude →",
                type="primary",
                disabled=not concept.strip(),
                key="steer_claude_btn",
            )
        if ask:
            with st.spinner("Asking Claude…"):
                try:
                    result = generate_inline_pairs(
                        intent=intent_key, concept=concept.strip(), n=int(n_pairs)
                    )
                    st.session_state["steer_inline_pairs"] = result.as_textarea()
                    st.session_state["steer_method"] = result.method_hint
                    st.session_state["steer_alpha"] = float(result.alpha_hint)
                    st.session_state["steer_claude_notes"] = result.notes
                    st.rerun()
                except Exception as e:
                    st.error(f"Claude call failed: {type(e).__name__}: {e}")
        if st.session_state.get("steer_claude_notes"):
            st.caption(f"_{st.session_state['steer_claude_notes']}_")
    else:
        st.caption(
            "_Set `ANTHROPIC_API_KEY` in your shell to also get a "
            "'Generate pairs with Claude' button here._"
        )


# ── Quality tips ─────────────────────────────────────────────────────────────


with st.expander("**Quality tips** — what makes a good steering run", expanded=False):
    st.markdown(
        """
- **How many pairs?** 8–12 is the sweet spot. 5 minimum (mean estimates get
  noisy below that). Beyond ~20 you hit diminishing returns; spend the budget
  on diverse subjects instead.
- **What makes good pairs?** Each pair should differ in **one** thing — the
  target concept. Keep `pos` and `neg` structurally similar (same length, same
  subject, same grammar). Across pairs, use diverse subjects (different
  occupations, ages, contexts) so the direction generalises beyond one face.
- **LoReFT vs CAA?** **LoReFT** (low-rank adapter) trains a tiny network and
  injects layer-aware edits — best for *attributes* and *styles*. **CAA**
  computes mean(pos activations) − mean(neg activations) — best for *crisp
  directional shifts* (demographic, suppression with negative α).
- **Alpha tuning.** Start at the suggested value. If steered ≈ baseline → push
  alpha up. If outputs become noise or your prompt content gets overwhelmed →
  lower alpha or train on more samples. For *suppression*, alpha should be
  **negative** (e.g. −10) to subtract the direction.
- **Failure modes.** *Steered shows concept but loses prompt content* →
  alpha overpowered the prompt. *Steered looks identical to baseline* → alpha
  too low, or trained layer doesn't carry the concept (try a different
  `target_layer`). *Garbage / noise* → alpha too high or too few pairs.
"""
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
    "Inference prompts (one per line)",
    help="Prompts to generate — once as baseline, once steered. Separate from training pairs.",
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
st.sidebar.slider("Training samples", 10, 1000, key="steer_max_samples")
st.sidebar.slider("Training steps", 2, 500, key="steer_train_steps")

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
        height=200,
        key="steer_inline_pairs",
    )
    # Re-parse for live counter (the page-top parse was on the value at top of script).
    _live_pairs, _live_skipped = _parse_inline_pairs(
        str(st.session_state.get("steer_inline_pairs", ""))
    )
    if _live_pairs:
        st.caption(
            f"✅ **{len(_live_pairs)} valid pair(s)** parsed"
            + (f" · {len(_live_skipped)} skipped" if _live_skipped else "")
        )
    elif str(st.session_state.get("steer_inline_pairs", "")).strip():
        st.warning(
            "No `positive | negative` lines parsed — falling back to the HF dataset. "
            "Each line must contain a `|` separator.",
            icon="⚠️",
        )

# Pull session-state values for the rest of the page
steer_type = str(st.session_state["steer_method"])
prompts_raw = str(st.session_state["steer_prompts"])
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
alpha = float(st.session_state["steer_alpha"])
max_samples = int(st.session_state["steer_max_samples"])
train_steps = int(st.session_state["steer_train_steps"])
goal = str(st.session_state["steer_goal"])

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
