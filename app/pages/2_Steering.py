"""Steering playground. Train a concept direction and inject it during generation.

The page is structured around the four steps from paper Figure 1:
1. What you want (goal and intent)
2. Training data (inline pairs or HF dataset)
3. Run config (method, alpha, model, prompts)
4. Run + results
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import streamlit as st

from app.lib import (
    INTENT_TEMPLATES,
    apply_payload,
    collect_images,
    detect_concept,
    device_dtype_picker,
    generate_inline_pairs,
    has_unresolved_placeholders,
    load_fingerprint,
    model_preset_picker,
    pair_baseline_modified,
    parse_pipe_lines,
    render_app_footer,
    render_run_label_sidebar,
    run_workflow,
    sweep_old_streamlit_tempdirs,
)
from app.lib import (
    is_available as llm_is_available,
)

st.set_page_config(page_title="Steering • T2I-Interp", layout="wide")

# Opportunistic cleanup of stale tempdirs from previous Run clicks.
sweep_old_streamlit_tempdirs("streamlit_steer_")

# ── Defaults + recipe-payload intake ─────────────────────────────────────────

_STEER_DEFAULTS: dict[str, object] = {
    "steer_goal": "",
    "steer_method": "loreft",
    "steer_model_preset": "sdxl_turbo",
    "steer_prompts": "A photo of Jack Sparrow\nA photo of Simba",
    "steer_alpha": 10.0,
    "steer_max_samples": 100,
    "steer_train_steps": 50,
    "steer_inline_pairs": "",
}
apply_payload(
    st.session_state,
    prefix="steer",
    defaults=_STEER_DEFAULTS,
    workflow_name="Steering",
)


# Per-method default HF dataset (matches the YAML configs).
_METHOD_DEFAULT_DATASETS = {
    "caa": "nirmalendu01/spectacles-bias-prompts-headshot-captioned",
    "loreft": "nirmalendu01/spectacles-bias-prompts-headshot",
    "ksteer": "nirmalendu01/spectacles-bias-prompts-headshot-captioned",
}

# Verb per intent — used to render "add spectacles", "suppress cigarettes",
# etc. in goal banners and the Run button label.
_INTENT_VERBS = {
    "add_attribute": "add",
    "suppress_concept": "suppress",
    "shift_demographic": "shift toward",
    "apply_style": "apply",
}


def _goal_phrase(intent_key: str, concept: str | None) -> str:
    """Build a short phrase like 'add spectacles' or 'suppress cigarettes'."""
    verb = _INTENT_VERBS.get(intent_key, "steer toward")
    if concept:
        return f"{verb} {concept}"
    return f"{verb} the concept in your pairs"


# ── Page header ──────────────────────────────────────────────────────────────

st.title("Steering")
st.markdown("##### Train a concept direction once, then add or subtract it at generation time.")
st.caption(
    "Paper §3.2. Three methods: CAA (mean activation difference), K-Steer "
    "(classifier-guided gradient), LoReFT (low-rank adapter). The paper's "
    "Figure 2 spectacles result uses LoReFT on SDXL-Turbo."
)


# ── Step 1: What you want ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 1 · What you want")
    st.caption(
        "Pick the kind of change you want. Click **Use example pairs** to "
        "load 8 ready-to-run pairs for the default concept (spectacles, "
        "cigarettes, Black, or painterly), or describe a different concept "
        "and let Claude write the pairs."
    )

    intent_label_to_key = {
        "Add an attribute (spectacles, beard, long hair)": "add_attribute",
        "Suppress a concept (cigarettes, weapons, NSFW)": "suppress_concept",
        "Shift toward a demographic (Black, women, older adults)": "shift_demographic",
        "Apply an art style (painterly, watercolor, anime)": "apply_style",
    }
    chosen_label = st.radio(
        "I want to:",
        list(intent_label_to_key),
        index=0,
        horizontal=False,
        key="steer_intent_label",
    )
    intent_key = intent_label_to_key[chosen_label]
    tmpl = INTENT_TEMPLATES[intent_key]

    st.markdown(f"**Pair format**: {tmpl['format_hint']}")
    st.caption(tmpl["tip"])

    c_tmpl, c_claude = st.columns(2)
    starter_text = "\n".join(tmpl["starter_pairs"])  # type: ignore[arg-type]
    with c_tmpl:
        with st.popover("Show example pairs", use_container_width=True):
            st.code(starter_text, language="text")
        if st.button(
            "Use example pairs",
            help=(
                "Drops 8 working example pairs into Step 2. Press Run as-is "
                "to get the default concept for the chosen intent (spectacles, "
                "cigarettes, Black, or painterly). Or swap the concept word "
                "first and Run."
            ),
            key="steer_use_template_btn",
            use_container_width=True,
        ):
            st.session_state["steer_inline_pairs"] = starter_text
            st.session_state["steer_method"] = str(tmpl["method_hint"])
            st.session_state["steer_alpha"] = float(tmpl["alpha_hint"])  # type: ignore[arg-type]
            st.rerun()

    with c_claude:
        if llm_is_available():
            concept = st.text_input(
                "Or describe a concept",
                placeholder="e.g. 'tattoos', 'chef hat', 'older adults', 'oil-painting style'",
                key="steer_claude_concept",
                label_visibility="collapsed",
            )
            if st.button(
                "Generate pairs with Claude",
                type="secondary",
                disabled=not concept.strip(),
                key="steer_claude_btn",
                use_container_width=True,
            ):
                with st.spinner("Asking Claude..."):
                    try:
                        result = generate_inline_pairs(
                            intent=intent_key, concept=concept.strip(), n=8
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
                "Set `ANTHROPIC_API_KEY` in your shell for a "
                "'Generate pairs with Claude' button here."
            )


# ── Step 2: Training data ────────────────────────────────────────────────────

inline_pairs_text = str(st.session_state.get("steer_inline_pairs", ""))
_parsed_pairs, _inline_skipped = parse_pipe_lines(inline_pairs_text, require_separator=True)
inline_pairs: list[dict[str, str]] = [p for p in _parsed_pairs if isinstance(p, dict)]
unresolved_placeholders = has_unresolved_placeholders(inline_pairs_text)
_current_method = str(st.session_state.get("steer_method", "loreft"))

# What concept is the user actually steering toward? Detected from the diff
# between pos and neg prompts across all pairs. Together with the intent verb
# this gives us a single phrase ("add spectacles", "suppress cigarettes") that
# the rest of the page can reuse.
_detected_concept = detect_concept(inline_pairs) if inline_pairs else None
_goal = _goal_phrase(intent_key, _detected_concept)

with st.container(border=True):
    st.markdown("### Step 2 · Training data")
    if inline_pairs:
        st.success(
            f"**Goal: {_goal}.** Training on {len(inline_pairs)} inline prompt "
            "pairs. The model will learn a direction from `pos minus neg` "
            "across these pairs, then add it (scaled by alpha) when "
            "generating your inference prompts.",
            icon="🎯",
        )
    else:
        _hf_default = _METHOD_DEFAULT_DATASETS.get(_current_method, "(none)")
        st.info(
            f"**Training on HuggingFace dataset `{_hf_default}`** "
            f"(default for `{_current_method}`). Paste pairs below to train "
            "on your own concept instead.",
            icon="🌐",
        )

    if unresolved_placeholders:
        st.error(
            "**Unresolved placeholders in your pairs**: "
            f"{', '.join(f'`{p}`' for p in unresolved_placeholders)}. "
            "Replace these with real words (e.g. `spectacles`, `a beard`) "
            "before Run. Training on literal `<ATTRIBUTE>` text produces garbage.",
            icon="⚠️",
        )

    st.text_area(
        "Inline pairs (positive | negative, one per line)",
        help=(
            "When set, trains on these inline pairs instead of the HF "
            "dataset. For CAA each `positive` becomes a label=1 caption and "
            "each `negative` a label=0 caption. For LoReFT each pair becomes "
            "one (base=negative, teacher=positive) row.\n\nLeave empty to "
            "use the configured HF dataset."
        ),
        placeholder=(
            "A photo of Jack Sparrow with spectacles | A photo of Jack Sparrow\n"
            "A photo of Simba with spectacles | A photo of Simba"
        ),
        height=180,
        key="steer_inline_pairs",
        label_visibility="visible",
    )
    # Live counter.
    _live_pairs, _live_skipped = parse_pipe_lines(
        str(st.session_state.get("steer_inline_pairs", "")), require_separator=True
    )
    if _live_pairs:
        st.caption(
            f"**{len(_live_pairs)} valid pair(s)** parsed"
            + (f" · {len(_live_skipped)} skipped" if _live_skipped else "")
        )
    elif str(st.session_state.get("steer_inline_pairs", "")).strip():
        st.warning(
            "No `positive | negative` lines parsed. Falling back to the HF "
            "dataset. Each line must contain a `|` separator.",
            icon="⚠️",
        )


# ── Step 3: Run config ───────────────────────────────────────────────────────

# The sidebar holds the device/dtype and the lower-traffic knobs. The most
# important ones (method, alpha, inference prompts) live in Step 3 on the main
# page so the user sees them without hunting.
st.sidebar.header("Hardware")
device, dtype = device_dtype_picker(default_device="mps")
st.sidebar.header("Less-used knobs")
st.sidebar.slider("Training samples", 10, 1000, key="steer_max_samples")
st.sidebar.slider("Training steps", 2, 500, key="steer_train_steps")
preset = model_preset_picker(
    default=str(st.session_state.get("steer_model_preset", "sdxl_turbo")),
    key="steer_model_preset",
)
render_run_label_sidebar(key="steer_goal")

with st.container(border=True):
    st.markdown("### Step 3 · Run config")

    c_method, c_alpha = st.columns(2)
    with c_method:
        st.selectbox(
            "Steering method",
            ["loreft", "caa", "ksteer"],
            key="steer_method",
            help=(
                "LoReFT trains a tiny adapter, best for attributes and styles. "
                "CAA computes mean(pos) minus mean(neg), best for crisp shifts. "
                "K-Steer uses a classifier."
            ),
        )
    with c_alpha:
        st.slider(
            "Alpha (steering strength)",
            -30.0,
            30.0,
            step=0.5,
            help=(
                "0 means no steering. Higher means stronger. Negative "
                "subtracts the direction (suppression). 10 to 20 typical "
                "for SDXL-Turbo + LoReFT."
            ),
            key="steer_alpha",
        )

    st.text_area(
        "Inference prompts (one per line)",
        help=(
            "What gets generated, once as baseline and once steered. "
            "Separate from the training pairs in Step 2."
        ),
        key="steer_prompts",
        height=110,
    )

    with st.expander("Quality tips and failure modes", expanded=False):
        st.markdown(
            """
- **How many pairs?** 8 to 12 is the sweet spot. 5 minimum. Past 20,
  diminishing returns. Spend the budget on diverse subjects instead.
- **What makes good pairs?** Each pair should differ in one thing, the
  target concept. Keep `pos` and `neg` structurally similar. Across
  pairs, vary the subject (occupations, ages, contexts).
- **LoReFT vs CAA?** LoReFT (low-rank adapter) is best for attributes
  and styles. CAA (mean activation diff) is best for crisp directional
  shifts and suppression with negative alpha.
- **Alpha tuning.** Start at the suggested value. If steered looks like
  baseline, push alpha up. If outputs become noise, lower alpha or
  train on more pairs. For suppression, alpha should be **negative**
  (around -10) to subtract the direction.
- **Failure modes.** Steered shows the concept but loses prompt content:
  alpha overpowered the prompt. Steered looks identical to baseline:
  alpha too low. Garbage or noise: alpha too high.
"""
        )


# ── Pull values for the override list ────────────────────────────────────────
steer_type = str(st.session_state["steer_method"])
prompts_raw = str(st.session_state["steer_prompts"])
prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
alpha = float(st.session_state["steer_alpha"])
max_samples = int(st.session_state["steer_max_samples"])
train_steps = int(st.session_state["steer_train_steps"])
goal = str(st.session_state["steer_goal"])


def _build_overrides(out_dir: str) -> tuple[list[str], str | None, str]:
    """Build Hydra overrides + paths for the JSON files we will write.

    Returns `(overrides, inline_pairs_file_path_or_none, prompts_file_path)`.
    The caller materialises the JSON files only when Run is actually clicked
    so script reruns from widget changes don't litter tempdirs.
    """
    pairs_file: str | None = None
    if inline_pairs:
        pairs_file = os.path.join(out_dir, "inline_pairs.json")
    # Prompts go through a JSON file rather than `prompts=[a, b]` on the CLI —
    # Hydra splits list literals on commas and chokes on prompts containing
    # commas or spaces (e.g. "a photo of a person, smiling").
    prompts_file = os.path.join(out_dir, "prompts.json")
    ovs = [
        f"--config-name=steer/{steer_type}",
        f"device={device}",
        f"dtype={dtype}",
        f"alpha={alpha}",
        f"max_samples={max_samples}",
        f"train_steps={train_steps}",
        f"+prompts_file={prompts_file}",
        f"save_dir={out_dir}/cache",
        f"output_dir={out_dir}",
        f"hydra.run.dir={out_dir}/.hydra",
        "wandb.project=null",
    ]
    if preset:
        ovs.append(f"model={preset}")
    if pairs_file:
        ovs.append(f"inline_pairs_file={pairs_file}")
    return ovs, pairs_file, prompts_file


# ── Step 4: Run ──────────────────────────────────────────────────────────────

# Show the CLI preview with a placeholder out_dir so we don't create a new
# tempdir on every script rerun (only when Run is actually clicked).
_preview_overrides, _, _ = _build_overrides("/tmp/streamlit_steer_<auto>")

with st.container(border=True):
    st.markdown("### Step 4 · Run")
    _data_source = "inline pairs" if inline_pairs else "the configured HF dataset"
    if inline_pairs:
        st.markdown(
            f"Will train **{steer_type.upper()}** on {_data_source} to "
            f"**{_goal}** at `alpha={alpha:g}`, then generate "
            f"**{len(prompts)} prompt(s)** baseline and steered."
        )
    else:
        st.markdown(
            f"Will train **{steer_type.upper()}** on {_data_source} at "
            f"`alpha={alpha:g}`, then generate "
            f"**{len(prompts)} prompt(s)** baseline and steered."
        )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-steer " + " \\\n  ".join(_preview_overrides), language="bash")
    _btn_label = f"▶ Train and {_goal}" if inline_pairs else "▶ Train and generate"
    run_clicked = st.button(
        _btn_label,
        type="primary",
        use_container_width=True,
        disabled=bool(unresolved_placeholders),
        help=(
            "Replace the placeholder tokens in Step 2 first." if unresolved_placeholders else None
        ),
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    out_dir = tempfile.mkdtemp(prefix="streamlit_steer_")
    overrides, pairs_file, prompts_file = _build_overrides(out_dir)
    if pairs_file:
        with open(pairs_file, "w") as f:
            json.dump(inline_pairs, f)
    with open(prompts_file, "w") as f:
        json.dump(prompts, f)

    with st.status(f"Training {steer_type.upper()} and generating...", expanded=True) as status:
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
            status.update(label="Run failed. See logs above.", state="error")

    st.divider()
    st.subheader("Results")

    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    # `run_steer.py` appends `_<block>_alpha=<alpha>` to cfg.output_dir, so
    # the outputs land in a sibling of `out_dir`, not under it. Walk siblings.
    images = collect_images(out_dir, include_prefix_siblings=True)
    if images:
        triples = pair_baseline_modified(images, modified_kinds=("steered",), label_prefix="prompt")
        st.markdown(f"**{len(triples)} prompt(s)** generated.")
        for label, baseline, steered in triples:
            with st.container(border=True):
                st.markdown(f"##### {label}")
                c_b, c_s = st.columns(2)
                with c_b:
                    st.markdown("**Baseline** (no steering)")
                    if baseline is not None:
                        st.image(str(baseline), use_container_width=True)
                    else:
                        st.caption("(missing)")
                with c_s:
                    st.markdown(f"**Steered** (alpha = {alpha:g})")
                    if steered is not None:
                        st.image(str(steered), use_container_width=True)
                    else:
                        st.caption("(missing)")

        with st.expander("How to read these results", expanded=False):
            st.markdown(
                """
- **Baseline** is what the model produces normally for this prompt and seed.
- **Steered** applies the trained direction at `alpha`. It should keep
  the prompt's content while leaning toward the trained concept.
- **Steered looks like baseline**: alpha too low, or wrong training layer.
  Push alpha up to 15 or 20.
- **Steered looks like garbage**: alpha too high, or adapter overfit a
  tiny dataset. Lower alpha or add more pairs.
- **Steered shows the concept but loses prompt content** (e.g. you
  wanted "Jack Sparrow with spectacles" and got just spectacles): alpha
  overpowered the prompt. Reduce it.
"""
            )
    else:
        st.warning("No images produced. Check logs above.")

    fp = load_fingerprint(out_dir, include_prefix_siblings=True)
    if fp:
        with st.container(border=True):
            st.markdown("##### Run fingerprint")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hash", fp["fingerprint_hash"])
            c2.metric("Workflow", fp["workflow"])
            c3.metric("Alpha", str(fp["intervention"].get("alpha", "-")))
            with st.expander("Full fingerprint JSON", expanded=False):
                st.json(fp)

render_app_footer()
