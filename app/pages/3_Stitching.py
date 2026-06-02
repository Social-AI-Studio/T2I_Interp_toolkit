"""Stitching playground. Train an MLP mapper across two activation spaces."""

from __future__ import annotations

import json
import os
import tempfile
import streamlit as st

from app.lib import (
    apply_payload,
    collect_images,
    device_dtype_picker,
    load_fingerprint,
    load_metrics,
    load_wandb_run,
    model_preset_picker,
    parse_pipe_lines,
    render_app_footer,
    render_run_label_sidebar,
    render_wandb_panel,
    render_workflow_run,
    scenario_radio,
    sweep_old_streamlit_tempdirs,
    wandb_picker,
)
from app.lib.prompts import STITCH_GENERIC_PROMPTS

st.set_page_config(page_title="Stitching • T2I-Interp", layout="wide")

# Opportunistic cleanup of stale tempdirs from previous Run clicks.
sweep_old_streamlit_tempdirs("streamlit_stitch_")

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
    # Cross-model controls. Defaults preserve single-model behaviour; pick a
    # preset_b to opt into paper §4 cross-model stitching.
    "stitch_model_b_preset": "(single-model: same as model_a)",
    "stitch_layer_a": "text_encoder.text_model.final_layer_norm",
    "stitch_layer_b": "unet.conv_out",
    "stitch_mode": "train",
}

# (model_b preset label, model_key_b override, lora_b.repo, lora_b.scheduler).
# Each preset matches a documented setup in t2i_interp/config/stitch/run.yaml.
_STITCH_MODEL_B_PRESETS: dict[str, dict[str, str | None]] = {
    "(single-model: same as model_a)": {
        "model_key_b": None,
        "lora_repo": None,
        "lora_scheduler": None,
    },
    "SD1.5 base + LCM-LoRA (paper §4 fine-tune transfer)": {
        "model_key_b": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "lora_repo": "latent-consistency/lcm-lora-sdv1-5",
        "lora_scheduler": "LCMScheduler",
    },
}

# Curated cross-model-friendly hook sites for SD1.5. mid_block_attentions
# matches the paper Figure 4 case study.
_STITCH_LAYER_OPTIONS = [
    "text_encoder.text_model.final_layer_norm",
    "unet.mid_block.attentions.0.transformer_blocks.0.attn2",
    "unet.mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0",
    "unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2",
    "unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2",
    "unet.conv_out",
]
apply_payload(
    st.session_state,
    prefix="stitch",
    defaults=_STITCH_DEFAULTS,
    workflow_name="Stitching",
)


_STITCH_PRESETS: dict[str, dict[str, object]] = {
    "Test if two layers carry similar info (small mapper)": {
        "label": "Test if two layers carry similar information",
        "hidden_dim": 256,
        "max_samples": 50,
        "num_steps": 100,
        "inline_pairs": STITCH_GENERIC_PROMPTS,
        "hint": (
            "A small mapper that converges is real evidence the two "
            "layers carry comparable information. If you bump hidden_dim "
            "to 1024 you can paper over real incompatibility, so keep "
            "it small for an honest test."
        ),
    },
    "Transfer behaviour between two models (large mapper)": {
        "label": "Transfer behaviour between two models",
        "hidden_dim": 512,
        "max_samples": 100,
        "num_steps": 200,
        "inline_pairs": STITCH_GENERIC_PROMPTS,
        "hint": (
            "Larger mapper, more samples, more steps. Used for cross-"
            "model behaviour transfer in the paper's §4 case study. The "
            "mapper has to be expressive enough to translate between two "
            "different models."
        ),
    },
}


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
    st.markdown("### Step 1 · What you want to do")
    scenario_radio(
        presets=_STITCH_PRESETS,
        prefix="stitch",
        apply_keys=["hidden_dim", "max_samples", "num_steps", "inline_pairs"],
    )


# ── Step 2: Training data ────────────────────────────────────────────────────

_pre_inline, _pre_skipped = parse_pipe_lines(
    str(st.session_state.get("stitch_inline_pairs", "")),
    require_separator=False,
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
        placeholder="a photo of a person\na photo of a cat\na photo of a landscape",
        height=180,
        key="stitch_inline_pairs",
    )
    _live_parsed, _live_skipped = parse_pipe_lines(
        str(st.session_state.get("stitch_inline_pairs", "")),
        require_separator=False,
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
# 64 - 16384 covers everything from a small test mapper up to the paper
# Figure 4 setup (hidden_dim=8448 = 11*768). Step 64 keeps the slider usable.
st.sidebar.slider("Mapper hidden_dim", 64, 16384, step=64, key="stitch_hidden_dim")
st.sidebar.slider("Train samples", 10, 500, key="stitch_max_samples")
st.sidebar.slider("Mapper training steps", 5, 1000, key="stitch_num_steps")
st.sidebar.slider("Inference steps", 4, 50, key="stitch_num_inference_steps")
preset = model_preset_picker(
    default=str(st.session_state.get("stitch_model_preset", "sd15")),
    key="stitch_model_preset",
)
st.sidebar.header("Cross-model setup (paper §4)")
st.sidebar.selectbox(
    "model_b",
    list(_STITCH_MODEL_B_PRESETS.keys()),
    key="stitch_model_b_preset",
    help=(
        "model_a is selected above ('Model preset'). For paper §4 "
        "cross-model stitching, pick a different model here. Defaults to "
        "single-model stitching (model_b = model_a)."
    ),
)
_layer_a_options = list(_STITCH_LAYER_OPTIONS)
_current_la = str(st.session_state.get("stitch_layer_a", _STITCH_LAYER_OPTIONS[0]))
if _current_la not in _layer_a_options:
    _layer_a_options.append(_current_la)
st.sidebar.selectbox(
    "layer_a (source activations)",
    _layer_a_options,
    index=_layer_a_options.index(_current_la),
    key="stitch_layer_a",
    help="Where in model_a to capture activations the mapper will translate from.",
)
_layer_b_options = list(_STITCH_LAYER_OPTIONS)
_current_lb = str(st.session_state.get("stitch_layer_b", _STITCH_LAYER_OPTIONS[-1]))
if _current_lb not in _layer_b_options:
    _layer_b_options.append(_current_lb)
st.sidebar.selectbox(
    "layer_b (target activations)",
    _layer_b_options,
    index=_layer_b_options.index(_current_lb),
    key="stitch_layer_b",
    help="Where in model_b the mapped activations get injected.",
)
st.sidebar.selectbox(
    "Stitch mode",
    ["train", "steer_contrast", "steer_transfer"],
    key="stitch_mode",
    help=(
        "train: collect paired activations, fit mapper, run stitched inference. "
        "steer_contrast / steer_transfer: load a previously trained mapper "
        "and apply a steering direction (the paper's MODE=steer)."
    ),
)
wandb_project, wandb_entity = wandb_picker()
render_run_label_sidebar(key="stitch_goal")

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
layer_a = str(st.session_state["stitch_layer_a"])
layer_b = str(st.session_state["stitch_layer_b"])
stitch_mode = str(st.session_state["stitch_mode"])
model_b_cfg = _STITCH_MODEL_B_PRESETS[str(st.session_state["stitch_model_b_preset"])]

# Re-parse from session_state in case the textarea was edited after the top-of-page parse.
inline_pairs, _inline_skipped = parse_pipe_lines(
    str(st.session_state.get("stitch_inline_pairs", "")),
    require_separator=False,
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


def _build_overrides(out_dir: str) -> tuple[list[str], str | None, str]:
    pairs_file: str | None = None
    if inline_pairs:
        pairs_file = os.path.join(out_dir, "inline_pairs.json")
    prompts_file = os.path.join(out_dir, "prompts.json")
    ovs = [
        f"device={device}",
        f"dtype={dtype}",
        f"hidden_dim={hidden_dim}",
        f"max_samples={max_samples}",
        f"num_steps={num_steps}",
        f"num_inference_steps={num_inference_steps}",
        f"+prompts_file={prompts_file}",
        f"layer_a={layer_a}",
        f"layer_b={layer_b}",
        f"mode={stitch_mode}",
        f"save_dir={out_dir}/cache",
        f"output_dir={out_dir}",
        f"hydra.run.dir={out_dir}/.hydra",
    ]
    if wandb_project:
        ovs.append(f"wandb.project={wandb_project}")
        if wandb_entity:
            ovs.append(f"wandb.entity={wandb_entity}")
    else:
        ovs.append("wandb.project=null")
    if preset:
        ovs.append(f"model={preset}")
    if pairs_file:
        ovs.append(f"inline_pairs_file={pairs_file}")
    # Cross-model setup: pass model_key_b / lora_b.* when the user picked a
    # non-default model_b preset.
    if model_b_cfg.get("model_key_b"):
        ovs.append(f"model_key_b={model_b_cfg['model_key_b']}")
    if model_b_cfg.get("lora_repo"):
        ovs.append(f"lora_b.repo={model_b_cfg['lora_repo']}")
    if model_b_cfg.get("lora_scheduler"):
        ovs.append(f"lora_b.scheduler={model_b_cfg['lora_scheduler']}")
    return ovs, pairs_file, prompts_file


# ── Step 4: Run ──────────────────────────────────────────────────────────────

_preview_overrides, _, _ = _build_overrides("/tmp/streamlit_stitch_<auto>")

with st.container(border=True):
    st.markdown("### Step 4 · Run")
    st.markdown(
        f"Will train an MLP mapper (`hidden_dim={hidden_dim}`) on "
        f"{'inline prompts' if inline_pairs else 'the HF dataset'} for "
        f"`{num_steps}` steps, then generate **{len(prompts)} prompt(s)** "
        "with the mapper rewiring activations."
    )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-stitch " + " \\\n  ".join(_preview_overrides), language="bash")
    run_clicked = st.button(
        "▶ Train mapper and generate",
        type="primary",
        use_container_width=True,
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    out_dir = tempfile.mkdtemp(prefix="streamlit_stitch_")
    overrides, pairs_file, prompts_file = _build_overrides(out_dir)
    if pairs_file:
        with open(pairs_file, "w") as f:
            json.dump(inline_pairs, f)
    with open(prompts_file, "w") as f:
        json.dump(prompts, f)

    result, elapsed = render_workflow_run(
        "t2i-stitch",
        overrides,
        out_dir=out_dir,
        running_label="Training mapper and generating...",
    )

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

    metrics = load_metrics(out_dir)
    if metrics is not None:
        with st.container(border=True):
            st.markdown("##### Metrics")
            if metrics:
                with st.expander("Full JSON", expanded=False):
                    st.json(metrics)
            else:
                st.info(
                    "metrics.json is empty — CLIP / FID / LPIPS backends "
                    "aren't installed. Run `uv sync --extra metrics` to enable."
                )

    render_wandb_panel(load_wandb_run(out_dir))

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

render_app_footer()
