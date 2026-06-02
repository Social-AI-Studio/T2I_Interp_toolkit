"""Localisation playground. Scale a cross-attention head and see the effect."""

from __future__ import annotations

import tempfile

import streamlit as st

from app.lib import (
    apply_payload,
    collect_images,
    device_dtype_picker,
    load_fingerprint,
    load_wandb_run,
    model_preset_picker,
    pair_baseline_modified,
    render_app_footer,
    render_run_label_sidebar,
    render_wandb_panel,
    render_workflow_run,
    scenario_radio,
    sweep_old_streamlit_tempdirs,
    wandb_picker,
)

st.set_page_config(page_title="Localisation • T2I-Interp", layout="wide")

# Opportunistic cleanup of stale tempdirs from previous Run clicks. Skips the
# in-flight one (only sweeps dirs older than the configured cutoff).
sweep_old_streamlit_tempdirs("streamlit_loc_")

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
apply_payload(
    st.session_state,
    prefix="loc",
    defaults=_LOC_DEFAULTS,
    workflow_name="Localisation",
)

_LOC_PRESETS: dict[str, dict[str, object]] = {
    "Probe early UNet (composition)": {
        "label": "Probe early UNet (composition layer)",
        "prompt": "a unicorn in a forest",
        "target_layer": "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
        "target_head": 0,
        "factor": 0.0,
        "hint": (
            "Early down-blocks tend to carry rough composition (where "
            "objects sit, overall layout). Ablate head 0 here, then re-run "
            "with heads 1 to 7 to find which head matters."
        ),
    },
    "Probe mid UNet (object identity)": {
        "label": "Probe mid UNet (object identity)",
        "prompt": "a red apple on a wooden table",
        "target_layer": "mid_block_attentions_0_transformer_blocks_0_attn2_out",
        "target_head": 3,
        "factor": 0.0,
        "hint": (
            "The mid_block tends to carry object identity and colour "
            "binding. Head 3 is a common colour-binding suspect for SD 1.5. "
            "Ablate it and see whether 'red' stops appearing."
        ),
    },
    "Probe late UNet (texture and fine detail)": {
        "label": "Probe late UNet (texture and fine detail)",
        "prompt": "a busy city street at dusk",
        "target_layer": "up_blocks_2_attentions_0_transformer_blocks_0_attn2_out",
        "target_head": 0,
        "factor": 0.0,
        "hint": (
            "Late up-blocks tend to carry texture, lighting, and fine "
            "detail. Ablate heads here and watch what gets blurred or "
            "stripped from the image."
        ),
    },
    "Amplify a head (boost its effect)": {
        "label": "Amplify a head (boost its effect)",
        "prompt": "a red apple on a wooden table",
        "target_layer": "mid_block_attentions_0_transformer_blocks_0_attn2_out",
        "target_head": 3,
        "factor": 2.0,
        "hint": (
            "Instead of ablating, multiply the head's contribution by 2. "
            "Useful for confirming a head you suspect carries a concept. "
            "If the concept gets stronger, you've confirmed it."
        ),
    },
}


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
    st.markdown("### Step 1 · What you want to find out")
    scenario_radio(
        presets=_LOC_PRESETS,
        prefix="loc",
        apply_keys=["prompt", "target_layer", "target_head", "factor"],
    )
    st.text_input(
        "Prompt to test",
        help="Pick a prompt where the scenario's effect should be visible.",
        key="loc_prompt",
    )


# ── Step 2: Where to look ────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown("### Step 2 · Where to look")
    st.caption("Which layer and head to scale.")

    # Curated list of common cross-attn output sites — typo-safe dropdown
    # plus an explicit "Custom..." escape hatch. Heads count is preset-aware
    # (SD 1.5 has 8 heads per cross-attn; SDXL families have 10–20).
    _SD15_ATTN2_LAYERS = [
        "down_blocks_0_attentions_0_transformer_blocks_0_attn2_out",
        "down_blocks_0_attentions_1_transformer_blocks_0_attn2_out",
        "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
        "down_blocks_1_attentions_1_transformer_blocks_0_attn2_out",
        "down_blocks_2_attentions_0_transformer_blocks_0_attn2_out",
        "down_blocks_2_attentions_1_transformer_blocks_0_attn2_out",
        "mid_block_attentions_0_transformer_blocks_0_attn2_out",
        "up_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
        "up_blocks_1_attentions_1_transformer_blocks_0_attn2_out",
        "up_blocks_1_attentions_2_transformer_blocks_0_attn2_out",
        "up_blocks_2_attentions_0_transformer_blocks_0_attn2_out",
        "up_blocks_2_attentions_1_transformer_blocks_0_attn2_out",
        "up_blocks_2_attentions_2_transformer_blocks_0_attn2_out",
        "up_blocks_3_attentions_0_transformer_blocks_0_attn2_out",
        "up_blocks_3_attentions_1_transformer_blocks_0_attn2_out",
        "up_blocks_3_attentions_2_transformer_blocks_0_attn2_out",
    ]
    _CUSTOM_SENTINEL = "Custom..."
    _layer_options = [*_SD15_ATTN2_LAYERS, _CUSTOM_SENTINEL]

    _current_layer = str(st.session_state.get("loc_target_layer", _SD15_ATTN2_LAYERS[2]))
    if _current_layer not in _SD15_ATTN2_LAYERS:
        # Recipe payload or previous custom value — preselect Custom...
        _default_dropdown = _CUSTOM_SENTINEL
    else:
        _default_dropdown = _current_layer

    c_layer, c_head = st.columns([2, 1])
    with c_layer:
        _picked = st.selectbox(
            "Target layer (UNet cross-attn output)",
            _layer_options,
            index=_layer_options.index(_default_dropdown),
            help=(
                "`down_blocks_1` is early (rough composition). `mid_block` "
                "is mid (object identity). `up_blocks_X` is late (textures "
                "and fine detail). Suffix `_attn2_out` is the output of a "
                "cross-attention layer (image to text). Pick `Custom...` "
                "to type a non-standard hook path."
            ),
            key="loc_target_layer_choice",
        )
        if _picked == _CUSTOM_SENTINEL:
            st.text_input(
                "Custom UNet path",
                value=_current_layer if _current_layer not in _SD15_ATTN2_LAYERS else "",
                key="loc_target_layer",
            )
        else:
            st.session_state["loc_target_layer"] = _picked
    with c_head:
        # SD 1.5 cross-attn has 8 heads; SDXL cross-attn varies (10–20). Pick
        # the safe upper bound for the selected preset so the dropdown won't
        # offer head indices the model doesn't have.
        _preset_choice = str(st.session_state.get("loc_model_preset", "sd15"))
        _heads_by_preset = {"sd15": 8, "sdxl_turbo": 10}
        _n_heads = _heads_by_preset.get(_preset_choice, 8)
        st.selectbox(
            f"Head index (0-{_n_heads - 1})",
            list(range(_n_heads)),
            help=(
                f"This model preset has {_n_heads} cross-attn heads. "
                "Iterate over them to find the one that carries your concept."
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


# ── Sidebar config ───────────────────────────────────────────────────────────

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
    help="Same seed means same initial noise, so comparisons isolate the head's effect.",
    key="loc_seed",
)
preset = model_preset_picker(
    default=str(st.session_state.get("loc_model_preset", "(use config default)")),
    options=("sd15", "sdxl_turbo"),
    key="loc_model_preset",
)
wandb_project, wandb_entity = wandb_picker()
render_run_label_sidebar(key="loc_goal")

# Pull session-state values for the override list
prompt = str(st.session_state["loc_prompt"])
target_layer = str(st.session_state["loc_target_layer"])
target_head = int(st.session_state["loc_target_head"])
factor = float(st.session_state["loc_factor"])
n_steps = int(st.session_state["loc_n_steps"])
seed = int(st.session_state["loc_seed"])
goal = str(st.session_state["loc_goal"])


def _build_overrides(out_dir: str) -> list[str]:
    """Build Hydra overrides for the run."""
    ovs = [
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
    ]
    if wandb_project:
        ovs.append(f"wandb.project={wandb_project}")
        if wandb_entity:
            ovs.append(f"wandb.entity={wandb_entity}")
    else:
        ovs.append("wandb.project=null")
    if preset:
        ovs.append(f"model={preset}")
    return ovs


# ── Step 3: Run ──────────────────────────────────────────────────────────────

_preview_overrides = _build_overrides("/tmp/streamlit_loc_<auto>")

with st.container(border=True):
    st.markdown("### Step 3 · Run")
    st.markdown(
        f"Will generate the prompt twice. Once unmodified (baseline), once "
        f"with head **{target_head}** of `{target_layer}` scaled by "
        f"**{factor:g}**."
    )
    with st.expander("CLI equivalent", expanded=False):
        st.code("t2i-localise " + " \\\n  ".join(_preview_overrides), language="bash")
    run_clicked = st.button(
        "▶ Run head ablation",
        type="primary",
        use_container_width=True,
    )


# ── Results ──────────────────────────────────────────────────────────────────

if run_clicked:
    out_dir = tempfile.mkdtemp(prefix="streamlit_loc_")
    overrides = _build_overrides(out_dir)
    result, elapsed = render_workflow_run(
        "t2i-localise", overrides, out_dir=out_dir, running_label="Running localisation..."
    )

    st.divider()
    st.subheader("Results")
    if goal:
        st.markdown(f"**Goal:** _{goal}_")

    images = collect_images(out_dir)
    if images:
        # run_localisation.py writes a matplotlib composite grid for each
        # swept layer at `<layer[:80]>.png` — baseline + every modified head
        # in one figure. It's a convenience summary, not a separate result;
        # render it standalone above the per-head pairs so it doesn't appear
        # paired against an identical baseline.
        composites = [p for p in images if "__h" not in p.name and p.name != "baseline.png"]
        per_head = [p for p in images if p not in composites]
        triples = pair_baseline_modified(
            per_head,
            modified_kinds=("modified", "head", "layer", "ablated"),
            label_prefix="head",
        )
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
        if composites:
            with st.container(border=True):
                st.markdown("##### Composite sweep grid")
                st.caption(
                    "Side-by-side baseline + every per-head ablation for the "
                    "selected layer. Same data as the per-head containers above."
                )
                for grid in composites:
                    st.image(str(grid), caption=grid.stem, use_container_width=True)

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
