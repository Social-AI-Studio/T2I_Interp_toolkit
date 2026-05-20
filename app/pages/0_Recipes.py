"""Recipes — goal-first gallery of "what do you want to do?" → which workflow + preset.

Solves the "I see 4 tools but don't know which one I need" problem. Pick a
goal, see what it does + what settings to use, click through to the
matching workflow page with the right sliders pre-positioned.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

st.set_page_config(page_title="Recipes • T2I-Interp", layout="wide")

st.title("Recipes — pick a goal, get the right tool")

st.markdown(
    "Each card below is a concrete *objective* — something a researcher or "
    "engineer might want to do with a T2I model. The bullet underneath tells "
    "you which workflow this maps to + the rough config to use. Click the "
    "button to jump to that workflow's page with the right settings already "
    "filled in."
)

st.markdown(
    "**Don't see your use case?** The four workflows below the recipes cover the "
    "general patterns. Browse the sidebar pages directly if you want full control."
)

st.divider()


@dataclass
class Recipe:
    title: str
    objective: str  # "I want to…"
    description: str  # What this does, in 2-3 sentences
    workflow: str  # "Localisation" | "Steering" | "Stitching" | "SAE"
    settings: list[tuple[str, str]]  # [(label, value), …] shown as a config block
    page_path: str  # streamlit relative page path for st.switch_page
    preset_session_key: str | None = None  # if set, write True under this key before nav


# ── The actual catalogue ─────────────────────────────────────────────────────
RECIPES = [
    # —— Steering ——————————————————————————————————————————————————————————
    Recipe(
        title="Add an attribute to a portrait (the paper's headline)",
        objective="I want to make my generated portraits **wear spectacles**, "
        "without retraining the model.",
        description=(
            "Trains a tiny LoReFT adapter (a few thousand parameters) on "
            "paired prompts (with-spectacles vs without). At inference, "
            "injects the learned direction across UNet cross-attention "
            "blocks. Reproduces Figure 2 of the paper."
        ),
        workflow="Steering",
        settings=[
            ("Method", "loreft"),
            ("Model preset", "sdxl_turbo"),
            ("Alpha", "10.0"),
            ("Training samples", "200"),
            ("Training steps", "50"),
        ],
        page_path="pages/2_Steering.py",
        preset_session_key="steer_preset",  # the page already reads "fig2"/"smoke"
    ),
    Recipe(
        title="Shift generations toward a specific demographic",
        objective='I want "photo of a man" to lean toward **Black men** '
        "(or any other demographic group) without changing the model.",
        description=(
            "Uses CAA: compute the difference between average activations "
            "for `photo of a Black man` and `photo of a man`, then add that "
            "direction to the base model at generation time. Reproduces "
            "Figure 3 of the paper."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Model preset", "sd15"),
            ("Alpha", "8.0"),
            ("Positive prompts", "photo of a Black man, photo of a Black person"),
            ("Negative prompts", "photo of a man, photo of a person"),
        ],
        page_path="pages/2_Steering.py",
    ),
    Recipe(
        title="Suppress / erase a concept from your generations",
        objective="I want to make sure my model **doesn't** generate cigarettes / "
        "guns / NSFW content, even when the prompt is ambiguous.",
        description=(
            "Train a steering direction for the unwanted concept (positive "
            "= has-concept, negative = doesn't), then inject with a "
            "**negative alpha** at inference. This subtracts the direction "
            "instead of adding it."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Alpha", "-5.0 to -15.0  (negative!)"),
            ("Training samples", "100-200"),
        ],
        page_path="pages/2_Steering.py",
    ),
    # —— Localisation ——————————————————————————————————————————————————————
    Recipe(
        title="Find where a concept lives in the UNet",
        objective="I want to know **which attention heads** are responsible for "
        'a concept (e.g. "unicorn-ness", "redness", "face structure") so I '
        "can target them surgically.",
        description=(
            "Sweeps through all cross-attention heads in a chosen layer "
            "(or across down/mid/up blocks). For each head, generates the "
            "prompt with that head's contribution scaled to 0. Heads whose "
            "ablation breaks the concept are the ones carrying it."
        ),
        workflow="Localisation",
        settings=[
            ("Prompt", "your concept (e.g. 'a unicorn')"),
            ("Target heads", "[0..7]  (sweep all)"),
            ("Scale factor", "0.0  (zero-ablate)"),
        ],
        page_path="pages/1_Localisation.py",
    ),
    Recipe(
        title="Test a specific head you suspect carries a concept",
        objective="I have a hypothesis that head 3 of layer X binds colour "
        "words. I want to verify that.",
        description=(
            "Run two generations: one with the head at `factor=1.0` "
            "(no change, baseline) and one with `factor=0.0` (ablated). "
            "If the colour-related behaviour changes between the two, "
            "that head is doing it."
        ),
        workflow="Localisation",
        settings=[
            ("Target head", "your suspect head"),
            ("Scale factor", "0.0  (compare against the baseline)"),
        ],
        page_path="pages/1_Localisation.py",
    ),
    # —— SAE ———————————————————————————————————————————————————————————
    Recipe(
        title="Discover what features activate for my prompt",
        objective="I generated an apple and want to know **which sparse "
        "features** the model was using — what concepts the model is "
        "internally combining to produce that image.",
        description=(
            "Captures the SAE latents during one forward pass on your "
            "prompt, then surfaces the top-K most-active features. You "
            "get the feature indices + the modulation grid showing what "
            "each one encodes."
        ),
        workflow="SAE",
        settings=[
            ("Prompt", "your prompt"),
            ("Top features to modulate", "4-6"),
            ("Strengths", "[-5, +5]"),
        ],
        page_path="pages/4_SAE.py",
    ),
    Recipe(
        title="Amplify or suppress a specific visual feature",
        objective="I found feature 1338 makes apples shinier. I want to "
        "amplify it consistently across all my generations.",
        description=(
            "Once you've discovered a feature index that controls a "
            "concept you care about, fix the modulation strength on that "
            "feature and generate at scale. The feature acts as a "
            "permanent additive bias on the UNet's representation."
        ),
        workflow="SAE",
        settings=[
            ("Modulation grid", "1 row (the feature you found), 1 column (your strength)"),
            ("Strength", "tune by trial; ±5 is a strong push"),
        ],
        page_path="pages/4_SAE.py",
    ),
    # —— Stitching ————————————————————————————————————————————————————————
    Recipe(
        title="Transfer a behaviour between two models",
        objective="I have a fine-tuned SD 1.5 with a behaviour I like. Can "
        "I get that behaviour into a different model **without re-training**?",
        description=(
            "Train a small MLP mapper between the two models' activation "
            "spaces using paired forward passes. At inference, the source "
            "model's activations flow through the mapper into the target "
            "model's pipeline. Used as the §4 case study in the paper."
        ),
        workflow="Stitching",
        settings=[
            ("Mode", "train"),
            ("layer_a / layer_b", "matching cross-attn blocks"),
            ("Hidden dim", "256-1024"),
        ],
        page_path="pages/3_Stitching.py",
    ),
    Recipe(
        title="Check whether two layers encode comparable information",
        objective="Are the model's early UNet layers really redundant with "
        "the text encoder's final hidden state? Can I prove it?",
        description=(
            "If a small MLP mapper can be trained to translate layer A → "
            "layer B and the resulting stitched image stays coherent → "
            "the two layers carry comparable information. If the stitched "
            "image is noise → they don't, and the model is using them "
            "for distinct purposes."
        ),
        workflow="Stitching",
        settings=[
            ("Mode", "train"),
            (
                "Hidden dim",
                "small (128-256) — bigger means the mapper can paper over real incompatibility",
            ),
        ],
        page_path="pages/3_Stitching.py",
    ),
]


# ── Render ────────────────────────────────────────────────────────────────────

# Group by workflow for navigation
WORKFLOWS = ["Steering", "Localisation", "SAE", "Stitching"]

for wf in WORKFLOWS:
    wf_recipes = [r for r in RECIPES if r.workflow == wf]
    if not wf_recipes:
        continue
    st.subheader(f"{wf} recipes")
    for r in wf_recipes:
        with st.container(border=True):
            c_text, c_action = st.columns([4, 1])
            with c_text:
                st.markdown(f"#### {r.title}")
                st.markdown(f"**Goal.** {r.objective}")
                st.markdown(f"**What it does.** {r.description}")
                st.markdown("**Suggested config:**")
                cfg_lines = "\n".join(f"  - **{k}**: `{v}`" for k, v in r.settings)
                st.markdown(cfg_lines)
            with c_action:
                st.markdown("")  # spacer
                st.markdown("")
                if st.button(f"Open {r.workflow}", key=f"go_{r.title[:30]}"):
                    if r.preset_session_key:
                        # Steering page reads st.session_state.steer_preset ∈ {"fig2", "smoke"}
                        st.session_state[r.preset_session_key] = "fig2"
                    st.switch_page(r.page_path)
    st.markdown("")  # spacing between workflow groups

st.divider()

st.caption(
    "These recipes are illustrative — the workflow pages give you full "
    "control over every knob. Recipes just save you from staring at "
    "an empty sidebar wondering what to set first."
)
