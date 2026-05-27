"""Recipes — goal-first router. Three modes:

1. **Goal Wizard** — structured Q&A, no API needed (default tab)
2. **Describe your goal** — free-form text, analyzed by Claude
3. **Browse all recipes** — the card gallery

Solves "I see 4 tools but don't know which one I need" by always letting
the user start from a *goal* rather than a tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.lib import RecipeMatch, analyze_goal
from app.lib import is_available as llm_is_available

st.set_page_config(page_title="Recipes • T2I-Interp", layout="wide")

st.title("Recipes — pick a goal, get the right tool")

st.markdown(
    "Three ways to find the right workflow for your goal. Default tab is the "
    "structured wizard (no setup); use the *describe in your own words* tab "
    "for novel goals not covered by the catalogue (uses Claude — see footer)."
)

tab_wizard, tab_llm, tab_gallery = st.tabs(
    ["Goal wizard", "Describe your goal (Claude)", "Browse all recipes"]
)

# ── Shared recipe catalogue (used by both wizard + gallery) ──────────────────


@dataclass
class Recipe:
    title: str
    objective: str
    description: str
    workflow: str
    settings: list[tuple[str, str]]
    page_path: str
    preset_session_key: str | None = None


RECIPES = [
    Recipe(
        title="Add an attribute to a portrait (paper headline)",
        objective="I want my generated portraits to **wear spectacles**, without retraining.",
        description=(
            "Trains a tiny LoReFT adapter (a few thousand parameters) on paired prompts "
            "(with-spectacles vs without). At inference, injects the learned direction "
            "across UNet cross-attention blocks. Reproduces Figure 2 of the paper."
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
        preset_session_key="steer_preset",
    ),
    Recipe(
        title="Shift generations toward a specific demographic",
        objective='I want "photo of a man" to lean toward **Black men** without changing the model.',
        description=(
            "Uses CAA: compute the difference between average activations for "
            "`photo of a Black man` and `photo of a man`, then add that direction "
            "to the base model at generation time. Reproduces Figure 3."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Model preset", "sd15"),
            ("Alpha", "8.0"),
            ("Positive prompts", "photo of a Black man"),
            ("Negative prompts", "photo of a man"),
        ],
        page_path="pages/2_Steering.py",
    ),
    Recipe(
        title="Suppress / erase an unwanted concept",
        objective="I want my model to **not** generate cigarettes / weapons / NSFW content.",
        description=(
            "Train a steering direction for the unwanted concept (positive = has-concept), "
            "then inject with a **negative alpha** at inference. Subtracts the direction "
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
    Recipe(
        title="Find where a concept lives in the UNet",
        objective='Which **attention heads** are responsible for "unicorn-ness" / "redness" / face structure?',
        description=(
            "Sweeps cross-attention heads in a chosen layer. For each head, generates the "
            "prompt with the head scaled to 0. Heads whose ablation breaks the concept "
            "are the ones carrying it."
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
        objective="I have a hypothesis that head 3 of layer X binds colour words. Verify it.",
        description=(
            "Run two generations: one with the head at `factor=1.0` (baseline) and one at "
            "`factor=0.0` (ablated). If the colour-related behaviour changes between the two, "
            "that head is doing it."
        ),
        workflow="Localisation",
        settings=[
            ("Target head", "your suspect head"),
            ("Scale factor", "0.0  (compare against the baseline)"),
        ],
        page_path="pages/1_Localisation.py",
    ),
    Recipe(
        title="Discover what features activate for my prompt",
        objective="I want to know **which sparse features** the model uses for my prompt.",
        description=(
            "Captures the SAE latents during one forward pass, then surfaces the top-K "
            "most-active features. You get the feature indices + the modulation grid "
            "showing what each encodes."
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
        objective="I found feature 1338 makes apples shinier. Amplify it consistently.",
        description=(
            "Once you've found a feature index that controls a concept you care about, "
            "fix the modulation strength on that feature and generate at scale. The "
            "feature acts as a permanent additive bias."
        ),
        workflow="SAE",
        settings=[
            ("Modulation grid", "1 row (your feature), 1 column (your strength)"),
            ("Strength", "tune by trial; ±5 is a strong push"),
        ],
        page_path="pages/4_SAE.py",
    ),
    Recipe(
        title="Transfer a behaviour between two models",
        objective="I have a fine-tuned SD 1.5 with a behaviour I like. Get it into a different model.",
        description=(
            "Train a small MLP mapper between the two models' activation spaces. At "
            "inference, the source model's activations flow through the mapper into the "
            "target. Paper §4 case study."
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
        objective="Are the model's early UNet layers redundant with the text encoder? Prove it.",
        description=(
            "If a small mapper can be trained to translate layer A → layer B and the "
            "resulting stitched image stays coherent → the two layers carry comparable "
            "information. If noise → they don't."
        ),
        workflow="Stitching",
        settings=[
            ("Mode", "train"),
            ("Hidden dim", "small (128-256) — bigger = mapper can paper over real incompatibility"),
        ],
        page_path="pages/3_Stitching.py",
    ),
]


def _render_recipe_card(r: Recipe, key_suffix: str = "") -> None:
    """Render one recipe inside an st.container — used by all three tabs."""
    with st.container(border=True):
        c_text, c_action = st.columns([4, 1])
        with c_text:
            st.markdown(f"#### {r.title}")
            st.markdown(f"**Goal.** {r.objective}")
            st.markdown(f"**What it does.** {r.description}")
            st.markdown("**Suggested config:**")
            for k, v in r.settings:
                st.markdown(f"  - **{k}**: `{v}`")
        with c_action:
            st.markdown("")
            st.markdown("")
            if st.button(f"Open {r.workflow}", key=f"go_{r.title[:30]}{key_suffix}"):
                if r.preset_session_key:
                    st.session_state[r.preset_session_key] = "fig2"
                st.switch_page(r.page_path)


# ── Tab 1: Goal wizard (structured Q&A) ──────────────────────────────────────


with tab_wizard:
    st.markdown(
        "Answer two quick questions and we'll route you to the right workflow "
        "with a starting config."
    )

    intent = st.radio(
        "**What kind of thing are you trying to do?**",
        [
            "Change *what* gets generated (add/remove/shift a concept)",
            "Find *where* in the model a concept lives",
            "Understand *what* concepts the model uses internally",
            "Move behavior *between* two different models",
        ],
        index=0,
        key="wizard_intent",
    )

    st.markdown("---")

    if intent.startswith("Change"):
        action = st.radio(
            "**What kind of change?**",
            [
                "Add an attribute (spectacles, beard, a style…)",
                "Shift a demographic (gender, race, age…)",
                "Suppress / erase a concept (NSFW, bias, a style I don't want)",
            ],
            index=0,
            key="wizard_action",
        )
        if action.startswith("Add"):
            matching = [r for r in RECIPES if "Add an attribute" in r.title]
        elif action.startswith("Shift"):
            matching = [r for r in RECIPES if "demographic" in r.title]
        else:
            matching = [r for r in RECIPES if "Suppress" in r.title]

    elif intent.startswith("Find"):
        sub = st.radio(
            "**Are you exploring or testing a hypothesis?**",
            [
                "Exploring — I don't yet know which head matters",
                "Testing — I suspect a specific head/layer",
            ],
            index=0,
            key="wizard_loc_action",
        )
        matching = (
            [r for r in RECIPES if "Find where" in r.title]
            if sub.startswith("Exploring")
            else [r for r in RECIPES if "Test a specific" in r.title]
        )

    elif intent.startswith("Understand"):
        sub = st.radio(
            "**Are you discovering features or amplifying a known one?**",
            [
                "Discovering — what features even activate for my prompt?",
                "Amplifying — I already know a feature index I want to push on",
            ],
            index=0,
            key="wizard_sae_action",
        )
        matching = (
            [r for r in RECIPES if "Discover what features" in r.title]
            if sub.startswith("Discovering")
            else [r for r in RECIPES if "Amplify" in r.title]
        )

    else:  # "Move behavior"
        sub = st.radio(
            "**Why?**",
            [
                "Transfer a behavior from a fine-tuned model into a base model",
                "Diagnose whether two layers carry comparable information",
            ],
            index=0,
            key="wizard_stitch_action",
        )
        matching = (
            [r for r in RECIPES if "Transfer a behaviour" in r.title]
            if sub.startswith("Transfer")
            else [r for r in RECIPES if "comparable information" in r.title]
        )

    st.markdown("---")
    st.markdown(f"### Recommended workflow: **{matching[0].workflow}**")
    for r in matching:
        _render_recipe_card(r, key_suffix="_wiz")


# ── Tab 2: Describe your goal (LLM-routed) ───────────────────────────────────


with tab_llm:
    if not llm_is_available():
        st.warning(
            "**Claude routing requires `ANTHROPIC_API_KEY`.** Set it in your shell "
            "(`export ANTHROPIC_API_KEY=sk-ant-…`) and restart `make app`. "
            "You can still use the **Goal wizard** tab without an API key.",
            icon="⚠️",
        )
    else:
        st.markdown(
            "Describe your goal in your own words. Claude (haiku-4.5) reads the "
            "available workflows + recipes and picks the best match for you, "
            "with reasoning + a starting config."
        )

        goal_text = st.text_area(
            "Your goal",
            placeholder=(
                'e.g. "Make my generations look more painterly without retraining" '
                'or "Reduce gender stereotyping for occupation prompts"'
            ),
            height=100,
            key="llm_goal",
        )

        if st.button("Ask Claude →", type="primary", disabled=not goal_text.strip()):
            with st.spinner("Asking Claude…"):
                try:
                    match: RecipeMatch = analyze_goal(goal_text.strip())
                    st.session_state["last_llm_match"] = match
                except Exception as e:
                    st.error(f"Claude call failed: {type(e).__name__}: {e}")
                    st.session_state.pop("last_llm_match", None)

        match: RecipeMatch | None = st.session_state.get("last_llm_match")
        if match is not None:
            st.markdown("---")
            st.markdown(f"### Claude's recommendation: **{match.workflow}**")

            with st.container(border=True):
                st.markdown(f"**Why this workflow.** {match.reasoning}")
                if match.caveat:
                    st.warning(match.caveat, icon="⚠️")
                st.markdown("**Suggested starting config:**")
                for label, value in match.suggested_config:
                    st.markdown(f"  - **{label}**: `{value}`")

                if st.button(f"Open {match.workflow} →", type="primary", key="llm_open_btn"):
                    st.switch_page(match.page_path)


# ── Tab 3: Browse all recipes (the existing gallery) ─────────────────────────


with tab_gallery:
    st.markdown(
        "The full catalogue — every concrete objective the toolkit has a recipe for. "
        "Grouped by workflow."
    )

    for wf in ["Steering", "Localisation", "SAE", "Stitching"]:
        wf_recipes = [r for r in RECIPES if r.workflow == wf]
        if not wf_recipes:
            continue
        st.subheader(f"{wf} recipes")
        for r in wf_recipes:
            _render_recipe_card(r, key_suffix="_gal")
        st.markdown("")

st.divider()

st.caption(
    "Wizard + browse modes are 100% local — no network, no LLM call. "
    "The *describe your goal* tab uses Anthropic's Claude API; set "
    "`ANTHROPIC_API_KEY` to enable it. Recipe page paths and presets stay "
    "in sync with the workflow pages via st.session_state."
)
