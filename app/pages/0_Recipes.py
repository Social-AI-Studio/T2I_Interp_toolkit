"""Recipes — goal-first router. Three modes:

1. **Goal Wizard** — structured Q&A, no API needed (default tab)
2. **Describe your goal** — free-form text, analyzed by Claude
3. **Browse all recipes** — the card gallery

Each recipe has both human-readable `settings` (for display) and a
machine-readable `fields` dict (for pre-filling workflow page widgets).
Clicking *Open* drops a payload into `st.session_state["recipe_payload"]`
which the workflow page consumes on next render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    workflow: str  # "Steering" | "Localisation" | "Stitching" | "SAE"
    settings: list[tuple[str, str]]  # display strings (key, value)
    page_path: str
    # Machine-readable defaults — keys must match what the target workflow
    # page seeds into st.session_state. See the `_apply_payload` helper at
    # the top of each workflow page for the accepted keys.
    fields: dict[str, Any] = field(default_factory=dict)
    # Text seeded into the workflow page's `goal` field on arrival.
    goal_text: str = ""


# Standard prompt blocks reused across recipes.
_SPECTACLES_PROMPTS = "A photo of Jack Sparrow\nA photo of Simba\nA photo of Mickey Mouse"

# Inline pair catalogues (pos | neg, one per line). Used by recipes whose target
# concept doesn't have a published HuggingFace dataset in the CAA/LoReFT schema —
# run_steer.py builds an in-memory Dataset from these pairs.
_DEMOGRAPHIC_PAIRS = (
    "photo of a Black man | photo of a man\n"
    "portrait of a Black man | portrait of a man\n"
    "photograph of a Black father | photograph of a father\n"
    "headshot of a Black man | headshot of a man\n"
    "photo of a Black businessman | photo of a businessman\n"
    "portrait of a Black athlete | portrait of an athlete\n"
    "photo of a Black doctor | photo of a doctor\n"
    "photo of a Black teacher | photo of a teacher"
)
_CIGARETTE_PAIRS = (
    "a man holding a cigarette | a man holding a pen\n"
    "a person smoking | a person resting\n"
    "a man smoking outside | a man standing outside\n"
    "a woman with a cigarette | a woman with a coffee cup\n"
    "a person lighting a cigarette | a person lighting a candle\n"
    "close-up of someone smoking | close-up of someone yawning\n"
    "a hand holding a cigarette | a hand holding a phone\n"
    "a character with a cigarette | a character with a coffee"
)
_PAINTERLY_PAIRS = (
    "a painterly portrait of a man | a photo of a man\n"
    "an impressionist painting of a woman | a photo of a woman\n"
    "a painterly landscape with mountains | a photo of a landscape with mountains\n"
    "an oil painting of a still life with apples | a photo of a still life with apples\n"
    "a painterly portrait of a child | a photo of a child\n"
    "an impressionist garden scene | a photo of a garden scene\n"
    "a painterly seascape at sunset | a photo of a seascape at sunset\n"
    "a painterly street market | a photo of a street market"
)
_OCCUPATION_PAIRS = (
    "a woman doctor | a doctor\n"
    "a woman CEO | a CEO\n"
    "a woman engineer | an engineer\n"
    "a woman scientist | a scientist\n"
    "a woman pilot | a pilot\n"
    "a woman programmer | a programmer\n"
    "a woman surgeon | a surgeon\n"
    "a woman professor | a professor"
)


RECIPES: list[Recipe] = [
    # ── Steering ─────────────────────────────────────────────────────────────
    Recipe(
        title="Add spectacles to character portraits (paper Fig 2)",
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
        fields={
            "method": "loreft",
            "model_preset": "sdxl_turbo",
            "prompts": _SPECTACLES_PROMPTS,
            "alpha": 10.0,
            "max_samples": 200,
            "train_steps": 50,
        },
        goal_text="Add spectacles to character portraits (paper Fig 2 reproduction).",
    ),
    Recipe(
        title="Shift generations toward a specific demographic (paper Fig 3)",
        objective='I want "photo of a man" to lean toward **Black men** without changing the model.',
        description=(
            "Uses CAA on **8 inline prompt pairs** (Black-versioned vs neutral). "
            "Computes the mean activation difference between the two groups, "
            "then adds that direction to the base model at generation time. "
            "Inspired by Figure 3 of the paper."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Model preset", "sd15"),
            ("Alpha", "8.0"),
            ("Inline pairs", "8 demographic pairs (pre-filled)"),
            ("Inference prompts", "photo of a man\\nphoto of a person"),
        ],
        page_path="pages/2_Steering.py",
        fields={
            "method": "caa",
            "model_preset": "sd15",
            "prompts": "photo of a man\nphoto of a person",
            "alpha": 8.0,
            "max_samples": 100,
            "train_steps": 50,
            "inline_pairs": _DEMOGRAPHIC_PAIRS,
        },
        goal_text="Shift 'photo of a man' generations toward Black men (paper Fig 3, inline pairs).",
    ),
    Recipe(
        title="Suppress / erase an unwanted concept (cigarettes)",
        objective="I want my model to **not** generate cigarettes / smoking content.",
        description=(
            "Trains a CAA direction for the unwanted concept (positive = has-cigarette, "
            "negative = a benign substitute), then injects with a **negative alpha** "
            "at inference. Subtracts the direction instead of adding it. Swap the "
            "inline pairs to suppress any other concept (weapons, NSFW, …)."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Alpha", "-10.0 (negative!)"),
            ("Inline pairs", "8 cigarette pairs (pre-filled)"),
        ],
        page_path="pages/2_Steering.py",
        fields={
            "method": "caa",
            "model_preset": "sd15",
            "prompts": "a man holding a cigarette\na person smoking",
            "alpha": -10.0,
            "max_samples": 100,
            "train_steps": 50,
            "inline_pairs": _CIGARETTE_PAIRS,
        },
        goal_text="Suppress cigarettes via negative-alpha CAA (inline pairs).",
    ),
    Recipe(
        title="Apply a painterly art style without LoRA training",
        objective="I want my generations to look **more painterly / impressionist** without a LoRA.",
        description=(
            "Trains a LoReFT adapter on 8 inline (base, teacher) pairs where the "
            "teacher prompts add 'painterly'/'impressionist' modifiers. At "
            "inference the adapter biases the model toward the teacher style — "
            "faster and lighter than a LoRA, fully composable with prompt."
        ),
        workflow="Steering",
        settings=[
            ("Method", "loreft"),
            ("Model preset", "sdxl_turbo"),
            ("Alpha", "12.0"),
            ("Inline pairs", "8 painterly pairs (pre-filled)"),
        ],
        page_path="pages/2_Steering.py",
        fields={
            "method": "loreft",
            "model_preset": "sdxl_turbo",
            "prompts": "a photo of a person\na photo of a landscape",
            "alpha": 12.0,
            "max_samples": 100,
            "train_steps": 100,
            "inline_pairs": _PAINTERLY_PAIRS,
        },
        goal_text="Make generations look painterly / impressionist (LoReFT, inline pairs).",
    ),
    Recipe(
        title="Reduce gender stereotyping for occupation prompts",
        objective='I want "a doctor" / "a CEO" prompts to **stop defaulting to one gender**.',
        description=(
            "Trains a CAA direction on 8 inline pairs (`a woman <occupation>` "
            "positive vs `a <occupation>` negative). At inference, applies a "
            "small positive alpha to tilt the model toward the under-represented "
            "direction. Sweep alpha to find the balance."
        ),
        workflow="Steering",
        settings=[
            ("Method", "caa"),
            ("Model preset", "sd15"),
            ("Alpha", "5.0"),
            ("Inline pairs", "8 occupation pairs (pre-filled)"),
        ],
        page_path="pages/2_Steering.py",
        fields={
            "method": "caa",
            "model_preset": "sd15",
            "prompts": "a doctor\na CEO\nan engineer",
            "alpha": 5.0,
            "max_samples": 100,
            "train_steps": 50,
            "inline_pairs": _OCCUPATION_PAIRS,
        },
        goal_text="Reduce gender stereotyping for occupation prompts (CAA, inline pairs).",
    ),
    # ── Localisation ─────────────────────────────────────────────────────────
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
        fields={
            "prompt": "a unicorn in a forest",
            "target_layer": "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
            "target_head": 0,
            "factor": 0.0,
            "n_steps": 15,
            "seed": 42,
            "model_preset": "sd15",
        },
        goal_text="Sweep all attention heads to find which carry 'unicorn-ness'.",
    ),
    Recipe(
        title="Test a specific head you suspect carries a concept",
        objective="I have a hypothesis that head 3 of mid_block binds **colour words**. Verify it.",
        description=(
            "Run two generations: one with the head at `factor=1.0` (baseline) and one at "
            "`factor=0.0` (ablated). If the colour-related behaviour changes between the two, "
            "that head is doing it."
        ),
        workflow="Localisation",
        settings=[
            ("Target head", "3"),
            ("Target layer", "mid_block.attentions.0..."),
            ("Scale factor", "0.0 (compare to baseline)"),
        ],
        page_path="pages/1_Localisation.py",
        fields={
            "prompt": "a red apple on a wooden table",
            "target_layer": "mid_block_attentions_0_transformer_blocks_0_attn2_out",
            "target_head": 3,
            "factor": 0.0,
            "n_steps": 20,
            "seed": 42,
            "model_preset": "sd15",
        },
        goal_text="Test whether head 3 of mid_block binds colour words.",
    ),
    Recipe(
        title="Compare early vs late UNet layers",
        objective="Are early UNet layers about **composition** and late layers about **texture**?",
        description=(
            "Pick a single head, ablate it in an early `down_blocks` layer for one run, "
            "and a late `up_blocks` layer for another. Compare what changes between them — "
            "the type of change reveals what each layer's responsibilities are."
        ),
        workflow="Localisation",
        settings=[
            ("Prompt", "a busy city street at dusk"),
            ("Target layer", "(early: down_blocks_1, late: up_blocks_2)"),
            ("Scale factor", "0.0"),
        ],
        page_path="pages/1_Localisation.py",
        fields={
            "prompt": "a busy city street at dusk",
            "target_layer": "down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
            "target_head": 0,
            "factor": 0.0,
            "n_steps": 20,
            "seed": 42,
            "model_preset": "sd15",
        },
        goal_text="Compare an early UNet layer ablation against a late one for the same prompt.",
    ),
    # ── SAE ──────────────────────────────────────────────────────────────────
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
        fields={
            "prompt": "a red apple on a wooden table",
            "strength_lo": -5.0,
            "strength_hi": 5.0,
            "n_features_to_plot": 4,
            "n_top_features": 10,
            "model_preset": "sdxl_turbo",
        },
        goal_text="Discover the top sparse features that activate for my prompt.",
    ),
    Recipe(
        title="Find a feature that controls a specific visual property",
        objective="Which feature makes images **shinier** / **more textured** / **more saturated**?",
        description=(
            "Run feature discovery for a prompt where your target property is prominent "
            "(e.g. 'a glossy red apple' for shininess). Each row in the output grid shows "
            "one feature scaled across negative→positive. Watch the columns for the row "
            "where the property changes consistently."
        ),
        workflow="SAE",
        settings=[
            ("Prompt", "a glossy red apple"),
            ("Strengths", "[-8, +8] (wide sweep)"),
            ("Top features", "6"),
        ],
        page_path="pages/4_SAE.py",
        fields={
            "prompt": "a glossy red apple",
            "strength_lo": -8.0,
            "strength_hi": 8.0,
            "n_features_to_plot": 6,
            "n_top_features": 15,
            "model_preset": "sdxl_turbo",
        },
        goal_text="Find a sparse feature that controls a specific visual property (e.g. shininess).",
    ),
    Recipe(
        title="Amplify or suppress a known feature index",
        objective="I found feature 1338 makes apples shinier. Amplify it consistently.",
        description=(
            "Once you've found a feature index that controls a concept you care about, "
            "fix the modulation strength on that feature and generate at scale. The "
            "feature acts as a permanent additive bias."
        ),
        workflow="SAE",
        settings=[
            ("Prompt", "your generation prompt"),
            ("Strengths", "[+5] (single fixed amplification)"),
            ("Top features to modulate", "1"),
        ],
        page_path="pages/4_SAE.py",
        fields={
            "prompt": "a red apple on a wooden table",
            "strength_lo": 0.0,
            "strength_hi": 5.0,
            "n_features_to_plot": 1,
            "n_top_features": 5,
            "model_preset": "sdxl_turbo",
        },
        goal_text="Amplify a known sparse feature index that controls a concept I care about.",
    ),
    # ── Stitching ────────────────────────────────────────────────────────────
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
            ("Hidden dim", "512"),
            ("Train samples", "100"),
            ("Mapper steps", "200"),
        ],
        page_path="pages/3_Stitching.py",
        fields={
            "prompts": "a photo of a person\na photo of a landscape\na photo of a still life",
            "hidden_dim": 512,
            "max_samples": 100,
            "num_steps": 200,
            "num_inference_steps": 15,
            "model_preset": "sd15",
        },
        goal_text="Transfer a fine-tuned model's behaviour into a different model via a mapper.",
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
            ("Hidden dim", "256  (small — bigger = mapper can paper over real incompat)"),
            ("Mapper steps", "100"),
        ],
        page_path="pages/3_Stitching.py",
        fields={
            "prompts": "a photo of a person",
            "hidden_dim": 256,
            "max_samples": 50,
            "num_steps": 100,
            "num_inference_steps": 15,
            "model_preset": "sd15",
        },
        goal_text="Diagnose whether two layers encode comparable information (small mapper test).",
    ),
]


def _apply_recipe(r: Recipe) -> None:
    """Store recipe payload in session_state. Workflow page reads + pops it."""
    st.session_state["recipe_payload"] = {
        "workflow": r.workflow,
        "goal": r.goal_text or r.objective,
        "fields": r.fields,
    }


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
            if r.fields:
                st.caption(
                    "Clicking *Open* pre-fills these settings on the workflow page. "
                    "You can still adjust anything before pressing Run."
                )
        with c_action:
            st.markdown("")
            st.markdown("")
            if st.button(f"Open {r.workflow}", key=f"go_{r.title[:30]}{key_suffix}"):
                _apply_recipe(r)
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
                "Apply an art style (painterly, impressionist…)",
            ],
            index=0,
            key="wizard_action",
        )
        if action.startswith("Add"):
            matching = [r for r in RECIPES if "spectacles" in r.title.lower()]
        elif action.startswith("Shift"):
            matching = [r for r in RECIPES if "demographic" in r.title.lower()]
        elif action.startswith("Suppress"):
            matching = [r for r in RECIPES if "suppress" in r.title.lower()]
        else:
            matching = [r for r in RECIPES if "painterly" in r.title.lower()]

    elif intent.startswith("Find"):
        sub = st.radio(
            "**Are you exploring or testing a hypothesis?**",
            [
                "Exploring — I don't yet know which head matters",
                "Testing — I suspect a specific head/layer",
                "Mapping — compare what early vs late layers do",
            ],
            index=0,
            key="wizard_loc_action",
        )
        if sub.startswith("Exploring"):
            matching = [r for r in RECIPES if "Find where" in r.title]
        elif sub.startswith("Testing"):
            matching = [r for r in RECIPES if "Test a specific" in r.title]
        else:
            matching = [r for r in RECIPES if "early vs late" in r.title]

    elif intent.startswith("Understand"):
        sub = st.radio(
            "**Are you discovering features or amplifying a known one?**",
            [
                "Discovering — what features even activate for my prompt?",
                "Hunting — find the feature that controls a property I care about",
                "Amplifying — I already know a feature index I want to push on",
            ],
            index=0,
            key="wizard_sae_action",
        )
        if sub.startswith("Discovering"):
            matching = [r for r in RECIPES if "Discover what features" in r.title]
        elif sub.startswith("Hunting"):
            matching = [r for r in RECIPES if "controls a specific visual property" in r.title]
        else:
            matching = [r for r in RECIPES if "Amplify" in r.title]

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
    if matching:
        st.markdown(f"### Recommended workflow: **{matching[0].workflow}**")
        for r in matching:
            _render_recipe_card(r, key_suffix="_wiz")
    else:
        st.info("No recipe matches that combination yet — try Browse all recipes.")


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
            "with reasoning + a starting config. The *Open* button will pre-fill "
            "the workflow page with your goal text — adjust the rest before running."
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
                    st.session_state["last_llm_user_goal"] = goal_text.strip()
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
                st.caption(
                    "Claude's suggested config is shown above for reference. "
                    "Clicking *Open* will land you on the workflow page with your "
                    "goal text pre-filled — set the sliders/inputs to match the "
                    "suggestion (or your own values) before pressing Run."
                )

                if st.button(f"Open {match.workflow} →", type="primary", key="llm_open_btn"):
                    st.session_state["recipe_payload"] = {
                        "workflow": match.workflow,
                        "goal": st.session_state.get("last_llm_user_goal", ""),
                        "fields": {},  # Claude's suggestions are free-form strings,
                        # so we don't try to parse them into widget values — just
                        # show them in the result panel via the goal text.
                    }
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
