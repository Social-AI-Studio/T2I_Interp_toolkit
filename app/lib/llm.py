"""LLM-based goal → recipe matcher for the Streamlit playground.

Uses the Anthropic SDK with structured tool-use to translate a free-form
user goal ("I want my generations to look more colorful and painterly")
into a concrete workflow + suggested config.

Requires `ANTHROPIC_API_KEY` in the environment. Without it,
`is_available()` returns False and pages should fall back to the
structured wizard.

The system prompt is kept stable so Anthropic's prompt cache hits across
calls — only the user goal changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.lib.workflows import WORKFLOW_TO_PAGE

# Hard-coded recipe catalogue. Keep this in sync with app/pages/0_Recipes.py.
WORKFLOW_DESCRIPTIONS = """
The toolkit has four workflows:

1. **Steering** (Concept direction injection)
   Trains a steering vector (CAA), classifier (K-Steer), or low-rank
   adapter (LoReFT) from paired prompts, then injects it during generation.
   Use this when the user wants to:
   - Add an attribute to existing prompts (e.g. spectacles, beard, vintage style)
   - Remove / suppress an attribute (use negative alpha)
   - Shift outputs toward a demographic (e.g. more diverse, more Black, more female)
   - Apply a style (painterly, photorealistic)
   - De-bias generations
   Key knobs: method (caa/ksteer/loreft), alpha (strength), training prompts.

2. **Localisation** (Find where concepts live in the model)
   Scales individual cross-attention heads in the UNet. Use this when the
   user wants to:
   - Find which heads / layers carry a specific concept
   - Test a causal hypothesis about a head
   - Build a "concept map" of the model
   - Verify their understanding of a model's internals
   Key knobs: target_layer, target_heads, factor (0=ablate, 1=no change, >1 amplify).

3. **Stitching** (Cross-layer / cross-model activation mapping)
   Trains an MLP that translates activations between two layers or two models.
   Use this when the user wants to:
   - Transfer behavior between two models without retraining
   - Test whether two layers encode comparable information
   - Move a steering direction across models
   Key knobs: layer_a, layer_b, model_key_b, hidden_dim, num_steps.

4. **SAE** (Sparse Autoencoder feature discovery)
   Loads pretrained SAEs that decompose dense activations into ~5000 sparse
   features. Use this when the user wants to:
   - Discover what concepts the model represents internally
   - Find a feature that controls a specific visual property
   - Amplify/suppress a specific feature consistently
   - Understand what a particular activation "means"
   Key knobs: prompt, strengths grid, n_features_to_plot.
""".strip()


SYSTEM_PROMPT = f"""
You are a routing assistant for the T2I-Interp toolkit, a Python library for
interpretability of text-to-image diffusion models (Stable Diffusion, SDXL, etc.).

Your job: given a user's free-form goal, decide which of four workflows best
matches their intent, and propose a concrete starting config.

{WORKFLOW_DESCRIPTIONS}

When responding, call the `recommend_recipe` tool exactly once. Be concrete
in `reasoning` (mention the specific knob settings and *why* you chose them
for this goal). Keep the `suggested_config` list to 3–6 key/value pairs —
just the most important knobs for this goal, not every possible parameter.

If the goal is ambiguous, pick the most useful workflow + add a clarifying
question in `reasoning`. If the goal is clearly outside the toolkit's scope
(e.g. "train a new model from scratch"), still pick the closest workflow but
explain the mismatch in `reasoning`.
""".strip()


RECOMMEND_RECIPE_TOOL = {
    "name": "recommend_recipe",
    "description": (
        "Recommend which of the four T2I-Interp workflows the user should "
        "open + a concrete starting config for their goal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "enum": ["Steering", "Localisation", "Stitching", "SAE"],
                "description": "The matching workflow.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Short explanation (2–4 sentences) of why this workflow "
                    "is the right fit for the user's goal. Reference their "
                    "specific goal wording where natural."
                ),
            },
            "suggested_config": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["label", "value"],
                },
                "description": (
                    "3–6 key/value pairs giving a concrete starting "
                    "config (e.g. Method=loreft, Alpha=10, etc.)."
                ),
            },
            "caveat": {
                "type": "string",
                "description": (
                    "Optional. Use if the goal is ambiguous, partially out "
                    "of scope, or if the user should know about a tradeoff. "
                    "Empty string if no caveat."
                ),
            },
        },
        "required": ["workflow", "reasoning", "suggested_config", "caveat"],
    },
}


# NOTE: WORKFLOW_DESCRIPTIONS above is intentionally separate from each
# recipe's UI `description` in 0_Recipes.py — this prose is the LLM's
# routing context (full sentences, all four workflows in one block) while
# recipe `description`s are short card blurbs. WORKFLOW_TO_PAGE is the only
# value that must stay in sync, hence the shared import from
# `app/lib/workflows.py`.


@dataclass(frozen=True)
class RecipeMatch:
    workflow: str
    reasoning: str
    suggested_config: list[tuple[str, str]]
    caveat: str
    page_path: str


def is_available() -> bool:
    """Whether the LLM analyser is configured (i.e. ANTHROPIC_API_KEY set)."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def analyze_goal(goal: str, model: str = "claude-haiku-4-5-20251001") -> RecipeMatch:
    """Send `goal` to Claude and return the recommended recipe.

    The system prompt is cached on Anthropic's side (cache_control breakpoint
    on the workflow-descriptions block), so repeated analyses are fast and
    cheap. `claude-haiku-4-5` is the right speed/cost tradeoff for routing —
    bump to `claude-sonnet-4-6` if you want richer reasoning.

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    """
    if not is_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. "
            "`export ANTHROPIC_API_KEY=sk-ant-…` and restart Streamlit, "
            "or use the structured Goal Wizard instead."
        )

    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Cache the long, stable system prompt — ~5min TTL.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECOMMEND_RECIPE_TOOL],
        tool_choice={"type": "tool", "name": "recommend_recipe"},
        messages=[{"role": "user", "content": f"My goal: {goal}"}],
    )

    # Extract the structured tool use
    for block in resp.content:
        if block.type == "tool_use" and block.name == "recommend_recipe":
            args = block.input
            return RecipeMatch(
                workflow=args["workflow"],
                reasoning=args["reasoning"],
                suggested_config=[(c["label"], c["value"]) for c in args["suggested_config"]],
                caveat=args.get("caveat", ""),
                page_path=WORKFLOW_TO_PAGE[args["workflow"]],
            )

    raise RuntimeError(
        f"Claude didn't call recommend_recipe. Stop reason: {resp.stop_reason}. "
        f"Content: {resp.content}"
    )
