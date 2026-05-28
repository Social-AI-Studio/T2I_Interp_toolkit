"""LLM-based goal-to-recipe matcher for the Streamlit playground.

Uses the Anthropic SDK with structured tool-use to translate a free-form
user goal ("I want my generations to look more colorful and painterly")
into a concrete workflow + suggested config.

Requires `ANTHROPIC_API_KEY` in the environment. Without it,
`is_available()` returns False and pages should fall back to the
structured wizard.

The system prompt is kept stable so Anthropic's prompt cache hits across
calls. Only the user goal changes.
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
for this goal). Keep the `suggested_config` list to 3 to 6 key/value pairs.
Just the most important knobs for this goal, not every possible parameter.

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
                    "Short explanation (2 to 4 sentences) of why this workflow "
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
                    "3 to 6 key/value pairs giving a concrete starting "
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
# recipe's UI `description` in 0_Recipes.py. This prose is the LLM's
# routing context (full sentences, all four workflows in one block).
# Recipe `description`s are short card blurbs. WORKFLOW_TO_PAGE is the
# only value that must stay in sync, hence the shared import from
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
    cheap. `claude-haiku-4-5` is the right speed/cost tradeoff for routing.
    Bump to `claude-sonnet-4-6` if you want richer reasoning.

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    """
    if not is_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. "
            "`export ANTHROPIC_API_KEY=sk-ant-...` and restart Streamlit, "
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
                # Cache the long, stable system prompt. Roughly 5 minute TTL.
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


# ─────────────────────────────────────────────────────────────────────────────
# Inline-pair generation. Let the user describe a concept and get back
# N paired prompts ready to paste into the Steering page's textarea.
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_INTENT_GUIDANCE = {
    "add_attribute": (
        "User wants to ADD an attribute (e.g. spectacles, beard, long hair) to "
        "generated portraits. Each pair: `pos` is `<subject> with <attribute>`, "
        "`neg` is `<subject>` (the same subject, no attribute). Use diverse "
        "subjects (different occupations, characters, ages, contexts). Method: "
        "LoReFT works well; alpha 8 to 15."
    ),
    "suppress_concept": (
        "User wants to SUPPRESS an unwanted concept (e.g. cigarettes, weapons, "
        "violence). Each pair: `pos` is `<subject with the unwanted concept>`, "
        "`neg` is `<the same subject with a benign substitute>`. The trained "
        "direction will be SUBTRACTED at inference (negative alpha). Method: "
        "CAA with alpha = -5 to -15."
    ),
    "shift_demographic": (
        "User wants to SHIFT generations toward a specific demographic (e.g. "
        "Black people, women, older adults). Each pair: `pos` is `<demographic> "
        "<subject>`, `neg` is `<subject>` (no demographic qualifier). Cover "
        "diverse occupations/contexts. Method: CAA; alpha 5 to 10."
    ),
    "apply_style": (
        "User wants to APPLY an art style (painterly, watercolor, vintage, "
        "anime, etc.). Each pair: `pos` is `<style> <subject>`, `neg` is "
        "`a photo of <subject>` or similar plain rendering. Diverse subjects. "
        "Method: LoReFT; alpha 10 to 20."
    ),
}


GENERATE_PAIRS_TOOL = {
    "name": "generate_pairs",
    "description": (
        "Generate N paired positive/negative prompts for training a steering "
        "direction. Each pair shares structure with one varied element."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pos": {
                            "type": "string",
                            "description": "Prompt containing the target concept.",
                        },
                        "neg": {
                            "type": "string",
                            "description": (
                                "Matching prompt WITHOUT the target concept. "
                                "As similar as possible to `pos` so the contrast "
                                "isolates the concept."
                            ),
                        },
                    },
                    "required": ["pos", "neg"],
                },
                "description": "Exactly N (default 8) paired prompts.",
            },
            "method_hint": {
                "type": "string",
                "enum": ["loreft", "caa"],
                "description": (
                    "Which steering method fits this intent best. Drives the "
                    "method dropdown on the Steering page."
                ),
            },
            "alpha_hint": {
                "type": "number",
                "description": (
                    "Suggested alpha. Positive for add/shift/style; negative "
                    "(e.g. -10) for suppress."
                ),
            },
            "notes": {
                "type": "string",
                "description": "1 to 2 sentence note on what was generated and why.",
            },
        },
        "required": ["pairs", "method_hint", "alpha_hint", "notes"],
    },
}


@dataclass(frozen=True)
class PairGenerationResult:
    pairs: list[dict[str, str]]  # [{'pos': ..., 'neg': ...}, ...]
    method_hint: str  # "loreft" | "caa"
    alpha_hint: float
    notes: str

    def as_textarea(self) -> str:
        """Render as `pos | neg` lines for the Steering page's textarea."""
        return "\n".join(f"{p['pos']} | {p['neg']}" for p in self.pairs)


def generate_inline_pairs(
    intent: str,
    concept: str,
    n: int = 8,
    model: str = "claude-haiku-4-5-20251001",
) -> PairGenerationResult:
    """Ask Claude for N paired prompts for the given intent + concept.

    `intent` must be one of `_PAIR_INTENT_GUIDANCE`'s keys
    (`add_attribute`, `suppress_concept`, `shift_demographic`, `apply_style`).

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    """
    if not is_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. "
            "Use the manual templates instead, or "
            "`export ANTHROPIC_API_KEY=sk-ant-...`."
        )
    if intent not in _PAIR_INTENT_GUIDANCE:
        raise ValueError(f"intent must be one of {list(_PAIR_INTENT_GUIDANCE)}, got {intent!r}")

    import anthropic

    system = (
        "You generate paired positive/negative training prompts for "
        "interpretability steering in text-to-image diffusion models. "
        "Each pair should isolate ONE varied element so the steering direction "
        "captures that concept and nothing else. Keep prompts short (≤12 words), "
        "structurally similar, and use diverse subjects across the N pairs.\n\n"
        f"Intent for this request: {_PAIR_INTENT_GUIDANCE[intent]}"
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        tools=[GENERATE_PAIRS_TOOL],
        tool_choice={"type": "tool", "name": "generate_pairs"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate {n} paired prompts. Concept the user wants to target: {concept!r}."
                ),
            }
        ],
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "generate_pairs":
            args = block.input
            return PairGenerationResult(
                pairs=[{"pos": p["pos"], "neg": p["neg"]} for p in args["pairs"]],
                method_hint=args["method_hint"],
                alpha_hint=float(args["alpha_hint"]),
                notes=args.get("notes", ""),
            )

    raise RuntimeError(
        f"Claude didn't call generate_pairs. Stop reason: {resp.stop_reason}. "
        f"Content: {resp.content}"
    )


# Pure-Python intent-to-template fallback, used when ANTHROPIC_API_KEY is unset.
# Keep templates simple and obviously editable. Users adapt them to their concept.
INTENT_TEMPLATES: dict[str, dict[str, object]] = {
    "add_attribute": {
        "format_hint": "`<subject> with <attribute> | <subject>`",
        "method_hint": "loreft",
        "alpha_hint": 10.0,
        "starter_pairs": [
            "a man with <ATTRIBUTE> | a man",
            "a woman with <ATTRIBUTE> | a woman",
            "a child with <ATTRIBUTE> | a child",
            "a businessman with <ATTRIBUTE> | a businessman",
            "a scientist with <ATTRIBUTE> | a scientist",
            "a doctor with <ATTRIBUTE> | a doctor",
            "a teacher with <ATTRIBUTE> | a teacher",
            "a student with <ATTRIBUTE> | a student",
        ],
        "tip": (
            "Replace `<ATTRIBUTE>` with what you want to add (e.g. `spectacles` "
            "or `a beard`). Use LoReFT with alpha around 10."
        ),
    },
    "suppress_concept": {
        "format_hint": "`<subject with concept> | <subject with benign substitute>`",
        "method_hint": "caa",
        "alpha_hint": -10.0,
        "starter_pairs": [
            "a man holding a <CONCEPT> | a man holding a pen",
            "a person using <CONCEPT> | a person reading",
            "a woman with <CONCEPT> | a woman with a coffee",
            "close-up of <CONCEPT> | close-up of a flower",
            "a hand holding <CONCEPT> | a hand holding a phone",
            "a character with <CONCEPT> | a character with a book",
            "<CONCEPT> on a table | a vase on a table",
            "scene with <CONCEPT> | scene with a chair",
        ],
        "tip": (
            "Replace `<CONCEPT>` with what you want to remove. Use CAA with "
            "**negative** alpha (around -10) so the direction gets subtracted."
        ),
    },
    "shift_demographic": {
        "format_hint": "`<demographic> <subject> | <subject>`",
        "method_hint": "caa",
        "alpha_hint": 8.0,
        "starter_pairs": [
            "photo of a <DEMO> man | photo of a man",
            "portrait of a <DEMO> man | portrait of a man",
            "photo of a <DEMO> woman | photo of a woman",
            "photo of a <DEMO> businessman | photo of a businessman",
            "photo of a <DEMO> doctor | photo of a doctor",
            "photo of a <DEMO> teacher | photo of a teacher",
            "headshot of a <DEMO> person | headshot of a person",
            "portrait of a <DEMO> athlete | portrait of an athlete",
        ],
        "tip": "Replace `<DEMO>` with the demographic qualifier. Use CAA with alpha around 8.",
    },
    "apply_style": {
        "format_hint": "`<style> <subject> | a photo of <subject>`",
        "method_hint": "loreft",
        "alpha_hint": 12.0,
        "starter_pairs": [
            "a <STYLE> portrait of a man | a photo of a man",
            "a <STYLE> portrait of a woman | a photo of a woman",
            "a <STYLE> landscape | a photo of a landscape",
            "a <STYLE> still life | a photo of a still life",
            "a <STYLE> seascape | a photo of a seascape",
            "a <STYLE> garden scene | a photo of a garden",
            "a <STYLE> street market | a photo of a street market",
            "a <STYLE> portrait of a child | a photo of a child",
        ],
        "tip": (
            "Replace `<STYLE>` with the style adjective (e.g. `painterly`, "
            "`watercolor`, or `anime`). Use LoReFT with alpha around 12."
        ),
    },
}
