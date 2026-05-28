"""Shared recipe payloads referenced from more than one page.

The full RECIPES catalogue still lives in app/pages/0_Recipes.py (it's
where it gets rendered). This module is for payloads that need to be
dropped into st.session_state from outside the Recipes page itself, like
the home page's "Reproduce paper Fig 2" CTA.
"""

from __future__ import annotations

from typing import Any

from app.lib.prompts import SPECTACLES_INFERENCE_PROMPTS, SPECTACLES_INLINE_PAIRS

# The paper Fig 2 reproduction. Imported by both 0_Recipes.py (used as the
# Recipe.fields for the "Add spectacles to character portraits" card) and
# streamlit_app.py (used by the "Reproduce paper Fig 2" home-page CTA).
FIG2_SPECTACLES_PAYLOAD: dict[str, Any] = {
    "workflow": "Steering",
    "goal": "Add spectacles to character portraits (paper Fig 2 reproduction).",
    "fields": {
        "method": "loreft",
        "model_preset": "sdxl_turbo",
        "prompts": SPECTACLES_INFERENCE_PROMPTS,
        "alpha": 10.0,
        "max_samples": 100,
        "train_steps": 100,
        "inline_pairs": SPECTACLES_INLINE_PAIRS,
    },
}
