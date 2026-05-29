"""Shared recipe payloads referenced from more than one page.

The full RECIPES catalogue still lives in app/pages/0_Recipes.py (it's
where it gets rendered). This module is for payloads that need to be
dropped into st.session_state from outside the Recipes page itself, like
the home page's "Reproduce paper Fig 2" CTA.
"""

from __future__ import annotations

from typing import Any

from app.lib.prompts import SPECTACLES_INFERENCE_PROMPTS, SPECTACLES_INLINE_PAIRS

# Paper-faithful Figure 2 setup: LoReFT spectacles on SDXL-Turbo. The
# original Fig 2 trained on ~1k paired prompts; this payload uses the
# bundled 12 inline pairs as a runnable demo. The goal text calls that out
# so users don't read "reproduces Figure 2" and expect identical numbers.
# Imported by 0_Recipes.py (the "Add spectacles to character portraits"
# card) and streamlit_app.py (the home-page "Run the spectacles demo" CTA).
FIG2_SPECTACLES_PAYLOAD: dict[str, Any] = {
    "workflow": "Steering",
    "goal": (
        "Add spectacles to character portraits (paper Fig 2 setup; "
        "demo-scale: 12 inline pairs vs ~1k in the paper)."
    ),
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
