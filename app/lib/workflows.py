"""Single source of truth for the four workflow-to-page-path mappings.

Both `app/pages/0_Recipes.py` and `app/lib/llm.py` import from here so the
mapping doesn't drift if a page is renamed or a workflow is added/removed.
"""

from __future__ import annotations

WORKFLOW_TO_PAGE: dict[str, str] = {
    "Steering": "pages/2_Steering.py",
    "Localisation": "pages/1_Localisation.py",
    "Stitching": "pages/3_Stitching.py",
    "SAE": "pages/4_SAE.py",
}

# Iteration order used by the Recipes gallery (puts Steering first since
# it's the paper's headline workflow).
WORKFLOW_ORDER: tuple[str, ...] = ("Steering", "Localisation", "SAE", "Stitching")
