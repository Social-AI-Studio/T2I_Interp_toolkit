"""Shared utilities for the T2I-Interp Streamlit playground."""

from app.lib.llm import (
    INTENT_TEMPLATES,
    PairGenerationResult,
    RecipeMatch,
    analyze_goal,
    generate_inline_pairs,
    is_available,
)
from app.lib.outputs import (
    collect_images,
    load_fingerprint,
    load_metrics,
    pair_baseline_modified,
    scan_fingerprints,
)
from app.lib.pages import (
    apply_payload,
    render_app_footer,
    render_run_label_sidebar,
    scenario_radio,
)
from app.lib.parsing import detect_concept, has_unresolved_placeholders, parse_pipe_lines
from app.lib.recipes import FIG2_SPECTACLES_PAYLOAD
from app.lib.runner import render_workflow_run, run_workflow, sweep_old_streamlit_tempdirs
from app.lib.widgets import device_dtype_picker, model_preset_picker
from app.lib.workflows import WORKFLOW_TO_PAGE

__all__ = [
    "FIG2_SPECTACLES_PAYLOAD",
    "INTENT_TEMPLATES",
    "PairGenerationResult",
    "RecipeMatch",
    "WORKFLOW_TO_PAGE",
    "analyze_goal",
    "apply_payload",
    "collect_images",
    "detect_concept",
    "device_dtype_picker",
    "generate_inline_pairs",
    "has_unresolved_placeholders",
    "is_available",
    "load_fingerprint",
    "load_metrics",
    "model_preset_picker",
    "pair_baseline_modified",
    "parse_pipe_lines",
    "render_app_footer",
    "render_run_label_sidebar",
    "render_workflow_run",
    "run_workflow",
    "scan_fingerprints",
    "scenario_radio",
    "sweep_old_streamlit_tempdirs",
]
