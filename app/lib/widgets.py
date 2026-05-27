"""Reusable Streamlit widgets for picking model preset + device + dtype."""

from __future__ import annotations

import streamlit as st


def device_dtype_picker(default_device: str | None = None) -> tuple[str, str]:
    """Sidebar widgets for device + dtype. Auto-detects what the box supports.

    Returns the Hydra-override-friendly strings (e.g. "mps", "bfloat16").
    """
    import torch

    options: list[tuple[str, str, str]] = []  # (label, device, default_dtype)
    if torch.cuda.is_available():
        options.append(("CUDA (fast)", "cuda:0", "float16"))
    if torch.backends.mps.is_available():
        options.append(("Apple MPS", "mps", "bfloat16"))
    options.append(("CPU (slow but always works)", "cpu", "float32"))

    labels = [o[0] for o in options]
    default_idx = 0
    if default_device:
        for i, (_, dev, _) in enumerate(options):
            if dev == default_device:
                default_idx = i
                break

    choice = st.sidebar.selectbox("Device", labels, index=default_idx)
    device = next(d for lbl, d, _ in options if lbl == choice)
    default_dtype = next(dt for lbl, _, dt in options if lbl == choice)
    dtype = st.sidebar.selectbox(
        "Dtype",
        ["float16", "bfloat16", "float32"],
        index=["float16", "bfloat16", "float32"].index(default_dtype),
        help="bfloat16 is the safest choice on Apple Silicon; float32 is slowest but most accurate.",
    )
    return device, dtype


def model_preset_picker(
    default: str = "sdxl_turbo",
    options: tuple[str, ...] = ("sd15", "sdxl", "sdxl_turbo"),
    key: str | None = None,
) -> str | None:
    """Sidebar picker for the `model=...` Hydra preset. None = use the
    workflow's config-default model_key.

    If `key` is passed, the widget binds to `st.session_state[key]` — seed
    that key beforehand (e.g. from a Recipes payload) to pre-fill it.
    """
    labels = ["(use config default)"] + list(options)
    help_text = (
        "Set to `(use config default)` to keep whatever the workflow's "
        "run.yaml already points at; otherwise overrides via Hydra `model=...`."
    )
    if key is not None:
        # When binding to session_state, set the initial value there once —
        # passing both `index=` and `key=` triggers a warning if the key is
        # already set. Subsequent renders read directly from session_state.
        if key not in st.session_state:
            st.session_state[key] = default if default in labels else labels[0]
        pick = st.sidebar.selectbox("Model preset", labels, key=key, help=help_text)
    else:
        default_idx = labels.index(default) if default in labels else 0
        pick = st.sidebar.selectbox("Model preset", labels, index=default_idx, help=help_text)
    return None if pick == "(use config default)" else pick
