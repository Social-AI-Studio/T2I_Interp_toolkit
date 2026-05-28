"""End-to-end CLI tests for the four workflow scripts.

These spawn the actual `t2i-steer`, `t2i-localise`, `t2i-stitch` commands
in a subprocess (the same code path the Streamlit pages exercise via
`run_workflow`) and assert that the run completes successfully and
produces images. They catch class of bugs the unit and AppTest layers
can't see: model-load issues, real Hydra override quoting, the
suffix-appended output_dir behaviour of run_steer, the StopIteration
bug in the buffer that this branch fixed, etc.

Marked `@pytest.mark.slow` so they don't run by default. To opt in:

    uv run pytest tests/integration/test_e2e_cli.py -v -m slow

Each test:
- runs at the smallest possible scale (1 inference step, 2 training
  steps, batch_size=2, n_inference_steps=1) so a model already cached
  on disk finishes in 30-90s on M5 Max + MPS
- skips if MPS isn't available (no other accelerator path here matches
  the Streamlit defaults)
- uses tmp_path so the run leaves no artifacts behind
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return getattr(torch.backends.mps, "is_available", lambda: False)()


_needs_mps = pytest.mark.skipif(
    not _mps_available(),
    reason="MPS not available — e2e tests target Apple Silicon for fast model load",
)


def _run_cli(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Spawn a t2i-* CLI command from the repo root, with wandb disabled.

    Mirrors how `app/lib/runner.py` invokes the same scripts from the
    Streamlit pages, so any bug that breaks the Streamlit run also breaks
    this test.
    """
    env = {**os.environ, "WANDB_MODE": "disabled"}
    result = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if result.returncode != 0:
        # Surface stdout/stderr together — easier to diagnose than separate.
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result


def _find_output_dir(parent: Path, base_name: str) -> Path | None:
    """run_steer rewrites output_dir to `<base>_<block>_alpha=<a>`. Find the
    actual dir even when the script appended a suffix."""
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(base_name)]
    return candidates[0] if candidates else None


def _make_pairs_file(tmp_path: Path, pairs: list[dict[str, str]]) -> Path:
    f = tmp_path / "inline_pairs.json"
    f.write_text(json.dumps(pairs))
    return f


# Eight pairs — the minimum size the Steering page surfaces, and the exact
# count that triggered the StopIteration in the buffer loader before the fix.
_DEMOGRAPHIC_PAIRS = [
    {"pos": "photo of a Black man", "neg": "photo of a man"},
    {"pos": "portrait of a Black man", "neg": "portrait of a man"},
    {"pos": "photo of a Black woman", "neg": "photo of a woman"},
    {"pos": "photo of a Black businessman", "neg": "photo of a businessman"},
    {"pos": "photo of a Black doctor", "neg": "photo of a doctor"},
    {"pos": "photo of a Black teacher", "neg": "photo of a teacher"},
    {"pos": "headshot of a Black person", "neg": "headshot of a person"},
    {"pos": "portrait of a Black athlete", "neg": "portrait of an athlete"},
]


@pytest.mark.slow
@_needs_mps
def test_e2e_steering_caa_8_inline_pairs(tmp_path):
    """Real CAA Steering run with 8 inline pairs — the case that triggered
    the StopIteration before the buffer-loader fix in this branch."""
    pairs_file = _make_pairs_file(tmp_path, _DEMOGRAPHIC_PAIRS)
    out_base = tmp_path / "out"
    result = _run_cli(
        [
            "uv",
            "run",
            "t2i-steer",
            "--config-name=steer/caa",
            "device=mps",
            "dtype=bfloat16",
            f"inline_pairs_file={pairs_file}",
            "alpha=5.0",
            "train_steps=2",
            "max_samples=20",
            "batch_size=2",
            "prompts=[photo of a man]",
            f"save_dir={tmp_path}/cache",
            f"output_dir={out_base}",
            f"hydra.run.dir={tmp_path}/.hydra",
            "wandb.project=null",
        ],
        timeout=300,
    )
    assert result.returncode == 0, "t2i-steer CAA + 8 inline pairs returned non-zero"

    # run_steer rewrites output_dir to `<base>_<block>_alpha=<a>`. The
    # Streamlit page's `include_prefix_siblings` mirror of this logic.
    actual = _find_output_dir(tmp_path, "out")
    assert actual is not None, f"no output dir; tmp contents: {list(tmp_path.iterdir())}"
    images = list(actual.rglob("*.png"))
    assert images, f"no images in {actual}"
    names = {img.name for img in images}
    assert any(n.startswith("baseline") for n in names), f"no baseline image in {names}"
    assert any(n.startswith("steered") for n in names), f"no steered image in {names}"


@pytest.mark.slow
@_needs_mps
def test_e2e_steering_loreft_8_inline_pairs(tmp_path):
    """Real LoReFT Steering run with 8 inline pairs. Different code path
    from CAA (paired-column dataset rather than label-column)."""
    pairs_file = _make_pairs_file(
        tmp_path,
        [
            {"pos": f"a painterly photo of a {x}", "neg": f"a photo of a {x}"}
            for x in [
                "man",
                "woman",
                "child",
                "doctor",
                "scientist",
                "teacher",
                "student",
                "businessman",
            ]
        ],
    )
    out_base = tmp_path / "out"
    result = _run_cli(
        [
            "uv",
            "run",
            "t2i-steer",
            "--config-name=steer/loreft",
            "device=mps",
            "dtype=bfloat16",
            f"inline_pairs_file={pairs_file}",
            "alpha=5.0",
            "train_steps=2",
            "max_samples=20",
            "batch_size=2",
            "num_inference_steps=4",
            "steer_steps=4",
            "prompts=[a photo of a person]",
            f"save_dir={tmp_path}/cache",
            f"output_dir={out_base}",
            f"hydra.run.dir={tmp_path}/.hydra",
            "wandb.project=null",
        ],
        timeout=300,
    )
    assert result.returncode == 0, "t2i-steer LoReFT + 8 inline pairs returned non-zero"
    actual = _find_output_dir(tmp_path, "out")
    assert actual is not None
    images = list(actual.rglob("*.png"))
    assert images, f"no images in {actual}"


@pytest.mark.slow
@_needs_mps
def test_e2e_localisation_head_ablation(tmp_path):
    """Real Localisation run: one head ablation produces baseline + modified."""
    out_dir = tmp_path / "out"
    result = _run_cli(
        [
            "uv",
            "run",
            "t2i-localise",
            "device=mps",
            "dtype=bfloat16",
            "prompt=a unicorn in a forest",
            "target_layer=down_blocks_1_attentions_0_transformer_blocks_0_attn2_out",
            "target_heads=[0]",
            "factor=0.0",
            "num_inference_steps=4",
            "seed=42",
            f"output_dir={out_dir}",
            f"hydra.run.dir={tmp_path}/.hydra",
            "wandb.project=null",
        ],
        timeout=180,
    )
    assert result.returncode == 0, "t2i-localise returned non-zero"
    assert out_dir.exists(), f"output dir missing: {out_dir}"
    images = list(out_dir.rglob("*.png"))
    assert images, f"no images produced in {out_dir}"


@pytest.mark.slow
@_needs_mps
def test_e2e_stitching_inline_prompts(tmp_path):
    """Real Stitching run with 10 inline prompts (one per model)."""
    # Stitching accepts a list of prompts (same prompt fed into both models).
    inline_prompts = [
        "a photo of a person",
        "a photo of a cat",
        "a photo of a landscape",
        "a photo of a still life",
        "a photo of a city street",
        "a photo of a forest",
        "a photo of a beach",
        "a portrait of a woman",
        "a portrait of a man",
        "a photo of a sunset",
    ]
    pairs_file = tmp_path / "stitch_prompts.json"
    pairs_file.write_text(json.dumps(inline_prompts))
    out_dir = tmp_path / "out"
    result = _run_cli(
        [
            "uv",
            "run",
            "t2i-stitch",
            "device=mps",
            "dtype=bfloat16",
            f"inline_pairs_file={pairs_file}",
            "hidden_dim=128",
            "max_samples=20",
            "num_steps=5",
            "num_inference_steps=4",
            "batch_size=2",
            "prompts=[a photo of a person]",
            f"save_dir={tmp_path}/cache",
            f"output_dir={out_dir}",
            f"hydra.run.dir={tmp_path}/.hydra",
            "wandb.project=null",
        ],
        timeout=600,  # Stitching loads two models; slower than steer/localise.
    )
    assert result.returncode == 0, "t2i-stitch returned non-zero"
    assert out_dir.exists()
    images = list(out_dir.rglob("*.png"))
    assert images, f"no images produced in {out_dir}"
