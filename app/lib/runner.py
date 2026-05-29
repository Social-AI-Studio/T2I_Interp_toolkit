"""Run a t2i-* CLI as a subprocess with live log streaming to Streamlit.

Wraps subprocess.Popen so each stdout line is yielded as it arrives. Streamlit
pages display the lines inside `st.status(...)` for a collapsible progress
view. We use subprocess rather than calling main() in-process because
Hydra hijacks sys.argv and `@hydra.main` is not re-entrant in the same kernel.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def render_workflow_run(
    command: str,
    overrides: list[str],
    *,
    out_dir: str,
    running_label: str,
    done_label: str = "Done",
):
    """Streamlit helper: spawn a CLI, stream logs into `st.status`, return result.

    The 4 workflow pages used to inline 15+ identical lines apiece: open a
    `st.status`, drain `run_workflow` into a 20-line rolling tail in a code
    block, time the run, and re-label the status based on returncode. This
    helper centralises that scaffolding so the pages just supply the labels.

    Returns `(WorkflowResult | None, elapsed_seconds)`. Pages keep their own
    Results section because the per-workflow rendering is too different to
    share (image-pairing labels, metric tiles, fingerprint detail).
    """
    import time as _time

    import streamlit as st

    with st.status(running_label, expanded=True) as status:
        line_box = st.empty()
        recent: list[str] = []
        start = _time.time()
        result: WorkflowResult | None = None
        for event in run_workflow(command, overrides, output_dir=out_dir):
            if isinstance(event, str):
                recent.append(event)
                line_box.code("\n".join(recent[-20:]))
            else:
                result = event
        elapsed = _time.time() - start
        if result is not None and result.returncode == 0:
            status.update(label=f"{done_label} in {elapsed:.1f}s", state="complete")
        else:
            status.update(label="Run failed. See logs above.", state="error")
    return result, elapsed


def sweep_old_streamlit_tempdirs(prefix: str, max_age_seconds: float = 3600) -> int:
    """Remove `tempfile.gettempdir()/<prefix>*` directories older than the cutoff.

    Streamlit workflow pages mkdtemp a new `streamlit_<workflow>_<rand>` dir on
    every Run so the just-produced images stay browsable until the user clicks
    Run again. Without cleanup that means each click leaks a directory of
    activations/images/.hydra metadata.

    Call this at the top of each workflow page so old runs are pruned without
    touching the in-flight one. Returns the count removed (best-effort —
    failures swallowed since cleanup is opportunistic).
    """
    if not prefix:
        return 0
    root = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        candidates = list(root.glob(f"{prefix}*"))
    except OSError:
        return 0
    for path in candidates:
        try:
            if not path.is_dir():
                continue
            if path.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed


@dataclass
class WorkflowResult:
    returncode: int
    output_dir: str  # absolute path the CLI used as cfg.output_dir
    log: str  # full stdout+stderr text


def run_workflow(
    command: str,  # one of "t2i-localise" / "t2i-steer" / "t2i-stitch" / "t2i-sae"
    overrides: list[str],  # Hydra "key=value" overrides (also: "--config-name=...")
    output_dir: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> Iterator[str | WorkflowResult]:
    """Spawn the CLI and yield log lines as they arrive. Final yield is the
    `WorkflowResult` with returncode + collected log."""
    full_env = {**os.environ, "WANDB_MODE": "disabled", **(env or {})}
    cmd = [command, *overrides]
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    log_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line)
        yield line.rstrip()
    proc.wait()
    yield WorkflowResult(
        returncode=proc.returncode,
        output_dir=output_dir,
        log="".join(log_lines),
    )
