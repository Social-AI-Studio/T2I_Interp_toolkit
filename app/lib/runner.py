"""Run a t2i-* CLI as a subprocess with live log streaming to Streamlit.

Wraps subprocess.Popen so each stdout line is yielded as it arrives. Streamlit
pages display the lines inside `st.status(...)` for a collapsible progress
view. We use subprocess rather than calling main() in-process because
Hydra hijacks sys.argv and `@hydra.main` is not re-entrant in the same kernel.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass


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
