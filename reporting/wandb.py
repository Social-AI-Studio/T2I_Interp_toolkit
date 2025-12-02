from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import wandb
from PIL import Image

from reporting.base import Reporter
from utils.image_ops import _as_pil
from utils.output import Output


class WandbReporter(Reporter):
    def __init__(self, init_kwargs: dict):
        self.init_kwargs = init_kwargs
        self.run: wandb.sdk.wandb_run.Run | None = None

    def start(self) -> None:
        self.run = wandb.init(**self.init_kwargs)

    def _ensure_run(self) -> bool:
        """Start a run if none exists. Returns True iff we started it."""
        if self.run is None:
            self.run = wandb.init(**(self.init_kwargs or {}))
            return True
        return False

    @staticmethod
    def _to_scalar(v) -> float:
        if isinstance(v, (list, tuple)) and v:
            return float(v[0])
        return float(v)

    @staticmethod
    def _metric_keys(outputs: list[Output]) -> list[str]:
        keys: set[str] = set()
        for o in outputs:
            if not getattr(o, "metrics", None):
                continue
            keys.update(o.metrics.keys())
        return sorted(keys)

    def save_artifact(
        self, outputs: list[Output], artifact_name="all_images", artifact_type="dataset"
    ):
        def add_pil_to_artifact(pil: Image.Image, path_in_artifact: str):
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            with art.new_file(path_in_artifact, mode="wb") as f:
                f.write(buf.getbuffer())

        art = wandb.Artifact(artifact_name, type=artifact_type)
        # map (group, idx) -> artifact relative path
        refs: dict[tuple[str, int], str] = {}
        for o in outputs:
            name = getattr(o, "name", None) or "sample"
            imgs: Sequence[Any] = o.preds if isinstance(o.preds, (list, tuple)) else [o.preds]
            for i, img in enumerate(imgs):
                pil = _as_pil(img)
                fn = f"{name}/{i:04d}.png"
                add_pil_to_artifact(pil, fn)
                refs[(name, i)] = fn
        wandb.log_artifact(art)
        return refs, art

    def log_table(self, outputs: list[Output] | Output, add_links: bool = False, **kwargs):
        started_here = self._ensure_run()
        if not isinstance(outputs, list):
            outputs = [outputs]
        try:
            metric_cols = self._metric_keys(outputs)
            cols = (
                ["group", "idx", "baseline", "steered image"]
                + metric_cols
                + (["link"] if add_links else [])
            )
            table = wandb.Table(columns=cols)

            link_map = {}
            if add_links:
                link_map, art = self.save_artifact(outputs)

            for o in outputs:
                name = getattr(o, "name", None) or "sample"
                imgs: Sequence[Any] = o.preds
                baseline_imgs = o.baselines or []
                metrics: dict[str, list[float]] = o.metrics or {}
                for i, img in enumerate(imgs):
                    pred_pil = (
                        _as_pil(img) if img is not None else Image.new("RGB", (512, 512), (0, 0, 0))
                    )
                    pred_cell = wandb.Image(pred_pil, caption=f"{name} [{i}]")
                    if i < len(baseline_imgs) and baseline_imgs[i] is not None:
                        base_pil = _as_pil(baseline_imgs[i])
                        base_cell = wandb.Image(base_pil, caption=f"{name} baseline [{i}]")
                    else:
                        base_cell = ""  # blank cell is fine in W&B tables
                    # row = [name, i, wandb.Image(pred_pil, caption=f"{name} [{i}]")]
                    row = [name, i, base_cell, pred_cell]
                    if len(metric_cols) == 0 or len(metrics) == 0:
                        table.add_data(*row)
                        continue
                    for k in metric_cols:
                        v = metrics[k][i] if k in metrics and i < len(metrics[k]) else np.nan
                        try:
                            if isinstance(v, (list, tuple)) and v:
                                v = float(v[0])
                            else:
                                v = float(v)
                        except Exception:
                            v = np.nan
                        row.append(v)
                    if add_links:
                        # artifact paths render as clickable in the UI
                        row.append(
                            link_map.get((name, i), "")
                        )  # e.g., "artifact_name/dir/file.png"
                    table.add_data(*row)

            wandb.log({"report/master_table": table})
        finally:
            if started_here:
                self.finish()

    def log_summaries(self, scalar_metrics: dict[str, float]) -> None:
        pass

    def start_silent(self) -> None:
        """Convenience to suppress Diffusers progress bars if you have your own."""
        try:
            from diffusers.utils import logging as dlog

            dlog.disable_progress_bar()
        except Exception:
            pass
        self.start()

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
