from __future__ import annotations

import datetime
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from t2i_interp.reporting.config_loader import load_config
from t2i_interp.reporting.wandb import WandbReporter
from t2i_interp.utils.output import Output
from t2i_interp.utils.utils import _to_jsonable


def _ts():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


# def _safe_json(obj: Any):
#     def default(o):
#         try:
#             return asdict(o)  # dataclasses
#         except Exception:
#             return str(o)
#     return json.dumps(obj, default=default, ensure_ascii=False, indent=2)


@dataclass
# class OutputManagerConfig:
#     root_dir: Union[str, Path] = "./runs"

@dataclass
class OutputManager:
    # cfg: OutputManagerConfig
    run_dir: Path = field(init=False)
    paths: dict[str, Path] = field(init=False, default_factory=dict)

    def __init__(self, **kwargs):
        assert kwargs.get("workflow") is not None, "OutputManager requires a workflow argument"
        assert kwargs.get("run_name") is not None, "OutputManager requires a run_name argument"

        self.root_dir = kwargs.get("root_dir", "./runs")
        root = Path(self.root_dir)
        self.workflow = kwargs.get("workflow")
        run_name = f"{kwargs.get('run_name')}_{self.workflow}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.run_dir = root / run_name
        # create dirs
        self.paths = {
            "viz": self.run_dir / "viz",
            "report": self.run_dir / "report",
            "artifacts": self.run_dir / "artifacts",
            "logs": self.run_dir / "logs",
        }
        for p in [self.run_dir, *self.paths.values()]:
            p.mkdir(parents=True, exist_ok=True)

    # ---------- metadata ----------
    def write_metadata(self, out: Output, **kwargs):
        path = self.run_dir / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = dict(getattr(out, "run_metadata", {}) or {})
        if kwargs:
            meta.update(kwargs)

        # If meta may contain non-JSON types, run through your helper first:
        meta = _to_jsonable(meta)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- convenience writers ----------
    def save_json(self, relpath: str, obj: Any):
        p = self.run_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_to_jsonable(obj), encoding="utf-8")
        tmp.replace(p)
        return p

    def save_best_ckpt(self, out: Output, **kwargs):
        torch.save(out.best_ckpt, self.paths["artifacts"] / "best_ckpt.pt")
        return

    def write_to_wandb(self, out: Output, **kwargs):
        wb_cfg = kwargs.get("wandb_init_kwargs", load_config("reporting/config.yaml"))
        reporter = WandbReporter(init_kwargs=wb_cfg)
        reporter.log_table(out)

    def save_bytes(self, relpath: str, data: bytes):
        p = self.run_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p

    def copy_in(self, src: str | Path, rel_dst: str) -> Path:
        src = Path(src)
        dst = self.run_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def register_artifact(
        self, local_path: str | Path, *, subdir: str = "artifacts", name: str | None = None
    ) -> Path:
        src = Path(local_path)
        name = name or src.name
        return self.copy_in(src, f"{subdir}/{name}")

    # ---------- paths for your three buckets ----------
    @property
    def viz_dir(self) -> Path:
        return self.paths["viz"]

    @property
    def report_dir(self) -> Path:
        return self.paths["report"]

    @property
    def artifacts_dir(self) -> Path:
        return self.paths["artifacts"]

    @property
    def logs_dir(self) -> Path:
        return self.paths["logs"]
