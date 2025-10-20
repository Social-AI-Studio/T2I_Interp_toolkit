from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, List, Union, Sequence
import torch as th
from torch import nn
from contextlib import nullcontext
from tqdm import tqdm
from abc import ABC, abstractmethod
from loguru import logger
from pathlib import Path
from utils.utils import _to_jsonable
  
@dataclass(kw_only=True)
class Update:
    info: str = ""
    warning: bool = False
    
@dataclass 
class TrainUpdate(Update):
    step: int
    parts: Dict[str, float]        # loss components, lr, etc.
    extras: Dict[str, Any] | None = None
    
class Updater(ABC):
    @abstractmethod
    def start(self, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    @abstractmethod
    def log(self, update: Update) -> None: 
        pass
    @abstractmethod
    def done(self) -> None:
        pass
    
class WandbUpdater(Updater):
    def __init__(self, init_kwargs: Dict):
        import wandb
        self.wandb = wandb
        self.init_kwargs = init_kwargs
        self.start()
    def start(self):
        self.run = self.wandb.init(**self.init_kwargs)
    def log(self, u: Update) -> None:
        assert isinstance(u, TrainUpdate)
        data = {"step": u.step, **{k: v for k,v in u.parts.items()}}
        if u.extras: data.update(u.extras)
        self.wandb.log(data)
    def done(self):
        if self.run: self.run.finish()    
        
class SimpleUpdater(Updater):
    def __init__(self, *, bar_color: str | None = None, total: Optional[int] = None):
        self.start(bar_color=bar_color, total=total)

    def start(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        bar_color: str | None = None,
        total: Optional[int] = None,
    ) -> None:
        # colour is supported by tqdm>=4.64 (spelled 'colour')
        self.pbar = tqdm(
            total=total,
            desc=run_name or "",
            dynamic_ncols=True,
            leave=True,
            colour=bar_color,  # e.g., "green", "yellow", "red" (optional)
        )

    def _ansi(self, s: str, color: str) -> str:
        colors = {
            "yellow": "\033[33m",
            "red": "\033[31m",
            "green": "\033[32m",
            "blue": "\033[34m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color,'')}{s}{colors['reset']}"

    def log(self, u: "Update | TrainUpdate") -> None:
        if isinstance(u, Update):
            icon = "⚠️" if u.warning else "ℹ️"
            text = u.info or ""
            # Color only the message for warnings
            msg = self._ansi(text, "yellow") if u.warning else text
            self.pbar.set_postfix_str(f"{icon} {msg}", refresh=True)
            tqdm.write(f"{icon} {msg}")

            self.pbar.update(1)
            return

        # TrainUpdate
        data = {"step": u.step, **{k: v for k, v in u.parts.items()}}
        if getattr(u, "extras", None):
            data.update(u.extras)
     
        self.pbar.set_postfix(data, refresh=True)
        tqdm.write(f"ℹ️ {msg}")
        self.pbar.update(1)

    def done(self) -> None:
        self.pbar.close()   
        
class SimpleFileLogger:  # (Updater) if you want to inherit
    def __init__(
        self,
        *,
        log_path: Optional[str | Path] = "./logs/run.log",
        level: str = "INFO",
        args: Optional[Sequence[Any]] = None,
        **kwargs,
        ):
        self.sink_id = None
        self.start(log_path=log_path, level=level, args=args, **kwargs)
    
    def start(
        self,
        *,
        log_path: Optional[str | Path],
        level: str = "INFO",
        args: Optional[Sequence[Any]] = None,
        **kwargs,
    ) -> None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.sink_id = logger.add(str(p), level=level)
        
        logger.info("🔧 SimpleFileLogger started → {}", p)
        if args:
            logger.info("Args:")
            for i, a in enumerate(args):
                logger.info("  [{}] {}", i, _to_jsonable(a))
        # log all kwargs
        if kwargs:
            logger.info("Run config:")
            for k, v in kwargs.items():
                logger.info("  {} = {}", k, _to_jsonable(v))

    def log(self, u: "Update | TrainUpdate") -> None:
        if isinstance(u, TrainUpdate):
            data = {"step": u.step, **{k: v for k, v in u.parts.items()}}
            if getattr(u, "extras", None):
                data.update(u.extras)
            logger.info("train_step step={} {}", u.step, " ".join(f"{k}={v}" for k, v in data.items()))
        elif isinstance(u, Update):
            if u.warning:
                logger.warning("⚠️ {}", u.info or "")
            else:
                logger.info("ℹ️ {}", u.info or "")
            return    

    def done(self) -> None:
        logger.info("✅ Run complete")
        if self.sink_id is not None:
            logger.remove(self.sink_id)
            self.sink_id = None       