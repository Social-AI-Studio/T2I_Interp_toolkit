from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from t2i_interp.utils.utils import _to_jsonable


@dataclass(kw_only=True)
class Update:
    info: str = ""
    warning: bool = False


@dataclass
class TrainUpdate(Update):
    step: int
    parts: dict[str, float]  # loss components, lr, etc.
    extras: dict[str, Any] | None = None


class Updater(ABC):
    @abstractmethod
    def start(self, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        pass

    @abstractmethod
    def log(self, update: Update) -> None:
        pass

    @abstractmethod
    def done(self) -> None:
        pass


class WandbUpdater(Updater):
    def __init__(self, init_kwargs: dict):
        import wandb

        self.wandb = wandb
        self.init_kwargs = init_kwargs
        self.start()

    def start(self):
        self.run = self.wandb.init(**self.init_kwargs)

    def log(self, u: Update) -> None:
        # assert isinstance(u, TrainUpdate)
        if not isinstance(u, TrainUpdate):
            return
        data = {"step": u.step, **{k: v for k, v in u.parts.items()}}
        if u.extras:
            data.update(u.extras)
        self.wandb.log(data)

    def done(self):
        if self.run:
            self.run.finish()


class SimpleUpdater(Updater):
    def __init__(self, *, bar_color: str | None = None, total: int | None = None):
        self.start(bar_color=bar_color, total=total)

    def start(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        bar_color: str | None = None,
        total: int | None = None,
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
        return f"{colors.get(color, '')}{s}{colors['reset']}"

    def log(self, u: Update | TrainUpdate) -> None:
        # Check subclass TrainUpdate FIRST because it inherits from Update
        if isinstance(u, TrainUpdate):
            data = {"step": u.step, **{k: v for k, v in u.parts.items()}}
            if getattr(u, "extras", None):
                data.update(u.extras)
            self.pbar.set_postfix(data, refresh=True)
            
            if u.info:
                 tqdm.write(f"ℹ️ {u.info}")
            
            self.pbar.update(1)
            return

        # Check base class Update SECOND
        if isinstance(u, Update):
            icon = "⚠️" if u.warning else "ℹ️"
            text = u.info or ""
            # Color only the message for warnings
            msg = self._ansi(text, "yellow") if u.warning else text
            self.pbar.set_postfix_str(f"{icon} {msg}", refresh=True)
            tqdm.write(f"{icon} {msg}")

            self.pbar.update(1)
            return

    def done(self) -> None:
        self.pbar.close()


class SimpleFileLogger(Updater):  # (Updater) if you want to inherit
    def __init__(
        self,
        *,
        log_path: str | Path | None = "./logs/run.log",
        level: str = "INFO",
        args: Sequence[Any] | None = None,
        **kwargs,
    ):
        self.sink_id = None
        self.start(log_path=log_path, level=level, args=args, **kwargs)

    def start(
        self,
        *,
        log_path: str | Path | None,
        level: str = "INFO",
        args: Sequence[Any] | None = None,
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

    def log(self, u: Update | TrainUpdate) -> None:
        if isinstance(u, TrainUpdate):
            data = {"step": u.step, **{k: v for k, v in u.parts.items()}}
            if getattr(u, "extras", None):
                data.update(u.extras)
            logger.info(
                "train_step step={} {}", u.step, " ".join(f"{k}={v}" for k, v in data.items())
            )
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
