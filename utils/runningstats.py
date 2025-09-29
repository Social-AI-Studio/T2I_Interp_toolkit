from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, List, Union
import torch as th
from torch import nn
from contextlib import nullcontext
from tqdm import tqdm
from abc import ABC, abstractmethod

    
@dataclass
class TrainUpdate:
    step: int
    parts: Dict[str, float]        # loss components, lr, etc.
    extras: Dict[str, Any] | None = None
    
class Updater(ABC):
    @abstractmethod
    def start(self, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    @abstractmethod
    def log(self, update: TrainUpdate) -> None: 
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
    def log(self, u: TrainUpdate):
        # if u.step % self.log_every == 0:
        data = {"step": u.step, **{k: v for k,v in u.parts.items()}}
        if u.extras: data.update(u.extras)
        self.wandb.log(data, step=u.step)
    def done(self):
        if self.run: self.run.finish()    