from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Optional, Sequence, Tuple, List
import torch
from utils.output import Output
from utils. metrics import MetricBase
from utils.runningstats import TrainUpdate, Updater, WandbUpdater

TrainingFn = Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, Any]]

@dataclass
class TrainingSpec:
    fn: TrainingFn 
    stats_updaters: Optional[Sequence[Updater]] = field(default_factory=list[WandbUpdater])
    callback_fns: Optional[Sequence[Callable]] = field(default_factory=list[Callable])         
    name: Optional[str] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
class Training:
    # """
    # A thin wrapper around HF Trainer that calls a custom inference function
    # to produce predictions. We override prediction_step so Trainer handles
    # batching/DDP/AMP/gathering, while you provide the forward logic.
    # """
    def __init__(self, spec: TrainingSpec,*args, **kw):
        super().__init__(*args, **kw)
        self.training_spec = spec

    def run_trainer(self) -> tuple:
        gen = self.training_spec.fn(
            *self.training_spec.args,
            **self.training_spec.kwargs
        )
        while True:
            try:
                update = next(gen)
                assert isinstance(update, TrainUpdate)
                for su in self.training_spec.stats_updaters:
                    su.log(update)
            except StopIteration as e:  
                out = e.value
                for su in self.training_spec.stats_updaters:
                    su.done()
                break                    
        
        if len(self.training_spec.callback_fns)>0:
            for cb in self.training_spec.callback_fns:
                cb(out, **self.training_spec.kwargs)
                      
        return out   