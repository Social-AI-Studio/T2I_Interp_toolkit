from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Optional, Sequence
import torch
from utils.output import Output
from utils. metrics import MetricBase

TrainingFn = Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, Any]]

@dataclass
class TrainingSpec:
    fn: TrainingFn 
    metric_fns: Optional[Sequence[MetricBase.compute]] = field(default_factory=list[MetricBase.compute])
    callback_fns: Optional[Sequence[Callable]] = field(default_factory=list[Callable])         
    name: Optional[str] = None
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

    def process(self) -> tuple:
        out: Output = self.training_spec.fn(
            **self.training_spec.kwargs
        )
        if len(self.training_spec.metric_fns)>0:
            for cb in self.training_spec.metric_fns:
                cb(out, **self.training_spec.kwargs)
        out.name = self.training_spec.name        
        return out   