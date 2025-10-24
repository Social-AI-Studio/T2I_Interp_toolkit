from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Optional, Sequence, Tuple, List
import torch
from utils.output import Output
from utils. metrics import MetricBase

InferenceFn = Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, Any]]

@dataclass
class InferenceSpec:
    inference_fn: InferenceFn 
    metric_fns: Optional[Sequence[MetricBase.compute]] = field(default_factory=list[MetricBase.compute])
    callback_fns: Optional[Sequence[Callable]] = field(default_factory=list[Callable])         
    name: Optional[str] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
class Inference:
    # """
    # A thin wrapper around HF Trainer that calls a custom inference function
    # to produce predictions. We override prediction_step so Trainer handles
    # batching/DDP/AMP/gathering, while you provide the forward logic.
    # """
    def __init__(self, inference_spec: InferenceSpec,*args, **kw):
        super().__init__(*args, **kw)
        self.inference_spec = inference_spec

    def run_inference(self) -> tuple:
        # Move inputs to the right device the Trainer way
        # inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            out: Output = self.inference_spec.inference_fn(
                *self.inference_spec.args,
                **self.inference_spec.kwargs
            )
            if len(self.inference_spec.metric_fns)>0:
                for cb in self.inference_spec.metric_fns:
                    cb(out, **self.inference_spec.kwargs)
                    
            if len(self.inference_spec.callback_fns)>0:
                for cb in self.inference_spec.callback_fns:
                    cb(out, **self.inference_spec.kwargs)        
            out.name = self.inference_spec.name        
        return out   