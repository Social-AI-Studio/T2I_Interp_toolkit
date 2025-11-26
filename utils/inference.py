from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Optional, Sequence, Tuple, List
import torch
from utils.output import Output
from utils. metrics import MetricBase
import inspect
from utils.runningstats import TrainUpdate, Updater, WandbUpdater, SimpleUpdater, SimpleFileLogger, Update

InferenceFn = Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, Any]]

@dataclass
class InferenceSpec:
    inference_fn: InferenceFn 
    stats_updaters: Optional[Sequence[Updater]] = field(default_factory=list[Updater])
    metric_fns: Optional[Sequence[MetricBase.compute]] = field(default_factory=list[MetricBase.compute])
    callback_fns: Optional[Sequence[Callable]] = field(default_factory=list[Callable])         
    name: Optional[str] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
# class Inference:
#     # """
#     # A thin wrapper around HF Trainer that calls a custom inference function
#     # to produce predictions. We override prediction_step so Trainer handles
#     # batching/DDP/AMP/gathering, while you provide the forward logic.
#     # """
#     def __init__(self, inference_spec: InferenceSpec,*args, **kw):
#         super().__init__(*args, **kw)
#         self.inference_spec = inference_spec

#     def run_inference(self) -> "Output":
#         spec = self.inference_spec

#         stats_updaters = getattr(spec, "stats_updaters", []) or []
#         metric_fns     = getattr(spec, "metric_fns", []) or []
#         callback_fns   = getattr(spec, "callback_fns", []) or []
#         args           = getattr(spec, "args", ()) or ()
#         kwargs         = getattr(spec, "kwargs", {}) or {}

#         out = None

#         with torch.no_grad():
#             gen = spec.inference_fn(*args, **kwargs)
#             try:
#                 while True:
#                     update = next(gen)
#                     assert isinstance(update, Update), f"Expected Update, got {type(update)}"
#                     for su in stats_updaters:
#                         su.log(update)
#             except StopIteration as e:
#                 assert isinstance(e.value, Output), f"Expected Output, got {type(e.value)}"
#                 out = e.value
#             except TypeError as e:
#                 out = e.value    
#             except Exception as e:
#                 raise e
#             finally:
#                 print("Finalizing stats updaters...")
#                 for su in stats_updaters:
#                     if hasattr(su, "done"):
#                         su.done()

#         # Post-processing on final Output
#         if out is not None:
#             for mf in metric_fns:
#                 mf(out, **kwargs)
#             for cb in callback_fns:
#                 cb(out, **kwargs)
#             # propagate a friendly name if provided in spec
#             out.name = getattr(spec, "name", getattr(out, "name", None))

#         return out
    

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