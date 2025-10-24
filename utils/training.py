from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Protocol, Optional, Sequence, Tuple, List
import torch
from utils.output import Output
from utils. metrics import MetricBase
from utils.runningstats import TrainUpdate, Updater, WandbUpdater, SimpleUpdater, SimpleFileLogger, Update
import inspect

TrainingFn = Callable[[torch.nn.Module, Dict[str, Any]], Dict[str, Any]]

@dataclass
class TrainingSpec:
    fn: TrainingFn 
    stats_updaters: Optional[Sequence[Updater]] = field(default_factory=list[Updater])
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
        # add type simple logger doesnt exist in updaters, add it
        if not any(isinstance(su, SimpleFileLogger) for su in self.training_spec.stats_updaters):
            filelogger = SimpleFileLogger(args=self.training_spec.args, **self.training_spec.kwargs)
            self.training_spec.stats_updaters = list(self.training_spec.stats_updaters)
            self.training_spec.stats_updaters.insert(0, filelogger)


    def run_trainer(self) -> tuple:
        # assert inspect.isgeneratorfunction(self.training_spec.fn), "Training function must be a generator"
        gen = self.training_spec.fn(
            *self.training_spec.args,
            **self.training_spec.kwargs
        )
        
        try:
            while True:
                update = next(gen)
                assert isinstance(update, Update)
                for su in self.training_spec.stats_updaters:
                    su.log(update)
        except StopIteration as e:  
            assert isinstance(e.value, Output)
            out = e.value
        except Exception as e:
            raise e
        finally:    
            for su in self.training_spec.stats_updaters:
                su.done()                           
        
        if len(self.training_spec.callback_fns)>0:
            for cb in self.training_spec.callback_fns:
                cb(out, **self.training_spec.kwargs)
                      
        return out   