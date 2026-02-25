from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from t2i_interp.utils.output import Output
from t2i_interp.utils.runningstats import (
    SimpleFileLogger,
    Update,
    Updater,
)
from t2i_interp.utils.generic import call_with_filtered_kwargs

TrainingFn = Callable[[torch.nn.Module, dict[str, Any]], dict[str, Any]]


@dataclass
class TrainingSpec:
    training_function: TrainingFn
    stats_updaters: Sequence[Updater] | None = field(default_factory=list[Updater])
    callback_fns: Sequence[Callable] | None = field(default_factory=list[Callable])
    name: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class Training:
    # """
    # A thin wrapper around HF Trainer that calls a custom inference function
    # to produce predictions. We override prediction_step so Trainer handles
    # batching/DDP/AMP/gathering, while you provide the forward logic.
    # """
    def __init__(self, spec: TrainingSpec, *args, **kw):
        super().__init__(*args, **kw)
        self.training_spec = spec
        # add type simple logger doesnt exist in updaters, add it
        if not any(isinstance(su, SimpleFileLogger) for su in self.training_spec.stats_updaters):
            filelogger = SimpleFileLogger(args=self.training_spec.args, **self.training_spec.kwargs)
            self.training_spec.stats_updaters = list(self.training_spec.stats_updaters)
            self.training_spec.stats_updaters.insert(0, filelogger)

    def run_trainer(self) -> Output:
        out = Output()
        try:
            gen = call_with_filtered_kwargs(self.training_spec.training_function, *self.training_spec.args, **self.training_spec.kwargs)
            for item in gen:
                for su in self.training_spec.stats_updaters:
                    if isinstance(item, Update):
                        su.log(item)
                    else:
                        out.preds = item

            if len(self.training_spec.callback_fns) > 0:
                for cb in self.training_spec.callback_fns:
                    cb(out, **self.training_spec.kwargs)
        finally:
            for su in self.training_spec.stats_updaters:
                su.done()
        return out
