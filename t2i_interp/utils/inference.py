from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from t2i_interp.utils.metrics import MetricBase
from t2i_interp.utils.output import Output
from t2i_interp.utils.runningstats import (
    Updater,
)
from t2i_interp.utils.generic import call_with_filtered_kwargs

InferenceFn = Callable[[torch.nn.Module, dict[str, Any]], dict[str, Any]]


@dataclass
class InferenceSpec:
    inference_fn: InferenceFn
    stats_updaters: Sequence[Updater] | None = field(default_factory=list[Updater])
    metric_fns: Sequence[MetricBase.compute] | None = field(
        default_factory=list[MetricBase.compute]
    )
    callback_fns: Sequence[Callable] | None = field(default_factory=list[Callable])
    name: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class Inference:
    def __init__(self, inference_spec: InferenceSpec, *args, **kw):
        super().__init__(*args, **kw)
        self.inference_spec = inference_spec

    def run_inference(self) -> Output:
        with torch.no_grad():
            out = Output()
            out.name = self.inference_spec.name
            out.preds = call_with_filtered_kwargs(
                self.inference_spec.inference_fn, 
                **self.inference_spec.kwargs
            )
            if len(self.inference_spec.metric_fns) > 0:
                for cb in self.inference_spec.metric_fns:
                    call_with_filtered_kwargs(
                        cb, out, **self.inference_spec.kwargs
                    )

            if len(self.inference_spec.callback_fns) > 0:
                for cb in self.inference_spec.callback_fns:
                    call_with_filtered_kwargs(
                        cb, out, **self.inference_spec.kwargs
                    )
        return out
