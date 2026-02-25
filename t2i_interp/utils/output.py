from dataclasses import dataclass
from typing import Any

# @dataclass
# class IO:
#     name:Optional[str]


@dataclass
class Output:
    preds: Any = None
    baselines: Any = None
    labels: Any | None = None
    metrics: list[dict[str, float]] | None = None
    name: str | None = None
    run_metadata: dict[str, Any] | None = None
    best_ckpt: str | None = None

    # def __init__(self,**kwargs):
    #     self.preds = kwargs.get("preds")
    #     self.baselines = kwargs.get("baselines", None)
    #     self.labels = kwargs.get("labels", None)
    #     self.metrics = kwargs.get("metrics", None)
    #     self.name = kwargs.get("name", None)
    #     self.run_metadata = kwargs.get("run_metadata", None)
    #     self.best_ckpt = kwargs.get("best_ckpt", None)

    # def copy(self):
    #     return Output(preds=self.preds, metrics=dict(self.metrics or {}), labels=self.labels, )
