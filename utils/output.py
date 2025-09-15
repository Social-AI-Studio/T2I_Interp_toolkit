from dataclasses import dataclass
from typing import Any, Dict, Optional

# @dataclass
# class IO:
#     name:Optional[str]
    
@dataclass
class Output:
    preds: Any
    labels: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    name: Optional[str] = None
    
    def copy(self):
        return Output(preds=self.preds, metrics=dict(self.metrics or {}), labels=self.labels)