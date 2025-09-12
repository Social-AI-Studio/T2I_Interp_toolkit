from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Output:
    preds: Any
    labels: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None