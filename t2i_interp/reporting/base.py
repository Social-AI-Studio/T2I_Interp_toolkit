# reporting/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

from t2i_interp.utils.output import Output


class Reporter(ABC):
    """Abstract base for report backends (W&B, PDF, Streamlit, etc.)."""

    @abstractmethod
    def start(self) -> None:
        """Open the reporting session/run."""
        pass

    @abstractmethod
    def log_table(self, outputs: list[Output], metric: str) -> None:
        """Log a sortable table of (name, image, metric)."""
        pass

    @abstractmethod
    def log_summaries(self, scalar_metrics: dict[str, float]) -> None:
        """Log run-level summary numbers (means, stds, etc.)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Close out the session."""
        pass
