"""
Module accessors for reading/writing model activations.

Provides IOType enum and ModuleAccessor class for consistent
input/output access across model submodules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch as th
from nnsight import Envoy
from nnsight.intervention.tracing.globals import Object

TraceTensor = th.Tensor | Object


class IOType(Enum):
    """Enum to specify input or output access."""

    INPUT = "input"
    OUTPUT = "output"


class ModuleAccessor:
    """
    I/O accessor that provides unified input/output access with getter/setter.

    Wraps a model module (nn.Module or Envoy) and provides a consistent
    .value property for reading and writing activations.

    Args:
        module: The target module (nn.Module or Envoy).
        attr_name: Human-readable name for this accessor.
        io_type: Whether to access INPUT or OUTPUT of the module.
        returns_tuple: If True, unwrap the first element of tuple outputs.

    Example:
        accessor = ModuleAccessor(model.unet.conv_out, "conv_out", IOType.OUTPUT)
        # In a trace context:
        features = accessor.value  # reads module output
        accessor.value = features * 2.0  # writes modified output
    """

    def __init__(
        self,
        module: th.nn.Module | Envoy,
        attr_name: str | None,
        io_type: IOType | None,
        returns_tuple: bool = False,
    ):
        self.module = module
        self.attr_name = attr_name
        self.io_type = io_type
        self.returns_tuple = returns_tuple

    @property
    def value(self) -> TraceTensor | Envoy:
        """Get the activation value (input or output depending on io_type)."""
        if self.io_type is None:
            raise ValueError("Cannot get the value of a module accessor without io_type.")
        if self.io_type.value == "input":
            target = self.module.input
        elif self.io_type.value == "output":
            target = self.module.output
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")
        if self.returns_tuple:
            return target[0]
        return target

    @value.setter
    def value(self, new):
        """Set the activation value (input or output depending on io_type)."""
        if self.io_type is None:
            raise ValueError("Cannot set the value of a module accessor without io_type.")
        kind = getattr(self.io_type, "value", self.io_type)

        if kind == "input":
            if self.returns_tuple:
                old = self.module.input
                rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
                self.module.input = (new, *rest)
            else:
                self.module.input = new
        elif kind == "output":
            if self.returns_tuple:
                old = self.module.output
                rest = tuple(old[1:]) if isinstance(old, tuple) and len(old) > 1 else ()
                self.module.output = (new, *rest)
            else:
                self.module.output = new
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")

    @property
    def heads(self) -> int:
        """Get number of attention heads (if available on module)."""
        return getattr(self.module, "heads", None)

    @property
    def inputs(self) -> Any:
        """Access full inputs tuple of the module."""
        return self.module.inputs

    @inputs.setter
    def inputs(self, new):
        """Set the full inputs tuple of the module."""
        self.module.inputs = new


class AttentionAccessor:
    """Placeholder for future attention-specific accessor functionality."""

    def __init__(self):
        pass
