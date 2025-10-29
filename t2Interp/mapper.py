# stitcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import copy
import inspect
import re
import torch as th
import torch.nn as nn
from t2Interp.T2I import T2IModel

# class Mapper(nn.Module):
#     """
#     Tools for:
#       Training a mapper network to map between the outputs of two modules
#     """
#     # mapper training
#     def train():
#         pass
    
class AffineMapper(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.scale = nn.Parameter(th.ones(input_dim))
        self.shift = nn.Parameter(th.zeros(input_dim))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * self.scale + self.shift
    
class MLPMapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def device(self) -> th.device:
        # works after .to(device) or .cuda()
        try:
            return next(self.parameters()).device
        except StopIteration:  # no params? fall back to CPU or a buffer (if you add one)
            return th.device("cpu")
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.network(x)
    
class MLPMapperTwoHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],     # must be length 2
        hidden_dim: int = 512,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(output_dims) == 2, "output_dims must have length 2 (for two heads)."
        d1, d2 = output_dims

        # Shared trunk
        layers = [
            nn.Linear(input_dim, hidden_dim),
            activation,
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        # Two heads
        self.head1 = nn.Linear(hidden_dim, d1)
        self.head2 = nn.Linear(hidden_dim, d2)

    @property
    def device(self) -> th.device:
        p = next(self.parameters(), None)
        return p.device if p is not None else th.device("cpu")

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        h = self.trunk(x)
        y1 = self.head1(h)
        y2 = self.head2(h)
        return y1, y2   