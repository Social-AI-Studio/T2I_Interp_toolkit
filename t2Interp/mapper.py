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
    def __init__(self, input_dim: int,output_dim: int, hidden_dim: int = 512, num_layers: int = 2, device = 'cpu', dtype=th.bfloat16):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim, device=device, dtype=dtype))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        predicted = self.network(x)
        # if loss_fn is not None:
        #     return predicted, loss_fn(predicted, x)
        return predicted