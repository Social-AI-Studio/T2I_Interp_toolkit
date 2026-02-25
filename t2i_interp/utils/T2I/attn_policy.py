from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Mapping, Union, Tuple
import torch as t

Tensor = t.Tensor

def to_index(idx, length: int, device: t.device) -> Tensor:
    if idx is None:
        return t.arange(length, device=device, dtype=t.long)
    if isinstance(idx, slice):
        return t.arange(length, device=device, dtype=t.long)[idx]
    if t.is_tensor(idx):
        return idx.to(device=device, dtype=t.long).flatten()
    if isinstance(idx, int):
        return t.tensor([idx], device=device, dtype=t.long)
    return t.tensor(list(idx), device=device, dtype=t.long)

# def pick_heads(heads, module):
#     # allow heads as dict keyed by module.attr_name
#     if isinstance(heads, dict):
#         return heads.get(getattr(module, "attr_name", None), None)
#     return heads

@dataclass
class AttnScalePolicy:
    selection: Dict[str, Any]
    factor: float
    n_heads: Optional[int] = None

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        value = kwargs.get("value", None)
        if value is None:
            return x
        B, S, D = x.shape
        H = self.n_heads
        assert H is not None

        spatial = to_index(self.selection.get("spatial_location", None), S, x.device)
        heads   = self.selection.get("heads", None)
        heads   = to_index(heads, int(H), x.device)

        hs = x.view(B, S, int(H), -1).clone()  # (B,S,H,d)
        hs[:, spatial[:, None], heads[None, :], :] *= float(self.factor)
        return hs.view(B, S, -1)

@dataclass
class AttnReplaceSelectedPolicy:
    selection: Dict[str, Any]
    # value: Union[Tensor, float, int]   # simplest: full tensor (B,S,D) or scalar
    n_heads: Optional[int] = None

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        value = kwargs.get("value", None)
        if value is None:
            return x
        B, S, D = x.shape
        H = self.n_heads
        assert H is not None

        spatial = to_index(self.selection.get("spatial_location", None), S, x.device)
        heads   = self.selection.get("heads", None)
        heads   = to_index(heads, int(H), x.device)

        out = x.view(B, S, int(H), -1).clone()

        # v = value
        value = value.to(device=x.device, dtype=x.dtype)
        if value.shape != x.shape:
            # allow view if same number of elements
            assert value.numel() == x.numel()
            value = value.view_as(x)
        v4 = value.view(B, S, int(H), -1)
        out[:, spatial[:, None], heads[None, :], :] = v4[:, spatial[:, None], heads[None, :], :]
        return out.view(B, S, -1)
