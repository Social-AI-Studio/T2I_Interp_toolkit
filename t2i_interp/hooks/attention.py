import os
from inspect import isfunction

import torch
from einops import rearrange, repeat
from torch import einsum, nn

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Dummy(nn.Module):
    """Generic passthrough wrapper: returns whatever you give it."""

    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x


class HookedCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.hook_q_out = Dummy("hook_q")
        self.hook_k_out = Dummy("hook_k")
        self.hook_v_out = Dummy("hook_v")
        self.hook_attn_out = Dummy("hook_attn_out")

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        q = self.hook_q_out(q)
        context = default(context, x)
        k = self.to_k(context)
        k = self.hook_k_out(k)
        v = self.to_v(context)
        v = self.hook_v_out(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type="cuda"):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.hook_attn_out(out)
        return self.to_out(out)

    @classmethod
    def wrap_factory(old, name):
        return Dummy(old, name=name)
